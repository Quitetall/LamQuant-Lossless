//! Per-channel **RLS** (recursive least squares) lossless codec (ADR 0054).
//!
//! RLS is the SOTA lossless predictor (MPEG-4 ALS; the 2024 multichannel-EEG
//! SOTA) and the one HHI does NOT use (it uses LMS-16). RLS converges far faster
//! and tracks non-stationarity, which is exactly where our static 5/3+LPC
//! collapses — measured −27.9% on the hard 21-channel EEG (`ma`) and −9.3% on ECG
//! vs the current path, while keep-best at the container level makes it never
//! worse on the signals it doesn't help.
//!
//! **Lossless reconstruction:** the f64 prediction is rounded to an integer and
//! the integer residual `e = x − round(pred)` is coded; both encoder and decoder
//! run the *identical* RLS recursion on the (losslessly exact) reconstructed
//! history, so they form the same `pred` and reconstruct `x = e + round(pred)`
//! exactly. RLS uses only `+ − × ÷` (no fma/transcendentals), which IEEE-754
//! mandates as correctly-rounded ⇒ **deterministic across platforms**, so this is
//! `no_std`-decodable and host↔MCU bit-identical despite being float.
//!
//! Wire body: `[n_ch u16][t u32]` then per channel `[golomb_len u32][golomb bytes]`
//! where the golomb stream is `golomb::encode_dense(rls_residual)`.

// Index-based loops are the natural form for the RLS matrix recursion.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};

/// Predictor order (8 was the RD sweet spot — bigger gave no gain, more compute).
const ORDER: usize = 8;
/// Forgetting factor (0.999 = stable; faster forgetting diverges, per the sweep).
const LAMBDA: f64 = 0.999;
/// Initial inverse-correlation scale `P = (1/δ)·I`.
const DELTA: f64 = 1.0;
/// Periodic reset: standard RLS accumulates numerical error and the inverse-
/// correlation matrix loses positive-definiteness over long runs (measured: great
/// to ~30k samples, then diverges). Restart the recursion every `RESET_PERIOD`
/// samples — deterministic on both sides, so it stays bit-exact, and bounds the
/// drift while costing only a short re-adaptation transient.
const RESET_PERIOD: usize = 16384;

/// One channel's RLS state (fixed order ⇒ stack arrays).
struct Rls {
    w: [f64; ORDER],
    p: [[f64; ORDER]; ORDER],
    hist: [f64; ORDER],
}

impl Rls {
    fn new() -> Self {
        let mut p = [[0.0f64; ORDER]; ORDER];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = 1.0 / DELTA;
        }
        Self { w: [0.0; ORDER], p, hist: [0.0; ORDER] }
    }

    #[inline]
    fn predict(&self) -> f64 {
        let mut s = 0.0;
        for k in 0..ORDER {
            s += self.w[k] * self.hist[k];
        }
        s
    }

    /// RLS adaptation with the exact sample `x` and its prediction (encoder and
    /// decoder pass identical values ⇒ identical state evolution).
    fn adapt(&mut self, x: f64, pred: f64) {
        let mut px = [0.0f64; ORDER];
        for i in 0..ORDER {
            let mut s = 0.0;
            for j in 0..ORDER {
                s += self.p[i][j] * self.hist[j];
            }
            px[i] = s;
        }
        let mut denom = LAMBDA;
        for j in 0..ORDER {
            denom += self.hist[j] * px[j];
        }
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..ORDER {
            self.w[i] += px[i] * inv * e;
        }
        let ilam = 1.0 / LAMBDA;
        for i in 0..ORDER {
            let ki = px[i] * inv;
            for j in 0..ORDER {
                self.p[i][j] = (self.p[i][j] - ki * px[j]) * ilam;
            }
        }
    }

    #[inline]
    fn push(&mut self, x: f64) {
        for k in (1..ORDER).rev() {
            self.hist[k] = self.hist[k - 1];
        }
        self.hist[0] = x;
    }

    #[cfg(feature = "encode")]
    fn encode_sample(&mut self, x: i64) -> i64 {
        let pred = self.predict();
        let e = x - crate::wavelet97::round_i64(pred);
        self.adapt(x as f64, pred);
        self.push(x as f64);
        e
    }

    fn decode_sample(&mut self, e: i64) -> i64 {
        let pred = self.predict();
        let x = e + crate::wavelet97::round_i64(pred);
        self.adapt(x as f64, pred);
        self.push(x as f64);
        x
    }
}

/// RLS residual of a single integer sequence (with periodic reset) — used by the
/// lossy 9/7 path to adaptively predict the quantized indices.
#[cfg(feature = "encode")]
pub fn residual(seq: &[i64]) -> Vec<i64> {
    let mut rls = Rls::new();
    seq.iter()
        .enumerate()
        .map(|(i, &x)| {
            if i != 0 && i % RESET_PERIOD == 0 {
                rls = Rls::new();
            }
            rls.encode_sample(x)
        })
        .collect()
}

/// Reconstruct a sequence from its RLS [`residual`] (no_std). Inverse of [`residual`].
pub fn reconstruct(res: &[i64]) -> Vec<i64> {
    let mut rls = Rls::new();
    res.iter()
        .enumerate()
        .map(|(i, &e)| {
            if i != 0 && i % RESET_PERIOD == 0 {
                rls = Rls::new();
            }
            rls.decode_sample(e)
        })
        .collect()
}

/// Encode a multichannel signal with per-channel RLS prediction + entropy coding.
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(alloc::format!("rls n_ch={n_ch}")));
    }
    let t = signal[0].len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    for ch in signal {
        if ch.len() != t {
            return Err(LmlError::InvalidHeader("rls ragged channel".into()));
        }
        let mut rls = Rls::new();
        let res: Vec<i64> = ch
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i != 0 && i % RESET_PERIOD == 0 {
                    rls = Rls::new();
                }
                rls.encode_sample(x)
            })
            .collect();
        let g = crate::entropy::encode(&res)?;
        out.extend_from_slice(&(g.len() as u32).to_le_bytes());
        out.extend_from_slice(&g);
    }
    Ok(out)
}

/// Decode an RLS body. `no_std`-capable (f64 +−×÷ only).
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 6 {
        return Err(LmlError::Truncated { expected: 6, actual: body.len(), context: "rls header" });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u32::from_le_bytes([body[2], body[3], body[4], body[5]]) as usize;
    let mut pos = 6usize;
    let mut out = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        if pos + 4 > body.len() {
            return Err(LmlError::Truncated { expected: pos + 4, actual: body.len(), context: "rls ch len" });
        }
        let glen = u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if pos + glen > body.len() {
            return Err(LmlError::Truncated { expected: pos + glen, actual: body.len(), context: "rls ch data" });
        }
        let res = crate::entropy::decode(&body[pos..pos + glen])?;
        pos += glen;
        if res.len() != t {
            return Err(LmlError::InvalidHeader(alloc::format!("rls ch t={} != {t}", res.len())));
        }
        let mut rls = Rls::new();
        let ch: Vec<i64> = res
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i != 0 && i % RESET_PERIOD == 0 {
                    rls = Rls::new();
                }
                rls.decode_sample(e)
            })
            .collect();
        out.push(ch);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_sig(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        (0..n_ch)
            .map(|c| {
                (0..t)
                    .map(|i| {
                        // non-stationary: amplitude + frequency drift (the regime RLS wins)
                        let amp = 2000.0 + 1500.0 * ((i as f64 * 0.001).sin());
                        let f = 0.05 + 0.04 * ((i as f64 * 0.0007).cos());
                        (amp * ((i + c * 13) as f64 * f).sin()) as i64 + ((i * 7 + c) % 11) as i64 - 5
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn roundtrip_bit_exact() {
        for (n_ch, t) in [(1usize, 500usize), (3, 2000), (8, 777), (21, 1500)] {
            let sig = make_sig(n_ch, t);
            let body = encode(&sig).unwrap();
            assert_eq!(decode(&body).unwrap(), sig, "rls must be bit-exact ({n_ch}x{t})");
        }
    }

    #[test]
    fn constant_and_tiny() {
        for sig in [vec![vec![0i64; 100]], vec![vec![42i64; 50]], vec![vec![-7i64, 7, -7, 7]]] {
            let body = encode(&sig).unwrap();
            assert_eq!(decode(&body).unwrap(), sig);
        }
    }
}
