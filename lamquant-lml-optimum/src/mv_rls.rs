//! Multivariate (cross-channel) **RLS** lossless codec (ADR 0054).
//!
//! The 2024 multichannel-EEG SOTA predicts each channel from its own past AND the
//! other channels jointly. Causal/decodable: channels are coded in order, so
//! channel `c`'s regressor is `[own K past samples] + [the M most-recent prior
//! channels c−1..c−M at the SAME instant n]` — same-instant spatial correlation
//! (volume conduction) + temporal history, adapted by one RLS. This subsumes our
//! separate cross-channel prediction + per-channel RLS into one predictor; it
//! wins on hard non-stationary high-amplitude EEG (the `ma` 21ch case) where the
//! static best-of-prior path collapses.
//!
//! Integer-exact like [`crate::rls`] (f64 `+−×÷` only ⇒ no_std-decodable,
//! deterministic). Periodic reset bounds RLS long-run divergence. Decode runs the
//! identical recursion on the losslessly-exact prior channels + own history.
//!
//! Wire: `[n_ch u16][t u32][k u8][m u8][reset u32]` then per channel
//! `[golomb_len u32][golomb bytes]`.
#![allow(clippy::needless_range_loop)] // index loops are natural for the RLS matrix

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};

use crate::wavelet97::round_i64;

/// own temporal order / max cross-channel taps / reset — written into the header
/// by the encoder; decode reads them back, so these consts are encode-side.
#[cfg(feature = "encode")]
const K: usize = 8;
#[cfg(feature = "encode")]
const M: usize = 32;
const LAMBDA: f64 = 0.999;
#[cfg(feature = "encode")]
const RESET: usize = 8192;

/// Variable-order RLS (order = K + #cross-channel taps for the channel).
struct Rls {
    n: usize,
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
}

impl Rls {
    fn new(n: usize) -> Self {
        let mut p = alloc::vec![alloc::vec![0.0f64; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        Self { n, w: alloc::vec![0.0; n], p }
    }

    fn predict(&self, reg: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in 0..self.n {
            s += self.w[k] * reg[k];
        }
        s
    }

    fn adapt(&mut self, reg: &[f64], x: f64, pred: f64) {
        let n = self.n;
        let mut px = alloc::vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.p[i][j] * reg[j];
            }
            px[i] = s;
        }
        let mut denom = LAMBDA;
        for j in 0..n {
            denom += reg[j] * px[j];
        }
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..n {
            self.w[i] += px[i] * inv * e;
        }
        let ilam = 1.0 / LAMBDA;
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n {
                self.p[i][j] = (self.p[i][j] - ki * px[j]) * ilam;
            }
        }
    }
}

/// Build channel `c`'s regressor at time `n` from its own history + prior
/// channels' current samples. `prior[j]` is the (decoded/original) channel j.
#[inline]
fn regressor(own: &[f64], prior: &[Vec<i64>], refs: &[usize], n: usize) -> Vec<f64> {
    let mut reg = alloc::vec![0.0f64; own.len() + refs.len()];
    reg[..own.len()].copy_from_slice(own);
    for (i, &j) in refs.iter().enumerate() {
        reg[own.len() + i] = prior[j][n] as f64;
    }
    reg
}

#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader("mv_rls n_ch".into()));
    }
    let t = signal[0].len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    out.push(K as u8);
    out.push(M as u8);
    out.extend_from_slice(&(RESET as u32).to_le_bytes());

    for c in 0..n_ch {
        if signal[c].len() != t {
            return Err(LmlError::InvalidHeader("mv_rls ragged".into()));
        }
        let xref = c.min(M);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order);
        let mut own = alloc::vec![0.0f64; K];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % RESET == 0 {
                rls = Rls::new(order);
            }
            let reg = regressor(&own, signal, &refs, n);
            let pred = rls.predict(&reg);
            res.push(signal[c][n] - round_i64(pred));
            rls.adapt(&reg, signal[c][n] as f64, pred);
            for q in (1..K).rev() {
                own[q] = own[q - 1];
            }
            own[0] = signal[c][n] as f64;
        }
        let g = crate::entropy::encode(&res)?;
        out.extend_from_slice(&(g.len() as u32).to_le_bytes());
        out.extend_from_slice(&g);
    }
    Ok(out)
}

/// Decode. `no_std`-capable.
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 12 {
        return Err(LmlError::Truncated { expected: 12, actual: body.len(), context: "mv_rls header" });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u32::from_le_bytes([body[2], body[3], body[4], body[5]]) as usize;
    let k = body[6] as usize;
    let m = body[7] as usize;
    let reset = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    if k == 0 || reset == 0 {
        return Err(LmlError::InvalidHeader("mv_rls bad params".into()));
    }
    let mut pos = 12usize;
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        if pos + 4 > body.len() {
            return Err(LmlError::Truncated { expected: pos + 4, actual: body.len(), context: "mv_rls ch len" });
        }
        let glen = u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if pos + glen > body.len() {
            return Err(LmlError::Truncated { expected: pos + glen, actual: body.len(), context: "mv_rls ch data" });
        }
        let res = crate::entropy::decode(&body[pos..pos + glen])?;
        pos += glen;
        if res.len() != t {
            return Err(LmlError::InvalidHeader("mv_rls ch t".into()));
        }
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = k + xref;
        let mut rls = Rls::new(order);
        let mut own = alloc::vec![0.0f64; k];
        let mut ch = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % reset == 0 {
                rls = Rls::new(order);
            }
            let reg = regressor(&own, &out, &refs, n);
            let pred = rls.predict(&reg);
            let x = res[n] + round_i64(pred);
            ch.push(x);
            rls.adapt(&reg, x as f64, pred);
            for q in (1..k).rev() {
                own[q] = own[q - 1];
            }
            own[0] = x as f64;
        }
        out.push(ch);
    }
    Ok(out)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    fn make_sig(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        // shared base + per-channel detail ⇒ cross-channel + temporal structure
        let base: Vec<i64> = (0..t).map(|i| ((i as f64 * 0.03).sin() * 2500.0) as i64).collect();
        (0..n_ch)
            .map(|c| {
                (0..t)
                    .map(|i| {
                        let g = 0.5 + 0.07 * c as f64;
                        (g * base[i] as f64) as i64 + ((i + c * 5) as f64 * 0.7).sin() as i64 * 90 + ((i * 3 + c) % 9) as i64 - 4
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn roundtrip_bit_exact() {
        for (n_ch, t) in [(1usize, 400usize), (4, 2000), (21, 1500), (8, 17000)] {
            let sig = make_sig(n_ch, t);
            let body = encode(&sig).unwrap();
            assert_eq!(decode(&body).unwrap(), sig, "mv_rls bit-exact ({n_ch}x{t})");
        }
    }
}
