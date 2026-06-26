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

/// own temporal order — written into the header by the encoder; decode reads it
/// back, so this const is encode-side.
#[cfg(feature = "encode")]
const K: usize = 8;

/// Keep-best `(λ, reset, m)` adaptation configs — the chosen INDEX is signaled in
/// the header (packed into the cfg byte; see [`encode`]). `m` is the max
/// cross-channel tap count for that config (also written to the header `m` byte,
/// which decode reads directly — so decode never indexes `m` out of CONFIGS).
///
/// Index 0 is the prior slow default and index 1 is faster forgetting with a
/// tighter reset; both keep `m = 32` (the old `M` const) so they are byte-identical
/// to the pre-Lever-B format, which makes the grid never-worse. Indexes 2 through 4
/// add progressively faster forgetting with tighter resets to track the most
/// non-stationary recordings. Indexes 5 and 6 add a low-M axis: a lower
/// cross-channel order re-converges faster after each reset and overfits
/// non-stationary noise less (on recordings with at most 32 channels, `m = 32`
/// already spans every prior channel, so the only useful M move is DOWN). The
/// encoder tries each and keeps the smallest; the consts are identical on both
/// sides, hence deterministic and bit-exact.
const CONFIGS: &[(f64, usize, usize)] = &[
    (0.999, 8192, 32),
    (0.997, 4096, 32),
    (0.995, 2048, 32),
    (0.990, 1024, 32),
    (0.985, 512, 32),
    (0.990, 1024, 8),
    (0.985, 512, 4),
];

/// Number of change-point segmentation variants tried per config (Lever C). `0`
/// = the fixed periodic reset only (byte-identical to pre-Lever-C); `1` = ALSO
/// reset the RLS at signal-derived regime boundaries. Packed into the cfg byte as
/// `packed = cfg + seg * CONFIGS.len()`, so `seg = 0, cfg = 0` ⇒ packed `0` ⇒
/// byte-identical to the pre-Lever-B/C wire. The detector is causal over the
/// losslessly-exact reconstructed samples ⇒ decode recomputes identical reset
/// points with NO per-boundary side info.
const SEG_VARIANTS: usize = 2;

/// Variable-order RLS (order = K + #cross-channel taps for the channel).
struct Rls {
    n: usize,
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
    lambda: f64,
}

impl Rls {
    fn new(n: usize, lambda: f64) -> Self {
        let mut p = alloc::vec![alloc::vec![0.0f64; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        Self { n, w: alloc::vec![0.0; n], p, lambda }
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
        let mut denom = self.lambda;
        for j in 0..n {
            denom += reg[j] * px[j];
        }
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..n {
            self.w[i] += px[i] * inv * e;
        }
        let ilam = 1.0 / self.lambda;
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

/// Encode under one `(λ, reset, m)` config (index `cfg`) and segmentation mode
/// `seg` (0 = fixed periodic reset only; 1 = ALSO reset at change-points). Header:
/// `[n_ch u16][t u32][k u8][m u8][packed u8]` where `packed = cfg + seg·CONFIGS.len()`.
/// The per-config `m` is written to the header `m` byte (decode reads it there,
/// never indexing CONFIGS for `m`).
#[cfg(feature = "encode")]
fn encode_one(signal: &[Vec<i64>], cfg: usize, seg: usize) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let (lambda, reset, m) = CONFIGS[cfg];
    let packed = cfg + seg * CONFIGS.len();
    let mut out = Vec::new();
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    out.push(K as u8);
    out.push(m as u8);
    out.push(packed as u8);

    for c in 0..n_ch {
        if signal[c].len() != t {
            return Err(LmlError::InvalidHeader("mv_rls ragged".into()));
        }
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order, lambda);
        let mut own = alloc::vec![0.0f64; K];
        let mut det = crate::segmentation::ChangePoint::new();
        let mut cp_next = false;
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            // `cp_next` was decided from samples 0..n-1 (causal) on the prior
            // iteration, so the decoder — which has the exact samples 0..n-1 — can
            // reproduce it BEFORE it needs to predict sample n.
            if n != 0 && (n % reset == 0 || cp_next) {
                rls = Rls::new(order, lambda);
            }
            let reg = regressor(&own, signal, &refs, n);
            let pred = rls.predict(&reg);
            res.push(signal[c][n] - round_i64(pred));
            rls.adapt(&reg, signal[c][n] as f64, pred);
            cp_next = seg != 0 && det.observe(signal[c][n]);
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

/// Encode keeping the smallest over the `(λ, reset, m)` config grid × the
/// segmentation on/off axis (never-worse: `cfg = 0, seg = 0` is always tried and
/// is byte-identical to the pre-Lever-B/C format).
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader("mv_rls n_ch".into()));
    }
    let mut best: Option<Vec<u8>> = None;
    for seg in 0..SEG_VARIANTS {
        for cfg in 0..CONFIGS.len() {
            if let Ok(b) = encode_one(signal, cfg, seg) {
                if best.as_ref().map_or(true, |bb| b.len() < bb.len()) {
                    best = Some(b);
                }
            }
        }
    }
    best.ok_or(LmlError::InvalidHeader("mv_rls: no config encoded".into()))
}

/// Decode. `no_std`-capable.
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 9 {
        return Err(LmlError::Truncated { expected: 9, actual: body.len(), context: "mv_rls header" });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u32::from_le_bytes([body[2], body[3], body[4], body[5]]) as usize;
    let k = body[6] as usize;
    let m = body[7] as usize;
    let packed = body[8] as usize;
    let cfg = packed % CONFIGS.len();
    let seg = packed / CONFIGS.len();
    if k == 0 || seg >= SEG_VARIANTS {
        return Err(LmlError::InvalidHeader("mv_rls bad params".into()));
    }
    // `m` comes from the header byte (encode wrote CONFIGS[cfg].2 there), NOT from
    // indexing CONFIGS — keeps decode robust to the per-config M axis.
    let (lambda, reset, _m_cfg) = CONFIGS[cfg];
    let mut pos = 9usize;
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
        let mut rls = Rls::new(order, lambda);
        let mut own = alloc::vec![0.0f64; k];
        let mut det = crate::segmentation::ChangePoint::new();
        let mut cp_next = false;
        let mut ch = Vec::with_capacity(t);
        for n in 0..t {
            // `cp_next` was decided from the reconstructed samples 0..n-1 on the
            // prior iteration — identical to the encoder's causal decision.
            if n != 0 && (n % reset == 0 || cp_next) {
                rls = Rls::new(order, lambda);
            }
            let reg = regressor(&own, &out, &refs, n);
            let pred = rls.predict(&reg);
            let x = res[n] + round_i64(pred);
            ch.push(x);
            rls.adapt(&reg, x as f64, pred);
            cp_next = seg != 0 && det.observe(x);
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
