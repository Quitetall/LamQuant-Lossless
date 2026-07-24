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
        Self::new_p0(n, lambda, 1.0)
    }

    /// `p0` = the initial `P` diagonal = `1/ridge` (a weak prior ⇒ large `p0`). Setting it to
    /// match the batch hindsight's ridge makes `λ=1` RLS the EXACT VAW online-ridge forecaster
    /// with the tight `(d/2)log T` regret (research A.5); `new` keeps the shipped `p0=1`.
    fn new_p0(n: usize, lambda: f64, p0: f64) -> Self {
        let mut p = alloc::vec![alloc::vec![0.0f64; n]; n];
        for i in 0..n {
            p[i][i] = p0;
        }
        Self {
            n,
            w: alloc::vec![0.0; n],
            p,
            lambda,
        }
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
use rayon::prelude::*;

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

    // Per-channel encode runs across rayon workers — each channel reads `signal`
    // immutably and produces its own `[len u32][entropy bytes]` chunk; collecting
    // an ORDERED Vec preserves the serial layout ⇒ byte-identical output. This is
    // the embarrassingly-parallel keep-best the Optimum tier always should have had
    // (mirrors the Desktop tier's per-channel rayon path).
    let chunks: LmlResult<Vec<Vec<u8>>> = (0..n_ch)
        .into_par_iter()
        .map(|c| {
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
            let mut chunk = Vec::with_capacity(4 + g.len());
            chunk.extend_from_slice(&(g.len() as u32).to_le_bytes());
            chunk.extend_from_slice(&g);
            Ok(chunk)
        })
        .collect();
    for chunk in chunks? {
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// RESEARCH (E-A1): the per-channel MV-RLS residual for one `(cfg, seg)` — the
/// signal the container actually entropy-codes for the win regime. Used to probe
/// whether the SHIPPED residual retains structure a (nonlinear) predictor could
/// exploit, instead of the per-channel RLS residual the order-1 probe wrongly used.
#[cfg(feature = "encode")]
pub fn residuals(signal: &[Vec<i64>], cfg: usize, seg: usize) -> Vec<Vec<i64>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let (lambda, reset, m) = CONFIGS[cfg];
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order, lambda);
        let mut own = alloc::vec![0.0f64; K];
        let mut det = crate::segmentation::ChangePoint::new();
        let mut cp_next = false;
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
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
        out.push(res);
    }
    out
}

/// RESEARCH (A.5 regret harness): the MV-RLS residual under ARBITRARY `(λ, reset, m,
/// seg)` — like [`encode_len_params`] but RETURNS the residual, so the harness can code
/// it with a fixed coder and measure predictor redundancy at λ=1 (static — the pure
/// VAW/RLS setting Theorem A1 bounds) and λ<1 (adaptive/tracking, A2/A3). NOT a wire path.
#[cfg(feature = "encode")]
pub fn residuals_params(
    signal: &[Vec<i64>],
    lambda: f64,
    reset: usize,
    m: usize,
    seg: usize,
) -> Vec<Vec<i64>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order, lambda);
        let mut own = alloc::vec![0.0f64; K];
        let mut det = crate::segmentation::ChangePoint::new();
        let mut cp_next = false;
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && ((reset != 0 && n % reset == 0) || cp_next) {
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
        out.push(res);
    }
    out
}

/// RESEARCH (A.5 regret harness): the BEST FIXED `d`-tap linear predictor in HINDSIGHT
/// — the regret-bound comparator `min_w Σ (x − w·φ)²`. Per channel, a batch
/// least-squares fit (ridge `ridge`) over the WHOLE recording (the same `[own K] + [M
/// cross-channel]` features MV-RLS uses), then the residual under that single fixed `w`.
/// An ORACLE: the fit is non-causal and its `w` side-info is NOT counted — the `(d/2)log T`
/// the online predictor pays is exactly this parameter cost the oracle skips. NOT a wire path.
#[cfg(feature = "encode")]
pub fn residuals_hindsight(signal: &[Vec<i64>], m: usize, ridge: f64) -> Vec<Vec<i64>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        // Pass 1: accumulate the normal equations A = Σ φφᵀ (+ ridge), b = Σ φ·x.
        let mut a = alloc::vec![alloc::vec![0.0f64; order]; order];
        let mut b = alloc::vec![0.0f64; order];
        let mut own = alloc::vec![0.0f64; K];
        for n in 0..t {
            let reg = regressor(&own, signal, &refs, n);
            let x = signal[c][n] as f64;
            for i in 0..order {
                b[i] += reg[i] * x;
                for j in 0..order {
                    a[i][j] += reg[i] * reg[j];
                }
            }
            for q in (1..K).rev() {
                own[q] = own[q - 1];
            }
            own[0] = x;
        }
        for i in 0..order {
            a[i][i] += ridge;
        }
        let w = crate::crosschan::solve_spd_cholesky(&a, &b)
            .unwrap_or_else(|| alloc::vec![0.0f64; order]);
        // Pass 2: residual under the single fixed w.
        let mut own = alloc::vec![0.0f64; K];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            let reg = regressor(&own, signal, &refs, n);
            let mut pred = 0.0f64;
            for k in 0..order {
                pred += w[k] * reg[k];
            }
            res.push(signal[c][n] - round_i64(pred));
            for q in (1..K).rev() {
                own[q] = own[q - 1];
            }
            own[0] = signal[c][n] as f64;
        }
        out.push(res);
    }
    out
}

/// RESEARCH (A.5 regret harness): the EXACT VAW / online-ridge forecaster = `λ=1` RLS with
/// `P₀=(1/ridge)·I` matched to `residuals_hindsight`'s ridge. This is the algorithm whose
/// online-vs-batch regret Theorem A1 bounds by `(d/2)log T` — the honest A1 measurement.
/// (`residuals_growing_ls` is a block-refit heuristic that is NOT VAW-tight.) NOT a wire path.
#[cfg(feature = "encode")]
pub fn residuals_vaw(signal: &[Vec<i64>], m: usize, ridge: f64) -> Vec<Vec<i64>> {
    assert!(
        ridge > 0.0,
        "residuals_vaw needs ridge > 0 (P₀ = 1/ridge; ridge=0 ⇒ +inf ⇒ NaN)"
    );
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new_p0(order, 1.0, 1.0 / ridge);
        let mut own = alloc::vec![0.0f64; K];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            let reg = regressor(&own, signal, &refs, n);
            let pred = rls.predict(&reg);
            res.push(signal[c][n] - round_i64(pred));
            rls.adapt(&reg, signal[c][n] as f64, pred);
            for q in (1..K).rev() {
                own[q] = own[q - 1];
            }
            own[0] = signal[c][n] as f64;
        }
        out.push(res);
    }
    out
}

/// RESEARCH (A.5 regret harness): the numerically-STABLE online least-squares forecaster
/// — the honest A1 comparator (the `λ=1` RLS recursion degrades over long `T`; this does
/// not). At each `block` boundary it re-solves the ridge-LS from the accumulated `[0..n)`
/// normal equations (fresh Cholesky, no accumulated recursion error, ridge matched to the
/// hindsight) and predicts the next block with that causal fit. Its excess codelength over
/// `residuals_hindsight` (same ridge) is the online-vs-batch redundancy Theorem A1 bounds by
/// `(d/2)log T`. NOT a wire path.
#[cfg(feature = "encode")]
pub fn residuals_growing_ls(
    signal: &[Vec<i64>],
    m: usize,
    block: usize,
    ridge: f64,
) -> Vec<Vec<i64>> {
    assert!(
        block > 0,
        "residuals_growing_ls needs block > 0 (n % block)"
    );
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut out = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut a = alloc::vec![alloc::vec![0.0f64; order]; order]; // Σ φφᵀ over [0..n)
        let mut b = alloc::vec![0.0f64; order]; // Σ φ·x over [0..n)
        let mut w = alloc::vec![0.0f64; order]; // current causal fit (0 until first solve)
        let mut own = alloc::vec![0.0f64; K];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % block == 0 {
                let mut a2 = a.clone();
                for i in 0..order {
                    a2[i][i] += ridge;
                }
                if let Some(sol) = crate::crosschan::solve_spd_cholesky(&a2, &b) {
                    w = sol;
                }
            }
            let reg = regressor(&own, signal, &refs, n);
            let mut pred = 0.0f64;
            for k in 0..order {
                pred += w[k] * reg[k];
            }
            res.push(signal[c][n] - round_i64(pred));
            // accumulate n INTO [0..n+1) only AFTER predicting it (keeps the fit causal)
            let x = signal[c][n] as f64;
            for i in 0..order {
                b[i] += reg[i] * x;
                for j in 0..order {
                    a[i][j] += reg[i] * reg[j];
                }
            }
            for q in (1..K).rev() {
                own[q] = own[q - 1];
            }
            own[0] = x;
        }
        out.push(res);
    }
    out
}

/// RESEARCH (A.5-iii): predictor-COEFFICIENT drift — the non-stationarity measure that should
/// actually predict the tracking win (amplitude-energy drift did NOT). Per channel, fit a batch
/// ridge-LS predictor on each `window`-sample block (own history seeded causally from the block
/// start), then average the RELATIVE L2 coefficient change `‖w_k − w_{k-1}‖ / ‖w_k‖` across
/// blocks + channels. High ⇒ the best fixed predictor drifts ⇒ adaptivity pays. NOT a wire path.
#[cfg(feature = "encode")]
pub fn coeff_drift(signal: &[Vec<i64>], m: usize, window: usize, ridge: f64) -> f64 {
    assert!(window > 0, "coeff_drift needs window > 0");
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    if t < 2 * window {
        return 0.0;
    }
    let mut acc = 0.0f64;
    let mut cnt = 0usize;
    for c in 0..n_ch {
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let nb = t / window;
        let mut prev: Option<Vec<f64>> = None;
        for k in 0..nb {
            let lo = k * window;
            let hi = lo + window;
            let mut a = alloc::vec![alloc::vec![0.0f64; order]; order];
            let mut b = alloc::vec![0.0f64; order];
            let mut own = alloc::vec![0.0f64; K];
            for j in 0..K {
                let idx = lo as isize - 1 - j as isize;
                own[j] = if idx >= 0 {
                    signal[c][idx as usize] as f64
                } else {
                    0.0
                };
            }
            for n in lo..hi {
                let reg = regressor(&own, signal, &refs, n);
                let x = signal[c][n] as f64;
                for i in 0..order {
                    b[i] += reg[i] * x;
                    for jj in 0..order {
                        a[i][jj] += reg[i] * reg[jj];
                    }
                }
                for q in (1..K).rev() {
                    own[q] = own[q - 1];
                }
                own[0] = x;
            }
            for i in 0..order {
                a[i][i] += ridge;
            }
            let w = crate::crosschan::solve_spd_cholesky(&a, &b)
                .unwrap_or_else(|| alloc::vec![0.0f64; order]);
            if let Some(pw) = &prev {
                let mut dn = 0.0f64;
                let mut wn = 0.0f64;
                for i in 0..order {
                    let d = w[i] - pw[i];
                    dn += d * d;
                    wn += w[i] * w[i];
                }
                acc += dn.sqrt() / (wn.sqrt() + 1e-9);
                cnt += 1;
            }
            prev = Some(w);
        }
    }
    if cnt > 0 {
        acc / cnt as f64
    } else {
        0.0
    }
}

/// RESEARCH (TUH tuning): the exact encoded BYTE LENGTH under ARBITRARY
/// `(λ, reset, m, seg)` params — the search primitive for finding new configs to
/// ADD to `CONFIGS` (keep-best ⇒ never-worse). Mirrors `encode_one`'s body byte
/// count exactly (9-byte header + per-channel `4 + entropy(res).len()`), so a param
/// tuple that wins here is a real container candidate once added to the grid.
#[cfg(feature = "encode")]
pub fn encode_len_params(
    signal: &[Vec<i64>],
    lambda: f64,
    reset: usize,
    m: usize,
    seg: usize,
) -> usize {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut total = 9; // [n_ch u16][t u32][k u8][m u8][packed u8]
    for c in 0..n_ch {
        if signal[c].len() != t {
            return usize::MAX; // ragged — disqualify
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
        match crate::entropy::encode(&res) {
            Ok(g) => total += 4 + g.len(),
            Err(_) => return usize::MAX,
        }
    }
    total
}

/// Encode-only search set: the `(cfg, seg)` variants that EVER win the internal keep-best,
/// measured across the full 40-recording corpus (`grid_winners_broad`). The pruned variants —
/// the entire `seg=1` axis and `cfg6` — never won, so searching only these is byte-identical
/// to the full 14-variant grid at ~2.3× lower encode cost. `CONFIGS`/`SEG_VARIANTS` are kept
/// for DECODE (the packed-config wire byte `cfg + seg·CONFIGS.len()` is unchanged).
#[cfg(feature = "encode")]
const SEARCH_SET: &[(usize, usize)] = &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)];

/// Encode keeping the smallest over the `SEARCH_SET` (the winning `(cfg, seg)` variants).
/// Never-worse: `cfg = 0, seg = 0` is always tried and is byte-identical to the
/// pre-Lever-B/C format.
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader("mv_rls n_ch".into()));
    }
    // Run the config search across rayon workers too (nested with the per-channel
    // parallelism inside `encode_one` — rayon's work-stealing pool flattens
    // 6 configs × n_ch channels into one task set, saturating all cores). Collect
    // in SEARCH_SET order, then pick the smallest with first-wins ties (lowest
    // index) — identical to the serial keep-best ⇒ byte-identical output.
    let cands: Vec<Vec<u8>> = SEARCH_SET
        .par_iter()
        .filter_map(|&(cfg, seg)| encode_one(signal, cfg, seg).ok())
        .collect();
    let mut best: Option<Vec<u8>> = None;
    for b in cands {
        if best.as_ref().map_or(true, |bb| b.len() < bb.len()) {
            best = Some(b);
        }
    }
    best.ok_or(LmlError::InvalidHeader("mv_rls: no config encoded".into()))
}

/// Decode. `no_std`-capable.
pub fn decode(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 9 {
        return Err(LmlError::Truncated {
            expected: 9,
            actual: body.len(),
            context: "mv_rls header",
        });
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
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: body.len(),
                context: "mv_rls ch len",
            });
        }
        let glen =
            u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if pos + glen > body.len() {
            return Err(LmlError::Truncated {
                expected: pos + glen,
                actual: body.len(),
                context: "mv_rls ch data",
            });
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

/// Bias-cancel contexts for `CODER_MV_RLS_BC`. A per-channel byte holds 0 (off) or 1..=4
/// (index+1 into this table). Shared by encode + decode (not encode-gated).
const BC_CTXS: [usize; 4] = [8, 16, 32, 64];

/// `CODER_MV_RLS_BC` encode under one config: the SAME predictor + header as [`encode_one`]
/// (residual via [`residuals_params`], bit-identical), but each channel's residual is coded
/// keep-best over {plain, `bias_cancel`@ctx}, prefixed by a `[bc u8]` (0 = off, else BC_CTXS
/// index+1). A SEPARATE coder ⇒ `CODER_MV_RLS` stays byte-identical; the container's coder
/// keep-best keeps the whole thing never-worse. `encode_one` is untouched.
#[cfg(feature = "encode")]
fn encode_one_bc(signal: &[Vec<i64>], cfg: usize, seg: usize) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = signal[0].len();
    let (lambda, reset, m) = CONFIGS[cfg];
    let packed = cfg + seg * CONFIGS.len();
    let residuals = residuals_params(signal, lambda, reset, m, seg);
    let mut out = Vec::new();
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    out.push(K as u8);
    out.push(m as u8);
    out.push(packed as u8);
    for c in 0..n_ch {
        if signal[c].len() != t {
            return Err(LmlError::InvalidHeader("mv_rls_bc ragged".into()));
        }
        let res = &residuals[c];
        let mut best = crate::entropy::encode(res)?;
        let mut bc = 0u8;
        for (i, &ctx) in BC_CTXS.iter().enumerate() {
            let mut rc = res.clone();
            lamquant_lml_mcu::lpc::bias_cancel(&mut rc, ctx);
            if let Ok(g) = crate::entropy::encode(&rc) {
                if g.len() < best.len() {
                    best = g;
                    bc = (i + 1) as u8;
                }
            }
        }
        out.push(bc);
        out.extend_from_slice(&(best.len() as u32).to_le_bytes());
        out.extend_from_slice(&best);
    }
    Ok(out)
}

/// `CODER_MV_RLS_BC` encode: keep-best over the config `SEARCH_SET` (mirrors [`encode`]).
#[cfg(feature = "encode")]
pub fn encode_bc(signal: &[Vec<i64>]) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader("mv_rls_bc n_ch".into()));
    }
    let cands: Vec<Vec<u8>> = SEARCH_SET
        .par_iter()
        .filter_map(|&(cfg, seg)| encode_one_bc(signal, cfg, seg).ok())
        .collect();
    let mut best: Option<Vec<u8>> = None;
    for b in cands {
        if best.as_ref().map_or(true, |bb| b.len() < bb.len()) {
            best = Some(b);
        }
    }
    best.ok_or(LmlError::InvalidHeader(
        "mv_rls_bc: no config encoded".into(),
    ))
}

/// `CODER_MV_RLS_BC` decode. Reads the per-channel `[bc u8]`, `bias_restore`s the residual, then
/// runs the identical RLS synthesis as [`decode`]. `no_std`-capable.
pub fn decode_bc(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 9 {
        return Err(LmlError::Truncated {
            expected: 9,
            actual: body.len(),
            context: "mv_rls_bc header",
        });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u32::from_le_bytes([body[2], body[3], body[4], body[5]]) as usize;
    let k = body[6] as usize;
    let m = body[7] as usize;
    let packed = body[8] as usize;
    let cfg = packed % CONFIGS.len();
    let seg = packed / CONFIGS.len();
    if k == 0 || seg >= SEG_VARIANTS {
        return Err(LmlError::InvalidHeader("mv_rls_bc bad params".into()));
    }
    let (lambda, reset, _m_cfg) = CONFIGS[cfg];
    let mut pos = 9usize;
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        if pos + 5 > body.len() {
            return Err(LmlError::Truncated {
                expected: pos + 5,
                actual: body.len(),
                context: "mv_rls_bc ch hdr",
            });
        }
        let bc = body[pos];
        pos += 1;
        let glen =
            u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if pos + glen > body.len() {
            return Err(LmlError::Truncated {
                expected: pos + glen,
                actual: body.len(),
                context: "mv_rls_bc ch data",
            });
        }
        let mut res = crate::entropy::decode(&body[pos..pos + glen])?;
        pos += glen;
        if res.len() != t {
            return Err(LmlError::InvalidHeader("mv_rls_bc ch t".into()));
        }
        if bc != 0 {
            let idx = (bc - 1) as usize;
            if idx >= BC_CTXS.len() {
                return Err(LmlError::InvalidHeader("mv_rls_bc bad ctx".into()));
            }
            lamquant_lml_mcu::lpc::bias_restore(&mut res, BC_CTXS[idx]);
        }
        // identical RLS synthesis to `decode`
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = k + xref;
        let mut rls = Rls::new(order, lambda);
        let mut own = alloc::vec![0.0f64; k];
        let mut det = crate::segmentation::ChangePoint::new();
        let mut cp_next = false;
        let mut ch = Vec::with_capacity(t);
        for n in 0..t {
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

// ── mv_rls guaranteed-δ near-lossless (closed-loop DPCM) ──────────────────────────────
// The one regime that STRUCTURALLY beats H.BWC: H.BWC has no per-sample error bound (its TCQ
// optimizes RMS ⇒ outliers force it near-lossless to bound max|err|). Ours bounds every sample
// by construction: quantize the prediction residual to the (2δ+1)-grid and reconstruct on that
// grid. CLOSED-LOOP — the RLS predicts from RECONSTRUCTED samples (own + cross-channel), so
// encoder and decoder stay in lock-step and the error never accumulates. Fixed config (the one
// the near_lossless_mvrls_probe measured as the win): λ=0.999, reset=8192, m=32.
const NL_LAMBDA: f64 = 0.999;
const NL_RESET: usize = 8192;
const NL_M: usize = 32;

/// Encode `signal` near-lossless with a HARD per-sample bound `|x − x̂| ≤ delta`. Closed-loop
/// mv_rls DPCM: residual `r = x − round(pred)`, `q = round(r/(2δ+1))`, `x̂ = round(pred) + (2δ+1)·q`
/// (⇒ `|x − x̂| ≤ δ`); the RLS adapts on `x̂`. Codes `q` losslessly. Wire: `[n_ch u16][t u32]
/// [k u8][m u8][delta u32]` then per channel `[glen u32][entropy(q)]`.
#[cfg(feature = "encode")]
pub fn encode_bounded_mae(signal: &[Vec<i64>], delta: i64) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    if n_ch == 0 || n_ch > u16::MAX as usize {
        return Err(LmlError::InvalidHeader("mv_rls_nl n_ch".into()));
    }
    if delta < 0 {
        return Err(LmlError::InvalidHeader("mv_rls_nl delta < 0".into()));
    }
    if delta > u32::MAX as i64 {
        return Err(LmlError::InvalidHeader(
            "mv_rls_nl delta exceeds u32".into(),
        ));
    }
    let t = signal[0].len();
    let grid = 2 * delta + 1;
    let mut out = Vec::new();
    out.extend_from_slice(&(n_ch as u16).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    out.push(K as u8);
    out.push(NL_M as u8);
    out.extend_from_slice(&(delta as u32).to_le_bytes());
    let mut xhat: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        if signal[c].len() != t {
            return Err(LmlError::InvalidHeader("mv_rls_nl ragged".into()));
        }
        let xref = c.min(NL_M);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order, NL_LAMBDA);
        let mut own = alloc::vec![0.0f64; K];
        let mut q_res = Vec::with_capacity(t);
        let mut rec = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % NL_RESET == 0 {
                rls = Rls::new(order, NL_LAMBDA);
            }
            let reg = regressor(&own, &xhat, &refs, n);
            let pred = rls.predict(&reg);
            let pr = round_i64(pred);
            let r = signal[c][n] - pr;
            // integer round-half-away-from-zero (grid=2δ+1 is odd ⇒ ⌊grid/2⌋=δ): makes the
            // guarantee |r − grid·q| ≤ δ EXACT for every i64 r, with no f64-precision dependency.
            let q = if r >= 0 {
                (r + delta) / grid
            } else {
                -((-r + delta) / grid)
            };
            let xh = pr + grid * q;
            q_res.push(q);
            rec.push(xh);
            rls.adapt(&reg, xh as f64, pred);
            for k in (1..K).rev() {
                own[k] = own[k - 1];
            }
            own[0] = xh as f64;
        }
        let g = crate::entropy::encode(&q_res)?;
        out.extend_from_slice(&(g.len() as u32).to_le_bytes());
        out.extend_from_slice(&g);
        xhat.push(rec);
    }
    Ok(out)
}

/// Decode a [`encode_bounded_mae`] stream. Replays the identical closed loop: `x̂ = round(pred)
/// + (2δ+1)·q`, RLS adapts on `x̂`. `no_std`-capable. Header is 12 bytes.
pub fn decode_bounded_mae(body: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    if body.len() < 12 {
        return Err(LmlError::Truncated {
            expected: 12,
            actual: body.len(),
            context: "mv_rls_nl header",
        });
    }
    let n_ch = u16::from_le_bytes([body[0], body[1]]) as usize;
    let t = u32::from_le_bytes([body[2], body[3], body[4], body[5]]) as usize;
    let k = body[6] as usize;
    let m = body[7] as usize;
    let delta = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as i64;
    if k == 0 {
        return Err(LmlError::InvalidHeader("mv_rls_nl bad k".into()));
    }
    let grid = 2 * delta + 1;
    let mut pos = 12usize;
    let mut xhat: Vec<Vec<i64>> = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        if pos + 4 > body.len() {
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: body.len(),
                context: "mv_rls_nl ch len",
            });
        }
        let glen =
            u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;
        if pos + glen > body.len() {
            return Err(LmlError::Truncated {
                expected: pos + glen,
                actual: body.len(),
                context: "mv_rls_nl ch data",
            });
        }
        let q_res = crate::entropy::decode(&body[pos..pos + glen])?;
        pos += glen;
        if q_res.len() != t {
            return Err(LmlError::InvalidHeader("mv_rls_nl ch t".into()));
        }
        let xref = c.min(m);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = k + xref;
        let mut rls = Rls::new(order, NL_LAMBDA);
        let mut own = alloc::vec![0.0f64; k];
        let mut rec = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % NL_RESET == 0 {
                rls = Rls::new(order, NL_LAMBDA);
            }
            let reg = regressor(&own, &xhat, &refs, n);
            let pred = rls.predict(&reg);
            let pr = round_i64(pred);
            let xh = pr + grid * q_res[n];
            rec.push(xh);
            rls.adapt(&reg, xh as f64, pred);
            for q in (1..k).rev() {
                own[q] = own[q - 1];
            }
            own[0] = xh as f64;
        }
        xhat.push(rec);
    }
    Ok(xhat)
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    fn make_sig(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        // shared base + per-channel detail ⇒ cross-channel + temporal structure
        let base: Vec<i64> = (0..t)
            .map(|i| ((i as f64 * 0.03).sin() * 2500.0) as i64)
            .collect();
        (0..n_ch)
            .map(|c| {
                (0..t)
                    .map(|i| {
                        let g = 0.5 + 0.07 * c as f64;
                        (g * base[i] as f64) as i64
                            + ((i + c * 5) as f64 * 0.7).sin() as i64 * 90
                            + ((i * 3 + c) % 9) as i64
                            - 4
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

    #[test]
    fn bc_roundtrip_bit_exact() {
        // CODER_MV_RLS_BC (bias_cancel keep-best) must round-trip bit-exact for every ctx choice.
        for (n_ch, t) in [(1usize, 400usize), (4, 2000), (21, 1500), (8, 17000)] {
            let sig = make_sig(n_ch, t);
            let body = encode_bc(&sig).unwrap();
            assert_eq!(
                decode_bc(&body).unwrap(),
                sig,
                "mv_rls_bc bit-exact ({n_ch}x{t})"
            );
        }
    }

    #[test]
    fn bc_never_worse_than_plain() {
        // Keep-best over {plain, bias_cancel@ctx} ⇒ encode_bc ≤ encode per channel; the config
        // search can only help, so the total is ≤ plain mv_rls (the container then keep-bests too).
        for (n_ch, t) in [(4usize, 4000usize), (21, 3000)] {
            let sig = make_sig(n_ch, t);
            assert!(encode_bc(&sig).unwrap().len() <= encode(&sig).unwrap().len() + n_ch);
        }
    }

    #[test]
    fn nl_bounded_mae_guarantee_and_roundtrip() {
        // The HARD per-sample bound |x − x̂| ≤ δ must hold through the wire, and δ=0 must be
        // bit-exact. This is the structural guarantee H.BWC cannot make.
        for &delta in &[0i64, 1, 4, 16] {
            for (n_ch, t) in [(1usize, 400usize), (8, 5000), (21, 3000)] {
                let sig = make_sig(n_ch, t);
                let body = encode_bounded_mae(&sig, delta).unwrap();
                let dec = decode_bounded_mae(&body).unwrap();
                assert_eq!(dec.len(), n_ch);
                let mut maxerr = 0i64;
                for c in 0..n_ch {
                    assert_eq!(dec[c].len(), t);
                    for n in 0..t {
                        maxerr = maxerr.max((sig[c][n] - dec[c][n]).abs());
                    }
                }
                assert!(maxerr <= delta, "δ={delta} {n_ch}x{t}: maxErr {maxerr} > δ");
                if delta == 0 {
                    assert_eq!(dec, sig, "δ=0 must be bit-exact");
                }
            }
        }
    }
}
