//! Float **CDF 9/7** wavelet (the JPEG2000 *lossy* filter) — forward + inverse
//! lifting, host-encode and no_std-decode capable.
//!
//! ADR 0054 Phase 3, lever 2. The integer 5/3 Le Gall (in the `mcu` floor) is a
//! short, integer-*reversible* filter — ideal for lossless, but a poor energy
//! compactor for the lossy ratio attack. The 9/7 CDF wavelet has much better
//! frequency selectivity, at the cost of irrational lifting constants ⇒ it is
//! **float-only and lossy-only** (never bit-exact). That is exactly why it lives
//! here in LMO (the float-allowed Optimum tier) and never in the firmware floor.
//!
//! ## Layout parity with 5/3
//!
//! A single forward level splits a length-`n` signal into
//! `approx` (even samples, `ceil(n/2)`) and `detail` (odd samples, `floor(n/2)`),
//! identical to `lamquant_lml_mcu::lifting::forward`. A 3-level cascade therefore
//! produces the same `[a3, d3, d2, d1]` subband layout the rest of the PCRD
//! pipeline already understands — only the numbers (and their float-ness) differ.
//!
//! ## Boundary handling
//!
//! Whole-sample symmetric ("mirror about the edge sample") extension, per the
//! JPEG2000 reference. The 9-tap support reaches past the array edges; naive
//! zero-padding produces edge ringing. [`refl`] mirrors any virtual index back
//! into `[0, n)`.
//!
//! ## Invertibility
//!
//! Forward and inverse are exact algebraic inverses (lifting is structurally
//! invertible regardless of the constants), so the only reconstruction error is
//! f64 round-off — a few LSB on integer input. Correctness does not depend on the
//! precise constant values, only on forward/inverse using a *consistent* set; the
//! `roundtrip_*` self-tests enforce that.

use alloc::vec;
use alloc::vec::Vec;

// ── Canonical CDF 9/7 lifting constants (Daubechies–Sweldens / JPEG2000) ──
const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
/// Scaling constant `K`. Forward scales even (low-pass) by `1/K`, odd (high-pass)
/// by `K`; inverse reverses. The pairing — not the magnitude — is what makes the
/// transform invertible; the magnitude only redistributes subband energy (which
/// the empirical synthesis gains absorb).
const K: f64 = 1.230_174_104_914_001;
const INV_K: f64 = 1.0 / K;

/// Whole-sample symmetric reflection of a (possibly out-of-range) index into
/// `[0, n)`. Mirrors about the boundary *samples* (index 0 and n-1).
#[inline]
fn refl(i: isize, n: usize) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return 0;
    }
    let m = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= m {
            i = 2 * (m - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// In-place forward 9/7 lifting on the interleaved signal buffer.
#[inline]
fn fwd97_inplace(x: &mut [f64]) {
    let n = x.len();
    if n < 2 {
        return;
    }
    // predict odd (α)
    let mut i = 1;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] += ALPHA * (l + r);
        i += 2;
    }
    // update even (β)
    let mut i = 0;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] += BETA * (l + r);
        i += 2;
    }
    // predict odd (γ)
    let mut i = 1;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] += GAMMA * (l + r);
        i += 2;
    }
    // update even (δ)
    let mut i = 0;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] += DELTA * (l + r);
        i += 2;
    }
    // scale
    let mut i = 0;
    while i < n {
        x[i] *= INV_K;
        i += 2;
    }
    let mut i = 1;
    while i < n {
        x[i] *= K;
        i += 2;
    }
}

/// In-place inverse 9/7 lifting — exact reverse of [`fwd97_inplace`].
#[inline]
fn inv97_inplace(x: &mut [f64]) {
    let n = x.len();
    if n < 2 {
        return;
    }
    // unscale
    let mut i = 0;
    while i < n {
        x[i] *= K;
        i += 2;
    }
    let mut i = 1;
    while i < n {
        x[i] *= INV_K;
        i += 2;
    }
    // undo update even (δ)
    let mut i = 0;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] -= DELTA * (l + r);
        i += 2;
    }
    // undo predict odd (γ)
    let mut i = 1;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] -= GAMMA * (l + r);
        i += 2;
    }
    // undo update even (β)
    let mut i = 0;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] -= BETA * (l + r);
        i += 2;
    }
    // undo predict odd (α)
    let mut i = 1;
    while i < n {
        let l = x[refl(i as isize - 1, n)];
        let r = x[refl(i as isize + 1, n)];
        x[i] -= ALPHA * (l + r);
        i += 2;
    }
}

/// Single-level forward 9/7. Returns `(approx, detail)` in split-buffer form,
/// matching `lamquant_lml_mcu::lifting::forward`'s subband sizes.
pub fn forward_97(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n < 2 {
        return (signal.to_vec(), Vec::new());
    }
    let mut buf = signal.to_vec();
    fwd97_inplace(&mut buf);
    let n_approx = n.div_ceil(2);
    let n_detail = n / 2;
    let mut approx = Vec::with_capacity(n_approx);
    let mut detail = Vec::with_capacity(n_detail);
    for i in 0..n_approx {
        approx.push(buf[2 * i]);
    }
    for i in 0..n_detail {
        detail.push(buf[2 * i + 1]);
    }
    (approx, detail)
}

/// Single-level inverse 9/7. Exact inverse of [`forward_97`] (up to f64 round-off).
pub fn inverse_97(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n_approx = approx.len();
    let n_detail = detail.len();
    let n = n_approx + n_detail;
    if n < 2 {
        return approx.to_vec();
    }
    let mut buf = vec![0.0f64; n];
    for i in 0..n_approx {
        buf[2 * i] = approx[i];
    }
    for i in 0..n_detail {
        buf[2 * i + 1] = detail[i];
    }
    inv97_inplace(&mut buf);
    buf
}

/// Round half away from zero with pure f64 arithmetic (no std/libm `round`),
/// keeping the inverse no_std-clean.
#[inline]
pub fn round_i64(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// `n_levels`-level forward 9/7. Integer input is widened to f64; the cascade
/// runs on the approximation band each level. Returns ordered subbands
/// `[approx, detail_top, ..., detail_1]` — the same layout (and per-subband
/// sizes) the 5/3 PCRD pipeline produces, so downstream code is transform-blind.
pub fn forward_97_levels(signal: &[i64], n_levels: u8) -> Vec<Vec<f64>> {
    let mut approx: Vec<f64> = signal.iter().map(|&v| v as f64).collect();
    let mut details: Vec<Vec<f64>> = Vec::with_capacity(n_levels as usize);
    for _ in 0..n_levels {
        let (a, d) = forward_97(&approx);
        details.push(d);
        approx = a;
    }
    let mut out = Vec::with_capacity(n_levels as usize + 1);
    out.push(approx);
    for d in details.into_iter().rev() {
        out.push(d);
    }
    out
}

/// `n_levels`-level inverse 9/7. Takes ordered float subbands
/// `[approx, detail_top, ..., detail_1]` (e.g. dequantized coefficients),
/// reconstructs the float signal, and rounds to i64. `n_levels == 0` is identity.
pub fn inverse_97_levels(subs: &[Vec<f64>], n_levels: u8) -> Vec<i64> {
    if n_levels == 0 {
        return subs[0].iter().map(|&v| round_i64(v)).collect();
    }
    let mut approx = subs[0].clone();
    for lvl in 0..n_levels as usize {
        approx = inverse_97(&approx, &subs[1 + lvl]);
    }
    approx.iter().map(|&v| round_i64(v)).collect()
}

/// 3-level forward 9/7 — `[a3, d3, d2, d1]`. Thin wrapper over [`forward_97_levels`].
pub fn forward_97_3level(signal: &[i64]) -> [Vec<f64>; 4] {
    let v = forward_97_levels(signal, 3);
    debug_assert_eq!(v.len(), 4);
    let mut it = v.into_iter();
    [
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
        it.next().unwrap(),
    ]
}

/// 3-level inverse 9/7 — wrapper over [`inverse_97_levels`].
pub fn inverse_97_3level(subs: &[Vec<f64>]) -> Vec<i64> {
    debug_assert_eq!(subs.len(), 4, "9/7 3-level expects [a3, d3, d2, d1]");
    inverse_97_levels(subs, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[i64], b: &[i64]) -> i64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).max().unwrap_or(0)
    }

    #[test]
    fn single_level_roundtrip_various_lengths() {
        for n in [2usize, 3, 4, 7, 8, 10, 63, 128, 625, 1250, 2500] {
            let signal: Vec<f64> = (0..n).map(|i| ((i * 137) % 10000) as f64 - 5000.0).collect();
            let (a, d) = forward_97(&signal);
            assert_eq!(a.len(), n.div_ceil(2), "approx len at n={n}");
            assert_eq!(d.len(), n / 2, "detail len at n={n}");
            let recon = inverse_97(&a, &d);
            let err = signal
                .iter()
                .zip(&recon)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-6, "9/7 single-level roundtrip err {err} at n={n}");
        }
    }

    #[test]
    fn three_level_roundtrip_small_error() {
        // Smooth low-freq + high-freq + periodic spikes — real allocatable energy.
        let signal: Vec<i64> = (0..2500)
            .map(|i| {
                let lo = ((i as f64) * 0.05).sin() * 3000.0;
                let hi = ((i as f64) * 0.9).sin() * 250.0;
                let spike = if i % 101 == 0 { 1200.0 } else { 0.0 };
                (lo + hi + spike) as i64
            })
            .collect();
        let subs = forward_97_3level(&signal);
        // Layout parity with 5/3 [a3, d3, d2, d1]: subbands tile the signal.
        assert_eq!(subs.iter().map(Vec::len).sum::<usize>(), signal.len());
        let recon = inverse_97_3level(&subs);
        assert_eq!(recon.len(), signal.len());
        // Lossy by construction, but at q=1 (no quantization) only f64 round-off:
        // a couple of LSB at most.
        assert!(
            max_abs_err(&signal, &recon) <= 2,
            "9/7 3-level lossless-ish roundtrip max err {} > 2",
            max_abs_err(&signal, &recon)
        );
    }

    #[test]
    fn odd_length_roundtrip() {
        let signal: Vec<i64> = (0..1237).map(|i| ((i * 31) % 4096 - 2048) as i64).collect();
        let subs = forward_97_3level(&signal);
        let recon = inverse_97_3level(&subs);
        assert_eq!(recon.len(), signal.len());
        assert!(max_abs_err(&signal, &recon) <= 2);
    }

    #[test]
    fn zeros_stay_zero() {
        let signal = vec![0i64; 512];
        let subs = forward_97_3level(&signal);
        for sb in &subs {
            for &c in sb {
                assert!(c.abs() < 1e-9);
            }
        }
        assert_eq!(inverse_97_3level(&subs), signal);
    }
}
