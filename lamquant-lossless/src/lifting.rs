//! Integer Le Gall 5/3 lifting DWT — forward and inverse.
//!
//! Bit-identical to the Python/numba implementation.
//!
//! Inner loops use contiguous-buffer arithmetic so the compiler
//! auto-vectorizes on AVX2/NEON/WASM SIMD. No hand-written intrinsics,
//! no unsafe — works on every target including MCU (RP2350).

use alloc::vec;
use alloc::vec::Vec;

/// Forward 1D integer lifting. Returns (approx, detail).
///
/// Split-buffer implementation: even/odd elements are separated first,
/// then predict/update operate on contiguous slices. The compiler
/// vectorizes the bulk arithmetic; boundary handling stays scalar.
#[inline]
pub fn forward(signal: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let n = signal.len();
    if n < 2 {
        return (signal.to_vec(), Vec::new());
    }

    let n_detail = n / 2;
    let n_approx = n.div_ceil(2);

    // Split into contiguous even (approx) and odd (detail) buffers.
    // This is the key to auto-vectorization: stride-1 access.
    let mut approx: Vec<i64> = Vec::with_capacity(n_approx);
    let mut detail: Vec<i64> = Vec::with_capacity(n_detail);
    for i in 0..n_approx {
        approx.push(signal[2 * i]);
    }
    for i in 0..n_detail {
        detail.push(signal[2 * i + 1]);
    }

    // ── Predict step: detail[i] -= (approx[i] + approx[i+1]) >> 1 ──
    // Bulk: contiguous reads from approx[], contiguous write to detail[]
    // Compiler vectorizes this loop (4x i64 per AVX2 cycle).
    //
    // Audit-2026-05-11 Fix-#41: `n_detail - 1` is safe because we
    // bailed early on `n < 2` (line 20), so `n_detail = n / 2 >= 1`.
    // Document invariant + debug_assert so a future refactor of the
    // early-exit cannot silently underflow.
    debug_assert!(
        n_detail >= 1,
        "lifting forward: n_detail must be >= 1 (n={n})"
    );
    let bulk_end = if n % 2 == 0 { n_detail - 1 } else { n_detail };
    for i in 0..bulk_end.min(n_detail) {
        if i + 1 < n_approx {
            detail[i] -= (approx[i] + approx[i + 1]) >> 1;
        }
    }
    // Boundary: last detail when even-length signal
    if n % 2 == 0 && n_detail > 0 {
        detail[n_detail - 1] -= approx[n_detail - 1];
    }

    // ── Update step: approx[i] += (detail[i-1] + detail[i] + 2) >> 2 ──
    approx[0] += (detail[0] + 1) >> 1;
    // Bulk: contiguous reads from detail[], contiguous write to approx[]
    for i in 1..n_approx {
        if i - 1 < n_detail && i < n_detail {
            // Interior: two neighboring details available
            approx[i] += (detail[i - 1] + detail[i] + 2) >> 2;
        } else if i - 1 < n_detail {
            // Right boundary: only left detail available
            approx[i] += (detail[i - 1] + 1) >> 1;
        }
    }

    (approx, detail)
}

/// Inverse 1D integer lifting. Exact inverse of `forward`.
///
/// Same split-buffer strategy for SIMD-friendly inner loops.
#[inline]
pub fn inverse(approx: &[i64], detail: &[i64]) -> Vec<i64> {
    let n_approx = approx.len();
    let n_detail = detail.len();
    let n = n_approx + n_detail;

    if n < 2 {
        return approx.to_vec();
    }

    let mut a = approx.to_vec();
    let mut d = detail.to_vec();

    // ── Inverse update: undo approx[i] += ... ──
    for i in (1..n_approx).rev() {
        if i - 1 < n_detail && i < n_detail {
            a[i] -= (d[i - 1] + d[i] + 2) >> 2;
        } else if i - 1 < n_detail {
            a[i] -= (d[i - 1] + 1) >> 1;
        }
    }
    a[0] -= (d[0] + 1) >> 1;

    // ── Inverse predict: undo detail[i] -= ... ──
    if n % 2 == 0 && n_detail > 0 {
        d[n_detail - 1] += a[n_detail - 1];
    }
    let bulk_end = if n % 2 == 0 { n_detail - 1 } else { n_detail };
    for i in 0..bulk_end.min(n_detail) {
        if i + 1 < n_approx {
            d[i] += (a[i] + a[i + 1]) >> 1;
        }
    }

    // ── Interleave back ──
    let mut out = vec![0i64; n];
    for i in 0..n_approx {
        out[2 * i] = a[i];
    }
    for i in 0..n_detail {
        out[2 * i + 1] = d[i];
    }
    out
}

/// 3-level forward lifting.
#[inline]
pub fn forward_3level(signal: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let (l1_approx, l1_detail) = forward(signal);
    let (l2_approx, l2_detail) = forward(&l1_approx);
    let (l3_approx, l3_detail) = forward(&l2_approx);
    (l3_approx, l3_detail, l2_detail, l1_detail)
}

/// 3-level inverse lifting.
#[inline]
pub fn inverse_3level(
    l3_approx: &[i64],
    l3_detail: &[i64],
    l2_detail: &[i64],
    l1_detail: &[i64],
) -> Vec<i64> {
    let l2_approx = inverse(l3_approx, l3_detail);
    let l1_approx = inverse(&l2_approx, l2_detail);
    inverse(&l1_approx, l1_detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_various_lengths() {
        for n in [1, 2, 3, 4, 7, 8, 10, 63, 128, 625, 1250, 2500] {
            let signal: Vec<i64> = (0..n).map(|i| ((i * 137) % 10000 - 5000) as i64).collect();
            let (a, d) = forward(&signal);
            let recovered = inverse(&a, &d);
            assert_eq!(signal, recovered, "Failed at n={}", n);
        }
    }

    #[test]
    fn roundtrip_3level() {
        let signal: Vec<i64> = (0..2500)
            .map(|i| ((i * 137) % 10000 - 5000) as i64)
            .collect();
        let (a3, d3, d2, d1) = forward_3level(&signal);
        let recovered = inverse_3level(&a3, &d3, &d2, &d1);
        assert_eq!(signal, recovered);
    }

    #[test]
    fn roundtrip_zeros() {
        let signal = vec![0i64; 100];
        let (a, d) = forward(&signal);
        assert_eq!(inverse(&a, &d), signal);
    }
}
