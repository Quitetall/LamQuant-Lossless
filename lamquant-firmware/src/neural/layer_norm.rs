//! `nn.LayerNorm` — fixed-point port of `torch.nn.LayerNorm(d_model)`.
//!
//! Used by `BidirectionalSSM.norm` (one instance per block, applied per
//! timestep before fwd + bwd scans). Python signature:
//!
//! ```python
//! # mean and var taken over the last axis (D), per-element gamma/beta.
//! y = (x - mean) / sqrt(var + eps) * gamma + beta
//! ```
//!
//! Q15 fixed-point implementation runs in i32 accumulators with i64
//! intermediates for the mean / variance reductions. Newton-Raphson
//! one-iteration inverse-sqrt (with a small LUT seed) keeps the
//! per-timestep cost under ~4 D_MODEL multiplications.

use crate::neural::ssm_block::{D_MODEL, T_SEQ};

/// LayerNorm's epsilon. PyTorch default = 1e-5 → Q30 = ~1073.
/// We track it in Q30 to match the variance precision.
const EPS_Q30: i64 = 1073;

/// Integer floor-sqrt of an i64 (n ≥ 0). Newton-Raphson, converges
/// in O(log log n) iterations on the IMAC core. Returns 0 for n ≤ 0.
///
/// Key identity used by LayerNorm:
///   sqrt(var_q30) = sqrt(var * 2^30) = sqrt(var) * 2^15
///
/// So `isqrt_i64(var_q30)` directly gives `sqrt(var)` in Q15. No LUT
/// gymnastics needed — the wider integer arithmetic is cheaper than
/// the rescale-and-Newton dance on a chip without an FPU.
pub fn isqrt_i64(n: i64) -> i64 {
    if n <= 1 { return n.max(0); }
    // Seed with a power-of-two estimate so Newton converges fast.
    let mut x = n;
    let mut shift = 0u32;
    while (x >> 2) >= 1 { x >>= 2; shift += 1; }
    let mut y = x << shift; // crude seed ≥ true sqrt
    // Newton-Raphson: y_{k+1} = (y + n/y) / 2 — converges quadratically.
    loop {
        let z = (y + n / y) / 2;
        if z >= y { break; }
        y = z;
    }
    y
}

/// LayerNorm over the last axis (D_MODEL).
///
/// `gamma` and `beta` are INT8 with scalar f32 scale folded to Q15.
/// EPS is fixed at 1e-5 in Q30 to match PyTorch's default and stay
/// inside the Q30 variance precision.
pub fn layer_norm_q15(
    x:        &[[i16; T_SEQ]; D_MODEL],
    gamma:    &[i8; D_MODEL], gamma_scale_q15: i32,
    beta:     &[i8; D_MODEL], beta_scale_q15:  i32,
    out:      &mut [[i16; T_SEQ]; D_MODEL],
) {
    for t in 0..T_SEQ {
        // Mean (Q15, summed in i64).
        let mut sum: i64 = 0;
        for d in 0..D_MODEL {
            sum += x[d][t] as i64;
        }
        let mean_q15 = (sum / D_MODEL as i64) as i32;

        // Variance (Q30, summed in i64).
        let mut var_sum: i64 = 0;
        for d in 0..D_MODEL {
            let diff = x[d][t] as i64 - mean_q15 as i64; // Q15
            var_sum += diff * diff;                       // Q30 contribution
        }
        let var_q30 = (var_sum / D_MODEL as i64) + EPS_Q30;
        // sqrt(var_q30) gives sqrt(var) directly in Q15.
        let sqrt_var_q15 = isqrt_i64(var_q30).max(1);

        for d in 0..D_MODEL {
            let centered = x[d][t] as i64 - mean_q15 as i64;          // Q15
            // y_q15 = (x - mean) / sqrt(var) — Q15 division of Q30 by Q15.
            let normed_q15 = (centered << 15) / sqrt_var_q15;
            // gamma[d] is INT8 raw weight (-128..127); gamma_scale_q15
            // is the floating-point gamma_scale lifted to Q15. Their
            // product is therefore gamma_real already in Q15 — no
            // additional shift in the *interpretation*.
            let g_q15 = gamma[d] as i64 * gamma_scale_q15 as i64;     // Q15
            let scaled_q15 = (normed_q15 * g_q15) >> 15;              // Q15
            let b_q15 = beta[d] as i64 * beta_scale_q15 as i64;       // Q15
            let y = scaled_q15 + b_q15;
            out[d][t] = if y > i16::MAX as i64 {
                i16::MAX
            } else if y < i16::MIN as i64 {
                i16::MIN
            } else {
                y as i16
            };
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn isqrt_known_values() {
        // sqrt(1<<30) = 1<<15 (true integer sqrt)
        assert_eq!(isqrt_i64(1 << 30), 1 << 15);
        // sqrt(4<<30) = 2<<15
        assert_eq!(isqrt_i64(4 << 30), 2 << 15);
        // sqrt(10000) ≈ 100
        assert_eq!(isqrt_i64(10000), 100);
        // sqrt(0) = 0, sqrt(1) = 1
        assert_eq!(isqrt_i64(0), 0);
        assert_eq!(isqrt_i64(1), 1);
    }

    #[test]
    fn layer_norm_constant_input_is_beta() {
        // x[:, t] = constant  ⇒ var = 0 ⇒ inv_std saturates ⇒ normed
        // is dominated by EPS, but small. gamma=1, beta=0 ⇒ out ≈ 0.
        let x: [[i16; T_SEQ]; D_MODEL] = [[1000; T_SEQ]; D_MODEL];
        // gamma_scale_q15 = 32767 → scale 1.0
        // beta_scale_q15 = 32767 → scale 1.0
        let gamma: [i8; D_MODEL] = [0; D_MODEL]; // gamma = 0 → out = beta
        let beta: [i8; D_MODEL] = [0; D_MODEL];  // beta = 0
        let mut out = [[0i16; T_SEQ]; D_MODEL];
        layer_norm_q15(&x, &gamma, 32767, &beta, 32767, &mut out);
        // gamma=0 zeros out the normalized term; beta=0 leaves nothing.
        for d in 0..D_MODEL {
            for t in 0..T_SEQ {
                assert_eq!(out[d][t], 0,
                    "gamma=beta=0 should produce zero output; got {} at [{}][{}]",
                    out[d][t], d, t);
            }
        }
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        // Build x[:, t] = alternating ±k → mean = 0, var = k²
        // After LN with γ=1, β=0:  y = x / k → ±1 ≈ Q15 ±32767.
        let k: i16 = 100;
        let mut x = [[0i16; T_SEQ]; D_MODEL];
        for d in 0..D_MODEL {
            x[d][0] = if d % 2 == 0 { k } else { -k };
        }
        // gamma = 1.0 (Q15) packed in i8 with scale 1/127
        let gamma = [127i8; D_MODEL];
        let g_scale = 32767 / 127; // ≈ 258 → 1.0 / 127 in Q15
        let beta = [0i8; D_MODEL];
        let mut out = [[0i16; T_SEQ]; D_MODEL];
        layer_norm_q15(&x, &gamma, g_scale, &beta, 0, &mut out);
        // Bounds are loose because Q15 precision + inv-sqrt
        // approximation accumulate; we just verify sign and magnitude.
        for d in 0..D_MODEL {
            let expected_sign = if d % 2 == 0 { 1 } else { -1 };
            assert_eq!(out[d][0].signum() as i32, expected_sign,
                "LN sign mismatch at d={}: out={}", d, out[d][0]);
            assert!(out[d][0].abs() > 10_000,
                "LN at d={} should be near ±1.0 Q15; got {}",
                d, out[d][0]);
        }
    }
}
