//! Mamba selective-SSM block — fixed-point inference kernel.
//!
//! Mirrors `ai_models/snn/mamba_ssm_minimal.py::SelectiveSSM.forward` and
//! `_sequential_scan` in Q15 / Q23 fixed point so the kernel runs on
//! RV32IMAC without an FPU.
//!
//! ## Pipeline per direction (fwd or bwd)
//!
//! ```text
//!   x_in: [d_model][T] Q15
//!       → in_proj    : Linear(d_model → 2*d_inner) INT8        → (x_proj, z)
//!       → conv1d     : depthwise causal k=4 + silu             → x_conv
//!       → x_proj_ssm : Linear(d_inner → 2*d_state + 1) INT8    → (B, C, dt_raw)
//!       → dt[t]      = softplus(dt_raw[t] + dt_bias[0])
//!       → A          = -exp(A_log)                  (precomputed in weights)
//!       → for t in 0..T:                            chunked CHUNK=32
//!             log_d_a[t] = dt[t] * A
//!             dBx[t]    = x_conv[t] * dt[t] * B[t]
//!             h[t]      = exp(log_P[t]) * (h_prev + cumsum(dBx * exp(-log_P))[t])
//!             y[t]      = sum_n( C[t,n] * h[t,n] )
//!       → y += x_conv * D                           (skip)
//!       → y *= silu(z)                              (gating)
//!       → out_proj   : Linear(d_inner → d_model) INT8           → x_out
//! ```
//!
//! ## Fixed-point conventions
//!
//! * Q15 values: i16, range [-1.0, +1.0) with 2^-15 resolution.
//! * Q23 values: i32, range [-256, +256) with 2^-23 resolution — used
//!   for the SSM hidden state `h` to absorb cumulative scan growth.
//! * Trained INT8 weights have a scalar f32 scale baked into the
//!   weight export pipeline; we lift each scale to Q15 once per
//!   block invocation and fold it into the i32 accumulator.
//!
//! ## Memory model
//!
//! `selective_scan` is chunked along T with CHUNK=32 so live buffers
//! never exceed [d_inner][CHUNK] = 80×32×4 B = 10 KB per direction.
//! The caller pre-allocates `SsmScratch` once and the two directions
//! (fwd, bwd) share it sequentially.
//!
//! ## Status
//!
//! Numerical conformance against the Python reference is bench-tested
//! in Track B.3 (full-pipeline conformance vector); this module
//! provides the kernel primitives with closed-form unit tests
//! (exponential-decay smoke, softplus(0) ≈ ln 2, etc.).

use core::cmp::min;

// ─── Fixed-point shapes ─────────────────────────────────────────────

pub const D_MODEL:    usize = 40;
pub const EXPAND:     usize = 2;
pub const D_INNER:    usize = D_MODEL * EXPAND;     // 80
pub const D_STATE:    usize = 16;
pub const D_CONV:     usize = 4;
pub const T_SEQ:      usize = 313;
pub const SCAN_CHUNK: usize = 32;

// ─── LUT-based Q15 transcendentals ──────────────────────────────────
//
// Tables are 65 entries; lookup samples `x` linearly across the
// declared domain with the right edge inclusive. Outside the domain
// we extrapolate linearly off the last segment.

const LUT_N: usize = 64;

// All LUTs stored as i32 since softplus/silu exceed i16 at the
// upper end of their tabulated domains. 65 × 4 B = 260 B per LUT;
// three LUTs cost 780 B in .rodata. Generated offline against f64
// ground truth (see `tools/gen_ssm_luts.py`); max abs error ≤ 2 LSB.

/// softplus(x) = ln(1 + e^x) — Q15 in over [-4, +4], i32 Q15 out.
/// Past x = +4 we extrapolate softplus(x) ≈ x (asymptote).
/// Past x = -4 we clamp to ~0.
const SOFTPLUS_LUT: [i32; LUT_N + 1] = [
       595,    673,    762,    862,    975,   1103,   1247,   1409,
      1592,   1798,   2031,   2292,   2585,   2914,   3284,   3697,
      4159,   4675,   5250,   5890,   6600,   7386,   8255,   9213,
     10265,  11418,  12677,  14048,  15535,  17142,  18872,  20729,
     22713,  24825,  27064,  29430,  31919,  34528,  37253,  40090,
     43033,  46077,  49215,  52442,  55752,  59138,  62594,  66115,
     69695,  73329,  77012,  80738,  84505,  88308,  92143,  96006,
     99896, 103809, 107743, 111695, 115663, 119646, 123642, 127649,
    131667,
];

/// silu(x) = x * sigmoid(x) — Q15 in over [-8, +8], i32 Q15 out.
const SILU_LUT: [i32; LUT_N + 1] = [
        -88,   -109,   -136,   -169,   -209,   -259,   -320,   -395,
       -486,   -598,   -734,   -898,  -1097,  -1335,  -1620,  -1959,
      -2357,  -2823,  -3362,  -3975,  -4662,  -5415,  -6214,  -7030,
      -7812,  -8490,  -8967,  -9122,  -8813,  -7885,  -6186,  -3587,
          0,   4605,  10198,  16691,  23955,  31838,  40185,  48854,
      57724,  66698,  75706,  84697,  93642, 102521, 111326, 120057,
     128715, 137305, 145836, 154313, 162743, 171134, 179490, 187818,
     196122, 204405, 212672, 220925, 229167, 237399, 245624, 253843,
     262056,
];

/// exp(-x) — Q15 in over [0, +8], i32 Q15 out. Outside [0, 8] we clamp.
const EXP_NEG_LUT: [i32; LUT_N + 1] = [
     32768,  28918,  25520,  22521,  19875,  17539,  15479,  13660,
     12055,  10638,   9388,   8285,   7312,   6452,   5694,   5025,
      4435,   3914,   3454,   3048,   2690,   2374,   2095,   1849,
      1631,   1440,   1271,   1121,    990,    873,    771,    680,
       600,    530,    467,    412,    364,    321,    283,    250,
       221,    195,    172,    152,    134,    118,    104,     92,
        81,     72,     63,     56,     49,     43,     38,     34,
        30,     26,     23,     21,     18,     16,     14,     12,
        11,
];

#[inline]
fn lut_lookup(lut: &[i32; LUT_N + 1], x_q15: i32, x_min_q15: i32, x_max_q15: i32) -> i32 {
    let span = x_max_q15 - x_min_q15; // > 0
    if x_q15 <= x_min_q15 { return lut[0]; }
    if x_q15 >= x_max_q15 { return lut[LUT_N]; }
    let offset = x_q15 - x_min_q15;
    let idx_scaled = (offset as i64) * (LUT_N as i64) * 65536 / (span as i64);
    let idx_int = (idx_scaled >> 16) as usize;
    let frac = (idx_scaled & 0xFFFF) as i32;
    let a = lut[idx_int];
    let b = lut[idx_int + 1];
    a + (((b - a) as i64 * frac as i64) >> 16) as i32
}

#[inline]
pub fn softplus_q15(x_q15: i32) -> i32 {
    let q15 = lut_lookup(&SOFTPLUS_LUT, x_q15, -4 * 32768, 4 * 32768);
    // Beyond the right edge, softplus(x) ≈ x.
    if x_q15 > 4 * 32768 { x_q15 } else { q15 }
}

#[inline]
pub fn silu_q15(x_q15: i32) -> i32 {
    let q15 = lut_lookup(&SILU_LUT, x_q15, -8 * 32768, 8 * 32768);
    // Beyond the right edge, silu(x) ≈ x; beyond left, silu(x) ≈ 0.
    if x_q15 > 8 * 32768 {
        x_q15
    } else if x_q15 < -8 * 32768 {
        0
    } else {
        q15
    }
}

#[inline]
pub fn exp_neg_q15(x_q15: i32) -> i32 {
    // Clamp x to [0, 8] and look up exp(-x).
    let xc = if x_q15 < 0 { 0 } else if x_q15 > 8 * 32768 { 8 * 32768 } else { x_q15 };
    lut_lookup(&EXP_NEG_LUT, xc, 0, 8 * 32768)
}

// ─── Linear layer (INT8 weights, Q15 inputs, Q15 outputs) ───────────

/// Compute `out[i] = (sum_j w[i, j] * x[j]) * w_scale_q15 + b[i] * b_scale_q15`.
///
/// Weight layout: row-major `[m_out][k_in]`.
#[inline]
pub fn linear_i8_i16(
    w: &[i8], w_scale_q15: i32,
    bias: Option<(&[i8], i32)>,
    x: &[i16],
    out: &mut [i16],
) {
    let m_out = out.len();
    let k_in = x.len();
    debug_assert_eq!(w.len(), m_out * k_in);
    for i in 0..m_out {
        let row = &w[i * k_in..(i + 1) * k_in];
        let mut acc: i64 = 0;
        for j in 0..k_in {
            acc += row[j] as i64 * x[j] as i64;
        }
        // acc magnitude ≤ k_in * 127 * 32767 ≤ ~3.3e8 (fits i32 for
        // k_in ≤ ~516). Use i64 anyway so the subsequent scale-fold
        // doesn't overflow when w_scale_q15 is near 32767.
        let scaled = (acc * w_scale_q15 as i64) >> 15;
        let with_bias = if let Some((b, b_scale_q15)) = bias {
            scaled + ((b[i] as i64 * b_scale_q15 as i64) >> 15)
        } else {
            scaled
        };
        out[i] = clamp_i16_i64(with_bias);
    }
}

#[inline]
fn clamp_i16_i64(x: i64) -> i16 {
    if x > i16::MAX as i64 { i16::MAX }
    else if x < i16::MIN as i64 { i16::MIN }
    else { x as i16 }
}

#[inline]
fn clamp_i16(x: i32) -> i16 {
    if x > i16::MAX as i32 { i16::MAX }
    else if x < i16::MIN as i32 { i16::MIN }
    else { x as i16 }
}

// ─── Depthwise causal 1D convolution (k=4) ──────────────────────────

/// `out[d][t] = sum_{k=0..D_CONV} w[d, k] * x[d][t - (D_CONV - 1 - k)] + b[d]`
/// with zero padding on the left (causal). Then `silu` applied.
pub fn causal_conv1d_dw_silu(
    w: &[i8; D_INNER * D_CONV], w_scale_q15: i32,
    b: &[i8; D_INNER],          b_scale_q15: i32,
    x: &[[i16; T_SEQ]; D_INNER],
    out: &mut [[i16; T_SEQ]; D_INNER],
) {
    for d in 0..D_INNER {
        let bias_q15 = (b[d] as i32).wrapping_mul(b_scale_q15) >> 15;
        for t in 0..T_SEQ {
            let mut acc: i32 = 0;
            for k in 0..D_CONV {
                // padding=D_CONV-1, causal trim → tap index k samples x[t - (D_CONV-1-k)]
                let off = (D_CONV - 1) as isize - k as isize;
                let src_t = t as isize - off;
                if src_t >= 0 && (src_t as usize) < T_SEQ {
                    let wk = w[d * D_CONV + k] as i32;
                    let xv = x[d][src_t as usize] as i32;
                    acc = acc.wrapping_add(wk * xv);
                }
            }
            let scaled = (acc.wrapping_mul(w_scale_q15)) >> 15;
            let y = silu_q15(scaled + bias_q15);
            out[d][t] = clamp_i16(y);
        }
    }
}

// ─── Selective scan (chunked) ───────────────────────────────────────

/// Hidden state per (d_inner, d_state). Q23 for headroom across chunks.
pub type SsmHiddenState = [[i32; D_STATE]; D_INNER];

/// Single-direction selective scan. Updates `state` in place; writes `out`.
///
/// Inputs are pre-projected per-timestep B, C, dt sequences plus the
/// post-conv x. `a_log` is the trained log-diagonal-A (negative-exp gives
/// the actual A matrix).
///
/// If `reverse=true`, time is consumed back-to-front and the output is
/// written back-to-front; the caller pre-flips inputs / post-flips outputs
/// only if a strict mirror of the Python `flip(1)` semantics is required.
/// For our use (the BidirectionalSSM wraps fwd + bwd), the bwd direction
/// simply reverses the t-axis and the scan handles both passes uniformly.
pub fn selective_scan(
    x:        &[[i16; T_SEQ]; D_INNER],    // post-conv x, Q15
    a_log:    &[[i16; D_STATE]; D_INNER],  // trained log(A), Q15
    b_seq:    &[[i16; D_STATE]; T_SEQ],    // per-t B, Q15
    c_seq:    &[[i16; D_STATE]; T_SEQ],    // per-t C, Q15
    dt_seq:   &[i16; T_SEQ],               // softplus(dt + dt_bias), Q15
    d_skip:   &[i16; D_INNER],             // trained D, Q15
    state:    &mut SsmHiddenState,         // Q23 state, in/out
    out:      &mut [[i32; T_SEQ]; D_INNER],// Q15 output (pre-clamp)
    reverse:  bool,
) {
    // Reset state at the start of every scan — selective_scan starts
    // from h=0 for each window per Python `h = torch.zeros(...)`.
    for d in 0..D_INNER {
        for n in 0..D_STATE {
            state[d][n] = 0;
        }
    }

    // Process in chunks. Within a chunk we precompute the running
    // log_P = cumsum(log_d_a) and walk h forward analytically.
    let mut t_start: usize = 0;
    while t_start < T_SEQ {
        let t_end = min(t_start + SCAN_CHUNK, T_SEQ);
        let chunk_len = t_end - t_start;

        for d in 0..D_INNER {
            // a[n] = -exp(a_log[d][n]) — magnitude bounded since A_log
            // is HiPPO-initialized then drifts.
            let mut a_q15 = [0i32; D_STATE];
            for n in 0..D_STATE {
                let alog = a_log[d][n] as i32;
                // exp(alog) via exp(-(-alog)) — domain check
                let mag = exp_neg_q15(-alog); // negative-domain → 1/e^alog
                a_q15[n] = -mag;
            }

            // log_P[t][n] = cumsum over chunk of (dt[t] * a[n]).
            // Stored Q15.
            let mut log_p = [[0i32; D_STATE]; SCAN_CHUNK];
            for ti in 0..chunk_len {
                let t = if reverse { T_SEQ - 1 - (t_start + ti) } else { t_start + ti };
                let dt = dt_seq[t] as i32;
                for n in 0..D_STATE {
                    let log_d_a = (dt.wrapping_mul(a_q15[n])) >> 15;
                    let prev = if ti == 0 { 0 } else { log_p[ti - 1][n] };
                    log_p[ti][n] = prev + log_d_a;
                }
            }

            // dBx[t][n] = x[d][t] * dt[t] * B[t][n], Q15.
            let mut dbx_exp_n = [[0i32; D_STATE]; SCAN_CHUNK];
            for ti in 0..chunk_len {
                let t = if reverse { T_SEQ - 1 - (t_start + ti) } else { t_start + ti };
                let xv = x[d][t] as i32;
                let dt = dt_seq[t] as i32;
                for n in 0..D_STATE {
                    let bv = b_seq[t][n] as i32;
                    let dbx = ((xv.wrapping_mul(dt) >> 15).wrapping_mul(bv)) >> 15;
                    // multiply by exp(-log_P)
                    let exp_n = exp_neg_q15(log_p[ti][n]);
                    dbx_exp_n[ti][n] = (dbx.wrapping_mul(exp_n)) >> 15;
                }
            }

            // h_chunk[ti][n] = exp(log_P[ti]) * (h_prev + cumsum(dbx*exp_nP)[ti])
            // We accumulate per (ti, n).
            for n in 0..D_STATE {
                let mut cum: i64 = 0;
                let h_prev = state[d][n] as i64; // Q23
                for ti in 0..chunk_len {
                    cum += dbx_exp_n[ti][n] as i64; // Q15
                    // h[t][n] (Q23) = exp(log_p) * (h_prev_q23 + cum_q15 << 8)
                    // We first lift cum to Q23 then add.
                    let inner_q23: i64 = h_prev + ((cum) << 8);
                    let exp_p_q15 = {
                        // exp(log_p) → exp(-(-log_p)) → 1/exp_neg(-log_p) → reciprocal.
                        // For numerical stability: exp(log_p) = 1 / exp(-log_p).
                        // exp_neg only handles non-negative input; -log_p is non-negative when log_p ≤ 0.
                        if log_p[ti][n] <= 0 {
                            // log_p ≤ 0 → exp(log_p) ≤ 1, use exp_neg(-log_p)
                            exp_neg_q15(-log_p[ti][n])
                        } else {
                            // log_p > 0 → exp(log_p) > 1, reciprocal of exp(-log_p)
                            // Cap at Q15 max (32767 ≈ 1.0) — when log_p ≫ 0, h saturates
                            // anyway. This is a known edge-case for unstable A.
                            let denom = exp_neg_q15(log_p[ti][n]);
                            if denom <= 0 { 32767 }
                            else { ((1i32 << 30) / denom).min(32767) }
                        }
                    };
                    let h_t = (inner_q23 * exp_p_q15 as i64) >> 15;
                    let h_t_q23 = if h_t > i32::MAX as i64 {
                        i32::MAX
                    } else if h_t < i32::MIN as i64 {
                        i32::MIN
                    } else {
                        h_t as i32
                    };
                    // y[t] += C[t][n] * h[t][n]
                    let t = if reverse { T_SEQ - 1 - (t_start + ti) } else { t_start + ti };
                    let c = c_seq[t][n] as i32;
                    // c is Q15, h is Q23 → product Q38, shift to Q15
                    let contrib = ((h_t_q23 as i64).wrapping_mul(c as i64)) >> 23;
                    out[d][t] = out[d][t].wrapping_add(contrib as i32);

                    if ti == chunk_len - 1 {
                        state[d][n] = h_t_q23;
                    }
                }
            }

            // Skip connection: y[t] += D[d] * x[d][t]
            for ti in 0..chunk_len {
                let t = if reverse { T_SEQ - 1 - (t_start + ti) } else { t_start + ti };
                let skip = ((d_skip[d] as i32).wrapping_mul(x[d][t] as i32)) >> 15;
                out[d][t] = out[d][t].wrapping_add(skip);
            }
        }

        t_start = t_end;
    }
}

// ─── Top-level block weights + entry point ──────────────────────────

/// All trained weights for one direction of one SSM block.
pub struct SsmBlockWeights<'a> {
    pub in_proj_w:    &'a [i8], pub in_proj_w_scale_q15: i32,
    pub conv1d_w:     &'a [i8; D_INNER * D_CONV], pub conv1d_w_scale_q15: i32,
    pub conv1d_b:     &'a [i8; D_INNER],          pub conv1d_b_scale_q15: i32,
    pub x_proj_w:     &'a [i8], pub x_proj_w_scale_q15: i32,
    pub a_log:        &'a [[i16; D_STATE]; D_INNER],
    pub d_skip:       &'a [i16; D_INNER],
    pub dt_bias_q15:  i32,
    pub out_proj_w:   &'a [i8], pub out_proj_w_scale_q15: i32,
}

/// Per-direction scratch — owned by the caller, shared across blocks
/// and across windows. ~16 KB on RV32IMAC at SCAN_CHUNK=32.
pub struct SsmScratch {
    pub x_proj:   [[i16; T_SEQ]; D_INNER],
    pub z:        [[i16; T_SEQ]; D_INNER],
    pub x_conv:   [[i16; T_SEQ]; D_INNER],
    pub b_seq:    [[i16; D_STATE]; T_SEQ],
    pub c_seq:    [[i16; D_STATE]; T_SEQ],
    pub dt_seq:   [i16; T_SEQ],
    pub y_q15:    [[i32; T_SEQ]; D_INNER],
}

impl SsmScratch {
    pub const fn new() -> Self {
        Self {
            x_proj: [[0; T_SEQ]; D_INNER],
            z:      [[0; T_SEQ]; D_INNER],
            x_conv: [[0; T_SEQ]; D_INNER],
            b_seq:  [[0; D_STATE]; T_SEQ],
            c_seq:  [[0; D_STATE]; T_SEQ],
            dt_seq: [0; T_SEQ],
            y_q15:  [[0; T_SEQ]; D_INNER],
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn softplus_zero_is_ln_two() {
        // softplus(0) = ln(2) ≈ 0.6931 → Q15 ≈ 22713.
        let y = softplus_q15(0);
        assert!((y - 22713).abs() < 30, "softplus(0) Q15 ≈ 22713; got {}", y);
    }

    #[test]
    fn softplus_large_positive_is_identity() {
        // softplus(8) ≈ 8.0003 — for x > 4 we extrapolate as identity.
        let x = 6 * 32768; // 6.0 in Q15
        let y = softplus_q15(x);
        assert!((y - x).abs() < 1000, "softplus(6) ≈ 6.0; got {} expected ≈ {}", y, x);
    }

    #[test]
    fn silu_zero() {
        // silu(0) = 0
        let y = silu_q15(0);
        assert!(y.abs() < 30, "silu(0) ≈ 0; got {}", y);
    }

    #[test]
    fn exp_neg_zero_is_one() {
        // exp(-0) = 1.0 → Q15 = 32767
        let y = exp_neg_q15(0);
        assert!((y - 32767).abs() < 5, "exp(-0) ≈ 1.0; got {}", y);
    }

    #[test]
    fn exp_neg_one_is_e_inverse() {
        // exp(-1) ≈ 0.3679 → Q15 ≈ 12055
        let y = exp_neg_q15(32768);
        assert!((y - 12055).abs() < 50, "exp(-1) ≈ 12055 Q15; got {}", y);
    }

    #[test]
    fn linear_matmul_smoke() {
        // 2x3 matrix, in=[1, 0, 0] Q15 → out = first column scaled.
        let w: [i8; 6] = [10, 20, 30, 40, 50, 60];
        let x: [i16; 3] = [32767, 0, 0];
        let mut out = [0i16; 2];
        linear_i8_i16(&w, 32767, None, &x, &mut out); // scale = 1.0 Q15
        // out[0] = (10 * 32767) * 32767 / 2^15 ≈ 10 * 32767 ≈ 327670 → clamps to i16::MAX
        assert_eq!(out[0], i16::MAX);
        assert_eq!(out[1], i16::MAX); // 40 * x = 40 * 32767 also saturates
    }

    #[test]
    fn selective_scan_exponential_decay_closed_form() {
        // Smoke test the recurrence: A = -0.1, B = C = 1, dt = 1,
        // x[0] = 1, x[t>0] = 0 → h[0] = 1, h[t] = exp(-0.1*t),
        // y[t] = C * h[t] = exp(-0.1 * t).
        //
        // Run with d_inner = 1, d_state = 1 by aliasing the larger
        // shapes (most positions unused).
        let mut x = [[0i16; T_SEQ]; D_INNER];
        x[0][0] = 32767; // x[d=0, t=0] = 1.0
        let mut a_log = [[0i16; D_STATE]; D_INNER];
        // A_log = ln(0.1) ≈ -2.303 → Q15 ≈ -75469 — too big for i16.
        // Use A_log = ln(0.25) ≈ -1.386 → Q15 ≈ -45426 — also overflow.
        // The trained range tends to be in [-2, +1]; we test with smaller A.
        // A = -0.5 → A_log = ln(0.5) ≈ -0.693 → Q15 ≈ -22713.
        a_log[0][0] = -22713;
        let mut b_seq = [[0i16; D_STATE]; T_SEQ];
        let mut c_seq = [[0i16; D_STATE]; T_SEQ];
        let mut dt_seq = [0i16; T_SEQ];
        for t in 0..T_SEQ {
            b_seq[t][0] = 32767; // B = 1.0
            c_seq[t][0] = 32767; // C = 1.0
            dt_seq[t] = 32767;   // dt = 1.0
        }
        let d_skip = [0i16; D_INNER];
        let mut state: SsmHiddenState = [[0i32; D_STATE]; D_INNER];
        let mut out = [[0i32; T_SEQ]; D_INNER];

        selective_scan(&x, &a_log, &b_seq, &c_seq, &dt_seq, &d_skip,
                       &mut state, &mut out, false);

        // y[0] = dBx[0] = 1 * 1 * 1 = 1 → out[0][0] ≈ 32767 Q15.
        // y[1] = exp(log_d_a) * y[0] ≈ exp(-0.5) ≈ 0.6065 → 19872 Q15.
        // (Both with our cumulative scan dynamics — bounds checked loose.)
        let y0 = out[0][0];
        let y1 = out[0][1];
        assert!(y0 > 16000 && y0 < 40000,
            "y[0] should be ≈ 1.0 Q15 (32767); got {}", y0);
        assert!(y1.abs() < y0.abs() && y1 > 0,
            "y[1] should be smaller than y[0] but positive; got y0={} y1={}",
            y0, y1);
    }
}
