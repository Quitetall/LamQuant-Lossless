//! TNN encoder — focal modulation chain on the L3 approximation.
//!
//! Pipeline:
//!   L3 [21][313]
//!     → premix     (TernaryConv1d 21→21, k=1, s=1)
//!     → focal1     (TernaryConv1d 21→128, k=3, s=2)         [STUB — gap]
//!     → focal2     (TernaryConv1d 128→128, k=5, s=1) + GN + ReLU
//!     → focal3     (TernaryConv1d 128→128, k=7, s=2) + GN + ReLU
//!     → dw_gate    (Depthwise 128, k=3, s=1)
//!     → bneck      (GLU: bneck_v INT8 * sigmoid(bneck_g))
//!   → Latent [32][79]
//!
//! ## Gaps documented
//!
//! 1. **bneck_v / bneck_g biases NOT exported** — the C reference adds a
//!    per-channel bias before the GLU multiply. The current export emits
//!    only alphas/scales. The bias is approximated as zero. Adds a
//!    constant offset to the latent that may shift FSQ binning slightly
//!    relative to the Python reference. Acceptable for bring-up but should
//!    be wired before parity claims.
//!
//! ## Performance
//!
//! Naive O(out_ch × t × in_ch × k) inner loop with `ternary_mac::conv1d_channel`
//! on each (out_ch, t). The C reference uses bit-serial CPOP + DMA prefetch
//! for ~3-cycle MACs; this Rust port targets correctness first. Phase 6
//! perf tuning brings the same optimizations.

use alloc::vec;
use alloc::vec::Vec;

use lamquant_weights::generated::focal::{
    bneck_g, bneck_v, dw_gate, focal1_conv, focal2, focal3, premix,
};

// ─── Layer shapes (mirror generated const generics) ─────────────────

const WIDTH: usize = 128;          // focal block width
const T_IN: usize = 313;           // L3 approximation length
const T_FOCAL1: usize = 157;       // (T_IN + STRIDE-1) / STRIDE = (313+1)/2 = 157
const T_FOCAL2: usize = T_FOCAL1;  // focal2 stride 1
const T_LATENT: usize = 79;        // (T_FOCAL2 + STRIDE-1) / STRIDE = (157+1)/2 = 79
const LATENT_DIMS: usize = 32;
const N_INPUT_CHANNELS: usize = 21;

// ─── Helpers ────────────────────────────────────────────────────────

/// Q31 multiply, signed.
#[inline(always)]
fn mul_q31(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 31) as i32
}

/// Saturating cast i32 → i16.
#[inline(always)]
fn sat_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Sigmoid LUT — 257-entry table over [-8, +8] in Q15.
///
/// `sigmoid_lut[i]` = round(sigmoid(-8 + 16*i/256) * 32768) for i ∈ [0, 256].
/// Out-of-range: clamp to 0/32767.
fn sigmoid_q15(x_q15: i32) -> i16 {
    // Map Q15 input to LUT index. Saturate at ±8.
    const MAX_IN: i32 = 8 << 15;
    let xc = x_q15.clamp(-MAX_IN, MAX_IN);
    // Index into 257-entry table over [-8, +8]:  idx = ((xc + 8<<15) * 256) / (16<<15)
    //                                               = (xc + 8<<15) >> 12
    let idx = ((xc + MAX_IN) >> 12) as usize;
    SIGMOID_LUT[idx.min(256)]
}

// 257-entry sigmoid LUT (Q15). Index i covers x = -8 + i*(16/256).
// Computed once at compile time would be ideal; for now use a runtime
// const computed via small generator. Hard-coded to keep no_std clean.
const SIGMOID_LUT: [i16; 257] = generate_sigmoid_lut();

const fn generate_sigmoid_lut() -> [i16; 257] {
    let mut tbl = [0i16; 257];
    let mut i = 0;
    while i < 257 {
        // x = -8 + 16 * i / 256, in fixed point: avoid f64 in const.
        // Approximate sigmoid via piecewise-linear since const fn lacks math.
        // For correctness we just clamp: <-4 → 0, >+4 → 32767, in-between linear.
        let x_e3 = (i as i32 - 128) * 16 * 1000 / 256; // x * 1000 (millivolts)
        let v = if x_e3 < -4000 {
            0
        } else if x_e3 > 4000 {
            32767
        } else {
            // Linear approximation around 0: sigmoid(x) ≈ 0.5 + 0.25*x for |x|<1
            //   ⇒ Q15: 16384 + (8192 * x_e3 / 1000)
            // This is intentionally rough; runtime path will be replaced
            // with a proper PWL or Taylor expansion in Phase 3 polish.
            let v = 16384 + 8 * x_e3;
            if v < 0 {
                0
            } else if v > 32767 {
                32767
            } else {
                v as i32
            }
        };
        tbl[i] = v as i16;
        i += 1;
    }
    tbl
}

/// Per-channel ReLU: clamp to 0..i16::MAX.
#[inline(always)]
fn relu_i32(v: i32) -> i32 {
    if v < 0 {
        0
    } else {
        v
    }
}

/// Apply GroupNorm (per-channel) + ReLU + i16-cast in fused integer form.
///
/// `acc_q31` = accumulator (already alpha-scaled).
/// `norm_w_q7` = i8 weight, `norm_b_q15` = i16 bias.
/// Folded form: `out = relu((acc * w_q7 / 128) + (b << 16))`, saturate to i16.
#[inline(always)]
fn fold_groupnorm_relu(acc_q31: i32, norm_w_q7: i8, norm_b_q15: i16) -> i16 {
    let scaled = ((acc_q31 as i64 * norm_w_q7 as i64) >> 7) as i32;
    let biased = scaled.saturating_add((norm_b_q15 as i32) << 16);
    sat_i16(relu_i32(biased) >> 16) // back to Q15-ish range
}

// ─── Activation buffers ─────────────────────────────────────────────
//
// Single flat `alloc::Vec<i16>` of length `channels * t`, row-major
// (ch-major). Was `Vec<Vec<i16>>` which malloc'd N+1 times per
// `ActBuf::new`; the focal pipeline allocates 8 buffers per `forward()`
// → ~626 mallocs/window. Flat layout drops that to **8** mallocs
// (one per ActBuf), with identical numerical semantics. Cat A6 step 1
// (2026-05-21) — production firmware will swap these for section-
// pinned `static mut` buffers in SRAM6/7 once scheduler integrates.

struct ActBuf {
    channels: usize,
    t: usize,
    /// Row-major: `data[ch * t + t_idx]`.
    data: Vec<i16>,
}

impl ActBuf {
    fn new(channels: usize, t: usize) -> Self {
        // Caller passes compile-time-known dims (`N_INPUT_CHANNELS`,
        // `WIDTH`, `T_IN`, etc.) — overflow is impossible on any
        // supported target. `checked_mul + expect` makes that
        // invariant explicit instead of silently saturating into a
        // catastrophic giant allocation if a future refactor wires
        // runtime dimensions through. (lamu review fix on 88b7868.)
        let len = channels.checked_mul(t)
            .expect("ActBuf dims must fit in usize");
        Self {
            channels,
            t,
            data: vec![0i16; len],
        }
    }

    #[inline]
    fn get(&self, ch: usize, t: usize) -> i16 {
        self.data[ch * self.t + t]
    }

    #[inline]
    fn set(&mut self, ch: usize, t: usize, v: i16) {
        self.data[ch * self.t + t] = v;
    }
}

// ─── Convolution primitives ─────────────────────────────────────────

/// Gather a centered window of activations into a contiguous buffer for
/// the inner `conv1d_channel` kernel. Caller supplies the output buffer.
fn gather_window(
    src: &ActBuf,
    t_center: usize,
    in_channels: usize,
    kernel_size: usize,
    t_max: usize,
    out: &mut [i16],
) {
    let half_k = kernel_size as i32 / 2;
    let mut idx = 0;
    for ic in 0..in_channels {
        for ki in -half_k..=half_k {
            let ti = t_center as i32 + ki;
            out[idx] = if ti >= 0 && (ti as usize) < t_max {
                src.get(ic, ti as usize)
            } else {
                0
            };
            idx += 1;
        }
    }
}

/// Decode a single 2-bit ternary weight from the global packed stream.
///
/// `packed` is the full OUT_CH × IN_CH × K weight tensor packed at 2 bits
/// per weight, 4 weights per byte, in row-major order
/// `[oc][ic][k] → linear index = oc*IN_CH*K + ic*K + k`.
///
/// No per-OC byte alignment assumed (premix has 21*1=21 weights/OC =
/// 5.25 bytes, packs across byte boundaries).
#[inline(always)]
fn ternary_weight_at(packed: &[u8], idx: usize) -> i32 {
    let byte = packed[idx >> 2];
    let shift = (idx & 3) * 2;
    let w = ((byte >> shift) & 0b11) as u32;
    // 00 = 0, 01 = +1, 10 = -1, 11 = 0 (pad). Matches mac_byte_fast.
    let neg = -((w >> 1) as i32);
    let nonzero = ((w & 1) ^ (w >> 1)) as i32;
    // (1 ^ neg) - neg = +1 when w==01, -1 when w==10, ignored when nonzero==0.
    ((1i32 ^ neg) - neg) * nonzero
}

/// Depthwise ternary conv (groups == in_channels == out_channels).
/// One kernel per input channel; no cross-channel mixing.
/// Uses bit-level weight indexing — packed stream may not align to byte
/// boundaries between channels (3 weights/ch = 0.75 bytes for k=3).
fn depthwise_conv_layer(
    input: &ActBuf,
    output: &mut ActBuf,
    n_channels: usize,
    kernel_size: usize,
    t: usize,
    packed: &[u8],
    alphas_q15: &[i16],
) {
    let half_k = kernel_size as i32 / 2;
    for ch in 0..n_channels {
        let alpha_q31 = (alphas_q15[ch] as i32) << 16;
        let ch_weight_base = ch * kernel_size;

        for t_idx in 0..t {
            let mut acc = 0i32;
            for ki in 0..kernel_size {
                let ti = t_idx as i32 + ki as i32 - half_k;
                let a = if ti >= 0 && (ti as usize) < t {
                    input.get(ch, ti as usize) as i32
                } else {
                    0
                };
                let w = ternary_weight_at(packed, ch_weight_base + ki);
                acc = acc.wrapping_add(a.wrapping_mul(w));
            }
            let scaled = mul_q31(acc, alpha_q31);
            output.set(ch, t_idx, sat_i16(scaled >> 16));
        }
    }
}

// focal1 weights are now exported as `focal1_conv`. Layer:
//   TernaryConv1d(21 → 128, k=3, s=2) + GroupNorm + ReLU.
// Wired through `conv_focal_layer` like focal2/focal3.

// ─── INT8 conv (bneck_v) ────────────────────────────────────────────

fn int8_pointwise(
    input: &ActBuf,
    output: &mut ActBuf,
    in_ch: usize,
    out_ch: usize,
    t: usize,
    weights_i8: &[i8],
    scales_q15: &[i16],
) {
    for oc in 0..out_ch {
        let scale_q31 = (scales_q15[oc] as i32) << 16;
        let oc_w_start = oc * in_ch;
        let oc_w = &weights_i8[oc_w_start..oc_w_start + in_ch];

        for t_idx in 0..t {
            let mut acc = 0i32;
            for ic in 0..in_ch {
                acc = acc.wrapping_add((input.get(ic, t_idx) as i32) * (oc_w[ic] as i32));
            }
            // Apply per-OC scale. INT8 weights × i16 act → i32 accumulator.
            let scaled = mul_q31(acc, scale_q31);
            output.set(oc, t_idx, sat_i16(scaled));
        }
    }
}

// ─── Public entry point ─────────────────────────────────────────────

/// Run the TNN encoder forward pass on the L3 approximation.
///
/// Input:  `l3[21][313]` — i16 activations from the lifting front-end.
/// Output: `out_latent[32][79]` — i16 latent ready for FSQ + rANS.
pub fn forward(l3: &[[i16; T_IN]; N_INPUT_CHANNELS], out_latent: &mut [[i16; T_LATENT]; LATENT_DIMS]) {
    // ── premix (21 → 21, k=1, no norm) ──
    let mut act_in = ActBuf::new(N_INPUT_CHANNELS, T_IN);
    for ch in 0..N_INPUT_CHANNELS {
        for t in 0..T_IN {
            act_in.set(ch, t, l3[ch][t]);
        }
    }

    let mut act_premix = ActBuf::new(N_INPUT_CHANNELS, T_IN);
    conv_focal_layer(
        &act_in,
        &mut act_premix,
        premix::IN_CHANNELS,
        premix::OUT_CHANNELS,
        premix::KERNEL_SIZE,
        T_IN,
        T_IN,
        premix::STRIDE,
        premix::PACKED_WEIGHTS.as_slice(),
        &premix::ALPHAS_Q15,
        None, // premix has no GroupNorm
    );
    drop(act_in);

    // ── focal1_conv (21 → 128, k=3, s=2, norm + ReLU) ──
    let mut act_f1 = ActBuf::new(WIDTH, T_FOCAL1);
    conv_focal_layer(
        &act_premix,
        &mut act_f1,
        focal1_conv::IN_CHANNELS,
        focal1_conv::OUT_CHANNELS,
        focal1_conv::KERNEL_SIZE,
        T_IN,
        T_FOCAL1,
        focal1_conv::STRIDE,
        focal1_conv::PACKED_WEIGHTS.as_slice(),
        &focal1_conv::ALPHAS_Q15,
        Some((&focal1_conv::NORM_WEIGHT_Q7, &focal1_conv::NORM_BIAS_Q15)),
    );
    drop(act_premix);

    // ── focal2 (128 → 128, k=5, s=1, norm + ReLU) ──
    let mut act_f2 = ActBuf::new(WIDTH, T_FOCAL2);
    conv_focal_layer(
        &act_f1,
        &mut act_f2,
        focal2::IN_CHANNELS,
        focal2::OUT_CHANNELS,
        focal2::KERNEL_SIZE,
        T_FOCAL1,
        T_FOCAL2,
        focal2::STRIDE,
        focal2::PACKED_WEIGHTS.as_slice(),
        &focal2::ALPHAS_Q15,
        Some((&focal2::NORM_WEIGHT_Q7, &focal2::NORM_BIAS_Q15)),
    );
    drop(act_f1);

    // ── focal3 (128 → 128, k=7, s=2, norm + ReLU) ──
    let mut act_f3 = ActBuf::new(WIDTH, T_LATENT);
    conv_focal_layer(
        &act_f2,
        &mut act_f3,
        focal3::IN_CHANNELS,
        focal3::OUT_CHANNELS,
        focal3::KERNEL_SIZE,
        T_FOCAL2,
        T_LATENT,
        focal3::STRIDE,
        focal3::PACKED_WEIGHTS.as_slice(),
        &focal3::ALPHAS_Q15,
        Some((&focal3::NORM_WEIGHT_Q7, &focal3::NORM_BIAS_Q15)),
    );
    drop(act_f2);

    // ── dw_gate (depthwise 128, k=3, s=1) ──
    let mut act_dw = ActBuf::new(WIDTH, T_LATENT);
    depthwise_conv_layer(
        &act_f3,
        &mut act_dw,
        WIDTH,
        dw_gate::KERNEL_SIZE,
        T_LATENT,
        dw_gate::PACKED_WEIGHTS.as_slice(),
        &dw_gate::ALPHAS_Q15,
    );

    // ── bneck_g (gate path: ternary 128 → 32, k=1) ──
    let mut act_bg = ActBuf::new(LATENT_DIMS, T_LATENT);
    conv_focal_layer(
        &act_dw,
        &mut act_bg,
        bneck_g::IN_CHANNELS,
        bneck_g::OUT_CHANNELS,
        bneck_g::KERNEL_SIZE,
        T_LATENT,
        T_LATENT,
        bneck_g::STRIDE,
        bneck_g::PACKED_WEIGHTS.as_slice(),
        &bneck_g::ALPHAS_Q15,
        None, // bneck_g has no GroupNorm
    );

    // ── bneck_v (value path: INT8 128 → 32, k=1) ──
    let mut act_bv = ActBuf::new(LATENT_DIMS, T_LATENT);
    int8_pointwise(
        &act_dw,
        &mut act_bv,
        WIDTH,
        LATENT_DIMS,
        T_LATENT,
        &bneck_v::WEIGHTS_RAW,
        &bneck_v::SCALES_Q15,
    );
    drop(act_dw);

    // ── GLU bottleneck: latent = bneck_v * sigmoid(bneck_g) ──
    for d in 0..LATENT_DIMS {
        for t in 0..T_LATENT {
            let g = sigmoid_q15((act_bg.get(d, t) as i32) << 15);
            let v = act_bv.get(d, t) as i32;
            // (v * g) is i32 * i16 → keep in i32 then >> 15 to land back in i16 range.
            let prod = (v * g as i32) >> 15;
            out_latent[d][t] = sat_i16(prod);
        }
    }
}

/// Generic ternary conv1d layer using bit-level weight indexing.
///
/// Weights are read from the GLOBAL packed stream `packed` at offset
/// `oc * in_ch * kernel_size + ic * kernel_size + ki` (in 2-bit weights,
/// not bytes). This handles non-byte-aligned per-OC layouts (premix).
fn conv_focal_layer(
    input: &ActBuf,
    output: &mut ActBuf,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    t_in: usize,
    t_out: usize,
    stride: usize,
    packed: &[u8],
    alphas_q15: &[i16],
    norm: Option<(&[i8], &[i16])>,
) {
    let half_k = kernel_size as i32 / 2;
    let weights_per_oc = in_ch * kernel_size;

    for oc in 0..out_ch {
        let alpha_q31 = (alphas_q15[oc] as i32) << 16;
        let oc_weight_base = oc * weights_per_oc;

        for t_out_idx in 0..t_out {
            let t_center = (t_out_idx * stride) as i32;
            let mut acc: i32 = 0;
            for ic in 0..in_ch {
                let ic_weight_base = oc_weight_base + ic * kernel_size;
                for ki in 0..kernel_size {
                    let ti = t_center + ki as i32 - half_k;
                    let a = if ti >= 0 && (ti as usize) < t_in {
                        input.get(ic, ti as usize) as i32
                    } else {
                        0
                    };
                    let w = ternary_weight_at(packed, ic_weight_base + ki);
                    acc = acc.wrapping_add(a.wrapping_mul(w));
                }
            }
            // Apply LSQ alpha (Q31 scaling).
            let scaled = mul_q31(acc, alpha_q31);
            let v = if let Some((nw, nb)) = norm {
                fold_groupnorm_relu(scaled, nw[oc], nb[oc])
            } else {
                sat_i16(relu_i32(scaled) >> 16)
            };
            output.set(oc, t_out_idx, v);
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn forward_zero_input_finite() {
        let l3 = [[0i16; T_IN]; N_INPUT_CHANNELS];
        let mut out = [[0i16; T_LATENT]; LATENT_DIMS];
        forward(&l3, &mut out);
        // All values must be in i16 range — sat_i16 guarantees this; just
        // assert no panic and structural shape.
        for d in 0..LATENT_DIMS {
            for t in 0..T_LATENT {
                let _ = out[d][t]; // touch every element
            }
        }
    }

    #[test]
    fn forward_dc_input_finite() {
        let mut l3 = [[0i16; T_IN]; N_INPUT_CHANNELS];
        for ch in 0..N_INPUT_CHANNELS {
            for t in 0..T_IN {
                l3[ch][t] = 1000;
            }
        }
        let mut out = [[0i16; T_LATENT]; LATENT_DIMS];
        forward(&l3, &mut out);
    }

    #[test]
    fn forward_ramp_input_finite() {
        let mut l3 = [[0i16; T_IN]; N_INPUT_CHANNELS];
        for ch in 0..N_INPUT_CHANNELS {
            for t in 0..T_IN {
                l3[ch][t] = ((t as i32 * 30) - 4000) as i16;
            }
        }
        let mut out = [[0i16; T_LATENT]; LATENT_DIMS];
        forward(&l3, &mut out);
    }

    #[test]
    fn sigmoid_lut_monotonic() {
        for i in 1..257 {
            assert!(SIGMOID_LUT[i] >= SIGMOID_LUT[i - 1]);
        }
        // Endpoints
        assert_eq!(SIGMOID_LUT[0], 0);
        assert_eq!(SIGMOID_LUT[256], 32767);
    }
}
