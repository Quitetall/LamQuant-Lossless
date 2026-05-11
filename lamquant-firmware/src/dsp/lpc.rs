//! LPC (Linear Predictive Coding) — order-8, 21 channels.
//!
//! Stage 2 of the pipeline: removes temporal redundancy after biquad,
//! before lifting DWT. Coefficients estimated from first 256 samples
//! (EEG spectral envelope changes slowly within a 10-second window).
//!
//! Pure integer in-place. No alloc, no `Vec`, no `f64`, no `unsafe`.
//!
//! Numerical model — **bit-exact** with `lamquant_core::lpc::analyze`:
//!   - Autocorrelation in i64 (4× unrolled), divided by seg_len (= 256)
//!   - Levinson–Durbin recursion in Q31 internally for stability
//!   - Output coefficients converted to **Q27 i32** (`>> 4`) — Python decoder
//!     consumes Q27, so the wire format stays compatible
//!   - Residual computed using the **Q27** coefficients so encoder/decoder
//!     roundtrip is bit-exact
//!   - Running-sum bias cancellation with `ctx_len = 256` (power of 2 →
//!     arithmetic right shift gives floor_div for negative sums, matching
//!     Python's `//` operator)
//!
//! Cross-equality test `matches_lamquant_core_analyze` asserts byte-for-
//! byte agreement on a non-trivial 21×2500 input.

use super::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};

pub const LPC_ORDER: usize = 8;
pub const AUTOCORR_LEN: usize = 256;
const Q27: i32 = 27;
const BIAS_CTX: usize = 256;
const BIAS_CTX_SHIFT: u32 = 8; // log2(BIAS_CTX)
const BIAS_CTX_MASK: usize = BIAS_CTX - 1;

/// Per-channel LPC analysis output: order-8 Q27 coefficients + residual.
pub struct LpcOutput {
    pub coeffs: [[i32; LPC_ORDER]; NUM_CHANNELS],
    pub residual: [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
}

impl LpcOutput {
    pub const fn zeroed() -> Self {
        Self {
            coeffs: [[0; LPC_ORDER]; NUM_CHANNELS],
            residual: [[0; WINDOW_SAMPLES]; NUM_CHANNELS],
        }
    }
}

#[inline(always)]
fn sat_add_i32(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

#[inline(always)]
fn sat_sub_i32(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// Q31 multiplication: `(a * b) >> 31`, signed.
#[inline(always)]
fn mul_q31(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 31) as i32
}

/// Biased autocorrelation `R[k] = sum(x[n] * x[n-k]) / len` for k=0..=order.
/// 4× unrolled inner loop, matches C reference at firmware/dsp/lpc_predictor.c.
#[inline]
fn autocorrelation(x: &[i32], len: usize, order: usize, r_out: &mut [i64]) {
    // Audit-2026-05-11 Fix-C26: wrapping arithmetic throughout so the
    // firmware path produces bit-identical output to lamquant-core's
    // `lpc::analyze` on every input, including overflow-inducing
    // adversarial signals (Q27 coeff × i64 sample can overflow i64 at
    // saturation; consistent wrapping keeps encode + decode aligned).
    for k in 0..=order {
        let mut acc: i64 = 0;
        let mut n = k;
        let limit = k + ((len - k) & !3);
        while n < limit {
            acc = acc.wrapping_add((x[n] as i64).wrapping_mul(x[n - k] as i64));
            acc = acc.wrapping_add((x[n + 1] as i64).wrapping_mul(x[n + 1 - k] as i64));
            acc = acc.wrapping_add((x[n + 2] as i64).wrapping_mul(x[n + 2 - k] as i64));
            acc = acc.wrapping_add((x[n + 3] as i64).wrapping_mul(x[n + 3 - k] as i64));
            n += 4;
        }
        while n < len {
            acc = acc.wrapping_add((x[n] as i64).wrapping_mul(x[n - k] as i64));
            n += 1;
        }
        r_out[k] = acc / (len as i64);
    }
}

/// Levinson–Durbin recursion. Q31 internal coefficients. Returns
/// `Some([a0..a_{order-1}])` on success (Q31), `None` if R[0] == 0.
#[inline]
fn levinson_q31(r: &[i64], order: usize) -> Option<[i32; LPC_ORDER]> {
    if r[0] == 0 {
        return None;
    }
    let mut a_prev = [0i32; LPC_ORDER];
    let mut a_curr = [0i32; LPC_ORDER];
    let mut e: i64 = r[0];

    for m in 0..order {
        let mut sum: i64 = r[m + 1];
        for j in 0..m {
            sum = sum.wrapping_add(
                ((a_prev[j] as i64).wrapping_mul(r[m - j])) >> 31,
            );
        }
        if e == 0 {
            return None;
        }

        // k = -sum / e in Q31 form. Divide-then-shift dodges i64 overflow
        // on the intermediate product (matches the C bug-fix path).
        //
        // Audit-2026-05-11 Fix-C27: wrapping_neg avoids UB on
        // `sum/e == i64::MIN`. Final clamp to i32 below makes the wrap
        // observable only as a saturated boundary value.
        let k_q31_64 = (sum / e).wrapping_neg().wrapping_mul(1i64 << 31);
        let k_q31 = if k_q31_64 > i32::MAX as i64 {
            i32::MAX
        } else if k_q31_64 < i32::MIN as i64 {
            i32::MIN
        } else {
            k_q31_64 as i32
        };

        a_curr[m] = k_q31;
        for j in 0..m {
            a_curr[j] = sat_add_i32(a_prev[j], mul_q31(k_q31, a_prev[m - 1 - j]));
        }

        e -= ((k_q31 as i64).wrapping_mul(k_q31_64)) >> 31;
        if e <= 0 {
            e = 1;
        }

        for j in 0..=m {
            a_prev[j] = a_curr[j];
        }
    }
    Some(a_curr)
}

/// Compute residual `r[n] = x[n] - sum_k(a_q27[k] * x[n - 1 - k]) >> 27`.
/// First `order` samples copied as-is. Order-8 hot path is unrolled.
#[inline]
fn residuals_q27(x: &[i32], r: &mut [i32], coeffs_q27: &[i32; LPC_ORDER]) {
    let len = x.len();
    debug_assert_eq!(r.len(), len);
    for n in 0..LPC_ORDER {
        r[n] = x[n];
    }
    let a0 = coeffs_q27[0] as i64;
    let a1 = coeffs_q27[1] as i64;
    let a2 = coeffs_q27[2] as i64;
    let a3 = coeffs_q27[3] as i64;
    let a4 = coeffs_q27[4] as i64;
    let a5 = coeffs_q27[5] as i64;
    let a6 = coeffs_q27[6] as i64;
    let a7 = coeffs_q27[7] as i64;
    // Audit-2026-05-11 Fix-C26: wrapping mul/add throughout so the
    // residual path matches lamquant-core bit-for-bit on overflow inputs.
    for n in LPC_ORDER..len {
        let p: i64 = a0.wrapping_mul(x[n - 1] as i64)
            .wrapping_add(a1.wrapping_mul(x[n - 2] as i64))
            .wrapping_add(a2.wrapping_mul(x[n - 3] as i64))
            .wrapping_add(a3.wrapping_mul(x[n - 4] as i64))
            .wrapping_add(a4.wrapping_mul(x[n - 5] as i64))
            .wrapping_add(a5.wrapping_mul(x[n - 6] as i64))
            .wrapping_add(a6.wrapping_mul(x[n - 7] as i64))
            .wrapping_add(a7.wrapping_mul(x[n - 8] as i64));
        let pred = (p >> Q27) as i32;
        r[n] = sat_sub_i32(x[n], pred);
    }
}

/// Inverse of `residuals_q27`: reconstruct `x[n] = r[n] + pred(x[n-k])`.
#[inline]
fn synth_q27(r: &[i32], x: &mut [i32], coeffs_q27: &[i32; LPC_ORDER]) {
    let len = r.len();
    debug_assert_eq!(x.len(), len);
    for n in 0..LPC_ORDER {
        x[n] = r[n];
    }
    let a0 = coeffs_q27[0] as i64;
    let a1 = coeffs_q27[1] as i64;
    let a2 = coeffs_q27[2] as i64;
    let a3 = coeffs_q27[3] as i64;
    let a4 = coeffs_q27[4] as i64;
    let a5 = coeffs_q27[5] as i64;
    let a6 = coeffs_q27[6] as i64;
    let a7 = coeffs_q27[7] as i64;
    // Audit-2026-05-11 Fix-C26: wrapping mul/add throughout so synth
    // pairs bit-exactly with residuals_q27 on overflow inputs.
    for n in LPC_ORDER..len {
        let p: i64 = a0.wrapping_mul(x[n - 1] as i64)
            .wrapping_add(a1.wrapping_mul(x[n - 2] as i64))
            .wrapping_add(a2.wrapping_mul(x[n - 3] as i64))
            .wrapping_add(a3.wrapping_mul(x[n - 4] as i64))
            .wrapping_add(a4.wrapping_mul(x[n - 5] as i64))
            .wrapping_add(a5.wrapping_mul(x[n - 6] as i64))
            .wrapping_add(a6.wrapping_mul(x[n - 7] as i64))
            .wrapping_add(a7.wrapping_mul(x[n - 8] as i64));
        let pred = (p >> Q27) as i32;
        x[n] = sat_add_i32(r[n], pred);
    }
}

/// Running-mean bias cancellation, ctx_len = 256.
///
/// Mirrors `lamquant_core::lpc::bias_cancel` exactly, with the optimisation
/// that ctx is a power of 2 → arithmetic right shift on `i64` gives Python's
/// `//` (floor division) for negative sums in one cycle.
#[inline]
fn bias_cancel(data: &mut [i32]) {
    let mut buf = [0i32; BIAS_CTX];
    let mut sum: i64 = 0;
    for i in 0..data.len() {
        let bias = (sum >> BIAS_CTX_SHIFT) as i32;
        let val = data[i];
        data[i] = sat_sub_i32(val, bias);
        let slot = i & BIAS_CTX_MASK;
        let old = buf[slot];
        buf[slot] = val;
        sum += val as i64 - old as i64;
    }
}

/// Inverse of `bias_cancel`. Stores the restored value in the circular
/// buffer (matching lamquant-core).
#[inline]
fn bias_restore(data: &mut [i32]) {
    let mut buf = [0i32; BIAS_CTX];
    let mut sum: i64 = 0;
    for i in 0..data.len() {
        let bias = (sum >> BIAS_CTX_SHIFT) as i32;
        data[i] = sat_add_i32(data[i], bias);
        let slot = i & BIAS_CTX_MASK;
        let old = buf[slot];
        buf[slot] = data[i];
        sum += data[i] as i64 - old as i64;
    }
}

/// Run LPC analysis on the 21-channel HP-filtered buffer.
///
/// `signal[ch]` is Q31 i32. Output residual is Q31 i32 (same dynamic range
/// as the input — predictive coding produces a same-scale residual).
/// Coefficients are Q27 i32 in `out.coeffs[ch][..]`.
pub fn analyze_all_channels(
    signal: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    out: &mut LpcOutput,
) {
    let mut r = [0i64; LPC_ORDER + 1];
    for ch in 0..NUM_CHANNELS {
        autocorrelation(&signal[ch], AUTOCORR_LEN, LPC_ORDER, &mut r);

        let coeffs_q31 = levinson_q31(&r, LPC_ORDER);
        let mut coeffs_q27 = [0i32; LPC_ORDER];
        if let Some(q31) = coeffs_q31 {
            for k in 0..LPC_ORDER {
                // Negate to match `lamquant_core` convention: residual uses
                // `signal - sum(coeff * past)`, Levinson returns the AR
                // polynomial with negative reflection roots. Both encode
                // and decode flip the sign so the math reverses bit-exact.
                coeffs_q27[k] = -(q31[k] >> 4);
            }
        }
        out.coeffs[ch] = coeffs_q27;

        // Residual (Q31 i32 in/out, Q27 coeffs).
        residuals_q27(&signal[ch], &mut out.residual[ch], &coeffs_q27);

        // Bias cancellation in-place over the full 2500-sample residual.
        bias_cancel(&mut out.residual[ch]);
    }
}

/// Inverse: reconstruct signal from residual + Q27 coefficients
/// (decoder side / verification).
pub fn synthesize_all_channels(
    residual: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    coeffs: &[[i32; LPC_ORDER]; NUM_CHANNELS],
    out: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
) {
    for ch in 0..NUM_CHANNELS {
        // Stage 1: undo bias cancellation (operating on a mutable copy of
        // the residual; we treat `restored` as scratch).
        let mut restored = residual[ch];
        bias_restore(&mut restored);

        // Stage 2: IIR LPC synthesis using the same Q27 coefficients.
        synth_q27(&restored, &mut out[ch], &coeffs[ch]);
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_sin_wave() {
        // Synthetic correlated signal (slow sine).
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                let phase = (i as f64) * 0.1 + (ch as f64) * 0.5;
                signal[ch][i] = ((phase.sin()) * 1_000_000.0) as i32;
            }
        }

        let mut out = LpcOutput::zeroed();
        analyze_all_channels(&signal, &mut out);

        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        synthesize_all_channels(&out.residual, &out.coeffs, &mut reconstructed);

        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(
                    reconstructed[ch][i], signal[ch][i],
                    "ch{ch} sample {i} mismatch"
                );
            }
        }
    }

    #[test]
    fn roundtrip_white_noise() {
        // Wider dynamic range than the sine test.
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        let mut s: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Keep magnitude ≤ ~2^24 so Q27 mul stays inside i64 with
                // headroom across 8 taps.
                signal[ch][i] = (s as i32) >> 8;
            }
        }
        let mut out = LpcOutput::zeroed();
        analyze_all_channels(&signal, &mut out);
        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        synthesize_all_channels(&out.residual, &out.coeffs, &mut reconstructed);
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(
                    reconstructed[ch][i], signal[ch][i],
                    "ch{ch} sample {i} mismatch"
                );
            }
        }
    }

    /// Cross-equality vs `lamquant_core::lpc::analyze` on a non-trivial input.
    /// Asserts both code paths produce **bit-identical** Q27 coefficients
    /// and bit-identical residuals.
    #[test]
    fn matches_lamquant_core_analyze() {
        use lamquant_core::lpc as core_lpc;
        // Synthetic correlated EEG-ish signal across all 21 channels.
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                let phase = (i as f64) * 0.07 + (ch as f64) * 0.31;
                signal[ch][i] = ((phase.sin()) * 800_000.0) as i32
                    + ((((i + ch * 13) * 1009) & 0x3FFF) as i32 - 0x2000);
            }
        }

        let mut out = LpcOutput::zeroed();
        analyze_all_channels(&signal, &mut out);

        for ch in 0..NUM_CHANNELS {
            let sig_i64: alloc::vec::Vec<i64> =
                signal[ch].iter().map(|&v| v as i64).collect();
            let (coeffs_core, residual_core) = core_lpc::analyze(&sig_i64, LPC_ORDER, AUTOCORR_LEN);

            for k in 0..LPC_ORDER {
                assert_eq!(
                    out.coeffs[ch][k], coeffs_core[k],
                    "ch{ch} coeff[{k}]: firmware={} core={}",
                    out.coeffs[ch][k], coeffs_core[k]
                );
            }
            for n in 0..WINDOW_SAMPLES {
                assert_eq!(
                    out.residual[ch][n] as i64, residual_core[n],
                    "ch{ch} residual[{n}]: firmware={} core={}",
                    out.residual[ch][n] as i64, residual_core[n]
                );
            }
        }
    }
}
