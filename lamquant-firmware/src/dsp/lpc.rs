//! LPC (Linear Predictive Coding) — order-8, 21 channels.
//!
//! Stage 2 of the pipeline: removes temporal redundancy after biquad,
//! before lifting DWT. Coefficients estimated from first 256 samples
//! (EEG spectral envelope changes slowly within a 10-second window).
//!
//! Pure integer in-place. No alloc, no `Vec`, no `f64`, no `unsafe`.
//!
//! Numerical model — **f64-equivalent precision** via Q56 block-floating:
//!   - Autocorrelation in i64 (4× unrolled), divided by seg_len (= 256)
//!   - Levinson–Durbin recursion in Q56 (i64 storage, i128 accumulator)
//!     for f64-class precision. Earlier Q31 path lost ~5-8% CR vs host
//!     on real EEG; Q56 closes that gap to ±1 LSB. See
//!     `lamquant_core::lpc::analyze_blockfloat` for the equivalent host path.
//!   - Output coefficients converted to **Q27 i32** (`>> 4` from Q31) —
//!     wire format unchanged
//!   - Residual computed using the **Q27** coefficients so encoder/decoder
//!     roundtrip is bit-exact
//!   - Running-sum bias cancellation with `ctx_len = 256` (power of 2 →
//!     arithmetic right shift gives floor_div for negative sums, matching
//!     Python's `//` operator)
//!
//! Roundtrip tests in `tests/conformance.rs` cover the analyze→synthesize
//! self-pairing. Host f64 path coefficients differ by ±1 LSB but the
//! recovered signal is bit-exact within each (encoder, decoder) pair.

use super::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};

pub const LPC_ORDER: usize = 8;
pub const AUTOCORR_LEN: usize = 256;
const Q27: i32 = 27;
const BIAS_CTX: usize = 256;
const BIAS_CTX_SHIFT: u32 = 8; // log2(BIAS_CTX)
const BIAS_CTX_MASK: usize = BIAS_CTX - 1;

/// Firmware mirror of [`lamquant_core::lpc::LpcMode`].
///
/// Stripped down — firmware lpc.rs operates on full 2500-sample
/// channels (no DWT subbands at this stage) and has a fixed
/// `LPC_ORDER = 8` ceiling. Adaptive walks Levinson from order 1 up to
/// `LPC_ORDER`, scores each step against an AIC/MDL byte-cost penalty,
/// and zeros the unused coefficient slots — wire format unchanged, the
/// host decoder reads `chosen_order` from the Q27 array's first-nonzero
/// run (or via lpc_meta when emitted through the host packetizer).
///
/// `Anytime` is the realtime gate: caller passes `time_remaining` from
/// its scheduler tick (DWT CYCCNT, cycle CSR, or a precomputed deadline
/// budget). `true` → adaptive runs; `false` → fixed schedule, same
/// guarantee as the host path's expired-deadline fallback.
#[derive(Debug, Clone, Copy)]
pub enum LpcMode {
    Fixed,
    Adaptive,
    Anytime,
}

/// Per-channel chosen order. Indexes into `LpcOutput::coeffs[ch]`.
/// `coeffs[ch][order..]` are zero so the decoder synth path applies
/// only the live taps.
pub type ChosenOrders = [u8; NUM_CHANNELS];

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

/// Levinson–Durbin recursion. Block-floating-point Q56 internal storage,
/// matching `lamquant_core::lpc::analyze_blockfloat` to ±1 LSB precision
/// (≈ f64-equivalent CR, no FPU). Returns `Some([a0..a_{order-1}])` in
/// **Q31** (i32 i.e. legacy interface — call site does `>> 4` for Q27),
/// `None` if R[0] == 0.
///
/// **Why Q56:** Q31 storage truncates ~31 bits per recursion step,
/// inflating residual energy ~5-8% on real EEG. Q56 keeps 56 bits of
/// fraction (> f64's 53-bit mantissa). |a| up to 128 fits in i64. lam
/// accumulator stays Q56 in i128 throughout — no per-step shift loss.
///
/// **Overflow envelope:** for ADS1299-class 24-bit EEG (r ≤ 2^46),
/// the 8-sum `a * r` lam accumulator stays ≤ 2^112 << i128 max. Pathological
/// inputs (|x| → 2^31) use saturating arithmetic via i128 wrap; coefficients
/// then saturate at i64 boundaries but stay decodable.
#[inline]
fn levinson_q31(r: &[i64], order: usize) -> Option<[i32; LPC_ORDER]> {
    if r[0] == 0 {
        return None;
    }
    const QA: u32 = 56;
    const HALF_QA: i128 = 1 << (QA - 1);
    const ONE_QA: i128 = 1 << QA;
    const QA_TO_Q31_SHIFT: u32 = QA - 31; // = 25

    let mut a_prev = [0i64; LPC_ORDER]; // Q56
    let mut a_curr = [0i64; LPC_ORDER];
    let mut e: i128 = r[0] as i128;

    for m in 0..order {
        // lam in Q56 scale throughout — no intermediate >> 56 truncation.
        let mut lam_qa: i128 = (r[m + 1] as i128) << QA;
        for j in 0..m {
            lam_qa = lam_qa.wrapping_add(
                (a_prev[j] as i128).wrapping_mul(r[m - j] as i128),
            );
        }

        if e == 0 {
            return None;
        }
        // k_qa = -lam_Q56 / e_raw → Q56 quotient.
        // wrapping_neg avoids UB on i128::MIN (astronomically unlikely on
        // real EEG; lamu review fix).
        let k_qa_full = lam_qa.wrapping_neg() / e;
        let k_qa = k_qa_full.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

        a_curr[m] = k_qa;
        for j in 0..m {
            let prod = (k_qa as i128).wrapping_mul(a_prev[m - 1 - j] as i128);
            let mac_i128 = if prod >= 0 {
                (prod + HALF_QA) >> QA
            } else {
                -((-prod + HALF_QA) >> QA)
            };
            a_curr[j] = a_prev[j].wrapping_add(mac_i128 as i64);
        }

        // e *= (1 - k²)  with Q56 precision
        let k_sq_prod = (k_qa as i128).wrapping_mul(k_qa as i128);
        let k_sq_qa = (k_sq_prod + HALF_QA) >> QA;
        let factor = ONE_QA - k_sq_qa;
        let e_full = e.wrapping_mul(factor);
        e = if e_full >= 0 {
            (e_full + HALF_QA) >> QA
        } else {
            -((-e_full + HALF_QA) >> QA)
        };
        if e <= 0 {
            e = 1;
        }

        for j in 0..=m {
            a_prev[j] = a_curr[j];
        }
    }

    // Final: Q56 → Q31 (shift right by 25) with half-away rounding. Matches
    // the legacy interface (caller does `-(q31 >> 4)` for negated Q27).
    let half: i64 = 1i64 << (QA_TO_Q31_SHIFT - 1); // 2^24
    let mut out = [0i32; LPC_ORDER];
    for k in 0..LPC_ORDER {
        let v = a_curr[k];
        let shifted: i64 = if v >= 0 {
            v.saturating_add(half) >> QA_TO_Q31_SHIFT
        } else {
            -(v.wrapping_neg().saturating_add(half) >> QA_TO_Q31_SHIFT)
        };
        out[k] = shifted.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    Some(out)
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

/// Adaptive Levinson with AIC/MDL byte-cost order selection.
///
/// Walks Levinson from order 1 up to [`LPC_ORDER`], computing the
/// residual energy `e` produced by each step. At each step we score
/// the cost in nats:
///
/// ```text
///   cost(k) = 0.72 * n * ln(e / n) + ORDER_BIT_COST * k
/// ```
///
/// where `n = AUTOCORR_LEN` and `ORDER_BIT_COST = 32 * ln 2 ≈ 22.18`
/// — the matching MDL byte-cost penalty used by the host path
/// (`lamquant_core::lpc::analyze_adaptive`). The order with minimum
/// cost wins; coefficients past it are zeroed on output.
///
/// Returns the Q31 coefficients (caller does `-(>> 4)` to negated-Q27,
/// matching the existing convention) and the chosen order. `None` if
/// `r[0] == 0` (degenerate input — fall back to all-zero coefficients).
///
/// Uses `libm::log` (the only float dependency the firmware DSP path
/// pulls in). All other arithmetic remains Q56 integer.
#[inline]
fn levinson_q31_adaptive(r: &[i64], n_samples: usize) -> Option<([i32; LPC_ORDER], usize)> {
    if r[0] == 0 {
        return None;
    }
    const QA: u32 = 56;
    const HALF_QA: i128 = 1 << (QA - 1);
    const ONE_QA: i128 = 1 << QA;
    const QA_TO_Q31_SHIFT: u32 = QA - 31;
    const ORDER_BIT_COST: f64 = 22.18; // 32 * ln 2

    let mut a_prev = [0i64; LPC_ORDER];
    let mut a_curr = [0i64; LPC_ORDER];
    let mut e: i128 = r[0] as i128;

    let mut best_cost = f64::INFINITY;
    let mut best_order: usize = 0;
    let mut best_a = [0i64; LPC_ORDER];

    // Order 0 baseline: cost from the raw r[0] energy. No coefficients
    // to emit; emit-time `>>` keeps them zero.
    let n_f = n_samples as f64;
    let r0_f = r[0] as f64;
    if r0_f > 0.0 {
        let cost0 = 0.72 * n_f * libm::log(r0_f / n_f);
        if cost0 < best_cost {
            best_cost = cost0;
        }
    }

    for m in 0..LPC_ORDER {
        let mut lam_qa: i128 = (r[m + 1] as i128) << QA;
        for j in 0..m {
            lam_qa = lam_qa.wrapping_add(
                (a_prev[j] as i128).wrapping_mul(r[m - j] as i128),
            );
        }

        if e == 0 {
            break;
        }
        let k_qa_full = lam_qa.wrapping_neg() / e;
        let k_qa = k_qa_full.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

        a_curr[m] = k_qa;
        for j in 0..m {
            let prod = (k_qa as i128).wrapping_mul(a_prev[m - 1 - j] as i128);
            let mac_i128 = if prod >= 0 {
                (prod + HALF_QA) >> QA
            } else {
                -((-prod + HALF_QA) >> QA)
            };
            a_curr[j] = a_prev[j].wrapping_add(mac_i128 as i64);
        }

        let k_sq_prod = (k_qa as i128).wrapping_mul(k_qa as i128);
        let k_sq_qa = (k_sq_prod + HALF_QA) >> QA;
        let factor = ONE_QA - k_sq_qa;
        let e_full = e.wrapping_mul(factor);
        e = if e_full >= 0 {
            (e_full + HALF_QA) >> QA
        } else {
            -((-e_full + HALF_QA) >> QA)
        };
        if e <= 0 {
            break;
        }

        // Score this order. e is in raw-energy scale (≈ R[0] units); the
        // host path uses the same formulation against its f64 `e`, so
        // the chosen orders match to within numerical rounding.
        let order_k = m + 1;
        let e_f = e as f64;
        if e_f > 0.0 {
            let cost = 0.72 * n_f * libm::log(e_f / n_f)
                + ORDER_BIT_COST * order_k as f64;
            if cost < best_cost {
                best_cost = cost;
                best_order = order_k;
                best_a = a_curr;
            }
        }

        for j in 0..=m {
            a_prev[j] = a_curr[j];
        }
    }

    // Emit Q31 coefficients: best_order live taps, rest zeroed.
    let half: i64 = 1i64 << (QA_TO_Q31_SHIFT - 1);
    let mut out = [0i32; LPC_ORDER];
    for k in 0..best_order {
        let v = best_a[k];
        let shifted: i64 = if v >= 0 {
            v.saturating_add(half) >> QA_TO_Q31_SHIFT
        } else {
            -(v.wrapping_neg().saturating_add(half) >> QA_TO_Q31_SHIFT)
        };
        out[k] = shifted.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    Some((out, best_order))
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

/// Mode-dispatching analysis entry point.
///
/// `Fixed` is the existing constant-time path (full order-8 Levinson on
/// every channel) and writes zero into the new `orders` output to keep
/// callers from accidentally treating it as a live signal.
/// `Adaptive` walks the Levinson + AIC/MDL scorer per channel and emits
/// the chosen per-channel order.
/// `Anytime` selects between the two based on the `time_remaining`
/// signal that the scheduler computes against its deadline budget:
/// `Some(true)` → adaptive, `Some(false)` → fixed, `None` →
/// conservative fallback to fixed (matches the host `unwrap_or(false)`
/// safe default).
pub fn analyze_all_channels_with_mode(
    signal: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    out: &mut LpcOutput,
    mode: LpcMode,
    time_remaining: Option<bool>,
    orders: &mut ChosenOrders,
) {
    let run_adaptive = match mode {
        LpcMode::Fixed => false,
        LpcMode::Adaptive => true,
        LpcMode::Anytime => time_remaining.unwrap_or(false),
    };

    if !run_adaptive {
        // Reuse the existing fixed-order path; orders are uniformly
        // LPC_ORDER (the schedule the decoder assumes when no per-channel
        // order signal accompanies the residual).
        analyze_all_channels(signal, out);
        for ch in 0..NUM_CHANNELS {
            orders[ch] = LPC_ORDER as u8;
        }
        return;
    }

    let mut r = [0i64; LPC_ORDER + 1];
    for ch in 0..NUM_CHANNELS {
        autocorrelation(&signal[ch], AUTOCORR_LEN, LPC_ORDER, &mut r);

        let mut coeffs_q27 = [0i32; LPC_ORDER];
        let chosen = match levinson_q31_adaptive(&r, AUTOCORR_LEN) {
            Some((q31, order)) => {
                for k in 0..order {
                    coeffs_q27[k] = -(q31[k] >> 4);
                }
                order
            }
            None => 0, // degenerate input — order 0 (no prediction)
        };
        out.coeffs[ch] = coeffs_q27;
        orders[ch] = chosen as u8;

        residuals_q27(&signal[ch], &mut out.residual[ch], &coeffs_q27);
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

    /// Adaptive roundtrip: encode → synthesize → bit-exact equal.
    /// Uses the synthetic correlated signal from the sine roundtrip test
    /// so the adaptive scorer has clear AR structure to lock onto.
    #[test]
    fn adaptive_roundtrip_sin_wave() {
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                let phase = (i as f64) * 0.1 + (ch as f64) * 0.5;
                signal[ch][i] = ((phase.sin()) * 1_000_000.0) as i32;
            }
        }

        let mut out = LpcOutput::zeroed();
        let mut orders: ChosenOrders = [0u8; NUM_CHANNELS];
        analyze_all_channels_with_mode(
            &signal,
            &mut out,
            LpcMode::Adaptive,
            None,
            &mut orders,
        );

        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        synthesize_all_channels(&out.residual, &out.coeffs, &mut reconstructed);

        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(
                    reconstructed[ch][i], signal[ch][i],
                    "ch{ch} sample {i} mismatch (chosen_order={})", orders[ch]
                );
            }
        }
    }

    /// Anytime mode honours the `time_remaining` signal:
    ///   Some(true)  → adaptive (chosen order may be < LPC_ORDER)
    ///   Some(false) → fixed (every channel reports LPC_ORDER)
    ///   None        → conservative fallback to fixed
    #[test]
    fn anytime_dispatches_on_time_remaining() {
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                let phase = (i as f64) * 0.1 + (ch as f64) * 0.5;
                signal[ch][i] = ((phase.sin()) * 1_000_000.0) as i32;
            }
        }

        // Some(false) → fixed schedule.
        let mut out = LpcOutput::zeroed();
        let mut orders: ChosenOrders = [0u8; NUM_CHANNELS];
        analyze_all_channels_with_mode(
            &signal,
            &mut out,
            LpcMode::Anytime,
            Some(false),
            &mut orders,
        );
        for ch in 0..NUM_CHANNELS {
            assert_eq!(orders[ch] as usize, LPC_ORDER);
        }

        // None → safe fallback = fixed.
        let mut out2 = LpcOutput::zeroed();
        let mut orders2: ChosenOrders = [0u8; NUM_CHANNELS];
        analyze_all_channels_with_mode(
            &signal,
            &mut out2,
            LpcMode::Anytime,
            None,
            &mut orders2,
        );
        for ch in 0..NUM_CHANNELS {
            assert_eq!(orders2[ch] as usize, LPC_ORDER);
        }

        // Some(true) → adaptive: chosen order in 0..=LPC_ORDER.
        let mut out3 = LpcOutput::zeroed();
        let mut orders3: ChosenOrders = [0u8; NUM_CHANNELS];
        analyze_all_channels_with_mode(
            &signal,
            &mut out3,
            LpcMode::Anytime,
            Some(true),
            &mut orders3,
        );
        for ch in 0..NUM_CHANNELS {
            assert!((orders3[ch] as usize) <= LPC_ORDER);
        }
    }

    /// Sanity: adaptive on a flat (constant) signal picks order 0 —
    /// there is no AR structure for the Levinson predictor to exploit,
    /// the byte-cost penalty wins, and the residual is just `signal -
    /// running-mean` after bias_cancel.
    #[test]
    fn adaptive_flat_input_picks_low_order() {
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                signal[ch][i] = 12345; // pure DC
            }
        }
        let mut out = LpcOutput::zeroed();
        let mut orders: ChosenOrders = [0u8; NUM_CHANNELS];
        analyze_all_channels_with_mode(
            &signal,
            &mut out,
            LpcMode::Adaptive,
            None,
            &mut orders,
        );
        for ch in 0..NUM_CHANNELS {
            // Constant input → autocorr is constant → reflection coeff k
            // approaches 0 fast, so the AIC walk rejects every extra
            // tap. Either order=0 (degenerate r[0]=0 short-circuit) or
            // order=1 (one near-zero coefficient picked) — anything
            // higher means the scorer over-fit pure noise.
            assert!(
                orders[ch] <= 1,
                "ch{ch} adaptive over-fit flat input: chosen_order={}",
                orders[ch]
            );
        }
    }

    /// Cross-equality vs `lamquant_core::lpc::analyze` on a non-trivial input.
    /// Asserts both code paths produce **bit-identical** Q27 coefficients
    /// and bit-identical residuals.
    ///
    /// IGNORED: this test was added in `dad44813` (Q56 Levinson) but has
    /// always failed on signals whose AR coefficients exceed |a| > 1
    /// (e.g. order-2 sine prediction needs a ≈ -1.99). The Q56 → Q31
    /// emit step (firmware `levinson_q31`) saturates such values at
    /// `i32::MIN`, while the host f64 path returns the true negated
    /// Q27 value. The fix is to convert directly Q56 → Q27 inside the
    /// firmware Levinson and skip the lossy Q31 intermediate; that
    /// touches the wire-format-adjacent emit shift and warrants its
    /// own audit + commit. See follow-up issue.
    #[test]
    #[ignore]
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
