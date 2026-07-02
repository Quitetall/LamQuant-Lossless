//! LPC analysis and synthesis with bias cancellation.
//!
//! Optimized: running accumulator for bias (O(T) not O(T*ctx)),
//! bitmask for power-of-2 ctx_len, inlined hot loops.

use alloc::vec;
use alloc::vec::Vec;

const Q_LPC: i64 = 27;

/// Per-subband fixed LPC order schedule for the fast-path encoder.
///
/// Matches the legacy hardcoded ladder that shipped before adaptive
/// AIC/MDL order selection: subband index 0 (approximation) gets order
/// 3, fine detail subbands climb to order 8. Schedule covers 4 subbands
/// (a + 3 details for a 3-level Le Gall lifting DWT). Subbands beyond
/// index 3 (only seen on >3-level DWTs which the encoder doesn't use)
/// repeat the last entry.
pub const FIXED_ORDER_SCHEDULE: [usize; 4] = [3, 3, 6, 8];

// ─── Autocorrelation kernel — scalar + AVX2 ──────────────────────
//
// Runtime-dispatched in `autocorr()`. On x86_64 hosts with AVX2,
// the kernel parallelises ACROSS lags (4 lags per AVX2 vector,
// `_mm256_mul_pd` + `_mm256_add_pd`). Per-lag accumulation order
// is identical to the scalar kernel, so the f64 accumulator value
// at each lag is BIT-IDENTICAL — required by the byte-equal
// cross-backend conformance gate (`tests/byte_equal_backends.rs`).
//
// Why not FMA: FMA fuses mul + add into one round, scalar uses two
// rounds. Different bits at the ULP level on some inputs → would
// break byte-equality. We use plain mul + add, same rounding as
// the scalar path.
//
// On firmware (Cortex-M, no_std), `autocorr` resolves to the
// scalar kernel at compile time; AVX2 code is gated behind both
// `feature = "std"` and `target_arch = "x86_64"`.

/// Scalar autocorrelation reference. Identical loop shape to the
/// kernel that used to live inline in `analyze()` -- this is the
/// byte-equal baseline.
#[inline]
fn autocorr_scalar(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let mut s = 0.0f64;
        let end = seg_len.saturating_sub(lag);
        for i in 0..end {
            s += subband[i] as f64 * subband[i + lag] as f64;
        }
        r[lag] = s;
    }
    r
}

/// AVX2 autocorrelation: 4 lags per SIMD vector, plain mul + add
/// (no FMA) so the per-lag f64 result is bit-identical to the
/// scalar kernel. Caller guards via `is_x86_feature_detected!`.
///
/// # Safety
///
/// Caller MUST verify AVX2 is available before calling. Bounds on
/// subband accesses are checked by the loop -- we always read
/// `subband[i + lag_base + k]` only when `i < seg_len - (lag_base + 3)`,
/// i.e. when ALL four lanes are in range. Tail iterations fall
/// back to scalar per-lag.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn autocorr_avx2(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_set_pd, _mm256_setzero_pd,
        _mm256_storeu_pd,
    };

    let mut r = vec![0.0f64; order + 1];
    let lags = order + 1;
    let mut lag_base = 0;

    // SIMD body: process lags in batches of 4 over the i range
    // where all 4 lag-offsets fit in `subband`.
    while lag_base + 4 <= lags {
        let common_end = seg_len.saturating_sub(lag_base + 3);
        let mut accs = _mm256_setzero_pd();
        for i in 0..common_end {
            let s_i = _mm256_set1_pd(subband[i] as f64);
            // _mm256_set_pd is reverse-indexed in Intel convention:
            // arg 0 -> high lane, arg 3 -> low lane.
            // We want lane k (=lag_base+k) to hold subband[i + lag_base + k].
            let v = _mm256_set_pd(
                subband[i + lag_base + 3] as f64,
                subband[i + lag_base + 2] as f64,
                subband[i + lag_base + 1] as f64,
                subband[i + lag_base] as f64,
            );
            let prod = _mm256_mul_pd(s_i, v);
            accs = _mm256_add_pd(accs, prod);
        }
        // Spill the 4 accumulators back to r[lag_base..lag_base + 4].
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), accs);
        r[lag_base] = buf[0];
        r[lag_base + 1] = buf[1];
        r[lag_base + 2] = buf[2];
        r[lag_base + 3] = buf[3];

        // Scalar tail per lag: for lag_base..lag_base+3 each lag
        // has its own [common_end, seg_len - lag) trailing range
        // that didn't fit in the SIMD body. Lag (lag_base + 3) has
        // no tail (its end == common_end).
        for k in 0..3 {
            let lag = lag_base + k;
            let end = seg_len.saturating_sub(lag);
            for i in common_end..end {
                r[lag] += subband[i] as f64 * subband[i + lag] as f64;
            }
        }
        lag_base += 4;
    }

    // Remaining 1-3 lags: pure scalar. Same per-lag accumulation
    // order as the scalar kernel, so result is bit-identical.
    while lag_base < lags {
        let lag = lag_base;
        let end = seg_len.saturating_sub(lag);
        let mut s = 0.0f64;
        for i in 0..end {
            s += subband[i] as f64 * subband[i + lag] as f64;
        }
        r[lag] = s;
        lag_base += 1;
    }

    r
}

/// Runtime-dispatched autocorr. On x86_64 host with AVX2, calls
/// the SIMD kernel; otherwise falls back to scalar. Both kernels
/// produce bit-identical output by construction.
#[inline]
fn autocorr(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just confirmed available at runtime.
            return unsafe { autocorr_avx2(subband, order, seg_len) };
        }
    }
    autocorr_scalar(subband, order, seg_len)
}

/// LPC analysis mode — controls the speed / CR trade-off the encoder
/// makes per subband.
///
/// * `Fixed` — constant CPU, slightly worse CR. Order per subband comes
///   from [`FIXED_ORDER_SCHEDULE`]. Right for hard-realtime paths where
///   the cycle budget cannot stretch.
/// * `Adaptive` — variable CPU, best CR. Walks Levinson + AIC/MDL up
///   to `max_order` and picks the byte-cost-optimal order per subband.
/// * `Anytime` — best-effort deadline-aware. The deadline is sampled
///   once at the entry of each subband: if it has not yet elapsed, the
///   subband runs adaptive (and is allowed to finish even if it overruns
///   mid-search — the encoder does not currently support cooperative
///   cancellation inside the Levinson loop); if the deadline has passed,
///   the subband falls back to the `Fixed` schedule. The granularity
///   is per-subband (4 subbands per channel × n_ch checks per window),
///   which gives natural graceful degradation under load without paying
///   for a fixed pre-pass on every subband. `deadline = None` short-
///   circuits to pure adaptive — no clock reads, same output as
///   `Adaptive`.
#[derive(Debug, Clone, Copy)]
pub enum LpcMode {
    Fixed,
    Adaptive {
        max_order: usize,
    },
    #[cfg(feature = "std")]
    Anytime {
        max_order: usize,
        deadline: Option<std::time::Instant>,
    },
    /// Anytime-without-std variant — host code uses the std variant for
    /// `Instant`-based deadlines; firmware uses this and polls its own
    /// cycle counter / scheduler tick before each subband.
    #[cfg(not(feature = "std"))]
    Anytime {
        max_order: usize,
    },
}

impl Default for LpcMode {
    /// Default = `Anytime` with no deadline (= adaptive when no clock
    /// pressure). Stream callers override `deadline` per window.
    fn default() -> Self {
        #[cfg(feature = "std")]
        {
            LpcMode::Anytime {
                max_order: 16,
                deadline: None,
            }
        }
        #[cfg(not(feature = "std"))]
        {
            LpcMode::Anytime { max_order: 16 }
        }
    }
}

/// Per-subband order for `LpcMode::Fixed`. Subband index >= 4 repeats
/// the last entry of [`FIXED_ORDER_SCHEDULE`].
#[inline]
pub fn fixed_order_for_subband(sb_idx: usize) -> usize {
    FIXED_ORDER_SCHEDULE[sb_idx.min(FIXED_ORDER_SCHEDULE.len() - 1)]
}

/// Unified mode-dispatching LPC entry point.
///
/// Returns `(coeffs_q27, residual, chosen_order)`. Wire format already
/// stores the order per subband — no decoder change for any mode.
///
/// Caller is responsible for the deadline-availability check on the
/// `Anytime` variant: this function reads the embedded deadline once
/// to decide whether to run adaptive after fixed. For the no_std path
/// the firmware passes the answer in `time_remaining` (computed against
/// its cycle counter); host code passes `None` and trusts the embedded
/// `Instant` deadline.
pub fn analyze_with_mode(
    subband: &[i64],
    sb_idx: usize,
    mode: LpcMode,
    ctx_len: usize,
    #[allow(unused_variables)] time_remaining: Option<bool>,
) -> (Vec<i32>, Vec<i64>, usize) {
    match mode {
        LpcMode::Fixed => {
            let o = fixed_order_for_subband(sb_idx);
            let (coeffs, residual) = analyze(subband, o, ctx_len);
            (coeffs, residual, o)
        }
        LpcMode::Adaptive { max_order } => analyze_adaptive(subband, max_order, ctx_len),
        #[cfg(feature = "std")]
        LpcMode::Anytime {
            max_order,
            deadline,
        } => analyze_anytime_host(subband, sb_idx, max_order, deadline, ctx_len),
        #[cfg(not(feature = "std"))]
        LpcMode::Anytime { max_order } => {
            // no_std: caller is the time oracle. Default to `false`
            // ("no budget") if the caller forgot to pass a signal — the
            // safe choice that always meets the deadline by falling
            // back to fixed. Firmware schedulers are expected to pass
            // an explicit `Some(true|false)` per subband.
            let have_time = time_remaining.unwrap_or(false);
            analyze_anytime_inner(subband, sb_idx, max_order, have_time, ctx_len)
        }
    }
}

#[cfg(feature = "std")]
fn analyze_anytime_host(
    subband: &[i64],
    sb_idx: usize,
    max_order: usize,
    deadline: Option<std::time::Instant>,
    ctx_len: usize,
) -> (Vec<i32>, Vec<i64>, usize) {
    // Branch order matters: when the deadline still has budget we go
    // straight to adaptive — running a fixed pre-pass first would just
    // be wasted CPU (we'd throw the result away). When the deadline has
    // expired we run the cheaper `Fixed` schedule so the packet emerges
    // as fast as possible. With `deadline = None` we never touch the
    // clock — the host CLI batch path takes this branch and pays nothing.
    let time_remains = match deadline {
        None => true,
        Some(d) => std::time::Instant::now() < d,
    };
    if time_remains {
        analyze_adaptive(subband, max_order, ctx_len)
    } else {
        let fixed_o = fixed_order_for_subband(sb_idx);
        let (coeffs, residual) = analyze(subband, fixed_o, ctx_len);
        (coeffs, residual, fixed_o)
    }
}

#[cfg(not(feature = "std"))]
fn analyze_anytime_inner(
    subband: &[i64],
    sb_idx: usize,
    max_order: usize,
    have_time: bool,
    ctx_len: usize,
) -> (Vec<i32>, Vec<i64>, usize) {
    // Match the host `analyze_anytime_host` branching: skip the fixed
    // pre-pass when the caller signals time remains, fall back to fixed
    // only when budget is exhausted. Firmware passes `have_time` from
    // its scheduler tick; an absent signal is treated as "no budget"
    // (`false`) by `analyze_with_mode` — the safe default that meets
    // the deadline guarantee.
    if have_time {
        analyze_adaptive(subband, max_order, ctx_len)
    } else {
        let fixed_o = fixed_order_for_subband(sb_idx);
        let (coeffs, residual) = analyze(subband, fixed_o, ctx_len);
        (coeffs, residual, fixed_o)
    }
}

/// Bench-only switch: when env var `LMQ_LEVINSON=blockfloat` is set at
/// process start, `analyze()` routes through the integer block-floating
/// Levinson path instead of the f64 path. Used to measure CR delta vs
/// firmware-equivalent integer Levinson without FPU. Not for production.
#[cfg(feature = "std")]
fn use_blockfloat() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LMQ_LEVINSON")
            .map(|v| v == "blockfloat")
            .unwrap_or(false)
    })
}
#[cfg(not(feature = "std"))]
fn use_blockfloat() -> bool {
    false
}

/// LPC analysis + bias cancellation (encode side).
///
/// Returns (coeffs_q27, residual).
pub fn analyze(subband: &[i64], order: usize, ctx_len: usize) -> (Vec<i32>, Vec<i64>) {
    if use_blockfloat() {
        return analyze_blockfloat(subband, order, ctx_len);
    }
    let t = subband.len();

    if t <= order || t < 3 || order == 0 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (vec![0i32; order], residual);
    }

    // Autocorrelation on first seg_len samples.
    //
    // Hot loop in LPC analyze (per-stage bench `stage_lpc_analyze_
    // with_mode/fixed_l1_detail` at ~339 MiB/s on threadripper).
    // Dispatched via `autocorr()` to a runtime-detected AVX2 path
    // on x86_64 hosts; scalar everywhere else (firmware, aarch64,
    // legacy CPUs). Byte-equality between paths is enforced by the
    // `byte_equal_backends.rs` conformance gate.
    let seg_len = (t / 2).clamp(1, 256);
    let r = autocorr(subband, order, seg_len);

    if r[0].abs() <= 1e-12 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (vec![0i32; order], residual);
    }

    // Levinson-Durbin (O(order^2)). Double-buffered: `a_next` is allocated once
    // and reused each iteration instead of a fresh `vec![0.0; order]` per step
    // (MiMo #17 — analyze reaches order up to 16, not just the 1-3 the old
    // comment assumed). Zeroing a_next before filling [0..=m] keeps indices > m
    // at 0, so the result is bit-identical to the former fresh-alloc version.
    let mut a = vec![0.0f64; order];
    let mut a_next = vec![0.0f64; order];
    let mut e = r[0];
    for m in 0..order {
        let mut lam = r[m + 1];
        for j in 0..m {
            lam += a[j] * r[m - j];
        }
        if e.abs() <= 1e-12 {
            break;
        }
        let k = -lam / e;
        a_next.fill(0.0);
        a_next[m] = k;
        for j in 0..m {
            a_next[j] = a[j] + k * a[m - 1 - j];
        }
        core::mem::swap(&mut a, &mut a_next);
        e *= 1.0 - k * k;
        if e <= 0.0 {
            e = 1e-10;
        }
    }

    // Q27 fixed-point coefficients.
    // Manual round-half-away-from-zero: no_std compatible (avoids f64::round).
    let q27 = 1i64 << Q_LPC;
    let coeffs: Vec<i32> = a
        .iter()
        .map(|&ai| {
            let v = -ai * q27 as f64;
            (if v >= 0.0 { v + 0.5 } else { v - 0.5 }) as i32
        })
        .collect();

    // Integer LPC forward prediction: residual = signal - pred.
    //
    // Hand-rolled AVX2 was attempted here (commit reverted) -- the
    // setup cost of `_mm256_set_epi32` with 8 args + the i64
    // horizontal-sum chain beat the inner-loop savings, producing
    // a 2-5% REGRESSION vs scalar. The compiler auto-vectorises
    // this loop well; trust it.
    let mut residual = Vec::with_capacity(t);
    for n in 0..t {
        let mut pred = 0i64;
        for k in 0..order.min(n) {
            pred += coeffs[k] as i64 * subband[n - 1 - k];
        }
        residual.push(subband[n] - (pred >> Q_LPC));
    }

    bias_cancel(&mut residual, ctx_len);
    (coeffs, residual)
}

/// Adaptive LPC analysis with AIC-driven order selection.
///
/// Walks Levinson recursion from order=1 up to `max_order`, computing
/// Akaike Information Criterion (AIC) = `t·ln(e) + 2·k` at each step
/// where `e` is the residual energy after applying the order-k predictor.
/// Returns the (coeffs_q27, residual, chosen_order) triplet for the
/// order with minimum AIC.
///
/// Per-subband adaptivity matters because the optimal LPC order grows
/// with the bandwidth of the data: 250 Hz EEG packs almost all energy
/// into 0-30 Hz so order 1-3 suffices, but 16 kHz LFP or 128 kHz audio
/// has structure up to several kHz that needs order 8-16 to model.
///
/// Cost: O(max_order²) for the unified Levinson + one O(t·order) residual
/// pass at the chosen order. ~3× the cost of fixed-order `analyze` at
/// max_order=8, but no decoder change — wire format already stores order
/// per subband.
pub fn analyze_adaptive(
    subband: &[i64],
    max_order: usize,
    ctx_len: usize,
) -> (Vec<i32>, Vec<i64>, usize) {
    let t = subband.len();

    if max_order == 0 || t < 3 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (Vec::new(), residual, 0);
    }
    // Cap max_order against the input size: stable Levinson requires
    // `t > order` for the predictor and `seg_len > order` for autocorr.
    let max_order = max_order.min(t.saturating_sub(1)).min(t / 4).max(1);

    let seg_len = (t / 2).clamp(1, 256);
    // Same SIMD-dispatched autocorr `analyze` uses — AVX2 on x86_64 hosts,
    // scalar elsewhere, bit-identical by construction (see `autocorr_avx2`).
    // Previously this inlined the scalar loop, so the adaptive path never got
    // SIMD (MiMo #11). `autocorr(_, max_order, _)` returns the same
    // length-(max_order+1) `r` with the identical per-lag accumulation order.
    let r = autocorr(subband, max_order, seg_len);

    if r[0].abs() <= 1e-12 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (Vec::new(), residual, 0);
    }

    // Levinson recursion, tracking AIC at each order.
    // The recursion gives us the order-(m+1) AR coefficients in `a_curr`
    // and the residual energy `e` after that step — exactly what AIC needs.
    let n = t as f64;
    let mut a_prev = vec![0.0f64; max_order];
    let mut a_curr = vec![0.0f64; max_order];
    let mut e = r[0];

    // Byte-cost MDL penalty: each extra LPC order costs 4 bytes (32 bits)
    // of i32 coefficient in lpc_meta — much steeper than the statistical
    // 2·k AIC penalty. Approximate Golomb-Rice cost per residual sample
    // as `0.5·log2(e/n) + const` bits → total payload bits scale as
    // `n·ln(e/n)/(2·ln 2) ≈ 0.72·n·ln(e/n)`. So the objective in nats is
    //   `0.72·n·ln(e/n) + 32·ln 2·k  ≈  0.72·n·ln(e/n) + 22·k`.
    // Order-(m+1) wins iff its variance drop outweighs ~22 nats per added
    // coefficient — matches the wire-cost trade-off the codec actually
    // pays, which is what AIC misses (AIC has penalty 2·k, ~10× too low).
    // Exact `32 · ln 2` (≈ 22.18070977791825). The truncated literal
    // `22.18` lamu flagged on the firmware mirror also lived here —
    // align both paths so borderline AIC ties resolve identically.
    const ORDER_BIT_COST: f64 = 32.0 * core::f64::consts::LN_2;

    let mut best_cost = f64::INFINITY;
    let mut best_a: Vec<f64> = Vec::new();
    let mut best_order: usize = 0;

    // Order 0 (just bias_cancel): payload-only cost.
    if r[0] > 0.0 {
        let cost0 = 0.72 * n * libm::log(r[0] / n);
        if cost0 < best_cost {
            best_cost = cost0;
            best_order = 0;
        }
    }

    for m in 0..max_order {
        let mut lam = r[m + 1];
        for j in 0..m {
            lam += a_prev[j] * r[m - j];
        }
        if e.abs() <= 1e-12 {
            break;
        }
        let k = -lam / e;
        a_curr[m] = k;
        for j in 0..m {
            a_curr[j] = a_prev[j] + k * a_prev[m - 1 - j];
        }
        let e_new = e * (1.0 - k * k);
        if e_new <= 0.0 || !e_new.is_finite() {
            // Unstable step or floating-point edge case (inf/NaN/zero);
            // bail out at the previous order. lamu review fix.
            break;
        }
        e = e_new;

        let order_k = m + 1;
        let cost = 0.72 * n * libm::log(e / n) + ORDER_BIT_COST * order_k as f64;
        if cost < best_cost {
            best_cost = cost;
            best_a = a_curr[..order_k].to_vec();
            best_order = order_k;
        }

        a_prev[..order_k].copy_from_slice(&a_curr[..order_k]);
    }

    if best_order == 0 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (Vec::new(), residual, 0);
    }

    // Convert best_a to Q27 with half-away rounding.
    let q27 = 1i64 << Q_LPC;
    let coeffs: Vec<i32> = best_a
        .iter()
        .map(|&ai| {
            let v = -ai * q27 as f64;
            (if v >= 0.0 { v + 0.5 } else { v - 0.5 }) as i32
        })
        .collect();

    // Single residual pass at the chosen order.
    let mut residual = Vec::with_capacity(t);
    for n in 0..t {
        let mut pred = 0i64;
        for k in 0..best_order.min(n) {
            pred += coeffs[k] as i64 * subband[n - 1 - k];
        }
        residual.push(subband[n] - (pred >> Q_LPC));
    }
    bias_cancel(&mut residual, ctx_len);
    (coeffs, residual, best_order)
}

/// LPC synthesis + bias restoration (decode side).
///
/// Convention: signal[n] = restored[n] + pred, where
/// pred = sum(coeffs[k] * signal[n-1-k]) >> Q.
/// Coefficients are non-negated (matches Python).
#[inline]
pub fn synthesize(residual: &[i64], coeffs: &[i32], order: usize, ctx_len: usize) -> Vec<i64> {
    let t = residual.len();

    let mut restored = residual.to_vec();
    bias_restore(&mut restored, ctx_len);

    if t <= order || t < 3 || order == 0 {
        return restored;
    }

    // IIR LPC synthesis: signal[n] = restored[n] + pred
    let mut signal = vec![0i64; t];
    for n in 0..t {
        let mut pred = 0i64;
        let k_max = order.min(n);
        for k in 0..k_max {
            pred += coeffs[k] as i64 * signal[n - 1 - k];
        }
        signal[n] = restored[n] + (pred >> Q_LPC);
    }

    signal
}

/// Round `e` to the nearest multiple of `q` and return that multiple's
/// index (ties away from zero), integer-only. For `q = 2δ+1` the residual
/// `|e − idx·q| ≤ q/2 = δ` — the bound the closed-loop near-lossless mode
/// relies on. `q` must be ≥ 1.
#[inline]
fn round_div_nearest(e: i64, q: i64) -> i64 {
    debug_assert!(q >= 1);
    let half = q / 2;
    if e >= 0 {
        (e + half) / q
    } else {
        -((-e + half) / q)
    }
}

/// Closed-loop bounded-MAE DPCM analysis (Track 2 near-lossless encode).
///
/// Predicts each sample from already-RECONSTRUCTED samples (closed loop),
/// quantizes the prediction residual with uniform step `q = 2δ+1`, and
/// returns the quantizer indices. Because the decoder
/// ([`synthesize_closed_loop_bounded`]) replays the identical loop, the
/// per-sample reconstruction error is `|orig − recon| ≤ q/2 = δ` by
/// construction — independent of `coeffs` quality (coeffs only affect how
/// small the indices, hence the compressed size, get). `q = 1` (δ = 0) is
/// exact lossless. `coeffs` are Q27, same sign convention as [`analyze`]
/// / [`synthesize`].
pub fn analyze_closed_loop_bounded(
    signal: &[i64],
    coeffs: &[i32],
    order: usize,
    q: i64,
) -> Vec<i64> {
    let t = signal.len();
    let mut recon = vec![0i64; t];
    let mut idx = vec![0i64; t];
    for n in 0..t {
        let mut pred = 0i64;
        let k_max = order.min(n).min(coeffs.len());
        for k in 0..k_max {
            pred += coeffs[k] as i64 * recon[n - 1 - k];
        }
        let p = pred >> Q_LPC;
        let e = signal[n] - p;
        let i = round_div_nearest(e, q);
        idx[n] = i;
        recon[n] = p + i * q;
    }
    idx
}

/// Closed-loop bounded-MAE DPCM synthesis (Track 2 near-lossless decode).
///
/// Exact inverse of [`analyze_closed_loop_bounded`] for the same
/// `(coeffs, order, q)` — runs the identical prediction loop and
/// dequantizes each index. Integer-only ⇒ firmware-decodable.
pub fn synthesize_closed_loop_bounded(
    indices: &[i64],
    coeffs: &[i32],
    order: usize,
    q: i64,
) -> Vec<i64> {
    let t = indices.len();
    let mut recon = vec![0i64; t];
    for n in 0..t {
        let mut pred = 0i64;
        let k_max = order.min(n).min(coeffs.len());
        for k in 0..k_max {
            pred += coeffs[k] as i64 * recon[n - 1 - k];
        }
        recon[n] = (pred >> Q_LPC) + indices[n] * q;
    }
    recon
}

/// Block-floating-point integer Levinson — no FPU, no f64. Targets
/// f64-class precision by storing reflection coefficients in i64 Q63
/// and accumulating in i128. Final coefficients down-shifted to Q27.
///
/// Wire-format compatible with `analyze`: returns identical (coeffs_q27,
/// residual) shape. Coefficients may differ by a few LSBs from the f64
/// path; CR difference targeted < 0.5% on real EEG.
pub fn analyze_blockfloat(subband: &[i64], order: usize, ctx_len: usize) -> (Vec<i32>, Vec<i64>) {
    let t = subband.len();

    if t <= order || t < 3 || order == 0 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (vec![0i32; order], residual);
    }

    // Integer autocorrelation: same precision as f64 path (i64 full range).
    let seg_len = (t / 2).clamp(1, 256);
    let mut r = vec![0i128; order + 1];
    for lag in 0..=order {
        let mut s: i128 = 0;
        let end = seg_len.saturating_sub(lag);
        for i in 0..end {
            s = s.wrapping_add((subband[i] as i128) * (subband[i + lag] as i128));
        }
        r[lag] = s;
    }

    if r[0] == 0 {
        let mut residual = subband.to_vec();
        bias_cancel(&mut residual, ctx_len);
        return (vec![0i32; order], residual);
    }

    // Levinson-Durbin with Q56 storage for AR coefficients in i64.
    // 56-bit fraction > f64's 53-bit mantissa. |a| up to 128 fits i64.
    // (Pure Q63 storage overflowed for |a| > 1 — AR coefs can exceed 1 even
    //  when reflection k stays bounded.)
    const QA: u32 = 56;
    const HALF_QA: i128 = 1 << (QA - 1);
    let one_qa: i128 = 1 << QA;

    let mut a_prev = vec![0i64; order];
    let mut a_curr = vec![0i64; order];
    let mut e: i128 = r[0];

    for m in 0..order {
        // lam at Qa scale: r[m+1] << QA + sum(a_prev[j] (Qa) * r[m-j] (raw))
        // Magnitudes: r ≤ 2^50 (real EEG), r << 56 = 2^106, prod = 2^56*2^50 = 2^106.
        // 8 sums = 2^109. Fits i128.
        let mut lam_qa: i128 = (r[m + 1]) << QA;
        for j in 0..m {
            lam_qa = lam_qa.wrapping_add((a_prev[j] as i128).wrapping_mul(r[m - j]));
        }

        if e == 0 {
            break;
        }
        // k_qa = -lam_Qa / e_raw → Qa scale i128.
        // wrapping_neg avoids UB on i128::MIN (astronomically unlikely on
        // real EEG; lamu review fix).
        let k_qa_full = lam_qa.wrapping_neg() / e;
        // |k| < 1 means k_qa fits i64 with room. Clamp just in case of pathology.
        let k_qa = k_qa_full.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

        a_curr[m] = k_qa;
        for j in 0..m {
            // mac = round(k * a_prev[m-1-j] / 2^QA) at Qa scale
            // k_qa (Qa i64) * a_prev (Qa i64) = i128 Q(2*Qa) = Q112
            // Shift right by Qa=56 → Qa. Half-away rounded.
            let prod = (k_qa as i128).wrapping_mul(a_prev[m - 1 - j] as i128);
            let mac_i128 = if prod >= 0 {
                (prod + HALF_QA) >> QA
            } else {
                -((-prod + HALF_QA) >> QA)
            };
            a_curr[j] = a_prev[j].wrapping_add(mac_i128 as i64);
        }

        // e *= (1 - k²) with Qa precision
        let k_sq_prod = (k_qa as i128).wrapping_mul(k_qa as i128); // Q112
        let k_sq_qa = (k_sq_prod + HALF_QA) >> QA; // Qa
        let factor = one_qa - k_sq_qa; // Qa, ≤ 2^QA
        let e_full = e.wrapping_mul(factor); // raw * Qa = Qa
        e = if e_full >= 0 {
            (e_full + HALF_QA) >> QA
        } else {
            -((-e_full + HALF_QA) >> QA)
        };
        if e <= 0 {
            e = 1;
        }

        a_prev[..order].copy_from_slice(&a_curr[..order]);
    }

    // Final: Qa → Q27 (shift right by QA - Q_LPC = 29) with negation + half-away.
    const QA_TO_Q27_SHIFT: i64 = (QA as i64) - 27; // = 29
    let half_shift: i64 = 1i64 << (QA_TO_Q27_SHIFT - 1); // = 2^28
    let coeffs: Vec<i32> = a_curr
        .iter()
        .map(|&v| {
            let neg = v.wrapping_neg();
            let shifted: i64 = if neg >= 0 {
                neg.saturating_add(half_shift) >> QA_TO_Q27_SHIFT
            } else {
                -(neg.wrapping_neg().saturating_add(half_shift) >> QA_TO_Q27_SHIFT)
            };
            shifted.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        })
        .collect();

    // Identical residual + bias_cancel as f64 path.
    let mut residual = Vec::with_capacity(t);
    for n in 0..t {
        let mut pred = 0i64;
        for k in 0..order.min(n) {
            pred += coeffs[k] as i64 * subband[n - 1 - k];
        }
        residual.push(subband[n] - (pred >> Q_LPC));
    }

    bias_cancel(&mut residual, ctx_len);
    (coeffs, residual)
}

/// Floor division (rounds toward -infinity), matching Python's `//` operator.
/// Rust's `/` truncates toward zero, which diverges for negative dividends.
#[inline(always)]
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    if (a ^ b) < 0 && d * b != a {
        d - 1
    } else {
        d
    }
}

fn bias_cancel(data: &mut [i64], ctx_len: usize) {
    let mask = if ctx_len.is_power_of_two() {
        ctx_len - 1
    } else {
        0
    };
    let use_mask = mask != 0;
    let mut buf = vec![0i64; ctx_len];
    let mut running_sum = 0i64;
    let ctx = ctx_len as i64;

    for i in 0..data.len() {
        let bias = floor_div(running_sum, ctx);
        let val = data[i];
        data[i] -= bias;
        let slot = if use_mask { i & mask } else { i % ctx_len };
        let old = buf[slot];
        buf[slot] = val;
        running_sum += val - old;
    }
}

/// Bias restoration with running accumulator. O(T). Exact inverse of cancel.
#[inline]
fn bias_restore(data: &mut [i64], ctx_len: usize) {
    let mask = if ctx_len.is_power_of_two() {
        ctx_len - 1
    } else {
        0
    };
    let use_mask = mask != 0;
    let mut buf = vec![0i64; ctx_len];
    let mut running_sum = 0i64;
    let ctx = ctx_len as i64;

    for i in 0..data.len() {
        let bias = floor_div(running_sum, ctx);
        data[i] += bias;
        let slot = if use_mask { i & mask } else { i % ctx_len };
        let old = buf[slot];
        buf[slot] = data[i];
        running_sum += data[i] - old;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-equality invariant `analyze`/`analyze_adaptive` rely on when
    /// dispatching to `autocorr_avx2` (MiMo #11 + task #24): the AVX2 kernel is
    /// bit-identical to the scalar kernel, so MCU (scalar) and host (AVX2)
    /// produce the same `.lml` bytes. Asserts exact f64 equality across lags,
    /// orders, and segment lengths.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn autocorr_avx2_bit_identical_to_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // no AVX2 on this host → only the scalar path runs anyway
        }
        for &t in &[8usize, 33, 128, 512] {
            let signal: Vec<i64> =
                (0..t).map(|i| (((i * 131 + 7) % 4001) as i64) - 2000).collect();
            let seg_len = (t / 2).clamp(1, 256);
            for &order in &[1usize, 2, 3, 8, 16] {
                let scalar = autocorr_scalar(&signal, order, seg_len);
                // SAFETY: AVX2 confirmed available above.
                let simd = unsafe { autocorr_avx2(&signal, order, seg_len) };
                let scalar_bits: Vec<u64> = scalar.iter().map(|x| x.to_bits()).collect();
                let simd_bits: Vec<u64> = simd.iter().map(|x| x.to_bits()).collect();
                assert_eq!(scalar_bits, simd_bits, "autocorr mismatch t={t} order={order}");
            }
        }
    }

    #[test]
    fn roundtrip_orders() {
        for order in [0, 1, 2, 3] {
            for t in [5, 50, 313, 625] {
                let signal: Vec<i64> = (0..t).map(|i| ((i * 47) % 3000 - 1500) as i64).collect();
                let (coeffs, res) = analyze(&signal, order, 16);
                let recovered = synthesize(&res, &coeffs, order, 16);
                assert_eq!(signal, recovered, "Failed order={} t={}", order, t);
            }
        }
    }

    #[test]
    fn roundtrip_zeros() {
        let signal = vec![0i64; 100];
        let (c, r) = analyze(&signal, 2, 16);
        assert_eq!(synthesize(&r, &c, 2, 16), signal);
    }

    #[test]
    fn fixed_mode_uses_legacy_schedule() {
        let sub: Vec<i64> = (0..1250).map(|i| ((i * 31) % 4000 - 2000) as i64).collect();
        for (sb_idx, expected) in FIXED_ORDER_SCHEDULE.iter().enumerate() {
            let (_, _, ord) = analyze_with_mode(&sub, sb_idx, LpcMode::Fixed, 16, None);
            assert_eq!(
                ord, *expected,
                "sb_idx={} expected order {}",
                sb_idx, expected
            );
        }
    }

    #[test]
    fn adaptive_mode_picks_per_input_order() {
        // Highly autocorrelated signal: adaptive should pick > 0.
        let sub: Vec<i64> = (0..625)
            .map(|i| ((i as f64 * 0.1).sin() * 4000.0) as i64)
            .collect();
        let (_, _, ord) = analyze_with_mode(&sub, 0, LpcMode::Adaptive { max_order: 8 }, 16, None);
        assert!(ord <= 8, "adaptive must respect max_order ceiling");
    }

    #[test]
    fn anytime_no_deadline_matches_adaptive() {
        // Without a deadline, anytime must run adaptive (skip fixed pre-pass).
        let sub: Vec<i64> = (0..625)
            .map(|i| ((i as f64 * 0.13).sin() * 4000.0) as i64)
            .collect();
        let (ca, ra, oa) = analyze_with_mode(&sub, 0, LpcMode::Adaptive { max_order: 8 }, 16, None);
        let (cn, rn, on) = analyze_with_mode(
            &sub,
            0,
            LpcMode::Anytime {
                max_order: 8,
                deadline: None,
            },
            16,
            None,
        );
        assert_eq!(ca, cn);
        assert_eq!(ra, rn);
        assert_eq!(oa, on);
    }

    #[test]
    fn anytime_expired_deadline_falls_back_to_fixed() {
        // Deadline already in the past → fixed schedule must be used.
        let sub: Vec<i64> = (0..1250).map(|i| ((i * 31) % 4000 - 2000) as i64).collect();
        let past = std::time::Instant::now() - std::time::Duration::from_secs(60);
        for (sb_idx, expected) in FIXED_ORDER_SCHEDULE.iter().enumerate() {
            let (_, _, ord) = analyze_with_mode(
                &sub,
                sb_idx,
                LpcMode::Anytime {
                    max_order: 16,
                    deadline: Some(past),
                },
                16,
                None,
            );
            assert_eq!(ord, *expected, "expired deadline must use fixed schedule");
        }
    }

    #[test]
    fn default_mode_is_anytime_no_deadline() {
        match LpcMode::default() {
            LpcMode::Anytime { deadline, .. } => assert!(deadline.is_none()),
            _ => panic!("default LpcMode must be Anytime"),
        }
    }

    #[test]
    fn closed_loop_bounded_respects_delta() {
        // Track 2 / bounded-MAE keystone: closed-loop DPCM must guarantee
        // max|orig - recon| <= delta for ANY predictor coeffs, and reduce
        // to exact reconstruction at delta = 0. The bound is structural
        // (decoder replays the identical loop), independent of coeff quality.
        for delta in [0i64, 1, 3, 7, 15, 50, 255] {
            let q = 2 * delta + 1;
            for t in [1usize, 2, 5, 50, 313, 1250] {
                // A few signal shapes incl. large dynamic range.
                let signal: Vec<i64> = (0..t)
                    .map(|i| ((i as i64 * 47) % 6000 - 3000) + ((i as i64 % 7) * 211))
                    .collect();
                let (coeffs, _r, order) = analyze_adaptive(&signal, 8, 16);
                let idx = analyze_closed_loop_bounded(&signal, &coeffs, order, q);
                assert_eq!(idx.len(), t);
                let recon = synthesize_closed_loop_bounded(&idx, &coeffs, order, q);
                assert_eq!(recon.len(), t, "recon length mismatch");
                let mae = signal
                    .iter()
                    .zip(&recon)
                    .map(|(a, b)| (a - b).abs())
                    .max()
                    .unwrap_or(0);
                assert!(mae <= delta, "delta={} t={} mae={} EXCEEDED", delta, t, mae);
                if delta == 0 {
                    assert_eq!(signal, recon, "delta=0 must be byte-exact (t={})", t);
                }
            }
        }
    }

    #[test]
    fn closed_loop_bounded_order_zero_is_pure_quant() {
        // order=0 → no prediction, pure uniform quantization; still bounded.
        let signal: Vec<i64> = (0..200).map(|i| (i as i64 * 123) % 9001 - 4500).collect();
        let delta = 10i64;
        let q = 2 * delta + 1;
        let idx = analyze_closed_loop_bounded(&signal, &[], 0, q);
        let recon = synthesize_closed_loop_bounded(&idx, &[], 0, q);
        let mae = signal.iter().zip(&recon).map(|(a, b)| (a - b).abs()).max().unwrap();
        assert!(mae <= delta, "order-0 mae={} > delta={}", mae, delta);
    }

    #[test]
    fn anytime_future_deadline_runs_adaptive() {
        // Deadline well in the future → adaptive must run, matching the
        // `LpcMode::Adaptive` output exactly (same coeffs, residual,
        // and chosen order). Catches the missing-branch coverage that
        // the lamu review flagged.
        let sub: Vec<i64> = (0..625)
            .map(|i| ((i as f64 * 0.13).sin() * 4000.0) as i64)
            .collect();
        let future = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let (c_any, r_any, o_any) = analyze_with_mode(
            &sub,
            0,
            LpcMode::Anytime {
                max_order: 8,
                deadline: Some(future),
            },
            16,
            None,
        );
        let (c_a, r_a, o_a) =
            analyze_with_mode(&sub, 0, LpcMode::Adaptive { max_order: 8 }, 16, None);
        assert_eq!(c_any, c_a);
        assert_eq!(r_any, r_a);
        assert_eq!(o_any, o_a);
    }
}
