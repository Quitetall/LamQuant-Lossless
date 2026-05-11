//! 3-level Le Gall 5/3 integer lifting DWT (Stage 3 of pipeline).
//!
//! Decomposes 21×2500 LPC residual into:
//!   L3 approx  [21][313]   → TNN encoder input (Mode 1)
//!   L3 detail  [21][312]   → sparse encoding (Golomb-Rice)
//!   L2 detail  [21][625]   → sparse encoding
//!   L1 detail  [21][1250]  → sparse encoding
//!
//! Pure integer in-place. No alloc, no `Vec`, no `unsafe`. Bit-identical to
//! `lamquant_core::lifting::forward_3level` (Python decoder compat) — same
//! arithmetic-shift rounding in the update step, same boundary handling.
//!
//! All array access is bounds-checked. Wrapping i32 arithmetic keeps the
//! inner loops free of `as i64` promotion — input magnitudes are bounded
//! well below i32::MAX/2 by the upstream biquad + LPC residual chain.

use super::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};

pub const L3_APPROX_LEN: usize = 313;
pub const L3_DETAIL_LEN: usize = 312;
pub const L2_DETAIL_LEN: usize = 625;
pub const L1_DETAIL_LEN: usize = 1250;

/// Subbands produced by 3-level lifting on a 2500-sample window.
pub struct Subbands {
    pub l3_approx: [[i32; L3_APPROX_LEN]; NUM_CHANNELS],
    pub l3_detail: [[i32; L3_DETAIL_LEN]; NUM_CHANNELS],
    pub l2_detail: [[i32; L2_DETAIL_LEN]; NUM_CHANNELS],
    pub l1_detail: [[i32; L1_DETAIL_LEN]; NUM_CHANNELS],
}

impl Subbands {
    pub const fn zeroed() -> Self {
        Self {
            l3_approx: [[0; L3_APPROX_LEN]; NUM_CHANNELS],
            l3_detail: [[0; L3_DETAIL_LEN]; NUM_CHANNELS],
            l2_detail: [[0; L2_DETAIL_LEN]; NUM_CHANNELS],
            l1_detail: [[0; L1_DETAIL_LEN]; NUM_CHANNELS],
        }
    }
}

/// Caller-owned scratch space for the per-channel lifting pipeline.
///
/// Holds the level-2 / level-3 approximation buffers used between
/// de-interleave + next-level lift. Total 7.5 KB. Allocate once alongside
/// `Subbands` in the pipeline scheduler.
///
/// Level-1 lift runs in-place on the caller's `signal[ch]` buffer, so no
/// 2500-sample scratch is needed. Two scratch buffers (level 2 + level 3)
/// avoid alias hazards during de-interleave — the destination of
/// de-interleave never aliases its source.
pub struct LiftingScratch {
    pub level2: [i32; L1_DETAIL_LEN], // 5 KB — L1 approx → L2 lift
    pub level3: [i32; L2_DETAIL_LEN], // 2.5 KB — L2 approx → L3 lift
}

impl LiftingScratch {
    pub const fn zeroed() -> Self {
        Self {
            level2: [0; L1_DETAIL_LEN],
            level3: [0; L2_DETAIL_LEN],
        }
    }
}

impl Default for LiftingScratch {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Forward 1D integer 5/3 lifting in-place on `buf[..n]`.
///
/// Matches `lamquant_core::lifting::forward` exactly:
///   - Predict: `detail[i] -= (approx[i] + approx[i+1]) >> 1`
///   - Predict boundary (even n): `detail[last] -= approx[last]`
///   - Update first:  `approx[0] += (detail[0] + 1) >> 1`
///   - Update interior: `approx[i] += (detail[i-1] + detail[i] + 2) >> 2`
///   - Update right boundary: `approx[i] += (detail[i-1] + 1) >> 1`
///
/// After the call, even indices of `buf[..n]` hold the approx coefficients,
/// odd indices hold the detail coefficients (interleaved layout).
///
/// `#[inline(always)]` so per-call constants (n=2500, 1250, 625) constant-fold
/// inside `forward_all_channels`. Slicing `&mut buf[..n]` once at entry teaches
/// the optimiser the loop bound, eliminating per-iteration bounds checks.
#[inline(always)]
fn lift_inplace(buf: &mut [i32], n: usize) {
    if n < 2 {
        return;
    }
    let buf = &mut buf[..n];
    let n_detail = n / 2;
    let n_approx = (n + 1) / 2;

    // ── Predict step ───────────────────────────────────────────────
    let bulk_end = if n % 2 == 0 { n_detail - 1 } else { n_detail };
    for i in 0..bulk_end {
        let two_i = 2 * i;
        let a0 = buf[two_i];
        let a1 = buf[two_i + 2];
        let pred = a0.wrapping_add(a1) >> 1;
        buf[two_i + 1] = buf[two_i + 1].wrapping_sub(pred);
    }
    if n % 2 == 0 && n_detail > 0 {
        let last_d = 2 * (n_detail - 1) + 1;
        let last_a = 2 * (n_detail - 1);
        buf[last_d] = buf[last_d].wrapping_sub(buf[last_a]);
    }

    // ── Update step ────────────────────────────────────────────────
    {
        let d0 = buf[1];
        let upd = d0.wrapping_add(1) >> 1;
        buf[0] = buf[0].wrapping_add(upd);
    }
    let bulk_update_end = n_detail.min(n_approx);
    for i in 1..bulk_update_end {
        let two_i = 2 * i;
        let dl = buf[two_i - 1];
        let dr = buf[two_i + 1];
        let upd = dl.wrapping_add(dr).wrapping_add(2) >> 2;
        buf[two_i] = buf[two_i].wrapping_add(upd);
    }
    if n_approx > n_detail {
        let i = n_approx - 1;
        let dl = buf[2 * i - 1];
        let upd = dl.wrapping_add(1) >> 1;
        buf[2 * i] = buf[2 * i].wrapping_add(upd);
    }
}

/// Inverse 1D integer 5/3 lifting in-place. Exact inverse of `lift_inplace`.
#[inline(always)]
fn unlift_inplace(buf: &mut [i32], n: usize) {
    if n < 2 {
        return;
    }
    let buf = &mut buf[..n];
    let n_detail = n / 2;
    let n_approx = (n + 1) / 2;

    // ── Inverse update ─────────────────────────────────────────────
    if n_approx > n_detail {
        let i = n_approx - 1;
        let dl = buf[2 * i - 1];
        let upd = dl.wrapping_add(1) >> 1;
        buf[2 * i] = buf[2 * i].wrapping_sub(upd);
    }
    let bulk_update_end = n_detail.min(n_approx);
    for i in (1..bulk_update_end).rev() {
        let two_i = 2 * i;
        let dl = buf[two_i - 1];
        let dr = buf[two_i + 1];
        let upd = dl.wrapping_add(dr).wrapping_add(2) >> 2;
        buf[two_i] = buf[two_i].wrapping_sub(upd);
    }
    {
        let d0 = buf[1];
        let upd = d0.wrapping_add(1) >> 1;
        buf[0] = buf[0].wrapping_sub(upd);
    }

    // ── Inverse predict ────────────────────────────────────────────
    if n % 2 == 0 && n_detail > 0 {
        let last_d = 2 * (n_detail - 1) + 1;
        let last_a = 2 * (n_detail - 1);
        buf[last_d] = buf[last_d].wrapping_add(buf[last_a]);
    }
    let bulk_end = if n % 2 == 0 { n_detail - 1 } else { n_detail };
    for i in 0..bulk_end {
        let two_i = 2 * i;
        let a0 = buf[two_i];
        let a1 = buf[two_i + 2];
        let pred = a0.wrapping_add(a1) >> 1;
        buf[two_i + 1] = buf[two_i + 1].wrapping_add(pred);
    }
}

/// De-interleave `src[..n]` (even = approx, odd = detail) into `approx_out`
/// and `detail_out`.
#[inline]
fn deinterleave(src: &[i32], n: usize, approx_out: &mut [i32], detail_out: &mut [i32]) {
    let n_approx = (n + 1) / 2;
    let n_detail = n / 2;
    for i in 0..n_approx {
        approx_out[i] = src[2 * i];
    }
    for i in 0..n_detail {
        detail_out[i] = src[2 * i + 1];
    }
}

/// Forward 3-level lifting on all 21 channels.
///
/// Pure integer in-place. No allocation. Caller provides `scratch` once
/// (allocated alongside `Subbands` in the pipeline scheduler).
///
/// **Mutates `signal` in place.** After the call, `signal[ch][..]` holds
/// interleaved level-1 lifting output (intermediate state — not useful
/// downstream). Caller must capture any required snapshot of the residual
/// before invoking this function.
pub fn forward_all_channels(
    signal: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    scratch: &mut LiftingScratch,
    out: &mut Subbands,
) {
    for ch in 0..NUM_CHANNELS {
        // Level 1: lift signal[ch] in place, deinterleave into scratch.level2
        // (= L1 approx) and out.l1_detail[ch].
        lift_inplace(&mut signal[ch], WINDOW_SAMPLES);
        deinterleave(
            &signal[ch],
            WINDOW_SAMPLES,
            &mut scratch.level2[..L1_DETAIL_LEN],
            &mut out.l1_detail[ch],
        );

        // Level 2: lift on level2 (1250 samples), deinterleave.
        lift_inplace(&mut scratch.level2[..L1_DETAIL_LEN], L1_DETAIL_LEN);
        deinterleave(
            &scratch.level2[..L1_DETAIL_LEN],
            L1_DETAIL_LEN,
            &mut scratch.level3[..L2_DETAIL_LEN],
            &mut out.l2_detail[ch],
        );

        // Level 3: lift on level3 (625 samples), deinterleave directly into out.
        lift_inplace(&mut scratch.level3[..L2_DETAIL_LEN], L2_DETAIL_LEN);
        deinterleave(
            &scratch.level3[..L2_DETAIL_LEN],
            L2_DETAIL_LEN,
            &mut out.l3_approx[ch],
            &mut out.l3_detail[ch],
        );
    }
}

/// Inverse 3-level lifting (decoder side / verification).
///
/// Reuses the same scratch struct. Output buffer `out[ch]` doubles as the
/// level-1 work buffer (no separate 2500-sample scratch needed).
pub fn inverse_all_channels(
    subbands: &Subbands,
    scratch: &mut LiftingScratch,
    out: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
) {
    for ch in 0..NUM_CHANNELS {
        // Re-interleave L3 approx + L3 detail into level3 (625 samples).
        for i in 0..L3_APPROX_LEN {
            scratch.level3[2 * i] = subbands.l3_approx[ch][i];
        }
        for i in 0..L3_DETAIL_LEN {
            scratch.level3[2 * i + 1] = subbands.l3_detail[ch][i];
        }
        unlift_inplace(&mut scratch.level3[..L2_DETAIL_LEN], L2_DETAIL_LEN);

        // Re-interleave L2 approx (= scratch.level3 pre-detail) + L2 detail
        // into level2 (1250 samples).
        let n_a_l2 = (L1_DETAIL_LEN + 1) / 2; // 625
        for i in 0..n_a_l2 {
            scratch.level2[2 * i] = scratch.level3[i];
        }
        for i in 0..L2_DETAIL_LEN {
            scratch.level2[2 * i + 1] = subbands.l2_detail[ch][i];
        }
        unlift_inplace(&mut scratch.level2[..L1_DETAIL_LEN], L1_DETAIL_LEN);

        // Re-interleave L1 approx (= scratch.level2 pre-detail) + L1 detail
        // directly into out[ch] (2500 samples) — saves a 10 KB scratch.
        let n_a_l1 = (WINDOW_SAMPLES + 1) / 2; // 1250
        for i in 0..n_a_l1 {
            out[ch][2 * i] = scratch.level2[i];
        }
        for i in 0..L1_DETAIL_LEN {
            out[ch][2 * i + 1] = subbands.l1_detail[ch][i];
        }
        unlift_inplace(&mut out[ch], WINDOW_SAMPLES);
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn forward_inverse_roundtrip() {
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                signal[ch][i] = ((ch as i32 + 1) * (i as i32 % 137)) << 4;
            }
        }
        let signal_orig = signal;

        let mut scratch = LiftingScratch::zeroed();
        let mut subbands = Subbands::zeroed();
        forward_all_channels(&mut signal, &mut scratch, &mut subbands);

        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        inverse_all_channels(&subbands, &mut scratch, &mut reconstructed);

        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(
                    reconstructed[ch][i], signal_orig[ch][i],
                    "ch{ch} sample {i} mismatch"
                );
            }
        }
    }

    #[test]
    fn dc_concentrates_in_l3_approx() {
        let mut signal = [[1000i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        let mut scratch = LiftingScratch::zeroed();
        let mut subbands = Subbands::zeroed();
        forward_all_channels(&mut signal, &mut scratch, &mut subbands);

        for ch in 0..NUM_CHANNELS {
            for &v in &subbands.l1_detail[ch] {
                assert_eq!(v, 0, "L1 detail should be 0 for DC input");
            }
            for &v in &subbands.l2_detail[ch] {
                assert_eq!(v, 0, "L2 detail should be 0 for DC input");
            }
            for &v in &subbands.l3_detail[ch] {
                assert_eq!(v, 0, "L3 detail should be 0 for DC input");
            }
        }
    }

    /// Cross-check against `lamquant_core::lifting` for arbitrary-ish input.
    #[test]
    fn matches_lamquant_core_forward() {
        use lamquant_core::lifting as core;
        let mut scratch = LiftingScratch::zeroed();
        let mut subbands = Subbands::zeroed();
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                signal[ch][i] = (((i as i64) * 137 + (ch as i64) * 53) % 30000 - 15000) as i32;
            }
        }
        let signal_orig = signal;
        forward_all_channels(&mut signal, &mut scratch, &mut subbands);

        for ch in 0..NUM_CHANNELS {
            let sig_i64: alloc::vec::Vec<i64> =
                signal_orig[ch].iter().map(|&v| v as i64).collect();
            let (a3, d3, d2, d1) = core::forward_3level(&sig_i64);
            for i in 0..L3_APPROX_LEN {
                assert_eq!(subbands.l3_approx[ch][i] as i64, a3[i], "ch{ch} l3_a[{i}]");
            }
            for i in 0..L3_DETAIL_LEN {
                assert_eq!(subbands.l3_detail[ch][i] as i64, d3[i], "ch{ch} l3_d[{i}]");
            }
            for i in 0..L2_DETAIL_LEN {
                assert_eq!(subbands.l2_detail[ch][i] as i64, d2[i], "ch{ch} l2_d[{i}]");
            }
            for i in 0..L1_DETAIL_LEN {
                assert_eq!(subbands.l1_detail[ch][i] as i64, d1[i], "ch{ch} l1_d[{i}]");
            }
        }
    }
}
