//! 3-level Le Gall 5/3 integer lifting DWT (Stage 3 of pipeline).
//!
//! Decomposes 21×2500 LPC residual into:
//!   L3 approx  [21][313]   → TNN encoder input (Mode 1)
//!   L3 detail  [21][312]   → sparse encoding (Golomb-Rice)
//!   L2 detail  [21][625]   → sparse encoding
//!   L1 detail  [21][1250]  → sparse encoding
//!
//! Implementation reuses `lamquant_core::lifting::forward_3level`. The
//! firmware uses Q31 `i32` samples; lamquant-core uses `i64`. Conversion
//! at the boundary is lossless (Q31 fits in i64).

use alloc::vec::Vec;
use lamquant_core::lifting;

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

/// Forward 3-level lifting on all 21 channels.
pub fn forward_all_channels(
    signal: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    out: &mut Subbands,
) {
    for ch in 0..NUM_CHANNELS {
        let signal_i64: Vec<i64> = signal[ch].iter().map(|&v| v as i64).collect();
        let (l3_a, l3_d, l2_d, l1_d) = lifting::forward_3level(&signal_i64);

        copy_into(&l3_a, &mut out.l3_approx[ch]);
        copy_into(&l3_d, &mut out.l3_detail[ch]);
        copy_into(&l2_d, &mut out.l2_detail[ch]);
        copy_into(&l1_d, &mut out.l1_detail[ch]);
    }
}

/// Inverse 3-level lifting (decoder side).
pub fn inverse_all_channels(
    subbands: &Subbands,
    out: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
) {
    for ch in 0..NUM_CHANNELS {
        let l3_a: Vec<i64> = subbands.l3_approx[ch].iter().map(|&v| v as i64).collect();
        let l3_d: Vec<i64> = subbands.l3_detail[ch].iter().map(|&v| v as i64).collect();
        let l2_d: Vec<i64> = subbands.l2_detail[ch].iter().map(|&v| v as i64).collect();
        let l1_d: Vec<i64> = subbands.l1_detail[ch].iter().map(|&v| v as i64).collect();
        let signal = lifting::inverse_3level(&l3_a, &l3_d, &l2_d, &l1_d);
        for (i, &v) in signal.iter().enumerate().take(WINDOW_SAMPLES) {
            out[ch][i] = v as i32;
        }
    }
}

#[inline]
fn copy_into(src: &[i64], dst: &mut [i32]) {
    for (i, &v) in src.iter().enumerate().take(dst.len()) {
        dst[i] = v as i32;
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

        let mut subbands = Subbands::zeroed();
        forward_all_channels(&signal, &mut subbands);

        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        inverse_all_channels(&subbands, &mut reconstructed);

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
    fn dc_concentrates_in_l3_approx() {
        // Constant DC input should put all energy in L3 approx, zero in details.
        let mut signal = [[1000i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        let mut subbands = Subbands::zeroed();
        forward_all_channels(&signal, &mut subbands);

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
}
