//! SNN-driven hard thresholding of lifting DWT detail coefficients.
//!
//! Zero out small detail coefficients → sparser signal → fewer bits.
//! The threshold per subband adapts to the active QualityMode and uses
//! a fast MAD (median-absolute-deviation) estimate.
//!
//! L2 detail covers 31-62 Hz which includes 60 Hz mains content.
//! Thresholding this band replaces the Gen 7.0 notch filter for free.

use super::quality::QualityMode;

use crate::dsp::lifting::{L1_DETAIL_LEN, L2_DETAIL_LEN, L3_DETAIL_LEN};

const NUM_CHANNELS: usize = 21;

/// Threshold multipliers `[QualityMode][subband_level]` in Q8 (256 = 1.0×).
///   subband index: 0 = L3 detail, 1 = L2 detail, 2 = L1 detail
///   QualityMode index: 0 = Alerting, 1 = Monitoring, 2 = Clinical
const THRESH_MULT: [[u16; 3]; 3] = [
    // Alerting:    L1+L2 dropped entirely; multipliers don't matter.
    [384, 768, 768], // 1.5×, 3.0×, 3.0×
    // Monitoring:  L1 dropped, L2+L3 moderate threshold.
    [128, 256, 512], // 0.5×, 1.0×, 2.0×
    // Clinical:    all details, gentle threshold.
    [64, 128, 256], // 0.25×, 0.5×, 1.0×
];

/// Fast integer MAD estimate (subsampled mean-absolute, every 8th sample).
///
/// For Laplacian-distributed wavelet detail coefficients,
/// `median(|X|) ≈ 0.69 × mean(|X|)`. The threshold multiplier absorbs
/// the ratio, so we use the cheaper mean.
fn estimate_mad(coeffs: &[i32]) -> i32 {
    if coeffs.is_empty() {
        return 1;
    }
    let mut abs_sum: i64 = 0;
    let mut count: u32 = 0;
    let mut i = 0;
    while i < coeffs.len() {
        let v = coeffs[i];
        abs_sum += v.unsigned_abs() as i64;
        count += 1;
        i += 8;
    }
    if count == 0 {
        return 1;
    }
    (abs_sum / count as i64) as i32
}

/// Hard threshold in-place. Returns count of non-zero coefficients.
fn threshold_inplace(coeffs: &mut [i32], threshold: i32) -> usize {
    let mut nonzero = 0;
    for v in coeffs.iter_mut() {
        if v.unsigned_abs() < threshold as u32 {
            *v = 0;
        } else {
            nonzero += 1;
        }
    }
    nonzero
}

/// Apply thresholding to detail subbands. Modifies in-place.
///
/// Returns total non-zero coefficients across included subbands (for
/// downstream compressed-size estimation).
///
/// Subband inclusion by quality mode:
///   Alerting   → L3 only
///   Monitoring → L3 + L2
///   Clinical   → L3 + L2 + L1
pub fn apply(
    l3_detail: &mut [[i32; L3_DETAIL_LEN]; NUM_CHANNELS],
    l2_detail: &mut [[i32; L2_DETAIL_LEN]; NUM_CHANNELS],
    l1_detail: &mut [[i32; L1_DETAIL_LEN]; NUM_CHANNELS],
    mode: QualityMode,
) -> usize {
    let mode_idx = mode as usize;
    let mut total_nonzero = 0;

    for ch in 0..NUM_CHANNELS {
        // L3 detail (16-31 Hz): always included.
        {
            let mad = estimate_mad(&l3_detail[ch]);
            let thresh = ((mad as i64 * THRESH_MULT[mode_idx][0] as i64) >> 8).max(1) as i32;
            total_nonzero += threshold_inplace(&mut l3_detail[ch], thresh);
        }

        // L2 detail (31-62 Hz): monitoring + clinical.
        if mode >= QualityMode::Monitoring {
            let mad = estimate_mad(&l2_detail[ch]);
            let thresh = ((mad as i64 * THRESH_MULT[mode_idx][1] as i64) >> 8).max(1) as i32;
            total_nonzero += threshold_inplace(&mut l2_detail[ch], thresh);
        }

        // L1 detail (62-125 Hz): clinical only.
        if mode >= QualityMode::Clinical {
            let mad = estimate_mad(&l1_detail[ch]);
            let thresh = ((mad as i64 * THRESH_MULT[mode_idx][2] as i64) >> 8).max(1) as i32;
            total_nonzero += threshold_inplace(&mut l1_detail[ch], thresh);
        }
    }

    total_nonzero
}

/// Bitmask of subbands included for a given quality mode.
///   bit 0 = L3 detail, bit 1 = L2 detail, bit 2 = L1 detail.
pub const fn subband_mask(mode: QualityMode) -> u8 {
    match mode {
        QualityMode::Alerting => 0b001,   // L3 only
        QualityMode::Monitoring => 0b011, // L3 + L2
        QualityMode::Clinical => 0b111,   // L3 + L2 + L1
    }
}
