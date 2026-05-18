/*
 * LamQuant Gen 7.1 — Detail Coefficient Thresholding (Phase 4A)
 * ==============================================================
 * SNN-driven hard thresholding of lifting DWT detail coefficients.
 *
 * Purpose:
 *   - Zero out small detail coefficients → sparser signal → fewer bits
 *   - Threshold level adapts to SNN activity classification:
 *       QUIESCENT: aggressive threshold (most coefficients zeroed)
 *       ACTIVE:    moderate threshold (preserve transients)
 *       SEIZURE:   minimal threshold (preserve full morphology)
 *   - L2 detail subband (31-62 Hz) includes 60 Hz power-line content.
 *     Thresholding this band replaces the Gen 7.0 notch filter for free.
 *
 * Algorithm:
 *   For each coefficient c in a detail subband:
 *     if |c| < threshold → c = 0  (hard threshold)
 *     else → c unchanged
 *
 * The threshold is a multiple of the subband's median absolute value (MAD),
 * precomputed per-channel at the start of each frame.
 *
 * All integer arithmetic. No float.
 */

#include <stdint.h>
#include <stdbool.h>
#include "../snn/snn.h"
#include "detail_threshold.h"

/* ================================================================
 * Quality modes (selectable by SNN auto or user command)
 * ================================================================ */

static volatile quality_mode_t current_quality = QUALITY_CLINICAL;

/* ================================================================
 * Threshold multipliers per quality mode
 * ================================================================
 *
 * Threshold = multiplier * MAD(subband) where MAD = median(|coeffs|).
 *
 * For integer MAD estimation, we use the sorted-middle approach
 * on a subsampled set (every 8th coefficient) to keep it fast.
 *
 * The multipliers are in Q8 fixed-point (256 = 1.0×):
 *   Clinical:   0.5× MAD → preserve fine detail
 *   Monitoring: 1.5× MAD → moderate sparsification
 *   Alerting:   3.0× MAD → aggressive (but alerting drops L1+L2 anyway)
 *
 * Per-subband multipliers allow tighter control:
 *   L1 detail (62-125 Hz): can be aggressive — mostly noise above EEG band
 *   L2 detail (31-62 Hz):  moderate — contains 60 Hz, some gamma
 *   L3 detail (16-31 Hz):  gentle — contains beta rhythm
 */

/* Threshold multipliers: [quality_mode][subband_level]
 * subband_level: 0=L3 detail, 1=L2 detail, 2=L1 detail
 * Values in Q8 (256 = 1.0× MAD) */
static const uint16_t THRESH_MULT[3][3] = {
    /* ALERTING:   doesn't matter (L1+L2 dropped entirely) */
    { 384, 768, 768 },    /* 1.5×, 3.0×, 3.0× */
    /* MONITORING: L1 dropped, L2+L3 moderate threshold */
    { 128, 256, 512 },    /* 0.5×, 1.0×, 2.0× */
    /* CLINICAL:   all details, gentle threshold */
    {  64, 128, 256 },    /* 0.25×, 0.5×, 1.0× */
};

/* ================================================================
 * MAD estimation (fast integer median approximation)
 * ================================================================
 *
 * Full sorting is O(N log N) — too slow for real-time.
 * Instead we estimate MAD by:
 *   1. Subsample every 8th coefficient
 *   2. Compute mean of absolute values (cheaper than median, close enough)
 *   3. Scale by 1.2 to approximate median/mean ratio for Laplacian dist
 *
 * For a Laplacian distribution (common for wavelet detail coefficients):
 *   median(|X|) ≈ 0.69 × mean(|X|)
 * But mean_abs is faster and the threshold multiplier absorbs the ratio.
 */

static int32_t estimate_mad(const int32_t* coeffs, int len) {
    if (len == 0) return 1;

    int64_t abs_sum = 0;
    int count = 0;

    /* Subsample every 8th coefficient for speed */
    for (int i = 0; i < len; i += 8) {
        int32_t v = coeffs[i];
        abs_sum += (v >= 0) ? (int64_t)v : -(int64_t)v;
        count++;
    }

    if (count == 0) return 1;
    return (int32_t)(abs_sum / count);
}

/* ================================================================
 * Hard threshold in-place
 * ================================================================ */

static int threshold_inplace(int32_t* coeffs, int len, int32_t threshold) {
    int nonzero = 0;
    for (int i = 0; i < len; i++) {
        int32_t v = coeffs[i];
        int32_t abs_v = (v >= 0) ? v : -v;
        if (abs_v < threshold) {
            coeffs[i] = 0;
        } else {
            nonzero++;
        }
    }
    return nonzero;
}

/* ================================================================
 * Public API
 * ================================================================ */

/* Subband lengths (from lifting_2d.c) */
#define L3_DETAIL_LEN 312
#define L2_DETAIL_LEN 625
#define L1_DETAIL_LEN 1250
#define NUM_CHANNELS  21

/*
 * Apply SNN-driven thresholding to detail subbands.
 *
 * Modifies subbands in-place: small coefficients → 0.
 * Returns total number of non-zero coefficients across all subbands
 * (useful for estimating compressed size).
 *
 * @param l3_detail  [21][312] — level 3 detail coefficients
 * @param l2_detail  [21][625] — level 2 detail coefficients
 * @param l1_detail  [21][1250] — level 1 detail coefficients
 * @param mode       quality mode (alerting/monitoring/clinical)
 * @return total non-zero coefficients across all included subbands
 */
int detail_threshold_apply(
    int32_t l3_detail[][L3_DETAIL_LEN],
    int32_t l2_detail[][L2_DETAIL_LEN],
    int32_t l1_detail[][L1_DETAIL_LEN],
    quality_mode_t mode
) {
    int total_nonzero = 0;

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        /* L3 detail (16-31 Hz): always included */
        {
            int32_t mad = estimate_mad(l3_detail[ch], L3_DETAIL_LEN);
            int32_t thresh = (int32_t)(((int64_t)mad * THRESH_MULT[mode][0]) >> 8);
            if (thresh < 1) thresh = 1;
            total_nonzero += threshold_inplace(l3_detail[ch], L3_DETAIL_LEN, thresh);
        }

        /* L2 detail (31-62 Hz): included in monitoring + clinical */
        if (mode >= QUALITY_MONITORING) {
            int32_t mad = estimate_mad(l2_detail[ch], L2_DETAIL_LEN);
            int32_t thresh = (int32_t)(((int64_t)mad * THRESH_MULT[mode][1]) >> 8);
            if (thresh < 1) thresh = 1;
            total_nonzero += threshold_inplace(l2_detail[ch], L2_DETAIL_LEN, thresh);
        }

        /* L1 detail (62-125 Hz): included in clinical only */
        if (mode >= QUALITY_CLINICAL) {
            int32_t mad = estimate_mad(l1_detail[ch], L1_DETAIL_LEN);
            int32_t thresh = (int32_t)(((int64_t)mad * THRESH_MULT[mode][2]) >> 8);
            if (thresh < 1) thresh = 1;
            total_nonzero += threshold_inplace(l1_detail[ch], L1_DETAIL_LEN, thresh);
        }
    }

    return total_nonzero;
}

/*
 * Auto-select quality mode based on sustained SNN activity.
 *
 * Logic:
 *   - If any group shows HIGH activity → CLINICAL (preserve seizure)
 *   - If >25% of timesteps show ACTIVE → MONITORING
 *   - Otherwise → ALERTING (bandwidth conservation)
 *
 * Can be overridden by user via serial command.
 */
quality_mode_t detail_threshold_auto_mode(void) {
    const uint8_t (*amap)[SNN_T_LATENT] = snn_get_activity_map();
    int active_count = 0;
    int total = SNN_READOUT_DIM * SNN_T_LATENT;

    for (int g = 0; g < SNN_READOUT_DIM; g++) {
        for (int t = 0; t < SNN_T_LATENT; t++) {
            if (amap[g][t] >= ACTIVITY_HIGH) {
                return QUALITY_CLINICAL;  /* Any seizure → full detail */
            }
            if (amap[g][t] >= ACTIVITY_ACTIVE) {
                active_count++;
            }
        }
    }

    if (active_count * 4 > total) {  /* >25% active */
        return QUALITY_MONITORING;
    }

    return QUALITY_ALERTING;
}

/* Mode control */
void detail_threshold_set_mode(quality_mode_t mode) {
    current_quality = mode;
}

quality_mode_t detail_threshold_get_mode(void) {
    return current_quality;
}

/*
 * Get which subbands are included for the current quality mode.
 * Returns a bitmask: bit 0 = L3 detail, bit 1 = L2 detail, bit 2 = L1 detail.
 */
uint8_t detail_threshold_subband_mask(quality_mode_t mode) {
    switch (mode) {
        case QUALITY_ALERTING:   return 0x01;  /* L3 only */
        case QUALITY_MONITORING: return 0x03;  /* L3 + L2 */
        case QUALITY_CLINICAL:   return 0x07;  /* L3 + L2 + L1 */
        default:                 return 0x01;
    }
}
