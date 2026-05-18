/*
 * LamQuant Gen 7.1 — Adaptive FSQ (Finite Scalar Quantization)
 * =============================================================
 * Variable-L quantization driven by SNN activity_map:
 *
 *   QUIESCENT (L=2):   2 codewords/dim → 1.00 bits → 32 bits/group
 *   ACTIVE    (L=3):   3 codewords/dim → 1.58 bits → 50.6 bits/group
 *   HIGH      (L=5):   5 codewords/dim → 2.32 bits → 74.2 bits/group
 *   CLINICAL  (L=32):  32 codewords/dim → 5.00 bits → 160 bits/group
 *
 * Gen 7.1 adds L=32 for clinical mode (quality_mode == QUALITY_CLINICAL).
 * When clinical mode is active, ALL timesteps use L=32 regardless of
 * SNN activity level — maximum fidelity for diagnostic-quality recording.
 *
 * The activity_map from the SNN determines L per (group, timestep).
 * FSQ level bitmap is transmitted in the BLE packet header.
 *
 * All Q31. Zero float. Uses precomputed inverse ranges per level.
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "../snn/snn.h"
#include "../core/math_utils.h"
#include "detail_threshold.h"
#include "fsq_adaptive.h"

/* Import vmin/vmax from trained weights */
#include "../firmware_export/focal_net_weights.h"

/* ================================================================
 * FSQ level configurations
 * ================================================================ */

/* Per-level quantization parameters (precomputed at export time) */
typedef struct {
    uint32_t num_levels;        /* L */
    int32_t  inv_range_q31;     /* (L * 2^31) / (vmax - vmin) */
} fsq_level_config_t;

/*
 * Four level configurations (Gen 7.1: added L=32 for clinical mode).
 * inv_range is computed from the trained vmin/vmax at export time.
 */
#define FSQ_NUM_CONFIGS 4

/* Note: FSQ_CONFIGS was a static const initializer template but is unused;
 * active_configs[] below is the mutable version populated at init. */

/* Mutable config with computed inv_range */
static fsq_level_config_t active_configs[FSQ_NUM_CONFIGS];

/* quality_mode_t and detail_threshold_get_mode() come from detail_threshold.h */

/* ================================================================
 * Initialization
 * ================================================================ */

void fsq_adaptive_init(void) {
    /*
     * Compute inv_range for each level configuration.
     * inv_range_q31 = (L * 2^31) / (vmax - vmin)
     * fsq_vmin_q31 / fsq_vmax_q31 are from focal_net_weights.h.
     */
    int64_t range = (int64_t)fsq_vmax_q31 - (int64_t)fsq_vmin_q31;
    if (range <= 0) range = 1;

    static const uint32_t LEVELS[FSQ_NUM_CONFIGS] = {2, 3, 5, 32};
    for (int i = 0; i < FSQ_NUM_CONFIGS; i++) {
        active_configs[i].num_levels = LEVELS[i];
        active_configs[i].inv_range_q31 =
            (int32_t)(((int64_t)LEVELS[i] << 31) / range);
    }
}

/* ================================================================
 * Activity → FSQ level mapping
 * ================================================================ */

static inline int activity_to_config_idx(activity_level_t level) {
    /* In clinical mode, always use L=32 regardless of activity */
    if (detail_threshold_get_mode() == QUALITY_CLINICAL) {
        return 3;  /* L=32 */
    }

    switch (level) {
        case ACTIVITY_QUIESCENT: return 0;  /* L=2 */
        case ACTIVITY_ACTIVE:    return 1;  /* L=3 */
        case ACTIVITY_HIGH:      return 2;  /* L=5 */
        default:                 return 0;
    }
}

/* ================================================================
 * Quantize one latent value with variable L
 * ================================================================ */

static inline uint32_t fsq_quantize_adaptive(int32_t val, int config_idx) {
    const fsq_level_config_t *cfg = &active_configs[config_idx];

    /* Shift to zero-based: val - vmin */
    int32_t shifted = val - fsq_vmin_q31;
    if (shifted < 0) shifted = 0;

    /* Multiply by precomputed inverse range */
    int64_t product = (int64_t)shifted * (int64_t)cfg->inv_range_q31;
    int32_t bin = (int32_t)(product >> 31);

    /* Clamp */
    if (bin < 0) bin = 0;
    if ((uint32_t)bin >= cfg->num_levels) bin = (int32_t)(cfg->num_levels - 1);

    return (uint32_t)bin;
}

/* ================================================================
 * Encode full latent with adaptive FSQ
 * ================================================================
 *
 * latent: [32][312] Q31 from TNN encoder
 * The 32 latent dims are mapped to 8 spatial groups (4 dims each).
 * Each group gets its own L based on the SNN activity map.
 *
 * Returns: number of symbols written to output buffer.
 */

/* Latent dim → spatial group mapping (4 dims per group) */
#define DIMS_PER_GROUP 4

static uint32_t fsq_output[32 * SNN_T_LATENT];  /* Max symbols */
static uint32_t fsq_output_len;
static uint8_t  fsq_level_bitmap[SNN_T_LATENT];  /* L index per timestep (packed) */

uint32_t fsq_adaptive_encode(const int32_t latent[][79], int T_latent) {
    const uint8_t (*amap)[SNN_T_LATENT] = snn_get_activity_map();
    fsq_output_len = 0;

    for (int t = 0; t < T_latent && t < SNN_T_LATENT; t++) {
        /*
         * Compute per-timestep level: take max activity across groups.
         * This determines L for ALL dims at this timestep.
         * (Simpler than per-group — only 2 bits side info per timestep.)
         */
        activity_level_t max_activity = ACTIVITY_QUIESCENT;
        for (int g = 0; g < SNN_READOUT_DIM; g++) {
            activity_level_t a = (activity_level_t)amap[g][t];
            if (a > max_activity) max_activity = a;
        }

        int cfg_idx = activity_to_config_idx(max_activity);
        fsq_level_bitmap[t] = (uint8_t)cfg_idx;

        /* Quantize all 32 dims at this timestep */
        for (int d = 0; d < 32; d++) {
            fsq_output[fsq_output_len++] = fsq_quantize_adaptive(latent[d][t], cfg_idx);
        }
    }

    return fsq_output_len;
}

/* ================================================================
 * Accessors for rANS encoder
 * ================================================================ */

const uint32_t* fsq_get_symbols(void) {
    return fsq_output;
}

uint32_t fsq_get_symbol_count(void) {
    return fsq_output_len;
}

const uint8_t* fsq_get_level_bitmap(void) {
    return fsq_level_bitmap;
}

/* Get the number of levels used at timestep t */
uint32_t fsq_get_num_levels_at(int t) {
    if (t < 0 || t >= SNN_T_LATENT) return 2;
    return active_configs[fsq_level_bitmap[t]].num_levels;
}

/*
 * Build FSQ level summary for BLE packet header.
 * Returns 16-bit bitmap: 2 bits per group, packed into 2 bytes.
 * (For the whole frame, uses most frequent level per group.)
 */
uint16_t fsq_build_level_summary(void) {
    uint16_t summary = 0;

    for (int g = 0; g < 8; g++) {
        /* Count predominant activity level for this group */
        int counts[3] = {0, 0, 0};
        const uint8_t (*amap)[SNN_T_LATENT] = snn_get_activity_map();
        for (int t = 0; t < SNN_T_LATENT; t++) {
            int a = amap[g][t];
            if (a < 3) counts[a]++;
        }

        /* Pick most frequent */
        int best = 0;
        if (counts[1] > counts[best]) best = 1;
        if (counts[2] > counts[best]) best = 2;

        summary |= ((uint16_t)best & 0x03) << (g * 2);
    }

    return summary;
}
