/*
 * LamQuant Gen 7 — Q2D2 Hexagonal Paired Quantization
 * =====================================================
 * Pairs 32 latent dims into 16 pairs. Each pair is quantized onto a
 * hexagonal grid for ~15% lower quantization error vs scalar FSQ
 * (denser lattice packing: hex covers ~90.69% vs rectangular ~78.54%).
 *
 * Algorithm per pair (x, y):
 *   1. Compute hex row:    row = round(y / dy)
 *      where dy = sqrt(3)/2 * dx (precomputed as Q31)
 *   2. Compute hex column: col = round((x - offset) / dx)
 *      where offset = (row & 1) ? dx/2 : 0
 *   3. Check 3 nearest candidates, pick closest (Euclidean in Q31)
 *   4. Map (row, col) to linear symbol index
 *
 * ~10 integer ops per pair (vs 5 per scalar FSQ).
 * Hex grid LUTs for L=2, L=3, L=5.
 *
 * ZERO float. All Q31 fixed-point.
 */

#include <stdint.h>
#include <stdbool.h>
#include "../core/math_utils.h"
#include "fsq_hex.h"

/* ================================================================
 * Hex grid constants (Q31)
 * ================================================================
 *
 * For a hex grid with L levels per axis:
 *   dx = range / (L - 1)                  — horizontal spacing
 *   dy = dx * sqrt(3)/2                   — vertical spacing
 *   sqrt(3)/2 ≈ 0.866025 → Q31: 0x6ED9EBA1
 *
 * Precomputed at export time for the trained vmin/vmax range.
 * Placeholder values assume range = 2.0 (from -1.0 to +1.0).
 */

#define SQRT3_OVER_2_Q31  0x6ED9EBA1   /* sqrt(3)/2 in Q31 */

/* Per-level hex grid configuration */
typedef struct {
    uint32_t L;                /* Levels per axis */
    int32_t  dx_q31;           /* Horizontal spacing in Q31 */
    int32_t  dy_q31;           /* Vertical spacing in Q31 */
    int32_t  dx_half_q31;      /* dx/2 for odd-row offset */
    int32_t  inv_dx_q31;       /* 1/dx in Q31 for fast division */
    int32_t  inv_dy_q31;       /* 1/dy in Q31 for fast division */
    uint32_t total_symbols;    /* L*L codewords per pair */
} hex_config_t;

/*
 * Placeholder configs — will be filled by init from trained range.
 * For range=2.0:
 *   L=2: dx=2.0, dy=1.732  → 4 codewords per pair
 *   L=3: dx=1.0, dy=0.866  → 9 codewords per pair
 *   L=5: dx=0.5, dy=0.433  → 25 codewords per pair
 */
static hex_config_t hex_configs[3];

/* ================================================================
 * Initialization
 * ================================================================ */

void fsq_hex_init(int32_t vmin_q31, int32_t vmax_q31) {
    int64_t range = (int64_t)vmax_q31 - (int64_t)vmin_q31;
    if (range <= 0) range = (int64_t)0x7FFFFFFF * 2;  /* Default 2.0 */

    static const uint32_t LEVELS[3] = {2, 3, 5};

    for (int i = 0; i < 3; i++) {
        uint32_t L = LEVELS[i];
        hex_configs[i].L = L;

        /* dx = range / (L - 1), but L=2 → dx = range */
        int64_t dx64;
        if (L <= 1) {
            dx64 = range;
        } else {
            dx64 = range / (int64_t)(L - 1);
        }
        hex_configs[i].dx_q31 = (int32_t)dx64;
        hex_configs[i].dx_half_q31 = (int32_t)(dx64 / 2);

        /* dy = dx * sqrt(3)/2 */
        hex_configs[i].dy_q31 = (int32_t)(
            ((int64_t)hex_configs[i].dx_q31 * (int64_t)SQRT3_OVER_2_Q31) >> 31);

        /* Inverse: inv_dx = (1<<31) / dx */
        if (hex_configs[i].dx_q31 != 0) {
            hex_configs[i].inv_dx_q31 = (int32_t)(
                ((int64_t)1 << 62) / (int64_t)hex_configs[i].dx_q31);
        }
        if (hex_configs[i].dy_q31 != 0) {
            hex_configs[i].inv_dy_q31 = (int32_t)(
                ((int64_t)1 << 62) / (int64_t)hex_configs[i].dy_q31);
        }

        hex_configs[i].total_symbols = L * L;
    }
}

/* ================================================================
 * Q31 distance squared (no sqrt needed for comparison)
 * ================================================================ */

static inline int64_t dist_sq_q31(int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    int64_t dx = (int64_t)x0 - (int64_t)x1;
    int64_t dy = (int64_t)y0 - (int64_t)y1;
    return (dx * dx + dy * dy) >> 31;  /* Scale down to avoid overflow */
}

/* ================================================================
 * Hex grid nearest-neighbor quantization for one pair
 * ================================================================ */

static inline uint32_t hex_quantize_pair(
    int32_t x, int32_t y,
    int32_t vmin, int config_idx
) {
    const hex_config_t *cfg = &hex_configs[config_idx];

    /* Shift to zero-based */
    int32_t xs = x - vmin;
    int32_t ys = y - vmin;

    /* Approximate row: round(ys / dy) */
    int32_t row_approx = (int32_t)(((int64_t)ys * (int64_t)cfg->inv_dy_q31) >> 31);

    /* Clamp row */
    if (row_approx < 0) row_approx = 0;
    if ((uint32_t)row_approx >= cfg->L) row_approx = (int32_t)(cfg->L - 1);

    /* X offset for odd rows */
    int32_t x_offset = (row_approx & 1) ? cfg->dx_half_q31 : 0;

    /* Approximate column: round((xs - offset) / dx) */
    int32_t col_approx = (int32_t)(
        ((int64_t)(xs - x_offset) * (int64_t)cfg->inv_dx_q31) >> 31);

    if (col_approx < 0) col_approx = 0;
    if ((uint32_t)col_approx >= cfg->L) col_approx = (int32_t)(cfg->L - 1);

    /*
     * Check 3 nearest candidates:
     *   (row, col), (row, col+1), (row±1, col_adjusted)
     * Pick the one with minimum Euclidean distance.
     */
    int32_t best_row = row_approx;
    int32_t best_col = col_approx;
    int64_t best_dist = 0x7FFFFFFFFFFFFFFFLL;

    /* Candidate grid points to check */
    int32_t candidates[][2] = {
        {row_approx, col_approx},
        {row_approx, col_approx + 1},
        {row_approx - 1, col_approx},
        {row_approx + 1, col_approx},
    };

    for (int c = 0; c < 4; c++) {
        int32_t cr = candidates[c][0];
        int32_t cc = candidates[c][1];

        /* Bounds check */
        if (cr < 0 || (uint32_t)cr >= cfg->L) continue;
        if (cc < 0 || (uint32_t)cc >= cfg->L) continue;

        /* Compute grid point position */
        int32_t gx = (int32_t)((int64_t)cc * (int64_t)cfg->dx_q31 >> 0)
                    + ((cr & 1) ? cfg->dx_half_q31 : 0);
        int32_t gy = (int32_t)((int64_t)cr * (int64_t)cfg->dy_q31 >> 0);

        int64_t d = dist_sq_q31(xs, ys, gx, gy);
        if (d < best_dist) {
            best_dist = d;
            best_row = cr;
            best_col = cc;
        }
    }

    /* Linear symbol index: row * L + col */
    return (uint32_t)best_row * cfg->L + (uint32_t)best_col;
}

/* ================================================================
 * Quantize all 32 latent dims as 16 pairs
 * ================================================================ */

/* Pair mapping: (dim 0, dim 1), (dim 2, dim 3), ..., (dim 30, dim 31) */
#define NUM_PAIRS 16

static uint32_t hex_symbols[NUM_PAIRS * 312];  /* Max output */
static uint32_t hex_symbol_count;

uint32_t fsq_hex_encode(
    const int32_t latent[][312],
    int T_latent,
    const uint8_t *level_bitmap,  /* Per-timestep config index from fsq_adaptive */
    int32_t vmin_q31
) {
    hex_symbol_count = 0;

    for (int t = 0; t < T_latent; t++) {
        int cfg_idx = level_bitmap[t];
        if (cfg_idx > 2) cfg_idx = 0;

        for (int p = 0; p < NUM_PAIRS; p++) {
            int d0 = p * 2;
            int d1 = p * 2 + 1;

            uint32_t sym = hex_quantize_pair(
                latent[d0][t], latent[d1][t],
                vmin_q31, cfg_idx);

            hex_symbols[hex_symbol_count++] = sym;
        }
    }

    return hex_symbol_count;
}

/* Accessors */
const uint32_t* fsq_hex_get_symbols(void) {
    return hex_symbols;
}

uint32_t fsq_hex_get_symbol_count(void) {
    return hex_symbol_count;
}

uint32_t fsq_hex_get_total_codewords(int config_idx) {
    if (config_idx < 0 || config_idx > 2) return 4;
    return hex_configs[config_idx].total_symbols;
}
