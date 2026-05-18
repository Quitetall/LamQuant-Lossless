#include <stdint.h>
#include "lifting_2d.h"

/*
 * LamQuant Gen 7.1 — Multi-Level Le Gall 5/3 Lifting Wavelet
 * ============================================================
 *
 * Two modes of operation:
 *
 * 1. Gen 7.1 Subband Pipeline (new):
 *    3-level temporal lifting on 21-channel LPC residual [21][2500].
 *    Produces subband decomposition:
 *      L3 approximation [21][313]  → TNN encoder input
 *      L3 detail [21][312]         → sparse encoding (Golomb-Rice)
 *      L2 detail [21][625]         → sparse encoding
 *      L1 detail [21][1250]        → sparse encoding (mode-dependent)
 *
 * 2. Lightning Path (legacy, preserved):
 *    Single-level 2D lifting on 6x32 compressed sensing tile.
 *
 * Mathematical formulation (Le Gall 5/3 integer lifting):
 *   Predict (detail):        d[n] = x[2n+1] - (x[2n] + x[2n+2]) >> 1
 *   Update (approximation):  s[n] = x[2n] + (d[n-1] + d[n] + 2) >> 2
 *
 * All integer arithmetic. No float. No malloc.
 */

/* ================================================================
 * Core 1D lifting primitive (shared by all modes)
 * ================================================================ */

/**
 * In-place 1D Le Gall 5/3 lifting.
 * After transform, even indices hold approximation coefficients
 * and odd indices hold detail coefficients (interleaved).
 *
 * Handles both even and odd input lengths:
 *   even length N: N/2 approx + N/2 detail
 *   odd length N:  (N+1)/2 approx + N/2 detail
 *
 * @param signal  Buffer to transform in-place
 * @param length  Number of samples (must be >= 2)
 */
static void lifting_1d_53_inplace(int32_t* signal, int length) {
    if (length < 2) return;

    int n_detail = length / 2;          /* Number of detail (odd) coefficients */
    int n_approx = (length + 1) / 2;    /* Number of approx (even) coefficients */

    /* Step 1: Predict (detail coefficients at odd indices)
     * d[n] = x[2n+1] - (x[2n] + x[2n+2]) >> 1 */
    for (int n = 0; n < n_detail - 1; n++) {
        signal[2*n + 1] -= (signal[2*n] + signal[2*n + 2]) >> 1;
    }
    /* Last detail: mirrored boundary */
    if (n_detail > 0) {
        int last_odd = 2*(n_detail - 1) + 1;
        int last_even = 2*(n_detail - 1);
        if (last_odd < length) {
            /* If length is even, last odd mirrors from last even:
             * d[n] = x[2n+1] - (x[2n] + x[2n]) >> 1  (mirror boundary)
             * BUG FIX: was `signal[last_odd] -= signal[last_even]`
             * which skipped the >>1 averaging. */
            if (length % 2 == 0) {
                signal[last_odd] -= (signal[last_even] + signal[last_even]) >> 1;
            } else {
                /* Length is odd: x[2n+2] exists at index length-1 */
                signal[last_odd] -= (signal[last_even] + signal[last_odd + 1]) >> 1;
            }
        }
    }

    /* Step 2: Update (approximation coefficients at even indices)
     * s[n] = x[2n] + (d[n-1] + d[n] + 2) >> 2 */
    signal[0] += (signal[1] + 1) >> 1;  /* Boundary: mirror d[-1] = d[0] */

    for (int n = 1; n < n_approx; n++) {
        int left_detail = 2*n - 1;
        int right_detail = 2*n + 1;
        if (right_detail < length) {
            /* Symmetric rounding: round toward zero for both positive and
             * negative sums. The old `(sum + 2) >> 2` rounded negative values
             * away from zero, causing systematic bias on depression/sleep EEG. */
            int32_t sum = signal[left_detail] + signal[right_detail];
            signal[2*n] += (sum >= 0) ? (sum + 2) >> 2 : -(((-sum) + 2) >> 2);
        } else {
            /* Boundary: mirror right detail = left detail */
            signal[2*n] += (signal[left_detail] + 1) >> 1;
        }
    }
}

/* ================================================================
 * Gen 7.1 Subband Pipeline: 3-level temporal lifting on [21][2500]
 * ================================================================ */

/*
 * Subband output structure.
 * After 3-level lifting, the interleaved buffer can be de-interleaved
 * into separate subbands. This struct provides pointers and lengths.
 *
 * Level decomposition for 2500 samples:
 *   Level 1: 2500 → 1250 approx + 1250 detail
 *   Level 2: 1250 → 625 approx + 625 detail
 *   Level 3: 625  → 313 approx + 312 detail  (625 is ODD)
 *
 * The lifting operates in-place on the residual buffer. After all
 * 3 levels, the buffer is fully interleaved. We then de-interleave
 * into the output struct.
 */

/* Subband constants and lifting_subbands_t are defined in lifting_2d.h */

/* Static allocation in SRAM — shared workspace */
static lifting_subbands_t subbands
    __attribute__((aligned(4)));

/* Temporary buffers for de-interleaving one level at a time */
static int32_t approx_tmp[1250]  /* Max intermediate approx length */
    __attribute__((aligned(4)));
static int32_t detail_tmp[1250]  /* Max intermediate detail length */
    __attribute__((aligned(4)));

/**
 * De-interleave a lifted signal into separate approx and detail arrays.
 * After lifting, even indices = approximation, odd indices = detail.
 *
 * @param interleaved  Input buffer (in-place lifted)
 * @param length       Length of the interleaved buffer
 * @param approx       Output approximation coefficients
 * @param detail       Output detail coefficients
 * @param n_approx     Output: number of approximation coefficients
 * @param n_detail     Output: number of detail coefficients
 */
static void deinterleave(const int32_t* interleaved, int length,
                         int32_t* approx, int32_t* detail,
                         int* n_approx, int* n_detail) {
    *n_approx = (length + 1) / 2;
    *n_detail = length / 2;
    for (int i = 0; i < *n_approx; i++)
        approx[i] = interleaved[2 * i];
    for (int i = 0; i < *n_detail; i++)
        detail[i] = interleaved[2 * i + 1];
}

/**
 * 3-level temporal lifting DWT on 21-channel LPC residual.
 *
 * @param residual  Input: LPC prediction residual [21][2500] (Q31)
 *                  WARNING: This buffer is modified in-place during processing.
 * @return Pointer to the static subbands structure.
 *
 * Subband structure after decomposition:
 *   l3_approx[21][313]  — coarsest approximation (TNN input)
 *   l3_detail[21][312]  — finest detail at level 3
 *   l2_detail[21][625]  — detail at level 2
 *   l1_detail[21][1250] — detail at level 1 (widest band)
 */
const lifting_subbands_t* lifting_3level(int32_t residual[][2500]) {
    for (int ch = 0; ch < 21; ch++) {
        /*
         * Level 1: 2500 samples → 1250 approx + 1250 detail
         * 2500 is even, so n_approx = n_detail = 1250
         */
        lifting_1d_53_inplace(residual[ch], 2500);

        int n_a1, n_d1;
        deinterleave(residual[ch], 2500, approx_tmp, subbands.l1_detail[ch],
                     &n_a1, &n_d1);
        /* n_a1 = 1250, n_d1 = 1250 */

        /*
         * Level 2: 1250 approx → 625 approx + 625 detail
         * 1250 is even, so n_approx = n_detail = 625
         */
        lifting_1d_53_inplace(approx_tmp, n_a1);

        int n_a2, n_d2;
        deinterleave(approx_tmp, n_a1, detail_tmp, subbands.l2_detail[ch],
                     &n_a2, &n_d2);
        /* Swap: detail_tmp now holds level-2 approximation (625 samples) */
        /* n_a2 = 625, n_d2 = 625 */

        /*
         * Level 3: 625 approx → 313 approx + 312 detail
         * 625 is ODD: n_approx = (625+1)/2 = 313, n_detail = 625/2 = 312
         */
        lifting_1d_53_inplace(detail_tmp, n_a2);

        int n_a3, n_d3;
        deinterleave(detail_tmp, n_a2,
                     subbands.l3_approx[ch], subbands.l3_detail[ch],
                     &n_a3, &n_d3);
        /* n_a3 = 313, n_d3 = 312 */
    }

    return &subbands;
}

/* Accessors for downstream pipeline stages */
const int32_t (*lifting_get_l3_approx(void))[SUBBAND_L3_APPROX_LEN] {
    return (const int32_t (*)[SUBBAND_L3_APPROX_LEN])subbands.l3_approx;
}

int lifting_get_l3_approx_len(void) {
    return SUBBAND_L3_APPROX_LEN;
}

/* ================================================================
 * Legacy API: 2D lifting on 6x32 Lightning Path tile
 * ================================================================
 * Preserved for backward compatibility with scheduler_v7.c.
 */

void run_2d_lifting(int32_t tile[6][32]) {
    /* Vertical pass: temporal axis (32 samples per channel) */
    for (int ch = 0; ch < 6; ch++) {
        lifting_1d_53_inplace(tile[ch], 32);
    }

    /* Horizontal pass: spatial axis (6 channels cross-linked) */
    for (int t = 0; t < 32; t++) {
        int space_half_len = 3;

        /* Predict phase */
        for (int n = 0; n < space_half_len - 1; n++) {
            tile[2*n + 1][t] -= (tile[2*n][t] + tile[2*n + 2][t]) >> 1;
        }
        tile[5][t] -= tile[4][t];

        /* Update phase */
        tile[0][t] += (tile[1][t] + 1) >> 1;
        for (int n = 1; n < space_half_len; n++) {
            tile[2*n][t] += (tile[2*n - 1][t] + tile[2*n + 1][t] + 2) >> 2;
        }
    }
}

/* Legacy fallback stub */
void run_static_lpc_fallback(int32_t tile[6][32]) {
    (void)tile;
    __asm volatile("nop");
}
