#include <stdint.h>
#include <stdbool.h>
#include "../core/math_utils.h"
#include "lpc_predictor.h"

/*
 * LamQuant Gen 7.1 — LPC Predictor (Subband Pipeline Stage 2)
 * =============================================================
 * Order-8 Linear Predictive Coding on 21-channel EEG.
 *
 * Called after HP biquad, before 3-level lifting DWT.
 * Removes temporal redundancy per channel, producing prediction
 * residuals that decorrelate before wavelet decomposition.
 *
 * Optimization: autocorrelation computed on first 256 samples only.
 * EEG spectral envelope changes slowly within a 10-second window,
 * so coefficients estimated from 256 samples are nearly identical
 * to those from the full 2500. The prediction filter is then applied
 * to all 2500 samples.
 *
 * All Q31 integer arithmetic. No float, no malloc.
 *
 * Budget: ~0.8 ms for 21 channels at order 8 (with loop unrolling).
 *
 * ---- Legacy API (Lightning Path, order-4 on 6x32 tile) ----
 * void lpc_predict_only(int32_t tile[6][32])
 * This entry point is preserved for backward compatibility with
 * the Lightning Path scheduler.
 */

#define LPC_ORDER_MAX     8
#define LPC_AUTOCORR_LEN  256    /* Samples used for coefficient estimation */
#define NUM_CHANNELS      21
#define WINDOW_SAMPLES    2500

/* Legacy Lightning Path dimensions */
#define TILE_CHANNELS 6
#define TILE_SAMPLES  32
#define LPC_ORDER_LEGACY 4

/*
 * Compute biased autocorrelation R[0..order] from a signal segment.
 * Uses biased estimate (divide by N) which is required for Levinson-Durbin
 * to guarantee a positive-definite Toeplitz matrix and stable filter.
 * Accumulates in int64 to avoid overflow.
 */
/**
 * Autocorrelation with 4x loop unrolling.
 *
 * The inner loop is the hottest path in LPC analysis (~550K MACs for
 * 21 channels × 256 samples × 9 lags). Unrolling 4x reduces loop
 * overhead and lets Hazard3 pipeline the MULH instructions.
 */
static void autocorrelation(const int32_t* x, int len, int64_t* R, int order) {
    for (int k = 0; k <= order; k++) {
        int64_t acc = 0;
        int n = k;
        /* Unrolled: 4 MACs per iteration */
        int end4 = k + ((len - k) & ~3);  /* round down to multiple of 4 */
        for (; n < end4; n += 4) {
            acc += (int64_t)x[n]     * (int64_t)x[n - k];
            acc += (int64_t)x[n + 1] * (int64_t)x[n + 1 - k];
            acc += (int64_t)x[n + 2] * (int64_t)x[n + 2 - k];
            acc += (int64_t)x[n + 3] * (int64_t)x[n + 3 - k];
        }
        /* Remainder */
        for (; n < len; n++) {
            acc += (int64_t)x[n] * (int64_t)x[n - k];
        }
        R[k] = acc / len;
    }
}

/*
 * Levinson-Durbin recursion: solve for LPC coefficients from autocorrelation.
 *
 * Outputs 'order' LPC coefficients in Q31 format.
 * Returns true if successful, false if ill-conditioned (R[0] == 0).
 */
static bool levinson_durbin(const int64_t* R, int order, int32_t* a_q31) {
    if (R[0] == 0) return false;

    int64_t E = R[0];
    int32_t a_prev[LPC_ORDER_MAX];
    int32_t a_curr[LPC_ORDER_MAX];

    for (int i = 0; i < order; i++) {
        a_prev[i] = 0;
        a_curr[i] = 0;
    }

    for (int m = 0; m < order; m++) {
        /* Compute reflection coefficient k_m = -sum / E */
        int64_t sum = R[m + 1];
        for (int j = 0; j < m; j++) {
            sum += ((int64_t)a_prev[j] * R[m - j]) >> 31;
        }

        if (E == 0) return false;

        /* k in Q31: k = -sum / E, then scale to Q31.
         * BUG FIX: old code did `-(sum * (1LL<<31)) / E` which overflows
         * int64 for large sum values BEFORE the division, wrapping around
         * and producing wrong reflection coefficients for high-energy
         * signals. Dividing first avoids the overflow. */
        int64_t k_q31_64 = -(sum / E) * (int64_t)(1LL << 31);

        /* Clamp to Q31 range */
        if (k_q31_64 > INT32_MAX) k_q31_64 = INT32_MAX;
        if (k_q31_64 < INT32_MIN) k_q31_64 = INT32_MIN;
        int32_t k_q31 = (int32_t)k_q31_64;

        /* Update coefficients */
        a_curr[m] = k_q31;
        for (int j = 0; j < m; j++) {
            a_curr[j] = add_sat_q31(a_prev[j], mul_q31(k_q31, a_prev[m - 1 - j]));
        }

        /* Update error */
        E = E - ((int64_t)k_q31 * k_q31_64) / (1LL << 31);
        if (E <= 0) E = 1;

        /* Copy for next iteration */
        for (int j = 0; j <= m; j++) {
            a_prev[j] = a_curr[j];
        }
    }

    for (int i = 0; i < order; i++) {
        a_q31[i] = a_curr[i];
    }
    return true;
}

/*
 * Apply forward LPC prediction and compute residuals.
 *
 * Writes residuals to a separate output buffer (not in-place) so the
 * original signal remains available for reconstruction at the decoder.
 *
 * pred[n] = sum(a[k] * x[n-1-k], k=0..order-1)
 * residual[n] = x[n] - pred[n]
 *
 * First 'order' samples are copied as-is (no prediction possible).
 */
static void lpc_residuals(const int32_t* x, int32_t* residual, int len,
                          const int32_t* a_q31, int order) {
    /* First 'order' samples: copy directly */
    for (int n = 0; n < order; n++) {
        residual[n] = x[n];
    }

    /* Compute prediction residuals — unrolled for order 8.
     * Each mul_q31 is a MULH (single cycle on Hazard3). Unrolling
     * eliminates the inner k-loop branch overhead (8 branches/sample
     * → 0 branches/sample). For order != 8, falls back to generic. */
    if (order == 8) {
        for (int n = 8; n < len; n++) {
            int32_t pred = mul_q31(a_q31[0], x[n-1]);
            pred = add_sat_q31(pred, mul_q31(a_q31[1], x[n-2]));
            pred = add_sat_q31(pred, mul_q31(a_q31[2], x[n-3]));
            pred = add_sat_q31(pred, mul_q31(a_q31[3], x[n-4]));
            pred = add_sat_q31(pred, mul_q31(a_q31[4], x[n-5]));
            pred = add_sat_q31(pred, mul_q31(a_q31[5], x[n-6]));
            pred = add_sat_q31(pred, mul_q31(a_q31[6], x[n-7]));
            pred = add_sat_q31(pred, mul_q31(a_q31[7], x[n-8]));
            residual[n] = sub_sat_q31(x[n], pred);
        }
    } else {
        for (int n = order; n < len; n++) {
            int32_t pred = 0;
            for (int k = 0; k < order; k++) {
                pred = add_sat_q31(pred, mul_q31(a_q31[k], x[n - 1 - k]));
            }
            residual[n] = sub_sat_q31(x[n], pred);
        }
    }
}

/*
 * Apply forward LPC prediction in-place (legacy, for Lightning Path).
 * Processes backwards to avoid overwriting samples still needed.
 */
static void lpc_residuals_inplace(int32_t* x, int len,
                                  const int32_t* a_q31, int order) {
    for (int n = len - 1; n >= order; n--) {
        int32_t pred = 0;
        for (int k = 0; k < order; k++) {
            pred = add_sat_q31(pred, mul_q31(a_q31[k], x[n - 1 - k]));
        }
        x[n] = sub_sat_q31(x[n], pred);
    }
}

/* ================================================================
 * Gen 7.1 Public API — Order-8 on 21-channel EEG
 * ================================================================ */

/* Per-channel LPC coefficients, stored for packet transmission */
static int32_t lpc_coefficients[NUM_CHANNELS][LPC_ORDER_MAX] __attribute__((section(".scratch_x.lpc_coeffs")));

/*
 * Analyze 21-channel EEG: compute LPC coefficients and prediction residuals.
 *
 * - Autocorrelation on first 256 samples per channel (fast estimation)
 * - Levinson-Durbin to solve for order-8 coefficients
 * - Forward prediction filter on all 2500 samples
 *
 * Input:  input[21][2500]  — HP-filtered EEG (Q31, in raw_adc_buffer)
 * Output: coeffs[21][8]    — LPC coefficients per channel (Q31)
 *         residual[21][2500] — prediction residuals (Q31)
 *
 * If Levinson-Durbin fails for a channel (flat signal), that channel's
 * residual is a copy of the input and coefficients are zero.
 */
void lpc_analyze(const int32_t input[][2500],
                 int32_t coeffs[][LPC_ORDER_MAX],
                 int32_t residual[][2500]) {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        int64_t R[LPC_ORDER_MAX + 1];

        /* Autocorrelation on first 256 samples only */
        autocorrelation(input[ch], LPC_AUTOCORR_LEN, R, LPC_ORDER_MAX);

        if (levinson_durbin(R, LPC_ORDER_MAX, coeffs[ch])) {
            lpc_residuals(input[ch], residual[ch], WINDOW_SAMPLES,
                          coeffs[ch], LPC_ORDER_MAX);
        } else {
            /* Flat signal: copy input, zero coefficients */
            for (int i = 0; i < LPC_ORDER_MAX; i++)
                coeffs[ch][i] = 0;
            for (int n = 0; n < WINDOW_SAMPLES; n++)
                residual[ch][n] = input[ch][n];
        }

        /* Store coefficients for packet encoding */
        for (int i = 0; i < LPC_ORDER_MAX; i++)
            lpc_coefficients[ch][i] = coeffs[ch][i];
    }
}

/* Accessor for packet encoder */
const int32_t (*lpc_get_coefficients(void))[LPC_ORDER_MAX] {
    return (const int32_t (*)[LPC_ORDER_MAX])lpc_coefficients;
}

/* ================================================================
 * Legacy API — Order-4 on 6x32 Lightning Path tile
 * ================================================================
 * Preserved for backward compatibility with scheduler_v7.c.
 */

void lpc_predict_only(int32_t tile[TILE_CHANNELS][TILE_SAMPLES]) {
    for (int ch = 0; ch < TILE_CHANNELS; ch++) {
        int64_t R[LPC_ORDER_LEGACY + 1];
        int32_t a_q31[LPC_ORDER_LEGACY];

        autocorrelation(tile[ch], TILE_SAMPLES, R, LPC_ORDER_LEGACY);

        if (levinson_durbin(R, LPC_ORDER_LEGACY, a_q31)) {
            lpc_residuals_inplace(tile[ch], TILE_SAMPLES, a_q31, LPC_ORDER_LEGACY);
        }
    }
}
