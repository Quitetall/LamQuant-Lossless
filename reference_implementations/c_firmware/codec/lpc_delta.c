/*
 * LamQuant Gen 7.1 — LPC Coefficient Delta Encoding (Phase 5B)
 * =============================================================
 * Reduces LPC coefficient side-information from 672 bytes to ~168 bytes
 * by exploiting temporal redundancy across consecutive windows.
 *
 * EEG spectral envelope changes slowly over time. LPC coefficients
 * from adjacent 10-second windows are highly correlated. Delta encoding
 * transmits the difference between consecutive frames:
 *
 *   First window:       full Q31 coefficients (21 × 8 × 4 = 672 bytes)
 *   Subsequent windows: Q15 deltas (21 × 8 × 2 = 336 bytes)
 *                        or Q8 deltas (21 × 8 × 1 = 168 bytes) if small enough
 *
 * The encoder tracks whether deltas fit in Q8 (±0.5% of full range)
 * or Q15 (±50% of full range). If deltas exceed Q15, it falls back
 * to a full-coefficient keyframe.
 *
 * All integer arithmetic. No float.
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "lpc_delta.h"

#define LPC_ORDER     8
#define NUM_CHANNELS  21

/* ================================================================
 * State: previous frame's coefficients for delta computation
 * ================================================================ */

static int32_t prev_coeffs[NUM_CHANNELS][LPC_ORDER] __attribute__((section(".scratch_x.lpc_delta")));
static bool    has_prev = false;

/* ================================================================
 * Delta computation
 * ================================================================ */

/* Delta encoding mode */
typedef enum {
    LPC_DELTA_KEYFRAME = 0x00,   /* Full Q31 coefficients (672 bytes) */
    LPC_DELTA_Q15      = 0x01,   /* Q15 deltas (336 bytes) */
    LPC_DELTA_Q8       = 0x02,   /* Q8 deltas (168 bytes) */
} lpc_delta_mode_t;

/*
 * Compute deltas and determine encoding mode.
 *
 * @param curr       Current frame's LPC coefficients [21][8] (Q31)
 * @param deltas     Output: delta values [21][8] (Q31, to be quantized later)
 * @return encoding mode (keyframe, Q15, or Q8)
 */
static lpc_delta_mode_t compute_deltas(
    const int32_t curr[][LPC_ORDER],
    int32_t deltas[][LPC_ORDER]
) {
    if (!has_prev) {
        return LPC_DELTA_KEYFRAME;
    }

    int32_t max_abs_delta = 0;

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int k = 0; k < LPC_ORDER; k++) {
            int32_t d = curr[ch][k] - prev_coeffs[ch][k];
            deltas[ch][k] = d;

            int32_t abs_d = (d >= 0) ? d : -d;
            if (abs_d > max_abs_delta) max_abs_delta = abs_d;
        }
    }

    /*
     * Q8 range: [-128, 127] in Q31 units → max delta ≈ ±128 << 23 ≈ ±1.07e9
     * That covers ~50% of Q31 range — plenty for typical EEG LPC drift.
     *
     * Q8: delta >> 23, fits in int8_t → 168 bytes
     * Q15: delta >> 16, fits in int16_t → 336 bytes
     *
     * Check if deltas fit in Q8 first (most compact).
     */
    int32_t q8_max = (int32_t)(127) << 23;   /* ~1.065e9 */
    int32_t q15_max = (int32_t)(32767) << 16; /* ~2.147e9 (nearly full Q31) */

    if (max_abs_delta <= q8_max) {
        return LPC_DELTA_Q8;
    } else if (max_abs_delta <= q15_max) {
        return LPC_DELTA_Q15;
    } else {
        return LPC_DELTA_KEYFRAME;
    }
}

/* ================================================================
 * Encoding into byte buffer
 * ================================================================ */

/*
 * Encode LPC coefficients into a byte buffer.
 *
 * Format:
 *   Byte 0:      mode (0x00=keyframe, 0x01=Q15, 0x02=Q8)
 *   Bytes 1+:    payload
 *
 * @param curr     Current coefficients [21][8] Q31
 * @param out_buf  Output byte buffer (must be >= 673 bytes)
 * @return number of bytes written
 */
uint32_t lpc_delta_encode(
    const int32_t curr[][LPC_ORDER],
    uint8_t* out_buf
) {
    int32_t deltas[NUM_CHANNELS][LPC_ORDER];
    lpc_delta_mode_t mode = compute_deltas(curr, deltas);

    out_buf[0] = (uint8_t)mode;
    uint32_t pos = 1;

    switch (mode) {
        case LPC_DELTA_KEYFRAME:
            /* Full Q31 coefficients: 21 × 8 × 4 = 672 bytes */
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int32_t v = curr[ch][k];
                    out_buf[pos++] = (uint8_t)(v & 0xFF);
                    out_buf[pos++] = (uint8_t)((v >> 8) & 0xFF);
                    out_buf[pos++] = (uint8_t)((v >> 16) & 0xFF);
                    out_buf[pos++] = (uint8_t)((v >> 24) & 0xFF);
                }
            }
            break;

        case LPC_DELTA_Q15:
            /* Q15 deltas: 21 × 8 × 2 = 336 bytes */
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int16_t d16 = (int16_t)(deltas[ch][k] >> 16);
                    out_buf[pos++] = (uint8_t)(d16 & 0xFF);
                    out_buf[pos++] = (uint8_t)((d16 >> 8) & 0xFF);
                }
            }
            break;

        case LPC_DELTA_Q8:
            /* Q8 deltas: 21 × 8 × 1 = 168 bytes */
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int8_t d8 = (int8_t)(deltas[ch][k] >> 23);
                    out_buf[pos++] = (uint8_t)d8;
                }
            }
            break;
    }

    /* Update state for next frame */
    memcpy(prev_coeffs, curr, sizeof(prev_coeffs));
    has_prev = true;

    return pos;
}

/*
 * Decode LPC coefficients from a byte buffer.
 * Used by the base station / Python decoder.
 *
 * @param in_buf    Input byte buffer
 * @param out_coeffs Output coefficients [21][8] Q31
 * @return number of bytes consumed
 */
uint32_t lpc_delta_decode(
    const uint8_t* in_buf,
    int32_t out_coeffs[][LPC_ORDER]
) {
    lpc_delta_mode_t mode = (lpc_delta_mode_t)in_buf[0];
    uint32_t pos = 1;

    /* BUG FIX: If the decoder receives a delta frame (Q15/Q8) before any
     * keyframe, prev_coeffs is uninitialized → garbage LPC synthesis.
     * This can happen on packet loss or connection start. Require a
     * keyframe first; return 0 (error) otherwise. */
    if (!has_prev && mode != LPC_DELTA_KEYFRAME) {
        /* Cannot apply delta without a previous keyframe. Zero output
         * so the caller gets silence instead of garbage. */
        memset(out_coeffs, 0, sizeof(int32_t) * NUM_CHANNELS * LPC_ORDER);
        return 0;  /* 0 bytes consumed signals error to caller */
    }

    switch (mode) {
        case LPC_DELTA_KEYFRAME:
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int32_t v = (int32_t)in_buf[pos]
                              | ((int32_t)in_buf[pos+1] << 8)
                              | ((int32_t)in_buf[pos+2] << 16)
                              | ((int32_t)in_buf[pos+3] << 24);
                    out_coeffs[ch][k] = v;
                    pos += 4;
                }
            }
            break;

        case LPC_DELTA_Q15:
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int16_t d16 = (int16_t)((uint16_t)in_buf[pos]
                                          | ((uint16_t)in_buf[pos+1] << 8));
                    out_coeffs[ch][k] = prev_coeffs[ch][k] + ((int32_t)d16 << 16);
                    pos += 2;
                }
            }
            break;

        case LPC_DELTA_Q8:
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                for (int k = 0; k < LPC_ORDER; k++) {
                    int8_t d8 = (int8_t)in_buf[pos];
                    out_coeffs[ch][k] = prev_coeffs[ch][k] + ((int32_t)d8 << 23);
                    pos += 1;
                }
            }
            break;
    }

    /* Update state */
    memcpy(prev_coeffs, out_coeffs, sizeof(prev_coeffs));
    has_prev = true;

    return pos;
}

/* Reset state (e.g., after connection loss or power state change) */
void lpc_delta_reset(void) {
    memset(prev_coeffs, 0, sizeof(prev_coeffs));
    has_prev = false;
}
