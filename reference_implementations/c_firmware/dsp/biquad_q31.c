#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "../core/math_utils.h"
#include "dsp_config.h"
#include "biquad_q31.h"

/*
 * LamQuant Gen 7.1 — Biquad Prefilter (Stage 1)
 * ================================================
 * Single-stage highpass per channel for DC removal. The coefficient row is
 * picked at runtime from dsp_config (set by the host over the serial 'F'
 * command), so the user can pick HP_OFF / HP_0_1 / HP_0_5 / HP_1_0 without
 * reflashing. The remaining filter shaping the user used to ask for is now
 * done downstream:
 *
 *   - LP filtering is replaced by the 3-level lifting DWT (L3 approximation
 *     is inherently bandlimited to 0-31.25 Hz).
 *   - 60 Hz notch is replaced by detail coefficient thresholding (60 Hz
 *     content lives in L2 detail subband at 31-62 Hz).
 *
 * Per-channel enable: a 21-bit mask in the active config. Disabled channels
 * are zeroed in raw_adc_buffer at the end of run_biquad_prefilter() so
 * downstream stages see a stable signal and the SNN does not chase pickup.
 *
 * ALL coefficients are Q30 constants, precomputed by export_firmware.py.
 * Q30 format represents values in [-2.0, +2.0) with 30 fractional bits.
 * This is necessary because the HP biquad's b1 ≈ -1.98 overflows Q31's
 * [-1.0, +1.0) range.
 *
 * ZERO float at compile time or runtime. No math.h, no libm, no soft-float.
 */

typedef struct {
    int32_t b0, b1, b2;  // Numerator (Q30)
    int32_t a1, a2;       // Denominator (Q30)
    int32_t x1, x2;       // Input delay line
    int32_t y1, y2;       // Output delay line
} biquad_state_t;

// --- Q30 Direct Form 1 (pure integer) ---

static inline int32_t biquad_process(biquad_state_t* S, int32_t x0) {
    int32_t acc;
    acc = mul_q30(S->b0, x0);
    acc = add_sat_q31(acc, mul_q30(S->b1, S->x1));
    acc = add_sat_q31(acc, mul_q30(S->b2, S->x2));
    acc = sub_sat_q31(acc, mul_q30(S->a1, S->y1));
    acc = sub_sat_q31(acc, mul_q30(S->a2, S->y2));
    S->x2 = S->x1; S->x1 = x0;
    S->y2 = S->y1; S->y1 = acc;
    return acc;
}

static void init_filter(biquad_state_t* S, const int32_t coeffs[5]) {
    S->b0 = coeffs[0];
    S->b1 = coeffs[1];
    S->b2 = coeffs[2];
    S->a1 = coeffs[3];
    S->a2 = coeffs[4];
    S->x1 = S->x2 = S->y1 = S->y2 = 0;
}

#define NUM_CHANNELS 21

static biquad_state_t hp_filters[NUM_CHANNELS] __attribute__((section(".scratch_x.hp_filters")));
static bool filters_initialized = false;

/* Snapshot of the HP option the per-channel state was bound to, so we can
 * re-bind cheaply when the host picks a new option. */
static hp_filter_t bound_hp = HP_NUM_OPTIONS;

// 21-channel ADC buffer (DMA target, SRAM0)
int32_t raw_adc_buffer[NUM_CHANNELS][2500]
    __attribute__((aligned(4)));

/* --- Power state helper (called by power_states.c safe mode and by
 *     dsp_set_filter_config() when the host changes options) --- */

void dsp_reset_pipeline(void) {
    // Force re-initialization on next run_biquad_prefilter() call.
    // This zeros all 21 HP filter delay lines.
    filters_initialized = false;
}

/*
 * Stage 1 entry point. Single HP biquad on all 21 channels, in-place.
 * LP filtering and 60 Hz rejection are handled by the lifting DWT
 * and detail coefficient thresholding in Gen 7.1.
 *
 * Disabled channels (per the active mask) are zeroed after filtering so
 * downstream stages still see a clean buffer of 21 channels.
 */
void run_biquad_prefilter(int window_len) {
    const dsp_config_t* cfg = dsp_get_config();
    const int32_t* hp_co    = dsp_hp_coeffs(cfg->hp);

    bool needs_rebind = !filters_initialized || bound_hp != cfg->hp;

    if (needs_rebind) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            init_filter(&hp_filters[ch], hp_co);
        }
        bound_hp = cfg->hp;
        filters_initialized = true;
    }

    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        for (int i = 0; i < window_len; i++) {
            raw_adc_buffer[ch][i] =
                biquad_process(&hp_filters[ch], raw_adc_buffer[ch][i]);
        }
    }

    /* Software channel gate: zero anything the host has masked off. */
    const uint32_t mask = cfg->channel_mask;
    if (mask != DSP_DEFAULT_CHANNEL_MASK) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            if (((mask >> ch) & 1u) == 0u) {
                memset(raw_adc_buffer[ch], 0, sizeof(int32_t) * (size_t)window_len);
            }
        }
    }
}
