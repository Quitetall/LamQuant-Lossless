/*
 * LamQuant — Runtime DSP Configuration
 * =====================================
 * Owns the active dsp_config_t and a small ROM table of pre-computed Q30
 * highpass biquad coefficients. The host (GUI or serial 'F' command)
 * selects an option by index; biquad_q31.c reads dsp_get_config() at the
 * start of each window and picks coefficients out of this table.
 *
 * In Gen 7.1 the lowpass and notch filter selections are gone — the
 * lifting DWT bandlimits the L3 approximation and SNN-driven detail
 * thresholding handles 60 Hz mains.
 *
 * No floats, no libm. The numbers below were generated offline by Python
 * (Butterworth order 2 via the bilinear transform) and pasted as int32 Q30.
 *
 * To regenerate:
 *   import math
 *   Q30 = 1 << 30; fs = 250.0; SQRT2 = math.sqrt(2.0)
 *   def hp(fc):
 *       K = math.tan(math.pi * fc / fs); n = 1 + SQRT2*K + K*K
 *       return (1/n, -2/n, 1/n, 2*(K*K - 1)/n, (1 - SQRT2*K + K*K)/n)
 */

#include <stdint.h>
#include <stdbool.h>
#include "dsp_config.h"

extern void dsp_reset_pipeline(void);

/* Q30 = 2^30. Coefficients live in [-2.0, +2.0). Each row: { b0, b1, b2, a1, a2 }. */
static const int32_t HP_COEFFS[HP_NUM_OPTIONS][5] = {
    /* HP_OFF — passthrough (b0 = 1.0, all else zero) */
    {  1073741824,           0,           0,           0,           0 },
    /* HP_0_1_HZ Butterworth o2 fs=250 */
    {  1071835315, -2143670630,  1071835315, -2143667245,  1069932191 },
    /* HP_0_5_HZ Butterworth o2 fs=250 (default) — matches the legacy hardcoded
     * Gen 7.1 values in biquad_q31.c bit-for-bit. */
    {  1064243069, -2128486138,  1064243069, -2128402106,  1054828345 },
    /* HP_1_0_HZ Butterworth o2 fs=250 */
    {  1054828333, -2109656665,  1054828333, -2109323487,  1036248020 },
};

static const char* HP_LABELS[HP_NUM_OPTIONS] = {
    "off", "0.1 Hz", "0.5 Hz", "1.0 Hz",
};

/* The single active configuration, mutated only via the setters below. */
static dsp_config_t g_active = {
    .hp           = HP_0_5_HZ,
    .channel_mask = DSP_DEFAULT_CHANNEL_MASK,
};

void dsp_set_filter_config(hp_filter_t hp) {
    if ((unsigned)hp >= (unsigned)HP_NUM_OPTIONS) hp = HP_0_5_HZ;
    g_active.hp = hp;
    /* Discard delay-line history so the new response settles cleanly. */
    dsp_reset_pipeline();
}

void dsp_set_channel_mask(uint32_t mask) {
    g_active.channel_mask = mask & DSP_DEFAULT_CHANNEL_MASK;
}

const dsp_config_t* dsp_get_config(void) {
    return &g_active;
}

const char* dsp_hp_label(hp_filter_t hp) {
    if ((unsigned)hp >= (unsigned)HP_NUM_OPTIONS) return "?";
    return HP_LABELS[hp];
}

const int32_t* dsp_hp_coeffs(hp_filter_t hp) {
    if ((unsigned)hp >= (unsigned)HP_NUM_OPTIONS) hp = HP_0_5_HZ;
    return HP_COEFFS[hp];
}
