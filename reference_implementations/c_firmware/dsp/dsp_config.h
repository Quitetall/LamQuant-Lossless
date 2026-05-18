/*
 * LamQuant — Runtime DSP Configuration
 * =====================================
 * Allows the GUI / host to switch between pre-computed Q30 highpass biquad
 * coefficients (off / 0.1 / 0.5 / 1.0 Hz), mask out unused channels, and
 * reset filter state — all without recompiling.
 *
 * In Gen 7.0 this also owned the lowpass and notch filter selection. Gen 7.1
 * removed those stages — LP shaping is now provided by the 3-level lifting
 * DWT (L3 approximation is bandlimited to 0-31.25 Hz) and 60 Hz mains
 * rejection is provided by SNN-driven detail coefficient thresholding. So
 * the only filter the user picks here is the highpass corner.
 *
 * No floats, no libm. Coefficients are baked into a small ROM table at
 * compile time and selected by index at runtime.
 */

#ifndef DSP_CONFIG_H
#define DSP_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

/* --- Filter option enum (must match GUI side) --- */

typedef enum {
    HP_OFF       = 0,
    HP_0_1_HZ    = 1,
    HP_0_5_HZ    = 2,  /* default */
    HP_1_0_HZ    = 3,
    HP_NUM_OPTIONS
} hp_filter_t;

typedef struct {
    hp_filter_t hp;
    uint32_t    channel_mask;  /* bit i = channel i enabled (bits 0..20) */
} dsp_config_t;

/* All 21 channels enabled by default. */
#define DSP_DEFAULT_CHANNEL_MASK ((1u << 21) - 1u)

/*
 * Set the active highpass selection. Indices outside the valid range are
 * silently clamped to the default. Forces a filter-state reset so previous
 * delay-line history is discarded.
 *
 * Safe to call from the serial command handler on Core 0 between windows.
 */
void dsp_set_filter_config(hp_filter_t hp);

/*
 * Set the channel enable mask. Disabled channels are zeroed in
 * raw_adc_buffer at the end of each window so downstream stages see a
 * stable signal (and so the SNN does not chase pickup on unused leads).
 *
 * Note: this does NOT change the ADC sampling — all channels are still
 * acquired. The mask is a software gate.
 */
void dsp_set_channel_mask(uint32_t mask);

/* Read-only accessors (used by preflight report + EDF metadata). */
const dsp_config_t* dsp_get_config(void);
const char* dsp_hp_label(hp_filter_t hp);

/*
 * Coefficient row accessor. Returns a pointer to 5 Q30 int32 values laid
 * out as { b0, b1, b2, a1, a2 }. Used by biquad_q31.c to bind the active
 * filter selection into per-channel biquad state.
 */
const int32_t* dsp_hp_coeffs(hp_filter_t hp);

#endif /* DSP_CONFIG_H */
