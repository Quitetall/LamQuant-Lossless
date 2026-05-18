#include <stdint.h>
#include <stddef.h>
#include "../firmware_export/fsq_lattice.h"
#include "../firmware_export/focal_net_weights.h"
#include "../core/math_utils.h"
#include "fsq.h"

// Adaptive FSQ scaling: adjusts quantization grid based on rolling signal RMS.
// Ensures weak signals fill the FSQ bins (preventing dead codes) and strong
// signals don't clip.
static volatile int32_t adaptive_fsq_scale = FSQ_QUANT_SCALE_Q31;

/*
 * Integer square root (no float, no libm).
 * Uses binary search: O(16) iterations for 32-bit input.
 * Returns floor(sqrt(x)).
 */
static uint32_t isqrt32(uint32_t x) {
    if (x == 0) return 0;
    uint32_t result = 0;
    uint32_t bit = 1u << 30;  // Start from highest power of 4

    while (bit > x) bit >>= 2;
    while (bit != 0) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return result;
}

/*
 * Update adaptive FSQ gain from a rolling RMS estimate.
 *
 * Called once per window with the current ADC buffer (Q31 samples).
 * Computes integer RMS and adjusts the FSQ projection scale:
 *   - High RMS (>50 in signal units): compress scale to avoid clipping
 *   - Low/nominal RMS: use default FSQ_QUANT_SCALE_Q31
 *
 * The "50" threshold corresponds to the silicon shackle clamp range (±50).
 * Signals near the clamp boundary need reduced scale to avoid saturation
 * in the FSQ lattice.
 */
void fsq_update_adaptive_gain(const int32_t* eeg_buffer, int len) {
    if (len <= 0) return;

    int64_t sum_sq = 0;
    for (int i = 0; i < len; i++) {
        int32_t val = eeg_buffer[i];
        sum_sq += (int64_t)val * val;
    }

    // Integer RMS: isqrt(sum_sq / len)
    uint32_t mean_sq = (uint32_t)(sum_sq / (int64_t)len);
    uint32_t rms = isqrt32(mean_sq);

    // Adaptive scaling: compress when signal is near the ±50 shackle boundary.
    // The divisor (rms/50 + 1) is always >= 1, preventing division by zero
    // and ensuring scale never exceeds the nominal value.
    if (rms > 50) {
        adaptive_fsq_scale = FSQ_QUANT_SCALE_Q31 / (int32_t)((rms / 50) + 1);
    } else {
        adaptive_fsq_scale = FSQ_QUANT_SCALE_Q31;
    }
}

/*
 * Quantize a single Q31 activation to the nearest FSQ grid point.
 *
 * Input range: Q31 scaled neural activations (output of ternary MAC).
 * The adaptive_fsq_scale (Q31 format) projects the activation onto the
 * discrete FSQ grid [-levels/2, +levels/2].
 *
 * mul_q31(val, scale) performs: (int32_t)((int64_t)val * scale >> 31)
 * This maps the continuous Q31 value to an integer grid index.
 */
static inline int32_t fsq_quantize_scalar(int32_t val, int32_t levels) {
    int32_t half_bound = levels >> 1;

    // Project activation onto FSQ grid using adaptive Q31 scale
    int32_t grid_estimate = mul_q31(val, adaptive_fsq_scale);

    // Clamp to valid grid range
    if (grid_estimate > half_bound)  grid_estimate = half_bound;
    if (grid_estimate < -half_bound) grid_estimate = -half_bound;

    return grid_estimate;
}

/*
 * Run FSQ on a 4-element activation vector.
 *
 * Converts 4 quantized grid values into a single flat index using
 * mixed-radix encoding: index = sum(q_val[i] * stride[i])
 * where stride[i] = product(FSQ_LEVELS[0..i-1]).
 *
 * The flat index uniquely identifies one point in the 4D FSQ lattice
 * and is the input to the entropy coder.
 */
uint32_t run_fsq_translation(int32_t* network_activations_4d) {
    uint32_t single_index = 0;
    uint32_t implicit_stride = 1;

    for (int i = 0; i < 4; i++) {
        int32_t q_val = fsq_quantize_scalar(network_activations_4d[i], FSQ_LEVELS[i]);
        // Shift from [-half, +half] to [0, levels-1] for unsigned index
        int32_t base_index = q_val + (FSQ_LEVELS[i] >> 1);
        single_index += (uint32_t)base_index * implicit_stride;
        implicit_stride *= (uint32_t)FSQ_LEVELS[i];
    }
    return single_index;
}
