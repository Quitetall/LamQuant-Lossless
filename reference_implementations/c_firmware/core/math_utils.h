#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stdint.h>

/* W2A8 activation support — compile with -DUSE_INT8_ACTIVATIONS to enable */
#ifdef USE_INT8_ACTIVATIONS
typedef int8_t act_t;
#define ACT_MAX 127
#define ACT_MIN (-128)
#else
typedef int16_t act_t;
#define ACT_MAX 32767
#define ACT_MIN (-32768)
#endif

/**
 * @brief Standard Q31 Multiplication
 * 64-bit intermediate precision ensures no overflow before normalization.
 */
static inline int32_t mul_q31(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 31);
}

/**
 * @brief Q30 Multiplication — range [-2.0, +2.0)
 * Used for filter coefficients that exceed Q31 range (|coeff| > 1.0).
 * 1 bit less fractional precision than Q31, but covers coefficients
 * like the HP biquad's b1 ≈ -1.98 which overflows Q31.
 */
static inline int32_t mul_q30(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 30);
}

/**
 * @brief Saturating Addition for Q31
 */
static inline int32_t add_sat_q31(int32_t a, int32_t b) {
    int32_t res;
    if (__builtin_add_overflow(a, b, &res)) {
        res = (a < 0) ? INT32_MIN : INT32_MAX;
    }
    return res;
}

/**
 * @brief Saturating Subtraction for Q31
 */
static inline int32_t sub_sat_q31(int32_t a, int32_t b) {
    int32_t res;
    if (__builtin_sub_overflow(a, b, &res)) {
        res = (a < 0) ? INT32_MIN : INT32_MAX;
    }
    return res;
}

/**
 * @brief Saturating clamp to int16 range using Zbb MIN/MAX (Path 3)
 *
 * On Hazard3 with Zbb, `min` and `max` are single-cycle instructions.
 * This replaces the branch-based `if (v > 32767) v = 32767; if (v < -32768) v = -32768;`
 * pattern with 2 deterministic cycles, no branch misprediction.
 */
static inline int16_t sat_i16(int32_t v) {
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
    /* GCC with -Os and Zbb emits: max a0,a0,-32768; min a0,a0,32767 */
}

/**
 * @brief Saturating clamp to activation range (int16 or int8 depending on mode)
 *
 * When USE_INT8_ACTIVATIONS is defined, clamps to [-128, 127].
 * Otherwise identical to sat_i16 (clamps to [-32768, 32767]).
 */
static inline act_t sat_act(int32_t v) {
    if (v > ACT_MAX) v = ACT_MAX;
    if (v < ACT_MIN) v = ACT_MIN;
    return (act_t)v;
}

#endif // MATH_UTILS_H
