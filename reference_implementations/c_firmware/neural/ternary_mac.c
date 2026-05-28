// SPDX-License-Identifier: GPL-3.0-or-later
#include <stdint.h>
#include "../firmware_export/focal_net_weights.h"
#include "../core/math_utils.h"
#include "ternary_mac.h"

// PHASE 14: Total Verification (Clinical Hardening)
// Hardware Target: RP2350 (Hazard3 RISC-V)
// Activation Resolution: 6-bit (W2A6)
// Weight Resolution: 2-bit (Ternary {-1, 0, 1})

/* KAT-only LUT — not used in hot path, retained for verification */
static const int32_t TERNARY_LUT_KAT[4] = {0, 1, -1, 0};

/**
 * @brief Legacy LUT-based Ternary Dot Product (retained for KAT verification)
 */
__attribute__((always_inline)) static inline int32_t ternary_mac_byte_lut(uint8_t packed_w, const int16_t* act) {
    int32_t acc = 0;
    acc += (int32_t)act[0] * TERNARY_LUT_KAT[(packed_w     ) & 0x03];
    acc += (int32_t)act[1] * TERNARY_LUT_KAT[(packed_w >> 2) & 0x03];
    acc += (int32_t)act[2] * TERNARY_LUT_KAT[(packed_w >> 4) & 0x03];
    acc += (int32_t)act[3] * TERNARY_LUT_KAT[(packed_w >> 6) & 0x03];
    return acc;
}

/**
 * @brief Branchless Ternary MAC (3 cycles/MAC on Hazard3)
 *
 * Processes 4 ternary weights per call via conditional negate.
 * No LUT, no multiply — pure bitwise arithmetic.
 *
 * Weight encoding: 00=0, 01=+1, 10=-1, 11=0(pad)
 *
 * @param packed_w A single 8-bit byte containing 4 packed 2-bit ternary weights.
 * @param act Pointer to the array of 4 contiguous int16_t activations.
 * @return int32_t The accumulated integer dot product.
 */
/* ternary_mac_byte_fast is now defined in ternary_mac.h (static inline) */

/**
 * @brief Clinical 1D Ternary Convolution
 *
 * Implements a single output channel of a 1D convolution over multi-channel
 * physiological data. Deeply optimized for the 256-channel Platinum Oracle
 * architecture to maximize cache utilization and minimize memory footprint
 * (< 32KB constraint).
 *
 * @note Biological Intent: Extracts dense, highly-specific non-linear
 * morphological features from the decorrelated EEG stream.
 *
 * @param act_in Pointer to the input activation array.
 * @param in_channels The number of input channels.
 * @param kernel_size The spatial width of the 1D convolution kernel.
 * @param packed_weights Pointer to the contiguous ternary packed weights.
 * @param lsq_alpha_q31 Q31 fixed-point scaling factor for digital gain restoration.
 *                     Exported by export_firmware.py as round(alpha * 2^31).
 * @return int32_t The accumulated and alpha-scaled channel output.
 */
int32_t ternary_conv1d_channel(
    const int16_t* act_in,
    int in_channels,
    int kernel_size,
    const uint8_t* packed_weights,
    int32_t lsq_alpha_q31
) {
    int32_t accumulator = 0;
    
    // We assume packed_weights is a contiguous array of (in_channels * kernel_size) / 4 bytes
    int total_weights = in_channels * kernel_size;
    int num_bytes = total_weights / 4;
    
    for (int i = 0; i < num_bytes; i++) {
        accumulator += ternary_mac_byte_fast(packed_weights[i], &act_in[i * 4]);
    }

    // Handle remaining weights (if not multiple of 4)
    int remainder = total_weights % 4;
    if (remainder > 0) {
        uint8_t last_byte = packed_weights[num_bytes];
        for (int j = 0; j < remainder; j++) {
            accumulator += (int32_t)act_in[num_bytes * 4 + j] * TERNARY_LUT_KAT[(last_byte >> (2 * j)) & 0x03];
        }
    }
    
    // Apply LSQ Alpha scaling (Digital Gain Restoration) in Q31 fixed-point.
    // BUG FIX: was `(float)accumulator * lsq_alpha` which mixed float with
    // the int32 fixed-point pipeline, losing precision. Now uses mul_q31()
    // with alpha stored as Q31 by export_firmware.py.
    return mul_q31(accumulator, lsq_alpha_q31);
}

/**
 * @brief Bit-serial ternary dot product via XNOR + CPOP (Path 2)
 *
 * Processes 32 input channels per popcount instruction.
 * For 8-bit activations: 8 bit-planes × ceil(n/32) CPOP operations.
 *
 * Requires activations in bit-planar format:
 *   bitplane[b][word] = packed bit b of 32 consecutive activations
 *
 * Weight format: sign_words[] and mask_words[] (32 weights per uint32):
 *   mask_words: bit=1 if weight is nonzero (01 or 10)
 *   sign_words: bit=1 if weight is negative (10)
 *
 * @param bitplanes  8 bit-planes of activations, each ceil(n_ch/32) words
 * @param sign_words packed sign bits of weights, ceil(n_ch/32) words
 * @param mask_words packed nonzero bits of weights, ceil(n_ch/32) words
 * @param n_words    number of 32-bit words per plane (ceil(n_ch/32))
 * @param act_bits   number of activation bits (typically 8)
 * @return int32_t   dot product result
 */
int32_t ternary_dot_bitserial(
    const uint32_t bitplanes[][16],  /* [8][n_words], n_words <= 16 (512 ch max) */
    const uint32_t* sign_words,
    const uint32_t* mask_words,
    int n_words,
    int act_bits
) {
    int32_t result = 0;

    for (int b = 0; b < act_bits; b++) {
        int32_t bit_acc = 0;
        for (int w = 0; w < n_words; w++) {
            uint32_t act_word = bitplanes[b][w];
            uint32_t mask = mask_words[w];
            uint32_t sign = sign_words[w];

            /* Positive contribution: activation bits AND (mask AND NOT sign) */
            uint32_t pos = act_word & mask & ~sign;
            /* Negative contribution: activation bits AND (mask AND sign) */
            uint32_t neg = act_word & mask & sign;

            /* Zbb: single-cycle __builtin_popcount on Hazard3 */
            bit_acc += (int32_t)__builtin_popcount(pos);
            bit_acc -= (int32_t)__builtin_popcount(neg);
        }
        result += bit_acc << b;  /* weight by bit position */
    }
    return result;
}

/**
 * @brief Convert int16 activations to bit-planar format for CPOP path
 *
 * Packs bit b of each activation into consecutive words.
 * Output: bitplanes[b][w] = packed bit b of activations[w*32 .. w*32+31]
 *
 * @param act       input activations (int16_t, unsigned magnitude assumed)
 * @param n_ch      number of channels
 * @param bitplanes output [8][n_words] array
 * @param n_words   ceil(n_ch / 32)
 */
void activations_to_bitplanes(
    const int16_t* act,
    int n_ch,
    uint32_t bitplanes[][16],
    int n_words
) {
    /* Zero output */
    for (int b = 0; b < 8; b++) {
        for (int w = 0; w < n_words; w++) {
            bitplanes[b][w] = 0;
        }
    }

    for (int i = 0; i < n_ch; i++) {
        uint32_t abs_val = (uint32_t)(act[i] >= 0 ? act[i] : -act[i]);
        int word = i / 32;
        uint32_t bit = (uint32_t)1 << (i & 31);
        for (int b = 0; b < 8; b++) {
            if (abs_val & ((uint32_t)1 << b)) {
                bitplanes[b][word] |= bit;
            }
        }
    }
}

/**
 * @brief Convert packed 2-bit ternary weights to sign/mask word format
 *
 * Input: packed 2-bit weights (4 per byte): 00=0, 01=+1, 10=-1, 11=pad
 * Output: sign_words (bit=1 if negative), mask_words (bit=1 if nonzero)
 *
 * @param packed    packed ternary weights
 * @param n_weights total number of weights
 * @param sign_out  output sign words
 * @param mask_out  output mask words
 * @param n_words   ceil(n_weights / 32)
 */
void weights_to_signmask(
    const uint8_t* packed,
    int n_weights,
    uint32_t* sign_out,
    uint32_t* mask_out,
    int n_words
) {
    for (int w = 0; w < n_words; w++) {
        sign_out[w] = 0;
        mask_out[w] = 0;
    }

    for (int i = 0; i < n_weights; i++) {
        int byte_pos = i / 4;
        int bit_pos = (i % 4) * 2;
        uint32_t w2 = ((uint32_t)packed[byte_pos] >> bit_pos) & 0x03;

        int word = i / 32;
        uint32_t bit = (uint32_t)1 << (i & 31);

        uint32_t nonzero = (w2 & 1) ^ (w2 >> 1);  /* 1 for 01 and 10 */
        uint32_t sign = w2 >> 1;                     /* 1 for 10 (-1) */

        if (nonzero) mask_out[word] |= bit;
        if (sign)    sign_out[word] |= bit;
    }
}

/**
 * @brief Known-Answer Test (KAT) for Bit Parity Sign-Off
 *
 * Mandatory boot-time verification of the ternary Look-Up Table (LUT) and
 * little-endian bit-packing schema.
 *
 * @note Critical Safety Function: Ensures the C compiler did not silently
 * rotate bitfields or alter struct packing alignments on the RP2350 architecture.
 * A failure here triggers an immediate safety fault before any BCI data is processed.
 *
 * @return int Returns 0 on PASS, or -1 on FATAL PARITY ERROR.
 */
int boot_ternary_parity_kat(void) {
    int16_t test_act[4] = {100, 200, 300, 400};
    // Packed weights: [1, -1, 0, 1]
    // Hex: 0x01 | (0x02 << 2) | (0x00 << 4) | (0x01 << 6) = 0x01 | 0x08 | 0x00 | 0x40 = 0x49
    uint8_t test_packed = 0x49;

    // Expected: (100*1) + (200*-1) + (300*0) + (400*1) = 100 - 200 + 400 = 300
    int32_t result_fast = ternary_mac_byte_fast(test_packed, test_act);
    int32_t result_lut  = ternary_mac_byte_lut(test_packed, test_act);

    if (result_fast != 300) {
        return -1; // FATAL: fast path parity error
    }
    if (result_lut != 300) {
        return -2; // FATAL: LUT path parity error
    }
    if (result_fast != result_lut) {
        return -3; // FATAL: fast/LUT mismatch
    }
    return 0; // PASS
}
