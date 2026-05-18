#ifndef TERNARY_MAC_H
#define TERNARY_MAC_H

#include <stdint.h>

/**
 * @brief Branchless Ternary MAC — 4 weights per call, no LUT, no multiply.
 * Weight encoding: 00=0, 01=+1, 10=-1, 11=0(pad)
 * ~3 cycles/MAC on Hazard3 RISC-V.
 */
__attribute__((always_inline)) static inline int32_t ternary_mac_byte_fast(uint8_t packed_w, const int16_t* act) {
    int32_t acc = 0;
    #define _TMAC_ONE(i) do { \
        uint32_t w = (packed_w >> ((i) * 2)) & 3; \
        int32_t a = (int32_t)act[i]; \
        int32_t neg = -(int32_t)(w >> 1); \
        int32_t val = (a ^ neg) - neg; \
        uint32_t nonzero = (w & 1) ^ (w >> 1); \
        acc += val & (-(int32_t)nonzero); \
    } while(0)
    _TMAC_ONE(0); _TMAC_ONE(1); _TMAC_ONE(2); _TMAC_ONE(3);
    #undef _TMAC_ONE
    return acc;
}

/* Path 1+3: Branchless scalar MAC (SW-pipelined by caller, Zbb sat) */
int32_t ternary_conv1d_channel(const int16_t* act_in, int in_channels, int kernel_size,
                               const uint8_t* packed_weights, int32_t lsq_alpha_q31);

/* Path 2: Bit-serial CPOP MAC (for wide layers, n_ch >= 32) */
int32_t ternary_dot_bitserial(const uint32_t bitplanes[][16],
                              const uint32_t* sign_words,
                              const uint32_t* mask_words,
                              int n_words, int act_bits);
void activations_to_bitplanes(const int16_t* act, int n_ch,
                              uint32_t bitplanes[][16], int n_words);
void weights_to_signmask(const uint8_t* packed, int n_weights,
                         uint32_t* sign_out, uint32_t* mask_out, int n_words);

/* Boot verification */
int boot_ternary_parity_kat(void);

#endif /* TERNARY_MAC_H */
