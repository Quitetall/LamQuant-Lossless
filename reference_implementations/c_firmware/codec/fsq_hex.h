#ifndef FSQ_HEX_H
#define FSQ_HEX_H

#include <stdint.h>

void fsq_hex_init(int32_t vmin_q31, int32_t vmax_q31);
uint32_t fsq_hex_encode(const int32_t latent[][312], int T_latent, const uint8_t *level_bitmap, int32_t vmin_q31);
const uint32_t* fsq_hex_get_symbols(void);
uint32_t fsq_hex_get_symbol_count(void);
uint32_t fsq_hex_get_total_codewords(int config_idx);

#endif /* FSQ_HEX_H */
