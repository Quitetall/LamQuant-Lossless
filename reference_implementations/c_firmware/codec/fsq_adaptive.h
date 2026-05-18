#ifndef FSQ_ADAPTIVE_H
#define FSQ_ADAPTIVE_H

#include <stdint.h>

void fsq_adaptive_init(void);
uint32_t fsq_adaptive_encode(const int32_t latent[][79], int T_latent);
const uint32_t* fsq_get_symbols(void);
uint32_t fsq_get_symbol_count(void);
const uint8_t* fsq_get_level_bitmap(void);
uint32_t fsq_get_num_levels_at(int t);
uint16_t fsq_build_level_summary(void);

#endif /* FSQ_ADAPTIVE_H */
