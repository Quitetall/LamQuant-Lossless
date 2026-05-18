#ifndef HYBRID_ENTROPY_H
#define HYBRID_ENTROPY_H

#include <stdint.h>
#include <stdbool.h>

void encode_detail_subbands(const int32_t l3_detail[][312],
                            const int32_t l2_detail[][625],
                            const int32_t l1_detail[][1250],
                            uint8_t subband_mask);
void run_hybrid_entropy_encoder(bool degraded);
const uint8_t* get_entropy_buffer(void);
uint32_t get_entropy_buffer_used_bytes(void);

#endif /* HYBRID_ENTROPY_H */
