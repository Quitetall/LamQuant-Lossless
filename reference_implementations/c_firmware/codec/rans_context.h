#ifndef RANS_CONTEXT_H
#define RANS_CONTEXT_H

#include <stdint.h>

void rans_context_init(void);
uint32_t rans_context_encode(int T_latent);
const uint8_t* rans_context_get_buffer(void);
uint32_t rans_context_get_used_bytes(void);

#endif /* RANS_CONTEXT_H */
