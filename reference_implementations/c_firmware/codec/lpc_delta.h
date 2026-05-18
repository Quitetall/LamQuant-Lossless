#ifndef LPC_DELTA_H
#define LPC_DELTA_H

#include <stdint.h>

#define LPC_DELTA_ORDER 8

uint32_t lpc_delta_encode(const int32_t curr[][LPC_DELTA_ORDER], uint8_t* out_buf);
uint32_t lpc_delta_decode(const uint8_t* in_buf, int32_t out_coeffs[][LPC_DELTA_ORDER]);
void lpc_delta_reset(void);

#endif /* LPC_DELTA_H */
