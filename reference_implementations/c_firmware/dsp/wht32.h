#ifndef WHT32_H
#define WHT32_H

#include <stdint.h>

void wht32_forward(int32_t x[32]);
void wht32_inverse(int32_t x[32]);
void wht32_apply_latent(int32_t latent[][79], int T_latent);
void wht32_inverse_latent(int32_t latent[][79], int T_latent);

#endif /* WHT32_H */
