#ifndef LPC_PREDICTOR_H
#define LPC_PREDICTOR_H

#include <stdint.h>

#define LPC_ORDER_MAX 8

void lpc_analyze(const int32_t input[][2500], int32_t coeffs[][LPC_ORDER_MAX], int32_t residual[][2500]);
const int32_t (*lpc_get_coefficients(void))[LPC_ORDER_MAX];
void lpc_predict_only(int32_t tile[6][32]);

#endif /* LPC_PREDICTOR_H */
