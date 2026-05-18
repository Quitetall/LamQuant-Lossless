#ifndef LIFTING_2D_H
#define LIFTING_2D_H

#include <stdint.h>

#define SUBBAND_L3_APPROX_LEN 313
#define SUBBAND_L3_DETAIL_LEN 312
#define SUBBAND_L2_DETAIL_LEN 625
#define SUBBAND_L1_DETAIL_LEN 1250

typedef struct {
    int32_t l3_approx[21][SUBBAND_L3_APPROX_LEN];
    int32_t l3_detail[21][SUBBAND_L3_DETAIL_LEN];
    int32_t l2_detail[21][SUBBAND_L2_DETAIL_LEN];
    int32_t l1_detail[21][SUBBAND_L1_DETAIL_LEN];
} lifting_subbands_t;

const lifting_subbands_t* lifting_3level(int32_t residual[][2500]);
const int32_t (*lifting_get_l3_approx(void))[SUBBAND_L3_APPROX_LEN];
int lifting_get_l3_approx_len(void);
void run_2d_lifting(int32_t tile[6][32]);
void run_static_lpc_fallback(int32_t tile[6][32]);

#endif /* LIFTING_2D_H */
