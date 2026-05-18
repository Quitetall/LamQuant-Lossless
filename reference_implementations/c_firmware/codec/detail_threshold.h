#ifndef DETAIL_THRESHOLD_H
#define DETAIL_THRESHOLD_H

#include <stdint.h>

typedef enum {
    QUALITY_ALERTING   = 0,
    QUALITY_MONITORING = 1,
    QUALITY_CLINICAL   = 2,
} quality_mode_t;

int detail_threshold_apply(int32_t l3_detail[][312], int32_t l2_detail[][625], int32_t l1_detail[][1250], quality_mode_t mode);
quality_mode_t detail_threshold_auto_mode(void);
void detail_threshold_set_mode(quality_mode_t mode);
quality_mode_t detail_threshold_get_mode(void);
uint8_t detail_threshold_subband_mask(quality_mode_t mode);

#endif /* DETAIL_THRESHOLD_H */
