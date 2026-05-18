#ifndef BIQUAD_Q31_H
#define BIQUAD_Q31_H

#include <stdint.h>

/* 21-channel ADC buffer (DMA target, SRAM0) */
extern int32_t raw_adc_buffer[21][2500];

void dsp_reset_pipeline(void);
void run_biquad_prefilter(int window_len);

#endif /* BIQUAD_Q31_H */
