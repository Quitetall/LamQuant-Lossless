#ifndef SCHEDULER_V7_7_H
#define SCHEDULER_V7_7_H

#include <stdint.h>

typedef enum {
    MODE_NEURAL,    /* Lossy: TNN + adaptive rANS (.lmq) */
    MODE_LOSSLESS,  /* Bit-exact: LPC + lifting + Golomb-Rice (.lml) */
} codec_mode_t;

typedef enum {
    OUTPUT_COMPRESSED_ONLY,
    OUTPUT_RAW_ONLY,
    OUTPUT_DUAL,
} output_mode_t;

/* DMA ISR — called by hardware DMA interrupt */
void __isr on_adc_dma_complete(void);

/* Lifecycle */
void lamquant_scheduler_v7_7_init(void);
void lamquant_scheduler_v7_7_run(void);
void scheduler_v7_7_abort(void);
void scheduler_v7_7_reinit(void);

/* Mode control */
void scheduler_v7_7_set_codec_mode(codec_mode_t mode);
codec_mode_t scheduler_v7_7_get_codec_mode(void);
void scheduler_v7_7_set_output_mode(output_mode_t mode);
output_mode_t scheduler_v7_7_get_output_mode(void);

#endif /* SCHEDULER_V7_7_H */
