#ifndef SCHEDULER_V7_1_H
#define SCHEDULER_V7_1_H

#include <stdint.h>

typedef enum {
    OUTPUT_COMPRESSED_ONLY,
    OUTPUT_RAW_ONLY,
    OUTPUT_DUAL,
} output_mode_t;

void on_adc_dma_complete(void);
void lamquant_scheduler_v7_1_init(void);
void lamquant_scheduler_v7_1_run(void);
void scheduler_v7_1_set_output_mode(output_mode_t mode);
output_mode_t scheduler_v7_1_get_output_mode(void);
void scheduler_v7_1_abort(void);
void scheduler_v7_1_reinit(void);

#endif /* SCHEDULER_V7_1_H */
