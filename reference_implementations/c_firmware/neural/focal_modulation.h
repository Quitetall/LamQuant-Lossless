#ifndef FOCAL_MODULATION_H
#define FOCAL_MODULATION_H

#include <stdint.h>

void run_tnn_encoder_inference(const int32_t input[][313], int T);
int get_latent_temporal_length(void);
const int32_t (*get_latent_output(void))[79];

/* DMA weight prefetch ping-pong buffer accessor (infrastructure for Opt 2).
 * Compile with -DENABLE_DMA_PREFETCH to allocate 2 KB SRAM. */
#ifdef ENABLE_DMA_PREFETCH
uint8_t (*get_weight_prefetch_buf(int *active_idx))[1024];
#endif

#endif /* FOCAL_MODULATION_H */
