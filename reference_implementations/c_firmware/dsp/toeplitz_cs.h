#ifndef TOEPLITZ_CS_H
#define TOEPLITZ_CS_H

#include <stdint.h>

void toep_fast_jump_lfsr(uint32_t jumps);
void sync_toeplitz_frame(uint32_t received_frame_counter);
void cs_project_channel(const int32_t* input, int32_t* output, int M, int N, uint32_t seed);
void apply_toeplitz_sensing_all(const int32_t input[][2500], int32_t output[][32], int num_channels, int M, int N);
void apply_toeplitz_sensing(const int32_t* input, int32_t* output, int M, int N);

#endif /* TOEPLITZ_CS_H */
