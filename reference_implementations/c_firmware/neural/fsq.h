#ifndef FSQ_H
#define FSQ_H

#include <stdint.h>

void fsq_update_adaptive_gain(const int32_t* eeg_buffer, int len);
uint32_t run_fsq_translation(int32_t* network_activations_4d);

#endif /* FSQ_H */
