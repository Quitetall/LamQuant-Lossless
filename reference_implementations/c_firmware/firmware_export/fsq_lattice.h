#ifndef FSQ_LATTICE_H
#define FSQ_LATTICE_H

#include <stdint.h>

// FSQ lattice levels per dimension (4D product quantizer)
// Total codebook size: 8 * 6 * 5 * 5 = 1200
static const int32_t FSQ_LEVELS[4] = {8, 6, 5, 5};

// Default quantization scale (0.5 in Q31)
#define FSQ_QUANT_SCALE_Q31 1073741824

// Bounds size for CRC computation
#define FSQ_BOUNDS_SIZE 4

#endif // FSQ_LATTICE_H
