/*
 * Host-compilation stubs for RP2350-specific headers.
 * Allows firmware C code to compile and run on x86/ARM for unit testing.
 * Only stubs out hardware dependencies — all math/logic is real.
 */
#ifndef HOST_STUBS_H
#define HOST_STUBS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <limits.h>

/* Stub out the firmware-specific headers that reference RP2350 hardware */

/* --- Toeplitz seeds stub --- */
#define TOEP_NUM_CHANNELS 21
static const uint8_t toep_seeds_stub[21][16] = {{0}};
static const uint8_t* toep_seeds_all[21] = {
    toep_seeds_stub[0],  toep_seeds_stub[1],  toep_seeds_stub[2],
    toep_seeds_stub[3],  toep_seeds_stub[4],  toep_seeds_stub[5],
    toep_seeds_stub[6],  toep_seeds_stub[7],  toep_seeds_stub[8],
    toep_seeds_stub[9],  toep_seeds_stub[10], toep_seeds_stub[11],
    toep_seeds_stub[12], toep_seeds_stub[13], toep_seeds_stub[14],
    toep_seeds_stub[15], toep_seeds_stub[16], toep_seeds_stub[17],
    toep_seeds_stub[18], toep_seeds_stub[19], toep_seeds_stub[20],
};

/* --- Focal net weights stub --- */
static const uint8_t focal_net_weights[16] = {0x49, 0x00, 0x55, 0xAA,
                                                0xFF, 0x01, 0x02, 0x03,
                                                0x04, 0x05, 0x06, 0x07,
                                                0x08, 0x09, 0x0A, 0x0B};
#define FOCAL_NET_WEIGHTS_SIZE 16

/* --- FSQ lattice stub --- */
static const int32_t FSQ_LEVELS[4] = {8, 6, 5, 5};
#define FSQ_BOUNDS_SIZE (4 * sizeof(int32_t))
#define FSQ_QUANT_SCALE_Q31 1073741824  /* 0.5 in Q31 */

/* --- Firmware CRC stub --- */
#define FIRMWARE_CRC32 0x00000000u  /* Will not match — tests use crc32_update directly */

/* Stub out section attributes (no special sections on host) */
#define __attribute__(x)

#endif /* HOST_STUBS_H */
