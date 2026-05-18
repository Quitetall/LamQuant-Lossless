#ifndef TOEP_SEEDS_H
#define TOEP_SEEDS_H

#include <stdint.h>

// LFSR seeds for per-channel Toeplitz compressed sensing rows
static const uint32_t TOEP_SEEDS[21] = {
    0xACE1u, 0xBE37u, 0xCAFEu, 0xDEADu, 0xF00Du, 0x1337u, 
    0xB00Bu, 0xFACEu, 0xD00Du, 0xBEEFu, 0xC0DEu, 0xBAD1u, 
    0xFEEDu, 0xDAD1u, 0xAB1Eu, 0xACDCu, 0xB1A5u, 0xCA5Eu, 
    0xDE1Fu, 0xEF01u, 0xF1A7u, 
};

#define TOEP_NUM_CHANNELS 21

#endif // TOEP_SEEDS_H
