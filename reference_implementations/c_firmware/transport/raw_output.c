/*
 * LamQuant Gen 7 — Raw USB CDC Output
 * =====================================
 * Streams prefiltered 24-bit ADC samples over USB CDC (serial).
 * Used when a wired connection is available from the Pico to the base station.
 *
 * Packet format:
 *   Header (8 bytes):  'L','A','M','R' + channel_count + reserved + window_id
 *   Payload:           3 bytes per sample (24-bit packed, big-endian)
 *                      channels × 2500 samples per window
 *
 * Non-blocking: DMA queues data to USB endpoint (~2 ms).
 * Does not compete with the codec pipeline for CPU resources.
 */

#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "raw_output.h"

/* ADC buffer (populated by DMA, filtered by biquad) */
extern int32_t raw_adc_buffer[21][2500];

/* Sync header identifying raw mode packets */
static const uint8_t RAW_SYNC[4] = {'L', 'A', 'M', 'R'};

/* Byte-level raw write via putchar_raw (non-blocking USB CDC) */
static inline void raw_usb_write(const uint8_t *data, int len) {
    for (int i = 0; i < len; i++) {
        putchar_raw(data[i]);
    }
}

void trigger_raw_usb_tx(uint32_t window_id, int total_channels) {
    /* Header: sync + metadata */
    uint8_t header[8];
    header[0] = RAW_SYNC[0];
    header[1] = RAW_SYNC[1];
    header[2] = RAW_SYNC[2];
    header[3] = RAW_SYNC[3];               /* Raw mode identifier */
    header[4] = (uint8_t)total_channels;    /* Channel count */
    header[5] = 0;                          /* Reserved */
    header[6] = (uint8_t)(window_id >> 8);  /* Window ID high byte */
    header[7] = (uint8_t)(window_id & 0xFF);/* Window ID low byte */

    raw_usb_write(header, 8);

    /* Stream raw samples: 3 bytes per sample (24-bit packed, big-endian) */
    for (int ch = 0; ch < total_channels; ch++) {
        for (int s = 0; s < 2500; s++) {
            int32_t val = raw_adc_buffer[ch][s];
            uint8_t sample[3] = {
                (uint8_t)((val >> 16) & 0xFF),
                (uint8_t)((val >> 8)  & 0xFF),
                (uint8_t)( val        & 0xFF)
            };
            raw_usb_write(sample, 3);
        }
    }
}
