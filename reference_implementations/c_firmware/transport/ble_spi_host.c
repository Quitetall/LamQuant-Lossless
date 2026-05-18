#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "ble_spi_host.h"

/*
 * LamQuant Gen 7 — BLE Transport + Adaptive Channel Coding
 * =========================================================
 * Manages BLE packet transmission via SPI1 DMA to nRF52840
 * and adaptive coding rate based on real-time PSRR monitoring.
 *
 * SPI1 pin assignments (nRF52840 SPI slave):
 *   GPIO10 = SCK, GPIO11 = MOSI, GPIO12 = MISO, GPIO13 = CS
 *
 * All arithmetic is integer. PSRR is in Q10 format (1 LSB = 0.1 dB).
 * 600 Q10 = 60.0 dB.
 */

#define BLE_SPI_PORT    spi1
#define BLE_SPI_BAUD    8000000   // 8 MHz
#define BLE_PIN_SCK     10
#define BLE_PIN_MOSI    11
#define BLE_PIN_MISO    12
#define BLE_PIN_CS      13

static int ble_dma_channel = -1;
volatile bool ble_initialized = false;

/* ChannelCoding is defined in ble_spi_host.h */

static ChannelCoding current_coding = CODING_RATE_1_1;

// PSRR in Q10 format. 680 = 68.0 dB (nominal for clean environment)
static int32_t adc_psrr_db_q10 = 680;

// --- FSQ level reduction (implemented here, declared extern in fsq.c) ---

// When EMC is detected, reduce FSQ codebook size to lower entropy
// and increase error resilience. This trades compression ratio for
// guaranteed decodability under packet loss.
//
// Normal:   L=3 per dim ({-1, 0, +1}), 3^4 = 81 codewords per 4-dim group
// Degraded: L=2 per dim ({0, +1}),     2^4 = 16 codewords per 4-dim group

static bool fsq_reduced = false;

void adaptive_reduce_fsq_levels(void) {
    fsq_reduced = true;
    // The FSQ encoder checks fsq_reduced to use L=2 instead of L=3
}

void adaptive_restore_fsq_levels(void) {
    fsq_reduced = false;
}

bool fsq_is_reduced(void) {
    return fsq_reduced;
}

// --- Coding rate control ---

void ble_set_coding_rate(ChannelCoding rate) {
    current_coding = rate;
}

ChannelCoding ble_get_coding_rate(void) {
    return current_coding;
}

/*
 * Called by main loop when ADC PSRR measurement is updated.
 *
 * PSRR thresholds (Q10):
 *   >= 600 (60.0 dB): Clean environment, full rate
 *   < 600 (60.0 dB):  EMC detected, add FEC redundancy + reduce FSQ
 *   < 400 (40.0 dB):  Severe EMC, maximum redundancy
 */
void on_emc_detected(int32_t current_psrr_measure_q10) {
    adc_psrr_db_q10 = current_psrr_measure_q10;

    if (adc_psrr_db_q10 < 400) {
        // Severe EMC: max redundancy
        ble_set_coding_rate(CODING_RATE_1_2);
        adaptive_reduce_fsq_levels();
    } else if (adc_psrr_db_q10 < 600) {
        // Mild EMC: moderate redundancy
        ble_set_coding_rate(CODING_RATE_2_3);
        adaptive_reduce_fsq_levels();
    } else {
        // Clean: full rate
        ble_set_coding_rate(CODING_RATE_1_1);
        adaptive_restore_fsq_levels();
    }
}

// --- BLE emergency flush (called by power_states.c safe mode) ---

void ble_spi_emergency_flush(void) {
    // Drop all pending packets, reset SPI FIFO
    // Implementation depends on BLE module (e.g., nRF52840 via SPI)
    // For now: clear the coding state
    current_coding = CODING_RATE_1_1;
    fsq_reduced = false;
}

void ble_enter_standby(void) {
    if (!ble_initialized) return;

    // De-assert CS (active-low: set HIGH to deselect)
    gpio_put(BLE_PIN_CS, 1);

    // Disable SPI peripheral to save power
    spi_deinit(BLE_SPI_PORT);
    ble_initialized = false;
}

// --- BLE SPI initialization (called from main.c during boot) ---

void ble_spi_init(void) {
    spi_init(BLE_SPI_PORT, BLE_SPI_BAUD);
    spi_set_format(BLE_SPI_PORT, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    gpio_set_function(BLE_PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(BLE_PIN_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(BLE_PIN_MISO, GPIO_FUNC_SPI);

    // CS is manual (active low)
    gpio_init(BLE_PIN_CS);
    gpio_set_dir(BLE_PIN_CS, GPIO_OUT);
    gpio_put(BLE_PIN_CS, 1);  // Deselected

    // Claim a DMA channel for BLE TX
    ble_dma_channel = dma_claim_unused_channel(true);
    ble_initialized = true;
}

// --- BLE DMA TX (called by scheduler stage 5) ---

extern const uint8_t* get_entropy_buffer(void);
extern uint32_t get_entropy_buffer_used_bytes(void);

void trigger_ble_dma_tx(void) {
    if (!ble_initialized || ble_dma_channel < 0) return;

    const uint8_t* payload = get_entropy_buffer();
    uint32_t len = get_entropy_buffer_used_bytes();

    if (len == 0) return;

    // Assert CS (active-low)
    gpio_put(BLE_PIN_CS, 0);

    // Configure DMA: memory -> SPI1 TX FIFO
    dma_channel_config c = dma_channel_get_default_config((uint)ble_dma_channel);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_8);
    channel_config_set_dreq(&c, spi_get_dreq(BLE_SPI_PORT, true));
    channel_config_set_read_increment(&c, true);
    channel_config_set_write_increment(&c, false);

    dma_channel_configure(
        (uint)ble_dma_channel,
        &c,
        &spi_get_hw(BLE_SPI_PORT)->dr,  // Write to SPI TX FIFO
        payload,                          // Read from entropy buffer
        len,                              // Transfer length
        true                              // Start immediately
    );

    // Non-blocking: DMA runs while scheduler re-arms ADC.
    // CS is de-asserted in the DMA completion ISR or next trigger_ble_dma_tx call.
}

/*
 * Emit an SNN event packet (type 0x04) over the BLE SPI link.
 *
 * Layout (12 bytes fixed):
 *   [0] sync (0xAA)
 *   [1] type (0x04)
 *   [2] level (0x02 for HIGH, reserved otherwise)
 *   [3..6] timestamp_ms little-endian uint32
 *   [7..11] zero pad
 *
 * Best-effort fire-and-forget. Skipped silently if BLE is not
 * initialised (radio not present, safe-mode), so the rest of the
 * pipeline keeps running on the bench. The host phone app subscribes
 * to packet type 0x04 over the BLE characteristic and surfaces a
 * notification without needing to also stream the raw EEG.
 */
void ble_spi_send_snn_alert(uint8_t level, uint32_t timestamp_ms) {
    if (!ble_initialized) return;

    static uint8_t pkt[12];
    pkt[0] = 0xAA;
    pkt[1] = 0x04;
    pkt[2] = level;
    pkt[3] = (uint8_t)(timestamp_ms        & 0xff);
    pkt[4] = (uint8_t)((timestamp_ms >>  8) & 0xff);
    pkt[5] = (uint8_t)((timestamp_ms >> 16) & 0xff);
    pkt[6] = (uint8_t)((timestamp_ms >> 24) & 0xff);
    pkt[7] = 0; pkt[8] = 0; pkt[9] = 0; pkt[10] = 0; pkt[11] = 0;

    // Synchronous 12-byte write — the alert is short and non-blocking
    // would require sharing the DMA channel with trigger_ble_dma_tx().
    // 12 bytes at 8 MHz ~= 12 us, well below the scheduler's headroom.
    gpio_put(BLE_PIN_CS, 0);
    spi_write_blocking(BLE_SPI_PORT, pkt, sizeof pkt);
    gpio_put(BLE_PIN_CS, 1);
}
