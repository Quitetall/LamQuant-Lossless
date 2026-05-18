#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/watchdog.h"
#include "hardware/dma.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#if defined(ADC_BACKEND_LC_ADC)
#include "lc_adc_trigger.pio.h"
#endif
#include "power_states.h"
#include "integrity.h"
#include "stack_guard.h"
#include "scheduler_v7_7.h"

/* Forward declaration — defined in scheduler_v7_7.c */
extern void __isr on_adc_dma_complete(void);
#include "preflight_report.h"
#include "../firmware_export/firmware_crc.h"
#include "../dsp/dsp_config.h"
#include "../dsp/biquad_q31.h"
#include "../snn/snn.h"
#include "../neural/ternary_mac.h"
#include "../neural/fsq.h"
#include "../codec/fsq_adaptive.h"
#include "../codec/rans_context.h"
#include "../codec/detail_threshold.h"
#include "../codec/lpc_delta.h"
#include "../transport/ble_spi_host.h"

/*
 * LamQuant Gen 7.1 "Subband" — System Entry Point
 * =================================================
 * Core 0: Boot, watchdog, PSRR monitoring, scheduler dispatch
 * Core 1: SNN-driven golden path (TNN on L3 approx + FSQ + rANS)
 *
 * Hardware: RP2350 (Hazard3 RISC-V, RV32IMAC + Zbb)
 *
 * Gen 7.1 pipeline:
 *   HP biquad → LPC analysis → 3-level lifting DWT
 *   → SNN on L3 approx → TNN on L3 approx → WHT → FSQ → rANS
 *   + detail thresholding → Golomb-Rice
 *   + LPC delta encoding
 *
 * Single-byte commands:
 *   'R' — Raw USB only (wired, full-quality 24-bit)
 *   'C' — Compressed BLE only (wireless, default)
 *   'D' — Dual (both simultaneously)
 *   'V' — Version query
 *   'P' — Preflight report
 *   '0' — Quality: alerting   (L3 only,    L=8 FSQ)
 *   '1' — Quality: monitoring (L3+L2,      L=16 FSQ)
 *   '2' — Quality: clinical   (all details, L=32 FSQ)
 *   'A' — Quality: auto (SNN-driven)
 *
 * Payload-bearing commands (host sends opcode then N bytes):
 *   'F' + 1 byte                          — HP filter selection (0..3)
 *   'M' + 4 bytes little-endian uint32    — per-channel enable mask
 *   'S' + 1 byte                          — SNN sensitivity (0=low, 1=med, 2=high)
 */

#if defined(ADC_BACKEND_ADS1299)
extern void ads1299_init(void);                  // afe/ads1299_driver.c
extern void ads1299_start_continuous(void);      // afe/ads1299_driver.c
#endif

// ADC buffer (populated by DMA, read by scheduler)
// Pinned to SRAM0 domain to isolate from Core 1 / SRAM4 traffic
// Declared in biquad_q31.h

// Forward declaration — defined below main().
static void handle_serial_command(uint8_t cmd);

// DMA channel for LC-ADC → SRAM0 transfer
static int dma_chan = -1;

/*
 * Configure DMA to transfer ADC samples from PIO RX FIFO into raw_adc_buffer.
 * Triggered by PIO state machine when LC comparator fires.
 */
static void dma_adc_init(PIO pio, uint sm) {
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config((uint)dma_chan);

    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);
    channel_config_set_read_increment(&cfg, false);   // PIO FIFO is single address
    channel_config_set_write_increment(&cfg, true);    // Write sequentially into buffer
    channel_config_set_dreq(&cfg, pio_get_dreq(pio, sm, false));  // Pace by PIO RX FIFO

    dma_channel_configure(
        (uint)dma_chan,
        &cfg,
        raw_adc_buffer,                    // Write destination
        &pio->rxf[sm],                     // Read source (PIO RX FIFO)
        21 * 2500,                         // Transfer count (full window)
        false                              // Don't start yet
    );

    // Enable DMA completion interrupt → wakes scheduler
    dma_channel_set_irq0_enabled((uint)dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, on_adc_dma_complete);
    irq_set_enabled(DMA_IRQ_0, true);
}

/*
 * Re-arm DMA for the next acquisition window.
 * Called by scheduler after processing the current window.
 */
/* Prototype visible to scheduler_v7_7.c via extern */
void dma_adc_rearm(void);

void dma_adc_rearm(void) {
    if (dma_chan >= 0) {
        dma_channel_set_write_addr((uint)dma_chan, raw_adc_buffer, true);
    }
}

/*
 * Warm boot detection: if watchdog caused the reset, skip full init
 * and preserve SRAM4 TNN weights (they survive soft reset).
 */
static bool was_watchdog_reset(void) {
    return watchdog_caused_reboot();
}

/*
 * Boot-time self-tests. Any failure → safe mode (infinite WFI).
 */
static void boot_self_test(void) {
    // 1. Ternary MAC Known Answer Test
    bool kat_ok = (boot_ternary_parity_kat() == 0);
    if (!kat_ok) {
        preflight_set_boot_results(false, false, 0, 0);
        emit_preflight_report();
        enter_safe_mode();  // Never returns
    }

    // 2. Firmware CRC integrity
    bool crc_ok = verify_firmware_crc();
    // Record results for preflight report (CRC values set by verify_firmware_crc)
    preflight_set_boot_results(kat_ok, crc_ok, FIRMWARE_EXPECTED_CRC, 0);
    if (!crc_ok) {
        emit_preflight_report();
        enter_safe_mode();  // Never returns
    }
}

int main(void) {
    // --- Phase 1: Hardware init ---
    stdio_init_all();
    stack_setup_hardware_trap();  // PMP canary + NAPOT bounds

    // Watchdog: 500ms timeout, pet in main loop
    watchdog_enable(500, true);

    // --- Phase 2: Warm boot shortcut ---
    if (was_watchdog_reset()) {
        // SRAM4 weights survived — skip KAT + CRC, go straight to scheduler
        // (TNN weights are in .sram4_tnn section which persists across soft reset)
        preflight_set_boot_results(true, true, FIRMWARE_EXPECTED_CRC, FIRMWARE_EXPECTED_CRC);
    } else {
        // Cold boot: full self-test
        boot_self_test();
        // Emit preflight report on successful cold boot
        emit_preflight_report();
    }

    // --- Phase 3: ADC backend + BLE SPI init ---
    ble_spi_init();

#if defined(ADC_BACKEND_ADS1299)
    // Production path: ADS1299 8-channel 24-bit AFE via SPI0
    ads1299_init();
    ads1299_start_continuous();
#else
    // Development path: LC-ADC comparator via PIO + DMA
    PIO pio = pio0;
    uint sm = (uint)pio_claim_unused_sm(pio, true);
    uint offset = (uint)pio_add_program(pio, &lc_adc_trigger_program);
    uint lc_adc_pin = 2;  // GPIO2: LC comparator output
    lc_adc_trigger_program_init(pio, sm, offset, lc_adc_pin);
    dma_adc_init(pio, sm);

    // Arm first DMA transfer
    dma_adc_rearm();
#endif

    // --- Phase 3.5: Initialize Gen 7.1 subband pipeline ---
    fsq_adaptive_init();      // Compute FSQ inv_range for L=2,3,5,32
    rans_context_init();      // Compute rANS reciprocals for all frequency tables
    lpc_delta_reset();        // Reset LPC delta state
    lamquant_scheduler_v7_7_init();  // Launch dual-core scheduler

    // --- Phase 4: Main loop ---
    // The scheduler is event-driven: it sleeps (WFI) until DMA fires,
    // then processes one window through the pipeline. Output mode and
    // quality mode are selectable via USB serial commands.
    static uint8_t prev_activity = 0;
    while (1) {
        lamquant_scheduler_v7_7_run();
        watchdog_update();

        // After each scheduler tick, sample the SNN activity and emit
        // an alert packet over BLE on the rising edge into HIGH. The
        // BLE side is best-effort: the function returns immediately if
        // the radio is not present.
        uint8_t act = snn_activity_sum();
        uint8_t cur = (act >= 32) ? 2 : (act > 0 ? 1 : 0);
        if (cur >= 2 && prev_activity < 2) {
            ble_spi_send_snn_alert(cur, (uint32_t)to_ms_since_boot(get_absolute_time()));
        }
        prev_activity = cur;

        // Poll USB serial for commands
        int c = getchar_timeout_us(0);
        if (c != PICO_ERROR_TIMEOUT) {
            handle_serial_command((uint8_t)c);
        }
    }

    return 0;  // Unreachable
}

/*
 * USB serial command handler.
 * Single-byte commands for mode switching, quality, and diagnostics; plus
 * a small set of payload-bearing commands ('F', 'M', 'S') for runtime DSP /
 * SNN configuration.
 */
static const char VERSION_STRING[] = "LamQuant Gen7.1 v7.1.0\n";

static void send_version_string(void) {
    for (int i = 0; VERSION_STRING[i] != '\0'; i++) {
        putchar_raw(VERSION_STRING[i]);
    }
}

static bool quality_auto_mode = true;

/*
 * Read exactly `n` bytes from USB serial with a per-byte deadline of 5 ms.
 * Returns true on success, false on timeout (in which case the partial
 * payload is discarded by the caller).
 */
static bool read_payload(uint8_t* buf, int n) {
    for (int i = 0; i < n; i++) {
        int c = getchar_timeout_us(5000);
        if (c == PICO_ERROR_TIMEOUT) return false;
        buf[i] = (uint8_t)c;
    }
    return true;
}

static void handle_serial_command(uint8_t cmd) {
    switch (cmd) {
        /* Output mode */
        case 'R': scheduler_v7_7_set_output_mode(OUTPUT_RAW_ONLY);        break;
        case 'C': scheduler_v7_7_set_output_mode(OUTPUT_COMPRESSED_ONLY); break;
        case 'D': scheduler_v7_7_set_output_mode(OUTPUT_DUAL);            break;

        /* Quality mode (Gen 7.1) */
        case '0': detail_threshold_set_mode(QUALITY_ALERTING);   quality_auto_mode = false; break;
        case '1': detail_threshold_set_mode(QUALITY_MONITORING); quality_auto_mode = false; break;
        case '2': detail_threshold_set_mode(QUALITY_CLINICAL);   quality_auto_mode = false; break;
        case 'A': quality_auto_mode = true;                                                 break;

        /* Diagnostics */
        case 'V': send_version_string();    break;
        case 'P': emit_preflight_report();  break;

        /* Runtime DSP / SNN config */
        case 'F': {
            uint8_t p[1];
            if (read_payload(p, 1)) {
                dsp_set_filter_config((hp_filter_t)p[0]);
            }
            break;
        }

        case 'M': {
            uint8_t p[4];
            if (read_payload(p, 4)) {
                uint32_t mask = (uint32_t)p[0]
                              | ((uint32_t)p[1] <<  8)
                              | ((uint32_t)p[2] << 16)
                              | ((uint32_t)p[3] << 24);
                dsp_set_channel_mask(mask);
            }
            break;
        }

        case 'S': {
            uint8_t p[1];
            if (read_payload(p, 1)) {
                snn_set_sensitivity((snn_sensitivity_t)p[0]);
            }
            break;
        }
    }
}
