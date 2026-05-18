/*
 * LamQuant Gen 7.1 "Subband" — Dual-Core Pipeline Scheduler
 * ===========================================================
 * Replaces Gen 7.0 scheduler with LPC + lifting subband preprocessing.
 *
 * Core 0 (always runs):
 *   ADC DMA → HP biquad → LPC analysis → 3-level lifting DWT
 *   → SNN inference on L3 approximation → Lightning Path
 *   → dispatch Core 1 if SNN detects activity
 *
 * Core 1 (on-demand):
 *   WFE → TNN encoder on L3 approximation [21][313]
 *   → adaptive FSQ → context-adaptive rANS
 *   → write result to mailbox → WFE
 *
 * INVARIANT: Lightning Path ALWAYS runs. Core 1 failure = Lightning transmitted.
 *
 * Key change from v7.0:
 *   - Prefilter: HP-only biquad (LP/notch removed — lifting handles both)
 *   - LPC analysis: order-8, per-channel, 256-sample autocorrelation
 *   - 3-level lifting DWT on LPC residual: produces L3 approx [21][313]
 *   - TNN input: L3 approximation [21][313] (was [21][2500])
 *   - SNN input: L3 approximation [21][313] at stride 1 (was [21][2500] at stride 8)
 *   - Detail subbands encoded separately (Phase 4, future)
 *
 * Memory map:
 *   SRAM0-1: ADC buffer + LPC residual workspace
 *   SRAM2-3: TNN activation buffers (much smaller: 112×313 vs 96×2500)
 *   SRAM4:   TNN weights (~42 KB packed)
 *   SRAM5:   SNN weights (~8 KB) + lifting subbands + LPC coefficients
 *   SRAM6:   SNN state + activity_map
 *   SRAM7:   Golden output buffer + sigmoid LUT
 *   SRAM8:   Mailbox (32B) + Core 0 stack
 *   SRAM9:   Core 1 stack
 *
 * Peak SRAM: ~450 KB (58%) — down from 71.5% in Gen 7.0.
 */

#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/timer.h"
#include "hardware/irq.h"
#include "hardware/dma.h"
#include "mailbox.h"
#include "power_states.h"
#include "safety_features.h"
#include "scheduler_v7_1.h"
#include "../snn/snn.h"

#define DEADLINE_US              4000
#define GOLDEN_BUDGET_US         3500
#define CORE1_TIMEOUT_US         15000

#define LPC_ORDER 8
#define L3_APPROX_LEN 313

/* output_mode_t is defined in scheduler_v7_1.h */

static volatile output_mode_t output_mode = OUTPUT_COMPRESSED_ONLY;

/* Lightning path workspace (legacy) */
int32_t lifting_tile_2d[6][32] __attribute__((aligned(4)));

/* LPC residual buffer — aliases raw_adc_buffer (safe: raw USB TX completes
 * before LPC runs, and the pipeline is strictly sequential). */
extern int32_t raw_adc_buffer[21][2500];
#define lpc_residual raw_adc_buffer
static int32_t lpc_coeffs[21][LPC_ORDER] __attribute__((aligned(4)));

/* Mailbox */
mailbox_t shared_mailbox;

/* Pipeline state */
typedef enum {
    STATE_SLEEP,
    STATE_PREFILTER,
    STATE_LPC_ANALYSIS,
    STATE_LIFTING_DWT,
    STATE_SNN_INFERENCE,
    STATE_LIGHTNING_DSP,
    STATE_WAITING_CORE1,
    STATE_TX_READY,
} PipelineStage;

volatile PipelineStage current_state = STATE_SLEEP;
volatile uint32_t window_start_time_us = 0;
volatile bool adc_buffer_ready = false;
static uint32_t frame_sequence = 0;

/* lifting_subbands_t is defined in lifting_2d.h */
#include "../dsp/lifting_2d.h"

/* Module headers replace bare extern declarations */
#include "../dsp/biquad_q31.h"
#include "../dsp/lpc_predictor.h"
#include "../dsp/toeplitz_cs.h"
#include "../neural/focal_modulation.h"
#include "../transport/ble_spi_host.h"

/* Module headers for remaining externs */
#include "../transport/hybrid_entropy.h"
#include "../transport/raw_output.h"

extern void dma_adc_rearm(void);

/* raw_adc_buffer declared via biquad_q31.h */

/* Golden output buffer in SRAM7 */
static uint8_t golden_output_buf[240] __attribute__((section(".scratch_x.golden_buf")));

/* ================================================================
 * DMA ISR
 * ================================================================ */

void __isr on_adc_dma_complete(void) {
    dma_hw->ints0 = 1u << 0;
    adc_buffer_ready = true;
    window_start_time_us = time_us_32();
}

/* ================================================================
 * Core 1 entry point — TNN on L3 approximation
 * ================================================================ */

static void core1_entry(void) {
    while (1) {
        mbox_wait_for_dispatch();

        uint32_t start_cycles = time_us_32();

        /* Run TNN encoder on L3 approximation [21][313] */
        const int32_t (*l3)[L3_APPROX_LEN] = lifting_get_l3_approx();
        run_tnn_encoder_inference(l3, L3_APPROX_LEN);

        if (mbox_should_abort()) {
            mbox_signal_error();
            continue;
        }

        /* Run entropy encoder (golden path) */
        run_hybrid_entropy_encoder(false);

        /* Copy result to golden output buffer */
        const uint8_t *src = get_entropy_buffer();
        uint32_t len = get_entropy_buffer_used_bytes();
        for (uint32_t i = 0; i < len && i < 240; i++) {
            golden_output_buf[i] = src[i];
        }

        uint32_t elapsed = time_us_32() - start_cycles;

        mbox_signal_done(
            (uint32_t)golden_output_buf,
            len,
            elapsed);
    }
}

/* ================================================================
 * Lightning path (Core 0, always runs — legacy fallback)
 * ================================================================ */

static void run_lightning_path(void) {
    current_state = STATE_LIGHTNING_DSP;

    for (int c = 0; c < 6; c++) {
        apply_toeplitz_sensing(raw_adc_buffer[c], lifting_tile_2d[c], 32, 2500);
    }
    run_2d_lifting(lifting_tile_2d);
    lpc_predict_only(lifting_tile_2d);

    run_hybrid_entropy_encoder(true);
}

/* Safe mode hook — abort in-progress inference on Core 1 */
void scheduler_abort_inference(void) {
    /* Signal Core 1 to halt; actual implementation depends on mailbox state.
     * For now, the watchdog-based safe mode will reset both cores. */
}

/* ================================================================
 * Main scheduler loop (Core 0)
 * ================================================================ */

void lamquant_scheduler_v7_1_init(void) {
    snn_init();
    mbox_init();
    safety_init();
    frame_sequence = 0;

    /* Core 1 stack in SCRATCH_Y — zero-contention, frees 2 KB main RAM */
    static uint32_t core1_stack[512] __attribute__((section(".scratch_y.core1_stack")));
    multicore_launch_core1_with_stack(core1_entry, core1_stack, sizeof(core1_stack));
}

void lamquant_scheduler_v7_1_run(void) {
    if (!adc_buffer_ready) {
        current_state = STATE_SLEEP;
        enter_dormant_state();
        return;
    }
    adc_buffer_ready = false;

    int window_len = 2500;

    /* Stage 1: HP biquad prefilter (single stage, DC removal only) */
    current_state = STATE_PREFILTER;
    run_biquad_prefilter(window_len);

    /* Stage 2: Raw USB output (if enabled, non-blocking DMA) */
    if (output_mode == OUTPUT_RAW_ONLY ||
        output_mode == OUTPUT_DUAL) {
        trigger_raw_usb_tx(frame_sequence, 21);
    }

    /* Stage 3: Compressed BLE output (if enabled) */
    if (output_mode == OUTPUT_COMPRESSED_ONLY ||
        output_mode == OUTPUT_DUAL) {

        /* Stage 3a: LPC analysis — order-8, per-channel
         * Autocorrelation on first 256 samples, prediction on all 2500.
         * Output: lpc_coeffs[21][8] + lpc_residual[21][2500] */
        current_state = STATE_LPC_ANALYSIS;
        lpc_analyze((const int32_t (*)[2500])raw_adc_buffer,
                    lpc_coeffs, lpc_residual);

        /* Stage 3b: 3-level lifting DWT on LPC residual
         * Output: L3 approximation [21][313] + detail subbands
         * Note: lpc_residual is modified in-place during lifting */
        current_state = STATE_LIFTING_DWT;
        lifting_3level(lpc_residual);

        /* Stage 3c: SNN inference on L3 approximation
         * Input: [21][313] at stride 1 (same temporal resolution as
         * Gen 7.0's [21][2500] at stride 8, since 313 ≈ 2500/8) */
        current_state = STATE_SNN_INFERENCE;
        const int32_t (*l3)[L3_APPROX_LEN] = lifting_get_l3_approx();
        snn_inference(l3, L3_APPROX_LEN);

        /* Safety: archive L3 approximation for pre-ictal lookback */
        safety_push_preictal(l3, window_start_time_us / 1000);

        uint8_t activity = snn_activity_sum();

        /* Safety: log seizure onset when SNN detects activity */
        if (activity > 0) {
            safety_on_seizure_start(activity, 0);
        }

        /* Dispatch Core 1 for golden path if activity detected */
        bool golden_dispatched = false;
        if (activity > 0) {
            if (shared_mailbox.cmd == MBOX_IDLE || shared_mailbox.cmd == MBOX_DONE) {
                shared_mailbox.cmd = MBOX_IDLE;
                mbox_dispatch(activity, frame_sequence);
                golden_dispatched = true;
            }
        }

        /* Lightning path ALWAYS runs (Core 0, guaranteed fallback) */
        run_lightning_path();

        /* Check if Core 1 finished golden path */
        current_state = STATE_WAITING_CORE1;
        bool use_golden = false;

        if (golden_dispatched) {
            uint32_t wait_start = time_us_32();
            while (!mbox_is_done() &&
                   (time_us_32() - wait_start) < CORE1_TIMEOUT_US) {
                /* Wait for Core 1 golden path completion */
            }

            if (mbox_is_done()) {
                use_golden = true;
                shared_mailbox.cmd = MBOX_IDLE;
            }
        }

        /* Transmit best available compressed packet */
        current_state = STATE_TX_READY;

        if (use_golden) {
            /* Golden buffer already filled by Core 1 */
        }

        trigger_ble_dma_tx();

        /* Safety: push compressed packet to BLE retry buffer */
        {
            uint32_t pkt_len = get_entropy_buffer_used_bytes();
            if (pkt_len > 0) {
                safety_ble_push_packet(golden_output_buf, (uint16_t)pkt_len,
                                       frame_sequence);
            }
            safety.faults.total_windows_encoded++;
        }

        /* Safety: periodic impedance + battery check (every 10th window) */
        if ((frame_sequence & 0x0F) == 0) {
            /* TODO: read actual impedance from AFE; stub with 0 for now */
            for (uint8_t ch = 0; ch < IMPEDANCE_CHANNELS; ch++) {
                safety_update_impedance(ch, 0);
            }
            /* TODO: read actual battery from ADC; stub with 100% for now */
            safety_update_battery(3700, 100);
        }
    }

    dma_adc_rearm();
    frame_sequence = (frame_sequence + 1) & 0x7FFFFFFF;  /* 31-bit wrap — safe for dedup */
}

/* ================================================================
 * Output mode control
 * ================================================================ */

void scheduler_v7_1_set_output_mode(output_mode_t mode) {
    output_mode = mode;
}

output_mode_t scheduler_v7_1_get_output_mode(void) {
    return output_mode;
}

/* Power state helpers */
void scheduler_v7_1_abort(void) {
    mbox_abort();
    current_state = STATE_SLEEP;
    adc_buffer_ready = false;
    for (int c = 0; c < 6; c++)
        for (int t = 0; t < 32; t++)
            lifting_tile_2d[c][t] = 0;
}

void scheduler_v7_1_reinit(void) {
    current_state = STATE_SLEEP;
    window_start_time_us = 0;
    adc_buffer_ready = false;
    frame_sequence = 0;
}
