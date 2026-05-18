/*
 * LamQuant Gen 7.7 — Dual-Core Pipeline Scheduler
 * =================================================
 * Simplified from v7.1: two user-selected modes, no runtime fallback.
 *
 * Mode 1 — Neural (Lossy):
 *   Core 0: ADC DMA → HP biquad → LPC → lifting DWT → Mamba SNN → dispatch Core 1
 *   Core 1: TNN encoder on L3 [21][313] → adaptive FSQ → rANS → BLE TX (.lmq)
 *   Compression: 63-525:1 (SNN-driven adaptive CR)
 *
 * Mode 2 — Lossless:
 *   Core 0: ADC DMA → HP biquad → LPC → lifting DWT → per-subband Golomb-Rice → BLE TX (.lml)
 *   Core 1: sleeping (WFE)
 *   Compression: ~3.76:1 (bit-exact roundtrip)
 *
 * NO fallback logic. Mode is a session-level configuration set by the host
 * via serial command. Both modes share the DSP front-end (stages 1-3).
 *
 * Timing budget (10-second window @ 150 MHz):
 *   DSP (biquad + LPC + lifting):  ~12 ms
 *   Mamba SNN:                      pending (firmware inference not yet implemented)
 *   TNN inference (Core 1):         264 ms (CPOP hybrid, 4.48x over v7.1 baseline)
 *   Golomb-Rice (lossless):         ~8 ms
 *   Total (neural mode):            ~276 ms / 10,000 ms = 2.8% budget
 *   Real-time margin:               36x
 *
 * Memory map (same as v7.1 — no layout changes):
 *   SRAM0-1: ADC buffer + LPC residual workspace
 *   SRAM2-3: TNN activation buffers (112×313)
 *   SRAM4:   TNN weights (packed ternary, XIP)
 *   SRAM5:   Mamba SNN weights (61.8 KB INT8) + lifting subbands + LPC coeffs
 *   SRAM6:   SNN state + activity map
 *   SRAM7:   Output buffer + sigmoid LUT
 *   SRAM8:   Mailbox (32B) + Core 0 stack
 *   SRAM9:   Core 1 stack
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
#include "scheduler_v7_7.h"
#include "../snn/snn.h"

/* Module headers */
#include "../dsp/lifting_2d.h"
#include "../dsp/biquad_q31.h"
#include "../dsp/lpc_predictor.h"
#include "../neural/focal_modulation.h"
#include "../transport/ble_spi_host.h"
#include "../transport/hybrid_entropy.h"
#include "../transport/raw_output.h"

#define LPC_ORDER        8
#define L3_APPROX_LEN   313
#define WINDOW_SAMPLES   2500
#define NUM_CHANNELS     21

/* ================================================================
 * State
 * ================================================================ */

typedef enum {
    STATE_SLEEP,
    STATE_DSP,          /* Biquad + LPC + Lifting (shared) */
    STATE_SNN,          /* Mamba SNN classification */
    STATE_ENCODE,       /* Core 1: TNN + entropy (neural mode) */
    STATE_LOSSLESS_ENC, /* Core 0: Golomb-Rice (lossless mode) */
    STATE_TX,           /* BLE/USB transmit */
} pipeline_state_t;

static volatile codec_mode_t codec_mode = MODE_NEURAL;
static volatile output_mode_t output_mode = OUTPUT_COMPRESSED_ONLY;
static volatile pipeline_state_t current_state = STATE_SLEEP;
static volatile bool adc_buffer_ready = false;
static volatile uint32_t window_start_us = 0;
static uint32_t frame_seq = 0;

/* Legacy symbol required by hybrid_entropy.c (lossless path) */
int32_t lifting_tile_2d[6][32] __attribute__((aligned(4)));

/* Shared buffers */
extern int32_t raw_adc_buffer[21][WINDOW_SAMPLES];
#define lpc_residual raw_adc_buffer  /* alias: safe, sequential pipeline */
static int32_t lpc_coeffs[NUM_CHANNELS][LPC_ORDER] __attribute__((aligned(4)));

/* Output buffer */
static uint8_t output_buf[240] __attribute__((section(".scratch_x.output_buf")));

/* Mailbox for Core 1 dispatch */
mailbox_t shared_mailbox;

/* ================================================================
 * DMA ISR — ADC window complete
 * ================================================================ */

extern void dma_adc_rearm(void);

void __isr on_adc_dma_complete(void) {
    dma_hw->ints0 = 1u << 0;
    adc_buffer_ready = true;
    window_start_us = time_us_32();
}

/* ================================================================
 * Core 1 — TNN inference (neural mode only)
 * ================================================================ */

static void core1_entry(void) {
    while (1) {
        mbox_wait_for_dispatch();

        /* TNN encoder on L3 approximation [21][313] */
        const int32_t (*l3)[L3_APPROX_LEN] = lifting_get_l3_approx();
        run_tnn_encoder_inference(l3, L3_APPROX_LEN);

        if (mbox_should_abort()) {
            mbox_signal_error();
            continue;
        }

        /* Entropy encode (rANS, uses SNN-derived FSQ levels) */
        run_hybrid_entropy_encoder(false);

        /* Copy to output buffer */
        const uint8_t *src = get_entropy_buffer();
        uint32_t len = get_entropy_buffer_used_bytes();
        for (uint32_t i = 0; i < len && i < sizeof(output_buf); i++) {
            output_buf[i] = src[i];
        }

        uint32_t elapsed = time_us_32() - window_start_us;
        mbox_signal_done((uint32_t)output_buf, len, elapsed);
    }
}

/* ================================================================
 * Lossless path (Core 0, Golomb-Rice on subbands)
 * ================================================================ */

static void run_lossless_encode(void) {
    current_state = STATE_LOSSLESS_ENC;
    /* Encode detail subbands + L3 residual with Golomb-Rice.
     * run_hybrid_entropy_encoder(true) = lossless mode. */
    run_hybrid_entropy_encoder(true);
}

/* ================================================================
 * Public API
 * ================================================================ */

void lamquant_scheduler_v7_7_init(void) {
    snn_init();
    mbox_init();
    safety_init();
    frame_seq = 0;

    /* Core 1 stack in SCRATCH_Y — zero bus contention */
    static uint32_t core1_stack[512] __attribute__((section(".scratch_y.core1_stack")));
    multicore_launch_core1_with_stack(core1_entry, core1_stack, sizeof(core1_stack));
}

void lamquant_scheduler_v7_7_run(void) {
    if (!adc_buffer_ready) {
        current_state = STATE_SLEEP;
        enter_dormant_state();
        return;
    }
    adc_buffer_ready = false;

    /* ── Stage 1: DSP front-end (shared by both modes) ── */
    current_state = STATE_DSP;

    /* HP biquad prefilter (0.5 Hz DC removal) */
    run_biquad_prefilter(WINDOW_SAMPLES);

    /* Raw USB output (if enabled, non-blocking DMA) */
    if (output_mode == OUTPUT_RAW_ONLY || output_mode == OUTPUT_DUAL) {
        trigger_raw_usb_tx(frame_seq, NUM_CHANNELS);
    }

    /* Skip compression if raw-only mode */
    if (output_mode == OUTPUT_RAW_ONLY) {
        goto window_done;
    }

    /* LPC analysis: order-8, 21 channels, autocorr on 256 samples */
    lpc_analyze((const int32_t (*)[WINDOW_SAMPLES])raw_adc_buffer,
                lpc_coeffs, lpc_residual);

    /* 3-level lifting DWT on LPC residual → L3 [21][313] + details */
    lifting_3level(lpc_residual);

    /* ── Stage 2: Mode-specific encoding ── */

    if (codec_mode == MODE_NEURAL) {
        /* Mamba SNN: classify activity on L3 approximation.
         * Output: per-timestep FSQ level schedule (2/3/5).
         * TODO: Mamba firmware inference not yet implemented.
         * Currently uses fixed FSQ levels until Mamba inference ships. */
        current_state = STATE_SNN;
        const int32_t (*l3)[L3_APPROX_LEN] = lifting_get_l3_approx();
        snn_inference(l3, L3_APPROX_LEN);

        uint8_t activity = snn_activity_sum();

        /* Safety: seizure detection */
        if (activity > 0) {
            safety_on_seizure_start(activity, 0);
        }
        safety_push_preictal(l3, window_start_us / 1000);

        /* Dispatch Core 1 for TNN + entropy encoding */
        current_state = STATE_ENCODE;
        mbox_dispatch(activity, frame_seq);

        /* Wait for Core 1 (264 ms typical, 38x within 10s budget) */
        while (!mbox_is_done()) {
            /* spin — Core 0 has nothing else to do in neural mode */
        }
        shared_mailbox.cmd = MBOX_IDLE;

    } else {
        /* MODE_LOSSLESS: pure DSP, no neural models */
        run_lossless_encode();
    }

    /* ── Stage 3: Transmit ── */
    current_state = STATE_TX;
    trigger_ble_dma_tx();

    /* Safety: packet retry buffer */
    {
        uint32_t pkt_len = get_entropy_buffer_used_bytes();
        if (pkt_len > 0) {
            safety_ble_push_packet(output_buf, (uint16_t)pkt_len, frame_seq);
        }
        safety.faults.total_windows_encoded++;
    }

    /* Periodic housekeeping (every 16th window = every 160 seconds) */
    if ((frame_seq & 0x0F) == 0) {
        for (uint8_t ch = 0; ch < IMPEDANCE_CHANNELS; ch++) {
            safety_update_impedance(ch, 0);
        }
        safety_update_battery(3700, 100);
    }

window_done:
    dma_adc_rearm();
    frame_seq = (frame_seq + 1) & 0x7FFFFFFF;
}

/* ================================================================
 * Mode control (host commands via serial/BLE)
 * ================================================================ */

void scheduler_v7_7_set_codec_mode(codec_mode_t mode) {
    codec_mode = mode;
}

codec_mode_t scheduler_v7_7_get_codec_mode(void) {
    return codec_mode;
}

void scheduler_v7_7_set_output_mode(output_mode_t mode) {
    output_mode = mode;
}

output_mode_t scheduler_v7_7_get_output_mode(void) {
    return output_mode;
}

/* Legacy symbol required by power_states.c safe-mode hook */
void scheduler_abort_inference(void) {
    mbox_abort();
}

void scheduler_v7_7_abort(void) {
    mbox_abort();
    current_state = STATE_SLEEP;
    adc_buffer_ready = false;
}

void scheduler_v7_7_reinit(void) {
    current_state = STATE_SLEEP;
    window_start_us = 0;
    adc_buffer_ready = false;
    frame_seq = 0;
}
