#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "hardware/irq.h"
#include "hardware/dma.h"
#include "power_states.h"
#include <stddef.h>

/*
 * LamQuant Gen 6 — Event-Driven Pipeline Scheduler
 * =================================================
 * Two execution paths:
 *
 *   GOLDEN:    biquad → TNN encoder → FSQ → rANS → BLE
 *              Full quality, uses TernaryMobileNetV5 weights from SRAM4.
 *
 *   LIGHTNING: biquad → Toeplitz CS → 2D lifting → LPC predict → Golomb-Rice → BLE
 *              Guaranteed deadline. Used when golden path would breach 4ms,
 *              or during seizure bursts where LPC is sufficient.
 *
 * The system NEVER misses a window. If golden can't finish, lightning fires.
 */

#define DEADLINE_US              4000
#define GOLDEN_BUDGET_US         3500   // Must finish TNN by this point
#define INFERENCE_ABORT_US       3800   // Hard abort, switch to lightning residual

// Lightning path workspace (SRAM5, isolated from SRAM4 weights)
int32_t lifting_tile_2d[6][32] __attribute__((section(".workspace_sram5"), aligned(4)));

typedef enum {
    STATE_SLEEP,
    STATE_PREFILTER,
    STATE_INFERENCE,
    STATE_FSQ_ENCODE,
    STATE_LIGHTNING_DSP,
    STATE_TX_READY,
} PipelineStage;

volatile PipelineStage current_state = STATE_SLEEP;
volatile uint32_t window_start_time_us = 0;
volatile bool adc_buffer_ready = false;

// --- Golden path externs ---
extern void run_biquad_prefilter(int window_len);
extern void run_tnn_encoder_inference(const int32_t input[][2500], int T);
extern void run_hybrid_entropy_encoder(bool degraded);
extern void trigger_ble_dma_tx(void);
extern void dma_adc_rearm(void);

// ADC buffer (populated by DMA, filtered by biquad)
extern int32_t raw_adc_buffer[21][2500];

// --- Lightning path externs ---
extern void apply_toeplitz_sensing(const int32_t* input, int32_t* output, int M, int N);
extern void run_2d_lifting(int32_t tile[6][32]);
extern void lpc_predict_only(int32_t tile[6][32]);

// --- Seizure detection ---
#define SEIZURE_VARIANCE_THRESHOLD  5000000
#define DEGRADED_VARIANCE_THRESHOLD 2000000

typedef enum {
    MODE_NORMAL,
    MODE_DEGRADED,
    MODE_SEIZURE,
} SystemMode;

/*
 * Quick variance estimate from first 6 channels × 32 samples.
 * Runs before inference to select golden vs lightning.
 * O(192) — negligible latency.
 */
static SystemMode detect_mode_from_adc(void) {
    int64_t total_variance = 0;

    for (int c = 0; c < 6; c++) {
        int64_t sum = 0, sum_sq = 0;
        for (int i = 0; i < 32; i++) {
            int32_t v = raw_adc_buffer[c][i];
            sum += v;
            sum_sq += (int64_t)v * v;
        }
        total_variance += (sum_sq / 32) - ((sum / 32) * (sum / 32));
    }
    int64_t avg = total_variance / 6;

    if (avg > SEIZURE_VARIANCE_THRESHOLD)  return MODE_SEIZURE;
    if (avg > DEGRADED_VARIANCE_THRESHOLD) return MODE_DEGRADED;
    return MODE_NORMAL;
}

// DMA ISR: LC-ADC window complete
void __isr on_adc_dma_complete(void) {
    dma_hw->ints0 = 1u << 0;
    adc_buffer_ready = true;
    window_start_time_us = time_us_32();
}

/*
 * Lightning path: Toeplitz CS → lifting → LPC → Golomb-Rice.
 * Guaranteed to finish within deadline.
 */
static void run_lightning_path(void) {
    current_state = STATE_LIGHTNING_DSP;

    for (int c = 0; c < 6; c++) {
        apply_toeplitz_sensing(raw_adc_buffer[c], lifting_tile_2d[c], 32, 2500);
    }
    run_2d_lifting(lifting_tile_2d);
    lpc_predict_only(lifting_tile_2d);

    current_state = STATE_FSQ_ENCODE;
    run_hybrid_entropy_encoder(true);
}

/*
 * Golden path: TNN encoder → FSQ → rANS.
 * Full quality. Falls back to degraded flag if deadline is breached.
 */
static void run_golden_path(int window_len) {
    current_state = STATE_INFERENCE;
    run_tnn_encoder_inference(
        (const int32_t (*)[2500])raw_adc_buffer, window_len);

    uint32_t elapsed = time_us_32() - window_start_time_us;
    bool degraded = (elapsed > INFERENCE_ABORT_US);

    current_state = STATE_FSQ_ENCODE;
    run_hybrid_entropy_encoder(degraded);
}

/* --- Power state helpers (called by power_states.c safe mode) --- */

void scheduler_abort_inference(void) {
    current_state = STATE_SLEEP;
    adc_buffer_ready = false;
    // Zero the lightning workspace to prevent stale data on resume
    for (int c = 0; c < 6; c++)
        for (int t = 0; t < 32; t++)
            lifting_tile_2d[c][t] = 0;
}

void scheduler_reinit(void) {
    current_state = STATE_SLEEP;
    window_start_time_us = 0;
    adc_buffer_ready = false;
}

/*
 * Main entry point. Called in a loop from main().
 * One ADC window per invocation.
 */
void lamquant_scheduler_run(void) {
    if (!adc_buffer_ready) {
        current_state = STATE_SLEEP;
        enter_dormant_state();
        return;
    }
    adc_buffer_ready = false;

    int window_len = 2500;

    // Stage 1: Biquad prefilter (both paths need clean signal)
    current_state = STATE_PREFILTER;
    run_biquad_prefilter(window_len);

    // Path selection
    SystemMode mode = detect_mode_from_adc();
    uint32_t elapsed = time_us_32() - window_start_time_us;

    if (mode == MODE_SEIZURE || elapsed > GOLDEN_BUDGET_US) {
        run_lightning_path();
    } else {
        run_golden_path(window_len);
    }

    // Stage 5: BLE transmit
    current_state = STATE_TX_READY;
    trigger_ble_dma_tx();

    // Re-arm DMA for next window
    dma_adc_rearm();
}
