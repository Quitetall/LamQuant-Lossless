/*
 * LamQuant Gen 7 — dLIF SNN Activity Detector
 * =============================================
 * Dendritic Leaky Integrate-and-Fire spiking neural network.
 * Runs on Core 0. Classifies 21-channel EEG into activity levels
 * per spatial group (8 groups × T_latent timesteps).
 *
 * Architecture:
 *   Delta encoder → Ternary conv 21→64 k=5 → dLIF hidden (64, 2 dend)
 *   → dLIF readout (8, 2 dend) → activity_map[8][312]
 *
 * All Q31. Zero float. ~5 KB weights (SRAM5), ~1 KB state (SRAM6).
 */

#ifndef SNN_H
#define SNN_H

#include <stdint.h>
#include <stdbool.h>

/* Activity levels */
typedef enum {
    ACTIVITY_QUIESCENT = 0,
    ACTIVITY_ACTIVE    = 1,
    ACTIVITY_HIGH      = 2
} activity_level_t;

/* dLIF neuron state */
typedef struct {
    int32_t V_soma;          /* Q31 somatic membrane potential */
    int32_t V_dendrite[2];   /* Q31 dendritic compartments */
    int32_t V_threshold;     /* Q31 adaptive threshold */
    int32_t ema_noise;       /* Q31 noise floor tracker */
} dlif_neuron_t;

/* Dimensions */
#define SNN_INPUT_CH      21
#define SNN_HIDDEN_DIM    64
#define SNN_READOUT_DIM   8
#define SNN_CONV_KERNEL   5
#define SNN_NUM_DENDRITES 2
#define SNN_STRIDE_8      8
#define SNN_T_LATENT      312  /* 2500 / 8 */

/* Spike event thresholds (research §1.4). These are the *defaults*; the
 * effective values can be retuned at runtime via snn_set_sensitivity(). */
#define SNN_K_GLOBAL      16   /* Population spike count threshold */
#define SNN_K_LOCAL       4    /* Per-group spike rate threshold */

/* Sensitivity presets selected by the host. */
typedef enum {
    SNN_SENSITIVITY_LOW    = 0,  /* harder to trigger — fewer false alarms */
    SNN_SENSITIVITY_MEDIUM = 1,  /* default */
    SNN_SENSITIVITY_HIGH   = 2,  /* easier to trigger — catches subtle bursts */
    SNN_SENSITIVITY_NUM_OPTIONS
} snn_sensitivity_t;

/*
 * Initialize SNN state (call once at boot).
 * Zeros all neuron states, loads weights from snn_weights.h.
 */
void snn_init(void);

/*
 * Run SNN inference on one 10-second window.
 *
 * input: [21][2500] Q31 filtered EEG (from biquad stage)
 * T: number of temporal samples (normally 2500)
 *
 * After return, activity map is in SRAM6.
 */
void snn_inference(const int32_t input[][313], int T);

/*
 * Get sum of activity across all groups for current window.
 * Returns: 0 = fully quiescent, >0 = some activity detected.
 * Used by scheduler to decide whether to wake Core 1.
 */
uint8_t snn_activity_sum(void);

/*
 * Get per-group activity level for a specific latent timestep.
 */
activity_level_t snn_get_activity(int group, int t_latent);

/*
 * Get pointer to full activity map [8][312].
 * Used by adaptive FSQ to select quantization levels.
 */
const uint8_t (*snn_get_activity_map(void))[SNN_T_LATENT];

/*
 * Switch the spike-event thresholds at runtime. Out-of-range levels are
 * clamped to medium. Safe to call from the serial command handler on
 * Core 0 between windows.
 */
void snn_set_sensitivity(snn_sensitivity_t level);

/* Read-only accessor used by preflight report and EDF metadata. */
snn_sensitivity_t snn_get_sensitivity(void);

#endif /* SNN_H */
