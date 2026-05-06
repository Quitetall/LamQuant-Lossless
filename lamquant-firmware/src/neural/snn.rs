//! dLIF SNN — Dendritic Leaky Integrate-and-Fire activity classifier.
//!
//! Port of `firmware/snn/snn.c`. Classifies the L3 approximation
//! [21][313] into 8 activity groups × `T_LATENT` timesteps. The
//! scheduler reads `activity_sum()` to decide whether to wake Core 1
//! for full TNN inference, and `get_activity_map()` drives the
//! adaptive FSQ level table.
//!
//! ## Pipeline
//!
//!   delta-encode (spike threshold crossing per t)
//!     → ternary spatial conv (21 → 64 ch, k=5)
//!     → dLIF hidden layer (64 neurons × 2 dendrites)
//!     → dLIF readout (8 groups × 2 dendrites)
//!     → activity_map [8][312]
//!
//! ## Status: Phase 3 stub
//!
//! The weight-export pipeline does not yet emit SNN weights to the
//! Rust crate (only encoder is in `lamquant_weights::generated::focal`).
//! This module exposes the production API and runs the integer state
//! machine, but the spatial-conv step uses placeholder identity-mapping
//! weights so it never panics on missing tables. Once
//! `lamquant_weights::generated::snn` ships, swap the placeholder block
//! for real ternary conv calls.

use core::sync::atomic::{AtomicU8, Ordering};

use lamquant_weights::generated::snn::{readout, spatial_mix};

// ─── Public API constants ──────────────────────────────────────────

pub const INPUT_CH: usize = 21;
/// Mamba d_model — set by the trained checkpoint via `spatial_mix.weight`
/// shape `(D_MODEL, INPUT_CH) = (40, 21)`. Hidden state width.
pub const D_MODEL: usize = 40;
/// Legacy alias for the dLIF placeholder routing (unused by the production
/// path now that spatial_mix runs the real projection).
pub const HIDDEN_DIM: usize = D_MODEL;
pub const READOUT_DIM: usize = 8;
pub const T_LATENT: usize = 312; // 2500 / 8 in C path; 313/1 effectively in Rust
pub const NUM_DENDRITES: usize = 2;
pub const T_INPUT: usize = 313;

// ─── Sensitivity presets ──────────────────────────────────────────

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SnnSensitivity {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl SnnSensitivity {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Low,
            2 => Self::High,
            _ => Self::Medium,
        }
    }

    /// Spike threshold scale (Q15). Higher = harder to trigger.
    fn threshold_scale_q15(&self) -> i32 {
        match self {
            Self::Low => 1 << 16,    // 2.0× — quiet
            Self::Medium => 1 << 15, // 1.0×
            Self::High => 1 << 14,   // 0.5× — twitchy
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ActivityLevel {
    Quiescent = 0,
    Active = 1,
    High = 2,
}

// ─── State (RAM-resident across windows) ──────────────────────────

#[derive(Copy, Clone)]
struct DlifNeuron {
    /// Q31 somatic membrane potential.
    v_soma: i32,
    /// Q31 dendritic compartments.
    v_dendrite: [i32; NUM_DENDRITES],
    /// Q31 adaptive threshold.
    v_threshold: i32,
    /// Q31 noise-floor EMA tracker.
    ema_noise: i32,
}

impl DlifNeuron {
    const fn new() -> Self {
        Self {
            v_soma: 0,
            v_dendrite: [0; NUM_DENDRITES],
            // Threshold tuned for the stub-routing pipeline. Production
            // SNN weights set this from a baked Q31 calibration constant.
            v_threshold: 1 << 22,
            ema_noise: 0,
        }
    }
}

struct SnnState {
    hidden: [DlifNeuron; HIDDEN_DIM],
    readout: [DlifNeuron; READOUT_DIM],
    activity_map: [[u8; T_LATENT]; READOUT_DIM],
    prev_sample: [i32; INPUT_CH],
    activity_sum: u32,
    initialized: bool,
}

impl SnnState {
    const fn new() -> Self {
        Self {
            hidden: [DlifNeuron::new(); HIDDEN_DIM],
            readout: [DlifNeuron::new(); READOUT_DIM],
            activity_map: [[0; T_LATENT]; READOUT_DIM],
            prev_sample: [0; INPUT_CH],
            activity_sum: 0,
            initialized: false,
        }
    }
}

static mut STATE: SnnState = SnnState::new();
static SENSITIVITY: AtomicU8 = AtomicU8::new(SnnSensitivity::Medium as u8);

// ─── Public API ─────────────────────────────────────────────────────

/// Reset all neuron states and zero the activity map.
/// Call once at boot.
pub fn init() {
    // SAFETY: single-threaded init at boot, called before any inference.
    unsafe {
        STATE = SnnState::new();
        STATE.initialized = true;
    }
    SENSITIVITY.store(SnnSensitivity::Medium as u8, Ordering::Relaxed);
}

/// Configure spike-event sensitivity.
pub fn set_sensitivity(level: SnnSensitivity) {
    SENSITIVITY.store(level as u8, Ordering::Relaxed);
}

pub fn get_sensitivity() -> SnnSensitivity {
    SnnSensitivity::from_u8(SENSITIVITY.load(Ordering::Relaxed))
}

/// Run inference on one 10-second window of L3 approximation.
///
/// Input: 21 × 313 Q15-ish int16 samples.
/// Output: writes activity_map; readable via `get_activity_map()`.
pub fn inference(l3: &[[i16; T_INPUT]; INPUT_CH]) {
    // SAFETY: scheduler guarantees inference runs single-threaded on Core 0.
    let state = unsafe { &mut STATE };
    if !state.initialized {
        return;
    }

    let sensitivity = get_sensitivity();
    let threshold_scale = sensitivity.threshold_scale_q15();

    // Stage 1: delta-encode the input (binary spike train per (ch, t)).
    // Stays the same as before — the trained SNN expects spike features.
    let mut spike_train = [[0i16; T_INPUT]; INPUT_CH];
    for ch in 0..INPUT_CH {
        let mut prev = state.prev_sample[ch];
        for t in 0..T_INPUT {
            let curr = l3[ch][t] as i32;
            let delta = curr - prev;
            let spike_thresh = (threshold_scale * 100) >> 15;
            spike_train[ch][t] = if delta.abs() > spike_thresh {
                delta.signum() as i16
            } else {
                0
            };
            prev = curr;
        }
        state.prev_sample[ch] = l3[ch][T_INPUT - 1] as i32;
    }

    // Stage 2: spatial_mix — Linear(21 → D_MODEL=40) using TRAINED INT8
    // weights from `lamquant_weights::generated::snn::spatial_mix`.
    //
    // Layout: SPATIAL_MIX_WEIGHT is row-major [out_ch][in_ch].
    // Dequant: f32 = q8 * SCALE; we keep i32 accumulators and apply the
    // scale once at the end (folded into v_soma drive strength).
    //
    // hidden_drive[oc][t] = sum_ic(SPATIAL_MIX_WEIGHT[oc][ic] * spike_train[ic][t])
    //                       * SPATIAL_MIX_WEIGHT_SCALE  (logical f32 step)
    //                     + SPATIAL_MIX_BIAS[oc] * SPATIAL_MIX_BIAS_SCALE
    let mut hidden_spikes = [[0u8; T_INPUT]; D_MODEL];

    // Precompute Q15 versions of the i16-bucket scaled weights so the inner
    // loop stays integer-only (no f32 hot path on RV32IMAC).
    //   weight_scale_q15 = round(SPATIAL_MIX_WEIGHT_SCALE * 2^15)
    //   bias_scale_q15   = round(SPATIAL_MIX_BIAS_SCALE   * 2^15)
    // Each i32 accumulator is then `acc * weight_scale_q15 >> 15` to land
    // in roughly [-32768, +32767] before the leaky integrator drives it
    // up by `<< 20`.
    let w_scale_q15 = (spatial_mix::SPATIAL_MIX_WEIGHT_SCALE * 32768.0) as i32;
    let b_scale_q15 = (spatial_mix::SPATIAL_MIX_BIAS_SCALE * 32768.0) as i32;

    for oc in 0..D_MODEL {
        let neuron = &mut state.hidden[oc];
        // Bias contribution (scaled, applied each timestep so it doesn't
        // saturate the integrator).
        let bias_step =
            (spatial_mix::SPATIAL_MIX_BIAS[oc] as i32).wrapping_mul(b_scale_q15) >> 15;

        for t in 0..T_INPUT {
            // 21-input matmul; kept fully integer.
            let mut acc: i32 = 0;
            for ic in 0..INPUT_CH {
                let w = spatial_mix::SPATIAL_MIX_WEIGHT[oc * INPUT_CH + ic] as i32;
                acc = acc.wrapping_add(w.wrapping_mul(spike_train[ic][t] as i32));
            }
            let drive = (acc.wrapping_mul(w_scale_q15) >> 15).wrapping_add(bias_step);

            // Leaky integrator (replaces dLIF placeholder routing).
            // `drive` magnitude is bounded by INPUT_CH * 127 * 1 ≈ 2700
            // before scaling, then * w_scale_q15 (~1000) >> 15 ≈ 80. Shift
            // by 17 lifts that into the i32 region so the membrane
            // threshold (1 << 22) is reachable on real signal — too small
            // a shift left it gated below threshold and never fired.
            neuron.v_soma -= neuron.v_soma >> 4;
            neuron.v_soma = neuron.v_soma.saturating_add(drive << 17);
            neuron.ema_noise -= neuron.ema_noise >> 6;
            neuron.ema_noise += neuron.v_soma.abs() >> 6;
            let dynamic_thresh =
                neuron.v_threshold + (neuron.ema_noise >> 1);
            if neuron.v_soma > dynamic_thresh {
                hidden_spikes[oc][t] = 1;
                neuron.v_soma = 0;
            }
        }
    }

    // Stage 3: readout — Linear(D_MODEL → 8) using TRAINED INT8 weights
    // from `lamquant_weights::generated::snn::readout`.
    // Output shape: activity_map[g][t_latent] for g in 0..8.
    let r_w_scale_q15 = (readout::READOUT_WEIGHT_SCALE * 32768.0) as i32;
    let r_b_scale_q15 = (readout::READOUT_BIAS_SCALE * 32768.0) as i32;

    state.activity_sum = 0;
    for g in 0..READOUT_DIM {
        let neuron = &mut state.readout[g];
        let bias_step =
            (readout::READOUT_BIAS[g] as i32).wrapping_mul(r_b_scale_q15) >> 15;
        let mut group_active_count = 0u32;

        for t in 0..T_LATENT {
            // Sum the readout-projected hidden activity for timestep t.
            let mut acc: i32 = 0;
            for hc in 0..D_MODEL {
                let w = readout::READOUT_WEIGHT[g * D_MODEL + hc] as i32;
                let h_active = hidden_spikes[hc][t] as i32;
                acc = acc.wrapping_add(w.wrapping_mul(h_active));
            }
            let drive = (acc.wrapping_mul(r_w_scale_q15) >> 15).wrapping_add(bias_step);

            neuron.v_soma -= neuron.v_soma >> 4;
            neuron.v_soma = neuron.v_soma.saturating_add(drive << 18);

            let level = if neuron.v_soma > neuron.v_threshold {
                neuron.v_soma = 0;
                group_active_count += 1;
                // Strong recent drive → High; otherwise Active.
                if drive.abs() > 4 {
                    ActivityLevel::High as u8
                } else {
                    ActivityLevel::Active as u8
                }
            } else {
                ActivityLevel::Quiescent as u8
            };
            state.activity_map[g][t] = level;
        }
        if group_active_count > 8 {
            state.activity_sum += 1;
        }
    }
}

/// Sum of groups that crossed the activity threshold this window.
/// Scheduler dispatch: 0 → quiescent, > 0 → wake Core 1 / wake TNN.
pub fn activity_sum() -> u32 {
    // SAFETY: read of u32 from single-producer state.
    unsafe { STATE.activity_sum }
}

/// Per-(group, timestep) activity level for adaptive FSQ.
pub fn get_activity(group: usize, t_latent: usize) -> ActivityLevel {
    if group >= READOUT_DIM || t_latent >= T_LATENT {
        return ActivityLevel::Quiescent;
    }
    // SAFETY: bounded read from initialized state.
    let v = unsafe { STATE.activity_map[group][t_latent] };
    match v {
        2 => ActivityLevel::High,
        1 => ActivityLevel::Active,
        _ => ActivityLevel::Quiescent,
    }
}

/// Full activity map (`[8][312]`) for adaptive FSQ scheduling.
pub fn get_activity_map() -> &'static [[u8; T_LATENT]; READOUT_DIM] {
    // SAFETY: returns shared reference into static state. Scheduler
    // guarantees no concurrent writer while caller reads.
    unsafe { &STATE.activity_map }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // SNN owns global `static mut STATE` which is single-threaded by
    // contract on the embedded target. Cargo runs unit tests in parallel
    // by default, so guard the shared state behind a mutex for the host
    // suite. Production firmware never sees this lock.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn zero_input_produces_no_activity() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        init();
        let l3 = [[0i16; T_INPUT]; INPUT_CH];
        inference(&l3);
        assert_eq!(activity_sum(), 0, "quiescent input should produce no activity");
    }

    #[test]
    fn real_weights_path_runs_without_panic() {
        // Smoke test for the post-W8 path: spatial_mix and readout now
        // consume real INT8 weights from `lamquant_weights::generated::snn`.
        // We assert the inference completes and produces a well-formed
        // activity_map. Functional thresholding behaviour depends on the
        // full Mamba SSM dynamics (still stubbed via leaky integrator);
        // strict spike-count assertions belong with task #28 once the
        // SSM block is implemented properly.
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        init();
        let mut l3 = [[0i16; T_INPUT]; INPUT_CH];
        for ch in 0..INPUT_CH {
            for t in 0..T_INPUT {
                l3[ch][t] = ((t as i32) * 200 - 32000) as i16;
            }
        }
        inference(&l3);
        let map = get_activity_map();
        // Every entry must be a valid activity level (0/1/2).
        for row in map.iter() {
            for &v in row.iter() {
                assert!(v <= 2, "activity_map contains invalid level {}", v);
            }
        }
    }

    #[test]
    fn sensitivity_high_more_active_than_low() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut l3 = [[0i16; T_INPUT]; INPUT_CH];
        for ch in 0..INPUT_CH {
            for t in 0..T_INPUT {
                l3[ch][t] = (((t * 17 + ch * 113) % 4096) as i16) - 2048;
            }
        }

        init();
        set_sensitivity(SnnSensitivity::Low);
        inference(&l3);
        let low_total: u32 = get_activity_map()
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as u32))
            .sum();

        init();
        set_sensitivity(SnnSensitivity::High);
        inference(&l3);
        let high_total: u32 = get_activity_map()
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as u32))
            .sum();

        assert!(
            high_total >= low_total,
            "HIGH sensitivity should not detect fewer spikes than LOW \
             (high={} low={})",
            high_total,
            low_total
        );
    }

    #[test]
    fn sensitivity_round_trips() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        set_sensitivity(SnnSensitivity::High);
        assert_eq!(get_sensitivity(), SnnSensitivity::High);
        set_sensitivity(SnnSensitivity::Low);
        assert_eq!(get_sensitivity(), SnnSensitivity::Low);
        set_sensitivity(SnnSensitivity::Medium);
        assert_eq!(get_sensitivity(), SnnSensitivity::Medium);
    }
}
