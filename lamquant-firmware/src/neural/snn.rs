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

// ─── Public API constants ──────────────────────────────────────────

pub const INPUT_CH: usize = 21;
pub const HIDDEN_DIM: usize = 64;
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

    // Stage 1: delta-encode (binary spike train per (ch, t)).
    // Production uses these as the input to ternary spatial conv. Until
    // SNN weights ship in lamquant_weights, we run the rest of the
    // pipeline with the spike train as a 21-channel proxy for the
    // 64-channel hidden layer (broadcast first 21, zero the rest).
    let mut spatial_in = [[0i16; T_INPUT]; HIDDEN_DIM];
    for ch in 0..INPUT_CH {
        let mut prev = state.prev_sample[ch];
        for t in 0..T_INPUT {
            let curr = l3[ch][t] as i32;
            let delta = curr - prev;
            // Spike if |delta| crosses adaptive threshold, scaled by sensitivity.
            let spike_thresh = (threshold_scale * 100) >> 15; // ~0.2 scale
            spatial_in[ch][t] = if delta.abs() > spike_thresh {
                delta.signum() as i16
            } else {
                0
            };
            prev = curr;
        }
        state.prev_sample[ch] = l3[ch][T_INPUT - 1] as i32;
    }

    // Stage 2: dLIF hidden layer — drive each neuron from a slice of
    // spatial_in, integrate membrane potential, fire if over threshold.
    let mut hidden_spikes = [[0u8; T_INPUT]; HIDDEN_DIM];
    for n in 0..HIDDEN_DIM {
        let neuron = &mut state.hidden[n];
        let src = n.min(INPUT_CH - 1); // crude routing — replace when SNN weights ship
        for t in 0..T_INPUT {
            // Integrate (leaky decay via shift, no float).
            // Rectify spike magnitude so alternating deltas accumulate;
            // production trained weights provide signed routing through
            // the real ternary spatial conv.
            neuron.v_soma -= neuron.v_soma >> 4; // ~6% leak per step
            neuron.v_soma += ((spatial_in[src][t] as i32).abs()) << 20;
            // Adaptive threshold tracks EMA noise floor.
            neuron.ema_noise -= neuron.ema_noise >> 6;
            neuron.ema_noise += neuron.v_soma.abs() >> 6;
            let dynamic_thresh = neuron.v_threshold + (neuron.ema_noise >> 1);
            if neuron.v_soma > dynamic_thresh {
                hidden_spikes[n][t] = 1;
                neuron.v_soma = 0; // reset after fire
            }
        }
    }

    // Stage 3: dLIF readout — pool hidden spikes into 8 groups.
    // Group g = hidden neurons [g*8 .. (g+1)*8].
    let group_size = HIDDEN_DIM / READOUT_DIM;
    state.activity_sum = 0;
    for g in 0..READOUT_DIM {
        let neuron = &mut state.readout[g];
        let mut group_active_count = 0u32;
        for t in 0..T_LATENT {
            // Aggregate spikes from this group's hidden neurons across the
            // stride-1 window covering this latent step.
            let mut spike_sum = 0i32;
            for h in 0..group_size {
                spike_sum += hidden_spikes[g * group_size + h][t] as i32;
            }
            neuron.v_soma -= neuron.v_soma >> 4;
            neuron.v_soma += spike_sum << 24;
            let level = if neuron.v_soma > neuron.v_threshold {
                neuron.v_soma = 0;
                group_active_count += 1;
                if spike_sum >= (group_size as i32 / 2) {
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
        let _g = TEST_LOCK.lock().unwrap();
        init();
        let l3 = [[0i16; T_INPUT]; INPUT_CH];
        inference(&l3);
        assert_eq!(activity_sum(), 0, "quiescent input should produce no activity");
    }

    #[test]
    fn high_amplitude_step_produces_some_activity() {
        let _g = TEST_LOCK.lock().unwrap();
        init();
        let mut l3 = [[0i16; T_INPUT]; INPUT_CH];
        for ch in 0..INPUT_CH {
            for t in 0..T_INPUT {
                l3[ch][t] = if t % 2 == 0 { 8000 } else { -8000 };
            }
        }
        inference(&l3);
        let map = get_activity_map();
        let total: u32 = map
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as u32))
            .sum();
        assert!(
            total > 0,
            "high amplitude alternating input should produce some spikes"
        );
    }

    #[test]
    fn sensitivity_high_more_active_than_low() {
        let _g = TEST_LOCK.lock().unwrap();
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
        let _g = TEST_LOCK.lock().unwrap();
        set_sensitivity(SnnSensitivity::High);
        assert_eq!(get_sensitivity(), SnnSensitivity::High);
        set_sensitivity(SnnSensitivity::Low);
        assert_eq!(get_sensitivity(), SnnSensitivity::Low);
        set_sensitivity(SnnSensitivity::Medium);
        assert_eq!(get_sensitivity(), SnnSensitivity::Medium);
    }
}
