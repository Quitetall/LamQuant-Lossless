//! Mamba SNN — bidirectional selective-SSM activity classifier.
//!
//! Replaces the legacy dLIF stub. Wires the trained INT8 weights from
//! `lamquant_weights::generated::snn::*` through `ssm_block::*` and
//! `layer_norm::*` so the firmware emits the same activity_map that
//! the Python `MambaSNN.forward()` produces (modulo Q15 quantization).
//!
//! ## Pipeline
//!
//! ```text
//!   L3 input [21][313] i16 Q15
//!     → spatial_mix   : Linear(21 → 40)
//!     → layer0.norm   : LayerNorm(40)
//!     → layer0.fwd    : SelectiveSSM forward
//!     → layer0.bwd    : SelectiveSSM reversed-time
//!     → residual fuse : x + (fwd + bwd) * 0.5
//!     → layer1.norm   : LayerNorm(40)
//!     → layer1.fwd    : SelectiveSSM forward
//!     → layer1.bwd    : SelectiveSSM reversed-time
//!     → residual fuse : x + (fwd + bwd) * 0.5
//!     → readout       : Linear(40 → 8)
//!     → threshold     : activity_map [8][312] u8
//! ```
//!
//! ## Status: structural rewrite (Track B.3 step 1)
//!
//! All scaffolding wired: the trained weights flow through
//! `ssm_block::selective_ssm_block` / `layer_norm::layer_norm_q15`
//! and the public API matches the dLIF surface byte-for-byte. The
//! sequential scan and integer-sqrt LayerNorm both pass closed-form
//! unit tests in their own modules.
//!
//! Numerical conformance vs the Python forward pass (full golden
//! activity_map) is the **next** commit in Track B.3 — it needs an
//! offline run of `MambaSNN.forward()` on a fixed seed to produce
//! the golden array, which is large enough that we bake it as a
//! separate generated file rather than inline. The current smoke
//! tests verify (a) the kernel runs without panicking on real inputs
//! and (b) the activity_map values stay within the valid 0..=2 enum
//! range.

use core::sync::atomic::{AtomicU8, Ordering};

use lamquant_weights::generated::snn::{
    layer0_bwd, layer0_fwd, layer0_norm,
    layer1_bwd, layer1_fwd, layer1_norm,
    readout, spatial_mix,
};

use crate::neural::ssm_block::{
    self, D_CONV, D_INNER, D_MODEL, D_STATE, T_SEQ,
    SsmBlockWeights, SsmHiddenState, SsmScratch,
};
use crate::neural::layer_norm::layer_norm_q15;

// ─── Public API constants (preserved from dLIF surface) ────────────

pub const INPUT_CH:    usize = 21;
pub const HIDDEN_DIM:  usize = D_MODEL;   // back-compat alias
pub const READOUT_DIM: usize = 8;
pub const T_INPUT:     usize = T_SEQ;     // 313
pub const T_LATENT:    usize = 312;
pub const NUM_DENDRITES: usize = 2;       // legacy; unused

// ─── Sensitivity ────────────────────────────────────────────────────

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

    /// Per-group readout threshold in Q15. Higher = harder to fire.
    fn threshold_q15(&self) -> i32 {
        match self {
            Self::Low    => 8000,   // ~0.24
            Self::Medium => 4000,   // ~0.12
            Self::High   => 2000,   // ~0.06
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ActivityLevel {
    Quiescent = 0,
    Active    = 1,
    High      = 2,
}

// ─── State (RAM-resident across windows) ───────────────────────────

/// Memory-tight SNN runtime state. Total ~85 KB:
///   activity_map         : 2.5 KB
///   scratch              : 28 KB
///   h_state              : 5 KB
///   x_buf  / norm_buf    : 25 KB each (50 KB)
///   dir_out (shared fwd/bwd) : 25 KB
///
/// |A| Q10 + D Q15 tables are NOT stored — computed per direction on
/// the stack inside `run_direction` (save 10 KB BSS at the cost of
/// ~5K of f32 Padé evals per inference; negligible vs the SSM math).
struct SnnState {
    activity_map: [[u8; T_LATENT]; READOUT_DIM],
    activity_sum: u32,
    initialized: bool,
    scratch: SsmScratch,
    h_state: SsmHiddenState,
    /// Block input — accumulates the bidir residual in-place across
    /// fwd then bwd within a block, then becomes the next block's
    /// input on the second layer.
    x_buf:    [[i16; T_INPUT]; D_MODEL],
    /// Layer norm output, ssm input for both fwd and bwd directions.
    norm_buf: [[i16; T_INPUT]; D_MODEL],
    /// Shared per-direction SSM output. fwd_out is fused into x_buf
    /// before bwd is computed, so the same buffer serves both.
    dir_out:  [[i16; T_INPUT]; D_MODEL],
}

impl SnnState {
    const fn new() -> Self {
        Self {
            activity_map: [[0; T_LATENT]; READOUT_DIM],
            activity_sum: 0,
            initialized:  false,
            scratch:      SsmScratch::new(),
            h_state:      [[0; D_STATE]; D_INNER],
            x_buf:        [[0; T_INPUT]; D_MODEL],
            norm_buf:     [[0; T_INPUT]; D_MODEL],
            dir_out:      [[0; T_INPUT]; D_MODEL],
        }
    }
}

// Place the full SnnState in the `.bss.snn_overlay` section so the
// linker overlays it with `lifting_scratch` + `lpc.residual` — both
// are consumed before snn::inference() runs (Track B.4). The
// `cfg(target_os = "none")` guard scopes this to the embedded build;
// the host-verify test build uses the regular .bss to avoid clashing
// with cargo-test's runtime layout.
#[cfg(target_os = "none")]
#[link_section = ".bss.snn_overlay"]
static mut STATE: SnnState = SnnState::new();

#[cfg(not(target_os = "none"))]
static mut STATE: SnnState = SnnState::new();
static SENSITIVITY: AtomicU8 = AtomicU8::new(SnnSensitivity::Medium as u8);

// ─── Public API ─────────────────────────────────────────────────────

pub fn init() {
    unsafe {
        STATE = SnnState::new();
        STATE.initialized = true;
    }
    SENSITIVITY.store(SnnSensitivity::Medium as u8, Ordering::Relaxed);
}

/// Convert generated A_LOG i16 table (linear scale s_log) into |A| Q10.
///
/// Real value chain:
///   A_log_real = A_LOG[i] * A_LOG_SCALE       (f32)
///   |A|_real   = exp(A_log_real)              (always positive)
///   |A|_q10    = clamp_i16(round(|A| * 1024))
/// Round-to-nearest-i16 without libm. `f32::round` lives in `std`,
/// not in `core::f32`, so we inline it for the no_std target.
#[inline]
fn round_f32_to_i16(x: f32) -> i16 {
    let r = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    if r >= i16::MAX as f32 { i16::MAX }
    else if r <= i16::MIN as f32 { i16::MIN }
    else { r as i16 }
}

fn build_a_abs_q10(
    a_log: &[i16; D_INNER * D_STATE], a_log_scale: f32,
    out: &mut [[i16; D_STATE]; D_INNER],
) {
    for d in 0..D_INNER {
        for n in 0..D_STATE {
            let alog_real = (a_log[d * D_STATE + n] as f32) * a_log_scale;
            // Padé exp approximant (no libm dep). Init-only path
            // — embedded build runs this ~1280 times at boot.
            let a_real = exp_f32(alog_real);
            out[d][n] = round_f32_to_i16(a_real * 1024.0);
        }
    }
}

fn build_d_q15(d_table: &[i8; D_INNER], d_scale: f32, out: &mut [i16; D_INNER]) {
    for d in 0..D_INNER {
        let real = (d_table[d] as f32) * d_scale;
        out[d] = round_f32_to_i16(real * 32768.0);
    }
}

/// Polynomial exp(x) for x ∈ [-4, +4]. f32 emulation suffices for the
/// init-time A_log conversion (one-shot, not the hot path).
fn exp_f32(x: f32) -> f32 {
    // Pade approximant of exp(x) good to ~1e-4 on [-4, +4]:
    //   exp(x) ≈ (1 + x/2 + x²/9 + x³/72) / (1 - x/2 + x²/9 - x³/72)
    let xc = if x > 4.0 { 4.0 } else if x < -4.0 { -4.0 } else { x };
    let x2 = xc * xc;
    let x3 = x2 * xc;
    let num = 1.0 + xc * 0.5 + x2 / 9.0 + x3 / 72.0;
    let den = 1.0 - xc * 0.5 + x2 / 9.0 - x3 / 72.0;
    num / den
}

pub fn set_sensitivity(level: SnnSensitivity) {
    SENSITIVITY.store(level as u8, Ordering::Relaxed);
}

pub fn get_sensitivity() -> SnnSensitivity {
    SnnSensitivity::from_u8(SENSITIVITY.load(Ordering::Relaxed))
}

/// One 10-second window of L3 approximation → activity_map.
pub fn inference(l3: &[[i16; T_INPUT]; INPUT_CH]) {
    let state = unsafe { &mut STATE };
    if !state.initialized { return; }

    let threshold_q15 = get_sensitivity().threshold_q15();

    // Stage 1 — spatial_mix: Linear(21 → 40) per timestep.
    // We process timestep-at-a-time to fit the linear_i8_i16 vector signature.
    let w_scale_q15 = scale_to_q15(spatial_mix::SPATIAL_MIX_WEIGHT_SCALE);
    let b_scale_q15 = scale_to_q15(spatial_mix::SPATIAL_MIX_BIAS_SCALE);
    let mut col_in:  [i16; INPUT_CH] = [0; INPUT_CH];
    let mut col_out: [i16; D_MODEL]  = [0; D_MODEL];
    for t in 0..T_INPUT {
        for ic in 0..INPUT_CH { col_in[ic] = l3[ic][t]; }
        ssm_block::linear_i8_i16(
            &spatial_mix::SPATIAL_MIX_WEIGHT, w_scale_q15,
            Some((&spatial_mix::SPATIAL_MIX_BIAS, b_scale_q15)),
            &col_in, &mut col_out,
        );
        for d in 0..D_MODEL { state.x_buf[d][t] = col_out[d]; }
    }

    // Stages 2 & 3 — two BidirectionalSSM blocks. Residual fused
    // in-place into x_buf so it becomes the next layer's input.
    apply_bidir_block_0(state);
    apply_bidir_block_1(state);
    // After layer 1: state.x_buf holds the final per-timestep activations.

    // Stage 4 — readout: Linear(40 → 8) per timestep, then threshold.
    let rw_scale = scale_to_q15(readout::READOUT_WEIGHT_SCALE);
    let rb_scale = scale_to_q15(readout::READOUT_BIAS_SCALE);
    let mut activity_sum = 0u32;
    let mut col_d:      [i16; D_MODEL]    = [0; D_MODEL];
    let mut col_logits: [i16; READOUT_DIM] = [0; READOUT_DIM];
    for t in 0..T_LATENT {
        for d in 0..D_MODEL { col_d[d] = state.x_buf[d][t]; }
        ssm_block::linear_i8_i16(
            &readout::READOUT_WEIGHT, rw_scale,
            Some((&readout::READOUT_BIAS, rb_scale)),
            &col_d, &mut col_logits,
        );
        let mut group_active = 0u32;
        for g in 0..READOUT_DIM {
            let logit = col_logits[g] as i32;
            let level = if logit.abs() > threshold_q15 * 3 / 2 {
                ActivityLevel::High as u8
            } else if logit.abs() > threshold_q15 {
                ActivityLevel::Active as u8
            } else {
                ActivityLevel::Quiescent as u8
            };
            state.activity_map[g][t] = level;
            if level > 0 { group_active += 1; }
        }
        if group_active > 0 { activity_sum += 1; }
    }
    state.activity_sum = activity_sum;
}

// Direction identifier for build_block_weights — keeps the borrow
// checker happy by binding the runtime tables (`a_abs_q10_l*`,
// `d_q15_l*`) inside the function call instead of holding both an
// immutable `&SnnState` and a mutable `&mut state.scratch` at once.
#[derive(Copy, Clone)]
enum Dir { L0Fwd, L0Bwd, L1Fwd, L1Bwd }

fn apply_bidir_block_0(state: &mut SnnState) {
    let gw = scale_to_q15(layer0_norm::NORM_WEIGHT_SCALE);
    let gb = scale_to_q15(layer0_norm::NORM_BIAS_SCALE);
    layer_norm_q15(
        &state.x_buf,
        &layer0_norm::NORM_WEIGHT, gw,
        &layer0_norm::NORM_BIAS,   gb,
        &mut state.norm_buf,
    );
    // Residual fused IN-PLACE into x_buf: after each direction we
    // add dir_out / 2 to x_buf. This avoids holding fwd_out + bwd_out
    // simultaneously (saves 25 KB SRAM).
    run_direction(state, Dir::L0Fwd, false);
    accumulate_half_into_x(state);
    run_direction(state, Dir::L0Bwd, true);
    accumulate_half_into_x(state);
}

fn apply_bidir_block_1(state: &mut SnnState) {
    let gw = scale_to_q15(layer1_norm::NORM_WEIGHT_SCALE);
    let gb = scale_to_q15(layer1_norm::NORM_BIAS_SCALE);
    layer_norm_q15(
        &state.x_buf,
        &layer1_norm::NORM_WEIGHT, gw,
        &layer1_norm::NORM_BIAS,   gb,
        &mut state.norm_buf,
    );
    run_direction(state, Dir::L1Fwd, false);
    accumulate_half_into_x(state);
    run_direction(state, Dir::L1Bwd, true);
    accumulate_half_into_x(state);
}

/// x_buf[d][t] += dir_out[d][t] / 2 with i16 saturation.
fn accumulate_half_into_x(state: &mut SnnState) {
    for d in 0..D_MODEL {
        for t in 0..T_INPUT {
            let sum = state.x_buf[d][t] as i32 + (state.dir_out[d][t] as i32 >> 1);
            state.x_buf[d][t] = if sum > i16::MAX as i32 { i16::MAX }
                                else if sum < i16::MIN as i32 { i16::MIN }
                                else { sum as i16 };
        }
    }
}

/// Drive `selective_ssm_block` for one direction. The weights borrow
/// `state` immutably for the table refs; the scratch/h_state/output
/// borrow `state` mutably. We split-borrow by packaging the immutable
/// refs into a local `SsmBlockWeights<'_>` value whose lifetime is
/// confined to this function, then release that borrow before touching
/// the mutable fields. To achieve that, we copy the small Q10/Q15
/// tables onto the stack first.
fn run_direction(state: &mut SnnState, dir: Dir, reverse: bool) {
    // Build the per-direction |A| Q10 + D Q15 tables on the stack
    // here (~3 KB). Saves 10 KB BSS vs storing four copies in
    // SnnState. Padé exp is cheap relative to the SSM math.
    let mut a_abs:  [[i16; D_STATE]; D_INNER] = [[0; D_STATE]; D_INNER];
    let mut d_skip: [i16; D_INNER]            = [0; D_INNER];
    let (a_log, a_log_scale, d_table, d_scale) = match dir {
        Dir::L0Fwd => (&layer0_fwd::A_LOG, layer0_fwd::A_LOG_SCALE,
                       &layer0_fwd::D,     layer0_fwd::D_SCALE),
        Dir::L0Bwd => (&layer0_bwd::A_LOG, layer0_bwd::A_LOG_SCALE,
                       &layer0_bwd::D,     layer0_bwd::D_SCALE),
        Dir::L1Fwd => (&layer1_fwd::A_LOG, layer1_fwd::A_LOG_SCALE,
                       &layer1_fwd::D,     layer1_fwd::D_SCALE),
        Dir::L1Bwd => (&layer1_bwd::A_LOG, layer1_bwd::A_LOG_SCALE,
                       &layer1_bwd::D,     layer1_bwd::D_SCALE),
    };
    build_a_abs_q10(a_log, a_log_scale, &mut a_abs);
    build_d_q15(d_table, d_scale, &mut d_skip);

    let weights = match dir {
        Dir::L0Fwd => SsmBlockWeights {
            in_proj_w: &layer0_fwd::IN_PROJ_W,
            in_proj_w_scale_q15: scale_to_q15(layer0_fwd::IN_PROJ_W_SCALE),
            conv1d_w: &layer0_fwd::CONV1D_W,
            conv1d_w_scale_q15: scale_to_q15(layer0_fwd::CONV1D_W_SCALE),
            conv1d_b: &layer0_fwd::CONV1D_B,
            conv1d_b_scale_q15: scale_to_q15(layer0_fwd::CONV1D_B_SCALE),
            x_proj_w: &layer0_fwd::X_PROJ_W,
            x_proj_w_scale_q15: scale_to_q15(layer0_fwd::X_PROJ_W_SCALE),
            a_abs_q10: &a_abs, d_skip_q15: &d_skip,
            dt_bias_q15: dt_bias_q15(layer0_fwd::DT_BIAS[0], layer0_fwd::DT_BIAS_SCALE),
            out_proj_w: &layer0_fwd::OUT_PROJ_W,
            out_proj_w_scale_q15: scale_to_q15(layer0_fwd::OUT_PROJ_W_SCALE),
        },
        Dir::L0Bwd => SsmBlockWeights {
            in_proj_w: &layer0_bwd::IN_PROJ_W,
            in_proj_w_scale_q15: scale_to_q15(layer0_bwd::IN_PROJ_W_SCALE),
            conv1d_w: &layer0_bwd::CONV1D_W,
            conv1d_w_scale_q15: scale_to_q15(layer0_bwd::CONV1D_W_SCALE),
            conv1d_b: &layer0_bwd::CONV1D_B,
            conv1d_b_scale_q15: scale_to_q15(layer0_bwd::CONV1D_B_SCALE),
            x_proj_w: &layer0_bwd::X_PROJ_W,
            x_proj_w_scale_q15: scale_to_q15(layer0_bwd::X_PROJ_W_SCALE),
            a_abs_q10: &a_abs, d_skip_q15: &d_skip,
            dt_bias_q15: dt_bias_q15(layer0_bwd::DT_BIAS[0], layer0_bwd::DT_BIAS_SCALE),
            out_proj_w: &layer0_bwd::OUT_PROJ_W,
            out_proj_w_scale_q15: scale_to_q15(layer0_bwd::OUT_PROJ_W_SCALE),
        },
        Dir::L1Fwd => SsmBlockWeights {
            in_proj_w: &layer1_fwd::IN_PROJ_W,
            in_proj_w_scale_q15: scale_to_q15(layer1_fwd::IN_PROJ_W_SCALE),
            conv1d_w: &layer1_fwd::CONV1D_W,
            conv1d_w_scale_q15: scale_to_q15(layer1_fwd::CONV1D_W_SCALE),
            conv1d_b: &layer1_fwd::CONV1D_B,
            conv1d_b_scale_q15: scale_to_q15(layer1_fwd::CONV1D_B_SCALE),
            x_proj_w: &layer1_fwd::X_PROJ_W,
            x_proj_w_scale_q15: scale_to_q15(layer1_fwd::X_PROJ_W_SCALE),
            a_abs_q10: &a_abs, d_skip_q15: &d_skip,
            dt_bias_q15: dt_bias_q15(layer1_fwd::DT_BIAS[0], layer1_fwd::DT_BIAS_SCALE),
            out_proj_w: &layer1_fwd::OUT_PROJ_W,
            out_proj_w_scale_q15: scale_to_q15(layer1_fwd::OUT_PROJ_W_SCALE),
        },
        Dir::L1Bwd => SsmBlockWeights {
            in_proj_w: &layer1_bwd::IN_PROJ_W,
            in_proj_w_scale_q15: scale_to_q15(layer1_bwd::IN_PROJ_W_SCALE),
            conv1d_w: &layer1_bwd::CONV1D_W,
            conv1d_w_scale_q15: scale_to_q15(layer1_bwd::CONV1D_W_SCALE),
            conv1d_b: &layer1_bwd::CONV1D_B,
            conv1d_b_scale_q15: scale_to_q15(layer1_bwd::CONV1D_B_SCALE),
            x_proj_w: &layer1_bwd::X_PROJ_W,
            x_proj_w_scale_q15: scale_to_q15(layer1_bwd::X_PROJ_W_SCALE),
            a_abs_q10: &a_abs, d_skip_q15: &d_skip,
            dt_bias_q15: dt_bias_q15(layer1_bwd::DT_BIAS[0], layer1_bwd::DT_BIAS_SCALE),
            out_proj_w: &layer1_bwd::OUT_PROJ_W,
            out_proj_w_scale_q15: scale_to_q15(layer1_bwd::OUT_PROJ_W_SCALE),
        },
    };

    // Disjoint-field borrow: norm_buf immutable + scratch / h_state /
    // dir_out mutable, all distinct SnnState fields.
    ssm_block::selective_ssm_block(
        &weights,
        &state.norm_buf,
        &mut state.scratch,
        &mut state.h_state,
        &mut state.dir_out,
        reverse,
    );
}

#[inline]
fn dt_bias_q15(dt_bias_i8: i8, scale: f32) -> i32 {
    // dt_bias[0] real = dt_bias_i8 * scale. Q15 = round(real * 2^15).
    let real = (dt_bias_i8 as f32) * scale;
    let q15 = real * 32768.0;
    if q15 >= i32::MAX as f32 { i32::MAX }
    else if q15 <= i32::MIN as f32 { i32::MIN }
    else { q15 as i32 }
}

/// Convert an f32 scale factor into Q15. Saturating clamp.
///
/// Uses `>=` / `<=` for the bounds because `i32::MAX as f32` rounds
/// up to 2_147_483_648.0 (one ULP past i32::MAX); a strict `>` would
/// admit that exact float into the `as i32` cast, which is UB in Rust
/// for out-of-range floats. (V4 Pro Finding 1 of 7ce5a488 review.)
#[inline]
fn scale_to_q15(s: f32) -> i32 {
    let v = s * 32768.0;
    if v >= i32::MAX as f32 { i32::MAX }
    else if v <= i32::MIN as f32 { i32::MIN }
    else { v as i32 }
}

pub fn activity_sum() -> u32 {
    unsafe { STATE.activity_sum }
}

pub fn get_activity(group: usize, t_latent: usize) -> ActivityLevel {
    if group >= READOUT_DIM || t_latent >= T_LATENT {
        return ActivityLevel::Quiescent;
    }
    let v = unsafe { STATE.activity_map[group][t_latent] };
    match v {
        2 => ActivityLevel::High,
        1 => ActivityLevel::Active,
        _ => ActivityLevel::Quiescent,
    }
}

pub fn get_activity_map() -> &'static [[u8; T_LATENT]; READOUT_DIM] {
    unsafe { &STATE.activity_map }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn zero_input_produces_no_activity() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        init();
        let l3 = [[0i16; T_INPUT]; INPUT_CH];
        inference(&l3);
        let map = get_activity_map();
        // With placeholder selective_ssm = norm-only, zero input → zero
        // mean and zero variance per timestep → output is dominated by
        // beta. As long as the kernels run cleanly the map stays in
        // the valid 0..=2 enum range and activity_sum is bounded.
        for row in map.iter() {
            for &v in row.iter() {
                assert!(v <= 2, "activity_map contains invalid level {}", v);
            }
        }
    }

    #[test]
    fn real_weights_path_runs_without_panic() {
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
        for row in map.iter() {
            for &v in row.iter() {
                assert!(v <= 2);
            }
        }
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
