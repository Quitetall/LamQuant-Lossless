//! FSQ (Finite Scalar Quantization) — quantize latent dimensions to a
//! discrete grid, then encode 4-tuple as a single flat index.
//!
//! Adaptive scaling: tracks rolling RMS of the input EEG and adjusts the
//! Q31 projection scale to prevent dead codes (weak signal) and clipping
//! (strong signal). The "50" threshold matches the silicon shackle clamp
//! range (±50 in signal units).

use core::sync::atomic::{AtomicI32, Ordering};

/// Default Q31 projection scale. Wired from the trained model via
/// `firmware_export.py` in Phase 5; for now use a placeholder.
const DEFAULT_FSQ_QUANT_SCALE_Q31: i32 = 0x4000_0000; // 0.5 in Q31

/// FSQ levels per dimension (Gen 7.7: production preset).
/// Indexes into a 4D lattice. Phase 5 will expose runtime preset switching.
pub const FSQ_LEVELS: [i32; 4] = [8, 16, 16, 32];

/// Adaptive scale, rebuilt every window from RMS.
static ADAPTIVE_FSQ_SCALE: AtomicI32 = AtomicI32::new(DEFAULT_FSQ_QUANT_SCALE_Q31);

/// Q31 multiply: `(a * b) >> 31`, signed.
#[inline(always)]
fn mul_q31(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 31) as i32
}

/// Integer square root via shift-and-subtract. O(16) iterations for u32.
/// Returns `floor(sqrt(x))`.
fn isqrt32(mut x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    let mut result = 0u32;
    let mut bit = 1u32 << 30; // highest power of 4

    while bit > x {
        bit >>= 2;
    }
    while bit != 0 {
        if x >= result + bit {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

/// Update adaptive FSQ gain from rolling RMS of the current window.
///
/// Called once per window with the Q31 ADC buffer. Adjusts the projection
/// scale so weak signals fill bins and strong signals don't clip.
pub fn update_adaptive_gain(eeg_buffer: &[i32]) {
    if eeg_buffer.is_empty() {
        return;
    }

    let mut sum_sq: i64 = 0;
    for &val in eeg_buffer {
        sum_sq += (val as i64) * (val as i64);
    }

    let mean_sq = (sum_sq / eeg_buffer.len() as i64) as u32;
    let rms = isqrt32(mean_sq);

    let new_scale = if rms > 50 {
        DEFAULT_FSQ_QUANT_SCALE_Q31 / ((rms / 50) + 1) as i32
    } else {
        DEFAULT_FSQ_QUANT_SCALE_Q31
    };

    ADAPTIVE_FSQ_SCALE.store(new_scale, Ordering::Relaxed);
}

/// Quantize a single Q31 activation to the nearest FSQ grid point.
///
/// Maps continuous Q31 value to integer in `[-levels/2, +levels/2]`.
#[inline]
fn quantize_scalar(val: i32, levels: i32) -> i32 {
    let half_bound = levels >> 1;
    let scale = ADAPTIVE_FSQ_SCALE.load(Ordering::Relaxed);
    let grid_estimate = mul_q31(val, scale);
    grid_estimate.clamp(-half_bound, half_bound)
}

/// Run FSQ on a 4-element activation vector. Returns the flat lattice index.
///
/// Mixed-radix encoding: index = Σᵢ (q_val[i] · stride[i])
/// where stride[i] = ∏_{j<i} FSQ_LEVELS[j].
///
/// The flat index uniquely identifies one point in the 4D FSQ lattice and
/// is the input to the entropy coder (rANS context-adaptive table lookup).
pub fn run_translation(network_activations_4d: &[i32; 4]) -> u32 {
    let mut single_index: u32 = 0;
    let mut implicit_stride: u32 = 1;

    for i in 0..4 {
        let q_val = quantize_scalar(network_activations_4d[i], FSQ_LEVELS[i]);
        // Shift from [-half, +half] to [0, levels-1].
        let base_index = q_val + (FSQ_LEVELS[i] >> 1);
        single_index += (base_index as u32).wrapping_mul(implicit_stride);
        implicit_stride = implicit_stride.wrapping_mul(FSQ_LEVELS[i] as u32);
    }
    single_index
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn isqrt32_basics() {
        assert_eq!(isqrt32(0), 0);
        assert_eq!(isqrt32(1), 1);
        assert_eq!(isqrt32(4), 2);
        assert_eq!(isqrt32(9), 3);
        assert_eq!(isqrt32(15), 3); // floor
        assert_eq!(isqrt32(100), 10);
        assert_eq!(isqrt32(10000), 100);
    }

    #[test]
    fn quantize_clamps_to_grid_bounds() {
        // Max activation should clamp to +levels/2.
        let q = quantize_scalar(i32::MAX, 8);
        assert!(q.abs() <= 4, "out of grid range: {q}");
    }

    #[test]
    fn translation_returns_index_in_lattice_size() {
        let lattice_size: u32 = FSQ_LEVELS.iter().product::<i32>() as u32;
        let zeros = [0i32; 4];
        let idx = run_translation(&zeros);
        assert!(idx < lattice_size, "FSQ index {idx} >= {lattice_size}");
    }
}
