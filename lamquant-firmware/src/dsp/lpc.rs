//! LPC (Linear Predictive Coding) wrapper — order-8, 21 channels.
//!
//! Stage 2 of the pipeline: removes temporal redundancy after biquad,
//! before lifting DWT. Coefficients estimated from first 256 samples
//! (EEG spectral envelope changes slowly within a 10-second window).
//!
//! Implementation reuses `lamquant_core::lpc::analyze/synthesize`. The
//! firmware uses Q31 `i32` samples; lamquant-core uses `i64`. We convert
//! at the boundary. The Q27 fixed-point coefficient format is preserved.

use alloc::vec::Vec;
use lamquant_core::lpc;

pub const LPC_ORDER: usize = 8;
pub const AUTOCORR_LEN: usize = 256;

use super::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};

/// Per-channel LPC analysis output: order-8 Q27 coefficients + residual.
pub struct LpcOutput {
    pub coeffs: [[i32; LPC_ORDER]; NUM_CHANNELS],
    pub residual: [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
}

impl LpcOutput {
    pub const fn zeroed() -> Self {
        Self {
            coeffs: [[0; LPC_ORDER]; NUM_CHANNELS],
            residual: [[0; WINDOW_SAMPLES]; NUM_CHANNELS],
        }
    }
}

/// Run LPC analysis on the 21-channel HP-filtered buffer.
///
/// `signal[ch]` is Q31 i32. Output residual matches input dtype.
/// Coefficients are Q27 i32 in `out.coeffs[ch][0..LPC_ORDER]`.
pub fn analyze_all_channels(
    signal: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    out: &mut LpcOutput,
) {
    // lamquant-core works on i64. Convert per-channel (one transient Vec
    // per channel — bump allocator handles it within window budget).
    for ch in 0..NUM_CHANNELS {
        let signal_i64: Vec<i64> = signal[ch].iter().map(|&v| v as i64).collect();
        let (coeffs, residual) = lpc::analyze(&signal_i64, LPC_ORDER, AUTOCORR_LEN);

        // Copy coefficients into the per-channel slot.
        for k in 0..LPC_ORDER {
            out.coeffs[ch][k] = coeffs.get(k).copied().unwrap_or(0);
        }

        // Copy residual back into i32 (will fit — Q31 in, Q31 out).
        for (i, &v) in residual.iter().enumerate().take(WINDOW_SAMPLES) {
            out.residual[ch][i] = v as i32;
        }
    }
}

/// Inverse: reconstruct signal from residual + coefficients (decoder side).
pub fn synthesize_all_channels(
    residual: &[[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
    coeffs: &[[i32; LPC_ORDER]; NUM_CHANNELS],
    out: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
) {
    for ch in 0..NUM_CHANNELS {
        let resid_i64: Vec<i64> = residual[ch].iter().map(|&v| v as i64).collect();
        let signal = lpc::synthesize(&resid_i64, &coeffs[ch], LPC_ORDER, AUTOCORR_LEN);
        for (i, &v) in signal.iter().enumerate().take(WINDOW_SAMPLES) {
            out[ch][i] = v as i32;
        }
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_sin_wave() {
        // Synthetic correlated signal (slow sine).
        let mut signal = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                let phase = (i as f64) * 0.1 + (ch as f64) * 0.5;
                signal[ch][i] = ((phase.sin()) * 1_000_000.0) as i32;
            }
        }

        let mut out = LpcOutput::zeroed();
        analyze_all_channels(&signal, &mut out);

        let mut reconstructed = [[0i32; WINDOW_SAMPLES]; NUM_CHANNELS];
        synthesize_all_channels(&out.residual, &out.coeffs, &mut reconstructed);

        // Roundtrip should be exact (integer LPC).
        for ch in 0..NUM_CHANNELS {
            for i in 0..WINDOW_SAMPLES {
                assert_eq!(
                    reconstructed[ch][i], signal[ch][i],
                    "ch{ch} sample {i} mismatch"
                );
            }
        }
    }
}
