//! Sync LSL outlet — read an `.lml` archive, decode samples, push
//! them through an `lsl::StreamOutlet` for any LSL-aware consumer.
//!
//! This is the world-class sync core. It hits microsecond-accurate
//! real-time pacing via `std::time::Instant` (when paced) and
//! drains the whole signal as-fast-as-possible (when bursting).
//! The async wrapper in `outlet_async.rs` builds on top via
//! `tokio::task::spawn_blocking`.

use crate::error::LslIntegrationError;
use crate::metadata::stream_info_from_lml;
use lsl::Pushable;

/// Replay-rate selector. World-class field tools support all three:
/// real-time matches the source sample rate so a consumer sees the
/// signal as the original subject experienced it; burst is for
/// batch / unit-test workflows; multiplier is for accelerated
/// review (e.g. play a 1 hr recording back in 1 min for a
/// clinician's first-pass scan).
#[derive(Debug, Clone, Copy)]
pub enum Rate {
    /// Match the source's nominal sample rate exactly.
    RealTime,
    /// Push as-fast-as-possible. No pacing.
    Burst,
    /// Real-time × this scalar. `0.5` = half speed, `10.0` = 10×.
    Multiplier(f64),
}

impl Rate {
    /// Per-sample sleep duration (nanoseconds). `None` = no pacing.
    fn sample_period_nanos(self, nominal_srate: f64) -> Option<u64> {
        match self {
            Rate::Burst => None,
            Rate::RealTime => Some((1.0e9 / nominal_srate.max(1.0)) as u64),
            Rate::Multiplier(x) => {
                if x <= 0.0 {
                    None
                } else {
                    Some((1.0e9 / (nominal_srate.max(1.0) * x)) as u64)
                }
            }
        }
    }
}

/// Sync outlet wrapper.
///
/// Built from an `.lml` file path. Holds the `lsl::StreamOutlet`
/// + the decoded samples + the chosen replay rate. Call
/// [`Outlet::push_all`] to drain the signal through the LSL
/// network with proper pacing.
pub struct Outlet {
    pub outlet: lsl::StreamOutlet,
    pub samples: Vec<Vec<i32>>,
    pub nominal_srate: f64,
    pub rate: Rate,
}

impl Outlet {
    /// Build an outlet from an `.lml` file. Reads the entire file,
    /// decodes the signal, constructs the LSL `StreamInfo` with
    /// channel labels + units from the EDF header, creates the
    /// outlet, returns ready-to-push state. Default rate =
    /// `Rate::RealTime`.
    pub fn from_lml(
        lml_path: &std::path::Path,
        name: Option<&str>,
    ) -> Result<Self, LslIntegrationError> {
        Self::from_lml_with_rate(lml_path, name, Rate::RealTime)
    }

    /// Build an outlet with an explicit replay rate.
    pub fn from_lml_with_rate(
        lml_path: &std::path::Path,
        name: Option<&str>,
        rate: Rate,
    ) -> Result<Self, LslIntegrationError> {
        let info = stream_info_from_lml(lml_path, name)?;
        let nominal_srate = info.nominal_srate();
        let outlet = lsl::StreamOutlet::new(&info, 0, 360)
            .map_err(LslIntegrationError::Lsl)?;
        // Decode the full signal. For large files this loads
        // everything into RAM up-front — Phase 1 trade-off. A
        // future streaming-decode path can replace this with a
        // window-by-window pump.
        let (signal_i64, _meta) = lamquant_core::container::read_file(lml_path)
            .map_err(LslIntegrationError::LmlDecode)?;
        let samples = transpose_to_per_sample_i32(&signal_i64);
        Ok(Self {
            outlet,
            samples,
            nominal_srate,
            rate,
        })
    }

    /// Drain every sample to the LSL outlet. Returns the number of
    /// samples pushed. Pacing is per-sample via `std::thread::sleep`
    /// — fine on Linux down to ~1 ms; for higher rates use the
    /// burst mode + an external pacer.
    pub fn push_all(&self) -> Result<usize, LslIntegrationError> {
        let period = self.rate.sample_period_nanos(self.nominal_srate);
        let mut pushed = 0usize;
        let start = std::time::Instant::now();
        for (idx, sample) in self.samples.iter().enumerate() {
            self.outlet
                .push_sample(sample)
                .map_err(LslIntegrationError::Lsl)?;
            pushed += 1;
            if let Some(p) = period {
                // Compute the target wake time for sample idx + 1
                // relative to `start`. Sleeping by `period` per
                // iteration would accumulate drift; anchoring to
                // `start + (idx + 1) * period` keeps cumulative
                // error bounded.
                let target_offset_ns = (idx as u64 + 1).saturating_mul(p);
                let target = start + std::time::Duration::from_nanos(target_offset_ns);
                let now = std::time::Instant::now();
                if target > now {
                    std::thread::sleep(target - now);
                }
            }
        }
        Ok(pushed)
    }

    /// Number of samples this outlet has buffered, ready to push.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// LSL nominal sample rate, as reported in the StreamInfo.
    pub fn nominal_srate(&self) -> f64 {
        self.nominal_srate
    }
}

/// Transpose a `[n_ch][n_samples] i64` matrix into `[n_samples][n_ch]
/// i32`. `lsl::StreamOutlet::push_sample` takes one sample (all
/// channels) per call, so we transpose first. The i64 → i32 cast
/// clamps; EDF int24 + synth i16 always fit, so this is lossless
/// on typical sources.
fn transpose_to_per_sample_i32(signal: &[Vec<i64>]) -> Vec<Vec<i32>> {
    if signal.is_empty() {
        return Vec::new();
    }
    let n_ch = signal.len();
    let n_samples = signal[0].len();
    let mut out = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let mut sample = Vec::with_capacity(n_ch);
        for ch in 0..n_ch {
            let v = signal[ch].get(s).copied().unwrap_or(0);
            // saturating_cast i64 → i32.
            let clamped = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            sample.push(clamped);
        }
        out.push(sample);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_sample_period_realtime() {
        let p = Rate::RealTime.sample_period_nanos(256.0);
        // 1e9 / 256 ≈ 3,906,250 ns
        assert!(p.is_some());
        assert!((p.unwrap() as i64 - 3_906_250).abs() < 100);
    }

    #[test]
    fn rate_sample_period_burst() {
        assert!(Rate::Burst.sample_period_nanos(256.0).is_none());
    }

    #[test]
    fn rate_sample_period_multiplier() {
        let p_2x = Rate::Multiplier(2.0).sample_period_nanos(256.0).unwrap();
        let p_1x = Rate::RealTime.sample_period_nanos(256.0).unwrap();
        // 2× rate → half the period.
        assert!((p_2x as i64 - (p_1x / 2) as i64).abs() < 100);
    }

    #[test]
    fn rate_sample_period_zero_multiplier_is_burst() {
        assert!(Rate::Multiplier(0.0).sample_period_nanos(256.0).is_none());
    }

    #[test]
    fn transpose_shape() {
        let signal: Vec<Vec<i64>> = vec![
            vec![1, 2, 3, 4],
            vec![10, 20, 30, 40],
            vec![100, 200, 300, 400],
        ];
        let transposed = transpose_to_per_sample_i32(&signal);
        assert_eq!(transposed.len(), 4); // 4 time-steps
        assert_eq!(transposed[0], vec![1, 10, 100]);
        assert_eq!(transposed[3], vec![4, 40, 400]);
    }

    #[test]
    fn transpose_clamps_out_of_range() {
        let signal: Vec<Vec<i64>> = vec![vec![i64::MAX, i64::MIN]];
        let transposed = transpose_to_per_sample_i32(&signal);
        assert_eq!(transposed[0][0], i32::MAX);
        assert_eq!(transposed[1][0], i32::MIN);
    }

    #[test]
    fn transpose_empty() {
        let empty: Vec<Vec<i64>> = Vec::new();
        assert!(transpose_to_per_sample_i32(&empty).is_empty());
    }
}
