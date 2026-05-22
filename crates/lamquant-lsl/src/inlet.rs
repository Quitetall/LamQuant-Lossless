//! Sync LSL inlet — Phase 3.
//!
//! World-class LSL → `.lml` bridge. Subscribes to a discoverable
//! LSL stream by name (or stream descriptor), accumulates samples
//! into window-sized chunks the LamQuant codec can consume, and
//! flushes them through `lamquant_core::container::write_file` to
//! produce a `.lml` archive on disk.
//!
//! Design choices:
//!
//!   * **Sample buffering**: the codec wants windows of N samples
//!     across n_channels. LSL delivers samples one at a time (or
//!     in small chunks). The `SampleBuffer` accumulates samples
//!     until a full window's worth is ready, then hands the
//!     transposed `[n_channels][window_size] i64` matrix to the
//!     codec.
//!
//!   * **Timestamp metadata**: LSL timestamps + host wall-clock
//!     are captured on every sample. The host-wall-clock anchor
//!     of the first sample becomes the `.lml` metadata's
//!     `startdate` field; per-sample LSL timestamps go into the
//!     metadata JSON (Phase 3.x — `startdate` + offset for now).
//!
//!   * **Channel format mapping**: LSL's `cf_int32` ⇒ codec `i64`
//!     (zero-extended), `cf_float32` ⇒ rejected for Phase 3 (the
//!     lossless codec is integer-only).
//!
//! Phase 3 lands the offline pieces (`SampleBuffer` + transpose
//! helpers + the codec-flush plumbing). The actual LSL subscription
//! (`Inlet::subscribe`) is gated behind the `liblsl` feature and
//! wired in Phase 4 when CLI integration arrives.

use lamquant_core::lpc::LpcMode;

/// Sample buffer that batches per-sample observations into the
/// codec's window-sized matrix.
///
/// The buffer is shape `[window_size][n_channels] i64`, i.e. one
/// row per sample. When `push_sample` fills the last row, the
/// buffer transposes to `[n_channels][window_size] i64` and yields
/// it via `flush_if_ready`.
///
/// Buffering this way (sample-major then transposed on flush) keeps
/// the LSL push path branch-free: every `push_sample` does the
/// same fixed amount of work regardless of buffer occupancy.
pub struct SampleBuffer {
    /// `[row=sample_index][col=channel_index] = i64`. Length grows
    /// as samples arrive; capped at `window_size`.
    rows: Vec<Vec<i64>>,
    n_channels: usize,
    window_size: usize,
}

impl SampleBuffer {
    /// Build a buffer for the given channel count + window size.
    /// Both must be > 0; panicking would be wrong (LSL stream
    /// shapes are caller-supplied), so the constructor returns a
    /// Result.
    pub fn new(n_channels: usize, window_size: usize) -> Result<Self, &'static str> {
        if n_channels == 0 {
            return Err("SampleBuffer: n_channels must be > 0");
        }
        if window_size == 0 {
            return Err("SampleBuffer: window_size must be > 0");
        }
        Ok(Self {
            rows: Vec::with_capacity(window_size),
            n_channels,
            window_size,
        })
    }

    /// Push one sample-worth of channel readings. Returns
    /// `Err` if the sample width doesn't match the configured
    /// `n_channels` (an LSL stream shape change mid-recording).
    pub fn push_sample(&mut self, sample: &[i32]) -> Result<(), &'static str> {
        if sample.len() != self.n_channels {
            return Err("SampleBuffer: sample width mismatch");
        }
        if self.rows.len() == self.window_size {
            return Err(
                "SampleBuffer: window already full; call flush_if_ready before pushing more",
            );
        }
        // Widen i32 → i64 for the codec's signal API.
        let row: Vec<i64> = sample.iter().map(|&v| v as i64).collect();
        self.rows.push(row);
        Ok(())
    }

    /// If a full window's worth of samples is buffered, transpose
    /// into `[n_channels][window_size] i64` and return it, leaving
    /// the buffer empty for the next window.
    pub fn flush_if_ready(&mut self) -> Option<Vec<Vec<i64>>> {
        if self.rows.len() < self.window_size {
            return None;
        }
        let n_channels = self.n_channels;
        let window_size = self.window_size;
        let rows = std::mem::replace(
            &mut self.rows,
            Vec::with_capacity(window_size),
        );
        // Transpose: row[i][ch] → out[ch][i].
        let mut out: Vec<Vec<i64>> = (0..n_channels)
            .map(|_| Vec::with_capacity(window_size))
            .collect();
        for row in rows {
            for (ch, v) in row.into_iter().enumerate() {
                out[ch].push(v);
            }
        }
        Some(out)
    }

    /// Force-drain whatever's buffered, even if the buffer isn't
    /// full. The trailing partial window is padded with zeros so
    /// the codec sees a uniform window-size everywhere. Caller is
    /// responsible for recording the actual sample count in the
    /// container metadata so the decoder trims the padding.
    pub fn drain_padded(&mut self) -> Option<(Vec<Vec<i64>>, usize)> {
        if self.rows.is_empty() {
            return None;
        }
        let actual_samples = self.rows.len();
        // Pad rows up to window_size with zeros.
        while self.rows.len() < self.window_size {
            self.rows.push(vec![0i64; self.n_channels]);
        }
        let drained = self.flush_if_ready()?;
        Some((drained, actual_samples))
    }

    /// Current buffer occupancy.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True if no samples are buffered.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Encoding mode for the LSL → `.lml` flush.
#[derive(Debug, Clone, Copy)]
pub struct InletEncodeOpts {
    /// LPC mode passed through to the codec.
    pub lpc_mode: LpcMode,
    /// Codec `noise_bits` — 0 = strictly lossless. Match
    /// `lml::compress_with_mode`'s convention.
    pub noise_bits: u8,
}

impl Default for InletEncodeOpts {
    fn default() -> Self {
        Self {
            lpc_mode: LpcMode::default(),
            noise_bits: 0,
        }
    }
}

/// Encode a single window's worth of samples through the codec.
/// Returns the LML packet bytes (one per window). Caller stitches
/// these into a container via `lamquant_core::container::write_file`
/// once the full recording is captured.
pub fn encode_window(
    window: &[Vec<i64>],
    opts: InletEncodeOpts,
) -> Result<Vec<u8>, lamquant_core::error::LmlError> {
    lamquant_core::lml::compress_with_mode(window, opts.noise_bits, opts.lpc_mode)
}

// ─── Live LSL inlet (`liblsl` feature) ────────────────────────────

#[cfg(feature = "liblsl")]
pub use live::{Inlet, RecordSession};

#[cfg(feature = "liblsl")]
mod live {
    use super::*;
    use crate::error::LslIntegrationError;
    use lsl::Pullable;

    /// Live LSL inlet wrapping `lsl::StreamInlet`.
    ///
    /// Resolves a stream by name (or predicate string), pulls
    /// samples in a tight loop, and stages them through
    /// `SampleBuffer` ready for codec flushes. The actual
    /// `.lml`-writing record loop lives in `RecordSession` so the
    /// inlet itself stays focused on the LSL → samples translation.
    pub struct Inlet {
        inner: lsl::StreamInlet,
        n_channels: usize,
        nominal_srate: f64,
        channel_format: lsl::ChannelFormat,
    }

    impl Inlet {
        /// Resolve a stream by name + open an inlet. Blocks up to
        /// `timeout_sec` for the stream to appear on the network;
        /// pass `lsl::FOREVER` to wait indefinitely. Picks the first
        /// matching stream when multiple advertise the same name —
        /// callers needing finer control should use
        /// [`Self::from_info`] with a hand-built `StreamInfo`
        /// from `lsl::resolve_byprop`.
        pub fn resolve_by_name(
            name: &str,
            timeout_sec: f64,
        ) -> Result<Self, LslIntegrationError> {
            let predicate = format!("name='{}'", name);
            let streams = lsl::resolve_bypred(&predicate, 1, timeout_sec)
                .map_err(LslIntegrationError::Lsl)?;
            let info = streams.into_iter().next().ok_or_else(|| {
                LslIntegrationError::Other(format!(
                    "no LSL stream matched name='{}' within {} s",
                    name, timeout_sec
                ))
            })?;
            Self::from_info(info)
        }

        /// Open an inlet against a pre-resolved StreamInfo. Use
        /// when the caller has already discovered + filtered the
        /// streams (e.g. via `lsl::resolve_byprop` with a richer
        /// predicate, or a UI source picker).
        ///
        /// After opening the inlet, this re-fetches the full
        /// StreamInfo via `inlet.info()` — `resolve_bypred`'s
        /// discovery beacon may not carry the publisher's
        /// channel_format, but the post-connect info does. Without
        /// this re-fetch our channel_format dispatch routes to the
        /// wrong arm.
        pub fn from_info(info: lsl::StreamInfo) -> Result<Self, LslIntegrationError> {
            let inner = lsl::StreamInlet::new(
                &info, /* max_buflen = */ 360, /* max_chunklen = */ 0,
                /* recover = */ true,
            )
            .map_err(LslIntegrationError::Lsl)?;
            // Re-fetch full info from the publisher.
            let full = inner
                .info(lsl::FOREVER)
                .map_err(LslIntegrationError::Lsl)?;
            let n_channels = full.channel_count() as usize;
            let nominal_srate = full.nominal_srate();
            let channel_format = full.channel_format();
            Ok(Self {
                inner,
                n_channels,
                nominal_srate,
                channel_format,
            })
        }

        /// Pull one sample (all channels) with the given timeout
        /// (`lsl::FOREVER` to block). Returns the i32-cast samples
        /// + the LSL timestamp (microseconds since LSL epoch).
        /// Non-int32 streams are cast through f64 then truncated to
        /// i32 — lossless for cf_int8/int16/int32, lossy for
        /// cf_float32/double64 (a future codec mode could preserve
        /// floats once the lossless side handles them).
        pub fn pull_sample(&self, timeout_sec: f64) -> Result<(Vec<i32>, f64), LslIntegrationError> {
            match self.channel_format {
                lsl::ChannelFormat::Int8 | lsl::ChannelFormat::Int16
                | lsl::ChannelFormat::Int32 => {
                    let (sample, ts): (Vec<i32>, f64) = self
                        .inner
                        .pull_sample(timeout_sec)
                        .map_err(LslIntegrationError::Lsl)?;
                    Ok((sample, ts))
                }
                lsl::ChannelFormat::Int64 => {
                    let (sample, ts): (Vec<i64>, f64) = self
                        .inner
                        .pull_sample(timeout_sec)
                        .map_err(LslIntegrationError::Lsl)?;
                    let casted = sample
                        .into_iter()
                        .map(|v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                        .collect();
                    Ok((casted, ts))
                }
                lsl::ChannelFormat::Float32 | lsl::ChannelFormat::Double64
                | lsl::ChannelFormat::String | lsl::ChannelFormat::Undefined => {
                    Err(LslIntegrationError::Other(format!(
                        "channel_format {:?} not yet supported by the lossless inlet",
                        self.channel_format
                    )))
                }
            }
        }

        /// Channel count reported by the source.
        pub fn channel_count(&self) -> usize {
            self.n_channels
        }

        /// Sample rate reported by the source.
        pub fn nominal_srate(&self) -> f64 {
            self.nominal_srate
        }
    }

    /// Live recording session: subscribes to an LSL stream + flushes
    /// codec-window-sized chunks of samples into a `.lml` container
    /// on disk. Building block for the upcoming `lml record` CLI.
    pub struct RecordSession {
        inlet: Inlet,
        buffer: SampleBuffer,
        encoded_windows: Vec<Vec<u8>>,
        opts: InletEncodeOpts,
    }

    impl RecordSession {
        pub fn new(inlet: Inlet, window_size: usize, opts: InletEncodeOpts) -> Result<Self, LslIntegrationError> {
            let n_ch = inlet.channel_count();
            let buffer = SampleBuffer::new(n_ch, window_size)
                .map_err(|e| LslIntegrationError::Other(e.to_string()))?;
            Ok(Self {
                inlet,
                buffer,
                encoded_windows: Vec::new(),
                opts,
            })
        }

        /// Capture up to `max_samples` from the network, blocking
        /// forever per `pull_sample`. Use [`Self::capture_timeout`]
        /// for bounded-wait variants. Returns the number of
        /// windows successfully encoded + flushed during this call.
        pub fn capture(&mut self, max_samples: usize) -> Result<usize, LslIntegrationError> {
            self.capture_timeout(max_samples, lsl::FOREVER)
        }

        /// Capture up to `max_samples` with a per-sample timeout
        /// (seconds; `0.0` = immediate non-blocking poll;
        /// `lsl::FOREVER` = block indefinitely). Stops early
        /// when a `pull_sample` times out (e.g. the publisher
        /// disconnected mid-record) and returns the windows
        /// encoded so far. Use this in interactive UIs / event
        /// loops where blocking forever is unacceptable.
        pub fn capture_timeout(
            &mut self,
            max_samples: usize,
            per_sample_timeout_sec: f64,
        ) -> Result<usize, LslIntegrationError> {
            let mut windows = 0usize;
            for _ in 0..max_samples {
                let (sample, _ts) = self.inlet.pull_sample(per_sample_timeout_sec)?;
                // liblsl signals timeout by returning an empty
                // sample vec (not an `Err`). Treat that as "no
                // sample available within the timeout" + drain
                // the buffer + bail. Caller can resume by calling
                // capture_timeout again when more samples may
                // arrive.
                if sample.is_empty() {
                    break;
                }
                self.buffer
                    .push_sample(&sample)
                    .map_err(|e| LslIntegrationError::Other(e.to_string()))?;
                if let Some(window) = self.buffer.flush_if_ready() {
                    let bytes = encode_window(&window, self.opts)
                        .map_err(LslIntegrationError::LmlDecode)?;
                    self.encoded_windows.push(bytes);
                    windows += 1;
                }
            }
            Ok(windows)
        }

        /// Drain the trailing partial window (padded with zeros)
        /// and return all encoded window bytes. After this, the
        /// session is empty + can't be reused.
        pub fn finish(mut self) -> Result<Vec<Vec<u8>>, LslIntegrationError> {
            if let Some((window, _actual)) = self.buffer.drain_padded() {
                let bytes = encode_window(&window, self.opts)
                    .map_err(LslIntegrationError::LmlDecode)?;
                self.encoded_windows.push(bytes);
            }
            Ok(self.encoded_windows)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_new_validates_dimensions() {
        assert!(SampleBuffer::new(0, 100).is_err());
        assert!(SampleBuffer::new(4, 0).is_err());
        assert!(SampleBuffer::new(4, 100).is_ok());
    }

    #[test]
    fn buffer_push_validates_sample_width() {
        let mut buf = SampleBuffer::new(3, 4).unwrap();
        assert!(buf.push_sample(&[1, 2, 3]).is_ok());
        assert!(buf.push_sample(&[1, 2]).is_err()); // wrong width
        assert!(buf.push_sample(&[1, 2, 3, 4]).is_err()); // wrong width
    }

    #[test]
    fn buffer_flush_only_when_full() {
        let mut buf = SampleBuffer::new(2, 4).unwrap();
        buf.push_sample(&[1, 10]).unwrap();
        buf.push_sample(&[2, 20]).unwrap();
        assert!(buf.flush_if_ready().is_none());
        buf.push_sample(&[3, 30]).unwrap();
        buf.push_sample(&[4, 40]).unwrap();
        let flushed = buf.flush_if_ready().expect("should flush");
        // Transposed: channel-major.
        assert_eq!(flushed, vec![vec![1, 2, 3, 4], vec![10, 20, 30, 40]]);
        // Buffer is empty after flush.
        assert!(buf.is_empty());
    }

    #[test]
    fn buffer_drain_padded() {
        let mut buf = SampleBuffer::new(2, 4).unwrap();
        buf.push_sample(&[1, 10]).unwrap();
        buf.push_sample(&[2, 20]).unwrap();
        let (drained, actual) = buf.drain_padded().expect("should drain");
        assert_eq!(actual, 2);
        assert_eq!(drained, vec![vec![1, 2, 0, 0], vec![10, 20, 0, 0]]);
    }

    #[test]
    fn buffer_drain_empty() {
        let mut buf = SampleBuffer::new(2, 4).unwrap();
        assert!(buf.drain_padded().is_none());
    }

    #[test]
    fn buffer_full_refuses_more() {
        let mut buf = SampleBuffer::new(1, 2).unwrap();
        buf.push_sample(&[1]).unwrap();
        buf.push_sample(&[2]).unwrap();
        assert!(buf.push_sample(&[3]).is_err());
    }

    #[test]
    fn encode_window_roundtrip() {
        // Simple deterministic AR(1)-ish signal so the codec has
        // structure to compress.
        let window: Vec<Vec<i64>> = (0..2)
            .map(|ch| {
                (0..32i64)
                    .map(|t| t + ch as i64 * 100)
                    .collect()
            })
            .collect();
        let encoded = encode_window(&window, InletEncodeOpts::default())
            .expect("encode");
        // Decompress and verify bit-exact roundtrip.
        let decoded = lamquant_core::lml::decompress(&encoded).expect("decode");
        assert_eq!(decoded, window);
    }
}
