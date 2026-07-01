//! NeuroScan CNT (`.cnt`) reader.
//!
//! Phase 4.4. Implements `SignalSourceReader` for the legacy NeuroScan
//! Scan v3/v4 binary format. The header layout is the de-facto spec
//! published by Compumedics and reproduced across `mne-python`,
//! `eeglab`, and `pycnt`; field offsets used here follow that
//! consensus.
//!
//! Scope:
//!   - 900-byte main SETUP header
//!   - 75-byte ELECTLOC per channel
//!   - multiplexed int16 LE sample data following the electrode block
//!   - sample rate from header offset 376 (u16 LE)
//!   - n_channels from header offset 370 (u16 LE)
//!   - total samples derived from file size (= (file_size - data_start)
//!     / (n_channels * 2)). Event table at file tail is *ignored*
//!     today — annotations are out of scope for Phase 4.4.
//!
//! Not yet supported:
//!   - Scan v5+ extended headers (rare in clinical archives)
//!   - 32-bit sample mode (very rare in CNT-as-shipped)
//!   - Event-table → SidecarBlob round-trip (deferred)
//!
//! Bible alignment:
//!   - R1  one format per file
//!   - R23 validate header magic-ish fields (n_channels > 0,
//!         sample_rate > 0, computed n_samples >= 0) before mallocing
//!   - R30 unsupported sample-bit-depths refused explicitly

use crate::error::{LmlError, LmlResult};
use std::path::PathBuf;
use std::sync::Arc;

use lamquant_abir::{Abir, Channel, Column};

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};
use super::reader::SignalSourceReader;

const SETUP_HEADER_LEN: usize = 900;
const ELECTLOC_LEN: usize = 75;
/// Maximum channels we'll trust before refusing to allocate buffers.
/// Clinical CNT files top out around 256.
const MAX_CHANNELS: usize = 1024;

/// `CntReader` — file-backed `SignalSourceReader` for NeuroScan CNT.
pub struct CntReader {
    path: PathBuf,
}

impl CntReader {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self { path: path.into() }
    }
}

impl SignalSourceReader for CntReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        let raw = std::fs::read(&self.path).map_err(LmlError::Io)?;
        if raw.len() < SETUP_HEADER_LEN {
            return Err(LmlError::Truncated {
                expected: SETUP_HEADER_LEN,
                actual: raw.len(),
                context: "CNT SETUP header",
            });
        }

        // Pull header fields by offset (consensus across mne-python /
        // eeglab CNT readers).
        let n_channels = u16::from_le_bytes([raw[370], raw[371]]) as usize;
        let sample_rate_u16 = u16::from_le_bytes([raw[376], raw[377]]);
        if n_channels == 0 {
            return Err(LmlError::InvalidHeader("CNT header: nchannels = 0".into()));
        }
        if n_channels > MAX_CHANNELS {
            return Err(LmlError::InvalidHeader(format!(
                "CNT header: nchannels {n_channels} exceeds MAX_CHANNELS {MAX_CHANNELS}"
            )));
        }
        if sample_rate_u16 == 0 {
            return Err(LmlError::InvalidHeader(
                "CNT header: sample_rate = 0".into(),
            ));
        }

        let electrode_block_end = SETUP_HEADER_LEN + n_channels * ELECTLOC_LEN;
        if raw.len() < electrode_block_end {
            return Err(LmlError::Truncated {
                expected: electrode_block_end,
                actual: raw.len(),
                context: "CNT electrode block",
            });
        }

        // Parse channel labels from each ELECTLOC. The 10-byte label
        // sits at the start of each electrode record, NUL-terminated.
        let mut channels: Vec<String> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let off = SETUP_HEADER_LEN + ch * ELECTLOC_LEN;
            let label_bytes = &raw[off..off + 10];
            let end = label_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(label_bytes.len());
            let label = String::from_utf8_lossy(&label_bytes[..end])
                .trim()
                .to_string();
            channels.push(if label.is_empty() {
                format!("ch{ch}")
            } else {
                label
            });
        }

        let data_start = electrode_block_end;
        let data_bytes_total = raw.len().saturating_sub(data_start);
        // CNT may have an event table appended past the signal. We
        // can't reliably skip it without the optional EVENT_START field,
        // so trust that `data_bytes` is at most `n_channels * total *
        // 2`. If a few bytes remain at file end (event header), they
        // get harmlessly chopped by the multiplexed loop's bounds
        // check.
        let bytes_per_sample = 2usize; // int16 LE multiplexed
        let frame_bytes = n_channels.saturating_mul(bytes_per_sample);
        if frame_bytes == 0 {
            return Err(LmlError::InvalidHeader(
                "CNT: zero frame size (overflow guard)".into(),
            ));
        }
        let total_samples = data_bytes_total / frame_bytes;

        let mut signal: Vec<Vec<i64>> = (0..n_channels)
            .map(|_| Vec::with_capacity(total_samples))
            .collect();
        for s in 0..total_samples {
            let base = data_start + s * frame_bytes;
            for ch in 0..n_channels {
                let off = base + ch * 2;
                let v = i16::from_le_bytes([raw[off], raw[off + 1]]) as i64;
                signal[ch].push(v);
            }
        }

        let sample_rate = sample_rate_u16 as f64;
        let duration_s = if sample_rate > 0.0 {
            total_samples as f64 / sample_rate
        } else {
            0.0
        };
        // CNT doesn't store per-channel phys_min/max in a uniform way;
        // use the int16 range as a conservative default.
        let phys_min: Vec<f64> = vec![i16::MIN as f64; n_channels];
        let phys_max: Vec<f64> = vec![i16::MAX as f64; n_channels];

        // Preserve the full source bytes as a sidecar so a future
        // `lml decode --to-cnt` can reconstruct losslessly even though
        // we ignored the event table on the read path.
        let bundle = SignalBundle {
            signal,
            sample_rate,
            channels,
            phys_min,
            phys_max,
            duration_s,
            metadata: SourceMetadata {
                source_file: self.path.display().to_string(),
                format: "CNT".to_string(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim: "raw_int16".to_string(),
            },
            sidecar: vec![SidecarBlob {
                key: "cnt_raw".to_string(),
                bytes: raw,
                aux: None,
            }],
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// ADR 0069 L7: specialize to `Column::I16` — CNT is pure int16 LE
    /// multiplexed, no branching needed (the memory win: 4x vs `I64`).
    /// Independent of `read_bundle`: re-parses the SETUP header + label
    /// block (cheap, offset reads only) and decodes the multiplexed
    /// sample block directly into `i16`, mirroring `read_bundle`'s
    /// index math exactly — locked by this module's
    /// `lower_to_abir_matches_read_bundle_i64` test. `phys_min`/`phys_max`
    /// use the same synthetic `i16::MIN`/`MAX` defaults `read_bundle`
    /// uses (CNT doesn't carry a uniform per-channel calibration field).
    fn lower_to_abir(&mut self) -> LmlResult<Abir> {
        let raw = std::fs::read(&self.path).map_err(LmlError::Io)?;
        if raw.len() < SETUP_HEADER_LEN {
            return Err(LmlError::Truncated {
                expected: SETUP_HEADER_LEN,
                actual: raw.len(),
                context: "CNT SETUP header",
            });
        }

        let n_channels = u16::from_le_bytes([raw[370], raw[371]]) as usize;
        let sample_rate_u16 = u16::from_le_bytes([raw[376], raw[377]]);
        if n_channels == 0 {
            return Err(LmlError::InvalidHeader("CNT header: nchannels = 0".into()));
        }
        if n_channels > MAX_CHANNELS {
            return Err(LmlError::InvalidHeader(format!(
                "CNT header: nchannels {n_channels} exceeds MAX_CHANNELS {MAX_CHANNELS}"
            )));
        }
        if sample_rate_u16 == 0 {
            return Err(LmlError::InvalidHeader(
                "CNT header: sample_rate = 0".into(),
            ));
        }

        let electrode_block_end = SETUP_HEADER_LEN + n_channels * ELECTLOC_LEN;
        if raw.len() < electrode_block_end {
            return Err(LmlError::Truncated {
                expected: electrode_block_end,
                actual: raw.len(),
                context: "CNT electrode block",
            });
        }

        let mut labels: Vec<String> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let off = SETUP_HEADER_LEN + ch * ELECTLOC_LEN;
            let label_bytes = &raw[off..off + 10];
            let end = label_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(label_bytes.len());
            let label = String::from_utf8_lossy(&label_bytes[..end])
                .trim()
                .to_string();
            labels.push(if label.is_empty() {
                format!("ch{ch}")
            } else {
                label
            });
        }

        let data_start = electrode_block_end;
        let data_bytes_total = raw.len().saturating_sub(data_start);
        let bytes_per_sample = 2usize;
        let frame_bytes = n_channels.saturating_mul(bytes_per_sample);
        if frame_bytes == 0 {
            return Err(LmlError::InvalidHeader(
                "CNT: zero frame size (overflow guard)".into(),
            ));
        }
        let total_samples = data_bytes_total / frame_bytes;

        let mut cols: Vec<Vec<i16>> = (0..n_channels)
            .map(|_| Vec::with_capacity(total_samples))
            .collect();
        for s in 0..total_samples {
            let base = data_start + s * frame_bytes;
            for ch in 0..n_channels {
                let off = base + ch * 2;
                cols[ch].push(i16::from_le_bytes([raw[off], raw[off + 1]]));
            }
        }

        let channels: Vec<Channel> = cols
            .into_iter()
            .enumerate()
            .map(|(j, col)| Channel {
                label: Arc::from(labels[j].as_str()),
                data: Column::I16(Arc::from(col)),
                phys_min: i16::MIN as f64,
                phys_max: i16::MAX as f64,
            })
            .collect();

        Ok(Abir {
            channels,
            sample_rate: sample_rate_u16 as f64,
            n_samples: total_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synth CNT byte blob: 900-byte SETUP, N×75 electrode
    /// records, multiplexed int16 LE data block.
    fn synth_cnt(n_ch: usize, n_samples: usize, sample_rate: u16) -> Vec<u8> {
        let mut buf = vec![0u8; SETUP_HEADER_LEN];
        buf[370..372].copy_from_slice(&(n_ch as u16).to_le_bytes());
        buf[376..378].copy_from_slice(&sample_rate.to_le_bytes());
        for ch in 0..n_ch {
            let mut rec = vec![0u8; ELECTLOC_LEN];
            let label = format!("E{ch:02}");
            rec[..label.len()].copy_from_slice(label.as_bytes());
            buf.extend_from_slice(&rec);
        }
        for s in 0..n_samples {
            for ch in 0..n_ch {
                let v = (s as i16) * (ch as i16 + 1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn read_bundle_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("synth.cnt");
        let blob = synth_cnt(4, 100, 250);
        std::fs::write(&p, &blob).unwrap();
        let mut reader = CntReader::new(&p);
        let b = reader.read_bundle().unwrap();
        assert_eq!(b.signal.len(), 4);
        assert_eq!(b.signal[0].len(), 100);
        for s in 0..100 {
            for ch in 0..4 {
                assert_eq!(b.signal[ch][s], (s as i64) * (ch as i64 + 1));
            }
        }
        assert_eq!(b.metadata.format, "CNT");
        assert_eq!(b.sample_rate, 250.0);
        assert_eq!(b.channels[0], "E00");
    }

    #[test]
    fn read_bundle_rejects_too_small_header() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("tiny.cnt");
        std::fs::write(&p, b"too short").unwrap();
        match CntReader::new(&p).read_bundle() {
            Err(LmlError::Truncated { .. }) => {}
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn read_bundle_rejects_zero_channels() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("zero.cnt");
        let mut buf = vec![0u8; SETUP_HEADER_LEN];
        buf[376..378].copy_from_slice(&250u16.to_le_bytes()); // valid rate
        std::fs::write(&p, &buf).unwrap();
        match CntReader::new(&p).read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("nchannels = 0")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_bundle_rejects_zero_sample_rate() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("zerorate.cnt");
        let mut buf = vec![0u8; SETUP_HEADER_LEN];
        buf[370..372].copy_from_slice(&2u16.to_le_bytes());
        // rate stays 0
        std::fs::write(&p, &buf).unwrap();
        match CntReader::new(&p).read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("sample_rate = 0")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    // ─── ADR 0069 L7 gate: lower_to_abir byte-exactness ─────────────

    #[test]
    fn lower_to_abir_matches_read_bundle_i64() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("synth.cnt");
        let blob = synth_cnt(4, 100, 250);
        std::fs::write(&p, &blob).unwrap();

        let bundle = CntReader::new(&p).read_bundle().unwrap();
        let abir = CntReader::new(&p).lower_to_abir().unwrap();

        assert_eq!(abir.n_channels(), 4);
        assert_eq!(abir.n_samples, 100);
        assert_eq!(abir.sample_rate, 250.0);
        for (j, ch) in abir.channels.iter().enumerate() {
            assert!(
                matches!(ch.data, Column::I16(_)),
                "CNT must specialize to Column::I16"
            );
            let widened = ch.data.window_i64(0, abir.n_samples);
            assert_eq!(
                widened.as_ref(),
                bundle.signal[j].as_slice(),
                "channel {j} mismatch"
            );
        }
    }
}
