//! Private parsed form used only inside source-reader implementations.
//!
//! The codec input is `signal: Vec<Vec<i64>>`; everything else here
//! supports either roundtrip reconstruction (sidecar) or downstream
//! workflows (channels, phys units, patient ID).
//!
//! Public source seams return validated semantic ABIR. This module assembles
//! one native uniform recording before the same transaction validates it into
//! `SemanticRead`.

/// The privacy-safe `source_file` value for a reader: the file's BASENAME,
/// never its absolute path (#30 — embedding the full host path in the encoded
/// metadata leaks provenance/PII and makes the container non-portable). `""`
/// for a pathless source. Every reader that sets [`SourceMetadata::source_file`]
/// from a path MUST route through this (EDF already does the equivalent inline).
pub(crate) fn source_basename(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Provenance + channel-level facts that every reader can supply.
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    /// The BASENAME of the file the bytes came from — never the absolute path
    /// (#30). `"<stdin>"` / `"<s3://...>"` for non-file sources. Set via
    /// [`source_basename`].
    pub source_file: String,
    /// Source format identifier: `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`,
    /// `"BRAINVISION"`, `"CNT"`, `"DICOM"`, `"RAW"`, …
    pub format: String,
    /// Patient identifier as recorded. Treat as PII — `lml strip-pii`
    /// (Phase 3.8) will redact this field on demand.
    pub patient_id: String,
    /// Free-form recording context (hospital, equipment, technician).
    pub recording_info: String,
    /// Recording start date, format depends on source.
    pub startdate: String,
    /// SI unit for physical-min/max scaling (typically `"uV"`).
    pub phys_dim: String,
}

/// Opaque byte blob preserved across roundtrip. The `key` is the only
/// part the codec / archive layer interprets; bytes are pass-through.
///
/// Per-format convention (EDF/BDF readers populate):
/// - `"raw_header"`   — main + signal headers, lossless EDF reconstruction
/// - `"trailing_data"` — bytes past the last complete record
/// - `"non_eeg_chunk"` — non-EEG channel data (annotations / status);
///   `aux` carries the original channel index
/// - `"edf_meta"`      — JSON-encoded EDF-specific scalars
///   (n_data_records, record_duration, dig_min/max, eeg_indices, …)
#[derive(Debug, Clone)]
pub(crate) struct SidecarBlob {
    pub(crate) key: String,
    pub(crate) bytes: Vec<u8>,
    /// Optional integer tag — channel index, record number, etc. Format
    /// reader decides the meaning per `key`. `None` for blobs that
    /// don't need disambiguation.
    pub(crate) aux: Option<i64>,
}

/// Private native parse result. Never crosses a Module Seam.
#[derive(Debug, Clone)]
pub(crate) struct ParsedUniformSignal {
    /// `[n_channels][total_samples]` — the LML kernel input.
    pub(crate) signal: Vec<Vec<i64>>,
    /// Hz.
    pub(crate) sample_rate: f64,
    /// Length = `signal.len()`. Channel labels in source order.
    pub(crate) channels: Vec<String>,
    /// Length = `signal.len()`. Physical min per channel (e.g. -200 uV).
    pub(crate) phys_min: Vec<f64>,
    /// Length = `signal.len()`. Physical max per channel.
    pub(crate) phys_max: Vec<f64>,
    /// Wall-clock duration of the recording in seconds.
    pub(crate) duration_s: f64,
    /// Format-agnostic provenance.
    pub(crate) metadata: SourceMetadata,
    /// Format-specific preservation blobs. Order is reader-defined.
    pub(crate) sidecar: Vec<SidecarBlob>,
}

impl ParsedUniformSignal {
    /// Cross-field invariant check. Call at every trust boundary
    /// (after reading a bundle from an untrusted source, before
    /// passing one to the codec). Returns `Err` rather than panicking
    /// so a malformed reader can't crash the whole process — Bible R6
    /// strict types at the boundary, R7 fail gracefully.
    ///
    /// Invariants enforced:
    ///   - `signal.len() == channels.len() == phys_min.len() == phys_max.len()`
    ///   - all channel buffers have the same length (no ragged signal)
    ///   - `sample_rate > 0` and finite
    ///   - `duration_s >= 0` and finite
    pub(crate) fn validate(&self) -> crate::error::LmlResult<()> {
        let n = self.signal.len();
        if self.channels.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "parsed uniform signal: channels.len {} != signal.len {}",
                self.channels.len(),
                n
            )));
        }
        if self.phys_min.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "parsed uniform signal: phys_min.len {} != signal.len {}",
                self.phys_min.len(),
                n
            )));
        }
        if self.phys_max.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "parsed uniform signal: phys_max.len {} != signal.len {}",
                self.phys_max.len(),
                n
            )));
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "parsed uniform signal: sample_rate {} must be finite and > 0",
                self.sample_rate
            )));
        }
        if !self.duration_s.is_finite() || self.duration_s < 0.0 {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "parsed uniform signal: duration_s {} must be finite and >= 0",
                self.duration_s
            )));
        }
        if let Some(first) = self.signal.first() {
            let len = first.len();
            for (i, ch) in self.signal.iter().enumerate() {
                if ch.len() != len {
                    return Err(crate::error::LmlError::InvalidHeader(format!(
                        "parsed uniform signal: channel {i} has {} samples, expected {len} (ragged signal)",
                        ch.len()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #30: source_basename returns ONLY the file name, never the absolute path,
    /// so the encoded metadata can't leak the host directory tree / PII.
    #[test]
    fn source_basename_strips_the_directory() {
        use std::path::Path;
        assert_eq!(
            source_basename(Path::new("/home/alice/data/patient001.edf")),
            "patient001.edf"
        );
        assert_eq!(source_basename(Path::new("rec.vhdr")), "rec.vhdr");
        assert_eq!(
            source_basename(Path::new("../scratch/sub 02.set")),
            "sub 02.set"
        );
        assert_eq!(source_basename(Path::new("/")), "");
    }

    fn make_signal(n_ch: usize, n_samples: usize) -> ParsedUniformSignal {
        ParsedUniformSignal {
            signal: vec![vec![0i64; n_samples]; n_ch],
            sample_rate: 256.0,
            channels: (0..n_ch).map(|i| format!("ch{i}")).collect(),
            phys_min: vec![-200.0; n_ch],
            phys_max: vec![200.0; n_ch],
            duration_s: n_samples as f64 / 256.0,
            metadata: SourceMetadata {
                source_file: "test.edf".into(),
                format: "EDF".into(),
                patient_id: "anon".into(),
                recording_info: String::new(),
                startdate: "2026-05-16".into(),
                phys_dim: "uV".into(),
            },
            sidecar: vec![],
        }
    }

    #[test]
    fn n_channels_invariant_checked() {
        let b = make_signal(4, 100);
        assert_eq!(b.signal.len(), 4);
    }

    #[test]
    fn validate_passes_on_well_formed_bundle() {
        let b = make_signal(4, 100);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn validate_rejects_channels_length_mismatch() {
        let mut b = make_signal(4, 100);
        b.channels.pop();
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_ragged_signal() {
        let mut b = make_signal(2, 100);
        b.signal[1].pop();
        let err = b.validate().unwrap_err().to_string();
        assert!(err.contains("ragged"), "got: {err}");
    }

    #[test]
    fn validate_rejects_non_finite_sample_rate() {
        let mut b = make_signal(1, 10);
        b.sample_rate = f64::NAN;
        assert!(b.validate().is_err());
        b.sample_rate = f64::INFINITY;
        assert!(b.validate().is_err());
        b.sample_rate = -1.0;
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_duration() {
        let mut b = make_signal(1, 10);
        b.duration_s = -0.001;
        assert!(b.validate().is_err());
    }

    #[test]
    fn sidecar_first_finds_match() {
        let mut b = make_signal(1, 10);
        b.sidecar.push(SidecarBlob {
            key: "raw_header".into(),
            bytes: vec![0xAA; 256],
            aux: None,
        });
        b.sidecar.push(SidecarBlob {
            key: "non_eeg_chunk".into(),
            bytes: vec![0xBB; 32],
            aux: Some(3),
        });
        let h = b
            .sidecar
            .iter()
            .find(|sidecar| sidecar.key == "raw_header")
            .unwrap();
        assert_eq!(h.bytes.len(), 256);
        assert_eq!(h.bytes[0], 0xAA);
    }

    #[test]
    fn sidecar_first_missing_returns_none() {
        let b = make_signal(1, 10);
        assert!(b.sidecar.iter().all(|sidecar| sidecar.key != "nonexistent"));
    }

    #[test]
    fn sidecar_all_collects_in_source_order() {
        let mut b = make_signal(1, 10);
        for ch in 0..3 {
            b.sidecar.push(SidecarBlob {
                key: "non_eeg_chunk".into(),
                bytes: vec![ch as u8; 4],
                aux: Some(ch),
            });
        }
        b.sidecar.push(SidecarBlob {
            key: "raw_header".into(),
            bytes: vec![],
            aux: None,
        });
        let chunks: Vec<&SidecarBlob> = b
            .sidecar
            .iter()
            .filter(|sidecar| sidecar.key == "non_eeg_chunk")
            .collect();
        assert_eq!(chunks.len(), 3);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.aux, Some(i as i64));
        }
    }
}
