//! Format-agnostic carrier for "I read a physiology recording from
//! somewhere, here are the samples + metadata".
//!
//! The codec input is `signal: Vec<Vec<i64>>`; everything else here
//! supports either roundtrip reconstruction (sidecar) or downstream
//! workflows (channels, phys units, patient ID).
//!
//! Design constraint (Bible R6 / R30): readers that want to admit
//! format-specific bytes do so through `SidecarBlob`, NOT by exposing
//! their own struct shape upstream. The codec never inspects sidecar
//! contents; the matching reader's `From<SignalBundle>` impl decodes
//! them back into the rich format-specific type when needed (e.g.
//! `lml decode --to-edf`).

/// Provenance + channel-level facts that every reader can supply.
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    /// Absolute or relative path the bytes came from. `"<stdin>"` /
    /// `"<s3://...>"` for non-file sources.
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
pub struct SidecarBlob {
    pub key: String,
    pub bytes: Vec<u8>,
    /// Optional integer tag — channel index, record number, etc. Format
    /// reader decides the meaning per `key`. `None` for blobs that
    /// don't need disambiguation.
    pub aux: Option<i64>,
}

/// The canonical "signal + metadata" tuple every reader produces.
///
/// Codec input is `signal`; the rest is preserved for roundtrip,
/// reporting, and downstream tools. Subsequent refactor phases will
/// migrate the LML container layer to consume `SignalBundle` directly
/// (today's container takes raw `&[Vec<i64>]` plus a JSON metadata
/// string — the bundle is the strongly-typed equivalent).
#[derive(Debug, Clone)]
pub struct SignalBundle {
    /// `[n_channels][total_samples]` — the LML kernel input.
    pub signal: Vec<Vec<i64>>,
    /// Hz.
    pub sample_rate: f64,
    /// Length = `signal.len()`. Channel labels in source order.
    pub channels: Vec<String>,
    /// Length = `signal.len()`. Physical min per channel (e.g. -200 uV).
    pub phys_min: Vec<f64>,
    /// Length = `signal.len()`. Physical max per channel.
    pub phys_max: Vec<f64>,
    /// Wall-clock duration of the recording in seconds.
    pub duration_s: f64,
    /// Format-agnostic provenance.
    pub metadata: SourceMetadata,
    /// Format-specific preservation blobs. Order is reader-defined.
    pub sidecar: Vec<SidecarBlob>,
}

impl SignalBundle {
    /// Find the first sidecar entry matching `key`. Used by per-format
    /// readers' `TryFrom<SignalBundle>` impls (e.g. EDF reconstruction).
    pub fn sidecar_first(&self, key: &str) -> Option<&SidecarBlob> {
        self.sidecar.iter().find(|s| s.key == key)
    }

    /// All sidecar entries matching `key`, in source order. EDF
    /// non-EEG channels arrive multiple times (one per channel).
    pub fn sidecar_all<'a>(&'a self, key: &'a str) -> impl Iterator<Item = &'a SidecarBlob> {
        self.sidecar.iter().filter(move |s| s.key == key)
    }

    /// Number of channels in the bundle. Fast-path accessor; assumes
    /// the cross-field invariant `signal.len == channels.len == phys_*.len`
    /// already holds. Use [`validate`] at trust boundaries to enforce
    /// it before consuming a bundle from an untrusted source.
    pub fn n_channels(&self) -> usize {
        let n = self.signal.len();
        debug_assert_eq!(n, self.channels.len(), "channels.len mismatch");
        debug_assert_eq!(n, self.phys_min.len(), "phys_min.len mismatch");
        debug_assert_eq!(n, self.phys_max.len(), "phys_max.len mismatch");
        n
    }

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
    pub fn validate(&self) -> crate::error::LmlResult<()> {
        let n = self.signal.len();
        if self.channels.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "SignalBundle: channels.len {} != signal.len {}",
                self.channels.len(),
                n
            )));
        }
        if self.phys_min.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "SignalBundle: phys_min.len {} != signal.len {}",
                self.phys_min.len(),
                n
            )));
        }
        if self.phys_max.len() != n {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "SignalBundle: phys_max.len {} != signal.len {}",
                self.phys_max.len(),
                n
            )));
        }
        if !self.sample_rate.is_finite() || self.sample_rate <= 0.0 {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "SignalBundle: sample_rate {} must be finite and > 0",
                self.sample_rate
            )));
        }
        if !self.duration_s.is_finite() || self.duration_s < 0.0 {
            return Err(crate::error::LmlError::InvalidHeader(format!(
                "SignalBundle: duration_s {} must be finite and >= 0",
                self.duration_s
            )));
        }
        if let Some(first) = self.signal.first() {
            let len = first.len();
            for (i, ch) in self.signal.iter().enumerate() {
                if ch.len() != len {
                    return Err(crate::error::LmlError::InvalidHeader(format!(
                        "SignalBundle: channel {i} has {} samples, expected {len} (ragged signal)",
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

    fn make_bundle(n_ch: usize, n_samples: usize) -> SignalBundle {
        SignalBundle {
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
        let b = make_bundle(4, 100);
        assert_eq!(b.n_channels(), 4);
    }

    #[test]
    fn validate_passes_on_well_formed_bundle() {
        let b = make_bundle(4, 100);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn validate_rejects_channels_length_mismatch() {
        let mut b = make_bundle(4, 100);
        b.channels.pop();
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_ragged_signal() {
        let mut b = make_bundle(2, 100);
        b.signal[1].pop();
        let err = b.validate().unwrap_err().to_string();
        assert!(err.contains("ragged"), "got: {err}");
    }

    #[test]
    fn validate_rejects_non_finite_sample_rate() {
        let mut b = make_bundle(1, 10);
        b.sample_rate = f64::NAN;
        assert!(b.validate().is_err());
        b.sample_rate = f64::INFINITY;
        assert!(b.validate().is_err());
        b.sample_rate = -1.0;
        assert!(b.validate().is_err());
    }

    #[test]
    fn validate_rejects_negative_duration() {
        let mut b = make_bundle(1, 10);
        b.duration_s = -0.001;
        assert!(b.validate().is_err());
    }

    #[test]
    fn sidecar_first_finds_match() {
        let mut b = make_bundle(1, 10);
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
        let h = b.sidecar_first("raw_header").unwrap();
        assert_eq!(h.bytes.len(), 256);
        assert_eq!(h.bytes[0], 0xAA);
    }

    #[test]
    fn sidecar_first_missing_returns_none() {
        let b = make_bundle(1, 10);
        assert!(b.sidecar_first("nonexistent").is_none());
    }

    #[test]
    fn sidecar_all_collects_in_source_order() {
        let mut b = make_bundle(1, 10);
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
        let chunks: Vec<&SidecarBlob> = b.sidecar_all("non_eeg_chunk").collect();
        assert_eq!(chunks.len(), 3);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.aux, Some(i as i64));
        }
    }
}
