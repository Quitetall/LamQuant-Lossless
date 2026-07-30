//! Temporary source-construction inputs used while readers build validated
//! ABIR datasets.
//!
//! These types never cross the public reader seam. Source bytes become
//! content-bound ABIR capsules; source metadata becomes typed catalog keys.

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

/// Opaque source bytes promoted into content-bound ABIR capsules.
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
}
