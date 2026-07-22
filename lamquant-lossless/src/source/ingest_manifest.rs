//! ADR 0074 · Track I — the per-dataset ingest manifest (host-only, `archive`).
//!
//! A serde config declaring, per dataset, the AUTHORITATIVE modality — so a
//! dataset receives an explicit ABIR modality concept instead of depending on
//! filename or channel-label heuristics.
//!
//! Parsed from JSON (`serde_json`, already linked under `archive`). A dataset is
//! matched by a path substring (its directory / name); first match wins. The
//! reader-override + reader-args fields are a later phase (I3, `FormatDescriptor`).

use std::path::Path;

use serde::Deserialize;

use super::descriptor::FormatDescriptor;

/// A modality declaration mapped to the canonical ABIR concept registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModalityDecl {
    Eeg,
    Ieeg,
    Ecog,
    Seeg,
    Ecg,
    Emg,
    Eog,
    Resp,
    Accel,
    Other,
    Untyped,
}

impl ModalityDecl {
    pub const fn concept(self) -> &'static str {
        match self {
            Self::Eeg => "abir:modality/eeg",
            Self::Ieeg => "abir:modality/ieeg",
            Self::Ecog => "abir:modality/ecog",
            Self::Seeg => "abir:modality/seeg",
            Self::Ecg => "abir:modality/ecg",
            Self::Emg => "abir:modality/emg",
            Self::Eog => "abir:modality/eog",
            Self::Resp => "abir:modality/respiration",
            Self::Accel => "abir:modality/acceleration",
            Self::Other => "abir:modality/other",
            Self::Untyped => "abir:modality/unknown",
        }
    }
}

/// One dataset's ingest rule.
#[derive(Debug, Clone, Deserialize)]
pub struct DatasetEntry {
    /// A path substring (dataset directory or name) this rule applies to.
    #[serde(rename = "match")]
    pub match_substr: String,
    /// The authoritative modality (stamped `ModalitySource::Manual` at ingest).
    pub modality: ModalityDecl,
    /// Optional (ADR 0074 Track I / I3): parse this dataset with a declared
    /// [`FormatDescriptor`] — a data-driven reader for a custom fixed-layout
    /// binary — INSTEAD of the file-extension dispatch. Absent = extension
    /// dispatch (EDF/BrainVision/RAW/CNT/…), today's behavior. This is what
    /// finally wires the proven-but-unwired `FormatDescriptor` into production.
    #[serde(default)]
    pub descriptor: Option<FormatDescriptor>,
}

/// The only manifest schema version this build understands.
pub const SUPPORTED_VERSION: u32 = 1;

/// Manifest load failure: bad JSON, or a `version` this build doesn't understand
/// (fail-closed — a future schema must NOT parse into partial/default fields).
#[derive(Debug)]
pub enum ManifestError {
    Json(serde_json::Error),
    UnsupportedVersion(u32),
}

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "ingest manifest: invalid JSON: {e}"),
            Self::UnsupportedVersion(v) => write!(
                f,
                "ingest manifest: unsupported version {v} (this build understands {SUPPORTED_VERSION})"
            ),
        }
    }
}

impl std::error::Error for ManifestError {}

/// The per-dataset ingest manifest. First-match-wins over `datasets`, in
/// declaration order (so a reproducible resolution).
#[derive(Debug, Clone, Deserialize)]
pub struct IngestManifest {
    /// Schema version (participates in the config hash / provenance).
    pub version: u32,
    #[serde(default)]
    pub datasets: Vec<DatasetEntry>,
}

impl IngestManifest {
    /// Parse from JSON, rejecting an unsupported `version` fail-closed (so a v2
    /// schema can't silently parse into default fields — MiMo review).
    pub fn from_json(text: &str) -> Result<Self, ManifestError> {
        let m: Self = serde_json::from_str(text).map_err(ManifestError::Json)?;
        if m.version != SUPPORTED_VERSION {
            return Err(ManifestError::UnsupportedVersion(m.version));
        }
        Ok(m)
    }

    /// Resolve a path to its full ingest rule (first match wins). `None` means no
    /// rule matched → the caller keeps today's behavior (born `Untyped`, extension
    /// dispatch), so an absent or non-matching manifest is byte-for-byte the
    /// current path.
    pub fn resolve_entry(&self, path: &Path) -> Option<&DatasetEntry> {
        let p = path.to_string_lossy();
        self.datasets.iter().find(|d| p.contains(&d.match_substr))
    }

    /// Resolve a path to its declared modality (first match wins).
    pub fn resolve(&self, path: &Path) -> Option<ModalityDecl> {
        self.resolve_entry(path).map(|d| d.modality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_resolves_first_match_wins() {
        let json = r#"{
            "version": 1,
            "datasets": [
                { "match": "chbmit", "modality": "eeg" },
                { "match": "ptbxl",  "modality": "ecg" }
            ]
        }"#;
        let m = IngestManifest::from_json(json).unwrap();
        assert_eq!(m.version, 1);
        assert_eq!(
            m.resolve(Path::new("/data/chbmit/chb01_03.edf")),
            Some(ModalityDecl::Eeg)
        );
        assert_eq!(
            m.resolve(Path::new("/data/ptbxl/rec_001.edf")),
            Some(ModalityDecl::Ecg)
        );
        assert_eq!(m.resolve(Path::new("/data/unknown/x.edf")), None);
        assert_eq!(ModalityDecl::Eeg.concept(), "abir:modality/eeg");
        assert_eq!(ModalityDecl::Ecg.concept(), "abir:modality/ecg");
        assert_eq!(ModalityDecl::Untyped.concept(), "abir:modality/unknown");
    }

    #[test]
    fn resolves_a_dataset_with_a_format_descriptor() {
        // I3: a dataset can declare a FormatDescriptor (custom binary layout).
        let json = r#"{
            "version": 1,
            "datasets": [
                { "match": "myraw", "modality": "eeg",
                  "descriptor": {
                      "format_name": "RAWBIN", "dtype": "I16", "endian": "Little",
                      "orientation": "Multiplexed",
                      "channel_count": {"Fixed": 2}, "sample_rate": {"Fixed": 256.0}
                  } }
            ]
        }"#;
        let m = IngestManifest::from_json(json).unwrap();
        let entry = m
            .resolve_entry(Path::new("/data/myraw/rec.bin"))
            .expect("match");
        assert_eq!(entry.modality, ModalityDecl::Eeg);
        assert_eq!(
            entry
                .descriptor
                .as_ref()
                .expect("descriptor present")
                .format_name,
            "RAWBIN"
        );
        // A dataset WITHOUT a descriptor → None (falls to the extension dispatch).
        let plain = IngestManifest::from_json(
            r#"{"version":1,"datasets":[{"match":"x","modality":"ecg"}]}"#,
        )
        .unwrap();
        assert!(plain
            .resolve_entry(Path::new("/x/y"))
            .unwrap()
            .descriptor
            .is_none());
    }

    #[test]
    fn empty_datasets_default_bad_json_and_version_guard() {
        // Empty datasets (valid v1) → no matches.
        let m = IngestManifest::from_json(r#"{"version": 1}"#).unwrap();
        assert!(m.datasets.is_empty());
        assert!(m.resolve(Path::new("/anything")).is_none());
        // Bad JSON → Json error.
        assert!(matches!(
            IngestManifest::from_json("{ not json"),
            Err(ManifestError::Json(_))
        ));
        // Unsupported version → fail-closed (NOT a silent parse into defaults).
        assert!(matches!(
            IngestManifest::from_json(r#"{"version": 2}"#),
            Err(ManifestError::UnsupportedVersion(2))
        ));
    }
}
