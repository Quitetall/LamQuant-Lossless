//! Source-format support: shared parse helpers plus the ABIR-first
//! `SignalSourceReader` seam implemented by EDF, BrainVision, NeuroScan CNT,
//! DICOM, EEGLAB, and custom raw readers.
//!
//! This module is the validation chassis. The intent is that adding a
//! new reader = one new file under `source/` that reuses every
//! primitive here; no copy-paste of UTF-8 / integer / float / int24
//! parsing logic. Bible R15 (modularity) + R1 (one thing well).
//!
//! Submodules:
//! - `ascii`       — Phase 0.2: parse_usize, parse_i64, parse_float
//! - `bitstream`   — Phase 0.2: read_i24_le
//! - `metadata`    — private inputs used to construct ABIR catalog semantics
//! - `reader`      — mandatory semantic-ABIR `SignalSourceReader` trait
//! - `edf_reader`  — `EdfReader`; direct validated ABIR lowering

pub mod ascii;
pub mod bitstream;
pub mod brainvision;
pub mod cnt;
mod metadata;
// ADR 0069 Pillar 3 / S5 Increment 3 (task #20): the format-description
// DSL — declares a fixed-layout reader as `serde`-derivable DATA
// (`FormatDescriptor`) instead of hand-written Rust, interpreted by
// `lower_descriptor_to_abir`. Same
// `archive`-feature gate as every other module here (inherited from
// `pub mod source;` in lib.rs); no additional cfg needed.
pub mod descriptor;
// ADR 0074 Track I: the per-dataset ingest manifest (serde JSON → authoritative
// modality). Host-only — `serde_json` is linked under `archive`.
#[cfg(feature = "dicom")]
pub mod dicom;
pub mod edf_reader;
pub mod eeglab;
#[cfg(feature = "archive")]
pub mod ingest_manifest;
pub mod raw;
pub mod reader;
pub mod semantic;

pub use brainvision::BrainVisionReader;
pub use cnt::CntReader;
pub use descriptor::{
    lower_descriptor_to_abir, lower_descriptor_to_abir_with_modality, ChannelCount,
    ChannelModality, ChannelModalityRule, DescriptorDtype, DescriptorError, DescriptorOrientation,
    Endian, FormatDescriptor, SampleRateSpec,
};
#[cfg(feature = "dicom")]
pub use dicom::DicomWaveformReader;
pub use edf_reader::EdfReader;
pub use eeglab::EeglabReader;
pub use metadata::{SidecarBlob, SourceMetadata};
pub use raw::RawReader;
pub use reader::SignalSourceReader;
pub use semantic::{
    from_owned_uniform_signal, from_owned_uniform_signal_with_interchange_bound_sources,
    from_owned_uniform_signal_with_overlays, from_owned_uniform_signal_with_semantics,
    from_uniform_signal_view, with_interchange_bound_sources, SemanticChannelMapping,
    SemanticEventMapping, SemanticFidelityReport, SemanticMappingReport, SemanticRead,
    SemanticSourceCapsule, SemanticSourceObject, SemanticTimedEvent,
};
