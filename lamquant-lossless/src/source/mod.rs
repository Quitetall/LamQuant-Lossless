//! Source-format support: shared parse helpers + `SignalSourceReader`
//! trait that EDF, BrainVision, NeuroScan CNT, and custom raw readers
//! all implement.
//!
//! This module is the validation chassis. The intent is that adding a
//! new reader = one new file under `source/` that reuses every
//! primitive here; no copy-paste of UTF-8 / integer / float / int24
//! parsing logic. Bible R15 (modularity) + R1 (one thing well).
//!
//! Submodules:
//! - `ascii`       — Phase 0.2: parse_usize, parse_i64, parse_float
//! - `bitstream`   — Phase 0.2: read_i24_le
//! - `bundle`      — private native parse storage plus `SourceMetadata`
//! - `reader`      — Phase 0.3: `SignalSourceReader` trait
//! - `edf_reader`  — Phase 0.3: `EdfReader` (first impl); the legacy
//!   free function `crate::edf::read_edf` continues to exist for
//!   non-migrated callers

pub mod ascii;
pub mod bitstream;
pub mod brainvision;
mod bundle;
pub mod cnt;
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
pub use bundle::SourceMetadata;
#[cfg(test)]
pub(crate) use bundle::{ParsedUniformSignal, SidecarBlob};
pub use cnt::CntReader;
pub use descriptor::{
    lower_descriptor_to_abir, lower_descriptor_to_abir_with_options, ChannelCount, ChannelModality,
    ChannelModalityRule, DescriptorDtype, DescriptorError, DescriptorOrientation, Endian,
    FormatDescriptor, SampleRateSpec,
};
#[cfg(feature = "dicom")]
pub use dicom::DicomWaveformReader;
pub use edf_reader::EdfReader;
pub use eeglab::EeglabReader;
pub use raw::RawReader;
pub use reader::SignalSourceReader;
pub use semantic::{
    from_owned_uniform_signal, from_uniform_signal_view, SemanticChannelMapping,
    SemanticEventMapping, SemanticFidelityReport, SemanticLoweringOptions, SemanticMappingReport,
    SemanticRead, SemanticSidecarView, SemanticSourceCapsule, SemanticSourceObject,
    SemanticTimedEvent, UniformSignalView,
};
