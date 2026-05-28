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
//! - `bundle`      — Phase 0.3: `SignalBundle`, `SourceMetadata`,
//!   `SidecarBlob` (the codec-agnostic carrier)
//! - `reader`      — Phase 0.3: `SignalSourceReader` trait
//! - `edf_reader`  — Phase 0.3: `EdfReader` (first impl); the legacy
//!   free function `crate::edf::read_edf` continues to exist for
//!   non-migrated callers

pub mod ascii;
pub mod bitstream;
pub mod brainvision;
pub mod bundle;
pub mod cnt;
#[cfg(feature = "dicom")]
pub mod dicom;
pub mod edf_reader;
pub mod eeglab;
pub mod raw;
pub mod reader;

pub use brainvision::BrainVisionReader;
pub use bundle::{SidecarBlob, SignalBundle, SourceMetadata};
pub use cnt::CntReader;
#[cfg(feature = "dicom")]
pub use dicom::DicomWaveformReader;
pub use edf_reader::EdfReader;
pub use eeglab::EeglabReader;
pub use raw::RawReader;
pub use reader::SignalSourceReader;
