//! LamQuant common primitives shared across the LamQuant EEG codec family
//! (`lamquant-lml` lossless and `lamquant-lmq` neural).
//!
//! `no_std` + `alloc` by default; the `host` feature adds std-only helpers.
//!
//! Modules:
//!   - [`crc32`] — CRC-32 (ISO 3309), the integrity check used across LML/LMA.
//!   - `paths` *(host)* — path utilities for the CLI tooling.
//!   - `ingest` *(host)* — non-EDF signal ingest (ASCII int lines →
//!     synthesized EDF wrappers) so codecs consume one canonical format.
//!
//! The EDF reader, LMA archive, and codec DSP live in the codec crates, not
//! here. Extracted from `lamquant-core` during the workspace decomposition.

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod crc32;

// Path utilities for host CLI tools. host-only (uses std::path).
#[cfg(feature = "host")]
pub mod paths;

// Non-EDF EEG ingestion (ASCII int lines, synthesized EDF wrappers).
// Codec-agnostic — both lossless and neural would consume the synthesized
// EDF rather than parsing raw ASCII themselves.
#[cfg(feature = "host")]
pub mod ingest;
