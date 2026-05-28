//! LamQuant common primitives shared between lossless and neural codecs.
//!
//! This crate holds the no_std + alloc-compatible building blocks that both
//! `lamquant-lossless` and `lamquant-neural` need:
//!   - CRC-32 (ISO 3309) integrity check
//!   - EDF / EEG container readers (under `host` feature)
//!   - LMA archive primitives
//!   - DSP-level traits and pipeline abstractions
//!
//! Migrated incrementally from `lamquant-core` as part of the 8-repo
//! decomposition (Phase 2, decomp/lossless-extract branch).

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
