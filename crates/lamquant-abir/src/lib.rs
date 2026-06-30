#![cfg_attr(not(feature = "std"), no_std)]
//! ABIR — the Atomic Biosignal Intermediate Representation (ADR 0069).
//!
//! Foundational, **no_std-first** crate that both LamQuant codecs depend DOWN on
//! (graph: `lamquant-common` ← `lamquant-abir` ← {LML tiers, LMQ} ← py). S2a seeds
//! it with the two self-contained codec-seam enums — [`Format`] (the wire-format
//! discriminator) and [`Mode`] (the codec operation mode). The typed IR atoms,
//! width-typed columns, modality types, and the shared `Codec` trait land in later
//! increments (S2b/S3). These enums are re-exported at
//! `lamquant_lml_mcu::codec::{Format, Mode}`, so the relocation is byte-identical
//! and no downstream consumer changes.

/// Which deterministic wire format a stream is, decided by its leading magic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// LML — the cheap-decode integer floor / interchange standard.
    Lml,
    /// LMO — the Optimum max-compression-ratio ceiling.
    Lmo,
}

/// The functional surface shared by both formats (ADR 0052). Each variant maps
/// to an LML entry point today; LMO mirrors the same surface with its own
/// machinery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Bit-exact, MAE = 0 (H.BWC WP0). Integer-only, available everywhere.
    Lossless,
    /// Near-lossless: every reconstructed sample within `delta` of the original
    /// (bounded MAE). Integer-only, available everywhere.
    BoundedMae(u64),
    /// Rate-targeted: minimize distortion subject to a bit-per-sample ceiling
    /// (H.BWC WP1–WP8). Needs the host RD search (`archive` feature); a no_std
    /// build returns `CodecError::ModeUnsupported`.
    TargetBps(f64),
}
