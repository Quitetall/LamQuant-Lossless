#![cfg_attr(not(feature = "std"), no_std)]
//! ABIR — the Atomic Biosignal Intermediate Representation (ADR 0069).
//!
//! Foundational, **no_std-first** crate that both LamQuant codecs depend DOWN on
//! (graph: `lamquant-common` ← `abir` ← {LML tiers, LMQ} ← py). S2a seeds
//! it with the two self-contained codec-seam enums — [`Format`] (the wire-format
//! discriminator) and [`Mode`] (the codec operation mode). The typed IR atoms,
//! width-typed columns, modality types, and the shared `Codec` trait land in later
//! increments (S2b/S3). These enums are re-exported at
//! `lamquant_lml_mcu::codec::{Format, Mode}`, so the relocation is byte-identical
//! and no downstream consumer changes.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// The ABIR atoms — the columnar, width-typed, zero-copy signal currency (Pillar 2).
pub mod atoms;
pub use atoms::{Abir, Channel, Column};

/// The modality trust model — the `Abir<M>` typestate (Pillar 1, ADR 0069 S3a).
pub mod modality;
pub use modality::{
    name_for_tag, Accel, Ecg, Ecog, Eeg, Emg, Eog, Ieeg, Modality, ModalityProvenance,
    ModalitySource, Other, Resp, Seeg, Untyped, VerifyError,
};

/// The reversibility markers — the `Reversible`/`Lossy` typestate (Pillar 3).
/// The no_std vocabulary; the host `Pass`/`LmlPipeline` machinery that gates on
/// it stays in `lamquant-lossless` (ADR 0074).
pub mod reversibility;
pub use reversibility::{Lossy, Reversibility, Reversible};

/// The BCS1 neutral wire header (ADR 0069/0071 L9) — the ONE deliberate byte
/// change: a 40-byte typed header (born-typed modality + codec descriptor +
/// mode + tier) wrapping the byte-unchanged JSON metadata → window index →
/// LML per-window payloads → `LMLFOOT1` footer. `no_std`-clean by
/// construction (pure `to_le_bytes`/`from_le_bytes`, no I/O).
pub mod bcs1;
pub use bcs1::{
    Bcs1Header, Bcs1ParseError, CodecDescriptor, BCS1_FLAG_HAS_FOOTER, BCS1_HEADER_LEN, BCS1_MAGIC,
    BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53, CODEC_LMO_97, CODEC_LMO_LOSSLESS,
    CODEC_LMQ_FSQ, CODEC_OPTIMUM_V2,
};

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
    /// build returns [`CodecError::ModeUnsupported`].
    TargetBps(f64),
}

/// Errors at the codec seam — **format-agnostic** (ADR 0069 L2). The contract does
/// NOT couple to any one format's error type: a backend codec (LML / LMO) maps its
/// own error into [`CodecError::Backend`] at the boundary, while the typed format
/// error (`LmlError`, …) stays available on the format-specific path (`lml::compress`).
/// This breaks the old `CodecError::Lml(LmlError)` coupling — `LmlError`/`GolombError`
/// remain kernel-internal to `lamquant-lml-mcu`.
#[derive(Debug)]
pub enum CodecError {
    /// A backend (format-specific) codec error, described textually so the seam stays
    /// decoupled from the kernels' error types.
    Backend(String),
    /// The stream is an LMO stream but this build has no LMO decoder linked.
    /// (The ADR 0052 "module not installed" outcome — never a mis-parse.)
    OptimumNotInstalled,
    /// The leading bytes match no known format magic.
    UnknownFormat,
    /// The requested [`Mode`] is not compiled into this build (e.g.
    /// [`Mode::TargetBps`] without the host RD search).
    ModeUnsupported,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CodecError::Backend(s) => write!(f, "backend codec error: {s}"),
            CodecError::OptimumNotInstalled => write!(f, "LMO decoder not installed in this build"),
            CodecError::UnknownFormat => write!(f, "unknown stream format (no magic match)"),
            CodecError::ModeUnsupported => {
                write!(f, "requested codec mode not available in this build")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CodecError {}

/// The shared encode/decode contract — implemented by `LmlCodec` (LML, in
/// `lamquant-lml-mcu`) and `LmoCodec` (LMO, in `lamquant-lml-optimum`). A universal
/// magic-dispatch `decode` routes a stream to the right one.
pub trait Codec {
    /// The wire format this codec produces and reads.
    fn format(&self) -> Format;

    /// Encode `signal` under `mode` into a self-describing (magic-stamped) stream.
    fn encode(&self, signal: &[Vec<i64>], mode: Mode) -> Result<Vec<u8>, CodecError>;

    /// Decode a stream of this codec's format back into the signal.
    fn decode(&self, bytes: &[u8]) -> Result<Vec<Vec<i64>>, CodecError>;
}
