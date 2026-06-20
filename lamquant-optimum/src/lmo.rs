//! The LMO wire format + codec (ADR 0052 Tier 3, ADR 0054 Phase 2).
//!
//! **Phase 2 = parity baseline.** LMO is a self-describing binary container
//! whose WP0 payload is, for now, a *faithful re-encode of the LML pipeline* —
//! i.e. the container wraps a genuine LML stream. This makes WP0 bit-exact by
//! construction (the inner codec is the proven integer floor) and gives the
//! ratio attack (Phases 3–4) a trustworthy baseline to beat before any deeper
//! transform / PCRD allocation / LMO-native entropy lands.
//!
//! Container layout (v1):
//!
//! ```text
//!   [0..4]  LMO_MAGIC = b"LMO1"
//!   [4]     version   = LMO_VERSION
//!   [5]     mode_tag  = 0 lossless | 1 bounded_mae | 2 target_bps  (informational)
//!   [6..]   inner LML stream (the Phase-2 payload)
//! ```
//!
//! Decode is `no_std`-capable (it just strips the 6-byte header and delegates
//! to the integer `lml::decompress`). Encode is host-only (`encode` feature):
//! it may invoke the f64 RD search for `TargetBps`.

use alloc::vec::Vec;

use lamquant_lossless_core::codec::{Codec, CodecError, Format, Mode, Signal, LMO_MAGIC};
use lamquant_lossless_core::error::LmlError;
use lamquant_lossless_core::lml;

/// LMO container version. Bumped only on a wire-format change.
pub const LMO_VERSION: u8 = 1;

/// Fixed container header length (`magic(4) + version(1) + mode_tag(1)`).
pub const LMO_HEADER_LEN: usize = 6;

/// The Optimum (LMO) codec. Implements the shared [`Codec`] seam.
#[derive(Debug, Default, Clone, Copy)]
pub struct LmoCodec;

/// Informational mode tag stored in the container header.
#[cfg(feature = "encode")]
fn mode_tag(mode: Mode) -> u8 {
    match mode {
        Mode::Lossless => 0,
        Mode::BoundedMae(_) => 1,
        Mode::TargetBps(_) => 2,
    }
}

impl Codec for LmoCodec {
    fn format(&self) -> Format {
        Format::Lmo
    }

    fn encode(&self, signal: &[Vec<i64>], mode: Mode) -> Result<Vec<u8>, CodecError> {
        #[cfg(feature = "encode")]
        {
            encode(signal, mode)
        }
        #[cfg(not(feature = "encode"))]
        {
            let _ = (signal, mode);
            Err(CodecError::ModeUnsupported)
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<Signal, CodecError> {
        decode(bytes)
    }
}

/// Encode `signal` under `mode` into an LMO container (host-only).
///
/// Phase 2: the payload is a faithful LML re-encode in the same mode, so WP0 is
/// bit-exact and the δ-bound / BPS-ceiling guarantees are inherited from LML.
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>], mode: Mode) -> Result<Vec<u8>, CodecError> {
    // Delegate to the LML floor for the payload (parity baseline). TargetBps
    // pulls the host RD search via `lamquant-lossless-core/archive`, enabled by
    // this crate's `encode` feature.
    let inner = lamquant_lossless_core::codec::LmlCodec.encode(signal, mode)?;

    let mut out = Vec::with_capacity(LMO_HEADER_LEN + inner.len());
    out.extend_from_slice(LMO_MAGIC.as_slice());
    out.push(LMO_VERSION);
    out.push(mode_tag(mode));
    out.extend_from_slice(&inner);
    Ok(out)
}

/// Decode an LMO container back to the signal (`no_std`-capable).
pub fn decode(bytes: &[u8]) -> Result<Signal, CodecError> {
    if bytes.len() < LMO_HEADER_LEN {
        return Err(CodecError::UnknownFormat);
    }
    if &bytes[0..4] != LMO_MAGIC.as_slice() {
        return Err(CodecError::UnknownFormat);
    }
    let version = bytes[4];
    if version != LMO_VERSION {
        // No back-compat history yet; an unknown version is not decodable here.
        return Err(CodecError::Lml(LmlError::UnsupportedVersion(version)));
    }
    // bytes[5] = mode_tag, informational — the inner LML stream self-describes.
    let inner = &bytes[LMO_HEADER_LEN..];
    lml::decompress(inner).map_err(Into::into)
}

/// Universal magic-dispatch decode for a build that has the LMO decoder linked.
///
/// Routes LML streams to the integer floor and LMO streams here. This is the
/// "full" dispatch (no [`CodecError::OptimumNotInstalled`]) the Desktop profile
/// — and a Firmware build that opted into LMO decode — exposes, in contrast to
/// `lamquant_lossless_core::codec::decode` which returns *not installed* for LMO.
pub fn decode_any(bytes: &[u8]) -> Result<Signal, CodecError> {
    match lamquant_lossless_core::codec::peek_format(bytes) {
        Some(Format::Lml) => lml::decompress(bytes).map_err(Into::into),
        Some(Format::Lmo) => decode(bytes),
        None => Err(CodecError::UnknownFormat),
    }
}
