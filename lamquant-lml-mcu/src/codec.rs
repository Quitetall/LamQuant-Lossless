//! The shared **codec seam** (ADR 0052): one functional surface over the two
//! deterministic wire formats (LML floor / LMO ceiling).
//!
//! The seam is the **I/O contract, not the implementation**: a [`Codec`]
//! encodes a signal under a [`Mode`] and decodes a stream back, but each format
//! owns its own transform / entropy / rate-control machinery. This module lives
//! in `-core` (the universal reader every profile links) and provides:
//!
//!   * [`Format`] + [`peek_format`] — self-describing streams via the leading
//!     4-byte magic.
//!   * [`Codec`] — the trait both `LmlCodec` (here) and `LmoCodec`
//!     (`lamquant-optimum`) implement.
//!   * [`LmlCodec`] — the LML implementation (integer, no_std).
//!   * [`decode`] — the **universal magic-dispatch** entry point. LML streams
//!     decode inline; LMO streams return [`CodecError::OptimumNotInstalled`]
//!     here, because `-core` has no dependency on `lamquant-optimum`. A build
//!     that *does* link the LMO decoder (Desktop, or a Firmware that opts in)
//!     provides its own dispatch that routes LMO to the Optimum crate. This is
//!     the typed "LMO decoder not installed" story of ADR 0052 — a Firmware
//!     stream is never mis-parsed.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::lpc;

/// A multi-channel integer signal `[n_channels][n_samples]` — the shared input
/// abstraction both formats consume and reconstruct.
pub type Signal = Vec<Vec<i64>>;

/// The LML format magic (`b"LML1"`). Re-exported from [`crate::lml`] so the
/// format registry has a single home alongside [`LMO_MAGIC`].
pub const LML_MAGIC: &[u8; 4] = lml::MAGIC;

/// The LMO (Optimum) format magic. Canonical definition: `lamquant-optimum`
/// re-exports *this* constant rather than declaring its own, so the format
/// registry has exactly one source of truth.
pub const LMO_MAGIC: &[u8; 4] = b"LMO1";

// The codec seam — `Format`, `Mode`, the `Codec` trait, and the format-agnostic
// `CodecError` — lives in the foundational `abir` crate (ADR 0069 S2a/L2).
// Re-exported here so `lamquant_lml_mcu::codec::{Format, Mode, Codec, CodecError}` —
// and every downstream path (`lamquant_core::codec::*`, `lamquant_lml_optimum::*`,
// firmware) — keeps resolving with zero consumer edits. L2 decoupled `CodecError`
// from `LmlError`: the LML/LMO impls map their kernel error into
// `CodecError::Backend(_)` at the boundary via [`lml_err`].
pub use abir::{Codec, CodecError, Format, Mode};

/// Map a kernel [`LmlError`] into the format-agnostic [`CodecError`] at the seam
/// boundary (ADR 0069 L2: the contract no longer wraps `LmlError`). Uses the
/// `Display` rendering — human-readable and no_std-clean.
fn lml_err(e: LmlError) -> CodecError {
    CodecError::Backend(e.to_string())
}

/// The LML codec — integer, cheap-decode, no_std. The interchange floor.
#[derive(Debug, Default, Clone, Copy)]
pub struct LmlCodec;

impl Codec for LmlCodec {
    fn format(&self) -> Format {
        Format::Lml
    }

    fn encode(&self, signal: &[Vec<i64>], mode: Mode) -> Result<Vec<u8>, CodecError> {
        match mode {
            Mode::Lossless => lml::compress(signal, 0).map_err(lml_err),
            Mode::BoundedMae(delta) => {
                lml::compress_bounded_mae(signal, delta, lpc::LpcMode::default()).map_err(lml_err)
            }
            Mode::TargetBps(bps) => {
                #[cfg(feature = "std")]
                {
                    lml::compress_target_bps(signal, bps, lpc::LpcMode::default())
                        .map_err(lml_err)
                }
                #[cfg(not(feature = "std"))]
                {
                    let _ = bps;
                    Err(CodecError::ModeUnsupported)
                }
            }
        }
    }

    fn decode(&self, bytes: &[u8]) -> Result<Signal, CodecError> {
        lml::decompress(bytes).map_err(lml_err)
    }
}

/// Peek the format of a stream from its leading bytes, without decoding.
///
/// An LML stream begins either with the raw `LML1` magic or with the
/// human-readable ASCII prefix `"LML | …CRC-32\n"` that precedes it — both
/// start with the 3 bytes `b"LML"`. An LMO stream is a binary container that
/// begins with the raw `LMO1` magic. The `LML`/`LMO` 3-byte prefixes are
/// distinct (`'L'` vs `'O'` at index 2), so the classification is unambiguous.
/// Returns `None` for anything else (too short, or no match).
pub fn peek_format(bytes: &[u8]) -> Option<Format> {
    if bytes.len() < 4 {
        return None;
    }
    let m = &bytes[0..4];
    if m == LMO_MAGIC.as_slice() {
        return Some(Format::Lmo);
    }
    // Covers raw "LML1" and the ASCII-prefixed "LML | …" header alike.
    if &m[0..3] == b"LML" {
        return Some(Format::Lml);
    }
    None
}

/// Universal magic-dispatch decode (the `-core` half: LML only).
///
/// * LML stream → decoded inline via [`lml::decompress`].
/// * LMO stream → [`CodecError::OptimumNotInstalled`] — `-core` cannot reach
///   the Optimum decoder. A build that links `lamquant-optimum` provides a
///   fuller dispatch (see the facade's `decode`).
/// * unknown magic → [`CodecError::UnknownFormat`].
pub fn decode(bytes: &[u8]) -> Result<Signal, CodecError> {
    match peek_format(bytes) {
        Some(Format::Lml) => lml::decompress(bytes).map_err(lml_err),
        Some(Format::Lmo) => Err(CodecError::OptimumNotInstalled),
        None => Err(CodecError::UnknownFormat),
    }
}

/// Convenience: encode via the LML codec (the always-available floor).
/// LMO encode lives in `lamquant-optimum` behind its `encode` feature.
pub fn encode_lml(signal: &[Vec<i64>], mode: Mode) -> LmlResult<Vec<u8>> {
    // ADR 0069 L2: call the LML encode path DIRECTLY so it returns the native
    // `LmlError` — no round-trip through the (now format-agnostic) `CodecError`.
    match mode {
        Mode::Lossless => lml::compress(signal, 0),
        Mode::BoundedMae(delta) => {
            lml::compress_bounded_mae(signal, delta, lpc::LpcMode::default())
        }
        Mode::TargetBps(bps) => {
            #[cfg(feature = "std")]
            {
                lml::compress_target_bps(signal, bps, lpc::LpcMode::default())
            }
            #[cfg(not(feature = "std"))]
            {
                let _ = bps;
                Err(LmlError::InvalidHeader(alloc::string::String::from(
                    "requested mode unsupported in this build",
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn ramp(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        (0..n_ch)
            .map(|c| (0..t).map(|i| ((i * 7 + c * 13) % 101) as i64 - 50).collect())
            .collect()
    }

    #[test]
    fn lml_lossless_roundtrip_via_trait() {
        let sig = ramp(4, 256);
        let stream = LmlCodec.encode(&sig, Mode::Lossless).expect("encode");
        assert_eq!(peek_format(&stream), Some(Format::Lml));
        let back = LmlCodec.decode(&stream).expect("decode");
        assert_eq!(back, sig);
    }

    #[test]
    fn universal_decode_routes_lml() {
        let sig = ramp(2, 128);
        let stream = LmlCodec.encode(&sig, Mode::Lossless).unwrap();
        let back = decode(&stream).expect("dispatch decode");
        assert_eq!(back, sig);
    }

    #[test]
    fn lmo_stream_reports_not_installed_in_core() {
        // Hand-craft a minimal LMO-magic stream. -core must classify it as LMO
        // and report OptimumNotInstalled — never mis-parse it as LML.
        let mut lmo = LMO_MAGIC.to_vec();
        lmo.extend_from_slice(&[0x01, 0xAA, 0xBB]);
        assert_eq!(peek_format(&lmo), Some(Format::Lmo));
        match decode(&lmo) {
            Err(CodecError::OptimumNotInstalled) => {}
            other => panic!("expected OptimumNotInstalled, got {:?}", other),
        }
    }

    #[test]
    fn unknown_magic_rejected() {
        assert_eq!(peek_format(b"\x00\x01\x02\x03"), None);
        assert_eq!(peek_format(b"XZ"), None);
        match decode(b"NOPEnope") {
            Err(CodecError::UnknownFormat) => {}
            other => panic!("expected UnknownFormat, got {:?}", other),
        }
    }

    #[test]
    fn bounded_mae_respects_delta() {
        let sig = vec![(0..512).map(|i| (i as i64 * 37) % 9001 - 4500).collect::<Vec<_>>()];
        let delta = 8u64;
        let stream = LmlCodec.encode(&sig, Mode::BoundedMae(delta)).expect("encode");
        let back = LmlCodec.decode(&stream).expect("decode");
        for (o, r) in sig[0].iter().zip(back[0].iter()) {
            assert!((o - r).unsigned_abs() <= delta, "|{o}-{r}| exceeds {delta}");
        }
    }
}
