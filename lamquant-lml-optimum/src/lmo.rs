//! The LMO wire format + codec (ADR 0052 Tier 3, ADR 0054 Phase 2).
//!
//! **Phase 2 = parity baseline.** LMO is a self-describing binary container
//! whose WP0 payload is, for now, a *faithful re-encode of the LML pipeline* —
//! i.e. the container wraps a genuine LML stream. This makes WP0 bit-exact by
//! construction (the inner codec is the proven integer floor) and gives the
//! ratio attack (Phases 3–4) a trustworthy baseline to beat before any deeper
//! transform / PCRD allocation / LMO-native entropy lands.
//!
//! Container layout (v2):
//!
//! ```text
//!   [0..4]  LMO_MAGIC = b"LMO1"
//!   [4]     version       = LMO_VERSION
//!   [5]     mode_tag      = 0 lossless | 1 bounded_mae | 2 target_bps  (informational)
//!   [6]     transform_id  = 0 inner LML stream (5/3 integer floor)
//!                         | 1 LMO-native 9/7 float body (ADR 0054 lever 2)
//!   [7..]   inner payload (an LML stream when transform_id=0, a 9/7 PCRD body when =1)
//! ```
//!
//! Decode is `no_std`-capable: it strips the 7-byte header and routes on
//! `transform_id` — `0` → the integer `lml::decompress`, `1` → the float
//! `lmo_pcrd97::decode_97` (float lifting needs no std/libm). Encode is host-only
//! (`encode` feature): for `TargetBps` it runs BOTH the 5/3 and 9/7 ratio attacks
//! and keeps whichever reconstructs at lower PRD (the 9/7 transform wins ~6% PRD
//! on EEG; auto-pick guarantees the container is never worse than the 5/3 floor).

use alloc::vec::Vec;

use lamquant_lml_mcu::codec::{Codec, CodecError, Format, Mode, Signal, LMO_MAGIC};
use lamquant_lml_mcu::error::LmlError;
use lamquant_lml_mcu::lml;

/// LMO container version. Bumped only on a wire-format change. v2 adds the
/// `transform_id` byte (ADR 0054 lever 2). LMO is research-tier with in-process
/// tests only, so there is no on-disk v1 to keep readable.
pub const LMO_VERSION: u8 = 2;

/// Fixed container header length (`magic(4) + version(1) + mode_tag(1) + transform_id(1)`).
pub const LMO_HEADER_LEN: usize = 7;

/// `transform_id` = the inner payload is an LML stream (integer 5/3 floor).
const TRANSFORM_LML_53: u8 = 0;
/// `transform_id` = the inner payload is an LMO-native 9/7 float PCRD body.
const TRANSFORM_LMO_97: u8 = 1;

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

/// CfP §4 mean-removed PRD (%), for the encode-side transform auto-pick.
#[cfg(feature = "encode")]
fn prd(orig: &[Vec<i64>], recon: &[Vec<i64>]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (o, r) in orig.iter().zip(recon.iter()) {
        let m = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
        for (a, b) in o.iter().zip(r.iter()) {
            let e = (*a - *b) as f64;
            num += e * e;
            den += (*a as f64 - m) * (*a as f64 - m);
        }
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * (num / den).sqrt()
    }
}

/// Encode `signal` under `mode` into an LMO container (host-only).
///
/// - **`TargetBps` (the lossy ratio attack, ADR 0054 Phase 3):** runs BOTH ratio
///   attacks at the target rate — the integer 5/3 per-subband **PCRD**
///   (`lml::compress_target_bps_pcrd`) and the float **9/7 CDF** PCRD
///   (`lmo_pcrd97::encode_target_bps_97`, lever 2) — decodes each, and keeps the
///   one with lower PRD. The container records the winner in `transform_id`, so a
///   9/7 stream is never worse than the 5/3 floor.
/// - **`Lossless` / `BoundedMae`:** the LML integer floor (WP0 bit-exact / δ-bound
///   inherited; 9/7 is float ⇒ not bit-exact, so it never applies here).
#[cfg(feature = "encode")]
pub fn encode(signal: &[Vec<i64>], mode: Mode) -> Result<Vec<u8>, CodecError> {
    use lamquant_lml_mcu::lpc::LpcMode;

    let (transform_id, inner) = match mode {
        Mode::TargetBps(bps) => {
            // Candidate 1: the integer 5/3 floor (always valid).
            let i53 = lml::compress_target_bps_pcrd(signal, bps, LpcMode::default())?;
            // Candidate 2: the float 9/7 ratio attack (lossy-only). Auto-pick the
            // lower-PRD reconstruction at the matched rate ceiling.
            match crate::lmo_pcrd97::encode_target_bps_97(signal, bps, LpcMode::default()) {
                Ok(b97) => {
                    let prd53 = prd(signal, &lml::decompress(&i53)?);
                    let pick_97 = match crate::lmo_pcrd97::decode_97(&b97) {
                        Ok(r97) => prd(signal, &r97) < prd53,
                        Err(_) => false,
                    };
                    if pick_97 {
                        (TRANSFORM_LMO_97, b97)
                    } else {
                        (TRANSFORM_LML_53, i53)
                    }
                }
                Err(_) => (TRANSFORM_LML_53, i53),
            }
        }
        other => (
            TRANSFORM_LML_53,
            lamquant_lml_mcu::codec::LmlCodec.encode(signal, other)?,
        ),
    };

    let mut out = Vec::with_capacity(LMO_HEADER_LEN + inner.len());
    out.extend_from_slice(LMO_MAGIC.as_slice());
    out.push(LMO_VERSION);
    out.push(mode_tag(mode));
    out.push(transform_id);
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
    // bytes[5] = mode_tag, informational. bytes[6] = transform_id routes the body.
    let transform_id = bytes[6];
    let inner = &bytes[LMO_HEADER_LEN..];
    match transform_id {
        TRANSFORM_LML_53 => lml::decompress(inner).map_err(Into::into),
        TRANSFORM_LMO_97 => crate::lmo_pcrd97::decode_97(inner).map_err(Into::into),
        other => Err(CodecError::Lml(LmlError::InvalidHeader(alloc::format!(
            "unknown LMO transform_id 0x{other:02X}"
        )))),
    }
}

/// Universal magic-dispatch decode for a build that has the LMO decoder linked.
///
/// Routes LML streams to the integer floor and LMO streams here. This is the
/// "full" dispatch (no [`CodecError::OptimumNotInstalled`]) the Desktop profile
/// — and a Firmware build that opted into LMO decode — exposes, in contrast to
/// `lamquant_lml_mcu::codec::decode` which returns *not installed* for LMO.
pub fn decode_any(bytes: &[u8]) -> Result<Signal, CodecError> {
    match lamquant_lml_mcu::codec::peek_format(bytes) {
        Some(Format::Lml) => lml::decompress(bytes).map_err(Into::into),
        Some(Format::Lmo) => decode(bytes),
        None => Err(CodecError::UnknownFormat),
    }
}
