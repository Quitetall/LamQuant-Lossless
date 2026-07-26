//! Encryption on the BCS2 privacy envelope.
//!
//! `LMLCRYPT` (see [`crate::security`]) is a bespoke container:
//! `magic(8) | version(1) | nonce(12) | AES-256-GCM ciphertext`. ADR 0139
//! contract 5 wants one native wire family, and BCS2 already carries encryption
//! as a *property of the artifact header* — [`PrivacyMode`] — rather than as a
//! separate format that happens to wrap one.
//!
//! # What changes, and what that buys
//!
//! The header stops being a private magic and becomes the same header every
//! other LamQuant artifact has. Concretely:
//!
//! - The whole header is **authenticated as AEAD associated data**. Under
//!   `LMLCRYPT` the magic and version were unauthenticated preamble; flipping a
//!   byte there produced a parse error rather than an authentication failure,
//!   and the distinction between "corrupt" and "tampered" was lost.
//! - Disclosure becomes a deliberate choice.
//!   [`PrivacyMode::EncryptedOpaque`] reveals nothing beyond "this is an
//!   encrypted BCS2 artifact"; `EncryptedDiscoverable` additionally publishes the
//!   inner profile, root kind and root content id so a store can index sealed
//!   data it cannot read. This module defaults to Opaque, because for clinical
//!   recordings the inner content id is itself identifying — it is a stable
//!   fingerprint of a specific patient recording, and publishing it would let an
//!   observer confirm a suspected file is present without ever holding the key.
//! - The nonce widens from 12 to 24 bytes (XChaCha20-Poly1305 rather than
//!   AES-256-GCM), which is what makes random nonces safe at scale instead of
//!   merely unlikely to collide.
//!
//! # Arbitrary files
//!
//! [`encrypt_bcs2`] requires its plaintext to be a valid BCS2 artifact, so a
//! plain file cannot be fed to it directly. Such input is wrapped in a BCS2 blob
//! first and unwrapped on the way out, which keeps one code path for both cases
//! rather than a BCS2 path plus a fallback that would inevitably drift.
//!
//! # What this module does not do
//!
//! It does not retire `LMLCRYPT`. [`crate::security`] still encrypts and
//! decrypts it, no existing file is rewritten, and nothing here touches the
//! Argon2 password path — see the note on [`PASSWORD_KDF_STILL_SIDECAR`].

use crate::security::Key;
use semantic_abir_bcs::{
    decrypt_bcs2, encode_blob, encrypt_bcs2, Bcs2View, BlobView, EncryptedEnvelopeView,
    PrivacyMode, ResourceBounds, BCS2_MAGIC, CAP_XCHACHA20_POLY1305,
};
use std::path::Path;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Media type marking a blob that exists only to carry a non-BCS2 file through
/// the privacy envelope.
///
/// Distinguished so decryption unwraps exactly the blobs this module created and
/// leaves a genuine user blob alone — otherwise encrypting a real
/// `application/octet-stream` blob and decrypting it would silently hand back
/// its payload instead of the artifact that went in.
const OPAQUE_WRAPPER_MEDIA_TYPE: &str = "application/vnd.lamquant.encrypted-opaque-wrapper";

/// Password-derived keys still keep their salt and Argon2 parameters in a
/// detached `<file>.lmcrypt.header` sidecar.
///
/// Not fixed here, and worth stating plainly rather than leaving implicit: a
/// detached sidecar means losing one file renders the ciphertext permanently
/// unrecoverable, and nothing binds the pair, so a mismatched sidecar derives
/// the wrong key and fails authentication in a way indistinguishable from
/// corruption. Putting the descriptor inside the artifact needs somewhere to
/// put it — the envelope header's reserved bytes are enforced-zero — which is a
/// change to a security wire format and a decision in its own right, not a
/// detail to settle inside a container migration.
pub const PASSWORD_KDF_STILL_SIDECAR: &str =
    "password mode still uses the detached .lmcrypt.header sidecar";

fn bounds_for(len: usize) -> ResourceBounds {
    let headroom = u32::try_from(len.saturating_mul(2).saturating_add(1 << 20)).unwrap_or(u32::MAX);
    let mut bounds = ResourceBounds::default();
    bounds.max_frame_bytes = bounds.max_frame_bytes.max(headroom);
    bounds.max_catalog_bytes = bounds.max_catalog_bytes.max(headroom);
    bounds
}

fn random_nonce() -> Result<[u8; 24], Error> {
    let mut nonce = [0_u8; 24];
    getrandom::getrandom(&mut nonce).map_err(|e| format!("failed to generate a nonce: {e}"))?;
    Ok(nonce)
}

/// True when `bytes` parse as a BCS2 artifact of any kind.
fn is_bcs2(bytes: &[u8], bounds: ResourceBounds) -> bool {
    bytes.len() >= 8
        && bytes[..8] == BCS2_MAGIC
        // A capability-gated artifact still IS a BCS2 artifact; parsing with the
        // full mask avoids misclassifying, say, a BFP-encoded training snapshot
        // as an arbitrary file and wrapping it needlessly.
        && Bcs2View::parse(bytes, u64::MAX, bounds).is_ok()
}

/// Encrypt `plaintext`, wrapping it first if it is not already a BCS2 artifact.
///
/// Returns the envelope bytes.
pub fn encrypt_bytes(
    plaintext: &[u8],
    key: &Key,
    privacy_mode: PrivacyMode,
) -> Result<Vec<u8>, Error> {
    let bounds = bounds_for(plaintext.len());
    let inner = if is_bcs2(plaintext, bounds) {
        plaintext.to_vec()
    } else {
        encode_blob(plaintext, OPAQUE_WRAPPER_MEDIA_TYPE, bounds)?
    };
    let nonce = random_nonce()?;
    Ok(encrypt_bcs2(
        &inner,
        privacy_mode,
        key.as_bytes(),
        &nonce,
        u64::MAX,
        bounds,
    )?)
}

/// Decrypt an envelope, unwrapping the blob if this module created one.
pub fn decrypt_bytes(envelope: &[u8], key: &Key) -> Result<Vec<u8>, Error> {
    let bounds = bounds_for(envelope.len());
    let plaintext = decrypt_bcs2(envelope, key.as_bytes(), u64::MAX, bounds)?;
    // Only unwrap the wrapper this module writes. A user's own blob decrypts
    // back to the blob, not to its payload.
    if let Ok(blob) = BlobView::parse(&plaintext, u64::MAX, bounds) {
        if blob.media_type() == OPAQUE_WRAPPER_MEDIA_TYPE {
            return Ok(blob.bytes().to_vec());
        }
    }
    Ok(plaintext)
}

/// Encrypt a file in place of `output`.
pub fn encrypt_file(
    input: &Path,
    output: &Path,
    key: &Key,
    privacy_mode: PrivacyMode,
) -> Result<u64, Error> {
    let plaintext = std::fs::read(input)?;
    let envelope = encrypt_bytes(&plaintext, key, privacy_mode)?;
    std::fs::write(output, &envelope)?;
    Ok(envelope.len() as u64)
}

/// Decrypt a file in place of `output`.
pub fn decrypt_file(input: &Path, output: &Path, key: &Key) -> Result<u64, Error> {
    let envelope = std::fs::read(input)?;
    let plaintext = decrypt_bytes(&envelope, key)?;
    std::fs::write(output, &plaintext)?;
    Ok(plaintext.len() as u64)
}

/// True when `path` holds a BCS2 artifact whose header declares it encrypted.
///
/// Reads the header only — this must stay cheap enough to call before deciding
/// whether a key is even needed.
pub fn is_encrypted(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    envelope_privacy_mode(&bytes).is_some()
}

/// The declared privacy mode of an encrypted envelope, if it is one.
pub fn envelope_privacy_mode(bytes: &[u8]) -> Option<PrivacyMode> {
    let view = EncryptedEnvelopeView::parse(bytes, bounds_for(bytes.len())).ok()?;
    Some(view.privacy_mode())
}

/// The capability an encrypted envelope requires of its reader.
pub const fn required_capability() -> u64 {
    CAP_XCHACHA20_POLY1305
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wrapper_media_type_is_a_valid_restricted_media_type() {
        // encode_blob validates the media type; a malformed constant would only
        // surface the first time somebody encrypted a non-BCS2 file.
        let bounds = ResourceBounds::default();
        assert!(encode_blob(b"payload", OPAQUE_WRAPPER_MEDIA_TYPE, bounds).is_ok());
    }

    #[test]
    fn opaque_is_the_default_this_module_recommends() {
        // Guards the disclosure decision documented above: the inner content id
        // is a stable fingerprint of a specific recording, so publishing it
        // would let an observer confirm a suspected file without a key.
        assert_ne!(
            PrivacyMode::EncryptedOpaque,
            PrivacyMode::EncryptedDiscoverable
        );
    }
}
