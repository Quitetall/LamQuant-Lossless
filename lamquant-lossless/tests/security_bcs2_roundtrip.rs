//! Encryption round-trips through the BCS2 privacy envelope.
//!
//! What is pinned here is not "encrypt then decrypt returns the input" — that is
//! the easy part and any AEAD gives it. The properties that matter are the ones
//! the container change is *for*:
//!
//! 1. The entire header is authenticated. Under `LMLCRYPT` the magic and version
//!    were unauthenticated preamble, so tampering there produced a parse error
//!    rather than an authentication failure and the difference between "corrupt"
//!    and "attacked" was unrecoverable.
//! 2. `EncryptedOpaque` genuinely discloses nothing about the plaintext.
//! 3. `EncryptedDiscoverable` discloses exactly what it promises and no more.
//! 4. A wrong key fails loudly rather than returning plausible bytes.
//! 5. Wrapping a non-BCS2 file is invisible on the way back out, and does not
//!    swallow a user's own blob.

#![cfg(all(feature = "security", feature = "archive"))]

use lamquant_core::security::Key;
use lamquant_core::security_bcs2;
use semantic_abir_bcs::{encode_blob, PrivacyMode, ResourceBounds};

fn key(seed: u8) -> Key {
    Key::from_bytes([seed; 32])
}

/// A non-BCS2 payload with a recognisable marker, so a leak is detectable by
/// searching the envelope for it.
const SECRET: &[u8] = b"PATIENT-0042-SEIZURE-ONSET-0913Z-CONFIDENTIAL";

#[test]
fn an_arbitrary_file_round_trips_byte_for_byte() {
    let envelope =
        security_bcs2::encrypt_bytes(SECRET, &key(1), PrivacyMode::EncryptedOpaque).unwrap();
    let recovered = security_bcs2::decrypt_bytes(&envelope, &key(1)).unwrap();
    assert_eq!(recovered, SECRET, "the wrapper must be invisible");
}

#[test]
fn a_bcs2_artifact_round_trips_without_being_wrapped() {
    let blob = encode_blob(
        b"signal bytes",
        "application/octet-stream",
        ResourceBounds::default(),
    )
    .unwrap();
    let envelope =
        security_bcs2::encrypt_bytes(&blob, &key(2), PrivacyMode::EncryptedOpaque).unwrap();
    let recovered = security_bcs2::decrypt_bytes(&envelope, &key(2)).unwrap();

    assert_eq!(
        recovered, blob,
        "a user's own blob must decrypt back to the blob, not to its payload"
    );
}

#[test]
fn an_opaque_envelope_discloses_nothing_about_the_plaintext() {
    let envelope =
        security_bcs2::encrypt_bytes(SECRET, &key(3), PrivacyMode::EncryptedOpaque).unwrap();

    assert!(
        !envelope
            .windows(SECRET.len())
            .any(|window| window == SECRET),
        "plaintext must not appear in the envelope"
    );
    // The header's disclosure fields must be genuinely empty, not merely
    // unread: profile at [16..20], root kind at [40], root content id at
    // [96..128].
    assert_eq!(&envelope[16..20], &[0, 0, 0, 0], "profile must not leak");
    assert_eq!(envelope[40], 0, "root kind must not leak");
    assert!(
        envelope[96..128].iter().all(|byte| *byte == 0),
        "the inner content id is a stable fingerprint of a specific recording; \
         an opaque envelope must not publish it"
    );
    assert_eq!(
        security_bcs2::envelope_privacy_mode(&envelope),
        Some(PrivacyMode::EncryptedOpaque)
    );
}

#[test]
fn a_discoverable_envelope_publishes_exactly_what_it_promises() {
    let blob = encode_blob(
        SECRET,
        "application/octet-stream",
        ResourceBounds::default(),
    )
    .unwrap();
    let envelope =
        security_bcs2::encrypt_bytes(&blob, &key(4), PrivacyMode::EncryptedDiscoverable).unwrap();

    assert_eq!(
        security_bcs2::envelope_privacy_mode(&envelope),
        Some(PrivacyMode::EncryptedDiscoverable)
    );
    assert_ne!(
        &envelope[96..128],
        &[0_u8; 32],
        "discoverable must publish the inner root content id"
    );
    // Disclosure is metadata only: the plaintext itself still must not appear.
    assert!(
        !envelope
            .windows(SECRET.len())
            .any(|window| window == SECRET),
        "discoverable discloses metadata, never content"
    );
    assert_eq!(
        security_bcs2::decrypt_bytes(&envelope, &key(4)).unwrap(),
        blob
    );
}

#[test]
fn the_wrong_key_is_refused_rather_than_returning_plausible_bytes() {
    let envelope =
        security_bcs2::encrypt_bytes(SECRET, &key(5), PrivacyMode::EncryptedOpaque).unwrap();
    assert!(
        security_bcs2::decrypt_bytes(&envelope, &key(6)).is_err(),
        "a wrong key must fail authentication, not produce garbage"
    );
}

#[test]
fn tampering_with_the_header_is_an_authentication_failure() {
    // The property the container change buys. The whole header is AEAD
    // associated data, so a flipped byte anywhere in it is detected as
    // tampering rather than shrugged off as a parse quirk.
    let envelope =
        security_bcs2::encrypt_bytes(SECRET, &key(7), PrivacyMode::EncryptedOpaque).unwrap();

    // Byte 42 is the privacy mode: flip Opaque to Discoverable. Under a format
    // with an unauthenticated preamble this would be a free downgrade of the
    // artifact's own disclosure claim.
    let mut tampered = envelope.clone();
    tampered[42] = PrivacyMode::EncryptedDiscoverable as u8;
    assert!(
        security_bcs2::decrypt_bytes(&tampered, &key(7)).is_err(),
        "the declared privacy mode must be authenticated, not advisory"
    );
}

#[test]
fn tampering_with_the_ciphertext_is_an_authentication_failure() {
    let envelope =
        security_bcs2::encrypt_bytes(SECRET, &key(8), PrivacyMode::EncryptedOpaque).unwrap();
    let mut tampered = envelope.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    assert!(security_bcs2::decrypt_bytes(&tampered, &key(8)).is_err());
}

#[test]
fn two_encryptions_of_one_plaintext_differ() {
    // Nonces are random per encryption. Identical envelopes would mean a fixed
    // nonce, which for a stream cipher leaks the XOR of two plaintexts.
    let first =
        security_bcs2::encrypt_bytes(SECRET, &key(9), PrivacyMode::EncryptedOpaque).unwrap();
    let second =
        security_bcs2::encrypt_bytes(SECRET, &key(9), PrivacyMode::EncryptedOpaque).unwrap();
    assert_ne!(first, second, "nonce reuse would be catastrophic here");
}

#[test]
fn files_round_trip_and_are_recognised_as_encrypted() {
    let dir = tempfile::tempdir().unwrap();
    let plain = dir.path().join("recording.edf");
    let sealed = dir.path().join("recording.edf.bcs2");
    let restored = dir.path().join("restored.edf");
    std::fs::write(&plain, SECRET).unwrap();

    security_bcs2::encrypt_file(&plain, &sealed, &key(10), PrivacyMode::EncryptedOpaque).unwrap();
    assert!(security_bcs2::is_encrypted(&sealed));
    assert!(
        !security_bcs2::is_encrypted(&plain),
        "a plain file must not be mistaken for an envelope"
    );

    security_bcs2::decrypt_file(&sealed, &restored, &key(10)).unwrap();
    assert_eq!(std::fs::read(&restored).unwrap(), SECRET);
}

#[test]
fn an_lmlcrypt_file_is_not_mistaken_for_a_bcs2_envelope() {
    // The retired container must stay distinguishable, since both are reachable
    // during the transition and picking the wrong reader would report a
    // decryption failure for a perfectly good file.
    let lmlcrypt = lamquant_core::security::encrypt_aes_gcm(&key(11), SECRET).unwrap();
    assert!(security_bcs2::envelope_privacy_mode(&lmlcrypt).is_none());
}
