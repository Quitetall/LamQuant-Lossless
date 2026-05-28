//! Adversarial-input safety tests for the EDF/BDF reader.
//!
//! ADR 0021 Phase 3 (L1/L2): tests that exercise the typed-error
//! paths added to `crate::edf::read_edf`. Each fixture is a
//! hand-crafted 256+ byte buffer written to a tempfile; the
//! reader is expected to reject the file with an `InvalidHeader`
//! diagnostic that includes the specific reason.
//!
//! Tests that PASSED against the pre-ADR-0021 code (i.e. would
//! have silently accepted the malformed input) are marked in the
//! commit message so they serve as regression guards if anyone
//! tries to re-introduce the silent-fallback pattern.

use lamquant_core::edf::read_edf;
use lamquant_core::error::LmlError;
use std::io::Write;

/// Helper: write `bytes` to a tempfile and return the path.
fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut tf = tempfile::NamedTempFile::new().unwrap();
    tf.write_all(bytes).unwrap();
    tf.flush().unwrap();
    tf
}

/// Build a 256-byte EDF main header padded with spaces. The EDF
/// spec requires fixed-width ASCII fields; this helper makes the
/// padding explicit so tests can override one field at a time.
fn ascii_header(version: &str) -> Vec<u8> {
    let mut h = Vec::with_capacity(256);
    // version: 8 chars, "0       " for EDF
    let mut v = version.as_bytes().to_vec();
    v.resize(8, b' ');
    h.extend_from_slice(&v);
    // patient_id, recording_id: 80 + 80 = 160, all spaces
    h.resize(8 + 80 + 80, b' ');
    // startdate (8), starttime (8)
    h.resize(8 + 80 + 80 + 8 + 8, b' ');
    // header_bytes (8), reserved (44), n_data_records (8), dur_record (8), n_signals (4)
    h.resize(256, b' ');
    h
}

#[test]
fn read_edf_rejects_bad_bdf_magic() {
    // 0xFF prefix without "BIOSEMI" used to be silently accepted
    // as BDF by the broken slice comparison at edf.rs:74. Now
    // requires bytes 1-7 to spell "BIOSEMI".
    let mut bytes = vec![0xFFu8];
    bytes.extend_from_slice(b"GARBAGE"); // 7 bytes after the 0xFF prefix
    bytes.resize(256, b' ');
    let tf = write_tmp(&bytes);
    let result = read_edf(tf.path());
    match result {
        Err(LmlError::InvalidHeader(msg)) => {
            assert!(
                msg.contains("BIOSEMI") || msg.contains("not a valid BDF"),
                "wrong error message: {}",
                msg
            );
        }
        Err(e) => panic!("expected InvalidHeader, got {}", e),
        Ok(_) => panic!("expected InvalidHeader, got Ok"),
    }
}

#[test]
fn read_edf_rejects_utf16_bom_as_bdf() {
    // UTF-16 LE BOM is 0xFF 0xFE. Old code claimed to catch this
    // but the slice compare was always-false. Now the BIOSEMI
    // check covers it.
    let mut bytes = vec![0xFF, 0xFE];
    bytes.extend(b"OTHER  "); // pad to 8 bytes (NOT BIOSEMI)
    bytes.resize(256, b' ');
    let tf = write_tmp(&bytes);
    let result = read_edf(tf.path());
    let ok = matches!(result, Err(LmlError::InvalidHeader(_)));
    assert!(ok, "UTF-16 BOM should be rejected as bad BDF magic");
}

#[test]
fn read_edf_rejects_file_too_small() {
    // Anything under the 256-byte main header is malformed.
    let tf = write_tmp(b"too short");
    let result = read_edf(tf.path());
    assert!(result.is_err(), "tiny file should fail");
}
