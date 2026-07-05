//! #28 — the LamQuant conformance vectors (`specs/conformance/vectors/`) run in
//! Rust, porting `specs/conformance/verify.py`. Decode-forever: each vector is a
//! committed LML1 `.lml` + an `.expected.json`. Positive vectors (`expected_error_kind`
//! null) must decode OK with the specified shape; negative (corruption) vectors must
//! decode-ERROR with the specified kind. The committed `.lml` bytes are pinned by
//! sha256. This is the third-party-reader conformance the spec invites, now guarding
//! the Rust reader (decode-forever) in CI.
#![cfg(feature = "archive")]

use lamquant_core::container;
use lamquant_core::error::LmlError;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specs/conformance/vectors")
}

fn sha(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

fn error_kind(e: &LmlError) -> &'static str {
    match e {
        LmlError::CrcMismatch { .. } => "CrcMismatch",
        LmlError::Truncated { .. } => "Truncated",
        LmlError::InvalidMagic(_) => "InvalidMagic",
        LmlError::UnsupportedVersion(_) => "UnsupportedVersion",
        LmlError::InvalidHeader(_) => "InvalidHeader",
        // No vector expects any other variant; a real mismatch is still caught by
        // the assert below (a new kind maps to "Other", failing != the expected kind).
        _ => "Other",
    }
}

#[test]
fn conformance_vectors_decode_forever_as_specified() {
    let dir = vectors_dir();
    let mut checked = 0usize;
    for entry in std::fs::read_dir(&dir).expect("conformance vectors dir") {
        let path = entry.expect("read dir entry").path();
        let name_os = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name_os.ends_with(".expected.json") {
            continue; // skip .lml, manifest.lml.json, etc.
        }
        let spec: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).expect("valid vector json");
        let name = spec["name"].as_str().unwrap();
        let lml = std::fs::read(dir.join(format!("{name}.lml"))).expect("vector .lml");

        // 1. The committed .lml bytes are the byte-exact golden.
        assert_eq!(sha(&lml), spec["lml_sha256"].as_str().unwrap(), "{name}: .lml sha drifted");

        // 2. Decode behavior matches the spec.
        let expected_err = spec["expected_error_kind"].as_str();
        match (container::read_bytes(&lml), expected_err) {
            (Ok((signal, _meta)), None) => {
                assert_eq!(
                    signal.len(),
                    spec["n_channels"].as_u64().unwrap() as usize,
                    "{name}: n_channels"
                );
                // total_samples in the vectors is the PER-CHANNEL sample count.
                assert_eq!(
                    signal[0].len(),
                    spec["total_samples"].as_u64().unwrap() as usize,
                    "{name}: per-channel total_samples"
                );
            }
            (Err(e), Some(kind)) => {
                assert_eq!(error_kind(&e), kind, "{name}: expected error {kind}, got {e:?}");
            }
            (Ok(_), Some(kind)) => panic!("{name}: expected error {kind}, but decoded OK"),
            (Err(e), None) => panic!("{name}: expected OK decode, got error {e:?}"),
        }
        checked += 1;
    }
    // Exact count: a vector renamed/moved (so the glob stops matching it) must fail
    // here rather than silently reduce coverage. Bump this when a vector is added.
    assert_eq!(checked, 13, "expected exactly 13 conformance vectors, ran {checked}");
}
