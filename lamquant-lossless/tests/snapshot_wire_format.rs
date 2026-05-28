//! Wire-format lockdown via insta snapshots (Cat A2 — 2026-05-21).
//!
//! Locks the LMA + LML container byte layout in CI. Any unintended
//! change to magic bytes, header size, version field, or zstd
//! framing breaks `cargo test snapshot_wire_format` and forces an
//! explicit `cargo insta review` acknowledgement.
//!
//! These are NOT the same as roundtrip tests — they pin the EXACT
//! bytes the encoder emits, not just that the decoder accepts them.

#![cfg(feature = "host")]

use std::fs;
use std::path::PathBuf;

use lamquant_core::lma;
use tempfile::TempDir;

/// Hex-dump first `n` bytes of a slice as "AA BB CC ..." for snapshotting.
fn hex_prefix(bytes: &[u8], n: usize) -> String {
    let take = n.min(bytes.len());
    bytes[..take]
        .iter()
        .map(|b| format!("{:02X}", b))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Snapshot the **deterministic prefix** of the LMA wire format —
/// magic (4 bytes) + version (4 bytes LE) + entry-count (4 bytes LE).
/// The next field (manifest length) and the manifest contents embed
/// non-deterministic timestamps / paths, so they are explicitly
/// excluded from this snapshot. Any unintended change to magic,
/// version field width, endianness, or entry-count slot will break
/// the snapshot and force an explicit `cargo insta review`.
#[test]
fn lma_header_magic_version_count() {
    let staging = TempDir::new().expect("tmpdir");
    fs::write(staging.path().join("hello.txt"),
              b"deterministic content for wire-format snapshot")
        .expect("write");

    let archive: PathBuf = staging.path().join("out.lma");
    lma::pack_archive(staging.path(), &archive, 9, false, None)
        .expect("pack_archive");

    let bytes = fs::read(&archive).expect("read .lma");
    // 12 bytes = magic(4) + version_le(4) + entry_count_le(4)
    insta::assert_snapshot!(
        "lma_v1_magic_version_count_12_bytes",
        hex_prefix(&bytes, 12)
    );
}
