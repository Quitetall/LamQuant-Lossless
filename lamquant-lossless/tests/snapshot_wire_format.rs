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
use lamquant_core::lpc::LpcMode;
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
    fs::write(
        staging.path().join("hello.txt"),
        b"deterministic content for wire-format snapshot",
    )
    .expect("write");

    let archive: PathBuf = staging.path().join("out.lma");
    lma::pack_archive(staging.path(), &archive, 9, false, None).expect("pack_archive");

    let bytes = fs::read(&archive).expect("read .lma");
    // 12 bytes = magic(4) + version_le(4) + entry_count_le(4). The writer
    // now emits the LMA v2 streaming format: magic "LMA2", version 2, and a
    // header entry_count of 0 — in v2 the manifest/entry index moved to the
    // EOCD footer (single-pass, 1x-disk pack), so the header count field is
    // no longer populated. (Snapshot was stale at v1 — the `--lib`-only CI
    // never exercised this integration suite; the widened suite does.)
    insta::assert_snapshot!(
        "lma_v2_magic_version_count_12_bytes",
        hex_prefix(&bytes, 12)
    );
}

/// ADR 0139/0143 — snapshot the deterministic 40-byte prefix of the current
/// canonical ABIR/BCS2 codec bundle. Deterministic inputs: a fixed synthetic
/// signal, 250 Hz, 256-sample windows, `noise_bits=0`, `LpcMode::default()`
/// (`Anytime{deadline:None}`, never reads a clock), `"{}"` metadata (so
/// the semantic envelope is stable). Packet bytes are pinned separately.
#[test]
fn abir_bcs2_header_40_bytes() {
    let sig: Vec<Vec<i64>> = vec![(0..600i64).map(|t| ((t * 37) % 4001) - 2000).collect()];
    let mut sink = Vec::new();
    lamquant_core::container::write_into(&mut sink, &sig, 250.0, 256, 0, "{}", LpcMode::default())
        .expect("write_into");
    insta::assert_snapshot!("abir_bcs2_header_40_bytes", hex_prefix(&sink, 40));
}
