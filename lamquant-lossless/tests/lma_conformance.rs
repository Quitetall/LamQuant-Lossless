//! LMA archive conformance — integration tests.
//!
//! Complement to the unit tests in `src/lma.rs::tests`. These tests exercise
//! the public archive API end-to-end on synthetic file trees that mix every
//! compression method (Lml / Zstd / Store) plus deeply-nested directory
//! layouts. Drift here means an archive produced by one revision can no
//! longer be read by another, or that integrity checks pass when they
//! should not.
//!
//! The tests deliberately avoid fabricating EDF data — the LML codec path
//! is exercised via `tests/codec/cross_lang/test_lma_cross_lang.py` once
//! the PyO3 wheel is available. Here we focus on the archive container,
//! method dispatch, and SHA chain.

use lamquant_core::lma::{
    choose_method, list_archive, pack_archive, pack_lml_with_siblings, unpack_archive, Method,
    SiblingEntryKind,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn write_tree(root: &Path, files: &[(&str, &[u8])]) {
    for (rel, data) in files {
        let full = root.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, data).unwrap();
    }
}

// ─── 1. Round-trip on a mixed-method tree ──────────────────────────

#[test]
fn pack_unpack_byte_exact_on_mixed_methods() {
    let src = tempfile::tempdir().unwrap();
    let files: Vec<(&str, &[u8])> = vec![
        ("notes.csv", b"a,b,c\n1,2,3\n"),           // → Zstd
        ("blob.bin", &[0x42u8; 256]),               // → Zstd (unknown ext)
        ("dir/sub.zst", &[0xCCu8; 128]),            // → Store (already-compressed)
        ("dir/sub/deeper.txt", b"hello, archive!"), // → Zstd
        ("packaged.zip", b"PK\x03\x04..."),         // → Store
    ];
    write_tree(src.path(), &files);

    let archive = tempfile::NamedTempFile::new().unwrap();
    let summary = pack_archive(src.path(), archive.path(), 9, false, None).unwrap();
    assert_eq!(summary.n_files, files.len());
    assert!(summary.archive_bytes > 0);
    assert!(
        summary.errors.is_empty(),
        "pack reported errors: {:?}",
        summary.errors
    );

    // Method selection must follow choose_method on each file.
    let entries = list_archive(archive.path()).unwrap();
    assert_eq!(entries.len(), files.len());
    for entry in &entries {
        let expected = choose_method(Path::new(&entry.path));
        assert_eq!(
            entry.method, expected,
            "{}: archived method {:?} != choose_method {:?}",
            entry.path, entry.method, expected
        );
    }

    // Method count tallies match summary.
    let lml_n = entries.iter().filter(|e| e.method == Method::Lml).count();
    let zstd_n = entries.iter().filter(|e| e.method == Method::Zstd).count();
    let store_n = entries.iter().filter(|e| e.method == Method::Store).count();
    assert_eq!(lml_n, summary.counts_lml);
    assert_eq!(zstd_n, summary.counts_zstd);
    assert_eq!(store_n, summary.counts_store);

    // Per-entry SHA-256 must match SHA of the actual input bytes.
    for (rel, data) in &files {
        let entry = entries.iter().find(|e| e.path == *rel).unwrap();
        assert_eq!(entry.sha256, sha256_hex(data), "SHA mismatch for {}", rel);
        assert_eq!(entry.original_size as usize, data.len());
    }

    // Unpack into a fresh tempdir and compare every file byte-exactly.
    let dst = tempfile::tempdir().unwrap();
    let unpack = unpack_archive(archive.path(), dst.path(), true, false, None).unwrap();
    assert_eq!(unpack.n_files, files.len());
    assert!(
        unpack.errors.is_empty(),
        "unpack reported errors: {:?}",
        unpack.errors
    );
    for (rel, data) in &files {
        let recovered = fs::read(dst.path().join(rel)).unwrap();
        assert_eq!(&recovered[..], *data, "byte drift for {}", rel);
    }
}

// ─── 2. Manifest is non-empty, deterministic, sorted ───────────────

#[test]
fn manifest_lists_files_in_deterministic_sorted_order() {
    let src = tempfile::tempdir().unwrap();
    write_tree(
        src.path(),
        &[
            ("z_last.txt", b"z"),
            ("a_first.txt", b"a"),
            ("dir/b.txt", b"b"),
            ("dir/a.txt", b"a"),
        ],
    );

    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(src.path(), archive.path(), 9, false, None).unwrap();
    let entries = list_archive(archive.path()).unwrap();

    let names: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        names, sorted,
        "manifest entries not in sorted order: {:?}",
        names
    );
}

// ─── 3. Empty / weird-content files round-trip ─────────────────────

#[test]
fn empty_and_unicode_files_roundtrip() {
    let src = tempfile::tempdir().unwrap();
    write_tree(
        src.path(),
        &[
            ("empty.txt", b""),
            ("unicode_\u{2713}.txt", "✓ ok ✓".as_bytes()),
            ("nul_in_data.bin", b"\x00\x01\x02\x00\x00\x03"),
            ("trailing_newline.txt", b"hello\n"),
        ],
    );

    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

    let dst = tempfile::tempdir().unwrap();
    unpack_archive(archive.path(), dst.path(), true, false, None).unwrap();

    assert_eq!(fs::read(dst.path().join("empty.txt")).unwrap(), b"");
    assert_eq!(
        fs::read(dst.path().join("unicode_\u{2713}.txt")).unwrap(),
        "✓ ok ✓".as_bytes()
    );
    assert_eq!(
        fs::read(dst.path().join("nul_in_data.bin")).unwrap(),
        b"\x00\x01\x02\x00\x00\x03"
    );
    assert_eq!(
        fs::read(dst.path().join("trailing_newline.txt")).unwrap(),
        b"hello\n"
    );
}

// ─── 4. Nested directories preserved ───────────────────────────────

#[test]
fn nested_directory_paths_preserved() {
    let src = tempfile::tempdir().unwrap();
    write_tree(
        src.path(),
        &[
            ("a/b/c/d/deep.txt", b"deep"),
            ("a/b/peer.txt", b"peer"),
            ("a/sibling.txt", b"sibling"),
        ],
    );

    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

    let dst = tempfile::tempdir().unwrap();
    unpack_archive(archive.path(), dst.path(), true, false, None).unwrap();

    assert_eq!(
        fs::read(dst.path().join("a/b/c/d/deep.txt")).unwrap(),
        b"deep"
    );
    assert_eq!(fs::read(dst.path().join("a/b/peer.txt")).unwrap(), b"peer");
    assert_eq!(
        fs::read(dst.path().join("a/sibling.txt")).unwrap(),
        b"sibling"
    );
}

// ─── 5. Archive-level SHA chain catches header tampering ───────────

#[test]
fn archive_sha_chain_catches_header_byte_flip() {
    let src = tempfile::tempdir().unwrap();
    write_tree(src.path(), &[("data.bin", &[0x55u8; 128])]);

    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

    // Flip a byte in the manifest region (after the 16-byte magic+version+count
    // header but well before the trailing 32-byte archive SHA-256).
    let mut bytes = fs::read(archive.path()).unwrap();
    let flip_idx = 20.min(bytes.len() - 33);
    bytes[flip_idx] ^= 0x01;
    fs::write(archive.path(), &bytes).unwrap();

    let dst = tempfile::tempdir().unwrap();
    let result = unpack_archive(archive.path(), dst.path(), true, false, None);
    let detected = match result {
        Err(_) => true,
        Ok(summary) => !summary.errors.is_empty(),
    };
    assert!(
        detected,
        "manifest-region byte flip was NOT detected by verify=true"
    );
}

// ─── 6. Per-entry SHA-256 catches payload tampering ────────────────

#[test]
fn per_entry_sha_catches_payload_byte_flip() {
    let src = tempfile::tempdir().unwrap();
    let payload = vec![0xA5u8; 1024];
    write_tree(src.path(), &[("victim.bin", &payload)]);

    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(src.path(), archive.path(), 9, false, None).unwrap();

    // Flip a byte in the back half (likely inside the entry payload).
    let mut bytes = fs::read(archive.path()).unwrap();
    let flip_idx = bytes.len() - 64; // 64 bytes back from the trailing archive-SHA
    bytes[flip_idx] ^= 0x10;
    fs::write(archive.path(), &bytes).unwrap();

    let dst = tempfile::tempdir().unwrap();
    let result = unpack_archive(archive.path(), dst.path(), true, false, None);
    let detected = match result {
        Err(_) => true,
        Ok(summary) => !summary.errors.is_empty(),
    };
    assert!(
        detected,
        "payload byte flip was NOT detected by verify=true"
    );
}

// ─── 7. choose_method dispatch matches every archived entry ────────

#[test]
fn choose_method_matches_archived_method_by_extension() {
    let cases: Vec<(&str, Method)> = vec![
        ("a.csv", Method::Zstd),
        ("a.json", Method::Zstd),
        ("a.md", Method::Zstd),
        ("no_ext_file", Method::Zstd),
        ("a.zip", Method::Store),
        ("a.png", Method::Store),
        ("a.mp4", Method::Store),
        ("a.lma", Method::Store),
        ("a.gz", Method::Store),
    ];
    for (name, expected) in &cases {
        let got = choose_method(Path::new(name));
        assert_eq!(
            got, *expected,
            "{}: expected {:?}, got {:?}",
            name, expected, got
        );
    }
}

// ─── 8. pack_lml_with_siblings — copy-only path round trip ─────────
//
// The LML-encode path delegates to `encode_edf_to_lml`, which is
// already locked by byte_equal_backends + the PyO3 cross-lang
// fixtures. The new code surface in pack_lml_with_siblings is the
// SIBLING-COPY + manifest-write path. Exercise that here with a tree
// that has no EDFs so we don't need to fabricate one. End-to-end:
// inputs SHA-match outputs, manifest contains every entry, counts
// add up.

#[test]
fn pack_lml_with_siblings_copies_non_edf_byte_exact() {
    let src = tempfile::tempdir().unwrap();
    let files: Vec<(&str, &[u8])> = vec![
        ("notes.csv", b"a,b,c\n1,2,3\n"), // Zstd-method but copied here
        ("dir/labels.tse", b"0.0 1.0 seiz\n1.0 2.0 bckg"), // Zstd-method
        ("dir/sub/blob.bin", &[0x42u8; 256]), // Zstd-method (unknown ext)
        ("packaged.zip", b"PK\x03\x04..."), // Store-method
    ];
    write_tree(src.path(), &files);

    let dst = tempfile::tempdir().unwrap();
    // Pre-existing tempdir is empty -> function must accept it.
    let summary = pack_lml_with_siblings(src.path(), dst.path(), false, None).expect("pack ok");

    assert_eq!(summary.counts_lml, 0, "no EDFs in fixture");
    assert_eq!(summary.counts_copied, files.len());
    assert!(
        summary.errors.is_empty(),
        "unexpected errors: {:?}",
        summary.errors
    );

    // Every input file must exist at the mirrored output position
    // with byte-exact SHA-256.
    for (rel, data) in &files {
        let out = dst.path().join(rel);
        let recovered = fs::read(&out).unwrap_or_else(|_| panic!("missing: {}", rel));
        assert_eq!(&recovered[..], *data, "byte drift for {}", rel);
        let entry = summary.entries.iter().find(|e| e.src_rel == *rel).unwrap();
        assert!(matches!(entry.kind, SiblingEntryKind::Copied));
        assert_eq!(entry.sha256, sha256_hex(data));
        assert_eq!(entry.original_size as usize, data.len());
        assert_eq!(entry.written_size as usize, data.len());
    }

    // Manifest must exist and reference every entry.
    let manifest = fs::read_to_string(dst.path().join("MANIFEST.json")).expect("manifest");
    for (rel, _) in &files {
        assert!(
            manifest.contains(rel),
            "manifest missing entry for {}: {}",
            rel,
            manifest,
        );
    }
}

#[test]
fn pack_lml_with_siblings_refuses_non_empty_output() {
    let src = tempfile::tempdir().unwrap();
    write_tree(src.path(), &[("a.txt", b"hello")]);
    let dst = tempfile::tempdir().unwrap();
    // Drop a file in dst -- pack must refuse rather than clobber.
    fs::write(dst.path().join("existing.txt"), b"do not lose me").unwrap();

    let result = pack_lml_with_siblings(src.path(), dst.path(), false, None);
    assert!(
        result.is_err(),
        "pack should refuse non-empty output; got {:?}",
        result.map(|s| s.counts_copied),
    );
    // Existing file must still be there -- function shouldn't have
    // touched it.
    let existing = fs::read(dst.path().join("existing.txt")).unwrap();
    assert_eq!(&existing[..], b"do not lose me");
}

#[test]
fn pack_lml_with_siblings_refuses_empty_input() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let result = pack_lml_with_siblings(src.path(), dst.path(), false, None);
    assert!(
        result.is_err(),
        "pack should reject empty input dir; got {:?}",
        result.map(|s| s.counts_copied),
    );
}

/// ADR 0023 Track A-3: a `.txt` file matching the ASCII int-per-line
/// shape is routed through the ingest pipeline (synthesize 1-channel
/// EDF + run LML codec) instead of the zstd fallback. The extracted
/// bytes must match the original byte-for-byte.
///
/// Source: checked-in fixture at
/// `specs/conformance/ascii_vectors/bonn_z_like_4097s.txt` (~20 KiB,
/// 4097 samples, CRLF-terminated — same record length as real Bonn
/// files). Smaller fixtures get caught by the "ingest must strictly
/// beat zstd" rule because the LML header overhead exceeds the
/// gains on tiny files.
#[test]
fn ingest_ascii_int_lines_roundtrip() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("specs")
        .join("conformance")
        .join("ascii_vectors")
        .join("bonn_z_like_4097s.txt");
    assert!(fixture.exists(), "fixture not found: {}", fixture.display());
    let original_bytes = fs::read(&fixture).expect("read fixture");
    let original_sha = sha256_hex(&original_bytes);

    // Build a directory with the fixture in it for pack_archive.
    let src_dir = tempfile::tempdir().unwrap();
    let src_file = src_dir.path().join("Z001.txt"); // Bonn naming → 173.61 Hz hint
    fs::write(&src_file, &original_bytes).unwrap();

    let archive = tempfile::Builder::new()
        .prefix("ingest_ascii_")
        .suffix(".lma")
        .tempfile()
        .unwrap();
    let summary = pack_archive(src_dir.path(), archive.path(), 9, false, None).expect("pack");

    // pack_archive picks the smaller of (zstd, ingest-LML) per the
    // "no regression" rule (ADR 0023). For this particular fixture
    // either path may win depending on the codec version (the byte-
    // equal backends gate keeps the choice deterministic within a
    // build, but smoothness of the synthesised signal means zstd
    // can edge out LML by tens of bytes here). The deterministic
    // ingest-wins case is covered by
    // `ingest_fires_when_lml_strictly_beats_zstd`. Here we only
    // assert the SHA-256 byte-roundtrip — the safety invariant
    // independent of method choice.
    assert_eq!(
        summary.counts_lml + summary.counts_zstd + summary.counts_store,
        1,
        "expected exactly 1 entry, got counts: lml={} zstd={} store={}",
        summary.counts_lml,
        summary.counts_zstd,
        summary.counts_store
    );

    // Extract and verify the bytes survived intact.
    let dst = tempfile::tempdir().unwrap();
    let unpack_summary =
        unpack_archive(archive.path(), dst.path(), true, false, None).expect("unpack");
    assert!(
        unpack_summary.errors.is_empty(),
        "unpack reported errors: {:?}",
        unpack_summary.errors
    );

    let extracted_bytes = fs::read(dst.path().join("Z001.txt")).expect("read extracted");
    assert_eq!(
        sha256_hex(&extracted_bytes),
        original_sha,
        "extracted bytes don't match original SHA-256"
    );
    assert_eq!(
        extracted_bytes,
        original_bytes,
        "byte-for-byte roundtrip mismatch (len {} vs {})",
        extracted_bytes.len(),
        original_bytes.len()
    );
}

/// Force-exercise the ingest path by generating ASCII content where
/// LPC strictly beats zstd (linear ramp — LPC-1 predicts perfectly,
/// residuals all zero; zstd has to encode thousands of unique values).
/// The fixture is deliberately long enough to amortize the authenticated
/// ABIR/BCS2 semantic envelope. This validates that try_ingest_to_lml is
/// wired in and produces a valid LML payload when it wins, and that
/// the extract path re-emits byte-exact ASCII.
#[test]
fn ingest_fires_when_lml_strictly_beats_zstd() {
    let src = tempfile::tempdir().unwrap();
    let path = src.path().join("Z001.txt"); // matches Bonn naming → 173.61 Hz hint

    // Linear ramp 0..32767 with CRLF. Each line is a different int
    // (zstd-hostile, LPC predictable). Range in i16 trivially.
    let mut buf = Vec::with_capacity(32768 * 8);
    for i in 0..32768_i32 {
        buf.extend_from_slice(format!("{}\r\n", i).as_bytes());
    }
    let original_sha = sha256_hex(&buf);
    fs::write(&path, &buf).unwrap();

    let archive = tempfile::Builder::new()
        .prefix("ingest_force_")
        .suffix(".lma")
        .tempfile()
        .unwrap();
    let summary = pack_archive(src.path(), archive.path(), 9, false, None).expect("pack");
    assert_eq!(
        summary.counts_lml, 1,
        "ingest should fire on linear-ramp ASCII (LPC-friendly, zstd-hostile); got counts: \
         lml={} zstd={} store={}",
        summary.counts_lml, summary.counts_zstd, summary.counts_store
    );

    // Confirm the manifest entry carries synthetic_from.
    let entries = list_archive(archive.path()).expect("list");
    assert_eq!(entries.len(), 1);
    let sf = entries[0]
        .synthetic_from
        .as_ref()
        .expect("synthetic_from must be set when ingest fires");
    assert_eq!(sf.format, "ascii_int_lines");
    assert!((sf.sample_rate - 173.61).abs() < 1e-9);

    // Extract + byte-exact roundtrip.
    let dst = tempfile::tempdir().unwrap();
    let unpack_summary =
        unpack_archive(archive.path(), dst.path(), true, false, None).expect("unpack");
    assert!(
        unpack_summary.errors.is_empty(),
        "unpack errors: {:?}",
        unpack_summary.errors
    );
    let extracted = fs::read(dst.path().join("Z001.txt")).expect("read extracted");
    assert_eq!(
        sha256_hex(&extracted),
        original_sha,
        "byte-roundtrip SHA mismatch"
    );
    assert_eq!(extracted, buf);
}

/// Files that don't look like ASCII int-per-line (random binary noise,
/// other text patterns) must NOT trigger the ingest path. They stay
/// on the zstd fallback so the archive doesn't pay synth+encode cost
/// for nothing.
#[test]
fn ingest_skips_non_matching_files() {
    let src = tempfile::tempdir().unwrap();
    fs::write(
        src.path().join("README.txt"),
        b"This is a readme.\nNot integers.\nLicense info.\n",
    )
    .unwrap();
    fs::write(
        src.path().join("data.bin"),
        &[0u8, 1, 2, 3, 4, 5, 0xFF, 0xFE],
    )
    .unwrap();

    let archive = tempfile::Builder::new()
        .prefix("ingest_skip_")
        .suffix(".lma")
        .tempfile()
        .unwrap();
    let summary = pack_archive(src.path(), archive.path(), 9, false, None).expect("pack");
    // Both files should go through zstd, not LML.
    assert_eq!(summary.counts_lml, 0);
    assert!(summary.counts_zstd + summary.counts_store == 2);
}
