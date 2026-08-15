//! LMA directory archives round-trip through the BCS2 forensic-capsule wire.
//!
//! The claim under test is narrow and total: **every archived file comes back
//! byte-for-byte**, across all three storage forms, with metadata intact — and
//! a reader that cannot decode a frame is refused rather than handed bytes it
//! will misinterpret.
//!
//! These tests exercise the real encoder cascade, not a mock. `Method::Store`,
//! `Method::Zstd` and `Method::Lml` are selected by file extension exactly as
//! they are in production, so the fixture directory is built to force all three.

#![cfg(feature = "archive")]

use lamquant_core::lma_forensic::{self, READER_CAPABILITIES};
use semantic_abir_bcs::{
    encode_forensic_tree, raw_content_id, Bcs2Error, ForensicContentTransform, ForensicEntry,
    ForensicFileType, ForensicTree, ForensicTreeView, ForensicXattr, LmaSyntheticLineEnding,
    LmaSyntheticReemitParametersV1, ResourceBounds, CAP_LMA_SYNTHETIC_REEMIT,
    CAP_LML1_LEGACY_MATERIALIZE, CAP_LML_LOSSLESS_V1, CAP_ZSTD,
};
use std::collections::BTreeMap;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::time::Duration;

/// A minimal but structurally valid EDF the LML encoder will accept.
///
/// Built rather than checked in so the test stays readable, and so a header
/// field change breaks compilation here instead of silently producing a file
/// that quietly falls back to zstd — which would make the `Method::Lml` arm of
/// this test vacuous while still passing.
fn synthetic_edf(channels: usize, samples_per_record: usize, records: usize) -> Vec<u8> {
    fn pad(field: &str, width: usize) -> Vec<u8> {
        let mut out = field.as_bytes().to_vec();
        out.resize(width, b' ');
        out
    }

    let mut edf = Vec::new();
    edf.extend(pad("0", 8)); // version
    edf.extend(pad("X X X X", 80)); // patient
    edf.extend(pad("Startdate 01-JAN-2026 X X X", 80)); // recording
    edf.extend(pad("01.01.26", 8)); // start date
    edf.extend(pad("00.00.00", 8)); // start time
    let header_bytes = 256 + 256 * channels;
    edf.extend(pad(&header_bytes.to_string(), 8));
    edf.extend(pad("EDF+C", 44)); // reserved
    edf.extend(pad(&records.to_string(), 8));
    edf.extend(pad("1", 8)); // record duration, seconds
    edf.extend(pad(&channels.to_string(), 4));

    for channel in 0..channels {
        edf.extend(pad(&format!("EEG C{channel}"), 16));
    }
    for _ in 0..channels {
        edf.extend(pad("AgAgCl", 80));
    }
    for _ in 0..channels {
        edf.extend(pad("uV", 8));
    }
    for _ in 0..channels {
        edf.extend(pad("-32768", 8)); // physical min
    }
    for _ in 0..channels {
        edf.extend(pad("32767", 8)); // physical max
    }
    for _ in 0..channels {
        edf.extend(pad("-32768", 8)); // digital min
    }
    for _ in 0..channels {
        edf.extend(pad("32767", 8)); // digital max
    }
    for _ in 0..channels {
        edf.extend(pad("HP:0.1Hz LP:75Hz", 80));
    }
    for _ in 0..channels {
        edf.extend(pad(&samples_per_record.to_string(), 8));
    }
    for _ in 0..channels {
        edf.extend(pad("", 32)); // reserved
    }

    // Deterministic, mildly correlated samples so the codec has real structure
    // to work with rather than noise.
    for record in 0..records {
        for channel in 0..channels {
            for sample in 0..samples_per_record {
                let phase = (record * samples_per_record + sample) as f64 / 16.0;
                let value = ((phase + channel as f64).sin() * 1000.0) as i16;
                edf.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    edf
}

/// Fixture forcing every storage form plus a nested directory.
///
/// Returns the tempdir and the exact bytes written, keyed by relative path, so
/// the round-trip can be asserted against the originals rather than against a
/// re-read of the same tree.
fn fixture() -> (tempfile::TempDir, BTreeMap<String, Vec<u8>>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let mut expected = BTreeMap::new();

    // Method::Lml — .edf routes through the codec.
    let edf = synthetic_edf(2, 64, 4);
    std::fs::create_dir_all(root.join("recordings")).unwrap();
    std::fs::write(root.join("recordings/session.edf"), &edf).unwrap();
    expected.insert("recordings/session.edf".to_string(), edf);

    // Method::Zstd — an unknown extension with compressible content.
    let notes = "subject notes\n".repeat(400).into_bytes();
    std::fs::write(root.join("recordings/notes.txt"), &notes).unwrap();
    expected.insert("recordings/notes.txt".to_string(), notes);

    // Method::Store — .zst is on the already-compressed list.
    let opaque: Vec<u8> = (0..4096u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    std::fs::write(root.join("blob.zst"), &opaque).unwrap();
    expected.insert("blob.zst".to_string(), opaque);

    // An empty file, historically a good source of off-by-one bugs.
    std::fs::write(root.join("recordings/empty.txt"), b"").unwrap();
    expected.insert("recordings/empty.txt".to_string(), Vec::new());

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            root.join("recordings"),
            std::fs::Permissions::from_mode(0o710),
        )
        .unwrap();
    }

    (dir, expected)
}

fn capsule_of(root: &Path, out: &Path) -> lamquant_core::lma::ArchiveSummary {
    lma_forensic::pack_directory(root, out, 3, false, None).expect("pack to capsule")
}

fn owned_capsule(platform: &str, mut entry: ForensicEntry) -> Vec<u8> {
    entry.path = b"entry.bin".to_vec();
    encode_forensic_tree(
        &ForensicTree {
            platform: platform.to_owned(),
            entries: vec![entry],
        },
        ResourceBounds::default(),
    )
    .expect("encode owned capsule")
}

fn owned_regular(content: &[u8]) -> ForensicEntry {
    ForensicEntry {
        path: Vec::new(),
        file_type: ForensicFileType::Regular,
        mode: 0o644,
        owner: None,
        timestamps: [None; 4],
        acl: None,
        xattrs: Vec::new(),
        hardlink_target: None,
        symlink_target: None,
        sparse_extents: Vec::new(),
        flags: 0,
        device: None,
        special_type: None,
        content: Some(content.to_vec()),
        content_transform: None,
    }
}

#[test]
fn every_storage_form_round_trips_byte_for_byte() {
    let (source, expected) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("archive.bcs2");

    let summary = capsule_of(source.path(), &capsule);
    assert_eq!(summary.n_files, expected.len(), "every file archived");
    assert!(
        summary.errors.is_empty(),
        "pack reported errors: {:?}",
        summary.errors
    );
    let verified = lma_forensic::verify_capsule(&capsule, false).expect("verify capsule");
    assert_eq!(verified.n_files, expected.len());
    assert_eq!(verified.original_bytes, summary.original_bytes);

    let restored = work.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    let unpacked = lma_forensic::unpack_capsule(&capsule, &restored, false).expect("unpack");
    assert!(
        unpacked.errors.is_empty(),
        "unpack reported errors: {:?}",
        unpacked.errors
    );
    assert_eq!(unpacked.n_files, expected.len());

    for (rel, original) in &expected {
        let actual = std::fs::read(restored.join(rel))
            .unwrap_or_else(|e| panic!("restored {rel} unreadable: {e}"));
        assert_eq!(
            &actual, original,
            "{rel} did not survive the capsule round trip byte-for-byte"
        );
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(restored.join("recordings"))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o710
        );
    }
}

#[test]
fn all_three_storage_forms_are_actually_exercised() {
    // Without this the round-trip test could pass while every entry silently
    // fell back to a single method, leaving two thirds of the mapping untested.
    let (source, _) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("archive.bcs2");
    let summary = capsule_of(source.path(), &capsule);

    assert!(summary.counts_lml >= 1, "no entry took the LML path");
    assert!(summary.counts_zstd >= 1, "no entry took the zstd path");
    assert!(summary.counts_store >= 1, "no entry was stored verbatim");
}

#[test]
fn the_custody_anchor_is_the_file_not_the_stored_bytes() {
    // The property the whole mapping exists to preserve.
    let (source, expected) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("archive.bcs2");
    capsule_of(source.path(), &capsule);

    let bytes = std::fs::read(&capsule).unwrap();
    let bounds = ResourceBounds {
        max_frame_bytes: u32::MAX,
        max_catalog_bytes: u32::MAX,
        ..ResourceBounds::default()
    };
    let view =
        ForensicTreeView::parse(&bytes, READER_CAPABILITIES, bounds).expect("capsule parses");

    for entry in view.entries() {
        if entry.file_type != semantic_abir_bcs::ForensicFileType::Regular {
            continue;
        }
        let rel = String::from_utf8_lossy(&entry.path).into_owned();
        let original = expected
            .get(&rel)
            .unwrap_or_else(|| panic!("unknown {rel}"));

        assert_eq!(
            entry.content_id,
            Some(semantic_abir_bcs::raw_content_id(original)),
            "{rel}: content_id must identify the FILE"
        );
        assert_eq!(
            entry.content_len,
            Some(original.len() as u64),
            "{rel}: content_len must be the file's length"
        );
        assert!(
            entry.xattrs.is_empty(),
            "{rel}: internal metadata leaked into xattrs"
        );

        if let Some(stored) = entry.stored_form {
            assert_ne!(
                stored.stored_content_id,
                entry.content_id.unwrap(),
                "{rel}: a transformed entry must not claim the file's own id for its frame"
            );
            assert_ne!(stored.capabilities, 0);
            if stored.capabilities & CAP_LML_LOSSLESS_V1 != 0 {
                assert_eq!(
                    stored.capabilities & CAP_LML1_LEGACY_MATERIALIZE,
                    0,
                    "freshly packed LML must not claim retired LML1 materialization"
                );
            }
        }
    }
}

#[test]
fn a_reader_missing_a_capability_is_refused_rather_than_misled() {
    let (source, _) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("archive.bcs2");
    capsule_of(source.path(), &capsule);

    let bytes = std::fs::read(&capsule).unwrap();
    let bounds = ResourceBounds {
        max_frame_bytes: u32::MAX,
        max_catalog_bytes: u32::MAX,
        ..ResourceBounds::default()
    };

    // The archive contains both LML and zstd frames, so a reader offering
    // neither must be refused at the envelope.
    match ForensicTreeView::parse(&bytes, 0, bounds) {
        Err(Bcs2Error::UnsupportedCapabilities(missing)) => {
            assert_ne!(missing & (CAP_ZSTD | CAP_LML_LOSSLESS_V1), 0);
        }
        other => panic!("expected a capability refusal, got {other:?}"),
    }

    // Offering only zstd is still not enough: the LML frame remains unreadable.
    assert!(
        matches!(
            ForensicTreeView::parse(&bytes, CAP_ZSTD, bounds),
            Err(Bcs2Error::UnsupportedCapabilities(_))
        ),
        "a partially equipped reader must not be admitted"
    );
}

#[test]
fn listing_reports_the_same_files_and_sizes_the_archive_holds() {
    let (source, expected) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("archive.bcs2");
    capsule_of(source.path(), &capsule);

    let inspection = lma_forensic::inspect_capsule(&capsule).expect("inspect");
    assert_eq!(inspection.root_content_id.to_string().len(), 64);
    assert_eq!(
        inspection.content_domain,
        lma_forensic::FORENSIC_CAPSULE_CONTENT_DOMAIN
    );
    let listed = inspection.entries;
    assert_eq!(listed.len(), expected.len());
    for entry in &listed {
        let original = expected
            .get(&entry.path)
            .unwrap_or_else(|| panic!("listed unknown path {}", entry.path));
        assert_eq!(
            entry.original_size,
            original.len() as u64,
            "{}: listed size must be the file's size, not the frame's",
            entry.path
        );
        assert!(
            entry.content_id.is_some(),
            "{}: logical ContentId must survive conversion",
            entry.path
        );
    }
}

#[test]
fn converting_an_lma_archive_preserves_every_file() {
    // The migration path for archives that already exist. Non-destructive: the
    // source archive must still be readable afterwards.
    let (source, expected) = fixture();
    let work = tempfile::tempdir().unwrap();
    let lma = work.path().join("archive.lma");
    lamquant_core::lma::pack_archive(source.path(), &lma, 3, false, None).expect("pack lma");

    let capsule = work.path().join("archive.bcs2");
    let summary = lma_forensic::convert_archive(&lma, &capsule, 3).expect("convert");
    assert!(
        summary.errors.is_empty(),
        "conversion reported errors: {:?}",
        summary.errors
    );
    assert_eq!(summary.n_files, expected.len());

    // Source untouched.
    let still_listed = lamquant_core::lma::list_archive(&lma).expect("source still readable");
    assert_eq!(still_listed.len(), expected.len());

    // Migration preserves physical stored frames too; verification decodes
    // them only to prove logical-file hashes before carrying their extents.
    let capsule_bytes = std::fs::read(&capsule).unwrap();
    let bounds = ResourceBounds {
        max_frame_bytes: u32::MAX,
        max_catalog_bytes: u32::MAX,
        ..ResourceBounds::default()
    };
    let view = ForensicTreeView::parse(&capsule_bytes, READER_CAPABILITIES, bounds).unwrap();
    let mut source_archive = lamquant_core::lma::LmaArchive::open(&lma).unwrap();
    for source_entry in source_archive.entries().to_vec() {
        let target_entry = view
            .entries()
            .iter()
            .find(|entry| entry.path == source_entry.path.as_bytes())
            .unwrap_or_else(|| panic!("converted capsule missing {}", source_entry.path));
        assert_eq!(
            view.stored_bytes(target_entry).unwrap(),
            source_archive.read_stored(&source_entry.path).unwrap(),
            "{} stored frame changed during conversion",
            source_entry.path
        );
    }

    let restored = work.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    lma_forensic::unpack_capsule(&capsule, &restored, false).expect("unpack converted");
    for (rel, original) in &expected {
        let actual = std::fs::read(restored.join(rel))
            .unwrap_or_else(|e| panic!("converted {rel} unreadable: {e}"));
        assert_eq!(&actual, original, "{rel} did not survive LMA → capsule");
    }
}

#[test]
fn conversion_preserves_empty_directories_and_refuses_existing_output() {
    let (source, _) = fixture();
    let empty = source.path().join("empty/nested");
    std::fs::create_dir_all(&empty).unwrap();
    let expected_mtime = 1_700_000_123_i64;
    filetime::set_file_mtime(
        &empty,
        filetime::FileTime::from_unix_time(expected_mtime, 0),
    )
    .unwrap();

    let work = tempfile::tempdir().unwrap();
    let lma = work.path().join("archive.lma");
    lamquant_core::lma::pack_archive(source.path(), &lma, 3, false, None).unwrap();
    let capsule = work.path().join("archive.bcs2");
    lma_forensic::convert_archive(&lma, &capsule, 3).unwrap();

    let bytes = std::fs::read(&capsule).unwrap();
    let view = ForensicTreeView::parse(
        &bytes,
        READER_CAPABILITIES,
        ResourceBounds {
            max_frame_bytes: u32::MAX,
            max_catalog_bytes: u32::MAX,
            ..ResourceBounds::default()
        },
    )
    .unwrap();
    let nested = view
        .entries()
        .iter()
        .find(|entry| entry.path == b"empty/nested")
        .expect("empty directory retained");
    assert_eq!(
        nested.timestamps[1].map(|stamp| stamp.seconds),
        Some(expected_mtime)
    );

    let restored = work.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    lma_forensic::unpack_capsule(&capsule, &restored, false).unwrap();
    assert!(restored.join("empty/nested").is_dir());
    let restored_mtime = filetime::FileTime::from_last_modification_time(
        &std::fs::metadata(restored.join("empty/nested")).unwrap(),
    );
    assert_eq!(restored_mtime.unix_seconds(), expected_mtime);

    let sentinel = work.path().join("sentinel.bcs2");
    std::fs::write(&sentinel, b"caller-owned").unwrap();
    assert!(lma_forensic::convert_archive(&lma, &sentinel, 3).is_err());
    assert_eq!(std::fs::read(&sentinel).unwrap(), b"caller-owned");
}

#[test]
fn a_capsule_is_distinguishable_from_an_lma_container() {
    let (source, _) = fixture();
    let work = tempfile::tempdir().unwrap();
    let lma = work.path().join("archive.lma");
    let capsule = work.path().join("archive.bcs2");
    lamquant_core::lma::pack_archive(source.path(), &lma, 3, false, None).unwrap();
    capsule_of(source.path(), &capsule);

    assert!(lma_forensic::is_capsule(&capsule));
    assert!(
        !lma_forensic::is_capsule(&lma),
        "an LMA container must not be mistaken for a capsule"
    );
}

#[test]
fn direct_pack_never_clobbers_an_existing_capsule_path() {
    let (source, _) = fixture();
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("existing.bcs2");
    std::fs::write(&capsule, b"keep-me").unwrap();

    let error = lma_forensic::pack_directory(source.path(), &capsule, 3, false, None)
        .expect_err("packing must publish with no-clobber semantics");

    assert!(
        error.to_string().contains("exists"),
        "unexpected error: {error}"
    );
    assert_eq!(std::fs::read(capsule).unwrap(), b"keep-me");
}

#[test]
fn direct_pack_is_byte_deterministic_across_snapshot_staging() {
    let (source, _) = fixture();
    let work = tempfile::tempdir().unwrap();
    let first = work.path().join("first.bcs2");
    let second = work.path().join("second.bcs2");

    capsule_of(source.path(), &first);
    capsule_of(source.path(), &second);

    assert_eq!(
        std::fs::read(first).unwrap(),
        std::fs::read(second).unwrap()
    );
}

#[test]
fn exact_restore_requires_existing_empty_destination() {
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("owned.bcs2");
    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    std::fs::write(
        &capsule,
        owned_capsule(&platform, owned_regular(b"payload")),
    )
    .unwrap();
    let missing = work.path().join("missing");

    let error = lma_forensic::unpack_capsule(&capsule, &missing, false).unwrap_err();
    assert!(error.to_string().contains("must already exist"));
    assert!(!missing.exists());
}

#[test]
fn exact_restore_preflight_rejects_unreproducible_metadata_without_writes() {
    let work = tempfile::tempdir().unwrap();
    let capsule = work.path().join("xattr.bcs2");
    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let mut entry = owned_regular(b"payload");
    entry.xattrs.push(ForensicXattr {
        name: b"user.example".to_vec(),
        value: b"value".to_vec(),
    });
    std::fs::write(&capsule, owned_capsule(&platform, entry)).unwrap();
    let restored = work.path().join("restored");
    std::fs::create_dir(&restored).unwrap();

    let error = lma_forensic::unpack_capsule(&capsule, &restored, false).unwrap_err();
    assert!(error.to_string().contains("cannot reproduce"));
    assert!(std::fs::read_dir(restored).unwrap().next().is_none());
}

#[test]
fn exact_restore_preflight_rejects_platform_and_storage_contradictions() {
    let work = tempfile::tempdir().unwrap();
    let foreign = work.path().join("foreign.bcs2");
    std::fs::write(
        &foreign,
        owned_capsule("foreign-platform", owned_regular(b"payload")),
    )
    .unwrap();
    let verified = lma_forensic::verify_capsule(&foreign, false)
        .expect("logical verification must not require local metadata restoration");
    assert_eq!(verified.n_files, 1);
    let foreign_out = work.path().join("foreign-out");
    std::fs::create_dir(&foreign_out).unwrap();
    let error = lma_forensic::unpack_capsule(&foreign, &foreign_out, false).unwrap_err();
    assert!(error.to_string().contains("platform mismatch"));
    assert!(std::fs::read_dir(foreign_out).unwrap().next().is_none());

    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let contradictory = work.path().join("contradictory.bcs2");
    let stored = b"encoded-payload";
    let logical = b"logical-payload";
    let mut entry = owned_regular(stored);
    entry.content_transform = Some(ForensicContentTransform::new(
        CAP_ZSTD | CAP_LML_LOSSLESS_V1,
        raw_content_id(logical),
        logical.len() as u64,
    ));
    std::fs::write(&contradictory, owned_capsule(&platform, entry)).unwrap();
    let contradictory_out = work.path().join("contradictory-out");
    std::fs::create_dir(&contradictory_out).unwrap();
    let error =
        lma_forensic::unpack_capsule(&contradictory, &contradictory_out, false).unwrap_err();
    assert!(error.to_string().contains("conflicting storage methods"));
    assert!(std::fs::read_dir(contradictory_out)
        .unwrap()
        .next()
        .is_none());
}

#[cfg(target_os = "linux")]
fn legacy_v1_lml(path: &str, stored: &[u8], original: &[u8]) -> Vec<u8> {
    legacy_v1_lml_with_synthetic(path, stored, original, None)
}

#[cfg(target_os = "linux")]
fn legacy_v1_lml_with_synthetic(
    path: &str,
    stored: &[u8],
    original: &[u8],
    synthetic_from: Option<serde_json::Value>,
) -> Vec<u8> {
    use sha2::{Digest, Sha256};

    let manifest = serde_json::json!({
        "compressor": "zstd",
        "files": [{
            "path": path,
            "original_size": original.len(),
            "compressed_size": stored.len(),
            "method": "lml",
            "sha256": format!("{:x}", Sha256::digest(original)),
            "offset": 0,
            "synthetic_from": synthetic_from,
        }]
    });
    let manifest = zstd::encode_all(serde_json::to_vec(&manifest).unwrap().as_slice(), 3).unwrap();
    let mut archive = Vec::new();
    archive.extend_from_slice(b"LMA1");
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
    archive.extend_from_slice(&manifest);
    archive.extend_from_slice(stored);
    let digest = Sha256::digest(&archive);
    archive.extend_from_slice(&digest);
    archive
}

#[cfg(target_os = "linux")]
fn fake_legacy_adapter(root: &Path, stored: &[u8], original: &[u8]) -> std::path::PathBuf {
    use base64::Engine;
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::PermissionsExt;

    let script = root.join("fake-legacy-adapter.py");
    let calls = root.join("adapter-calls.log");
    let output = base64::engine::general_purpose::STANDARD.encode(original);
    let source_blake3 = blake3::hash(stored).to_hex();
    let output_sha256 = format!("{:x}", Sha256::digest(original));
    let body = format!(
        r#"#!/usr/bin/env python3
import base64, json, pathlib, sys
request = json.load(sys.stdin)
with open(r"{}", "a", encoding="utf-8") as calls:
    calls.write(request["operation"] + "\n")
if request["operation"] == "manifest":
    print(json.dumps({{
        "status": "ok-manifest",
        "value": {{
            "schema": "lamquant.legacy-capabilities/v1",
            "process_protocol": "abir.adapter-process/v1",
            "capabilities": [{{
                "profile": "legacy.lml1.v1",
                "parent_verified_materialization": True
            }}]
        }}
    }}))
    raise SystemExit(0)
pathlib.Path(request["destination"]).write_bytes(base64.b64decode("{output}"))
print(json.dumps({{
    "status": "ok-materialization",
    "value": {{
        "profile": "legacy.lml1.v1",
        "source_blake3": "{source_blake3}",
        "source_bytes": {},
        "output_sha256": "{output_sha256}",
        "output_bytes": {},
        "source_preserved": True,
        "exact_original_bytes": request.get("expected_sha256") is not None
    }}
}}))
"#,
        calls.display(),
        stored.len(),
        original.len(),
    );
    std::fs::write(&script, body).unwrap();
    let mut permissions = std::fs::metadata(&script).unwrap().permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&script, permissions).unwrap();
    script
}

#[cfg(target_os = "linux")]
#[test]
fn retired_lml1_conversion_and_restore_use_supervised_adapter_process() {
    let root = tempfile::tempdir().unwrap();
    let original = b"exact retired EDF bytes with header and trailing data";
    let stored = b"LML1-retired-process-fixture";
    let archive = root.path().join("retired.lma");
    std::fs::write(&archive, legacy_v1_lml("recording.edf", stored, original)).unwrap();
    let source_before = std::fs::read(&archive).unwrap();
    let adapter = fake_legacy_adapter(root.path(), stored, original);
    let config = lma_forensic::LegacyAdapterConfig {
        executable: adapter,
        timeout: Duration::from_secs(5),
        max_rss_bytes: 512 * 1024 * 1024,
    };

    let capsule = root.path().join("retired.bcs2");
    let summary = lma_forensic::convert_archive_with_legacy_config(&archive, &capsule, 3, &config)
        .expect("supervised conversion");
    assert!(summary.errors.is_empty());
    assert_eq!(summary.counts_lml, 1);
    assert_eq!(std::fs::read(&archive).unwrap(), source_before);

    let capsule_bytes = std::fs::read(&capsule).unwrap();
    let bounds = ResourceBounds {
        max_frame_bytes: u32::MAX,
        max_catalog_bytes: u32::MAX,
        ..ResourceBounds::default()
    };
    let view = ForensicTreeView::parse(&capsule_bytes, READER_CAPABILITIES, bounds).unwrap();
    let entry = view
        .entries()
        .iter()
        .find(|entry| entry.path == b"recording.edf")
        .unwrap();
    assert_ne!(
        entry.required_capabilities() & CAP_LML1_LEGACY_MATERIALIZE,
        0
    );
    assert!(matches!(
        ForensicTreeView::parse(
            &capsule_bytes,
            READER_CAPABILITIES & !CAP_LML1_LEGACY_MATERIALIZE,
            bounds
        ),
        Err(Bcs2Error::UnsupportedCapabilities(_))
    ));

    let restored = root.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    let unpacked =
        lma_forensic::unpack_capsule_with_legacy_config(&capsule, &restored, false, &config)
            .expect("supervised restore");
    assert!(
        unpacked.errors.is_empty(),
        "restore errors: {:?}",
        unpacked.errors
    );
    assert_eq!(
        std::fs::read(restored.join("recording.edf")).unwrap(),
        original
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("adapter-calls.log")).unwrap(),
        "materialize-exact\nmanifest\nmaterialize-exact\n"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn retired_candidate_receipt_cannot_bypass_parent_content_identity() {
    let root = tempfile::tempdir().unwrap();
    let original = b"exact retired EDF bytes with header and trailing data";
    let stored = b"LML1-retired-process-fixture";
    let archive = root.path().join("retired.lma");
    std::fs::write(&archive, legacy_v1_lml("recording.edf", stored, original)).unwrap();
    let good_config = lma_forensic::LegacyAdapterConfig {
        executable: fake_legacy_adapter(root.path(), stored, original),
        timeout: Duration::from_secs(5),
        max_rss_bytes: 512 * 1024 * 1024,
    };
    let capsule = root.path().join("retired.bcs2");
    lma_forensic::convert_archive_with_legacy_config(&archive, &capsule, 3, &good_config)
        .expect("build authenticated capsule");

    let mut wrong = original.to_vec();
    wrong[0] ^= 0x01;
    let bad_config = lma_forensic::LegacyAdapterConfig {
        executable: fake_legacy_adapter(root.path(), stored, &wrong),
        timeout: Duration::from_secs(5),
        max_rss_bytes: 512 * 1024 * 1024,
    };
    let restored = root.path().join("wrong-restored");
    std::fs::create_dir(&restored).unwrap();
    let error =
        lma_forensic::unpack_capsule_with_legacy_config(&capsule, &restored, false, &bad_config)
            .unwrap_err();
    assert!(error.to_string().contains("archived content id"));
    assert!(!restored.join("recording.edf").exists());
}

#[cfg(target_os = "linux")]
#[test]
fn retired_restore_refuses_adapter_without_parent_verification_capability() {
    let root = tempfile::tempdir().unwrap();
    let original = b"exact retired EDF bytes with header and trailing data";
    let stored = b"LML1-retired-process-fixture";
    let archive = root.path().join("retired.lma");
    std::fs::write(&archive, legacy_v1_lml("recording.edf", stored, original)).unwrap();
    let adapter = fake_legacy_adapter(root.path(), stored, original);
    let config = lma_forensic::LegacyAdapterConfig {
        executable: adapter.clone(),
        timeout: Duration::from_secs(5),
        max_rss_bytes: 512 * 1024 * 1024,
    };
    let capsule = root.path().join("retired.bcs2");
    lma_forensic::convert_archive_with_legacy_config(&archive, &capsule, 3, &config)
        .expect("build authenticated capsule");
    let script = std::fs::read_to_string(&adapter).unwrap().replace(
        "\"parent_verified_materialization\": True",
        "\"parent_verified_materialization\": False",
    );
    std::fs::write(&adapter, script).unwrap();

    let restored = root.path().join("unsupported-restored");
    std::fs::create_dir(&restored).unwrap();
    let error =
        lma_forensic::unpack_capsule_with_legacy_config(&capsule, &restored, false, &config)
            .unwrap_err();
    assert!(error
        .to_string()
        .contains("does not advertise parent-verified"));
    assert!(std::fs::read_dir(restored).unwrap().next().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn conversion_fails_without_publishing_when_adapter_cannot_start() {
    let root = tempfile::tempdir().unwrap();
    let original = b"exact retired bytes";
    let stored = b"LML1-retired-process-fixture";
    let archive = root.path().join("retired.lma");
    std::fs::write(&archive, legacy_v1_lml("recording.edf", stored, original)).unwrap();
    let output = root.path().join("must-not-exist.bcs2");
    let config = lma_forensic::LegacyAdapterConfig {
        executable: root.path().join("missing-adapter"),
        timeout: Duration::from_secs(1),
        max_rss_bytes: 128 * 1024 * 1024,
    };

    assert!(
        lma_forensic::convert_archive_with_legacy_config(&archive, &output, 3, &config).is_err()
    );
    assert!(!output.exists());
}

#[cfg(target_os = "linux")]
#[test]
fn retired_synthetic_lml1_uses_explicit_exact_re_emission_operation() {
    let root = tempfile::tempdir().unwrap();
    let original = b"-12\r\n0\r\n34";
    let stored = b"LML1-retired-synthetic-fixture";
    let archive = root.path().join("synthetic.lma");
    let synthetic = serde_json::json!({
        "format": "ascii_int_lines",
        "sample_rate": 250.0,
        "template": {
            "line_ending": "CrLf",
            "leading_whitespace": 0,
            "field_width": 0,
            "trailing_newline": false
        }
    });
    std::fs::write(
        &archive,
        legacy_v1_lml_with_synthetic("recording.txt", stored, original, Some(synthetic)),
    )
    .unwrap();
    let config = lma_forensic::LegacyAdapterConfig {
        executable: fake_legacy_adapter(root.path(), stored, original),
        timeout: Duration::from_secs(5),
        max_rss_bytes: 512 * 1024 * 1024,
    };

    let capsule = root.path().join("synthetic.bcs2");
    let converted =
        lma_forensic::convert_archive_with_legacy_config(&archive, &capsule, 3, &config).unwrap();
    assert!(converted.errors.is_empty(), "{:?}", converted.errors);
    let capsule_bytes = std::fs::read(&capsule).unwrap();
    let view = ForensicTreeView::parse(
        &capsule_bytes,
        READER_CAPABILITIES,
        ResourceBounds {
            max_frame_bytes: u32::MAX,
            max_catalog_bytes: u32::MAX,
            ..ResourceBounds::default()
        },
    )
    .unwrap();
    let entry = view
        .entries()
        .iter()
        .find(|entry| entry.path == b"recording.txt")
        .unwrap();
    assert!(entry.xattrs.is_empty());
    let stored_form = entry.stored_form.expect("synthetic stored form");
    assert_ne!(stored_form.capabilities & CAP_LMA_SYNTHETIC_REEMIT, 0);
    assert_eq!(
        LmaSyntheticReemitParametersV1::decode(stored_form.parameters).unwrap(),
        LmaSyntheticReemitParametersV1::new(LmaSyntheticLineEnding::CrLf, 0, 0, false)
    );
    let restored = root.path().join("restored");
    std::fs::create_dir_all(&restored).expect("make empty restore destination");
    let unpacked =
        lma_forensic::unpack_capsule_with_legacy_config(&capsule, &restored, false, &config)
            .unwrap();
    assert!(unpacked.errors.is_empty(), "{:?}", unpacked.errors);
    assert_eq!(
        std::fs::read(restored.join("recording.txt")).unwrap(),
        original
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("adapter-calls.log")).unwrap(),
        "materialize-synthetic-exact\nmanifest\nmaterialize-synthetic-exact\n"
    );
}
