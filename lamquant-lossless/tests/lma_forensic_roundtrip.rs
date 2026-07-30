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
    Bcs2Error, ForensicTreeView, ResourceBounds, CAP_LML_LOSSLESS_V1, CAP_ZSTD,
};
use std::collections::BTreeMap;
use std::path::Path;

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

    (dir, expected)
}

fn capsule_of(root: &Path, out: &Path) -> lamquant_core::lma::ArchiveSummary {
    lma_forensic::pack_directory(root, out, 3, false, None).expect("pack to capsule")
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

    let restored = work.path().join("restored");
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

        if let Some(stored) = entry.stored_form {
            assert_ne!(
                stored.stored_content_id,
                entry.content_id.unwrap(),
                "{rel}: a transformed entry must not claim the file's own id for its frame"
            );
            assert_ne!(stored.capabilities, 0);
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

    let listed = lma_forensic::list_capsule(&capsule).expect("list");
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
            entry.sha256.is_some(),
            "{}: the archive-time sha256 must survive conversion",
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

    let restored = work.path().join("restored");
    lma_forensic::unpack_capsule(&capsule, &restored, false).expect("unpack converted");
    for (rel, original) in &expected {
        let actual = std::fs::read(restored.join(rel))
            .unwrap_or_else(|e| panic!("converted {rel} unreadable: {e}"));
        assert_eq!(&actual, original, "{rel} did not survive LMA → capsule");
    }
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
