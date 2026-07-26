//! The training window pack round-trips through the BCS2 `TRAINING_*` wire.
//!
//! The claim: every window's stored record survives conversion byte-for-byte,
//! the catalog describes each row by its *logical* real-valued shape rather than
//! by the quantised bytes it happens to hold, and a consumer that cannot
//! dequantise is refused instead of handed mantissas it would read as
//! amplitudes.
//!
//! All three pack dtypes are exercised, because the block-floating-point arms
//! and the exact `F32` arm take different paths through the dequantiser.

#![cfg(feature = "archive")]

use lamquant_core::tensor_pack::{quantize_window, PackDtype, PackWriter};
use lamquant_core::tensor_pack_bcs2::{self, SnapshotPack};
use semantic_abir_bcs::{Bcs2View, ResourceBounds};
use semantic_abir_training::TrainingProfile;
use std::path::{Path, PathBuf};

const CHANNELS: usize = 3;
const WINDOW_LEN: usize = 8;
const WINDOWS: usize = 4;

/// Deterministic, mildly correlated windows — real structure, no randomness.
fn source_windows() -> Vec<Vec<f32>> {
    (0..WINDOWS)
        .map(|window| {
            (0..CHANNELS * WINDOW_LEN)
                .map(|index| {
                    let channel = index / WINDOW_LEN;
                    let sample = index % WINDOW_LEN;
                    ((window * WINDOW_LEN + sample) as f32 / 4.0 + channel as f32).sin() * 100.0
                })
                .collect()
        })
        .collect()
}

fn write_pack(dir: &Path, dtype: PackDtype) -> (PathBuf, Vec<Vec<u8>>) {
    let path = dir.join(format!("pack-{}.lqtp", dtype.to_u8()));
    let mut writer = PackWriter::create(&path, dtype, CHANNELS, WINDOW_LEN, WINDOWS, [7_u8; 32])
        .expect("pack writer");
    let mut records = Vec::new();
    for window in source_windows() {
        writer.write_window(&window).expect("write window");
        // Recompute the record the writer produced, so the expectation is built
        // from the same quantiser rather than read back out of the file.
        let (scales, mantissas) = quantize_window(&window, CHANNELS, WINDOW_LEN, dtype);
        let mut record = Vec::new();
        for scale in &scales {
            record.extend_from_slice(&scale.to_le_bytes());
        }
        record.extend_from_slice(&mantissas);
        records.push(record);
    }
    writer.finish().expect("finish pack");
    (path, records)
}

fn convert(dir: &Path, dtype: PackDtype) -> (PathBuf, Vec<Vec<u8>>) {
    let (pack, records) = write_pack(dir, dtype);
    let snapshot = dir.join(format!("pack-{}.bcs2", dtype.to_u8()));
    tensor_pack_bcs2::convert_pack(&pack, &snapshot, TrainingProfile::Balanced)
        .expect("conversion succeeds");
    (snapshot, records)
}

#[test]
fn every_dtype_round_trips_its_records_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    for dtype in [PackDtype::Int8, PackDtype::Int16, PackDtype::F32] {
        let (snapshot_path, expected) = convert(dir.path(), dtype);
        let pack = SnapshotPack::open(&snapshot_path).expect("open snapshot");
        assert_eq!(pack.len().expect("len"), WINDOWS, "{dtype:?}: window count");

        let mut actual = pack.records().expect("records");
        // The sealed catalog orders rows by logical id, which is a hash and so
        // does not follow write order. Compare as sets of records, which is the
        // property that matters: no record altered, added or lost.
        actual.sort();
        let mut expected = expected;
        expected.sort();
        assert_eq!(
            actual, expected,
            "{dtype:?}: stored records must survive conversion unchanged"
        );
    }
}

#[test]
fn the_catalog_describes_rows_by_their_logical_shape() {
    // The point of the mapping: a consumer reading the catalog sees a
    // real-valued window, not the integers that happen to be stored.
    let dir = tempfile::tempdir().unwrap();
    let (snapshot_path, _) = convert(dir.path(), PackDtype::Int16);
    let bytes = std::fs::read(&snapshot_path).unwrap();
    let text = String::from_utf8_lossy(&bytes);

    assert!(
        text.contains("\"shape\":[3,8]"),
        "each row must declare the logical [channels, samples] shape"
    );
    assert!(
        text.contains("\"element\":\"f32\""),
        "each row must declare its logical element type, not the mantissa's"
    );
    assert!(
        text.contains("\"encoding\""),
        "each row must record that its frame is encoded"
    );
}

#[test]
fn dequantisation_matches_the_source_windows() {
    // F32 packs are exact; the BFP dtypes are lossy by design, so each is held
    // to the tolerance its mantissa width can actually deliver. A single shared
    // tolerance would either be too loose to catch an int8 bug or too tight for
    // int8 to ever pass.
    let dir = tempfile::tempdir().unwrap();
    let source = source_windows();
    for (dtype, tolerance) in [
        (PackDtype::F32, 0.0_f32),
        (PackDtype::Int16, 0.02),
        (PackDtype::Int8, 2.0),
    ] {
        let (snapshot_path, _) = convert(dir.path(), dtype);
        let pack = SnapshotPack::open(&snapshot_path).expect("open");
        let recovered = pack.windows(dtype).expect("dequantise");
        assert_eq!(recovered.len(), source.len());

        // Row order follows the hashed logical ids, so match each recovered
        // window to its nearest source window rather than assuming an order.
        for window in &recovered {
            let best = source
                .iter()
                .map(|original| {
                    original
                        .iter()
                        .zip(window)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max)
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                best <= tolerance,
                "{dtype:?}: no source window within {tolerance} (closest max error {best})"
            );
        }
    }
}

#[test]
fn a_consumer_without_the_capability_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (snapshot_path, _) = convert(dir.path(), PackDtype::Int16);
    let bytes = std::fs::read(&snapshot_path).unwrap();
    let mut bounds = ResourceBounds::default();
    bounds.max_frame_bytes = u32::MAX;
    bounds.max_catalog_bytes = u32::MAX;

    assert!(
        Bcs2View::parse(&bytes, 0, bounds).is_err(),
        "mantissas must not be handed to a consumer that cannot dequantise them"
    );
}

#[test]
fn a_snapshot_is_distinguishable_from_an_lqtp_pack() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, _) = write_pack(dir.path(), PackDtype::Int16);
    let (snapshot, _) = convert(dir.path(), PackDtype::Int16);

    assert!(tensor_pack_bcs2::is_snapshot(&snapshot));
    assert!(
        !tensor_pack_bcs2::is_snapshot(&pack),
        "an LQTP1 pack must not be mistaken for a snapshot"
    );
}

#[test]
fn the_source_pack_is_left_untouched() {
    // Conversion is non-destructive: the pack a training run may still be
    // mmapping must be readable and unchanged afterwards.
    let dir = tempfile::tempdir().unwrap();
    let (pack, _) = write_pack(dir.path(), PackDtype::Int16);
    let before = std::fs::read(&pack).unwrap();

    let snapshot = dir.path().join("converted.bcs2");
    tensor_pack_bcs2::convert_pack(&pack, &snapshot, TrainingProfile::Balanced).expect("convert");

    assert_eq!(std::fs::read(&pack).unwrap(), before);
}
