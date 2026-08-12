//! ADR 0092 A14 after ADR 0139/0141 cutover.
//!
//! BCS1 and the in-main legacy parser were retired. Current conformance means:
//! `abir-bcs` owns one BCS2 grammar for borrowed memory and seekable sources,
//! the LML profile decoder consumes that authenticated view, and retired wires
//! fail closed without restoring their parsers to the product graph.

mod common;

use common::encode_uniform;
use lamquant_core::container;
use lamquant_core::lpc::LpcMode;
use semantic_abir_bcs::{Bcs2FileIndex, Bcs2View, ResourceBounds};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};

const BASELINE_CAPABILITIES: u64 = 0;

fn signal() -> Vec<Vec<i64>> {
    vec![
        (0..600).map(|value| value as i64 * 3 - 700).collect(),
        (0..600).map(|value| 900 - value as i64 * 2).collect(),
        (0..600).map(|value| (value as i64 % 31) - 15).collect(),
    ]
}

fn artifact() -> (Vec<Vec<i64>>, Vec<u8>) {
    let signal = signal();
    let bytes = encode_uniform(
        &signal,
        250.0,
        128,
        r#"{"source":"a14-bcs2"}"#,
        LpcMode::Fixed,
    );
    (signal, bytes)
}

#[test]
fn borrowed_seekable_and_file_views_share_one_bcs2_plan() {
    let (_signal, bytes) = artifact();
    let borrowed =
        Bcs2View::parse(&bytes, BASELINE_CAPABILITIES, ResourceBounds::default()).unwrap();

    let mut cursor = Cursor::new(&bytes);
    let indexed = Bcs2FileIndex::open(
        &mut cursor,
        BASELINE_CAPABILITIES,
        ResourceBounds::default(),
    )
    .unwrap();
    assert_eq!(indexed.profile(), borrowed.profile());
    assert_eq!(indexed.root_kind(), borrowed.root_kind());
    assert_eq!(indexed.root_content_id(), borrowed.root_content_id());
    assert_eq!(indexed.semantic_json(), borrowed.semantic_json());
    assert_eq!(indexed.references(), borrowed.references());
    assert_eq!(indexed.artifact_len(), bytes.len() as u64);
    assert_eq!(indexed.frames().len(), borrowed.frames().len());

    for (location, frame) in indexed.frames().iter().zip(borrowed.frames()) {
        let mut actual = vec![0_u8; location.len() as usize];
        cursor.seek(SeekFrom::Start(location.offset())).unwrap();
        cursor.read_exact(&mut actual).unwrap();
        assert_eq!(actual, frame.bytes());
        assert_eq!(location.content_id(), frame.content_id());
        assert_eq!(location.storage_id(), frame.storage_id());
    }

    let mut temporary = tempfile::NamedTempFile::new().unwrap();
    temporary.write_all(&bytes).unwrap();
    let mut file = std::fs::File::open(temporary.path()).unwrap();
    let file_index =
        Bcs2FileIndex::open(&mut file, BASELINE_CAPABILITIES, ResourceBounds::default()).unwrap();
    assert_eq!(file_index.root_content_id(), borrowed.root_content_id());
    assert_eq!(file_index.frames(), indexed.frames());
}

#[test]
fn profile_decode_and_indexed_windows_consume_authenticated_bcs2() {
    let (expected, bytes) = artifact();
    let (decoded, metadata) = container::read_bytes(&bytes).unwrap();
    assert_eq!(decoded, expected);
    assert_eq!(metadata, r#"{"source":"a14-bcs2"}"#);

    let (second, header) = container::read_window_from_bytes(&bytes, 1).unwrap();
    assert_eq!(header.n_channels, expected.len());
    assert_eq!(header.total_samples, expected[0].len());
    assert_eq!(header.n_windows, 5);
    assert_eq!(header.window_size, 128);
    for channel in 0..expected.len() {
        assert_eq!(second[channel], expected[channel][128..256]);
    }
}

#[test]
fn truncation_corruption_and_retired_wires_fail_closed() {
    let (_signal, bytes) = artifact();
    for length in [0, 1, 3, 39, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            container::open(&bytes[..length]).is_err(),
            "truncated artifact of {length} bytes was accepted"
        );
    }

    let borrowed =
        Bcs2View::parse(&bytes, BASELINE_CAPABILITIES, ResourceBounds::default()).unwrap();
    let first = borrowed.frames().first().expect("LML bundle has frames");
    let offset = first.bytes().as_ptr() as usize - bytes.as_ptr() as usize;
    let mut corrupt = bytes.clone();
    corrupt[offset] ^= 0x80;
    assert!(container::open(&corrupt).is_err());

    let legacy = container::open(b"LML1-retired").unwrap_err().to_string();
    assert!(legacy.contains("retired wire"));
    assert!(legacy.contains("legacy Adapter"));
    assert!(container::open(b"BCS1-retired").is_err());
}

#[test]
fn current_reader_contract_does_not_restore_retired_parser_clones() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(!crate_root.join("src/container_reader.rs").exists());
    assert!(!crate_root.join("src/bcs1_stream.rs").exists());
    assert!(!crate_root.join("src/abir_container.rs").exists());

    let profile = include_str!("../src/bcs2_container.rs");
    assert!(!profile.contains("Bcs1Header"));
    assert!(!profile.contains("try_read_footer"));
    assert!(!profile.contains("SeekFrom"));
    assert!(profile.contains("FROZEN_LEGACY_MAGICS"));
}
