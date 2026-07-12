use abir::{Bcs1Header, BCS1_HEADER_LEN};
use lamquant_core::container;
use lamquant_core::container_reader::{ContainerFormat, ContainerReader};
use lamquant_core::error::LmlError;
use lamquant_core::lpc::LpcMode;
use lamquant_core::range::{RangeQuery, RangeReader};
use std::io::{Cursor, Write};

fn signal() -> Vec<Vec<i64>> {
    vec![
        (0..600).map(|value| value as i64 * 3 - 700).collect(),
        (0..600).map(|value| 900 - value as i64 * 2).collect(),
        (0..600).map(|value| (value as i64 % 31) - 15).collect(),
    ]
}

fn bcs1_bytes() -> (Vec<Vec<i64>>, Vec<u8>) {
    let signal = signal();
    let mut bytes = Vec::new();
    container::write_into(
        &mut bytes,
        &signal,
        250.0,
        128,
        0,
        r#"{"source":"a14"}"#,
        LpcMode::Fixed,
    )
    .unwrap();
    (signal, bytes)
}

#[test]
fn bcs1_memory_reader_owns_sequential_indexed_and_range_access() {
    let (expected, bytes) = bcs1_bytes();
    let mut reader = ContainerReader::from_source(Cursor::new(bytes.clone())).unwrap();
    assert_eq!(reader.format(), ContainerFormat::Bcs1);
    assert!(reader.header().metadata.contains(r#""source":"a14""#));
    assert_eq!(reader.decode_all().unwrap(), expected);

    let second = reader.read_window(1).unwrap();
    for channel in 0..expected.len() {
        assert_eq!(second[channel], expected[channel][128..256]);
    }

    let windows = reader.windows_for_range(150, 300).unwrap();
    assert_eq!(windows.len(), 2);

    let mut range = RangeReader::open_from_source(Cursor::new(bytes)).unwrap();
    let slice = range
        .read(&RangeQuery::new(150, 300, Some(vec![2, 0])).unwrap())
        .unwrap();
    assert_eq!(slice.channels.as_deref(), Some(&[0, 2][..]));
    assert_eq!(slice.signal[0], expected[0][150..300]);
    assert_eq!(slice.signal[1], expected[2][150..300]);
}

#[test]
fn file_and_buffer_facades_share_the_same_reader_contract() {
    let (expected, bytes) = bcs1_bytes();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();

    let (from_memory, memory_metadata) = container::read_bytes(&bytes).unwrap();
    let (from_file, file_metadata) = container::read_file(file.path()).unwrap();
    assert_eq!(from_memory, expected);
    assert_eq!(from_file, expected);
    assert_eq!(memory_metadata, file_metadata);
}

#[test]
fn calibrated_decode_and_window_read_use_normalized_plan() {
    let (expected, bytes) = bcs1_bytes();
    let calibration = [
        -1000.0, 1000.0, -1.0, 1.0, -1000.0, 1000.0, -2.0, 2.0, -1000.0, 1000.0, -3.0, 3.0,
    ];
    let mut output = vec![0.0f32; expected.len() * expected[0].len()];
    let header =
        container::read_bytes_into_f32_calibrated(&bytes, &mut output, &calibration).unwrap();
    assert_eq!(header.n_ch, expected.len());
    assert!((output[0] - expected[0][0] as f32 / 1000.0).abs() < 1e-6);

    let (window, window_header) = container::read_window_from_bytes(&bytes, 3).unwrap();
    assert_eq!(window_header.n_windows, 5);
    for channel in 0..expected.len() {
        assert_eq!(window[channel], expected[channel][384..512]);
    }
}

#[test]
fn frozen_legacy_fixture_matches_independent_legacy_decoder() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/legacy_payload_crc.lml"
    ));
    let (expected, expected_metadata) = lamquant_lml_legacy::container::read_bytes(bytes).unwrap();
    let mut reader = ContainerReader::from_source(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.format(), ContainerFormat::LegacyLml1);
    assert_eq!(reader.header().metadata, expected_metadata);
    assert_eq!(reader.decode_all().unwrap(), expected);
}

#[test]
fn invalid_metadata_and_index_offsets_fail_closed() {
    let (_signal, mut invalid_metadata) = bcs1_bytes();
    invalid_metadata[40] = 0xff;
    assert!(matches!(
        ContainerReader::from_source(Cursor::new(invalid_metadata)),
        Err(LmlError::InvalidHeader(_))
    ));

    let (_signal, mut invalid_index) = bcs1_bytes();
    let header = Bcs1Header::parse(&invalid_index).unwrap();
    let index = BCS1_HEADER_LEN + header.metadata_length as usize;
    invalid_index[index..index + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut reader = ContainerReader::from_source(Cursor::new(invalid_index)).unwrap();
    assert!(matches!(
        reader.read_window(0),
        Err(LmlError::Truncated {
            context: "window length",
            ..
        })
    ));

    let (_signal, mut invalid_payload) = bcs1_bytes();
    let header = Bcs1Header::parse(&invalid_payload).unwrap();
    let payload = BCS1_HEADER_LEN + header.metadata_length as usize + header.n_windows as usize * 4;
    invalid_payload[payload..payload + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut reader = ContainerReader::from_source(Cursor::new(invalid_payload)).unwrap();
    assert!(matches!(
        reader.next_window(),
        Some(Err(LmlError::Truncated {
            context: "window payload",
            ..
        }))
    ));
}

#[test]
fn short_sources_return_truncated_without_panicking() {
    for length in 0..4 {
        assert!(matches!(
            ContainerReader::from_source(Cursor::new(vec![0u8; length])),
            Err(LmlError::Truncated {
                context: "container header",
                ..
            })
        ));
    }
}

#[test]
fn compatibility_facade_contains_no_format_parser_or_footer_clone() {
    let compatibility = include_str!("../src/bcs1_stream.rs");
    assert!(!compatibility.contains("Bcs1Header::parse"));
    assert!(!compatibility.contains("try_read_footer"));
    assert!(!compatibility.contains("SeekFrom"));

    let owner = include_str!("../src/container_reader.rs");
    assert_eq!(owner.matches("fn read_footer").count(), 1);
    assert_eq!(owner.matches("fn read_plan").count(), 1);
}
