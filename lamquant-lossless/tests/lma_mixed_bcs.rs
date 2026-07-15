#![cfg(feature = "archive")]

use std::sync::Arc;

use abir::{
    encode_bcs2, BiosignalWireVersion, ChannelDescriptor, Clock, ClockKind, ModalityId, Rational,
    RecordingBuilder, RecordingIdentity, SampleBuffer, SignalSeries, SignalStream, TimeAxis, Unit,
};
use lamquant_core::lma::{pack_archive, read_recording_entry};
use lamquant_core::lpc::LpcMode;

fn bcs1_bytes() -> Vec<u8> {
    let signal = vec![vec![-2_i64, -1, 0, 1], vec![10, 11, 12, 13]];
    let metadata = serde_json::json!({
        "channels": ["Fp1", "Cz"],
        "patient_id": "legacy-subject",
        "phys_dim": "uV",
        "phys_min": [-200.0, -150.0],
        "phys_max": [200.0, 150.0],
        "recording_info": "legacy BCS1 fixture",
        "startdate": "2026-07-15"
    })
    .to_string();
    let mut bytes = Vec::new();
    lamquant_core::container::write_into(
        &mut bytes,
        &signal,
        250.0,
        4,
        0,
        &metadata,
        LpcMode::Fixed,
    )
    .unwrap();
    bytes
}

fn bcs2_bytes() -> Vec<u8> {
    let rate = Rational::new(512, 1).unwrap();
    let mut builder = RecordingBuilder::new(RecordingIdentity::new(
        "semantic-subject",
        Some("session-1"),
        Some("run-2"),
    ));
    builder
        .add_clock(Clock::new("clock:eeg", ClockKind::Relative, rate))
        .unwrap();
    builder
        .add_signal_stream(
            SignalStream::new("stream:eeg", ModalityId::eeg()).with_series(SignalSeries::new(
                ChannelDescriptor::new("channel:f3", "F3", ModalityId::eeg(), Unit::ucum("uV")),
                TimeAxis::uniform("clock:eeg", 0, rate),
                SampleBuffer::from_i16(Arc::from([-3_i16, 0, 7, 11])),
            )),
        )
        .unwrap();
    encode_bcs2(&builder.freeze().unwrap()).unwrap()
}

#[test]
fn one_lma_dispatches_bcs1_and_bcs2_into_the_same_recording_boundary() {
    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join("legacy.bcs1"), bcs1_bytes()).unwrap();
    std::fs::write(source.path().join("semantic.bcs2"), bcs2_bytes()).unwrap();
    std::fs::write(source.path().join("notes.txt"), b"not a recording").unwrap();
    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(source.path(), archive.path(), 9, false, None).unwrap();

    let legacy = read_recording_entry(archive.path(), "legacy.bcs1").unwrap();
    assert_eq!(legacy.wire_version(), BiosignalWireVersion::Bcs1);
    assert_eq!(legacy.recording().identity().subject(), "legacy-subject");
    assert_eq!(legacy.recording().signal_streams().len(), 1);
    assert_eq!(legacy.recording().signal_streams()[0].series().len(), 2);
    assert!(matches!(
        legacy.recording().signal_streams()[0].series()[0].samples(),
        SampleBuffer::I64(values) if values.as_ref() == [-2, -1, 0, 1]
    ));

    let semantic = read_recording_entry(archive.path(), "semantic.bcs2").unwrap();
    assert_eq!(semantic.wire_version(), BiosignalWireVersion::Bcs2);
    assert_eq!(
        semantic.recording().identity().subject(),
        "semantic-subject"
    );
    assert_eq!(semantic.recording().identity().session(), Some("session-1"));
    assert!(matches!(
        semantic.recording().signal_streams()[0].series()[0].samples(),
        SampleBuffer::I16(values) if values.as_ref() == [-3, 0, 7, 11]
    ));

    let error = read_recording_entry(archive.path(), "notes.txt").unwrap_err();
    assert!(error.to_string().contains("neither BCS1 nor BCS2"));
}

#[test]
fn explicit_bcs_extension_mismatch_fails_closed() {
    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join("wrong.bcs1"), bcs2_bytes()).unwrap();
    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(source.path(), archive.path(), 9, false, None).unwrap();

    let error = read_recording_entry(archive.path(), "wrong.bcs1").unwrap_err();
    assert!(error.to_string().contains("declares BCS1"));
    assert!(error.to_string().contains("contains BCS2"));
}

#[test]
fn malformed_bcs1_metadata_fails_before_semantic_projection() {
    let signal = vec![vec![1_i64, 2, 3, 4]];
    let mut bytes = Vec::new();
    lamquant_core::container::write_into(
        &mut bytes,
        &signal,
        250.0,
        4,
        0,
        "not-json",
        LpcMode::Fixed,
    )
    .unwrap();
    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join("bad.bcs1"), bytes).unwrap();
    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(source.path(), archive.path(), 9, false, None).unwrap();

    let error = read_recording_entry(archive.path(), "bad.bcs1").unwrap_err();
    assert!(error.to_string().contains("metadata JSON"));
}

#[test]
fn malformed_typed_bcs1_metadata_fails_closed() {
    let signal = vec![vec![1_i64, 2, 3, 4]];
    let mut bytes = Vec::new();
    lamquant_core::container::write_into(
        &mut bytes,
        &signal,
        250.0,
        4,
        0,
        r#"{"channels":["Fp1"],"patient_id":17}"#,
        LpcMode::Fixed,
    )
    .unwrap();
    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join("bad-typed.bcs1"), bytes).unwrap();
    let archive = tempfile::NamedTempFile::new().unwrap();
    pack_archive(source.path(), archive.path(), 9, false, None).unwrap();

    let error = read_recording_entry(archive.path(), "bad-typed.bcs1").unwrap_err();
    assert!(error.to_string().contains("patient_id"));
    assert!(error.to_string().contains("must be text"));
}
