#![cfg(feature = "archive")]

use abir::{decode_bcs2, encode_bcs2, SampleBuffer, SemanticDisposition, Value};
use lamquant_core::source::{
    recording_from_signal_bundle, recording_from_signal_bundle_with_options, EdfReader,
    RecordingAdapterOptions, SidecarBlob, SignalBundle, SignalSourceReader, SourceMetadata,
};

fn bundle(sample_rate: f64) -> SignalBundle {
    SignalBundle {
        signal: vec![
            vec![-2, -1, 0, 1],
            vec![10, 11, 12, 13],
            vec![20, 21, 22, 23],
        ],
        sample_rate,
        channels: vec!["Fp1".into(), "EEG Cz".into(), "ECG II".into()],
        phys_min: vec![-200.0, -150.0, -5.0],
        phys_max: vec![200.0, 150.0, 5.0],
        duration_s: 4.0 / sample_rate,
        metadata: SourceMetadata {
            source_file: "sub-01_task-rest.edf".into(),
            format: "EDF+C".into(),
            patient_id: "source-subject".into(),
            recording_info: "test recording".into(),
            startdate: "2026-07-15".into(),
            phys_dim: "uV".into(),
        },
        sidecar: vec![SidecarBlob {
            key: "raw_header".into(),
            bytes: vec![0x41, 0x42, 0x43],
            aux: Some(7),
        }],
    }
}

#[test]
fn bundle_lowers_once_into_typed_immutable_graph() {
    let recording = recording_from_signal_bundle(bundle(250.0)).unwrap();

    assert_eq!(recording.identity().subject(), "source-subject");
    assert_eq!(recording.clocks().len(), 1);
    assert_eq!(recording.signal_streams().len(), 2);
    let modalities = recording
        .signal_streams()
        .iter()
        .map(|stream| stream.modality().as_str())
        .collect::<Vec<_>>();
    assert_eq!(modalities, vec!["ecg", "eeg"]);
    let eeg = recording
        .signal_streams()
        .iter()
        .find(|stream| stream.modality().as_str() == "eeg")
        .unwrap();
    assert_eq!(eeg.series().len(), 2);
    assert!(matches!(
        eeg.series()[0].samples(),
        SampleBuffer::I64(values) if values.as_ref() == [-2, -1, 0, 1]
    ));
    let calibration = recording
        .tables()
        .iter()
        .find(|table| table.id() == "table:channel-calibration")
        .unwrap();
    assert_eq!(calibration.row_count(), 3);
    assert_eq!(calibration.column_count(), 6);
    assert_eq!(recording.attachments()[0].bytes(), &[0x41, 0x42, 0x43]);
    assert_eq!(
        recording.loss_receipts()[0].disposition(),
        SemanticDisposition::PreservedAsExtension
    );
    assert!(recording.verify().is_ok());
}

#[test]
fn source_adapter_is_deterministic_through_bcs2() {
    let options = RecordingAdapterOptions {
        subject: Some("sub-01".into()),
        session: Some("ses-02".into()),
        run: Some("run-03".into()),
        declared_modality: None,
    };
    let first = recording_from_signal_bundle_with_options(bundle(512.5), options.clone()).unwrap();
    let second = recording_from_signal_bundle_with_options(bundle(512.5), options).unwrap();

    let first_bytes = encode_bcs2(&first).unwrap();
    assert_eq!(first_bytes, encode_bcs2(&second).unwrap());
    let decoded = decode_bcs2(&first_bytes).unwrap();
    assert_eq!(decoded.identity().subject(), "sub-01");
    assert_eq!(decoded.identity().session(), Some("ses-02"));
    assert_eq!(decoded.identity().run(), Some("run-03"));
    assert_eq!(decoded.signal_streams().len(), 2);
    assert_eq!(decoded.attachments()[0].bytes(), &[0x41, 0x42, 0x43]);
}

#[test]
fn rational_approximation_is_explicit_and_original_bits_survive() {
    let rate = 250.123_456_789_123_f64;
    let recording = recording_from_signal_bundle(bundle(rate)).unwrap();

    let receipt = recording
        .loss_receipts()
        .iter()
        .find(|receipt| receipt.id() == "receipt:sample-rate-rationalization")
        .unwrap();
    assert_eq!(receipt.disposition(), SemanticDisposition::Approximated);
    let property = recording
        .extensions()
        .properties()
        .iter()
        .find(|property| property.name().local() == "sample_rate_f64_bits")
        .unwrap();
    assert!(matches!(property.value(), Value::U64(bits) if *bits == rate.to_bits()));
}

#[test]
fn unrepresentable_rate_fails_closed() {
    let mut source = bundle(f64::MIN_POSITIVE);
    source.duration_s = 0.0;
    let error = recording_from_signal_bundle(source).unwrap_err();
    assert!(error.to_string().contains("cannot be represented"));
}

#[test]
fn declared_modality_overrides_channel_inference() {
    let options = RecordingAdapterOptions {
        declared_modality: Some("iEEG".into()),
        ..RecordingAdapterOptions::default()
    };
    let recording = recording_from_signal_bundle_with_options(bundle(250.0), options).unwrap();

    assert_eq!(recording.signal_streams().len(), 1);
    assert_eq!(recording.signal_streams()[0].modality().as_str(), "ieeg");
    assert_eq!(recording.signal_streams()[0].series().len(), 3);
}

#[test]
fn malformed_semantic_input_fails_before_graph_construction() {
    let mut source = bundle(250.0);
    source.phys_min[1] = f64::NAN;
    let error = recording_from_signal_bundle(source).unwrap_err();
    assert!(error.to_string().contains("physical range"));

    let options = RecordingAdapterOptions {
        session: Some("  ".into()),
        ..RecordingAdapterOptions::default()
    };
    let error = recording_from_signal_bundle_with_options(bundle(250.0), options).unwrap_err();
    assert!(error.to_string().contains("session"));
}

#[test]
fn real_edf_reader_uses_the_same_abir2_boundary() {
    let samples = [-4_i16, -1, 0, 7, i16::MAX];
    let bytes = lamquant_core::ingest::synth_single_channel_edf(&samples, 250.0);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.edf");
    std::fs::write(&path, bytes).unwrap();

    let recording = EdfReader::new(path).lower_to_recording().unwrap();

    assert_eq!(recording.signal_streams().len(), 1);
    assert_eq!(recording.signal_streams()[0].modality().as_str(), "eeg");
    assert!(matches!(
        recording.signal_streams()[0].series()[0].samples(),
        SampleBuffer::I64(values)
            if values.as_ref() == samples.map(i64::from)
    ));
    assert!(recording
        .attachments()
        .iter()
        .any(|attachment| attachment.bytes().len() >= 512));
}
