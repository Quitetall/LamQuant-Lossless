#![cfg(feature = "archive")]

use abir::{decode_bcs2, encode_bcs2, SemanticDisposition, Value};
use lamquant_core::source::BidsRecordingReader;

fn write_bids_fixture(root: &std::path::Path) -> std::path::PathBuf {
    std::fs::write(
        root.join("dataset_description.json"),
        br#"{"Name":"ABIR2 fixture","BIDSVersion":"1.9.0"}"#,
    )
    .unwrap();
    let eeg_dir = root.join("sub-01").join("ses-02").join("eeg");
    std::fs::create_dir_all(&eeg_dir).unwrap();
    let stem = "sub-01_ses-02_task-rest_run-03";
    let signal_path = eeg_dir.join(format!("{stem}_eeg.edf"));
    let samples = [-4_i16, -1, 0, 7, i16::MAX];
    std::fs::write(
        &signal_path,
        lamquant_core::ingest::synth_single_channel_edf(&samples, 250.0),
    )
    .unwrap();
    std::fs::write(
        eeg_dir.join(format!("{stem}_events.tsv")),
        "onset\tduration\ttrial_type\tvalue\n0.008\t0.012\tstimulus\tleft\n0.020\tn/a\tresponse\tok\n",
    )
    .unwrap();
    std::fs::write(
        eeg_dir.join(format!("{stem}_channels.tsv")),
        "name\ttype\tunits\tstatus\nEEG ch0\tEEG\tuV\tgood\n",
    )
    .unwrap();
    std::fs::write(
        eeg_dir.join("sub-01_ses-02_electrodes.tsv"),
        "name\tx\ty\tz\nEEG ch0\t1.0\t-2.5\t3.25\n",
    )
    .unwrap();
    std::fs::write(
        eeg_dir.join("sub-01_ses-02_coordsystem.json"),
        br#"{"EEGCoordinateSystem":"CapTrak","EEGCoordinateUnits":"mm"}"#,
    )
    .unwrap();
    signal_path
}

#[test]
fn bids_reader_builds_typed_semantics_and_preserves_sidecars() {
    let directory = tempfile::tempdir().unwrap();
    let signal_path = write_bids_fixture(directory.path());

    let recording = BidsRecordingReader::new(&signal_path)
        .read_recording()
        .unwrap();

    assert_eq!(recording.identity().subject(), "01");
    assert_eq!(recording.identity().session(), Some("02"));
    assert_eq!(recording.identity().run(), Some("03"));
    assert_eq!(recording.signal_streams()[0].modality().as_str(), "eeg");

    assert_eq!(recording.events().len(), 2);
    assert_eq!(recording.events()[0].tick(), 2);
    assert_eq!(recording.events()[0].label().local(), "stimulus");
    assert_eq!(recording.events()[1].tick(), 5);
    assert_eq!(recording.intervals().len(), 1);
    assert_eq!(recording.intervals()[0].start_tick(), 2);
    assert_eq!(recording.intervals()[0].end_tick(), 5);

    let event_table = recording
        .tables()
        .iter()
        .find(|table| table.id() == "table:bids-events")
        .unwrap();
    assert_eq!(event_table.row_count(), 2);
    let channel_table = recording
        .tables()
        .iter()
        .find(|table| table.id() == "table:bids-channels")
        .unwrap();
    assert_eq!(channel_table.row_count(), 1);

    assert_eq!(recording.coordinate_frames().len(), 1);
    assert_eq!(recording.coordinate_frames()[0].system().local(), "CapTrak");
    assert_eq!(recording.coordinates().len(), 1);
    assert_eq!(recording.coordinates()[0].values(), &[1.0, -2.5, 3.25]);
    assert_eq!(
        recording.coordinates()[0].object_id(),
        "signal:channel:000000"
    );

    let preserved = recording
        .attachments()
        .iter()
        .filter(|attachment| attachment.id().starts_with("attachment:bids:"))
        .collect::<Vec<_>>();
    assert_eq!(preserved.len(), 5);
    assert!(preserved
        .iter()
        .any(|attachment| attachment.media_type() == "text/tab-separated-values"));
    assert!(recording.loss_receipts().iter().any(|receipt| {
        receipt.label().namespace() == "bids" && receipt.disposition() == SemanticDisposition::Exact
    }));
    assert!(recording.verify().is_ok());
}

#[test]
fn inherited_sidecar_resolution_is_specific_and_deterministic() {
    let directory = tempfile::tempdir().unwrap();
    let signal_path = write_bids_fixture(directory.path());
    std::fs::write(
        directory.path().join("task-rest_eeg.json"),
        br#"{"PowerLineFrequency":50,"SamplingFrequency":250}"#,
    )
    .unwrap();
    let local = signal_path.with_extension("json");
    std::fs::write(&local, br#"{"PowerLineFrequency":60}"#).unwrap();

    let first = BidsRecordingReader::new(&signal_path)
        .read_recording()
        .unwrap();
    let second = BidsRecordingReader::new(&signal_path)
        .read_recording()
        .unwrap();

    assert_eq!(encode_bcs2(&first).unwrap(), encode_bcs2(&second).unwrap());
    let decoded = decode_bcs2(&encode_bcs2(&first).unwrap()).unwrap();
    let power_line = decoded
        .extensions()
        .properties()
        .iter()
        .find(|property| property.name().local() == "PowerLineFrequency")
        .unwrap();
    assert!(matches!(power_line.value(), Value::U64(60)));
    let sampling_frequency = decoded
        .extensions()
        .properties()
        .iter()
        .find(|property| property.name().local() == "SamplingFrequency")
        .unwrap();
    assert!(matches!(sampling_frequency.value(), Value::U64(250)));
}

#[test]
fn malformed_or_ambiguous_bids_metadata_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let signal_path = write_bids_fixture(directory.path());
    let events = signal_path
        .parent()
        .unwrap()
        .join("sub-01_ses-02_task-rest_run-03_events.tsv");
    std::fs::write(&events, "onset\tduration\nnot-a-number\t0\n").unwrap();

    let error = BidsRecordingReader::new(&signal_path)
        .read_recording()
        .unwrap_err();
    assert!(error.to_string().contains("events.tsv onset"));

    let bad_name = signal_path.parent().unwrap().join("recording.edf");
    std::fs::copy(&signal_path, &bad_name).unwrap();
    let error = BidsRecordingReader::new(&bad_name)
        .read_recording()
        .unwrap_err();
    assert!(error.to_string().contains("BIDS entity"));
}
