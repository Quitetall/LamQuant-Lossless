use std::sync::Arc;

use abir::{
    ChannelDescriptor, ModalityId, Rational, RecordingBuilder, RecordingIdentity, SampleBuffer,
    SampleDtype, SignalSeries, SignalStream, TimeAxis, Unit,
};

#[test]
fn frozen_recording_preserves_mixed_rates_ragged_lengths_and_native_widths() {
    let mut builder = RecordingBuilder::new(RecordingIdentity::new(
        "subject-01",
        Some("session-01"),
        Some("run-01"),
    ));

    let eeg = SignalStream::new("signals/eeg", ModalityId::eeg()).with_series(SignalSeries::new(
        ChannelDescriptor::new("channels/fz", "Fz", ModalityId::eeg(), Unit::ucum("uV")),
        TimeAxis::uniform("clocks/device", 0, Rational::new(250, 1).unwrap()),
        SampleBuffer::from_i16(Arc::from([1_i16, -2, 3, -4])),
    ));
    let ecg = SignalStream::new("signals/ecg", ModalityId::ecg()).with_series(SignalSeries::new(
        ChannelDescriptor::new(
            "channels/ecg-i",
            "ECG I",
            ModalityId::ecg(),
            Unit::ucum("mV"),
        ),
        TimeAxis::uniform("clocks/device", 0, Rational::new(500, 1).unwrap()),
        SampleBuffer::from_i32(Arc::from([10_i32, 20, 30, 40, 50, 60])),
    ));

    builder.add_signal_stream(eeg).unwrap();
    builder.add_signal_stream(ecg).unwrap();
    let recording = builder.freeze().unwrap();

    assert_eq!(recording.identity().subject(), "subject-01");
    assert_eq!(recording.signal_streams().len(), 2);
    assert_eq!(recording.signal_streams()[0].series()[0].len(), 4);
    assert_eq!(recording.signal_streams()[1].series()[0].len(), 6);
    assert_eq!(
        recording.signal_streams()[0].series()[0].samples().dtype(),
        SampleDtype::I16
    );
    assert_eq!(
        recording.signal_streams()[1].series()[0].samples().dtype(),
        SampleDtype::I32
    );
    assert_eq!(
        recording.signal_streams()[0].series()[0]
            .time_axis()
            .sample_rate(),
        Some(Rational::new(250, 1).unwrap())
    );
    assert_eq!(recording.streams_by_modality(&ModalityId::eeg()).len(), 1);
    recording.verify().unwrap();
}

#[test]
fn freeze_rejects_duplicate_entity_ids_and_bad_explicit_time_axes() {
    let make_stream = |id: &str, channel_id: &str| {
        SignalStream::new(id, ModalityId::eeg()).with_series(SignalSeries::new(
            ChannelDescriptor::new(channel_id, "Fz", ModalityId::eeg(), Unit::ucum("uV")),
            TimeAxis::explicit("clocks/device", Arc::from([0_i64, 4_000_000])),
            SampleBuffer::from_i16(Arc::from([1_i16, 2, 3])),
        ))
    };

    let mut builder = RecordingBuilder::new(RecordingIdentity::new("subject-01", None, None));
    builder
        .add_signal_stream(make_stream("signals/eeg", "channels/fz"))
        .unwrap();
    assert!(builder
        .add_signal_stream(make_stream("signals/eeg", "channels/fz-duplicate"))
        .is_err());
    assert!(builder.freeze().is_err());
}
