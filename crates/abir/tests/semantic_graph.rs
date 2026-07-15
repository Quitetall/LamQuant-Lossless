use std::sync::Arc;

use abir::{
    Attachment, ChannelDescriptor, Clock, ClockKind, CoordinateFrame, CoordinatePoint, Event,
    Interval, LossReceipt, ModalityId, Property, PropertyBag, ProvenanceActivity, QualifiedName,
    Rational, RecordingBuilder, RecordingError, RecordingIdentity, ReferenceEdge, ReferenceNode,
    ReferenceNodeKind, Relationship, SampleBuffer, SampleDtype, SemanticDisposition, SignalSeries,
    SignalStream, Table, TableColumn, Tensor, TensorBuffer, TensorDataType, TimeAxis, Unit, Value,
    ValueType,
};

fn name(namespace: &str, local: &str) -> QualifiedName {
    QualifiedName::new(namespace, local)
}

fn base_builder() -> RecordingBuilder {
    let mut builder = RecordingBuilder::new(RecordingIdentity::new(
        "subject-01",
        Some("session-01"),
        Some("run-01"),
    ));
    builder
        .add_clock(Clock::new(
            "clock:device",
            ClockKind::Device,
            Rational::new(1_000_000, 1).unwrap(),
        ))
        .unwrap();
    builder
        .add_signal_stream(
            SignalStream::new("stream:eeg", ModalityId::eeg()).with_series(SignalSeries::new(
                ChannelDescriptor::new("channel:C3", "C3", ModalityId::eeg(), Unit::ucum("uV")),
                TimeAxis::uniform("clock:device", 0, Rational::new(250, 1).unwrap()),
                SampleBuffer::from_i16(Arc::from([1_i16, -2, 3, -4])),
            )),
        )
        .unwrap();
    builder
}

#[test]
fn frozen_recording_exposes_full_typed_semantic_graph() {
    let mut builder = base_builder();
    let trial_type = name("bids", "trial_type");
    let stimulus = name("neuro", "stimulus");

    builder
        .add_event(
            Event::new("event:stimulus-1", "clock:device", 20_000, stimulus.clone())
                .with_properties(PropertyBag::new(vec![Property::new(
                    trial_type.clone(),
                    Value::text("left_hand"),
                )])),
        )
        .unwrap();
    builder
        .add_interval(Interval::new(
            "interval:artifact-1",
            "clock:device",
            80_000,
            120_000,
            name("neuro", "motion_artifact"),
        ))
        .unwrap();
    builder
        .add_table(
            Table::new("table:trials")
                .with_column(TableColumn::new(
                    trial_type.clone(),
                    ValueType::Text,
                    Arc::from([Value::text("left_hand"), Value::text("right_hand")]),
                ))
                .with_column(TableColumn::new(
                    name("bids", "onset_tick"),
                    ValueType::I64,
                    Arc::from([Value::I64(20_000), Value::I64(220_000)]),
                )),
        )
        .unwrap();
    builder
        .add_tensor(Tensor::new(
            "tensor:leadfield",
            Arc::from([2_u64, 3]),
            TensorBuffer::from_f32(Arc::from([0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])),
        ))
        .unwrap();
    builder
        .add_coordinate_frame(CoordinateFrame::new(
            "frame:head",
            3,
            name("neuro", "head_cartesian"),
        ))
        .unwrap();
    builder
        .add_coordinate(CoordinatePoint::new(
            "coordinate:C3",
            "frame:head",
            "channel:C3",
            Arc::from([12.5_f64, -4.0, 80.25]),
            Unit::ucum("mm"),
        ))
        .unwrap();
    builder
        .add_reference_node(ReferenceNode::new(
            "reference:average",
            ReferenceNodeKind::DerivedReference,
        ))
        .unwrap();
    builder
        .add_reference_edge(ReferenceEdge::new(
            "reference-edge:C3-average",
            "channel:C3",
            "reference:average",
            name("neuro", "subtract_reference"),
        ))
        .unwrap();
    builder
        .add_relationship(Relationship::new(
            "relationship:event-trials",
            "event:stimulus-1",
            name("abir", "described_by"),
            "table:trials",
        ))
        .unwrap();
    builder
        .add_attachment(Attachment::new(
            "attachment:source-header",
            "application/dicom",
            Arc::from(&b"DICM"[..]),
        ))
        .unwrap();
    builder
        .add_provenance(
            ProvenanceActivity::new(
                "provenance:ingest",
                name("abir", "ingest"),
                "lamquant-ingest/0.1",
            )
            .with_input("attachment:source-header")
            .with_output("stream:eeg"),
        )
        .unwrap();
    builder
        .add_loss_receipt(LossReceipt::new(
            "loss:dicom-private-tag",
            name("dicom", "0019,1001"),
            SemanticDisposition::PreservedAsExtension,
            Some(name("vendor.example", "private_0019_1001")),
            "Private field retained as a typed extension",
        ))
        .unwrap();
    builder.set_extensions(PropertyBag::new(vec![Property::new(
        name("vendor.example", "private_0019_1001"),
        Value::bytes(Arc::from(&[0x10_u8, 0x20][..])),
    )]));

    let recording = builder.freeze().unwrap();

    assert_eq!(recording.clocks()[0].tick_rate().numerator(), 1_000_000);
    assert_eq!(recording.events()[0].label(), &stimulus);
    assert_eq!(
        recording.events()[0]
            .properties()
            .get(&trial_type)
            .and_then(Value::as_text),
        Some("left_hand")
    );
    assert_eq!(recording.intervals()[0].end_tick(), 120_000);
    assert_eq!(recording.tables()[0].row_count(), 2);
    assert_eq!(recording.tensors()[0].shape(), &[2, 3]);
    assert_eq!(recording.coordinates()[0].values(), &[12.5, -4.0, 80.25]);
    assert_eq!(recording.reference_edges()[0].from(), "channel:C3");
    assert_eq!(recording.relationships()[0].object(), "table:trials");
    assert_eq!(recording.attachments()[0].bytes(), b"DICM");
    assert_eq!(
        recording.provenance()[0].outputs(),
        &[Arc::from("stream:eeg")]
    );
    assert_eq!(
        recording.loss_receipts()[0].disposition(),
        SemanticDisposition::PreservedAsExtension
    );
    assert_eq!(
        recording
            .extensions()
            .get(&name("vendor.example", "private_0019_1001"))
            .and_then(Value::as_bytes),
        Some(&[0x10_u8, 0x20][..])
    );
}

#[test]
fn freeze_rejects_dangling_time_and_graph_references() {
    let mut missing_clock = RecordingBuilder::new(RecordingIdentity::new("s", None, None));
    missing_clock
        .add_event(Event::new(
            "event:orphan",
            "clock:missing",
            0,
            name("neuro", "stimulus"),
        ))
        .unwrap();
    assert!(matches!(
        missing_clock.freeze(),
        Err(RecordingError::UnknownClockId { .. })
    ));

    let mut dangling_relation = base_builder();
    dangling_relation
        .add_relationship(Relationship::new(
            "relationship:dangling",
            "event:missing",
            name("abir", "described_by"),
            "stream:eeg",
        ))
        .unwrap();
    assert!(matches!(
        dangling_relation.freeze(),
        Err(RecordingError::UnknownNodeId { .. })
    ));
}

#[test]
fn graph_references_are_independent_of_node_family_storage_order() {
    let mut builder = base_builder();
    builder
        .add_relationship(Relationship::new(
            "relationship:event-attachment",
            "event:late",
            name("abir", "derived_from"),
            "attachment:late",
        ))
        .unwrap();
    builder
        .add_event(Event::new(
            "event:late",
            "clock:device",
            1,
            name("neuro", "late_event"),
        ))
        .unwrap();
    builder
        .add_attachment(Attachment::new(
            "attachment:late",
            "application/octet-stream",
            Arc::from(&[1_u8, 2, 3][..]),
        ))
        .unwrap();

    builder.freeze().unwrap();
}

#[test]
fn freeze_rejects_invalid_interval_table_tensor_and_coordinates() {
    let mut invalid_interval = base_builder();
    invalid_interval
        .add_interval(Interval::new(
            "interval:backwards",
            "clock:device",
            50,
            49,
            name("neuro", "artifact"),
        ))
        .unwrap();
    assert!(matches!(
        invalid_interval.freeze(),
        Err(RecordingError::InvalidIntervalBounds { .. })
    ));

    let mut ragged_table = base_builder();
    ragged_table
        .add_table(
            Table::new("table:ragged")
                .with_column(TableColumn::new(
                    name("test", "a"),
                    ValueType::I64,
                    Arc::from([Value::I64(1), Value::I64(2)]),
                ))
                .with_column(TableColumn::new(
                    name("test", "b"),
                    ValueType::Text,
                    Arc::from([Value::text("only-one")]),
                )),
        )
        .unwrap();
    assert!(matches!(
        ragged_table.freeze(),
        Err(RecordingError::TableColumnLengthMismatch { .. })
    ));

    let mut bad_tensor = base_builder();
    bad_tensor
        .add_tensor(Tensor::new(
            "tensor:bad",
            Arc::from([2_u64, 2]),
            TensorBuffer::from_i16(Arc::from([1_i16, 2, 3])),
        ))
        .unwrap();
    assert!(matches!(
        bad_tensor.freeze(),
        Err(RecordingError::TensorElementCountMismatch { .. })
    ));

    let mut bad_coordinate = base_builder();
    bad_coordinate
        .add_coordinate_frame(CoordinateFrame::new(
            "frame:head",
            3,
            name("neuro", "head_cartesian"),
        ))
        .unwrap();
    bad_coordinate
        .add_coordinate(CoordinatePoint::new(
            "coordinate:C3",
            "frame:head",
            "channel:C3",
            Arc::from([1.0_f64, 2.0]),
            Unit::ucum("mm"),
        ))
        .unwrap();
    assert!(matches!(
        bad_coordinate.freeze(),
        Err(RecordingError::CoordinateDimensionMismatch { .. })
    ));
}

#[test]
fn primitive_vocabulary_preserves_native_numeric_and_nested_extension_values() {
    let samples = SampleBuffer::from_f32(Arc::from([1.25_f32, -2.5]));
    assert_eq!(samples.dtype(), SampleDtype::F32);
    assert_eq!(samples.as_f32(), Some(&[1.25_f32, -2.5][..]));

    let tensor = TensorBuffer::from_f64(Arc::from([1.0_f64, 2.0]));
    assert_eq!(tensor.dtype(), TensorDataType::F64);
    assert_eq!(tensor.as_f64(), Some(&[1.0_f64, 2.0][..]));

    let nested = Value::record(PropertyBag::new(vec![Property::new(
        name("dicom", "channel_sensitivity"),
        Value::rational(Rational::new(3, 2).unwrap()),
    )]));
    assert_eq!(nested.value_type(), Some(ValueType::Record));
    assert_eq!(Value::U64(42).value_type(), Some(ValueType::U64));
    assert_eq!(
        Value::list(Arc::from([Value::Bool(true), Value::Null])).value_type(),
        Some(ValueType::List)
    );

    assert_ne!(
        SemanticDisposition::Approximated,
        SemanticDisposition::Dropped
    );
    assert_ne!(
        ReferenceNodeKind::Ground,
        ReferenceNodeKind::PhysicalReference
    );
    assert_ne!(ClockKind::UnixUtc, ClockKind::Relative);
}

#[test]
fn freeze_rejects_zero_rates_missing_extension_targets_and_excessive_nesting() {
    let mut zero_clock = RecordingBuilder::new(RecordingIdentity::new("s", None, None));
    zero_clock
        .add_clock(Clock::new(
            "clock:stopped",
            ClockKind::Device,
            Rational::new(0, 1).unwrap(),
        ))
        .unwrap();
    assert!(matches!(
        zero_clock.freeze(),
        Err(RecordingError::InvalidRate { .. })
    ));

    let mut missing_extension = base_builder();
    missing_extension
        .add_loss_receipt(LossReceipt::new(
            "loss:missing-extension",
            name("dicom", "private"),
            SemanticDisposition::PreservedAsExtension,
            Some(name("vendor.example", "absent")),
            "claimed but absent",
        ))
        .unwrap();
    assert!(matches!(
        missing_extension.freeze(),
        Err(RecordingError::MissingExtensionTarget { .. })
    ));

    let mut nested = Value::Null;
    for _ in 0..65 {
        nested = Value::list(Arc::from([nested]));
    }
    let mut excessive_nesting = base_builder();
    excessive_nesting.set_extensions(PropertyBag::new(vec![Property::new(
        name("test", "nested"),
        nested,
    )]));
    assert!(matches!(
        excessive_nesting.freeze(),
        Err(RecordingError::ValueNestingLimit { .. })
    ));
}
