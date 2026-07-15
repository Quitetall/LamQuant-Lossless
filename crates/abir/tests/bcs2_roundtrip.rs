use std::sync::Arc;

use abir::{
    decode_bcs2, encode_bcs2, Attachment, Bcs2View, BiosignalWireVersion, ChannelDescriptor, Clock,
    ClockKind, CoordinateFrame, CoordinatePoint, Event, Interval, LossReceipt, ModalityId,
    Property, PropertyBag, ProvenanceActivity, QualifiedName, Rational, Recording,
    RecordingBuilder, RecordingIdentity, ReferenceEdge, ReferenceNode, ReferenceNodeKind,
    Relationship, SampleBuffer, SampleDtype, SectionKind, SemanticDisposition, SignalSeries,
    SignalStream, Table, TableColumn, Tensor, TensorBuffer, TensorDataType, TimeAxis, Unit, Value,
    ValueType, BCS1_MAGIC, BCS2_HEADER_LEN, BCS2_MAGIC,
};

fn name(namespace: &str, local: &str) -> QualifiedName {
    QualifiedName::new(namespace, local)
}

fn recording(reverse_unordered_nodes: bool) -> Recording {
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
                TimeAxis::uniform("clock:device", 10, Rational::new(250, 1).unwrap()),
                SampleBuffer::from_i16(Arc::from([1_i16, -2, 3, -4])),
            )),
        )
        .unwrap();

    let event_a = Event::new("event:a", "clock:device", 20_000, name("neuro", "stimulus"))
        .with_properties(PropertyBag::new(vec![Property::new(
            name("bids", "trial_type"),
            Value::text("left_hand"),
        )]));
    let event_b = Event::new("event:b", "clock:device", 30_000, name("neuro", "response"));
    if reverse_unordered_nodes {
        builder.add_event(event_b).unwrap();
        builder.add_event(event_a).unwrap();
    } else {
        builder.add_event(event_a).unwrap();
        builder.add_event(event_b).unwrap();
    }

    builder
        .add_interval(Interval::new(
            "interval:artifact",
            "clock:device",
            40_000,
            50_000,
            name("neuro", "artifact"),
        ))
        .unwrap();
    builder
        .add_table(Table::new("table:trials").with_column(TableColumn::new(
            name("bids", "onset_tick"),
            ValueType::I64,
            Arc::from([Value::I64(20_000), Value::I64(30_000)]),
        )))
        .unwrap();
    builder
        .add_tensor(Tensor::new(
            "tensor:features",
            Arc::from([2_u64, 2]),
            TensorBuffer::from_f32(Arc::from([1.0_f32, 2.0, 3.0, 4.0])),
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
            Arc::from([1.0_f64, 2.0, 3.0]),
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
            "relationship:event-table",
            "event:a",
            name("abir", "described_by"),
            "table:trials",
        ))
        .unwrap();
    builder
        .add_attachment(Attachment::new(
            "attachment:source",
            "application/dicom",
            Arc::from(&b"DICM-private-payload"[..]),
        ))
        .unwrap();
    builder
        .add_provenance(
            ProvenanceActivity::new(
                "provenance:ingest",
                name("abir", "ingest"),
                "lamquant-ingest/0.1",
            )
            .with_input("attachment:source")
            .with_output("stream:eeg"),
        )
        .unwrap();
    builder
        .add_loss_receipt(LossReceipt::new(
            "loss:private-tag",
            name("dicom", "0019,1001"),
            SemanticDisposition::PreservedAsExtension,
            Some(name("vendor.example", "private_0019_1001")),
            "retained as extension",
        ))
        .unwrap();

    let first = Property::new(name("vendor.example", "alpha"), Value::U64(1));
    let second = Property::new(
        name("vendor.example", "private_0019_1001"),
        Value::bytes(Arc::from(&[0x10_u8, 0x20][..])),
    );
    builder.set_extensions(if reverse_unordered_nodes {
        PropertyBag::new(vec![second, first])
    } else {
        PropertyBag::new(vec![first, second])
    });
    builder.freeze().unwrap()
}

#[test]
fn bcs2_is_deterministic_indexed_and_full_graph_round_trips() {
    let canonical = encode_bcs2(&recording(false)).unwrap();
    let reordered = encode_bcs2(&recording(true)).unwrap();
    assert_eq!(canonical, reordered);
    assert_eq!(&canonical[..4], BCS2_MAGIC);
    assert!(canonical.len() > BCS2_HEADER_LEN);

    let view = Bcs2View::parse(&canonical).unwrap();
    assert!(view.has_section(SectionKind::SignalSeries));
    assert!(view.has_section(SectionKind::Attachments));
    assert!(view
        .record_bytes(SectionKind::SignalSeries, "channel:C3")
        .unwrap()
        .is_some());

    let selected = view.decode_signal_series("channel:C3").unwrap().unwrap();
    assert_eq!(selected.channel().label(), "C3");
    assert_eq!(selected.samples().len(), 4);

    let decoded = decode_bcs2(&canonical).unwrap();
    decoded.verify().unwrap();
    assert_eq!(decoded.identity().subject(), "subject-01");
    assert_eq!(
        decoded.clocks()[0].tick_rate(),
        Rational::new(1_000_000, 1).unwrap()
    );
    assert_eq!(decoded.signal_streams()[0].series()[0].len(), 4);
    assert_eq!(decoded.events().len(), 2);
    assert_eq!(decoded.intervals()[0].end_tick(), 50_000);
    assert_eq!(decoded.tables()[0].row_count(), 2);
    assert_eq!(decoded.tensors()[0].shape(), &[2, 2]);
    assert_eq!(decoded.coordinates()[0].values(), &[1.0, 2.0, 3.0]);
    assert_eq!(decoded.reference_edges()[0].from(), "channel:C3");
    assert_eq!(decoded.attachments()[0].bytes(), b"DICM-private-payload");
    assert_eq!(decoded.provenance()[0].outputs()[0].as_ref(), "stream:eeg");
    assert_eq!(
        decoded.loss_receipts()[0].disposition(),
        SemanticDisposition::PreservedAsExtension
    );
}

#[test]
fn bcs2_rejects_corruption_and_truncation_fail_closed() {
    let bytes = encode_bcs2(&recording(false)).unwrap();
    for len in [0, 3, BCS2_HEADER_LEN - 1, bytes.len() - 1] {
        assert!(Bcs2View::parse(&bytes[..len]).is_err(), "len={len}");
    }

    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0x80;
    assert!(Bcs2View::parse(&corrupt).is_err());

    let mut impossible_directory = bytes;
    impossible_directory[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Bcs2View::parse(&impossible_directory).is_err());
}

#[test]
fn bcs2_round_trips_every_native_buffer_and_value_tag() {
    let mut builder = RecordingBuilder::new(RecordingIdentity::new("all-tags", None, None));
    builder
        .add_clock(Clock::new(
            "clock:device",
            ClockKind::Device,
            Rational::new(1_000, 1).unwrap(),
        ))
        .unwrap();

    let buffers = vec![
        ("00-i8", SampleBuffer::from_i8(Arc::from([-1_i8]))),
        ("01-u8", SampleBuffer::from_u8(Arc::from([1_u8]))),
        ("02-i16", SampleBuffer::from_i16(Arc::from([-2_i16]))),
        ("03-u16", SampleBuffer::from_u16(Arc::from([2_u16]))),
        ("04-i32", SampleBuffer::from_i32(Arc::from([-3_i32]))),
        ("05-u32", SampleBuffer::from_u32(Arc::from([3_u32]))),
        ("06-i64", SampleBuffer::from_i64(Arc::from([-4_i64]))),
        ("07-f32", SampleBuffer::from_f32(Arc::from([1.25_f32]))),
        ("08-f64", SampleBuffer::from_f64(Arc::from([2.5_f64]))),
    ];
    let mut stream = SignalStream::new("stream:all", ModalityId::eeg());
    for (suffix, buffer) in buffers {
        stream = stream.with_series(SignalSeries::new(
            ChannelDescriptor::new(
                format!("channel:{suffix}"),
                suffix,
                ModalityId::eeg(),
                Unit::ucum("uV"),
            ),
            TimeAxis::uniform("clock:device", 0, Rational::new(250, 1).unwrap()),
            buffer,
        ));
    }
    builder.add_signal_stream(stream).unwrap();

    let tensors = vec![
        ("00-i8", TensorBuffer::from_i8(Arc::from([-1_i8]))),
        ("01-u8", TensorBuffer::from_u8(Arc::from([1_u8]))),
        ("02-i16", TensorBuffer::from_i16(Arc::from([-2_i16]))),
        ("03-u16", TensorBuffer::from_u16(Arc::from([2_u16]))),
        ("04-i32", TensorBuffer::from_i32(Arc::from([-3_i32]))),
        ("05-u32", TensorBuffer::from_u32(Arc::from([3_u32]))),
        ("06-i64", TensorBuffer::from_i64(Arc::from([-4_i64]))),
        ("07-f32", TensorBuffer::from_f32(Arc::from([1.25_f32]))),
        ("08-f64", TensorBuffer::from_f64(Arc::from([2.5_f64]))),
    ];
    for (suffix, buffer) in tensors {
        builder
            .add_tensor(Tensor::new(
                format!("tensor:{suffix}"),
                Arc::from([1_u64]),
                buffer,
            ))
            .unwrap();
    }

    builder.set_extensions(PropertyBag::new(vec![
        Property::new(name("test", "00-null"), Value::Null),
        Property::new(name("test", "01-bool"), Value::Bool(true)),
        Property::new(name("test", "02-i64"), Value::I64(-9)),
        Property::new(name("test", "03-u64"), Value::U64(9)),
        Property::new(name("test", "04-f64"), Value::from(3.5_f64)),
        Property::new(
            name("test", "05-rational"),
            Value::rational(Rational::new(3, 7).unwrap()),
        ),
        Property::new(name("test", "06-text"), Value::text("value")),
        Property::new(
            name("test", "07-bytes"),
            Value::bytes(Arc::from(&[0_u8, 1, 2][..])),
        ),
        Property::new(
            name("test", "08-list"),
            Value::list(Arc::from([Value::Bool(false), Value::I64(4)])),
        ),
        Property::new(
            name("test", "09-record"),
            Value::record(PropertyBag::new(vec![Property::new(
                name("test", "nested"),
                Value::text("ok"),
            )])),
        ),
    ]));

    let recording = builder.freeze().unwrap();
    let bytes = encode_bcs2(&recording).unwrap();
    let decoded = decode_bcs2(&bytes).unwrap();
    assert_eq!(encode_bcs2(&decoded).unwrap(), bytes);
    assert_eq!(
        decoded.signal_streams()[0]
            .series()
            .iter()
            .map(|series| series.samples().dtype())
            .collect::<Vec<_>>(),
        vec![
            SampleDtype::I8,
            SampleDtype::U8,
            SampleDtype::I16,
            SampleDtype::U16,
            SampleDtype::I32,
            SampleDtype::U32,
            SampleDtype::I64,
            SampleDtype::F32,
            SampleDtype::F64,
        ]
    );
    assert_eq!(
        decoded
            .tensors()
            .iter()
            .map(|tensor| tensor.buffer().dtype())
            .collect::<Vec<_>>(),
        vec![
            TensorDataType::I8,
            TensorDataType::U8,
            TensorDataType::I16,
            TensorDataType::U16,
            TensorDataType::I32,
            TensorDataType::U32,
            TensorDataType::I64,
            TensorDataType::F32,
            TensorDataType::F64,
        ]
    );
    assert_eq!(decoded.extensions().properties().len(), 10);
}

#[test]
fn wire_detection_keeps_bcs1_and_bcs2_distinct() {
    assert_eq!(
        BiosignalWireVersion::detect(BCS1_MAGIC),
        Some(BiosignalWireVersion::Bcs1)
    );
    assert_eq!(
        BiosignalWireVersion::detect(BCS2_MAGIC),
        Some(BiosignalWireVersion::Bcs2)
    );
    assert_eq!(BiosignalWireVersion::detect(b"LML1"), None);
}

#[test]
fn bcs2_minimal_wire_golden() {
    let recording = RecordingBuilder::new(RecordingIdentity::new("golden", None, None))
        .freeze()
        .unwrap();
    let bytes = encode_bcs2(&recording).unwrap();
    let actual = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let expected = "424353320200010140000000120000004000000000000000d002000000000000b104000000000000a8a08ecf20e75b92000000000000000000000000000000000100010001000000100300000000000051000000000000000300000000000000a89c813c00000000020001000100000061030000000000003400000000000000010000000000000054b48ee40000000003000100010000009503000000000000100000000000000000000000000000007ce1149a000000000400010001000000a503000000000000100000000000000000000000000000007ce1149a000000000500010001000000b503000000000000100000000000000000000000000000007ce1149a000000000600010001000000c503000000000000100000000000000000000000000000007ce1149a000000000700010001000000d503000000000000100000000000000000000000000000007ce1149a000000000800010001000000e503000000000000100000000000000000000000000000007ce1149a000000000900010001000000f503000000000000100000000000000000000000000000007ce1149a000000000a000100010000000504000000000000100000000000000000000000000000007ce1149a000000000b000100010000001504000000000000100000000000000000000000000000007ce1149a000000000c000100010000002504000000000000100000000000000000000000000000007ce1149a000000000d000100010000003504000000000000100000000000000000000000000000007ce1149a000000000e000100010000004504000000000000100000000000000000000000000000007ce1149a000000000f000100010000005504000000000000100000000000000000000000000000007ce1149a0000000010000100010000006504000000000000100000000000000000000000000000007ce1149a0000000011000100010000007504000000000000100000000000000000000000000000007ce1149a00000000120001000100000085040000000000002c0000000000000001000000000000006e12eaca000000000300000000000000300000000000000000000000000000000e000000000000001b0000000000000021000000000000005f5f657874656e73696f6e735f5f5f5f7265636f7264696e675f5f676f6c64656e01000000180000002800000000000000010000000000000028000000000000000c0000000000000002000000ffffffffffffffff0000000018000000100000000000000000000000180000001000000000000000000000001800000010000000000000000000000018000000100000000000000000000000180000001000000000000000000000001800000010000000000000000000000018000000100000000000000000000000180000001000000000000000000000001800000010000000000000000000000018000000100000000000000000000000180000001000000000000000000000001800000010000000000000000000000018000000100000000000000000000000180000001000000000000000000000001800000010000000000000000100000018000000280000000000000000000000000000002800000000000000040000000000000000000000";
    assert_eq!(actual, expected);
}
