extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Attachment, ChannelDescriptor, Clock, ClockKind, CoordinateFrame, CoordinatePoint, Event,
    Interval, LossReceipt, ModalityId, Property, PropertyBag, ProvenanceActivity, QualifiedName,
    Rational, Recording, RecordingBuilder, RecordingIdentity, ReferenceEdge, ReferenceNode,
    ReferenceNodeKind, Relationship, SampleBuffer, SemanticDisposition, SignalSeries, SignalStream,
    Table, TableColumn, Tensor, TensorBuffer, TimeAxis, Unit, Value, ValueType,
};

use super::{
    crc32, encode_header, Bcs2Error, Bcs2View, DirectoryEntry, SectionKind, BCS2_FLAG_CRC32,
    BCS2_HEADER_LEN, DIRECTORY_ENTRY_LEN, INDEX_ENTRY_LEN, NO_STRING_ID, SECTION_HEADER_LEN,
};

const RECORDING_ID: &str = "__recording__";
const EXTENSIONS_ID: &str = "__extensions__";

struct StringTable {
    values: Vec<String>,
}

impl StringTable {
    fn from_recording(recording: &Recording) -> Result<Self, Bcs2Error> {
        let mut values = BTreeSet::new();
        collect_recording_strings(recording, &mut values);
        if values.len() > u32::MAX as usize {
            return Err(Bcs2Error::LimitExceeded("string count"));
        }
        Ok(Self {
            values: values.into_iter().collect(),
        })
    }

    fn id(&self, value: &str) -> Result<u32, Bcs2Error> {
        let index = self
            .values
            .binary_search_by(|candidate| candidate.as_str().cmp(value))
            .map_err(|_| Bcs2Error::InvalidLayout("uninterned string"))?;
        u32::try_from(index).map_err(|_| Bcs2Error::LimitExceeded("string id"))
    }

    fn encode(&self) -> Result<Vec<u8>, Bcs2Error> {
        let count = u32::try_from(self.values.len())
            .map_err(|_| Bcs2Error::LimitExceeded("string count"))?;
        let offsets_bytes = (self.values.len() + 1)
            .checked_mul(8)
            .ok_or(Bcs2Error::IntegerOverflow("string offsets"))?;
        let data_offset = SECTION_HEADER_LEN
            .checked_add(offsets_bytes)
            .ok_or(Bcs2Error::IntegerOverflow("string data offset"))?;
        let mut out = Vec::new();
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&(data_offset as u64).to_le_bytes());
        let mut cursor = 0_u64;
        out.extend_from_slice(&cursor.to_le_bytes());
        for value in self.values.iter() {
            cursor = cursor
                .checked_add(value.len() as u64)
                .ok_or(Bcs2Error::IntegerOverflow("string data"))?;
            out.extend_from_slice(&cursor.to_le_bytes());
        }
        for value in self.values.iter() {
            out.extend_from_slice(value.as_bytes());
        }
        Ok(out)
    }
}

struct Record {
    id: String,
    bytes: Vec<u8>,
}

struct BuiltSection {
    kind: SectionKind,
    item_count: u64,
    bytes: Vec<u8>,
}

/// Encode a verified ABIR2 recording into canonical BCS2 bytes.
pub fn encode_bcs2(recording: &Recording) -> Result<Vec<u8>, Bcs2Error> {
    recording
        .verify()
        .map_err(|error| Bcs2Error::Graph(error.to_string()))?;
    let strings = StringTable::from_recording(recording)?;
    let mut sections = Vec::new();
    sections.push(BuiltSection {
        kind: SectionKind::Strings,
        item_count: strings.values.len() as u64,
        bytes: strings.encode()?,
    });
    sections.push(build_indexed_section(
        SectionKind::Identity,
        vec![Record {
            id: RECORDING_ID.to_string(),
            bytes: encode_identity(recording.identity(), &strings)?,
        }],
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Clocks,
        recording.clocks().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_clock(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::SignalStreams,
        recording.signal_streams().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_stream(item, &strings)?,
            })
        }),
        &strings,
    )?);
    let mut series_records = Vec::new();
    for stream in recording.signal_streams() {
        for series in stream.series() {
            series_records.push(Record {
                id: series.channel().id().to_string(),
                bytes: encode_series(stream.id(), series, &strings)?,
            });
        }
    }
    sections.push(build_indexed_section(
        SectionKind::SignalSeries,
        series_records,
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Events,
        recording.events().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_event(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Intervals,
        recording.intervals().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_interval(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Tables,
        recording.tables().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_table(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Tensors,
        recording.tensors().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_tensor(item)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::CoordinateFrames,
        recording.coordinate_frames().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_coordinate_frame(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Coordinates,
        recording.coordinates().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_coordinate(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::ReferenceNodes,
        recording.reference_nodes().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_reference_node(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::ReferenceEdges,
        recording.reference_edges().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_reference_edge(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Relationships,
        recording.relationships().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_relationship(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Attachments,
        recording.attachments().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_attachment(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::Provenance,
        recording.provenance().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_provenance(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_records(
        SectionKind::LossReceipts,
        recording.loss_receipts().iter().map(|item| {
            Ok(Record {
                id: item.id().to_string(),
                bytes: encode_loss_receipt(item, &strings)?,
            })
        }),
        &strings,
    )?);
    sections.push(build_indexed_section(
        SectionKind::Extensions,
        vec![Record {
            id: EXTENSIONS_ID.to_string(),
            bytes: encode_properties(recording.extensions(), &strings)?,
        }],
        &strings,
    )?);
    sections.sort_by_key(|section| section.kind);

    let section_count =
        u32::try_from(sections.len()).map_err(|_| Bcs2Error::LimitExceeded("section count"))?;
    let directory_length = sections
        .len()
        .checked_mul(DIRECTORY_ENTRY_LEN)
        .ok_or(Bcs2Error::IntegerOverflow("directory length"))?;
    let mut next_offset = BCS2_HEADER_LEN
        .checked_add(directory_length)
        .ok_or(Bcs2Error::IntegerOverflow("first section offset"))?
        as u64;
    let mut entries = Vec::with_capacity(sections.len());
    for section in sections.iter() {
        let length = section.bytes.len() as u64;
        entries.push(DirectoryEntry {
            kind_raw: section.kind as u16,
            version: 1,
            flags: BCS2_FLAG_CRC32 as u32,
            offset: next_offset,
            length,
            item_count: section.item_count,
            checksum: crc32(&section.bytes),
        });
        next_offset = next_offset
            .checked_add(length)
            .ok_or(Bcs2Error::IntegerOverflow("file length"))?;
    }

    let mut payload = Vec::new();
    for entry in entries.iter() {
        entry.encode(&mut payload);
    }
    for section in sections.iter() {
        payload.extend_from_slice(&section.bytes);
    }
    let file_length = (BCS2_HEADER_LEN as u64)
        .checked_add(payload.len() as u64)
        .ok_or(Bcs2Error::IntegerOverflow("file length"))?;
    let header = encode_header(section_count, file_length, &payload);
    let mut out = Vec::with_capacity(BCS2_HEADER_LEN + payload.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&payload);
    Ok(out)
}

fn build_records<I>(
    kind: SectionKind,
    records: I,
    strings: &StringTable,
) -> Result<BuiltSection, Bcs2Error>
where
    I: IntoIterator<Item = Result<Record, Bcs2Error>>,
{
    build_indexed_section(
        kind,
        records.into_iter().collect::<Result<Vec<_>, _>>()?,
        strings,
    )
}

fn build_indexed_section(
    kind: SectionKind,
    mut records: Vec<Record>,
    strings: &StringTable,
) -> Result<BuiltSection, Bcs2Error> {
    records.sort_by(|left, right| left.id.cmp(&right.id));
    if let Some(pair) = records.windows(2).find(|pair| pair[0].id == pair[1].id) {
        return Err(Bcs2Error::DuplicateId(pair[0].id.clone()));
    }
    let count =
        u32::try_from(records.len()).map_err(|_| Bcs2Error::LimitExceeded("record count"))?;
    let payload_offset = SECTION_HEADER_LEN
        .checked_add(
            records
                .len()
                .checked_mul(INDEX_ENTRY_LEN)
                .ok_or(Bcs2Error::IntegerOverflow("record index"))?,
        )
        .ok_or(Bcs2Error::IntegerOverflow("record payload offset"))?;
    let mut out = Vec::new();
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(INDEX_ENTRY_LEN as u32).to_le_bytes());
    out.extend_from_slice(&(payload_offset as u64).to_le_bytes());
    let mut offset = payload_offset as u64;
    for record in records.iter() {
        out.extend_from_slice(&strings.id(&record.id)?.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(record.bytes.len() as u64).to_le_bytes());
        offset = offset
            .checked_add(record.bytes.len() as u64)
            .ok_or(Bcs2Error::IntegerOverflow("record offset"))?;
    }
    for record in records {
        out.extend_from_slice(&record.bytes);
    }
    Ok(BuiltSection {
        kind,
        item_count: u64::from(count),
        bytes: out,
    })
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn string(&mut self, value: &str, strings: &StringTable) -> Result<(), Bcs2Error> {
        self.u32(strings.id(value)?);
        Ok(())
    }
    fn qname(&mut self, value: &QualifiedName, strings: &StringTable) -> Result<(), Bcs2Error> {
        self.string(value.namespace(), strings)?;
        self.string(value.local(), strings)
    }
    fn bytes(self) -> Vec<u8> {
        self.bytes
    }
}

fn encode_identity(
    identity: &RecordingIdentity,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(identity.subject(), strings)?;
    out.u32(
        identity
            .session()
            .map_or(Ok(NO_STRING_ID), |value| strings.id(value))?,
    );
    out.u32(
        identity
            .run()
            .map_or(Ok(NO_STRING_ID), |value| strings.id(value))?,
    );
    Ok(out.bytes())
}

fn encode_clock(clock: &Clock, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    match clock.kind() {
        ClockKind::Device => out.u8(1),
        ClockKind::UnixUtc => out.u8(2),
        ClockKind::Relative => out.u8(3),
        ClockKind::Other(name) => {
            out.u8(4);
            out.qname(name, strings)?;
        }
    }
    out.u64(clock.tick_rate().numerator());
    out.u64(clock.tick_rate().denominator());
    Ok(out.bytes())
}

fn encode_stream(stream: &SignalStream, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(stream.modality().as_str(), strings)?;
    out.u32(
        u32::try_from(stream.series().len())
            .map_err(|_| Bcs2Error::LimitExceeded("series count"))?,
    );
    for series in stream.series() {
        out.string(series.channel().id(), strings)?;
    }
    Ok(out.bytes())
}

fn encode_series(
    stream_id: &str,
    series: &SignalSeries,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(stream_id, strings)?;
    out.string(series.channel().label(), strings)?;
    out.string(series.channel().modality().as_str(), strings)?;
    out.string(series.channel().unit().system(), strings)?;
    out.string(series.channel().unit().as_str(), strings)?;
    match series.time_axis() {
        TimeAxis::Uniform { .. } => {
            out.u8(1);
            out.string(series.time_axis().clock_id(), strings)?;
            out.i64(
                series
                    .time_axis()
                    .start_tick()
                    .ok_or(Bcs2Error::InvalidLayout("uniform start tick"))?,
            );
            let rate = series
                .time_axis()
                .sample_rate()
                .ok_or(Bcs2Error::InvalidLayout("uniform rate"))?;
            out.u64(rate.numerator());
            out.u64(rate.denominator());
        }
        TimeAxis::Explicit { .. } => {
            out.u8(2);
            out.string(series.time_axis().clock_id(), strings)?;
            let ticks = series
                .time_axis()
                .explicit_ticks()
                .ok_or(Bcs2Error::InvalidLayout("explicit ticks"))?;
            out.u64(ticks.len() as u64);
            for tick in ticks {
                out.i64(*tick);
            }
        }
    }
    encode_sample_buffer(series.samples(), &mut out);
    Ok(out.bytes())
}

fn encode_sample_buffer(buffer: &SampleBuffer, out: &mut Encoder) {
    match buffer {
        SampleBuffer::I8(values) => {
            out.u8(1);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.u8(*value as u8);
            }
        }
        SampleBuffer::U8(values) => {
            out.u8(2);
            out.u64(values.len() as u64);
            out.bytes.extend_from_slice(values);
        }
        SampleBuffer::I16(values) => {
            out.u8(3);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        SampleBuffer::U16(values) => {
            out.u8(4);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        SampleBuffer::I32(values) => {
            out.u8(5);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        SampleBuffer::U32(values) => {
            out.u8(6);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        SampleBuffer::I64(values) => {
            out.u8(7);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        SampleBuffer::F32(values) => {
            out.u8(8);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        SampleBuffer::F64(values) => {
            out.u8(9);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
}

fn encode_event(event: &Event, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(event.clock_id(), strings)?;
    out.i64(event.tick());
    out.qname(event.label(), strings)?;
    out.bytes
        .extend_from_slice(&encode_properties(event.properties(), strings)?);
    Ok(out.bytes())
}

fn encode_interval(interval: &Interval, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(interval.clock_id(), strings)?;
    out.i64(interval.start_tick());
    out.i64(interval.end_tick());
    out.qname(interval.label(), strings)?;
    Ok(out.bytes())
}

fn encode_table(table: &Table, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.u32(
        u32::try_from(table.columns().len())
            .map_err(|_| Bcs2Error::LimitExceeded("table columns"))?,
    );
    for column in table.columns() {
        out.qname(column.name(), strings)?;
        out.u8(value_type_tag(column.value_type()));
        out.u64(column.values().len() as u64);
        for value in column.values() {
            encode_value(value, strings, &mut out)?;
        }
    }
    Ok(out.bytes())
}

fn encode_tensor(tensor: &Tensor) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.u32(
        u32::try_from(tensor.shape().len()).map_err(|_| Bcs2Error::LimitExceeded("tensor rank"))?,
    );
    for dimension in tensor.shape() {
        out.u64(*dimension);
    }
    encode_tensor_buffer(tensor.buffer(), &mut out);
    Ok(out.bytes())
}

fn encode_tensor_buffer(buffer: &TensorBuffer, out: &mut Encoder) {
    match buffer {
        TensorBuffer::I8(values) => {
            out.u8(1);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.u8(*value as u8);
            }
        }
        TensorBuffer::U8(values) => {
            out.u8(2);
            out.u64(values.len() as u64);
            out.bytes.extend_from_slice(values);
        }
        TensorBuffer::I16(values) => {
            out.u8(3);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        TensorBuffer::U16(values) => {
            out.u8(4);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        TensorBuffer::I32(values) => {
            out.u8(5);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        TensorBuffer::U32(values) => {
            out.u8(6);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        TensorBuffer::I64(values) => {
            out.u8(7);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        TensorBuffer::F32(values) => {
            out.u8(8);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        TensorBuffer::F64(values) => {
            out.u8(9);
            out.u64(values.len() as u64);
            for value in values.iter() {
                out.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
}

fn encode_coordinate_frame(
    frame: &CoordinateFrame,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.u64(frame.dimension() as u64);
    out.qname(frame.system(), strings)?;
    Ok(out.bytes())
}

fn encode_coordinate(point: &CoordinatePoint, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(point.frame_id(), strings)?;
    out.string(point.object_id(), strings)?;
    out.u32(
        u32::try_from(point.values().len())
            .map_err(|_| Bcs2Error::LimitExceeded("coordinate dimension"))?,
    );
    for value in point.values() {
        out.u64(value.to_bits());
    }
    out.string(point.unit().system(), strings)?;
    out.string(point.unit().as_str(), strings)?;
    Ok(out.bytes())
}

fn encode_reference_node(
    node: &ReferenceNode,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    match node.kind() {
        ReferenceNodeKind::Channel => out.u8(1),
        ReferenceNodeKind::PhysicalReference => out.u8(2),
        ReferenceNodeKind::DerivedReference => out.u8(3),
        ReferenceNodeKind::Ground => out.u8(4),
        ReferenceNodeKind::Other(name) => {
            out.u8(5);
            out.qname(name, strings)?;
        }
    }
    Ok(out.bytes())
}

fn encode_reference_edge(
    edge: &ReferenceEdge,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(edge.from(), strings)?;
    out.string(edge.to(), strings)?;
    out.qname(edge.label(), strings)?;
    Ok(out.bytes())
}

fn encode_relationship(
    relation: &Relationship,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(relation.subject(), strings)?;
    out.qname(relation.predicate(), strings)?;
    out.string(relation.object(), strings)?;
    Ok(out.bytes())
}

fn encode_attachment(attachment: &Attachment, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.string(attachment.media_type(), strings)?;
    out.u64(attachment.bytes().len() as u64);
    out.bytes.extend_from_slice(attachment.bytes());
    Ok(out.bytes())
}

fn encode_provenance(
    activity: &ProvenanceActivity,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.qname(activity.activity(), strings)?;
    out.string(activity.software(), strings)?;
    out.u32(
        u32::try_from(activity.inputs().len())
            .map_err(|_| Bcs2Error::LimitExceeded("provenance inputs"))?,
    );
    for input in activity.inputs() {
        out.string(input, strings)?;
    }
    out.u32(
        u32::try_from(activity.outputs().len())
            .map_err(|_| Bcs2Error::LimitExceeded("provenance outputs"))?,
    );
    for output in activity.outputs() {
        out.string(output, strings)?;
    }
    Ok(out.bytes())
}

fn encode_loss_receipt(receipt: &LossReceipt, strings: &StringTable) -> Result<Vec<u8>, Bcs2Error> {
    let mut out = Encoder::new();
    out.qname(receipt.label(), strings)?;
    out.u8(disposition_tag(receipt.disposition()));
    match receipt.extension() {
        Some(name) => {
            out.u8(1);
            out.qname(name, strings)?;
        }
        None => out.u8(0),
    }
    out.string(receipt.details(), strings)?;
    Ok(out.bytes())
}

fn encode_properties(
    properties: &PropertyBag,
    strings: &StringTable,
) -> Result<Vec<u8>, Bcs2Error> {
    let mut ordered: Vec<&Property> = properties.properties().iter().collect();
    ordered.sort_by(|left, right| {
        (left.name().namespace(), left.name().local())
            .cmp(&(right.name().namespace(), right.name().local()))
    });
    let mut out = Encoder::new();
    out.u32(u32::try_from(ordered.len()).map_err(|_| Bcs2Error::LimitExceeded("property count"))?);
    for property in ordered {
        out.qname(property.name(), strings)?;
        encode_value(property.value(), strings, &mut out)?;
    }
    Ok(out.bytes())
}

fn encode_value(value: &Value, strings: &StringTable, out: &mut Encoder) -> Result<(), Bcs2Error> {
    match value {
        Value::Null => out.u8(0),
        Value::Bool(value) => {
            out.u8(1);
            out.u8(u8::from(*value));
        }
        Value::I64(value) => {
            out.u8(2);
            out.i64(*value);
        }
        Value::U64(value) => {
            out.u8(3);
            out.u64(*value);
        }
        Value::F64(bits) => {
            out.u8(4);
            out.u64(*bits);
        }
        Value::Rational(value) => {
            out.u8(5);
            out.u64(value.numerator());
            out.u64(value.denominator());
        }
        Value::Text(value) => {
            out.u8(6);
            out.string(value, strings)?;
        }
        Value::Bytes(value) => {
            out.u8(7);
            out.u64(value.len() as u64);
            out.bytes.extend_from_slice(value);
        }
        Value::List(values) => {
            out.u8(8);
            out.u32(
                u32::try_from(values.len()).map_err(|_| Bcs2Error::LimitExceeded("value list"))?,
            );
            for value in values.iter() {
                encode_value(value, strings, out)?;
            }
        }
        Value::Record(properties) => {
            out.u8(9);
            out.bytes
                .extend_from_slice(&encode_properties(properties, strings)?);
        }
    }
    Ok(())
}

fn value_type_tag(value: ValueType) -> u8 {
    match value {
        ValueType::Null => 0,
        ValueType::Bool => 1,
        ValueType::I64 => 2,
        ValueType::U64 => 3,
        ValueType::F64 => 4,
        ValueType::Rational => 5,
        ValueType::Text => 6,
        ValueType::Bytes => 7,
        ValueType::List => 8,
        ValueType::Record => 9,
    }
}

fn disposition_tag(value: SemanticDisposition) -> u8 {
    match value {
        SemanticDisposition::Exact => 1,
        SemanticDisposition::Normalized => 2,
        SemanticDisposition::PreservedAsExtension => 3,
        SemanticDisposition::Approximated => 4,
        SemanticDisposition::Dropped => 5,
    }
}

fn collect_recording_strings(recording: &Recording, values: &mut BTreeSet<String>) {
    insert(values, RECORDING_ID);
    insert(values, EXTENSIONS_ID);
    insert(values, recording.identity().subject());
    if let Some(value) = recording.identity().session() {
        insert(values, value);
    }
    if let Some(value) = recording.identity().run() {
        insert(values, value);
    }
    for clock in recording.clocks() {
        insert(values, clock.id());
        if let ClockKind::Other(name) = clock.kind() {
            collect_qname(name, values);
        }
    }
    for stream in recording.signal_streams() {
        insert(values, stream.id());
        insert(values, stream.modality().as_str());
        for series in stream.series() {
            let channel = series.channel();
            insert(values, channel.id());
            insert(values, channel.label());
            insert(values, channel.modality().as_str());
            insert(values, channel.unit().system());
            insert(values, channel.unit().as_str());
            insert(values, series.time_axis().clock_id());
        }
    }
    for event in recording.events() {
        insert(values, event.id());
        insert(values, event.clock_id());
        collect_qname(event.label(), values);
        collect_properties(event.properties(), values);
    }
    for interval in recording.intervals() {
        insert(values, interval.id());
        insert(values, interval.clock_id());
        collect_qname(interval.label(), values);
    }
    for table in recording.tables() {
        insert(values, table.id());
        for column in table.columns() {
            collect_qname(column.name(), values);
            for value in column.values() {
                collect_value(value, values);
            }
        }
    }
    for tensor in recording.tensors() {
        insert(values, tensor.id());
    }
    for frame in recording.coordinate_frames() {
        insert(values, frame.id());
        collect_qname(frame.system(), values);
    }
    for point in recording.coordinates() {
        insert(values, point.id());
        insert(values, point.frame_id());
        insert(values, point.object_id());
        insert(values, point.unit().system());
        insert(values, point.unit().as_str());
    }
    for node in recording.reference_nodes() {
        insert(values, node.id());
        if let ReferenceNodeKind::Other(name) = node.kind() {
            collect_qname(name, values);
        }
    }
    for edge in recording.reference_edges() {
        insert(values, edge.id());
        insert(values, edge.from());
        insert(values, edge.to());
        collect_qname(edge.label(), values);
    }
    for relation in recording.relationships() {
        insert(values, relation.id());
        insert(values, relation.subject());
        insert(values, relation.object());
        collect_qname(relation.predicate(), values);
    }
    for attachment in recording.attachments() {
        insert(values, attachment.id());
        insert(values, attachment.media_type());
    }
    for activity in recording.provenance() {
        insert(values, activity.id());
        collect_qname(activity.activity(), values);
        insert(values, activity.software());
        for input in activity.inputs() {
            insert(values, input);
        }
        for output in activity.outputs() {
            insert(values, output);
        }
    }
    for receipt in recording.loss_receipts() {
        insert(values, receipt.id());
        collect_qname(receipt.label(), values);
        if let Some(name) = receipt.extension() {
            collect_qname(name, values);
        }
        insert(values, receipt.details());
    }
    collect_properties(recording.extensions(), values);
}

fn insert(values: &mut BTreeSet<String>, value: &str) {
    values.insert(value.to_string());
}
fn collect_qname(name: &QualifiedName, values: &mut BTreeSet<String>) {
    insert(values, name.namespace());
    insert(values, name.local());
}
fn collect_properties(properties: &PropertyBag, values: &mut BTreeSet<String>) {
    for property in properties.properties() {
        collect_qname(property.name(), values);
        collect_value(property.value(), values);
    }
}
fn collect_value(value: &Value, values: &mut BTreeSet<String>) {
    match value {
        Value::Text(value) => insert(values, value),
        Value::List(items) => {
            for item in items.iter() {
                collect_value(item, values);
            }
        }
        Value::Record(properties) => collect_properties(properties, values),
        _ => {}
    }
}

// ---- decode ----

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn take(&mut self, length: usize, context: &'static str) -> Result<&'a [u8], Bcs2Error> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(Bcs2Error::IntegerOverflow(context))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(Bcs2Error::Truncated {
                context,
                needed: end,
                available: self.bytes.len(),
            })?;
        self.offset = end;
        Ok(value)
    }
    fn u8(&mut self) -> Result<u8, Bcs2Error> {
        Ok(self.take(1, "u8")?[0])
    }
    fn u32(&mut self) -> Result<u32, Bcs2Error> {
        let b = self.take(4, "u32")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, Bcs2Error> {
        let b = self.take(8, "u64")?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn i64(&mut self) -> Result<i64, Bcs2Error> {
        let b = self.take(8, "i64")?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    fn finish(self) -> Result<(), Bcs2Error> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(Bcs2Error::InvalidLayout("trailing record bytes"))
        }
    }
}

/// Decode a complete BCS2 recording graph.
pub fn decode_bcs2(bytes: &[u8]) -> Result<Recording, Bcs2Error> {
    let view = Bcs2View::parse(bytes)?;
    let identity_records = view.records(SectionKind::Identity)?;
    if identity_records.len() != 1 || identity_records[0].0 != RECORDING_ID {
        return Err(Bcs2Error::InvalidLayout("identity section"));
    }
    let identity = decode_identity(&view, identity_records[0].1)?;
    let mut builder = RecordingBuilder::new(identity);
    for (id, record) in view.records(SectionKind::Clocks)? {
        builder
            .add_clock(decode_clock(&view, id, record)?)
            .map_err(graph_error)?;
    }

    let mut streams = Vec::new();
    for (id, record) in view.records(SectionKind::SignalStreams)? {
        streams.push(decode_stream(&view, id, record)?);
    }
    let mut series_by_id: BTreeMap<String, (String, SignalSeries)> = BTreeMap::new();
    for (id, record) in view.records(SectionKind::SignalSeries)? {
        let (stream_id, series) = decode_signal_series_record(&view, id, record)?;
        series_by_id.insert(id.to_string(), (stream_id.to_string(), series));
    }
    for descriptor in streams {
        let mut stream = SignalStream::new(&descriptor.id, descriptor.modality);
        for channel_id in descriptor.channels {
            let (actual_stream, series) = series_by_id.remove(&channel_id).ok_or_else(|| {
                Bcs2Error::Graph(format!(
                    "stream '{}' references missing series '{}'",
                    descriptor.id, channel_id
                ))
            })?;
            if actual_stream != descriptor.id {
                return Err(Bcs2Error::Graph(format!(
                    "series '{}' belongs to '{}' not '{}'",
                    channel_id, actual_stream, descriptor.id
                )));
            }
            stream = stream.with_series(series);
        }
        builder.add_signal_stream(stream).map_err(graph_error)?;
    }
    if !series_by_id.is_empty() {
        return Err(Bcs2Error::Graph("orphan signal series".to_string()));
    }

    for (id, record) in view.records(SectionKind::Events)? {
        builder
            .add_event(decode_event(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Intervals)? {
        builder
            .add_interval(decode_interval(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Tables)? {
        builder
            .add_table(decode_table(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Tensors)? {
        builder
            .add_tensor(decode_tensor(id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::CoordinateFrames)? {
        builder
            .add_coordinate_frame(decode_coordinate_frame(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Coordinates)? {
        builder
            .add_coordinate(decode_coordinate(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::ReferenceNodes)? {
        builder
            .add_reference_node(decode_reference_node(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::ReferenceEdges)? {
        builder
            .add_reference_edge(decode_reference_edge(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Relationships)? {
        builder
            .add_relationship(decode_relationship(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Attachments)? {
        builder
            .add_attachment(decode_attachment(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::Provenance)? {
        builder
            .add_provenance(decode_provenance(&view, id, record)?)
            .map_err(graph_error)?;
    }
    for (id, record) in view.records(SectionKind::LossReceipts)? {
        builder
            .add_loss_receipt(decode_loss_receipt(&view, id, record)?)
            .map_err(graph_error)?;
    }
    let extension_records = view.records(SectionKind::Extensions)?;
    if extension_records.len() != 1 || extension_records[0].0 != EXTENSIONS_ID {
        return Err(Bcs2Error::InvalidLayout("extensions section"));
    }
    let mut cursor = Cursor::new(extension_records[0].1);
    let extensions = decode_properties(&view, &mut cursor, 0)?;
    cursor.finish()?;
    builder.set_extensions(extensions);
    builder.freeze().map_err(graph_error)
}

fn graph_error(error: crate::RecordingError) -> Bcs2Error {
    Bcs2Error::Graph(error.to_string())
}

struct StreamDescriptor {
    id: String,
    modality: ModalityId,
    channels: Vec<String>,
}

fn decode_identity(view: &Bcs2View<'_>, bytes: &[u8]) -> Result<RecordingIdentity, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let subject = view.string(c.u32()?)?;
    let session = decode_optional_string(view, c.u32()?)?;
    let run = decode_optional_string(view, c.u32()?)?;
    c.finish()?;
    Ok(RecordingIdentity::new(subject, session, run))
}
fn decode_optional_string<'a>(
    view: &'a Bcs2View<'a>,
    id: u32,
) -> Result<Option<&'a str>, Bcs2Error> {
    if id == NO_STRING_ID {
        Ok(None)
    } else {
        Ok(Some(view.string(id)?))
    }
}
fn decode_qname(view: &Bcs2View<'_>, c: &mut Cursor<'_>) -> Result<QualifiedName, Bcs2Error> {
    Ok(QualifiedName::new(
        view.string(c.u32()?)?,
        view.string(c.u32()?)?,
    ))
}
fn rational(n: u64, d: u64) -> Result<Rational, Bcs2Error> {
    Rational::new(n, d).ok_or(Bcs2Error::InvalidLayout("zero rational denominator"))
}

fn decode_clock(view: &Bcs2View<'_>, id: &str, bytes: &[u8]) -> Result<Clock, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let tag = c.u8()?;
    let kind = match tag {
        1 => ClockKind::Device,
        2 => ClockKind::UnixUtc,
        3 => ClockKind::Relative,
        4 => ClockKind::Other(decode_qname(view, &mut c)?),
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "clock kind",
                tag,
            })
        }
    };
    let rate = rational(c.u64()?, c.u64()?)?;
    c.finish()?;
    Ok(Clock::new(id, kind, rate))
}

fn decode_stream(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<StreamDescriptor, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let modality = ModalityId::new(view.string(c.u32()?)?);
    let raw_count = c.u32()?;
    let count = bounded_count(&c, u64::from(raw_count), 4, "stream channel count")?;
    let mut channels = Vec::with_capacity(count);
    for _ in 0..count {
        channels.push(view.string(c.u32()?)?.to_string());
    }
    c.finish()?;
    Ok(StreamDescriptor {
        id: id.to_string(),
        modality,
        channels,
    })
}

pub(crate) fn decode_signal_series_record<'a>(
    view: &Bcs2View<'a>,
    channel_id: &str,
    bytes: &[u8],
) -> Result<(&'a str, SignalSeries), Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let stream_id = view.string(c.u32()?)?;
    let label = view.string(c.u32()?)?;
    let modality = ModalityId::new(view.string(c.u32()?)?);
    let unit = Unit::new(view.string(c.u32()?)?, view.string(c.u32()?)?);
    let axis_tag = c.u8()?;
    let clock = view.string(c.u32()?)?;
    let axis = match axis_tag {
        1 => {
            let start = c.i64()?;
            let rate = rational(c.u64()?, c.u64()?)?;
            TimeAxis::uniform(clock, start, rate)
        }
        2 => {
            let raw_count = c.u64()?;
            let count = bounded_count(&c, raw_count, 8, "tick count")?;
            let mut ticks = Vec::with_capacity(count);
            for _ in 0..count {
                ticks.push(c.i64()?);
            }
            TimeAxis::explicit(clock, ticks.into())
        }
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "time axis",
                tag: axis_tag,
            })
        }
    };
    let samples = decode_sample_buffer(&mut c)?;
    c.finish()?;
    Ok((
        stream_id,
        SignalSeries::new(
            ChannelDescriptor::new(channel_id, label, modality, unit),
            axis,
            samples,
        ),
    ))
}

fn decode_sample_buffer(c: &mut Cursor<'_>) -> Result<SampleBuffer, Bcs2Error> {
    let tag = c.u8()?;
    let raw_count = c.u64()?;
    let width = scalar_width(tag).ok_or(Bcs2Error::InvalidTag {
        context: "sample dtype",
        tag,
    })?;
    let count = bounded_count(c, raw_count, width, "sample count")?;
    macro_rules! read_vec {
        ($ty:ty,$size:expr,$conv:expr) => {{
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let b = c.take($size, "sample data")?;
                v.push($conv(b));
            }
            v.into()
        }};
    }
    Ok(match tag {
        1 => SampleBuffer::from_i8(read_vec!(i8, 1, |b: &[u8]| b[0] as i8)),
        2 => SampleBuffer::from_u8(Arc::from(c.take(count, "sample data")?)),
        3 => SampleBuffer::from_i16(read_vec!(i16, 2, |b: &[u8]| i16::from_le_bytes([
            b[0], b[1]
        ]))),
        4 => SampleBuffer::from_u16(read_vec!(u16, 2, |b: &[u8]| u16::from_le_bytes([
            b[0], b[1]
        ]))),
        5 => SampleBuffer::from_i32(read_vec!(i32, 4, |b: &[u8]| i32::from_le_bytes([
            b[0], b[1], b[2], b[3]
        ]))),
        6 => SampleBuffer::from_u32(read_vec!(u32, 4, |b: &[u8]| u32::from_le_bytes([
            b[0], b[1], b[2], b[3]
        ]))),
        7 => SampleBuffer::from_i64(read_vec!(i64, 8, |b: &[u8]| i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
        ]))),
        8 => SampleBuffer::from_f32(read_vec!(f32, 4, |b: &[u8]| f32::from_bits(
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        ))),
        9 => SampleBuffer::from_f64(read_vec!(f64, 8, |b: &[u8]| f64::from_bits(
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        ))),
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "sample dtype",
                tag,
            })
        }
    })
}

fn decode_event(view: &Bcs2View<'_>, id: &str, bytes: &[u8]) -> Result<Event, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let clock = view.string(c.u32()?)?;
    let tick = c.i64()?;
    let label = decode_qname(view, &mut c)?;
    let props = decode_properties(view, &mut c, 0)?;
    c.finish()?;
    Ok(Event::new(id, clock, tick, label).with_properties(props))
}
fn decode_interval(view: &Bcs2View<'_>, id: &str, bytes: &[u8]) -> Result<Interval, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let value = Interval::new(
        id,
        view.string(c.u32()?)?,
        c.i64()?,
        c.i64()?,
        decode_qname(view, &mut c)?,
    );
    c.finish()?;
    Ok(value)
}
fn decode_table(view: &Bcs2View<'_>, id: &str, bytes: &[u8]) -> Result<Table, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let raw_count = c.u32()?;
    let count = bounded_count(&c, u64::from(raw_count), 17, "table column count")?;
    let mut table = Table::new(id);
    for _ in 0..count {
        let name = decode_qname(view, &mut c)?;
        let ty = decode_value_type(c.u8()?)?;
        let raw_values_count = c.u64()?;
        let values_count = bounded_count(&c, raw_values_count, 1, "table values")?;
        let mut values = Vec::with_capacity(values_count);
        for _ in 0..values_count {
            values.push(decode_value(view, &mut c, 0)?);
        }
        table = table.with_column(TableColumn::new(name, ty, values.into()));
    }
    c.finish()?;
    Ok(table)
}
fn decode_tensor(id: &str, bytes: &[u8]) -> Result<Tensor, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let raw_rank = c.u32()?;
    let rank = bounded_count(&c, u64::from(raw_rank), 8, "tensor rank")?;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(c.u64()?);
    }
    let buffer = decode_tensor_buffer(&mut c)?;
    c.finish()?;
    Ok(Tensor::new(id, shape.into(), buffer))
}
fn decode_tensor_buffer(c: &mut Cursor<'_>) -> Result<TensorBuffer, Bcs2Error> {
    let tag = c.u8()?;
    let raw_count = c.u64()?;
    let width = scalar_width(tag).ok_or(Bcs2Error::InvalidTag {
        context: "tensor dtype",
        tag,
    })?;
    let count = bounded_count(c, raw_count, width, "tensor elements")?;
    macro_rules! read_vec {
        ($size:expr,$conv:expr) => {{
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                let b = c.take($size, "tensor data")?;
                v.push($conv(b));
            }
            v.into()
        }};
    }
    Ok(match tag {
        1 => TensorBuffer::from_i8(read_vec!(1, |b: &[u8]| b[0] as i8)),
        2 => TensorBuffer::from_u8(Arc::from(c.take(count, "tensor data")?)),
        3 => TensorBuffer::from_i16(read_vec!(2, |b: &[u8]| i16::from_le_bytes([b[0], b[1]]))),
        4 => TensorBuffer::from_u16(read_vec!(2, |b: &[u8]| u16::from_le_bytes([b[0], b[1]]))),
        5 => TensorBuffer::from_i32(read_vec!(4, |b: &[u8]| i32::from_le_bytes([
            b[0], b[1], b[2], b[3]
        ]))),
        6 => TensorBuffer::from_u32(read_vec!(4, |b: &[u8]| u32::from_le_bytes([
            b[0], b[1], b[2], b[3]
        ]))),
        7 => TensorBuffer::from_i64(read_vec!(8, |b: &[u8]| i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
        ]))),
        8 => TensorBuffer::from_f32(read_vec!(4, |b: &[u8]| f32::from_bits(u32::from_le_bytes(
            [b[0], b[1], b[2], b[3]]
        )))),
        9 => TensorBuffer::from_f64(read_vec!(8, |b: &[u8]| f64::from_bits(u64::from_le_bytes(
            [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]
        )))),
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "tensor dtype",
                tag,
            })
        }
    })
}

fn decode_coordinate_frame(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<CoordinateFrame, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let dimension =
        usize::try_from(c.u64()?).map_err(|_| Bcs2Error::LimitExceeded("coordinate dimension"))?;
    let system = decode_qname(view, &mut c)?;
    c.finish()?;
    Ok(CoordinateFrame::new(id, dimension, system))
}
fn decode_coordinate(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<CoordinatePoint, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let frame = view.string(c.u32()?)?;
    let object = view.string(c.u32()?)?;
    let raw_count = c.u32()?;
    let count = bounded_count(&c, u64::from(raw_count), 8, "coordinate dimension")?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(f64::from_bits(c.u64()?));
    }
    let unit = Unit::new(view.string(c.u32()?)?, view.string(c.u32()?)?);
    c.finish()?;
    Ok(CoordinatePoint::new(id, frame, object, values.into(), unit))
}
fn decode_reference_node(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<ReferenceNode, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let tag = c.u8()?;
    let kind = match tag {
        1 => ReferenceNodeKind::Channel,
        2 => ReferenceNodeKind::PhysicalReference,
        3 => ReferenceNodeKind::DerivedReference,
        4 => ReferenceNodeKind::Ground,
        5 => ReferenceNodeKind::Other(decode_qname(view, &mut c)?),
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "reference node",
                tag,
            })
        }
    };
    c.finish()?;
    Ok(ReferenceNode::new(id, kind))
}
fn decode_reference_edge(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<ReferenceEdge, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let from = view.string(c.u32()?)?;
    let to = view.string(c.u32()?)?;
    let label = decode_qname(view, &mut c)?;
    c.finish()?;
    Ok(ReferenceEdge::new(id, from, to, label))
}
fn decode_relationship(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<Relationship, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let subject = view.string(c.u32()?)?;
    let predicate = decode_qname(view, &mut c)?;
    let object = view.string(c.u32()?)?;
    c.finish()?;
    Ok(Relationship::new(id, subject, predicate, object))
}
fn decode_attachment(view: &Bcs2View<'_>, id: &str, bytes: &[u8]) -> Result<Attachment, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let media = view.string(c.u32()?)?;
    let length = decode_len(c.u64()?, "attachment")?;
    let payload = Arc::from(c.take(length, "attachment")?);
    c.finish()?;
    Ok(Attachment::new(id, media, payload))
}
fn decode_provenance(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<ProvenanceActivity, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let mut value =
        ProvenanceActivity::new(id, decode_qname(view, &mut c)?, view.string(c.u32()?)?);
    let inputs = c.u32()?;
    bounded_count(&c, u64::from(inputs), 4, "provenance inputs")?;
    for _ in 0..inputs {
        value = value.with_input(view.string(c.u32()?)?);
    }
    let outputs = c.u32()?;
    bounded_count(&c, u64::from(outputs), 4, "provenance outputs")?;
    for _ in 0..outputs {
        value = value.with_output(view.string(c.u32()?)?);
    }
    c.finish()?;
    Ok(value)
}
fn decode_loss_receipt(
    view: &Bcs2View<'_>,
    id: &str,
    bytes: &[u8],
) -> Result<LossReceipt, Bcs2Error> {
    let mut c = Cursor::new(bytes);
    let label = decode_qname(view, &mut c)?;
    let disposition = decode_disposition(c.u8()?)?;
    let extension = match c.u8()? {
        0 => None,
        1 => Some(decode_qname(view, &mut c)?),
        tag => {
            return Err(Bcs2Error::InvalidTag {
                context: "loss extension",
                tag,
            })
        }
    };
    let details = view.string(c.u32()?)?;
    c.finish()?;
    Ok(LossReceipt::new(id, label, disposition, extension, details))
}

fn decode_properties(
    view: &Bcs2View<'_>,
    c: &mut Cursor<'_>,
    depth: u8,
) -> Result<PropertyBag, Bcs2Error> {
    if depth >= 64 {
        return Err(Bcs2Error::LimitExceeded("value nesting"));
    }
    let raw_count = c.u32()?;
    let count = bounded_count(c, u64::from(raw_count), 9, "property count")?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let name = decode_qname(view, c)?;
        if values.last().is_some_and(|previous: &Property| {
            (previous.name().namespace(), previous.name().local())
                >= (name.namespace(), name.local())
        }) {
            return Err(Bcs2Error::InvalidLayout("property sort order"));
        }
        values.push(Property::new(name, decode_value(view, c, depth + 1)?));
    }
    Ok(PropertyBag::new(values))
}
fn decode_value(view: &Bcs2View<'_>, c: &mut Cursor<'_>, depth: u8) -> Result<Value, Bcs2Error> {
    if depth >= 64 {
        return Err(Bcs2Error::LimitExceeded("value nesting"));
    }
    let tag = c.u8()?;
    Ok(match tag {
        0 => Value::Null,
        1 => match c.u8()? {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            tag => {
                return Err(Bcs2Error::InvalidTag {
                    context: "bool",
                    tag,
                })
            }
        },
        2 => Value::I64(c.i64()?),
        3 => Value::U64(c.u64()?),
        4 => Value::F64(c.u64()?),
        5 => Value::Rational(rational(c.u64()?, c.u64()?)?),
        6 => Value::text(view.string(c.u32()?)?),
        7 => {
            let length = decode_len(c.u64()?, "value bytes")?;
            Value::bytes(Arc::from(c.take(length, "value bytes")?))
        }
        8 => {
            let raw_count = c.u32()?;
            let count = bounded_count(c, u64::from(raw_count), 1, "value list")?;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(decode_value(view, c, depth + 1)?);
            }
            Value::list(items.into())
        }
        9 => Value::record(decode_properties(view, c, depth + 1)?),
        _ => {
            return Err(Bcs2Error::InvalidTag {
                context: "value",
                tag,
            })
        }
    })
}
fn decode_value_type(tag: u8) -> Result<ValueType, Bcs2Error> {
    match tag {
        0 => Ok(ValueType::Null),
        1 => Ok(ValueType::Bool),
        2 => Ok(ValueType::I64),
        3 => Ok(ValueType::U64),
        4 => Ok(ValueType::F64),
        5 => Ok(ValueType::Rational),
        6 => Ok(ValueType::Text),
        7 => Ok(ValueType::Bytes),
        8 => Ok(ValueType::List),
        9 => Ok(ValueType::Record),
        _ => Err(Bcs2Error::InvalidTag {
            context: "value type",
            tag,
        }),
    }
}
fn decode_disposition(tag: u8) -> Result<SemanticDisposition, Bcs2Error> {
    match tag {
        1 => Ok(SemanticDisposition::Exact),
        2 => Ok(SemanticDisposition::Normalized),
        3 => Ok(SemanticDisposition::PreservedAsExtension),
        4 => Ok(SemanticDisposition::Approximated),
        5 => Ok(SemanticDisposition::Dropped),
        _ => Err(Bcs2Error::InvalidTag {
            context: "semantic disposition",
            tag,
        }),
    }
}
fn decode_len(value: u64, context: &'static str) -> Result<usize, Bcs2Error> {
    usize::try_from(value).map_err(|_| Bcs2Error::LimitExceeded(context))
}

fn bounded_count(
    cursor: &Cursor<'_>,
    value: u64,
    minimum_item_bytes: usize,
    context: &'static str,
) -> Result<usize, Bcs2Error> {
    let count = decode_len(value, context)?;
    if minimum_item_bytes != 0 && count <= cursor.remaining() / minimum_item_bytes {
        Ok(count)
    } else {
        Err(Bcs2Error::InvalidLayout(context))
    }
}

fn scalar_width(tag: u8) -> Option<usize> {
    match tag {
        1 | 2 => Some(1),
        3 | 4 => Some(2),
        5 | 6 | 8 => Some(4),
        7 | 9 => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{bounded_count, Bcs2Error, Cursor};

    #[test]
    fn bounded_count_rejects_hostile_allocation_claims() {
        let cursor = Cursor::new(&[0_u8; 8]);
        assert_eq!(
            bounded_count(&cursor, 9, 1, "hostile count"),
            Err(Bcs2Error::InvalidLayout("hostile count"))
        );
        assert_eq!(
            bounded_count(&cursor, 1, 0, "zero width"),
            Err(Bcs2Error::InvalidLayout("zero width"))
        );
    }
}
