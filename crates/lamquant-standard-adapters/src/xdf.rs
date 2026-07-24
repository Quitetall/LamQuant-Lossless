// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class XDF 1.0 adapter (ADR 0143).
//!
//! XDF is a container for MANY simultaneously recorded streams, each with its
//! own clock, and the whole point of the format is that those clocks are
//! reconciled explicitly rather than assumed synchronous. So this adapter maps:
//!
//! * every XDF stream to its own ABIR `Stream`, atom and `Clock` -- collapsing
//!   them into one stream would assert a shared time base the file denies;
//! * every `ClockOffset` chunk series to a `ClockRelation` from that stream's
//!   clock to the recording host clock, with the observed spread carried as the
//!   relation's uncertainty and the FULL series preserved as its provenance
//!   payload, so no measurement is summarised away;
//! * every `Boundary` chunk to an `Event` on the host clock, because a boundary
//!   marks a real discontinuity in the recording rather than a formatting
//!   detail;
//! * each stream's header and footer XML to an exact blob plus source keys, so
//!   vendor metadata this adapter does not model survives byte-exact.
//!
//! Sample values are promoted in the stream's own declared element type. A
//! `string` stream is carried as an exact blob rather than reinterpreted into
//! numbers it never held.

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use semantic_abir::{
    interchange_content_id, payload_content_id as abir_payload_id, AbirDataset, Atom, AtomTag,
    BlobIntegrity, BlobRef, ByteOrder, Clock, ClockRelation, ClockRelationTag, ClockTag, ConceptId,
    DatasetDraft, DatasetTag, ElementType, Event, EventTag, Layout, ObjectId, PayloadDescriptor,
    Presence, Rational, Recording, RecordingTag, SignalBlock, SourceCapsule, SourceKey, Stream,
    StreamTag, TimeAxis, ValidationLimits,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{binding_namespace, payload_content_id, plan_id, valid_relative_path};

const PROFILE: &str = "xdf.1.0";
const MAGIC: &[u8; 4] = b"XDF:";
const TAG_FILE_HEADER: u16 = 1;
const TAG_STREAM_HEADER: u16 = 2;
const TAG_SAMPLES: u16 = 3;
const TAG_CLOCK_OFFSET: u16 = 4;
const TAG_BOUNDARY: u16 = 5;
const TAG_STREAM_FOOTER: u16 = 6;
/// The fixed UUID every boundary chunk carries as its whole content.
const BOUNDARY_UUID: [u8; 16] = [
    0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E, 0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4,
];
/// Chunk-count ceiling. A malformed length field can otherwise describe an
/// unbounded chunk stream; refusing early keeps a bad file from becoming a
/// memory-exhaustion vector.
const MAX_CHUNKS: usize = 1_000_000;

pub struct XdfAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
}

/// One XDF stream, as read off the wire.
struct XdfStream {
    id: u32,
    header_xml: String,
    footer_xml: Option<String>,
    name: String,
    kind: String,
    channel_count: usize,
    /// Declared nominal rate as EXACT text, so a rational conversion never
    /// rounds through binary floating point.
    nominal_srate: String,
    format: ChannelFormat,
    /// Sample values, channel-major: `values[channel][sample]`.
    values: Vec<Vec<f64>>,
    /// String-format payloads, kept verbatim.
    strings: Vec<Vec<String>>,
    /// Explicit per-sample timestamps, where the file carried them.
    timestamps: Vec<Option<f64>>,
    /// `(collection_time, offset_value)` from every ClockOffset chunk.
    offsets: Vec<(f64, f64)>,
    channel_labels: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChannelFormat {
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Double64,
    StringValues,
}

impl ChannelFormat {
    fn parse(text: &str) -> Result<Self, AdapterError> {
        Ok(match text.trim() {
            "int8" => Self::Int8,
            "int16" => Self::Int16,
            "int32" => Self::Int32,
            "int64" => Self::Int64,
            "float32" => Self::Float32,
            "double64" => Self::Double64,
            "string" => Self::StringValues,
            other => {
                return Err(AdapterError::InvalidSource(format!(
                    "unknown XDF channel_format: {other}"
                )))
            }
        })
    }

    const fn width(self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 | Self::Double64 => 8,
            Self::StringValues => 0,
        }
    }

    const fn element(self) -> ElementType {
        match self {
            Self::Int8 => ElementType::I8,
            Self::Int16 => ElementType::I16,
            Self::Int32 => ElementType::I32,
            Self::Int64 => ElementType::I64,
            Self::Float32 => ElementType::F32,
            Self::Double64 => ElementType::F64,
            Self::StringValues => ElementType::Utf8,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float32 => "float32",
            Self::Double64 => "double64",
            Self::StringValues => "string",
        }
    }
}

struct ParsedXdf {
    file_header_xml: String,
    streams: Vec<XdfStream>,
    boundaries: usize,
}

/// A cursor that never reads past the end and never panics on a short file.
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], AdapterError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| AdapterError::InvalidSource("XDF length overflow".to_owned()))?;
        if end > self.bytes.len() {
            return Err(AdapterError::InvalidSource(
                "XDF chunk runs past the end of the file".to_owned(),
            ));
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, AdapterError> {
        Ok(self.take(1)?[0])
    }

    fn u16le(&mut self) -> Result<u16, AdapterError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32le(&mut self) -> Result<u32, AdapterError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f64le(&mut self) -> Result<f64, AdapterError> {
        let bytes = self.take(8)?;
        let mut buffer = [0_u8; 8];
        buffer.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(buffer))
    }

    /// XDF stores counts as `NumBytes` (1, 4 or 8) followed by that many
    /// little-endian bytes. Any other width is a malformed file.
    fn variable_count(&mut self) -> Result<u64, AdapterError> {
        let width = self.byte()?;
        let bytes = match width {
            1 | 4 | 8 => self.take(width as usize)?,
            other => {
                return Err(AdapterError::InvalidSource(format!(
                    "XDF variable-length count declares {other} bytes; only 1, 4 and 8 are valid"
                )))
            }
        };
        let mut buffer = [0_u8; 8];
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(u64::from_le_bytes(buffer))
    }

    fn done(&self) -> bool {
        self.position >= self.bytes.len()
    }
}

/// Extract the text of the FIRST `<tag>` element at any depth.
///
/// A full XML parser would buy nothing here: XDF stream headers are a fixed,
/// shallow, generated vocabulary, and the exact bytes are preserved verbatim
/// alongside the extracted fields, so anything this misses is still recoverable
/// from the blob rather than silently lost.
fn element_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml[start..end].trim().to_owned())
}

fn channel_labels(xml: &str) -> Vec<String> {
    let mut labels = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<channel>") {
        let after = &rest[start + "<channel>".len()..];
        let Some(end) = after.find("</channel>") else {
            break;
        };
        if let Some(label) = element_text(&after[..end], "label") {
            labels.push(label);
        }
        rest = &after[end..];
    }
    labels
}

/// Parse a decimal literal into an exact rational. XDF writes rates as decimal
/// text; going through `f64` would make an exactly-representable rate like
/// `0.1` inexact for no reason.
fn decimal_rational(text: &str) -> Result<Rational, AdapterError> {
    let trimmed = text.trim();
    let (sign, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1_i128, rest),
        None => (1_i128, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let (whole, fraction) = match digits.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (digits, ""),
    };
    if whole.is_empty() && fraction.is_empty() {
        return Err(AdapterError::InvalidSource(format!(
            "XDF nominal_srate is not a decimal: {text}"
        )));
    }
    let mut numerator: i128 = 0;
    for character in whole.chars().chain(fraction.chars()) {
        let digit = character.to_digit(10).ok_or_else(|| {
            AdapterError::InvalidSource(format!("XDF rate is not decimal: {text}"))
        })?;
        numerator = numerator
            .checked_mul(10)
            .and_then(|value| value.checked_add(i128::from(digit)))
            .ok_or_else(|| AdapterError::InvalidSource("XDF rate overflows".to_owned()))?;
    }
    let mut denominator: i128 = 1;
    for _ in 0..fraction.len() {
        denominator = denominator
            .checked_mul(10)
            .ok_or_else(|| AdapterError::InvalidSource("XDF rate overflows".to_owned()))?;
    }
    Rational::new(sign * numerator, denominator)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn parse_xdf(bytes: &[u8]) -> Result<ParsedXdf, AdapterError> {
    if bytes.len() < 4 || &bytes[..4] != MAGIC {
        return Err(AdapterError::InvalidSource(
            "not an XDF file (missing XDF: magic)".to_owned(),
        ));
    }
    let mut reader = Reader { bytes, position: 4 };
    let mut file_header_xml = String::new();
    let mut streams: BTreeMap<u32, XdfStream> = BTreeMap::new();
    let mut boundaries = 0_usize;
    let mut chunks = 0_usize;

    while !reader.done() {
        chunks += 1;
        if chunks > MAX_CHUNKS {
            return Err(AdapterError::InvalidSource(
                "XDF file declares more chunks than this adapter will read".to_owned(),
            ));
        }
        let length = reader.variable_count()?;
        if length < 2 {
            return Err(AdapterError::InvalidSource(
                "XDF chunk length excludes its own tag".to_owned(),
            ));
        }
        let tag = reader.u16le()?;
        let content_len = usize::try_from(length - 2)
            .map_err(|_| AdapterError::InvalidSource("XDF chunk is too large".to_owned()))?;
        let content = reader.take(content_len)?;
        match tag {
            TAG_FILE_HEADER => {
                file_header_xml = String::from_utf8(content.to_vec()).map_err(|_| {
                    AdapterError::InvalidSource("XDF file header is not UTF-8".to_owned())
                })?;
            }
            TAG_STREAM_HEADER => {
                let mut inner = Reader {
                    bytes: content,
                    position: 0,
                };
                let stream_id = inner.u32le()?;
                let xml = String::from_utf8(content[4..].to_vec()).map_err(|_| {
                    AdapterError::InvalidSource("XDF stream header is not UTF-8".to_owned())
                })?;
                let channel_count = element_text(&xml, "channel_count")
                    .ok_or_else(|| {
                        AdapterError::InvalidSource(
                            "XDF stream header declares no channel_count".to_owned(),
                        )
                    })?
                    .parse::<usize>()
                    .map_err(|_| {
                        AdapterError::InvalidSource(
                            "XDF channel_count is not an integer".to_owned(),
                        )
                    })?;
                if channel_count == 0 {
                    return Err(AdapterError::InvalidSource(
                        "XDF stream declares zero channels".to_owned(),
                    ));
                }
                let format = ChannelFormat::parse(
                    &element_text(&xml, "channel_format").ok_or_else(|| {
                        AdapterError::InvalidSource(
                            "XDF stream header declares no channel_format".to_owned(),
                        )
                    })?,
                )?;
                if streams.contains_key(&stream_id) {
                    return Err(AdapterError::InvalidSource(format!(
                        "XDF file declares stream {stream_id} twice"
                    )));
                }
                streams.insert(
                    stream_id,
                    XdfStream {
                        id: stream_id,
                        name: element_text(&xml, "name").unwrap_or_default(),
                        kind: element_text(&xml, "type").unwrap_or_default(),
                        nominal_srate: element_text(&xml, "nominal_srate")
                            .unwrap_or_else(|| "0".to_owned()),
                        channel_labels: channel_labels(&xml),
                        header_xml: xml,
                        footer_xml: None,
                        channel_count,
                        format,
                        values: vec![Vec::new(); channel_count],
                        strings: vec![Vec::new(); channel_count],
                        timestamps: Vec::new(),
                        offsets: Vec::new(),
                    },
                );
            }
            TAG_SAMPLES => {
                let mut inner = Reader {
                    bytes: content,
                    position: 0,
                };
                let stream_id = inner.u32le()?;
                let stream = streams.get_mut(&stream_id).ok_or_else(|| {
                    AdapterError::InvalidSource(format!(
                        "XDF samples reference stream {stream_id} before its header"
                    ))
                })?;
                let count = inner.variable_count()?;
                let count = usize::try_from(count).map_err(|_| {
                    AdapterError::InvalidSource("XDF sample count is too large".to_owned())
                })?;
                for _ in 0..count {
                    let stamp_bytes = inner.byte()?;
                    let timestamp = match stamp_bytes {
                        0 => None,
                        8 => Some(inner.f64le()?),
                        other => {
                            return Err(AdapterError::InvalidSource(format!(
                                "XDF timestamp declares {other} bytes; only 0 and 8 are valid"
                            )))
                        }
                    };
                    stream.timestamps.push(timestamp);
                    for channel in 0..stream.channel_count {
                        if stream.format == ChannelFormat::StringValues {
                            let length = inner.variable_count()?;
                            let length = usize::try_from(length).map_err(|_| {
                                AdapterError::InvalidSource(
                                    "XDF string value is too large".to_owned(),
                                )
                            })?;
                            let raw = inner.take(length)?;
                            stream.strings[channel].push(String::from_utf8(raw.to_vec()).map_err(
                                |_| {
                                    AdapterError::InvalidSource(
                                        "XDF string value is not UTF-8".to_owned(),
                                    )
                                },
                            )?);
                        } else {
                            let raw = inner.take(stream.format.width())?;
                            stream.values[channel].push(decode_value(stream.format, raw));
                        }
                    }
                }
            }
            TAG_CLOCK_OFFSET => {
                let mut inner = Reader {
                    bytes: content,
                    position: 0,
                };
                let stream_id = inner.u32le()?;
                let collection = inner.f64le()?;
                let offset = inner.f64le()?;
                let stream = streams.get_mut(&stream_id).ok_or_else(|| {
                    AdapterError::InvalidSource(format!(
                        "XDF clock offset references unknown stream {stream_id}"
                    ))
                })?;
                stream.offsets.push((collection, offset));
            }
            TAG_BOUNDARY => {
                if content != BOUNDARY_UUID {
                    return Err(AdapterError::InvalidSource(
                        "XDF boundary chunk carries the wrong UUID".to_owned(),
                    ));
                }
                boundaries += 1;
            }
            TAG_STREAM_FOOTER => {
                let mut inner = Reader {
                    bytes: content,
                    position: 0,
                };
                let stream_id = inner.u32le()?;
                let xml = String::from_utf8(content[4..].to_vec()).map_err(|_| {
                    AdapterError::InvalidSource("XDF stream footer is not UTF-8".to_owned())
                })?;
                let stream = streams.get_mut(&stream_id).ok_or_else(|| {
                    AdapterError::InvalidSource(format!(
                        "XDF footer references unknown stream {stream_id}"
                    ))
                })?;
                stream.footer_xml = Some(xml);
            }
            other => {
                return Err(AdapterError::InvalidSource(format!(
                    "unknown XDF chunk tag {other}"
                )))
            }
        }
    }
    if file_header_xml.is_empty() {
        return Err(AdapterError::InvalidSource(
            "XDF file carries no FileHeader chunk".to_owned(),
        ));
    }
    if streams.is_empty() {
        return Err(AdapterError::InvalidSource(
            "XDF file declares no streams".to_owned(),
        ));
    }
    Ok(ParsedXdf {
        file_header_xml,
        streams: streams.into_values().collect(),
        boundaries,
    })
}

fn decode_value(format: ChannelFormat, raw: &[u8]) -> f64 {
    match format {
        ChannelFormat::Int8 => f64::from(raw[0] as i8),
        ChannelFormat::Int16 => f64::from(i16::from_le_bytes([raw[0], raw[1]])),
        ChannelFormat::Int32 => f64::from(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
        ChannelFormat::Int64 => {
            let mut buffer = [0_u8; 8];
            buffer.copy_from_slice(raw);
            i64::from_le_bytes(buffer) as f64
        }
        ChannelFormat::Float32 => f64::from(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
        ChannelFormat::Double64 => {
            let mut buffer = [0_u8; 8];
            buffer.copy_from_slice(raw);
            f64::from_le_bytes(buffer)
        }
        ChannelFormat::StringValues => 0.0,
    }
}

/// Re-encode one stream's values in its OWN declared element type. Promoting
/// everything to f64 would silently widen an int64 stream past the precision
/// the file guaranteed.
fn encode_values(stream: &XdfStream) -> Vec<u8> {
    let samples = stream.values.first().map_or(0, Vec::len);
    let mut bytes = Vec::with_capacity(samples * stream.channel_count * stream.format.width());
    for channel in &stream.values {
        for value in channel {
            match stream.format {
                ChannelFormat::Int8 => bytes.push((*value as i8) as u8),
                ChannelFormat::Int16 => bytes.extend_from_slice(&(*value as i16).to_le_bytes()),
                ChannelFormat::Int32 => bytes.extend_from_slice(&(*value as i32).to_le_bytes()),
                ChannelFormat::Int64 => bytes.extend_from_slice(&(*value as i64).to_le_bytes()),
                ChannelFormat::Float32 => bytes.extend_from_slice(&(*value as f32).to_le_bytes()),
                ChannelFormat::Double64 => bytes.extend_from_slice(&value.to_le_bytes()),
                ChannelFormat::StringValues => {}
            }
        }
    }
    bytes
}

fn id<T>(seed: &blake3::Hash, domain: &[u8], index: u64) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(seed.as_bytes());
    hasher.update(domain);
    hasher.update(&index.to_le_bytes());
    let digest = hasher.finalize();
    let mut raw = [0_u8; 16];
    raw.copy_from_slice(&digest.as_bytes()[..16]);
    ObjectId::from_bytes(raw)
}

fn concept(value: &str) -> Result<ConceptId, AdapterError> {
    ConceptId::new(value).map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn source_key(namespace: &str, value: &str) -> Result<SourceKey, AdapterError> {
    SourceKey::new(namespace, value).map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn exact(source_path: String, target: String) -> MappingEntry {
    MappingEntry {
        source_path,
        target,
        disposition: MappingDisposition::Exact,
        reason: None,
    }
}

/// Convert an observed clock reading to an exact rational at microsecond
/// resolution. XDF measures these in seconds as f64; rounding to a fixed grid
/// keeps the recorded relation reproducible instead of platform-dependent.
fn microsecond_ticks(value: f64) -> Result<i64, AdapterError> {
    if !value.is_finite() {
        return Err(AdapterError::InvalidSource(
            "XDF timestamp is not finite".to_owned(),
        ));
    }
    let micros = (value * 1_000_000.0).round();
    if micros.abs() > 9.0e15 {
        return Err(AdapterError::InvalidSource(
            "XDF timestamp is out of range".to_owned(),
        ));
    }
    Ok(micros as i64)
}

fn seconds_rational(value: f64) -> Result<Rational, AdapterError> {
    if !value.is_finite() {
        return Err(AdapterError::InvalidSource(
            "XDF clock offset is not finite".to_owned(),
        ));
    }
    let micros = (value * 1_000_000.0).round();
    if micros.abs() > 9.0e15 {
        return Err(AdapterError::InvalidSource(
            "XDF clock offset is out of range".to_owned(),
        ));
    }
    Rational::new(micros as i128, 1_000_000)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

struct ParsedDataset {
    dataset: AbirDataset,
    payloads: Vec<PayloadObject>,
    mappings: Vec<MappingEntry>,
    stream_count: u64,
    offset_count: u64,
    boundary_count: u64,
}

impl XdfAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: AdapterProfile {
                id: ProfileId(PROFILE.to_owned()),
                standard: "XDF".to_owned(),
                edition: "1.0".to_owned(),
                media_types: vec!["application/x-xdf".to_owned()],
                status: ProfileStatus::Semantic,
                required_validator: "pyxdf".to_owned(),
                capabilities: BTreeSet::from([
                    AdapterCapability::Inspect,
                    AdapterCapability::Import,
                    AdapterCapability::PlanExport,
                    AdapterCapability::Export,
                    AdapterCapability::Validate,
                ]),
            },
            max_source_bytes,
        }
    }

    fn entry<'a>(&self, source: &'a ForeignObject) -> Result<&'a ForeignEntry, AdapterError> {
        if source.profile != self.profile.id {
            return Err(AdapterError::ProfileMismatch {
                expected: self.profile.id.clone(),
                actual: source.profile.clone(),
            });
        }
        if source.entries.len() != 1 {
            return Err(AdapterError::InvalidSource(
                "XDF semantic profile requires exactly one file".to_owned(),
            ));
        }
        let entry = &source.entries[0];
        if !valid_relative_path(&entry.path) {
            return Err(AdapterError::InvalidPath(entry.path.clone()));
        }
        if u64::try_from(entry.bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?
            > self.max_source_bytes
        {
            return Err(AdapterError::SourceTooLarge);
        }
        Ok(entry)
    }

    fn parse(
        &self,
        entry: &ForeignEntry,
        limits: ValidationLimits,
    ) -> Result<ParsedDataset, AdapterError> {
        let parsed = parse_xdf(&entry.bytes)?;
        let seed = blake3::hash(&entry.bytes);
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let host_clock_id = id::<ClockTag>(&seed, b"host-clock", 0);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();
        let mut stream_ids = Vec::new();
        let mut offset_total = 0_u64;

        // The recording host clock every stream clock is related TO. XDF's
        // whole synchronisation model is "each stream has its own clock and the
        // file records how it relates to the host", so the host clock must be a
        // real object rather than an implied one.
        draft.add_clock(Clock::new(
            host_clock_id,
            concept("xdf:clock/recording-host")?,
            None,
            Rational::new(0, 1).expect("zero is a rational"),
            Rational::new(1, 1).expect("unit rate is a rational"),
            Rational::new(0, 1).expect("zero is a rational"),
        ));

        for (index, stream) in parsed.streams.iter().enumerate() {
            let position = index as u64;
            let stream_id = id::<StreamTag>(&seed, b"stream", position);
            let atom_id = id::<AtomTag>(&seed, b"atom", position);
            let clock_id = id::<ClockTag>(&seed, b"stream-clock", position);
            let metadata_id = id::<AtomTag>(&seed, b"stream-metadata", position);

            let samples = if stream.format == ChannelFormat::StringValues {
                stream.strings.first().map_or(0, Vec::len)
            } else {
                stream.values.first().map_or(0, Vec::len)
            } as u64;
            if samples == 0 {
                return Err(AdapterError::InvalidSource(format!(
                    "XDF stream {} carries no samples",
                    stream.id
                )));
            }

            // Every stream gets its own clock: they are NOT synchronous.
            draft.add_clock(Clock::new(
                clock_id,
                concept("xdf:clock/stream-source")?,
                None,
                Rational::new(0, 1).expect("zero is a rational"),
                Rational::new(1, 1).expect("unit rate is a rational"),
                Rational::new(0, 1).expect("zero is a rational"),
            ));

            // The explicit per-sample timestamps, where the file carried them.
            let explicit: Vec<f64> = stream.timestamps.iter().flatten().copied().collect();
            let has_explicit = explicit.len() as u64 == samples;
            let rate = decimal_rational(&stream.nominal_srate)?;

            let (atom, timestamp_payload) = if rate.is_positive() && !has_explicit {
                // A declared rate with deduced timestamps is exactly a regular axis.
                let segment = semantic_abir::TimeSegment::new(
                    Rational::new(0, 1).expect("zero is a rational"),
                    rate,
                    samples,
                )
                .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
                (TimeAxis::Regular(segment), None)
            } else {
                // Irregular, or explicitly stamped: carry the timestamps
                // themselves rather than inventing a rate they do not have.
                if !has_explicit {
                    return Err(AdapterError::UnsupportedMeaning(format!(
                        "XDF stream {} has neither a positive nominal_srate nor a timestamp for every sample",
                        stream.id
                    )));
                }
                // ABIR carries explicit timestamps as exact integer ticks, so
                // the f64 seconds XDF records are pinned to a microsecond grid
                // rather than handed on as binary floating point.
                let mut bytes = Vec::with_capacity(explicit.len() * 8);
                for value in &explicit {
                    bytes.extend_from_slice(&microsecond_ticks(*value)?.to_le_bytes());
                }
                let content_id = abir_payload_id(ElementType::I64, &bytes);
                (
                    TimeAxis::Explicit {
                        timestamps: content_id,
                        count: samples,
                    },
                    Some((content_id, bytes)),
                )
            };

            let value_bytes = if stream.format == ChannelFormat::StringValues {
                serde_json::to_vec(&stream.strings)
                    .map_err(|error| AdapterError::InvalidSource(error.to_string()))?
            } else {
                encode_values(stream)
            };
            let content_id = abir_payload_id(stream.format.element(), &value_bytes);
            let descriptor = PayloadDescriptor::new(
                content_id,
                u64::try_from(value_bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?,
                stream.format.element(),
                ByteOrder::Little,
                vec![stream.channel_count as u64, samples],
                Layout::DenseRowMajor,
                Some(concept("abir:encoding/raw")?),
                None,
            );
            if stream.format == ChannelFormat::StringValues {
                // String values are not a numeric signal; carrying them as one
                // would assert a magnitude they do not have.
                draft.add_atom(Atom::BlobRef(BlobRef::new(
                    atom_id,
                    Presence::Present,
                    Some(PayloadDescriptor::new(
                        content_id,
                        value_bytes.len() as u64,
                        ElementType::Bytes,
                        ByteOrder::Little,
                        vec![value_bytes.len() as u64],
                        Layout::DenseRowMajor,
                        Some(concept("xdf:encoding/string-values-json")?),
                        Some("application/json".to_owned()),
                    )),
                    "application/json".to_owned(),
                    BlobIntegrity::new(concept("abir:integrity/blake3-256")?, content_id),
                )));
            } else {
                draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                    atom_id,
                    Presence::Present,
                    Some(descriptor),
                    atom,
                    None,
                )));
            }
            payloads.push(PayloadObject {
                content_id,
                bytes: value_bytes,
            });
            // An explicit time axis names a companion payload, and ABIR
            // requires that payload to belong to a real atom -- a dangling
            // reference is exactly the failure this prevents.
            let mut atom_ids = vec![atom_id];
            if let Some((stamp_id, stamp_bytes)) = timestamp_payload {
                let stamp_atom = id::<AtomTag>(&seed, b"timestamps", position);
                draft.add_atom(Atom::Tensor(semantic_abir::Tensor::new(
                    stamp_atom,
                    Presence::Present,
                    Some(PayloadDescriptor::new(
                        stamp_id,
                        stamp_bytes.len() as u64,
                        ElementType::I64,
                        ByteOrder::Little,
                        vec![samples],
                        Layout::DenseRowMajor,
                        Some(concept("abir:encoding/raw")?),
                        None,
                    )),
                    vec![semantic_abir::SemanticAxis::new(
                        concept("abir:axis/sample")?,
                        samples,
                    )],
                )));
                payloads.push(PayloadObject {
                    content_id: stamp_id,
                    bytes: stamp_bytes,
                });
                atom_ids.push(stamp_atom);
            }

            // The stream's own XML, byte-exact. Fields this adapter models are
            // ALSO promoted, but nothing is dropped by not being modelled.
            let mut metadata = stream.header_xml.clone().into_bytes();
            if let Some(footer) = &stream.footer_xml {
                metadata.extend_from_slice(b"\n");
                metadata.extend_from_slice(footer.as_bytes());
            }
            let metadata_content = abir_payload_id(ElementType::Bytes, &metadata);
            draft.add_atom(Atom::BlobRef(BlobRef::new(
                metadata_id,
                Presence::Present,
                Some(PayloadDescriptor::new(
                    metadata_content,
                    metadata.len() as u64,
                    ElementType::Bytes,
                    ByteOrder::Little,
                    vec![metadata.len() as u64],
                    Layout::DenseRowMajor,
                    Some(concept("xdf:encoding/stream-xml")?),
                    Some("application/xml".to_owned()),
                )),
                "application/xml".to_owned(),
                BlobIntegrity::new(concept("abir:integrity/blake3-256")?, metadata_content),
            )));
            payloads.push(PayloadObject {
                content_id: metadata_content,
                bytes: metadata,
            });

            let modality = if stream.kind.eq_ignore_ascii_case("EEG") {
                "abir:modality/eeg"
            } else {
                "abir:modality/unknown"
            };
            draft.add_stream(Stream::new(
                stream_id,
                recording_id,
                concept(modality)?,
                {
                    atom_ids.push(metadata_id);
                    atom_ids
                },
                Some(clock_id),
                None,
                None,
            ));
            stream_ids.push(stream_id);

            // The clock offsets: one relation, and the whole observed series
            // kept as its provenance so no measurement is summarised away.
            if !stream.offsets.is_empty() {
                let mut series = Vec::with_capacity(stream.offsets.len() * 16);
                let mut smallest = f64::INFINITY;
                let mut largest = f64::NEG_INFINITY;
                for (collection, offset) in &stream.offsets {
                    series.extend_from_slice(&collection.to_le_bytes());
                    series.extend_from_slice(&offset.to_le_bytes());
                    smallest = smallest.min(*offset);
                    largest = largest.max(*offset);
                }
                let series_id = abir_payload_id(ElementType::F64, &series);
                let (last_collection, last_offset) = *stream
                    .offsets
                    .last()
                    .expect("the series was checked non-empty");
                let spread = seconds_rational(largest - smallest)?;
                draft.add_clock_relation(ClockRelation::new(
                    id::<ClockRelationTag>(&seed, b"clock-relation", position),
                    clock_id,
                    host_clock_id,
                    seconds_rational(last_offset)?,
                    Rational::new(1, 1).expect("unit rate is a rational"),
                    spread,
                    concept("xdf:clock-method/measured-offset")?,
                    seconds_rational(last_collection)?,
                    None,
                    series_id,
                ));
                payloads.push(PayloadObject {
                    content_id: series_id,
                    bytes: series,
                });
                offset_total = offset_total
                    .checked_add(stream.offsets.len() as u64)
                    .ok_or(AdapterError::SourceTooLarge)?;
                mappings.push(exact(
                    format!("stream[{}].clock-offsets", stream.id),
                    format!("clock-relation:{clock_id}->{host_clock_id}"),
                ));
            }

            mappings.push(exact(
                format!("stream[{}].samples", stream.id),
                format!("atom:{atom_id}"),
            ));
            mappings.push(exact(
                format!("stream[{}].xml", stream.id),
                format!("atom:{metadata_id}"),
            ));
        }

        // Boundaries are real discontinuities in the recording, not framing.
        for index in 0..parsed.boundaries {
            let event_id = id::<EventTag>(&seed, b"boundary", index as u64);
            draft.add_event(Event::new(
                event_id,
                concept("xdf:event/boundary")?,
                host_clock_id,
                Rational::new(index as i128, 1).expect("index is a rational"),
                Rational::new(index as i128, 1).expect("index is a rational"),
                Rational::new(0, 1).expect("zero is a rational"),
            ));
        }
        if parsed.boundaries > 0 {
            mappings.push(exact(
                "boundary-chunks".to_owned(),
                "events:xdf:event/boundary".to_owned(),
            ));
        }

        let mut recording = Recording::new(recording_id, stream_ids);
        recording.add_source_key(source_key("xdf.file-header", &parsed.file_header_xml)?);
        recording.add_source_key(source_key(
            "xdf.stream-count",
            &parsed.streams.len().to_string(),
        )?);
        recording.add_source_key(source_key(
            "xdf.boundary-count",
            &parsed.boundaries.to_string(),
        )?);
        for stream in &parsed.streams {
            recording.add_source_key(source_key(
                &format!("xdf.stream.{}", stream.id),
                &format!(
                    "name={};type={};channels={};format={};srate={};labels={}",
                    stream.name,
                    stream.kind,
                    stream.channel_count,
                    stream.format.name(),
                    stream.nominal_srate,
                    stream.channel_labels.join("|"),
                ),
            )?);
        }
        draft.add_recording(recording);

        let semantic = draft
            .clone()
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        let interchange = interchange_content_id(&semantic)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let namespace = format!("adapter.{PROFILE}.binding.{interchange}");
        let source_content = payload_content_id(&entry.bytes);
        draft.add_source_capsule(SourceCapsule::new(
            source_key(&namespace, &entry.path)?,
            source_content,
            entry.media_type.as_deref(),
        ));
        let dataset = draft
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        mappings.push(exact(
            entry.path.clone(),
            format!("source-capsule:{source_content}"),
        ));
        payloads.push(PayloadObject {
            content_id: source_content,
            bytes: entry.bytes.clone(),
        });
        Ok(ParsedDataset {
            dataset,
            payloads,
            mappings,
            stream_count: parsed.streams.len() as u64,
            offset_count: offset_total,
            boundary_count: parsed.boundaries as u64,
        })
    }

    fn capsules<'a>(
        &self,
        dataset: &'a AbirDataset,
    ) -> Result<Vec<&'a semantic_abir::SourceCapsule>, AdapterError> {
        let namespace = binding_namespace(&self.profile.id, dataset)?;
        Ok(dataset
            .source_capsules()
            .iter()
            .filter(|capsule| capsule.source().namespace() == namespace)
            .collect())
    }
}

impl Adapter for XdfAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = parse_xdf(&entry.bytes)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: Vec::new(),
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("streams".to_owned(), parsed.streams.len() as u64),
                ("boundaries".to_owned(), parsed.boundaries as u64),
                (
                    "clock-offsets".to_owned(),
                    parsed
                        .streams
                        .iter()
                        .map(|stream| stream.offsets.len() as u64)
                        .sum(),
                ),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = self.parse(entry, limits)?;
        Ok(ImportOutcome {
            dataset: parsed.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ProjectedSemantic,
                entries: parsed.mappings,
                preserved_unknowns: 1,
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: parsed.payloads,
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let unsupported = capsules.len() != 1;
        let mappings = capsules
            .iter()
            .map(|capsule| {
                exact(
                    capsule.source().value().to_owned(),
                    capsule.source().value().to_owned(),
                )
            })
            .collect();
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings,
            requires_user_acceptance: false,
            unsupported,
            plan_id: String::new(),
        };
        plan.plan_id = plan_id(&plan);
        Ok(plan)
    }

    fn export(
        &self,
        dataset: &AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        let expected = self.plan_export(dataset)?;
        if expected != *plan || plan_id(plan) != plan.plan_id {
            return Err(AdapterError::ExportPlanMismatch);
        }
        if !plan.accepts_without_loss() {
            return Err(AdapterError::UnsupportedMeaning(
                "dataset lacks one exact XDF source capsule".to_owned(),
            ));
        }
        let capsule = self.capsules(dataset)?[0];
        let bytes = payloads.resolve(capsule.content_id())?;
        if payload_content_id(&bytes) != capsule.content_id() {
            return Err(AdapterError::MissingPayload(capsule.content_id()));
        }
        // Re-parse before handing the bytes back: matching the capsule
        // ContentId proves they are unchanged, not that they are still XDF.
        parse_xdf(&bytes)?;
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries: vec![ForeignEntry {
                    path: capsule.source().value().to_owned(),
                    media_type: capsule.media_type().map(str::to_owned),
                    bytes,
                }],
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: true,
                semantic_equivalence: true,
                output_content_ids: vec![capsule.content_id().to_string()],
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self.entry(source).and_then(|entry| {
            self.parse(entry, ValidationLimits::default())
                .map(|parsed| {
                    (
                        parsed.stream_count,
                        parsed.offset_count,
                        parsed.boundary_count,
                    )
                })
        });
        let diagnostics = match &result {
            Ok((streams, offsets, boundaries)) => vec![format!(
                "streams={streams} clock-offsets={offsets} boundaries={boundaries}"
            )],
            Err(error) => vec![error.to_string()],
        };
        ValidationArtifact {
            profile: self.profile.id.clone(),
            internal_valid: result.is_ok(),
            independent_validator: self.profile.required_validator.clone(),
            independent_valid: None,
            diagnostics,
        }
    }
}
