//! BCS2: deterministic indexed serialization for immutable ABIR2 recordings.
//!
//! BCS2 is additive. It does not reinterpret or replace the frozen BCS1 wire.
//! The fixed header and directory make every graph family independently
//! addressable, while each family carries a per-ID record index for selective
//! reads without materializing the complete recording.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use crate::SignalSeries;

mod codec;
pub use codec::{decode_bcs2, encode_bcs2};

/// BCS2 file magic.
pub const BCS2_MAGIC: &[u8; 4] = b"BCS2";
/// Fixed BCS2 header size.
pub const BCS2_HEADER_LEN: usize = 64;
/// BCS2 major version.
pub const BCS2_VERSION_MAJOR: u8 = 2;
/// BCS2 minor version.
pub const BCS2_VERSION_MINOR: u8 = 0;

pub(crate) const BCS2_ENDIAN_LITTLE: u8 = 1;
pub(crate) const BCS2_FLAG_CRC32: u8 = 1;
pub(crate) const DIRECTORY_ENTRY_LEN: usize = 40;
pub(crate) const SECTION_HEADER_LEN: usize = 16;
pub(crate) const INDEX_ENTRY_LEN: usize = 24;
pub(crate) const NO_STRING_ID: u32 = u32::MAX;

/// Wire family identified from leading magic only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BiosignalWireVersion {
    /// Frozen BCS1 container.
    Bcs1,
    /// Indexed ABIR2 graph container.
    Bcs2,
}

impl BiosignalWireVersion {
    /// Detect BCS1 or BCS2 without interpreting any other format.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        match bytes.get(..4) {
            Some(magic) if magic == crate::BCS1_MAGIC => Some(Self::Bcs1),
            Some(magic) if magic == BCS2_MAGIC => Some(Self::Bcs2),
            _ => None,
        }
    }
}

/// Stable BCS2 section identifiers. Unknown numeric values remain skippable.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u16)]
pub enum SectionKind {
    Strings = 1,
    Identity = 2,
    Clocks = 3,
    SignalStreams = 4,
    SignalSeries = 5,
    Events = 6,
    Intervals = 7,
    Tables = 8,
    Tensors = 9,
    CoordinateFrames = 10,
    Coordinates = 11,
    ReferenceNodes = 12,
    ReferenceEdges = 13,
    Relationships = 14,
    Attachments = 15,
    Provenance = 16,
    LossReceipts = 17,
    Extensions = 18,
}

impl SectionKind {
    pub(crate) const fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Strings),
            2 => Some(Self::Identity),
            3 => Some(Self::Clocks),
            4 => Some(Self::SignalStreams),
            5 => Some(Self::SignalSeries),
            6 => Some(Self::Events),
            7 => Some(Self::Intervals),
            8 => Some(Self::Tables),
            9 => Some(Self::Tensors),
            10 => Some(Self::CoordinateFrames),
            11 => Some(Self::Coordinates),
            12 => Some(Self::ReferenceNodes),
            13 => Some(Self::ReferenceEdges),
            14 => Some(Self::Relationships),
            15 => Some(Self::Attachments),
            16 => Some(Self::Provenance),
            17 => Some(Self::LossReceipts),
            18 => Some(Self::Extensions),
            _ => None,
        }
    }
}

/// Parsed fixed BCS2 header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bcs2Header {
    section_count: u32,
    directory_offset: u64,
    directory_length: u64,
    file_length: u64,
    payload_crc32: u32,
}

impl Bcs2Header {
    /// Number of section directory entries.
    pub const fn section_count(&self) -> u32 {
        self.section_count
    }

    /// Declared complete file length.
    pub const fn file_length(&self) -> u64 {
        self.file_length
    }

    fn parse(bytes: &[u8]) -> Result<Self, Bcs2Error> {
        if bytes.len() < BCS2_HEADER_LEN {
            return Err(Bcs2Error::Truncated {
                context: "header",
                needed: BCS2_HEADER_LEN,
                available: bytes.len(),
            });
        }
        if &bytes[..4] != BCS2_MAGIC {
            return Err(Bcs2Error::InvalidMagic);
        }
        if bytes[4] != BCS2_VERSION_MAJOR || bytes[5] != BCS2_VERSION_MINOR {
            return Err(Bcs2Error::UnsupportedVersion {
                major: bytes[4],
                minor: bytes[5],
            });
        }
        if bytes[6] != BCS2_ENDIAN_LITTLE {
            return Err(Bcs2Error::UnsupportedEndianness(bytes[6]));
        }
        if bytes[7] != BCS2_FLAG_CRC32 {
            return Err(Bcs2Error::UnsupportedFlags(bytes[7]));
        }
        if read_u32(bytes, 8)? as usize != BCS2_HEADER_LEN {
            return Err(Bcs2Error::InvalidLayout("header length"));
        }
        if bytes[48..64].iter().any(|byte| *byte != 0) {
            return Err(Bcs2Error::InvalidLayout("non-zero reserved header bytes"));
        }

        let stored_header_crc = read_u32(bytes, 44)?;
        let mut header = [0_u8; BCS2_HEADER_LEN];
        header.copy_from_slice(&bytes[..BCS2_HEADER_LEN]);
        header[44..48].fill(0);
        if crc32(&header) != stored_header_crc {
            return Err(Bcs2Error::ChecksumMismatch("header"));
        }

        let parsed = Self {
            section_count: read_u32(bytes, 12)?,
            directory_offset: read_u64(bytes, 16)?,
            directory_length: read_u64(bytes, 24)?,
            file_length: read_u64(bytes, 32)?,
            payload_crc32: read_u32(bytes, 40)?,
        };
        if parsed.section_count > 1_024 {
            return Err(Bcs2Error::LimitExceeded("section count"));
        }
        let expected_directory_length = u64::from(parsed.section_count)
            .checked_mul(DIRECTORY_ENTRY_LEN as u64)
            .ok_or(Bcs2Error::IntegerOverflow("directory length"))?;
        if parsed.directory_offset != BCS2_HEADER_LEN as u64
            || parsed.directory_length != expected_directory_length
        {
            return Err(Bcs2Error::InvalidLayout("section directory"));
        }
        if parsed.file_length != bytes.len() as u64 {
            return Err(Bcs2Error::InvalidLayout("file length"));
        }
        if crc32(&bytes[BCS2_HEADER_LEN..]) != parsed.payload_crc32 {
            return Err(Bcs2Error::ChecksumMismatch("payload"));
        }
        Ok(parsed)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DirectoryEntry {
    pub kind_raw: u16,
    pub version: u16,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub item_count: u64,
    pub checksum: u32,
}

impl DirectoryEntry {
    fn parse(bytes: &[u8], offset: usize) -> Result<Self, Bcs2Error> {
        let slice = checked_slice(bytes, offset, DIRECTORY_ENTRY_LEN, "directory entry")?;
        let entry = Self {
            kind_raw: read_u16(slice, 0)?,
            version: read_u16(slice, 2)?,
            flags: read_u32(slice, 4)?,
            offset: read_u64(slice, 8)?,
            length: read_u64(slice, 16)?,
            item_count: read_u64(slice, 24)?,
            checksum: read_u32(slice, 32)?,
        };
        if read_u32(slice, 36)? != 0 {
            return Err(Bcs2Error::InvalidLayout("directory reserved bytes"));
        }
        if entry.flags != BCS2_FLAG_CRC32 as u32 {
            return Err(Bcs2Error::InvalidLayout("section flags"));
        }
        Ok(entry)
    }

    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.kind_raw.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.length.to_le_bytes());
        out.extend_from_slice(&self.item_count.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
    }
}

#[derive(Clone, Copy, Debug)]
struct StringTableView<'a> {
    bytes: &'a [u8],
    count: u32,
    data_offset: usize,
}

impl<'a> StringTableView<'a> {
    fn parse(bytes: &'a [u8], declared_count: u64) -> Result<Self, Bcs2Error> {
        if bytes.len() < SECTION_HEADER_LEN {
            return Err(Bcs2Error::Truncated {
                context: "string table",
                needed: SECTION_HEADER_LEN,
                available: bytes.len(),
            });
        }
        let count = read_u32(bytes, 0)?;
        if u64::from(count) != declared_count || read_u32(bytes, 4)? != 0 {
            return Err(Bcs2Error::InvalidLayout("string table count"));
        }
        let data_offset_u64 = read_u64(bytes, 8)?;
        let offset_count = u64::from(count)
            .checked_add(1)
            .ok_or(Bcs2Error::IntegerOverflow("string offset count"))?;
        let expected_data_offset = (SECTION_HEADER_LEN as u64)
            .checked_add(
                offset_count
                    .checked_mul(8)
                    .ok_or(Bcs2Error::IntegerOverflow("string offsets"))?,
            )
            .ok_or(Bcs2Error::IntegerOverflow("string data offset"))?;
        if data_offset_u64 != expected_data_offset {
            return Err(Bcs2Error::InvalidLayout("string data offset"));
        }
        let data_offset = usize::try_from(data_offset_u64)
            .map_err(|_| Bcs2Error::LimitExceeded("string data offset"))?;
        if data_offset > bytes.len() {
            return Err(Bcs2Error::Truncated {
                context: "string offsets",
                needed: data_offset,
                available: bytes.len(),
            });
        }
        let view = Self {
            bytes,
            count,
            data_offset,
        };
        if view.offset(0)? != 0 || view.offset(count)? != bytes.len() - data_offset {
            return Err(Bcs2Error::InvalidLayout("string terminal offsets"));
        }
        let mut previous: Option<&str> = None;
        for id in 0..count {
            let current = view.get(id)?;
            if previous.is_some_and(|value| value >= current) {
                return Err(Bcs2Error::InvalidLayout("string sort order"));
            }
            previous = Some(current);
        }
        Ok(view)
    }

    fn offset(&self, index: u32) -> Result<usize, Bcs2Error> {
        if index > self.count {
            return Err(Bcs2Error::InvalidStringId(index));
        }
        let index_bytes = (index as usize)
            .checked_mul(8)
            .ok_or(Bcs2Error::IntegerOverflow("string offset index"))?;
        let offset = SECTION_HEADER_LEN
            .checked_add(index_bytes)
            .ok_or(Bcs2Error::IntegerOverflow("string offset index"))?;
        usize::try_from(read_u64(self.bytes, offset)?)
            .map_err(|_| Bcs2Error::LimitExceeded("string offset"))
    }

    fn get(&self, id: u32) -> Result<&'a str, Bcs2Error> {
        if id >= self.count {
            return Err(Bcs2Error::InvalidStringId(id));
        }
        let start = self.offset(id)?;
        let end = self.offset(id + 1)?;
        if start > end {
            return Err(Bcs2Error::InvalidLayout("string offsets not monotonic"));
        }
        let absolute_start = self
            .data_offset
            .checked_add(start)
            .ok_or(Bcs2Error::IntegerOverflow("string start"))?;
        let bytes = checked_slice(self.bytes, absolute_start, end - start, "string bytes")?;
        core::str::from_utf8(bytes).map_err(|_| Bcs2Error::InvalidUtf8)
    }

    fn id(&self, target: &str) -> Result<Option<u32>, Bcs2Error> {
        let mut low = 0_u32;
        let mut high = self.count;
        while low < high {
            let mid = low + (high - low) / 2;
            match self.get(mid)?.cmp(target) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }
}

/// Borrowed, fully validated BCS2 index.
#[derive(Debug)]
pub struct Bcs2View<'a> {
    bytes: &'a [u8],
    header: Bcs2Header,
    directory: Vec<DirectoryEntry>,
    strings: StringTableView<'a>,
}

impl<'a> Bcs2View<'a> {
    /// Validate the complete envelope, directory, section checksums, string
    /// table, and every per-section item index.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Bcs2Error> {
        let header = Bcs2Header::parse(bytes)?;
        let directory_start = usize::try_from(header.directory_offset)
            .map_err(|_| Bcs2Error::LimitExceeded("directory offset"))?;
        let directory_end = directory_start
            .checked_add(
                usize::try_from(header.directory_length)
                    .map_err(|_| Bcs2Error::LimitExceeded("directory length"))?,
            )
            .ok_or(Bcs2Error::IntegerOverflow("directory end"))?;
        if directory_end > bytes.len() {
            return Err(Bcs2Error::Truncated {
                context: "directory",
                needed: directory_end,
                available: bytes.len(),
            });
        }

        let mut directory = Vec::with_capacity(header.section_count as usize);
        let mut previous_kind = 0_u16;
        let mut previous_end = directory_end as u64;
        for index in 0..header.section_count as usize {
            let entry =
                DirectoryEntry::parse(bytes, directory_start + index * DIRECTORY_ENTRY_LEN)?;
            if SectionKind::from_u16(entry.kind_raw).is_some() && entry.version != 1 {
                return Err(Bcs2Error::InvalidLayout("known section version"));
            }
            if index > 0 && entry.kind_raw <= previous_kind {
                return Err(Bcs2Error::InvalidLayout("directory sort order"));
            }
            previous_kind = entry.kind_raw;
            let end = entry
                .offset
                .checked_add(entry.length)
                .ok_or(Bcs2Error::IntegerOverflow("section end"))?;
            if entry.offset != previous_end || end > header.file_length {
                return Err(Bcs2Error::InvalidLayout(
                    "gapped, overlapping, or out-of-range section",
                ));
            }
            let section = checked_slice(
                bytes,
                usize::try_from(entry.offset)
                    .map_err(|_| Bcs2Error::LimitExceeded("section offset"))?,
                usize::try_from(entry.length)
                    .map_err(|_| Bcs2Error::LimitExceeded("section length"))?,
                "section",
            )?;
            if crc32(section) != entry.checksum {
                return Err(Bcs2Error::ChecksumMismatch("section"));
            }
            previous_end = end;
            directory.push(entry);
        }
        if previous_end != header.file_length {
            return Err(Bcs2Error::InvalidLayout("unindexed trailing bytes"));
        }

        let string_entry = directory
            .iter()
            .find(|entry| entry.kind_raw == SectionKind::Strings as u16)
            .ok_or(Bcs2Error::MissingSection(SectionKind::Strings))?;
        let string_bytes = section_slice(bytes, string_entry)?;
        let strings = StringTableView::parse(string_bytes, string_entry.item_count)?;
        let view = Self {
            bytes,
            header,
            directory,
            strings,
        };
        for entry in view.directory.iter() {
            if SectionKind::from_u16(entry.kind_raw).is_some()
                && entry.kind_raw != SectionKind::Strings as u16
            {
                view.validate_indexed_section(entry)?;
            }
        }
        Ok(view)
    }

    /// Parsed fixed header.
    pub const fn header(&self) -> &Bcs2Header {
        &self.header
    }

    /// Whether a known section is present.
    pub fn has_section(&self, kind: SectionKind) -> bool {
        self.entry(kind).is_some()
    }

    /// Borrow one indexed record by its canonical ID.
    pub fn record_bytes(&self, kind: SectionKind, id: &str) -> Result<Option<&'a [u8]>, Bcs2Error> {
        let Some(entry) = self.entry(kind) else {
            return Ok(None);
        };
        if kind == SectionKind::Strings {
            return Err(Bcs2Error::InvalidLayout("string table has no record index"));
        }
        let Some(id_number) = self.strings.id(id)? else {
            return Ok(None);
        };
        let section = section_slice(self.bytes, entry)?;
        let count = read_u32(section, 0)?;
        let mut low = 0_u32;
        let mut high = count;
        while low < high {
            let mid = low + (high - low) / 2;
            let index_offset = SECTION_HEADER_LEN + mid as usize * INDEX_ENTRY_LEN;
            let candidate = read_u32(section, index_offset)?;
            match candidate.cmp(&id_number) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => {
                    let offset = usize::try_from(read_u64(section, index_offset + 8)?)
                        .map_err(|_| Bcs2Error::LimitExceeded("record offset"))?;
                    let length = usize::try_from(read_u64(section, index_offset + 16)?)
                        .map_err(|_| Bcs2Error::LimitExceeded("record length"))?;
                    return Ok(Some(checked_slice(section, offset, length, "record")?));
                }
            }
        }
        Ok(None)
    }

    /// Decode one channel series without decoding the recording graph or any
    /// unrelated attachment/tensor payload.
    pub fn decode_signal_series(
        &self,
        channel_id: &str,
    ) -> Result<Option<SignalSeries>, Bcs2Error> {
        let Some(bytes) = self.record_bytes(SectionKind::SignalSeries, channel_id)? else {
            return Ok(None);
        };
        let (_, series) = codec::decode_signal_series_record(self, channel_id, bytes)?;
        Ok(Some(series))
    }

    pub(crate) fn string(&self, id: u32) -> Result<&'a str, Bcs2Error> {
        self.strings.get(id)
    }

    pub(crate) fn records(&self, kind: SectionKind) -> Result<Vec<(&'a str, &'a [u8])>, Bcs2Error> {
        let Some(entry) = self.entry(kind) else {
            return Ok(Vec::new());
        };
        let section = section_slice(self.bytes, entry)?;
        let count = read_u32(section, 0)?;
        let mut records = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let index_offset = SECTION_HEADER_LEN + index * INDEX_ENTRY_LEN;
            let id = self.string(read_u32(section, index_offset)?)?;
            let offset = usize::try_from(read_u64(section, index_offset + 8)?)
                .map_err(|_| Bcs2Error::LimitExceeded("record offset"))?;
            let length = usize::try_from(read_u64(section, index_offset + 16)?)
                .map_err(|_| Bcs2Error::LimitExceeded("record length"))?;
            records.push((id, checked_slice(section, offset, length, "record")?));
        }
        Ok(records)
    }

    fn entry(&self, kind: SectionKind) -> Option<&DirectoryEntry> {
        self.directory
            .iter()
            .find(|entry| entry.kind_raw == kind as u16)
    }

    fn validate_indexed_section(&self, entry: &DirectoryEntry) -> Result<(), Bcs2Error> {
        let section = section_slice(self.bytes, entry)?;
        if section.len() < SECTION_HEADER_LEN {
            return Err(Bcs2Error::Truncated {
                context: "section header",
                needed: SECTION_HEADER_LEN,
                available: section.len(),
            });
        }
        let count = read_u32(section, 0)?;
        if u64::from(count) != entry.item_count || read_u32(section, 4)? as usize != INDEX_ENTRY_LEN
        {
            return Err(Bcs2Error::InvalidLayout(
                "section item count or index width",
            ));
        }
        let index_bytes = (count as usize)
            .checked_mul(INDEX_ENTRY_LEN)
            .ok_or(Bcs2Error::IntegerOverflow("section index"))?;
        let expected_payload_offset = SECTION_HEADER_LEN
            .checked_add(index_bytes)
            .ok_or(Bcs2Error::IntegerOverflow("section index"))?;
        if usize::try_from(read_u64(section, 8)?)
            .map_err(|_| Bcs2Error::LimitExceeded("section payload offset"))?
            != expected_payload_offset
        {
            return Err(Bcs2Error::InvalidLayout("section payload offset"));
        }
        if expected_payload_offset > section.len() {
            return Err(Bcs2Error::Truncated {
                context: "section index",
                needed: expected_payload_offset,
                available: section.len(),
            });
        }
        let mut previous_id: Option<u32> = None;
        let mut expected_record_offset = expected_payload_offset;
        for index in 0..count as usize {
            let offset = SECTION_HEADER_LEN + index * INDEX_ENTRY_LEN;
            let id = read_u32(section, offset)?;
            self.string(id)?;
            if previous_id.is_some_and(|previous| previous >= id) {
                return Err(Bcs2Error::InvalidLayout("record index sort order"));
            }
            previous_id = Some(id);
            if read_u32(section, offset + 4)? != 0 {
                return Err(Bcs2Error::InvalidLayout("record flags"));
            }
            let record_offset = usize::try_from(read_u64(section, offset + 8)?)
                .map_err(|_| Bcs2Error::LimitExceeded("record offset"))?;
            let record_length = usize::try_from(read_u64(section, offset + 16)?)
                .map_err(|_| Bcs2Error::LimitExceeded("record length"))?;
            if record_offset != expected_record_offset {
                return Err(Bcs2Error::InvalidLayout("non-canonical record offsets"));
            }
            expected_record_offset = record_offset
                .checked_add(record_length)
                .ok_or(Bcs2Error::IntegerOverflow("record end"))?;
            if expected_record_offset > section.len() {
                return Err(Bcs2Error::Truncated {
                    context: "record payload",
                    needed: expected_record_offset,
                    available: section.len(),
                });
            }
        }
        if expected_record_offset != section.len() {
            return Err(Bcs2Error::InvalidLayout("trailing section bytes"));
        }
        Ok(())
    }
}

/// Fail-closed BCS2 parse/encode/decode error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Bcs2Error {
    Truncated {
        context: &'static str,
        needed: usize,
        available: usize,
    },
    InvalidMagic,
    UnsupportedVersion {
        major: u8,
        minor: u8,
    },
    UnsupportedEndianness(u8),
    UnsupportedFlags(u8),
    InvalidLayout(&'static str),
    ChecksumMismatch(&'static str),
    MissingSection(SectionKind),
    InvalidStringId(u32),
    InvalidUtf8,
    InvalidTag {
        context: &'static str,
        tag: u8,
    },
    DuplicateId(String),
    LimitExceeded(&'static str),
    IntegerOverflow(&'static str),
    Graph(String),
}

impl fmt::Display for Bcs2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated {
                context,
                needed,
                available,
            } => write!(
                f,
                "BCS2 {context} truncated: need {needed}, have {available}"
            ),
            Self::InvalidMagic => f.write_str("BCS2 invalid magic"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "BCS2 unsupported version {major}.{minor}")
            }
            Self::UnsupportedEndianness(value) => {
                write!(f, "BCS2 unsupported endianness {value}")
            }
            Self::UnsupportedFlags(value) => write!(f, "BCS2 unsupported flags {value:#x}"),
            Self::InvalidLayout(context) => write!(f, "BCS2 invalid {context}"),
            Self::ChecksumMismatch(context) => write!(f, "BCS2 {context} checksum mismatch"),
            Self::MissingSection(kind) => write!(f, "BCS2 missing section {kind:?}"),
            Self::InvalidStringId(id) => write!(f, "BCS2 invalid string id {id}"),
            Self::InvalidUtf8 => f.write_str("BCS2 invalid UTF-8"),
            Self::InvalidTag { context, tag } => {
                write!(f, "BCS2 invalid {context} tag {tag}")
            }
            Self::DuplicateId(id) => write!(f, "BCS2 duplicate record id '{id}'"),
            Self::LimitExceeded(context) => {
                write!(f, "BCS2 {context} exceeds implementation limit")
            }
            Self::IntegerOverflow(context) => write!(f, "BCS2 integer overflow in {context}"),
            Self::Graph(message) => write!(f, "BCS2 graph validation failed: {message}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Bcs2Error {}

pub(crate) fn encode_header(
    section_count: u32,
    file_length: u64,
    payload: &[u8],
) -> [u8; BCS2_HEADER_LEN] {
    let mut header = [0_u8; BCS2_HEADER_LEN];
    header[..4].copy_from_slice(BCS2_MAGIC);
    header[4] = BCS2_VERSION_MAJOR;
    header[5] = BCS2_VERSION_MINOR;
    header[6] = BCS2_ENDIAN_LITTLE;
    header[7] = BCS2_FLAG_CRC32;
    header[8..12].copy_from_slice(&(BCS2_HEADER_LEN as u32).to_le_bytes());
    header[12..16].copy_from_slice(&section_count.to_le_bytes());
    header[16..24].copy_from_slice(&(BCS2_HEADER_LEN as u64).to_le_bytes());
    header[24..32]
        .copy_from_slice(&(u64::from(section_count) * DIRECTORY_ENTRY_LEN as u64).to_le_bytes());
    header[32..40].copy_from_slice(&file_length.to_le_bytes());
    header[40..44].copy_from_slice(&crc32(payload).to_le_bytes());
    let checksum = crc32(&header);
    header[44..48].copy_from_slice(&checksum.to_le_bytes());
    header
}

pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn section_slice<'a>(bytes: &'a [u8], entry: &DirectoryEntry) -> Result<&'a [u8], Bcs2Error> {
    checked_slice(
        bytes,
        usize::try_from(entry.offset).map_err(|_| Bcs2Error::LimitExceeded("section offset"))?,
        usize::try_from(entry.length).map_err(|_| Bcs2Error::LimitExceeded("section length"))?,
        "section",
    )
}

pub(crate) fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    context: &'static str,
) -> Result<&'a [u8], Bcs2Error> {
    let end = offset
        .checked_add(length)
        .ok_or(Bcs2Error::IntegerOverflow(context))?;
    bytes.get(offset..end).ok_or(Bcs2Error::Truncated {
        context,
        needed: end,
        available: bytes.len(),
    })
}

pub(crate) fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, Bcs2Error> {
    let data = checked_slice(bytes, offset, 2, "u16")?;
    Ok(u16::from_le_bytes([data[0], data[1]]))
}

pub(crate) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Bcs2Error> {
    let data = checked_slice(bytes, offset, 4, "u32")?;
    Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

pub(crate) fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Bcs2Error> {
    let data = checked_slice(bytes, offset, 8, "u64")?;
    Ok(u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]))
}
