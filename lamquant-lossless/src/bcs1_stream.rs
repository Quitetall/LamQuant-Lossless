//! Compatibility facades over the unified container reader.
//!
//! New sources always route through [`ContainerReader`]. The legacy enum arms
//! remain only so callers that already constructed a frozen `LmlReader` or a
//! `Bcs1StreamReader` can hand ownership across without an API cliff.

use crate::container_reader::{ContainerFormat, ContainerReader};
use crate::error::{LmlError, LmlResult};
use crate::offset_table::OffsetTable;
use crate::stream::{ContainerHeader, LmlReader};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

pub struct Bcs1StreamReader<R: Read + Seek>(ContainerReader<R>);

impl Bcs1StreamReader<BufReader<File>> {
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::from_source(BufReader::new(file))
    }
}

impl<R: Read + Seek> Bcs1StreamReader<R> {
    pub fn from_source(source: R) -> LmlResult<Self> {
        let reader = ContainerReader::from_source(source)?;
        if reader.format() != ContainerFormat::Bcs1 {
            return Err(LmlError::InvalidHeader(
                "Bcs1StreamReader requires a BCS1 container".into(),
            ));
        }
        Ok(Self(reader))
    }

    pub fn header(&self) -> &ContainerHeader {
        self.0.header()
    }

    pub fn offset_table(&self) -> Option<&OffsetTable> {
        self.0.offset_table()
    }

    pub fn seek_to_window(&mut self, index: usize) -> LmlResult<()> {
        self.0.seek_to_window(index)
    }

    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        self.0.windows_for_range(start, end_exclusive)
    }

    pub fn rewind(&mut self) -> LmlResult<()> {
        self.0.rewind()
    }

    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        self.0.next_window()
    }
}

impl<R: Read + Seek> Iterator for Bcs1StreamReader<R> {
    type Item = LmlResult<Vec<Vec<i64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_window()
    }
}

/// Back-compatible facade. New `open`/`from_source` calls always select the
/// unified arm; the format-specific arms only adapt already-built readers.
pub enum AnyLmlReader<R: Read + Seek> {
    Legacy(LmlReader<R>),
    Bcs1(Bcs1StreamReader<R>),
    Unified(ContainerReader<R>),
}

impl AnyLmlReader<BufReader<File>> {
    pub fn open(path: &Path) -> LmlResult<Self> {
        Ok(Self::Unified(ContainerReader::open(path)?))
    }
}

impl<R: Read + Seek> AnyLmlReader<R> {
    pub fn from_source(source: R) -> LmlResult<Self> {
        Ok(Self::Unified(ContainerReader::from_source(source)?))
    }

    pub fn format(&self) -> ContainerFormat {
        match self {
            Self::Legacy(_) => ContainerFormat::LegacyLml1,
            Self::Bcs1(_) => ContainerFormat::Bcs1,
            Self::Unified(reader) => reader.format(),
        }
    }

    pub fn header(&self) -> &ContainerHeader {
        match self {
            Self::Legacy(reader) => reader.header(),
            Self::Bcs1(reader) => reader.header(),
            Self::Unified(reader) => reader.header(),
        }
    }

    pub fn offset_table(&self) -> Option<&OffsetTable> {
        match self {
            Self::Legacy(reader) => reader.offset_table(),
            Self::Bcs1(reader) => reader.offset_table(),
            Self::Unified(reader) => reader.offset_table(),
        }
    }

    pub fn seek_to_window(&mut self, index: usize) -> LmlResult<()> {
        match self {
            Self::Legacy(reader) => reader.seek_to_window(index),
            Self::Bcs1(reader) => reader.seek_to_window(index),
            Self::Unified(reader) => reader.seek_to_window(index),
        }
    }

    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        match self {
            Self::Legacy(reader) => reader.windows_for_range(start, end_exclusive),
            Self::Bcs1(reader) => reader.windows_for_range(start, end_exclusive),
            Self::Unified(reader) => reader.windows_for_range(start, end_exclusive),
        }
    }

    pub fn rewind(&mut self) -> LmlResult<()> {
        match self {
            Self::Legacy(reader) => reader.rewind(),
            Self::Bcs1(reader) => reader.rewind(),
            Self::Unified(reader) => reader.rewind(),
        }
    }

    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        match self {
            Self::Legacy(reader) => reader.next_window(),
            Self::Bcs1(reader) => reader.next_window(),
            Self::Unified(reader) => reader.next_window(),
        }
    }
}

impl<R: Read + Seek> Iterator for AnyLmlReader<R> {
    type Item = LmlResult<Vec<Vec<i64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_window()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container;
    use crate::lpc::LpcMode;

    fn signal() -> Vec<Vec<i64>> {
        vec![
            (0..5000).map(|value| value as i64 - 2500).collect(),
            (0..5000).map(|value| 1200 - value as i64).collect(),
        ]
    }

    #[test]
    fn bcs1_compatibility_reader_uses_unified_core() {
        let input = signal();
        let mut bytes = Vec::new();
        container::write_into(&mut bytes, &input, 250.0, 2500, 0, "{}", LpcMode::Fixed).unwrap();
        let mut reader = Bcs1StreamReader::from_source(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(reader.header().n_channels, input.len());
        assert!(reader.next_window().unwrap().is_ok());
    }

    #[test]
    fn dispatcher_reports_normalized_format() {
        let input = signal();
        let mut bytes = Vec::new();
        container::write_into(&mut bytes, &input, 250.0, 2500, 0, "{}", LpcMode::Fixed).unwrap();
        let reader = AnyLmlReader::from_source(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(reader.format(), ContainerFormat::Bcs1);
        assert!(matches!(reader, AnyLmlReader::Unified(_)));
    }

    #[test]
    fn bcs1_reader_rejects_legacy_source() {
        let legacy = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/legacy_payload_crc.lml"
        ));
        let result = Bcs1StreamReader::from_source(std::io::Cursor::new(legacy));
        assert!(matches!(result, Err(LmlError::InvalidHeader(_))));
    }
}
