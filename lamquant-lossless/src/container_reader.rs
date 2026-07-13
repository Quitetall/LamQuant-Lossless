//! Unified LML1/BCS1 container reader.
//!
//! Wire-format adapters normalize version-specific headers into one plan.
//! Metadata, index/footer handling, payload decoding, seeking, and iteration
//! are owned here once for memory, file, and other `Read + Seek` sources.

use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::offset_table::{OffsetTable, ENTRY_SIZE, FOOTER_MAGIC, FOOTER_SIZE};
use crate::stream::ContainerHeader;
use abir::{Bcs1Header, CodecDescriptor, BCS1_HEADER_LEN, BCS1_MAGIC, BCS1_VERSION_MAJOR};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const LEGACY_MAX_HEADER_LEN: usize = 32;
const LEGACY_MIN_HEADER_LEN: usize = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    LegacyLml1,
    Bcs1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormatDetails {
    Legacy,
    Bcs1 {
        modality_tag: u8,
        modality_source: u8,
        codec_descriptor: u8,
        mode: u8,
        tier: u8,
        decode_capability: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FooterInspection {
    NotDeclared,
    MissingMagic,
    Valid {
        entries: usize,
        exceeds_header_windows: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerInspection {
    pub format: ContainerFormat,
    pub details: ContainerFormatDetails,
    pub version_major: u8,
    pub version_minor: u8,
    pub n_channels: usize,
    pub n_windows: usize,
    pub total_samples: usize,
    pub window_size: usize,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub flags: u8,
    pub metadata: String,
    pub source_len: u64,
    pub footer: FooterInspection,
}

#[derive(Debug, Clone, Copy)]
struct ContainerPlan {
    format: ContainerFormat,
    details: ContainerFormatDetails,
    version_major: u8,
    version_minor: u8,
    header_len: usize,
    metadata_len: usize,
    n_channels: usize,
    n_windows: usize,
    total_samples: usize,
    window_size: usize,
    sample_rate_mhz: u32,
    bit_depth: u8,
    flags: u8,
}

impl ContainerPlan {
    fn validate(self) -> LmlResult<Self> {
        if self.n_channels == 0 || self.n_channels > 1024 {
            return Err(LmlError::InvalidHeader(format!(
                "channel count: {}",
                self.n_channels
            )));
        }
        if self.n_windows == 0 {
            return Err(LmlError::InvalidHeader("zero windows".into()));
        }
        if self.total_samples == 0 {
            return Err(LmlError::InvalidHeader("zero samples".into()));
        }
        let max_samples = (self.n_windows as u64)
            .checked_mul(self.window_size as u64)
            .ok_or_else(|| {
                LmlError::InvalidHeader("n_windows * window_size overflows u64".into())
            })?;
        if self.total_samples as u64 > max_samples {
            return Err(LmlError::InvalidHeader(format!(
                "total_samples {} exceeds n_windows*window_size {max_samples}",
                self.total_samples
            )));
        }
        Ok(self)
    }

    fn payload_start(self) -> LmlResult<u64> {
        let index_len = (self.n_windows as u64)
            .checked_mul(4)
            .ok_or_else(|| LmlError::InvalidHeader("window index length overflows u64".into()))?;
        (self.header_len as u64)
            .checked_add(self.metadata_len as u64)
            .and_then(|value| value.checked_add(index_len))
            .ok_or_else(|| LmlError::InvalidHeader("payload offset overflows u64".into()))
    }
}

fn validate_bcs1(header: &Bcs1Header) -> LmlResult<()> {
    if header.version_major > BCS1_VERSION_MAJOR {
        return Err(LmlError::UnsupportedVersion(header.version_major));
    }
    if header.decode_capability > 0 {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 decode_capability {} exceeds this reader's max (0 = integer floor)",
            header.decode_capability
        )));
    }
    if header.codec_descriptor != CodecDescriptor::Lml53.to_u8() {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 codec_descriptor {} is not installed in this reader",
            header.codec_descriptor
        )));
    }
    Ok(())
}

fn bcs1_plan(header: &[u8]) -> LmlResult<ContainerPlan> {
    let parsed = Bcs1Header::parse(header)
        .map_err(|error| LmlError::InvalidHeader(format!("BCS1 header: {error}")))?;
    validate_bcs1(&parsed)?;
    ContainerPlan {
        format: ContainerFormat::Bcs1,
        details: ContainerFormatDetails::Bcs1 {
            modality_tag: parsed.modality_tag,
            modality_source: parsed.modality_source,
            codec_descriptor: parsed.codec_descriptor,
            mode: parsed.mode,
            tier: parsed.tier,
            decode_capability: parsed.decode_capability,
        },
        version_major: parsed.version_major,
        version_minor: parsed.version_minor,
        header_len: BCS1_HEADER_LEN,
        metadata_len: parsed.metadata_length as usize,
        n_channels: parsed.n_channels as usize,
        n_windows: parsed.n_windows as usize,
        total_samples: parsed.total_samples as usize,
        window_size: parsed.window_size as usize,
        sample_rate_mhz: parsed.sample_rate_mhz,
        bit_depth: parsed.bit_depth,
        flags: parsed.flags,
    }
    .validate()
}

fn legacy_plan(header: &[u8]) -> LmlResult<ContainerPlan> {
    if header.len() < LEGACY_MIN_HEADER_LEN {
        return Err(LmlError::Truncated {
            expected: LEGACY_MIN_HEADER_LEN,
            actual: header.len(),
            context: "container header",
        });
    }
    if &header[..3] != b"LML" {
        return Err(LmlError::InvalidMagic([
            header[0], header[1], header[2], header[3],
        ]));
    }
    if header[3] != b'1' {
        return if header[3].is_ascii_digit() {
            Err(LmlError::UnsupportedVersion(header[3]))
        } else {
            Err(LmlError::InvalidMagic([
                header[0], header[1], header[2], header[3],
            ]))
        };
    }

    let probe = u16::from_le_bytes([header[4], header[5]]);
    let (header_len, n_channels, n_windows, total_samples, window_size, metadata_len) =
        if probe == 1 && header.len() >= 32 && matches!(header[20], 16 | 24 | 32) {
            (
                32,
                u16::from_le_bytes([header[6], header[7]]) as usize,
                u16::from_le_bytes([header[8], header[9]]) as usize,
                u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize,
                u16::from_le_bytes([header[14], header[15]]) as usize,
                u32::from_le_bytes([header[22], header[23], header[24], header[25]]) as usize,
            )
        } else if probe == 1 {
            if header.len() < 20 {
                return Err(LmlError::Truncated {
                    expected: 20,
                    actual: header.len(),
                    context: "container header (20-byte)",
                });
            }
            (
                20,
                u16::from_le_bytes([header[6], header[7]]) as usize,
                u16::from_le_bytes([header[8], header[9]]) as usize,
                u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize,
                u16::from_le_bytes([header[14], header[15]]) as usize,
                u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize,
            )
        } else {
            (
                18,
                probe as usize,
                u16::from_le_bytes([header[6], header[7]]) as usize,
                u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize,
                u16::from_le_bytes([header[12], header[13]]) as usize,
                u32::from_le_bytes([header[14], header[15], header[16], header[17]]) as usize,
            )
        };
    let (version_major, version_minor, sample_rate_mhz, bit_depth, flags) = if header_len == 32 {
        (
            header[4],
            header[5],
            u32::from_le_bytes([header[16], header[17], header[18], header[19]]),
            header[20],
            header[21],
        )
    } else {
        (1, 0, 0, 0, 0)
    };

    ContainerPlan {
        format: ContainerFormat::LegacyLml1,
        details: ContainerFormatDetails::Legacy,
        version_major,
        version_minor,
        header_len,
        metadata_len,
        n_channels,
        n_windows,
        total_samples,
        window_size,
        sample_rate_mhz,
        bit_depth,
        flags,
    }
    .validate()
}

fn read_plan<R: Read + Seek>(source: &mut R) -> LmlResult<(ContainerPlan, u64)> {
    source.seek(SeekFrom::Start(0)).map_err(LmlError::Io)?;
    let end = source.seek(SeekFrom::End(0)).map_err(LmlError::Io)?;
    source.seek(SeekFrom::Start(0)).map_err(LmlError::Io)?;
    let prefix_len = usize::try_from(end.min(BCS1_HEADER_LEN as u64))
        .map_err(|_| LmlError::InvalidHeader("container length does not fit usize".into()))?;
    let mut prefix = vec![0u8; prefix_len];
    source.read_exact(&mut prefix).map_err(LmlError::Io)?;
    let plan = if prefix.starts_with(BCS1_MAGIC) {
        bcs1_plan(&prefix)?
    } else {
        legacy_plan(&prefix[..prefix.len().min(LEGACY_MAX_HEADER_LEN)])?
    };
    Ok((plan, end))
}

fn read_footer<R: Read + Seek>(source: &mut R) -> LmlResult<Option<OffsetTable>> {
    let end = source.seek(SeekFrom::End(0)).map_err(LmlError::Io)?;
    if end < FOOTER_SIZE as u64 {
        return Ok(None);
    }
    source
        .seek(SeekFrom::End(-(FOOTER_SIZE as i64)))
        .map_err(LmlError::Io)?;
    let mut footer = [0u8; FOOTER_SIZE];
    source.read_exact(&mut footer).map_err(LmlError::Io)?;
    if &footer[..8] != FOOTER_MAGIC {
        return Ok(None);
    }
    let n_windows = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]) as u64;
    let table_bytes = n_windows
        .checked_mul(ENTRY_SIZE as u64)
        .ok_or_else(|| LmlError::InvalidHeader("footer table length overflows u64".into()))?;
    let combined_len = table_bytes
        .checked_add(FOOTER_SIZE as u64)
        .ok_or_else(|| LmlError::InvalidHeader("footer length overflows u64".into()))?;
    if combined_len > end {
        return Err(LmlError::InvalidHeader(format!(
            "LMLFOOT1 requires {combined_len} bytes but source is {end} bytes"
        )));
    }
    let mut combined = vec![0u8; combined_len as usize];
    source
        .seek(SeekFrom::Start(end - combined_len))
        .map_err(LmlError::Io)?;
    source.read_exact(&mut combined).map_err(LmlError::Io)?;
    OffsetTable::read_from_buffer(&combined)
}

pub struct ContainerReader<R: Read + Seek> {
    source: R,
    format: ContainerFormat,
    header: ContainerHeader,
    windows_read: usize,
    window_offsets: Vec<u32>,
    offset_table: Option<OffsetTable>,
    first_payload_pos: u64,
    source_len: u64,
    details: ContainerFormatDetails,
    version_major: u8,
    version_minor: u8,
    sample_rate_mhz: u32,
    bit_depth: u8,
    flags: u8,
    footer: FooterInspection,
}

impl ContainerReader<BufReader<File>> {
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::from_source(BufReader::new(file))
    }
}

impl<R: Read + Seek> ContainerReader<R> {
    pub fn from_source(mut source: R) -> LmlResult<Self> {
        let (plan, source_len) = read_plan(&mut source)?;
        source
            .seek(SeekFrom::Start(plan.header_len as u64))
            .map_err(LmlError::Io)?;
        let mut metadata = vec![0u8; plan.metadata_len];
        source.read_exact(&mut metadata).map_err(LmlError::Io)?;
        let metadata = String::from_utf8(metadata).map_err(|error| {
            LmlError::InvalidHeader(format!("metadata is not valid UTF-8: {error}"))
        })?;
        let mut window_offsets = Vec::with_capacity(plan.n_windows);
        for _ in 0..plan.n_windows {
            let mut offset = [0u8; 4];
            source.read_exact(&mut offset).map_err(LmlError::Io)?;
            window_offsets.push(u32::from_le_bytes(offset));
        }
        let first_payload_pos = plan.payload_start()?;
        source
            .seek(SeekFrom::Start(first_payload_pos))
            .map_err(LmlError::Io)?;
        let offset_table = read_footer(&mut source)?;
        let footer = if let Some(table) = offset_table.as_ref() {
            FooterInspection::Valid {
                entries: table.len(),
                exceeds_header_windows: table.len() > plan.n_windows,
            }
        } else if plan.flags & 0b0000_0001 != 0 {
            FooterInspection::MissingMagic
        } else {
            FooterInspection::NotDeclared
        };
        source
            .seek(SeekFrom::Start(first_payload_pos))
            .map_err(LmlError::Io)?;
        Ok(Self {
            source,
            format: plan.format,
            header: ContainerHeader {
                n_channels: plan.n_channels,
                n_windows: plan.n_windows,
                total_samples: plan.total_samples,
                window_size: plan.window_size,
                metadata,
            },
            windows_read: 0,
            window_offsets,
            offset_table,
            first_payload_pos,
            source_len,
            details: plan.details,
            version_major: plan.version_major,
            version_minor: plan.version_minor,
            sample_rate_mhz: plan.sample_rate_mhz,
            bit_depth: plan.bit_depth,
            flags: plan.flags,
            footer,
        })
    }

    pub fn format(&self) -> ContainerFormat {
        self.format
    }

    pub fn header(&self) -> &ContainerHeader {
        &self.header
    }

    pub fn inspection(&self) -> ContainerInspection {
        ContainerInspection {
            format: self.format,
            details: self.details,
            version_major: self.version_major,
            version_minor: self.version_minor,
            n_channels: self.header.n_channels,
            n_windows: self.header.n_windows,
            total_samples: self.header.total_samples,
            window_size: self.header.window_size,
            sample_rate_mhz: self.sample_rate_mhz,
            bit_depth: self.bit_depth,
            flags: self.flags,
            metadata: self.header.metadata.clone(),
            source_len: self.source_len,
            footer: self.footer,
        }
    }

    pub fn offset_table(&self) -> Option<&OffsetTable> {
        self.offset_table.as_ref()
    }

    pub fn container_header(&self) -> LmlResult<lamquant_lml_legacy::container::ContainerHeader> {
        Ok(lamquant_lml_legacy::container::ContainerHeader {
            n_ch: self.header.n_channels,
            n_windows: self.header.n_windows,
            total_samples: self.header.total_samples,
            window_size: self.header.window_size,
            metadata: self.header.metadata.clone(),
            payload_start: usize::try_from(self.first_payload_pos)
                .map_err(|_| LmlError::InvalidHeader("payload offset does not fit usize".into()))?,
        })
    }

    pub fn seek_to_window(&mut self, index: usize) -> LmlResult<()> {
        let table = self.offset_table.as_ref().ok_or_else(|| {
            LmlError::InvalidHeader("container has no LMLFOOT1 seek table".into())
        })?;
        let entry = table.entries().get(index).ok_or_else(|| {
            LmlError::InvalidHeader(format!("window {index} out of range (len {})", table.len()))
        })?;
        self.source
            .seek(SeekFrom::Start(entry.abs_offset))
            .map_err(LmlError::Io)?;
        self.windows_read = index;
        Ok(())
    }

    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        let range = self
            .offset_table
            .as_ref()
            .ok_or_else(|| LmlError::InvalidHeader("container has no LMLFOOT1 seek table".into()))?
            .windows_for_range(start, end_exclusive)
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "empty or past-EOF range [{start}, {end_exclusive})"
                ))
            })?;
        let mut windows = Vec::with_capacity(range.end() - range.start() + 1);
        for index in range {
            self.seek_to_window(index)?;
            windows.push(self.next_window().ok_or_else(|| {
                LmlError::InvalidHeader(format!("window {index} missing after successful seek"))
            })??);
        }
        Ok(windows)
    }

    pub fn rewind(&mut self) -> LmlResult<()> {
        self.source
            .seek(SeekFrom::Start(self.first_payload_pos))
            .map_err(LmlError::Io)?;
        self.windows_read = 0;
        Ok(())
    }

    pub fn read_window(&mut self, index: usize) -> LmlResult<Vec<Vec<i64>>> {
        let relative = *self.window_offsets.get(index).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "window {index} out of range (len {})",
                self.window_offsets.len()
            ))
        })? as u64;
        let position = self
            .first_payload_pos
            .checked_add(relative)
            .ok_or_else(|| LmlError::InvalidHeader("window offset overflows u64".into()))?;
        let length_end = position
            .checked_add(4)
            .ok_or_else(|| LmlError::InvalidHeader("window length offset overflows u64".into()))?;
        if length_end > self.source_len {
            return Err(LmlError::Truncated {
                expected: usize::try_from(length_end).unwrap_or(usize::MAX),
                actual: usize::try_from(self.source_len).unwrap_or(usize::MAX),
                context: "window length",
            });
        }
        self.source
            .seek(SeekFrom::Start(position))
            .map_err(LmlError::Io)?;
        self.windows_read = index;
        let window = self.next_window().ok_or_else(|| {
            LmlError::InvalidHeader(format!("window {index} missing after indexed seek"))
        })??;
        if window.len() != self.header.n_channels {
            return Err(LmlError::InvalidHeader(format!(
                "window {index}: decoded channel count {} != header {}",
                window.len(),
                self.header.n_channels
            )));
        }
        Ok(window)
    }

    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        if self.windows_read >= self.header.n_windows {
            return None;
        }
        let mut length = [0u8; 4];
        if let Err(error) = self.source.read_exact(&mut length) {
            return Some(Err(LmlError::Io(error)));
        }
        let payload_len = u32::from_le_bytes(length) as u64;
        let payload_start = match self.source.stream_position() {
            Ok(position) => position,
            Err(error) => return Some(Err(LmlError::Io(error))),
        };
        if payload_len > self.source_len.saturating_sub(payload_start) {
            return Some(Err(LmlError::Truncated {
                expected: usize::try_from(payload_start.saturating_add(payload_len))
                    .unwrap_or(usize::MAX),
                actual: usize::try_from(self.source_len).unwrap_or(usize::MAX),
                context: "window payload",
            }));
        }
        let mut payload = vec![0u8; payload_len as usize];
        if let Err(error) = self.source.read_exact(&mut payload) {
            return Some(Err(LmlError::Io(error)));
        }
        self.windows_read += 1;
        Some(lml::decompress(&payload))
    }

    pub fn decode_all(&mut self) -> LmlResult<Vec<Vec<i64>>> {
        self.rewind()?;
        let mut signal = vec![vec![0i64; self.header.total_samples]; self.header.n_channels];
        let window_size = self.header.window_size;
        for window_index in 0..self.header.n_windows {
            let window = self.next_window().ok_or_else(|| {
                LmlError::InvalidHeader(format!("missing window {window_index}"))
            })??;
            if window.len() != self.header.n_channels {
                return Err(LmlError::InvalidHeader(format!(
                    "window {window_index}: decoded channel count {} != header {}",
                    window.len(),
                    self.header.n_channels
                )));
            }
            let start = window_index.checked_mul(window_size).ok_or_else(|| {
                LmlError::InvalidHeader("window sample offset overflows usize".into())
            })?;
            if start >= self.header.total_samples {
                continue;
            }
            for (channel, samples) in window.iter().enumerate() {
                let count = samples.len().min(self.header.total_samples - start);
                signal[channel][start..start + count].copy_from_slice(&samples[..count]);
            }
        }
        Ok(signal)
    }

    pub fn decode_all_f32_calibrated(
        &mut self,
        out: &mut [f32],
        calibration: &[f32],
    ) -> LmlResult<()> {
        let channels = self.header.n_channels;
        let total = self.header.total_samples;
        if out.len() != channels * total {
            return Err(LmlError::InvalidHeader(format!(
                "output buffer size mismatch: expected {} got {}",
                channels * total,
                out.len()
            )));
        }
        if calibration.len() != channels * 4 {
            return Err(LmlError::InvalidHeader(format!(
                "calib length {} != n_ch*4 ({})",
                calibration.len(),
                channels * 4
            )));
        }
        let mut scale = vec![0.0f32; channels];
        let mut offset = vec![0.0f32; channels];
        for channel in 0..channels {
            let digital_min = calibration[channel * 4];
            let digital_max = calibration[channel * 4 + 1];
            let physical_min = calibration[channel * 4 + 2];
            let physical_max = calibration[channel * 4 + 3];
            let digital_range = digital_max - digital_min;
            if digital_range != 0.0 {
                scale[channel] = (physical_max - physical_min) / digital_range;
                offset[channel] = physical_min - digital_min * scale[channel];
            }
        }

        self.rewind()?;
        for window_index in 0..self.header.n_windows {
            let window = self.next_window().ok_or_else(|| {
                LmlError::InvalidHeader(format!("missing window {window_index}"))
            })??;
            if window.len() != channels {
                return Err(LmlError::InvalidHeader(format!(
                    "window {window_index}: decoded channel count {} != header {channels}",
                    window.len()
                )));
            }
            let start = window_index
                .checked_mul(self.header.window_size)
                .ok_or_else(|| {
                    LmlError::InvalidHeader("window sample offset overflows usize".into())
                })?;
            if start >= total {
                continue;
            }
            for channel in 0..channels {
                let count = window[channel].len().min(total - start);
                let destination = channel * total + start;
                for (index, sample) in window[channel][..count].iter().enumerate() {
                    out[destination + index] = *sample as f32 * scale[channel] + offset[channel];
                }
            }
        }
        Ok(())
    }
}

impl<R: Read + Seek> Iterator for ContainerReader<R> {
    type Item = LmlResult<Vec<Vec<i64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_window()
    }
}
