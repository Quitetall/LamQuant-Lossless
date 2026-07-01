//! BCS1-aware streaming reader (ADR 0069/0071 L9 — read-side completion).
//!
//! `lamquant_lml_legacy::stream::LmlReader` is FROZEN and understands only
//! the legacy 32-byte `LML1` header — its very first read (`from_source`)
//! assumes a 32-byte header and its magic check rejects anything else with
//! `InvalidMagic`. Since `abir_container::write_abir` now emits `BCS1` by
//! default (the L9 wire change), every STREAMING (window-at-a-time,
//! `Read + Seek`) consumer needs a BCS1-aware counterpart — the whole-file
//! facade (`abir_container::{read_file,read_bytes,read_from}`) already
//! dispatches, but `LmlReader` and everything built on it (`range::
//! RangeReader`, `bin/lml.rs`'s streaming decode) did not.
//!
//! [`Bcs1StreamReader`] is that counterpart: a small, deliberate CLONE of
//! `LmlReader`'s window-streaming machinery with the ONE byte that changed
//! threaded through — header length 32 -> [`lamquant_abir::BCS1_HEADER_LEN`]
//! (40), header fields read off [`lamquant_abir::Bcs1Header::parse`] instead
//! of hand-rolled byte offsets. Everything AFTER the header — metadata JSON,
//! window-length index, per-window `LML1` packets, `LMLFOOT1` footer — is
//! byte-identical to the legacy wire (see `abir_container` module docs), so
//! the footer-probe / seek / windows_for_range bodies below are copied
//! verbatim from `LmlReader`, not reimplemented. `lamquant-lml-legacy` stays
//! untouched — this is new code in the live `lamquant-lossless` crate.
//!
//! [`AnyLmlReader`] is the magic-dispatching facade every streaming call
//! site should use instead of hardcoding `stream::LmlReader::open`: it
//! peeks the leading 4 bytes of the source and routes to
//! [`Bcs1StreamReader`] or the frozen `LmlReader`, then exposes one uniform
//! `header()`/`next_window()`/`seek_to_window()`/`windows_for_range()`/
//! `rewind()` surface regardless of which wire format it opened — the same
//! routing rule `abir_container::read_bytes` applies to a whole-file
//! buffer, applied here to a `Read + Seek` stream.

use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::offset_table::{OffsetTable, ENTRY_SIZE, FOOTER_MAGIC, FOOTER_SIZE};
use crate::stream::{ContainerHeader, LmlReader};
use lamquant_abir::{Bcs1Header, BCS1_HEADER_LEN, BCS1_MAGIC};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Streaming reader for a `BCS1` container — the BCS1 counterpart of
/// `lamquant_lml_legacy::stream::LmlReader`. See module docs.
///
/// Reuses `lamquant_lml_legacy::stream::ContainerHeader` as its header type
/// (rather than defining a parallel struct with the identical field set) so
/// [`AnyLmlReader::header`] can return one type regardless of which reader
/// it dispatched to.
pub struct Bcs1StreamReader<R: Read + Seek> {
    reader: R,
    header: ContainerHeader,
    windows_read: usize,
    /// `LMLFOOT1` seek table parsed from EOF when present — same semantics
    /// as `LmlReader::offset_table`.
    offset_table: Option<OffsetTable>,
    /// Absolute file offset of the first window payload's length prefix.
    first_payload_pos: u64,
}

impl Bcs1StreamReader<BufReader<File>> {
    /// Open a BCS1 file for streaming decode. Back-compat-shaped shortcut
    /// mirroring `LmlReader::open`.
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::from_source(BufReader::new(file))
    }
}

impl<R: Read + Seek> Bcs1StreamReader<R> {
    /// Construct from any `Read + Seek` source positioned at byte 0 of a
    /// `BCS1` container (mirrors `LmlReader::from_source`'s contract, with
    /// the 40-byte typed header in place of the legacy 32-byte one).
    pub fn from_source(mut reader: R) -> LmlResult<Self> {
        let mut hdr = [0u8; BCS1_HEADER_LEN];
        reader.read_exact(&mut hdr).map_err(LmlError::Io)?;
        let parsed = Bcs1Header::parse(&hdr)
            .map_err(|e| LmlError::InvalidHeader(format!("BCS1 header: {e}")))?;

        let n_channels = parsed.n_channels as usize;
        let n_windows = parsed.n_windows as usize;
        let total_samples = parsed.total_samples as usize;
        let window_size = parsed.window_size as usize;
        let meta_len = parsed.metadata_length as usize;

        // Read metadata
        let mut meta_buf = vec![0u8; meta_len];
        reader.read_exact(&mut meta_buf).map_err(LmlError::Io)?;
        let metadata = String::from_utf8_lossy(&meta_buf).to_string();

        // Skip window index
        let skip = n_windows as u64 * 4;
        reader
            .seek(SeekFrom::Current(skip as i64))
            .map_err(LmlError::Io)?;

        let first_payload_pos = reader.stream_position().map_err(LmlError::Io)?;

        // Probe EOF for an LMLFOOT1 seek table — identical body to
        // `LmlReader::try_read_footer` (see there for the security
        // rationale on why the on-disk `table_start` field is ignored).
        let offset_table = Self::try_read_footer(&mut reader)?;

        // Seek back to the first payload so sequential reads continue to
        // work from window 0.
        reader
            .seek(SeekFrom::Start(first_payload_pos))
            .map_err(LmlError::Io)?;

        Ok(Self {
            reader,
            header: ContainerHeader {
                n_channels,
                n_windows,
                total_samples,
                window_size,
                metadata,
            },
            windows_read: 0,
            offset_table,
            first_payload_pos,
        })
    }

    /// Attempt to parse the optional `LMLFOOT1` seek table at EOF. Copied
    /// verbatim from `LmlReader::try_read_footer` (the legacy crate is
    /// frozen, so this can't be shared by reference) — including the
    /// security fix that derives `table_start` from `n_windows` instead of
    /// trusting the unprotected on-disk field.
    fn try_read_footer(reader: &mut R) -> LmlResult<Option<OffsetTable>> {
        let end = reader.seek(SeekFrom::End(0)).map_err(LmlError::Io)?;
        if end < FOOTER_SIZE as u64 {
            return Ok(None);
        }
        reader
            .seek(SeekFrom::End(-(FOOTER_SIZE as i64)))
            .map_err(LmlError::Io)?;
        let mut footer = [0u8; FOOTER_SIZE];
        reader.read_exact(&mut footer).map_err(LmlError::Io)?;
        if &footer[0..8] != FOOTER_MAGIC {
            return Ok(None);
        }
        let n_windows =
            u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]) as u64;
        let table_bytes = n_windows.checked_mul(ENTRY_SIZE as u64).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "LMLFOOT1: n_windows {n_windows} * ENTRY_SIZE overflows u64"
            ))
        })?;
        let combined_len = table_bytes.checked_add(FOOTER_SIZE as u64).ok_or_else(|| {
            LmlError::InvalidHeader("LMLFOOT1: combined_len overflows u64".into())
        })?;
        if combined_len > end {
            return Err(LmlError::InvalidHeader(format!(
                "LMLFOOT1: n_windows {n_windows} requires {combined_len} bytes \
                 but file is only {end} bytes"
            )));
        }
        // Derive the seek target ourselves — do NOT trust the unprotected
        // `table_start` field at footer[20..24] (Lamu V4 Pro security
        // review of 4d347b0, same finding the legacy reader fixes).
        let derived_table_start = end - combined_len;
        let mut combined = vec![0u8; combined_len as usize];
        reader
            .seek(SeekFrom::Start(derived_table_start))
            .map_err(LmlError::Io)?;
        reader.read_exact(&mut combined).map_err(LmlError::Io)?;
        OffsetTable::read_from_buffer(&combined)
    }

    /// Returns the parsed `LMLFOOT1` seek table if present.
    pub fn offset_table(&self) -> Option<&OffsetTable> {
        self.offset_table.as_ref()
    }

    /// Random-access seek to window `idx`. Requires the file carry a seek
    /// table (every `write_abir` output does).
    pub fn seek_to_window(&mut self, idx: usize) -> LmlResult<()> {
        let table = self.offset_table.as_ref().ok_or_else(|| {
            LmlError::InvalidHeader(
                "seek_to_window: BCS1 file has no LMLFOOT1 seek table; \
                 fall back to sequential decode via next_window"
                    .into(),
            )
        })?;
        let entry = table.entries().get(idx).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "seek_to_window: window {idx} out of range (len {})",
                table.len()
            ))
        })?;
        self.reader
            .seek(SeekFrom::Start(entry.abs_offset))
            .map_err(LmlError::Io)?;
        self.windows_read = idx;
        Ok(())
    }

    /// Decode every window that intersects the sample range
    /// `[start, end_exclusive)`. See `LmlReader::windows_for_range`.
    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        let range = self
            .offset_table
            .as_ref()
            .ok_or_else(|| {
                LmlError::InvalidHeader(
                    "windows_for_range: BCS1 file has no LMLFOOT1 seek table".into(),
                )
            })?
            .windows_for_range(start, end_exclusive)
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "windows_for_range: empty / past-EOF range [{start}, {end_exclusive})"
                ))
            })?;
        let mut out = Vec::with_capacity(range.end() - range.start() + 1);
        for w in range {
            self.seek_to_window(w)?;
            match self.next_window() {
                Some(Ok(decoded)) => out.push(decoded),
                Some(Err(e)) => return Err(e),
                None => {
                    return Err(LmlError::InvalidHeader(format!(
                        "windows_for_range: seek_to_window({w}) succeeded but next_window EOF"
                    )))
                }
            }
        }
        Ok(out)
    }

    /// Rewind to sequential-decode start (window 0).
    pub fn rewind(&mut self) -> LmlResult<()> {
        self.reader
            .seek(SeekFrom::Start(self.first_payload_pos))
            .map_err(LmlError::Io)?;
        self.windows_read = 0;
        Ok(())
    }

    /// Get container header info.
    pub fn header(&self) -> &ContainerHeader {
        &self.header
    }

    /// Read and decompress the next window. Returns None at EOF.
    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        if self.windows_read >= self.header.n_windows {
            return None;
        }

        let mut len_buf = [0u8; 4];
        if let Err(e) = self.reader.read_exact(&mut len_buf) {
            return Some(Err(LmlError::Io(e)));
        }
        let payload_len = u32::from_le_bytes(len_buf) as usize;

        let mut payload = vec![0u8; payload_len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            return Some(Err(LmlError::Io(e)));
        }

        self.windows_read += 1;
        Some(lml::decompress(&payload))
    }
}

impl<R: Read + Seek> Iterator for Bcs1StreamReader<R> {
    type Item = LmlResult<Vec<Vec<i64>>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_window()
    }
}

/// Magic-dispatching streaming reader (ADR 0069/0071 L9). Routes to
/// [`Bcs1StreamReader`] or the frozen `LmlReader` based on the leading 4
/// bytes of the source, then exposes one uniform surface — this is the
/// entry point streaming call sites should use instead of hardcoding
/// `stream::LmlReader::open`/`from_source`.
pub enum AnyLmlReader<R: Read + Seek> {
    Legacy(LmlReader<R>),
    Bcs1(Bcs1StreamReader<R>),
}

impl AnyLmlReader<BufReader<File>> {
    /// Open a file, dispatching on its leading 4 bytes.
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::from_source(BufReader::new(file))
    }
}

impl<R: Read + Seek> AnyLmlReader<R> {
    /// Construct from any `Read + Seek` source positioned at byte 0. Peeks
    /// the leading 4 bytes then seeks back to 0 before handing off to
    /// whichever reader understands them — same routing rule as
    /// `abir_container::read_bytes`, applied to a streaming source instead
    /// of an in-memory buffer.
    pub fn from_source(mut source: R) -> LmlResult<Self> {
        let mut magic = [0u8; 4];
        source.read_exact(&mut magic).map_err(LmlError::Io)?;
        source.seek(SeekFrom::Start(0)).map_err(LmlError::Io)?;
        if magic == *BCS1_MAGIC {
            Ok(Self::Bcs1(Bcs1StreamReader::from_source(source)?))
        } else {
            Ok(Self::Legacy(LmlReader::from_source(source)?))
        }
    }

    pub fn header(&self) -> &ContainerHeader {
        match self {
            Self::Legacy(r) => r.header(),
            Self::Bcs1(r) => r.header(),
        }
    }

    pub fn offset_table(&self) -> Option<&OffsetTable> {
        match self {
            Self::Legacy(r) => r.offset_table(),
            Self::Bcs1(r) => r.offset_table(),
        }
    }

    pub fn seek_to_window(&mut self, idx: usize) -> LmlResult<()> {
        match self {
            Self::Legacy(r) => r.seek_to_window(idx),
            Self::Bcs1(r) => r.seek_to_window(idx),
        }
    }

    pub fn windows_for_range(
        &mut self,
        start: u32,
        end_exclusive: u32,
    ) -> LmlResult<Vec<Vec<Vec<i64>>>> {
        match self {
            Self::Legacy(r) => r.windows_for_range(start, end_exclusive),
            Self::Bcs1(r) => r.windows_for_range(start, end_exclusive),
        }
    }

    pub fn rewind(&mut self) -> LmlResult<()> {
        match self {
            Self::Legacy(r) => r.rewind(),
            Self::Bcs1(r) => r.rewind(),
        }
    }

    pub fn next_window(&mut self) -> Option<LmlResult<Vec<Vec<i64>>>> {
        match self {
            Self::Legacy(r) => r.next_window(),
            Self::Bcs1(r) => r.next_window(),
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

    fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut sig = vec![Vec::with_capacity(t); n_ch];
        for ch in 0..n_ch {
            for _ in 0..t {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                sig[ch].push(((state >> 33) as i32) as i64 % 8000);
            }
        }
        sig
    }

    fn write_synth(n_ch: usize, t: usize, window: usize, seed: u64) -> (Vec<Vec<i64>>, Vec<u8>) {
        let sig = synth_signal(n_ch, t, seed);
        let mut sink: Vec<u8> = Vec::new();
        container::write_into(&mut sink, &sig, 250.0, window, 0, "{}", LpcMode::default()).unwrap();
        (sig, sink)
    }

    #[test]
    fn bcs1_stream_reader_round_trips_full_signal() {
        let (sig, sink) = write_synth(3, 384, 128, 11);
        assert_eq!(&sink[0..4], BCS1_MAGIC, "container::write_into must emit BCS1");
        let mut reconstructed: Vec<Vec<i64>> = vec![Vec::new(); 3];
        let reader = Bcs1StreamReader::from_source(std::io::Cursor::new(sink)).unwrap();
        for window in reader {
            let w = window.unwrap();
            for (ch, samples) in w.iter().enumerate() {
                reconstructed[ch].extend_from_slice(samples);
            }
        }
        for ch in 0..3 {
            assert_eq!(&reconstructed[ch][..384], &sig[ch][..]);
        }
    }

    #[test]
    fn bcs1_stream_reader_seek_to_window_decodes_only_target() {
        let (sig, sink) = write_synth(2, 384, 128, 7);
        let mut reader = Bcs1StreamReader::from_source(std::io::Cursor::new(sink)).unwrap();
        reader.seek_to_window(2).unwrap();
        let w = reader.next_window().unwrap().unwrap();
        for ch in 0..2 {
            assert_eq!(w[ch], sig[ch][256..384], "channel {ch} window 2 drift");
        }
    }

    #[test]
    fn any_lml_reader_dispatches_bcs1() {
        let (sig, sink) = write_synth(2, 256, 128, 3);
        let mut reconstructed: Vec<Vec<i64>> = vec![Vec::new(); 2];
        let reader = AnyLmlReader::from_source(std::io::Cursor::new(sink)).unwrap();
        assert!(matches!(reader, AnyLmlReader::Bcs1(_)));
        for window in reader {
            let w = window.unwrap();
            for (ch, samples) in w.iter().enumerate() {
                reconstructed[ch].extend_from_slice(samples);
            }
        }
        for ch in 0..2 {
            assert_eq!(&reconstructed[ch][..256], &sig[ch][..]);
        }
    }

    #[test]
    fn any_lml_reader_dispatches_legacy_lml1() {
        // Hand-craft a minimal LML1 buffer distinct enough to exercise the
        // legacy branch without linking the retiring `legacy-encode`
        // writer (this test file doesn't gate on that feature). Uses the
        // same shape `stream::open_rejects_invalid_magic`'s sibling tests
        // rely on: a 32-byte header + zero windows is enough to prove
        // DISPATCH routes to `AnyLmlReader::Legacy`, independent of
        // whether the payload itself decodes.
        let mut buf = vec![0u8; 32];
        buf[0..4].copy_from_slice(b"LML1");
        buf[4] = 1; // version_major probe
        buf[6..8].copy_from_slice(&1u16.to_le_bytes()); // n_channels
        buf[8..10].copy_from_slice(&0u16.to_le_bytes()); // n_windows = 0
        buf[10..14].copy_from_slice(&0u32.to_le_bytes()); // total_samples = 0
        buf[14..16].copy_from_slice(&128u16.to_le_bytes()); // window_size
        buf[20] = 16; // bit_depth
        // meta_len = 0 at [22..26]
        let reader = AnyLmlReader::from_source(std::io::Cursor::new(buf));
        // total_samples=0/n_windows=0 aren't validated by `LmlReader::
        // from_source` (only `read_bytes`/`parse_header` guard those) — the
        // point here is purely that dispatch chose the Legacy branch, not
        // BCS1.
        assert!(matches!(reader, Ok(AnyLmlReader::Legacy(_))));
    }
}
