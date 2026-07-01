//! Random-access range reader — sample-and-channel slicing on top of
//! `LmlReader::windows_for_range`.
//!
//! Phase 3.1 builds the host-side, value-typed API the partial-decode
//! CLI flags (`--channels`, `--time-range`) sit on. `LmlReader`
//! already exposes window-level random access via the LMLFOOT1 seek
//! table from Phase 0.6/0.7; this module collapses that into a single
//! "give me sample range R on channel subset C" call that returns a
//! stitched `RangeSlice`.
//!
//! Bible alignment:
//! - R6  strict types at boundaries: `RangeQuery` is a newtype, not
//!   a tuple of `(u32, u32, Option<Vec<usize>>)`; `RangeSlice` is
//!   distinct from `Vec<Vec<i64>>` (callers can't accidentally pass
//!   a raw decode result where a range slice is expected, and vice-
//!   versa).
//! - R23 validate at both ends: `RangeQuery::new` rejects empty +
//!   inverted ranges + out-of-bounds channel indices at construction.
//!   `RangeReader::read` re-validates against the actual container
//!   header (channels and sample count are unknown at query-build time).
//! - R30 hostile-caller interface: out-of-range channels return a
//!   typed error, never silently degrade to "all channels". An empty
//!   channel list is rejected at construction (no implicit "all").
//! - R31 idempotency: the same `RangeQuery` against the same file
//!   always returns the same `RangeSlice` — no hidden internal state
//!   leaks across calls.

use crate::bcs1_stream::{AnyLmlReader, Bcs1StreamReader};
use crate::error::{LmlError, LmlResult};
use crate::stream::LmlReader;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// Inclusive-start, exclusive-end sample range plus optional channel
/// subset.
///
/// Construction is fallible — invalid ranges are caught at query-build
/// time so the reader never has to defend against `start > end` or
/// duplicate channel indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeQuery {
    start_sample: u32,
    end_sample_exclusive: u32,
    /// `None` = all channels. `Some(_)` = subset, sorted + de-duplicated.
    channels: Option<Vec<usize>>,
}

impl RangeQuery {
    /// New range over `[start, end_exclusive)` samples. `channels=None`
    /// selects every channel; `Some(list)` selects the listed channels
    /// in the order they appear after sort+dedup (which preserves
    /// underlying channel ordering — useful for stable byte-output).
    ///
    /// Errors:
    /// - `InvalidHeader` if `end <= start` (empty range refused; callers
    ///   that want "everything" should not construct a range at all).
    /// - `InvalidHeader` if `channels` is `Some(vec![])` (empty subset
    ///   refused — semantics would be "decode nothing", which is a bug
    ///   shape, not a feature).
    pub fn new(
        start_sample: u32,
        end_sample_exclusive: u32,
        channels: Option<Vec<usize>>,
    ) -> LmlResult<Self> {
        if end_sample_exclusive <= start_sample {
            return Err(LmlError::InvalidHeader(format!(
                "RangeQuery: end {end_sample_exclusive} <= start {start_sample}"
            )));
        }
        let channels = match channels {
            None => None,
            Some(mut v) => {
                if v.is_empty() {
                    return Err(LmlError::InvalidHeader(
                        "RangeQuery: channel list is empty — pass None for 'all channels'".into(),
                    ));
                }
                v.sort_unstable();
                v.dedup();
                Some(v)
            }
        };
        Ok(Self {
            start_sample,
            end_sample_exclusive,
            channels,
        })
    }

    pub fn start_sample(&self) -> u32 {
        self.start_sample
    }

    pub fn end_sample_exclusive(&self) -> u32 {
        self.end_sample_exclusive
    }

    pub fn channels(&self) -> Option<&[usize]> {
        self.channels.as_deref()
    }

    pub fn sample_count(&self) -> u32 {
        self.end_sample_exclusive - self.start_sample
    }
}

/// Decoded slice — channels selected, samples trimmed to the requested
/// range.
///
/// `signal[ch_idx_within_selection][sample_offset_from_start]`.
/// If the original query carried `channels=Some(vec![3, 7])`, then
/// `signal[0]` is channel 3 and `signal[1]` is channel 7 of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeSlice {
    /// Selected channels (post sort+dedup). `None` = all source channels.
    pub channels: Option<Vec<usize>>,
    /// Absolute sample range covered (matches the query exactly).
    pub start_sample: u32,
    pub end_sample_exclusive: u32,
    /// Decoded signal: `[selected_ch][sample_offset]`.
    pub signal: Vec<Vec<i64>>,
}

impl RangeSlice {
    pub fn n_channels(&self) -> usize {
        self.signal.len()
    }

    pub fn n_samples(&self) -> usize {
        self.signal.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Thin facade that owns an [`AnyLmlReader`] and exposes range-based reads.
///
/// ADR 0069/0071 L9: `RangeReader` used to wrap `LmlReader` (frozen,
/// LML1-only) directly. Since `write_abir` now emits `BCS1` by default, the
/// inner reader must be able to dispatch to either wire format — it now
/// wraps `AnyLmlReader`, the magic-dispatching facade in `bcs1_stream`.
/// `RangeReader` does NOT re-parse the container itself; the underlying
/// reader retains its position across calls.
pub struct RangeReader<R: Read + Seek> {
    inner: AnyLmlReader<R>,
}

impl<R: Read + Seek> RangeReader<R> {
    /// Build over an existing legacy `LmlReader`. Errors if the reader's
    /// container has no LMLFOOT1 seek table (legacy file written before
    /// Phase 0.6/0.7). Kept for back-compat call sites that already hold a
    /// constructed `LmlReader`; new call sites should prefer
    /// [`RangeReader::open`] / [`RangeReader::open_from_source`], which
    /// dispatch on magic automatically instead of requiring the caller to
    /// pick a reader type up front.
    pub fn new(inner: LmlReader<R>) -> LmlResult<Self> {
        Self::from_any(AnyLmlReader::Legacy(inner))
    }

    /// Build over an existing [`Bcs1StreamReader`]. The BCS1 counterpart of
    /// [`RangeReader::new`].
    pub fn new_bcs1(inner: Bcs1StreamReader<R>) -> LmlResult<Self> {
        Self::from_any(AnyLmlReader::Bcs1(inner))
    }

    fn from_any(inner: AnyLmlReader<R>) -> LmlResult<Self> {
        if inner.offset_table().is_none() {
            return Err(LmlError::InvalidHeader(
                "RangeReader: container has no LMLFOOT1 seek table; legacy files \
                 cannot be range-decoded — re-encode or fall back to sequential decode"
                    .into(),
            ));
        }
        Ok(Self { inner })
    }

    /// Magic-dispatching construction directly from a `Read + Seek` source
    /// positioned at byte 0 (ADR 0069/0071 L9) — peeks the leading 4 bytes
    /// and routes to the BCS1-aware or legacy-LML1 streaming reader before
    /// wrapping it, so callers don't need to construct an
    /// `LmlReader`/`Bcs1StreamReader` themselves first.
    pub fn open_from_source(source: R) -> LmlResult<Self> {
        Self::from_any(AnyLmlReader::from_source(source)?)
    }

    pub fn header(&self) -> &crate::stream::ContainerHeader {
        self.inner.header()
    }

    /// Decode `query` against this file. Returns a stitched, trimmed
    /// slice. Errors on out-of-bounds channel index or range past EOF.
    pub fn read(&mut self, query: &RangeQuery) -> LmlResult<RangeSlice> {
        // Snapshot the header fields up front so the borrow doesn't
        // collide with `windows_for_range`'s `&mut self.inner` below.
        let (total_samples, n_channels_file) = {
            let hdr = self.inner.header();
            (hdr.total_samples, hdr.n_channels)
        };
        let total_samples_u32: u32 = total_samples.try_into().map_err(|_| {
            LmlError::InvalidHeader(format!(
                "RangeReader: total_samples {total_samples} > u32::MAX"
            ))
        })?;
        // Re-validate against the live header — `RangeQuery::new` cannot
        // know channel-count or sample-count at construction time.
        if query.start_sample >= total_samples_u32 {
            return Err(LmlError::InvalidHeader(format!(
                "RangeReader: start_sample {} past total_samples {total_samples_u32}",
                query.start_sample
            )));
        }
        // Clamp end to total_samples — callers that ask for "the rest"
        // should not need to know the file length up front.
        let end = query.end_sample_exclusive.min(total_samples_u32);
        if end <= query.start_sample {
            return Err(LmlError::InvalidHeader(format!(
                "RangeReader: clamped end {end} <= start {}",
                query.start_sample
            )));
        }
        if let Some(chans) = query.channels() {
            for &c in chans {
                if c >= n_channels_file {
                    return Err(LmlError::InvalidHeader(format!(
                        "RangeReader: channel {c} out of range (file has {n_channels_file} channels)"
                    )));
                }
            }
        }
        let windows = self.inner.windows_for_range(query.start_sample, end)?;
        if windows.is_empty() {
            return Err(LmlError::InvalidHeader(format!(
                "RangeReader: no windows intersect range [{}, {end})",
                query.start_sample
            )));
        }
        // Figure out where the first returned window starts in absolute
        // sample coordinates so we can trim the head correctly. The
        // offset table tells us; we already know windows_for_range
        // returns a contiguous run starting at the first intersecting
        // window.
        let table = self
            .inner
            .offset_table()
            .expect("RangeReader::new guarantees offset_table present");
        let range_idx = table
            .windows_for_range(query.start_sample, end)
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "RangeReader: empty window-range [{}, {end})",
                    query.start_sample
                ))
            })?;
        let first_window_idx = *range_idx.start();
        let first_window_abs_sample = table.entries()[first_window_idx].first_sample_idx;

        // Stitch + trim. Output channel ordering follows the query's
        // sort+dedup'd channel list (or 0..n_channels when None).
        let selected: Vec<usize> = match query.channels() {
            Some(chs) => chs.to_vec(),
            None => (0..n_channels_file).collect(),
        };
        let n_samples_out = (end - query.start_sample) as usize;
        let mut out: Vec<Vec<i64>> = (0..selected.len())
            .map(|_| Vec::with_capacity(n_samples_out))
            .collect();

        let mut abs_sample_cursor = first_window_abs_sample;
        for w in &windows {
            // `w` is `[n_ch][T_window]` (full file channels, not the
            // selected subset). Per-window length comes from the
            // payload itself.
            let w_len = w.first().map(|c| c.len()).unwrap_or(0) as u32;
            let window_start = abs_sample_cursor;
            let window_end_excl = window_start + w_len;

            // Compute intersection with the query in absolute coords.
            let head_skip = query.start_sample.saturating_sub(window_start);
            let tail_clip = window_end_excl.saturating_sub(end);
            let take_len = (w_len.saturating_sub(head_skip).saturating_sub(tail_clip)) as usize;
            if take_len > 0 {
                let head = head_skip as usize;
                for (out_idx, &src_ch) in selected.iter().enumerate() {
                    let src = &w[src_ch][head..head + take_len];
                    out[out_idx].extend_from_slice(src);
                }
            }
            abs_sample_cursor = window_end_excl;
        }
        // Defensive: the stitch loop must have produced exactly the
        // requested length. Bible R7 — catch logic bugs at the sink,
        // not three layers downstream.
        debug_assert!(
            out.iter().all(|c| c.len() == n_samples_out),
            "RangeReader: stitched length mismatch"
        );
        Ok(RangeSlice {
            channels: query.channels().map(|c| c.to_vec()),
            start_sample: query.start_sample,
            end_sample_exclusive: end,
            signal: out,
        })
    }
}

impl RangeReader<BufReader<File>> {
    /// Open a file by path, dispatching on its leading 4 bytes (ADR
    /// 0069/0071 L9) — the entry point most callers should use instead of
    /// constructing an `LmlReader`/`Bcs1StreamReader` + calling
    /// `RangeReader::new` themselves. Replaces the old two-step
    /// `LmlReader::open(path)` + `RangeReader::new(reader)` pattern (which
    /// hard-failed with `InvalidMagic` on a `BCS1` file before
    /// `RangeReader::new` was ever reached).
    pub fn open(path: &Path) -> LmlResult<Self> {
        let file = File::open(path).map_err(LmlError::Io)?;
        Self::open_from_source(BufReader::new(file))
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
    fn query_rejects_empty_range() {
        assert!(RangeQuery::new(100, 100, None).is_err());
        assert!(RangeQuery::new(100, 50, None).is_err());
    }

    #[test]
    fn query_rejects_empty_channel_list() {
        let err = RangeQuery::new(0, 100, Some(vec![])).unwrap_err();
        match err {
            LmlError::InvalidHeader(msg) => assert!(msg.contains("empty"), "got: {msg}"),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn query_sorts_and_dedups_channels() {
        let q = RangeQuery::new(0, 100, Some(vec![5, 1, 3, 1, 5])).unwrap();
        assert_eq!(q.channels(), Some(&[1usize, 3, 5][..]));
    }

    #[test]
    fn read_full_range_all_channels_matches_original() {
        let (sig, sink) = write_synth(3, 384, 128, 7);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        let q = RangeQuery::new(0, 384, None).unwrap();
        let slice = rr.read(&q).unwrap();
        assert_eq!(slice.n_channels(), 3);
        assert_eq!(slice.n_samples(), 384);
        for ch in 0..3 {
            assert_eq!(slice.signal[ch], sig[ch], "channel {ch} drift");
        }
    }

    #[test]
    fn read_mid_window_range_trims_head_and_tail() {
        // Windows of 128 each; range [150, 400) → trim 22 off window 1's
        // head and 16 off window 3's tail.
        let (sig, sink) = write_synth(2, 512, 128, 23);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        let q = RangeQuery::new(150, 400, None).unwrap();
        let slice = rr.read(&q).unwrap();
        assert_eq!(slice.start_sample, 150);
        assert_eq!(slice.end_sample_exclusive, 400);
        assert_eq!(slice.n_samples(), 250);
        for ch in 0..2 {
            assert_eq!(slice.signal[ch], sig[ch][150..400]);
        }
    }

    #[test]
    fn read_channel_subset_returns_only_selected() {
        let (sig, sink) = write_synth(4, 256, 128, 31);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        let q = RangeQuery::new(0, 256, Some(vec![2, 0])).unwrap();
        let slice = rr.read(&q).unwrap();
        assert_eq!(slice.n_channels(), 2);
        // Output channel order = sort+dedup'd, so [0, 2].
        assert_eq!(slice.channels.as_ref().unwrap(), &[0usize, 2]);
        assert_eq!(slice.signal[0], sig[0]);
        assert_eq!(slice.signal[1], sig[2]);
    }

    #[test]
    fn read_clamps_end_to_total_samples() {
        let (sig, sink) = write_synth(1, 200, 128, 41);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        // Ask for [50, 9_999) — must clamp to [50, 200).
        let q = RangeQuery::new(50, 9_999, None).unwrap();
        let slice = rr.read(&q).unwrap();
        assert_eq!(slice.end_sample_exclusive, 200);
        assert_eq!(slice.n_samples(), 150);
        assert_eq!(slice.signal[0], sig[0][50..200]);
    }

    #[test]
    fn read_errors_on_start_past_eof() {
        let (_sig, sink) = write_synth(1, 128, 128, 43);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        let q = RangeQuery::new(9_999, 10_000, None).unwrap();
        match rr.read(&q) {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("past total_samples"), "got: {msg}")
            }
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_errors_on_out_of_range_channel() {
        let (_sig, sink) = write_synth(2, 128, 64, 47);
        let mut rr = RangeReader::open_from_source(std::io::Cursor::new(sink)).unwrap();
        let q = RangeQuery::new(0, 64, Some(vec![5])).unwrap();
        match rr.read(&q) {
            Err(LmlError::InvalidHeader(msg)) => {
                assert!(msg.contains("out of range"), "got: {msg}")
            }
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }
}
