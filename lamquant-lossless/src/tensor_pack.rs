//! LQTP1 — the LamQuant training tensor pack (ADR 0075, Part B / the window-pack).
//!
//! A derived, immutable, memory-mappable store of exactly the manifest's pre-normalized
//! windows, so training reads them RAW (mmap → random window → dequant on GPU) with zero
//! decode / normalize / fallback in the hot path. The `.lma` stays the 24-bit lossless
//! archival truth; the pack is rebuilt from it when the normalize or dtype changes.
//!
//! Values are stored as **block floating point (BFP)** — a per-window-per-channel f32
//! scale + integer mantissas — configurable at build time:
//!   - `Int16` (16 honest bits, dynamic-range-adaptive; beats fp16's 11) — the default,
//!   - `Int8`  (matches the model's bf16 compute; ~half the size again),
//!   - `F32`   (24-bit-faithful; the high-R rebuild target — scales are 1.0, mantissas
//!     are the raw f32).
//!
//! This module is B1: the BFP codec + the fixed-stride format + a fail-closed, mmap'd
//! reader. The offline builder (B2) and the PyO3 reader (B3) sit on top.

use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// The mantissa dtype of a pack (block floating point unless `F32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackDtype {
    /// 8-bit BFP mantissa (per-channel f32 scale). Matches bf16 compute; smallest.
    Int8,
    /// 16-bit BFP mantissa (per-channel f32 scale). 16 honest bits; the default.
    Int16,
    /// Raw f32 (scale = 1.0). 24-bit-faithful; the high-R rebuild target.
    F32,
}

impl PackDtype {
    pub fn to_u8(self) -> u8 {
        match self {
            PackDtype::Int8 => 1,
            PackDtype::Int16 => 2,
            PackDtype::F32 => 3,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(PackDtype::Int8),
            2 => Some(PackDtype::Int16),
            3 => Some(PackDtype::F32),
            _ => None,
        }
    }

    /// Parse the CLI/build spelling.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "int8" | "i8" => Some(PackDtype::Int8),
            "int16" | "i16" => Some(PackDtype::Int16),
            "f32" | "float32" => Some(PackDtype::F32),
            _ => None,
        }
    }

    /// Bytes per mantissa element.
    pub fn mant_size(self) -> usize {
        match self {
            PackDtype::Int8 => 1,
            PackDtype::Int16 => 2,
            PackDtype::F32 => 4,
        }
    }

    /// The symmetric max mantissa magnitude for the integer dtypes (unused for `F32`).
    pub fn mant_max(self) -> f32 {
        match self {
            PackDtype::Int8 => i8::MAX as f32,   // 127
            PackDtype::Int16 => i16::MAX as f32, // 32767
            PackDtype::F32 => 1.0,
        }
    }
}

/// Quantize one `[n_ch, t]` (row-major) f32 window to BFP: a per-channel f32 scale plus
/// packed integer (or raw f32) mantissa bytes. Scale = `max(|row|)/mant_max`; a silent
/// channel gets scale 0 + zero mantissas. For `F32`, scale is 1.0 and the mantissa is the
/// raw f32 bytes (exact).
pub fn quantize_window(x: &[f32], n_ch: usize, t: usize, dtype: PackDtype) -> (Vec<f32>, Vec<u8>) {
    debug_assert_eq!(x.len(), n_ch * t);
    let mut scales = vec![0.0f32; n_ch];
    let mut mant = Vec::with_capacity(n_ch * t * dtype.mant_size());
    for c in 0..n_ch {
        let row = &x[c * t..(c + 1) * t];
        match dtype {
            PackDtype::F32 => {
                scales[c] = 1.0;
                for &v in row {
                    mant.extend_from_slice(&v.to_le_bytes());
                }
            }
            PackDtype::Int8 | PackDtype::Int16 => {
                let mant_max = dtype.mant_max();
                let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                if amax == 0.0 {
                    scales[c] = 0.0;
                    mant.resize(mant.len() + t * dtype.mant_size(), 0);
                } else {
                    let scale = amax / mant_max;
                    scales[c] = scale;
                    for &v in row {
                        let q = (v / scale).round().clamp(-mant_max, mant_max);
                        match dtype {
                            PackDtype::Int8 => mant.push(q as i8 as u8),
                            PackDtype::Int16 => mant.extend_from_slice(&(q as i16).to_le_bytes()),
                            PackDtype::F32 => unreachable!(),
                        }
                    }
                }
            }
        }
    }
    (scales, mant)
}

/// Inverse of [`quantize_window`]: dequantize BFP mantissas + scales back to `[n_ch, t]`
/// f32 (`mantissa * scale`, or the raw f32 for `F32`).
pub fn dequantize_window(
    scales: &[f32],
    mant: &[u8],
    n_ch: usize,
    t: usize,
    dtype: PackDtype,
) -> Vec<f32> {
    debug_assert_eq!(scales.len(), n_ch);
    debug_assert_eq!(mant.len(), n_ch * t * dtype.mant_size());
    let mut out = vec![0.0f32; n_ch * t];
    for c in 0..n_ch {
        let scale = scales[c];
        for i in 0..t {
            let idx = c * t + i;
            out[idx] = match dtype {
                PackDtype::Int8 => (mant[idx] as i8) as f32 * scale,
                PackDtype::Int16 => {
                    let o = idx * 2;
                    i16::from_le_bytes([mant[o], mant[o + 1]]) as f32 * scale
                }
                PackDtype::F32 => {
                    let o = idx * 4;
                    f32::from_le_bytes([mant[o], mant[o + 1], mant[o + 2], mant[o + 3]])
                }
            };
        }
    }
    out
}

/// Errors from the pack format layer.
#[derive(Debug)]
pub enum PackError {
    BadMagic,
    BadVersion(u8),
    BadDtype(u8),
    Truncated { expected: usize, actual: usize },
    ManifestMismatch,
    ShapeMismatch(String),
    Io(std::io::Error),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::BadMagic => write!(f, "not an LQTP pack (bad magic)"),
            PackError::BadVersion(v) => write!(f, "unsupported LQTP version {v}"),
            PackError::BadDtype(v) => write!(f, "unknown pack dtype tag {v}"),
            PackError::Truncated { expected, actual } => {
                write!(
                    f,
                    "pack truncated: expected >= {expected} bytes, got {actual}"
                )
            }
            PackError::ManifestMismatch => {
                write!(
                    f,
                    "pack manifest hash != loaded manifest (rebuild the pack)"
                )
            }
            PackError::ShapeMismatch(s) => write!(f, "pack shape mismatch: {s}"),
            PackError::Io(e) => write!(f, "pack I/O error: {e}"),
        }
    }
}

impl std::error::Error for PackError {}

impl From<std::io::Error> for PackError {
    fn from(e: std::io::Error) -> Self {
        PackError::Io(e)
    }
}

/// LQTP1 magic / version / fixed header length.
pub const LQTP_MAGIC: &[u8; 4] = b"LQTP";
pub const LQTP_VERSION: u8 = 1;
pub const LQTP_HEADER_LEN: usize = 64;

/// The fixed 64-byte LQTP1 header. The window index is IMPLICIT (manifest order: row `i`
/// is manifest entry `i`), so the only stored index is `manifest_sha256` — the loader
/// refuses a pack whose hash != the loaded manifest, making the v11-desync OOM class
/// structurally impossible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackHeader {
    pub dtype: PackDtype,
    pub n_channels: usize,
    pub window_len: usize,
    pub n_windows: usize,
    pub record_stride: usize,
    pub manifest_sha256: [u8; 32],
}

impl PackHeader {
    /// Bytes per window record: `n_ch` f32 scales, then `n_ch*window_len` mantissas.
    pub fn record_stride(dtype: PackDtype, n_channels: usize, window_len: usize) -> usize {
        n_channels * 4 + n_channels * window_len * dtype.mant_size()
    }

    /// Byte offset of window `row`'s record.
    pub fn record_offset(&self, row: usize) -> usize {
        LQTP_HEADER_LEN + row * self.record_stride
    }

    /// Total pack file length for `n_windows` records.
    pub fn total_len(&self) -> usize {
        LQTP_HEADER_LEN + self.n_windows * self.record_stride
    }

    pub fn to_bytes(&self) -> Result<[u8; LQTP_HEADER_LEN], PackError> {
        // try_from (not `as`): a shape that overflows its wire width must ERROR, never
        // wrap silently — a truncated n_windows would slip past the stride cross-check.
        let n_ch = u16::try_from(self.n_channels).map_err(|_| {
            PackError::ShapeMismatch(format!("n_channels {} > u16::MAX", self.n_channels))
        })?;
        let win_len = u32::try_from(self.window_len).map_err(|_| {
            PackError::ShapeMismatch(format!("window_len {} > u32::MAX", self.window_len))
        })?;
        let n_win = u32::try_from(self.n_windows).map_err(|_| {
            PackError::ShapeMismatch(format!("n_windows {} > u32::MAX", self.n_windows))
        })?;
        let stride = u64::try_from(self.record_stride).map_err(|_| {
            PackError::ShapeMismatch(format!("record_stride {} > u64::MAX", self.record_stride))
        })?;
        let mut h = [0u8; LQTP_HEADER_LEN];
        h[0..4].copy_from_slice(LQTP_MAGIC);
        h[4] = LQTP_VERSION;
        h[5] = self.dtype.to_u8();
        h[6..8].copy_from_slice(&n_ch.to_le_bytes());
        h[8..12].copy_from_slice(&win_len.to_le_bytes());
        h[12..16].copy_from_slice(&n_win.to_le_bytes());
        h[16..24].copy_from_slice(&stride.to_le_bytes());
        h[24..56].copy_from_slice(&self.manifest_sha256);
        // [56..64] reserved (zero).
        Ok(h)
    }

    pub fn parse(data: &[u8]) -> Result<PackHeader, PackError> {
        if data.len() < LQTP_HEADER_LEN {
            return Err(PackError::Truncated {
                expected: LQTP_HEADER_LEN,
                actual: data.len(),
            });
        }
        if &data[0..4] != LQTP_MAGIC {
            return Err(PackError::BadMagic);
        }
        if data[4] != LQTP_VERSION {
            return Err(PackError::BadVersion(data[4]));
        }
        let dtype = PackDtype::from_u8(data[5]).ok_or(PackError::BadDtype(data[5]))?;
        let n_channels = u16::from_le_bytes([data[6], data[7]]) as usize;
        let window_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let n_windows = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let record_stride = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]) as usize;
        let mut manifest_sha256 = [0u8; 32];
        manifest_sha256.copy_from_slice(&data[24..56]);
        // Cross-check the stride against dtype/shape so a corrupt header can't drive an
        // out-of-bounds record layout downstream.
        let expected = Self::record_stride(dtype, n_channels, window_len);
        if record_stride != expected {
            return Err(PackError::ShapeMismatch(format!(
                "record_stride {record_stride} != n_ch*4 + n_ch*t*{} = {expected}",
                dtype.mant_size()
            )));
        }
        Ok(PackHeader {
            dtype,
            n_channels,
            window_len,
            n_windows,
            record_stride,
            manifest_sha256,
        })
    }
}

/// Sequential writer for an LQTP1 pack — writes the header, then `n_windows` BFP records in
/// manifest order. `finish` refuses to close a short pack.
pub struct PackWriter {
    file: Option<BufWriter<File>>,
    header: PackHeader,
    written: usize,
    partial_path: std::path::PathBuf,
    final_path: std::path::PathBuf,
    done: bool,
}

impl PackWriter {
    pub fn create(
        path: &Path,
        dtype: PackDtype,
        n_channels: usize,
        window_len: usize,
        n_windows: usize,
        manifest_sha256: [u8; 32],
    ) -> Result<Self, PackError> {
        let header = PackHeader {
            dtype,
            n_channels,
            window_len,
            n_windows,
            record_stride: PackHeader::record_stride(dtype, n_channels, window_len),
            manifest_sha256,
        };
        // Write to a sibling `.partial` and rename into place only on finish(), so a
        // dropped or aborted build never leaves a corrupt pack at the final path. The
        // Drop impl removes the `.partial` if finish() never succeeded.
        let mut partial = path.as_os_str().to_owned();
        partial.push(".partial");
        let partial_path = std::path::PathBuf::from(partial);
        let mut file = BufWriter::new(File::create(&partial_path)?);
        file.write_all(&header.to_bytes()?)?;
        Ok(Self {
            file: Some(file),
            header,
            written: 0,
            partial_path,
            final_path: path.to_path_buf(),
            done: false,
        })
    }

    /// Quantize + append one `[n_ch, window_len]` (row-major) window at the next row.
    pub fn write_window(&mut self, x: &[f32]) -> Result<(), PackError> {
        if self.written >= self.header.n_windows {
            return Err(PackError::ShapeMismatch(format!(
                "wrote more than the declared {} windows",
                self.header.n_windows
            )));
        }
        if x.len() != self.header.n_channels * self.header.window_len {
            return Err(PackError::ShapeMismatch(format!(
                "window len {} != n_ch*window_len ({})",
                x.len(),
                self.header.n_channels * self.header.window_len
            )));
        }
        let (scales, mant) = quantize_window(
            x,
            self.header.n_channels,
            self.header.window_len,
            self.header.dtype,
        );
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| PackError::ShapeMismatch("writer already finished".into()))?;
        for s in &scales {
            file.write_all(&s.to_le_bytes())?;
        }
        file.write_all(&mant)?;
        self.written += 1;
        Ok(())
    }

    /// Flush, fsync, and atomically rename the `.partial` into place. Refuses a short pack
    /// (a truncated build must fail, not ship silently); on any early return the `.partial`
    /// is removed by Drop, and the final path is never created.
    pub fn finish(mut self) -> Result<(), PackError> {
        if self.written != self.header.n_windows {
            return Err(PackError::ShapeMismatch(format!(
                "wrote {} of the declared {} windows",
                self.written, self.header.n_windows
            )));
        }
        let mut file = self
            .file
            .take()
            .ok_or_else(|| PackError::ShapeMismatch("writer already finished".into()))?;
        file.flush()?;
        let f = file
            .into_inner()
            .map_err(|e| PackError::Io(e.into_error()))?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&self.partial_path, &self.final_path)?;
        self.done = true;
        Ok(())
    }
}

impl Drop for PackWriter {
    fn drop(&mut self) {
        // Aborted / short / errored build — remove the `.partial` so it doesn't linger.
        // On success finish() already renamed it away and set `done`, so this no-ops.
        if !self.done {
            let _ = std::fs::remove_file(&self.partial_path);
        }
    }
}

/// Memory-mapped, read-only, shared-across-workers LQTP1 reader. Verifies length + (if a
/// manifest hash is supplied) the fail-closed `manifest_sha256`. `window_raw` returns
/// BORROWED scale + mantissa slices of the map (zero-copy; the PyO3 layer hands these to
/// numpy/torch and dequantizes on the GPU).
pub struct PackReader {
    mmap: memmap2::Mmap,
    header: PackHeader,
}

impl PackReader {
    pub fn open(
        path: &Path,
        expected_manifest_sha256: Option<[u8; 32]>,
    ) -> Result<Self, PackError> {
        let file = File::open(path)?;
        // SAFETY: read-only mapping of a file treated as immutable for this handle.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let header = PackHeader::parse(&mmap)?;
        let total = header.total_len();
        // Exact-length format (fail-closed): too short => Truncated; trailing bytes => a
        // valid LQTP1 pack is exactly header + records, so reject the surplus too.
        if mmap.len() != total {
            return Err(if mmap.len() < total {
                PackError::Truncated {
                    expected: total,
                    actual: mmap.len(),
                }
            } else {
                PackError::ShapeMismatch(format!(
                    "pack has {} trailing bytes past the {total} declared",
                    mmap.len() - total
                ))
            });
        }
        if let Some(exp) = expected_manifest_sha256 {
            if header.manifest_sha256 != exp {
                return Err(PackError::ManifestMismatch);
            }
        }
        Ok(Self { mmap, header })
    }

    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    pub fn n_windows(&self) -> usize {
        self.header.n_windows
    }

    /// Borrowed `(scale_bytes[n_ch*4], mantissa_bytes[n_ch*window_len*mant_size])` of the
    /// map for `row`. No copy, no dequant.
    pub fn window_raw(&self, row: usize) -> Result<(&[u8], &[u8]), PackError> {
        if row >= self.header.n_windows {
            return Err(PackError::ShapeMismatch(format!(
                "row {row} >= n_windows {}",
                self.header.n_windows
            )));
        }
        let start = self.header.record_offset(row);
        let scale_bytes = self.header.n_channels * 4;
        let scales = &self.mmap[start..start + scale_bytes];
        let mant = &self.mmap[start + scale_bytes..start + self.header.record_stride];
        Ok((scales, mant))
    }

    /// Dequantize `row` to `[n_ch, window_len]` f32 (Rust-side convenience; the training
    /// hot path dequantizes on the GPU from the raw borrowed bytes instead).
    pub fn dequantize_window(&self, row: usize) -> Result<Vec<f32>, PackError> {
        let (scale_bytes, mant) = self.window_raw(row)?;
        let scales: Vec<f32> = scale_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(dequantize_window(
            &scales,
            mant,
            self.header.n_channels,
            self.header.window_len,
            self.header.dtype,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_window(n_ch: usize, t: usize) -> Vec<f32> {
        (0..n_ch)
            .flat_map(|c| {
                // Per-channel amplitude spread (µV-to-artifact range) to exercise the
                // per-channel BFP scale; channel 3 is silent (all zero).
                (0..t).map(move |i| {
                    if c == 3 {
                        0.0
                    } else {
                        let amp = 10.0_f32.powi(c as i32 - 1); // 0.1 .. 10^(n-2)
                        amp * ((i as f32 * 0.021 + c as f32).sin())
                    }
                })
            })
            .collect()
    }

    #[test]
    fn bfp_roundtrip_within_bound() {
        let (n_ch, t) = (6usize, 400usize);
        let x = synth_window(n_ch, t);
        for dtype in [PackDtype::Int8, PackDtype::Int16, PackDtype::F32] {
            let (scales, mant) = quantize_window(&x, n_ch, t, dtype);
            assert_eq!(scales.len(), n_ch);
            assert_eq!(mant.len(), n_ch * t * dtype.mant_size());
            let y = dequantize_window(&scales, &mant, n_ch, t, dtype);

            for c in 0..n_ch {
                // Per-channel error bound: half the quantization step (= scale/2), or
                // exact for F32.
                let bound = match dtype {
                    PackDtype::F32 => 0.0,
                    _ => scales[c] * 0.5 + 1e-6, // +eps for round-half-to-even ties
                };
                for i in 0..t {
                    let idx = c * t + i;
                    let d = (x[idx] - y[idx]).abs();
                    assert!(
                        d <= bound,
                        "dtype {dtype:?} ch {c} sample {i}: |Δ|={d} > bound {bound} (scale {})",
                        scales[c]
                    );
                }
            }
        }
    }

    #[test]
    fn silent_channel_is_zero() {
        let (n_ch, t) = (4usize, 100usize);
        let x = synth_window(n_ch, t); // ch 3 is silent
        for dtype in [PackDtype::Int8, PackDtype::Int16] {
            let (scales, mant) = quantize_window(&x, n_ch, t, dtype);
            assert_eq!(scales[3], 0.0, "silent channel scale must be 0 ({dtype:?})");
            let y = dequantize_window(&scales, &mant, n_ch, t, dtype);
            for i in 0..t {
                assert_eq!(y[3 * t + i], 0.0, "silent channel must dequant to 0");
            }
        }
    }

    #[test]
    fn dtype_tags_round_trip() {
        for d in [PackDtype::Int8, PackDtype::Int16, PackDtype::F32] {
            assert_eq!(PackDtype::from_u8(d.to_u8()), Some(d));
        }
        assert_eq!(PackDtype::from_u8(0), None);
        assert_eq!(PackDtype::from_u8(4), None);
        assert_eq!(PackDtype::parse("int16"), Some(PackDtype::Int16));
        assert_eq!(PackDtype::parse("nope"), None);
    }

    fn sha_hex(b: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b);
        format!("{:x}", h.finalize())
    }

    fn multi_window_signal(n_ch: usize, t: usize, n_win: usize) -> Vec<Vec<f32>> {
        (0..n_win)
            .map(|w| {
                (0..n_ch)
                    .flat_map(|c| {
                        (0..t).map(move |i| {
                            ((i + w * 7) as f32 * 0.03 + c as f32).sin() * (c + 1) as f32
                        })
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn writer_reader_roundtrip() {
        let (n_ch, t, n_win) = (5usize, 300usize, 4usize);
        let windows = multi_window_signal(n_ch, t, n_win);
        let hash = [0x5au8; 32];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        for dtype in [PackDtype::Int8, PackDtype::Int16, PackDtype::F32] {
            {
                let mut w = PackWriter::create(tmp.path(), dtype, n_ch, t, n_win, hash).unwrap();
                for win in &windows {
                    w.write_window(win).unwrap();
                }
                w.finish().unwrap();
            }
            let r = PackReader::open(tmp.path(), Some(hash)).unwrap();
            assert_eq!(r.n_windows(), n_win);
            assert_eq!(r.header().dtype, dtype);
            for (row, win) in windows.iter().enumerate() {
                let deq = r.dequantize_window(row).unwrap();
                let (scales, _) = quantize_window(win, n_ch, t, dtype);
                for c in 0..n_ch {
                    let bound = if dtype == PackDtype::F32 {
                        0.0
                    } else {
                        scales[c] * 0.5 + 1e-6
                    };
                    for i in 0..t {
                        assert!(
                            (win[c * t + i] - deq[c * t + i]).abs() <= bound,
                            "{dtype:?} row {row} ch {c} sample {i} out of BFP bound"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn index_hash_mismatch_refused() {
        let (n_ch, t, n_win) = (3usize, 100usize, 2usize);
        let windows = multi_window_signal(n_ch, t, n_win);
        let hash = [0x11u8; 32];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut w =
                PackWriter::create(tmp.path(), PackDtype::Int16, n_ch, t, n_win, hash).unwrap();
            for win in &windows {
                w.write_window(win).unwrap();
            }
            w.finish().unwrap();
        }
        // Matching hash opens; a wrong hash is refused FAIL-CLOSED; None = no check.
        assert!(PackReader::open(tmp.path(), Some(hash)).is_ok());
        assert!(matches!(
            PackReader::open(tmp.path(), Some([0x22u8; 32])),
            Err(PackError::ManifestMismatch)
        ));
        assert!(PackReader::open(tmp.path(), None).is_ok());
        // A short build must refuse to finish (never ship a truncated pack).
        let mut w2 = PackWriter::create(tmp.path(), PackDtype::Int16, n_ch, t, 3, hash).unwrap();
        w2.write_window(&windows[0]).unwrap();
        assert!(w2.finish().is_err(), "short pack must not finish");
    }

    #[test]
    fn pack_layout_golden() {
        // A tiny deterministic int16 pack; frozen sha over the whole file pins the LQTP1
        // wire. Regenerate ONLY with a deliberate version bump.
        let (n_ch, t, n_win) = (2usize, 3usize, 2usize);
        let windows: Vec<Vec<f32>> = vec![
            vec![1.0, -2.0, 3.0, 0.5, 0.0, -0.5],
            vec![0.0, 0.0, 0.0, 10.0, -20.0, 30.0], // ch0 silent
        ];
        let hash = [0u8; 32];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut w =
                PackWriter::create(tmp.path(), PackDtype::Int16, n_ch, t, n_win, hash).unwrap();
            for win in &windows {
                w.write_window(win).unwrap();
            }
            w.finish().unwrap();
        }
        let bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(
            bytes.len(),
            LQTP_HEADER_LEN + n_win * (n_ch * 4 + n_ch * t * 2)
        );
        assert_eq!(
            sha_hex(&bytes),
            "cb35ecf71056e2440236dcdd46358492a467dac3a9d1adb621de519fb87394b5",
            "LQTP1 layout drifted (regen deliberately with a version bump)"
        );
    }

    #[test]
    fn aborted_build_removes_partial() {
        let (n_ch, t) = (2usize, 50usize);
        let hash = [0u8; 32];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("aborted.lqtp");
        let partial = dir.path().join("aborted.lqtp.partial");
        // Drop without finish (an aborted build) → Drop must remove the .partial and
        // never create the final path.
        {
            let mut w = PackWriter::create(&path, PackDtype::Int16, n_ch, t, 3, hash).unwrap();
            w.write_window(&vec![1.0f32; n_ch * t]).unwrap();
        }
        assert!(
            !partial.exists(),
            ".partial must be removed on drop-without-finish"
        );
        assert!(!path.exists(), "final pack must never appear on abort");
        // A successful build renames the partial away — no leftover either.
        {
            let mut w = PackWriter::create(&path, PackDtype::Int16, n_ch, t, 1, hash).unwrap();
            w.write_window(&vec![2.0f32; n_ch * t]).unwrap();
            w.finish().unwrap();
        }
        assert!(path.exists(), "finish must create the final pack");
        assert!(
            !partial.exists(),
            "finish renames partial → final, no leftover"
        );
    }
}
