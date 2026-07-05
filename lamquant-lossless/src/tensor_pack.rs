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
pub fn dequantize_window(scales: &[f32], mant: &[u8], n_ch: usize, t: usize, dtype: PackDtype) -> Vec<f32> {
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
                write!(f, "pack truncated: expected >= {expected} bytes, got {actual}")
            }
            PackError::ManifestMismatch => {
                write!(f, "pack manifest hash != loaded manifest (rebuild the pack)")
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
}
