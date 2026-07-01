//! BrainVision Core file-format reader (`.vhdr` / `.vmrk` / `.eeg`).
//!
//! Three-file convention used by Brain Products' analyzer:
//!   - `.vhdr`  text header, INI-style, references the data + marker
//!     files by relative path. Mandatory.
//!   - `.eeg`   binary signal data (INT_16, INT_32, or IEEE_FLOAT_32),
//!     multiplexed (sample-major) or vectorized (channel-major).
//!   - `.vmrk`  text marker file (annotations). Preserved verbatim
//!     in a sidecar blob; not parsed into samples.
//!
//! Phase 4.1 scope:
//!   - INT_16 + INT_32 binary formats
//!   - Multiplexed data orientation (the universal default)
//!   - Single-file resolution & sample rate read from `[Common Infos]`
//!     / `[Binary Infos]` / `[Channel Infos]`
//!   - Per-channel resolution captured for phys_min / phys_max scaling
//!   - `.vmrk` stored as `markers_raw` sidecar (lossless roundtrip)
//!   - `.vhdr` stored as `vhdr_raw` sidecar
//!
//! Out of scope (later iterations):
//!   - IEEE_FLOAT_32 (needs lossy quantisation strategy)
//!   - Vectorized orientation (uncommon in clinical files)
//!   - Marker-to-EDF-TAL conversion
//!
//! Bible alignment:
//!   - R1 — one format per file under `source/`
//!   - R6 — emit `SignalBundle`, never expose BrainVision-specific
//!     types upstream
//!   - R23 — validate header invariants before allocating sample buffers
//!   - R30 — refuse unsupported sub-formats explicitly with typed errors

use crate::error::{LmlError, LmlResult};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lamquant_abir::{Abir, Channel, Column};

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};
use super::reader::SignalSourceReader;

/// Maximum `.vhdr` size we'll trust (text headers are tiny in practice;
/// 4 MiB is a paranoid ceiling for adversarial inputs).
const MAX_VHDR_BYTES: u64 = 4 * 1024 * 1024;
/// Maximum `.vmrk` size we'll trust before refusing to ingest.
const MAX_VMRK_BYTES: u64 = 64 * 1024 * 1024;

/// Binary data type carried in the `.eeg` file (read from `[Binary Infos]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryFormat {
    Int16,
    Int32,
}

impl BinaryFormat {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "INT_16" => Some(Self::Int16),
            "INT_32" => Some(Self::Int32),
            _ => None,
        }
    }
    fn bytes_per_sample(self) -> usize {
        match self {
            Self::Int16 => 2,
            Self::Int32 => 4,
        }
    }
}

/// Sample-major vs channel-major data layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Multiplexed, // sample-major: ch0,ch1,...,chN,ch0,ch1,...
    Vectorized,  // channel-major: all of ch0, then all of ch1
}

impl Orientation {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim() {
            "MULTIPLEXED" => Some(Self::Multiplexed),
            "VECTORIZED" => Some(Self::Vectorized),
            _ => None,
        }
    }
}

/// Parsed `.vhdr` header. Built by `parse_vhdr`; everything in here is
/// derived from the text-format file before any `.eeg` byte is touched.
#[derive(Debug)]
struct VhdrHeader {
    data_file: String,   // relative path to .eeg
    marker_file: String, // relative path to .vmrk (may be empty)
    n_channels: usize,
    /// Sample interval in MICROSECONDS per sample (BrainVision spec).
    sample_interval_us: f64,
    binary_format: BinaryFormat,
    orientation: Orientation,
    /// Per-channel resolution (multiplier from raw int → physical unit).
    /// `Ch1=Fp1,REF,0.5,µV` → resolution = 0.5.
    channels: Vec<BrainVisionChannel>,
}

#[derive(Debug, Clone)]
struct BrainVisionChannel {
    name: String,
    /// Reference name (often empty); preserved for diagnostics.
    _reference: String,
    resolution: f64,
    unit: String,
}

/// Parse a `.vhdr` INI-style header. Hand-rolled (no INI crate dep) so
/// the parse contract stays auditable.
fn parse_vhdr(raw: &str) -> LmlResult<VhdrHeader> {
    let mut data_file = String::new();
    let mut marker_file = String::new();
    let mut n_channels: usize = 0;
    let mut sample_interval_us: f64 = 0.0;
    let mut data_format: String = String::new();
    let mut binary_format: Option<BinaryFormat> = None;
    let mut orientation: Option<Orientation> = None;
    let mut section = String::new();
    let mut channel_lines: Vec<(usize, String)> = Vec::new();

    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with(';') {
            continue;
        }
        if t.starts_with('[') && t.ends_with(']') {
            section = t[1..t.len() - 1].to_string();
            continue;
        }
        let (k, v) = match t.split_once('=') {
            Some(p) => (p.0.trim(), p.1.trim()),
            None => continue,
        };
        match section.as_str() {
            "Common Infos" => match k {
                "DataFile" => data_file = v.to_string(),
                "MarkerFile" => marker_file = v.to_string(),
                "DataFormat" => data_format = v.to_string(),
                "DataOrientation" => {
                    orientation = Some(Orientation::from_str(v).ok_or_else(|| {
                        LmlError::InvalidHeader(format!(
                            "BrainVision .vhdr: unsupported DataOrientation '{v}' \
                             (this reader supports MULTIPLEXED, VECTORIZED)"
                        ))
                    })?);
                }
                "NumberOfChannels" => {
                    n_channels = v.parse().map_err(|e| {
                        LmlError::InvalidHeader(format!(
                            "BrainVision .vhdr: NumberOfChannels '{v}' not an integer ({e})"
                        ))
                    })?;
                }
                "SamplingInterval" => {
                    sample_interval_us = v.parse().map_err(|e| {
                        LmlError::InvalidHeader(format!(
                            "BrainVision .vhdr: SamplingInterval '{v}' not a number ({e})"
                        ))
                    })?;
                }
                _ => {}
            },
            "Binary Infos" => {
                if k == "BinaryFormat" {
                    binary_format = Some(BinaryFormat::from_str(v).ok_or_else(|| {
                        LmlError::InvalidHeader(format!(
                            "BrainVision .vhdr: BinaryFormat '{v}' not supported \
                             (this reader supports INT_16, INT_32; IEEE_FLOAT_32 \
                             requires lossy quantisation — out of scope here)"
                        ))
                    })?);
                }
            }
            "Channel Infos" => {
                // Ch<n>=name,reference,resolution,unit  (n is 1-based)
                if let Some(num_str) = k.strip_prefix("Ch") {
                    if let Ok(idx) = num_str.parse::<usize>() {
                        channel_lines.push((idx, v.to_string()));
                    }
                }
            }
            _ => {}
        }
    }

    if !data_format.eq_ignore_ascii_case("BINARY") {
        return Err(LmlError::InvalidHeader(format!(
            "BrainVision .vhdr: DataFormat '{data_format}' not supported \
             (only BINARY is implemented; ASCII export is rare and out of scope)"
        )));
    }
    if data_file.is_empty() {
        return Err(LmlError::InvalidHeader(
            "BrainVision .vhdr: DataFile= missing".into(),
        ));
    }
    if n_channels == 0 {
        return Err(LmlError::InvalidHeader(
            "BrainVision .vhdr: NumberOfChannels= missing or zero".into(),
        ));
    }
    if sample_interval_us <= 0.0 || !sample_interval_us.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "BrainVision .vhdr: SamplingInterval '{sample_interval_us}' must be > 0 and finite"
        )));
    }
    let binary_format = binary_format.ok_or_else(|| {
        LmlError::InvalidHeader("BrainVision .vhdr: BinaryFormat= missing".into())
    })?;
    let orientation = orientation.unwrap_or(Orientation::Multiplexed);

    // Sort channel lines by 1-based index, then parse each.
    channel_lines.sort_by_key(|(i, _)| *i);
    let mut channels = Vec::with_capacity(n_channels);
    for (expected, (idx, line)) in (1..=n_channels).zip(channel_lines.iter()) {
        if *idx != expected {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .vhdr: channel sequence broken — \
                 expected Ch{expected}=, got Ch{idx}="
            )));
        }
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() < 4 {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .vhdr: Ch{idx}={line} — \
                 expected 4 comma-separated fields (name, reference, resolution, unit)"
            )));
        }
        let resolution: f64 = parts[2].trim().parse().map_err(|e| {
            LmlError::InvalidHeader(format!(
                "BrainVision .vhdr: Ch{idx} resolution '{}' not a number ({e})",
                parts[2]
            ))
        })?;
        channels.push(BrainVisionChannel {
            name: parts[0].trim().to_string(),
            _reference: parts[1].trim().to_string(),
            resolution,
            unit: parts[3].trim().to_string(),
        });
    }
    if channels.len() != n_channels {
        return Err(LmlError::InvalidHeader(format!(
            "BrainVision .vhdr: NumberOfChannels={n_channels} but \
             Channel Infos section listed {} channels",
            channels.len()
        )));
    }
    Ok(VhdrHeader {
        data_file,
        marker_file,
        n_channels,
        sample_interval_us,
        binary_format,
        orientation,
        channels,
    })
}

/// `BrainVisionReader` — file-backed `SignalSourceReader` impl.
pub struct BrainVisionReader {
    vhdr_path: PathBuf,
}

impl BrainVisionReader {
    pub fn new<P: Into<PathBuf>>(vhdr_path: P) -> Self {
        Self {
            vhdr_path: vhdr_path.into(),
        }
    }
}

impl SignalSourceReader for BrainVisionReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        // 1. Read + parse the .vhdr.
        let vhdr_meta = std::fs::metadata(&self.vhdr_path).map_err(LmlError::Io)?;
        if vhdr_meta.len() > MAX_VHDR_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .vhdr {} too large ({} bytes, max {})",
                self.vhdr_path.display(),
                vhdr_meta.len(),
                MAX_VHDR_BYTES
            )));
        }
        let vhdr_raw = std::fs::read(&self.vhdr_path).map_err(LmlError::Io)?;
        let vhdr_text = std::str::from_utf8(&vhdr_raw).map_err(|e| {
            LmlError::InvalidHeader(format!("BrainVision .vhdr: invalid UTF-8 ({e})"))
        })?;
        let hdr = parse_vhdr(vhdr_text)?;

        // 2. Resolve sibling paths (relative to the .vhdr's directory).
        let dir = self.vhdr_path.parent().unwrap_or(Path::new("."));
        let eeg_path = dir.join(&hdr.data_file);
        let vmrk_path = if hdr.marker_file.is_empty() {
            None
        } else {
            Some(dir.join(&hdr.marker_file))
        };

        // 3. Read the .eeg binary payload.
        let eeg_meta = std::fs::metadata(&eeg_path).map_err(LmlError::Io)?;
        let eeg_len = eeg_meta.len() as usize;
        let bps = hdr.binary_format.bytes_per_sample();
        if eeg_len % (hdr.n_channels * bps) != 0 {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .eeg {} length {} is not a multiple of n_channels * bps ({} * {})",
                eeg_path.display(),
                eeg_len,
                hdr.n_channels,
                bps
            )));
        }
        let n_samples = eeg_len / (hdr.n_channels * bps);
        let raw = std::fs::read(&eeg_path).map_err(LmlError::Io)?;

        // 4. Decode into `[n_channels][n_samples] i64`.
        let mut signal: Vec<Vec<i64>> = (0..hdr.n_channels)
            .map(|_| Vec::with_capacity(n_samples))
            .collect();
        match (hdr.binary_format, hdr.orientation) {
            (BinaryFormat::Int16, Orientation::Multiplexed) => {
                for sample_idx in 0..n_samples {
                    let base = sample_idx * hdr.n_channels * 2;
                    for ch in 0..hdr.n_channels {
                        let off = base + ch * 2;
                        let v = i16::from_le_bytes([raw[off], raw[off + 1]]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (BinaryFormat::Int16, Orientation::Vectorized) => {
                for ch in 0..hdr.n_channels {
                    let ch_base = ch * n_samples * 2;
                    for s in 0..n_samples {
                        let off = ch_base + s * 2;
                        let v = i16::from_le_bytes([raw[off], raw[off + 1]]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (BinaryFormat::Int32, Orientation::Multiplexed) => {
                for sample_idx in 0..n_samples {
                    let base = sample_idx * hdr.n_channels * 4;
                    for ch in 0..hdr.n_channels {
                        let off = base + ch * 4;
                        let v = i32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (BinaryFormat::Int32, Orientation::Vectorized) => {
                for ch in 0..hdr.n_channels {
                    let ch_base = ch * n_samples * 4;
                    for s in 0..n_samples {
                        let off = ch_base + s * 4;
                        let v = i32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
        }

        // 5. Build phys_min / phys_max from the int range * per-channel
        //    resolution. Approximate but stable across files.
        let (raw_min, raw_max): (f64, f64) = match hdr.binary_format {
            BinaryFormat::Int16 => (i16::MIN as f64, i16::MAX as f64),
            BinaryFormat::Int32 => (i32::MIN as f64, i32::MAX as f64),
        };
        let phys_min: Vec<f64> = hdr
            .channels
            .iter()
            .map(|c| raw_min * c.resolution)
            .collect();
        let phys_max: Vec<f64> = hdr
            .channels
            .iter()
            .map(|c| raw_max * c.resolution)
            .collect();
        let channels: Vec<String> = hdr.channels.iter().map(|c| c.name.clone()).collect();
        let phys_dim = hdr
            .channels
            .first()
            .map(|c| c.unit.clone())
            .unwrap_or_else(|| "uV".to_string());

        let sample_rate = 1_000_000.0 / hdr.sample_interval_us;
        let duration_s = if sample_rate > 0.0 {
            n_samples as f64 / sample_rate
        } else {
            0.0
        };

        // 6. Sidecars: full .vhdr + optional .vmrk + the .eeg payload
        //    plus filename anchors so the encoder can write byte-exact
        //    preservation copies with the original filenames (the .vhdr
        //    internally references `DataFile=` and `MarkerFile=` by
        //    name; if we rename, the .vhdr text drifts from its
        //    siblings and the lossless roundtrip breaks).
        let mut sidecar = Vec::new();
        sidecar.push(SidecarBlob {
            key: "vhdr_raw".to_string(),
            bytes: vhdr_raw,
            aux: None,
        });
        sidecar.push(SidecarBlob {
            key: "eeg_raw".to_string(),
            bytes: raw.clone(),
            aux: None,
        });
        // Filename anchor for `.eeg`. The encoder uses this exact name
        // when emitting the preservation copy so the byte-equal .vhdr
        // still points at a real file post-extract.
        sidecar.push(SidecarBlob {
            key: "eeg_filename".to_string(),
            bytes: hdr.data_file.as_bytes().to_vec(),
            aux: None,
        });
        if let Some(vmrk_path) = vmrk_path {
            if vmrk_path.exists() {
                let vmrk_meta = std::fs::metadata(&vmrk_path).map_err(LmlError::Io)?;
                if vmrk_meta.len() > MAX_VMRK_BYTES {
                    return Err(LmlError::InvalidHeader(format!(
                        "BrainVision .vmrk {} too large ({} bytes, max {})",
                        vmrk_path.display(),
                        vmrk_meta.len(),
                        MAX_VMRK_BYTES
                    )));
                }
                let vmrk_bytes = std::fs::read(&vmrk_path).map_err(LmlError::Io)?;
                sidecar.push(SidecarBlob {
                    key: "vmrk_raw".to_string(),
                    bytes: vmrk_bytes,
                    aux: None,
                });
                sidecar.push(SidecarBlob {
                    key: "vmrk_filename".to_string(),
                    bytes: hdr.marker_file.as_bytes().to_vec(),
                    aux: None,
                });
            }
        }

        let bundle = SignalBundle {
            signal,
            sample_rate,
            channels,
            phys_min,
            phys_max,
            duration_s,
            metadata: SourceMetadata {
                source_file: self.vhdr_path.display().to_string(),
                format: "BRAINVISION".to_string(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim,
            },
            sidecar,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// ADR 0069 L7: specialize to the `.eeg` file's native `BinaryFormat`
    /// instead of the trait default's all-`I64` `Abir` (the memory win —
    /// `Column::I16`/`I32` instead of always widening to `i64`).
    ///
    /// Independent of `read_bundle`: re-parses the (tiny, KB-scale)
    /// `.vhdr` text header, then decodes the `.eeg` binary payload
    /// directly into native-width buffers via `from_le_bytes` element-
    /// wise — NOT `cast_slice`/transmute, which would assume the host is
    /// little-endian and isn't a portable reinterpretation of a `&[u8]`
    /// buffer. This never materializes the intermediate `Vec<Vec<i64>>`
    /// `read_bundle` builds; the decode loops below mirror
    /// `read_bundle`'s index math exactly (same `base`/`off` formulas per
    /// `(BinaryFormat, Orientation)` arm), so the two paths stay
    /// byte-exact by construction — locked by this module's
    /// `lower_to_abir_*_matches_read_bundle` tests.
    ///
    /// `resolution` (the `.vhdr` per-channel int→physical-unit
    /// multiplier) is applied ONLY to `phys_min`/`phys_max`, never to the
    /// sample lane itself — the raw ints are what the codec/decoder need
    /// bit-exact.
    fn lower_to_abir(&mut self) -> LmlResult<Abir> {
        let vhdr_meta = std::fs::metadata(&self.vhdr_path).map_err(LmlError::Io)?;
        if vhdr_meta.len() > MAX_VHDR_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .vhdr {} too large ({} bytes, max {})",
                self.vhdr_path.display(),
                vhdr_meta.len(),
                MAX_VHDR_BYTES
            )));
        }
        let vhdr_raw = std::fs::read(&self.vhdr_path).map_err(LmlError::Io)?;
        let vhdr_text = std::str::from_utf8(&vhdr_raw).map_err(|e| {
            LmlError::InvalidHeader(format!("BrainVision .vhdr: invalid UTF-8 ({e})"))
        })?;
        let hdr = parse_vhdr(vhdr_text)?;

        let dir = self.vhdr_path.parent().unwrap_or(Path::new("."));
        let eeg_path = dir.join(&hdr.data_file);
        let eeg_meta = std::fs::metadata(&eeg_path).map_err(LmlError::Io)?;
        let eeg_len = eeg_meta.len() as usize;
        let bps = hdr.binary_format.bytes_per_sample();
        if eeg_len % (hdr.n_channels * bps) != 0 {
            return Err(LmlError::InvalidHeader(format!(
                "BrainVision .eeg {} length {} is not a multiple of n_channels * bps ({} * {})",
                eeg_path.display(),
                eeg_len,
                hdr.n_channels,
                bps
            )));
        }
        let n_samples = eeg_len / (hdr.n_channels * bps);
        let raw = std::fs::read(&eeg_path).map_err(LmlError::Io)?;

        let (raw_min, raw_max): (f64, f64) = match hdr.binary_format {
            BinaryFormat::Int16 => (i16::MIN as f64, i16::MAX as f64),
            BinaryFormat::Int32 => (i32::MIN as f64, i32::MAX as f64),
        };

        let channels: Vec<Channel> = match hdr.binary_format {
            BinaryFormat::Int16 => {
                let mut cols: Vec<Vec<i16>> = (0..hdr.n_channels)
                    .map(|_| Vec::with_capacity(n_samples))
                    .collect();
                match hdr.orientation {
                    Orientation::Multiplexed => {
                        for sample_idx in 0..n_samples {
                            let base = sample_idx * hdr.n_channels * 2;
                            for ch in 0..hdr.n_channels {
                                let off = base + ch * 2;
                                cols[ch].push(i16::from_le_bytes([raw[off], raw[off + 1]]));
                            }
                        }
                    }
                    Orientation::Vectorized => {
                        for ch in 0..hdr.n_channels {
                            let ch_base = ch * n_samples * 2;
                            for s in 0..n_samples {
                                let off = ch_base + s * 2;
                                cols[ch].push(i16::from_le_bytes([raw[off], raw[off + 1]]));
                            }
                        }
                    }
                }
                cols.into_iter()
                    .enumerate()
                    .map(|(j, col)| Channel {
                        label: Arc::from(hdr.channels[j].name.as_str()),
                        data: Column::I16(Arc::from(col)),
                        phys_min: raw_min * hdr.channels[j].resolution,
                        phys_max: raw_max * hdr.channels[j].resolution,
                    })
                    .collect()
            }
            BinaryFormat::Int32 => {
                let mut cols: Vec<Vec<i32>> = (0..hdr.n_channels)
                    .map(|_| Vec::with_capacity(n_samples))
                    .collect();
                match hdr.orientation {
                    Orientation::Multiplexed => {
                        for sample_idx in 0..n_samples {
                            let base = sample_idx * hdr.n_channels * 4;
                            for ch in 0..hdr.n_channels {
                                let off = base + ch * 4;
                                cols[ch].push(i32::from_le_bytes([
                                    raw[off],
                                    raw[off + 1],
                                    raw[off + 2],
                                    raw[off + 3],
                                ]));
                            }
                        }
                    }
                    Orientation::Vectorized => {
                        for ch in 0..hdr.n_channels {
                            let ch_base = ch * n_samples * 4;
                            for s in 0..n_samples {
                                let off = ch_base + s * 4;
                                cols[ch].push(i32::from_le_bytes([
                                    raw[off],
                                    raw[off + 1],
                                    raw[off + 2],
                                    raw[off + 3],
                                ]));
                            }
                        }
                    }
                }
                cols.into_iter()
                    .enumerate()
                    .map(|(j, col)| Channel {
                        label: Arc::from(hdr.channels[j].name.as_str()),
                        data: Column::I32(Arc::from(col)),
                        phys_min: raw_min * hdr.channels[j].resolution,
                        phys_max: raw_max * hdr.channels[j].resolution,
                    })
                    .collect()
            }
        };

        let sample_rate = 1_000_000.0 / hdr.sample_interval_us;
        Ok(Abir::from_parts(channels, sample_rate, n_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn synth_vhdr_int16_multiplexed(n_ch: usize, sample_interval_us: f64) -> String {
        let mut s = String::new();
        s.push_str("Brain Vision Data Exchange Header File Version 1.0\n");
        s.push_str("[Common Infos]\n");
        s.push_str("DataFile=test.eeg\n");
        s.push_str("MarkerFile=test.vmrk\n");
        s.push_str("DataFormat=BINARY\n");
        s.push_str("DataOrientation=MULTIPLEXED\n");
        s.push_str(&format!("NumberOfChannels={n_ch}\n"));
        s.push_str(&format!("SamplingInterval={sample_interval_us}\n"));
        s.push_str("\n[Binary Infos]\n");
        s.push_str("BinaryFormat=INT_16\n");
        s.push_str("\n[Channel Infos]\n");
        for i in 1..=n_ch {
            s.push_str(&format!("Ch{i}=Ch{i}_name,REF,0.5,uV\n"));
        }
        s
    }

    #[test]
    fn parse_vhdr_extracts_fields() {
        let raw = synth_vhdr_int16_multiplexed(3, 4000.0);
        let h = parse_vhdr(&raw).unwrap();
        assert_eq!(h.n_channels, 3);
        assert_eq!(h.sample_interval_us, 4000.0);
        assert_eq!(h.binary_format, BinaryFormat::Int16);
        assert_eq!(h.orientation, Orientation::Multiplexed);
        assert_eq!(h.channels[0].name, "Ch1_name");
        assert_eq!(h.channels[0].resolution, 0.5);
    }

    #[test]
    fn parse_vhdr_rejects_missing_data_file() {
        let raw = "[Common Infos]\nDataFormat=BINARY\nNumberOfChannels=1\nSamplingInterval=1000\n[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\nCh1=a,REF,1,uV";
        match parse_vhdr(raw) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("DataFile=")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn parse_vhdr_rejects_unsupported_binary_format() {
        let raw = "[Common Infos]\nDataFile=x.eeg\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=1\nSamplingInterval=1000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=a,REF,1,uV";
        match parse_vhdr(raw) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("IEEE_FLOAT_32")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn parse_vhdr_rejects_channel_gap() {
        let raw = "[Common Infos]\nDataFile=x.eeg\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=1000\n[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\nCh1=a,REF,1,uV\nCh3=c,REF,1,uV";
        match parse_vhdr(raw) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("channel sequence broken")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_bundle_int16_multiplexed_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 3usize;
        let n_samples = 100usize;
        // Write synthetic .eeg (multiplexed int16 LE).
        let mut eeg_bytes: Vec<u8> = Vec::with_capacity(n_ch * n_samples * 2);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                let v = (s as i16) * (ch as i16 + 1);
                eeg_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join("test.eeg"), &eeg_bytes).unwrap();
        // .vhdr referencing test.eeg + test.vmrk.
        let vhdr = synth_vhdr_int16_multiplexed(n_ch, 4000.0);
        std::fs::write(tmp.path().join("test.vhdr"), vhdr.as_bytes()).unwrap();
        // .vmrk a few stub bytes.
        std::fs::write(tmp.path().join("test.vmrk"), b"; vmrk stub").unwrap();
        let mut reader = BrainVisionReader::new(tmp.path().join("test.vhdr"));
        let b = reader.read_bundle().unwrap();
        assert_eq!(b.signal.len(), n_ch);
        assert_eq!(b.signal[0].len(), n_samples);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                assert_eq!(b.signal[ch][s], (s as i64) * (ch as i64 + 1));
            }
        }
        assert!((b.sample_rate - 250.0).abs() < 1e-9, "sr={}", b.sample_rate);
        assert_eq!(b.metadata.format, "BRAINVISION");
        // Sidecars present.
        assert!(b.sidecar_first("vhdr_raw").is_some());
        assert!(b.sidecar_first("vmrk_raw").is_some());
    }

    #[test]
    fn read_bundle_handles_missing_marker_file() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 1usize;
        let n_samples = 8usize;
        let mut eeg_bytes: Vec<u8> = Vec::new();
        for s in 0..n_samples {
            for _ in 0..n_ch {
                eeg_bytes.extend_from_slice(&(s as i16).to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join("a.eeg"), &eeg_bytes).unwrap();
        // No MarkerFile= line, no .vmrk file present.
        let mut vhdr = String::new();
        vhdr.push_str("[Common Infos]\n");
        vhdr.push_str("DataFile=a.eeg\n");
        vhdr.push_str("DataFormat=BINARY\n");
        vhdr.push_str("DataOrientation=MULTIPLEXED\n");
        vhdr.push_str("NumberOfChannels=1\n");
        vhdr.push_str("SamplingInterval=4000\n");
        vhdr.push_str("[Binary Infos]\nBinaryFormat=INT_16\n");
        vhdr.push_str("[Channel Infos]\nCh1=A,REF,0.5,uV\n");
        let p = tmp.path().join("a.vhdr");
        std::fs::write(&p, vhdr.as_bytes()).unwrap();
        let mut reader = BrainVisionReader::new(&p);
        let b = reader.read_bundle().unwrap();
        assert!(b.sidecar_first("vmrk_raw").is_none());
    }

    #[test]
    fn read_bundle_rejects_eeg_length_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        // .vhdr expects 2 channels INT_16, but .eeg has an odd byte count.
        let vhdr = synth_vhdr_int16_multiplexed(2, 4000.0);
        let p = tmp.path().join("test.vhdr");
        std::fs::write(&p, vhdr.as_bytes()).unwrap();
        std::fs::write(tmp.path().join("test.eeg"), b"abc").unwrap(); // 3 bytes
        let mut w = std::fs::File::create(tmp.path().join("test.vmrk")).unwrap();
        let _ = w.write_all(b";");
        let mut reader = BrainVisionReader::new(&p);
        match reader.read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("multiple of")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    // ─── ADR 0069 L7 gate: lower_to_abir byte-exactness ─────────────

    fn synth_vhdr_int32_multiplexed(n_ch: usize, sample_interval_us: f64) -> String {
        let mut s = synth_vhdr_int16_multiplexed(n_ch, sample_interval_us);
        s = s.replace("BinaryFormat=INT_16", "BinaryFormat=INT_32");
        s
    }

    #[test]
    fn lower_to_abir_int16_multiplexed_matches_read_bundle_i64() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 3usize;
        let n_samples = 100usize;
        let mut eeg_bytes: Vec<u8> = Vec::with_capacity(n_ch * n_samples * 2);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                let v = (s as i16).wrapping_mul(ch as i16 + 1).wrapping_sub(50);
                eeg_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join("test.eeg"), &eeg_bytes).unwrap();
        let vhdr = synth_vhdr_int16_multiplexed(n_ch, 4000.0);
        let p = tmp.path().join("test.vhdr");
        std::fs::write(&p, vhdr.as_bytes()).unwrap();
        std::fs::write(tmp.path().join("test.vmrk"), b"; vmrk stub").unwrap();

        let bundle = BrainVisionReader::new(&p).read_bundle().unwrap();
        let abir = BrainVisionReader::new(&p).lower_to_abir().unwrap();

        assert_eq!(abir.n_channels(), n_ch);
        assert_eq!(abir.n_samples, n_samples);
        for (j, ch) in abir.channels.iter().enumerate() {
            assert!(
                matches!(ch.data, Column::I16(_)),
                "INT_16 must specialize to Column::I16"
            );
            let widened = ch.data.window_i64(0, abir.n_samples);
            assert_eq!(
                widened.as_ref(),
                bundle.signal[j].as_slice(),
                "channel {j} mismatch"
            );
        }
    }

    #[test]
    fn lower_to_abir_int16_vectorized_matches_read_bundle_i64() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 2usize;
        let n_samples = 64usize;
        let mut eeg_bytes: Vec<u8> = Vec::with_capacity(n_ch * n_samples * 2);
        for ch in 0..n_ch {
            for s in 0..n_samples {
                let v = (s as i16).wrapping_add((ch as i16) * 1000);
                eeg_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join("v.eeg"), &eeg_bytes).unwrap();
        let mut vhdr = synth_vhdr_int16_multiplexed(n_ch, 2000.0);
        vhdr = vhdr.replace("DataOrientation=MULTIPLEXED", "DataOrientation=VECTORIZED");
        vhdr = vhdr.replace("DataFile=test.eeg", "DataFile=v.eeg");
        let p = tmp.path().join("v.vhdr");
        std::fs::write(&p, vhdr.as_bytes()).unwrap();

        let bundle = BrainVisionReader::new(&p).read_bundle().unwrap();
        let abir = BrainVisionReader::new(&p).lower_to_abir().unwrap();
        for (j, ch) in abir.channels.iter().enumerate() {
            assert!(matches!(ch.data, Column::I16(_)));
            let widened = ch.data.window_i64(0, abir.n_samples);
            assert_eq!(
                widened.as_ref(),
                bundle.signal[j].as_slice(),
                "channel {j} mismatch"
            );
        }
    }

    #[test]
    fn lower_to_abir_int32_multiplexed_matches_read_bundle_i64() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 2usize;
        let n_samples = 80usize;
        let mut eeg_bytes: Vec<u8> = Vec::with_capacity(n_ch * n_samples * 4);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                let v = (s as i32).wrapping_mul(100_000).wrapping_add(ch as i32) - 40_000;
                eeg_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join("i32.eeg"), &eeg_bytes).unwrap();
        let mut vhdr = synth_vhdr_int32_multiplexed(n_ch, 4000.0);
        vhdr = vhdr.replace("DataFile=test.eeg", "DataFile=i32.eeg");
        let p = tmp.path().join("i32.vhdr");
        std::fs::write(&p, vhdr.as_bytes()).unwrap();

        let bundle = BrainVisionReader::new(&p).read_bundle().unwrap();
        let abir = BrainVisionReader::new(&p).lower_to_abir().unwrap();
        for (j, ch) in abir.channels.iter().enumerate() {
            assert!(
                matches!(ch.data, Column::I32(_)),
                "INT_32 must specialize to Column::I32"
            );
            let widened = ch.data.window_i64(0, abir.n_samples);
            assert_eq!(
                widened.as_ref(),
                bundle.signal[j].as_slice(),
                "channel {j} mismatch"
            );
        }
    }
}
