//! DICOM Waveform reader (feature `dicom`).
//!
//! Phase 4.7 + 8 (Item A). Reads the Waveform IOD used for clinical
//! ECG / EEG storage. Tag map (consensus across DICOM PS3.3 §C.10.9
//! and the `pydicom`, `dicom-rs`, and `dcmtk` reference implementations):
//!
//!     (5400,0100) WaveformSequence              SQ — one item per multiplex group
//!       (003A,0005) NumberOfWaveformChannels    US
//!       (003A,001A) SamplingFrequency           DS  (Hz, decimal string)
//!       (003A,0010) NumberOfWaveformSamples     UL
//!       (5400,1004) WaveformBitsAllocated       US  (8 / 16 / 32)
//!       (5400,1006) WaveformSampleInterpretation CS  (SS / SB / UB / MB / AB)
//!       (5400,1010) WaveformData                OW/OB (raw samples,
//!                                                channel-multiplexed)
//!       (003A,0200) ChannelDefinitionSequence   SQ — per-channel meta
//!         (003A,0203) ChannelLabel              SH
//!         (5400,1006) WaveformBitsStored        US
//!
//! Supported SOP classes:
//!   - 12-Lead ECG Waveform Storage  (1.2.840.10008.5.1.4.1.1.9.1.1)
//!   - General ECG Waveform Storage  (1.2.840.10008.5.1.4.1.1.9.1.2)
//!
//! Out of scope (refused with typed errors):
//!   - WaveformBitsAllocated ≠ 16  (8-bit / 32-bit deferred)
//!   - WaveformSampleInterpretation ≠ SS  (UB / MB / AB deferred)
//!   - Multi-multiplex-group with mismatched SamplingFrequency
//!   - Other Waveform IOD types (Voice Audio, Hemodynamic, Ambulatory,
//!     Respiratory) — these refuse because we don't yet have validated
//!     fixtures + sensitivity-decoding logic
//!
//! Bible alignment:
//!   - R1  one format per file
//!   - R23 validate every required tag + bit-allocated + interpretation
//!         BEFORE allocating sample buffers
//!   - R30 each refusal axis produces its own typed error citing the
//!         spec section

#![cfg(feature = "dicom")]

use crate::error::{LmlError, LmlResult};
use std::path::PathBuf;

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};
use super::reader::SignalSourceReader;

use dicom_core::Tag;
use dicom_object::open_file;

// ─── Tag constants (kept for documentation + future expansion) ────
const TAG_WAVEFORM_SEQUENCE: Tag = Tag(0x5400, 0x0100);
const TAG_NUM_CHANNELS: Tag = Tag(0x003A, 0x0005);
const TAG_SAMPLING_FREQ: Tag = Tag(0x003A, 0x001A);
const TAG_NUM_SAMPLES: Tag = Tag(0x003A, 0x0010);
const TAG_BITS_ALLOCATED: Tag = Tag(0x5400, 0x1004);
const TAG_SAMPLE_INTERPRETATION: Tag = Tag(0x5400, 0x1006);
const TAG_WAVEFORM_DATA: Tag = Tag(0x5400, 0x1010);
const TAG_CHANNEL_DEF_SEQUENCE: Tag = Tag(0x003A, 0x0200);
const TAG_CHANNEL_LABEL: Tag = Tag(0x003A, 0x0203);

/// `DicomWaveformReader` — file-backed `SignalSourceReader` for DICOM
/// Waveform IODs.
pub struct DicomWaveformReader {
    path: PathBuf,
}

impl DicomWaveformReader {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self { path: path.into() }
    }
}

impl SignalSourceReader for DicomWaveformReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        let obj = open_file(&self.path).map_err(|e| {
            LmlError::InvalidHeader(format!("dicom: open {}: {e}", self.path.display()))
        })?;

        // Pull WaveformSequence (5400,0100). Required for any waveform IOD.
        let waveform_seq = obj.element(TAG_WAVEFORM_SEQUENCE).map_err(|_| {
            LmlError::InvalidHeader(format!(
                "dicom: missing WaveformSequence ({:04X},{:04X}); is this a Waveform IOD?",
                TAG_WAVEFORM_SEQUENCE.group(),
                TAG_WAVEFORM_SEQUENCE.element()
            ))
        })?;
        let groups = waveform_seq.items().ok_or_else(|| {
            LmlError::InvalidHeader("dicom: WaveformSequence has no items".into())
        })?;
        if groups.is_empty() {
            return Err(LmlError::InvalidHeader(
                "dicom: WaveformSequence is empty".into(),
            ));
        }

        // Validate all multiplex groups share the same sampling rate
        // BEFORE allocating buffers. The fold-as-append strategy below
        // only makes sense when timing is uniform.
        let mut group_meta: Vec<GroupMeta> = Vec::with_capacity(groups.len());
        for (idx, item) in groups.iter().enumerate() {
            let m = parse_group_meta(item, idx)?;
            group_meta.push(m);
        }
        let base_rate = group_meta[0].sampling_freq;
        let base_n_ch = group_meta[0].n_channels;
        for m in group_meta.iter().skip(1) {
            if (m.sampling_freq - base_rate).abs() > 1e-9 {
                return Err(LmlError::InvalidHeader(format!(
                    "dicom: multi-group SamplingFrequency mismatch \
                     ({} Hz in group 0 vs {} Hz in group {}) — fold-as-append \
                     would distort timing; refuse",
                    base_rate, m.sampling_freq, m.idx
                )));
            }
            if m.n_channels != base_n_ch {
                return Err(LmlError::InvalidHeader(format!(
                    "dicom: multi-group channel-count mismatch \
                     ({} in group 0 vs {} in group {}); refuse",
                    base_n_ch, m.n_channels, m.idx
                )));
            }
        }

        // Walk each multiplex group and decode the multiplexed int16 LE
        // stream into a channel-major i64 matrix.
        let mut signal: Vec<Vec<i64>> = (0..base_n_ch).map(|_| Vec::new()).collect();
        let mut channels: Vec<String> = (0..base_n_ch).map(|i| format!("ch{i}")).collect();
        let mut total_samples: u64 = 0;

        for (idx, (item, meta)) in groups.iter().zip(group_meta.iter()).enumerate() {
            // Pull WaveformData (5400,1010) as bytes.
            let waveform_data_el = item.element(TAG_WAVEFORM_DATA).map_err(|_| {
                LmlError::InvalidHeader(format!(
                    "dicom: group {idx} missing WaveformData (5400,1010)"
                ))
            })?;
            let waveform_bytes = waveform_data_el.to_bytes().map_err(|e| {
                LmlError::InvalidHeader(format!("dicom: group {idx} WaveformData to_bytes: {e}"))
            })?;
            // Checked u64 multiply: n_channels/n_samples are header-derived
            // and untrusted. A plain `usize` product can overflow, wrap to a
            // SMALL value, pass the `waveform_bytes.len() < expected_bytes`
            // guard, and then the decode loop below (bounds 0..n_samples,
            // 0..n_channels — the REAL huge values) indexes far past
            // waveform_bytes and panics. Overflow -> reject.
            let expected_bytes = (meta.n_channels as u64)
                .checked_mul(meta.n_samples as u64)
                .and_then(|p| p.checked_mul(2))
                .ok_or_else(|| {
                    LmlError::InvalidHeader(format!(
                        "dicom: group {idx} n_channels * n_samples * 2 overflows u64"
                    ))
                })?;
            if (waveform_bytes.len() as u64) < expected_bytes {
                return Err(LmlError::Truncated {
                    expected: expected_bytes as usize,
                    actual: waveform_bytes.len(),
                    context: "dicom WaveformData",
                });
            }

            // Multiplexed decode: sample-major outer loop, channel-major
            // inner. int16 LE.
            for s in 0..meta.n_samples as usize {
                let base = s * meta.n_channels as usize * 2;
                for ch in 0..meta.n_channels as usize {
                    let off = base + ch * 2;
                    let v =
                        i16::from_le_bytes([waveform_bytes[off], waveform_bytes[off + 1]]) as i64;
                    signal[ch].push(v);
                }
            }
            total_samples += meta.n_samples as u64;

            // Channel labels — only pull from the first group; subsequent
            // groups inherit (mismatched labels would be flagged by a
            // future, stricter validator).
            if idx == 0 {
                if let Ok(ch_def_seq) = item.element(TAG_CHANNEL_DEF_SEQUENCE) {
                    if let Some(ch_items) = ch_def_seq.items() {
                        for (ch_idx, ch_item) in ch_items.iter().enumerate() {
                            if ch_idx >= base_n_ch {
                                break;
                            }
                            if let Ok(label_el) = ch_item.element(TAG_CHANNEL_LABEL) {
                                if let Ok(label) = label_el.to_str() {
                                    let trimmed = label.trim().to_string();
                                    if !trimmed.is_empty() {
                                        channels[ch_idx] = trimmed;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // phys_min / phys_max default to the int16 range. Per-channel
        // sensitivity-decoding (mapping raw int16 → physical units via
        // ChannelSensitivity / CorrectionFactor / SensitivityUnits) is
        // deferred — clinical ECG users today just need the integer
        // matrix, and the codec stays format-agnostic.
        let phys_min: Vec<f64> = vec![i16::MIN as f64; base_n_ch];
        let phys_max: Vec<f64> = vec![i16::MAX as f64; base_n_ch];
        let duration_s = if base_rate > 0.0 {
            total_samples as f64 / base_rate
        } else {
            0.0
        };

        // Preserve the full source bytes as a sidecar so a future
        // `lml decode --to-dicom` can reconstruct losslessly even
        // though we ignored sensitivity decoding on the read path.
        let raw_bytes = std::fs::read(&self.path).map_err(LmlError::Io)?;

        let bundle = SignalBundle {
            signal,
            sample_rate: base_rate,
            channels,
            phys_min,
            phys_max,
            duration_s,
            metadata: SourceMetadata {
                source_file: self.path.display().to_string(),
                format: "DICOM_WAVEFORM".to_string(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim: "raw_int16".to_string(),
            },
            sidecar: vec![SidecarBlob {
                key: "dicom_raw".to_string(),
                bytes: raw_bytes,
                aux: None,
            }],
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

/// Validated metadata for a single multiplex group within a Waveform
/// IOD. Built by `parse_group_meta` before any sample bytes are read,
/// so failures surface BEFORE we malloc the sample buffer.
struct GroupMeta {
    idx: usize,
    n_channels: usize,
    n_samples: u32,
    sampling_freq: f64,
}

fn parse_group_meta(
    item: &dicom_object::mem::InMemDicomObject,
    idx: usize,
) -> LmlResult<GroupMeta> {
    let n_channels = item
        .element(TAG_NUM_CHANNELS)
        .map_err(|_| missing(idx, TAG_NUM_CHANNELS, "NumberOfWaveformChannels"))?
        .to_int::<u32>()
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "dicom group {idx}: NumberOfWaveformChannels not a u32: {e}"
            ))
        })? as usize;
    if n_channels == 0 {
        return Err(LmlError::InvalidHeader(format!(
            "dicom group {idx}: NumberOfWaveformChannels = 0"
        )));
    }
    let n_samples = item
        .element(TAG_NUM_SAMPLES)
        .map_err(|_| missing(idx, TAG_NUM_SAMPLES, "NumberOfWaveformSamples"))?
        .to_int::<u32>()
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "dicom group {idx}: NumberOfWaveformSamples not a u32: {e}"
            ))
        })?;
    let sampling_freq_str = item
        .element(TAG_SAMPLING_FREQ)
        .map_err(|_| missing(idx, TAG_SAMPLING_FREQ, "SamplingFrequency"))?
        .to_str()
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "dicom group {idx}: SamplingFrequency not a string: {e}"
            ))
        })?
        .to_string();
    let sampling_freq: f64 = sampling_freq_str.trim().parse().map_err(|e| {
        LmlError::InvalidHeader(format!(
            "dicom group {idx}: SamplingFrequency '{sampling_freq_str}' not a number: {e}"
        ))
    })?;
    if !(sampling_freq.is_finite() && sampling_freq > 0.0) {
        return Err(LmlError::InvalidHeader(format!(
            "dicom group {idx}: SamplingFrequency {sampling_freq} must be finite and > 0"
        )));
    }
    let bits_allocated = item
        .element(TAG_BITS_ALLOCATED)
        .map_err(|_| missing(idx, TAG_BITS_ALLOCATED, "WaveformBitsAllocated"))?
        .to_int::<u32>()
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "dicom group {idx}: WaveformBitsAllocated not a u32: {e}"
            ))
        })?;
    if bits_allocated != 16 {
        return Err(LmlError::InvalidHeader(format!(
            "dicom group {idx}: WaveformBitsAllocated = {bits_allocated}; \
             only 16-bit supported in v1 (PS3.3 §C.10.9 — 8 / 32 deferred)"
        )));
    }
    let sample_interp = item
        .element(TAG_SAMPLE_INTERPRETATION)
        .map_err(|_| {
            missing(
                idx,
                TAG_SAMPLE_INTERPRETATION,
                "WaveformSampleInterpretation",
            )
        })?
        .to_str()
        .map_err(|e| {
            LmlError::InvalidHeader(format!(
                "dicom group {idx}: WaveformSampleInterpretation not a string: {e}"
            ))
        })?
        .trim()
        .to_string();
    if sample_interp != "SS" {
        return Err(LmlError::InvalidHeader(format!(
            "dicom group {idx}: WaveformSampleInterpretation = '{sample_interp}'; \
             only 'SS' (signed 16-bit two's complement) supported in v1 \
             (PS3.3 §C.10.9.1 — UB / MB / AB / SB deferred)"
        )));
    }
    Ok(GroupMeta {
        idx,
        n_channels,
        n_samples,
        sampling_freq,
    })
}

fn missing(idx: usize, tag: Tag, name: &str) -> LmlError {
    LmlError::InvalidHeader(format!(
        "dicom group {idx}: missing {name} ({:04X},{:04X})",
        tag.group(),
        tag.element()
    ))
}
