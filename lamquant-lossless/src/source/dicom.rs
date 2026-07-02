//! DICOM Waveform reader (feature `dicom`).
//!
//! Phase 4.7 + 8 (Item A). Reads the Waveform IOD used for clinical
//! ECG / EEG storage. Tag map (consensus across DICOM PS3.3 §C.10.9
//! and the `pydicom`, `dicom-rs`, and `dcmtk` reference implementations):
//!
//! ```text
//! (5400,0100) WaveformSequence              SQ — one item per multiplex group
//!   (003A,0005) NumberOfWaveformChannels    US
//!   (003A,001A) SamplingFrequency           DS  (Hz, decimal string)
//!   (003A,0010) NumberOfWaveformSamples     UL
//!   (5400,1004) WaveformBitsAllocated       US  (8 / 16 / 32)
//!   (5400,1006) WaveformSampleInterpretation CS  (SS / SB / UB / MB / AB)
//!   (5400,1010) WaveformData                OW/OB (raw samples,
//!                                            channel-multiplexed)
//!   (003A,0200) ChannelDefinitionSequence   SQ — per-channel meta
//!     (003A,0203) ChannelLabel              SH
//!     (5400,1006) WaveformBitsStored        US
//! ```
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
use std::sync::Arc;

use abir::{Abir, Channel, Column};

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
/// `(0008,0060) Modality` — DICOM's own declared-recording-type field
/// (e.g. `"ECG"`). Read (best-effort; ADR 0069 S3b) so a reader that
/// genuinely KNOWS its modality can pass it through to
/// `Abir::with_inferred_modality` as a `FormatDeclared` hint, rather than
/// relying purely on channel-label inference.
const TAG_MODALITY: Tag = Tag(0x0008, 0x0060);

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
                source_file: crate::source::bundle::source_basename(&self.path),
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

    /// ADR 0069 L7: specialize to `Column::I16` — every supported group
    /// is validated `WaveformBitsAllocated == 16` +
    /// `WaveformSampleInterpretation == "SS"` (signed 16-bit two's
    /// complement) before any sample byte is touched, so the whole
    /// multiplex-group decode is a plain `i16` widen with no calibration
    /// folded in (the memory win: 4x vs `I64`).
    ///
    /// Independent of `read_bundle`: re-walks the same
    /// `WaveformSequence` and decodes each group's `WaveformData`
    /// directly into `i16`, mirroring `read_bundle`'s index math exactly
    /// — locked by this module's `lower_to_abir_matches_read_bundle_i64`
    /// test. `phys_min`/`phys_max` use the same synthetic `i16::MIN`/
    /// `MAX` defaults `read_bundle` uses (sensitivity-decoding is
    /// deferred on both paths).
    fn lower_to_abir(&mut self) -> LmlResult<Abir> {
        let obj = open_file(&self.path).map_err(|e| {
            LmlError::InvalidHeader(format!("dicom: open {}: {e}", self.path.display()))
        })?;

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

        let mut cols: Vec<Vec<i16>> = (0..base_n_ch).map(|_| Vec::new()).collect();
        let mut labels: Vec<String> = (0..base_n_ch).map(|i| format!("ch{i}")).collect();

        for (idx, (item, meta)) in groups.iter().zip(group_meta.iter()).enumerate() {
            let waveform_data_el = item.element(TAG_WAVEFORM_DATA).map_err(|_| {
                LmlError::InvalidHeader(format!(
                    "dicom: group {idx} missing WaveformData (5400,1010)"
                ))
            })?;
            let waveform_bytes = waveform_data_el.to_bytes().map_err(|e| {
                LmlError::InvalidHeader(format!("dicom: group {idx} WaveformData to_bytes: {e}"))
            })?;
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

            for s in 0..meta.n_samples as usize {
                let base = s * meta.n_channels as usize * 2;
                for ch in 0..meta.n_channels as usize {
                    let off = base + ch * 2;
                    cols[ch].push(i16::from_le_bytes([
                        waveform_bytes[off],
                        waveform_bytes[off + 1],
                    ]));
                }
            }

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
                                        labels[ch_idx] = trimmed;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let n_samples = cols.first().map(|c| c.len()).unwrap_or(0);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let channels: Vec<Channel> = cols
            .into_iter()
            .enumerate()
            .map(|(j, col)| Channel {
                label: Arc::from(labels[j].as_str()),
                data: Column::I16(Arc::from(col)),
                phys_min: i16::MIN as f64,
                phys_max: i16::MAX as f64,
            })
            .collect();

        // ADR 0069 S3b: unlike the other readers, DICOM DOES have a real
        // declared-modality field — `(0008,0060) Modality` (e.g. `"ECG"`
        // for the Waveform IODs this reader supports). Best-effort read:
        // a missing/malformed tag falls back to `None`, and
        // `infer_modality` then falls back to the channel-label pass.
        let modality_hint: Option<String> = obj
            .element(TAG_MODALITY)
            .ok()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        Ok(Abir::from_parts(channels, base_rate, n_samples)
            .with_inferred_modality(&label_refs, modality_hint.as_deref()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_core::{dicom_value, PrimitiveValue};
    use std::path::PathBuf;

    /// Same fixture directory `tests/dicom_parity.rs` uses — real +
    /// synthesized `.dcm` files committed under `tests/fixtures/dicom/`.
    /// `general_ecg.dcm` was synthesized via
    /// `tools/make_general_ecg_fixture.py` (module doc: "General ECG
    /// Waveform Storage"), 3ch/2500 samples — small and fast for this
    /// gate.
    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dicom")
            .join(name)
    }

    // ─── ADR 0069 L7 gate: lower_to_abir byte-exactness ─────────────

    #[test]
    fn lower_to_abir_matches_read_bundle_i64() {
        let path = fixture("general_ecg.dcm");
        if !path.exists() {
            eprintln!(
                "SKIP lower_to_abir_matches_read_bundle_i64: fixture missing at {}",
                path.display()
            );
            return;
        }
        let bundle = DicomWaveformReader::new(&path).read_bundle().unwrap();
        let abir = DicomWaveformReader::new(&path).lower_to_abir().unwrap();

        assert_eq!(abir.n_channels(), bundle.signal.len());
        assert_eq!(abir.sample_rate, bundle.sample_rate);
        for (j, ch) in abir.channels.iter().enumerate() {
            assert!(
                matches!(ch.data, Column::I16(_)),
                "DICOM SS/16-bit must specialize to Column::I16"
            );
            let widened = ch.data.window_i64(0, abir.n_samples);
            assert_eq!(
                widened.as_ref(),
                bundle.signal[j].as_slice(),
                "channel {j} mismatch"
            );
        }
    }

    // ─── ADR 0069 S3b gate: born-typed lowering (modality inference) ───

    #[test]
    fn lower_to_abir_infers_ecg_from_dicom_modality_tag() {
        use abir::{Ecg, Eeg, Modality, ModalitySource};

        let path = fixture("general_ecg.dcm");
        if !path.exists() {
            eprintln!(
                "SKIP lower_to_abir_infers_ecg_from_dicom_modality_tag: fixture missing at {}",
                path.display()
            );
            return;
        }
        // This fixture's channel labels are non-standard ("Lead I/J/K" —
        // not the 12-lead names `infer_modality` recognizes), so the ONLY
        // reliable signal here is the DICOM `(0008,0060) Modality`
        // attribute itself (`"ECG"` for this SOP class) — exercising the
        // `FormatDeclared` path, not the channel-label path.
        let abir = DicomWaveformReader::new(&path).lower_to_abir().unwrap();
        assert_eq!(abir.provenance().tag, Ecg::TAG);
        assert_eq!(abir.provenance().source, ModalitySource::FormatDeclared);
        assert!(
            abir.clone().try_into_modality::<Eeg>().is_err(),
            "an ECG-declared Abir must refuse promotion to Eeg"
        );
        assert!(abir.try_into_modality::<Ecg>().is_ok());
    }

    // ─── Task #33 (ADR 0069 L9 hardening): negative-path refusal ───────
    //
    // `parse_group_meta` (called by BOTH `read_bundle` and `lower_to_abir`)
    // refuses any group whose `WaveformBitsAllocated != 16` or
    // `WaveformSampleInterpretation != "SS"` BEFORE a single sample byte
    // is decoded — that check is what keeps `lower_to_abir`'s unconditional
    // `Column::I16` narrow safe (a 32-bit or unsigned sample would silently
    // corrupt an `i16` narrow otherwise). These tests mutate a real,
    // otherwise-valid fixture (`general_ecg.dcm`) via `dicom-object`'s
    // `update_value_at` so the byte layout stays spec-valid except for the
    // one tag under test, then assert BOTH entry points reject it.

    /// Mutate one primitive-valued tag inside multiplex group 0 of a
    /// DICOM Waveform IOD fixture and write the result to a fresh temp
    /// file. Panics (test setup failure, not the thing under test) if the
    /// tag isn't present in the fixture.
    fn mutate_group0_tag(
        src: &std::path::Path,
        tag: Tag,
        new_value: PrimitiveValue,
        out_name: &str,
    ) -> PathBuf {
        let mut obj = open_file(src).expect("fixture must open");
        obj.update_value_at((TAG_WAVEFORM_SEQUENCE, 0u32, tag), |v| {
            if let Some(p) = v.primitive_mut() {
                *p = new_value.clone();
            }
        })
        .expect("tag must be present in group 0 of the fixture");
        let tmp = tempfile::tempdir().unwrap();
        // Leak the tempdir so the file survives past this function's
        // return (test-only; the OS reaps it at process exit).
        let out = Box::leak(Box::new(tmp)).path().join(out_name);
        obj.write_to_file(&out).expect("write mutated fixture");
        out
    }

    #[test]
    fn read_bundle_and_lower_to_abir_reject_bits_allocated_ne_16() {
        let path = fixture("general_ecg.dcm");
        if !path.exists() {
            eprintln!(
                "SKIP read_bundle_and_lower_to_abir_reject_bits_allocated_ne_16: \
                 fixture missing at {}",
                path.display()
            );
            return;
        }
        let out = mutate_group0_tag(
            &path,
            TAG_BITS_ALLOCATED,
            dicom_value!(U16, [8u16]),
            "bits_allocated_8.dcm",
        );

        assert!(
            DicomWaveformReader::new(&out).read_bundle().is_err(),
            "WaveformBitsAllocated=8 must be refused by read_bundle (only 16-bit \
             supported — PS3.3 §C.10.9)"
        );
        assert!(
            DicomWaveformReader::new(&out).lower_to_abir().is_err(),
            "WaveformBitsAllocated=8 must be refused by lower_to_abir — an \
             oversized value must never reach the unconditional Column::I16 narrow"
        );
    }

    #[test]
    fn read_bundle_and_lower_to_abir_reject_sample_interpretation_ne_ss() {
        let path = fixture("general_ecg.dcm");
        if !path.exists() {
            eprintln!(
                "SKIP read_bundle_and_lower_to_abir_reject_sample_interpretation_ne_ss: \
                 fixture missing at {}",
                path.display()
            );
            return;
        }
        let out = mutate_group0_tag(
            &path,
            TAG_SAMPLE_INTERPRETATION,
            PrimitiveValue::from("UB"),
            "interp_ub.dcm",
        );

        assert!(
            DicomWaveformReader::new(&out).read_bundle().is_err(),
            "WaveformSampleInterpretation=UB must be refused by read_bundle \
             (only 'SS' supported — PS3.3 §C.10.9.1)"
        );
        assert!(
            DicomWaveformReader::new(&out).lower_to_abir().is_err(),
            "WaveformSampleInterpretation=UB must be refused by lower_to_abir"
        );
    }
}
