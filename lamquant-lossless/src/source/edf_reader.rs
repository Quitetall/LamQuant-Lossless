//! `EdfReader` — `SignalSourceReader` impl for EDF/EDF+/BDF.
//!
//! Today this wraps the existing `crate::edf::read_edf` (which parses
//! a `&Path` directly). Phase 0.5 generalises to `&mut dyn LmlSource`
//! so stdin and S3 sources work. The current implementation keeps the
//! existing byte-parser intact and lowers its private native form into ABIR.
//!
//! Reconstruction consumes a validated `UniformSignalView`; source-preservation
//! bytes remain content-addressed inside ABIR.

use std::path::{Path, PathBuf};

use crate::edf::{read_edf, EdfFile};
use crate::error::{LmlError, LmlResult};

use super::bundle::{ParsedUniformSignal, SidecarBlob, SourceMetadata};
use super::reader::SignalSourceReader;
use super::semantic::{
    lower_parsed_uniform_signal, SemanticLoweringOptions, SemanticRead, UniformSignalView,
};

/// Reader for EDF / EDF+ / BDF files.
///
/// Holds the path at construction. Repeated lowering re-parses the unchanged
/// file; `read_edf` is idempotent for unchanged bytes.
#[derive(Debug, Clone)]
pub struct EdfReader {
    path: PathBuf,
}

impl EdfReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Lower once with Adapter-provided semantic overlays and source binding.
    pub fn lower_to_abir_with_options(
        &mut self,
        options: SemanticLoweringOptions,
    ) -> LmlResult<SemanticRead> {
        lower_parsed_uniform_signal(self.read_parsed_signal()?, options)
    }

    fn read_parsed_signal(&mut self) -> LmlResult<ParsedUniformSignal> {
        let edf = read_edf(&self.path)?;
        Ok(edf.into())
    }
}

impl SignalSourceReader for EdfReader {
    fn lower_to_abir(&mut self) -> LmlResult<SemanticRead> {
        self.lower_to_abir_with_options(SemanticLoweringOptions::default())
    }
}

// ─── Lossless EdfFile ↔ semantic native view conversion ───────────────

/// `EdfFile` → private parsed signal. Infallible because:
///   - All numeric fields (`n_signals_total`, `n_data_records`,
///     `dig_min/max`) are integers that serde_json serialises trivially.
///   - `record_duration: f64` is finite by construction — `read_edf`
///     parses it via `crate::source::ascii::parse_float`, which rejects
///     NaN / Inf at the source (Phase 0.2 strengthening). serde_json
///     fails to serialise NaN/Inf, so the parse-time rejection is the
///     load-bearing guarantee here.
///   - All strings come from EDF ASCII fields that `parse_*` already
///     promoted to valid UTF-8 (or rejected).
///
/// If any invariant above weakens, this expect would panic — a future
/// audit should re-check this list before changing the ASCII parsers.
impl From<EdfFile> for ParsedUniformSignal {
    fn from(e: EdfFile) -> Self {
        let mut sidecar = Vec::with_capacity(2 + e.non_eeg_data.len() + 1);
        sidecar.push(SidecarBlob {
            key: "raw_header".into(),
            bytes: e.raw_header,
            aux: None,
        });
        sidecar.push(SidecarBlob {
            key: "trailing_data".into(),
            bytes: e.trailing_data,
            aux: None,
        });
        for (ch_idx, raw) in e.non_eeg_data {
            sidecar.push(SidecarBlob {
                key: "non_eeg_chunk".into(),
                bytes: raw,
                aux: Some(ch_idx as i64),
            });
        }
        // EDF-specific scalars/vectors needed to reconstruct an
        // EdfFile. JSON keeps the schema explicit and human-readable
        // when dumped from `lml info`. Reconstruction lives in the
        // matching `TryFrom` below.
        let edf_meta = serde_json::json!({
            "n_signals_total": e.n_signals_total,
            "n_data_records": e.n_data_records,
            "record_duration": e.record_duration,
            "all_labels": e.all_labels,
            "all_ns_per_rec": e.all_ns_per_rec,
            "eeg_indices": e.eeg_indices,
            "dig_min": e.dig_min,
            "dig_max": e.dig_max,
            "is_bdf": e.is_bdf,
        });
        let edf_meta_bytes = serde_json::to_vec(&edf_meta)
            .expect("serde_json::to_vec on a hand-built tree of scalars cannot fail");
        sidecar.push(SidecarBlob {
            key: "edf_meta".into(),
            bytes: edf_meta_bytes,
            aux: None,
        });
        Self {
            signal: e.signal,
            sample_rate: e.sample_rate,
            channels: e.channels,
            phys_min: e.phys_min,
            phys_max: e.phys_max,
            duration_s: e.duration_s,
            metadata: SourceMetadata {
                source_file: e.source_file,
                format: e.format,
                patient_id: e.patient_id,
                recording_info: e.recording_info,
                startdate: e.startdate,
                phys_dim: e.phys_dim,
            },
            sidecar,
        }
    }
}

/// ABIR-tied native signal → `EdfFile`. Fails when preservation capsules are
/// missing or malformed.
impl TryFrom<UniformSignalView<'_>> for EdfFile {
    type Error = LmlError;

    fn try_from(signal: UniformSignalView<'_>) -> LmlResult<Self> {
        let raw_header = signal
            .sidecar_first("raw_header")
            .ok_or_else(|| LmlError::InvalidHeader("ABIR → EdfFile: raw_header missing".into()))?
            .bytes
            .to_vec();
        let trailing_data = signal
            .sidecar_first("trailing_data")
            .ok_or_else(|| LmlError::InvalidHeader("ABIR → EdfFile: trailing_data missing".into()))?
            .bytes
            .to_vec();
        let mut non_eeg_data: Vec<(usize, Vec<u8>)> = Vec::new();
        for chunk in signal.sidecar_all("non_eeg_chunk") {
            let ch_idx = chunk.aux.ok_or_else(|| {
                LmlError::InvalidHeader(
                    "ABIR → EdfFile: non_eeg_chunk capsule missing aux index".into(),
                )
            })? as usize;
            non_eeg_data.push((ch_idx, chunk.bytes.to_vec()));
        }
        let edf_meta_bytes = signal
            .sidecar_first("edf_meta")
            .ok_or_else(|| {
                LmlError::InvalidHeader("ABIR → EdfFile: edf_meta capsule missing".into())
            })?
            .bytes;
        let meta: serde_json::Value = serde_json::from_slice(edf_meta_bytes)
            .map_err(|e| LmlError::InvalidHeader(format!("ABIR → EdfFile: edf_meta json: {e}")))?;
        let get_usize = |k: &str| -> LmlResult<usize> {
            meta[k].as_u64().map(|v| v as usize).ok_or_else(|| {
                LmlError::InvalidHeader(format!("edf_meta.{k}: missing or not a u64"))
            })
        };
        let get_f64 = |k: &str| -> LmlResult<f64> {
            meta[k]
                .as_f64()
                .ok_or_else(|| LmlError::InvalidHeader(format!("edf_meta.{k}: not a number")))
        };
        let get_bool = |k: &str| -> LmlResult<bool> {
            meta[k]
                .as_bool()
                .ok_or_else(|| LmlError::InvalidHeader(format!("edf_meta.{k}: not a bool")))
        };
        let get_str_vec = |k: &str| -> LmlResult<Vec<String>> {
            meta[k]
                .as_array()
                .ok_or_else(|| LmlError::InvalidHeader(format!("edf_meta.{k}: not an array")))?
                .iter()
                .map(|v| {
                    v.as_str().map(str::to_string).ok_or_else(|| {
                        LmlError::InvalidHeader(format!("edf_meta.{k}: element not a string"))
                    })
                })
                .collect()
        };
        let get_usize_vec = |k: &str| -> LmlResult<Vec<usize>> {
            meta[k]
                .as_array()
                .ok_or_else(|| LmlError::InvalidHeader(format!("edf_meta.{k}: not an array")))?
                .iter()
                .map(|v| {
                    v.as_u64().map(|x| x as usize).ok_or_else(|| {
                        LmlError::InvalidHeader(format!("edf_meta.{k}: element not a u64"))
                    })
                })
                .collect()
        };
        let get_i32_vec = |k: &str| -> LmlResult<Vec<i32>> {
            meta[k]
                .as_array()
                .ok_or_else(|| LmlError::InvalidHeader(format!("edf_meta.{k}: not an array")))?
                .iter()
                .map(|v| {
                    let raw = v.as_i64().ok_or_else(|| {
                        LmlError::InvalidHeader(format!("edf_meta.{k}: element not an i64"))
                    })?;
                    // Defensive: a tampered sidecar could supply an
                    // out-of-i32 integer. EDF spec bounds dig_min/max
                    // to [-32768, 32767] (or BDF [-8388608, 8388607]),
                    // both well inside i32; reject explicitly rather
                    // than `as i32` truncate (Bible R30 hostile-caller).
                    i32::try_from(raw).map_err(|_| {
                        LmlError::InvalidHeader(format!("edf_meta.{k}: {raw} out of i32 range"))
                    })
                })
                .collect()
        };
        // Total samples is the per-channel sample count, which the EDF
        // spec requires to be uniform across channels. After Phase 0.2,
        // `read_edf` already enforces this; we recompute here so the
        // round-trip preserves it instead of leaving it 0.
        let total_samples = signal.signal().first().map(Vec::len).unwrap_or(0);
        let n_channels = signal.signal().len();
        let metadata = signal.source_metadata();
        Ok(EdfFile {
            signal: signal.signal().to_vec(),
            channels: signal.channels().to_vec(),
            sample_rate: signal.sample_rate(),
            n_channels,
            total_samples,
            duration_s: signal.duration_s(),
            source_file: metadata.source_file.clone(),
            patient_id: metadata.patient_id.clone(),
            raw_header,
            non_eeg_data,
            n_signals_total: get_usize("n_signals_total")?,
            n_data_records: get_usize("n_data_records")?,
            record_duration: get_f64("record_duration")?,
            all_labels: get_str_vec("all_labels")?,
            all_ns_per_rec: get_usize_vec("all_ns_per_rec")?,
            eeg_indices: get_usize_vec("eeg_indices")?,
            recording_info: metadata.recording_info.clone(),
            startdate: metadata.startdate.clone(),
            format: metadata.format.clone(),
            phys_min: signal.physical_minima().to_vec(),
            phys_max: signal.physical_maxima().to_vec(),
            dig_min: get_i32_vec("dig_min")?,
            dig_max: get_i32_vec("dig_max")?,
            phys_dim: metadata.phys_dim.clone(),
            trailing_data,
            is_bdf: get_bool("is_bdf")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but valid EdfFile for round-trip testing.
    fn make_edf() -> EdfFile {
        EdfFile {
            signal: vec![vec![1i64, 2, 3, 4]],
            channels: vec!["Fp1".into()],
            sample_rate: 256.0,
            n_channels: 1,
            total_samples: 4,
            duration_s: 4.0 / 256.0,
            source_file: "/tmp/test.edf".into(),
            patient_id: "X".into(),
            raw_header: vec![0xAA; 512],
            non_eeg_data: vec![(2, vec![0xBB; 16])],
            n_signals_total: 2,
            n_data_records: 1,
            record_duration: 1.0,
            all_labels: vec!["Fp1".into(), "EDF Annotations".into()],
            all_ns_per_rec: vec![4, 8],
            eeg_indices: vec![0],
            recording_info: "Startdate 16-MAY-2026".into(),
            startdate: "16.05.26".into(),
            format: "EDF+C".into(),
            phys_min: vec![-200.0],
            phys_max: vec![200.0],
            dig_min: vec![-32768],
            dig_max: vec![32767],
            phys_dim: "uV".into(),
            trailing_data: vec![0xCC, 0xDD],
            is_bdf: false,
        }
    }

    fn lower_fixture(edf: EdfFile) -> SemanticRead {
        lower_parsed_uniform_signal(edf.into(), SemanticLoweringOptions::default()).unwrap()
    }

    #[test]
    fn edf_to_abir_preserves_signal() {
        let read = lower_fixture(make_edf());
        let signal = read.uniform_signal().unwrap();
        assert_eq!(signal.signal(), [vec![1i64, 2, 3, 4]]);
        assert_eq!(signal.sample_rate(), 256.0);
        assert_eq!(signal.channels(), ["Fp1"]);
    }

    #[test]
    fn edf_to_abir_populates_content_addressed_capsules() {
        let read = lower_fixture(make_edf());
        let signal = read.uniform_signal().unwrap();
        let h = signal.sidecar_first("raw_header").unwrap();
        assert_eq!(h.bytes.len(), 512);
        let trailing = signal.sidecar_first("trailing_data").unwrap();
        assert_eq!(trailing.bytes, vec![0xCC, 0xDD]);
        let chunks: Vec<_> = signal.sidecar_all("non_eeg_chunk").collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].aux, Some(2));
        assert_eq!(chunks[0].bytes.len(), 16);
        assert!(signal.sidecar_first("edf_meta").is_some());
    }

    #[test]
    fn roundtrip_edf_to_abir_to_edf() {
        // `EdfFile` is not `Clone` (intentional — it owns large header
        // buffers). Build a fresh fixture for the comparison so we
        // don't need to derive Clone on the public struct.
        let edf = make_edf();
        let reference = make_edf();
        let read = lower_fixture(edf);
        let rt: EdfFile = read
            .uniform_signal()
            .unwrap()
            .try_into()
            .expect("ABIR → EdfFile must succeed");
        // Total_samples is preserved: the TryFrom recomputes it from
        // the per-channel sample count (EDF spec guarantees uniformity).
        assert_eq!(rt.total_samples, reference.total_samples);
        assert_eq!(rt.signal, reference.signal);
        assert_eq!(rt.channels, reference.channels);
        assert_eq!(rt.sample_rate, reference.sample_rate);
        assert_eq!(rt.duration_s, reference.duration_s);
        assert_eq!(rt.raw_header, reference.raw_header);
        assert_eq!(rt.non_eeg_data, reference.non_eeg_data);
        assert_eq!(rt.n_signals_total, reference.n_signals_total);
        assert_eq!(rt.n_data_records, reference.n_data_records);
        assert_eq!(rt.record_duration, reference.record_duration);
        assert_eq!(rt.all_labels, reference.all_labels);
        assert_eq!(rt.all_ns_per_rec, reference.all_ns_per_rec);
        assert_eq!(rt.eeg_indices, reference.eeg_indices);
        assert_eq!(rt.format, reference.format);
        assert_eq!(rt.phys_min, reference.phys_min);
        assert_eq!(rt.phys_max, reference.phys_max);
        assert_eq!(rt.dig_min, reference.dig_min);
        assert_eq!(rt.dig_max, reference.dig_max);
        assert_eq!(rt.phys_dim, reference.phys_dim);
        assert_eq!(rt.trailing_data, reference.trailing_data);
        assert_eq!(rt.is_bdf, reference.is_bdf);
        assert_eq!(rt.patient_id, reference.patient_id);
        assert_eq!(rt.recording_info, reference.recording_info);
        assert_eq!(rt.startdate, reference.startdate);
    }

    #[test]
    fn ragged_native_signal_never_exposes_an_abir_view() {
        let mut parsed: ParsedUniformSignal = make_edf().into();
        parsed.signal.push(vec![999i64]);
        parsed.channels.push("Fp2".into());
        parsed.phys_min.push(-200.0);
        parsed.phys_max.push(200.0);
        let err =
            lower_parsed_uniform_signal(parsed, SemanticLoweringOptions::default()).unwrap_err();
        assert!(
            err.to_string().contains("ragged") || err.to_string().contains("samples"),
            "expected validate() to reject ragged signal, got: {err}"
        );
    }

    #[test]
    fn abir_to_edf_missing_raw_header_errs() {
        let mut parsed: ParsedUniformSignal = make_edf().into();
        parsed.sidecar.retain(|s| s.key != "raw_header");
        let read = lower_parsed_uniform_signal(parsed, SemanticLoweringOptions::default()).unwrap();
        let r: LmlResult<EdfFile> = read.uniform_signal().unwrap().try_into();
        assert!(r.is_err());
    }

    #[test]
    fn abir_to_edf_missing_edf_meta_errs() {
        let mut parsed: ParsedUniformSignal = make_edf().into();
        parsed.sidecar.retain(|s| s.key != "edf_meta");
        let read = lower_parsed_uniform_signal(parsed, SemanticLoweringOptions::default()).unwrap();
        let r: LmlResult<EdfFile> = read.uniform_signal().unwrap().try_into();
        assert!(r.is_err());
    }

    #[test]
    fn abir_to_edf_corrupt_edf_meta_errs() {
        let mut parsed: ParsedUniformSignal = make_edf().into();
        if let Some(s) = parsed.sidecar.iter_mut().find(|s| s.key == "edf_meta") {
            s.bytes = b"not valid json".to_vec();
        }
        let read = lower_parsed_uniform_signal(parsed, SemanticLoweringOptions::default()).unwrap();
        let r: LmlResult<EdfFile> = read.uniform_signal().unwrap().try_into();
        assert!(r.is_err());
    }
}
