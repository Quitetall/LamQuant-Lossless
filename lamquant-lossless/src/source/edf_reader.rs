//! EDF/EDF+/BDF ingest into canonical semantic ABIR.
//!
//! `EdfFile` remains reader-private native state. Public source seams return
//! [`super::semantic::SemanticRead`]; exact headers, annotation channels, and
//! trailing bytes become content-bound ABIR source capsules.

use std::path::{Path, PathBuf};

use crate::edf::{read_edf, EdfFile};
use crate::error::{LmlError, LmlResult};

use super::metadata::{SidecarBlob, SourceMetadata};
use super::reader::SignalSourceReader;
use super::semantic::{from_owned_uniform_signal, SemanticRead};

/// Reader for EDF / EDF+ / BDF files.
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
}

impl SignalSourceReader for EdfReader {
    fn lower_to_abir(&mut self) -> LmlResult<SemanticRead> {
        lower_edf_file(
            read_edf(&self.path)?,
            semantic_abir::ValidationLimits::default(),
        )
    }
}

/// Lower one parsed EDF-family value without exposing a second common carrier.
pub fn lower_edf_file(
    edf: EdfFile,
    limits: semantic_abir::ValidationLimits,
) -> LmlResult<SemanticRead> {
    let EdfFile {
        signal,
        channels,
        sample_rate,
        n_channels: _,
        total_samples: _,
        duration_s,
        source_file,
        patient_id,
        raw_header,
        non_eeg_data,
        n_signals_total,
        n_data_records,
        record_duration,
        all_labels,
        all_ns_per_rec,
        eeg_indices,
        recording_info,
        startdate,
        format,
        phys_min,
        phys_max,
        dig_min,
        dig_max,
        phys_dim,
        trailing_data,
        is_bdf,
    } = edf;

    let mut sidecar = Vec::with_capacity(3 + non_eeg_data.len());
    sidecar.push(SidecarBlob {
        key: "raw_header".into(),
        bytes: raw_header,
        aux: None,
    });
    sidecar.push(SidecarBlob {
        key: "trailing_data".into(),
        bytes: trailing_data,
        aux: None,
    });
    for (channel_index, bytes) in non_eeg_data {
        sidecar.push(SidecarBlob {
            key: "non_eeg_chunk".into(),
            bytes,
            aux: Some(channel_index as i64),
        });
    }
    let edf_meta = serde_json::json!({
        "n_signals_total": n_signals_total,
        "n_data_records": n_data_records,
        "record_duration": record_duration,
        "all_labels": all_labels,
        "all_ns_per_rec": all_ns_per_rec,
        "eeg_indices": eeg_indices,
        "dig_min": dig_min,
        "dig_max": dig_max,
        "is_bdf": is_bdf,
    });
    sidecar.push(SidecarBlob {
        key: "edf_meta".into(),
        bytes: serde_json::to_vec(&edf_meta)
            .map_err(|error| LmlError::InvalidHeader(format!("EDF metadata: {error}")))?,
        aux: None,
    });

    from_owned_uniform_signal(
        signal,
        sample_rate,
        channels,
        phys_min,
        phys_max,
        duration_s,
        SourceMetadata {
            source_file,
            format,
            patient_id,
            recording_info,
            startdate,
            phys_dim,
        },
        sidecar,
        limits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edf() -> EdfFile {
        EdfFile {
            signal: vec![vec![1_i64, 2, 3, 4]],
            channels: vec!["Fp1".into()],
            sample_rate: 256.0,
            n_channels: 1,
            total_samples: 4,
            duration_s: 4.0 / 256.0,
            source_file: "test.edf".into(),
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

    #[test]
    fn edf_lowers_directly_to_validated_abir() {
        let read = lower_edf_file(make_edf(), semantic_abir::ValidationLimits::default()).unwrap();
        assert_eq!(read.native_i64(), Some(&[vec![1, 2, 3, 4]][..]));
        assert_eq!(read.mapping.source_format, "EDF+C");
        assert_eq!(read.mapping.channel_count, 1);
        assert_eq!(read.mapping.source_capsule_count, 4);
        assert_eq!(read.opened.dataset().recordings().len(), 1);
        assert_eq!(read.opened.dataset().streams().len(), 1);
        assert!(read.fidelity.sample_values_exact);
    }

    #[test]
    fn edf_forensic_parts_are_content_bound_source_capsules() {
        let read = lower_edf_file(make_edf(), semantic_abir::ValidationLimits::default()).unwrap();
        let values = read
            .opened
            .dataset()
            .source_capsules()
            .iter()
            .map(|capsule| capsule.source().value())
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            vec!["raw_header", "trailing_data", "non_eeg_chunk#2", "edf_meta"]
        );
        let raw_header_id =
            semantic_abir::payload_content_id(semantic_abir::ElementType::Bytes, &[0xAA; 512]);
        assert_eq!(
            read.opened.dataset().source_capsules()[0].content_id(),
            raw_header_id
        );
    }

    #[test]
    fn malformed_edf_native_shape_fails_closed() {
        let mut edf = make_edf();
        edf.signal.clear();
        edf.channels.clear();
        edf.phys_min.clear();
        edf.phys_max.clear();
        let error = lower_edf_file(edf, semantic_abir::ValidationLimits::default()).unwrap_err();
        assert!(error.to_string().contains("at least one channel"));
    }
}
