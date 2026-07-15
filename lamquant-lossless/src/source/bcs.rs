//! BCS1/BCS2 compatibility at the immutable ABIR2 recording boundary.
//!
//! BCS1 is the frozen equal-rate integer container. BCS2 already serializes an
//! ABIR2 graph directly. This module is the only place that lifts legacy BCS1
//! semantics into that graph, so LMA and other callers never branch on the
//! original recording version after decode.

use std::path::Path;

use abir::{
    decode_bcs2, name_for_tag, Bcs1Header, BiosignalWireVersion, Recording, BCS1_HEADER_LEN,
};

use crate::error::{LmlError, LmlResult};

use super::abir2::{recording_from_signal_bundle_with_options, RecordingAdapterOptions};
use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};

/// A decoded recording plus the wire family that carried it.
#[derive(Clone, Debug)]
pub struct BiosignalRecording {
    wire_version: BiosignalWireVersion,
    recording: Recording,
}

impl BiosignalRecording {
    /// Source wire family, detected from the authoritative leading magic.
    pub fn wire_version(&self) -> BiosignalWireVersion {
        self.wire_version
    }

    /// Canonical immutable ABIR2 graph.
    pub fn recording(&self) -> &Recording {
        &self.recording
    }

    /// Consume the wrapper and return the canonical graph.
    pub fn into_recording(self) -> Recording {
        self.recording
    }
}

/// Decode BCS1 or BCS2 bytes into one canonical immutable recording.
///
/// `source_name` is descriptive provenance only. Its basename is retained;
/// absolute host paths are never embedded. Wire dispatch always uses magic.
pub fn decode_biosignal_recording(
    bytes: &[u8],
    source_name: Option<&str>,
) -> LmlResult<BiosignalRecording> {
    let wire_version = BiosignalWireVersion::detect(bytes).ok_or_else(|| {
        LmlError::InvalidHeader("biosignal recording entry is neither BCS1 nor BCS2".to_string())
    })?;
    let recording = match wire_version {
        BiosignalWireVersion::Bcs1 => decode_bcs1_recording(bytes, source_name)?,
        BiosignalWireVersion::Bcs2 => decode_bcs2(bytes)
            .map_err(|error| LmlError::InvalidHeader(format!("BCS2 recording: {error}")))?,
    };
    Ok(BiosignalRecording {
        wire_version,
        recording,
    })
}

fn decode_bcs1_recording(bytes: &[u8], source_name: Option<&str>) -> LmlResult<Recording> {
    let header = Bcs1Header::parse(bytes)
        .map_err(|error| LmlError::InvalidHeader(format!("BCS1 header: {error}")))?;
    let (signal, metadata_json) = crate::container::bcs1_read_bytes(bytes)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
        .map_err(|error| LmlError::InvalidHeader(format!("BCS1 metadata JSON: {error}")))?;
    let object = metadata.as_object().ok_or_else(|| {
        LmlError::InvalidHeader("BCS1 metadata JSON must be an object".to_string())
    })?;
    let channel_count = signal.len();
    let channels =
        string_array(object.get("channels"), "channels", channel_count)?.unwrap_or_else(|| {
            (0..channel_count)
                .map(|index| format!("channel-{index:04}"))
                .collect()
        });
    let phys_min = number_array(object.get("phys_min"), "phys_min", channel_count)?
        .unwrap_or_else(|| vec![0.0; channel_count]);
    let phys_max = number_array(object.get("phys_max"), "phys_max", channel_count)?
        .unwrap_or_else(|| vec![0.0; channel_count]);
    let sample_rate = header.sample_rate_mhz as f64 / 1_000.0;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 sample_rate_mhz {} is not positive",
            header.sample_rate_mhz
        )));
    }
    let sample_count = signal.first().map(Vec::len).unwrap_or(0);
    let source_file = source_name
        .map(Path::new)
        .and_then(Path::file_name)
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    let patient_id = text_field(object, "patient_id")?.unwrap_or_default();
    let bundle = SignalBundle {
        signal,
        sample_rate,
        channels,
        phys_min,
        phys_max,
        duration_s: sample_count as f64 / sample_rate,
        metadata: SourceMetadata {
            source_file,
            format: "BCS1".into(),
            patient_id: patient_id.clone(),
            recording_info: text_field(object, "recording_info")?.unwrap_or_default(),
            startdate: text_field(object, "startdate")?.unwrap_or_default(),
            phys_dim: text_field(object, "phys_dim")?.unwrap_or_else(|| "source".into()),
        },
        sidecar: vec![
            SidecarBlob {
                key: "bcs1_header".into(),
                bytes: bytes[..BCS1_HEADER_LEN].to_vec(),
                aux: None,
            },
            SidecarBlob {
                key: "bcs1_metadata_json".into(),
                bytes: metadata_json.into_bytes(),
                aux: None,
            },
        ],
    };
    let declared_modality = name_for_tag(header.modality_tag)
        .filter(|value| *value != "untyped")
        .map(str::to_owned);
    recording_from_signal_bundle_with_options(
        bundle,
        RecordingAdapterOptions {
            subject: (!patient_id.is_empty()).then_some(patient_id),
            declared_modality,
            ..RecordingAdapterOptions::default()
        },
    )
}

fn text_field(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> LmlResult<Option<String>> {
    match object.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value.as_str().map(str::to_owned).map(Some).ok_or_else(|| {
            LmlError::InvalidHeader(format!("BCS1 metadata field {field:?} must be text"))
        }),
    }
}

fn string_array(
    value: Option<&serde_json::Value>,
    field: &str,
    expected: usize,
) -> LmlResult<Option<Vec<String>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        LmlError::InvalidHeader(format!("BCS1 metadata field {field:?} must be an array"))
    })?;
    if values.len() != expected {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 metadata field {field:?} has {} values, expected {expected}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "BCS1 metadata field {field:?}[{index}] must be text"
                ))
            })
        })
        .collect::<LmlResult<Vec<_>>>()
        .map(Some)
}

fn number_array(
    value: Option<&serde_json::Value>,
    field: &str,
    expected: usize,
) -> LmlResult<Option<Vec<f64>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let values = value.as_array().ok_or_else(|| {
        LmlError::InvalidHeader(format!("BCS1 metadata field {field:?} must be an array"))
    })?;
    if values.len() != expected {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 metadata field {field:?} has {} values, expected {expected}",
            values.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let number = value.as_f64().ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "BCS1 metadata field {field:?}[{index}] must be numeric"
                ))
            })?;
            if !number.is_finite() {
                return Err(LmlError::InvalidHeader(format!(
                    "BCS1 metadata field {field:?}[{index}] must be finite"
                )));
            }
            Ok(number)
        })
        .collect::<LmlResult<Vec<_>>>()
        .map(Some)
}
