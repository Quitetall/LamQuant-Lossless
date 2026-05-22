//! `lsl::StreamInfo` builder. Wraps `metadata_lite::stream_spec_from_lml`
//! and constructs the real `lsl::StreamInfo`, populating channel
//! descriptors in the XML so LabRecorder + other LSL consumers
//! display human-readable channel labels.
//!
//! Only compiled with the `liblsl` Cargo feature.

use crate::error::LslIntegrationError;
use crate::metadata_lite::{stream_spec_from_lml, ChannelFormatLite};
use lsl::{ChannelFormat, StreamInfo};

/// Convert our zero-dep `ChannelFormatLite` into the real
/// `lsl::ChannelFormat`. Phase 1 handles the two formats LamQuant
/// emits (int32 from EDF/synth-i16, float32 reserved for future
/// float sources); other LSL formats (string, int8, int16, int64,
/// double64) are not yet exercised.
fn channel_format(lite: ChannelFormatLite) -> ChannelFormat {
    match lite {
        ChannelFormatLite::Int32 => ChannelFormat::Int32,
        ChannelFormatLite::Float32 => ChannelFormat::Float32,
    }
}

/// Build a populated `lsl::StreamInfo` directly from an `.lml` file.
/// World-class detail: writes per-channel label + unit + type into
/// the StreamInfo's XML metadata so consumers display "Fp1-F7" not
/// "ch0".
pub fn stream_info_from_lml(
    lml_path: &std::path::Path,
    name: Option<&str>,
) -> Result<StreamInfo, LslIntegrationError> {
    let source_id = crate::stream_id::stream_id_from_lml(lml_path)?;
    let spec = stream_spec_from_lml(lml_path, name, Some("EEG"), &source_id)?;
    let mut info = StreamInfo::new(
        &spec.name,
        &spec.stream_type,
        spec.channel_count,
        spec.nominal_srate,
        channel_format(spec.channel_format),
        &spec.source_id,
    )
    .map_err(LslIntegrationError::Lsl)?;
    let mut desc = info.desc();
    let mut channels = desc.append_child("channels");
    for (i, label) in spec.channel_labels.iter().enumerate() {
        let display = if label.is_empty() {
            format!("ch{}", i)
        } else {
            label.clone()
        };
        let mut ch = channels.append_child("channel");
        ch.append_child_value("label", &display);
        ch.append_child_value("unit", &spec.channel_unit);
        ch.append_child_value("type", "EEG");
    }
    Ok(info)
}
