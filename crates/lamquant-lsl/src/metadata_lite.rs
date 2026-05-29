//! Feature-gate-free metadata parser. Builds a `StreamSpec` from an
//! `.lml` container without depending on the `lsl` crate, so the
//! parsing logic is testable + reachable even when the `liblsl`
//! feature is off. The `lsl::StreamInfo` construction (which DOES
//! need `lsl`) lives in `metadata.rs` and wraps this.
//!
//! World-class detail: this preserves the EDF signal-header
//! metadata (channel labels, physical units) end-to-end so
//! LabRecorder + downstream tools display "Fp1-F7" not "ch0" out
//! of the box.

use crate::error::LslIntegrationError;

/// Wire format for the LSL `channel_format` field. Mirrors
/// `lsl::ChannelFormat` so callers can construct it without
/// pulling the lsl crate in. The `liblsl`-feature build converts
/// to the real enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelFormatLite {
    Int32,
    Float32,
}

/// Stream specification — everything needed to construct an
/// `lsl::StreamInfo`. Built from `.lml` metadata.
#[derive(Debug, Clone)]
pub struct StreamSpec {
    pub name: String,
    pub stream_type: String,
    pub channel_count: u32,
    pub nominal_srate: f64,
    pub channel_format: ChannelFormatLite,
    pub source_id: String,
    /// Channel labels in order. Empty string for unlabeled channels.
    pub channel_labels: Vec<String>,
    /// Channel units (e.g. "uV"). One per channel (same unit if EDF
    /// reports one phys_dim for all signals).
    pub channel_unit: String,
}

impl StreamSpec {
    /// Build a `StreamSpec` directly from caller-supplied channel
    /// labels + sample rate — the codec-independent path. Vision is
    /// the LSL bridge per the project vision, so an outlet must be
    /// constructable from any decoded / live buffer (a device serial
    /// stream, an in-memory decode, a Python-decoded window) without
    /// touching the `.lml` codec at all.
    ///
    /// `channel_labels` drives both the channel count and the LSL
    /// XML `<channels>` block. `name` is the human-readable stream
    /// name LabRecorder shows in its source picker. `source_id` is
    /// the LSL dedup key — pass a stable per-source string (e.g. the
    /// device serial or a content hash) so reconnects dedup; empty
    /// disables LSL's recovery/dedup.
    ///
    /// Channel format is fixed to `Int32` (the lossless wire format
    /// LamQuant emits — EDF int24 + synth i16 both fit). `stream_type`
    /// defaults to `"EEG"`; `channel_unit` to `"uV"`.
    pub fn from_channels(
        name: &str,
        nominal_srate: f64,
        channel_labels: Vec<String>,
        source_id: &str,
    ) -> Result<Self, LslIntegrationError> {
        if channel_labels.is_empty() {
            return Err(LslIntegrationError::MissingMetadata(
                "StreamSpec::from_channels: at least one channel label required".to_string(),
            ));
        }
        if !nominal_srate.is_finite() || nominal_srate < 0.0 {
            return Err(LslIntegrationError::MissingMetadata(format!(
                "StreamSpec::from_channels: nominal_srate {} must be finite + >= 0 \
                 (0 == lsl::IRREGULAR_RATE)",
                nominal_srate
            )));
        }
        let channel_count = u32::try_from(channel_labels.len()).map_err(|_| {
            LslIntegrationError::MissingMetadata(format!(
                "channel count {} doesn't fit in u32",
                channel_labels.len()
            ))
        })?;
        Ok(StreamSpec {
            name: name.to_string(),
            stream_type: "EEG".to_string(),
            channel_count,
            nominal_srate,
            channel_format: ChannelFormatLite::Int32,
            source_id: source_id.to_string(),
            channel_labels,
            channel_unit: "uV".to_string(),
        })
    }
}

/// Build a StreamSpec by reading the LML container header +
/// metadata JSON. No liblsl dep, so this is reachable from any
/// build configuration.
pub fn stream_spec_from_lml(
    lml_path: &std::path::Path,
    name: Option<&str>,
    stream_type: Option<&str>,
    source_id: &str,
) -> Result<StreamSpec, LslIntegrationError> {
    let bytes = std::fs::read(lml_path)?;
    let header = lamquant_core::container::parse_header(&bytes)
        .map_err(LslIntegrationError::LmlDecode)?;

    let channel_count = u32::try_from(header.n_ch).map_err(|_| {
        LslIntegrationError::MissingMetadata(format!(
            "channel count {} doesn't fit in u32",
            header.n_ch
        ))
    })?;

    let metadata = &header.metadata;
    let sample_rate = parse_number_field(metadata, "sample_rate").unwrap_or_else(|| {
        let duration = parse_number_field(metadata, "duration_s").unwrap_or(1.0);
        if duration > 0.0 {
            header.total_samples as f64 / duration
        } else {
            256.0
        }
    });

    let channel_labels = parse_string_array_field(metadata, "channels");
    let channel_unit = extract_str_field(metadata, "phys_dim")
        .unwrap_or_else(|| "uV".to_string())
        .trim()
        .to_string();

    Ok(StreamSpec {
        name: name.unwrap_or("LamQuant").to_string(),
        stream_type: stream_type.unwrap_or("EEG").to_string(),
        channel_count,
        nominal_srate: sample_rate,
        channel_format: ChannelFormatLite::Int32,
        source_id: source_id.to_string(),
        channel_labels,
        channel_unit,
    })
}

// ─── Tiny zero-dep JSON field extractors ──────────────────────────
//
// The LamQuant metadata JSON is hand-written + flat (no nesting),
// so substring scans suffice and we avoid pulling serde_json into
// this crate just to read a handful of fields.

pub(crate) fn extract_str_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

pub(crate) fn parse_number_field(json: &str, field: &str) -> Option<f64> {
    let needle = format!("\"{}\":", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

pub(crate) fn parse_string_array_field(json: &str, field: &str) -> Vec<String> {
    let needle = format!("\"{}\":[", field);
    let Some(start) = json.find(&needle) else {
        return Vec::new();
    };
    let body_start = start + needle.len();
    let rest = &json[body_start..];
    let Some(end) = rest.find(']') else {
        return Vec::new();
    };
    let array_body = &rest[..end];
    let mut out = Vec::new();
    let mut cursor = 0;
    let bytes = array_body.as_bytes();
    while cursor < bytes.len() {
        if bytes[cursor] == b'"' {
            let value_start = cursor + 1;
            let mut value_end = value_start;
            while value_end < bytes.len() && bytes[value_end] != b'"' {
                if bytes[value_end] == b'\\' && value_end + 1 < bytes.len() {
                    value_end += 2;
                    continue;
                }
                value_end += 1;
            }
            if value_end <= bytes.len() {
                if let Ok(s) = std::str::from_utf8(&bytes[value_start..value_end]) {
                    out.push(s.to_string());
                }
            }
            cursor = value_end + 1;
        } else {
            cursor += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_number_field_basic() {
        let json = r#"{"sample_rate":256.5,"duration":10}"#;
        assert_eq!(parse_number_field(json, "sample_rate"), Some(256.5));
        assert_eq!(parse_number_field(json, "duration"), Some(10.0));
        assert_eq!(parse_number_field(json, "missing"), None);
    }

    #[test]
    fn parse_number_field_int_form() {
        let json = r#"{"n_channels":23,"trailing":0}"#;
        assert_eq!(parse_number_field(json, "n_channels"), Some(23.0));
    }

    #[test]
    fn parse_string_array_field_basic() {
        let json = r#"{"channels":["Fp1","F7","T3"],"other":1}"#;
        let labels = parse_string_array_field(json, "channels");
        assert_eq!(labels, vec!["Fp1", "F7", "T3"]);
    }

    #[test]
    fn parse_string_array_field_missing() {
        let json = r#"{"other":1}"#;
        assert!(parse_string_array_field(json, "channels").is_empty());
    }

    #[test]
    fn extract_str_field_basic() {
        let json = r#"{"phys_dim":"uV","other":"def"}"#;
        assert_eq!(extract_str_field(json, "phys_dim"), Some("uV".into()));
        assert_eq!(extract_str_field(json, "missing"), None);
    }

    #[test]
    fn from_channels_basic() {
        let spec = StreamSpec::from_channels(
            "MyAmp",
            256.0,
            vec!["Fp1".into(), "F7".into(), "T3".into()],
            "lamquant:abc123",
        )
        .expect("valid spec");
        assert_eq!(spec.name, "MyAmp");
        assert_eq!(spec.nominal_srate, 256.0);
        assert_eq!(spec.channel_count, 3);
        assert_eq!(spec.channel_format, ChannelFormatLite::Int32);
        assert_eq!(spec.channel_labels, vec!["Fp1", "F7", "T3"]);
        assert_eq!(spec.source_id, "lamquant:abc123");
        assert_eq!(spec.stream_type, "EEG");
        assert_eq!(spec.channel_unit, "uV");
    }

    #[test]
    fn from_channels_rejects_empty() {
        assert!(StreamSpec::from_channels("x", 256.0, Vec::new(), "id").is_err());
    }

    #[test]
    fn from_channels_rejects_negative_srate() {
        assert!(
            StreamSpec::from_channels("x", -1.0, vec!["ch0".into()], "id").is_err()
        );
    }

    #[test]
    fn from_channels_allows_irregular_rate() {
        // 0.0 == lsl::IRREGULAR_RATE — legitimate for marker streams.
        assert!(StreamSpec::from_channels("x", 0.0, vec!["ch0".into()], "id").is_ok());
    }
}
