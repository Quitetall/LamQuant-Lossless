//! Deterministic LSL `source_id` generation.
//!
//! LSL identifies streams by their `source_id`. Two streams sharing
//! a source_id are considered duplicates by LabRecorder + other
//! consumers — they'll record only one, deduplicating noisy
//! networks. For LamQuant replay this is exactly the property we
//! want: the same `.lml` file always produces the same UID, so
//! restarting a replay doesn't create a duplicate recording in the
//! consumer's session.
//!
//! The UID is derived from the LML container's `signal_sha256`
//! metadata field — a hash over the channel data the encoder
//! computed at write time. Same file content → same UID;
//! different recordings → different UIDs.

use sha2::{Digest, Sha256};

/// Length of the UID returned (hex characters). 16 hex chars = 64
/// bits of source-id entropy — overkill for collision avoidance
/// across any realistic deployment, short enough to log + paste.
const UID_HEX_LEN: usize = 16;

/// Build a deterministic LSL `source_id` from an already-extracted
/// signal SHA-256 hex string. This is the lower-level form;
/// [`stream_id_from_lml`] is the convenience wrapper that reads
/// the SHA out of an `.lml` file on disk.
///
/// The output is namespaced with the `lamquant:` prefix so a
/// shared LSL network with many sources shows LamQuant-emitted
/// streams clearly in LabRecorder's source picker.
pub fn stream_id_from_components(signal_sha256_hex: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(signal_sha256_hex.as_bytes());
    let full = hasher.finalize();
    let hex = format!("{:x}", full);
    format!("lamquant:{}", &hex[..UID_HEX_LEN])
}

/// Build a UID directly from an `.lml` file path. Parses the
/// container header + reads the `signal_sha256` field from the
/// embedded metadata JSON. Phase-1 implementation reads the whole
/// file; a future optimisation reads only the header.
pub fn stream_id_from_lml(
    lml_path: &std::path::Path,
) -> Result<String, crate::error::LslIntegrationError> {
    let bytes = std::fs::read(lml_path)?;
    let header = lamquant_core::container::parse_header(&bytes)
        .map_err(crate::error::LslIntegrationError::LmlDecode)?;
    // The metadata field is a JSON string serialised by
    // `encode_edf_to_lml`. Pull out `signal_sha256` (hex string).
    // Falls back to a marker UID if the field is absent — old
    // archives without the field still get a stable id.
    let signal_sha = extract_str_field(&header.metadata, "signal_sha256")
        .unwrap_or_else(|| "no-sha-in-metadata".to_string());
    Ok(stream_id_from_components(&signal_sha))
}

/// Tiny zero-dep JSON string-value extractor. The LamQuant metadata
/// JSON is hand-written + flat (no nesting), so a substring search
/// is sufficient. Avoids pulling serde_json into the crate just to
/// read one field. Returns `None` if the field is missing or its
/// value isn't a quoted string.
fn extract_str_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_yield_same_id() {
        let a = stream_id_from_components("abc123");
        let b = stream_id_from_components("abc123");
        assert_eq!(a, b);
    }

    #[test]
    fn different_signal_yields_different_id() {
        let a = stream_id_from_components("abc123");
        let b = stream_id_from_components("def456");
        assert_ne!(a, b);
    }

    #[test]
    fn id_has_lamquant_prefix() {
        let id = stream_id_from_components("abc");
        assert!(id.starts_with("lamquant:"));
    }

    #[test]
    fn id_hex_length() {
        let id = stream_id_from_components("abc");
        // "lamquant:" prefix + 16 hex chars = 25.
        assert_eq!(id.len(), "lamquant:".len() + UID_HEX_LEN);
    }

    #[test]
    fn extract_str_field_basic() {
        let json = r#"{"signal_sha256":"abc123","other":"def"}"#;
        assert_eq!(extract_str_field(json, "signal_sha256"), Some("abc123".into()));
        assert_eq!(extract_str_field(json, "other"), Some("def".into()));
        assert_eq!(extract_str_field(json, "missing"), None);
    }
}
