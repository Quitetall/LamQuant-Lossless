//! Custom raw-binary reader with JSON sidecar.
//!
//! For users with proprietary acquisition pipelines (custom amplifiers,
//! lab-grown DAQ stacks, post-processed exports) whose data lives in
//! plain `.raw` byte files alongside a small JSON sidecar describing
//! shape and units. Schema:
//!
//! ```json
//! {
//!   "n_channels": 4,
//!   "sample_rate": 250.0,
//!   "dtype": "int16" | "int32",
//!   "orientation": "multiplexed" | "vectorized",
//!   "channels": ["Fp1", "Fp2", "C3", "C4"],
//!   "phys_min": [-200.0, -200.0, -200.0, -200.0],
//!   "phys_max": [ 200.0,  200.0,  200.0,  200.0],
//!   "phys_dim": "uV"
//! }
//! ```
//!
//! Sidecar lookup: `<basename>.json` first, then `<basename>.meta.json`.
//! `<basename>` is the `.raw` path minus its final extension.
//!
//! Bible alignment:
//!   - R1 — one format per file under `source/`
//!   - R23 — schema validation before allocating sample buffers
//!   - R30 — refuse mismatched lengths / unknown dtype / orientation
//!     with typed errors

use crate::error::{LmlError, LmlResult};
use std::path::{Path, PathBuf};

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};
use super::reader::SignalSourceReader;

const MAX_SIDECAR_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    Int16,
    Int32,
}

impl Dtype {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "int16" | "i16" => Some(Self::Int16),
            "int32" | "i32" => Some(Self::Int32),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Orientation {
    Multiplexed,
    Vectorized,
}

impl Orientation {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "multiplexed" => Some(Self::Multiplexed),
            "vectorized" => Some(Self::Vectorized),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct RawSidecar {
    n_channels: usize,
    sample_rate: f64,
    dtype: Dtype,
    orientation: Orientation,
    channels: Vec<String>,
    phys_min: Vec<f64>,
    phys_max: Vec<f64>,
    phys_dim: String,
}

fn parse_raw_sidecar(json: &str) -> LmlResult<RawSidecar> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| LmlError::InvalidHeader(format!("raw sidecar: not valid JSON ({e})")))?;
    let obj = v.as_object().ok_or_else(|| {
        LmlError::InvalidHeader("raw sidecar: must be a top-level JSON object".into())
    })?;

    let n_channels = obj
        .get("n_channels")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `n_channels`".into()))?
        as usize;
    if n_channels == 0 {
        return Err(LmlError::InvalidHeader(
            "raw sidecar: n_channels must be > 0".into(),
        ));
    }
    let sample_rate = obj
        .get("sample_rate")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `sample_rate`".into()))?;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "raw sidecar: sample_rate {sample_rate} must be finite and > 0"
        )));
    }
    let dtype_str = obj
        .get("dtype")
        .and_then(|x| x.as_str())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `dtype`".into()))?;
    let dtype = Dtype::from_str(dtype_str).ok_or_else(|| {
        LmlError::InvalidHeader(format!(
            "raw sidecar: dtype '{dtype_str}' not supported (use 'int16' or 'int32')"
        ))
    })?;
    let orientation_str = obj
        .get("orientation")
        .and_then(|x| x.as_str())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `orientation`".into()))?;
    let orientation = Orientation::from_str(orientation_str).ok_or_else(|| {
        LmlError::InvalidHeader(format!(
            "raw sidecar: orientation '{orientation_str}' not supported \
             (use 'multiplexed' or 'vectorized')"
        ))
    })?;
    let channels: Vec<String> = obj
        .get("channels")
        .and_then(|x| x.as_array())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `channels` array".into()))?
        .iter()
        .map(|c| c.as_str().unwrap_or("").to_string())
        .collect();
    if channels.len() != n_channels {
        return Err(LmlError::InvalidHeader(format!(
            "raw sidecar: channels.len {} != n_channels {n_channels}",
            channels.len()
        )));
    }
    let phys_min: Vec<f64> = obj
        .get("phys_min")
        .and_then(|x| x.as_array())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `phys_min` array".into()))?
        .iter()
        .filter_map(|c| c.as_f64())
        .collect();
    if phys_min.len() != n_channels {
        return Err(LmlError::InvalidHeader(format!(
            "raw sidecar: phys_min.len {} != n_channels {n_channels}",
            phys_min.len()
        )));
    }
    let phys_max: Vec<f64> = obj
        .get("phys_max")
        .and_then(|x| x.as_array())
        .ok_or_else(|| LmlError::InvalidHeader("raw sidecar: missing `phys_max` array".into()))?
        .iter()
        .filter_map(|c| c.as_f64())
        .collect();
    if phys_max.len() != n_channels {
        return Err(LmlError::InvalidHeader(format!(
            "raw sidecar: phys_max.len {} != n_channels {n_channels}",
            phys_max.len()
        )));
    }
    let phys_dim = obj
        .get("phys_dim")
        .and_then(|x| x.as_str())
        .unwrap_or("uV")
        .to_string();
    Ok(RawSidecar {
        n_channels,
        sample_rate,
        dtype,
        orientation,
        channels,
        phys_min,
        phys_max,
        phys_dim,
    })
}

/// Locate the sidecar JSON next to a `.raw` file. Tries `<stem>.json`,
/// `<stem>.meta.json`, and `<raw_path>.json` (suffix appended) in order.
fn locate_sidecar(raw_path: &Path) -> Option<PathBuf> {
    let candidates = [
        raw_path.with_extension("json"),
        raw_path.with_extension("meta.json"),
        {
            let mut p = raw_path.as_os_str().to_os_string();
            p.push(".json");
            PathBuf::from(p)
        },
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// `RawReader` — file-backed `SignalSourceReader` for `<basename>.raw`
/// + sibling sidecar JSON.
pub struct RawReader {
    raw_path: PathBuf,
    /// Override sidecar path. `None` triggers automatic sibling lookup.
    sidecar_override: Option<PathBuf>,
}

impl RawReader {
    pub fn new<P: Into<PathBuf>>(raw_path: P) -> Self {
        Self {
            raw_path: raw_path.into(),
            sidecar_override: None,
        }
    }
    pub fn with_sidecar<P: Into<PathBuf>>(mut self, sidecar_path: P) -> Self {
        self.sidecar_override = Some(sidecar_path.into());
        self
    }
}

impl SignalSourceReader for RawReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        let sidecar_path = self
            .sidecar_override
            .clone()
            .or_else(|| locate_sidecar(&self.raw_path))
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "raw reader: no sidecar JSON found next to {} \
                     (tried <stem>.json, <stem>.meta.json, <path>.json)",
                    self.raw_path.display()
                ))
            })?;
        let sidecar_meta = std::fs::metadata(&sidecar_path).map_err(LmlError::Io)?;
        if sidecar_meta.len() > MAX_SIDECAR_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "raw sidecar {} too large ({} bytes, max {})",
                sidecar_path.display(),
                sidecar_meta.len(),
                MAX_SIDECAR_BYTES
            )));
        }
        let sidecar_bytes = std::fs::read(&sidecar_path).map_err(LmlError::Io)?;
        let sidecar_text = std::str::from_utf8(&sidecar_bytes)
            .map_err(|e| LmlError::InvalidHeader(format!("raw sidecar: invalid UTF-8 ({e})")))?;
        let sc = parse_raw_sidecar(sidecar_text)?;

        let raw_meta = std::fs::metadata(&self.raw_path).map_err(LmlError::Io)?;
        let raw_len = raw_meta.len() as usize;
        let bps = sc.dtype.bytes_per_sample();
        if raw_len % (sc.n_channels * bps) != 0 {
            return Err(LmlError::InvalidHeader(format!(
                "raw {}: length {} not multiple of n_channels * bps ({} * {})",
                self.raw_path.display(),
                raw_len,
                sc.n_channels,
                bps
            )));
        }
        let n_samples = raw_len / (sc.n_channels * bps);
        let raw_bytes = std::fs::read(&self.raw_path).map_err(LmlError::Io)?;

        let mut signal: Vec<Vec<i64>> = (0..sc.n_channels)
            .map(|_| Vec::with_capacity(n_samples))
            .collect();
        match (sc.dtype, sc.orientation) {
            (Dtype::Int16, Orientation::Multiplexed) => {
                for s in 0..n_samples {
                    let base = s * sc.n_channels * 2;
                    for ch in 0..sc.n_channels {
                        let off = base + ch * 2;
                        let v = i16::from_le_bytes([raw_bytes[off], raw_bytes[off + 1]]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (Dtype::Int16, Orientation::Vectorized) => {
                for ch in 0..sc.n_channels {
                    let ch_base = ch * n_samples * 2;
                    for s in 0..n_samples {
                        let off = ch_base + s * 2;
                        let v = i16::from_le_bytes([raw_bytes[off], raw_bytes[off + 1]]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (Dtype::Int32, Orientation::Multiplexed) => {
                for s in 0..n_samples {
                    let base = s * sc.n_channels * 4;
                    for ch in 0..sc.n_channels {
                        let off = base + ch * 4;
                        let v = i32::from_le_bytes([
                            raw_bytes[off],
                            raw_bytes[off + 1],
                            raw_bytes[off + 2],
                            raw_bytes[off + 3],
                        ]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
            (Dtype::Int32, Orientation::Vectorized) => {
                for ch in 0..sc.n_channels {
                    let ch_base = ch * n_samples * 4;
                    for s in 0..n_samples {
                        let off = ch_base + s * 4;
                        let v = i32::from_le_bytes([
                            raw_bytes[off],
                            raw_bytes[off + 1],
                            raw_bytes[off + 2],
                            raw_bytes[off + 3],
                        ]) as i64;
                        signal[ch].push(v);
                    }
                }
            }
        }

        let duration_s = if sc.sample_rate > 0.0 {
            n_samples as f64 / sc.sample_rate
        } else {
            0.0
        };

        let bundle = SignalBundle {
            signal,
            sample_rate: sc.sample_rate,
            channels: sc.channels,
            phys_min: sc.phys_min,
            phys_max: sc.phys_max,
            duration_s,
            metadata: SourceMetadata {
                source_file: crate::source::bundle::source_basename(&self.raw_path),
                format: "RAW".to_string(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim: sc.phys_dim,
            },
            sidecar: vec![
                SidecarBlob {
                    key: "raw_sidecar_json".to_string(),
                    bytes: sidecar_bytes,
                    aux: None,
                },
                // Byte-exact preservation of the `.raw` payload itself.
                // The codec losslessly reconstructs the signal from the
                // i64 sample matrix + sidecar JSON, so this entry is
                // strictly redundant for roundtrip. We carry it for
                // the "no byte ever lost" invariant: an `lml extract`
                // of a per-recording `.lma` recovers the literal `.raw`
                // bytes the user gave us, regardless of dtype /
                // orientation reconstruction quirks.
                SidecarBlob {
                    key: "raw_payload_raw".to_string(),
                    bytes: raw_bytes,
                    aux: None,
                },
                // Filename anchor so the encoder can write the
                // preservation copy with the exact source filename
                // (not just `<lml_stem>.raw`).
                SidecarBlob {
                    key: "raw_payload_filename".to_string(),
                    bytes: self
                        .raw_path
                        .file_name()
                        .map(|n| n.to_string_lossy().as_bytes().to_vec())
                        .unwrap_or_default(),
                    aux: None,
                },
                // Filename of the actual sidecar JSON we located
                // (`<stem>.json` / `<stem>.meta.json` / etc.) so the
                // encoder can preserve that exact name on extract.
                SidecarBlob {
                    key: "raw_sidecar_filename".to_string(),
                    bytes: sidecar_path
                        .file_name()
                        .map(|n| n.to_string_lossy().as_bytes().to_vec())
                        .unwrap_or_default(),
                    aux: None,
                },
            ],
        };
        bundle.validate()?;
        Ok(bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_sidecar(n_ch: usize, dtype: &str, orient: &str) -> String {
        let chans: Vec<String> = (0..n_ch).map(|i| format!("\"ch{i}\"")).collect();
        let pmin = vec!["-200.0"; n_ch].join(",");
        let pmax = vec!["200.0"; n_ch].join(",");
        format!(
            "{{\"n_channels\":{n_ch},\"sample_rate\":250.0,\"dtype\":\"{dtype}\",\
             \"orientation\":\"{orient}\",\"channels\":[{}],\
             \"phys_min\":[{pmin}],\"phys_max\":[{pmax}],\"phys_dim\":\"uV\"}}",
            chans.join(",")
        )
    }

    #[test]
    fn parse_sidecar_extracts_fields() {
        let sc = parse_raw_sidecar(&good_sidecar(2, "int16", "multiplexed")).unwrap();
        assert_eq!(sc.n_channels, 2);
        assert_eq!(sc.dtype, Dtype::Int16);
        assert_eq!(sc.orientation, Orientation::Multiplexed);
    }

    #[test]
    fn parse_sidecar_rejects_unknown_dtype() {
        let json = good_sidecar(1, "float32", "multiplexed");
        match parse_raw_sidecar(&json) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("float32")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn parse_sidecar_rejects_length_mismatch() {
        let json = r#"{"n_channels":3,"sample_rate":250.0,"dtype":"int16","orientation":"multiplexed","channels":["a","b"],"phys_min":[-1.0,-1.0,-1.0],"phys_max":[1.0,1.0,1.0],"phys_dim":"uV"}"#;
        match parse_raw_sidecar(json) {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("channels.len")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_bundle_int16_multiplexed_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 3usize;
        let n_samples = 200usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(n_ch * n_samples * 2);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                let v = (s as i16) * (ch as i16 + 1);
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let raw = tmp.path().join("data.raw");
        let json = tmp.path().join("data.json");
        std::fs::write(&raw, &bytes).unwrap();
        std::fs::write(&json, good_sidecar(n_ch, "int16", "multiplexed")).unwrap();
        let mut reader = RawReader::new(&raw);
        let b = reader.read_bundle().unwrap();
        assert_eq!(b.signal.len(), n_ch);
        assert_eq!(b.signal[0].len(), n_samples);
        for s in 0..n_samples {
            for ch in 0..n_ch {
                assert_eq!(b.signal[ch][s], (s as i64) * (ch as i64 + 1));
            }
        }
        assert_eq!(b.metadata.format, "RAW");
    }

    #[test]
    fn read_bundle_missing_sidecar_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let raw = tmp.path().join("nope.raw");
        std::fs::write(&raw, b"\0\0\0\0").unwrap();
        let mut reader = RawReader::new(&raw);
        match reader.read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("no sidecar JSON")),
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    // ─── ADR 0069 L7 gate: lower_to_legacy_recording byte-exactness ─────────────

    // ─── ADR 0069 S3b gate: born-typed lowering (modality inference) ───
}
