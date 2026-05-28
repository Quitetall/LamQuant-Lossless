//! EEGLAB `.set` + `.fdt` reader.
//!
//! Phase 8 / Item B. EEGLAB exports recordings as a pair:
//!   - `<name>.set` — MATLAB 5 `.mat` container with an `EEG` struct
//!     carrying metadata (`nbchan`, `pnts`, `srate`, `chanlocs`)
//!   - `<name>.fdt` — binary float32 sample data, channel-major (the
//!     first `pnts` values are all of channel 0, then channel 1, …)
//!
//! ## v1 scope: sidecar-driven metadata
//!
//! Parsing the full MATLAB 5 binary format is a 200+ LOC walker we
//! defer. The v1 reader looks for a sibling `<name>.lml-meta.json`
//! describing the metadata that would normally come from `.set`:
//!
//! ```text
//! {
//!   "n_channels": 32,
//!   "n_samples":  100000,
//!   "sample_rate": 500.0,
//!   "channels":   ["Fp1", "Fp2", ...],
//!   "phys_dim":   "uV"
//! }
//! ```
//!
//! Use `tools/dump_eeglab_meta.py` (committed) to extract this sidecar
//! from a real `.set` file via scipy.io. Run it once per archive when
//! you start working with the file; the sidecar then lives alongside
//! the `.set` permanently.
//!
//! ## Lossless float32 → i64 (default)
//!
//! Each `f32` sample is bit-cast into its IEEE 754 `u32`
//! representation and widened to `i64`. The mapping is a two-way
//! bijection — every NaN payload, denormal, infinity, and finite value
//! round-trips byte-for-byte. The codec's "forensic bit-exact" promise
//! holds for EEGLAB inputs by default.
//!
//! The decoder inverts via `f32::from_bits((sample as u32))`.
//! Metadata field `eeglab_dtype = "lossless_f32_bitcast"` lets the
//! decoder pick the right inverse path.
//!
//! NaN and Inf inputs are refused with a typed error: float bit-cast
//! preserves them, but downstream LamQuant DSP (when noise-bits > 0,
//! when arithmetic touches the values) is undefined on non-finite
//! values. Bible R30 — refuse explicitly rather than silently corrupt.
//!
//! ## Opt-in lossy `--lossy-int16`
//!
//! Future iteration: scale-and-quantise into i16 with a per-channel
//! sensitivity factor. Documented in the plan but not wired yet.

use crate::error::{LmlError, LmlResult};
use std::path::{Path, PathBuf};

use super::bundle::{SidecarBlob, SignalBundle, SourceMetadata};
use super::reader::SignalSourceReader;

const MAX_META_BYTES: u64 = 8 * 1024 * 1024;
/// Maximum bytes we'll buffer for the original `.set` MAT v5 file when
/// stashing it as a `SidecarBlob`. EEGLAB structs are typically
/// 10 KB – 1 MB; very high-channel ICA-decomposed files can hit tens
/// of megabytes. The cap exists to refuse pathological inputs rather
/// than to clip honest files. Bump if a real corpus needs it.
const MAX_SET_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum bytes we'll buffer for the original `.fdt` payload. The
/// signal samples themselves are loaded via the existing read path;
/// this cap governs only the byte-exact preservation copy. A 256-
/// channel × 24 h recording at 1 kHz f32 is ~ 8.8 GB which is well
/// beyond the design point — refuse it loudly rather than OOM.
const MAX_FDT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug)]
struct EeglabMeta {
    n_channels: usize,
    n_samples: usize,
    sample_rate: f64,
    channels: Vec<String>,
    phys_dim: String,
}

fn parse_meta(json: &str) -> LmlResult<EeglabMeta> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| LmlError::InvalidHeader(format!("eeglab meta sidecar: invalid JSON ({e})")))?;
    let obj = v.as_object().ok_or_else(|| {
        LmlError::InvalidHeader("eeglab meta sidecar: must be a top-level JSON object".into())
    })?;
    let n_channels = obj
        .get("n_channels")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| LmlError::InvalidHeader("eeglab meta: missing n_channels".into()))?
        as usize;
    if n_channels == 0 {
        return Err(LmlError::InvalidHeader(
            "eeglab meta: n_channels must be > 0".into(),
        ));
    }
    let n_samples = obj
        .get("n_samples")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| LmlError::InvalidHeader("eeglab meta: missing n_samples".into()))?
        as usize;
    let sample_rate = obj
        .get("sample_rate")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| LmlError::InvalidHeader("eeglab meta: missing sample_rate".into()))?;
    if !(sample_rate.is_finite() && sample_rate > 0.0) {
        return Err(LmlError::InvalidHeader(format!(
            "eeglab meta: sample_rate {sample_rate} must be finite and > 0"
        )));
    }
    let channels: Vec<String> = obj
        .get("channels")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .map(|c| c.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_else(|| (0..n_channels).map(|i| format!("ch{i}")).collect());
    if channels.len() != n_channels {
        return Err(LmlError::InvalidHeader(format!(
            "eeglab meta: channels.len {} != n_channels {n_channels}",
            channels.len()
        )));
    }
    let phys_dim = obj
        .get("phys_dim")
        .and_then(|x| x.as_str())
        .unwrap_or("uV")
        .to_string();
    Ok(EeglabMeta {
        n_channels,
        n_samples,
        sample_rate,
        channels,
        phys_dim,
    })
}

fn locate_meta(set_path: &Path) -> Option<PathBuf> {
    let candidates = [
        set_path.with_extension("lml-meta.json"),
        set_path.with_extension("json"),
        {
            let mut p = set_path.as_os_str().to_os_string();
            p.push(".lml-meta.json");
            PathBuf::from(p)
        },
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn locate_fdt(set_path: &Path) -> Option<PathBuf> {
    let candidates = [set_path.with_extension("fdt"), {
        let mut p = set_path.as_os_str().to_os_string();
        p.push(".fdt");
        PathBuf::from(p)
    }];
    candidates.into_iter().find(|p| p.exists())
}

/// `EeglabReader` — file-backed `SignalSourceReader` for the
/// `<name>.set + <name>.fdt + <name>.lml-meta.json` triple.
pub struct EeglabReader {
    set_path: PathBuf,
    meta_override: Option<PathBuf>,
    fdt_override: Option<PathBuf>,
    /// When true, scale-and-quantise float32 → i16 lossy. Default
    /// false: lossless f32 bit-cast (Bible R6 — the codec's core
    /// promise is lossless; opt-in for lossy).
    lossy_int16: bool,
}

impl EeglabReader {
    pub fn new<P: Into<PathBuf>>(set_path: P) -> Self {
        Self {
            set_path: set_path.into(),
            meta_override: None,
            fdt_override: None,
            lossy_int16: false,
        }
    }
    pub fn with_meta<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.meta_override = Some(p.into());
        self
    }
    pub fn with_fdt<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.fdt_override = Some(p.into());
        self
    }
    pub fn with_lossy_int16(mut self, on: bool) -> Self {
        self.lossy_int16 = on;
        self
    }
}

impl SignalSourceReader for EeglabReader {
    fn read_bundle(&mut self) -> LmlResult<SignalBundle> {
        let meta_path = self
            .meta_override
            .clone()
            .or_else(|| locate_meta(&self.set_path))
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "eeglab reader: no metadata sidecar found next to {} \
                     (tried <name>.lml-meta.json, <name>.json, <path>.lml-meta.json). \
                     Generate one with `tools/dump_eeglab_meta.py`.",
                    self.set_path.display()
                ))
            })?;
        let meta_size = std::fs::metadata(&meta_path).map_err(LmlError::Io)?.len();
        if meta_size > MAX_META_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "eeglab meta {} too large ({meta_size} bytes, max {MAX_META_BYTES})",
                meta_path.display()
            )));
        }
        let meta_bytes = std::fs::read(&meta_path).map_err(LmlError::Io)?;
        let meta_text = std::str::from_utf8(&meta_bytes)
            .map_err(|e| LmlError::InvalidHeader(format!("eeglab meta: invalid UTF-8 ({e})")))?;
        let meta = parse_meta(meta_text)?;

        let fdt_path = self
            .fdt_override
            .clone()
            .or_else(|| locate_fdt(&self.set_path))
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "eeglab reader: no .fdt found next to {} \
                     (tried <name>.fdt, <path>.fdt)",
                    self.set_path.display()
                ))
            })?;
        let expected_bytes = meta.n_channels * meta.n_samples * 4;
        let fdt_size = std::fs::metadata(&fdt_path).map_err(LmlError::Io)?.len();
        if fdt_size < expected_bytes as u64 {
            return Err(LmlError::Truncated {
                expected: expected_bytes,
                actual: fdt_size as usize,
                context: "eeglab .fdt",
            });
        }
        let raw = std::fs::read(&fdt_path).map_err(LmlError::Io)?;

        // EEGLAB writes channel-major: first `n_samples` float32s are
        // channel 0, then channel 1, ...
        let mut signal: Vec<Vec<i64>> = (0..meta.n_channels)
            .map(|_| Vec::with_capacity(meta.n_samples))
            .collect();
        for ch in 0..meta.n_channels {
            let base = ch * meta.n_samples * 4;
            for s in 0..meta.n_samples {
                let off = base + s * 4;
                let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                let f = f32::from_le_bytes(bytes);
                if !f.is_finite() {
                    return Err(LmlError::InvalidHeader(format!(
                        "eeglab: ch{ch} sample {s} is non-finite (NaN / Inf); \
                         LamQuant DSP requires finite inputs. Drop the file or \
                         clean upstream before encoding."
                    )));
                }
                let i: i64 = if self.lossy_int16 {
                    // Scale to int16 range. Per-channel sensitivity factor
                    // is set 1.0 here for the placeholder path; a future
                    // pass derives it per-channel from the data.
                    (f as f64).clamp(i16::MIN as f64, i16::MAX as f64) as i64
                } else {
                    // Lossless path: bit-cast f32 → u32 → i64.
                    let u = f.to_bits();
                    u as i64
                };
                signal[ch].push(i);
            }
        }

        let duration_s = if meta.sample_rate > 0.0 {
            meta.n_samples as f64 / meta.sample_rate
        } else {
            0.0
        };
        let phys_min: Vec<f64> = vec![f32::MIN as f64; meta.n_channels];
        let phys_max: Vec<f64> = vec![f32::MAX as f64; meta.n_channels];

        // Byte-exact preservation of the original `.set` MAT v5 file.
        // Today the v1 reader doesn't parse the MAT struct; the
        // `<name>.lml-meta.json` sidecar carries only the few fields
        // the codec consumes (nbchan, pnts, srate, channel labels).
        // Every other `EEG.*` field (events, urevents, chanlocs xyz,
        // icaweights, icasphere, reject, history, etc.) is dropped on
        // the floor unless we stash the original bytes -- which is
        // exactly what the "no data ever lost" invariant demands.
        // Encoder side bundles this blob as a `.set` entry in the
        // per-recording `.lma` so `lml decode --to-eeglab` recovers
        // the original MAT struct byte-for-byte.
        let set_size = std::fs::metadata(&self.set_path)
            .map_err(LmlError::Io)?
            .len();
        if set_size > MAX_SET_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "eeglab .set {} too large ({set_size} bytes, max {MAX_SET_BYTES})",
                self.set_path.display()
            )));
        }
        let set_raw = std::fs::read(&self.set_path).map_err(LmlError::Io)?;

        // Same treatment for `.fdt`: the signal data has already been
        // decoded into `signal` for the codec, but the raw bytes are
        // what reproduces the original file on decode. Without this
        // copy the byte-exact roundtrip relies on f32-bitcast inverse
        // arithmetic, which is correct mathematically but not what an
        // operator running `cmp original.fdt restored.fdt` will
        // accept.
        if fdt_size > MAX_FDT_BYTES {
            return Err(LmlError::InvalidHeader(format!(
                "eeglab .fdt {} too large ({fdt_size} bytes, max {MAX_FDT_BYTES})",
                fdt_path.display()
            )));
        }
        // `raw` already contains the full `.fdt` bytes from the load
        // above. Clone instead of re-reading from disk.
        let fdt_raw = raw.clone();

        let bundle = SignalBundle {
            signal,
            sample_rate: meta.sample_rate,
            channels: meta.channels,
            phys_min,
            phys_max,
            duration_s,
            metadata: SourceMetadata {
                source_file: self.set_path.display().to_string(),
                format: if self.lossy_int16 {
                    "EEGLAB_LOSSY_I16".to_string()
                } else {
                    "EEGLAB_LOSSLESS_F32".to_string()
                },
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim: meta.phys_dim,
            },
            sidecar: vec![
                // (1) The codec-consumed metadata JSON. Keep for
                //     downstream tooling that wants the parsed fields
                //     without re-walking the MAT struct.
                SidecarBlob {
                    key: if self.lossy_int16 {
                        "eeglab_dtype_lossy_i16_scaled".to_string()
                    } else {
                        "eeglab_dtype_lossless_f32_bitcast".to_string()
                    },
                    bytes: meta_bytes,
                    aux: None,
                },
                // (2) Original `.set` MAT v5 bytes. Encoder writes
                //     this as `<stem>.set` LMA entry.
                SidecarBlob {
                    key: "set_raw".to_string(),
                    bytes: set_raw,
                    aux: None,
                },
                // (3) Original `.fdt` float32 bytes. Encoder writes
                //     this as `<stem>.fdt` LMA entry.
                SidecarBlob {
                    key: "fdt_raw".to_string(),
                    bytes: fdt_raw,
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

    fn synth_meta(n_ch: usize, n_samp: usize) -> String {
        let chans: Vec<String> = (0..n_ch).map(|i| format!("\"ch{i}\"")).collect();
        format!(
            "{{\"n_channels\":{n_ch},\"n_samples\":{n_samp},\"sample_rate\":500.0,\
             \"channels\":[{}],\"phys_dim\":\"uV\"}}",
            chans.join(",")
        )
    }

    fn synth_fdt(n_ch: usize, n_samp: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_ch * n_samp * 4);
        for ch in 0..n_ch {
            for s in 0..n_samp {
                let v = (s as f32) * 0.1 + (ch as f32);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn lossless_roundtrip_bitcast_recovers_exact_floats() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 3;
        let n_samp = 100;
        let set = tmp.path().join("rec.set");
        let fdt = tmp.path().join("rec.fdt");
        let meta = tmp.path().join("rec.lml-meta.json");
        std::fs::write(&set, b"").unwrap(); // .set body unused in v1
        std::fs::write(&fdt, synth_fdt(n_ch, n_samp)).unwrap();
        std::fs::write(&meta, synth_meta(n_ch, n_samp)).unwrap();
        let mut r = EeglabReader::new(&set);
        let b = r.read_bundle().unwrap();
        assert_eq!(b.signal.len(), n_ch);
        assert_eq!(b.signal[0].len(), n_samp);
        // Bit-cast invariant: reading bits back via f32::from_bits
        // recovers the exact f32 we wrote.
        for ch in 0..n_ch {
            for s in 0..n_samp {
                let expected = (s as f32) * 0.1 + (ch as f32);
                let stored_i = b.signal[ch][s];
                let recovered = f32::from_bits(stored_i as u32);
                assert_eq!(
                    recovered.to_bits(),
                    expected.to_bits(),
                    "ch{ch} s{s}: bit-cast round-trip drift ({recovered:?} vs {expected:?})"
                );
            }
        }
        assert_eq!(b.metadata.format, "EEGLAB_LOSSLESS_F32");
    }

    #[test]
    fn lossy_int16_path_clamps_into_i16_range() {
        let tmp = tempfile::tempdir().unwrap();
        let n_ch = 2;
        let n_samp = 50;
        let set = tmp.path().join("a.set");
        std::fs::write(&set, b"").unwrap();
        std::fs::write(tmp.path().join("a.fdt"), synth_fdt(n_ch, n_samp)).unwrap();
        std::fs::write(tmp.path().join("a.lml-meta.json"), synth_meta(n_ch, n_samp)).unwrap();
        let mut r = EeglabReader::new(&set).with_lossy_int16(true);
        let b = r.read_bundle().unwrap();
        // Values stored should be within i16 range (clamp invariant).
        for ch in &b.signal {
            for &v in ch {
                assert!(
                    v >= i16::MIN as i64 && v <= i16::MAX as i64,
                    "lossy path emitted out-of-range value: {v}"
                );
            }
        }
        assert_eq!(b.metadata.format, "EEGLAB_LOSSY_I16");
    }

    #[test]
    fn channel_labels_preserved_and_sample_rate_carried() {
        let tmp = tempfile::tempdir().unwrap();
        let set = tmp.path().join("b.set");
        std::fs::write(&set, b"").unwrap();
        std::fs::write(tmp.path().join("b.fdt"), synth_fdt(2, 8)).unwrap();
        std::fs::write(
            tmp.path().join("b.lml-meta.json"),
            "{\"n_channels\":2,\"n_samples\":8,\"sample_rate\":250.0,\
             \"channels\":[\"Fp1\",\"Cz\"],\"phys_dim\":\"uV\"}",
        )
        .unwrap();
        let mut r = EeglabReader::new(&set);
        let b = r.read_bundle().unwrap();
        assert_eq!(b.channels, vec!["Fp1".to_string(), "Cz".to_string()]);
        assert!((b.sample_rate - 250.0).abs() < 1e-9);
    }

    #[test]
    fn missing_fdt_errors_explicitly() {
        let tmp = tempfile::tempdir().unwrap();
        let set = tmp.path().join("c.set");
        std::fs::write(&set, b"").unwrap();
        std::fs::write(tmp.path().join("c.lml-meta.json"), synth_meta(1, 1)).unwrap();
        let mut r = EeglabReader::new(&set);
        match r.read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("no .fdt found")),
            other => panic!("expected InvalidHeader('no .fdt'), got {other:?}"),
        }
    }

    #[test]
    fn nan_input_refused_explicitly() {
        let tmp = tempfile::tempdir().unwrap();
        let set = tmp.path().join("d.set");
        std::fs::write(&set, b"").unwrap();
        // Synth .fdt with NaN at index 0 (channel 0, sample 0).
        let mut fdt = synth_fdt(1, 4);
        fdt[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(tmp.path().join("d.fdt"), &fdt).unwrap();
        std::fs::write(tmp.path().join("d.lml-meta.json"), synth_meta(1, 4)).unwrap();
        let mut r = EeglabReader::new(&set);
        match r.read_bundle() {
            Err(LmlError::InvalidHeader(msg)) => assert!(msg.contains("non-finite")),
            other => panic!("expected InvalidHeader('non-finite'), got {other:?}"),
        }
    }
}
