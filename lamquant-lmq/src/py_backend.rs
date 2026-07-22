//! ADR 0074 Track N — `PyBackend`: drives the Python `SubbandCodec` inference over
//! a SUBPROCESS (the only Rust→Python precedent in the repo — see
//! `training/backends/.../python_backend.rs`; NOT in-process pyo3 embedding).
//!
//! The shell owns the wire; this backend owns only the network, in Python, for now
//! (fast R&D). It spawns a Python helper, exchanges a JSON request/response over
//! stdin/stdout (numeric arrays inline; `backend_meta` as a byte array), and maps
//! the result into [`NeuralTokens`] / a reconstructed signal. Swapping in the Rust
//! backend later is a drop-in — the wire never changes.
//!
//! Host-only (feature `python`): needs `std` (process) + `serde_json`. codec-neural
//! is imported by the helper, never edited; weights resolve via `$LAMQUANT_WEIGHTS_DIR`.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::string::{String, ToString};
use std::vec::Vec;

use semantic_abir_bcs::ModelProvenance;
use serde_json::{json, Value};

use crate::backend::{BackendError, NeuralBackend, NeuralTokens};

/// A subprocess-driven Python neural backend.
pub struct PyBackend {
    /// The Python executable (e.g. `"python3"`).
    python: String,
    /// Path to the inference helper script (`lmq_infer.py`).
    helper: PathBuf,
    /// `"selftest"` (deterministic, no weights — proves the bridge) or `"model"`
    /// (the real `SubbandCodec`, env-gated).
    mode: String,
    model: ModelProvenance,
}

impl PyBackend {
    /// Drive the real `SubbandCodec` (`mode = "model"`).
    pub fn model(
        python: impl Into<String>,
        helper: impl Into<PathBuf>,
        model: ModelProvenance,
    ) -> Self {
        Self {
            python: python.into(),
            helper: helper.into(),
            mode: "model".to_string(),
            model,
        }
    }
    /// Drive the helper's deterministic self-test transform (`mode = "selftest"`) —
    /// no weights, for verifying the subprocess bridge itself.
    pub fn selftest(
        python: impl Into<String>,
        helper: impl Into<PathBuf>,
        model: ModelProvenance,
    ) -> Self {
        Self {
            python: python.into(),
            helper: helper.into(),
            mode: "selftest".to_string(),
            model,
        }
    }

    // NB: `call` blocks on `wait_with_output` with NO timeout — a hung helper
    // (model-load stall, infinite loop) blocks the caller. Acceptable for the R&D
    // path; a watchdog-thread kill is a tracked follow-up before any unattended use.
    fn call(&self, mut request: Value) -> Result<Value, BackendError> {
        request["mode"] = json!(self.mode);
        if self.mode == "model" {
            request["expected_checkpoint_sha256"] =
                json!(encode_hex(&self.model.checkpoint_sha256));
        }
        let child = Command::new(&self.python)
            .arg(&self.helper)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                BackendError(format!(
                    "spawn `{} {}`: {e}",
                    self.python,
                    self.helper.display()
                ))
            })?;
        {
            use std::io::Write;
            let mut stdin = child
                .stdin
                .as_ref()
                .ok_or_else(|| BackendError("no child stdin".to_string()))?;
            let buf = serde_json::to_vec(&request)
                .map_err(|e| BackendError(format!("serialize request: {e}")))?;
            stdin
                .write_all(&buf)
                .map_err(|e| BackendError(format!("write request: {e}")))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| BackendError(format!("wait for helper: {e}")))?;
        if !out.status.success() {
            return Err(BackendError(format!(
                "helper exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        let response: Value = serde_json::from_slice(&out.stdout).map_err(|e| {
            BackendError(format!(
                "parse response: {e} (stderr: {})",
                String::from_utf8_lossy(&out.stderr)
            ))
        })?;
        if self.mode == "model"
            && response.get("checkpoint_sha256").and_then(Value::as_str)
                != Some(encode_hex(&self.model.checkpoint_sha256).as_str())
        {
            return Err(BackendError(
                "helper executed a checkpoint different from model provenance".to_string(),
            ));
        }
        Ok(response)
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

impl NeuralBackend for PyBackend {
    fn model_provenance(&self) -> ModelProvenance {
        self.model.clone()
    }

    fn encode(&self, signal: &[Vec<i64>], sample_rate: f64) -> Result<NeuralTokens, BackendError> {
        let resp = self.call(json!({
            "op": "encode",
            "sample_rate": sample_rate,
            "signal": signal,
        }))?;
        Ok(NeuralTokens {
            tokens: i32_array(&resp, "tokens")?,
            schedule: u8_array(&resp, "schedule")?,
            alphabet: u16_field(&resp, "alphabet")?,
            n_channels: u16_field(&resp, "n_channels")?,
            n_samples: u32_field(&resp, "n_samples")?,
            backend_meta: u8_array(&resp, "backend_meta")?,
        })
    }

    fn decode(&self, t: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
        let resp = self.call(json!({
            "op": "decode",
            "tokens": t.tokens,
            "schedule": t.schedule,
            "alphabet": t.alphabet,
            "n_channels": t.n_channels,
            "n_samples": t.n_samples,
            "backend_meta": t.backend_meta,
        }))?;
        let rows = resp
            .get("signal")
            .and_then(|v| v.as_array())
            .ok_or_else(|| BackendError("response missing `signal` array".to_string()))?;
        rows.iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| BackendError("signal row is not an array".to_string()))?
                    .iter()
                    .map(|x| {
                        x.as_i64()
                            .ok_or_else(|| BackendError("signal sample not an i64".to_string()))
                    })
                    .collect::<Result<Vec<i64>, _>>()
            })
            .collect()
    }
}

fn u64_field(v: &Value, key: &str) -> Result<u64, BackendError> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .ok_or_else(|| BackendError(format!("response missing u64 field `{key}`")))
}

fn u16_field(v: &Value, key: &str) -> Result<u16, BackendError> {
    u16::try_from(u64_field(v, key)?).map_err(|_| BackendError(format!("`{key}` out of u16 range")))
}

fn u32_field(v: &Value, key: &str) -> Result<u32, BackendError> {
    u32::try_from(u64_field(v, key)?).map_err(|_| BackendError(format!("`{key}` out of u32 range")))
}

fn i32_array(v: &Value, key: &str) -> Result<Vec<i32>, BackendError> {
    v.get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| BackendError(format!("response missing array `{key}`")))?
        .iter()
        .map(|x| {
            let n = x
                .as_i64()
                .ok_or_else(|| BackendError(format!("`{key}`: element not an int")))?;
            i32::try_from(n)
                .map_err(|_| BackendError(format!("`{key}`: element {n} out of i32 range")))
        })
        .collect()
}

fn u8_array(v: &Value, key: &str) -> Result<Vec<u8>, BackendError> {
    v.get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| BackendError(format!("response missing array `{key}`")))?
        .iter()
        .map(|x| {
            let n = x
                .as_u64()
                .ok_or_else(|| BackendError(format!("`{key}`: element not a uint")))?;
            u8::try_from(n)
                .map_err(|_| BackendError(format!("`{key}`: element {n} out of u8 range")))
        })
        .collect()
}
