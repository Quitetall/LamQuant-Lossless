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

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::string::{String, ToString};
use std::thread;
use std::time::{Duration, Instant};
use std::vec::Vec;

use semantic_abir_bcs::ModelProvenance;
use serde_json::{json, Value};

use crate::backend::{
    BackendCapabilities, BackendError, ChannelSupport, NeuralBackend, NeuralTokens,
};

/// How long a single helper invocation may take before it is killed.
///
/// Generous on purpose: loading a checkpoint onto CPU and running inference over
/// a window is slow, and a timeout that fires during ordinary work is worse than
/// none — it would turn a slow machine into a failing one. The value that
/// matters here is that it is finite. A helper that stalls forever used to stall
/// its caller forever with it.
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(600);

/// Poll interval while waiting for the helper to exit. Short enough that a kill
/// is prompt, long enough that waiting ten minutes costs no measurable CPU.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(20);

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
    /// Upper bound on one `call`. See [`DEFAULT_CALL_TIMEOUT`].
    timeout: Duration,
    /// Channel count this backend's checkpoint requires, when the caller
    /// knows it. `None` leaves the constraint undeclared rather than absent.
    expected_channels: Option<u16>,
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
            timeout: DEFAULT_CALL_TIMEOUT,
            expected_channels: None,
        }
    }

    /// Declare the channel count this backend's checkpoint requires.
    ///
    /// Turns a constraint the shell could not see into one it can enforce
    /// before spawning anything. Nothing verifies the claim against the
    /// checkpoint -- that needs provenance the checkpoint does not carry yet --
    /// so a wrong value trades one late failure for one early one.
    #[must_use]
    pub fn expecting_channels(mut self, channels: u16) -> Self {
        self.expected_channels = Some(channels);
        self
    }

    /// Override how long one helper invocation may run before it is killed.
    ///
    /// Exit is detected by polling every [`EXIT_POLL_INTERVAL`], so the deadline
    /// is accurate to about that interval and a timeout near or below it will
    /// overshoot proportionally. That is fine for the seconds-to-minutes range
    /// this is for; it is not a general-purpose sub-millisecond deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
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
            timeout: DEFAULT_CALL_TIMEOUT,
            expected_channels: None,
        }
    }

    /// Wait for the helper to exit, killing it if it outlasts the deadline.
    ///
    /// After a kill this still reaps the child. Skipping that would leave a
    /// zombie for every timed-out call, which under a retry loop is its own
    /// slow failure.
    fn wait_bounded(&self, child: &mut Child) -> Result<ExitStatus, BackendError> {
        let deadline = Instant::now() + self.timeout;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {}
                Err(error) => return Err(BackendError(format!("wait for helper: {error}"))),
            }
            if Instant::now() >= deadline {
                let killed = child.kill().and_then(|()| child.wait().map(|_| ()));
                return Err(BackendError(format!(
                    "helper exceeded {:?} and was killed{}",
                    self.timeout,
                    match killed {
                        Ok(()) => String::new(),
                        Err(error) => format!(" (kill/reap failed: {error})"),
                    }
                )));
            }
            thread::sleep(EXIT_POLL_INTERVAL);
        }
    }

    /// Run one request through the helper, bounded by [`Self::timeout`].
    ///
    /// A hung helper used to hang the caller with it, and there are TWO ways it
    /// could, which is why this does not simply wrap `wait_with_output` in a
    /// watchdog:
    ///
    /// 1. `wait` never returns because the helper never exits (a model-load
    ///    stall, an infinite loop).
    /// 2. `write_all` never returns because the helper never *reads*. A pipe
    ///    holds 64 KiB on Linux and a 21-channel window is far larger, so the
    ///    write blocks once the buffer fills — before any wait is reached. A
    ///    timeout guarding only the wait would never fire.
    ///
    /// So stdin, stdout and stderr each get a thread and the main thread does
    /// nothing but watch the clock. Every blocking operation is off the path
    /// that enforces the deadline; killing the child closes the pipes, which is
    /// what lets those threads finish rather than leaking.
    fn call(&self, mut request: Value) -> Result<Value, BackendError> {
        request["mode"] = json!(self.mode);
        if self.mode == "model" {
            request["expected_checkpoint_sha256"] =
                json!(encode_hex(&self.model.checkpoint_sha256));
        }
        let payload = serde_json::to_vec(&request)
            .map_err(|e| BackendError(format!("serialize request: {e}")))?;
        let mut child = Command::new(&self.python)
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

        // All three pipes are claimed BEFORE any thread starts. Interleaving the
        // two would leave a window where the writer thread is already running
        // against a live child and an early return abandons both — the thread
        // detached and the child never reaped. Failing here instead costs
        // nothing, because nothing has been started yet.
        let (mut stdin, stdout, stderr) =
            match (child.stdin.take(), child.stdout.take(), child.stderr.take()) {
                (Some(stdin), Some(stdout), Some(stderr)) => (stdin, stdout, stderr),
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(BackendError(
                        "helper was spawned without all three pipes".to_string(),
                    ));
                }
            };
        let writer = thread::spawn(move || {
            let outcome = stdin.write_all(&payload);
            // Closing the pipe is how the helper learns the request has ended.
            drop(stdin);
            outcome
        });
        let stdout = drain(stdout);
        let stderr = drain(stderr);

        let status = self.wait_bounded(&mut child)?;
        let out_bytes = join_stream(stdout, "stdout")?;
        let err_bytes = join_stream(stderr, "stderr")?;
        let write_outcome = writer
            .join()
            .map_err(|_| BackendError("helper stdin writer panicked".to_string()))?;

        if !status.success() {
            // The helper's own stderr is the real diagnosis. Report it ahead of
            // any write error: a helper that dies on import makes our write fail
            // with a broken pipe, and "broken pipe" would bury the traceback
            // that actually says what went wrong.
            return Err(BackendError(format!(
                "helper exited {}: {}",
                status,
                String::from_utf8_lossy(&err_bytes)
            )));
        }
        if let Err(error) = write_outcome {
            return Err(BackendError(format!("write request: {error}")));
        }
        let response: Value = serde_json::from_slice(&out_bytes).map_err(|e| {
            BackendError(format!(
                "parse response: {e} (stderr: {})",
                String::from_utf8_lossy(&err_bytes)
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

/// Read a child pipe to EOF on its own thread.
///
/// Not merely for the timeout: a helper writing more than a pipe-buffer of
/// stdout while nobody drains it blocks forever, so draining concurrently is
/// what makes the exchange safe at any size.
fn drain(mut pipe: impl Read + Send + 'static) -> thread::JoinHandle<std::io::Result<Vec<u8>>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).map(|_| bytes)
    })
}

fn join_stream(
    handle: thread::JoinHandle<std::io::Result<Vec<u8>>>,
    which: &str,
) -> Result<Vec<u8>, BackendError> {
    handle
        .join()
        .map_err(|_| BackendError(format!("helper {which} reader panicked")))?
        .map_err(|error| BackendError(format!("read helper {which}: {error}")))
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
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            channels: match (self.mode.as_str(), self.expected_channels) {
                // Declared by whoever knows the checkpoint. Once stated, a
                // mismatched recording is refused before a subprocess is spawned.
                (_, Some(expected)) => ChannelSupport::Exactly(expected),
                // The self-test transform is `x mod 5` per sample. It genuinely
                // does not care how many channels there are.
                ("selftest", None) => ChannelSupport::Any,
                // The real architecture fixes a channel count, and this process
                // cannot see it without loading the checkpoint. Saying so is not
                // the same as saying anything goes.
                _ => ChannelSupport::DeclaredByCheckpoint,
            },
        }
    }

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
