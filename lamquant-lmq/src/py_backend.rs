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
//! Process-lifetime and write containment uses Bubblewrap PID namespaces on
//! Linux and Job Objects on Windows. This is not a confidentiality sandbox:
//! the helper inherits its environment, and Linux exposes the host root
//! read-only so Python, libraries, and weights remain discoverable. Helper code
//! and model artifacts must therefore be trusted. Other Unix hosts fail closed
//! until the native Rust backend replaces this temporary subprocess path.

use std::io::Read;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::string::{String, ToString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};
use std::vec::Vec;

use semantic_abir::Rational;
use semantic_abir_bcs::ModelProvenance;
use serde::de::{DeserializeSeed, Error as DeError, IgnoredAny, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{json, Value};

use crate::backend::{
    BackendError, BackendTarget, NeuralBackend, NeuralBackendCapabilities, NeuralTokens,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const DEFAULT_REQUEST_BYTES: u64 = 32 * 1024 * 1024;
const DEFAULT_STDOUT_BYTES: u64 = 32 * 1024 * 1024;
const DEFAULT_STDERR_BYTES: u64 = 1024 * 1024;
const JSON_I64_BYTES: u64 = 22;
const JSON_I32_BYTES: u64 = 12;
const JSON_U8_BYTES: u64 = 4;
const JSON_OVERHEAD_BYTES: u64 = 4096;
#[cfg(target_os = "linux")]
const BUBBLEWRAP_PATH: &str = "/usr/bin/bwrap";

/// Bounded subprocess transport allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendIoLimits {
    pub maximum_request_bytes: u64,
    pub maximum_stdout_bytes: u64,
    pub maximum_stderr_bytes: u64,
}

impl Default for BackendIoLimits {
    fn default() -> Self {
        Self {
            maximum_request_bytes: DEFAULT_REQUEST_BYTES,
            maximum_stdout_bytes: DEFAULT_STDOUT_BYTES,
            maximum_stderr_bytes: DEFAULT_STDERR_BYTES,
        }
    }
}

/// Shareable cooperative cancellation signal for one or more backend calls.
#[derive(Clone, Debug, Default)]
pub struct BackendCancellation {
    cancelled: Arc<AtomicBool>,
}

impl BackendCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

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
    timeout: Duration,
    cancellation: BackendCancellation,
    capabilities: NeuralBackendCapabilities,
    io_limits: BackendIoLimits,
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
            timeout: DEFAULT_TIMEOUT,
            cancellation: BackendCancellation::default(),
            capabilities: model_capabilities(),
            io_limits: BackendIoLimits::default(),
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
            timeout: DEFAULT_TIMEOUT,
            cancellation: BackendCancellation::default(),
            capabilities: selftest_capabilities(),
            io_limits: BackendIoLimits::default(),
        }
    }

    /// Override the hard wall-clock deadline for each helper invocation.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Bind cooperative cancellation controlled by the caller.
    pub fn with_cancellation(mut self, cancellation: BackendCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Override request/stdout/stderr allocation ceilings.
    pub fn with_io_limits(mut self, io_limits: BackendIoLimits) -> Self {
        self.io_limits = io_limits;
        self
    }

    fn call(&self, mut request: Value) -> Result<HelperOutput, BackendError> {
        let started = Instant::now();
        self.check_active(started)?;
        request["mode"] = json!(self.mode);
        if self.mode == "model" {
            request["expected_checkpoint_sha256"] =
                json!(encode_hex(&self.model.checkpoint_sha256));
        }
        let request = serde_json::to_vec(&request)
            .map_err(|e| BackendError(format!("serialize request: {e}")))?;
        enforce_io_limit(
            "request",
            request.len(),
            self.io_limits.maximum_request_bytes,
        )?;
        self.check_active(started)?;
        let mut command = helper_command(&self.python, &self.helper)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_process_tree(&mut command);
        let mut child = command.spawn().map_err(|e| {
            BackendError(format!(
                "spawn `{} {}`: {e}",
                self.python,
                self.helper.display()
            ))
        })?;
        let process_tree = match ProcessTree::attach(&child) {
            Ok(process_tree) => process_tree,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let mut child = SupervisedChild::new(child, process_tree);
        let stdin = match child.child.stdin.take() {
            Some(stdin) => stdin,
            None => return terminate_with(&mut child, "no child stdin"),
        };
        let stdout = match child.child.stdout.take() {
            Some(stdout) => stdout,
            None => return terminate_with(&mut child, "no child stdout"),
        };
        let stderr = match child.child.stderr.take() {
            Some(stderr) => stderr,
            None => return terminate_with(&mut child, "no child stderr"),
        };
        let stdin_result = match write_request(stdin, request) {
            Ok(receiver) => receiver,
            Err(error) => {
                child.terminate();
                return Err(error);
            }
        };
        let stdout_result =
            match read_pipe_bounded(stdout, self.io_limits.maximum_stdout_bytes, "stdout") {
                Ok(receiver) => receiver,
                Err(error) => {
                    child.terminate();
                    return Err(error);
                }
            };
        let stderr_result =
            match read_pipe_bounded(stderr, self.io_limits.maximum_stderr_bytes, "stderr") {
                Ok(receiver) => receiver,
                Err(error) => {
                    child.terminate();
                    return Err(error);
                }
            };
        let mut stdin_done = false;
        let mut stdout = None;
        let mut stderr = None;
        let mut status = None;
        loop {
            if self.cancellation.is_cancelled() {
                child.terminate();
                return Err(BackendError("helper invocation cancelled".to_string()));
            }
            if started.elapsed() >= self.timeout {
                child.terminate();
                return Err(BackendError("helper invocation timed out".to_string()));
            }
            if !stdin_done {
                match stdin_result.try_recv() {
                    Ok(Ok(())) => stdin_done = true,
                    Ok(Err(error)) => {
                        child.terminate();
                        return Err(error);
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => {
                        child.terminate();
                        return Err(BackendError("stdin writer disconnected".to_string()));
                    }
                }
            }
            poll_pipe(&stdout_result, &mut stdout, "stdout").inspect_err(|_error| {
                child.terminate();
            })?;
            poll_pipe(&stderr_result, &mut stderr, "stderr").inspect_err(|_error| {
                child.terminate();
            })?;
            // Do not reap the child until every pipe has closed. A reaped PID
            // may be reused immediately; retaining the unreaped child keeps its
            // PID/PGID unavailable while timeout/cancellation cleanup can still
            // signal the numeric process group safely.
            if status.is_none() && stdin_done && stdout.is_some() && stderr.is_some() {
                status = match child.child.try_wait() {
                    Ok(status) => status,
                    Err(error) => {
                        child.terminate();
                        return Err(BackendError(format!("poll helper: {error}")));
                    }
                };
            }
            if stdin_done && status.is_some() && stdout.is_some() && stderr.is_some() {
                break;
            }
            thread::sleep(POLL_INTERVAL.min(self.timeout.saturating_sub(started.elapsed())));
        }
        let status = status.expect("loop exits only with helper status");
        let stdout = stdout.expect("loop exits only with stdout");
        let stderr = stderr.expect("loop exits only with stderr");
        // `try_wait` already reaped the direct child. Linux Bubblewrap has
        // torn down its PID namespace before this status becomes observable;
        // Windows Job cleanup remains bound to its owned handle. Disarm numeric
        // process-group signaling so a later Drop cannot target a reused PGID.
        child.mark_reaped();
        if !status.success() {
            return Err(BackendError(format!(
                "helper exited {}: {}",
                status,
                String::from_utf8_lossy(&stderr)
            )));
        }
        Ok(HelperOutput {
            stdout,
            stderr,
            started,
        })
    }

    fn check_active(&self, started: Instant) -> Result<(), BackendError> {
        if self.cancellation.is_cancelled() {
            return Err(BackendError("helper invocation cancelled".to_string()));
        }
        if started.elapsed() >= self.timeout {
            return Err(BackendError("helper invocation timed out".to_string()));
        }
        Ok(())
    }

    fn validate_checkpoint(&self, actual: Option<&str>) -> Result<(), BackendError> {
        if self.mode == "model"
            && actual != Some(encode_hex(&self.model.checkpoint_sha256).as_str())
        {
            return Err(BackendError(
                "helper executed a checkpoint different from model provenance".to_string(),
            ));
        }
        Ok(())
    }
}

struct HelperOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    started: Instant,
}

fn enforce_io_limit(name: &str, actual: usize, limit: u64) -> Result<(), BackendError> {
    if actual as u128 > u128::from(limit) {
        Err(BackendError(format!(
            "helper {name} exceeds byte limit ({actual} > {limit})"
        )))
    } else {
        Ok(())
    }
}

fn write_request(
    mut stdin: ChildStdin,
    request: Vec<u8>,
) -> Result<Receiver<Result<(), BackendError>>, BackendError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::Builder::new()
        .name("lmq-helper-stdin".to_string())
        .spawn(move || {
            use std::io::Write;
            let result = stdin
                .write_all(&request)
                .map_err(|error| BackendError(format!("write request: {error}")));
            let _ = sender.send(result);
        })
        .map_err(|error| BackendError(format!("spawn helper stdin writer: {error}")))?;
    Ok(receiver)
}

fn read_pipe_bounded(
    mut pipe: impl Read + Send + 'static,
    limit: u64,
    name: &str,
) -> Result<Receiver<Result<Vec<u8>, BackendError>>, BackendError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let name = name.to_string();
    let thread_name = format!("lmq-helper-{name}");
    let spawn_name = name.clone();
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 8 * 1024];
            let result = loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break Ok(bytes),
                    Ok(read) => {
                        let Some(total) = bytes.len().checked_add(read) else {
                            break Err(BackendError(format!("helper {name} exceeds byte limit")));
                        };
                        if total as u128 > u128::from(limit) {
                            break Err(BackendError(format!(
                                "helper {name} exceeds byte limit ({total} > {limit})"
                            )));
                        }
                        bytes.extend_from_slice(&chunk[..read]);
                    }
                    Err(error) => {
                        break Err(BackendError(format!("read helper {name}: {error}")));
                    }
                }
            };
            let _ = sender.send(result);
        })
        .map_err(|error| BackendError(format!("spawn helper {spawn_name} reader: {error}")))?;
    Ok(receiver)
}

fn poll_pipe(
    receiver: &Receiver<Result<Vec<u8>, BackendError>>,
    output: &mut Option<Vec<u8>>,
    name: &str,
) -> Result<(), BackendError> {
    if output.is_some() {
        return Ok(());
    }
    match receiver.try_recv() {
        Ok(Ok(bytes)) => {
            *output = Some(bytes);
            Ok(())
        }
        Ok(Err(error)) => Err(error),
        Err(TryRecvError::Empty) => Ok(()),
        Err(TryRecvError::Disconnected) => Err(BackendError(format!("{name} reader disconnected"))),
    }
}

fn terminate_with<T>(child: &mut SupervisedChild, message: &str) -> Result<T, BackendError> {
    child.terminate();
    Err(BackendError(message.to_string()))
}

#[cfg(target_os = "linux")]
fn helper_command(python: &str, helper: &PathBuf) -> Result<Command, BackendError> {
    // PID-namespace init owns every descendant, including children that call
    // setsid(2). When the helper exits or bwrap is killed, namespace teardown
    // kills all remaining processes. Read-only host mount preserves Python and
    // weight discovery; /tmp remains writable for ordinary runtime scratch.
    // This constrains process lifetime and host writes, not confidentiality:
    // helper code can read user-readable host files and inherits its environment.
    verify_linux_containment()?;
    let mut command = Command::new(BUBBLEWRAP_PATH);
    command
        .arg("--unshare-pid")
        .arg("--die-with-parent")
        .arg("--ro-bind")
        .arg("/")
        .arg("/")
        .arg("--dev-bind")
        .arg("/dev")
        .arg("/dev")
        .arg("--bind")
        .arg("/tmp")
        .arg("/tmp")
        .arg("--proc")
        .arg("/proc")
        .arg("--")
        .arg(python)
        .arg(helper);
    Ok(command)
}

#[cfg(target_os = "linux")]
fn verify_linux_containment() -> Result<(), BackendError> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    static PREFLIGHT: OnceLock<Result<(), String>> = OnceLock::new();
    PREFLIGHT
        .get_or_init(|| {
            let metadata = std::fs::metadata(BUBBLEWRAP_PATH).map_err(|error| {
                format!("trusted Bubblewrap launcher `{BUBBLEWRAP_PATH}` unavailable: {error}")
            })?;
            let mode = metadata.permissions().mode();
            if !metadata.is_file() || metadata.uid() != 0 || mode & 0o111 == 0 || mode & 0o022 != 0
            {
                return Err(format!(
                    "trusted Bubblewrap launcher `{BUBBLEWRAP_PATH}` must be a \
                     root-owned executable not writable by group or other"
                ));
            }
            let status = Command::new(BUBBLEWRAP_PATH)
                .arg("--unshare-pid")
                .arg("--die-with-parent")
                .arg("--ro-bind")
                .arg("/")
                .arg("/")
                .arg("--proc")
                .arg("/proc")
                .arg("--")
                .arg("/bin/true")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map_err(|error| format!("Bubblewrap containment preflight failed: {error}"))?;
            if !status.success() {
                return Err(format!(
                    "Bubblewrap PID-namespace containment unavailable (status {status})"
                ));
            }
            Ok(())
        })
        .clone()
        .map_err(BackendError)
}

#[cfg(windows)]
fn helper_command(python: &str, helper: &PathBuf) -> Result<Command, BackendError> {
    let mut command = Command::new(python);
    command.arg(helper);
    Ok(command)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn helper_command(_python: &str, _helper: &PathBuf) -> Result<Command, BackendError> {
    Err(BackendError(
        "bounded helper process containment is unsupported on this Unix platform".to_string(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn helper_command(_python: &str, _helper: &PathBuf) -> Result<Command, BackendError> {
    Err(BackendError(
        "bounded helper process containment is unsupported on this platform".to_string(),
    ))
}

struct SupervisedChild {
    child: Child,
    process_tree: ProcessTree,
    terminated: bool,
}

impl SupervisedChild {
    fn new(child: Child, process_tree: ProcessTree) -> Self {
        Self {
            child,
            process_tree,
            terminated: false,
        }
    }

    fn terminate(&mut self) {
        if !self.terminated {
            self.process_tree.terminate(&mut self.child);
            self.terminated = true;
        }
    }

    fn mark_reaped(&mut self) {
        self.terminated = true;
    }
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(unix)]
fn configure_process_tree(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(windows)]
fn configure_process_tree(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;
    command.creation_flags(CREATE_SUSPENDED);
}

#[cfg(not(any(unix, windows)))]
fn configure_process_tree(_command: &mut Command) {}

#[cfg(unix)]
struct ProcessTree {
    process_group: i32,
}

#[cfg(unix)]
impl ProcessTree {
    fn attach(child: &Child) -> Result<Self, BackendError> {
        let process_group = i32::try_from(child.id())
            .map_err(|_| BackendError("helper pid exceeds process-group range".to_string()))?;
        Ok(Self { process_group })
    }

    fn terminate(&self, child: &mut Child) {
        // SAFETY: child was spawned into a new process group whose id is its
        // pid. A negative pid targets that group only.
        unsafe {
            libc::kill(-self.process_group, libc::SIGKILL);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(windows)]
struct ProcessTree {
    job: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
impl ProcessTree {
    fn attach(child: &Child) -> Result<Self, BackendError> {
        use std::mem::{size_of, zeroed};
        use std::os::windows::io::AsRawHandle;
        use std::ptr::{null, null_mut};
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };

        #[link(name = "ntdll")]
        extern "system" {
            fn NtResumeProcess(process_handle: HANDLE) -> i32;
        }

        // SAFETY: all handles and structure sizes follow the Win32 API
        // contracts; every failure closes the newly created job handle.
        unsafe {
            let job = CreateJobObjectW(null(), null());
            if job == null_mut() as HANDLE {
                return Err(BackendError("create helper job object failed".to_string()));
            }
            let mut information: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &information as *const _ as *const core::ffi::c_void,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                CloseHandle(job);
                return Err(BackendError(
                    "configure helper job object failed".to_string(),
                ));
            }
            let process = child.as_raw_handle() as HANDLE;
            if process == null_mut() as HANDLE || AssignProcessToJobObject(job, process) == 0 {
                CloseHandle(job);
                return Err(BackendError(
                    "assign helper to job object failed".to_string(),
                ));
            }
            if NtResumeProcess(process) < 0 {
                TerminateJobObject(job, 1);
                CloseHandle(job);
                return Err(BackendError(
                    "resume job-bound helper process failed".to_string(),
                ));
            }
            Ok(Self { job })
        }
    }

    fn terminate(&self, child: &mut Child) {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;
        // SAFETY: `job` remains owned by `self` until Drop.
        unsafe {
            TerminateJobObject(self.job, 1);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        // SAFETY: `job` is a live owned handle and is closed exactly once.
        unsafe {
            CloseHandle(self.job);
        }
    }
}

#[cfg(not(any(unix, windows)))]
struct ProcessTree;

#[cfg(not(any(unix, windows)))]
impl ProcessTree {
    fn attach(_child: &Child) -> Result<Self, BackendError> {
        Err(BackendError(
            "bounded helper process trees are unsupported on this platform".to_string(),
        ))
    }

    fn terminate(&self, child: &mut Child) {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn selftest_capabilities() -> NeuralBackendCapabilities {
    NeuralBackendCapabilities {
        target: BackendTarget::HostSubprocess,
        operational: cfg!(any(target_os = "linux", windows)),
        minimum_channels: 1,
        maximum_channels: u16::MAX,
        minimum_samples: 1,
        maximum_samples: u32::MAX,
        minimum_sample_rate: Rational::new(1, i64::MAX.into()).expect("positive denominator"),
        maximum_sample_rate: Rational::new(i64::MAX.into(), 1).expect("positive denominator"),
        maximum_tokens: u32::MAX,
        maximum_schedule_bytes: u32::MAX,
        maximum_backend_metadata_bytes: 2,
        minimum_alphabet: 5,
        maximum_alphabet: 5,
    }
}

fn model_capabilities() -> NeuralBackendCapabilities {
    NeuralBackendCapabilities {
        target: BackendTarget::HostSubprocess,
        operational: cfg!(any(target_os = "linux", windows)),
        minimum_channels: 21,
        maximum_channels: 21,
        minimum_samples: 2_500,
        maximum_samples: 2_500,
        minimum_sample_rate: Rational::new(250, 1).expect("valid model sample rate"),
        maximum_sample_rate: Rational::new(250, 1).expect("valid model sample rate"),
        maximum_tokens: 32 * 79,
        maximum_schedule_bytes: 79,
        maximum_backend_metadata_bytes: 16 * 1024 * 1024,
        minimum_alphabet: 32,
        maximum_alphabet: 32,
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
    fn capabilities(&self) -> NeuralBackendCapabilities {
        self.capabilities
    }

    fn model_provenance(&self) -> ModelProvenance {
        self.model.clone()
    }

    fn encode(
        &self,
        signal: &[Vec<i64>],
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        if self.cancellation.is_cancelled() {
            return Err(BackendError("helper invocation cancelled".to_string()));
        }
        let channels = u16::try_from(signal.len())
            .map_err(|_| BackendError("signal channel count exceeds u16".to_string()))?;
        let samples = signal.first().map_or(0, Vec::len);
        let samples_u32 = u32::try_from(samples)
            .map_err(|_| BackendError("signal sample count exceeds u32".to_string()))?;
        self.capabilities
            .validate_input(channels, samples_u32, sample_rate)
            .map_err(|error| BackendError(format!("helper capability mismatch: {error:?}")))?;
        preflight_encode_request(signal, self.io_limits.maximum_request_bytes)?;
        let (rate_numerator, rate_denominator) = sample_rate.parts();
        let sample_rate = rate_numerator as f64 / rate_denominator as f64;
        let output = self.call(json!({
            "op": "encode",
            "sample_rate": sample_rate,
            "signal": signal,
        }))?;
        self.check_active(output.started)?;
        let envelope: EncodeResponse<'_> = parse_envelope(&output)?;
        self.validate_checkpoint(envelope.checkpoint_sha256)?;
        let input_elements = signal
            .len()
            .checked_mul(samples)
            .ok_or_else(|| BackendError("signal response limit overflow".to_string()))?;
        let tokens = parse_bounded_array(
            envelope.tokens,
            usize::try_from(self.capabilities.maximum_tokens)
                .unwrap_or(usize::MAX)
                .min(input_elements),
            None,
            "tokens",
        )?;
        let schedule = parse_bounded_array(
            envelope.schedule,
            usize::try_from(self.capabilities.maximum_schedule_bytes)
                .unwrap_or(usize::MAX)
                .min(samples),
            None,
            "schedule",
        )?;
        let backend_meta = parse_bounded_array(
            envelope.backend_meta,
            usize::try_from(self.capabilities.maximum_backend_metadata_bytes).unwrap_or(usize::MAX),
            None,
            "backend_meta",
        )?;
        let tokens = NeuralTokens {
            tokens,
            schedule,
            alphabet: envelope.alphabet,
            n_channels: envelope.n_channels,
            n_samples: envelope.n_samples,
            backend_meta,
        };
        self.capabilities
            .validate_output(&tokens)
            .map_err(|error| BackendError(format!("helper capability mismatch: {error:?}")))?;
        if usize::from(tokens.n_channels) != signal.len()
            || usize::try_from(tokens.n_samples).ok() != Some(samples)
        {
            return Err(BackendError(
                "helper response shape differs from input signal".to_string(),
            ));
        }
        self.check_active(output.started)?;
        Ok(tokens)
    }

    fn decode(&self, t: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
        self.capabilities
            .validate_output(t)
            .map_err(|error| BackendError(format!("helper capability mismatch: {error:?}")))?;
        preflight_decode_request(t, self.io_limits.maximum_request_bytes)?;
        preflight_decode_response(t, self.io_limits.maximum_stdout_bytes)?;
        let output = self.call(json!({
            "op": "decode",
            "tokens": t.tokens,
            "schedule": t.schedule,
            "alphabet": t.alphabet,
            "n_channels": t.n_channels,
            "n_samples": t.n_samples,
            "backend_meta": t.backend_meta,
        }))?;
        self.check_active(output.started)?;
        let envelope: DecodeResponse<'_> = parse_envelope(&output)?;
        self.validate_checkpoint(envelope.checkpoint_sha256)?;
        let signal = parse_bounded_matrix(
            envelope.signal,
            usize::from(t.n_channels),
            usize::try_from(t.n_samples)
                .map_err(|_| BackendError("sample count exceeds host usize".to_string()))?,
            "signal",
        )?;
        self.check_active(output.started)?;
        Ok(signal)
    }
}

#[derive(Deserialize)]
struct EncodeResponse<'a> {
    #[serde(borrow)]
    tokens: &'a RawValue,
    #[serde(borrow)]
    schedule: &'a RawValue,
    alphabet: u16,
    n_channels: u16,
    n_samples: u32,
    #[serde(borrow)]
    backend_meta: &'a RawValue,
    #[serde(default)]
    checkpoint_sha256: Option<&'a str>,
}

#[derive(Deserialize)]
struct DecodeResponse<'a> {
    #[serde(borrow)]
    signal: &'a RawValue,
    #[serde(default)]
    checkpoint_sha256: Option<&'a str>,
}

fn parse_envelope<'a, T>(output: &'a HelperOutput) -> Result<T, BackendError>
where
    T: Deserialize<'a>,
{
    serde_json::from_slice(&output.stdout).map_err(|error| {
        BackendError(format!(
            "parse response envelope: {error} (stderr: {})",
            String::from_utf8_lossy(&output.stderr)
        ))
    })
}

struct BoundedArraySeed<T> {
    maximum: usize,
    exact: Option<usize>,
    name: &'static str,
    marker: PhantomData<T>,
}

impl<T> BoundedArraySeed<T> {
    fn new(maximum: usize, exact: Option<usize>, name: &'static str) -> Self {
        Self {
            maximum,
            exact,
            name,
            marker: PhantomData,
        }
    }
}

impl<'de, T> DeserializeSeed<'de> for BoundedArraySeed<T>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, T> Visitor<'de> for BoundedArraySeed<T>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "bounded `{}` array", self.name)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if sequence.size_hint().is_some_and(|hint| hint > self.maximum) {
            return Err(A::Error::custom(format_args!(
                "`{}` exceeds element limit {}",
                self.name, self.maximum
            )));
        }
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.maximum));
        while let Some(value) = sequence.next_element()? {
            if values.len() == self.maximum {
                return Err(A::Error::custom(format_args!(
                    "`{}` exceeds element limit {}",
                    self.name, self.maximum
                )));
            }
            values.push(value);
        }
        if self.exact.is_some_and(|exact| values.len() != exact) {
            return Err(A::Error::custom(format_args!(
                "`{}` length {} differs from required {}",
                self.name,
                values.len(),
                self.exact.expect("checked Some")
            )));
        }
        Ok(values)
    }
}

fn parse_bounded_array<T>(
    raw: &RawValue,
    maximum: usize,
    exact: Option<usize>,
    name: &'static str,
) -> Result<Vec<T>, BackendError>
where
    T: for<'de> Deserialize<'de>,
{
    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
    let values = BoundedArraySeed::new(maximum, exact, name)
        .deserialize(&mut deserializer)
        .map_err(|error| BackendError(format!("parse `{name}`: {error}")))?;
    deserializer
        .end()
        .map_err(|error| BackendError(format!("parse `{name}` trailing data: {error}")))?;
    Ok(values)
}

struct BoundedMatrixSeed {
    rows: usize,
    columns: usize,
    name: &'static str,
}

impl<'de> DeserializeSeed<'de> for BoundedMatrixSeed {
    type Value = Vec<Vec<i64>>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for BoundedMatrixSeed {
    type Value = Vec<Vec<i64>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "bounded `{}` matrix", self.name)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        if sequence.size_hint().is_some_and(|hint| hint > self.rows) {
            return Err(A::Error::custom(format_args!(
                "`{}` exceeds row limit {}",
                self.name, self.rows
            )));
        }
        let mut rows = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.rows));
        while rows.len() < self.rows {
            let Some(row) = sequence.next_element_seed(BoundedArraySeed::new(
                self.columns,
                Some(self.columns),
                "signal row",
            ))?
            else {
                break;
            };
            rows.push(row);
        }
        if sequence.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::custom(format_args!(
                "`{}` exceeds row limit {}",
                self.name, self.rows
            )));
        }
        if rows.len() != self.rows {
            return Err(A::Error::custom(format_args!(
                "`{}` row count {} differs from required {}",
                self.name,
                rows.len(),
                self.rows
            )));
        }
        Ok(rows)
    }
}

fn parse_bounded_matrix(
    raw: &RawValue,
    rows: usize,
    columns: usize,
    name: &'static str,
) -> Result<Vec<Vec<i64>>, BackendError> {
    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
    let values = BoundedMatrixSeed {
        rows,
        columns,
        name,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| BackendError(format!("parse `{name}`: {error}")))?;
    deserializer
        .end()
        .map_err(|error| BackendError(format!("parse `{name}` trailing data: {error}")))?;
    Ok(values)
}

fn preflight_encode_request(
    signal: &[Vec<i64>],
    maximum_request_bytes: u64,
) -> Result<(), BackendError> {
    let samples = signal.first().map_or(0, Vec::len);
    if signal.is_empty() || samples == 0 || signal.iter().any(|row| row.len() != samples) {
        return Err(BackendError(
            "signal must be non-empty rectangular channels".to_string(),
        ));
    }
    let elements = (signal.len() as u128)
        .checked_mul(samples as u128)
        .ok_or_else(|| BackendError("signal request size overflow".to_string()))?;
    let estimate = elements
        .checked_mul(u128::from(JSON_I64_BYTES))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| BackendError("signal request size overflow".to_string()))?;
    if estimate > u128::from(maximum_request_bytes) {
        return Err(BackendError(format!(
            "helper request exceeds byte limit ({estimate} estimated > {maximum_request_bytes})"
        )));
    }
    Ok(())
}

fn preflight_decode_request(
    tokens: &NeuralTokens,
    maximum_request_bytes: u64,
) -> Result<(), BackendError> {
    let token_bytes = (tokens.tokens.len() as u128)
        .checked_mul(u128::from(JSON_I32_BYTES))
        .ok_or_else(|| BackendError("token request size overflow".to_string()))?;
    let schedule_bytes = (tokens.schedule.len() as u128)
        .checked_mul(u128::from(JSON_U8_BYTES))
        .ok_or_else(|| BackendError("schedule request size overflow".to_string()))?;
    let metadata_bytes = (tokens.backend_meta.len() as u128)
        .checked_mul(u128::from(JSON_U8_BYTES))
        .ok_or_else(|| BackendError("metadata request size overflow".to_string()))?;
    let estimate = token_bytes
        .checked_add(schedule_bytes)
        .and_then(|bytes| bytes.checked_add(metadata_bytes))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| BackendError("decode request size overflow".to_string()))?;
    if estimate > u128::from(maximum_request_bytes) {
        return Err(BackendError(format!(
            "helper request exceeds byte limit ({estimate} estimated > {maximum_request_bytes})"
        )));
    }
    Ok(())
}

fn preflight_decode_response(
    tokens: &NeuralTokens,
    maximum_stdout_bytes: u64,
) -> Result<(), BackendError> {
    let elements = u128::from(tokens.n_channels)
        .checked_mul(u128::from(tokens.n_samples))
        .ok_or_else(|| BackendError("decoded response size overflow".to_string()))?;
    let estimate = elements
        .checked_mul(u128::from(JSON_I64_BYTES))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| BackendError("decoded response size overflow".to_string()))?;
    if estimate > u128::from(maximum_stdout_bytes) {
        return Err(BackendError(format!(
            "helper decoded response exceeds stdout byte limit \
             ({estimate} estimated > {maximum_stdout_bytes})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_abir::ContentId;
    use semantic_abir_bcs::PccpStatus;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn model() -> ModelProvenance {
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([1; 32]),
            checkpoint_sha256: [2; 32],
            pccp_change_id: "test".to_string(),
            pccp_evidence_id: ContentId::from_bytes([3; 32]),
            pccp_status: PccpStatus::Candidate,
        }
    }

    fn rate() -> Rational {
        Rational::new(250, 1).unwrap()
    }

    #[test]
    fn pre_cancelled_helper_never_spawns() {
        let cancellation = BackendCancellation::default();
        cancellation.cancel();
        let backend = PyBackend::selftest("missing-executable", "missing-helper", model())
            .with_cancellation(cancellation);
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("cancelled backend must fail");
        assert!(error.0.contains("cancelled"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn model_backend_enforces_exact_capabilities_before_spawn() {
        let backend = PyBackend::model("missing-executable", "missing-helper", model());
        let error = backend
            .encode(&[vec![1]], Rational::new(1, 1).unwrap())
            .expect_err("unsupported model shape and rate must fail before spawn");
        assert!(
            error.0.contains("ChannelCount"),
            "unexpected capability failure: {}",
            error.0
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn zero_timeout_helper_never_spawns() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let marker = std::env::temp_dir().join(format!("lmq-zero-timeout-{unique}.marker"));
        let helper = std::env::temp_dir().join(format!("lmq-zero-timeout-{unique}.sh"));
        fs::write(
            &helper,
            format!("#!/bin/sh\ntouch '{}'\n", marker.display()),
        )
        .unwrap();
        let backend = PyBackend::selftest("sh", &helper, model()).with_timeout(Duration::ZERO);
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("zero timeout must fail before spawn");
        assert!(error.0.contains("timed out"));
        assert!(
            !marker.exists(),
            "helper performed a pre-timeout side effect"
        );
        let _ = fs::remove_file(helper);
    }

    #[test]
    fn response_array_limit_applies_before_full_materialization() {
        let output = HelperOutput {
            stdout: br#"{"tokens":[0,0,0,0,0],"schedule":[5],"alphabet":5,"n_channels":1,"n_samples":1,"backend_meta":[]}"#.to_vec(),
            stderr: Vec::new(),
            started: Instant::now(),
        };
        let envelope: EncodeResponse<'_> = parse_envelope(&output).unwrap();
        let error = parse_bounded_array::<i32>(envelope.tokens, 4, None, "tokens")
            .expect_err("fifth token must be rejected at parser boundary");
        assert!(error.0.contains("exceeds element limit 4"), "{}", error.0);
    }

    #[test]
    fn decoded_response_shape_is_bounded_before_spawn() {
        let backend = PyBackend::selftest("missing-executable", "missing-helper", model());
        let tokens = NeuralTokens {
            tokens: vec![0],
            schedule: vec![5],
            alphabet: 5,
            n_channels: u16::MAX,
            n_samples: u32::MAX,
            backend_meta: Vec::new(),
        };
        let error = backend
            .decode(&tokens)
            .expect_err("huge reconstructed shape must fail before helper spawn");
        assert!(
            error
                .0
                .contains("decoded response exceeds stdout byte limit"),
            "{}",
            error.0
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hung_helper_is_killed_at_deadline() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-hung-helper-{unique}.sh"));
        fs::write(&helper, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
        let backend =
            PyBackend::selftest("sh", &helper, model()).with_timeout(Duration::from_millis(50));
        let started = Instant::now();
        let error = backend
            .encode(&[vec![1; 10_000]], rate())
            .expect_err("hung helper must time out");
        let _ = fs::remove_file(helper);
        assert!(error.0.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn helper_output_is_bounded_before_json_parse() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-output-helper-{unique}.sh"));
        fs::write(
            &helper,
            "#!/bin/sh\nwhile :; do printf 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'; done\n",
        )
        .unwrap();
        let backend = PyBackend::selftest("sh", &helper, model())
            .with_timeout(Duration::from_secs(2))
            .with_io_limits(BackendIoLimits {
                maximum_stdout_bytes: 1_024,
                ..BackendIoLimits::default()
            });
        let started = Instant::now();
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("oversized output must fail");
        let _ = fs::remove_file(helper);
        assert!(error.0.contains("stdout exceeds byte limit"), "{}", error.0);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn request_is_bounded_before_spawn() {
        let backend = PyBackend::selftest("missing-executable", "missing-helper", model())
            .with_io_limits(BackendIoLimits {
                maximum_request_bytes: 4_096,
                ..BackendIoLimits::default()
            });
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("estimated request at limit must fail before spawn");
        assert!(error.0.contains("request exceeds byte limit"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn timeout_kills_descendants_holding_pipes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-tree-helper-{unique}.sh"));
        let heartbeat = std::env::temp_dir().join(format!("lmq-tree-helper-{unique}.heartbeat"));
        fs::write(
            &helper,
            format!(
                "#!/bin/sh\nsetsid sh -c 'while :; do printf x >> \"{}\"; sleep 0.01; done' \
                 >/dev/null 2>&1 &\nwait\n",
                heartbeat.display()
            ),
        )
        .unwrap();
        let backend =
            PyBackend::selftest("sh", &helper, model()).with_timeout(Duration::from_millis(50));
        let started = Instant::now();
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("process tree must time out");
        assert!(error.0.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
        let size_after_return = fs::metadata(&heartbeat).map_or(0, |metadata| metadata.len());
        thread::sleep(Duration::from_millis(100));
        let size_after_wait = fs::metadata(&heartbeat).map_or(0, |metadata| metadata.len());
        assert_eq!(
            size_after_wait, size_after_return,
            "new-session descendant remained alive after timeout"
        );
        let _ = fs::remove_file(helper);
        let _ = fs::remove_file(heartbeat);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn successful_helper_cannot_leave_new_session_descendants() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-success-tree-{unique}.sh"));
        let heartbeat = std::env::temp_dir().join(format!("lmq-success-tree-{unique}.heartbeat"));
        fs::write(
            &helper,
            format!(
                "#!/bin/sh\nsetsid sh -c 'while :; do printf x >> \"{}\"; sleep 0.01; done' \
                 >/dev/null 2>&1 &\ncat >/dev/null\nprintf '%s' \
                 '{{\"tokens\":[1],\"schedule\":[5],\"alphabet\":5,\"n_channels\":1,\
                 \"n_samples\":1,\"backend_meta\":[]}}'\n",
                heartbeat.display()
            ),
        )
        .unwrap();
        let backend = PyBackend::selftest("sh", &helper, model());
        backend
            .encode(&[vec![1]], rate())
            .expect("valid helper response");
        let size_after_return = fs::metadata(&heartbeat).map_or(0, |metadata| metadata.len());
        thread::sleep(Duration::from_millis(100));
        let size_after_wait = fs::metadata(&heartbeat).map_or(0, |metadata| metadata.len());
        assert_eq!(
            size_after_wait, size_after_return,
            "new-session descendant remained alive after helper return"
        );
        let _ = fs::remove_file(helper);
        let _ = fs::remove_file(heartbeat);
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    #[test]
    fn unsupported_unix_backend_fails_closed_before_spawn() {
        let backend = PyBackend::selftest("missing-executable", "missing-helper", model());
        let error = backend
            .encode(&[vec![1]], rate())
            .expect_err("unsupported Unix process containment must fail closed");
        assert!(error.0.contains("Unavailable"), "{}", error.0);
    }
}
