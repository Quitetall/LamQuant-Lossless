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
//! Developer-mode process/write containment uses Bubblewrap PID namespaces on
//! Linux and Job Objects on Windows; it is not a confidentiality sandbox.
//! [`ProductionPyBackend`] instead requires a complete read-only rootfs closure,
//! clears its environment, hides accelerator devices, and joins an empty
//! cgroup-v2 child before `exec`. Other Unix hosts fail closed until an equally
//! enforceable production loader exists. Host kernel and privileged host
//! administrators remain trusted; containment isolates unprivileged model code.

use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::num::{NonZeroU16, NonZeroU64};
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
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::backend::{
    BackendError, BackendErrorKind, BackendModel, BackendTarget, NeuralBackend,
    NeuralBackendCapabilities, NeuralSignal, NeuralTokens, SignalDomain, TrainedModelArtifact,
};

mod production;

pub use production::{
    ProductionBackendLoadError, ProductionDeploymentAttestation, ProductionPyBackend,
    ProductionPyBackendConfig,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const HELPER_MODEL_REJECTION_EXIT_CODE: i32 = 64;
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

/// Enforced subprocess ceilings bound into production deployment identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendProcessLimits {
    pub memory_bytes: NonZeroU64,
    pub cpu_slots: NonZeroU16,
    pub maximum_tasks: NonZeroU16,
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
    /// (the real `SubbandCodec`; optional in developer tests, mandatory in the
    /// package evidence gate).
    mode: String,
    model: PyBackendModel,
    timeout: Duration,
    cancellation: BackendCancellation,
    capabilities: NeuralBackendCapabilities,
    io_limits: BackendIoLimits,
    execution: PyExecutionEnvironment,
}

enum PyExecutionEnvironment {
    Developer,
    #[cfg(target_os = "linux")]
    ProductionLinux {
        rootfs: PathBuf,
        cgroup: PathBuf,
        weights_dir: PathBuf,
        process_limits: BackendProcessLimits,
    },
}

enum PyBackendModel {
    ModelFree(ModelProvenance),
    Trained(Box<TrainedModelArtifact>),
}

#[derive(Serialize)]
struct HelperRequest<'a, T> {
    #[serde(flatten)]
    operation: T,
    mode: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_checkpoint_sha256: Option<String>,
}

#[derive(Serialize)]
struct EncodeRequest<'a> {
    op: &'static str,
    sample_rate: f64,
    signal: &'a [Vec<i64>],
    signal_domain: &'static str,
}

#[derive(Serialize)]
struct DecodeRequest<'a> {
    op: &'static str,
    tokens: &'a [i32],
    schedule: &'a [u8],
    alphabet: u16,
    n_channels: u16,
    n_samples: u32,
    backend_meta: &'a [u8],
    signal_domain: &'static str,
}

impl PyBackendModel {
    const fn provenance(&self) -> &ModelProvenance {
        match self {
            Self::ModelFree(provenance) => provenance,
            Self::Trained(artifact) => artifact.provenance(),
        }
    }
}

impl PyBackend {
    /// Drive the real `SubbandCodec` (`mode = "model"`).
    pub fn model(
        python: impl Into<String>,
        helper: impl Into<PathBuf>,
        model: TrainedModelArtifact,
    ) -> Self {
        Self {
            python: python.into(),
            helper: helper.into(),
            mode: "model".to_string(),
            model: PyBackendModel::Trained(Box::new(model)),
            timeout: DEFAULT_TIMEOUT,
            cancellation: BackendCancellation::default(),
            capabilities: model_capabilities(),
            io_limits: BackendIoLimits::default(),
            execution: PyExecutionEnvironment::Developer,
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
            model: PyBackendModel::ModelFree(model),
            timeout: DEFAULT_TIMEOUT,
            cancellation: BackendCancellation::default(),
            capabilities: selftest_capabilities(),
            io_limits: BackendIoLimits::default(),
            execution: PyExecutionEnvironment::Developer,
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

    fn call<T: Serialize>(
        &self,
        operation: T,
        request_capacity: usize,
    ) -> Result<HelperOutput, BackendError> {
        let started = Instant::now();
        self.check_active(started)?;
        let request = HelperRequest {
            operation,
            mode: &self.mode,
            expected_checkpoint_sha256: (self.mode == "model")
                .then(|| encode_hex(&self.model.provenance().checkpoint_sha256)),
        };
        let maximum_request_bytes =
            usize::try_from(self.io_limits.maximum_request_bytes).map_err(|_| {
                BackendError::new(
                    BackendErrorKind::ResourceLimit,
                    "request limit exceeds host usize",
                )
            })?;
        if request_capacity > maximum_request_bytes {
            return Err(BackendError::new(
                BackendErrorKind::ResourceLimit,
                "request capacity exceeds configured byte limit".to_string(),
            ));
        }
        let mut request_bytes = Vec::new();
        request_bytes
            .try_reserve_exact(request_capacity)
            .map_err(|error| {
                BackendError::new(
                    BackendErrorKind::ResourceLimit,
                    format!("reserve request buffer: {error}"),
                )
            })?;
        serde_json::to_writer(&mut request_bytes, &request).map_err(|error| {
            BackendError::new(
                BackendErrorKind::Protocol,
                format!("serialize request: {error}"),
            )
        })?;
        enforce_io_limit(
            "request",
            request_bytes.len(),
            self.io_limits.maximum_request_bytes,
        )?;
        self.check_active(started)?;
        let mut command = helper_command(&self.python, &self.helper, &self.execution)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        configure_process_tree(&mut command);
        let cgroup_membership = configure_process_limits(&mut command, &self.execution)?;
        let mut child = command.spawn().map_err(|e| {
            BackendError::new(
                BackendErrorKind::Process,
                format!("spawn `{} {}`: {e}", self.python, self.helper.display()),
            )
        })?;
        drop(cgroup_membership);
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
        let stdin_result = match write_request(stdin, request_bytes) {
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
                return Err(BackendError::new(
                    BackendErrorKind::Cancelled,
                    "helper invocation cancelled",
                ));
            }
            if started.elapsed() >= self.timeout {
                child.terminate();
                return Err(BackendError::new(
                    BackendErrorKind::Timeout,
                    "helper invocation timed out",
                ));
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
                        return Err(BackendError::new(
                            BackendErrorKind::Process,
                            "stdin writer disconnected",
                        ));
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
                        return Err(BackendError::new(
                            BackendErrorKind::Process,
                            format!("poll helper: {error}"),
                        ));
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
            let kind = if status.code() == Some(HELPER_MODEL_REJECTION_EXIT_CODE) {
                BackendErrorKind::Model
            } else {
                BackendErrorKind::Process
            };
            return Err(BackendError::new(
                kind,
                format!(
                    "helper exited {}: {}",
                    status,
                    String::from_utf8_lossy(&stderr)
                ),
            ));
        }
        Ok(HelperOutput {
            stdout,
            stderr,
            started,
        })
    }

    fn check_active(&self, started: Instant) -> Result<(), BackendError> {
        if self.cancellation.is_cancelled() {
            return Err(BackendError::new(
                BackendErrorKind::Cancelled,
                "helper invocation cancelled",
            ));
        }
        if started.elapsed() >= self.timeout {
            return Err(BackendError::new(
                BackendErrorKind::Timeout,
                "helper invocation timed out",
            ));
        }
        Ok(())
    }

    fn validate_checkpoint(&self, actual: Option<&str>) -> Result<(), BackendError> {
        if self.mode == "model"
            && actual != Some(encode_hex(&self.model.provenance().checkpoint_sha256).as_str())
        {
            return Err(BackendError::new(
                BackendErrorKind::Model,
                "helper executed a checkpoint different from model provenance".to_string(),
            ));
        }
        Ok(())
    }
}

impl PyBackend {
    #[cfg(target_os = "linux")]
    fn with_production_linux_execution(
        mut self,
        rootfs: PathBuf,
        cgroup: PathBuf,
        weights_dir: PathBuf,
        process_limits: BackendProcessLimits,
    ) -> Self {
        self.execution = PyExecutionEnvironment::ProductionLinux {
            rootfs,
            cgroup,
            weights_dir,
            process_limits,
        };
        self
    }
}

struct HelperOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    started: Instant,
}

fn enforce_io_limit(name: &str, actual: usize, limit: u64) -> Result<(), BackendError> {
    if actual as u128 > u128::from(limit) {
        Err(BackendError::new(
            BackendErrorKind::ResourceLimit,
            format!("helper {name} exceeds byte limit ({actual} > {limit})"),
        ))
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
            let result = stdin.write_all(&request).map_err(|error| {
                BackendError::new(BackendErrorKind::Process, format!("write request: {error}"))
            });
            let _ = sender.send(result);
        })
        .map_err(|error| {
            BackendError::new(
                BackendErrorKind::Process,
                format!("spawn helper stdin writer: {error}"),
            )
        })?;
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
                            break Err(BackendError::new(
                                BackendErrorKind::ResourceLimit,
                                format!("helper {name} exceeds byte limit"),
                            ));
                        };
                        if total as u128 > u128::from(limit) {
                            break Err(BackendError::new(
                                BackendErrorKind::ResourceLimit,
                                format!("helper {name} exceeds byte limit ({total} > {limit})"),
                            ));
                        }
                        bytes.extend_from_slice(&chunk[..read]);
                    }
                    Err(error) => {
                        break Err(BackendError::new(
                            BackendErrorKind::Process,
                            format!("read helper {name}: {error}"),
                        ));
                    }
                }
            };
            let _ = sender.send(result);
        })
        .map_err(|error| {
            BackendError::new(
                BackendErrorKind::Process,
                format!("spawn helper {spawn_name} reader: {error}"),
            )
        })?;
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
        Err(TryRecvError::Disconnected) => Err(BackendError::new(
            BackendErrorKind::Process,
            format!("{name} reader disconnected"),
        )),
    }
}

fn terminate_with<T>(child: &mut SupervisedChild, message: &str) -> Result<T, BackendError> {
    child.terminate();
    Err(BackendError::new(
        BackendErrorKind::Internal,
        message.to_string(),
    ))
}

#[cfg(target_os = "linux")]
fn helper_command(
    python: &str,
    helper: &PathBuf,
    execution: &PyExecutionEnvironment,
) -> Result<Command, BackendError> {
    verify_linux_containment()?;
    let mut command = Command::new(BUBBLEWRAP_PATH);
    match execution {
        PyExecutionEnvironment::Developer => {
            // Developer bridge: process/write containment only. Host filesystem
            // remains visible and inherited environment remains trusted.
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
        }
        PyExecutionEnvironment::ProductionLinux {
            rootfs,
            weights_dir,
            process_limits,
            ..
        } => {
            command
                .env_clear()
                .env("PYTHONNOUSERSITE", "1")
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .env("PYTHONHASHSEED", "0")
                .env("CUDA_VISIBLE_DEVICES", "")
                .env("HIP_VISIBLE_DEVICES", "")
                .env("ROCR_VISIBLE_DEVICES", "")
                .env("LAMQUANT_WEIGHTS_DIR", weights_dir)
                .env(
                    "OMP_NUM_THREADS",
                    process_limits.cpu_slots.get().to_string(),
                )
                .env(
                    "MKL_NUM_THREADS",
                    process_limits.cpu_slots.get().to_string(),
                )
                .env(
                    "OPENBLAS_NUM_THREADS",
                    process_limits.cpu_slots.get().to_string(),
                )
                .env(
                    "NUMEXPR_NUM_THREADS",
                    process_limits.cpu_slots.get().to_string(),
                )
                .arg("--unshare-all")
                .arg("--cap-drop")
                .arg("ALL")
                .arg("--die-with-parent")
                .arg("--new-session")
                .arg("--ro-bind")
                .arg(rootfs)
                .arg("/")
                .arg("--dev")
                .arg("/dev")
                .arg("--proc")
                .arg("/proc")
                .arg("--tmpfs")
                .arg("/tmp")
                .arg("--chdir")
                .arg("/")
                .arg("--")
                .arg(python)
                .arg("-I")
                .arg(helper);
        }
    }
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
        .map_err(|error| BackendError::new(BackendErrorKind::Deployment, error))
}

#[cfg(windows)]
fn helper_command(
    python: &str,
    helper: &PathBuf,
    _execution: &PyExecutionEnvironment,
) -> Result<Command, BackendError> {
    let mut command = Command::new(python);
    command.arg(helper);
    Ok(command)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn helper_command(
    _python: &str,
    _helper: &PathBuf,
    _execution: &PyExecutionEnvironment,
) -> Result<Command, BackendError> {
    Err(BackendError::new(
        BackendErrorKind::Deployment,
        "bounded helper process containment is unsupported on this Unix platform".to_string(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn helper_command(
    _python: &str,
    _helper: &PathBuf,
    _execution: &PyExecutionEnvironment,
) -> Result<Command, BackendError> {
    Err(BackendError::new(
        BackendErrorKind::Deployment,
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

#[cfg(target_os = "linux")]
fn configure_process_limits(
    command: &mut Command,
    execution: &PyExecutionEnvironment,
) -> Result<Option<File>, BackendError> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;

    let PyExecutionEnvironment::ProductionLinux { cgroup, .. } = execution else {
        return Ok(None);
    };
    let membership = OpenOptions::new()
        .write(true)
        .open(cgroup.join("cgroup.procs"))
        .map_err(|error| {
            BackendError::new(
                BackendErrorKind::Deployment,
                format!("open production cgroup membership: {error}"),
            )
        })?;
    let membership_fd = membership.as_raw_fd();
    // SAFETY: callback invokes only async-signal-safe libc operations and uses
    // stack storage. The descriptor is live at fork; the child inherits its
    // own copy. The parent closes its copy after spawn returns.
    unsafe {
        command.pre_exec(move || write_own_pid_to_cgroup(membership_fd));
    }
    Ok(Some(membership))
}

#[cfg(target_os = "linux")]
fn write_own_pid_to_cgroup(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    let pid = unsafe { libc::getpid() };
    let mut digits = [0_u8; 32];
    let mut cursor = digits.len();
    let mut value = pid as u32;
    loop {
        cursor -= 1;
        digits[cursor] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    let bytes = &digits[cursor..];
    // SAFETY: `fd` is a live cgroup.procs descriptor and `bytes` is valid.
    let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
    if written == bytes.len() as isize {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_process_limits(
    _command: &mut Command,
    _execution: &PyExecutionEnvironment,
) -> Result<Option<File>, BackendError> {
    Ok(None)
}

#[cfg(unix)]
struct ProcessTree {
    process_group: i32,
}

#[cfg(unix)]
impl ProcessTree {
    fn attach(child: &Child) -> Result<Self, BackendError> {
        let process_group = i32::try_from(child.id()).map_err(|_| {
            BackendError::new(
                BackendErrorKind::Process,
                "helper pid exceeds process-group range".to_string(),
            )
        })?;
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
                return Err(BackendError::new(
                    BackendErrorKind::Deployment,
                    "create helper job object failed".to_string(),
                ));
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
                return Err(BackendError::new(
                    BackendErrorKind::Deployment,
                    "configure helper job object failed".to_string(),
                ));
            }
            let process = child.as_raw_handle() as HANDLE;
            if process == null_mut() as HANDLE || AssignProcessToJobObject(job, process) == 0 {
                CloseHandle(job);
                return Err(BackendError::new(
                    BackendErrorKind::Deployment,
                    "assign helper to job object failed".to_string(),
                ));
            }
            if NtResumeProcess(process) < 0 {
                TerminateJobObject(job, 1);
                CloseHandle(job);
                return Err(BackendError::new(
                    BackendErrorKind::Process,
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
        Err(BackendError::new(
            BackendErrorKind::Deployment,
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
        signal_domain: SignalDomain::DigitalInteger,
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
        signal_domain: SignalDomain::PhysicalMicrovoltQ16,
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

    fn model(&self) -> BackendModel<'_> {
        match &self.model {
            PyBackendModel::ModelFree(provenance) => BackendModel::ModelFree(provenance.clone()),
            PyBackendModel::Trained(artifact) => BackendModel::trained(artifact),
        }
    }

    fn encode(
        &self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        if self.cancellation.is_cancelled() {
            return Err(BackendError::new(
                BackendErrorKind::Cancelled,
                "helper invocation cancelled",
            ));
        }
        let samples = signal.channels.first().map_or(0, Vec::len);
        self.capabilities
            .validate_signal(signal, sample_rate)
            .map_err(|error| {
                BackendError::new(
                    BackendErrorKind::Capability,
                    format!("helper capability mismatch: {error:?}"),
                )
            })?;
        let request_capacity =
            preflight_encode_request(&signal.channels, self.io_limits.maximum_request_bytes)?;
        let (rate_numerator, rate_denominator) = sample_rate.parts();
        let sample_rate = rate_numerator as f64 / rate_denominator as f64;
        let output = self.call(
            EncodeRequest {
                op: "encode",
                sample_rate,
                signal: &signal.channels,
                signal_domain: signal.domain.protocol_name(),
            },
            request_capacity,
        )?;
        self.check_active(output.started)?;
        let envelope: EncodeResponse<'_> = parse_envelope(&output)?;
        self.validate_checkpoint(envelope.checkpoint_sha256)?;
        let input_elements = signal.channels.len().checked_mul(samples).ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "signal response limit overflow",
            )
        })?;
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
            .map_err(|error| {
                BackendError::new(
                    BackendErrorKind::Capability,
                    format!("helper capability mismatch: {error:?}"),
                )
            })?;
        if usize::from(tokens.n_channels) != signal.channels.len()
            || usize::try_from(tokens.n_samples).ok() != Some(samples)
        {
            return Err(BackendError::new(
                BackendErrorKind::Capability,
                "helper response shape differs from input signal".to_string(),
            ));
        }
        self.check_active(output.started)?;
        Ok(tokens)
    }

    fn decode(&self, t: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
        self.capabilities.validate_output(t).map_err(|error| {
            BackendError::new(
                BackendErrorKind::Capability,
                format!("helper capability mismatch: {error:?}"),
            )
        })?;
        let request_capacity = preflight_decode_request(t, self.io_limits.maximum_request_bytes)?;
        preflight_decode_response(t, self.io_limits.maximum_stdout_bytes)?;
        let output = self.call(
            DecodeRequest {
                op: "decode",
                tokens: &t.tokens,
                schedule: &t.schedule,
                alphabet: t.alphabet,
                n_channels: t.n_channels,
                n_samples: t.n_samples,
                backend_meta: &t.backend_meta,
                signal_domain: self.capabilities.signal_domain.protocol_name(),
            },
            request_capacity,
        )?;
        self.check_active(output.started)?;
        let envelope: DecodeResponse<'_> = parse_envelope(&output)?;
        self.validate_checkpoint(envelope.checkpoint_sha256)?;
        let signal = parse_bounded_matrix(
            envelope.signal,
            usize::from(t.n_channels),
            usize::try_from(t.n_samples).map_err(|_| {
                BackendError::new(
                    BackendErrorKind::ResourceLimit,
                    "sample count exceeds host usize",
                )
            })?,
            "signal",
        )?;
        self.check_active(output.started)?;
        Ok(NeuralSignal {
            domain: self.capabilities.signal_domain,
            channels: signal,
        })
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
        BackendError::new(
            BackendErrorKind::Protocol,
            format!(
                "parse response envelope: {error} (stderr: {})",
                String::from_utf8_lossy(&output.stderr)
            ),
        )
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
            return Err(DeError::custom(format_args!(
                "`{}` exceeds element limit {}",
                self.name, self.maximum
            )));
        }
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.maximum));
        while let Some(value) = sequence.next_element()? {
            if values.len() == self.maximum {
                return Err(DeError::custom(format_args!(
                    "`{}` exceeds element limit {}",
                    self.name, self.maximum
                )));
            }
            values.push(value);
        }
        if self.exact.is_some_and(|exact| values.len() != exact) {
            return Err(DeError::custom(format_args!(
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
        .map_err(|error| {
            BackendError::new(
                BackendErrorKind::Protocol,
                format!("parse `{name}`: {error}"),
            )
        })?;
    deserializer.end().map_err(|error| {
        BackendError::new(
            BackendErrorKind::Protocol,
            format!("parse `{name}` trailing data: {error}"),
        )
    })?;
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
            return Err(DeError::custom(format_args!(
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
            return Err(DeError::custom(format_args!(
                "`{}` exceeds row limit {}",
                self.name, self.rows
            )));
        }
        if rows.len() != self.rows {
            return Err(DeError::custom(format_args!(
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
    .map_err(|error| {
        BackendError::new(
            BackendErrorKind::Protocol,
            format!("parse `{name}`: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        BackendError::new(
            BackendErrorKind::Protocol,
            format!("parse `{name}` trailing data: {error}"),
        )
    })?;
    Ok(values)
}

fn preflight_encode_request(
    signal: &[Vec<i64>],
    maximum_request_bytes: u64,
) -> Result<usize, BackendError> {
    let samples = signal.first().map_or(0, Vec::len);
    if signal.is_empty() || samples == 0 || signal.iter().any(|row| row.len() != samples) {
        return Err(BackendError::new(
            BackendErrorKind::Capability,
            "signal must be non-empty rectangular channels".to_string(),
        ));
    }
    let elements = (signal.len() as u128)
        .checked_mul(samples as u128)
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "signal request size overflow".to_string(),
            )
        })?;
    let estimate = elements
        .checked_mul(u128::from(JSON_I64_BYTES))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "signal request size overflow".to_string(),
            )
        })?;
    if estimate > u128::from(maximum_request_bytes) {
        return Err(BackendError::new(
            BackendErrorKind::ResourceLimit,
            format!(
                "helper request exceeds byte limit ({estimate} estimated > {maximum_request_bytes})"
            ),
        ));
    }
    usize::try_from(estimate).map_err(|_| {
        BackendError::new(
            BackendErrorKind::ResourceLimit,
            "signal request size exceeds host usize".to_string(),
        )
    })
}

fn preflight_decode_request(
    tokens: &NeuralTokens,
    maximum_request_bytes: u64,
) -> Result<usize, BackendError> {
    let token_bytes = (tokens.tokens.len() as u128)
        .checked_mul(u128::from(JSON_I32_BYTES))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "token request size overflow".to_string(),
            )
        })?;
    let schedule_bytes = (tokens.schedule.len() as u128)
        .checked_mul(u128::from(JSON_U8_BYTES))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "schedule request size overflow".to_string(),
            )
        })?;
    let metadata_bytes = (tokens.backend_meta.len() as u128)
        .checked_mul(u128::from(JSON_U8_BYTES))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "metadata request size overflow".to_string(),
            )
        })?;
    let estimate = token_bytes
        .checked_add(schedule_bytes)
        .and_then(|bytes| bytes.checked_add(metadata_bytes))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "decode request size overflow".to_string(),
            )
        })?;
    if estimate > u128::from(maximum_request_bytes) {
        return Err(BackendError::new(
            BackendErrorKind::ResourceLimit,
            format!(
                "helper request exceeds byte limit ({estimate} estimated > {maximum_request_bytes})"
            ),
        ));
    }
    usize::try_from(estimate).map_err(|_| {
        BackendError::new(
            BackendErrorKind::ResourceLimit,
            "decode request size exceeds host usize".to_string(),
        )
    })
}

fn preflight_decode_response(
    tokens: &NeuralTokens,
    maximum_stdout_bytes: u64,
) -> Result<(), BackendError> {
    let elements = u128::from(tokens.n_channels)
        .checked_mul(u128::from(tokens.n_samples))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "decoded response size overflow".to_string(),
            )
        })?;
    let estimate = elements
        .checked_mul(u128::from(JSON_I64_BYTES))
        .and_then(|bytes| bytes.checked_add(u128::from(JSON_OVERHEAD_BYTES)))
        .ok_or_else(|| {
            BackendError::new(
                BackendErrorKind::ResourceLimit,
                "decoded response size overflow".to_string(),
            )
        })?;
    if estimate > u128::from(maximum_stdout_bytes) {
        return Err(BackendError::new(
            BackendErrorKind::ResourceLimit,
            format!(
                "helper decoded response exceeds stdout byte limit \
                 ({estimate} estimated > {maximum_stdout_bytes})"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelInputContract;
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

    fn trained_model() -> TrainedModelArtifact {
        TrainedModelArtifact::new(model(), model_input_contract())
    }

    fn rate() -> Rational {
        Rational::new(250, 1).unwrap()
    }

    fn digital(channels: Vec<Vec<i64>>) -> NeuralSignal {
        NeuralSignal::digital(channels)
    }

    fn model_input_contract() -> ModelInputContract {
        ModelInputContract::new(
            semantic_abir::ConceptId::new("abir:modality/eeg").unwrap(),
            (0..21)
                .map(|index| {
                    semantic_abir::ConceptId::new(format!("lamquant:test-channel/{index}")).unwrap()
                })
                .collect(),
            ContentId::from_bytes([4; 32]),
            Rational::new(250, 1).unwrap(),
            2_500,
            SignalDomain::PhysicalMicrovoltQ16,
            semantic_abir::ConceptId::new("lamquant:operation/model-input-v1").unwrap(),
            semantic_abir::ConceptId::new("lamquant:proof/model-input-v1").unwrap(),
            semantic_abir::ConceptId::new("lamquant:backend-pipeline/subband-v1").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn pre_cancelled_helper_never_spawns() {
        let cancellation = BackendCancellation::default();
        cancellation.cancel();
        let backend = PyBackend::selftest("missing-executable", "missing-helper", model())
            .with_cancellation(cancellation);
        let error = backend
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("cancelled backend must fail");
        assert!(error.message().contains("cancelled"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn model_backend_enforces_exact_capabilities_before_spawn() {
        let backend = PyBackend::model("missing-executable", "missing-helper", trained_model());
        let backend_model = backend.model();
        let contract = backend_model
            .input_contract()
            .expect("trained model must bind semantic input");
        assert_eq!(contract.channel_concepts().len(), 21);
        assert_eq!(contract.sample_rate(), Rational::new(250, 1).unwrap());
        assert_eq!(contract.samples(), 2_500);
        assert_eq!(contract.signal_domain(), SignalDomain::PhysicalMicrovoltQ16);
        let error = backend
            .encode(
                &NeuralSignal::physical_microvolt_q16(vec![vec![1]]),
                Rational::new(1, 1).unwrap(),
            )
            .expect_err("unsupported model shape and rate must fail before spawn");
        assert!(
            error.message().contains("ChannelCount"),
            "unexpected capability failure: {}",
            error.message()
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
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("zero timeout must fail before spawn");
        assert!(error.message().contains("timed out"));
        assert!(
            !marker.exists(),
            "helper performed a pre-timeout side effect"
        );
        let _ = fs::remove_file(helper);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn helper_rejection_is_a_nonretryable_model_failure() {
        use std::os::unix::fs::PermissionsExt;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-reject-helper-{unique}.sh"));
        fs::write(
            &helper,
            "#!/bin/sh\ncat >/dev/null\nprintf 'model rejected request' >&2\nexit 64\n",
        )
        .unwrap();
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
        let backend = PyBackend::selftest("sh", &helper, model());
        let error = backend
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("helper rejection must fail");
        assert_eq!(error.kind(), BackendErrorKind::Model);
        assert!(error.message().contains("model rejected request"));
        let _ = fs::remove_file(helper);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn helper_signal_termination_is_a_process_failure() {
        use std::os::unix::fs::PermissionsExt;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let helper = std::env::temp_dir().join(format!("lmq-signal-helper-{unique}.sh"));
        fs::write(&helper, "#!/bin/sh\ncat >/dev/null\nkill -KILL $$\n").unwrap();
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
        let backend = PyBackend::selftest("sh", &helper, model());
        let error = backend
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("signal termination must fail");
        assert_eq!(error.kind(), BackendErrorKind::Process);
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
        assert!(
            error.message().contains("exceeds element limit 4"),
            "{}",
            error.message()
        );
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
                .message()
                .contains("decoded response exceeds stdout byte limit"),
            "{}",
            error.message()
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
            .encode(&digital(vec![vec![1; 10_000]]), rate())
            .expect_err("hung helper must time out");
        let _ = fs::remove_file(helper);
        assert!(error.message().contains("timed out"));
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
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("oversized output must fail");
        let _ = fs::remove_file(helper);
        assert!(
            error.message().contains("stdout exceeds byte limit"),
            "{}",
            error.message()
        );
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
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("estimated request at limit must fail before spawn");
        assert!(error.message().contains("request exceeds byte limit"));
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
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("process tree must time out");
        assert!(error.message().contains("timed out"));
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
            .encode(&digital(vec![vec![1]]), rate())
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
            .encode(&digital(vec![vec![1]]), rate())
            .expect_err("unsupported Unix process containment must fail closed");
        assert!(
            error.message().contains("Unavailable"),
            "{}",
            error.message()
        );
    }
}
