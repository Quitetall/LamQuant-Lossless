use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use semantic_abir::{ContentId, ElementType, Rational};
use serde::de::{DeserializeSeed, Error as DeError, SeqAccess, Visitor};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use super::{
    BackendCancellation, BackendIoLimits, BackendProcessLimits, PyBackend, PyExecutionEnvironment,
};
use crate::backend::{
    BackendError, BackendErrorKind, BackendModel, NeuralBackend, NeuralBackendCapabilities,
    NeuralSignal, NeuralTokens, TrainedModelArtifact,
};

pub struct ProductionPyBackendConfig {
    pub rootfs: PathBuf,
    pub runtime_manifest: PathBuf,
    /// Absolute path inside `rootfs`.
    pub python: PathBuf,
    /// Absolute path inside `rootfs`.
    pub helper: PathBuf,
    /// Absolute path inside `rootfs`.
    pub checkpoint: PathBuf,
    pub cgroup: PathBuf,
    pub model: TrainedModelArtifact,
    pub io_limits: BackendIoLimits,
    pub timeout: std::time::Duration,
    pub process_limits: BackendProcessLimits,
    pub cancellation: BackendCancellation,
}

/// Loader-produced runtime closure and enforced-resource attestation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionDeploymentAttestation {
    closure_id: ContentId,
    execution_id: ContentId,
    checkpoint_content_id: ContentId,
    checkpoint_sha256: [u8; 32],
    process_limits: BackendProcessLimits,
    io_limits: BackendIoLimits,
}

impl ProductionDeploymentAttestation {
    pub const fn closure_id(&self) -> ContentId {
        self.closure_id
    }

    pub const fn execution_id(&self) -> ContentId {
        self.execution_id
    }

    pub const fn checkpoint_content_id(&self) -> ContentId {
        self.checkpoint_content_id
    }

    pub const fn checkpoint_sha256(&self) -> [u8; 32] {
        self.checkpoint_sha256
    }

    pub const fn process_limits(&self) -> BackendProcessLimits {
        self.process_limits
    }

    pub const fn io_limits(&self) -> BackendIoLimits {
        self.io_limits
    }
}

/// Production Python backend. Construction fails unless runtime closure is
/// complete, immutable, checkpoint-bound, and executed under kernel-enforced
/// Linux cgroup-v2 ceilings.
pub struct ProductionPyBackend {
    backend: PyBackend,
    attestation: ProductionDeploymentAttestation,
    runtime: VerifiedRuntimeClosure,
    execution_lock: std::sync::Mutex<ProductionExecutionState>,
}

struct ProductionExecutionState {
    last_full_revalidation: std::time::Instant,
}

#[derive(Debug)]
pub struct ProductionBackendLoadError(String);

impl std::fmt::Display for ProductionBackendLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ProductionBackendLoadError {}

#[derive(Clone)]
struct VerifiedRuntimeFile {
    relative_path: PathBuf,
    sha256: [u8; 32],
    bytes: u64,
    executable: bool,
}

#[derive(Clone)]
struct VerifiedRuntimeClosure {
    rootfs: PathBuf,
    files: Vec<VerifiedRuntimeFile>,
    bubblewrap_sha256: [u8; 32],
    closure_id: ContentId,
    root_device: u64,
    root_inode: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifest {
    version: u16,
    files: Vec<RuntimeManifestFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifestFile {
    path: String,
    sha256: String,
    bytes: u64,
    executable: bool,
}

const MAX_RUNTIME_FILES: usize = 200_000;
const MAX_RUNTIME_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
const FULL_RUNTIME_REVALIDATION_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(5 * 60);
const PRODUCTION_EXECUTION_POLICY: &[u8] =
    b"org.quitetall.lamquant.lmq.production-py-execution-policy-v2";

impl ProductionPyBackend {
    pub fn load(config: ProductionPyBackendConfig) -> Result<Self, ProductionBackendLoadError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = config;
            return Err(ProductionBackendLoadError(
                "production Python backend currently requires Linux cgroup-v2 and Bubblewrap"
                    .into(),
            ));
        }
        #[cfg(target_os = "linux")]
        {
            validate_production_limits(config.process_limits)?;
            validate_io_limits(config.io_limits)?;
            if config.timeout.is_zero() {
                return Err(ProductionBackendLoadError(
                    "production backend timeout must be nonzero".into(),
                ));
            }
            let rootfs = config
                .rootfs
                .canonicalize()
                .map_err(|error| load_error("canonicalize runtime root", error))?;
            if !rootfs.is_dir() || !filesystem_is_read_only(&rootfs)? {
                return Err(ProductionBackendLoadError(
                    "production runtime root must be a read-only mounted filesystem".into(),
                ));
            }
            let (root_device, root_inode) = runtime_root_identity(&rootfs)?;
            let manifest_bytes = read_bounded_file(
                &config.runtime_manifest,
                MAX_RUNTIME_MANIFEST_BYTES,
                "runtime manifest",
            )?;
            reject_duplicate_json_members(&manifest_bytes)?;
            let manifest: RuntimeManifest =
                serde_json::from_slice(&manifest_bytes).map_err(|error| {
                    ProductionBackendLoadError(format!("parse runtime manifest: {error}"))
                })?;
            if manifest.version != 1
                || manifest.files.is_empty()
                || manifest.files.len() > MAX_RUNTIME_FILES
            {
                return Err(ProductionBackendLoadError(
                    "runtime manifest version/count is invalid".into(),
                ));
            }
            let files = normalize_runtime_manifest(manifest.files)?;
            let bubblewrap_sha256 = sha256_file(Path::new(super::BUBBLEWRAP_PATH))?.0;
            super::verify_linux_containment()
                .map_err(|error| ProductionBackendLoadError(error.message().to_string()))?;
            let closure_id = verify_runtime_tree(&rootfs, &files, bubblewrap_sha256)?;

            let python = virtual_runtime_path(&rootfs, &config.python, "python")?;
            let helper = virtual_runtime_path(&rootfs, &config.helper, "helper")?;
            let checkpoint = virtual_runtime_path(&rootfs, &config.checkpoint, "checkpoint")?;
            if config.checkpoint.file_name().and_then(|name| name.to_str())
                != Some("student_subband.ckpt")
            {
                return Err(ProductionBackendLoadError(
                    "production checkpoint must use helper-resolved name student_subband.ckpt"
                        .into(),
                ));
            }
            let python_relative = virtual_relative_path(&config.python, "python")?;
            let helper_relative = virtual_relative_path(&config.helper, "helper")?;
            let checkpoint_relative = virtual_relative_path(&config.checkpoint, "checkpoint")?;
            let by_path = files
                .iter()
                .map(|file| (file.relative_path.as_path(), file))
                .collect::<BTreeMap<_, _>>();
            let python_entry = by_path.get(python_relative.as_path()).ok_or_else(|| {
                ProductionBackendLoadError("python absent from runtime closure".into())
            })?;
            if !python_entry.executable {
                return Err(ProductionBackendLoadError(
                    "production Python interpreter is not executable".into(),
                ));
            }
            if !by_path.contains_key(helper_relative.as_path())
                || !by_path.contains_key(checkpoint_relative.as_path())
            {
                return Err(ProductionBackendLoadError(
                    "helper or checkpoint absent from runtime closure".into(),
                ));
            }

            let checkpoint_content_id_expected = config.model.provenance().checkpoint_content_id;
            let checkpoint_sha256_expected = config.model.provenance().checkpoint_sha256;
            let checkpoint_entry = by_path
                .get(checkpoint_relative.as_path())
                .expect("checked checkpoint closure membership");
            if checkpoint_entry.sha256 != checkpoint_sha256_expected {
                return Err(ProductionBackendLoadError(
                    "checkpoint SHA-256 differs from trained model provenance".into(),
                ));
            }
            let checkpoint_content_id = payload_content_id_streaming(&checkpoint)?;
            if checkpoint_content_id != checkpoint_content_id_expected {
                return Err(ProductionBackendLoadError(
                    "checkpoint ContentId differs from trained model provenance".into(),
                ));
            }
            let execution_id = production_execution_id(
                closure_id,
                &python_relative,
                &helper_relative,
                &checkpoint_relative,
                checkpoint_content_id,
                checkpoint_sha256_expected,
                config.process_limits,
                config.io_limits,
                config.timeout,
            );

            let cgroup = configure_linux_cgroup(&config.cgroup, config.process_limits)?;
            let python_virtual = config.python.to_string_lossy().into_owned();
            let helper_virtual = config.helper.clone();
            let backend = PyBackend::model(python_virtual, helper_virtual, config.model)
                .with_timeout(config.timeout)
                .with_io_limits(config.io_limits)
                .with_cancellation(config.cancellation)
                .with_production_linux_execution(
                    rootfs.clone(),
                    cgroup,
                    config
                        .checkpoint
                        .parent()
                        .expect("absolute checkpoint has a parent")
                        .to_path_buf(),
                    config.process_limits,
                );
            debug_assert_eq!(python, rootfs.join(python_relative));
            debug_assert_eq!(helper, rootfs.join(helper_relative));
            Ok(Self {
                backend,
                attestation: ProductionDeploymentAttestation {
                    closure_id,
                    execution_id,
                    checkpoint_content_id,
                    checkpoint_sha256: checkpoint_sha256_expected,
                    process_limits: config.process_limits,
                    io_limits: config.io_limits,
                },
                runtime: VerifiedRuntimeClosure {
                    rootfs,
                    files,
                    bubblewrap_sha256,
                    closure_id,
                    root_device,
                    root_inode,
                },
                execution_lock: std::sync::Mutex::new(ProductionExecutionState {
                    last_full_revalidation: std::time::Instant::now(),
                }),
            })
        }
    }

    pub const fn deployment_attestation(&self) -> &ProductionDeploymentAttestation {
        &self.attestation
    }

    pub fn revalidate_deployment(&self) -> Result<(), ProductionBackendLoadError> {
        let mut execution = self
            .execution_lock
            .lock()
            .map_err(|_| ProductionBackendLoadError("production execution lock poisoned".into()))?;
        self.revalidate_deployment_locked(&mut execution)
    }

    fn revalidate_deployment_locked(
        &self,
        execution: &mut ProductionExecutionState,
    ) -> Result<(), ProductionBackendLoadError> {
        #[cfg(not(target_os = "linux"))]
        {
            Err(ProductionBackendLoadError(
                "production Python backend is unavailable on this platform".into(),
            ))
        }
        #[cfg(target_os = "linux")]
        {
            if !filesystem_is_read_only(&self.runtime.rootfs)? {
                return Err(ProductionBackendLoadError(
                    "production runtime root is no longer read-only".into(),
                ));
            }
            if runtime_root_identity(&self.runtime.rootfs)?
                != (self.runtime.root_device, self.runtime.root_inode)
            {
                return Err(ProductionBackendLoadError(
                    "production runtime root identity changed".into(),
                ));
            }
            if execution.last_full_revalidation.elapsed() >= FULL_RUNTIME_REVALIDATION_INTERVAL {
                let actual = verify_runtime_tree(
                    &self.runtime.rootfs,
                    &self.runtime.files,
                    self.runtime.bubblewrap_sha256,
                )?;
                if actual != self.runtime.closure_id {
                    return Err(ProductionBackendLoadError(
                        "production runtime closure identity changed".into(),
                    ));
                }
                execution.last_full_revalidation = std::time::Instant::now();
            }
            let PyExecutionEnvironment::ProductionLinux {
                cgroup,
                process_limits,
                ..
            } = &self.backend.execution
            else {
                return Err(ProductionBackendLoadError(
                    "production backend lost enforced execution environment".into(),
                ));
            };
            verify_linux_cgroup(cgroup, *process_limits)?;
            if !read_trimmed(&cgroup.join("cgroup.procs"))?.is_empty() {
                return Err(ProductionBackendLoadError(
                    "production inference cgroup is occupied".into(),
                ));
            }
            let current_bubblewrap_sha256 = sha256_file(Path::new(super::BUBBLEWRAP_PATH))?.0;
            if current_bubblewrap_sha256 != self.runtime.bubblewrap_sha256 {
                return Err(ProductionBackendLoadError(
                    "trusted Bubblewrap executable changed".into(),
                ));
            }
            Ok(())
        }
    }
}

impl NeuralBackend for ProductionPyBackend {
    fn capabilities(&self) -> NeuralBackendCapabilities {
        self.backend.capabilities()
    }

    fn model(&self) -> BackendModel<'_> {
        self.backend.model()
    }

    fn encode(
        &self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        let mut execution = self.execution_lock.lock().map_err(|_| {
            BackendError::new(
                BackendErrorKind::Internal,
                "production execution lock poisoned",
            )
        })?;
        self.revalidate_deployment_locked(&mut execution)
            .map_err(|error| BackendError::new(BackendErrorKind::Deployment, error.to_string()))?;
        self.backend.encode(signal, sample_rate)
    }

    fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
        let mut execution = self.execution_lock.lock().map_err(|_| {
            BackendError::new(
                BackendErrorKind::Internal,
                "production execution lock poisoned",
            )
        })?;
        self.revalidate_deployment_locked(&mut execution)
            .map_err(|error| BackendError::new(BackendErrorKind::Deployment, error.to_string()))?;
        self.backend.decode(tokens)
    }
}

#[cfg(target_os = "linux")]
fn validate_production_limits(
    limits: BackendProcessLimits,
) -> Result<(), ProductionBackendLoadError> {
    if limits.maximum_tasks.get() < limits.cpu_slots.get()
        || limits.memory_bytes.get() < 64 * 1024 * 1024
    {
        return Err(ProductionBackendLoadError(
            "process limits require tasks >= CPU slots and at least 64 MiB".into(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_io_limits(limits: BackendIoLimits) -> Result<(), ProductionBackendLoadError> {
    if limits.maximum_request_bytes == 0
        || limits.maximum_stdout_bytes == 0
        || limits.maximum_stderr_bytes == 0
    {
        return Err(ProductionBackendLoadError(
            "production backend I/O limits must be nonzero".into(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn normalize_runtime_manifest(
    entries: Vec<RuntimeManifestFile>,
) -> Result<Vec<VerifiedRuntimeFile>, ProductionBackendLoadError> {
    let mut files = Vec::with_capacity(entries.len());
    let mut paths = BTreeSet::new();
    for entry in entries {
        let relative_path = normalized_relative_path(&entry.path)?;
        if !paths.insert(relative_path.clone()) || entry.bytes == 0 {
            return Err(ProductionBackendLoadError(
                "runtime manifest contains duplicate paths or empty files".into(),
            ));
        }
        let sha256 = parse_sha256(&entry.sha256).ok_or_else(|| {
            ProductionBackendLoadError("runtime manifest SHA-256 is invalid".into())
        })?;
        files.push(VerifiedRuntimeFile {
            relative_path,
            sha256,
            bytes: entry.bytes,
            executable: entry.executable,
        });
    }
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

#[cfg(target_os = "linux")]
fn normalized_relative_path(value: &str) -> Result<PathBuf, ProductionBackendLoadError> {
    if value.is_empty() || value.contains('\\') {
        return Err(ProductionBackendLoadError(
            "runtime path must use nonempty POSIX-relative syntax".into(),
        ));
    }
    let path = Path::new(value);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ProductionBackendLoadError(
            "runtime path must be normalized and relative".into(),
        ));
    }
    Ok(path.to_path_buf())
}

#[cfg(target_os = "linux")]
fn virtual_relative_path(value: &Path, name: &str) -> Result<PathBuf, ProductionBackendLoadError> {
    let mut components = value.components();
    if !matches!(components.next(), Some(Component::RootDir)) {
        return Err(ProductionBackendLoadError(format!(
            "production {name} path must be absolute inside rootfs"
        )));
    }
    let mut relative = PathBuf::new();
    for component in components {
        let Component::Normal(component) = component else {
            return Err(ProductionBackendLoadError(format!(
                "production {name} path is not normalized"
            )));
        };
        relative.push(component);
    }
    if relative.as_os_str().is_empty() {
        return Err(ProductionBackendLoadError(format!(
            "production {name} path is empty"
        )));
    }
    Ok(relative)
}

#[cfg(target_os = "linux")]
fn virtual_runtime_path(
    rootfs: &Path,
    value: &Path,
    name: &str,
) -> Result<PathBuf, ProductionBackendLoadError> {
    let relative = virtual_relative_path(value, name)?;
    let host = rootfs.join(relative);
    if !host.is_file() {
        return Err(ProductionBackendLoadError(format!(
            "production {name} path is not a regular file"
        )));
    }
    Ok(host)
}

#[cfg(target_os = "linux")]
fn verify_runtime_tree(
    rootfs: &Path,
    expected: &[VerifiedRuntimeFile],
    bubblewrap_sha256: [u8; 32],
) -> Result<ContentId, ProductionBackendLoadError> {
    let mut actual_paths = Vec::new();
    let mut entries_seen = 0;
    collect_runtime_files(
        rootfs,
        rootfs,
        &mut actual_paths,
        &mut entries_seen,
        MAX_RUNTIME_FILES,
    )?;
    actual_paths.sort();
    let expected_paths = expected
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    if actual_paths != expected_paths {
        return Err(ProductionBackendLoadError(
            "runtime manifest does not cover exact rootfs file closure".into(),
        ));
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmq.production-py-closure-v1\0");
    hasher.update(&bubblewrap_sha256);
    for file in expected {
        let host_path = rootfs.join(&file.relative_path);
        let (sha256, bytes) = sha256_file(&host_path)?;
        if sha256 != file.sha256 || bytes != file.bytes {
            return Err(ProductionBackendLoadError(format!(
                "runtime file differs from manifest: {}",
                file.relative_path.display()
            )));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executable = fs::metadata(&host_path)
                .map_err(|error| load_error("read runtime file metadata", error))?
                .permissions()
                .mode()
                & 0o111
                != 0;
            if executable != file.executable {
                return Err(ProductionBackendLoadError(format!(
                    "runtime executable bit differs from manifest: {}",
                    file.relative_path.display()
                )));
            }
        }
        hash_closure_field(&mut hasher, file.relative_path.to_string_lossy().as_bytes());
        hash_closure_field(&mut hasher, &file.sha256);
        hash_closure_field(&mut hasher, &file.bytes.to_le_bytes());
        hash_closure_field(&mut hasher, &[u8::from(file.executable)]);
    }
    Ok(ContentId::from_bytes(*hasher.finalize().as_bytes()))
}

#[cfg(target_os = "linux")]
fn collect_runtime_files(
    rootfs: &Path,
    current: &Path,
    output: &mut Vec<PathBuf>,
    entries_seen: &mut usize,
    maximum_entries: usize,
) -> Result<(), ProductionBackendLoadError> {
    let entries =
        fs::read_dir(current).map_err(|error| load_error("read runtime directory", error))?;
    for entry in entries {
        if *entries_seen >= maximum_entries {
            return Err(ProductionBackendLoadError(
                "runtime closure exceeds entry-count limit".into(),
            ));
        }
        *entries_seen += 1;
        let entry = entry.map_err(|error| load_error("read runtime directory entry", error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| load_error("read runtime entry metadata", error))?;
        if metadata.file_type().is_symlink() {
            return Err(ProductionBackendLoadError(
                "runtime closure must not contain symbolic links".into(),
            ));
        }
        if metadata.is_dir() {
            collect_runtime_files(rootfs, &path, output, entries_seen, maximum_entries)?;
        } else if metadata.is_file() {
            output.push(
                path.strip_prefix(rootfs)
                    .map_err(|_| ProductionBackendLoadError("runtime path escaped root".into()))?
                    .to_path_buf(),
            );
        } else {
            return Err(ProductionBackendLoadError(
                "runtime closure contains a non-file object".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn read_bounded_file(
    path: &Path,
    maximum: u64,
    name: &str,
) -> Result<Vec<u8>, ProductionBackendLoadError> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| load_error(&format!("open {name}"), error))?;
    let metadata = file
        .metadata()
        .map_err(|error| load_error(&format!("read {name} metadata"), error))?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(ProductionBackendLoadError(format!(
            "{name} is not a bounded regular file"
        )));
    }
    let read_limit = maximum
        .checked_add(1)
        .ok_or_else(|| ProductionBackendLoadError(format!("{name} byte limit overflow")))?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .map_err(|_| ProductionBackendLoadError(format!("{name} size exceeds host usize")))?,
    );
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| load_error(&format!("read {name}"), error))?;
    if bytes.len() as u64 > maximum {
        return Err(ProductionBackendLoadError(format!(
            "{name} exceeds byte limit"
        )));
    }
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn sha256_file(path: &Path) -> Result<([u8; 32], u64), ProductionBackendLoadError> {
    let mut file = File::open(path).map_err(|error| load_error("open closure file", error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| load_error("hash closure file", error))?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| ProductionBackendLoadError("closure file size overflow".into()))?;
        hasher.update(&buffer[..read]);
    }
    Ok((hasher.finalize().into(), bytes))
}

#[cfg(target_os = "linux")]
fn payload_content_id_streaming(path: &Path) -> Result<ContentId, ProductionBackendLoadError> {
    let mut file = File::open(path).map_err(|error| load_error("open checkpoint", error))?;
    let mut hasher = semantic_abir::PayloadContentHasher::new(ElementType::Bytes);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| load_error("hash checkpoint ContentId", error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize())
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn production_execution_id(
    closure_id: ContentId,
    python_relative: &Path,
    helper_relative: &Path,
    checkpoint_relative: &Path,
    checkpoint_content_id: ContentId,
    checkpoint_sha256: [u8; 32],
    process_limits: BackendProcessLimits,
    io_limits: BackendIoLimits,
    timeout: std::time::Duration,
) -> ContentId {
    use std::os::unix::ffi::OsStrExt;

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmq.production-py-execution-v1\0");
    hash_closure_field(&mut hasher, PRODUCTION_EXECUTION_POLICY);
    hash_closure_field(&mut hasher, closure_id.as_bytes());
    hash_closure_field(&mut hasher, python_relative.as_os_str().as_bytes());
    hash_closure_field(&mut hasher, helper_relative.as_os_str().as_bytes());
    hash_closure_field(&mut hasher, checkpoint_relative.as_os_str().as_bytes());
    hash_closure_field(&mut hasher, checkpoint_content_id.as_bytes());
    hash_closure_field(&mut hasher, &checkpoint_sha256);
    hash_closure_field(
        &mut hasher,
        &process_limits.memory_bytes.get().to_le_bytes(),
    );
    hash_closure_field(&mut hasher, &process_limits.cpu_slots.get().to_le_bytes());
    hash_closure_field(
        &mut hasher,
        &process_limits.maximum_tasks.get().to_le_bytes(),
    );
    hash_closure_field(&mut hasher, &io_limits.maximum_request_bytes.to_le_bytes());
    hash_closure_field(&mut hasher, &io_limits.maximum_stdout_bytes.to_le_bytes());
    hash_closure_field(&mut hasher, &io_limits.maximum_stderr_bytes.to_le_bytes());
    hash_closure_field(&mut hasher, &timeout.as_nanos().to_le_bytes());
    ContentId::from_bytes(*hasher.finalize().as_bytes())
}

#[cfg(target_os = "linux")]
fn hash_closure_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(target_os = "linux")]
fn parse_sha256(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

#[cfg(target_os = "linux")]
fn load_error(context: &str, error: impl std::fmt::Display) -> ProductionBackendLoadError {
    ProductionBackendLoadError(format!("{context}: {error}"))
}

#[cfg(target_os = "linux")]
fn filesystem_is_read_only(path: &Path) -> Result<bool, ProductionBackendLoadError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| ProductionBackendLoadError("runtime root contains NUL".into()))?;
    let mut stats = core::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is NUL-terminated and `stats` points to writable storage.
    let result = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return Err(load_error(
            "inspect runtime filesystem",
            std::io::Error::last_os_error(),
        ));
    }
    // SAFETY: successful `statvfs` initialized `stats`.
    let stats = unsafe { stats.assume_init() };
    Ok(stats.f_flag & libc::ST_RDONLY as libc::c_ulong != 0)
}

#[cfg(target_os = "linux")]
fn runtime_root_identity(path: &Path) -> Result<(u64, u64), ProductionBackendLoadError> {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::symlink_metadata(path)
        .map_err(|error| load_error("read runtime root metadata", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ProductionBackendLoadError(
            "production runtime root is not a concrete directory".into(),
        ));
    }
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(target_os = "linux")]
fn configure_linux_cgroup(
    path: &Path,
    limits: BackendProcessLimits,
) -> Result<PathBuf, ProductionBackendLoadError> {
    let path = path
        .canonicalize()
        .map_err(|error| load_error("canonicalize cgroup", error))?;
    let cgroup_root = Path::new("/sys/fs/cgroup")
        .canonicalize()
        .map_err(|error| load_error("canonicalize cgroup-v2 root", error))?;
    if path == cgroup_root
        || !path.starts_with(&cgroup_root)
        || !cgroup_root.join("cgroup.controllers").is_file()
    {
        return Err(ProductionBackendLoadError(
            "production cgroup must be a delegated cgroup-v2 child".into(),
        ));
    }
    if !read_trimmed(&path.join("cgroup.procs"))?.is_empty() {
        return Err(ProductionBackendLoadError(
            "production cgroup must be empty at load".into(),
        ));
    }
    write_cgroup_value(
        &path.join("memory.max"),
        &limits.memory_bytes.get().to_string(),
    )?;
    write_cgroup_value(&path.join("memory.swap.max"), "0")?;
    write_cgroup_value(
        &path.join("pids.max"),
        &limits.maximum_tasks.get().to_string(),
    )?;
    let quota = u64::from(limits.cpu_slots.get())
        .checked_mul(100_000)
        .ok_or_else(|| ProductionBackendLoadError("CPU quota overflow".into()))?;
    write_cgroup_value(&path.join("cpu.max"), &format!("{quota} 100000"))?;
    verify_linux_cgroup(&path, limits)?;
    Ok(path)
}

#[cfg(target_os = "linux")]
fn verify_linux_cgroup(
    path: &Path,
    limits: BackendProcessLimits,
) -> Result<(), ProductionBackendLoadError> {
    let memory = read_trimmed(&path.join("memory.max"))?;
    let swap = read_trimmed(&path.join("memory.swap.max"))?;
    let tasks = read_trimmed(&path.join("pids.max"))?;
    let cpu = read_trimmed(&path.join("cpu.max"))?;
    let quota = u64::from(limits.cpu_slots.get())
        .checked_mul(100_000)
        .ok_or_else(|| ProductionBackendLoadError("CPU quota overflow".into()))?;
    let expected_cpu = format!("{quota} 100000");
    if memory != limits.memory_bytes.get().to_string()
        || swap != "0"
        || tasks != limits.maximum_tasks.get().to_string()
        || cpu.split_ascii_whitespace().collect::<Vec<_>>()
            != expected_cpu.split_ascii_whitespace().collect::<Vec<_>>()
    {
        return Err(ProductionBackendLoadError(
            "kernel cgroup limits differ from deployment attestation".into(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_cgroup_value(path: &Path, value: &str) -> Result<(), ProductionBackendLoadError> {
    fs::write(path, value).map_err(|error| load_error("configure cgroup limit", error))
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: &Path) -> Result<String, ProductionBackendLoadError> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_string())
        .map_err(|error| load_error("read cgroup value", error))
}

#[cfg(target_os = "linux")]
fn reject_duplicate_json_members(bytes: &[u8]) -> Result<(), ProductionBackendLoadError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    UniqueJsonSeed
        .deserialize(&mut deserializer)
        .map_err(|error| ProductionBackendLoadError(format!("parse runtime manifest: {error}")))?;
    deserializer
        .end()
        .map_err(|error| ProductionBackendLoadError(format!("parse runtime manifest: {error}")))
}

#[derive(Clone, Copy)]
struct UniqueJsonSeed;

impl<'de> DeserializeSeed<'de> for UniqueJsonSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JSON with unique object members")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(UniqueJsonSeed)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(DeError::custom("duplicate JSON object member"));
            }
            map.next_value_seed(UniqueJsonSeed)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelInputContract;
    use crate::backend::SignalDomain;
    use semantic_abir::ContentId;
    use semantic_abir::Rational;
    use semantic_abir_bcs::ModelProvenance;
    use std::fs;
    use std::num::{NonZeroU16, NonZeroU64};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn model() -> ModelProvenance {
        ModelProvenance {
            checkpoint_content_id: ContentId::from_bytes([1; 32]),
            checkpoint_sha256: [2; 32],
            pccp_change_id: "test".to_string(),
            pccp_evidence_id: ContentId::from_bytes([3; 32]),
            pccp_status: semantic_abir_bcs::PccpStatus::Candidate,
        }
    }

    fn trained_model() -> TrainedModelArtifact {
        TrainedModelArtifact::new(model(), model_input_contract())
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
    fn production_manifest_rejects_duplicate_members_and_path_escape() {
        assert!(reject_duplicate_json_members(br#"{"version":1,"version":1,"files":[]}"#).is_err());
        assert!(normalize_runtime_manifest(vec![RuntimeManifestFile {
            path: "../escape".into(),
            sha256: "00".repeat(32),
            bytes: 1,
            executable: false,
        }])
        .is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bounded_manifest_reader_uses_one_descriptor_and_hard_byte_limit() {
        use std::os::unix::fs::symlink;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lmq-bounded-read-{unique}"));
        fs::create_dir(&root).unwrap();
        let oversized = root.join("oversized.json");
        File::create(&oversized).unwrap().set_len(17).unwrap();
        assert!(read_bounded_file(&oversized, 16, "runtime manifest").is_err());

        let target = root.join("target.json");
        fs::write(&target, b"{}").unwrap();
        let alias = root.join("alias.json");
        symlink(&target, &alias).unwrap();
        assert!(read_bounded_file(&alias, 16, "runtime manifest").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn production_execution_identity_binds_selected_roles_and_exact_limits() {
        let closure = ContentId::from_bytes([0x11; 32]);
        let checkpoint = ContentId::from_bytes([0x22; 32]);
        let process = BackendProcessLimits {
            memory_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
            cpu_slots: NonZeroU16::new(2).unwrap(),
            maximum_tasks: NonZeroU16::new(4).unwrap(),
        };
        let io = BackendIoLimits {
            maximum_request_bytes: 1_024,
            maximum_stdout_bytes: 2_048,
            maximum_stderr_bytes: 512,
        };
        let identity = production_execution_id(
            closure,
            Path::new("opt/python"),
            Path::new("opt/helper.py"),
            Path::new("weights/student_subband.ckpt"),
            checkpoint,
            [0x33; 32],
            process,
            io,
            Duration::from_secs(30),
        );
        assert_ne!(
            identity,
            production_execution_id(
                closure,
                Path::new("opt/python"),
                Path::new("opt/alternate-helper.py"),
                Path::new("weights/student_subband.ckpt"),
                checkpoint,
                [0x33; 32],
                process,
                io,
                Duration::from_secs(30),
            )
        );
        assert_ne!(
            identity,
            production_execution_id(
                closure,
                Path::new("opt/python"),
                Path::new("opt/helper.py"),
                Path::new("weights/student_subband.ckpt"),
                checkpoint,
                [0x33; 32],
                process,
                BackendIoLimits {
                    maximum_request_bytes: 512,
                    maximum_stdout_bytes: 2_560,
                    maximum_stderr_bytes: 512,
                },
                Duration::from_secs(30),
            )
        );
        assert_ne!(
            identity,
            production_execution_id(
                closure,
                Path::new("opt/python"),
                Path::new("opt/helper.py"),
                Path::new("weights/student_subband.ckpt"),
                checkpoint,
                [0x33; 32],
                process,
                io,
                Duration::from_secs(31),
            )
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn runtime_tree_traversal_rejects_before_exceeding_entry_bound() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lmq-runtime-entry-bound-{unique}"));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("first"), b"1").unwrap();
        fs::write(root.join("second"), b"2").unwrap();
        let mut paths = Vec::new();
        let mut entries_seen = 0;
        assert!(
            collect_runtime_files(&root, &root, &mut paths, &mut entries_seen, 1).is_err(),
            "second directory entry must fail before retention"
        );
        assert!(paths.len() <= 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn production_runtime_closure_binds_every_file_and_mode() {
        use std::os::unix::fs::PermissionsExt;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lmq-runtime-closure-{unique}"));
        fs::create_dir_all(root.join("opt/lamquant")).unwrap();
        let python = root.join("opt/lamquant/python");
        let helper = root.join("opt/lamquant/helper.py");
        fs::write(&python, b"python-runtime").unwrap();
        fs::write(&helper, b"print('bound helper')").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o555)).unwrap();
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o444)).unwrap();

        let (python_sha256, python_bytes) = sha256_file(&python).unwrap();
        let (helper_sha256, helper_bytes) = sha256_file(&helper).unwrap();
        let files = vec![
            VerifiedRuntimeFile {
                relative_path: PathBuf::from("opt/lamquant/helper.py"),
                sha256: helper_sha256,
                bytes: helper_bytes,
                executable: false,
            },
            VerifiedRuntimeFile {
                relative_path: PathBuf::from("opt/lamquant/python"),
                sha256: python_sha256,
                bytes: python_bytes,
                executable: true,
            },
        ];
        let bubblewrap_sha256 = [0x77; 32];
        let first = verify_runtime_tree(&root, &files, bubblewrap_sha256).unwrap();
        assert_eq!(
            first,
            verify_runtime_tree(&root, &files, bubblewrap_sha256).unwrap()
        );
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o644)).unwrap();
        fs::write(&helper, b"mutated helper").unwrap();
        assert!(verify_runtime_tree(&root, &files, bubblewrap_sha256).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn production_loader_rejects_writable_runtime_root_before_execution() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("lmq-production-root-{unique}"));
        fs::create_dir(&root).unwrap();
        let error = super::ProductionPyBackend::load(ProductionPyBackendConfig {
            rootfs: root.clone(),
            runtime_manifest: root.join("manifest.json"),
            python: PathBuf::from("/usr/bin/python3"),
            helper: PathBuf::from("/opt/lamquant/lmq_infer.py"),
            checkpoint: PathBuf::from("/opt/lamquant/model.ckpt"),
            cgroup: PathBuf::from("/sys/fs/cgroup/missing-lmq-test"),
            model: trained_model(),
            io_limits: super::BackendIoLimits::default(),
            timeout: std::time::Duration::from_secs(1),
            process_limits: BackendProcessLimits {
                memory_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
                cpu_slots: NonZeroU16::new(1).unwrap(),
                maximum_tasks: NonZeroU16::new(4).unwrap(),
            },
            cancellation: super::BackendCancellation::default(),
        })
        .err()
        .expect("writable runtime roots cannot mint production attestation");
        assert!(error.to_string().contains("read-only"));
        fs::remove_dir_all(root).unwrap();
    }
}
