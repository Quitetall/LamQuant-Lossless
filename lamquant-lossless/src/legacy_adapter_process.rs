//! Bounded process boundary for retired codec operations.
//!
//! Current code sends one JSON request to the independent legacy Adapter,
//! enforces wall-clock and memory ceilings, captures bounded diagnostics, and
//! verifies the returned receipt plus destination bytes. No retired decoder is
//! linked into this crate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

type Error = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_MAX_RSS_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_PROTOCOL_BYTES: usize = 1024 * 1024;
const IO_CHUNK_BYTES: usize = 64 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Supervision policy for one retired-format Adapter process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyAdapterConfig {
    /// Adapter binary path. Defaults to `lamquant-legacy-adapter` on `PATH`.
    pub executable: PathBuf,
    /// Per-entry wall-clock ceiling.
    pub timeout: Duration,
    /// Per-process resident-memory ceiling. Linux additionally applies the same
    /// value as an address-space hard limit before `exec`.
    pub max_rss_bytes: u64,
}

impl LegacyAdapterConfig {
    /// Build policy from bounded environment overrides.
    ///
    /// - `LAMQUANT_LEGACY_ADAPTER`: executable path
    /// - `LAMQUANT_LEGACY_ADAPTER_TIMEOUT_SECS`: positive integer
    /// - `LAMQUANT_LEGACY_ADAPTER_MAX_RSS_MIB`: positive integer
    pub fn from_env() -> Result<Self, Error> {
        let executable = std::env::var_os("LAMQUANT_LEGACY_ADAPTER")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("lamquant-legacy-adapter"));
        let timeout_seconds = parse_positive_env(
            "LAMQUANT_LEGACY_ADAPTER_TIMEOUT_SECS",
            DEFAULT_TIMEOUT_SECONDS,
        )?;
        let max_rss_mib = parse_positive_env(
            "LAMQUANT_LEGACY_ADAPTER_MAX_RSS_MIB",
            DEFAULT_MAX_RSS_BYTES / (1024 * 1024),
        )?;
        let max_rss_bytes = max_rss_mib
            .checked_mul(1024 * 1024)
            .ok_or("legacy Adapter RSS limit overflows u64")?;
        Ok(Self {
            executable,
            timeout: Duration::from_secs(timeout_seconds),
            max_rss_bytes,
        })
    }

    fn validate(&self) -> Result<(), Error> {
        platform_supervision_supported()?;
        if self.executable.as_os_str().is_empty() {
            return Err("legacy Adapter executable is empty".into());
        }
        if self.timeout.is_zero() {
            return Err("legacy Adapter timeout must be positive".into());
        }
        if self.max_rss_bytes == 0 {
            return Err("legacy Adapter RSS limit must be positive".into());
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn platform_supervision_supported() -> Result<(), Error> {
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn platform_supervision_supported() -> Result<(), Error> {
    Err("bounded legacy Adapter execution currently requires Linux process supervision".into())
}

fn parse_positive_env(name: &str, default: u64) -> Result<u64, Error> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    let text = value
        .into_string()
        .map_err(|_| format!("{name} is not valid UTF-8"))?;
    let parsed = text
        .parse::<u64>()
        .map_err(|error| format!("{name} must be a positive integer: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "kebab-case")]
enum ProcessRequest<'a> {
    Manifest,
    MaterializeExact(MaterializeRequest<'a>),
    MaterializeSyntheticExact(SyntheticMaterializeRequest<'a>),
}

#[derive(Serialize)]
struct MaterializeRequest<'a> {
    source: &'a Path,
    destination: &'a Path,
    accept_fidelity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_sha256: Option<&'a str>,
    original_size: u64,
    max_source_bytes: u64,
    max_decoded_bytes: u64,
    max_output_bytes: u64,
}

#[derive(Serialize)]
struct SyntheticMaterializeRequest<'a> {
    source: &'a Path,
    destination: &'a Path,
    accept_fidelity: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_sha256: Option<&'a str>,
    original_size: u64,
    max_source_bytes: u64,
    max_decoded_bytes: u64,
    max_intermediate_bytes: u64,
    max_output_bytes: u64,
    format: &'a str,
    template: &'a serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct MaterializeReceipt {
    pub profile: String,
    pub source_blake3: String,
    pub source_bytes: u64,
    pub output_sha256: String,
    pub output_bytes: u64,
    pub source_preserved: bool,
    pub exact_original_bytes: bool,
}

#[derive(Deserialize)]
#[serde(tag = "status", content = "value", rename_all = "kebab-case")]
enum ProcessResponse {
    OkManifest(CapabilityManifest),
    OkMaterialization(MaterializeReceipt),
    Error { code: String, message: String },
}

#[derive(Deserialize)]
struct CapabilityManifest {
    schema: String,
    process_protocol: String,
    capabilities: Vec<Capability>,
}

#[derive(Deserialize)]
struct Capability {
    profile: String,
    #[serde(default)]
    parent_verified_materialization: bool,
}

/// Require explicit support before issuing a digest-free staged request.
pub(crate) fn require_parent_verified_materialization(
    config: &LegacyAdapterConfig,
) -> Result<(), Error> {
    config.validate()?;
    let request = serde_json::to_vec(&ProcessRequest::Manifest)?;
    let output = run_process(config, &request)?;
    let response: ProcessResponse = serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "legacy Adapter returned invalid manifest JSON: {error}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    let manifest = match response {
        ProcessResponse::OkManifest(manifest) => manifest,
        ProcessResponse::Error { code, message } => {
            return Err(format!("legacy Adapter refused manifest [{code}]: {message}").into())
        }
        ProcessResponse::OkMaterialization(_) => {
            return Err("legacy Adapter returned materialization for manifest request".into())
        }
    };
    if manifest.schema != "lamquant.legacy-capabilities/v1"
        || manifest.process_protocol != "abir.adapter-process/v1"
        || !manifest.capabilities.iter().any(|capability| {
            capability.profile == "legacy.lml1.v1" && capability.parent_verified_materialization
        })
    {
        return Err("legacy Adapter does not advertise parent-verified materialization".into());
    }
    Ok(())
}

/// Materialize one exact retired source file and verify every receipt field.
pub(crate) fn materialize_exact(
    config: &LegacyAdapterConfig,
    source: &Path,
    destination: &Path,
    expected_sha256: Option<&str>,
    original_size: u64,
    max_decoded_bytes: u64,
) -> Result<MaterializeReceipt, Error> {
    config.validate()?;
    if destination.exists() {
        return Err(format!(
            "legacy Adapter destination already exists: {}",
            destination.display()
        )
        .into());
    }
    let source_metadata = std::fs::symlink_metadata(source)?;
    if !source_metadata.file_type().is_file() {
        return Err(format!(
            "legacy Adapter source is not a regular file: {}",
            source.display()
        )
        .into());
    }
    let source_bytes = source_metadata.len();
    let source_blake3 = blake3_file(source)?;
    let request = ProcessRequest::MaterializeExact(MaterializeRequest {
        source,
        destination,
        accept_fidelity: true,
        expected_sha256,
        original_size,
        max_source_bytes: source_bytes,
        max_decoded_bytes,
        max_output_bytes: original_size,
    });
    let request = serde_json::to_vec(&request)?;
    if request.len() > MAX_PROTOCOL_BYTES {
        return Err("legacy Adapter request exceeds 1 MiB protocol limit".into());
    }
    let output = run_process(config, &request)?;
    let response: ProcessResponse = serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "legacy Adapter returned invalid JSON: {error}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    let receipt = match response {
        ProcessResponse::OkMaterialization(receipt) => receipt,
        ProcessResponse::OkManifest(_) => {
            return Err("legacy Adapter returned manifest for materialization request".into())
        }
        ProcessResponse::Error { code, message } => {
            return Err(
                format!("legacy Adapter refused materialization [{code}]: {message}").into(),
            )
        }
    };
    if receipt.profile != "legacy.lml1.v1"
        || receipt.source_blake3 != source_blake3
        || receipt.source_bytes != source_bytes
        || receipt.output_bytes != original_size
        || !receipt.source_preserved
        || receipt.exact_original_bytes != expected_sha256.is_some()
        || expected_sha256.is_some_and(|expected| receipt.output_sha256 != expected)
    {
        return Err("legacy Adapter returned a contradictory materialization receipt".into());
    }

    let output_metadata = std::fs::symlink_metadata(destination)?;
    if !output_metadata.file_type().is_file() || output_metadata.len() != original_size {
        return Err("legacy Adapter output extent contradicts its receipt".into());
    }
    if sha256_file(destination)? != receipt.output_sha256 {
        return Err("legacy Adapter output SHA-256 contradicts its receipt".into());
    }
    Ok(receipt)
}

/// Materialize one exact pre-synthesis source file from retired LML1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_synthetic_exact(
    config: &LegacyAdapterConfig,
    source: &Path,
    destination: &Path,
    expected_sha256: Option<&str>,
    original_size: u64,
    max_decoded_bytes: u64,
    format: &str,
    template: &serde_json::Value,
) -> Result<MaterializeReceipt, Error> {
    config.validate()?;
    if destination.exists() {
        return Err(format!(
            "legacy Adapter destination already exists: {}",
            destination.display()
        )
        .into());
    }
    let source_metadata = std::fs::symlink_metadata(source)?;
    if !source_metadata.file_type().is_file() {
        return Err(format!(
            "legacy Adapter source is not a regular file: {}",
            source.display()
        )
        .into());
    }
    let source_bytes = source_metadata.len();
    let source_blake3 = blake3_file(source)?;
    let request = ProcessRequest::MaterializeSyntheticExact(SyntheticMaterializeRequest {
        source,
        destination,
        accept_fidelity: true,
        expected_sha256,
        original_size,
        max_source_bytes: source_bytes,
        max_decoded_bytes,
        max_intermediate_bytes: max_decoded_bytes,
        max_output_bytes: original_size,
        format,
        template,
    });
    let request = serde_json::to_vec(&request)?;
    if request.len() > MAX_PROTOCOL_BYTES {
        return Err("legacy Adapter request exceeds 1 MiB protocol limit".into());
    }
    let output = run_process(config, &request)?;
    let response: ProcessResponse = serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "legacy Adapter returned invalid JSON: {error}; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    let receipt = match response {
        ProcessResponse::OkMaterialization(receipt) => receipt,
        ProcessResponse::OkManifest(_) => {
            return Err("legacy Adapter returned manifest for materialization request".into())
        }
        ProcessResponse::Error { code, message } => {
            return Err(
                format!("legacy Adapter refused materialization [{code}]: {message}").into(),
            )
        }
    };
    if receipt.profile != "legacy.lml1.v1"
        || receipt.source_blake3 != source_blake3
        || receipt.source_bytes != source_bytes
        || receipt.output_bytes != original_size
        || !receipt.source_preserved
        || receipt.exact_original_bytes != expected_sha256.is_some()
        || expected_sha256.is_some_and(|expected| receipt.output_sha256 != expected)
    {
        return Err("legacy Adapter returned a contradictory materialization receipt".into());
    }
    let output_metadata = std::fs::symlink_metadata(destination)?;
    if !output_metadata.file_type().is_file() || output_metadata.len() != original_size {
        return Err("legacy Adapter output extent contradicts its receipt".into());
    }
    if sha256_file(destination)? != receipt.output_sha256 {
        return Err("legacy Adapter output SHA-256 contradicts its receipt".into());
    }
    Ok(receipt)
}

struct ProcessOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_process(config: &LegacyAdapterConfig, request: &[u8]) -> Result<ProcessOutput, Error> {
    let deadline = Instant::now()
        .checked_add(config.timeout)
        .ok_or("legacy Adapter timeout overflows monotonic clock")?;
    let mut command = Command::new(&config.executable);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_child(&mut command, config.max_rss_bytes)?;
    let mut child = command.spawn().map_err(|error| {
        format!(
            "cannot start legacy Adapter `{}`: {error}",
            config.executable.display()
        )
    })?;
    let stdout = child.stdout.take().ok_or("legacy Adapter stdout missing")?;
    let stderr = child.stderr.take().ok_or("legacy Adapter stderr missing")?;
    let stdout_reader = thread::spawn(move || read_capped(stdout, MAX_PROTOCOL_BYTES));
    let stderr_reader = thread::spawn(move || read_capped(stderr, MAX_PROTOCOL_BYTES));
    let mut stdin = child.stdin.take().ok_or("legacy Adapter stdin missing")?;
    let request = request.to_vec();
    let writer = thread::spawn(move || stdin.write_all(&request));

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => break Err(format!("cannot inspect legacy Adapter process: {error}")),
        }
        if Instant::now() >= deadline {
            break Err(format!(
                "legacy Adapter exceeded {:.3} second timeout",
                config.timeout.as_secs_f64()
            ));
        }
        #[cfg(target_os = "linux")]
        if let Some(rss_bytes) = match linux_resident_bytes(child.id()) {
            Ok(rss_bytes) => rss_bytes,
            Err(error) => {
                break Err(error.to_string());
            }
        } {
            if rss_bytes > config.max_rss_bytes {
                break Err(format!(
                    "legacy Adapter RSS {rss_bytes} exceeds {} byte limit",
                    config.max_rss_bytes
                ));
            }
        }
        thread::sleep(POLL_INTERVAL);
    };

    // Always kill the process group before joining pipes. A helper process may
    // otherwise inherit stdout/stderr and keep readers blocked after its parent
    // exits or the supervision policy fires.
    terminate_child(&mut child);
    let _ = child.wait();
    let write_result = writer
        .join()
        .map_err(|_| "legacy Adapter stdin writer panicked")?;
    let stdout = join_reader(stdout_reader, "stdout");
    let stderr = join_reader(stderr_reader, "stderr");
    let status = status.map_err(|error| -> Error { error.into() })?;
    write_result.map_err(|error| format!("cannot write legacy Adapter request: {error}"))?;
    let stdout = stdout?;
    let stderr = stderr?;
    if !status.success() {
        return Err(process_failure(status, &stderr).into());
    }
    Ok(ProcessOutput { stdout, stderr })
}

fn read_capped(mut source: impl Read, limit: usize) -> std::io::Result<Vec<u8>> {
    let mut captured = Vec::new();
    let mut exceeded = false;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if !exceeded {
            let available = limit.saturating_add(1).saturating_sub(captured.len());
            captured.extend_from_slice(&buffer[..count.min(available)]);
            exceeded = captured.len() > limit;
        }
    }
    if exceeded {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legacy Adapter output exceeds 1 MiB protocol limit",
        ))
    } else {
        Ok(captured)
    }
}

fn join_reader(
    handle: thread::JoinHandle<std::io::Result<Vec<u8>>>,
    label: &str,
) -> Result<Vec<u8>, Error> {
    handle
        .join()
        .map_err(|_| format!("legacy Adapter {label} reader panicked"))?
        .map_err(|error| format!("legacy Adapter {label}: {error}").into())
}

fn process_failure(status: ExitStatus, stderr: &[u8]) -> String {
    format!(
        "legacy Adapter exited with {status}: {}",
        String::from_utf8_lossy(stderr).trim()
    )
}

#[cfg(unix)]
fn configure_child(command: &mut Command, max_rss_bytes: u64) -> Result<(), Error> {
    use std::os::unix::process::CommandExt;

    #[cfg(target_os = "linux")]
    let ceiling = rlimit_from_u64(max_rss_bytes)?;

    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            #[cfg(target_os = "linux")]
            {
                let ceiling = libc::rlimit {
                    rlim_cur: ceiling,
                    rlim_max: ceiling,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &ceiling) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[allow(clippy::unnecessary_cast)]
fn rlimit_from_u64(value: u64) -> Result<libc::rlim_t, Error> {
    let converted = value as libc::rlim_t;
    if converted as u128 != u128::from(value) {
        return Err("legacy Adapter RSS limit does not fit this platform's rlim_t".into());
    }
    Ok(converted)
}

#[cfg(not(unix))]
fn configure_child(_command: &mut Command, _max_rss_bytes: u64) -> Result<(), Error> {
    platform_supervision_supported()
}

fn terminate_child(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        let _ = libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(target_os = "linux")]
fn linux_resident_bytes(pid: u32) -> Result<Option<u64>, Error> {
    let path = PathBuf::from(format!("/proc/{pid}/status"));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display()).into()),
    };
    // Process may become a zombie between `try_wait` and this read; Linux then
    // keeps `/proc/<pid>/status` briefly but omits VmRSS. Next loop observes exit.
    let Some(line) = text.lines().find(|line| line.starts_with("VmRSS:")) else {
        return Ok(None);
    };
    let kib = line
        .split_whitespace()
        .nth(1)
        .ok_or("VmRSS line lacks a value")?
        .parse::<u64>()?;
    Ok(Some(
        kib.checked_mul(1024)
            .ok_or("legacy Adapter RSS overflows u64")?,
    ))
}

fn blake3_file(path: &Path) -> Result<String, Error> {
    let mut source = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; IO_CHUNK_BYTES];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut source = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; IO_CHUNK_BYTES];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn timeout_covers_a_child_that_never_reads_stdin() {
        let root = tempfile::tempdir().unwrap();
        let adapter = root.path().join("blocked-adapter.sh");
        std::fs::write(&adapter, "#!/bin/sh\nsleep 60\n").unwrap();
        let mut permissions = std::fs::metadata(&adapter).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&adapter, permissions).unwrap();
        let config = LegacyAdapterConfig {
            executable: adapter,
            timeout: Duration::from_millis(50),
            max_rss_bytes: 128 * 1024 * 1024,
        };
        let request = vec![b'x'; MAX_PROTOCOL_BYTES];
        let started = Instant::now();
        let error = match run_process(&config, &request) {
            Ok(_) => panic!("blocked adapter unexpectedly completed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("timeout"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
