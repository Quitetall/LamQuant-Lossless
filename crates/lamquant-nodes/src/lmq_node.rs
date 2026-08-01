//! PCCP-bound production LMQ node and host execution adapter.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use blut_graph_core::{
    AbirRootType, AbirSemanticType, AbirViewType, Capability, CheckpointMode, CompileError,
    CompiledNode, ConfigField, ConfigSchema, ConfigType, ConfigValue, Determinism, Effect,
    ExecutionError, ExtentContract, FailureContract, FailureEvidence, FidelityContract,
    ImplementationId, KernelDescriptor, KernelId, KernelRegistry, Layout, LeaseAccess,
    LeaseContract, LeaseLifetime, NodeDescriptor, NodeTypeRef, Partiality, PolicyContract,
    PortDescriptor, ProofContract, ResourceEnvelope, StateContract, StateScope, StructuredFailure,
    Target,
};
use ed25519_dalek::{Signature, VerifyingKey};
use fs2::FileExt;
use lamquant_lmq::{
    backend::{
        BackendError, BackendErrorKind, BackendModel, BackendTarget, NeuralBackend,
        NeuralBackendCapabilities, NeuralSignal, NeuralTokens,
    },
    shell,
};
use semantic_abir::{ContentId, Rational};
use semantic_abir_bcs::{CodecFidelity, CodecImplementation, PccpStatus};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::{LamQuantNodeValue, FEATURE_SET, SOURCE_ID};

mod deployment;
mod evidence;
#[cfg(test)]
mod tests;

use deployment::LmqBackendDeployment;
pub use deployment::LmqBackendSession;
#[cfg(test)]
pub(crate) use deployment::{LmqAttestedBackend, LmqBackendDeploymentManifest};
pub use evidence::{verify_pccp_gate_evidence, VerifiedPccpEvidence};

pub const LMQ_NODE_TYPE: &str = "org.quitetall.lamquant.lmq.encode.progressive";
pub const LMQ_MODEL_INPUT_PROOF: &str = "lamquant:proof/model-input-v1";
pub const LMQ_GENERATED_PROOF: &str = "org.quitetall.lamquant.proof.lmq-progressive-generated-v1";
pub const LMQ_CURRENT_PCCP_POLICY: &str =
    "org.quitetall.lamquant.policy.pccp-model-current-authorized-v1";

const CAP_ABIR: &str = "abir.semantic-v1";
const CAP_LMQ: &str = "bcs.lmq.progressive-v1";
const FAILURE_DOMAIN: &str = "org.quitetall.lamquant.lmq.encode";
const INPUT_SEMANTIC_TYPE: &str = "abir.dataset";
const OUTPUT_SEMANTIC_TYPE: &str = "bcs2.bundle.lmq-progressive";
const MODEL_ARTIFACT_CONFIG: &str = "model_artifact_content_id";
const PCCP_EVIDENCE_CONFIG: &str = "pccp_evidence_content_id";
const BACKEND_DEPLOYMENT_CONFIG: &str = "backend_deployment_content_id";
const LMQ_FIDELITY_METRIC: &str = "pccp-validated-model-reconstruction";
const BCS2_TWO_FRAME_OVERHEAD_BYTES: u64 = 128 + 48 + 2 * 128;
const HOST_KERNEL: KernelId = KernelId(0x4c51_0101);
const BLUT_DURABLE_KERNEL: KernelId = KernelId(0x4c51_0102);
const PCCP_AUTHORIZATION_DOMAIN: &[u8] = b"org.quitetall.lamquant.pccp-authorization-snapshot-v1";
const MAX_PCCP_AUTHORIZATION_ENTRIES: usize = 4_096;
const AUTHORITY_MARKER_FILE: &str = "authority";
const AUTHORIZATION_CURRENT_FILE: &str = "current";
const AUTHORIZATION_EPOCH_PREFIX: &str = "epoch-";
const AUTHORIZATION_LOCK_FILE: &str = ".lock";
const AUTHORIZATION_STAGING_PREFIX: &str = ".staging-";
static AUTHORIZATION_STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);
pub const LMQ_PCCP_AUTHORIZATION_MAX_LIFETIME_SECONDS: u64 = 24 * 60 * 60;
pub const LMQ_PCCP_AUTHORIZATION_SNAPSHOT_SCHEMA: &str =
    "org.quitetall.lamquant.pccp-authorization-snapshot-v1";
const MAX_PCCP_AUTHORIZATION_SNAPSHOT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LmqNodeProfileError {
    TrainedArtifactRequired,
    PccpGateNotPassed,
    InvalidModelProvenance,
    InvalidPccpEvidence,
    PccpEvidenceMismatch,
    ModelInputProofMismatch,
    InvalidBackendDeployment,
    InvalidResourceBounds,
    ResourceExtentOverflow,
    BackendArtifactMismatch,
    BackendDeploymentMismatch,
}

impl fmt::Display for LmqNodeProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LmqNodeProfileError {}

/// Immutable identities checked against current PCCP authorization at runtime.
#[derive(Clone, Copy, Debug)]
struct LmqPccpAuthorizationRequest<'a> {
    pub model_artifact_content_id: ContentId,
    pub checkpoint_content_id: ContentId,
    pub checkpoint_sha256: [u8; 32],
    pub pccp_change_id: &'a str,
    pub pccp_evidence_id: ContentId,
}

/// One model authorization carried inside a signed external ledger snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LmqPccpAuthorizationEntry {
    model_artifact_content_id: ContentId,
    checkpoint_content_id: ContentId,
    checkpoint_sha256: [u8; 32],
    pccp_change_id: String,
    pccp_evidence_id: ContentId,
}

impl LmqPccpAuthorizationEntry {
    pub fn new(
        model_artifact_content_id: ContentId,
        checkpoint_content_id: ContentId,
        checkpoint_sha256: [u8; 32],
        pccp_change_id: impl Into<String>,
        pccp_evidence_id: ContentId,
    ) -> Result<Self, LmqNodeProfileError> {
        let pccp_change_id = pccp_change_id.into();
        if model_artifact_content_id.to_bytes() == [0; 32]
            || checkpoint_content_id.to_bytes() == [0; 32]
            || checkpoint_sha256 == [0; 32]
            || pccp_evidence_id.to_bytes() == [0; 32]
            || pccp_change_id.is_empty()
            || pccp_change_id.len() > 128
            || pccp_change_id.trim() != pccp_change_id
            || pccp_change_id.chars().any(char::is_control)
        {
            return Err(LmqNodeProfileError::InvalidPccpEvidence);
        }
        Ok(Self {
            model_artifact_content_id,
            checkpoint_content_id,
            checkpoint_sha256,
            pccp_change_id,
            pccp_evidence_id,
        })
    }

    pub fn from_profile(profile: &LmqNodeProfile) -> Self {
        Self::new(
            profile.model_artifact_content_id,
            profile.checkpoint_content_id,
            profile.checkpoint_sha256,
            profile.pccp_change_id.clone(),
            profile.pccp_evidence_id,
        )
        .expect("validated LMQ profile always yields a valid authorization entry")
    }

    pub const fn model_artifact_content_id(&self) -> ContentId {
        self.model_artifact_content_id
    }

    pub const fn checkpoint_content_id(&self) -> ContentId {
        self.checkpoint_content_id
    }

    pub const fn checkpoint_sha256(&self) -> [u8; 32] {
        self.checkpoint_sha256
    }

    pub fn pccp_change_id(&self) -> &str {
        &self.pccp_change_id
    }

    pub const fn pccp_evidence_id(&self) -> ContentId {
        self.pccp_evidence_id
    }

    fn matches(&self, request: &LmqPccpAuthorizationRequest<'_>) -> bool {
        self.model_artifact_content_id == request.model_artifact_content_id
            && self.checkpoint_content_id == request.checkpoint_content_id
            && self.checkpoint_sha256 == request.checkpoint_sha256
            && self.pccp_change_id == request.pccp_change_id
            && self.pccp_evidence_id == request.pccp_evidence_id
    }
}

/// Signature-bound external snapshot of current model authorization state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedLmqPccpAuthorizationSnapshot {
    epoch: u64,
    not_before_unix_seconds: u64,
    not_after_unix_seconds: u64,
    entries: Vec<LmqPccpAuthorizationEntry>,
    signature: [u8; 64],
}

impl SignedLmqPccpAuthorizationSnapshot {
    pub fn new(
        epoch: u64,
        not_before_unix_seconds: u64,
        not_after_unix_seconds: u64,
        entries: Vec<LmqPccpAuthorizationEntry>,
        signature: [u8; 64],
    ) -> Result<Self, LmqPccpAuthorizationLedgerError> {
        let entries = normalize_authorization_entries(entries)?;
        validate_authorization_window(epoch, not_before_unix_seconds, not_after_unix_seconds)?;
        Ok(Self {
            epoch,
            not_before_unix_seconds,
            not_after_unix_seconds,
            entries,
            signature,
        })
    }

    /// Domain-separated digest signed by the external PCCP authority.
    pub fn signing_message(
        epoch: u64,
        not_before_unix_seconds: u64,
        not_after_unix_seconds: u64,
        entries: Vec<LmqPccpAuthorizationEntry>,
    ) -> Result<[u8; 32], LmqPccpAuthorizationLedgerError> {
        let entries = normalize_authorization_entries(entries)?;
        validate_authorization_window(epoch, not_before_unix_seconds, not_after_unix_seconds)?;
        Ok(authorization_snapshot_digest(
            epoch,
            not_before_unix_seconds,
            not_after_unix_seconds,
            &entries,
        ))
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub const fn not_before_unix_seconds(&self) -> u64 {
        self.not_before_unix_seconds
    }

    pub const fn not_after_unix_seconds(&self) -> u64 {
        self.not_after_unix_seconds
    }

    pub fn entries(&self) -> &[LmqPccpAuthorizationEntry] {
        &self.entries
    }

    pub const fn signature(&self) -> &[u8; 64] {
        &self.signature
    }
}

/// Parse one strict external authorization snapshot without selecting trust.
///
/// Signature trust remains caller-provided. JSON rejects duplicate or unknown
/// members and accepts only exact lowercase hexadecimal identities.
pub fn load_lmq_pccp_authorization_snapshot_json(
    document: &[u8],
) -> Result<SignedLmqPccpAuthorizationSnapshot, LmqPccpAuthorizationLedgerError> {
    if document.is_empty()
        || document.len() > MAX_PCCP_AUTHORIZATION_SNAPSHOT_BYTES
        || !evidence::has_unique_object_members(document)
    {
        return Err(LmqPccpAuthorizationLedgerError::InvalidSnapshot);
    }
    let value: JsonValue = serde_json::from_slice(document)
        .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
    let root = value
        .as_object()
        .filter(|root| {
            has_exact_json_members(
                root,
                &[
                    "schema",
                    "epoch",
                    "not_before_unix_seconds",
                    "not_after_unix_seconds",
                    "entries",
                    "signature",
                ],
            )
        })
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
    if root.get("schema").and_then(JsonValue::as_str)
        != Some(LMQ_PCCP_AUTHORIZATION_SNAPSHOT_SCHEMA)
    {
        return Err(LmqPccpAuthorizationLedgerError::InvalidSnapshot);
    }
    let epoch = required_json_u64(root, "epoch")?;
    let not_before_unix_seconds = required_json_u64(root, "not_before_unix_seconds")?;
    let not_after_unix_seconds = required_json_u64(root, "not_after_unix_seconds")?;
    let raw_entries = root
        .get("entries")
        .and_then(JsonValue::as_array)
        .filter(|entries| entries.len() <= MAX_PCCP_AUTHORIZATION_ENTRIES)
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
    let mut entries = Vec::with_capacity(raw_entries.len());
    for raw_entry in raw_entries {
        let entry = raw_entry
            .as_object()
            .filter(|entry| {
                has_exact_json_members(
                    entry,
                    &[
                        "model_artifact_content_id",
                        "checkpoint_content_id",
                        "checkpoint_sha256",
                        "pccp_change_id",
                        "pccp_evidence_id",
                    ],
                )
            })
            .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
        let model_artifact_content_id =
            ContentId::from_bytes(required_json_hex_32(entry, "model_artifact_content_id")?);
        let checkpoint_content_id =
            ContentId::from_bytes(required_json_hex_32(entry, "checkpoint_content_id")?);
        let checkpoint_sha256 = required_json_hex_32(entry, "checkpoint_sha256")?;
        let pccp_change_id = entry
            .get("pccp_change_id")
            .and_then(JsonValue::as_str)
            .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
        let pccp_evidence_id =
            ContentId::from_bytes(required_json_hex_32(entry, "pccp_evidence_id")?);
        entries.push(
            LmqPccpAuthorizationEntry::new(
                model_artifact_content_id,
                checkpoint_content_id,
                checkpoint_sha256,
                pccp_change_id,
                pccp_evidence_id,
            )
            .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidSnapshot)?,
        );
    }
    let signature = root
        .get("signature")
        .and_then(JsonValue::as_str)
        .and_then(|value| decode_hex_64(value.as_bytes()))
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)?;
    SignedLmqPccpAuthorizationSnapshot::new(
        epoch,
        not_before_unix_seconds,
        not_after_unix_seconds,
        entries,
        signature,
    )
}

/// Verify signature and validity window against caller-provisioned trust.
pub fn verify_current_lmq_pccp_authorization_snapshot(
    trusted_verifying_key: [u8; 32],
    snapshot: &SignedLmqPccpAuthorizationSnapshot,
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let verifying_key = VerifyingKey::from_bytes(&trusted_verifying_key)
        .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidVerifyingKey)?;
    verify_authorization_snapshot(&verifying_key, snapshot)?;
    ensure_authorization_snapshot_current(snapshot)
}

fn has_exact_json_members(root: &JsonMap<String, JsonValue>, required: &[&str]) -> bool {
    root.len() == required.len() && required.iter().all(|name| root.contains_key(*name))
}

fn required_json_u64(
    root: &JsonMap<String, JsonValue>,
    name: &str,
) -> Result<u64, LmqPccpAuthorizationLedgerError> {
    root.get(name)
        .and_then(JsonValue::as_u64)
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)
}

fn required_json_hex_32(
    root: &JsonMap<String, JsonValue>,
    name: &str,
) -> Result<[u8; 32], LmqPccpAuthorizationLedgerError> {
    root.get(name)
        .and_then(JsonValue::as_str)
        .and_then(|value| decode_hex_32(value.as_bytes()))
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidSnapshot)
}

#[derive(Debug)]
struct LmqPccpAuthorizationState {
    epoch: u64,
    not_before_unix_seconds: u64,
    not_after_unix_seconds: u64,
    entries: Vec<LmqPccpAuthorizationEntry>,
}

/// Durable anti-rollback state for one trusted PCCP signing key.
///
/// Directory must already exist, be owned by the service EUID, and reside below
/// an ancestry owned only by that EUID or root. Writable shared ancestors must
/// use sticky-directory semantics. Store must use a local filesystem providing
/// advisory `flock`, atomic same-directory rename/link, and durable directory
/// `fsync`; same-UID and root processes are inside the provisioning trust
/// boundary.
/// Epoch records are append-only; library never deletes or rewrites them.
#[derive(Debug)]
pub struct LmqPccpAuthorizationEpochStore {
    root: PathBuf,
    root_identity: LmqPccpAuthorizationRootIdentity,
    verifying_key: VerifyingKey,
}

impl LmqPccpAuthorizationEpochStore {
    pub fn open(
        root: impl AsRef<Path>,
        verifying_key: [u8; 32],
    ) -> Result<Self, LmqPccpAuthorizationLedgerError> {
        let verifying_key = VerifyingKey::from_bytes(&verifying_key)
            .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidVerifyingKey)?;
        let root =
            fs::canonicalize(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let root_identity = LmqPccpAuthorizationRootIdentity::capture(&root)?;
        let authority_id = authorization_authority_id(&verifying_key);
        let store = Self {
            root,
            root_identity,
            verifying_key,
        };
        let _lock = store.lock_exclusive()?;
        cleanup_epoch_store_staging(&store.root)?;
        ensure_epoch_store_authority(&store.root, authority_id)?;
        recover_epoch_store(&store.root)?;
        Ok(store)
    }

    fn admit_locked(
        &self,
        snapshot: &SignedLmqPccpAuthorizationSnapshot,
        _lock: &LmqPccpAuthorizationStoreLock,
    ) -> Result<(), LmqPccpAuthorizationLedgerError> {
        let digest = authorization_store_record_digest(snapshot);
        let current = recover_epoch_store(&self.root)?;
        if let Some((epoch, current_digest)) = current {
            if snapshot.epoch < epoch {
                return Err(LmqPccpAuthorizationLedgerError::SnapshotRollback);
            }
            if snapshot.epoch == epoch {
                return if digest == current_digest {
                    Ok(())
                } else {
                    Err(LmqPccpAuthorizationLedgerError::SnapshotRollback)
                };
            }
        }
        publish_epoch_store_head(&self.root, snapshot.epoch, digest)?;
        ensure_epoch_audit_record(&self.root, snapshot.epoch, digest)
    }

    fn current_epoch_locked(
        &self,
        _lock: &LmqPccpAuthorizationStoreLock,
    ) -> Result<u64, LmqPccpAuthorizationLedgerError> {
        read_epoch_store_head(&self.root)?
            .map(|(epoch, _)| epoch)
            .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
    }

    fn lock_shared(
        &self,
    ) -> Result<LmqPccpAuthorizationStoreLock, LmqPccpAuthorizationLedgerError> {
        LmqPccpAuthorizationStoreLock::acquire(&self.root, &self.root_identity, false)
    }

    fn lock_exclusive(
        &self,
    ) -> Result<LmqPccpAuthorizationStoreLock, LmqPccpAuthorizationLedgerError> {
        LmqPccpAuthorizationStoreLock::acquire(&self.root, &self.root_identity, true)
    }
}

#[derive(Debug)]
struct LmqPccpAuthorizationRootIdentity {
    directory: File,
}

impl LmqPccpAuthorizationRootIdentity {
    fn capture(root: &Path) -> Result<Self, LmqPccpAuthorizationLedgerError> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let directory = options
            .open(root)
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let identity = Self { directory };
        identity.validate_path(root)?;
        Ok(identity)
    }

    fn validate_path(&self, root: &Path) -> Result<(), LmqPccpAuthorizationLedgerError> {
        let pinned = self
            .directory
            .metadata()
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let current =
            fs::metadata(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        if !pinned.is_dir() || !current.is_dir() {
            return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
        }
        #[cfg(unix)]
        {
            let effective_uid = rustix::process::geteuid().as_raw();
            if pinned.dev() != current.dev()
                || pinned.ino() != current.ino()
                || pinned.uid() != current.uid()
                || pinned.uid() != effective_uid
                || pinned.permissions().mode() & 0o022 != 0
                || current.permissions().mode() & 0o022 != 0
            {
                return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
            }
            let mut ancestor = root;
            while let Some(parent) = ancestor.parent() {
                let parent_metadata = fs::symlink_metadata(parent)
                    .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
                let parent_mode = parent_metadata.permissions().mode();
                let parent_uid = parent_metadata.uid();
                if !parent_metadata.is_dir()
                    || (parent_uid != 0 && parent_uid != effective_uid)
                    || (parent_mode & 0o022 != 0 && parent_mode & 0o1000 == 0)
                {
                    return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
                }
                ancestor = parent;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct LmqPccpAuthorizationStoreLock {
    file: File,
}

impl LmqPccpAuthorizationStoreLock {
    fn acquire(
        root: &Path,
        root_identity: &LmqPccpAuthorizationRootIdentity,
        exclusive: bool,
    ) -> Result<Self, LmqPccpAuthorizationLedgerError> {
        root_identity.validate_path(root)?;
        let path = root.join(AUTHORIZATION_LOCK_FILE);
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = options
            .open(&path)
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let metadata = file
            .metadata()
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let path_metadata = fs::symlink_metadata(&path)
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        validate_epoch_store_file(root, &metadata, &path_metadata)?;
        if exclusive {
            FileExt::lock_exclusive(&file)
        } else {
            FileExt::lock_shared(&file)
        }
        .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        root_identity.validate_path(root)?;
        Ok(Self { file })
    }
}

impl Drop for LmqPccpAuthorizationStoreLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

/// Thread-safe verifier-backed view of signed external PCCP authorization.
///
/// Read leases block a newer signed snapshot until active inference completes.
#[derive(Debug)]
pub struct LmqPccpAuthorizationLedger {
    epoch_store: LmqPccpAuthorizationEpochStore,
    current: RwLock<LmqPccpAuthorizationState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LmqPccpAuthorizationLedgerError {
    InvalidEntry,
    InvalidSnapshot,
    InvalidVerifyingKey,
    InvalidEpochStore,
    EpochStoreIo,
    UntrustedSignature,
    SnapshotNotCurrent,
    SnapshotRollback,
    ClockUnavailable,
    Poisoned,
}

impl LmqPccpAuthorizationLedger {
    pub fn open(
        epoch_store: LmqPccpAuthorizationEpochStore,
        snapshot: SignedLmqPccpAuthorizationSnapshot,
    ) -> Result<Self, LmqPccpAuthorizationLedgerError> {
        verify_authorization_snapshot(&epoch_store.verifying_key, &snapshot)?;
        ensure_authorization_snapshot_current(&snapshot)?;
        let store_lock = epoch_store.lock_exclusive()?;
        epoch_store.admit_locked(&snapshot, &store_lock)?;
        drop(store_lock);
        Ok(Self {
            epoch_store,
            current: RwLock::new(snapshot.into_state()),
        })
    }

    pub fn apply_signed_snapshot(
        &self,
        snapshot: SignedLmqPccpAuthorizationSnapshot,
    ) -> Result<(), LmqPccpAuthorizationLedgerError> {
        verify_authorization_snapshot(&self.epoch_store.verifying_key, &snapshot)?;
        ensure_authorization_snapshot_current(&snapshot)?;
        let store_lock = self.epoch_store.lock_exclusive()?;
        let mut current = self
            .current
            .write()
            .map_err(|_| LmqPccpAuthorizationLedgerError::Poisoned)?;
        if snapshot.epoch <= current.epoch {
            return Err(LmqPccpAuthorizationLedgerError::SnapshotRollback);
        }
        self.epoch_store.admit_locked(&snapshot, &store_lock)?;
        *current = snapshot.into_state();
        Ok(())
    }
}

struct LedgerAuthorizationLease<'a> {
    _guard: RwLockReadGuard<'a, LmqPccpAuthorizationState>,
    _store_lock: LmqPccpAuthorizationStoreLock,
}

impl LmqPccpAuthorizationLedger {
    fn acquire<'a>(
        &'a self,
        request: &LmqPccpAuthorizationRequest<'_>,
    ) -> Option<LedgerAuthorizationLease<'a>> {
        let store_lock = self.epoch_store.lock_shared().ok()?;
        let durable_epoch = self.epoch_store.current_epoch_locked(&store_lock).ok()?;
        let guard = self.current.read().ok()?;
        let now = current_unix_seconds().ok()?;
        if durable_epoch != guard.epoch
            || now < guard.not_before_unix_seconds
            || now > guard.not_after_unix_seconds
            || !guard.entries.iter().any(|entry| entry.matches(request))
        {
            return None;
        }
        Some(LedgerAuthorizationLease {
            _guard: guard,
            _store_lock: store_lock,
        })
    }
}

impl SignedLmqPccpAuthorizationSnapshot {
    fn into_state(self) -> LmqPccpAuthorizationState {
        LmqPccpAuthorizationState {
            epoch: self.epoch,
            not_before_unix_seconds: self.not_before_unix_seconds,
            not_after_unix_seconds: self.not_after_unix_seconds,
            entries: self.entries,
        }
    }
}

fn normalize_authorization_entries(
    mut entries: Vec<LmqPccpAuthorizationEntry>,
) -> Result<Vec<LmqPccpAuthorizationEntry>, LmqPccpAuthorizationLedgerError> {
    if entries.len() > MAX_PCCP_AUTHORIZATION_ENTRIES {
        return Err(LmqPccpAuthorizationLedgerError::InvalidSnapshot);
    }
    entries.sort_by(|left, right| {
        left.model_artifact_content_id
            .as_bytes()
            .cmp(right.model_artifact_content_id.as_bytes())
    });
    if entries
        .windows(2)
        .any(|pair| pair[0].model_artifact_content_id == pair[1].model_artifact_content_id)
    {
        return Err(LmqPccpAuthorizationLedgerError::InvalidSnapshot);
    }
    Ok(entries)
}

fn validate_authorization_window(
    epoch: u64,
    not_before_unix_seconds: u64,
    not_after_unix_seconds: u64,
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    if epoch == 0
        || not_before_unix_seconds > not_after_unix_seconds
        || not_after_unix_seconds.saturating_sub(not_before_unix_seconds)
            > LMQ_PCCP_AUTHORIZATION_MAX_LIFETIME_SECONDS
    {
        return Err(LmqPccpAuthorizationLedgerError::InvalidSnapshot);
    }
    Ok(())
}

fn authorization_snapshot_digest(
    epoch: u64,
    not_before_unix_seconds: u64,
    not_after_unix_seconds: u64,
    entries: &[LmqPccpAuthorizationEntry],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PCCP_AUTHORIZATION_DOMAIN);
    hasher.update(&epoch.to_le_bytes());
    hasher.update(&not_before_unix_seconds.to_le_bytes());
    hasher.update(&not_after_unix_seconds.to_le_bytes());
    hasher.update(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        hasher.update(entry.model_artifact_content_id.as_bytes());
        hasher.update(entry.checkpoint_content_id.as_bytes());
        hasher.update(&entry.checkpoint_sha256);
        hasher.update(&(entry.pccp_change_id.len() as u64).to_le_bytes());
        hasher.update(entry.pccp_change_id.as_bytes());
        hasher.update(entry.pccp_evidence_id.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn verify_authorization_snapshot(
    verifying_key: &VerifyingKey,
    snapshot: &SignedLmqPccpAuthorizationSnapshot,
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let message = authorization_snapshot_digest(
        snapshot.epoch,
        snapshot.not_before_unix_seconds,
        snapshot.not_after_unix_seconds,
        &snapshot.entries,
    );
    verifying_key
        .verify_strict(&message, &Signature::from_bytes(&snapshot.signature))
        .map_err(|_| LmqPccpAuthorizationLedgerError::UntrustedSignature)
}

fn ensure_authorization_snapshot_current(
    snapshot: &SignedLmqPccpAuthorizationSnapshot,
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let now = current_unix_seconds()?;
    if now < snapshot.not_before_unix_seconds || now > snapshot.not_after_unix_seconds {
        return Err(LmqPccpAuthorizationLedgerError::SnapshotNotCurrent);
    }
    Ok(())
}

fn current_unix_seconds() -> Result<u64, LmqPccpAuthorizationLedgerError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| LmqPccpAuthorizationLedgerError::ClockUnavailable)
}

fn authorization_authority_id(verifying_key: &VerifyingKey) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.pccp-authorization-authority-v1");
    hasher.update(verifying_key.as_bytes());
    *hasher.finalize().as_bytes()
}

fn authorization_store_record_digest(snapshot: &SignedLmqPccpAuthorizationSnapshot) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.pccp-authorization-epoch-record-v1");
    hasher.update(&authorization_snapshot_digest(
        snapshot.epoch,
        snapshot.not_before_unix_seconds,
        snapshot.not_after_unix_seconds,
        &snapshot.entries,
    ));
    hasher.update(&snapshot.signature);
    *hasher.finalize().as_bytes()
}

fn ensure_epoch_store_authority(
    root: &Path,
    authority_id: [u8; 32],
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let path = root.join(AUTHORITY_MARKER_FILE);
    let expected = format!("{}\n", encode_hex(&authority_id));
    if let Some(actual) = try_read_epoch_store_record(&path)? {
        return if actual == authority_id {
            Ok(())
        } else {
            Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
        };
    }
    publish_immutable_record(root, &path, expected.as_bytes())?;
    let actual = read_epoch_store_record(&path)?;
    if actual == authority_id {
        Ok(())
    } else {
        Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
    }
}

fn recover_epoch_store(
    root: &Path,
) -> Result<Option<(u64, [u8; 32])>, LmqPccpAuthorizationLedgerError> {
    cleanup_epoch_store_staging(root)?;
    let audit = scan_epoch_audit_records(root)?;
    let head = read_epoch_store_head(root)?;
    let recovered = match (head, audit) {
        (None, None) => return Ok(None),
        (Some(head), None) => head,
        (None, Some(audit)) => {
            publish_epoch_store_head(root, audit.0, audit.1)?;
            audit
        }
        (Some(head), Some(audit)) if head.0 == audit.0 => {
            if head.1 != audit.1 {
                return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
            }
            head
        }
        (Some(head), Some(audit)) if head.0 > audit.0 => head,
        (Some(_), Some(audit)) => {
            publish_epoch_store_head(root, audit.0, audit.1)?;
            audit
        }
    };
    ensure_epoch_audit_record(root, recovered.0, recovered.1)?;
    Ok(Some(recovered))
}

fn scan_epoch_audit_records(
    root: &Path,
) -> Result<Option<(u64, [u8; 32])>, LmqPccpAuthorizationLedgerError> {
    let mut high_water = None;
    for entry in fs::read_dir(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)? {
        let entry = entry.map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let file_type = entry
            .file_type()
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        if !file_type.is_file() {
            return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
        }
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
        if matches!(
            name.as_str(),
            AUTHORITY_MARKER_FILE | AUTHORIZATION_CURRENT_FILE | AUTHORIZATION_LOCK_FILE
        ) {
            continue;
        }
        let epoch = name
            .strip_prefix(AUTHORIZATION_EPOCH_PREFIX)
            .filter(|digits| digits.len() == 20 && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .and_then(|digits| digits.parse::<u64>().ok())
            .filter(|epoch| *epoch != 0)
            .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
        let digest = read_epoch_store_record(&entry.path())?;
        if high_water
            .as_ref()
            .is_none_or(|(current_epoch, _)| epoch > *current_epoch)
        {
            high_water = Some((epoch, digest));
        }
    }
    Ok(high_water)
}

fn read_epoch_store_head(
    root: &Path,
) -> Result<Option<(u64, [u8; 32])>, LmqPccpAuthorizationLedgerError> {
    let path = root.join(AUTHORIZATION_CURRENT_FILE);
    let Some(bytes) = read_bounded_record_if_exists(&path, 86)? else {
        return Ok(None);
    };
    if bytes.len() != 86 || bytes[20] != b':' || bytes[85] != b'\n' {
        return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
    }
    let epoch = core::str::from_utf8(&bytes[..20])
        .ok()
        .filter(|digits| digits.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|digits| digits.parse::<u64>().ok())
        .filter(|epoch| *epoch != 0)
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
    let digest =
        decode_hex_32(&bytes[21..85]).ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
    Ok(Some((epoch, digest)))
}

fn publish_epoch_store_head(
    root: &Path,
    epoch: u64,
    digest: [u8; 32],
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let record = format!("{epoch:020}:{}\n", encode_hex(&digest));
    let staging = write_staged_record(root, record.as_bytes())?;
    let result = fs::rename(&staging, root.join(AUTHORIZATION_CURRENT_FILE))
        .and_then(|_| sync_directory(root));
    if result.is_err() {
        let _ = fs::remove_file(&staging);
        return Err(LmqPccpAuthorizationLedgerError::EpochStoreIo);
    }
    Ok(())
}

fn ensure_epoch_audit_record(
    root: &Path,
    epoch: u64,
    digest: [u8; 32],
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let path = root.join(format!("{AUTHORIZATION_EPOCH_PREFIX}{epoch:020}"));
    if let Some(existing) = try_read_epoch_store_record(&path)? {
        return if existing == digest {
            Ok(())
        } else {
            Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
        };
    }
    let record = format!("{}\n", encode_hex(&digest));
    publish_immutable_record(root, &path, record.as_bytes())?;
    if read_epoch_store_record(&path)? == digest {
        Ok(())
    } else {
        Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
    }
}

fn publish_immutable_record(
    root: &Path,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let staging = write_staged_record(root, bytes)?;
    let linked = fs::hard_link(&staging, destination);
    match linked {
        Ok(()) => {
            sync_directory(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
            fs::remove_file(&staging)
                .and_then(|_| sync_directory(root))
                .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(&staging).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
            Ok(())
        }
        Err(_) => {
            let _ = fs::remove_file(&staging);
            Err(LmqPccpAuthorizationLedgerError::EpochStoreIo)
        }
    }
}

fn write_staged_record(
    root: &Path,
    bytes: &[u8],
) -> Result<PathBuf, LmqPccpAuthorizationLedgerError> {
    for _ in 0..128 {
        let sequence = AUTHORIZATION_STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "{AUTHORIZATION_STAGING_PREFIX}{}-{sequence:020}",
            std::process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&path) {
            Ok(mut file) => {
                if file.write_all(bytes).and_then(|_| file.sync_all()).is_err() {
                    let _ = fs::remove_file(&path);
                    return Err(LmqPccpAuthorizationLedgerError::EpochStoreIo);
                }
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(LmqPccpAuthorizationLedgerError::EpochStoreIo),
        }
    }
    Err(LmqPccpAuthorizationLedgerError::EpochStoreIo)
}

fn cleanup_epoch_store_staging(root: &Path) -> Result<(), LmqPccpAuthorizationLedgerError> {
    let mut removed = false;
    for entry in fs::read_dir(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)? {
        let entry = entry.map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
        if !name.starts_with(AUTHORIZATION_STAGING_PREFIX) {
            continue;
        }
        if !entry
            .file_type()
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?
            .is_file()
        {
            return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
        }
        let metadata = entry
            .metadata()
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        let path_metadata = fs::symlink_metadata(entry.path())
            .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        validate_epoch_store_file(root, &metadata, &path_metadata)?;
        fs::remove_file(entry.path()).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        removed = true;
    }
    if removed {
        sync_directory(root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
    }
    Ok(())
}

fn sync_directory(root: &Path) -> std::io::Result<()> {
    File::open(root)?.sync_all()
}

fn read_epoch_store_record(path: &Path) -> Result<[u8; 32], LmqPccpAuthorizationLedgerError> {
    try_read_epoch_store_record(path)?.ok_or(LmqPccpAuthorizationLedgerError::EpochStoreIo)
}

fn try_read_epoch_store_record(
    path: &Path,
) -> Result<Option<[u8; 32]>, LmqPccpAuthorizationLedgerError> {
    let Some(bytes) = read_bounded_record_if_exists(path, 65)? else {
        return Ok(None);
    };
    if bytes.len() != 65 || bytes[64] != b'\n' {
        return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
    }
    decode_hex_32(&bytes[..64])
        .map(Some)
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)
}

fn read_bounded_record_if_exists(
    path: &Path,
    expected_len: usize,
) -> Result<Option<Vec<u8>>, LmqPccpAuthorizationLedgerError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(LmqPccpAuthorizationLedgerError::EpochStoreIo),
    };
    let root = path
        .parent()
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
    let metadata = file
        .metadata()
        .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
    let path_metadata =
        fs::symlink_metadata(path).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
    validate_epoch_store_file(root, &metadata, &path_metadata)?;
    let capacity = expected_len
        .checked_add(1)
        .ok_or(LmqPccpAuthorizationLedgerError::InvalidEpochStore)?;
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(&mut file)
        .take(capacity as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
    Ok(Some(bytes))
}

fn validate_epoch_store_file(
    _root: &Path,
    metadata: &fs::Metadata,
    path_metadata: &fs::Metadata,
) -> Result<(), LmqPccpAuthorizationLedgerError> {
    if !metadata.is_file() || !path_metadata.file_type().is_file() {
        return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
    }
    #[cfg(unix)]
    {
        let root_metadata =
            fs::metadata(_root).map_err(|_| LmqPccpAuthorizationLedgerError::EpochStoreIo)?;
        if metadata.dev() != path_metadata.dev()
            || metadata.ino() != path_metadata.ino()
            || metadata.uid() != root_metadata.uid()
            || path_metadata.uid() != root_metadata.uid()
            || metadata.permissions().mode() & 0o022 != 0
            || path_metadata.permissions().mode() & 0o022 != 0
        {
            return Err(LmqPccpAuthorizationLedgerError::InvalidEpochStore);
        }
    }
    Ok(())
}

fn encode_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn decode_hex_32(bytes: &[u8]) -> Option<[u8; 32]> {
    if bytes.len() != 64 {
        return None;
    }
    let mut output = [0_u8; 32];
    for (index, output_byte) in output.iter_mut().enumerate() {
        let high = decode_hex_nibble(bytes[index * 2])?;
        let low = decode_hex_nibble(bytes[index * 2 + 1])?;
        *output_byte = (high << 4) | low;
    }
    Some(output)
}

fn decode_hex_64(bytes: &[u8]) -> Option<[u8; 64]> {
    if bytes.len() != 128 {
        return None;
    }
    let mut output = [0_u8; 64];
    for (index, output_byte) in output.iter_mut().enumerate() {
        let high = decode_hex_nibble(bytes[index * 2])?;
        let low = decode_hex_nibble(bytes[index * 2 + 1])?;
        *output_byte = (high << 4) | low;
    }
    Some(output)
}

const fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Immutable production binding between one trained artifact, PCCP evidence,
/// calibrated fidelity, implementation identity, and resource ceilings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LmqNodeProfile {
    model_artifact_content_id: ContentId,
    checkpoint_content_id: ContentId,
    checkpoint_sha256: [u8; 32],
    pccp_change_id: String,
    pccp_evidence_id: ContentId,
    pearson_floor: Rational,
    backend_capabilities: NeuralBackendCapabilities,
    backend_deployment: LmqBackendDeployment,
    fidelity: CodecFidelity,
    implementation: CodecImplementation,
    bounds: shell::LmqResourceBounds,
    resources: ResourceEnvelope,
    input_max_bytes: u64,
    output_max_bytes: u64,
    maximum_loss: u16,
}

impl LmqNodeProfile {
    pub fn from_session(
        session: &LmqBackendSession<'_>,
        evidence: &VerifiedPccpEvidence,
        bounds: shell::LmqResourceBounds,
    ) -> Result<Self, LmqNodeProfileError> {
        let deployment = session.deployment();
        let artifact = session.artifact();
        let provenance = artifact.provenance();
        if provenance.pccp_status != PccpStatus::GatePass {
            return Err(LmqNodeProfileError::PccpGateNotPassed);
        }
        if provenance.checkpoint_content_id.to_bytes() == [0; 32]
            || provenance.checkpoint_sha256 == [0; 32]
            || provenance.pccp_change_id.is_empty()
            || provenance.pccp_change_id.len() > 128
            || provenance.pccp_change_id.trim() != provenance.pccp_change_id
            || provenance.pccp_change_id.chars().any(char::is_control)
        {
            return Err(LmqNodeProfileError::InvalidModelProvenance);
        }
        if evidence.evidence_id != provenance.pccp_evidence_id
            || evidence.checkpoint_sha256 != provenance.checkpoint_sha256
            || evidence.change_id != provenance.pccp_change_id
        {
            return Err(LmqNodeProfileError::PccpEvidenceMismatch);
        }
        let evidence_id = evidence.evidence_id;
        let pearson_floor = evidence.pearson_floor;
        if artifact.input_contract().upstream_claim_kind().as_str() != LMQ_MODEL_INPUT_PROOF {
            return Err(LmqNodeProfileError::ModelInputProofMismatch);
        }
        let capabilities = session.capabilities();
        if shell::validate_resource_profile(bounds, capabilities).is_err() {
            return Err(LmqNodeProfileError::InvalidResourceBounds);
        }

        let input_max_bytes = bounds
            .max_signal_bytes
            .checked_add(u64::from(bounds.bundle.max_frame_bytes))
            .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let output_max_bytes = u64::from(bounds.bundle.max_catalog_bytes)
            .checked_add(u64::from(bounds.bundle.max_frame_bytes) * 2)
            .and_then(|value| value.checked_add(BCS2_TWO_FRAME_OVERHEAD_BYTES))
            .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let token_bytes = u64::from(bounds.max_tokens)
            .checked_mul(12)
            .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let scratch_bytes = [
            bounds.max_signal_bytes,
            token_bytes,
            u64::from(bounds.max_schedule_bytes),
            u64::from(bounds.max_backend_meta_bytes),
            bounds.max_body_internal_working_bytes,
            u64::from(bounds.bundle.max_frame_bytes) * 2,
            output_max_bytes,
            deployment.max_working_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, value| total.checked_add(value))
        .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let peak_bytes = input_max_bytes;

        let fidelity = shell::transformed_fidelity(LMQ_FIDELITY_METRIC);
        let mut implementation = shell::implementation_identity(deployment.build_id.clone());
        let mut implementation_hasher = blake3::Hasher::new();
        hash_field(
            &mut implementation_hasher,
            b"org.quitetall.lamquant.nodes.lmq-emitter-implementation-v1",
        );
        hash_field(
            &mut implementation_hasher,
            implementation.implementation_id.as_bytes(),
        );
        hash_field(
            &mut implementation_hasher,
            deployment.implementation_id.as_bytes(),
        );
        implementation.implementation_id =
            ContentId::from_bytes(*implementation_hasher.finalize().as_bytes());

        Ok(Self {
            model_artifact_content_id: artifact.content_id(),
            checkpoint_content_id: provenance.checkpoint_content_id,
            checkpoint_sha256: provenance.checkpoint_sha256,
            pccp_change_id: provenance.pccp_change_id.clone(),
            pccp_evidence_id: evidence_id,
            pearson_floor,
            backend_capabilities: capabilities,
            backend_deployment: deployment.clone(),
            fidelity,
            implementation,
            bounds,
            resources: ResourceEnvelope {
                peak_bytes,
                scratch_bytes,
                threads: deployment.max_threads,
                device: deployment.device.clone(),
            },
            input_max_bytes,
            output_max_bytes,
            maximum_loss: u16::MAX,
        })
    }

    pub const fn model_artifact_content_id(&self) -> ContentId {
        self.model_artifact_content_id
    }

    pub const fn checkpoint_content_id(&self) -> ContentId {
        self.checkpoint_content_id
    }

    pub const fn checkpoint_sha256(&self) -> [u8; 32] {
        self.checkpoint_sha256
    }

    pub fn pccp_change_id(&self) -> &str {
        &self.pccp_change_id
    }

    pub const fn pccp_evidence_id(&self) -> ContentId {
        self.pccp_evidence_id
    }

    pub const fn pearson_floor(&self) -> Rational {
        self.pearson_floor
    }

    pub fn fidelity(&self) -> &CodecFidelity {
        &self.fidelity
    }

    pub fn implementation(&self) -> &CodecImplementation {
        &self.implementation
    }

    pub const fn bounds(&self) -> shell::LmqResourceBounds {
        self.bounds
    }

    pub const fn maximum_loss(&self) -> u16 {
        self.maximum_loss
    }

    pub(crate) fn verify_session(
        &self,
        session: &LmqBackendSession<'_>,
    ) -> Result<(), LmqNodeProfileError> {
        let artifact = session.artifact();
        if artifact.content_id() != self.model_artifact_content_id
            || artifact.provenance().pccp_evidence_id != self.pccp_evidence_id
            || artifact.provenance().pccp_status != PccpStatus::GatePass
            || session.capabilities() != self.backend_capabilities
            || !session.live_deployment_matches()
        {
            return Err(LmqNodeProfileError::BackendArtifactMismatch);
        }
        if session.deployment() != &self.backend_deployment {
            return Err(LmqNodeProfileError::BackendDeploymentMismatch);
        }
        Ok(())
    }

    fn authorization_request(&self) -> LmqPccpAuthorizationRequest<'_> {
        LmqPccpAuthorizationRequest {
            model_artifact_content_id: self.model_artifact_content_id,
            checkpoint_content_id: self.checkpoint_content_id,
            checkpoint_sha256: self.checkpoint_sha256,
            pccp_change_id: &self.pccp_change_id,
            pccp_evidence_id: self.pccp_evidence_id,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LmqRuntime<'a> {
    session: &'a LmqBackendSession<'a>,
    authorizer: &'a LmqPccpAuthorizationLedger,
    profile: &'a LmqNodeProfile,
}

impl<'a> LmqRuntime<'a> {
    pub(crate) fn new(
        session: &'a LmqBackendSession<'a>,
        authorizer: &'a LmqPccpAuthorizationLedger,
        profile: &'a LmqNodeProfile,
    ) -> Result<Self, LmqNodeProfileError> {
        profile.verify_session(session)?;
        Ok(Self {
            session,
            authorizer,
            profile,
        })
    }
}

struct AuthorizingBackend<'a> {
    backend: &'a dyn NeuralBackend,
    authorizer: &'a LmqPccpAuthorizationLedger,
    request: LmqPccpAuthorizationRequest<'a>,
    denied: &'a Cell<bool>,
}

impl NeuralBackend for AuthorizingBackend<'_> {
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
        let Some(_lease) = self.authorizer.acquire(&self.request) else {
            self.denied.set(true);
            return Err(BackendError::new(
                BackendErrorKind::Capability,
                "current PCCP authorization denied",
            ));
        };
        self.backend.encode(signal, sample_rate)
    }

    fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
        let Some(_lease) = self.authorizer.acquire(&self.request) else {
            self.denied.set(true);
            return Err(BackendError::new(
                BackendErrorKind::Capability,
                "current PCCP authorization denied",
            ));
        };
        self.backend.decode(tokens)
    }
}

pub fn lmq_node_config(profile: &LmqNodeProfile) -> BTreeMap<String, ConfigValue> {
    BTreeMap::from([
        (
            MODEL_ARTIFACT_CONFIG.into(),
            ConfigValue::Text(format!("{}", profile.model_artifact_content_id)),
        ),
        (
            PCCP_EVIDENCE_CONFIG.into(),
            ConfigValue::Text(format!("{}", profile.pccp_evidence_id)),
        ),
        (
            BACKEND_DEPLOYMENT_CONFIG.into(),
            ConfigValue::Text(format!("{}", profile.backend_deployment.implementation_id)),
        ),
    ])
}

pub fn lmq_descriptor(profile: &LmqNodeProfile) -> NodeDescriptor {
    let fidelity = FidelityContract {
        minimum_input: u16::MAX,
        maximum_loss: profile.maximum_loss,
    };
    NodeDescriptor {
        type_name: LMQ_NODE_TYPE.into(),
        version: 1,
        inputs: vec![dataset_port(profile)],
        outputs: vec![bundle_port(profile)],
        capabilities: vec![Capability(CAP_ABIR.into()), Capability(CAP_LMQ.into())],
        targets: profile.backend_deployment.allowed_targets.clone(),
        resources: profile.resources.clone(),
        determinism: Determinism::NumericallyEquivalent,
        config: config_schema(),
        state: StateContract {
            scope: StateScope::Stateless,
            max_bytes: 0,
            checkpoint: blut_graph_core::CheckpointContract {
                mode: CheckpointMode::Disabled,
                max_snapshot_bytes: 0,
                max_interval_invocations: 0,
            },
        },
        subgraph: None,
        proof: ProofContract {
            requires: vec![LMQ_MODEL_INPUT_PROOF.into()],
            provides: vec![LMQ_GENERATED_PROOF.into()],
            invalidates: vec![],
        },
        policy: current_pccp_policy(),
        fidelity,
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: vec![FAILURE_DOMAIN.into()],
        },
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

pub fn register_lmq_node(
    registry: &mut KernelRegistry,
    profile: &LmqNodeProfile,
) -> Result<(), CompileError> {
    registry.register_descriptor(lmq_descriptor(profile))?;
    for target in &profile.backend_deployment.allowed_targets {
        let (id, target) = match target {
            Target::Host => (HOST_KERNEL, Target::Host),
            Target::BlutDurable => (BLUT_DURABLE_KERNEL, Target::BlutDurable),
            Target::McuAot => continue,
        };
        registry.register_kernel(lmq_kernel(id, target, profile))?;
    }
    Ok(())
}

pub(crate) fn execute_lmq<'a>(
    node: &CompiledNode,
    inputs: &[Option<&LamQuantNodeValue<'a>>],
    runtime: Option<LmqRuntime<'a>>,
) -> Result<Vec<LamQuantNodeValue<'a>>, ExecutionError> {
    let runtime = runtime.ok_or_else(|| {
        kernel_failure(
            node,
            "runtime-unavailable",
            "LMQ backend and calibrated profile were not installed",
        )
    })?;
    if node.semantic_types.len() != 1
        || node.semantic_configs.len() != 1
        || node.semantic_types[0].type_name != LMQ_NODE_TYPE
        || node.semantic_types[0].version != 1
    {
        return Err(kernel_failure(
            node,
            "invalid-plan",
            "LMQ executor requires one version-1 LMQ semantic node",
        ));
    }
    let target = match node.kernel {
        HOST_KERNEL => Target::Host,
        BLUT_DURABLE_KERNEL => Target::BlutDurable,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-plan",
                "compiled LMQ node selected an unknown kernel",
            ));
        }
    };
    if node.implementation_id != implementation_id(target, runtime.profile) {
        return Err(kernel_failure(
            node,
            "model-binding-mismatch",
            "compiled LMQ implementation identity does not match runtime profile",
        ));
    }
    if !runtime
        .profile
        .backend_deployment
        .allowed_targets
        .contains(&target)
    {
        return Err(kernel_failure(
            node,
            "deployment-realm-mismatch",
            "compiled LMQ target is not authorized by backend deployment",
        ));
    }
    runtime
        .profile
        .verify_session(runtime.session)
        .map_err(|error| {
            kernel_failure(
                node,
                "model-binding-mismatch",
                &format!("LMQ backend no longer matches compiled profile: {error}"),
            )
        })?;
    let config = node.semantic_configs.first().ok_or_else(|| {
        kernel_failure(
            node,
            "invalid-plan",
            "LMQ semantic configuration is missing",
        )
    })?;
    if config != &lmq_node_config(runtime.profile) {
        return Err(kernel_failure(
            node,
            "model-binding-mismatch",
            "compiled LMQ model or PCCP evidence identity does not match runtime profile",
        ));
    }
    let input = match inputs {
        [Some(LamQuantNodeValue::AbirDataset(value))] => value,
        _ => {
            return Err(kernel_failure(
                node,
                "invalid-input",
                "LMQ node requires one owned ABIR dataset payload closure",
            ));
        }
    };
    if input.retained_bytes() > runtime.profile.input_max_bytes {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "LMQ ABIR payload closure exceeds compiled input extent",
        ));
    }
    let denied = Cell::new(false);
    let session_backend = runtime.session.backend();
    let backend = AuthorizingBackend {
        backend: &session_backend,
        authorizer: runtime.authorizer,
        request: runtime.profile.authorization_request(),
        denied: &denied,
    };
    let bundle = shell::encode_bundle_bounded(
        input.dataset(),
        input.payloads(),
        &backend,
        runtime.profile.fidelity.clone(),
        runtime.profile.implementation.clone(),
        runtime.profile.bounds,
    )
    .map_err(|error| {
        if denied.get() {
            kernel_failure(
                node,
                "authorization-denied",
                "current PCCP authorization was denied before neural inference",
            )
        } else {
            lmq_shell_failure(node, error)
        }
    })?;
    if bundle.len() as u64 > runtime.profile.output_max_bytes {
        return Err(kernel_failure(
            node,
            "resource-limit",
            "LMQ bundle exceeds compiled output extent",
        ));
    }
    Ok(vec![LamQuantNodeValue::Bcs2(bundle)])
}

fn config_schema() -> ConfigSchema {
    ConfigSchema {
        fields: vec![
            ConfigField {
                name: MODEL_ARTIFACT_CONFIG.into(),
                value_type: ConfigType::Text { max_bytes: 64 },
                required: true,
                default: None,
            },
            ConfigField {
                name: PCCP_EVIDENCE_CONFIG.into(),
                value_type: ConfigType::Text { max_bytes: 64 },
                required: true,
                default: None,
            },
            ConfigField {
                name: BACKEND_DEPLOYMENT_CONFIG.into(),
                value_type: ConfigType::Text { max_bytes: 64 },
                required: true,
                default: None,
            },
        ],
    }
}

fn dataset_port(profile: &LmqNodeProfile) -> PortDescriptor {
    PortDescriptor {
        name: "dataset".into(),
        semantic_type: INPUT_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::Opaque],
        max_bytes: profile.input_max_bytes,
        abir: AbirSemanticType {
            root: AbirRootType::Dataset,
            view: AbirViewType::Root,
        },
        proof: ProofContract {
            requires: vec![LMQ_MODEL_INPUT_PROOF.into()],
            provides: vec![],
            invalidates: vec![],
        },
        policy: current_pccp_policy(),
        fidelity: FidelityContract {
            minimum_input: u16::MAX,
            maximum_loss: 0,
        },
        extent: opaque_extent(),
        lease: read_lease(false),
    }
}

fn bundle_port(profile: &LmqNodeProfile) -> PortDescriptor {
    PortDescriptor {
        name: "bundle".into(),
        semantic_type: OUTPUT_SEMANTIC_TYPE.into(),
        optional: false,
        layouts: vec![Layout::Packed],
        max_bytes: profile.output_max_bytes,
        abir: AbirSemanticType {
            root: AbirRootType::Dataset,
            view: AbirViewType::Root,
        },
        proof: ProofContract {
            requires: vec![],
            provides: vec![LMQ_GENERATED_PROOF.into()],
            invalidates: vec![],
        },
        policy: current_pccp_policy(),
        fidelity: FidelityContract {
            minimum_input: u16::MAX.saturating_sub(profile.maximum_loss),
            maximum_loss: profile.maximum_loss,
        },
        extent: byte_extent(profile.output_max_bytes),
        lease: read_lease(false),
    }
}

fn lmq_kernel(id: KernelId, target: Target, profile: &LmqNodeProfile) -> KernelDescriptor {
    KernelDescriptor {
        id,
        implements: vec![NodeTypeRef {
            type_name: LMQ_NODE_TYPE.into(),
            version: 1,
        }],
        implementation_id: implementation_id(target, profile),
        conversion: None,
        target,
        input_layouts: vec![Layout::Opaque],
        output_layouts: vec![Layout::Packed],
        resources: profile.resources.clone(),
        determinism: Determinism::NumericallyEquivalent,
        lowering: "fused:lmq:validated-abir+trained-backend+bcs2:v1".into(),
    }
}

fn implementation_id(target: Target, profile: &LmqNodeProfile) -> ImplementationId {
    let mut hasher = blake3::Hasher::new();
    hash_field(
        &mut hasher,
        b"org.quitetall.lamquant.nodes.lmq-implementation-v1",
    );
    hash_field(&mut hasher, SOURCE_ID.as_bytes());
    hash_field(&mut hasher, FEATURE_SET.as_bytes());
    hash_field(&mut hasher, LMQ_NODE_TYPE.as_bytes());
    hash_field(
        &mut hasher,
        &[match target {
            Target::McuAot => 0,
            Target::Host => 1,
            Target::BlutDurable => 2,
        }],
    );
    hash_field(&mut hasher, profile.model_artifact_content_id.as_bytes());
    hash_field(&mut hasher, profile.pccp_evidence_id.as_bytes());
    hash_field(&mut hasher, profile.checkpoint_content_id.as_bytes());
    hash_field(&mut hasher, &profile.checkpoint_sha256);
    hash_field(&mut hasher, profile.pccp_change_id.as_bytes());
    hash_field(&mut hasher, &profile.maximum_loss.to_le_bytes());
    let (floor_numerator, floor_denominator) = profile.pearson_floor.parts();
    hash_field(&mut hasher, &floor_numerator.to_le_bytes());
    hash_field(&mut hasher, &floor_denominator.to_le_bytes());
    hash_field(
        &mut hasher,
        profile.backend_deployment.implementation_id.as_bytes(),
    );
    hash_field(
        &mut hasher,
        profile.backend_deployment.executable_content_id.as_bytes(),
    );
    hash_field(
        &mut hasher,
        &profile.backend_deployment.max_working_bytes.to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        &profile.backend_deployment.max_threads.to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        profile
            .backend_deployment
            .device
            .as_deref()
            .unwrap_or_default()
            .as_bytes(),
    );
    for target in &profile.backend_deployment.allowed_targets {
        hash_field(
            &mut hasher,
            &[match target {
                Target::McuAot => 0,
                Target::Host => 1,
                Target::BlutDurable => 2,
            }],
        );
    }
    hash_backend_capabilities(&mut hasher, profile.backend_capabilities);
    hash_field(
        &mut hasher,
        profile.implementation.implementation_id.as_bytes(),
    );
    hash_field(&mut hasher, profile.implementation.build_id.as_bytes());
    hash_bounds(&mut hasher, profile.bounds);
    ImplementationId(*hasher.finalize().as_bytes())
}

fn hash_backend_capabilities(hasher: &mut blake3::Hasher, capabilities: NeuralBackendCapabilities) {
    hash_field(
        hasher,
        &[match capabilities.target {
            BackendTarget::Reference => 0,
            BackendTarget::HostSubprocess => 1,
            BackendTarget::HostNative => 2,
            BackendTarget::McuNative => 3,
            _ => u8::MAX,
        }],
    );
    hash_field(
        hasher,
        &[match capabilities.signal_domain {
            lamquant_lmq::backend::SignalDomain::DigitalInteger => 0,
            lamquant_lmq::backend::SignalDomain::PhysicalMicrovoltQ16 => 1,
            _ => u8::MAX,
        }],
    );
    hash_field(hasher, &[u8::from(capabilities.operational)]);
    for value in [
        u64::from(capabilities.minimum_channels),
        u64::from(capabilities.maximum_channels),
        u64::from(capabilities.minimum_samples),
        u64::from(capabilities.maximum_samples),
        u64::from(capabilities.maximum_tokens),
        u64::from(capabilities.maximum_schedule_bytes),
        u64::from(capabilities.maximum_backend_metadata_bytes),
        u64::from(capabilities.minimum_alphabet),
        u64::from(capabilities.maximum_alphabet),
    ] {
        hash_field(hasher, &value.to_le_bytes());
    }
    for rational in [
        capabilities.minimum_sample_rate,
        capabilities.maximum_sample_rate,
    ] {
        let (numerator, denominator) = rational.parts();
        hash_field(hasher, &numerator.to_le_bytes());
        hash_field(hasher, &denominator.to_le_bytes());
    }
}

fn hash_bounds(hasher: &mut blake3::Hasher, bounds: shell::LmqResourceBounds) {
    for value in [
        u64::from(bounds.bundle.max_catalog_bytes),
        u64::from(bounds.bundle.max_index_entries),
        u64::from(bounds.bundle.max_frame_bytes),
        u64::from(bounds.bundle.max_generations),
        bounds.max_signal_bytes,
        u64::from(bounds.max_signal_channels),
        u64::from(bounds.max_tokens),
        u64::from(bounds.max_schedule_bytes),
        u64::from(bounds.max_backend_meta_bytes),
        u64::from(bounds.max_alphabet),
        u64::from(bounds.max_model_total),
        u64::from(bounds.max_model_basis_channels),
        u64::from(bounds.max_model_basis_terms),
        u64::from(bounds.max_model_derivations),
        u64::from(bounds.max_model_claims),
        u64::from(bounds.max_model_derivation_output_edges),
        bounds.max_body_internal_working_bytes,
    ] {
        hash_field(hasher, &value.to_le_bytes());
    }
}

fn hash_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn current_pccp_policy() -> PolicyContract {
    PolicyContract {
        requires: vec![LMQ_CURRENT_PCCP_POLICY.into()],
        adds: vec![],
    }
}

fn opaque_extent() -> ExtentContract {
    ExtentContract {
        rank: 0,
        maximum_shape: vec![],
        max_elements: 1,
        ragged: false,
        sparse: false,
    }
}

fn byte_extent(maximum: u64) -> ExtentContract {
    ExtentContract {
        rank: 1,
        maximum_shape: vec![maximum],
        max_elements: maximum,
        ragged: false,
        sparse: false,
    }
}

fn read_lease(zero_copy_permitted: bool) -> LeaseContract {
    LeaseContract {
        access: LeaseAccess::ReadOnly,
        lifetime: LeaseLifetime::Invocation,
        zero_copy_permitted,
        contiguous_required: false,
    }
}

fn kernel_failure(node: &CompiledNode, code: &str, message: &str) -> ExecutionError {
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: code.into(),
            message: message.into(),
            retryable: false,
            evidence: Vec::new(),
        },
    }
}

fn lmq_shell_failure(node: &CompiledNode, error: shell::LmqError) -> ExecutionError {
    let (code, retryable, evidence) = match &error {
        shell::LmqError::ResourceLimit {
            resource,
            actual,
            limit,
        } => (
            "resource-limit",
            false,
            vec![FailureEvidence {
                semantic_type: "org.quitetall.lamquant.failure-evidence.resource-limit-v1+json"
                    .into(),
                payload: format!(
                    "{{\"actual\":{actual},\"limit\":{limit},\"resource\":\"{resource:?}\"}}"
                )
                .into_bytes(),
            }],
        ),
        shell::LmqError::InvalidResourceProfile(_) => {
            ("resource-profile-invalid", false, Vec::new())
        }
        shell::LmqError::Backend(error) => match error.kind() {
            BackendErrorKind::Cancelled => ("backend-cancelled", false, Vec::new()),
            BackendErrorKind::Timeout => ("backend-timeout", true, Vec::new()),
            BackendErrorKind::ResourceLimit => ("backend-resource-limit", false, Vec::new()),
            BackendErrorKind::Deployment => ("backend-deployment", false, Vec::new()),
            BackendErrorKind::Process => ("backend-process", true, Vec::new()),
            BackendErrorKind::Protocol => ("backend-protocol", false, Vec::new()),
            BackendErrorKind::Model => ("backend-model", false, Vec::new()),
            BackendErrorKind::Capability => ("backend-contract", false, Vec::new()),
            BackendErrorKind::Deferred => ("backend-deferred", false, Vec::new()),
            BackendErrorKind::Internal => ("backend-internal", false, Vec::new()),
            _ => ("backend-unknown", false, Vec::new()),
        },
        shell::LmqError::BackendCapability(_) | shell::LmqError::SignalShapeMismatch => {
            ("backend-contract", false, Vec::new())
        }
        shell::LmqError::ModelInputContract(_)
        | shell::LmqError::UnsupportedSemantics(_)
        | shell::LmqError::SemanticValidation => ("semantic-contract", false, Vec::new()),
        shell::LmqError::PayloadAccess(_) => ("payload-access", false, Vec::new()),
        shell::LmqError::PayloadIdentityMismatch => ("payload-integrity", false, Vec::new()),
        shell::LmqError::Body(_)
        | shell::LmqError::Bundle(_)
        | shell::LmqError::CatalogContract
        | shell::LmqError::Header
        | shell::LmqError::BadTokens
        | shell::LmqError::SemanticEncoding => ("codec-contract", false, Vec::new()),
        _ => ("codec-failure", false, Vec::new()),
    };
    ExecutionError::KernelFailed {
        kernel: node.kernel,
        failure: StructuredFailure {
            domain: FAILURE_DOMAIN.into(),
            code: code.into(),
            message: format!("LMQ bundle encode failed: {error}"),
            retryable,
            evidence,
        },
    }
}
