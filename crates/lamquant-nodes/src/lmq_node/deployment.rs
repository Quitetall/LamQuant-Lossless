//! Loader-owned backend deployment and immutable session bindings.

#[cfg(feature = "lmq-python")]
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
#[cfg(not(any(test, feature = "lmq-python")))]
use core::marker::PhantomData;

use blut_graph_core::Target;
use lamquant_lmq::backend::{
    BackendError, BackendModel, NeuralBackend, NeuralBackendCapabilities, NeuralSignal,
    NeuralTokens, TrainedModelArtifact,
};
#[cfg(feature = "lmq-python")]
use lamquant_lmq::py_backend::ProductionPyBackend;
use semantic_abir::{ContentId, Rational};

#[cfg(test)]
use semantic_abir::{payload_content_id, ElementType};

#[cfg(any(test, feature = "lmq-python"))]
use super::hash_field;
#[cfg(any(test, feature = "lmq-python"))]
use super::LmqNodeProfileError;

/// Opaque loader-verified deployment identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LmqBackendDeployment {
    pub(super) executable_content_id: ContentId,
    pub(super) implementation_id: ContentId,
    pub(super) build_id: String,
    pub(super) max_working_bytes: u64,
    pub(super) max_threads: u16,
    pub(super) device: Option<String>,
    pub(super) allowed_targets: Vec<Target>,
}

enum SessionBackend<'a> {
    #[cfg(feature = "lmq-python")]
    ProductionPy(&'a ProductionPyBackend),
    #[cfg(test)]
    Test {
        backend: &'a dyn LmqAttestedBackend,
        executable_content_id: ContentId,
    },
    #[cfg(not(any(test, feature = "lmq-python")))]
    #[allow(dead_code)]
    Unavailable(PhantomData<&'a ()>),
}

impl SessionBackend<'_> {
    fn backend(&self) -> &dyn NeuralBackend {
        match self {
            #[cfg(feature = "lmq-python")]
            Self::ProductionPy(backend) => *backend,
            #[cfg(test)]
            Self::Test { backend, .. } => *backend,
            #[cfg(not(any(test, feature = "lmq-python")))]
            Self::Unavailable(_) => unreachable!("no production LMQ backend feature enabled"),
        }
    }

    fn live_deployment_matches(&self, _deployment: &LmqBackendDeployment) -> bool {
        match self {
            #[cfg(feature = "lmq-python")]
            Self::ProductionPy(backend) => {
                backend.deployment_attestation().execution_id() == _deployment.executable_content_id
            }
            #[cfg(test)]
            Self::Test {
                backend,
                executable_content_id,
            } => {
                backend.executable_content_id() == *executable_content_id
                    && *executable_content_id == _deployment.executable_content_id
            }
            #[cfg(not(any(test, feature = "lmq-python")))]
            Self::Unavailable(_) => false,
        }
    }
}

/// Immutable backend binding. Public construction exists only for loader-sealed
/// production backends; raw `NeuralBackend` values cannot mint a Node profile.
pub struct LmqBackendSession<'a> {
    backend: SessionBackend<'a>,
    artifact: TrainedModelArtifact,
    capabilities: NeuralBackendCapabilities,
    deployment: LmqBackendDeployment,
}

impl<'a> LmqBackendSession<'a> {
    #[cfg(feature = "lmq-python")]
    pub fn from_production_py_backend(
        backend: &'a ProductionPyBackend,
    ) -> Result<Self, LmqNodeProfileError> {
        backend
            .revalidate_deployment()
            .map_err(|_| LmqNodeProfileError::InvalidBackendDeployment)?;
        let artifact = backend
            .model()
            .trained_artifact()
            .cloned()
            .ok_or(LmqNodeProfileError::TrainedArtifactRequired)?;
        let capabilities = backend.capabilities();
        let attestation = backend.deployment_attestation();
        let process_limits = attestation.process_limits();
        let io_limits = attestation.io_limits();
        if attestation.closure_id().to_bytes() == [0; 32]
            || attestation.execution_id().to_bytes() == [0; 32]
            || attestation.checkpoint_content_id() != artifact.provenance().checkpoint_content_id
            || attestation.checkpoint_sha256() != artifact.provenance().checkpoint_sha256
            || !matches!(
                capabilities.target,
                lamquant_lmq::backend::BackendTarget::HostSubprocess
            )
        {
            return Err(LmqNodeProfileError::InvalidBackendDeployment);
        }
        let max_working_bytes = process_limits
            .memory_bytes
            .get()
            .checked_add(io_limits.maximum_request_bytes)
            .and_then(|bytes| bytes.checked_add(io_limits.maximum_stdout_bytes))
            .and_then(|bytes| bytes.checked_add(io_limits.maximum_stderr_bytes))
            .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let max_threads = process_limits
            .maximum_tasks
            .get()
            .checked_add(3)
            .ok_or(LmqNodeProfileError::ResourceExtentOverflow)?;
        let build_id = format!("lmq-production-py:{}", attestation.execution_id());
        let deployment = deployment_identity(
            attestation.execution_id(),
            &build_id,
            max_working_bytes,
            max_threads,
            Some("cpu"),
            &[Target::Host],
        );
        Ok(Self {
            backend: SessionBackend::ProductionPy(backend),
            artifact,
            capabilities,
            deployment,
        })
    }

    pub(super) const fn artifact(&self) -> &TrainedModelArtifact {
        &self.artifact
    }

    pub(super) const fn capabilities(&self) -> NeuralBackendCapabilities {
        self.capabilities
    }

    pub(super) fn backend(&self) -> SessionNeuralBackend<'_, 'a> {
        SessionNeuralBackend { session: self }
    }

    pub(super) fn live_deployment_matches(&self) -> bool {
        self.backend.live_deployment_matches(&self.deployment)
    }

    pub(super) const fn deployment(&self) -> &LmqBackendDeployment {
        &self.deployment
    }
}

/// Immutable model/capability projection captured by production loading.
#[derive(Clone, Copy)]
pub(super) struct SessionNeuralBackend<'session, 'backend> {
    session: &'session LmqBackendSession<'backend>,
}

impl NeuralBackend for SessionNeuralBackend<'_, '_> {
    fn capabilities(&self) -> NeuralBackendCapabilities {
        self.session.capabilities
    }

    fn model(&self) -> BackendModel<'_> {
        BackendModel::trained(&self.session.artifact)
    }

    fn encode(
        &self,
        signal: &NeuralSignal,
        sample_rate: Rational,
    ) -> Result<NeuralTokens, BackendError> {
        self.session.backend.backend().encode(signal, sample_rate)
    }

    fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
        self.session.backend.backend().decode(tokens)
    }
}

#[cfg(any(test, feature = "lmq-python"))]
fn deployment_identity(
    executable_content_id: ContentId,
    build_id: &str,
    max_working_bytes: u64,
    max_threads: u16,
    device: Option<&str>,
    allowed_targets: &[Target],
) -> LmqBackendDeployment {
    let mut hasher = blake3::Hasher::new();
    hash_field(
        &mut hasher,
        b"org.quitetall.lamquant.nodes.lmq-backend-deployment-v2",
    );
    hash_field(&mut hasher, executable_content_id.as_bytes());
    hash_field(&mut hasher, build_id.as_bytes());
    hash_field(&mut hasher, &max_working_bytes.to_le_bytes());
    hash_field(&mut hasher, &max_threads.to_le_bytes());
    hash_field(&mut hasher, device.unwrap_or_default().as_bytes());
    for target in allowed_targets {
        hash_field(
            &mut hasher,
            &[match target {
                Target::McuAot => 0,
                Target::Host => 1,
                Target::BlutDurable => 2,
            }],
        );
    }
    LmqBackendDeployment {
        executable_content_id,
        implementation_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        build_id: build_id.into(),
        max_working_bytes,
        max_threads,
        device: device.map(String::from),
        allowed_targets: allowed_targets.to_vec(),
    }
}

#[cfg(test)]
const MAX_BUILD_ID_BYTES: usize = 256;
#[cfg(test)]
const MAX_DEVICE_ID_BYTES: usize = 256;

/// Test-only manifest. Production callers cannot assert deployment properties.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LmqBackendDeploymentManifest {
    executable_content_id: ContentId,
    build_id: String,
    max_working_bytes: u64,
    max_threads: u16,
    device: Option<String>,
    allowed_targets: Vec<Target>,
}

#[cfg(test)]
impl LmqBackendDeploymentManifest {
    pub(crate) fn new(
        executable_content_id: ContentId,
        build_id: impl Into<String>,
        max_working_bytes: u64,
        max_threads: u16,
        device: Option<String>,
        allowed_targets: Vec<Target>,
    ) -> Self {
        Self {
            executable_content_id,
            build_id: build_id.into(),
            max_working_bytes,
            max_threads,
            device,
            allowed_targets,
        }
    }
}

#[cfg(test)]
pub(crate) trait LmqAttestedBackend: NeuralBackend {
    fn executable_content_id(&self) -> ContentId;
}

#[cfg(test)]
impl<'a> LmqBackendSession<'a> {
    pub(crate) fn verify_test(
        backend: &'a dyn LmqAttestedBackend,
        manifest: LmqBackendDeploymentManifest,
        executable: &[u8],
    ) -> Result<Self, LmqNodeProfileError> {
        let artifact = backend
            .model()
            .trained_artifact()
            .cloned()
            .ok_or(LmqNodeProfileError::TrainedArtifactRequired)?;
        let capabilities = backend.capabilities();
        let actual_content_id = payload_content_id(ElementType::Bytes, executable);
        if executable.is_empty()
            || actual_content_id != manifest.executable_content_id
            || actual_content_id != backend.executable_content_id()
            || manifest.executable_content_id.to_bytes() == [0; 32]
            || !valid_bounded_text(&manifest.build_id, MAX_BUILD_ID_BYTES)
            || manifest.max_working_bytes == 0
            || manifest.max_threads == 0
            || manifest
                .device
                .as_deref()
                .is_some_and(|device| !valid_bounded_text(device, MAX_DEVICE_ID_BYTES))
            || manifest.allowed_targets.is_empty()
            || manifest.allowed_targets.contains(&Target::McuAot)
        {
            return Err(LmqNodeProfileError::InvalidBackendDeployment);
        }
        let mut allowed_targets = manifest.allowed_targets;
        allowed_targets.sort_unstable_by_key(|target| match target {
            Target::McuAot => 0_u8,
            Target::Host => 1,
            Target::BlutDurable => 2,
        });
        if allowed_targets
            .windows(2)
            .any(|targets| targets[0] == targets[1])
        {
            return Err(LmqNodeProfileError::InvalidBackendDeployment);
        }
        let deployment = deployment_identity(
            actual_content_id,
            &manifest.build_id,
            manifest.max_working_bytes,
            manifest.max_threads,
            manifest.device.as_deref(),
            &allowed_targets,
        );
        Ok(Self {
            backend: SessionBackend::Test {
                backend,
                executable_content_id: actual_content_id,
            },
            artifact,
            capabilities,
            deployment,
        })
    }
}

#[cfg(test)]
fn valid_bounded_text(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_bytes
        && value.trim() == value
        && !value.chars().any(char::is_control)
}
