//! Current-ABIR LMQ shell.
//!
//! The neural backend owns inference only. This module owns the deterministic
//! token packet and seals it with canonical ABIR semantics in the registered
//! `bcs.lmq.progressive.v1` BCS2 profile.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use semantic_abir::{
    canonical_debug_json, parse_canonical_dataset, verify_payload_content, AbirDataset, Atom,
    ByteOrder, ChannelTag, ConceptId, ContentId, ElementType, InMemoryPayloadAccess, Layout,
    ObjectId, OpenedDataset, PayloadAccess, PayloadDescriptor, PayloadLease, Presence, Rational,
    ReferenceKind, SemanticRef, TimeAxis,
};
use semantic_abir_bcs::{
    encode_codec_bundle, CodecBundleError, CodecBundleInput, CodecBundleView, CodecFidelity,
    CodecFidelityKind, CodecImplementation, CodecParameter, CodecParameterValue, CodecProfile,
    ResourceBounds,
};

use crate::backend::{
    hash_field, BackendCapabilityError, BackendError, BackendModel, ModelInputContractError,
    NeuralBackend, NeuralSignal, NeuralTokens, SignalDomain, TrainedModelArtifact,
};
use crate::body::{decode_body_bounded, encode_body_bounded, BodyBounds, BodyError};
use crate::calibration::{AffineDomainTransform, CalibrationDomainError};
use crate::reconstruction::{
    build_reconstructed_dataset, codec_fidelity_statement, ReconstructionContext,
};

pub const LMQ_KERNEL_ID: &str = "org.quitetall.lamquant.lmq.fsq-rans-v1";
pub const LMQ_FIDELITY_CONTRACT: &str =
    "org.quitetall.lamquant.bcs2.lmq.explicit-nonexact-reconstruction-v1";
pub const RANS_MODEL_TOTAL: u64 = 4096;
/// Smallest admitted BCS2 catalog budget for the required LMQ identity,
/// fidelity, model-provenance, parameter, semantics, and packet bindings.
pub const MIN_LMQ_CATALOG_BYTES: u32 = 512;
/// Smallest possible LMQP1 frame: 15-byte packet header, 15-byte body prefix,
/// one four-byte model count, and the four-byte terminal rANS state.
pub const MIN_LMQ_PACKET_FRAME_BYTES: u32 = 38;
const PACKET_MAGIC: &[u8; 4] = b"LMQP";
const PACKET_VERSION: u8 = 1;
const PACKET_HEADER_LEN: usize = 15;
const LMQ_WIRE_ABIR_REVISION: &str = "c101513167ad8d7cdefa6387b20c644fdaf66432";
const LINKED_ABIR_REVISION: &str = "c82228ea1a28ad48488a62c2073344a8ff40265f";
const MODEL_CHANNEL_BASIS_DOMAIN: &[u8] = b"lamquant.lmq.model-channel-basis.v1";

/// LMQ-specific resource ceilings layered over BCS2 frame/catalog limits.
///
/// [`from_bundle`](Self::from_bundle) intentionally applies the BCS2 frame
/// ceiling to decoded I64 signal materialization too. This hardens the legacy
/// wrapper against inputs whose decoded working set dwarfs their compressed
/// output. Callers needing independent ceilings must use the bounded APIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LmqResourceBounds {
    pub bundle: ResourceBounds,
    pub max_signal_bytes: u64,
    /// Maximum signal atoms/channels resolved for one LMQ stream.
    pub max_signal_channels: u16,
    /// Also bounds the shell's temporary eight-byte-per-token I64 staging.
    pub max_tokens: u32,
    pub max_schedule_bytes: u32,
    pub max_backend_meta_bytes: u32,
    pub max_alphabet: u16,
    pub max_model_total: u32,
    /// Maximum channel records indexed while validating a portable model basis.
    pub max_model_basis_channels: u32,
    /// Maximum aggregate weighted terms hashed for one model basis.
    pub max_model_basis_terms: u32,
    /// Maximum derivation records inspected for one trained-model contract.
    pub max_model_derivations: u32,
    /// Maximum typed claim records indexed for one trained-model contract.
    pub max_model_claims: u32,
    /// Maximum aggregate derivation output edges inspected.
    pub max_model_derivation_output_edges: u32,
    /// Bounds allocations internal to the body codec. Shell token staging is
    /// governed separately by `max_tokens`.
    pub max_body_internal_working_bytes: u64,
}

impl LmqResourceBounds {
    pub const fn from_bundle(bundle: ResourceBounds) -> Self {
        Self {
            bundle,
            max_signal_bytes: bundle.max_frame_bytes as u64,
            max_signal_channels: u16::MAX,
            max_tokens: lamquant_lml_mcu::rans::MAX_RANS_SYMBOLS as u32,
            max_schedule_bytes: bundle.max_frame_bytes,
            max_backend_meta_bytes: bundle.max_frame_bytes,
            max_alphabet: RANS_MODEL_TOTAL as u16,
            max_model_total: RANS_MODEL_TOTAL as u32,
            max_model_basis_channels: u16::MAX as u32,
            max_model_basis_terms: lamquant_lml_mcu::rans::MAX_RANS_SYMBOLS as u32,
            max_model_derivations: u16::MAX as u32,
            max_model_claims: u16::MAX as u32,
            max_model_derivation_output_edges: 1 << 20,
            max_body_internal_working_bytes: bundle.max_frame_bytes as u64 + 17 * 1024 * 1024,
        }
    }

    fn body(self, max_body_bytes: u32) -> BodyBounds {
        BodyBounds {
            max_symbols: self.max_tokens,
            max_schedule_bytes: self.max_schedule_bytes,
            max_rans_bytes: self.bundle.max_frame_bytes,
            max_alphabet: self.max_alphabet,
            max_model_total: self.max_model_total,
            max_working_bytes: self.max_body_internal_working_bytes,
            max_body_bytes,
        }
    }
}

impl Default for LmqResourceBounds {
    fn default() -> Self {
        Self::from_bundle(ResourceBounds::default())
    }
}

/// Explicit compatibility policy for catalogs written before model contracts.
///
/// Current APIs reject legacy catalogs by default. Compatibility bridges must
/// opt in and still pass complete dataset/model-contract validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyModelContractPolicy {
    Reject,
    AllowPreContractCatalog,
}

#[derive(Debug)]
pub struct OpenedLmqBundle<'a> {
    bundle: CodecBundleView<'a>,
    source_dataset: AbirDataset,
    reconstructed: OpenedDataset<InMemoryPayloadAccess>,
}

impl<'a> OpenedLmqBundle<'a> {
    /// Canonical semantics sealed at encode time. Payload identities here refer
    /// to the original signal and are intentionally not resolved by this
    /// decoded object.
    pub const fn source_dataset(&self) -> &AbirDataset {
        &self.source_dataset
    }

    /// ABIR semantics and payload access for the actual lossy reconstruction.
    /// Its payload ContentIds are derived from decoded bytes, never copied from
    /// the source dataset.
    pub const fn reconstructed(&self) -> &OpenedDataset<InMemoryPayloadAccess> {
        &self.reconstructed
    }

    pub const fn bundle(&self) -> &CodecBundleView<'a> {
        &self.bundle
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum LmqError {
    Backend(BackendError),
    BackendCapability(BackendCapabilityError),
    ModelInputContract(ModelInputContractError),
    Body(BodyError),
    Bundle(CodecBundleError),
    CatalogContract,
    Header,
    BadTokens,
    PayloadAccess(semantic_abir::PayloadAccessError),
    PayloadIdentityMismatch,
    SemanticEncoding,
    SemanticValidation,
    SignalShapeMismatch,
    InvalidResourceProfile(&'static str),
    ResourceLimit {
        resource: LmqResource,
        actual: u64,
        limit: u64,
    },
    UnsupportedSemantics(&'static str),
}

/// Runtime resource governed by [`LmqResourceBounds`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LmqResource {
    SignalBytes,
    SignalChannels,
    TokenCount,
    Alphabet,
    ModelTotal,
    ModelBasisChannels,
    ModelBasisTerms,
    ModelDerivations,
    ModelClaims,
    ModelDerivationOutputEdges,
    ScheduleBytes,
    BackendMetadataBytes,
    SemanticFrameBytes,
    PacketBytes,
}

impl From<BodyError> for LmqError {
    fn from(error: BodyError) -> Self {
        Self::Body(error)
    }
}

impl From<CalibrationDomainError> for LmqError {
    fn from(error: CalibrationDomainError) -> Self {
        match error {
            CalibrationDomainError::Range => {
                Self::UnsupportedSemantics("calibration exceeds bounded model-domain arithmetic")
            }
            CalibrationDomainError::UnsupportedUnit => Self::UnsupportedSemantics(
                "physical model domain requires a recognized voltage unit",
            ),
        }
    }
}

impl fmt::Display for LmqError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(formatter, "LMQ backend failed: {error}"),
            Self::BackendCapability(error) => {
                write!(formatter, "LMQ backend capability mismatch: {error:?}")
            }
            Self::ModelInputContract(error) => {
                write!(formatter, "LMQ model input contract mismatch: {error:?}")
            }
            Self::Body(error) => write!(formatter, "LMQ token body failed: {error:?}"),
            Self::Bundle(error) => error.fmt(formatter),
            Self::PayloadAccess(error) => error.fmt(formatter),
            Self::ResourceLimit {
                resource,
                actual,
                limit,
            } => {
                write!(
                    formatter,
                    "LMQ resource limit exceeded: {resource:?} ({actual} > {limit})"
                )
            }
            Self::UnsupportedSemantics(reason) => {
                write!(formatter, "unsupported LMQ ABIR semantics: {reason}")
            }
            Self::InvalidResourceProfile(reason) => {
                write!(formatter, "invalid LMQ resource profile: {reason}")
            }
            other => write!(formatter, "LMQ BCS2 bundle error: {other:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmqError {}

pub fn implementation_identity(build_id: impl Into<String>) -> CodecImplementation {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmq.implementation-v1\0");
    hasher.update(LINKED_ABIR_REVISION.as_bytes());
    hasher.update(LMQ_KERNEL_ID.as_bytes());
    CodecImplementation {
        build_id: build_id.into(),
        implementation_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        kernel_id: LMQ_KERNEL_ID.to_string(),
    }
}

pub fn transformed_fidelity(metric: impl Into<String>) -> CodecFidelity {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmq.fidelity-v1\0");
    hasher.update(LMQ_FIDELITY_CONTRACT.as_bytes());
    CodecFidelity {
        bound: None,
        contract_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        kind: CodecFidelityKind::Transformed,
        metric: Some(metric.into()),
    }
}

/// Validate shell-owned bounds before payload access or neural inference.
///
/// This checks invariants independent of any backend. It rejects profiles that
/// cannot hold even one valid LMQP1 packet or required BCS2 catalog.
pub fn validate_resource_bounds(bounds: LmqResourceBounds) -> Result<(), LmqError> {
    if bounds.bundle.max_catalog_bytes < MIN_LMQ_CATALOG_BYTES {
        return Err(LmqError::InvalidResourceProfile(
            "catalog budget is below the LMQ structural minimum",
        ));
    }
    if bounds.bundle.max_index_entries < 2 {
        return Err(LmqError::InvalidResourceProfile(
            "LMQ bundles require semantics and packet index entries",
        ));
    }
    if bounds.bundle.max_frame_bytes < MIN_LMQ_PACKET_FRAME_BYTES {
        return Err(LmqError::InvalidResourceProfile(
            "frame budget is below the minimum LMQP1 packet",
        ));
    }
    if bounds.bundle.max_generations == 0 || bounds.max_model_total < RANS_MODEL_TOTAL as u32 {
        return Err(LmqError::InvalidResourceProfile(
            "generation or fixed-model ceiling is undersized",
        ));
    }
    if bounds.max_tokens > lamquant_lml_mcu::rans::MAX_RANS_SYMBOLS as u32 {
        return Err(LmqError::InvalidResourceProfile(
            "token ceiling exceeds the hard rANS symbol limit",
        ));
    }
    if u64::from(bounds.max_alphabet) > RANS_MODEL_TOTAL {
        return Err(LmqError::InvalidResourceProfile(
            "alphabet ceiling exceeds the fixed rANS model total",
        ));
    }
    Ok(())
}

/// Validate one backend's complete production admission envelope.
///
/// Unlike the generic shell, a production Node promises its declared backend
/// capability range fits the compiled resource envelope. This function keeps
/// that contract in the shell that owns LMQ framing and hard limits.
pub fn validate_resource_profile(
    bounds: LmqResourceBounds,
    capabilities: crate::backend::NeuralBackendCapabilities,
) -> Result<(), LmqError> {
    validate_resource_bounds(bounds)?;
    capabilities
        .validate_input(
            capabilities.minimum_channels,
            capabilities.minimum_samples,
            capabilities.minimum_sample_rate,
        )
        .map_err(LmqError::BackendCapability)?;
    capabilities
        .validate_input(
            capabilities.maximum_channels,
            capabilities.maximum_samples,
            capabilities.maximum_sample_rate,
        )
        .map_err(LmqError::BackendCapability)?;
    if !matches!(
        capabilities.target,
        crate::backend::BackendTarget::HostNative | crate::backend::BackendTarget::HostSubprocess
    ) {
        return Err(LmqError::InvalidResourceProfile(
            "production LMQ Node requires a host backend",
        ));
    }
    if capabilities.maximum_tokens == 0
        || capabilities.minimum_alphabet == 0
        || capabilities.minimum_alphabet > capabilities.maximum_alphabet
        || u64::from(capabilities.maximum_alphabet) > RANS_MODEL_TOTAL
        || bounds.max_signal_bytes == 0
        || bounds.max_signal_channels == 0
        || bounds.max_tokens == 0
        || bounds.max_schedule_bytes == 0
        || bounds.max_alphabet == 0
        || bounds.max_model_basis_channels == 0
        || bounds.max_model_basis_terms == 0
        || bounds.max_model_derivations == 0
        || bounds.max_model_claims == 0
        || bounds.max_model_derivation_output_edges == 0
        || bounds.max_body_internal_working_bytes == 0
        || bounds.max_signal_channels < capabilities.maximum_channels
        || bounds.max_tokens < capabilities.maximum_tokens
        || bounds.max_schedule_bytes < capabilities.maximum_schedule_bytes
        || bounds.max_backend_meta_bytes < capabilities.maximum_backend_metadata_bytes
        || bounds.max_alphabet < capabilities.maximum_alphabet
    {
        return Err(LmqError::InvalidResourceProfile(
            "backend capability range exceeds LMQ resource bounds",
        ));
    }
    let maximum_signal_bytes = u64::from(capabilities.maximum_channels)
        .checked_mul(u64::from(capabilities.maximum_samples))
        .and_then(|samples| samples.checked_mul(8))
        .ok_or(LmqError::InvalidResourceProfile(
            "backend signal extent overflows",
        ))?;
    if maximum_signal_bytes > bounds.max_signal_bytes {
        return Err(LmqError::InvalidResourceProfile(
            "backend maximum signal exceeds decoded-signal budget",
        ));
    }
    let maximum_rans_bytes = u64::from(capabilities.maximum_tokens)
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or(LmqError::InvalidResourceProfile(
            "backend rANS extent overflows",
        ))?;
    let maximum_body_bytes = 15_u64
        .checked_add(u64::from(capabilities.maximum_alphabet) * 4)
        .and_then(|bytes| bytes.checked_add(u64::from(capabilities.maximum_schedule_bytes)))
        .and_then(|bytes| bytes.checked_add(maximum_rans_bytes))
        .ok_or(LmqError::InvalidResourceProfile(
            "backend body extent overflows",
        ))?;
    let maximum_packet_bytes = 15_u64
        .checked_add(u64::from(capabilities.maximum_backend_metadata_bytes))
        .and_then(|bytes| bytes.checked_add(maximum_body_bytes))
        .ok_or(LmqError::InvalidResourceProfile(
            "backend packet extent overflows",
        ))?;
    if maximum_packet_bytes > u64::from(bounds.bundle.max_frame_bytes) {
        return Err(LmqError::InvalidResourceProfile(
            "frame budget cannot hold maximum backend packet",
        ));
    }
    let maximum_body_working_bytes = u64::from(capabilities.maximum_alphabet)
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(maximum_rans_bytes))
        .and_then(|bytes| bytes.checked_add(maximum_body_bytes))
        .ok_or(LmqError::InvalidResourceProfile(
            "backend body working extent overflows",
        ))?;
    if maximum_body_working_bytes > bounds.max_body_internal_working_bytes {
        return Err(LmqError::InvalidResourceProfile(
            "body working budget cannot hold maximum backend output",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Compatibility-signature wrapper. `bounds.max_frame_bytes` also caps decoded
/// I64 signal materialization; use [`encode_bundle_bounded`] when compressed
/// frame and decoded working-set ceilings differ.
pub fn encode_bundle<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    backend: &dyn NeuralBackend,
    fidelity: CodecFidelity,
    implementation: CodecImplementation,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmqError> {
    encode_bundle_bounded(
        dataset,
        access,
        backend,
        fidelity,
        implementation,
        LmqResourceBounds::from_bundle(bounds),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_bundle_bounded<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    backend: &dyn NeuralBackend,
    fidelity: CodecFidelity,
    implementation: CodecImplementation,
    bounds: LmqResourceBounds,
) -> Result<Vec<u8>, LmqError> {
    validate_resource_bounds(bounds)?;
    if fidelity.kind == CodecFidelityKind::Exact || implementation.kernel_id != LMQ_KERNEL_ID {
        return Err(LmqError::CatalogContract);
    }
    // Refuse catalogs that cannot be represented truthfully in reconstructed
    // ABIR before payload access or backend work. This guarantees every emitted
    // bundle can pass the decoder's fidelity projection.
    codec_fidelity_statement(dataset.id(), &fidelity)?;
    let semantics = canonical_debug_json(dataset).map_err(|_| LmqError::SemanticEncoding)?;
    enforce_lmq_limit(
        LmqResource::SemanticFrameBytes,
        semantics.len() as u64,
        u64::from(bounds.bundle.max_frame_bytes),
    )?;
    let capabilities = backend.capabilities();
    let expected_channels = preflight_signal_channel_count(dataset, bounds)?;
    if expected_channels < capabilities.minimum_channels
        || expected_channels > capabilities.maximum_channels
    {
        return Err(LmqError::BackendCapability(
            BackendCapabilityError::ChannelCount,
        ));
    }
    let layout = resolve_signal_layout(dataset, bounds)?;
    let expected_samples = layout.samples;
    let sample_rate = layout.sample_rate;
    let model = backend.model();
    capabilities
        .validate_input(expected_channels, expected_samples, sample_rate)
        .map_err(LmqError::BackendCapability)?;
    validate_model_input_contract(
        dataset,
        &model,
        capabilities.signal_domain,
        expected_channels,
        sample_rate,
        expected_samples,
        bounds,
    )?;
    let domain_plan = compile_signal_domain_plan(&layout, capabilities.signal_domain)?;
    let signal = read_signal(&layout, access, bounds.max_signal_bytes, &domain_plan)?;
    let tokens = backend
        .encode(&signal, sample_rate)
        .map_err(LmqError::Backend)?;
    capabilities
        .validate_output(&tokens)
        .map_err(LmqError::BackendCapability)?;
    if tokens.n_channels != expected_channels || tokens.n_samples != expected_samples {
        return Err(LmqError::SignalShapeMismatch);
    }
    let packet = encode_packet_bounded(&tokens, bounds)?;
    let packets = [&packet[..]];
    encode_codec_bundle(
        CodecBundleInput {
            // Baseline kernels: any reader of the profile can decode these packets.
            required_capabilities: 0,
            canonical_semantics: &semantics,
            fidelity,
            implementation,
            model_provenance: Some(model.provenance().clone()),
            packets: &packets,
            parameters: canonical_parameters(capabilities.signal_domain, model.trained_artifact()),
            profile: CodecProfile::LmqProgressive,
        },
        bounds.bundle,
    )
    .map_err(LmqError::Bundle)
}

pub fn open_bundle<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: ResourceBounds,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    open_bundle_bounded_with_policy(
        bytes,
        backend,
        LmqResourceBounds::from_bundle(bounds),
        LegacyModelContractPolicy::Reject,
    )
}

/// Open one pre-contract catalog through explicit compatibility policy.
pub fn open_bundle_with_legacy_contract<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: ResourceBounds,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    open_bundle_bounded_with_policy(
        bytes,
        backend,
        LmqResourceBounds::from_bundle(bounds),
        LegacyModelContractPolicy::AllowPreContractCatalog,
    )
}

pub fn open_bundle_bounded<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: LmqResourceBounds,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    open_bundle_bounded_with_policy(bytes, backend, bounds, LegacyModelContractPolicy::Reject)
}

pub fn open_bundle_bounded_with_policy<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: LmqResourceBounds,
    legacy_policy: LegacyModelContractPolicy,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    validate_resource_bounds(bounds)?;
    let bundle = CodecBundleView::open(bytes, bounds.bundle).map_err(LmqError::Bundle)?;
    let catalog = bundle.catalog();
    let capabilities = backend.capabilities();
    let model = backend.model();
    if catalog.profile() != CodecProfile::LmqProgressive
        || catalog.packet_count() != 1
        || catalog.model_provenance() != Some(model.provenance())
        || catalog.fidelity().kind == CodecFidelityKind::Exact
        || catalog.implementation().kernel_id != LMQ_KERNEL_ID
        || !catalog_parameters_supported(
            catalog.parameters(),
            capabilities.signal_domain,
            &model,
            legacy_policy,
        )
    {
        return Err(LmqError::CatalogContract);
    }
    let dataset = parse_canonical_dataset(bundle.canonical_semantics())
        .map_err(|_| LmqError::SemanticEncoding)?;
    let expected_channels = preflight_signal_channel_count(&dataset, bounds)?;
    if expected_channels < capabilities.minimum_channels
        || expected_channels > capabilities.maximum_channels
    {
        return Err(LmqError::BackendCapability(
            BackendCapabilityError::ChannelCount,
        ));
    }
    let layout = resolve_signal_layout(&dataset, bounds)?;
    let expected_samples = layout.samples;
    let sample_rate = layout.sample_rate;
    capabilities
        .validate_input(expected_channels, expected_samples, sample_rate)
        .map_err(LmqError::BackendCapability)?;
    validate_model_input_contract(
        &dataset,
        &model,
        capabilities.signal_domain,
        expected_channels,
        sample_rate,
        expected_samples,
        bounds,
    )?;
    let domain_plan = compile_signal_domain_plan(&layout, capabilities.signal_domain)?;
    enforce_signal_bound(expected_channels, expected_samples, bounds.max_signal_bytes)?;
    let packet = bundle.packet(0).ok_or(LmqError::Header)?;
    let tokens = decode_packet_bounded(packet, bounds)?;
    capabilities
        .validate_output(&tokens)
        .map_err(LmqError::BackendCapability)?;
    if tokens.n_channels != expected_channels || tokens.n_samples != expected_samples {
        return Err(LmqError::SignalShapeMismatch);
    }
    let signal = backend.decode(&tokens).map_err(LmqError::Backend)?;
    capabilities
        .validate_signal(&signal, sample_rate)
        .map_err(LmqError::BackendCapability)?;
    if signal.channels.len() != usize::from(tokens.n_channels)
        || signal
            .channels
            .iter()
            .any(|channel| channel.len() != tokens.n_samples as usize)
    {
        return Err(LmqError::SignalShapeMismatch);
    }
    let signal = model_signal_to_source_digital(signal, &domain_plan)?;
    let reconstructed = build_reconstructed_dataset(
        &dataset,
        &signal,
        ReconstructionContext {
            fidelity: catalog.fidelity(),
            implementation: catalog.implementation(),
            model: catalog
                .model_provenance()
                .ok_or(LmqError::CatalogContract)?,
            source_semantic_id: catalog.source_semantic_id(),
            source_interchange_id: catalog.source_interchange_id(),
        },
    )?;
    Ok(OpenedLmqBundle {
        bundle,
        source_dataset: dataset,
        reconstructed,
    })
}

struct ResolvedSignalLayout<'a> {
    atoms: Vec<&'a Atom>,
    channels: u16,
    samples: u32,
    sample_rate: Rational,
}

fn preflight_signal_channel_count(
    dataset: &AbirDataset,
    bounds: LmqResourceBounds,
) -> Result<u16, LmqError> {
    if dataset.recordings().len() != 1 || dataset.streams().len() != 1 {
        return Err(LmqError::UnsupportedSemantics(
            "requires exactly one recording and one stream",
        ));
    }
    let recording = &dataset.recordings()[0];
    let stream = &dataset.streams()[0];
    if recording.streams() != [stream.id()]
        || stream.recording_id() != recording.id()
        || stream.atoms().is_empty()
        || stream.atoms().len() != dataset.atoms().len()
    {
        return Err(LmqError::UnsupportedSemantics(
            "stream must own every atom exactly once",
        ));
    }
    enforce_lmq_limit(
        LmqResource::SignalChannels,
        stream.atoms().len() as u64,
        u64::from(bounds.max_signal_channels),
    )?;
    u16::try_from(stream.atoms().len()).map_err(|_| LmqError::ResourceLimit {
        resource: LmqResource::SignalChannels,
        actual: stream.atoms().len() as u64,
        limit: u64::from(bounds.max_signal_channels),
    })
}

fn resolve_signal_layout(
    dataset: &AbirDataset,
    bounds: LmqResourceBounds,
) -> Result<ResolvedSignalLayout<'_>, LmqError> {
    let channels = preflight_signal_channel_count(dataset, bounds)?;
    let stream = &dataset.streams()[0];
    let mut atom_by_id = dataset
        .atoms()
        .iter()
        .map(|atom| (atom.id(), atom))
        .collect::<Vec<_>>();
    atom_by_id.sort_unstable_by_key(|(id, _)| *id);
    if atom_by_id.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(LmqError::UnsupportedSemantics(
            "dataset atom identities are not unique",
        ));
    }
    let mut atoms = Vec::with_capacity(usize::from(channels));
    let mut membership = Vec::with_capacity(usize::from(channels));
    for atom_id in stream.atoms() {
        let index = atom_by_id
            .binary_search_by_key(atom_id, |(id, _)| *id)
            .map_err(|_| LmqError::UnsupportedSemantics("unresolved stream atom"))?;
        atoms.push(atom_by_id[index].1);
        membership.push(*atom_id);
    }
    membership.sort_unstable();
    if membership.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(LmqError::UnsupportedSemantics(
            "stream atom membership is not bijective",
        ));
    }

    let mut samples = None;
    let mut start = None;
    let mut sample_rate = None;
    for atom in &atoms {
        let Atom::SignalBlock(block) = atom else {
            return Err(LmqError::UnsupportedSemantics(
                "only SignalBlock atoms are supported",
            ));
        };
        if atom.presence() != Presence::Present {
            return Err(LmqError::UnsupportedSemantics(
                "only present signal blocks are supported",
            ));
        }
        let descriptor = atom
            .payload()
            .ok_or(LmqError::UnsupportedSemantics("signal has no payload"))?;
        validate_descriptor(descriptor)?;
        let TimeAxis::Regular(segment) = block.time_axis() else {
            return Err(LmqError::UnsupportedSemantics(
                "LMQ requires a regular time axis",
            ));
        };
        if descriptor.shape().last().copied() != Some(segment.samples())
            || samples
                .replace(segment.samples())
                .is_some_and(|prior| prior != segment.samples())
        {
            return Err(LmqError::SignalShapeMismatch);
        }
        if start
            .replace(segment.start())
            .is_some_and(|prior| prior != segment.start())
            || sample_rate
                .replace(segment.rate())
                .is_some_and(|prior| prior != segment.rate())
        {
            return Err(LmqError::UnsupportedSemantics(
                "LMQ requires aligned starts and uniform rates",
            ));
        }
    }
    let samples = u32::try_from(samples.ok_or(LmqError::SignalShapeMismatch)?)
        .map_err(|_| LmqError::SignalShapeMismatch)?;
    Ok(ResolvedSignalLayout {
        atoms,
        channels,
        samples,
        sample_rate: sample_rate.ok_or(LmqError::SignalShapeMismatch)?,
    })
}

fn canonical_parameters(
    signal_domain: SignalDomain,
    model: Option<&TrainedModelArtifact>,
) -> Vec<CodecParameter> {
    let mut parameters = vec![
        CodecParameter {
            name: "abir.revision".to_string(),
            value: CodecParameterValue::Text {
                value: LMQ_WIRE_ABIR_REVISION.to_string(),
            },
        },
        CodecParameter {
            name: "lmq.packet_grammar".to_string(),
            value: CodecParameterValue::Text {
                value: "LMQP1".to_string(),
            },
        },
        CodecParameter {
            name: "neural.signal-domain".to_string(),
            value: CodecParameterValue::Text {
                value: signal_domain.protocol_name().to_string(),
            },
        },
        CodecParameter {
            name: "semantic.fidelity-contract".to_string(),
            value: CodecParameterValue::Text {
                value: LMQ_FIDELITY_CONTRACT.to_string(),
            },
        },
    ];
    if let Some(artifact) = model {
        parameters.push(CodecParameter {
            name: "neural.input-contract".to_string(),
            value: CodecParameterValue::Text {
                value: artifact.input_contract().content_id().to_string(),
            },
        });
        parameters.push(CodecParameter {
            name: "neural.model-artifact".to_string(),
            value: CodecParameterValue::Text {
                value: artifact.content_id().to_string(),
            },
        });
    }
    parameters.sort_by(|left, right| left.name.cmp(&right.name));
    parameters
}

fn catalog_parameters_supported(
    actual: &[CodecParameter],
    signal_domain: SignalDomain,
    model: &BackendModel<'_>,
    legacy_policy: LegacyModelContractPolicy,
) -> bool {
    actual == canonical_parameters(signal_domain, model.trained_artifact())
        || (legacy_policy == LegacyModelContractPolicy::AllowPreContractCatalog
            && model.input_contract().is_some()
            && actual == canonical_parameters(signal_domain, None))
}

fn validate_model_input_contract(
    dataset: &AbirDataset,
    model: &BackendModel<'_>,
    signal_domain: SignalDomain,
    expected_channels: u16,
    sample_rate: Rational,
    samples: u32,
    bounds: LmqResourceBounds,
) -> Result<(), LmqError> {
    let Some(contract) = model.input_contract() else {
        return Ok(());
    };
    if contract.signal_domain() != signal_domain {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::SignalDomain,
        ));
    }
    if contract.sample_rate() != sample_rate {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::SampleRate,
        ));
    }
    if contract.samples() != samples {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::SampleCount,
        ));
    }
    let stream = dataset
        .streams()
        .first()
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::Modality,
        ))?;
    if stream.modality() != contract.modality() {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::Modality,
        ));
    }
    let basis_id = stream
        .channel_basis_id()
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::MissingChannelBasis,
        ))?;
    let basis = dataset
        .channel_bases()
        .iter()
        .find(|basis| basis.id() == basis_id)
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::MissingChannelBasis,
        ))?;
    let expected_channels = usize::from(expected_channels);
    if basis.channels().len() != expected_channels
        || contract.channel_concepts().len() != expected_channels
    {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::ChannelCount,
        ));
    }
    if basis
        .channels()
        .iter()
        .map(|channel| channel.concept())
        .ne(contract.channel_concepts())
    {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::ChannelOrder,
        ));
    }
    if model_channel_basis_content_id_bounded(dataset, bounds)?
        != contract.model_channel_basis_content_id()
    {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::ChannelBasis,
        ));
    }
    enforce_lmq_limit(
        LmqResource::ModelDerivations,
        dataset.derivations().len() as u64,
        u64::from(bounds.max_model_derivations),
    )?;
    enforce_lmq_limit(
        LmqResource::ModelClaims,
        dataset.proofs().len() as u64,
        u64::from(bounds.max_model_claims),
    )?;
    let mut claim_subjects = dataset
        .proofs()
        .iter()
        .filter(|proof| proof.kind() == contract.upstream_claim_kind())
        .map(|proof| proof.subject())
        .collect::<Vec<_>>();
    claim_subjects.sort_unstable();
    claim_subjects.dedup();

    let stream_ref = SemanticRef::of(stream.id());
    let mut saw_output = false;
    let mut saw_operation = false;
    let mut matched_derivation = None;
    let mut output_edges = 0_u64;
    for derivation in dataset.derivations() {
        output_edges = output_edges
            .checked_add(derivation.outputs().len() as u64)
            .ok_or(LmqError::ResourceLimit {
                resource: LmqResource::ModelDerivationOutputEdges,
                actual: u64::MAX,
                limit: u64::from(bounds.max_model_derivation_output_edges),
            })?;
        enforce_lmq_limit(
            LmqResource::ModelDerivationOutputEdges,
            output_edges,
            u64::from(bounds.max_model_derivation_output_edges),
        )?;
        if !derivation.outputs().contains(&stream_ref) {
            continue;
        }
        saw_output = true;
        if derivation.operation() != contract.upstream_derivation() {
            continue;
        }
        saw_operation = true;
        let derivation_ref = SemanticRef::of(derivation.id());
        if claim_subjects.binary_search(&derivation_ref).is_ok()
            && matched_derivation.replace(derivation).is_some()
        {
            return Err(LmqError::ModelInputContract(
                ModelInputContractError::Derivation,
            ));
        }
    }
    if !saw_output {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::MissingDerivation,
        ));
    }
    if !saw_operation {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::Derivation,
        ));
    }
    matched_derivation.ok_or(LmqError::ModelInputContract(
        ModelInputContractError::MissingDerivationClaim,
    ))?;
    Ok(())
}

/// Portable model-role identity of exact weighted basis used by first stream.
///
/// ABIR `ObjectId<ChannelTag>` values retain source-observation identity. Model
/// contracts intentionally project those instance identities onto channel-role
/// concepts so one contract applies across recordings. Every referenced role
/// must resolve uniquely; duplicate roles fail closed rather than collapsing
/// distinct observations.
pub fn model_channel_basis_content_id(dataset: &AbirDataset) -> Result<ContentId, LmqError> {
    model_channel_basis_content_id_bounded(dataset, LmqResourceBounds::default())
}

pub fn model_channel_basis_content_id_bounded(
    dataset: &AbirDataset,
    bounds: LmqResourceBounds,
) -> Result<ContentId, LmqError> {
    let stream = dataset
        .streams()
        .first()
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::Modality,
        ))?;
    let basis_id = stream
        .channel_basis_id()
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::MissingChannelBasis,
        ))?;
    let basis = dataset
        .channel_bases()
        .iter()
        .find(|basis| basis.id() == basis_id)
        .ok_or(LmqError::ModelInputContract(
            ModelInputContractError::MissingChannelBasis,
        ))?;
    let construction = basis.construction().ok_or(LmqError::ModelInputContract(
        ModelInputContractError::MissingChannelBasisConstruction,
    ))?;
    if construction.len() != basis.channels().len() {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::ChannelBasis,
        ));
    }
    enforce_lmq_limit(
        LmqResource::ModelBasisChannels,
        dataset.channels().len() as u64,
        u64::from(bounds.max_model_basis_channels),
    )?;
    enforce_lmq_limit(
        LmqResource::ModelBasisChannels,
        basis.channels().len() as u64,
        u64::from(bounds.max_model_basis_channels),
    )?;

    // Build one bounded typed-source index. Portable role ambiguity is checked
    // only across source records actually referenced by this basis; unrelated
    // catalog channels cannot invalidate an otherwise exact construction.
    let mut source_by_id =
        Vec::<(ObjectId<ChannelTag>, &ConceptId)>::with_capacity(dataset.channels().len());
    for channel in dataset.channels() {
        source_by_id.push((channel.id(), channel.kind()));
    }
    source_by_id.sort_by_key(|(id, _)| *id);

    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, MODEL_CHANNEL_BASIS_DOMAIN);
    hash_field(&mut hasher, reference_kind_name(basis.reference()));
    let output_count = u32::try_from(basis.channels().len())
        .map_err(|_| LmqError::ModelInputContract(ModelInputContractError::ChannelCount))?;
    hash_field(&mut hasher, &output_count.to_le_bytes());

    let mut total_terms = 0_u64;
    let mut referenced_sources = Vec::new();
    for (output, vector) in basis.channels().iter().zip(construction) {
        hash_field(&mut hasher, output.concept().as_str().as_bytes());
        let term_count = u32::try_from(vector.terms().len())
            .map_err(|_| LmqError::ModelInputContract(ModelInputContractError::ChannelBasis))?;
        total_terms =
            total_terms
                .checked_add(u64::from(term_count))
                .ok_or(LmqError::ResourceLimit {
                    resource: LmqResource::ModelBasisTerms,
                    actual: u64::MAX,
                    limit: u64::from(bounds.max_model_basis_terms),
                })?;
        enforce_lmq_limit(
            LmqResource::ModelBasisTerms,
            total_terms,
            u64::from(bounds.max_model_basis_terms),
        )?;
        hash_field(&mut hasher, &term_count.to_le_bytes());
        let mut terms = Vec::<(&ConceptId, Rational)>::with_capacity(vector.terms().len());
        for term in vector.terms() {
            let source_index = source_by_id
                .binary_search_by_key(&term.source(), |(id, _)| *id)
                .map_err(|_| {
                    LmqError::ModelInputContract(ModelInputContractError::MissingChannelBasisSource)
                })?;
            let source = source_by_id[source_index].1;
            referenced_sources.push((source, term.source()));
            terms.push((source, term.coefficient()));
        }
        terms.sort_by(|left, right| {
            left.0.cmp(right.0).then_with(|| {
                let (left_numerator, left_denominator) = left.1.parts();
                let (right_numerator, right_denominator) = right.1.parts();
                (left_numerator, left_denominator).cmp(&(right_numerator, right_denominator))
            })
        });
        if terms.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(LmqError::ModelInputContract(
                ModelInputContractError::AmbiguousChannelBasisSource,
            ));
        }
        for (source, coefficient) in terms {
            hash_field(&mut hasher, source.as_str().as_bytes());
            let (numerator, denominator) = coefficient.parts();
            hash_field(&mut hasher, &numerator.to_le_bytes());
            hash_field(&mut hasher, &denominator.to_le_bytes());
        }
    }
    referenced_sources.sort_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(&right.1)));
    if referenced_sources
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1)
    {
        return Err(LmqError::ModelInputContract(
            ModelInputContractError::AmbiguousChannelBasisSource,
        ));
    }
    Ok(ContentId::from_bytes(*hasher.finalize().as_bytes()))
}

const fn reference_kind_name(reference: ReferenceKind) -> &'static [u8] {
    match reference {
        ReferenceKind::Absolute => b"absolute",
        ReferenceKind::Common => b"common",
        ReferenceKind::Differential => b"differential",
        ReferenceKind::Unknown => b"unknown",
    }
}

/// Encode one backend token envelope into exact production LMQP1 bytes.
///
/// Evaluation tooling uses this seam to measure real packet bytes without
/// constructing a BCS2 catalog. Runtime bundle encoding calls the same bounded
/// implementation, so measured entropy/header bytes cannot drift into a
/// parallel estimator.
pub fn encode_token_packet(tokens: &NeuralTokens) -> Result<Vec<u8>, LmqError> {
    encode_packet_bounded(tokens, LmqResourceBounds::default())
}

#[cfg(test)]
fn encode_packet(tokens: &NeuralTokens) -> Result<Vec<u8>, LmqError> {
    encode_token_packet(tokens)
}

fn encode_packet_bounded(
    tokens: &NeuralTokens,
    bounds: LmqResourceBounds,
) -> Result<Vec<u8>, LmqError> {
    if tokens.n_channels == 0 || tokens.n_samples == 0 {
        return Err(LmqError::SignalShapeMismatch);
    }
    enforce_lmq_limit(
        LmqResource::TokenCount,
        tokens.tokens.len() as u64,
        u64::from(bounds.max_tokens),
    )?;
    enforce_lmq_limit(
        LmqResource::ScheduleBytes,
        tokens.schedule.len() as u64,
        u64::from(bounds.max_schedule_bytes),
    )?;
    enforce_lmq_limit(
        LmqResource::BackendMetadataBytes,
        tokens.backend_meta.len() as u64,
        u64::from(bounds.max_backend_meta_bytes),
    )?;
    enforce_lmq_limit(
        LmqResource::Alphabet,
        u64::from(tokens.alphabet),
        u64::from(bounds.max_alphabet),
    )?;
    enforce_lmq_limit(
        LmqResource::ModelTotal,
        RANS_MODEL_TOTAL,
        u64::from(bounds.max_model_total),
    )?;
    let packet_prefix = PACKET_HEADER_LEN
        .checked_add(tokens.backend_meta.len())
        .ok_or(LmqError::ResourceLimit {
            resource: LmqResource::PacketBytes,
            actual: u64::MAX,
            limit: u64::from(bounds.bundle.max_frame_bytes),
        })?;
    let body_budget = (bounds.bundle.max_frame_bytes as usize)
        .checked_sub(packet_prefix)
        .ok_or(LmqError::ResourceLimit {
            resource: LmqResource::PacketBytes,
            actual: packet_prefix as u64,
            limit: u64::from(bounds.bundle.max_frame_bytes),
        })?;
    let counts = histogram(&tokens.tokens, tokens.alphabet)?;
    let symbols = tokens
        .tokens
        .iter()
        .map(|&token| i64::from(token))
        .collect::<Vec<_>>();
    let body = encode_body_bounded(
        &symbols,
        &tokens.schedule,
        &counts,
        bounds.body(body_budget as u32),
    )?;
    let meta_len = u32::try_from(tokens.backend_meta.len()).map_err(|_| LmqError::Header)?;
    let packet_len = PACKET_HEADER_LEN
        .checked_add(tokens.backend_meta.len())
        .and_then(|bytes| bytes.checked_add(body.len()))
        .ok_or(LmqError::ResourceLimit {
            resource: LmqResource::PacketBytes,
            actual: u64::MAX,
            limit: u64::from(bounds.bundle.max_frame_bytes),
        })?;
    enforce_lmq_limit(
        LmqResource::PacketBytes,
        packet_len as u64,
        u64::from(bounds.bundle.max_frame_bytes),
    )?;
    let mut packet = Vec::with_capacity(packet_len);
    packet.extend_from_slice(PACKET_MAGIC);
    packet.push(PACKET_VERSION);
    packet.extend_from_slice(&tokens.n_channels.to_le_bytes());
    packet.extend_from_slice(&tokens.n_samples.to_le_bytes());
    packet.extend_from_slice(&meta_len.to_le_bytes());
    packet.extend_from_slice(&tokens.backend_meta);
    packet.extend_from_slice(&body);
    Ok(packet)
}

#[cfg(test)]
fn decode_packet(packet: &[u8]) -> Result<NeuralTokens, LmqError> {
    decode_packet_bounded(packet, LmqResourceBounds::default())
}

fn decode_packet_bounded(
    packet: &[u8],
    bounds: LmqResourceBounds,
) -> Result<NeuralTokens, LmqError> {
    if packet.get(..4) != Some(PACKET_MAGIC) || packet.get(4) != Some(&PACKET_VERSION) {
        return Err(LmqError::Header);
    }
    let n_channels = u16::from_le_bytes(
        packet
            .get(5..7)
            .ok_or(LmqError::Header)?
            .try_into()
            .map_err(|_| LmqError::Header)?,
    );
    let n_samples = u32::from_le_bytes(
        packet
            .get(7..11)
            .ok_or(LmqError::Header)?
            .try_into()
            .map_err(|_| LmqError::Header)?,
    );
    let meta_len = u32::from_le_bytes(
        packet
            .get(11..15)
            .ok_or(LmqError::Header)?
            .try_into()
            .map_err(|_| LmqError::Header)?,
    ) as usize;
    if n_channels == 0 || n_samples == 0 {
        return Err(LmqError::SignalShapeMismatch);
    }
    enforce_lmq_limit(
        LmqResource::BackendMetadataBytes,
        meta_len as u64,
        u64::from(bounds.max_backend_meta_bytes),
    )?;
    let after_header = packet.get(PACKET_HEADER_LEN..).ok_or(LmqError::Header)?;
    let backend_meta = after_header
        .get(..meta_len)
        .ok_or(LmqError::Header)?
        .to_vec();
    let body = after_header.get(meta_len..).ok_or(LmqError::Header)?;
    let body_budget = (bounds.bundle.max_frame_bytes as usize)
        .checked_sub(PACKET_HEADER_LEN)
        .and_then(|bytes| bytes.checked_sub(meta_len))
        .ok_or(LmqError::ResourceLimit {
            resource: LmqResource::PacketBytes,
            actual: packet.len() as u64,
            limit: u64::from(bounds.bundle.max_frame_bytes),
        })?;
    let (symbols, schedule, alphabet) = decode_body_bounded(body, bounds.body(body_budget as u32))?;
    let mut tokens = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        if symbol < 0 || symbol >= i64::from(alphabet) {
            return Err(LmqError::BadTokens);
        }
        tokens.push(i32::try_from(symbol).map_err(|_| LmqError::BadTokens)?);
    }
    Ok(NeuralTokens {
        tokens,
        schedule,
        alphabet,
        n_channels,
        n_samples,
        backend_meta,
    })
}

fn read_signal<A: PayloadAccess>(
    layout: &ResolvedSignalLayout<'_>,
    access: &A,
    max_signal_bytes: u64,
    domain_plan: &SignalDomainPlan,
) -> Result<NeuralSignal, LmqError> {
    enforce_signal_bound(layout.channels, layout.samples, max_signal_bytes)?;
    let mut signal = Vec::with_capacity(layout.atoms.len());
    for (index, atom) in layout.atoms.iter().enumerate() {
        let descriptor = atom
            .payload()
            .ok_or(LmqError::UnsupportedSemantics("signal has no payload"))?;
        let lease = access.lease(descriptor).map_err(LmqError::PayloadAccess)?;
        verify_payload_content(descriptor, lease.bytes())
            .map_err(|_| LmqError::PayloadIdentityMismatch)?;
        let mut channel = decode_integer_payload(descriptor, lease.bytes())?;
        if channel.len() != layout.samples as usize {
            return Err(LmqError::SignalShapeMismatch);
        }
        if domain_plan.domain == SignalDomain::PhysicalMicrovoltQ16 {
            let transform = *domain_plan
                .transforms
                .get(index)
                .ok_or(LmqError::SignalShapeMismatch)?;
            for sample in &mut channel {
                *sample = transform.digital_to_model(*sample)?;
            }
        }
        signal.push(channel);
    }
    Ok(NeuralSignal {
        domain: domain_plan.domain,
        channels: signal,
    })
}

struct SignalDomainPlan {
    domain: SignalDomain,
    transforms: Vec<AffineDomainTransform>,
}

fn compile_signal_domain_plan(
    layout: &ResolvedSignalLayout<'_>,
    domain: SignalDomain,
) -> Result<SignalDomainPlan, LmqError> {
    if domain == SignalDomain::DigitalInteger {
        return Ok(SignalDomainPlan {
            domain,
            transforms: Vec::new(),
        });
    }
    let mut transforms = Vec::with_capacity(layout.atoms.len());
    for atom in &layout.atoms {
        let Atom::SignalBlock(block) = atom else {
            return Err(LmqError::UnsupportedSemantics(
                "only SignalBlock atoms are supported",
            ));
        };
        let calibration = block.calibration().ok_or(LmqError::UnsupportedSemantics(
            "physical model domain requires exact per-channel calibration",
        ))?;
        transforms.push(AffineDomainTransform::compile(calibration)?);
    }
    Ok(SignalDomainPlan { domain, transforms })
}

fn model_signal_to_source_digital(
    signal: NeuralSignal,
    domain_plan: &SignalDomainPlan,
) -> Result<Vec<Vec<i64>>, LmqError> {
    if signal.domain != domain_plan.domain {
        return Err(LmqError::SignalShapeMismatch);
    }
    if signal.domain == SignalDomain::DigitalInteger {
        if !domain_plan.transforms.is_empty() {
            return Err(LmqError::SignalShapeMismatch);
        }
        return Ok(signal.channels);
    }
    if domain_plan.transforms.len() != signal.channels.len() {
        return Err(LmqError::SignalShapeMismatch);
    }
    let mut output = Vec::with_capacity(signal.channels.len());
    for (mut channel, transform) in signal.channels.into_iter().zip(&domain_plan.transforms) {
        for sample in &mut channel {
            *sample = (*transform).model_to_digital(*sample)?;
        }
        output.push(channel);
    }
    Ok(output)
}

fn enforce_signal_bound(
    channels: u16,
    samples: u32,
    max_signal_bytes: u64,
) -> Result<(), LmqError> {
    let decoded_bytes = u64::from(channels)
        .checked_mul(u64::from(samples))
        .and_then(|count| count.checked_mul(8))
        .ok_or(LmqError::ResourceLimit {
            resource: LmqResource::SignalBytes,
            actual: u64::MAX,
            limit: max_signal_bytes,
        })?;
    enforce_lmq_limit(LmqResource::SignalBytes, decoded_bytes, max_signal_bytes)
}

fn enforce_lmq_limit(resource: LmqResource, actual: u64, limit: u64) -> Result<(), LmqError> {
    if actual > limit {
        Err(LmqError::ResourceLimit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn validate_descriptor(descriptor: &PayloadDescriptor) -> Result<(), LmqError> {
    if !matches!(
        descriptor.element(),
        ElementType::I8 | ElementType::I16 | ElementType::I24 | ElementType::I32 | ElementType::I64
    ) || !matches!(descriptor.byte_order(), ByteOrder::Little | ByteOrder::Big)
        || !matches!(
            descriptor.layout(),
            Layout::DenseRowMajor | Layout::DenseColumnMajor
        )
        || descriptor.encoding().is_some()
        || !matches!(descriptor.shape(), [_] | [1, _])
    {
        return Err(LmqError::UnsupportedSemantics(
            "payload must be dense, unencoded signed integers with shape [T] or [1,T]",
        ));
    }
    Ok(())
}

fn decode_integer_payload(
    descriptor: &PayloadDescriptor,
    bytes: &[u8],
) -> Result<Vec<i64>, LmqError> {
    let width = descriptor
        .element()
        .byte_width()
        .ok_or(LmqError::SignalShapeMismatch)? as usize;
    if bytes.len() % width != 0 {
        return Err(LmqError::SignalShapeMismatch);
    }
    bytes
        .chunks_exact(width)
        .map(|chunk| decode_integer(descriptor.element(), descriptor.byte_order(), chunk))
        .collect()
}

fn decode_integer(element: ElementType, order: ByteOrder, bytes: &[u8]) -> Result<i64, LmqError> {
    match (element, order) {
        (ElementType::I8, _) => Ok(i64::from(i8::from_ne_bytes([bytes[0]]))),
        (ElementType::I16, ByteOrder::Little) => {
            Ok(i64::from(i16::from_le_bytes([bytes[0], bytes[1]])))
        }
        (ElementType::I16, ByteOrder::Big) => {
            Ok(i64::from(i16::from_be_bytes([bytes[0], bytes[1]])))
        }
        (ElementType::I24, ByteOrder::Little) => {
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
            Ok(i64::from(((raw << 8) as i32) >> 8))
        }
        (ElementType::I24, ByteOrder::Big) => {
            let raw = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
            Ok(i64::from(((raw << 8) as i32) >> 8))
        }
        (ElementType::I32, ByteOrder::Little) => Ok(i64::from(i32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| LmqError::SignalShapeMismatch)?,
        ))),
        (ElementType::I32, ByteOrder::Big) => Ok(i64::from(i32::from_be_bytes(
            bytes
                .try_into()
                .map_err(|_| LmqError::SignalShapeMismatch)?,
        ))),
        (ElementType::I64, ByteOrder::Little) => Ok(i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| LmqError::SignalShapeMismatch)?,
        )),
        (ElementType::I64, ByteOrder::Big) => Ok(i64::from_be_bytes(
            bytes
                .try_into()
                .map_err(|_| LmqError::SignalShapeMismatch)?,
        )),
        _ => Err(LmqError::UnsupportedSemantics(
            "unsupported integer payload",
        )),
    }
}

fn histogram(tokens: &[i32], alphabet: u16) -> Result<Vec<i32>, LmqError> {
    let alphabet = usize::from(alphabet);
    if alphabet == 0 || alphabet as u64 > RANS_MODEL_TOTAL {
        return Err(LmqError::BadTokens);
    }
    let mut raw = vec![0_u64; alphabet];
    for &token in tokens {
        let index = usize::try_from(token).map_err(|_| LmqError::BadTokens)?;
        let count = raw.get_mut(index).ok_or(LmqError::BadTokens)?;
        *count += 1;
    }
    let mut frequencies = vec![1_i32; alphabet];
    let total: u64 = raw.iter().sum();
    if total == 0 {
        return Ok(frequencies);
    }
    let budget = RANS_MODEL_TOTAL - alphabet as u64;
    let mut assigned = 0_u64;
    for (frequency, count) in frequencies.iter_mut().zip(&raw) {
        let extra = count.saturating_mul(budget) / total;
        *frequency += extra as i32;
        assigned += extra;
    }
    let remainder = budget - assigned;
    if remainder > 0 {
        let best = (0..alphabet).max_by_key(|&index| raw[index]).unwrap_or(0);
        frequencies[best] += remainder as i32;
    }
    Ok(frequencies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ModelInputContract, StubBackend};
    use alloc::format;
    use semantic_abir::{
        payload_content_id, Acquisition, AcquisitionTag, AtomTag, Calibration, Channel,
        ChannelBasis, ChannelBasisTag, ChannelBasisTerm, ChannelBasisVector, ChannelSpec,
        ChannelTag, Clock, ClockRelation, ClockRelationTag, ClockTag, ConceptDictionary,
        ConceptDictionaryTag, ConceptId, CoordinateFrame, CoordinateFrameTag, DatasetDraft,
        DatasetTag, Derivation, DerivationTag, DerivedArtifact, DerivedArtifactTag, Device,
        DeviceTag, Event, EventTag, ExactNumber, ExecutionRecord, Fidelity, FidelityKind,
        FrameTransform, FrameTransformTag, InMemoryPayloadAccess, ObjectId, OpenedDataset, Patient,
        PatientTag, Policy, PolicyTag, Proof, ProofTag, Rational, Recording, RecordingTag,
        ReferenceKind, SemanticRef, Sensor, SensorTag, Session, SessionTag, SignalBlock,
        SourceCapsule, SourceKey, SourceRelationship, Stream, StreamTag, Subject, SubjectTag,
        TimeSegment, ValidationLimits,
    };
    use semantic_abir_bcs::{ModelProvenance, PccpStatus, BCS2_MAGIC};

    fn stub_model_provenance() -> ModelProvenance {
        match StubBackend::default().model() {
            BackendModel::ModelFree(provenance) => provenance,
            BackendModel::Trained(_) => unreachable!(),
        }
    }

    fn stub_model_artifact(contract: &ModelInputContract) -> TrainedModelArtifact {
        TrainedModelArtifact::new(stub_model_provenance(), contract.clone())
    }

    fn fixture() -> OpenedDataset<InMemoryPayloadAccess> {
        fixture_with_starts(&[0, 0, 0, 0])
    }

    fn fixture_with_starts(starts: &[i128]) -> OpenedDataset<InMemoryPayloadAccess> {
        let (mut draft, access, atom_ids, recording_id, stream_id) = fixture_parts(starts);
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            ConceptId::new("abir:modality/eeg").unwrap(),
            atom_ids,
            None,
            None,
            None,
        ));
        OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
    }

    fn fixture_parts(
        starts: &[i128],
    ) -> (
        DatasetDraft,
        InMemoryPayloadAccess,
        Vec<ObjectId<AtomTag>>,
        ObjectId<RecordingTag>,
        ObjectId<StreamTag>,
    ) {
        fixture_parts_with_calibration(starts, None)
    }

    fn fixture_parts_with_calibration(
        starts: &[i128],
        calibration: Option<Calibration>,
    ) -> (
        DatasetDraft,
        InMemoryPayloadAccess,
        Vec<ObjectId<AtomTag>>,
        ObjectId<RecordingTag>,
        ObjectId<StreamTag>,
    ) {
        let signal = (0..4)
            .map(|channel| {
                (0..500)
                    .map(|sample| ((sample * 3 + channel * 7) % 40) as i64 - 20)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([1; 16]));
        let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
        let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
        let mut access = InMemoryPayloadAccess::new();
        let mut atom_ids = Vec::new();
        for (index, channel) in signal.iter().enumerate() {
            let bytes = channel
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>();
            let content_id = payload_content_id(ElementType::I64, &bytes);
            access.insert(content_id, bytes);
            let mut id = [0_u8; 16];
            id[15] = (index + 1) as u8;
            let atom_id = ObjectId::<AtomTag>::from_bytes(id);
            atom_ids.push(atom_id);
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(PayloadDescriptor::new(
                    content_id,
                    (channel.len() * 8) as u64,
                    ElementType::I64,
                    ByteOrder::Little,
                    vec![1, channel.len() as u64],
                    Layout::DenseRowMajor,
                    None,
                    None,
                )),
                TimeAxis::Regular(
                    TimeSegment::new(
                        Rational::new(starts[index], 1).unwrap(),
                        Rational::new(250, 1).unwrap(),
                        channel.len() as u64,
                    )
                    .unwrap(),
                ),
                calibration.clone(),
            )));
        }
        (draft, access, atom_ids, recording_id, stream_id)
    }

    fn id<T>(byte: u8) -> ObjectId<T> {
        ObjectId::from_bytes([byte; 16])
    }

    fn concept(value: &str) -> ConceptId {
        ConceptId::new(value).unwrap()
    }

    fn calibration(scale: (i128, i128), offset: (i128, i128), unit: &str) -> Calibration {
        Calibration::new(
            Rational::new(scale.0, scale.1).unwrap(),
            Rational::new(offset.0, offset.1).unwrap(),
            concept(unit),
        )
        .unwrap()
    }

    struct DomainBackend {
        domain: SignalDomain,
        encoded: core::cell::RefCell<Option<NeuralSignal>>,
        decoded: NeuralSignal,
    }

    impl DomainBackend {
        fn new(domain: SignalDomain, decoded: Vec<Vec<i64>>) -> Self {
            Self {
                domain,
                encoded: core::cell::RefCell::new(None),
                decoded: NeuralSignal {
                    domain,
                    channels: decoded,
                },
            }
        }
    }

    impl NeuralBackend for DomainBackend {
        fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
            let mut capabilities = StubBackend::default().capabilities();
            capabilities.signal_domain = self.domain;
            capabilities
        }

        fn model(&self) -> BackendModel<'_> {
            BackendModel::ModelFree(stub_model_provenance())
        }

        fn encode(
            &self,
            signal: &NeuralSignal,
            _sample_rate: Rational,
        ) -> Result<NeuralTokens, BackendError> {
            self.encoded.replace(Some(signal.clone()));
            let n_channels = u16::try_from(signal.channels.len()).unwrap();
            let n_samples = u32::try_from(signal.channels[0].len()).unwrap();
            Ok(NeuralTokens {
                tokens: vec![0; usize::from(n_channels) * n_samples as usize],
                schedule: vec![5; n_samples as usize],
                alphabet: 5,
                n_channels,
                n_samples,
                backend_meta: Vec::new(),
            })
        }

        fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
            Ok(self.decoded.clone())
        }
    }

    struct ContractStubBackend {
        artifact: TrainedModelArtifact,
    }

    impl ContractStubBackend {
        fn new(contract: ModelInputContract) -> Self {
            Self {
                artifact: stub_model_artifact(&contract),
            }
        }
    }

    impl NeuralBackend for ContractStubBackend {
        fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
            StubBackend::default().capabilities()
        }

        fn model(&self) -> BackendModel<'_> {
            BackendModel::trained(&self.artifact)
        }

        fn encode(
            &self,
            signal: &NeuralSignal,
            sample_rate: Rational,
        ) -> Result<NeuralTokens, BackendError> {
            StubBackend::default().encode(signal, sample_rate)
        }

        fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
            StubBackend::default().decode(tokens)
        }
    }

    fn calibrated_fixture() -> OpenedDataset<InMemoryPayloadAccess> {
        let calibration = calibration((2, 1), (1, 1), "ucum:uV");
        let (mut draft, access, atom_ids, recording_id, stream_id) =
            fixture_parts_with_calibration(&[0, 0, 0, 0], Some(calibration));
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            ConceptId::new("abir:modality/eeg").unwrap(),
            atom_ids,
            None,
            None,
            None,
        ));
        OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
    }

    fn test_input_contract(
        dataset: &AbirDataset,
        channel_concepts: Vec<ConceptId>,
    ) -> ModelInputContract {
        ModelInputContract::new(
            concept("abir:modality/eeg"),
            channel_concepts,
            model_channel_basis_content_id(dataset).unwrap(),
            Rational::new(250, 1).unwrap(),
            500,
            SignalDomain::DigitalInteger,
            concept("lamquant:operation/model-input-v1"),
            concept("lamquant:proof/model-input-v1"),
            concept("lamquant:backend-pipeline/test-v1"),
        )
        .unwrap()
    }

    fn contract_fixture(
        channel_concepts: Vec<ConceptId>,
        reference: ReferenceKind,
    ) -> OpenedDataset<InMemoryPayloadAccess> {
        contract_fixture_with_identity(channel_concepts, reference, 83, 84, true)
    }

    fn contract_fixture_with_identity(
        channel_concepts: Vec<ConceptId>,
        reference: ReferenceKind,
        reference_byte: u8,
        source_start: u8,
        include_proof: bool,
    ) -> OpenedDataset<InMemoryPayloadAccess> {
        contract_fixture_with_identity_and_extra_derivation(
            channel_concepts,
            reference,
            reference_byte,
            source_start,
            include_proof,
            false,
            false,
        )
    }

    fn contract_fixture_with_identity_and_extra_derivation(
        channel_concepts: Vec<ConceptId>,
        reference: ReferenceKind,
        reference_byte: u8,
        source_start: u8,
        include_proof: bool,
        include_unrelated_derivation: bool,
        include_unrelated_duplicate_role: bool,
    ) -> OpenedDataset<InMemoryPayloadAccess> {
        let (mut draft, access, atom_ids, recording_id, stream_id) = fixture_parts(&[0, 0, 0, 0]);
        let basis_id = id::<ChannelBasisTag>(82);
        let reference_id = id::<ChannelTag>(reference_byte);
        draft.add_channel(Channel::new(
            reference_id,
            concept("lamquant:test-source/reference"),
        ));
        let mut vectors = Vec::with_capacity(channel_concepts.len());
        for (index, _) in channel_concepts.iter().enumerate() {
            let source_id = ObjectId::<ChannelTag>::from_bytes([source_start + index as u8; 16]);
            draft.add_channel(Channel::new(
                source_id,
                concept(&format!("lamquant:test-source/{index}")),
            ));
            vectors.push(
                ChannelBasisVector::new(vec![
                    ChannelBasisTerm::new(source_id, Rational::new(1, 1).unwrap()).unwrap(),
                    ChannelBasisTerm::new(reference_id, Rational::new(-1, 1).unwrap()).unwrap(),
                ])
                .unwrap(),
            );
        }
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_channel_basis(
            ChannelBasis::new(
                basis_id,
                channel_concepts.into_iter().map(ChannelSpec::new).collect(),
                reference,
            )
            .with_construction(vectors)
            .unwrap(),
        );
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            concept("abir:modality/eeg"),
            atom_ids.clone(),
            None,
            Some(basis_id),
            None,
        ));
        let derivation_id = id::<DerivationTag>(90);
        draft.add_derivation(Derivation::new(
            derivation_id,
            concept("lamquant:operation/model-input-v1"),
            atom_ids.into_iter().map(SemanticRef::of).collect(),
            vec![SemanticRef::of(stream_id)],
        ));
        if include_proof {
            draft.add_proof(Proof::new(
                id::<ProofTag>(91),
                concept("lamquant:proof/model-input-v1"),
                SemanticRef::of(derivation_id),
                ContentId::from_bytes([92; 32]),
            ));
        }
        if include_unrelated_derivation {
            draft.add_derivation(Derivation::new(
                id::<DerivationTag>(93),
                concept("future:operation/unrelated-v1"),
                Vec::new(),
                vec![SemanticRef::of(stream_id)],
            ));
        }
        if include_unrelated_duplicate_role {
            draft.add_channel(Channel::new(
                id::<ChannelTag>(94),
                concept("lamquant:test-source/0"),
            ));
        }
        OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
    }

    #[test]
    fn model_channel_basis_identity_is_portable_across_abir_object_ids() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let first =
            contract_fixture_with_identity(channels.clone(), ReferenceKind::Common, 83, 84, true);
        // Reverse relative source/reference ID ordering. ABIR canonicalizes
        // terms by ObjectId, while LMQ contract identity must remain semantic.
        let second =
            contract_fixture_with_identity(channels, ReferenceKind::Common, 120, 100, true);
        assert_eq!(
            model_channel_basis_content_id(first.dataset()).unwrap(),
            model_channel_basis_content_id(second.dataset()).unwrap()
        );
    }

    #[test]
    fn unrelated_duplicate_source_role_does_not_change_model_basis_identity() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let baseline = contract_fixture_with_identity_and_extra_derivation(
            channels.clone(),
            ReferenceKind::Common,
            83,
            84,
            true,
            false,
            false,
        );
        let with_unrelated_duplicate = contract_fixture_with_identity_and_extra_derivation(
            channels,
            ReferenceKind::Common,
            83,
            84,
            true,
            false,
            true,
        );
        assert_eq!(
            model_channel_basis_content_id(baseline.dataset()).unwrap(),
            model_channel_basis_content_id(with_unrelated_duplicate.dataset()).unwrap()
        );
    }

    #[test]
    fn model_channel_basis_metadata_is_bounded_before_inference() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened = contract_fixture(channels.clone(), ReferenceKind::Common);
        let backend = ContractStubBackend::new(test_input_contract(opened.dataset(), channels));
        let channel_bounds = LmqResourceBounds {
            max_model_basis_channels: 1,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            encode_bundle_bounded(
                opened.dataset(),
                opened.access(),
                &backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                channel_bounds,
            ),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::ModelBasisChannels,
                ..
            })
        ));
        let term_bounds = LmqResourceBounds {
            max_model_basis_terms: 1,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            encode_bundle_bounded(
                opened.dataset(),
                opened.access(),
                &backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                term_bounds,
            ),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::ModelBasisTerms,
                ..
            })
        ));
        for (bounds, resource) in [
            (
                LmqResourceBounds {
                    max_model_derivations: 0,
                    ..LmqResourceBounds::default()
                },
                LmqResource::ModelDerivations,
            ),
            (
                LmqResourceBounds {
                    max_model_claims: 0,
                    ..LmqResourceBounds::default()
                },
                LmqResource::ModelClaims,
            ),
            (
                LmqResourceBounds {
                    max_model_derivation_output_edges: 0,
                    ..LmqResourceBounds::default()
                },
                LmqResource::ModelDerivationOutputEdges,
            ),
        ] {
            assert!(matches!(
                encode_bundle_bounded(
                    opened.dataset(),
                    opened.access(),
                    &backend,
                    transformed_fidelity("test-residue"),
                    implementation_identity("test-build"),
                    bounds,
                ),
                Err(LmqError::ResourceLimit {
                    resource: actual,
                    ..
                }) if actual == resource
            ));
        }
    }

    #[test]
    fn calibrated_model_domain_round_trips_through_public_shell() {
        let opened = calibrated_fixture();
        let decoded_q16 = (0..4)
            .map(|_| {
                (0..500)
                    .map(|sample| if sample % 2 == 0 { 655_360 } else { 786_432 })
                    .collect()
            })
            .collect();
        let backend = DomainBackend::new(SignalDomain::PhysicalMicrovoltQ16, decoded_q16);

        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("calibrated-test"),
            implementation_identity("calibrated-test"),
            ResourceBounds::default(),
        )
        .unwrap();
        let encoded = backend.encoded.borrow();
        let encoded = encoded.as_ref().unwrap();
        assert_eq!(encoded.domain, SignalDomain::PhysicalMicrovoltQ16);
        assert_eq!(encoded.channels[0][0], -2_555_904);
        assert_eq!(encoded.channels[1][0], -1_638_400);

        let decoded = open_bundle(&bytes, &backend, ResourceBounds::default()).unwrap();
        for atom_id in decoded.reconstructed().dataset().streams()[0].atoms() {
            let block = decoded.reconstructed().block_view(*atom_id).unwrap();
            let samples = block
                .bytes()
                .chunks_exact(8)
                .map(|sample| i64::from_le_bytes(sample.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(&samples[..4], &[4, 6, 4, 6]);
            let atom = decoded
                .reconstructed()
                .dataset()
                .atoms()
                .iter()
                .find(|atom| atom.id() == *atom_id)
                .unwrap();
            let Atom::SignalBlock(block) = atom else {
                panic!("reconstruction must retain signal blocks");
            };
            assert_eq!(
                block.calibration(),
                Some(&calibration((2, 1), (1, 1), "ucum:uV"))
            );
        }
    }

    #[test]
    fn catalog_binds_backend_signal_domain() {
        let opened = calibrated_fixture();
        let physical =
            DomainBackend::new(SignalDomain::PhysicalMicrovoltQ16, vec![vec![0; 500]; 4]);
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &physical,
            transformed_fidelity("calibrated-test"),
            implementation_identity("calibrated-test"),
            ResourceBounds::default(),
        )
        .unwrap();
        let digital = DomainBackend::new(SignalDomain::DigitalInteger, vec![vec![0; 500]; 4]);

        assert!(matches!(
            open_bundle(&bytes, &digital, ResourceBounds::default()),
            Err(LmqError::CatalogContract)
        ));
    }

    fn rich_fixture() -> OpenedDataset<InMemoryPayloadAccess> {
        let (mut draft, access, atom_ids, recording_id, stream_id) = fixture_parts(&[0, 0, 0, 0]);
        let subject_id = id::<SubjectTag>(20);
        let patient_id = id::<PatientTag>(21);
        let session_id = id::<SessionTag>(22);
        let acquisition_id = id::<AcquisitionTag>(23);
        let device_id = id::<DeviceTag>(24);
        let sensor_id = id::<SensorTag>(25);
        let channel_id = id::<ChannelTag>(26);
        let clock_id = id::<ClockTag>(27);
        let reference_clock_id = id::<ClockTag>(41);
        let frame_a = id::<CoordinateFrameTag>(28);
        let frame_b = id::<CoordinateFrameTag>(29);
        let basis_id = id::<ChannelBasisTag>(30);
        let policy_id = id::<PolicyTag>(31);
        let safe_proof_id = id::<ProofTag>(32);
        let unsafe_proof_id = id::<ProofTag>(33);
        let safe_derivation_id = id::<DerivationTag>(34);
        let unsafe_derivation_id = id::<DerivationTag>(35);
        let safe_artifact_id = id::<DerivedArtifactTag>(36);
        let unsafe_artifact_id = id::<DerivedArtifactTag>(37);

        draft.add_subject(
            Subject::new(subject_id, concept("abir:subject/human"))
                .with_source_key(SourceKey::new("bids.subject", "sub-01").unwrap()),
        );
        draft.add_patient(Patient::new(patient_id, concept("abir:patient/clinical")));
        draft.add_session(Session::new(session_id, concept("abir:session/recording")));
        draft.add_acquisition(Acquisition::new(
            acquisition_id,
            concept("abir:acquisition/eeg"),
        ));
        draft.add_device(Device::new(device_id, concept("abir:device/amplifier")));
        draft.add_sensor(Sensor::new(sensor_id, concept("abir:sensor/electrode")));
        draft.add_channel(Channel::new(channel_id, concept("eeg:channel/fp1")));
        draft.add_concept_dictionary(ConceptDictionary::new(
            id::<ConceptDictionaryTag>(38),
            concept("abir:dictionary/semantic-v1"),
        ));
        draft.add_clock(Clock::new(
            clock_id,
            concept("abir:clock/device"),
            None,
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 1_000_000).unwrap(),
        ));
        draft.add_clock(Clock::new(
            reference_clock_id,
            concept("abir:clock/reference"),
            None,
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 10_000_000).unwrap(),
        ));
        draft.add_clock_relation(ClockRelation::new(
            id::<ClockRelationTag>(42),
            clock_id,
            reference_clock_id,
            Rational::new(1, 1_000).unwrap(),
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 1_000_000).unwrap(),
            concept("abir:clock-relation/measured"),
            Rational::new(0, 1).unwrap(),
            Some(Rational::new(10, 1).unwrap()),
            ContentId::from_bytes([46; 32]),
        ));
        let identity = [
            ExactNumber::Integer(1),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(1),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(1),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(0),
            ExactNumber::Integer(1),
        ];
        draft.add_coordinate_frame(CoordinateFrame::new(
            frame_a,
            concept("abir:frame/head"),
            None,
            Some(identity),
            Rational::new(1, 1_000).unwrap(),
        ));
        draft.add_coordinate_frame(CoordinateFrame::new(
            frame_b,
            concept("abir:frame/sensor"),
            Some(frame_a),
            Some(identity),
            Rational::new(1, 10_000).unwrap(),
        ));
        draft.add_frame_transform(FrameTransform::new(
            id::<FrameTransformTag>(39),
            frame_b,
            frame_a,
            identity,
            Rational::new(1, 1_000).unwrap(),
            concept("abir:frame-transform/measured"),
        ));
        draft.add_channel_basis(ChannelBasis::new(
            basis_id,
            (0..4)
                .map(|index| {
                    ChannelSpec::new(concept(&format!("eeg:channel/test-{index}")))
                        .with_coordinate_frame(frame_a)
                })
                .collect(),
            ReferenceKind::Differential,
        ));
        draft.add_policy(Policy::new(
            policy_id,
            None,
            vec![concept("abir:policy/research-only")],
        ));
        draft.add_event(Event::new(
            id::<EventTag>(40),
            concept("abir:event/stimulus"),
            clock_id,
            Rational::new(1, 2).unwrap(),
            Rational::new(3, 4).unwrap(),
            Rational::new(1, 1_000).unwrap(),
        ));

        draft.add_source_relationship(SourceRelationship::PatientSubject {
            patient_id,
            subject_id,
        });
        draft.add_source_relationship(SourceRelationship::SessionPatient {
            session_id,
            patient_id,
        });
        draft.add_source_relationship(SourceRelationship::AcquisitionSession {
            acquisition_id,
            session_id,
        });
        draft.add_source_relationship(SourceRelationship::AcquisitionDevice {
            acquisition_id,
            device_id,
        });
        draft.add_source_relationship(SourceRelationship::DeviceSensor {
            device_id,
            sensor_id,
        });
        draft.add_source_relationship(SourceRelationship::SensorChannel {
            sensor_id,
            channel_id,
        });
        draft.add_source_relationship(SourceRelationship::AcquisitionRecording {
            acquisition_id,
            recording_id,
        });
        draft.add_source_relationship(SourceRelationship::ChannelBasisMember {
            channel_id,
            basis_id,
            position: 0,
        });

        draft.add_proof(Proof::new(
            safe_proof_id,
            concept("abir:proof/policy-attestation"),
            SemanticRef::of(policy_id),
            ContentId::from_bytes([41; 32]),
        ));
        draft.add_proof(Proof::new(
            unsafe_proof_id,
            concept("future:proof/source-signal"),
            SemanticRef::of(atom_ids[0]),
            ContentId::from_bytes([42; 32]),
        ));
        draft.add_derivation(Derivation::new(
            safe_derivation_id,
            concept("future:operation/context-derive"),
            vec![SemanticRef::of(policy_id)],
            vec![SemanticRef::of(safe_artifact_id)],
        ));
        draft.add_derived_artifact(DerivedArtifact::new(
            safe_artifact_id,
            ContentId::from_bytes([43; 32]),
            safe_derivation_id,
        ));
        draft.add_derivation(Derivation::new(
            unsafe_derivation_id,
            concept("future:operation/signal-derive"),
            vec![SemanticRef::of(atom_ids[0])],
            vec![SemanticRef::of(unsafe_artifact_id)],
        ));
        draft.add_derived_artifact(DerivedArtifact::new(
            unsafe_artifact_id,
            ContentId::from_bytes([44; 32]),
            unsafe_derivation_id,
        ));
        draft.add_fidelity(Fidelity::new(
            SemanticRef::of(policy_id),
            FidelityKind::Exact,
            None,
            None,
        ));
        draft.add_fidelity(Fidelity::new(
            SemanticRef::of(atom_ids[0]),
            FidelityKind::Exact,
            None,
            None,
        ));
        draft.add_source_capsule(SourceCapsule::new(
            SourceKey::new("nwb.object", "acquisition/eeg").unwrap(),
            ContentId::from_bytes([45; 32]),
            Some("application/x-hdf5"),
        ));
        draft.add_observed_execution(ExecutionRecord::new(
            concept("future:operation/validate"),
            "rich-fixture",
        ));

        let mut recording = Recording::new(recording_id, vec![stream_id]);
        recording.add_source_key(SourceKey::new("edf.recording", "fixture").unwrap());
        draft.add_recording(recording);
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            concept("abir:modality/eeg"),
            atom_ids,
            Some(clock_id),
            Some(basis_id),
            Some(policy_id),
        ));
        OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access)
    }

    #[test]
    fn unaligned_channel_origins_fail_before_backend_inference() {
        let opened = fixture_with_starts(&[0, 0, 1, 0]);
        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &StubBackend::default(),
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::UnsupportedSemantics(_))
        ));
    }

    #[test]
    fn duplicate_stream_atom_membership_is_rejected() {
        let (mut draft, access, mut atom_ids, recording_id, stream_id) =
            fixture_parts(&[0, 0, 0, 0]);
        atom_ids[1] = atom_ids[0];
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            concept("abir:modality/eeg"),
            atom_ids,
            None,
            None,
            None,
        ));
        let opened =
            OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access);

        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &StubBackend::default(),
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::UnsupportedSemantics(
                "stream atom membership is not bijective"
            ))
        ));
    }

    #[test]
    fn backend_input_contract_fails_before_payload_or_inference() {
        struct ContractRejectingBackend;
        impl NeuralBackend for ContractRejectingBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                let mut capabilities = StubBackend::default().capabilities();
                capabilities.minimum_channels = 21;
                capabilities.maximum_channels = 21;
                capabilities
            }

            fn model(&self) -> BackendModel<'_> {
                BackendModel::ModelFree(stub_model_provenance())
            }

            fn encode(
                &self,
                _signal: &NeuralSignal,
                _sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                panic!("backend must not run after capability rejection")
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                unreachable!()
            }
        }

        let opened = fixture();
        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &ContractRejectingBackend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::BackendCapability(
                BackendCapabilityError::ChannelCount
            ))
        ));
    }

    #[test]
    fn semantic_model_input_contract_fails_before_payload_or_inference() {
        struct PanicLease;
        impl PayloadLease for PanicLease {
            fn bytes(&self) -> &[u8] {
                unreachable!()
            }
        }
        struct PanicAccess;
        impl PayloadAccess for PanicAccess {
            type Lease<'a>
                = PanicLease
            where
                Self: 'a;

            fn lease<'a>(
                &'a self,
                _descriptor: &PayloadDescriptor,
            ) -> Result<Self::Lease<'a>, semantic_abir::PayloadAccessError> {
                panic!("payload lease must not run after model-contract rejection")
            }
        }
        let expected = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let canonical = contract_fixture(expected.clone(), ReferenceKind::Common);
        let contract = test_input_contract(canonical.dataset(), expected.clone());
        let mut wrong_order = expected.clone();
        wrong_order.swap(1, 2);
        let opened = contract_fixture(wrong_order, ReferenceKind::Common);
        let backend = ContractStubBackend::new(contract);

        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                &PanicAccess,
                &backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::ModelInputContract(
                ModelInputContractError::ChannelOrder
            ))
        ));
        let differential = contract_fixture(expected, ReferenceKind::Differential);
        assert!(matches!(
            encode_bundle(
                differential.dataset(),
                &PanicAccess,
                &backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::ModelInputContract(
                ModelInputContractError::ChannelBasis
            ))
        ));
        let one_channel = vec![concept("lamquant:test-channel/c0")];
        let underspecified = contract_fixture(one_channel.clone(), ReferenceKind::Common);
        let underspecified_backend =
            ContractStubBackend::new(test_input_contract(underspecified.dataset(), one_channel));
        assert!(matches!(
            encode_bundle(
                underspecified.dataset(),
                &PanicAccess,
                &underspecified_backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::ModelInputContract(
                ModelInputContractError::ChannelCount
            ))
        ));
    }

    #[test]
    fn semantic_model_input_contract_requires_upstream_derivation_claim() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened =
            contract_fixture_with_identity(channels.clone(), ReferenceKind::Common, 83, 84, false);
        let backend = ContractStubBackend::new(test_input_contract(opened.dataset(), channels));

        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::ModelInputContract(
                ModelInputContractError::MissingDerivationClaim
            ))
        ));
    }

    #[test]
    fn semantic_model_input_contract_allows_unrelated_derivations() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened = contract_fixture_with_identity_and_extra_derivation(
            channels.clone(),
            ReferenceKind::Common,
            83,
            84,
            true,
            true,
            false,
        );
        let backend = ContractStubBackend::new(test_input_contract(opened.dataset(), channels));

        encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
    }

    #[test]
    fn catalog_binds_model_input_contract_identity() {
        let expected = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened = contract_fixture(expected.clone(), ReferenceKind::Common);
        let backend =
            ContractStubBackend::new(test_input_contract(opened.dataset(), expected.clone()));
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        let mut wrong_order = expected;
        wrong_order.swap(1, 2);
        let wrong = ContractStubBackend::new(test_input_contract(opened.dataset(), wrong_order));
        assert!(matches!(
            open_bundle(&bytes, &wrong, ResourceBounds::default()),
            Err(LmqError::CatalogContract)
        ));
    }

    #[test]
    fn trained_catalog_dual_reads_legacy_but_single_writes_contract_identity() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened = contract_fixture(channels.clone(), ReferenceKind::Common);
        let contract = test_input_contract(opened.dataset(), channels);
        let artifact = stub_model_artifact(&contract);
        let semantics = BackendModel::trained(&artifact);
        let current = canonical_parameters(SignalDomain::DigitalInteger, Some(&artifact));
        let legacy = canonical_parameters(SignalDomain::DigitalInteger, None);

        assert!(catalog_parameters_supported(
            &current,
            SignalDomain::DigitalInteger,
            &semantics,
            LegacyModelContractPolicy::Reject,
        ));
        assert!(!catalog_parameters_supported(
            &legacy,
            SignalDomain::DigitalInteger,
            &semantics,
            LegacyModelContractPolicy::Reject,
        ));
        assert!(catalog_parameters_supported(
            &legacy,
            SignalDomain::DigitalInteger,
            &semantics,
            LegacyModelContractPolicy::AllowPreContractCatalog,
        ));
        assert_ne!(current, legacy);
    }

    #[test]
    fn explicit_legacy_open_revalidates_complete_model_contract() {
        let channels = ["c0", "c1", "c2", "c3"]
            .map(|name| concept(&format!("lamquant:test-channel/{name}")))
            .to_vec();
        let opened = contract_fixture(channels.clone(), ReferenceKind::Common);
        let backend =
            ContractStubBackend::new(test_input_contract(opened.dataset(), channels.clone()));
        let current = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        let current = CodecBundleView::open(&current, ResourceBounds::default()).unwrap();
        let packets = [current.packet(0).unwrap()];
        let legacy = encode_codec_bundle(
            CodecBundleInput {
                required_capabilities: 0,
                canonical_semantics: current.canonical_semantics(),
                fidelity: current.catalog().fidelity().clone(),
                implementation: current.catalog().implementation().clone(),
                model_provenance: current.catalog().model_provenance().cloned(),
                packets: &packets,
                parameters: canonical_parameters(SignalDomain::DigitalInteger, None),
                profile: CodecProfile::LmqProgressive,
            },
            ResourceBounds::default(),
        )
        .unwrap();

        assert!(matches!(
            open_bundle(&legacy, &backend, ResourceBounds::default()),
            Err(LmqError::CatalogContract)
        ));
        open_bundle_with_legacy_contract(&legacy, &backend, ResourceBounds::default()).unwrap();

        let mut wrong_order = channels;
        wrong_order.swap(1, 2);
        let wrong = contract_fixture(wrong_order, ReferenceKind::Common);
        let wrong_semantics = canonical_debug_json(wrong.dataset()).unwrap();
        let invalid_legacy = encode_codec_bundle(
            CodecBundleInput {
                required_capabilities: 0,
                canonical_semantics: &wrong_semantics,
                fidelity: current.catalog().fidelity().clone(),
                implementation: current.catalog().implementation().clone(),
                model_provenance: current.catalog().model_provenance().cloned(),
                packets: &packets,
                parameters: canonical_parameters(SignalDomain::DigitalInteger, None),
                profile: CodecProfile::LmqProgressive,
            },
            ResourceBounds::default(),
        )
        .unwrap();
        assert!(matches!(
            open_bundle_with_legacy_contract(&invalid_legacy, &backend, ResourceBounds::default()),
            Err(LmqError::ModelInputContract(
                ModelInputContractError::ChannelOrder
            ))
        ));
    }

    #[test]
    fn physical_model_domain_requires_calibration_before_payload_or_inference() {
        struct PhysicalBackend;
        impl NeuralBackend for PhysicalBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                let mut capabilities = StubBackend::default().capabilities();
                capabilities.signal_domain = SignalDomain::PhysicalMicrovoltQ16;
                capabilities
            }

            fn model(&self) -> BackendModel<'_> {
                BackendModel::ModelFree(stub_model_provenance())
            }

            fn encode(
                &self,
                _signal: &NeuralSignal,
                _sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                panic!("backend must not run without calibrated input")
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                unreachable!()
            }
        }

        let opened = fixture();
        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &PhysicalBackend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::UnsupportedSemantics(
                "physical model domain requires exact per-channel calibration"
            ))
        ));
    }

    #[test]
    fn implementation_identity_tracks_linked_abir_while_wire_catalog_stays_frozen() {
        assert_ne!(LINKED_ABIR_REVISION, LMQ_WIRE_ABIR_REVISION);

        let mut linked_hasher = blake3::Hasher::new();
        linked_hasher.update(b"org.quitetall.lamquant.lmq.implementation-v1\0");
        linked_hasher.update(LINKED_ABIR_REVISION.as_bytes());
        linked_hasher.update(LMQ_KERNEL_ID.as_bytes());
        let linked_id = ContentId::from_bytes(*linked_hasher.finalize().as_bytes());
        assert_eq!(
            implementation_identity("same-build").implementation_id,
            linked_id
        );

        let parameters = canonical_parameters(SignalDomain::DigitalInteger, None);
        assert!(matches!(
            &parameters[0].value,
            CodecParameterValue::Text { value }
                if value == LMQ_WIRE_ABIR_REVISION
        ));
    }

    #[test]
    fn shell_uses_bcs2_and_round_trips_lossy_reconstruction() {
        let opened = fixture();
        let backend = StubBackend { alphabet: 5 };
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        assert!(bytes.starts_with(&BCS2_MAGIC));
        let decoded = open_bundle(&bytes, &backend, ResourceBounds::default()).unwrap();
        let reconstructed = decoded.reconstructed();
        assert_eq!(reconstructed.dataset().atoms().len(), 5);
        for atom_id in reconstructed.dataset().streams()[0].atoms() {
            let block = reconstructed.block_view(*atom_id).unwrap();
            assert!(block.bytes().chunks_exact(8).all(|sample| {
                (0..5).contains(&i64::from_le_bytes(sample.try_into().unwrap()))
            }));
            verify_payload_content(block.descriptor(), block.bytes()).unwrap();
        }
        assert_ne!(
            reconstructed.dataset().payload_content_ids(),
            opened.dataset().payload_content_ids()
        );
        assert_eq!(
            canonical_debug_json(decoded.source_dataset()).unwrap(),
            canonical_debug_json(opened.dataset()).unwrap()
        );
    }

    #[test]
    fn reconstruction_preserves_context_and_invalidates_signal_claims() {
        let opened = rich_fixture();
        let backend = StubBackend { alphabet: 5 };
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        let decoded = open_bundle(&bytes, &backend, ResourceBounds::default()).unwrap();
        let source = decoded.source_dataset();
        let reconstructed = decoded.reconstructed().dataset();

        assert_eq!(reconstructed.clocks(), source.clocks());
        assert_eq!(
            reconstructed.coordinate_frames(),
            source.coordinate_frames()
        );
        assert_eq!(reconstructed.channel_bases(), source.channel_bases());
        assert_eq!(reconstructed.policies(), source.policies());
        assert_eq!(reconstructed.subjects(), source.subjects());
        assert_eq!(reconstructed.patients(), source.patients());
        assert_eq!(reconstructed.sessions(), source.sessions());
        assert_eq!(reconstructed.acquisitions(), source.acquisitions());
        assert_eq!(reconstructed.devices(), source.devices());
        assert_eq!(reconstructed.sensors(), source.sensors());
        assert_eq!(reconstructed.channels(), source.channels());
        assert_eq!(source.clock_relations().len(), 1);
        assert!(reconstructed.clock_relations().is_empty());
        assert_eq!(reconstructed.frame_transforms(), source.frame_transforms());
        assert_eq!(reconstructed.events(), source.events());
        assert_eq!(
            reconstructed.concept_dictionaries(),
            source.concept_dictionaries()
        );

        let source_stream = &source.streams()[0];
        let reconstructed_stream = &reconstructed.streams()[0];
        assert_eq!(reconstructed_stream.clock_id(), source_stream.clock_id());
        assert_eq!(
            reconstructed_stream.channel_basis_id(),
            source_stream.channel_basis_id()
        );
        assert_eq!(reconstructed_stream.policy_id(), source_stream.policy_id());
        assert_eq!(
            reconstructed.recordings()[0].source_keys(),
            source.recordings()[0].source_keys()
        );
        assert!(reconstructed
            .source_relationships()
            .iter()
            .any(|relationship| matches!(
                relationship,
                SourceRelationship::AcquisitionRecording { recording_id, .. }
                    if *recording_id == reconstructed.recordings()[0].id()
            )));
        assert!(!reconstructed
            .source_relationships()
            .iter()
            .any(|relationship| matches!(
                relationship,
                SourceRelationship::AcquisitionRecording { recording_id, .. }
                    if *recording_id == source.recordings()[0].id()
            )));

        assert_eq!(source.proofs().len(), 2);
        assert!(reconstructed.proofs().is_empty());
        assert_eq!(source.derivations().len(), 2);
        assert!(reconstructed.derivations().is_empty());
        assert_eq!(source.derived_artifacts().len(), 2);
        assert!(reconstructed.derived_artifacts().is_empty());

        assert_eq!(source.fidelity().len(), 2);
        assert_eq!(reconstructed.fidelity().len(), 2);
        assert!(reconstructed.fidelity().iter().any(|statement| {
            statement.subject() == SemanticRef::of(id::<PolicyTag>(31))
                && statement.kind() == FidelityKind::Exact
        }));
        assert!(reconstructed.fidelity().iter().any(|statement| {
            statement.subject() == SemanticRef::of(reconstructed.id())
                && statement.kind() == FidelityKind::Transformed
                && statement
                    .metric()
                    .is_some_and(|metric| metric.as_str() == "lamquant:metric/test-residue")
        }));
        assert!(!reconstructed
            .fidelity()
            .iter()
            .any(|statement| { statement.subject() == SemanticRef::of(source.atoms()[0].id()) }));

        assert_eq!(reconstructed.source_capsules().len(), 1);
        let receipt = &reconstructed.source_capsules()[0];
        assert_eq!(
            receipt.media_type(),
            Some("application/vnd.quitetall.lamquant.lmq-reconstruction-receipt-v1")
        );
        let receipt_descriptor = reconstructed
            .atoms()
            .iter()
            .filter_map(Atom::payload)
            .find(|descriptor| descriptor.content_id() == receipt.content_id())
            .unwrap();
        let receipt_bytes = decoded
            .reconstructed()
            .access()
            .lease(receipt_descriptor)
            .unwrap();
        let receipt_text = core::str::from_utf8(receipt_bytes.bytes()).unwrap();
        assert!(receipt_text.starts_with("LMQ-RECONSTRUCTION-PROJECTION-V1\n"));
        assert!(receipt_text.contains("pccp-status=candidate\n"));
        assert!(receipt_text.contains(
            "checkpoint-sha256=5252525252525252525252525252525252525252525252525252525252525252\n"
        ));
        assert!(receipt_text.contains(
            "invalidated-proofs=20202020202020202020202020202020,21212121212121212121212121212121\n"
        ));
        assert!(
            receipt_text.contains("invalidated-clock-relations=2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a2a\n")
        );
        assert!(receipt_text.contains(
            "invalidated-derived-artifacts=24242424242424242424242424242424,25252525252525252525252525252525\n"
        ));
        assert!(receipt_text.contains("invalidated-fidelity-subjects=atom:"));
        for descriptor in reconstructed.atoms().iter().filter_map(Atom::payload) {
            assert!(decoded.reconstructed().access().lease(descriptor).is_ok());
        }
        assert_eq!(
            reconstructed.observed_execution().len(),
            source.observed_execution().len() + 1
        );
        let execution = reconstructed.observed_execution().last().unwrap();
        assert_eq!(
            execution.operation().as_str(),
            "lamquant:operation/lmq-decode"
        );
        assert_eq!(
            execution.implementation(),
            "org.quitetall.lamquant.lmq.fsq-rans-v1@test-build"
        );
    }

    #[test]
    fn codec_fidelity_mapping_is_exact_and_fail_closed() {
        let dataset_id = id::<DatasetTag>(90);
        let bounded = CodecFidelity {
            bound: Some(CodecParameterValue::Rational {
                denominator: "1000".to_string(),
                numerator: "75".to_string(),
            }),
            contract_id: ContentId::from_bytes([91; 32]),
            kind: CodecFidelityKind::Bounded,
            metric: Some("prd".to_string()),
        };
        let statement = codec_fidelity_statement(dataset_id, &bounded).unwrap();
        assert_eq!(statement.subject(), SemanticRef::of(dataset_id));
        assert_eq!(statement.kind(), FidelityKind::Bounded);
        assert_eq!(
            statement.metric().map(ConceptId::as_str),
            Some("lamquant:metric/prd")
        );
        assert_eq!(
            statement.bound(),
            Some(ExactNumber::Rational(Rational::new(3, 40).unwrap()))
        );

        let invalid_bound = CodecFidelity {
            bound: Some(CodecParameterValue::Text {
                value: "approximately-small".to_string(),
            }),
            ..bounded.clone()
        };
        assert!(matches!(
            codec_fidelity_statement(dataset_id, &invalid_bound),
            Err(LmqError::CatalogContract)
        ));
        let opened = fixture();
        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &StubBackend::default(),
                invalid_bound,
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::CatalogContract)
        ));
        let invalid_metric = CodecFidelity {
            metric: Some("not a canonical metric".to_string()),
            ..bounded
        };
        assert!(matches!(
            codec_fidelity_statement(dataset_id, &invalid_metric),
            Err(LmqError::CatalogContract)
        ));
    }

    #[test]
    fn pccp_evidence_state_changes_reconstruction_identity() {
        struct StatusBackend(PccpStatus);

        impl NeuralBackend for StatusBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                StubBackend::default().capabilities()
            }

            fn model(&self) -> BackendModel<'_> {
                let mut provenance = stub_model_provenance();
                provenance.pccp_status = self.0;
                BackendModel::ModelFree(provenance)
            }

            fn encode(
                &self,
                signal: &NeuralSignal,
                sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                StubBackend::default().encode(signal, sample_rate)
            }

            fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                StubBackend::default().decode(tokens)
            }
        }

        let opened = fixture();
        let build = |backend: &StatusBackend| {
            let bytes = encode_bundle(
                opened.dataset(),
                opened.access(),
                backend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                ResourceBounds::default(),
            )
            .unwrap();
            open_bundle(&bytes, backend, ResourceBounds::default())
                .unwrap()
                .reconstructed()
                .dataset()
                .source_capsules()[0]
                .content_id()
        };

        let candidate = build(&StatusBackend(PccpStatus::Candidate));
        let rejected = build(&StatusBackend(PccpStatus::Rejected));
        assert_ne!(candidate, rejected);
    }

    #[test]
    fn wrong_model_and_exact_fidelity_fail_closed() {
        let opened = fixture();
        let backend = StubBackend::default();
        let exact = CodecFidelity {
            bound: None,
            contract_id: ContentId::from_bytes([10; 32]),
            kind: CodecFidelityKind::Exact,
            metric: None,
        };
        assert!(matches!(
            encode_bundle(
                opened.dataset(),
                opened.access(),
                &backend,
                exact,
                implementation_identity("test-build"),
                ResourceBounds::default(),
            ),
            Err(LmqError::CatalogContract)
        ));
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        struct WrongModelBackend(StubBackend);
        impl NeuralBackend for WrongModelBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                self.0.capabilities()
            }

            fn model(&self) -> BackendModel<'_> {
                let mut provenance = match self.0.model() {
                    BackendModel::ModelFree(provenance) => provenance,
                    BackendModel::Trained(_) => unreachable!(),
                };
                provenance.checkpoint_sha256 = [11; 32];
                BackendModel::ModelFree(provenance)
            }

            fn encode(
                &self,
                signal: &NeuralSignal,
                sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                self.0.encode(signal, sample_rate)
            }

            fn decode(&self, tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                self.0.decode(tokens)
            }
        }
        let wrong = WrongModelBackend(StubBackend::default());
        assert!(matches!(
            open_bundle(&bytes, &wrong, ResourceBounds::default()),
            Err(LmqError::CatalogContract)
        ));
    }

    #[test]
    fn packet_corruption_and_shape_mismatch_fail_closed() {
        let opened = fixture();
        let backend = StubBackend::default();
        let mut bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert!(matches!(
            open_bundle(&bytes, &backend, ResourceBounds::default()),
            Err(LmqError::Bundle(_))
        ));
    }

    #[test]
    fn histogram_total_is_bounded_regardless_of_token_count() {
        let tokens = (0..5_000_000).map(|index| index % 5).collect::<Vec<_>>();
        let counts = histogram(&tokens, 5).unwrap();
        assert_eq!(
            counts.iter().map(|&count| i64::from(count)).sum::<i64>() as u64,
            RANS_MODEL_TOTAL
        );
        assert!(counts.iter().all(|&count| count >= 1));
    }

    #[test]
    fn packet_allows_latent_density_independent_of_reconstruction_shape() {
        let tokens = NeuralTokens {
            tokens: vec![1, 2, 3],
            schedule: vec![5],
            alphabet: 5,
            n_channels: 4,
            n_samples: 500,
            backend_meta: vec![9, 8, 7],
        };
        let packet = encode_packet(&tokens).unwrap();
        assert_eq!(decode_packet(&packet).unwrap(), tokens);
    }

    #[test]
    fn encode_rejects_signal_before_payload_or_backend_work() {
        struct PanicLease;
        impl PayloadLease for PanicLease {
            fn bytes(&self) -> &[u8] {
                unreachable!()
            }
        }
        struct PanicAccess;
        impl PayloadAccess for PanicAccess {
            type Lease<'a>
                = PanicLease
            where
                Self: 'a;

            fn lease<'a>(
                &'a self,
                _descriptor: &PayloadDescriptor,
            ) -> Result<Self::Lease<'a>, semantic_abir::PayloadAccessError> {
                panic!("payload lease must not run after failed signal preflight")
            }
        }
        struct PanicBackend;
        impl NeuralBackend for PanicBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                StubBackend::default().capabilities()
            }

            fn model(&self) -> BackendModel<'_> {
                BackendModel::ModelFree(stub_model_provenance())
            }

            fn encode(
                &self,
                _signal: &NeuralSignal,
                _sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                panic!("backend must not run after failed signal preflight")
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                unreachable!()
            }
        }
        let opened = fixture();
        let bounds = LmqResourceBounds {
            max_signal_bytes: 1,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            encode_bundle_bounded(
                opened.dataset(),
                &PanicAccess,
                &PanicBackend,
                transformed_fidelity("test-residue"),
                implementation_identity("test-build"),
                bounds,
            ),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::SignalBytes,
                ..
            })
        ));
    }

    #[test]
    fn packet_limits_reject_backend_output_and_headers_before_copy() {
        let tokens = NeuralTokens {
            tokens: vec![1, 2, 3],
            schedule: vec![5, 5],
            alphabet: 5,
            n_channels: 1,
            n_samples: 1,
            backend_meta: vec![9, 8],
        };
        let token_bounds = LmqResourceBounds {
            max_tokens: 2,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            encode_packet_bounded(&tokens, token_bounds),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::TokenCount,
                ..
            })
        ));
        let alphabet_bounds = LmqResourceBounds {
            max_alphabet: 4,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            encode_packet_bounded(&tokens, alphabet_bounds),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::Alphabet,
                actual: 5,
                limit: 4,
            })
        ));
        let packet_bounds = LmqResourceBounds::from_bundle(ResourceBounds {
            max_frame_bytes: 32,
            ..ResourceBounds::default()
        });
        assert!(matches!(
            encode_packet_bounded(&tokens, packet_bounds),
            Err(LmqError::Body(BodyError::ResourceLimit {
                resource: crate::body::BodyResource::BodyBytes,
                ..
            }))
        ));

        let packet = encode_packet(&tokens).unwrap();
        let meta_bounds = LmqResourceBounds {
            max_backend_meta_bytes: 1,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            decode_packet_bounded(&packet, meta_bounds),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::BackendMetadataBytes,
                ..
            })
        ));
    }

    #[test]
    fn decode_rejects_reconstruction_before_backend_work() {
        let opened = fixture();
        let stub = StubBackend::default();
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &stub,
            transformed_fidelity("test-residue"),
            implementation_identity("test-build"),
            ResourceBounds::default(),
        )
        .unwrap();

        struct PanicDecodeBackend;
        impl NeuralBackend for PanicDecodeBackend {
            fn capabilities(&self) -> crate::backend::NeuralBackendCapabilities {
                StubBackend::default().capabilities()
            }

            fn model(&self) -> BackendModel<'_> {
                BackendModel::ModelFree(stub_model_provenance())
            }

            fn encode(
                &self,
                _signal: &NeuralSignal,
                _sample_rate: Rational,
            ) -> Result<NeuralTokens, BackendError> {
                unreachable!()
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<NeuralSignal, BackendError> {
                panic!("backend must not run after failed reconstruction preflight")
            }
        }
        let bounds = LmqResourceBounds {
            max_signal_bytes: 1,
            ..LmqResourceBounds::default()
        };
        assert!(matches!(
            open_bundle_bounded(&bytes, &PanicDecodeBackend, bounds),
            Err(LmqError::ResourceLimit {
                resource: LmqResource::SignalBytes,
                ..
            })
        ));
    }
}
