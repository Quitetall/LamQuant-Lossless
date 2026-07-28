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
    canonical_debug_json, parse_canonical_dataset, payload_content_id, verify_payload_content,
    AbirDataset, Atom, AtomTag, ByteOrder, ContentId, DatasetDraft, DatasetTag, ElementType,
    InMemoryPayloadAccess, Layout, ObjectId, OpenedDataset, PayloadAccess, PayloadDescriptor,
    PayloadLease, Presence, Recording, RecordingTag, SignalBlock, Stream, StreamTag, TimeAxis,
    ValidationLimits,
};
use semantic_abir_bcs::{
    encode_codec_bundle, CodecBundleError, CodecBundleInput, CodecBundleView, CodecFidelity,
    CodecFidelityKind, CodecImplementation, CodecParameter, CodecParameterValue, CodecProfile,
    ResourceBounds,
};

use crate::backend::{BackendError, NeuralBackend, NeuralTokens};
use crate::body::{decode_body_bounded, encode_body_bounded, BodyBounds, BodyError};

pub const LMQ_KERNEL_ID: &str = "org.quitetall.lamquant.lmq.fsq-rans-v1";
pub const LMQ_FIDELITY_CONTRACT: &str =
    "org.quitetall.lamquant.bcs2.lmq.explicit-nonexact-reconstruction-v1";
pub const RANS_MODEL_TOTAL: u64 = 4096;
const PACKET_MAGIC: &[u8; 4] = b"LMQP";
const PACKET_VERSION: u8 = 1;
const PACKET_HEADER_LEN: usize = 15;
const LMQ_WIRE_ABIR_REVISION: &str = "c101513167ad8d7cdefa6387b20c644fdaf66432";
const LINKED_ABIR_REVISION: &str = "a02ad44fa36899dcb7d53d95c9e640f17e885ffc";

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
    /// Also bounds the shell's temporary eight-byte-per-token I64 staging.
    pub max_tokens: u32,
    pub max_schedule_bytes: u32,
    pub max_backend_meta_bytes: u32,
    pub max_alphabet: u16,
    pub max_model_total: u32,
    /// Bounds allocations internal to the body codec. Shell token staging is
    /// governed separately by `max_tokens`.
    pub max_body_internal_working_bytes: u64,
}

impl LmqResourceBounds {
    pub const fn from_bundle(bundle: ResourceBounds) -> Self {
        Self {
            bundle,
            max_signal_bytes: bundle.max_frame_bytes as u64,
            max_tokens: lamquant_lml_mcu::rans::MAX_RANS_SYMBOLS as u32,
            max_schedule_bytes: bundle.max_frame_bytes,
            max_backend_meta_bytes: bundle.max_frame_bytes,
            max_alphabet: u16::MAX,
            max_model_total: RANS_MODEL_TOTAL as u32,
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
    TokenCount,
    Alphabet,
    ModelTotal,
    ScheduleBytes,
    BackendMetadataBytes,
    PacketBytes,
}

impl From<BodyError> for LmqError {
    fn from(error: BodyError) -> Self {
        Self::Body(error)
    }
}

impl fmt::Display for LmqError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(formatter, "LMQ backend failed: {}", error.0),
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
    if fidelity.kind == CodecFidelityKind::Exact || implementation.kernel_id != LMQ_KERNEL_ID {
        return Err(LmqError::CatalogContract);
    }
    let (signal, sample_rate) = read_signal(dataset, access, bounds.max_signal_bytes)?;
    // Ask before spending. The shape check below inspects what the backend
    // RETURNED; this one refuses a signal the backend already said it cannot
    // take, before a subprocess is spawned and a checkpoint is loaded to reach
    // the same conclusion the slow way.
    let channels = u16::try_from(signal.len()).map_err(|_| LmqError::SignalShapeMismatch)?;
    if !backend.capabilities().admits_channels(channels) {
        return Err(LmqError::SignalShapeMismatch);
    }
    let model = backend.model_provenance();
    let tokens = backend
        .encode(&signal, sample_rate)
        .map_err(LmqError::Backend)?;
    if usize::from(tokens.n_channels) != signal.len()
        || usize::try_from(tokens.n_samples).ok() != signal.first().map(Vec::len)
    {
        return Err(LmqError::SignalShapeMismatch);
    }
    let packet = encode_packet_bounded(&tokens, bounds)?;
    let semantics = canonical_debug_json(dataset).map_err(|_| LmqError::SemanticEncoding)?;
    let packets = [&packet[..]];
    encode_codec_bundle(
        CodecBundleInput {
            // Baseline kernels: any reader of the profile can decode these packets.
            required_capabilities: 0,
            canonical_semantics: &semantics,
            fidelity,
            implementation,
            model_provenance: Some(model),
            packets: &packets,
            parameters: canonical_parameters(),
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
    open_bundle_bounded(bytes, backend, LmqResourceBounds::from_bundle(bounds))
}

pub fn open_bundle_bounded<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: LmqResourceBounds,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    let bundle = CodecBundleView::open(bytes, bounds.bundle).map_err(LmqError::Bundle)?;
    let catalog = bundle.catalog();
    if catalog.profile() != CodecProfile::LmqProgressive
        || catalog.packet_count() != 1
        || catalog.model_provenance() != Some(&backend.model_provenance())
        || catalog.fidelity().kind == CodecFidelityKind::Exact
        || catalog.implementation().kernel_id != LMQ_KERNEL_ID
        || catalog.parameters() != canonical_parameters()
    {
        return Err(LmqError::CatalogContract);
    }
    let dataset = parse_canonical_dataset(bundle.canonical_semantics())
        .map_err(|_| LmqError::SemanticEncoding)?;
    let (expected_channels, expected_samples) = reconstruction_shape(&dataset)?;
    enforce_signal_bound(expected_channels, expected_samples, bounds.max_signal_bytes)?;
    let packet = bundle.packet(0).ok_or(LmqError::Header)?;
    let tokens = decode_packet_bounded(packet, bounds)?;
    if tokens.n_channels != expected_channels || tokens.n_samples != expected_samples {
        return Err(LmqError::SignalShapeMismatch);
    }
    let signal = backend.decode(&tokens).map_err(LmqError::Backend)?;
    if signal.len() != usize::from(tokens.n_channels)
        || signal
            .iter()
            .any(|channel| channel.len() != tokens.n_samples as usize)
    {
        return Err(LmqError::SignalShapeMismatch);
    }
    let reconstructed = build_reconstructed_dataset(&dataset, &signal)?;
    Ok(OpenedLmqBundle {
        bundle,
        source_dataset: dataset,
        reconstructed,
    })
}

fn build_reconstructed_dataset(
    source: &AbirDataset,
    signal: &[Vec<i64>],
) -> Result<OpenedDataset<InMemoryPayloadAccess>, LmqError> {
    let source_recording = &source.recordings()[0];
    let source_stream = &source.streams()[0];
    if signal.len() != source_stream.atoms().len() {
        return Err(LmqError::SignalShapeMismatch);
    }

    let mut access = InMemoryPayloadAccess::new();
    let mut payloads = Vec::with_capacity(signal.len());
    for channel in signal {
        let bytes = channel
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let content_id = payload_content_id(ElementType::I64, &bytes);
        access.insert(content_id, bytes);
        payloads.push(content_id);
    }

    let dataset_id =
        derived_object_id::<DatasetTag>(b"dataset", source.id().as_bytes(), &payloads, 0);
    let recording_id =
        derived_object_id::<RecordingTag>(b"recording", source.id().as_bytes(), &payloads, 0);
    let stream_id = derived_object_id::<StreamTag>(b"stream", source.id().as_bytes(), &payloads, 0);
    let mut draft = DatasetDraft::new(dataset_id);
    let mut atom_ids = Vec::with_capacity(signal.len());

    for (index, ((source_atom_id, channel), content_id)) in source_stream
        .atoms()
        .iter()
        .zip(signal)
        .zip(payloads.iter().copied())
        .enumerate()
    {
        let source_atom = source
            .atoms()
            .iter()
            .find(|atom| atom.id() == *source_atom_id)
            .ok_or(LmqError::UnsupportedSemantics("unresolved source atom"))?;
        let Atom::SignalBlock(source_block) = source_atom else {
            return Err(LmqError::UnsupportedSemantics(
                "only SignalBlock atoms are supported",
            ));
        };
        let atom_id =
            derived_object_id::<AtomTag>(b"signal-block", source.id().as_bytes(), &payloads, index);
        atom_ids.push(atom_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(PayloadDescriptor::new(
                content_id,
                u64::try_from(channel.len())
                    .ok()
                    .and_then(|samples| samples.checked_mul(8))
                    .ok_or(LmqError::SignalShapeMismatch)?,
                ElementType::I64,
                ByteOrder::Little,
                vec![1, channel.len() as u64],
                Layout::DenseRowMajor,
                None,
                None,
            )),
            source_block.time_axis().clone(),
            source_block.calibration().cloned(),
        )));
    }

    let mut recording = Recording::new(recording_id, vec![stream_id]);
    for source_key in source_recording.source_keys() {
        recording.add_source_key(source_key.clone());
    }
    draft.add_recording(recording);

    // Carry the source's interpretive context instead of dropping it.
    //
    // The reconstructed stream used to be built with `None, None, None` for
    // `clock_id`, `channel_basis_id` and `policy_id`. Each omission changes what
    // the decoded dataset MEANS:
    //
    //   * without a channel basis it says "21 channels" and no longer says which
    //     electrodes they are, so the reconstruction cannot be compared against
    //     the source it came from;
    //   * without a clock its timestamps no longer name the timebase they are
    //     on;
    //   * without a policy, any consent, retention or use restriction attached
    //     to the source silently fails to travel with the decoded copy. That one
    //     is not a fidelity question. A lossy copy that has quietly shed its
    //     governing policy is the failure the policy exists to prevent.
    //
    // The collections are copied whole rather than chased through the reference
    // graph. They are metadata, not payload, and copying all of them keeps
    // referential closure true by construction — resolving only what looked
    // reachable would make correctness depend on this function's model of
    // ABIR's reference graph staying in step with ABIR's.
    for clock in source.clocks() {
        draft.add_clock(clock.clone());
    }
    for frame in source.coordinate_frames() {
        draft.add_coordinate_frame(frame.clone());
    }
    for basis in source.channel_bases() {
        draft.add_channel_basis(basis.clone());
    }
    for policy in source.policies() {
        draft.add_policy(policy.clone());
    }
    for channel in source.channels() {
        draft.add_channel(channel.clone());
    }

    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        source_stream.modality().clone(),
        atom_ids,
        source_stream.clock_id(),
        source_stream.channel_basis_id(),
        source_stream.policy_id(),
    ));
    let dataset = draft
        .validate(ValidationLimits::default())
        .map_err(|_| LmqError::SemanticValidation)?;
    Ok(OpenedDataset::new(dataset, access))
}

fn derived_object_id<T>(
    role: &[u8],
    source_dataset_id: &[u8; 16],
    payloads: &[ContentId],
    index: usize,
) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmq.reconstruction-object-v1\0");
    hasher.update(role);
    hasher.update(&[0]);
    hasher.update(source_dataset_id);
    hasher.update(&(index as u64).to_le_bytes());
    for payload in payloads {
        hasher.update(payload.as_bytes());
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    ObjectId::from_bytes(bytes)
}

fn reconstruction_shape(dataset: &AbirDataset) -> Result<(u16, u32), LmqError> {
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
    let channels =
        u16::try_from(stream.atoms().len()).map_err(|_| LmqError::SignalShapeMismatch)?;
    let mut samples = None;
    let mut start = None;
    for atom_id in stream.atoms() {
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or(LmqError::UnsupportedSemantics("unresolved stream atom"))?;
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
            || start
                .replace(segment.start())
                .is_some_and(|prior| prior != segment.start())
        {
            return Err(LmqError::SignalShapeMismatch);
        }
    }
    let samples = u32::try_from(samples.ok_or(LmqError::SignalShapeMismatch)?)
        .map_err(|_| LmqError::SignalShapeMismatch)?;
    Ok((channels, samples))
}

fn canonical_parameters() -> Vec<CodecParameter> {
    vec![
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
            name: "semantic.fidelity-contract".to_string(),
            value: CodecParameterValue::Text {
                value: LMQ_FIDELITY_CONTRACT.to_string(),
            },
        },
    ]
}

#[cfg(test)]
fn encode_packet(tokens: &NeuralTokens) -> Result<Vec<u8>, LmqError> {
    encode_packet_bounded(tokens, LmqResourceBounds::default())
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
    dataset: &AbirDataset,
    access: &A,
    max_signal_bytes: u64,
) -> Result<(Vec<Vec<i64>>, f64), LmqError> {
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
    let channel_count =
        u16::try_from(stream.atoms().len()).map_err(|_| LmqError::SignalShapeMismatch)?;
    let minimum_signal_bytes =
        u64::from(channel_count)
            .checked_mul(8)
            .ok_or(LmqError::ResourceLimit {
                resource: LmqResource::SignalBytes,
                actual: u64::MAX,
                limit: max_signal_bytes,
            })?;
    enforce_lmq_limit(
        LmqResource::SignalBytes,
        minimum_signal_bytes,
        max_signal_bytes,
    )?;
    let mut channels = Vec::with_capacity(usize::from(channel_count));
    let mut decoded_bytes = 0_u64;
    let mut sample_rate = None;
    let mut sample_count = None;
    let mut start = None;
    for atom_id in stream.atoms() {
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or(LmqError::UnsupportedSemantics("unresolved stream atom"))?;
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
        u32::try_from(segment.samples()).map_err(|_| LmqError::SignalShapeMismatch)?;
        let rate = segment.rate();
        if sample_rate.replace(rate).is_some_and(|prior| prior != rate)
            || sample_count
                .replace(segment.samples())
                .is_some_and(|prior| prior != segment.samples())
            || start
                .replace(segment.start())
                .is_some_and(|prior| prior != segment.start())
        {
            return Err(LmqError::UnsupportedSemantics(
                "LMQ requires aligned starts, uniform rates, and sample counts",
            ));
        }
        if descriptor.shape().last().copied() != Some(segment.samples()) {
            return Err(LmqError::SignalShapeMismatch);
        }
        decoded_bytes = decoded_bytes
            .checked_add(
                segment
                    .samples()
                    .checked_mul(8)
                    .ok_or(LmqError::ResourceLimit {
                        resource: LmqResource::SignalBytes,
                        actual: u64::MAX,
                        limit: max_signal_bytes,
                    })?,
            )
            .ok_or(LmqError::ResourceLimit {
                resource: LmqResource::SignalBytes,
                actual: u64::MAX,
                limit: max_signal_bytes,
            })?;
        channels.push((descriptor, segment.samples()));
    }
    enforce_lmq_limit(LmqResource::SignalBytes, decoded_bytes, max_signal_bytes)?;

    let mut signal = Vec::with_capacity(channels.len());
    for (descriptor, samples) in channels {
        let lease = access.lease(descriptor).map_err(LmqError::PayloadAccess)?;
        verify_payload_content(descriptor, lease.bytes())
            .map_err(|_| LmqError::PayloadIdentityMismatch)?;
        let channel = decode_integer_payload(descriptor, lease.bytes())?;
        if channel.len() as u64 != samples {
            return Err(LmqError::SignalShapeMismatch);
        }
        signal.push(channel);
    }
    let (numerator, denominator) = sample_rate.ok_or(LmqError::SignalShapeMismatch)?.parts();
    let rate = numerator as f64 / denominator as f64;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(LmqError::UnsupportedSemantics("invalid sample rate"));
    }
    Ok((signal, rate))
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
    use crate::backend::StubBackend;
    use semantic_abir::{
        payload_content_id, AtomTag, ConceptId, DatasetDraft, DatasetTag, InMemoryPayloadAccess,
        ObjectId, OpenedDataset, Rational, Recording, RecordingTag, SignalBlock, Stream, StreamTag,
        TimeSegment, ValidationLimits,
    };
    use semantic_abir_bcs::{ModelProvenance, BCS2_MAGIC};

    fn fixture() -> OpenedDataset<InMemoryPayloadAccess> {
        fixture_with_starts(&[0, 0, 0, 0])
    }

    fn fixture_with_starts(starts: &[i128]) -> OpenedDataset<InMemoryPayloadAccess> {
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
                None,
            )));
        }
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

    /// A governed source must not decode into an ungoverned copy.
    ///
    /// The reconstructed stream used to be built with `None, None, None` for
    /// clock, channel basis and policy. Every test above passes either way,
    /// because none of their fixtures carried any of the three -- which is
    /// exactly how the omission survived. This one carries all three.
    #[test]
    fn reconstruction_carries_the_clock_basis_and_policy_of_its_source() {
        use semantic_abir::{
            ChannelBasis, ChannelBasisTag, ChannelSpec, Clock, ClockTag, Policy, PolicyTag,
            ReferenceKind,
        };

        let signal: Vec<Vec<i64>> = (0..2).map(|c| vec![c as i64; 8]).collect();
        let mut draft = DatasetDraft::new(ObjectId::<DatasetTag>::from_bytes([1; 16]));
        let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
        let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
        let clock_id = ObjectId::<ClockTag>::from_bytes([4; 16]);
        let basis_id = ObjectId::<ChannelBasisTag>::from_bytes([5; 16]);
        let policy_id = ObjectId::<PolicyTag>::from_bytes([6; 16]);
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
                        Rational::new(0, 1).unwrap(),
                        Rational::new(250, 1).unwrap(),
                        channel.len() as u64,
                    )
                    .unwrap(),
                ),
                None,
            )));
        }
        draft.add_clock(Clock::new(
            clock_id,
            ConceptId::new("abir:clock/acquisition").unwrap(),
            None,
            Rational::new(0, 1).unwrap(),
            Rational::new(250, 1).unwrap(),
            Rational::new(0, 1).unwrap(),
        ));
        draft.add_channel_basis(ChannelBasis::new(
            basis_id,
            (0..signal.len())
                .map(|_| ChannelSpec::new(ConceptId::new("abir:channel/eeg").unwrap()))
                .collect(),
            ReferenceKind::Unknown,
        ));
        // The restriction is the point: it must reach the decoded copy.
        draft.add_policy(Policy::new(
            policy_id,
            None,
            vec![ConceptId::new("abir:restriction/no-redistribution").unwrap()],
        ));
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            ConceptId::new("abir:modality/eeg").unwrap(),
            atom_ids,
            Some(clock_id),
            Some(basis_id),
            Some(policy_id),
        ));
        let opened =
            OpenedDataset::new(draft.validate(ValidationLimits::default()).unwrap(), access);

        let backend = StubBackend::default();
        let bytes = encode_bundle(
            opened.dataset(),
            opened.access(),
            &backend,
            transformed_fidelity("test-residue"),
            implementation_identity("test"),
            ResourceBounds::default(),
        )
        .expect("encode");
        let decoded = open_bundle(&bytes, &backend, ResourceBounds::default()).expect("decode");
        let stream = &decoded.reconstructed().dataset().streams()[0];

        assert_eq!(stream.clock_id(), Some(clock_id), "clock was dropped");
        assert_eq!(
            stream.channel_basis_id(),
            Some(basis_id),
            "channel basis was dropped: the copy no longer says which electrodes these are"
        );
        assert_eq!(
            stream.policy_id(),
            Some(policy_id),
            "policy was dropped: a governed recording decoded into an ungoverned copy"
        );
        // The references must resolve, not merely be present as ids.
        let reconstructed = decoded.reconstructed().dataset();
        assert!(reconstructed.policies().iter().any(|p| p.id() == policy_id));
        assert!(reconstructed.clocks().iter().any(|c| c.id() == clock_id));
        assert!(reconstructed
            .channel_bases()
            .iter()
            .any(|b| b.id() == basis_id));
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

        let parameters = canonical_parameters();
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
        assert_eq!(reconstructed.dataset().atoms().len(), 4);
        for atom in reconstructed.dataset().atoms() {
            let block = reconstructed.block_view(atom.id()).unwrap();
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
            fn model_provenance(&self) -> ModelProvenance {
                let mut provenance = self.0.model_provenance();
                provenance.checkpoint_sha256 = [11; 32];
                provenance
            }

            fn encode(
                &self,
                signal: &[Vec<i64>],
                sample_rate: f64,
            ) -> Result<NeuralTokens, BackendError> {
                self.0.encode(signal, sample_rate)
            }

            fn decode(&self, tokens: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
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
            fn model_provenance(&self) -> ModelProvenance {
                StubBackend::default().model_provenance()
            }

            fn encode(
                &self,
                _signal: &[Vec<i64>],
                _sample_rate: f64,
            ) -> Result<NeuralTokens, BackendError> {
                panic!("backend must not run after failed signal preflight")
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
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
            fn model_provenance(&self) -> ModelProvenance {
                StubBackend::default().model_provenance()
            }

            fn encode(
                &self,
                _signal: &[Vec<i64>],
                _sample_rate: f64,
            ) -> Result<NeuralTokens, BackendError> {
                unreachable!()
            }

            fn decode(&self, _tokens: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
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
