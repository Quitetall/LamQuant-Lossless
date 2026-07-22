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
use crate::body::{decode_body, encode_body, BodyError};

pub const LMQ_KERNEL_ID: &str = "org.quitetall.lamquant.lmq.fsq-rans-v1";
pub const LMQ_FIDELITY_CONTRACT: &str =
    "org.quitetall.lamquant.bcs2.lmq.explicit-nonexact-reconstruction-v1";
pub const RANS_MODEL_TOTAL: u64 = 4096;
const PACKET_MAGIC: &[u8; 4] = b"LMQP";
const PACKET_VERSION: u8 = 1;
const PACKET_HEADER_LEN: usize = 15;
const ABIR_REVISION: &str = "c101513167ad8d7cdefa6387b20c644fdaf66432";

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
    UnsupportedSemantics(&'static str),
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
    hasher.update(ABIR_REVISION.as_bytes());
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
pub fn encode_bundle<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    backend: &dyn NeuralBackend,
    fidelity: CodecFidelity,
    implementation: CodecImplementation,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmqError> {
    if fidelity.kind == CodecFidelityKind::Exact || implementation.kernel_id != LMQ_KERNEL_ID {
        return Err(LmqError::CatalogContract);
    }
    let (signal, sample_rate) = read_signal(dataset, access)?;
    let model = backend.model_provenance();
    let tokens = backend
        .encode(&signal, sample_rate)
        .map_err(LmqError::Backend)?;
    if usize::from(tokens.n_channels) != signal.len()
        || usize::try_from(tokens.n_samples).ok() != signal.first().map(Vec::len)
    {
        return Err(LmqError::SignalShapeMismatch);
    }
    let packet = encode_packet(&tokens)?;
    let semantics = canonical_debug_json(dataset).map_err(|_| LmqError::SemanticEncoding)?;
    let packets = [&packet[..]];
    encode_codec_bundle(
        CodecBundleInput {
            canonical_semantics: &semantics,
            fidelity,
            implementation,
            model_provenance: Some(model),
            packets: &packets,
            parameters: canonical_parameters(),
            profile: CodecProfile::LmqProgressive,
        },
        bounds,
    )
    .map_err(LmqError::Bundle)
}

pub fn open_bundle<'a>(
    bytes: &'a [u8],
    backend: &dyn NeuralBackend,
    bounds: ResourceBounds,
) -> Result<OpenedLmqBundle<'a>, LmqError> {
    let bundle = CodecBundleView::open(bytes, bounds).map_err(LmqError::Bundle)?;
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
    let packet = bundle.packet(0).ok_or(LmqError::Header)?;
    let tokens = decode_packet(packet)?;
    if tokens.n_channels != expected_channels || tokens.n_samples != expected_samples {
        return Err(LmqError::SignalShapeMismatch);
    }
    let reconstruction_bytes = u64::from(expected_channels)
        .checked_mul(u64::from(expected_samples))
        .and_then(|samples| samples.checked_mul(8))
        .ok_or(LmqError::SignalShapeMismatch)?;
    if reconstruction_bytes > u64::from(bounds.max_frame_bytes) {
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
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        source_stream.modality().clone(),
        atom_ids,
        None,
        None,
        None,
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
                value: ABIR_REVISION.to_string(),
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

fn encode_packet(tokens: &NeuralTokens) -> Result<Vec<u8>, LmqError> {
    if tokens.n_channels == 0 || tokens.n_samples == 0 {
        return Err(LmqError::SignalShapeMismatch);
    }
    let counts = histogram(&tokens.tokens, tokens.alphabet)?;
    let symbols = tokens
        .tokens
        .iter()
        .map(|&token| i64::from(token))
        .collect::<Vec<_>>();
    let body = encode_body(&symbols, &tokens.schedule, &counts)?;
    let meta_len = u32::try_from(tokens.backend_meta.len()).map_err(|_| LmqError::Header)?;
    let mut packet = Vec::with_capacity(
        PACKET_HEADER_LEN
            .saturating_add(tokens.backend_meta.len())
            .saturating_add(body.len()),
    );
    packet.extend_from_slice(PACKET_MAGIC);
    packet.push(PACKET_VERSION);
    packet.extend_from_slice(&tokens.n_channels.to_le_bytes());
    packet.extend_from_slice(&tokens.n_samples.to_le_bytes());
    packet.extend_from_slice(&meta_len.to_le_bytes());
    packet.extend_from_slice(&tokens.backend_meta);
    packet.extend_from_slice(&body);
    Ok(packet)
}

fn decode_packet(packet: &[u8]) -> Result<NeuralTokens, LmqError> {
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
    let after_header = packet.get(PACKET_HEADER_LEN..).ok_or(LmqError::Header)?;
    let backend_meta = after_header
        .get(..meta_len)
        .ok_or(LmqError::Header)?
        .to_vec();
    let body = after_header.get(meta_len..).ok_or(LmqError::Header)?;
    let (symbols, schedule, alphabet) = decode_body(body)?;
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
    let mut signal = Vec::with_capacity(stream.atoms().len());
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
        let lease = access.lease(descriptor).map_err(LmqError::PayloadAccess)?;
        verify_payload_content(descriptor, lease.bytes())
            .map_err(|_| LmqError::PayloadIdentityMismatch)?;
        let channel = decode_integer_payload(descriptor, lease.bytes())?;
        if descriptor.shape().last().copied() != Some(segment.samples())
            || channel.len() as u64 != segment.samples()
        {
            return Err(LmqError::SignalShapeMismatch);
        }
        signal.push(channel);
    }
    let (numerator, denominator) = sample_rate.ok_or(LmqError::SignalShapeMismatch)?.parts();
    let rate = numerator as f64 / denominator as f64;
    if !rate.is_finite() || rate <= 0.0 {
        return Err(LmqError::UnsupportedSemantics("invalid sample rate"));
    }
    if signal.len() > u16::MAX as usize || signal.first().map_or(0, Vec::len) > u32::MAX as usize {
        return Err(LmqError::SignalShapeMismatch);
    }
    Ok((signal, rate))
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
}
