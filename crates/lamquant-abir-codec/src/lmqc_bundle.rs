//! Non-destructive migration of legacy `LMQC` containers into
//! `bcs.lmq.progressive.v1`.
//!
//! Packet 0 carries the opaque neural payload. Canonical ABIR semantics carry
//! montage, channel order, and decoded shape. Packet 1 retains only legacy
//! framing metadata (prefix plus CRC), permitting byte-exact restoration
//! without duplicating the neural payload or keeping legacy framing active.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use lamquant_abir_montage::{
    channel_spec_for, frames_for_montage, montage_root_frame, MontageError,
};
use lamquant_lml_mcu::{
    crc32::{crc32, crc32_update, CRC32_INIT},
    lmqc::{
        decode_lmqc, LmqcContainer, LmqcError, FLAG_COORDS, FLAG_NAMES, HEADER_SIZE,
        PAYLOAD_FP16_LATENT, PAYLOAD_FSQ_TOKENS,
    },
};
use semantic_abir::{
    canonical_debug_json, payload_content_id, AbirDataset, Atom, AtomTag, ByteOrder, ChannelBasis,
    ChannelBasisTag, Clock, ClockTag, ConceptId, ContentId, DatasetDraft, DatasetTag,
    DecodedSemantics, ElementType, EncodedBlock, Layout, ObjectId, PayloadDescriptor, Presence,
    Rational, Recording, RecordingTag, ReferenceKind, SourceCapsule, SourceKey, Stream, StreamTag,
    ValidationLimits,
};
use semantic_abir_bcs::{
    encode_codec_bundle, raw_content_id, Bcs2View, CodecBundleError, CodecBundleInput,
    CodecBundleView, CodecFidelity, CodecFidelityKind, CodecImplementation, CodecParameter,
    CodecParameterValue, CodecProfile, ModelProvenance, ResourceBounds, CAP_LMQC_LEGACY_V1,
};

const LMQC_SOURCE_MEDIA_TYPE: &str = "application/vnd.quitetall.lmqc";
const LMQC_SOURCE_NAMESPACE: &str = "lmqc.container";
const LMQC_CHANNEL_NAMESPACE: &str = "lmqc.channel-label";
const LMQC_ESCAPED_CHANNEL_NAMESPACE: &str = "lmqc.channel-label.utf8-hex";
const NEURAL_PAYLOAD_ORDINAL: usize = 0;
const REEMIT_METADATA_ORDINAL: usize = 1;
const RAW_CONTENT_HASH_DOMAIN: &[u8] = b"org.quitetall.abir.bcs2.raw-content\0";

/// Capability required to interpret legacy LMQC payload and re-emit frames.
pub const LMQC_READER_CAPABILITIES: u64 = CAP_LMQC_LEGACY_V1;
/// Largest channel count representable by the legacy LMQC wire.
///
/// Actual imports remain bounded by the caller's BCS2 resource and catalog
/// limits; this converter does not impose a narrower scientific ceiling.
pub const MAX_LMQC_CHANNELS: u16 = u16::MAX;

/// Semantic kind carried by the legacy LMQC payload discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LmqcPayloadKind {
    Fp16Latent,
    FsqTokens,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LegacyLayout {
    payload_start: usize,
    payload_end: usize,
}

impl LmqcPayloadKind {
    fn parse(value: u8) -> Result<Self, LmqcBundleError> {
        match value {
            PAYLOAD_FP16_LATENT => Ok(Self::Fp16Latent),
            PAYLOAD_FSQ_TOKENS => Ok(Self::FsqTokens),
            _ => Err(LmqcBundleError::InvalidLegacy("unknown payload kind")),
        }
    }

    const fn encoding_concept(self) -> &'static str {
        match self {
            Self::Fp16Latent => "lamquant:encoding/lmqc-fp16-latent-v1",
            Self::FsqTokens => "lamquant:encoding/lmqc-fsq-tokens-v1",
        }
    }

    const fn wire_name(self) -> &'static str {
        match self {
            Self::Fp16Latent => "fp16-latent",
            Self::FsqTokens => "fsq-tokens",
        }
    }

    /// Original one-byte LMQC discriminator.
    pub const fn wire_value(self) -> u8 {
        match self {
            Self::Fp16Latent => PAYLOAD_FP16_LATENT,
            Self::FsqTokens => PAYLOAD_FSQ_TOKENS,
        }
    }
}

/// Metadata absent from legacy LMQC and therefore supplied by migration policy.
#[derive(Clone, Debug)]
pub struct LmqcBundleInput {
    /// Measurement uncertainty in metres. LMQC coordinates do not carry it.
    pub coordinate_uncertainty: Rational,
    pub fidelity: CodecFidelity,
    pub implementation: CodecImplementation,
    pub model_provenance: ModelProvenance,
}

/// Failure while projecting LMQC into ABIR or proving inverse closure.
#[derive(Debug)]
pub enum LmqcBundleError {
    Bundle(CodecBundleError),
    Legacy(LmqcError),
    Montage(MontageError),
    InvalidLegacy(&'static str),
    SemanticEncoding,
    SemanticMismatch,
    SemanticValidation,
}

impl fmt::Display for LmqcBundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(error) => error.fmt(formatter),
            Self::Legacy(error) => write!(formatter, "invalid legacy LMQC: {error:?}"),
            Self::Montage(error) => write!(formatter, "invalid LMQC montage: {error:?}"),
            Self::InvalidLegacy(reason) => write!(formatter, "unsupported legacy LMQC: {reason}"),
            Self::SemanticEncoding => formatter.write_str("LMQC ABIR semantics could not encode"),
            Self::SemanticMismatch => {
                formatter.write_str("LMQC bundle semantics do not match source capsule")
            }
            Self::SemanticValidation => {
                formatter.write_str("LMQC ABIR semantics failed validation")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmqcBundleError {}

/// Fully validated borrowed view over one migrated legacy LMQC bundle.
#[derive(Debug)]
pub struct OpenedLmqcBcs2<'a> {
    bundle: CodecBundleView<'a>,
    metadata: LmqcMetadata,
}

/// Decoded legacy header and montage metadata without copied neural payload.
#[derive(Clone, Debug, PartialEq)]
pub struct LmqcMetadata {
    pub version: u8,
    pub n_channels: u16,
    pub latent_c: u16,
    pub latent_t: u16,
    pub sample_rate: u16,
    pub window_samples: u32,
    pub payload_kind: LmqcPayloadKind,
    pub coords: Option<Vec<f32>>,
    pub channels: Option<Vec<alloc::string::String>>,
    pub payload_len: usize,
}

impl LmqcMetadata {
    fn from_parts(
        container: LmqcContainer,
        payload_kind: LmqcPayloadKind,
        payload_len: usize,
    ) -> Self {
        Self {
            version: container.version,
            n_channels: container.n_channels,
            latent_c: container.latent_c,
            latent_t: container.latent_t,
            sample_rate: container.sample_rate,
            window_samples: container.window_samples,
            payload_kind,
            coords: container.coords,
            channels: container.channels,
            payload_len,
        }
    }
}

impl<'a> OpenedLmqcBcs2<'a> {
    pub const fn dataset(&self) -> &AbirDataset {
        self.bundle.dataset()
    }

    /// Opaque neural payload, separated from legacy montage framing.
    pub fn neural_payload(&self) -> &'a [u8] {
        self.bundle
            .packet(NEURAL_PAYLOAD_ORDINAL)
            .expect("validated LMQC bundles contain a neural payload")
    }

    /// Legacy header and montage metadata. Neural payload remains borrowed.
    pub const fn container(&self) -> &LmqcMetadata {
        &self.metadata
    }

    pub const fn bundle(&self) -> &CodecBundleView<'a> {
        &self.bundle
    }
}

/// Project canonical LMQC bytes into a semantic LMQ progressive BCS2 bundle.
///
/// Producer, model, and fidelity evidence are explicit inputs because legacy
/// LMQC stores none of them. Conversion never fabricates those claims.
pub fn lmqc_to_bcs2(
    lmqc: &[u8],
    input: LmqcBundleInput,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmqcBundleError> {
    if input.fidelity.kind == CodecFidelityKind::Exact {
        return Err(LmqcBundleError::InvalidLegacy(
            "lossy LMQC cannot claim exact fidelity",
        ));
    }
    if input.coordinate_uncertainty.parts().0 < 0 {
        return Err(LmqcBundleError::InvalidLegacy(
            "negative coordinate uncertainty",
        ));
    }
    let layout = preflight_lmqc(lmqc, bounds)?;
    let payload = &lmqc[layout.payload_start..layout.payload_end];
    let reemit_metadata = reemit_metadata(lmqc, layout)?;
    let (container, source_id, payload_kind) =
        decode_split_lmqc(payload, &reemit_metadata, bounds)?;
    if source_id != raw_content_id(lmqc) {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    let dataset = dataset_from_lmqc(
        source_id,
        &container,
        payload,
        payload_kind,
        input.coordinate_uncertainty,
        semantic_limits(bounds),
    )?;
    let semantics =
        canonical_debug_json(&dataset).map_err(|_| LmqcBundleError::SemanticEncoding)?;
    let packets = [payload, &reemit_metadata[..]];
    encode_codec_bundle(
        CodecBundleInput {
            canonical_semantics: &semantics,
            fidelity: input.fidelity,
            implementation: input.implementation,
            model_provenance: Some(input.model_provenance),
            packets: &packets,
            parameters: legacy_parameters(&container, payload_kind, input.coordinate_uncertainty),
            profile: CodecProfile::LmqProgressive,
            required_capabilities: CAP_LMQC_LEGACY_V1,
        },
        bounds,
    )
    .map_err(LmqcBundleError::Bundle)
}

/// Open an imported LMQC bundle and prove packet, capsule, and semantic closure.
pub fn open_lmqc_bcs2(
    bytes: &[u8],
    bounds: ResourceBounds,
) -> Result<OpenedLmqcBcs2<'_>, LmqcBundleError> {
    let bundle = CodecBundleView::open_with_capabilities(bytes, CAP_LMQC_LEGACY_V1, bounds)
        .map_err(LmqcBundleError::Bundle)?;
    let catalog = bundle.catalog();
    if catalog.profile() != CodecProfile::LmqProgressive
        || catalog.packet_count() != 2
        || catalog.model_provenance().is_none()
        || catalog.fidelity().kind == CodecFidelityKind::Exact
    {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    verify_packet_capabilities(bytes, catalog, bounds)?;
    let coordinate_uncertainty = catalog_coordinate_uncertainty(catalog.parameters())?;

    let payload = bundle
        .packet(NEURAL_PAYLOAD_ORDINAL)
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    let metadata = bundle
        .packet(REEMIT_METADATA_ORDINAL)
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    let (container, source_id, payload_kind) = decode_split_lmqc(payload, metadata, bounds)?;
    if catalog.parameters() != legacy_parameters(&container, payload_kind, coordinate_uncertainty) {
        return Err(LmqcBundleError::SemanticMismatch);
    }

    let expected = dataset_from_lmqc(
        source_id,
        &container,
        payload,
        payload_kind,
        coordinate_uncertainty,
        semantic_limits(bounds),
    )?;
    let expected_semantics =
        canonical_debug_json(&expected).map_err(|_| LmqcBundleError::SemanticEncoding)?;
    if bundle.canonical_semantics() != expected_semantics {
        return Err(LmqcBundleError::SemanticMismatch);
    }

    let metadata = LmqcMetadata::from_parts(container, payload_kind, payload.len());
    Ok(OpenedLmqcBcs2 { bundle, metadata })
}

/// Recover legacy bytes only after the BCS2 bundle closes over its source
/// capsule, inner payload, and reconstructed ABIR semantics.
pub fn bcs2_to_lmqc(bcs2: &[u8], bounds: ResourceBounds) -> Result<Vec<u8>, LmqcBundleError> {
    let opened = open_lmqc_bcs2(bcs2, bounds)?;
    let metadata = opened
        .bundle()
        .packet(REEMIT_METADATA_ORDINAL)
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    restore_legacy(opened.neural_payload(), metadata, bounds)
}

fn preflight_lmqc(bytes: &[u8], bounds: ResourceBounds) -> Result<LegacyLayout, LmqcBundleError> {
    if bytes.len() > bounds.max_frame_bytes as usize {
        return Err(LmqcBundleError::InvalidLegacy(
            "source container exceeds resource bound",
        ));
    }
    if bytes.len() < HEADER_SIZE + 8 {
        return Err(LmqcBundleError::Legacy(LmqcError::TooShort));
    }
    let payload_start = payload_start_from_prefix(bytes, bounds)?;
    let payload_len = u32::from_le_bytes(
        bytes[payload_start - 4..payload_start]
            .try_into()
            .map_err(|_| LmqcBundleError::Legacy(LmqcError::Truncated))?,
    );
    let payload_len =
        usize::try_from(payload_len).map_err(|_| LmqcBundleError::Legacy(LmqcError::Truncated))?;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
    if payload_end.checked_add(4) != Some(bytes.len()) {
        return Err(LmqcBundleError::InvalidLegacy(
            "non-canonical payload framing",
        ));
    }
    Ok(LegacyLayout {
        payload_start,
        payload_end,
    })
}

fn validate_header_fields(bytes: &[u8]) -> Result<(), LmqcBundleError> {
    if let Some(flags) = bytes.get(5) {
        if flags & !(FLAG_NAMES | FLAG_COORDS) != 0 {
            return Err(LmqcBundleError::InvalidLegacy("unknown LMQC header flags"));
        }
    }
    if bytes.get(7).is_some_and(|reserved| *reserved != 0) {
        return Err(LmqcBundleError::InvalidLegacy("nonzero LMQC reserved byte"));
    }
    Ok(())
}

fn payload_start_from_prefix(
    bytes: &[u8],
    bounds: ResourceBounds,
) -> Result<usize, LmqcBundleError> {
    if bytes.len() < HEADER_SIZE {
        return Err(LmqcBundleError::Legacy(LmqcError::TooShort));
    }
    validate_header_fields(bytes)?;
    let n_channels = u16::from_le_bytes([bytes[8], bytes[9]]);
    if usize::from(n_channels) > bounds.max_catalog_bytes as usize {
        return Err(LmqcBundleError::InvalidLegacy(
            "channel count exceeds catalog bound",
        ));
    }
    let flags = bytes[5];
    let mut offset = HEADER_SIZE;
    let mut escaped_label_expansion = 0_usize;
    if flags & FLAG_COORDS != 0 {
        let coordinate_bytes = usize::from(n_channels)
            .checked_mul(12)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
        offset = offset
            .checked_add(coordinate_bytes)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
    }
    if flags & FLAG_NAMES != 0 {
        let length_end = offset
            .checked_add(4)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
        let encoded = bytes
            .get(offset..length_end)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
        let names_len = u32::from_le_bytes(
            encoded
                .try_into()
                .map_err(|_| LmqcBundleError::Legacy(LmqcError::Truncated))?,
        );
        let names_len = usize::try_from(names_len)
            .map_err(|_| LmqcBundleError::Legacy(LmqcError::Truncated))?;
        offset = length_end
            .checked_add(names_len)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
        let names_bytes = bytes
            .get(length_end..offset)
            .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
        let names = core::str::from_utf8(names_bytes)
            .map_err(|_| LmqcBundleError::Legacy(LmqcError::BadUtf8))?;
        let name_count = if names.is_empty() {
            0
        } else {
            names.split('\n').count()
        };
        if name_count != usize::from(n_channels) {
            return Err(LmqcBundleError::Legacy(LmqcError::BadNamesLen));
        }
        for name in names.split('\n') {
            if name.chars().any(char::is_control) {
                escaped_label_expansion = escaped_label_expansion.checked_add(name.len()).ok_or(
                    LmqcBundleError::InvalidLegacy("projected channel labels exceed catalog bound"),
                )?;
            }
        }
    }
    let payload_start = offset
        .checked_add(4)
        .ok_or(LmqcBundleError::Legacy(LmqcError::Truncated))?;
    let projected_metadata = payload_start.checked_add(escaped_label_expansion).ok_or(
        LmqcBundleError::InvalidLegacy("projected channel labels exceed catalog bound"),
    )?;
    if projected_metadata > bounds.max_catalog_bytes as usize {
        return Err(LmqcBundleError::InvalidLegacy(
            if escaped_label_expansion == 0 {
                "legacy metadata exceeds catalog bound"
            } else {
                "projected channel labels exceed catalog bound"
            },
        ));
    }
    if payload_start > bytes.len() {
        return Err(LmqcBundleError::Legacy(LmqcError::Truncated));
    }
    Ok(payload_start)
}

fn validate_container(
    container: &LmqcContainer,
    payload: &[u8],
) -> Result<LmqcPayloadKind, LmqcBundleError> {
    if container.n_channels == 0 {
        return Err(LmqcBundleError::InvalidLegacy("zero channels"));
    }
    if container.latent_c == 0 || container.latent_t == 0 {
        return Err(LmqcBundleError::InvalidLegacy("zero latent extent"));
    }
    if container.sample_rate == 0 || container.window_samples == 0 {
        return Err(LmqcBundleError::InvalidLegacy("zero decoded extent"));
    }
    if payload.is_empty() {
        return Err(LmqcBundleError::InvalidLegacy("empty neural payload"));
    }
    LmqcPayloadKind::parse(container.payload_kind)
}

fn decode_split_lmqc(
    payload: &[u8],
    metadata: &[u8],
    bounds: ResourceBounds,
) -> Result<(LmqcContainer, ContentId, LmqcPayloadKind), LmqcBundleError> {
    let (prefix, stored_crc) = split_reemit_metadata(metadata)?;
    if prefix.len() < HEADER_SIZE + 4 {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    let payload_start = payload_start_from_prefix(prefix, bounds)?;
    if payload_start != prefix.len() {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    let declared_len = u32::from_le_bytes(
        prefix[prefix.len() - 4..]
            .try_into()
            .map_err(|_| LmqcBundleError::SemanticMismatch)?,
    );
    if usize::try_from(declared_len).ok() != Some(payload.len()) {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    let legacy_len = prefix
        .len()
        .checked_add(payload.len())
        .and_then(|size| size.checked_add(4))
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    if legacy_len > bounds.max_frame_bytes as usize {
        return Err(LmqcBundleError::InvalidLegacy(
            "restored container exceeds resource bound",
        ));
    }

    let expected_crc = u32::from_le_bytes(
        stored_crc
            .try_into()
            .map_err(|_| LmqcBundleError::SemanticMismatch)?,
    );
    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, prefix);
    crc_state = crc32_update(crc_state, payload);
    if crc_state ^ CRC32_INIT != expected_crc {
        return Err(LmqcBundleError::Legacy(LmqcError::CrcMismatch));
    }

    let mut metadata_only = prefix.to_vec();
    let payload_len_offset = metadata_only.len() - 4;
    metadata_only[payload_len_offset..].copy_from_slice(&0_u32.to_le_bytes());
    let checksum = crc32(&metadata_only);
    metadata_only.extend_from_slice(&checksum.to_le_bytes());
    let container = decode_lmqc(&metadata_only).map_err(LmqcBundleError::Legacy)?;
    drop(metadata_only);
    let payload_kind = validate_container(&container, payload)?;
    let source_id = raw_content_id_parts(prefix, payload, stored_crc);
    Ok((container, source_id, payload_kind))
}

fn raw_content_id_parts(prefix: &[u8], payload: &[u8], crc: &[u8]) -> ContentId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RAW_CONTENT_HASH_DOMAIN);
    hasher.update(prefix);
    hasher.update(payload);
    hasher.update(crc);
    ContentId::from_bytes(*hasher.finalize().as_bytes())
}

fn verify_packet_capabilities(
    bytes: &[u8],
    catalog: &semantic_abir_bcs::CodecBundleCatalog,
    bounds: ResourceBounds,
) -> Result<(), LmqcBundleError> {
    let packet_ids = (0..catalog.packet_count())
        .map(|ordinal| {
            catalog
                .packet_content_id(ordinal)
                .ok_or(LmqcBundleError::SemanticMismatch)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let wire = Bcs2View::parse(bytes, CAP_LMQC_LEGACY_V1, bounds)
        .map_err(|error| LmqcBundleError::Bundle(CodecBundleError::Bcs2(error)))?;
    let mut seen = BTreeSet::new();
    for frame in wire.frames() {
        if packet_ids.contains(&frame.content_id()) {
            if frame.required_capabilities() != CAP_LMQC_LEGACY_V1 {
                return Err(LmqcBundleError::SemanticMismatch);
            }
            seen.insert(frame.content_id());
        } else if frame.required_capabilities() != 0 {
            return Err(LmqcBundleError::SemanticMismatch);
        }
    }
    if seen != packet_ids {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    Ok(())
}

fn reemit_metadata(legacy: &[u8], layout: LegacyLayout) -> Result<Vec<u8>, LmqcBundleError> {
    let prefix_len = u32::try_from(layout.payload_start)
        .map_err(|_| LmqcBundleError::InvalidLegacy("legacy prefix exceeds u32"))?;
    let mut metadata = Vec::with_capacity(
        4_usize
            .checked_add(layout.payload_start)
            .and_then(|size| size.checked_add(4))
            .ok_or(LmqcBundleError::InvalidLegacy("metadata extent overflow"))?,
    );
    metadata.extend_from_slice(&prefix_len.to_le_bytes());
    metadata.extend_from_slice(&legacy[..layout.payload_start]);
    metadata.extend_from_slice(&legacy[layout.payload_end..]);
    Ok(metadata)
}

fn restore_legacy(
    payload: &[u8],
    metadata: &[u8],
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmqcBundleError> {
    let (prefix, suffix) = split_reemit_metadata(metadata)?;
    let legacy_len = prefix
        .len()
        .checked_add(payload.len())
        .and_then(|size| size.checked_add(4))
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    if legacy_len > bounds.max_frame_bytes as usize {
        return Err(LmqcBundleError::InvalidLegacy(
            "restored container exceeds resource bound",
        ));
    }
    let mut legacy = Vec::with_capacity(legacy_len);
    legacy.extend_from_slice(prefix);
    legacy.extend_from_slice(payload);
    legacy.extend_from_slice(suffix);
    Ok(legacy)
}

fn split_reemit_metadata(metadata: &[u8]) -> Result<(&[u8], &[u8]), LmqcBundleError> {
    if metadata.len() < 8 {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    let prefix_len =
        u32::from_le_bytes(metadata[..4].try_into().expect("four-byte prefix length")) as usize;
    let suffix_offset = 4_usize
        .checked_add(prefix_len)
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    if suffix_offset.checked_add(4) != Some(metadata.len()) {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    Ok((&metadata[4..suffix_offset], &metadata[suffix_offset..]))
}

fn dataset_from_lmqc(
    source_id: ContentId,
    container: &LmqcContainer,
    neural_payload: &[u8],
    payload_kind: LmqcPayloadKind,
    coordinate_uncertainty: Rational,
    limits: ValidationLimits,
) -> Result<AbirDataset, LmqcBundleError> {
    let dataset_id = derived_id::<DatasetTag>(source_id, b"dataset");
    let recording_id = derived_id::<RecordingTag>(source_id, b"recording");
    let stream_id = derived_id::<StreamTag>(source_id, b"stream");
    let atom_id = derived_id::<AtomTag>(source_id, b"encoded-block");
    let basis_id = derived_id::<ChannelBasisTag>(source_id, b"channel-basis");
    let clock_id = derived_id::<ClockTag>(source_id, b"sample-clock");
    let mut draft = DatasetDraft::new(dataset_id);
    let zero = Rational::new(0, 1).map_err(|_| LmqcBundleError::SemanticValidation)?;

    let coordinates = container
        .coords
        .as_ref()
        .map(|flat| {
            flat.chunks_exact(3)
                .map(|xyz| [xyz[0], xyz[1], xyz[2]])
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let (frames, montage_root) = if coordinates.is_empty() {
        (vec![None; usize::from(container.n_channels)], None)
    } else {
        let digest = montage_digest(container);
        let root = montage_root_frame(&digest).map_err(LmqcBundleError::Montage)?;
        (
            frames_for_montage(
                &digest,
                &coordinates,
                Some(root.id()),
                coordinate_uncertainty,
            )
            .map_err(LmqcBundleError::Montage)?,
            Some(root),
        )
    };

    let channel_concept =
        ConceptId::new("abir:channel/eeg").map_err(|_| LmqcBundleError::SemanticValidation)?;
    let mut channel_specs = Vec::with_capacity(usize::from(container.n_channels));
    let mut montage_root_added = false;
    for (index, frame) in frames.into_iter().enumerate() {
        let frame_id = frame.as_ref().map(|value| value.id());
        if let Some(frame) = frame {
            if !montage_root_added {
                draft.add_coordinate_frame(
                    montage_root
                        .clone()
                        .expect("located frame carries montage root"),
                );
                montage_root_added = true;
            }
            draft.add_coordinate_frame(frame);
        }
        let mut spec = channel_spec_for(channel_concept.clone(), frame_id);
        if let Some(name) = container
            .channels
            .as_ref()
            .and_then(|names| names.get(index))
        {
            spec = spec.with_source_key(source_key_for_label(name)?);
        }
        channel_specs.push(spec);
    }
    draft.add_channel_basis(ChannelBasis::new(
        basis_id,
        channel_specs,
        ReferenceKind::Unknown,
    ));

    let payload_len = u64::try_from(neural_payload.len())
        .map_err(|_| LmqcBundleError::InvalidLegacy("payload extent overflow"))?;
    let payload = PayloadDescriptor::new(
        payload_content_id(ElementType::Bytes, neural_payload),
        payload_len,
        ElementType::Bytes,
        ByteOrder::NotApplicable,
        vec![payload_len],
        Layout::DenseRowMajor,
        Some(
            ConceptId::new(payload_kind.encoding_concept())
                .map_err(|_| LmqcBundleError::SemanticValidation)?,
        ),
        Some("application/octet-stream".into()),
    );
    let decoded = DecodedSemantics::new(
        ConceptId::new("abir:atom/signal-block")
            .map_err(|_| LmqcBundleError::SemanticValidation)?,
        ElementType::F32,
        vec![
            u64::from(container.n_channels),
            u64::from(container.window_samples),
        ],
    );
    draft.add_atom(Atom::EncodedBlock(EncodedBlock::new(
        atom_id,
        Presence::Present,
        Some(payload),
        decoded,
    )));
    draft.add_recording(Recording::new(recording_id, vec![stream_id]));
    draft.add_clock(Clock::new(
        clock_id,
        ConceptId::new("abir:clock/sample").map_err(|_| LmqcBundleError::SemanticValidation)?,
        None,
        zero,
        Rational::new(i128::from(container.sample_rate), 1)
            .map_err(|_| LmqcBundleError::SemanticValidation)?,
        zero,
    ));
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        ConceptId::new("abir:modality/eeg").map_err(|_| LmqcBundleError::SemanticValidation)?,
        vec![atom_id],
        Some(clock_id),
        Some(basis_id),
        None,
    ));
    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new(LMQC_SOURCE_NAMESPACE, source_id.to_string())
            .map_err(|_| LmqcBundleError::SemanticValidation)?,
        source_id,
        Some(LMQC_SOURCE_MEDIA_TYPE),
    ));

    draft
        .validate(limits)
        .map_err(|_| LmqcBundleError::SemanticValidation)
}

fn semantic_limits(bounds: ResourceBounds) -> ValidationLimits {
    ValidationLimits {
        max_recordings: 1,
        max_streams: 1,
        max_atoms: 1,
        max_catalog_records: 0,
        max_relationships: 0,
        max_governance_records: 1,
        max_channels: usize::from(MAX_LMQC_CHANNELS),
        max_rank: 2,
        max_nesting_depth: 8,
        max_metadata_bytes: bounds.max_catalog_bytes as usize,
        max_logical_payload_bytes: u64::from(bounds.max_frame_bytes),
    }
}

fn source_key_for_label(name: &str) -> Result<SourceKey, LmqcBundleError> {
    if let Ok(key) = SourceKey::new(LMQC_CHANNEL_NAMESPACE, name) {
        return Ok(key);
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut escaped = String::new();
    escaped
        .try_reserve(
            name.len()
                .checked_mul(2)
                .ok_or(LmqcBundleError::SemanticValidation)?,
        )
        .map_err(|_| LmqcBundleError::SemanticValidation)?;
    for byte in name.as_bytes() {
        escaped.push(char::from(HEX[usize::from(byte >> 4)]));
        escaped.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    SourceKey::new(LMQC_ESCAPED_CHANNEL_NAMESPACE, escaped)
        .map_err(|_| LmqcBundleError::SemanticValidation)
}

fn derived_id<T>(source: ContentId, role: &[u8]) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmqc-abir-id-v1\0");
    hasher.update(source.as_bytes());
    hasher.update(&[0]);
    hasher.update(role);
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    ObjectId::from_bytes(bytes)
}

fn montage_digest(container: &LmqcContainer) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.lmqc-montage-v1\0");
    hasher.update(&container.n_channels.to_le_bytes());
    if let Some(coords) = &container.coords {
        for value in coords {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    if let Some(names) = &container.channels {
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update(&[0]);
        }
    }
    *hasher.finalize().as_bytes()
}

fn legacy_parameters(
    container: &LmqcContainer,
    payload_kind: LmqcPayloadKind,
    coordinate_uncertainty: Rational,
) -> Vec<CodecParameter> {
    vec![
        text_parameter("container-format", "LMQC/1"),
        integer_parameter("latent-channels", container.latent_c),
        integer_parameter("latent-frames", container.latent_t),
        rational_parameter("montage-coordinate-uncertainty-m", coordinate_uncertainty),
        integer_parameter("neural-payload-ordinal", NEURAL_PAYLOAD_ORDINAL),
        text_parameter("payload-kind", payload_kind.wire_name()),
        text_parameter("reemit-metadata-format", "lmqc-prefix-crc-v1"),
        integer_parameter("reemit-metadata-ordinal", REEMIT_METADATA_ORDINAL),
        integer_parameter("sample-rate-hz", container.sample_rate),
        text_parameter("source-capsule-mode", "reconstructable"),
        integer_parameter("window-samples", container.window_samples),
    ]
}

fn rational_parameter(name: &str, value: Rational) -> CodecParameter {
    let (numerator, denominator) = value.parts();
    CodecParameter {
        name: name.into(),
        value: CodecParameterValue::Rational {
            denominator: denominator.to_string(),
            numerator: numerator.to_string(),
        },
    }
}

fn catalog_coordinate_uncertainty(
    parameters: &[CodecParameter],
) -> Result<Rational, LmqcBundleError> {
    let parameter = parameters
        .iter()
        .find(|parameter| parameter.name == "montage-coordinate-uncertainty-m")
        .ok_or(LmqcBundleError::SemanticMismatch)?;
    let CodecParameterValue::Rational {
        denominator,
        numerator,
    } = &parameter.value
    else {
        return Err(LmqcBundleError::SemanticMismatch);
    };
    let numerator = numerator
        .parse::<i128>()
        .map_err(|_| LmqcBundleError::SemanticMismatch)?;
    let denominator = denominator
        .parse::<i128>()
        .map_err(|_| LmqcBundleError::SemanticMismatch)?;
    let value =
        Rational::new(numerator, denominator).map_err(|_| LmqcBundleError::SemanticMismatch)?;
    if value.parts().0 < 0 {
        return Err(LmqcBundleError::SemanticMismatch);
    }
    Ok(value)
}

fn integer_parameter(name: &str, value: impl ToString) -> CodecParameter {
    CodecParameter {
        name: name.into(),
        value: CodecParameterValue::Integer {
            value: value.to_string(),
        },
    }
}

fn text_parameter(name: &str, value: &str) -> CodecParameter {
    CodecParameter {
        name: name.into(),
        value: CodecParameterValue::Text {
            value: value.into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use lamquant_lml_mcu::lmqc::encode_lmqc;
    use semantic_abir_bcs::{CodecFidelityKind, PccpStatus};

    fn policy() -> LmqcBundleInput {
        LmqcBundleInput {
            coordinate_uncertainty: Rational::new(1, 1_000).unwrap(),
            fidelity: CodecFidelity {
                bound: Some(CodecParameterValue::Rational {
                    denominator: "1000".to_string(),
                    numerator: "75".to_string(),
                }),
                contract_id: ContentId::from_bytes([0x11; 32]),
                kind: CodecFidelityKind::Bounded,
                metric: Some("prd".to_string()),
            },
            implementation: CodecImplementation {
                build_id: "test-lmqc-build".to_string(),
                implementation_id: ContentId::from_bytes([0x12; 32]),
                kernel_id: "org.quitetall.lamquant.test-lmq-v1".to_string(),
            },
            model_provenance: ModelProvenance {
                checkpoint_content_id: ContentId::from_bytes([0x13; 32]),
                checkpoint_sha256: [0x14; 32],
                pccp_change_id: "TEST-LMQC-001".to_string(),
                pccp_evidence_id: ContentId::from_bytes([0x15; 32]),
                pccp_status: PccpStatus::GatePass,
            },
        }
    }

    fn fixture() -> Vec<u8> {
        let names = ["Fp1", "Fp2"]
            .into_iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        encode_lmqc(
            2,
            32,
            79,
            250,
            2500,
            PAYLOAD_FP16_LATENT,
            Some(&[0.081, 0.0, 0.034, f32::NAN, f32::NAN, f32::NAN]),
            Some(&names),
            &[0xde, 0xad, 0xbe, 0xef],
        )
        .expect("canonical LMQC fixture")
    }

    #[test]
    fn canonical_lmqc_round_trips_byte_exact_through_bcs2() {
        let legacy = fixture();
        let bundle =
            lmqc_to_bcs2(&legacy, policy(), ResourceBounds::default()).expect("LMQC projects");
        let recovered = bcs2_to_lmqc(&bundle, ResourceBounds::default()).expect("LMQC restores");
        assert_eq!(recovered, legacy);
    }

    #[test]
    fn split_open_borrows_payload_and_inverse_helper_enforces_bound() {
        let legacy = fixture();
        let layout = preflight_lmqc(&legacy, ResourceBounds::default()).unwrap();
        let payload = &legacy[layout.payload_start..layout.payload_end];
        let metadata = reemit_metadata(&legacy, layout).unwrap();
        let (split, source_id, payload_kind) =
            decode_split_lmqc(payload, &metadata, ResourceBounds::default()).unwrap();
        assert!(
            split.payload.is_empty(),
            "split parser must not copy the neural payload"
        );
        assert_eq!(source_id, raw_content_id(&legacy));
        assert_eq!(payload_kind, LmqcPayloadKind::Fp16Latent);

        let bounds = ResourceBounds {
            max_frame_bytes: u32::try_from(legacy.len() - 1).unwrap(),
            ..ResourceBounds::default()
        };
        assert!(matches!(
            restore_legacy(payload, &metadata, bounds),
            Err(LmqcBundleError::InvalidLegacy(
                "restored container exceeds resource bound"
            ))
        ));
    }
}
