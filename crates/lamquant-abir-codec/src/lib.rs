#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! Deterministic LamQuant codec packets carried as semantic ABIR BCS2 Bundles.
//!
//! This crate is deliberately an integration layer. It calls the existing
//! `lamquant-lml-mcu` packet encoder/decoder without changing the LML1 grammar
//! or its hot path. The outer BCS2 Bundle binds canonical ABIR semantics,
//! packet bytes, codec implementation identity, and the exact-fidelity
//! contract. Opening is fail-closed: the LML packet is decoded and every
//! channel is re-hashed as its declared ABIR payload before data is exposed.
//!
//! Legacy lossy LMQC containers use the same contract through
//! [`lmqc_to_bcs2`]: neural payload bytes remain opaque and borrowed, while
//! montage, decoded shape, fidelity, provenance, and byte-exact re-emission
//! metadata become explicit, capability-gated ABIR semantics.

extern crate alloc;

pub mod lmqc_bundle;
pub use lmqc_bundle::{
    bcs2_to_lmqc, lmqc_to_bcs2, open_lmqc_bcs2, LmqcBundleError, LmqcBundleInput, LmqcMetadata,
    LmqcPayloadKind, OpenedLmqcBcs2, LMQC_READER_CAPABILITIES,
};

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use semantic_abir::{
    canonical_debug_json, verify_payload_content, AbirDataset, Atom, ByteOrder, ContentId,
    ElementType, Layout, PayloadAccess, PayloadDescriptor, PayloadLease, Presence,
};
use semantic_abir_bcs::{
    encode_codec_bundle, Bcs2View, CodecBundleError, CodecBundleInput, CodecBundleView,
    CodecFidelity, CodecFidelityKind, CodecImplementation, CodecParameter, CodecParameterValue,
    CodecProfile, ResourceBounds, CAP_LML_ARITHMETIC_V1,
};

/// Stable algorithm identity. Build-specific identity is recorded separately.
pub const LML_KERNEL_ID: &str = "org.quitetall.lamquant.lml-mcu.lossless-v1";

/// Kernel id for the Optimum (LMO) tier.
///
/// A DISTINCT kernel id under the SAME profile, which is the whole shape of the
/// decision recorded when `CAP_LML_OPTIMUM_V1` was allocated: optimum produces
/// the same semantics as baseline LML -- exact fidelity, the same decoded
/// signal -- from a materially different bitstream. Profiles answer "what kind
/// of artifact is this"; the kernel id and the capability answer "can this
/// reader decode it at all". Giving optimum its own profile would have implied
/// a different KIND of artifact where there is only a different compressor.
#[cfg(feature = "optimum")]
pub const OPTIMUM_KERNEL_ID: &str = "org.quitetall.lamquant.lml-optimum.lossless-v1";
/// Exact semantic-to-packet closure enforced by this integration crate.
pub const LML_FIDELITY_CONTRACT: &str =
    "org.quitetall.lamquant.bcs2.lml.exact-signal-block-closure-v1";
const SOURCE_ID: &str = env!("LAMQUANT_ABIR_CODEC_SOURCE_ID");
const BUILD_ID: &str = env!("LAMQUANT_ABIR_CODEC_BUILD_ID");
// Existing LML/Optimum BCS2 catalogs use this as a wire-contract marker.
// Dependency updates do not rewrite it: old artifacts must remain readable.
const LML_WIRE_ABIR_REVISION: &str = "c101513167ad8d7cdefa6387b20c644fdaf66432";
// Linked source identity may advance independently from stable wire contracts.
const LINKED_ABIR_REVISION: &str = "a02ad44fa36899dcb7d53d95c9e640f17e885ffc";
const HEADER_SIZE: usize = 22;
/// Maximum sample count representable by one LML1 packet header.
pub const MAX_PACKET_SAMPLES: usize = u16::MAX as usize;
/// Maximum channel count accepted by the LML1 packet decoder.
pub const MAX_PACKET_CHANNELS: usize = 1024;
/// Maximum decoded matrix retained by one BCS2 LML bundle.
///
/// This mirrors the LML decoder's one-gibibyte allocation ceiling across the
/// complete packet sequence. Canonical ABIR descriptors can declare much more
/// logical data than the encoded packet frames occupy, so the trust envelope
/// must reject that expansion before reserving the output matrix.
const MAX_DECODED_BUNDLE_BYTES: usize = 1024 * 1024 * 1024;
/// Maximum encoded BCS2 bundle produced by graph-facing explicit encoders.
pub const MAX_ENCODED_BUNDLE_BYTES: usize = 1024 * 1024 * 1024;
const BCS2_HEADER_BYTES: usize = 128;
const BCS2_INDEX_HEADER_BYTES: usize = 48;
const BCS2_INDEX_ENTRY_BYTES: usize = 128;
/// Conservative logical-output ceiling for Optimum bundles.
///
/// Optimum transform-2 decode retains residual/reconstruction scratch in
/// addition to returned samples. Keeping logical output at 64 MiB bounds peak
/// working memory to a host-safe envelope until streaming decode lands.
#[cfg(feature = "optimum")]
const MAX_OPTIMUM_DECODED_BUNDLE_BYTES: usize = 64 * 1024 * 1024;
#[cfg(feature = "optimum")]
const LMO_HEADER_SIZE: usize = 7;
#[cfg(feature = "optimum")]
const LMO_VERSION: u8 = 2;
#[cfg(feature = "optimum")]
const LMO_MODE_LOSSLESS: u8 = 0;
#[cfg(feature = "optimum")]
const LMO_TRANSFORM_LML_53: u8 = 0;
#[cfg(feature = "optimum")]
const LMO_TRANSFORM_OPTIMUM_LOSSLESS: u8 = 2;

/// A validated BCS2 LML bundle with decoded, semantics-verified samples.
#[derive(Debug)]
pub struct OpenedLmlBundle<'a> {
    bundle: CodecBundleView<'a>,
    packet_sample_counts: Vec<usize>,
    signal: Vec<Vec<i64>>,
}

impl<'a> OpenedLmlBundle<'a> {
    pub const fn dataset(&self) -> &AbirDataset {
        self.bundle.dataset()
    }

    pub fn signal(&self) -> &[Vec<i64>] {
        &self.signal
    }

    pub fn packet(&self) -> &'a [u8] {
        self.bundle
            .packet(0)
            .expect("validated LML bundles contain at least one packet")
    }

    pub fn packets(&self) -> impl ExactSizeIterator<Item = &'a [u8]> + '_ {
        self.bundle.packets()
    }

    pub fn packet_sample_counts(&self) -> &[usize] {
        &self.packet_sample_counts
    }

    pub const fn bundle(&self) -> &CodecBundleView<'a> {
        &self.bundle
    }
}

/// Validated optimum BCS2 bundle with exact decoded samples.
#[cfg(feature = "optimum")]
#[derive(Debug)]
pub struct OpenedOptimumBundle<'a> {
    bundle: CodecBundleView<'a>,
    packet_sample_counts: Vec<usize>,
    signal: Vec<Vec<i64>>,
}

#[cfg(feature = "optimum")]
impl<'a> OpenedOptimumBundle<'a> {
    pub const fn dataset(&self) -> &AbirDataset {
        self.bundle.dataset()
    }

    pub fn signal(&self) -> &[Vec<i64>] {
        &self.signal
    }

    pub fn packet(&self) -> &'a [u8] {
        self.bundle
            .packet(0)
            .expect("validated optimum bundles contain at least one packet")
    }

    pub fn packets(&self) -> impl ExactSizeIterator<Item = &'a [u8]> + '_ {
        self.bundle.packets()
    }

    pub fn packet_sample_counts(&self) -> &[usize] {
        &self.packet_sample_counts
    }

    pub const fn bundle(&self) -> &CodecBundleView<'a> {
        &self.bundle
    }
}

/// Encode the supported uniform integer SignalBlock subset with the existing
/// LML1 lossless kernel, then seal it as `bcs.lml.lossless.v1`.
pub fn encode_lml_bundle<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    encode_lml_bundle_with_window_size(dataset, access, MAX_PACKET_SAMPLES, bounds)
}

/// Encode a uniform integer dataset into an ordered sequence of bounded LML1
/// packets inside one authenticated BCS2 bundle.
pub fn encode_lml_bundle_with_window_size<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    window_size: usize,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    encode_lml_bundle_with_window_size_and_mode(
        dataset,
        access,
        window_size,
        lamquant_lml_mcu::lpc::LpcMode::default(),
        bounds,
    )
}

/// Encode a uniform integer dataset with explicit packet extent and predictor
/// mode while resolving samples exclusively through the ABIR payload contract.
pub fn encode_lml_bundle_with_window_size_and_mode<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    if window_size == 0 {
        return Err(LmlBundleError::PacketExtent);
    }
    let descriptors = ordered_descriptors(dataset)?;
    let mut signal = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let lease = access
            .lease(descriptor)
            .map_err(LmlBundleError::PayloadAccess)?;
        verify_payload_content(descriptor, lease.bytes())
            .map_err(|_| LmlBundleError::PayloadIdentityMismatch)?;
        signal.push(decode_integer_payload(descriptor, lease.bytes())?);
    }
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    encode_lml_bundle_from_verified_signal(
        dataset,
        &views,
        window_size,
        mode,
        EncoderSelection::Ambient,
        bounds,
    )
}

/// Encode caller-owned samples after proving they match the ABIR descriptors.
/// This avoids decoding another full matrix from an in-memory resolver.
pub fn encode_lml_bundle_from_signal(
    dataset: &AbirDataset,
    signal: &[Vec<i64>],
    window_size: usize,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    encode_lml_bundle_from_signal_with_mode(
        dataset,
        signal,
        window_size,
        lamquant_lml_mcu::lpc::LpcMode::default(),
        bounds,
    )
}

/// Encode caller-owned samples with an explicit LML predictor mode after
/// proving they match the ABIR descriptors.
pub fn encode_lml_bundle_from_signal_with_mode(
    dataset: &AbirDataset,
    signal: &[Vec<i64>],
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    encode_lml_bundle_from_views_with_mode(dataset, &views, window_size, mode, bounds)
}

/// Encode borrowed channel views without copying their sample matrices.
///
/// Descriptor closure is proved before the first packet is produced. Only the
/// small outer slice table and bounded packet buffers allocate.
pub fn encode_lml_bundle_from_views_with_mode(
    dataset: &AbirDataset,
    signal: &[&[i64]],
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    verify_lml_signal_views_closure(dataset, signal)?;
    encode_lml_bundle_from_verified_signal(
        dataset,
        signal,
        window_size,
        mode,
        EncoderSelection::Ambient,
        bounds,
    )
}

/// Encode borrowed views with explicit experimental encoder choices.
///
/// This is the deterministic graph-runtime seam: no process environment
/// variable can alter packet selection.
pub fn encode_lml_bundle_from_views_explicit(
    dataset: &AbirDataset,
    signal: &[&[i64]],
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    features: lamquant_lml_mcu::lml::EncodeFeatures,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    verify_lml_signal_views_closure(dataset, signal)?;
    encode_lml_bundle_from_verified_signal(
        dataset,
        signal,
        window_size,
        mode,
        EncoderSelection::Explicit(features),
        bounds,
    )
}

#[derive(Clone, Copy)]
enum EncoderSelection {
    Ambient,
    Explicit(lamquant_lml_mcu::lml::EncodeFeatures),
}

fn encode_lml_bundle_from_verified_signal(
    dataset: &AbirDataset,
    signal: &[&[i64]],
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    selection: EncoderSelection,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    let packet_samples = window_size.min(MAX_PACKET_SAMPLES);
    if packet_samples == 0 {
        return Err(LmlBundleError::PacketExtent);
    }
    let total_samples = signal.first().map_or(0, |channel| channel.len());
    let packet_count = total_samples.div_ceil(packet_samples);
    let semantics = canonical_debug_json(dataset).map_err(|_| LmlBundleError::SemanticEncoding)?;
    let packet_budget = encoded_packet_budget(semantics.len(), packet_count, bounds)?;
    let per_packet_budget = packet_budget
        .checked_div(packet_count.max(1))
        .map(|budget| budget.min(bounds.max_frame_bytes as usize))
        .filter(|budget| *budget > 0)
        .ok_or(LmlBundleError::EncodedResourceLimit)?;
    let bounded_selection = match selection {
        EncoderSelection::Ambient => EncoderSelection::Ambient,
        EncoderSelection::Explicit(mut features) => {
            features.max_packet_bytes = Some(
                features
                    .max_packet_bytes
                    .map_or(per_packet_budget, |limit| limit.min(per_packet_budget)),
            );
            EncoderSelection::Explicit(features)
        }
    };
    let mut encoded_packet_bytes = 0usize;
    let mut packets = Vec::with_capacity(packet_count);
    for start in (0..total_samples).step_by(packet_samples) {
        let end = start.saturating_add(packet_samples).min(total_samples);
        let window = signal
            .iter()
            .map(|channel| &channel[start..end])
            .collect::<Vec<_>>();
        let packet = compress_views(&window, mode, bounded_selection)?;
        encoded_packet_bytes = encoded_packet_bytes
            .checked_add(packet.len())
            .filter(|length| *length <= packet_budget)
            .ok_or(LmlBundleError::EncodedResourceLimit)?;
        packets.push(packet);
    }
    let packet_refs = packets.iter().map(Vec::as_slice).collect::<Vec<_>>();
    encode_verified_packets_with_semantics(&semantics, &packet_refs, bounds)
}

fn encoded_packet_budget(
    semantics_bytes: usize,
    packet_count: usize,
    bounds: ResourceBounds,
) -> Result<usize, LmlBundleError> {
    if semantics_bytes > bounds.max_frame_bytes as usize {
        return Err(LmlBundleError::EncodedResourceLimit);
    }
    let frame_count = packet_count
        .checked_add(1)
        .filter(|count| *count <= bounds.max_index_entries as usize)
        .ok_or(LmlBundleError::EncodedResourceLimit)?;
    let catalog_reserve = usize::try_from(bounds.max_catalog_bytes)
        .map_err(|_| LmlBundleError::EncodedResourceLimit)?;
    let index_reserve = frame_count
        .checked_mul(BCS2_INDEX_ENTRY_BYTES)
        .and_then(|entries| entries.checked_add(BCS2_INDEX_HEADER_BYTES))
        .ok_or(LmlBundleError::EncodedResourceLimit)?;
    let nonpacket_reserve = BCS2_HEADER_BYTES
        .checked_add(catalog_reserve)
        .and_then(|value| value.checked_add(semantics_bytes))
        .and_then(|value| value.checked_add(index_reserve))
        .ok_or(LmlBundleError::EncodedResourceLimit)?;
    MAX_ENCODED_BUNDLE_BYTES
        .checked_sub(nonpacket_reserve)
        .ok_or(LmlBundleError::EncodedResourceLimit)
}

#[cfg(feature = "std")]
fn compress_views(
    signal: &[&[i64]],
    mode: lamquant_lml_mcu::lpc::LpcMode,
    selection: EncoderSelection,
) -> Result<Vec<u8>, LmlBundleError> {
    use lamquant_lml_desktop::backend::{global_backend, ComputeBackend};

    // Rayon workers observe a live deadline at different instants, which can
    // select different predictor schedules. Keep live-deadline encoding on the
    // serial reference path; Fixed, Adaptive, and deadline-free Anytime remain
    // byte-equal on either backend.
    if matches!(
        mode,
        lamquant_lml_mcu::lpc::LpcMode::Anytime {
            deadline: Some(_),
            ..
        }
    ) {
        let result = match selection {
            EncoderSelection::Ambient => {
                lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode)
            }
            EncoderSelection::Explicit(features) => {
                lamquant_lml_mcu::lml::compress_with_mode_views_explicit(signal, 0, mode, features)
            }
        };
        return result.map_err(LmlBundleError::Lml);
    }

    let result = match global_backend() {
        ComputeBackend::Firmware => match selection {
            EncoderSelection::Ambient => {
                lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode)
            }
            EncoderSelection::Explicit(features) => {
                lamquant_lml_mcu::lml::compress_with_mode_views_explicit(signal, 0, mode, features)
            }
        },
        ComputeBackend::Desktop => match selection {
            EncoderSelection::Ambient => {
                lamquant_lml_desktop::compress_with_mode_parallel_views(signal, 0, mode)
            }
            EncoderSelection::Explicit(features) => {
                lamquant_lml_desktop::compress_with_mode_parallel_views_explicit(
                    signal, 0, mode, features,
                )
            }
        },
    };
    result.map_err(LmlBundleError::Lml)
}

#[cfg(not(feature = "std"))]
fn compress_views(
    signal: &[&[i64]],
    mode: lamquant_lml_mcu::lpc::LpcMode,
    selection: EncoderSelection,
) -> Result<Vec<u8>, LmlBundleError> {
    match selection {
        EncoderSelection::Ambient => {
            lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode)
        }
        EncoderSelection::Explicit(features) => {
            lamquant_lml_mcu::lml::compress_with_mode_views_explicit(signal, 0, mode, features)
        }
    }
    .map_err(LmlBundleError::Lml)
}

/// Seal one pre-existing LML1 packet after proving that its exact decoded
/// samples close over the supplied ABIR payload descriptors.
pub fn seal_lml_packet(
    dataset: &AbirDataset,
    packet: &[u8],
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    seal_lml_packets(dataset, &[packet], bounds)
}

/// Seal an ordered sequence of exact LML1 packets after proving that their
/// concatenated reconstruction closes over the supplied ABIR payloads.
pub fn seal_lml_packets(
    dataset: &AbirDataset,
    packets: &[&[u8]],
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    if packets.is_empty() {
        return Err(LmlBundleError::PacketCount);
    }
    let (signal, _) = decode_packet_sequence(dataset, packets.iter().copied())?;
    verify_signal_closure(dataset, &signal)?;
    encode_verified_packets(dataset, packets, bounds)
}

fn encode_verified_packets(
    dataset: &AbirDataset,
    packets: &[&[u8]],
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    let semantics = canonical_debug_json(dataset).map_err(|_| LmlBundleError::SemanticEncoding)?;
    encode_verified_packets_with_semantics(&semantics, packets, bounds)
}

fn encode_verified_packets_with_semantics(
    semantics: &[u8],
    packets: &[&[u8]],
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    for packet in packets {
        validate_strict_lossless_packet(packet)?;
    }
    let required_capabilities = if packets
        .iter()
        .any(|packet| lamquant_lml_mcu::lml::requires_arithmetic_coders(packet))
    {
        CAP_LML_ARITHMETIC_V1
    } else {
        0
    };
    let encoded = encode_codec_bundle(
        CodecBundleInput {
            // Capability follows packet bytes, not producer build features.
            required_capabilities,
            canonical_semantics: semantics,
            fidelity: exact_fidelity(),
            implementation: implementation_identity(),
            model_provenance: None,
            packets,
            parameters: canonical_parameters(),
            profile: CodecProfile::LmlLossless,
        },
        bounds,
    )
    .map_err(LmlBundleError::Bundle)?;
    if encoded.len() > MAX_ENCODED_BUNDLE_BYTES {
        return Err(LmlBundleError::EncodedResourceLimit);
    }
    Ok(encoded)
}

/// Open, authenticate, decode, and prove semantic closure before returning a
/// packet or reconstructed samples.
pub fn open_lml_bundle(
    bytes: &[u8],
    bounds: ResourceBounds,
) -> Result<OpenedLmlBundle<'_>, LmlBundleError> {
    let bundle = CodecBundleView::open_with_capabilities(bytes, lml_reader_capabilities(), bounds)
        .map_err(LmlBundleError::Bundle)?;
    validate_catalog(&bundle)?;
    validate_packet_capabilities(bytes, &bundle, bounds)?;
    let (signal, packet_sample_counts) =
        decode_packet_sequence(bundle.dataset(), bundle.packets())?;
    verify_signal_closure(bundle.dataset(), &signal)?;
    Ok(OpenedLmlBundle {
        bundle,
        packet_sample_counts,
        signal,
    })
}

fn validate_packet_capabilities(
    bytes: &[u8],
    bundle: &CodecBundleView<'_>,
    bounds: ResourceBounds,
) -> Result<(), LmlBundleError> {
    let required = if bundle
        .packets()
        .any(lamquant_lml_mcu::lml::requires_arithmetic_coders)
    {
        CAP_LML_ARITHMETIC_V1
    } else {
        0
    };
    let wire = Bcs2View::parse(bytes, lml_reader_capabilities(), bounds)
        .map_err(|error| LmlBundleError::Bundle(CodecBundleError::Bcs2(error)))?;
    let packet_ids = (0..bundle.catalog().packet_count())
        .map(|ordinal| {
            bundle
                .catalog()
                .packet_content_id(ordinal)
                .ok_or(LmlBundleError::CatalogContract)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    for frame in wire.frames() {
        let packet_frame = packet_ids.contains(&frame.content_id());
        let expected = if packet_frame { required } else { 0 };
        if frame.required_capabilities() != expected {
            return Err(LmlBundleError::CatalogContract);
        }
    }
    Ok(())
}

const fn lml_reader_capabilities() -> u64 {
    #[cfg(feature = "experimental-arithmetic")]
    {
        CAP_LML_ARITHMETIC_V1
    }
    #[cfg(not(feature = "experimental-arithmetic"))]
    {
        0
    }
}

fn validate_catalog(bundle: &CodecBundleView<'_>) -> Result<(), LmlBundleError> {
    let catalog = bundle.catalog();
    if catalog.profile() != CodecProfile::LmlLossless || catalog.packet_count() == 0 {
        return Err(LmlBundleError::PacketCount);
    }
    if catalog.model_provenance().is_some()
        || catalog.fidelity() != &exact_fidelity()
        || catalog.implementation().kernel_id != LML_KERNEL_ID
        || catalog.parameters() != canonical_parameters()
    {
        return Err(LmlBundleError::CatalogContract);
    }
    Ok(())
}

fn decode_packet_sequence<'a>(
    dataset: &AbirDataset,
    packets: impl ExactSizeIterator<Item = &'a [u8]>,
) -> Result<(Vec<Vec<i64>>, Vec<usize>), LmlBundleError> {
    let descriptors = ordered_descriptors(dataset)?;
    let expected_samples = descriptors
        .first()
        .and_then(|descriptor| descriptor.shape().last())
        .copied()
        .and_then(|samples| usize::try_from(samples).ok())
        .ok_or(LmlBundleError::SignalShapeMismatch)?;
    let decoded_bytes = descriptors
        .len()
        .checked_mul(expected_samples)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<i64>()))
        .ok_or(LmlBundleError::DecodedResourceLimit)?;
    if decoded_bytes > MAX_DECODED_BUNDLE_BYTES {
        return Err(LmlBundleError::DecodedResourceLimit);
    }
    let mut signal = (0..descriptors.len())
        .map(|_| {
            let mut channel = Vec::new();
            channel
                .try_reserve_exact(expected_samples)
                .map_err(|_| LmlBundleError::DecodedResourceLimit)?;
            Ok(channel)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut packet_sample_counts = Vec::with_capacity(packets.len());
    for packet in packets {
        validate_strict_lossless_packet(packet)?;
        let window = lamquant_lml_mcu::lml::decompress(packet).map_err(LmlBundleError::Lml)?;
        if window.len() != signal.len() {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        let samples = window.first().map_or(0, Vec::len);
        if samples == 0 || window.iter().any(|channel| channel.len() != samples) {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        for (output, channel) in signal.iter_mut().zip(window) {
            if output.len().saturating_add(samples) > expected_samples {
                return Err(LmlBundleError::SignalShapeMismatch);
            }
            output.extend(channel);
        }
        packet_sample_counts.push(samples);
    }
    if signal
        .iter()
        .any(|channel| channel.len() != expected_samples)
    {
        return Err(LmlBundleError::SignalShapeMismatch);
    }
    Ok((signal, packet_sample_counts))
}

/// Reproducible identity of this integration build and the linked LML kernel
/// sources. The implementation identity is source-stable; the build identity
/// additionally binds target, Cargo profile, enabled features, and rustc.
pub fn implementation_identity() -> CodecImplementation {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.abir-codec.implementation-v1\0");
    hasher.update(SOURCE_ID.as_bytes());
    hasher.update(&[0]);
    hasher.update(LINKED_ABIR_REVISION.as_bytes());
    CodecImplementation {
        build_id: format!("blake3:{BUILD_ID}"),
        implementation_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        kernel_id: LML_KERNEL_ID.to_string(),
    }
}

/// Linked Optimum source/build identity.
///
/// Use this as producer identity only when the exact linked Optimum encoder
/// emitted the packets. [`seal_optimum_packets`] accepts identity explicitly so
/// importing pre-existing packets cannot silently claim this build produced
/// them.
#[cfg(feature = "optimum")]
pub fn optimum_implementation_identity() -> CodecImplementation {
    CodecImplementation {
        kernel_id: OPTIMUM_KERNEL_ID.to_string(),
        ..implementation_identity()
    }
}

/// Seal optimum-coded packets after proving they reconstruct the dataset.
///
/// The proof is the point. Sealing without it would produce a bundle asserting
/// that its packets reproduce the declared signal on the strength of nobody
/// having checked -- and the optimum bitstream is precisely the one whose
/// correctness cannot be eyeballed. So this decodes what it is about to seal,
/// exactly as the baseline path does, which is why the crate needs the optimum
/// decoder at all.
///
/// The resulting bundle declares `CAP_LML_OPTIMUM_V1`, so a baseline-only
/// reader is refused at the envelope instead of failing somewhere inside a
/// bitstream it was never able to parse.
#[cfg(feature = "optimum")]
pub fn seal_optimum_packets(
    dataset: &AbirDataset,
    packets: &[&[u8]],
    producer: CodecImplementation,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    use semantic_abir_bcs::CAP_LML_OPTIMUM_V1;

    if packets.is_empty() {
        return Err(LmlBundleError::PacketCount);
    }
    if producer.kernel_id != OPTIMUM_KERNEL_ID {
        return Err(LmlBundleError::CatalogContract);
    }
    let (signal, _) = decode_optimum_sequence(dataset, packets)?;
    verify_signal_closure(dataset, &signal)?;

    let semantics = canonical_debug_json(dataset).map_err(|_| LmlBundleError::SemanticEncoding)?;
    encode_codec_bundle(
        CodecBundleInput {
            required_capabilities: CAP_LML_OPTIMUM_V1,
            canonical_semantics: &semantics,
            fidelity: exact_fidelity(),
            implementation: producer,
            model_provenance: None,
            packets,
            parameters: canonical_parameters(),
            profile: CodecProfile::LmlLossless,
        },
        bounds,
    )
    .map_err(LmlBundleError::Bundle)
}

/// Open an optimum-coded bundle, authenticate it, and prove semantic closure.
///
/// A separate entry point rather than a flag on [`open_lml_bundle`], because
/// the two differ in what the caller must be able to do. `open_lml_bundle`
/// advertises no capabilities and therefore refuses an optimum bundle outright
/// -- the correct default, since a baseline-only consumer calling it is
/// asserting exactly that it cannot decode optimum.
#[cfg(feature = "optimum")]
pub fn open_optimum_bundle(
    bytes: &[u8],
    bounds: ResourceBounds,
) -> Result<OpenedOptimumBundle<'_>, LmlBundleError> {
    use semantic_abir_bcs::CAP_LML_OPTIMUM_V1;

    let bundle = CodecBundleView::open_with_capabilities(bytes, CAP_LML_OPTIMUM_V1, bounds)
        .map_err(LmlBundleError::Bundle)?;
    let catalog = bundle.catalog();
    if catalog.profile() != CodecProfile::LmlLossless {
        return Err(LmlBundleError::CatalogContract);
    }
    if catalog.packet_count() == 0 {
        return Err(LmlBundleError::PacketCount);
    }
    if catalog.model_provenance().is_some()
        || catalog.fidelity() != &exact_fidelity()
        || catalog.implementation().kernel_id != OPTIMUM_KERNEL_ID
        || catalog.parameters() != canonical_parameters()
    {
        return Err(LmlBundleError::CatalogContract);
    }
    let wire = Bcs2View::parse(bytes, CAP_LML_OPTIMUM_V1, bounds)
        .map_err(|error| LmlBundleError::Bundle(CodecBundleError::Bcs2(error)))?;
    let packet_ids = (0..catalog.packet_count())
        .map(|ordinal| {
            catalog
                .packet_content_id(ordinal)
                .ok_or(LmlBundleError::CatalogContract)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut seen_packet_ids = BTreeSet::new();
    for frame in wire.frames() {
        if packet_ids.contains(&frame.content_id()) {
            if frame.required_capabilities() != CAP_LML_OPTIMUM_V1 {
                return Err(LmlBundleError::CatalogContract);
            }
            seen_packet_ids.insert(frame.content_id());
        } else if frame.required_capabilities() != 0 {
            return Err(LmlBundleError::CatalogContract);
        }
    }
    if seen_packet_ids != packet_ids {
        return Err(LmlBundleError::CatalogContract);
    }
    let packets: Vec<&[u8]> = bundle.packets().collect();
    let (signal, packet_sample_counts) = decode_optimum_sequence(bundle.dataset(), &packets)?;
    verify_signal_closure(bundle.dataset(), &signal)?;
    Ok(OpenedOptimumBundle {
        bundle,
        packet_sample_counts,
        signal,
    })
}

/// Decode an ordered optimum packet sequence into one concatenated signal.
///
/// Channel count must agree across packets: a sequence whose packets disagree
/// does not describe one recording, and concatenating it anyway would produce a
/// signal that passes closure by accident on the first channel.
#[cfg(feature = "optimum")]
fn decode_optimum_sequence(
    dataset: &AbirDataset,
    packets: &[&[u8]],
) -> Result<(Vec<Vec<i64>>, Vec<usize>), LmlBundleError> {
    let descriptors = ordered_descriptors(dataset)?;
    let expected_samples = descriptors
        .first()
        .and_then(|descriptor| descriptor.shape().last())
        .copied()
        .and_then(|samples| usize::try_from(samples).ok())
        .ok_or(LmlBundleError::SignalShapeMismatch)?;
    let decoded_bytes = descriptors
        .len()
        .checked_mul(expected_samples)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<i64>()))
        .ok_or(LmlBundleError::DecodedResourceLimit)?;
    if decoded_bytes > MAX_OPTIMUM_DECODED_BUNDLE_BYTES {
        return Err(LmlBundleError::DecodedResourceLimit);
    }
    let mut signal: Option<Vec<Vec<i64>>> = None;
    let mut packet_sample_counts = Vec::with_capacity(packets.len());

    for packet in packets {
        validate_strict_optimum_packet(packet)?;
        let accumulated = signal
            .as_ref()
            .and_then(|channels| channels.first())
            .map_or(0, Vec::len);
        let remaining = expected_samples
            .checked_sub(accumulated)
            .ok_or(LmlBundleError::SignalShapeMismatch)?;
        let decoded =
            lamquant_lml_optimum::decode_lossless_bounded(packet, descriptors.len(), remaining)
                .map_err(LmlBundleError::Optimum)?;
        if decoded.len() != descriptors.len() {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        let samples = decoded.first().map_or(0, Vec::len);
        if samples == 0 || decoded.iter().any(|channel| channel.len() != samples) {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        let end = accumulated
            .checked_add(samples)
            .ok_or(LmlBundleError::DecodedResourceLimit)?;
        if end > expected_samples {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        if let Some(output) = signal.as_mut() {
            for (channel, more) in output.iter_mut().zip(decoded) {
                channel
                    .try_reserve_exact(more.len())
                    .map_err(|_| LmlBundleError::DecodedResourceLimit)?;
                channel.extend(more);
            }
        } else {
            signal = Some(decoded);
        }
        packet_sample_counts.push(samples);
    }
    let signal = signal.ok_or(LmlBundleError::PacketCount)?;
    if signal
        .iter()
        .any(|channel| channel.len() != expected_samples)
    {
        return Err(LmlBundleError::SignalShapeMismatch);
    }
    Ok((signal, packet_sample_counts))
}

#[cfg(feature = "optimum")]
fn validate_strict_optimum_packet(packet: &[u8]) -> Result<(), LmlBundleError> {
    if packet.len() < LMO_HEADER_SIZE || &packet[..4] != b"LMO1" {
        return Err(LmlBundleError::NotOptimum);
    }
    if packet[4] != LMO_VERSION {
        return Err(LmlBundleError::NotOptimum);
    }
    if packet[5] != LMO_MODE_LOSSLESS {
        return Err(LmlBundleError::NotExactLossless);
    }
    match packet[6] {
        LMO_TRANSFORM_LML_53 => validate_strict_lossless_packet(&packet[LMO_HEADER_SIZE..]),
        LMO_TRANSFORM_OPTIMUM_LOSSLESS => Ok(()),
        _ => Err(LmlBundleError::NotExactLossless),
    }
}

fn exact_fidelity() -> CodecFidelity {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"org.quitetall.lamquant.abir-codec.fidelity-v1\0");
    hasher.update(LML_FIDELITY_CONTRACT.as_bytes());
    CodecFidelity {
        bound: None,
        contract_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        kind: CodecFidelityKind::Exact,
        metric: None,
    }
}

fn canonical_parameters() -> Vec<CodecParameter> {
    vec![
        CodecParameter {
            name: "abir.revision".to_string(),
            value: CodecParameterValue::Text {
                value: LML_WIRE_ABIR_REVISION.to_string(),
            },
        },
        CodecParameter {
            name: "lml.noise_bits".to_string(),
            value: CodecParameterValue::Integer {
                value: "0".to_string(),
            },
        },
        CodecParameter {
            name: "lml.packet_grammar".to_string(),
            value: CodecParameterValue::Text {
                value: "LML1".to_string(),
            },
        },
        CodecParameter {
            name: "semantic.closure".to_string(),
            value: CodecParameterValue::Text {
                value: LML_FIDELITY_CONTRACT.to_string(),
            },
        },
    ]
}

fn ordered_descriptors(dataset: &AbirDataset) -> Result<Vec<&PayloadDescriptor>, LmlBundleError> {
    if dataset.recordings().len() != 1 || dataset.streams().len() != 1 {
        return Err(LmlBundleError::UnsupportedSemantics(
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
        return Err(LmlBundleError::UnsupportedSemantics(
            "stream must own every atom exactly once",
        ));
    }
    if stream.atoms().len() > MAX_PACKET_CHANNELS {
        return Err(LmlBundleError::DecodedResourceLimit);
    }
    let mut descriptors = Vec::with_capacity(stream.atoms().len());
    let mut samples = None;
    for atom_id in stream.atoms() {
        if descriptors.iter().any(|(id, _)| *id == atom_id) {
            return Err(LmlBundleError::UnsupportedSemantics(
                "duplicate stream atom",
            ));
        }
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or(LmlBundleError::UnsupportedSemantics(
                "unresolved stream atom",
            ))?;
        if !matches!(atom, Atom::SignalBlock(_)) || atom.presence() != Presence::Present {
            return Err(LmlBundleError::UnsupportedSemantics(
                "only present SignalBlock atoms are supported",
            ));
        }
        let descriptor = atom.payload().ok_or(LmlBundleError::UnsupportedSemantics(
            "signal has no payload",
        ))?;
        validate_descriptor(descriptor)?;
        let channel_samples = *descriptor
            .shape()
            .last()
            .ok_or(LmlBundleError::UnsupportedSemantics("empty payload shape"))?;
        if samples
            .replace(channel_samples)
            .is_some_and(|n| n != channel_samples)
        {
            return Err(LmlBundleError::UnsupportedSemantics(
                "LML requires a uniform sample count",
            ));
        }
        descriptors.push((atom_id, descriptor));
    }
    Ok(descriptors
        .into_iter()
        .map(|(_, descriptor)| descriptor)
        .collect())
}

fn validate_descriptor(descriptor: &PayloadDescriptor) -> Result<(), LmlBundleError> {
    if !matches!(
        descriptor.element(),
        ElementType::I8 | ElementType::I16 | ElementType::I24 | ElementType::I32 | ElementType::I64
    ) {
        return Err(LmlBundleError::UnsupportedSemantics(
            "LML exact profile supports signed integer samples only",
        ));
    }
    if !matches!(descriptor.byte_order(), ByteOrder::Little | ByteOrder::Big)
        || !matches!(
            descriptor.layout(),
            Layout::DenseRowMajor | Layout::DenseColumnMajor
        )
        || descriptor.encoding().is_some()
        || !matches!(descriptor.shape(), [_] | [1, _])
    {
        return Err(LmlBundleError::UnsupportedSemantics(
            "payload must be unencoded dense signed integers with shape [T] or [1,T]",
        ));
    }
    Ok(())
}

fn verify_signal_closure(dataset: &AbirDataset, signal: &[Vec<i64>]) -> Result<(), LmlBundleError> {
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    verify_lml_signal_views_closure(dataset, &views)
}

/// Verify that borrowed signed-integer channel views exactly match dataset
/// payload descriptors and logical content identities.
pub fn verify_lml_signal_views_closure(
    dataset: &AbirDataset,
    signal: &[&[i64]],
) -> Result<(), LmlBundleError> {
    let descriptors = ordered_descriptors(dataset)?;
    if descriptors.len() != signal.len() {
        return Err(LmlBundleError::SignalShapeMismatch);
    }
    for (descriptor, channel) in descriptors.into_iter().zip(signal) {
        if descriptor.shape().last().copied() != Some(channel.len() as u64) {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        verify_integer_payload_content(descriptor, channel)?;
    }
    Ok(())
}

fn verify_integer_payload_content(
    descriptor: &PayloadDescriptor,
    samples: &[i64],
) -> Result<(), LmlBundleError> {
    // I64 is the hot container representation. Hash its canonical bytes in
    // place instead of allocating another full channel-sized buffer.
    if descriptor.element() == ElementType::I64 {
        let logical_bytes = samples
            .len()
            .checked_mul(8)
            .ok_or(LmlBundleError::SignalShapeMismatch)?;
        if u64::try_from(logical_bytes).ok() != Some(descriptor.logical_bytes()) {
            return Err(LmlBundleError::SignalShapeMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"abir.semantic-v1.payload\0");
        hasher.update(descriptor.element().semantic_tag());
        hasher.update(&[0]);
        let mut buffer = [0_u8; 8 * 1024];
        for chunk in samples.chunks(1024) {
            for (index, &sample) in chunk.iter().enumerate() {
                let bytes = match descriptor.byte_order() {
                    ByteOrder::Little => sample.to_le_bytes(),
                    ByteOrder::Big => sample.to_be_bytes(),
                    ByteOrder::NotApplicable => return Err(LmlBundleError::SampleRange),
                };
                let offset = index * 8;
                buffer[offset..offset + 8].copy_from_slice(&bytes);
            }
            hasher.update(&buffer[..chunk.len() * 8]);
        }
        if ContentId::from_bytes(*hasher.finalize().as_bytes()) != descriptor.content_id() {
            return Err(LmlBundleError::PayloadIdentityMismatch);
        }
        return Ok(());
    }
    let bytes = encode_integer_payload(descriptor, samples)?;
    verify_payload_content(descriptor, &bytes).map_err(|_| LmlBundleError::PayloadIdentityMismatch)
}

fn decode_integer_payload(
    descriptor: &PayloadDescriptor,
    bytes: &[u8],
) -> Result<Vec<i64>, LmlBundleError> {
    let width = descriptor
        .element()
        .byte_width()
        .ok_or(LmlBundleError::UnsupportedSemantics(
            "nonfixed sample width",
        ))? as usize;
    if bytes.len() % width != 0 {
        return Err(LmlBundleError::SignalShapeMismatch);
    }
    bytes
        .chunks_exact(width)
        .map(|chunk| decode_integer(descriptor.element(), descriptor.byte_order(), chunk))
        .collect()
}

fn decode_integer(
    element: ElementType,
    order: ByteOrder,
    bytes: &[u8],
) -> Result<i64, LmlBundleError> {
    let value = match (element, order) {
        (ElementType::I8, _) => i8::from_ne_bytes([bytes[0]]) as i64,
        (ElementType::I16, ByteOrder::Little) => i16::from_le_bytes([bytes[0], bytes[1]]) as i64,
        (ElementType::I16, ByteOrder::Big) => i16::from_be_bytes([bytes[0], bytes[1]]) as i64,
        (ElementType::I24, ByteOrder::Little) => {
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
            (((raw << 8) as i32) >> 8) as i64
        }
        (ElementType::I24, ByteOrder::Big) => {
            let raw = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
            (((raw << 8) as i32) >> 8) as i64
        }
        (ElementType::I32, ByteOrder::Little) => {
            i32::from_le_bytes(bytes.try_into().expect("validated width")) as i64
        }
        (ElementType::I32, ByteOrder::Big) => {
            i32::from_be_bytes(bytes.try_into().expect("validated width")) as i64
        }
        (ElementType::I64, ByteOrder::Little) => {
            i64::from_le_bytes(bytes.try_into().expect("validated width"))
        }
        (ElementType::I64, ByteOrder::Big) => {
            i64::from_be_bytes(bytes.try_into().expect("validated width"))
        }
        _ => {
            return Err(LmlBundleError::UnsupportedSemantics(
                "unsupported integer format",
            ))
        }
    };
    Ok(value)
}

fn encode_integer_payload(
    descriptor: &PayloadDescriptor,
    samples: &[i64],
) -> Result<Vec<u8>, LmlBundleError> {
    let width = descriptor
        .element()
        .byte_width()
        .ok_or(LmlBundleError::UnsupportedSemantics(
            "nonfixed sample width",
        ))? as usize;
    let capacity = samples
        .len()
        .checked_mul(width)
        .ok_or(LmlBundleError::SignalShapeMismatch)?;
    let mut bytes = Vec::with_capacity(capacity);
    for &sample in samples {
        match (descriptor.element(), descriptor.byte_order()) {
            (ElementType::I8, _) => {
                bytes.push(i8::try_from(sample).map_err(|_| LmlBundleError::SampleRange)? as u8)
            }
            (ElementType::I16, ByteOrder::Little) => bytes.extend_from_slice(
                &i16::try_from(sample)
                    .map_err(|_| LmlBundleError::SampleRange)?
                    .to_le_bytes(),
            ),
            (ElementType::I16, ByteOrder::Big) => bytes.extend_from_slice(
                &i16::try_from(sample)
                    .map_err(|_| LmlBundleError::SampleRange)?
                    .to_be_bytes(),
            ),
            (ElementType::I24, order) => {
                let value = i32::try_from(sample).map_err(|_| LmlBundleError::SampleRange)?;
                if !(-8_388_608..=8_388_607).contains(&value) {
                    return Err(LmlBundleError::SampleRange);
                }
                let encoded = match order {
                    ByteOrder::Little => value.to_le_bytes(),
                    ByteOrder::Big => value.to_be_bytes(),
                    ByteOrder::NotApplicable => return Err(LmlBundleError::SampleRange),
                };
                match order {
                    ByteOrder::Little => bytes.extend_from_slice(&encoded[..3]),
                    ByteOrder::Big => bytes.extend_from_slice(&encoded[1..]),
                    ByteOrder::NotApplicable => unreachable!(),
                }
            }
            (ElementType::I32, ByteOrder::Little) => bytes.extend_from_slice(
                &i32::try_from(sample)
                    .map_err(|_| LmlBundleError::SampleRange)?
                    .to_le_bytes(),
            ),
            (ElementType::I32, ByteOrder::Big) => bytes.extend_from_slice(
                &i32::try_from(sample)
                    .map_err(|_| LmlBundleError::SampleRange)?
                    .to_be_bytes(),
            ),
            (ElementType::I64, ByteOrder::Little) => bytes.extend_from_slice(&sample.to_le_bytes()),
            (ElementType::I64, ByteOrder::Big) => bytes.extend_from_slice(&sample.to_be_bytes()),
            _ => {
                return Err(LmlBundleError::UnsupportedSemantics(
                    "unsupported integer format",
                ))
            }
        }
    }
    Ok(bytes)
}

fn validate_strict_lossless_packet(packet: &[u8]) -> Result<(), LmlBundleError> {
    let offset = find_magic(packet).ok_or(LmlBundleError::NotLml1)?;
    let header_end = offset
        .checked_add(HEADER_SIZE)
        .ok_or(LmlBundleError::PacketExtent)?;
    if packet.len() < header_end {
        return Err(LmlBundleError::PacketExtent);
    }
    let header = &packet[offset..header_end];
    let flags = header[9];
    if flags & 0x02 != 0 || flags >> 2 != 0 {
        return Err(LmlBundleError::NotExactLossless);
    }
    let lpc_len = u32::from_le_bytes(header[10..14].try_into().expect("fixed header")) as usize;
    let payload_len = u32::from_le_bytes(header[14..18].try_into().expect("fixed header")) as usize;
    let expected = header_end
        .checked_add(lpc_len)
        .and_then(|length| length.checked_add(payload_len))
        .ok_or(LmlBundleError::PacketExtent)?;
    if expected != packet.len() {
        return Err(LmlBundleError::PacketExtent);
    }
    Ok(())
}

fn find_magic(packet: &[u8]) -> Option<usize> {
    if packet.starts_with(b"LML1") {
        return Some(0);
    }
    for index in 0..packet.len().min(128) {
        if packet.get(index) == Some(&b'\n')
            && packet.get(index + 1..index + 5) == Some(&b"LML1"[..])
            && packet[..index]
                .iter()
                .all(|byte| (0x20..=0x7e).contains(byte))
        {
            return Some(index + 1);
        }
    }
    None
}

#[derive(Debug)]
pub enum LmlBundleError {
    Bundle(CodecBundleError),
    CatalogContract,
    DecodedResourceLimit,
    EncodedResourceLimit,
    Lml(lamquant_lml_mcu::error::LmlError),
    NotExactLossless,
    NotLml1,
    #[cfg(feature = "optimum")]
    NotOptimum,
    #[cfg(feature = "optimum")]
    Optimum(lamquant_lml_mcu::codec::CodecError),
    PacketCount,
    PacketExtent,
    PayloadAccess(semantic_abir::PayloadAccessError),
    PayloadIdentityMismatch,
    SampleRange,
    SemanticEncoding,
    SignalShapeMismatch,
    UnsupportedSemantics(&'static str),
}

impl fmt::Display for LmlBundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(error) => error.fmt(formatter),
            Self::Lml(error) => error.fmt(formatter),
            #[cfg(feature = "optimum")]
            Self::Optimum(error) => error.fmt(formatter),
            Self::PayloadAccess(error) => error.fmt(formatter),
            Self::UnsupportedSemantics(reason) => {
                write!(formatter, "unsupported LML ABIR semantics: {reason}")
            }
            other => write!(formatter, "LML ABIR bundle error: {other:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LmlBundleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use semantic_abir::{
        payload_content_id, AtomTag, ConceptId, DatasetDraft, DatasetTag, InMemoryPayloadAccess,
        ObjectId, OpenedDataset, Rational, Recording, RecordingTag, SignalBlock, Stream, StreamTag,
        TimeAxis, TimeSegment, ValidationLimits,
    };

    fn fixture() -> OpenedDataset<InMemoryPayloadAccess> {
        fixture_from_signal(&[
            (ElementType::I16, vec![1_i64, -2, 3, -4, 5, -6, 7, -8]),
            (
                ElementType::I24,
                vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            ),
            (
                ElementType::I64,
                vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
            ),
        ])
    }

    #[test]
    fn encoded_budget_reserves_every_bcs2_index_entry() {
        let bounds = ResourceBounds::default();
        let one_packet = encoded_packet_budget(1024, 1, bounds).unwrap();
        let many_packets = encoded_packet_budget(1024, 131_072, bounds).unwrap();

        assert_eq!(
            one_packet - many_packets,
            (131_072 - 1) * BCS2_INDEX_ENTRY_BYTES
        );
    }

    fn fixture_from_signal(
        signal: &[(ElementType, Vec<i64>)],
    ) -> OpenedDataset<InMemoryPayloadAccess> {
        assert!(!signal.is_empty());
        let sample_count = signal[0].1.len();
        assert!(
            sample_count > 0
                && signal
                    .iter()
                    .all(|(_, samples)| samples.len() == sample_count)
        );
        let dataset_id = ObjectId::<DatasetTag>::from_bytes([1; 16]);
        let recording_id = ObjectId::<RecordingTag>::from_bytes([2; 16]);
        let stream_id = ObjectId::<StreamTag>::from_bytes([3; 16]);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut access = InMemoryPayloadAccess::new();
        let mut atom_ids = Vec::new();
        for (index, (element, samples)) in signal.iter().enumerate() {
            let placeholder = PayloadDescriptor::new(
                ContentId::from_bytes([0; 32]),
                (samples.len() as u64) * element.byte_width().unwrap(),
                *element,
                ByteOrder::Little,
                vec![1, samples.len() as u64],
                Layout::DenseRowMajor,
                None,
                None,
            );
            let bytes = encode_integer_payload(&placeholder, samples).unwrap();
            let content_id = payload_content_id(*element, &bytes);
            let descriptor = PayloadDescriptor::new(
                content_id,
                bytes.len() as u64,
                *element,
                ByteOrder::Little,
                vec![1, samples.len() as u64],
                Layout::DenseRowMajor,
                None,
                None,
            );
            access.insert(content_id, bytes);
            let mut id = [0_u8; 16];
            id[15] = (index + 1) as u8;
            let atom_id = ObjectId::<AtomTag>::from_bytes(id);
            atom_ids.push(atom_id);
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                TimeAxis::Regular(
                    TimeSegment::new(
                        Rational::new(0, 1).unwrap(),
                        Rational::new(256, 1).unwrap(),
                        samples.len() as u64,
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

    #[cfg(feature = "optimum")]
    fn optimum_packet(signal: &[Vec<i64>]) -> Vec<u8> {
        let inner = lamquant_lml_mcu::lml::compress(signal, 0).expect("baseline LML packet");
        let mut packet = Vec::with_capacity(7 + inner.len());
        packet.extend_from_slice(b"LMO1");
        packet.extend_from_slice(&[2, 0, 0]);
        packet.extend_from_slice(&inner);
        packet
    }

    #[cfg(feature = "optimum")]
    fn real_optimum_packet(signal: &[Vec<i64>]) -> Vec<u8> {
        let packet = lamquant_lml_optimum::encode(signal, lamquant_lml_optimum::Mode::Lossless)
            .expect("real optimum packet");
        assert_eq!(
            packet.get(6),
            Some(&LMO_TRANSFORM_OPTIMUM_LOSSLESS),
            "fixture must exercise transform-2 rather than wrapped baseline LML"
        );
        packet
    }

    fn oversized_semantic_dataset() -> AbirDataset {
        let dataset_id = ObjectId::<DatasetTag>::from_bytes([11; 16]);
        let recording_id = ObjectId::<RecordingTag>::from_bytes([12; 16]);
        let stream_id = ObjectId::<StreamTag>::from_bytes([13; 16]);
        let atom_id = ObjectId::<AtomTag>::from_bytes([14; 16]);
        let samples = (MAX_DECODED_BUNDLE_BYTES / core::mem::size_of::<i64>() + 1) as u64;
        let descriptor = PayloadDescriptor::new(
            ContentId::from_bytes([15; 32]),
            samples * core::mem::size_of::<i64>() as u64,
            ElementType::I64,
            ByteOrder::Little,
            vec![1, samples],
            Layout::DenseRowMajor,
            None,
            None,
        );
        let mut draft = DatasetDraft::new(dataset_id);
        draft.add_atom(Atom::SignalBlock(SignalBlock::new(
            atom_id,
            Presence::Present,
            Some(descriptor),
            TimeAxis::Regular(
                TimeSegment::new(
                    Rational::new(0, 1).unwrap(),
                    Rational::new(256, 1).unwrap(),
                    samples,
                )
                .unwrap(),
            ),
            None,
        )));
        draft.add_recording(Recording::new(recording_id, vec![stream_id]));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            ConceptId::new("abir:modality/eeg").unwrap(),
            vec![atom_id],
            None,
            None,
            None,
        ));
        draft.validate(ValidationLimits::default()).unwrap()
    }

    fn excessive_channel_dataset() -> AbirDataset {
        let dataset_id = ObjectId::<DatasetTag>::from_bytes([21; 16]);
        let recording_id = ObjectId::<RecordingTag>::from_bytes([22; 16]);
        let stream_id = ObjectId::<StreamTag>::from_bytes([23; 16]);
        let descriptor = PayloadDescriptor::new(
            ContentId::from_bytes([24; 32]),
            core::mem::size_of::<i64>() as u64,
            ElementType::I64,
            ByteOrder::Little,
            vec![1, 1],
            Layout::DenseRowMajor,
            None,
            None,
        );
        let mut draft = DatasetDraft::new(dataset_id);
        let mut atom_ids = Vec::with_capacity(MAX_PACKET_CHANNELS + 1);
        for index in 0..=MAX_PACKET_CHANNELS {
            let mut id = [0_u8; 16];
            id[0] = 25;
            id[8..].copy_from_slice(&(index as u64).to_be_bytes());
            let atom_id = ObjectId::<AtomTag>::from_bytes(id);
            atom_ids.push(atom_id);
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor.clone()),
                TimeAxis::Regular(
                    TimeSegment::new(
                        Rational::new(0, 1).unwrap(),
                        Rational::new(256, 1).unwrap(),
                        1,
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
        draft.validate(ValidationLimits::default()).unwrap()
    }

    #[test]
    fn existing_lml_packet_is_unchanged_inside_bundle_and_reopens_exactly() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = lamquant_lml_mcu::lml::compress(&signal, 0).expect("LML packet");
        let bytes = seal_lml_packet(mapped.dataset(), &packet, ResourceBounds::default())
            .expect("BCS2 bundle");
        let opened = open_lml_bundle(&bytes, ResourceBounds::default()).expect("open bundle");
        assert_eq!(opened.packet(), packet);
        assert_eq!(opened.signal(), signal);
        assert_eq!(
            canonical_debug_json(opened.dataset()).unwrap(),
            canonical_debug_json(mapped.dataset()).unwrap()
        );
        assert!(core::ptr::eq(opened.dataset(), opened.bundle().dataset()));
    }

    #[cfg(feature = "experimental-arithmetic")]
    #[test]
    fn baseline_packet_rejects_overdeclared_arithmetic_capability() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = lamquant_lml_mcu::lml::compress(&signal, 0).expect("LML packet");
        let semantics = canonical_debug_json(mapped.dataset()).expect("canonical semantics");
        let bytes = encode_codec_bundle(
            CodecBundleInput {
                required_capabilities: CAP_LML_ARITHMETIC_V1,
                canonical_semantics: &semantics,
                fidelity: exact_fidelity(),
                implementation: implementation_identity(),
                model_provenance: None,
                packets: &[packet.as_slice()],
                parameters: canonical_parameters(),
                profile: CodecProfile::LmlLossless,
            },
            ResourceBounds::default(),
        )
        .expect("syntactically valid overdeclared bundle");

        assert!(matches!(
            open_lml_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::CatalogContract)
        ));
    }

    #[cfg(feature = "experimental-arithmetic")]
    #[test]
    fn selected_arithmetic_packet_declares_capability_and_baseline_reader_refuses_it() {
        let signal = vec![0_i64; 4096];
        let mapped = fixture_from_signal(&[(ElementType::I64, signal.clone())]);
        let views = [signal.as_slice()];
        let bytes = encode_lml_bundle_from_views_explicit(
            mapped.dataset(),
            &views,
            u16::MAX as usize,
            lamquant_lml_mcu::lpc::LpcMode::Fixed,
            lamquant_lml_mcu::lml::EncodeFeatures {
                arithmetic: true,
                ..lamquant_lml_mcu::lml::EncodeFeatures::default()
            },
            ResourceBounds::default(),
        )
        .expect("arithmetic bundle");
        let opened =
            open_lml_bundle(&bytes, ResourceBounds::default()).expect("capable reader opens");

        assert!(opened
            .packets()
            .any(lamquant_lml_mcu::lml::requires_arithmetic_coders));
        assert!(
            CodecBundleView::open_with_capabilities(&bytes, 0, ResourceBounds::default()).is_err()
        );
    }

    #[test]
    fn frozen_pre_p8_c101_bundle_remains_readable() {
        let encoded = include_str!("../tests/fixtures/pre_p8_lml_bcs2.hex").trim();
        assert_eq!(encoded.len() % 2, 0);
        let bytes = encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = core::str::from_utf8(pair).expect("ASCII hex");
                u8::from_str_radix(pair, 16).expect("valid hex fixture")
            })
            .collect::<Vec<_>>();
        let opened =
            open_lml_bundle(&bytes, ResourceBounds::default()).expect("pre-P8 bundle opens");
        assert_eq!(
            opened.signal(),
            &[
                vec![1, -2, 3, -4, 5, -6, 7, -8],
                vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
                vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
            ]
        );
        assert_eq!(
            opened.bundle().catalog().parameters(),
            canonical_parameters()
        );
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_bundle_roundtrips_and_baseline_reader_refuses_it() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = optimum_packet(&signal);
        let bytes = seal_optimum_packets(
            mapped.dataset(),
            &[packet.as_slice()],
            optimum_implementation_identity(),
            ResourceBounds::default(),
        )
        .expect("optimum bundle");

        assert!(matches!(
            open_lml_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::Bundle(CodecBundleError::Bcs2(
                semantic_abir_bcs::Bcs2Error::UnsupportedCapabilities(
                    semantic_abir_bcs::CAP_LML_OPTIMUM_V1
                )
            )))
        ));
        let opened = open_optimum_bundle(&bytes, ResourceBounds::default()).expect("open optimum");
        assert_eq!(opened.signal(), signal);
        assert_eq!(opened.packet_sample_counts(), &[8]);
        assert_eq!(
            opened.bundle().catalog().implementation().kernel_id,
            OPTIMUM_KERNEL_ID
        );
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_sequence_rejects_disagreeing_channel_counts() {
        let mapped = fixture_from_signal(&[
            (ElementType::I64, vec![1, 2, 5, 6]),
            (ElementType::I64, vec![3, 4, 7, 8]),
        ]);
        let first = optimum_packet(&[vec![1, 2], vec![3, 4]]);
        let second = optimum_packet(&[vec![5, 6]]);
        assert!(matches!(
            decode_optimum_sequence(mapped.dataset(), &[first.as_slice(), second.as_slice()]),
            Err(LmlBundleError::Optimum(_))
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_sealer_rejects_raw_baseline_lml() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = lamquant_lml_mcu::lml::compress(&signal, 0).unwrap();
        assert!(matches!(
            seal_optimum_packets(
                mapped.dataset(),
                &[packet.as_slice()],
                optimum_implementation_identity(),
                ResourceBounds::default()
            ),
            Err(LmlBundleError::NotOptimum)
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_semantic_expansion_is_bounded_before_decode() {
        let packet = optimum_packet(&[vec![0]]);
        assert!(matches!(
            seal_optimum_packets(
                &oversized_semantic_dataset(),
                &[packet.as_slice()],
                optimum_implementation_identity(),
                ResourceBounds::default()
            ),
            Err(LmlBundleError::DecodedResourceLimit)
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_sequence_cannot_exceed_declared_sample_extent() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = optimum_packet(&signal);
        assert!(matches!(
            decode_optimum_sequence(mapped.dataset(), &[packet.as_slice(), packet.as_slice()]),
            Err(LmlBundleError::Optimum(_))
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn forged_inner_sample_extent_is_rejected_before_decode() {
        let mapped = fixture_from_signal(&[(ElementType::I64, vec![0; 8])]);
        let mut packet = b"LMO1\x02\x00\x02".to_vec();
        packet.extend_from_slice(&[3, 0, 1, 0]); // v3, feature=0, one channel
        packet.push(0); // channel 0 has no spatial references
        packet.push(1); // RLS residual coder
        packet.extend_from_slice(&1_u16.to_le_bytes());
        packet.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_optimum_sequence(mapped.dataset(), &[packet.as_slice()]),
            Err(LmlBundleError::Optimum(_))
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn forged_inner_encoded_extent_is_rejected_without_panicking() {
        let mapped = fixture_from_signal(&[(ElementType::I64, vec![0; 8])]);
        let mut packet = b"LMO1\x02\x00\x02".to_vec();
        packet.extend_from_slice(&[3, 0, 1, 0]); // v3, feature=0, one channel
        packet.push(0); // channel 0 has no spatial references
        packet.push(1); // RLS residual coder
        packet.extend_from_slice(&1_u16.to_le_bytes());
        packet.extend_from_slice(&8_u32.to_le_bytes());
        packet.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_optimum_sequence(mapped.dataset(), &[packet.as_slice()]),
            Err(LmlBundleError::Optimum(_))
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn transform_two_roundtrips_through_abir_closure() {
        let base = (0..2048)
            .map(|index| ((index * 17 + index * index * 3) % 4096) as i64 - 2048)
            .collect::<Vec<_>>();
        let signal = (0..8)
            .map(|channel| {
                base.iter()
                    .map(|sample| sample + channel as i64 * 7)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let semantic_signal = signal
            .iter()
            .cloned()
            .map(|channel| (ElementType::I64, channel))
            .collect::<Vec<_>>();
        let mapped = fixture_from_signal(&semantic_signal);
        let packet = real_optimum_packet(&signal);
        let bytes = seal_optimum_packets(
            mapped.dataset(),
            &[packet.as_slice()],
            optimum_implementation_identity(),
            ResourceBounds::default(),
        )
        .expect("transform-2 bundle");
        let opened =
            open_optimum_bundle(&bytes, ResourceBounds::default()).expect("open transform-2");
        assert_eq!(opened.signal(), signal);
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_open_rejects_missing_packet_capability() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = optimum_packet(&signal);
        let semantics = canonical_debug_json(mapped.dataset()).unwrap();
        let packets = [packet.as_slice()];
        let bytes = encode_codec_bundle(
            CodecBundleInput {
                required_capabilities: 0,
                canonical_semantics: &semantics,
                fidelity: exact_fidelity(),
                implementation: optimum_implementation_identity(),
                model_provenance: None,
                packets: &packets,
                parameters: canonical_parameters(),
                profile: CodecProfile::LmlLossless,
            },
            ResourceBounds::default(),
        )
        .unwrap();
        assert!(matches!(
            open_optimum_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::CatalogContract)
        ));
    }

    #[cfg(feature = "optimum")]
    #[test]
    fn optimum_open_reports_wrong_profile_as_catalog_contract() {
        use semantic_abir_bcs::CAP_LML_OPTIMUM_V1;

        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = optimum_packet(&signal);
        let semantics = canonical_debug_json(mapped.dataset()).unwrap();
        let packets = [packet.as_slice()];
        let bytes = encode_codec_bundle(
            CodecBundleInput {
                required_capabilities: CAP_LML_OPTIMUM_V1,
                canonical_semantics: &semantics,
                fidelity: CodecFidelity {
                    bound: None,
                    contract_id: ContentId::from_bytes([0x31; 32]),
                    kind: CodecFidelityKind::Transformed,
                    metric: Some("test-only".into()),
                },
                implementation: optimum_implementation_identity(),
                model_provenance: Some(semantic_abir_bcs::ModelProvenance {
                    checkpoint_content_id: ContentId::from_bytes([0x32; 32]),
                    checkpoint_sha256: [0x33; 32],
                    pccp_change_id: "test-only".into(),
                    pccp_evidence_id: ContentId::from_bytes([0x34; 32]),
                    pccp_status: semantic_abir_bcs::PccpStatus::Candidate,
                }),
                packets: &packets,
                parameters: canonical_parameters(),
                profile: CodecProfile::LmqProgressive,
            },
            ResourceBounds::default(),
        )
        .expect("wrong-profile fixture");
        assert!(matches!(
            open_optimum_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::CatalogContract)
        ));
    }

    #[test]
    fn encoder_uses_existing_lml_bytes() {
        let mapped = fixture();
        let bytes = encode_lml_bundle(mapped.dataset(), mapped.access(), ResourceBounds::default())
            .expect("encoded bundle");
        let opened = open_lml_bundle(&bytes, ResourceBounds::default()).expect("open bundle");
        let expected = lamquant_lml_mcu::lml::compress(opened.signal(), 0).unwrap();
        assert_eq!(opened.packet(), expected);
    }

    #[test]
    fn payload_access_encoder_preserves_explicit_window_and_mode() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let expected = encode_lml_bundle_from_signal_with_mode(
            mapped.dataset(),
            &signal,
            3,
            lamquant_lml_mcu::lpc::LpcMode::Fixed,
            ResourceBounds::default(),
        )
        .unwrap();
        let actual = encode_lml_bundle_with_window_size_and_mode(
            mapped.dataset(),
            mapped.access(),
            3,
            lamquant_lml_mcu::lpc::LpcMode::Fixed,
            ResourceBounds::default(),
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn semantic_packet_mismatch_fails_before_sealing() {
        let mapped = fixture();
        let mut wrong = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        wrong[0][0] += 1;
        let packet = lamquant_lml_mcu::lml::compress(&wrong, 0).unwrap();
        assert!(matches!(
            seal_lml_packet(mapped.dataset(), &packet, ResourceBounds::default()),
            Err(LmlBundleError::PayloadIdentityMismatch)
        ));
    }

    #[test]
    fn packet_tail_and_near_lossless_modes_fail_closed() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let mut packet = lamquant_lml_mcu::lml::compress(&signal, 0).unwrap();
        packet.push(0);
        assert!(matches!(
            seal_lml_packet(mapped.dataset(), &packet, ResourceBounds::default()),
            Err(LmlBundleError::PacketExtent)
        ));
        let near_lossless = lamquant_lml_mcu::lml::compress(&signal, 1).unwrap();
        assert!(matches!(
            seal_lml_packet(mapped.dataset(), &near_lossless, ResourceBounds::default()),
            Err(LmlBundleError::NotExactLossless)
        ));
    }

    #[test]
    fn bcs2_corruption_is_rejected_before_decode() {
        let mapped = fixture();
        let mut bytes =
            encode_lml_bundle(mapped.dataset(), mapped.access(), ResourceBounds::default())
                .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert!(matches!(
            open_lml_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::Bundle(_))
        ));
    }

    #[test]
    fn semantic_expansion_is_bounded_before_output_allocation() {
        let dataset = oversized_semantic_dataset();
        let semantics = canonical_debug_json(&dataset).unwrap();
        let packet = lamquant_lml_mcu::lml::compress(&[vec![0]], 0).unwrap();
        let packets = [&packet[..]];
        let bytes = encode_codec_bundle(
            CodecBundleInput {
                // Baseline kernels: any reader of the profile can decode these packets.
                required_capabilities: 0,
                canonical_semantics: &semantics,
                fidelity: exact_fidelity(),
                implementation: implementation_identity(),
                model_provenance: None,
                packets: &packets,
                parameters: canonical_parameters(),
                profile: CodecProfile::LmlLossless,
            },
            ResourceBounds::default(),
        )
        .unwrap();
        assert!(matches!(
            open_lml_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::DecodedResourceLimit)
        ));
    }

    #[test]
    fn excessive_channel_count_is_rejected_before_descriptor_allocation() {
        assert!(matches!(
            ordered_descriptors(&excessive_channel_dataset()),
            Err(LmlBundleError::DecodedResourceLimit)
        ));
    }

    #[test]
    fn generic_bundle_with_unregistered_kernel_is_not_an_lml_module_output() {
        let mapped = fixture();
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
        ];
        let packet = lamquant_lml_mcu::lml::compress(&signal, 0).unwrap();
        let semantics = canonical_debug_json(mapped.dataset()).unwrap();
        let packets = [&packet[..]];
        let mut implementation = implementation_identity();
        implementation.kernel_id = "unregistered-lookalike".to_string();
        let bytes = encode_codec_bundle(
            CodecBundleInput {
                // Baseline kernels: any reader of the profile can decode these packets.
                required_capabilities: 0,
                canonical_semantics: &semantics,
                fidelity: exact_fidelity(),
                implementation,
                model_provenance: None,
                packets: &packets,
                parameters: canonical_parameters(),
                profile: CodecProfile::LmlLossless,
            },
            ResourceBounds::default(),
        )
        .unwrap();
        assert!(matches!(
            open_lml_bundle(&bytes, ResourceBounds::default()),
            Err(LmlBundleError::CatalogContract)
        ));
    }

    #[test]
    fn signed_integer_payload_conversion_is_exact_in_both_byte_orders() {
        let cases = [
            (ElementType::I8, vec![-128, -1, 0, 127]),
            (ElementType::I16, vec![-32_768, -1, 0, 32_767]),
            (ElementType::I24, vec![-8_388_608, -1, 0, 8_388_607]),
            (
                ElementType::I32,
                vec![i32::MIN as i64, -1, 0, i32::MAX as i64],
            ),
            (ElementType::I64, vec![i64::MIN, -1, 0, i64::MAX]),
        ];
        for (element, samples) in cases {
            for order in [ByteOrder::Little, ByteOrder::Big] {
                let placeholder = PayloadDescriptor::new(
                    ContentId::from_bytes([1; 32]),
                    (samples.len() as u64) * element.byte_width().unwrap(),
                    element,
                    order,
                    vec![samples.len() as u64],
                    Layout::DenseRowMajor,
                    None,
                    None,
                );
                let bytes = encode_integer_payload(&placeholder, &samples).unwrap();
                let descriptor = PayloadDescriptor::new(
                    payload_content_id(element, &bytes),
                    bytes.len() as u64,
                    element,
                    order,
                    vec![samples.len() as u64],
                    Layout::DenseRowMajor,
                    None,
                    None,
                );
                assert_eq!(
                    decode_integer_payload(&descriptor, &bytes).unwrap(),
                    samples
                );
                verify_payload_content(&descriptor, &bytes).unwrap();
            }
        }
    }
}
