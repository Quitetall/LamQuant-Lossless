#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! Deterministic LML packets carried as semantic ABIR BCS2 Bundles.
//!
//! This crate is deliberately an integration layer. It calls the existing
//! `lamquant-lml-mcu` packet encoder/decoder without changing the LML1 grammar
//! or its hot path. The outer BCS2 Bundle binds canonical ABIR semantics,
//! packet bytes, codec implementation identity, and the exact-fidelity
//! contract. Opening is fail-closed: the LML packet is decoded and every
//! channel is re-hashed as its declared ABIR payload before data is exposed.

extern crate alloc;

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
    encode_codec_bundle, CodecBundleError, CodecBundleInput, CodecBundleView, CodecFidelity,
    CodecFidelityKind, CodecImplementation, CodecParameter, CodecParameterValue, CodecProfile,
    ResourceBounds,
};

/// Stable algorithm identity. Build-specific identity is recorded separately.
pub const LML_KERNEL_ID: &str = "org.quitetall.lamquant.lml-mcu.lossless-v1";
/// Exact semantic-to-packet closure enforced by this integration crate.
pub const LML_FIDELITY_CONTRACT: &str =
    "org.quitetall.lamquant.bcs2.lml.exact-signal-block-closure-v1";
const SOURCE_ID: &str = env!("LAMQUANT_ABIR_CODEC_SOURCE_ID");
const BUILD_ID: &str = env!("LAMQUANT_ABIR_CODEC_BUILD_ID");
const ABIR_REVISION: &str = "c101513167ad8d7cdefa6387b20c644fdaf66432";
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
    encode_lml_bundle_from_verified_signal(
        dataset,
        &signal,
        window_size,
        lamquant_lml_mcu::lpc::LpcMode::default(),
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
    verify_signal_closure(dataset, signal)?;
    encode_lml_bundle_from_verified_signal(dataset, signal, window_size, mode, bounds)
}

fn encode_lml_bundle_from_verified_signal(
    dataset: &AbirDataset,
    signal: &[Vec<i64>],
    window_size: usize,
    mode: lamquant_lml_mcu::lpc::LpcMode,
    bounds: ResourceBounds,
) -> Result<Vec<u8>, LmlBundleError> {
    let packet_samples = window_size.min(MAX_PACKET_SAMPLES);
    if packet_samples == 0 {
        return Err(LmlBundleError::PacketExtent);
    }
    let total_samples = signal.first().map_or(0, Vec::len);
    let mut packets = Vec::with_capacity(total_samples.div_ceil(packet_samples));
    for start in (0..total_samples).step_by(packet_samples) {
        let end = start.saturating_add(packet_samples).min(total_samples);
        let window = signal
            .iter()
            .map(|channel| &channel[start..end])
            .collect::<Vec<_>>();
        packets.push(compress_views(&window, mode)?);
    }
    let packet_refs = packets.iter().map(Vec::as_slice).collect::<Vec<_>>();
    encode_verified_packets(dataset, &packet_refs, bounds)
}

#[cfg(feature = "std")]
fn compress_views(
    signal: &[&[i64]],
    mode: lamquant_lml_mcu::lpc::LpcMode,
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
        return lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode)
            .map_err(LmlBundleError::Lml);
    }

    let result = match global_backend() {
        ComputeBackend::Firmware => {
            lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode)
        }
        ComputeBackend::Desktop => {
            lamquant_lml_desktop::compress_with_mode_parallel_views(signal, 0, mode)
        }
    };
    result.map_err(LmlBundleError::Lml)
}

#[cfg(not(feature = "std"))]
fn compress_views(
    signal: &[&[i64]],
    mode: lamquant_lml_mcu::lpc::LpcMode,
) -> Result<Vec<u8>, LmlBundleError> {
    lamquant_lml_mcu::lml::compress_with_mode_views(signal, 0, mode).map_err(LmlBundleError::Lml)
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
    for packet in packets {
        validate_strict_lossless_packet(packet)?;
    }
    let semantics = canonical_debug_json(dataset).map_err(|_| LmlBundleError::SemanticEncoding)?;
    encode_codec_bundle(
        CodecBundleInput {
            canonical_semantics: &semantics,
            fidelity: exact_fidelity(),
            implementation: implementation_identity(),
            model_provenance: None,
            packets,
            parameters: canonical_parameters(),
            profile: CodecProfile::LmlLossless,
        },
        bounds,
    )
    .map_err(LmlBundleError::Bundle)
}

/// Open, authenticate, decode, and prove semantic closure before returning a
/// packet or reconstructed samples.
pub fn open_lml_bundle(
    bytes: &[u8],
    bounds: ResourceBounds,
) -> Result<OpenedLmlBundle<'_>, LmlBundleError> {
    let bundle = CodecBundleView::open(bytes, bounds).map_err(LmlBundleError::Bundle)?;
    validate_catalog(&bundle)?;
    let (signal, packet_sample_counts) =
        decode_packet_sequence(bundle.dataset(), bundle.packets())?;
    verify_signal_closure(bundle.dataset(), &signal)?;
    Ok(OpenedLmlBundle {
        bundle,
        packet_sample_counts,
        signal,
    })
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
    hasher.update(ABIR_REVISION.as_bytes());
    CodecImplementation {
        build_id: format!("blake3:{BUILD_ID}"),
        implementation_id: ContentId::from_bytes(*hasher.finalize().as_bytes()),
        kernel_id: LML_KERNEL_ID.to_string(),
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
                value: ABIR_REVISION.to_string(),
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
    Lml(lamquant_lml_mcu::error::LmlError),
    NotExactLossless,
    NotLml1,
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
        let signal = [
            (ElementType::I16, vec![1_i64, -2, 3, -4, 5, -6, 7, -8]),
            (
                ElementType::I24,
                vec![-8_388_608, -100, -1, 0, 1, 100, 8_388_606, 8_388_607],
            ),
            (
                ElementType::I64,
                vec![-1_000_000, -4, -1, 0, 1, 4, 1_000_000, 42],
            ),
        ];
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
