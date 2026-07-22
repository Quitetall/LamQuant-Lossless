//! Canonical semantic-ABIR lowering for source readers.
//!
//! Reader-specific parsers may keep private native layouts while parsing, but
//! their public seam returns a validated [`semantic_abir::AbirDataset`] with an
//! owned payload resolver. The retired uniform recording IR is not exposed by
//! this module.

use semantic_abir as semantic;
use sha2::{Digest, Sha256};

use crate::error::{LmlError, LmlResult};

use super::bundle::{SignalBundle, SourceMetadata};

/// Owned semantic result returned by biosignal source readers.
#[derive(Debug)]
pub struct SemanticRead {
    pub opened: semantic::OpenedDataset<semantic::InMemoryPayloadAccess>,
    pub mapping: SemanticMappingReport,
    pub fidelity: SemanticFidelityReport,
}

/// Auditable source-to-ABIR mapping summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticMappingReport {
    pub source_format: String,
    pub recording_count: usize,
    pub stream_count: usize,
    pub channel_count: usize,
    pub source_capsule_count: usize,
    pub channels: Vec<SemanticChannelMapping>,
    pub events: Vec<SemanticEventMapping>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticChannelMapping {
    pub index: usize,
    pub atom_id: semantic::ObjectId<semantic::AtomTag>,
    pub content_id: semantic::ContentId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticEventMapping {
    pub index: usize,
    pub event_id: semantic::ObjectId<semantic::EventTag>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticSourceCapsule {
    pub namespace: String,
    pub value: String,
    pub bytes: Vec<u8>,
    pub media_type: Option<String>,
}

/// Foreign source object whose namespace will be bound to the semantic
/// interchange identity produced by the same lowering operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticSourceObject {
    pub value: String,
    pub bytes: Vec<u8>,
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticTimedEvent {
    pub kind: semantic::ConceptId,
    pub start: semantic::Rational,
    pub end: semantic::Rational,
    pub uncertainty: semantic::Rational,
}

/// Honest fidelity claims for the generic reader lowering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticFidelityReport {
    pub sample_values_exact: bool,
    pub sample_rate_exact: bool,
    pub channel_order_exact: bool,
    pub labels_exact: bool,
    pub physical_ranges_preserved_as_source_keys: bool,
    pub calibration_promoted: bool,
    pub source_capsules_content_bound: bool,
}

/// Lower a validated format-agnostic reader result into canonical semantic
/// ABIR. Payload bytes move into the returned in-memory resolver exactly once.
pub fn from_signal_bundle(
    bundle: SignalBundle,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    from_signal_bundle_with_semantics(
        bundle,
        semantic::ConceptId::new("abir:modality/unknown").expect("static concept is canonical"),
        Vec::new(),
        Vec::new(),
        limits,
    )
}

/// Lower a caller-owned uniform signal without cloning its channel matrix.
///
/// This is the container hot path: semantic payload bytes are still produced
/// once for the ABIR resolver, but the input `Vec<Vec<i64>>` remains borrowed.
#[allow(clippy::too_many_arguments)]
pub fn from_uniform_signal_view(
    signal: &[Vec<i64>],
    sample_rate: f64,
    channels: Vec<String>,
    phys_min: Vec<f64>,
    phys_max: Vec<f64>,
    duration_s: f64,
    metadata: SourceMetadata,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    validate_signal_view(
        signal,
        sample_rate,
        &channels,
        &phys_min,
        &phys_max,
        duration_s,
    )?;
    lower_signal_parts(
        SignalInput::Borrowed(signal),
        sample_rate,
        channels,
        phys_min,
        phys_max,
        metadata,
        Vec::new(),
        semantic::ConceptId::new("abir:modality/unknown").expect("static concept is canonical"),
        CapsuleBinding::Explicit(Vec::new()),
        Vec::new(),
        limits,
    )
}

pub fn from_signal_bundle_with_overlays(
    bundle: SignalBundle,
    extra_capsules: Vec<SemanticSourceCapsule>,
    events: Vec<SemanticTimedEvent>,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    from_signal_bundle_with_semantics(
        bundle,
        semantic::ConceptId::new("abir:modality/unknown").expect("static concept is canonical"),
        extra_capsules,
        events,
        limits,
    )
}

pub fn from_signal_bundle_with_semantics(
    bundle: SignalBundle,
    modality: semantic::ConceptId,
    extra_capsules: Vec<SemanticSourceCapsule>,
    events: Vec<SemanticTimedEvent>,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    lower_signal_bundle(
        bundle,
        modality,
        CapsuleBinding::Explicit(extra_capsules),
        events,
        limits,
    )
}

/// Lower a signal bundle once and bind foreign objects to its capsule-free
/// interchange identity. The `namespace_prefix` is prepended to that identity.
///
/// This avoids the adapter anti-pattern of lowering an unbound copy solely to
/// discover the identity and then lowering the full sample payload again.
pub fn from_signal_bundle_with_interchange_bound_sources(
    bundle: SignalBundle,
    modality: semantic::ConceptId,
    namespace_prefix: String,
    source_objects: Vec<SemanticSourceObject>,
    events: Vec<SemanticTimedEvent>,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    lower_signal_bundle(
        bundle,
        modality,
        CapsuleBinding::InterchangeBound {
            namespace_prefix,
            source_objects,
        },
        events,
        limits,
    )
}

enum CapsuleBinding {
    Explicit(Vec<SemanticSourceCapsule>),
    InterchangeBound {
        namespace_prefix: String,
        source_objects: Vec<SemanticSourceObject>,
    },
}

enum SignalInput<'a> {
    Owned(Vec<Vec<i64>>),
    Borrowed(&'a [Vec<i64>]),
}

fn lower_signal_bundle(
    bundle: SignalBundle,
    modality: semantic::ConceptId,
    capsule_binding: CapsuleBinding,
    events: Vec<SemanticTimedEvent>,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    bundle.validate()?;
    let SignalBundle {
        signal,
        sample_rate,
        channels,
        phys_min,
        phys_max,
        duration_s: _,
        metadata,
        sidecar,
    } = bundle;
    lower_signal_parts(
        SignalInput::Owned(signal),
        sample_rate,
        channels,
        phys_min,
        phys_max,
        metadata,
        sidecar,
        modality,
        capsule_binding,
        events,
        limits,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_signal_parts(
    signal: SignalInput<'_>,
    sample_rate: f64,
    channels: Vec<String>,
    phys_min: Vec<f64>,
    phys_max: Vec<f64>,
    metadata: SourceMetadata,
    sidecar: Vec<super::bundle::SidecarBlob>,
    modality: semantic::ConceptId,
    capsule_binding: CapsuleBinding,
    events: Vec<SemanticTimedEvent>,
    limits: semantic::ValidationLimits,
) -> LmlResult<SemanticRead> {
    let (channel_count, sample_count) = match &signal {
        SignalInput::Owned(channels) => (channels.len(), channels.first().map_or(0, Vec::len)),
        SignalInput::Borrowed(channels) => (channels.len(), channels.first().map_or(0, Vec::len)),
    };
    if channel_count == 0 || sample_count == 0 {
        return Err(invalid(
            "semantic ABIR requires at least one non-empty channel",
        ));
    }
    let samples = u64::try_from(sample_count)
        .map_err(|_| invalid("sample count exceeds semantic ABIR limits"))?;
    let rate = exact_positive_f64(sample_rate)?;

    let mut payloads = semantic::InMemoryPayloadAccess::new();
    let mut channel_payloads = Vec::with_capacity(channel_count);
    let mut add_channel = |channel: &[i64]| {
        let mut bytes = Vec::with_capacity(channel.len().saturating_mul(8));
        for sample in channel {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        let content_id = semantic::payload_content_id(semantic::ElementType::I64, &bytes);
        payloads.insert(content_id, bytes);
        channel_payloads.push(content_id);
    };
    match signal {
        SignalInput::Owned(channels) => {
            for channel in channels {
                add_channel(&channel);
            }
        }
        SignalInput::Borrowed(channels) => {
            for channel in channels {
                add_channel(channel);
            }
        }
    }

    let seed = semantic_seed(
        &channel_payloads,
        sample_rate,
        &channels,
        &phys_min,
        &phys_max,
        &metadata,
        &modality,
        &events,
    );
    let dataset_id = object_id(&seed, b"dataset", 0);
    let recording_id = object_id(&seed, b"recording", 0);
    let stream_id = object_id(&seed, b"stream", 0);
    let basis_id = object_id(&seed, b"channel-basis", 0);
    let event_clock_id = (!events.is_empty()).then(|| object_id(&seed, b"event-clock", 0));

    let mut draft = semantic::DatasetDraft::new(dataset_id);
    let mut recording = semantic::Recording::new(recording_id, vec![stream_id]);
    add_recording_key(&mut recording, "source.format", &metadata.format)?;
    add_recording_key(&mut recording, "source.file", &metadata.source_file)?;
    add_recording_key(&mut recording, "source.patient-id", &metadata.patient_id)?;
    add_recording_key(
        &mut recording,
        "source.recording-info",
        &metadata.recording_info,
    )?;
    add_recording_key(&mut recording, "source.startdate", &metadata.startdate)?;
    draft.add_recording(recording);

    let mut atom_ids = Vec::with_capacity(channel_payloads.len());
    let mut channel_specs = Vec::with_capacity(channel_payloads.len());
    let mut channel_mappings = Vec::with_capacity(channel_payloads.len());
    for (index, content_id) in channel_payloads.iter().copied().enumerate() {
        let atom_id = object_id(&seed, b"signal-block", index as u64);
        let descriptor = semantic::PayloadDescriptor::new(
            content_id,
            samples
                .checked_mul(8)
                .ok_or_else(|| invalid("logical payload byte count overflow"))?,
            semantic::ElementType::I64,
            semantic::ByteOrder::Little,
            vec![1, samples],
            semantic::Layout::DenseRowMajor,
            None,
            None,
        );
        let segment = semantic::TimeSegment::new(
            semantic::Rational::new(0, 1).expect("zero is canonical"),
            rate,
            samples,
        )
        .map_err(|error| invalid(&error.to_string()))?;
        draft.add_atom(semantic::Atom::SignalBlock(semantic::SignalBlock::new(
            atom_id,
            semantic::Presence::Present,
            Some(descriptor),
            semantic::TimeAxis::Regular(segment),
            None,
        )));
        atom_ids.push(atom_id);
        channel_mappings.push(SemanticChannelMapping {
            index,
            atom_id,
            content_id,
        });

        let label = channels.get(index).map(String::as_str).unwrap_or("");
        let mut spec = semantic::ChannelSpec::new(
            semantic::ConceptId::new(format!("source:channel/{index}"))
                .map_err(|error| invalid(&error.to_string()))?,
        );
        spec = spec.with_source_key(source_key("source.channel-label", label)?);
        spec = spec.with_source_key(source_key(
            "source.physical-min",
            &phys_min[index].to_string(),
        )?);
        spec = spec.with_source_key(source_key(
            "source.physical-max",
            &phys_max[index].to_string(),
        )?);
        spec = spec.with_source_key(source_key("source.physical-unit", &metadata.phys_dim)?);
        channel_specs.push(spec);
    }

    draft.add_stream(semantic::Stream::new(
        stream_id,
        recording_id,
        modality,
        atom_ids,
        event_clock_id,
        Some(basis_id),
        None,
    ));
    draft.add_channel_basis(semantic::ChannelBasis::new(
        basis_id,
        channel_specs,
        semantic::ReferenceKind::Unknown,
    ));

    let mut event_mappings = Vec::with_capacity(events.len());
    if let Some(clock_id) = event_clock_id {
        draft.add_clock(semantic::Clock::new(
            clock_id,
            semantic::ConceptId::new("abir:clock/recording-relative-seconds")
                .expect("static concept is canonical"),
            None,
            semantic::Rational::new(0, 1).expect("zero is canonical"),
            semantic::Rational::new(1, 1).expect("one is canonical"),
            semantic::Rational::new(0, 1).expect("zero is canonical"),
        ));
        for (index, event) in events.into_iter().enumerate() {
            let event_id = object_id(&seed, b"event", index as u64);
            draft.add_event(semantic::Event::new(
                event_id,
                event.kind,
                clock_id,
                event.start,
                event.end,
                event.uncertainty,
            ));
            event_mappings.push(SemanticEventMapping { index, event_id });
        }
    }

    let extra_capsules = match capsule_binding {
        CapsuleBinding::Explicit(capsules) => capsules,
        CapsuleBinding::InterchangeBound {
            namespace_prefix,
            source_objects,
        } => {
            // Interchange identity excludes source capsules by contract. A
            // clone of semantic metadata is validated here; sample payload
            // buffers remain solely in `payloads` and are never copied.
            let semantic_root = draft.clone().validate(limits).map_err(|report| {
                invalid(&format!("semantic ABIR validation failed: {report:?}"))
            })?;
            let semantic_id = semantic::interchange_content_id(&semantic_root)
                .map_err(|error| invalid(&error.to_string()))?;
            let namespace = format!("{namespace_prefix}{semantic_id}");
            source_objects
                .into_iter()
                .map(|source| SemanticSourceCapsule {
                    namespace: namespace.clone(),
                    value: source.value,
                    bytes: source.bytes,
                    media_type: source.media_type,
                })
                .collect()
        }
    };
    let source_capsule_count = sidecar.len().saturating_add(extra_capsules.len());
    for (index, capsule) in sidecar.into_iter().enumerate() {
        let content_id = semantic::payload_content_id(semantic::ElementType::Bytes, &capsule.bytes);
        payloads.insert(content_id, capsule.bytes);
        let value = capsule.aux.map_or_else(
            || capsule.key.clone(),
            |aux| format!("{}#{aux}", capsule.key),
        );
        draft.add_source_capsule(semantic::SourceCapsule::new(
            source_key(&format!("source.sidecar.{index}"), &value)?,
            content_id,
            Some("application/octet-stream"),
        ));
    }
    for capsule in extra_capsules {
        let content_id = semantic::payload_content_id(semantic::ElementType::Bytes, &capsule.bytes);
        payloads.insert(content_id, capsule.bytes);
        draft.add_source_capsule(semantic::SourceCapsule::new(
            source_key(&capsule.namespace, &capsule.value)?,
            content_id,
            capsule.media_type.as_deref(),
        ));
    }

    let dataset = draft
        .validate(limits)
        .map_err(|report| invalid(&format!("semantic ABIR validation failed: {report:?}")))?;
    Ok(SemanticRead {
        opened: semantic::OpenedDataset::new(dataset, payloads),
        mapping: SemanticMappingReport {
            source_format: metadata.format,
            recording_count: 1,
            stream_count: 1,
            channel_count: channel_payloads.len(),
            source_capsule_count,
            channels: channel_mappings,
            events: event_mappings,
        },
        fidelity: SemanticFidelityReport {
            sample_values_exact: true,
            sample_rate_exact: true,
            channel_order_exact: true,
            labels_exact: true,
            physical_ranges_preserved_as_source_keys: true,
            calibration_promoted: false,
            source_capsules_content_bound: true,
        },
    })
}

fn add_recording_key(
    recording: &mut semantic::Recording,
    namespace: &str,
    value: &str,
) -> LmlResult<()> {
    if !value.is_empty() {
        recording.add_source_key(source_key(namespace, value)?);
    }
    Ok(())
}

fn source_key(namespace: &str, value: &str) -> LmlResult<semantic::SourceKey> {
    semantic::SourceKey::new(namespace, value).map_err(|error| invalid(&error.to_string()))
}

#[allow(clippy::too_many_arguments)] // inputs form the complete semantic identity tuple
fn semantic_seed(
    payloads: &[semantic::ContentId],
    sample_rate: f64,
    channels: &[String],
    physical_minima: &[f64],
    physical_maxima: &[f64],
    metadata: &SourceMetadata,
    modality: &semantic::ConceptId,
    events: &[SemanticTimedEvent],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"org.quitetall.lamquant.source-semantic-abir-v1\0");
    digest.update(sample_rate.to_bits().to_le_bytes());
    digest.update((modality.as_str().len() as u64).to_le_bytes());
    digest.update(modality.as_str().as_bytes());
    for value in [
        metadata.source_file.as_str(),
        metadata.format.as_str(),
        metadata.patient_id.as_str(),
        metadata.recording_info.as_str(),
        metadata.startdate.as_str(),
        metadata.phys_dim.as_str(),
    ] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    for (index, (payload, label)) in payloads.iter().zip(channels).enumerate() {
        digest.update(payload.as_bytes());
        digest.update((label.len() as u64).to_le_bytes());
        digest.update(label.as_bytes());
        digest.update(physical_minima[index].to_bits().to_le_bytes());
        digest.update(physical_maxima[index].to_bits().to_le_bytes());
    }
    digest.update((events.len() as u64).to_le_bytes());
    for event in events {
        digest.update((event.kind.as_str().len() as u64).to_le_bytes());
        digest.update(event.kind.as_str().as_bytes());
        for value in [event.start, event.end, event.uncertainty] {
            let (numerator, denominator) = value.parts();
            digest.update(numerator.to_le_bytes());
            digest.update(denominator.to_le_bytes());
        }
    }
    digest.finalize().into()
}

fn object_id<T>(seed: &[u8; 32], domain: &[u8], index: u64) -> semantic::ObjectId<T> {
    let mut digest = Sha256::new();
    digest.update(b"org.quitetall.lamquant.source-object-id-v1\0");
    digest.update(seed);
    digest.update(domain);
    digest.update(index.to_le_bytes());
    let bytes: [u8; 32] = digest.finalize().into();
    let mut id = [0_u8; 16];
    id.copy_from_slice(&bytes[..16]);
    semantic::ObjectId::from_bytes(id)
}

fn exact_positive_f64(value: f64) -> LmlResult<semantic::Rational> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("sample rate is not a positive exact number"));
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mut numerator, mut power) = if exponent == 0 {
        (u128::from(fraction), -1074_i32)
    } else {
        (u128::from(fraction | (1_u64 << 52)), exponent - 1023 - 52)
    };
    if power < 0 {
        let reduce = numerator.trailing_zeros().min((-power) as u32);
        numerator >>= reduce;
        power += reduce as i32;
    }
    let (numerator, denominator) = if power >= 0 {
        (
            numerator
                .checked_shl(power as u32)
                .and_then(|number| i128::try_from(number).ok())
                .ok_or_else(|| invalid("sample rate exceeds exact-number limits"))?,
            1_i128,
        )
    } else {
        let shift = (-power) as u32;
        if shift > 126 {
            return Err(invalid("sample rate exceeds exact-number limits"));
        }
        (
            i128::try_from(numerator)
                .map_err(|_| invalid("sample rate exceeds exact-number limits"))?,
            1_i128 << shift,
        )
    };
    semantic::Rational::new(numerator, denominator).map_err(|error| invalid(&error.to_string()))
}

fn validate_signal_view(
    signal: &[Vec<i64>],
    sample_rate: f64,
    channels: &[String],
    phys_min: &[f64],
    phys_max: &[f64],
    duration_s: f64,
) -> LmlResult<()> {
    let n_channels = signal.len();
    if channels.len() != n_channels || phys_min.len() != n_channels || phys_max.len() != n_channels
    {
        return Err(invalid(
            "signal metadata lengths must equal the channel count",
        ));
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid("sample rate must be finite and positive"));
    }
    if !duration_s.is_finite() || duration_s < 0.0 {
        return Err(invalid("duration must be finite and nonnegative"));
    }
    let Some(sample_count) = signal.first().map(Vec::len) else {
        return Err(invalid("signal must contain at least one channel"));
    };
    if sample_count == 0 || signal.iter().any(|channel| channel.len() != sample_count) {
        return Err(invalid("signal channels must be non-empty and uniform"));
    }
    Ok(())
}

fn invalid(message: &str) -> LmlError {
    LmlError::InvalidHeader(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{SidecarBlob, SourceMetadata};

    fn fixture_bundle() -> SignalBundle {
        SignalBundle {
            signal: vec![vec![1, -2, 3], vec![4, 5, 6]],
            sample_rate: 250.0,
            channels: vec!["Fp1".to_owned(), "Cz".to_owned()],
            phys_min: vec![-100.0, -200.0],
            phys_max: vec![100.0, 200.0],
            duration_s: 3.0 / 250.0,
            metadata: SourceMetadata {
                source_file: "fixture.edf".to_owned(),
                format: "EDF".to_owned(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: String::new(),
                phys_dim: "uV".to_owned(),
            },
            sidecar: vec![SidecarBlob {
                key: "raw_header".to_owned(),
                bytes: vec![1, 2, 3],
                aux: None,
            }],
        }
    }

    #[test]
    fn bundle_lowering_returns_valid_owned_semantics() {
        let read =
            from_signal_bundle(fixture_bundle(), semantic::ValidationLimits::default()).unwrap();

        assert_eq!(read.opened.dataset().recordings().len(), 1);
        assert_eq!(read.opened.dataset().streams().len(), 1);
        assert_eq!(read.opened.dataset().atoms().len(), 2);
        assert_eq!(read.mapping.channel_count, 2);
        assert!(read.fidelity.sample_values_exact);
        let atom_id = read.opened.dataset().streams()[0].atoms()[0];
        let block = read.opened.block_view(atom_id).unwrap();
        assert_eq!(block.bytes().len(), 24);
    }

    #[test]
    fn borrowed_uniform_lowering_matches_owned_semantics() {
        let mut bundle = fixture_bundle();
        bundle.sidecar.clear();
        let owned =
            from_signal_bundle(bundle.clone(), semantic::ValidationLimits::default()).unwrap();
        let borrowed = from_uniform_signal_view(
            &bundle.signal,
            bundle.sample_rate,
            bundle.channels.clone(),
            bundle.phys_min.clone(),
            bundle.phys_max.clone(),
            bundle.duration_s,
            bundle.metadata.clone(),
            semantic::ValidationLimits::default(),
        )
        .unwrap();
        assert_eq!(
            semantic::canonical_debug_json(owned.opened.dataset()).unwrap(),
            semantic::canonical_debug_json(borrowed.opened.dataset()).unwrap()
        );
        assert_eq!(owned.mapping, borrowed.mapping);
        assert_eq!(owned.fidelity, borrowed.fidelity);
    }

    #[test]
    fn overlays_are_semantic_while_source_capsules_do_not_rename_the_dataset() {
        let event = SemanticTimedEvent {
            kind: semantic::ConceptId::new("bids:event/stimulus").unwrap(),
            start: semantic::Rational::new(1, 2).unwrap(),
            end: semantic::Rational::new(3, 5).unwrap(),
            uncertainty: semantic::Rational::new(0, 1).unwrap(),
        };
        let baseline = from_signal_bundle_with_overlays(
            fixture_bundle(),
            vec![],
            vec![event.clone()],
            semantic::ValidationLimits::default(),
        )
        .unwrap();
        let with_source = from_signal_bundle_with_overlays(
            fixture_bundle(),
            vec![SemanticSourceCapsule {
                namespace: "adapter.edf.binding".to_owned(),
                value: "fixture.edf".to_owned(),
                bytes: vec![9, 8, 7],
                media_type: Some("application/edf".to_owned()),
            }],
            vec![event],
            semantic::ValidationLimits::default(),
        )
        .unwrap();
        assert_eq!(
            baseline.opened.dataset().id(),
            with_source.opened.dataset().id()
        );
        assert_eq!(with_source.mapping.events.len(), 1);
        assert_eq!(with_source.opened.dataset().events().len(), 1);
        assert_eq!(with_source.opened.dataset().clocks().len(), 1);

        let unmodified =
            from_signal_bundle(fixture_bundle(), semantic::ValidationLimits::default()).unwrap();
        let mut changed = fixture_bundle();
        changed.phys_max[0] += 1.0;
        let changed = from_signal_bundle(changed, semantic::ValidationLimits::default()).unwrap();
        assert_ne!(
            unmodified.opened.dataset().id(),
            changed.opened.dataset().id()
        );
    }

    #[test]
    fn interchange_bound_sources_use_the_final_capsule_free_identity() {
        let read = from_signal_bundle_with_interchange_bound_sources(
            fixture_bundle(),
            semantic::ConceptId::new("abir:modality/eeg").unwrap(),
            "adapter.test.binding.".to_owned(),
            vec![SemanticSourceObject {
                value: "fixture.edf".to_owned(),
                bytes: vec![9, 8, 7],
                media_type: Some("application/edf".to_owned()),
            }],
            vec![],
            semantic::ValidationLimits::default(),
        )
        .unwrap();
        let semantic_id = semantic::interchange_content_id(read.opened.dataset()).unwrap();
        let capsule = read.opened.dataset().source_capsules().last().unwrap();
        assert_eq!(
            capsule.source().namespace(),
            format!("adapter.test.binding.{semantic_id}")
        );
        let block = read
            .opened
            .block_view(read.opened.dataset().streams()[0].atoms()[0])
            .unwrap();
        assert_eq!(
            block.bytes(),
            &[1_i64, -2, 3].map(i64::to_le_bytes).concat()
        );
    }
}
