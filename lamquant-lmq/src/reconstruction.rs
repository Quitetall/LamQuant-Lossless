//! ABIR projection for decoded LMQ signals.
//!
//! Source semantics remain available through `OpenedLmqBundle::source_dataset`.
//! This module builds a separately valid, payload-complete dataset for the
//! actual reconstruction. Context without external evidence is retained.
//! Claims whose evidence bytes are not carried by LMQ are invalidated and
//! listed in a deterministic reconstruction receipt.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write;

use semantic_abir::{
    payload_content_id, AbirDataset, Atom, AtomTag, BlobIntegrity, BlobRef, ByteOrder, ConceptId,
    ContentId, DatasetDraft, DatasetTag, ElementType, ExactNumber, ExecutionRecord,
    Fidelity as SemanticFidelity, FidelityKind as SemanticFidelityKind, InMemoryPayloadAccess,
    Layout, ObjectId, ObjectKind, OpenedDataset, PayloadDescriptor, Presence, Rational, Recording,
    RecordingTag, SemanticRef, SignalBlock, SourceCapsule, SourceKey, SourceRelationship, Stream,
    StreamTag, ValidationLimits,
};
use semantic_abir_bcs::{
    CodecFidelity, CodecFidelityKind, CodecImplementation, CodecParameterValue, ModelProvenance,
    PccpStatus,
};

use crate::shell::LmqError;

const RECEIPT_MEDIA_TYPE: &str = "application/vnd.quitetall.lamquant.lmq-reconstruction-receipt-v1";
const RECEIPT_SOURCE_NAMESPACE: &str = "lamquant.reconstruction.receipt";

pub(crate) struct ReconstructionContext<'a> {
    pub(crate) fidelity: &'a CodecFidelity,
    pub(crate) implementation: &'a CodecImplementation,
    pub(crate) model: &'a ModelProvenance,
    pub(crate) source_semantic_id: ContentId,
    pub(crate) source_interchange_id: ContentId,
}

struct ReconstructionIds {
    source_recording: ObjectId<RecordingTag>,
    recording: ObjectId<RecordingTag>,
    dataset: ObjectId<DatasetTag>,
    stream: ObjectId<StreamTag>,
}

pub(crate) fn build_reconstructed_dataset(
    source: &AbirDataset,
    signal: &[Vec<i64>],
    context: ReconstructionContext<'_>,
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

    let ids = ReconstructionIds {
        source_recording: source_recording.id(),
        dataset: derived_object_id::<DatasetTag>(b"dataset", source.id().as_bytes(), &payloads, 0),
        recording: derived_object_id::<RecordingTag>(
            b"recording",
            source.id().as_bytes(),
            &payloads,
            0,
        ),
        stream: derived_object_id::<StreamTag>(b"stream", source.id().as_bytes(), &payloads, 0),
    };
    let mut draft = DatasetDraft::new(ids.dataset);
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

    let mut recording = Recording::new(ids.recording, vec![ids.stream]);
    for source_key in source_recording.source_keys() {
        recording.add_source_key(source_key.clone());
    }
    draft.add_recording(recording);
    draft.add_stream(Stream::new(
        ids.stream,
        ids.recording,
        source_stream.modality().clone(),
        atom_ids,
        source_stream.clock_id(),
        source_stream.channel_basis_id(),
        source_stream.policy_id(),
    ));

    let projection = add_context_without_external_payloads(&mut draft, source, &ids, &context)?;
    let receipt = reconstruction_receipt(source, &projection, &context)?;
    let receipt_content_id = payload_content_id(ElementType::Bytes, &receipt);
    let receipt_len = u64::try_from(receipt.len()).map_err(|_| LmqError::SemanticValidation)?;
    access.insert(receipt_content_id, receipt);
    let receipt_atom_id = derived_object_id::<AtomTag>(
        b"reconstruction-receipt",
        source.id().as_bytes(),
        &[receipt_content_id],
        0,
    );
    draft.add_atom(Atom::BlobRef(BlobRef::new(
        receipt_atom_id,
        Presence::Present,
        Some(PayloadDescriptor::new(
            receipt_content_id,
            receipt_len,
            ElementType::Bytes,
            ByteOrder::NotApplicable,
            vec![receipt_len],
            Layout::DenseRowMajor,
            Some(
                ConceptId::new("lamquant:encoding/lmq-reconstruction-receipt-v1")
                    .map_err(|_| LmqError::SemanticValidation)?,
            ),
            Some(RECEIPT_MEDIA_TYPE.into()),
        )),
        RECEIPT_MEDIA_TYPE.into(),
        BlobIntegrity::new(
            ConceptId::new("abir:integrity/blake3-256")
                .map_err(|_| LmqError::SemanticValidation)?,
            receipt_content_id,
        ),
    )));
    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new(
            RECEIPT_SOURCE_NAMESPACE,
            lowercase_hex(context.source_semantic_id.as_bytes()),
        )
        .map_err(|_| LmqError::SemanticValidation)?,
        receipt_content_id,
        Some(RECEIPT_MEDIA_TYPE),
    ));

    let dataset = draft
        .validate(ValidationLimits::default())
        .map_err(|_| LmqError::SemanticValidation)?;
    Ok(OpenedDataset::new(dataset, access))
}

struct Projection {
    retained_derivations: BTreeSet<SemanticRef>,
    dropped_derivations: Vec<[u8; 16]>,
    dropped_fidelity_subjects: Vec<SemanticRef>,
}

fn add_context_without_external_payloads(
    draft: &mut DatasetDraft,
    source: &AbirDataset,
    ids: &ReconstructionIds,
    context: &ReconstructionContext<'_>,
) -> Result<Projection, LmqError> {
    let mut base_refs = BTreeSet::new();
    base_refs.insert(SemanticRef::of(ids.dataset));
    base_refs.insert(SemanticRef::of(ids.recording));
    base_refs.insert(SemanticRef::of(ids.stream));
    for atom in draft.atoms() {
        base_refs.insert(SemanticRef::of(atom.id()));
    }

    macro_rules! copy_catalog {
        ($values:expr, $add:ident) => {
            for value in $values {
                base_refs.insert(SemanticRef::of(value.id()));
                draft.$add(value.clone());
            }
        };
    }

    copy_catalog!(source.clocks(), add_clock);
    copy_catalog!(source.coordinate_frames(), add_coordinate_frame);
    copy_catalog!(source.channel_bases(), add_channel_basis);
    copy_catalog!(source.policies(), add_policy);
    copy_catalog!(source.subjects(), add_subject);
    copy_catalog!(source.patients(), add_patient);
    copy_catalog!(source.sessions(), add_session);
    copy_catalog!(source.acquisitions(), add_acquisition);
    copy_catalog!(source.devices(), add_device);
    copy_catalog!(source.sensors(), add_sensor);
    copy_catalog!(source.channels(), add_channel);
    copy_catalog!(source.frame_transforms(), add_frame_transform);
    copy_catalog!(source.events(), add_event);
    copy_catalog!(source.concept_dictionaries(), add_concept_dictionary);

    for relationship in source.source_relationships() {
        let relationship = match *relationship {
            SourceRelationship::AcquisitionRecording {
                acquisition_id,
                recording_id: related_recording_id,
            } => {
                if related_recording_id != ids.source_recording {
                    return Err(LmqError::UnsupportedSemantics(
                        "source relationship targets an unexpected recording",
                    ));
                }
                SourceRelationship::AcquisitionRecording {
                    acquisition_id,
                    recording_id: ids.recording,
                }
            }
            other => other,
        };
        draft.add_source_relationship(relationship);
    }

    for execution in source.observed_execution() {
        draft.add_observed_execution(execution.clone());
    }
    draft.add_observed_execution(ExecutionRecord::new(
        ConceptId::new("lamquant:operation/lmq-decode")
            .map_err(|_| LmqError::SemanticValidation)?,
        format!(
            "{}@{}",
            context.implementation.kernel_id, context.implementation.build_id
        ),
    ));

    // Derivations are retained only when every reference names copied,
    // payload-independent context. Proofs, derived artifacts, clock relations,
    // and source capsules require external evidence bytes absent from LMQ and
    // are therefore invalidated rather than emitted as dangling claims.
    let mut retained_derivations = BTreeSet::new();
    let mut dropped_derivations = Vec::new();
    for derivation in source.derivations() {
        if derivation
            .inputs()
            .iter()
            .chain(derivation.outputs())
            .all(|reference| base_refs.contains(reference))
        {
            let reference = SemanticRef::of(derivation.id());
            retained_derivations.insert(reference);
            base_refs.insert(reference);
            draft.add_derivation(derivation.clone());
        } else {
            dropped_derivations.push(derivation.id().to_bytes());
        }
    }

    let mut dropped_fidelity_subjects = Vec::new();
    for statement in source.fidelity() {
        if base_refs.contains(&statement.subject()) {
            draft.add_fidelity(statement.clone());
        } else {
            dropped_fidelity_subjects.push(statement.subject());
        }
    }
    draft.add_fidelity(codec_fidelity_statement(ids.dataset, context.fidelity)?);
    Ok(Projection {
        retained_derivations,
        dropped_derivations,
        dropped_fidelity_subjects,
    })
}

fn reconstruction_receipt(
    source: &AbirDataset,
    projection: &Projection,
    context: &ReconstructionContext<'_>,
) -> Result<Vec<u8>, LmqError> {
    let mut receipt = String::from("LMQ-RECONSTRUCTION-PROJECTION-V1\n");
    receipt_field_hex(
        &mut receipt,
        "source-semantic-id",
        context.source_semantic_id.as_bytes(),
    )?;
    receipt_field_hex(
        &mut receipt,
        "source-interchange-id",
        context.source_interchange_id.as_bytes(),
    )?;
    receipt_field_hex(
        &mut receipt,
        "fidelity-contract-id",
        context.fidelity.contract_id.as_bytes(),
    )?;
    receipt_field_hex(
        &mut receipt,
        "implementation-id",
        context.implementation.implementation_id.as_bytes(),
    )?;
    receipt_field(&mut receipt, "kernel-id", &context.implementation.kernel_id)?;
    receipt_field(&mut receipt, "build-id", &context.implementation.build_id)?;
    receipt_field_hex(
        &mut receipt,
        "checkpoint-content-id",
        context.model.checkpoint_content_id.as_bytes(),
    )?;
    receipt_field_hex(
        &mut receipt,
        "checkpoint-sha256",
        &context.model.checkpoint_sha256,
    )?;
    receipt_field(
        &mut receipt,
        "pccp-change-id",
        &context.model.pccp_change_id,
    )?;
    receipt_field_hex(
        &mut receipt,
        "pccp-evidence-id",
        context.model.pccp_evidence_id.as_bytes(),
    )?;
    receipt_field(
        &mut receipt,
        "pccp-status",
        match context.model.pccp_status {
            PccpStatus::Candidate => "candidate",
            PccpStatus::GatePass => "gate-pass",
            PccpStatus::Rejected => "rejected",
        },
    )?;

    receipt_object_ids(
        &mut receipt,
        "invalidated-clock-relations",
        source
            .clock_relations()
            .iter()
            .map(|value| value.id().to_bytes())
            .collect(),
    )?;
    receipt_object_ids(
        &mut receipt,
        "invalidated-proofs",
        source
            .proofs()
            .iter()
            .map(|value| value.id().to_bytes())
            .collect(),
    )?;
    receipt_object_ids(
        &mut receipt,
        "invalidated-derivations",
        projection.dropped_derivations.clone(),
    )?;
    receipt_object_ids(
        &mut receipt,
        "invalidated-derived-artifacts",
        source
            .derived_artifacts()
            .iter()
            .map(|value| value.id().to_bytes())
            .collect(),
    )?;
    receipt_content_ids(
        &mut receipt,
        "invalidated-source-capsules",
        source
            .source_capsules()
            .iter()
            .map(semantic_abir::SourceCapsule::content_id)
            .collect(),
    )?;
    receipt_semantic_refs(
        &mut receipt,
        "invalidated-fidelity-subjects",
        projection.dropped_fidelity_subjects.clone(),
    )?;
    receipt_object_ids(
        &mut receipt,
        "retained-derivations",
        projection
            .retained_derivations
            .iter()
            .map(|reference| reference.bytes())
            .collect(),
    )?;
    Ok(receipt.into_bytes())
}

fn receipt_field(receipt: &mut String, name: &str, value: &str) -> Result<(), LmqError> {
    if value.chars().any(char::is_control) {
        return Err(LmqError::CatalogContract);
    }
    writeln!(receipt, "{name}={value}").map_err(|_| LmqError::SemanticEncoding)
}

fn receipt_field_hex(receipt: &mut String, name: &str, bytes: &[u8]) -> Result<(), LmqError> {
    receipt_field(receipt, name, &lowercase_hex(bytes))
}

fn receipt_object_ids(
    receipt: &mut String,
    name: &str,
    mut ids: Vec<[u8; 16]>,
) -> Result<(), LmqError> {
    ids.sort_unstable();
    receipt_list(receipt, name, ids.iter().map(|value| lowercase_hex(value)))
}

fn receipt_content_ids(
    receipt: &mut String,
    name: &str,
    mut ids: Vec<ContentId>,
) -> Result<(), LmqError> {
    ids.sort_unstable();
    receipt_list(
        receipt,
        name,
        ids.iter().map(|value| lowercase_hex(value.as_bytes())),
    )
}

fn receipt_semantic_refs(
    receipt: &mut String,
    name: &str,
    mut references: Vec<SemanticRef>,
) -> Result<(), LmqError> {
    references.sort_unstable();
    receipt_list(
        receipt,
        name,
        references.iter().map(|reference| {
            format!(
                "{}:{}",
                object_kind_name(reference.kind()),
                lowercase_hex(&reference.bytes())
            )
        }),
    )
}

fn object_kind_name(kind: ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Dataset => "dataset",
        ObjectKind::Recording => "recording",
        ObjectKind::Stream => "stream",
        ObjectKind::Atom => "atom",
        ObjectKind::Clock => "clock",
        ObjectKind::CoordinateFrame => "coordinate-frame",
        ObjectKind::ChannelBasis => "channel-basis",
        ObjectKind::Policy => "policy",
        ObjectKind::Proof => "proof",
        ObjectKind::Derivation => "derivation",
        ObjectKind::Subject => "subject",
        ObjectKind::Patient => "patient",
        ObjectKind::Session => "session",
        ObjectKind::Acquisition => "acquisition",
        ObjectKind::Device => "device",
        ObjectKind::Sensor => "sensor",
        ObjectKind::Channel => "channel",
        ObjectKind::ClockRelation => "clock-relation",
        ObjectKind::FrameTransform => "frame-transform",
        ObjectKind::Event => "event",
        ObjectKind::ConceptDictionary => "concept-dictionary",
        ObjectKind::DerivedArtifact => "derived-artifact",
    }
}

fn receipt_list(
    receipt: &mut String,
    name: &str,
    values: impl Iterator<Item = String>,
) -> Result<(), LmqError> {
    write!(receipt, "{name}=").map_err(|_| LmqError::SemanticEncoding)?;
    let mut separator = "";
    for value in values {
        write!(receipt, "{separator}{value}").map_err(|_| LmqError::SemanticEncoding)?;
        separator = ",";
    }
    receipt.push('\n');
    Ok(())
}

fn lowercase_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(encoded, "{byte:02x}").expect("writing to String is infallible");
    }
    encoded
}

pub(crate) fn codec_fidelity_statement(
    dataset_id: ObjectId<DatasetTag>,
    fidelity: &CodecFidelity,
) -> Result<SemanticFidelity, LmqError> {
    let (kind, metric, bound) = match fidelity.kind {
        CodecFidelityKind::Exact => return Err(LmqError::CatalogContract),
        CodecFidelityKind::Transformed => (
            SemanticFidelityKind::Transformed,
            Some(codec_metric_concept(
                fidelity.metric.as_deref(),
                "lamquant:metric/lmq-transformed",
            )?),
            None,
        ),
        CodecFidelityKind::Bounded => (
            SemanticFidelityKind::Bounded,
            Some(codec_metric_concept(
                fidelity.metric.as_deref(),
                "lamquant:metric/lmq-bounded",
            )?),
            Some(codec_bound(
                fidelity.bound.as_ref().ok_or(LmqError::CatalogContract)?,
            )?),
        ),
    };
    Ok(SemanticFidelity::new(
        SemanticRef::of(dataset_id),
        kind,
        metric,
        bound,
    ))
}

fn codec_metric_concept(metric: Option<&str>, fallback: &str) -> Result<ConceptId, LmqError> {
    let metric = metric.unwrap_or(fallback);
    if let Ok(concept) = ConceptId::new(metric) {
        return Ok(concept);
    }
    ConceptId::new(format!("lamquant:metric/{metric}")).map_err(|_| LmqError::CatalogContract)
}

fn codec_bound(bound: &CodecParameterValue) -> Result<ExactNumber, LmqError> {
    match bound {
        CodecParameterValue::Integer { value } => {
            let value = value
                .parse::<i128>()
                .map_err(|_| LmqError::CatalogContract)?;
            if value < 0 {
                return Err(LmqError::CatalogContract);
            }
            Ok(ExactNumber::Integer(value))
        }
        CodecParameterValue::Rational {
            denominator,
            numerator,
        } => {
            let numerator = numerator
                .parse::<i128>()
                .map_err(|_| LmqError::CatalogContract)?;
            let denominator = denominator
                .parse::<i128>()
                .map_err(|_| LmqError::CatalogContract)?;
            if numerator < 0 {
                return Err(LmqError::CatalogContract);
            }
            Rational::new(numerator, denominator)
                .map(ExactNumber::Rational)
                .map_err(|_| LmqError::CatalogContract)
        }
        _ => Err(LmqError::CatalogContract),
    }
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
