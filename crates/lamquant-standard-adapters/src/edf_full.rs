use super::{binding_namespace, payload_content_id, plan_id, valid_relative_path};
use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use lamquant_core::edf::{read_edf, EdfFile};
use semantic_abir::{
    interchange_content_id, payload_content_id as abir_payload_id, Atom, AtomTag, BlobIntegrity,
    BlobRef, ByteOrder, Calibration, ChannelBasis, ChannelBasisTag, ChannelSpec, Clock, ClockTag,
    ConceptId, DatasetDraft, DatasetTag, ElementType, Event, EventTag, Layout, ObjectId,
    PayloadDescriptor, Presence, Rational, Recording, RecordingTag, ReferenceKind, SignalBlock,
    SourceCapsule, SourceKey, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

const PROFILE: &str = "edfplus.1";

pub struct EdfAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
}

struct ParsedEdf {
    dataset: semantic_abir::AbirDataset,
    payloads: Vec<PayloadObject>,
    mappings: Vec<MappingEntry>,
    channels: u64,
    annotations: u64,
    segments: u64,
}

#[derive(Clone)]
struct SignalHeader {
    label: String,
    unit: String,
    physical_min: Rational,
    physical_max: Rational,
    digital_min: i128,
    digital_max: i128,
    samples_per_record: usize,
}

#[derive(Clone)]
struct Annotation {
    onset: Rational,
    duration: Option<Rational>,
    text: String,
}

impl EdfAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: AdapterProfile {
                id: ProfileId(PROFILE.to_owned()),
                standard: "EDF/EDF+/BDF".to_owned(),
                edition: "EDF+ 1".to_owned(),
                media_types: vec!["application/edf".to_owned(), "application/bdf".to_owned()],
                status: ProfileStatus::Semantic,
                required_validator: "pyedflib".to_owned(),
                capabilities: BTreeSet::from([
                    AdapterCapability::Inspect,
                    AdapterCapability::Import,
                    AdapterCapability::PlanExport,
                    AdapterCapability::Export,
                    AdapterCapability::Validate,
                ]),
            },
            max_source_bytes,
        }
    }

    fn entry<'a>(&self, source: &'a ForeignObject) -> Result<&'a ForeignEntry, AdapterError> {
        if source.profile != self.profile.id {
            return Err(AdapterError::ProfileMismatch {
                expected: self.profile.id.clone(),
                actual: source.profile.clone(),
            });
        }
        if source.entries.len() != 1 {
            return Err(AdapterError::InvalidSource(
                "EDF semantic profile requires exactly one file".to_owned(),
            ));
        }
        let entry = &source.entries[0];
        if !valid_relative_path(&entry.path) {
            return Err(AdapterError::InvalidPath(entry.path.clone()));
        }
        if u64::try_from(entry.bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?
            > self.max_source_bytes
        {
            return Err(AdapterError::SourceTooLarge);
        }
        Ok(entry)
    }

    fn parse(
        &self,
        entry: &ForeignEntry,
        limits: ValidationLimits,
    ) -> Result<ParsedEdf, AdapterError> {
        let temporary =
            tempfile::tempdir().map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let extension = if entry.bytes.first() == Some(&0xff) {
            "bdf"
        } else {
            "edf"
        };
        let path = temporary.path().join(format!("source.{extension}"));
        fs::write(&path, &entry.bytes)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let edf =
            read_edf(&path).map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let headers = signal_headers(&edf)?;
        let (record_onsets, annotations) = annotations(&edf, &headers)?;
        if edf.format == "EDF+D" && record_onsets.len() != edf.n_data_records {
            return Err(AdapterError::InvalidSource(
                "every EDF+D record requires an exact timekeeping annotation".to_owned(),
            ));
        }

        let seed = blake3::hash(&entry.bytes);
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let stream_id = id::<StreamTag>(&seed, b"stream", 0);
        let clock_id = id::<ClockTag>(&seed, b"clock", 0);
        let basis_id = id::<ChannelBasisTag>(&seed, b"basis", 0);
        let record_duration = decimal(&ascii(&edf.raw_header[244..252])?)?;
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();
        let mut atom_ids = Vec::new();
        let mut channel_specs = Vec::new();
        let mut signal_count = 0_u64;
        let mut segment_count = 0_u64;
        for (source_index, header) in headers.iter().enumerate() {
            if is_annotation(&header.label) {
                continue;
            }
            let values = channel_values(&edf, source_index, header.samples_per_record)?;
            let samples = u64::try_from(values.len()).map_err(|_| AdapterError::SourceTooLarge)?;
            let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let content_id = abir_payload_id(ElementType::I64, &bytes);
            let atom_id = id::<AtomTag>(&seed, b"signal", source_index as u64);
            let rate = samples_per_second(header.samples_per_record, record_duration)?;
            let time_axis = if edf.format == "EDF+D" {
                let mut segments = Vec::with_capacity(edf.n_data_records);
                for onset in &record_onsets {
                    segments.push(
                        TimeSegment::new(
                            *onset,
                            rate,
                            u64::try_from(header.samples_per_record)
                                .map_err(|_| AdapterError::SourceTooLarge)?,
                        )
                        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?,
                    );
                }
                segment_count = segment_count
                    .checked_add(segments.len() as u64)
                    .ok_or(AdapterError::SourceTooLarge)?;
                TimeAxis::Piecewise(segments)
            } else {
                segment_count = segment_count
                    .checked_add(1)
                    .ok_or(AdapterError::SourceTooLarge)?;
                TimeAxis::Regular(
                    TimeSegment::new(Rational::new(0, 1).unwrap(), rate, samples)
                        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?,
                )
            };
            let calibration = calibration(header)?;
            let descriptor = PayloadDescriptor::new(
                content_id,
                u64::try_from(bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?,
                ElementType::I64,
                ByteOrder::Little,
                vec![1, samples],
                Layout::DenseRowMajor,
                Some(ConceptId::new("abir:encoding/raw").unwrap()),
                None,
            );
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                time_axis,
                Some(calibration),
            )));
            payloads.push(PayloadObject { content_id, bytes });
            atom_ids.push(atom_id);
            let mut spec = ChannelSpec::new(channel_concept(&header.label, source_index)?);
            spec = spec.with_source_key(source_key("edf.channel-label", &header.label)?);
            spec = spec.with_source_key(source_key("edf.physical-unit", &header.unit)?);
            channel_specs.push(spec);
            mappings.push(exact(
                format!("signal[{source_index}].samples"),
                format!("atom:{atom_id}"),
            ));
            mappings.push(exact(
                format!("signal[{source_index}].calibration"),
                format!("atom:{atom_id}:calibration"),
            ));
            signal_count += 1;
        }

        if !annotations.is_empty() {
            let annotation_id = id::<AtomTag>(&seed, b"annotations", 0);
            let encoded = serde_json::to_vec(
                &annotations
                    .iter()
                    .map(|annotation| {
                        serde_json::json!({
                            "onset": annotation.onset.to_string(),
                            "duration": annotation.duration.map(|value| value.to_string()),
                            "text": annotation.text,
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
            let content_id = abir_payload_id(ElementType::Bytes, &encoded);
            let descriptor = PayloadDescriptor::new(
                content_id,
                encoded.len() as u64,
                ElementType::Bytes,
                ByteOrder::Little,
                vec![encoded.len() as u64],
                Layout::DenseRowMajor,
                Some(ConceptId::new("edf:encoding/annotation-json").unwrap()),
                Some("application/json".to_owned()),
            );
            draft.add_atom(Atom::BlobRef(BlobRef::new(
                annotation_id,
                Presence::Present,
                Some(descriptor),
                "application/json".to_owned(),
                BlobIntegrity::new(
                    ConceptId::new("abir:integrity/blake3-256").unwrap(),
                    content_id,
                ),
            )));
            payloads.push(PayloadObject {
                content_id,
                bytes: encoded,
            });
            atom_ids.push(annotation_id);
            mappings.push(exact(
                "annotations".to_owned(),
                format!("atom:{annotation_id}"),
            ));
            for (index, annotation) in annotations.iter().enumerate() {
                let event_id = id::<EventTag>(&seed, b"annotation-event", index as u64);
                let end = match annotation.duration {
                    Some(duration) => add(annotation.onset, duration)?,
                    None => annotation.onset,
                };
                draft.add_event(Event::new(
                    event_id,
                    ConceptId::new("edf:event/annotation").unwrap(),
                    clock_id,
                    annotation.onset,
                    end,
                    Rational::new(0, 1).unwrap(),
                ));
            }
        }

        let mut recording = Recording::new(recording_id, vec![stream_id]);
        for (namespace, value) in [
            ("edf.format", edf.format.as_str()),
            ("edf.patient-id", edf.patient_id.as_str()),
            ("edf.recording-info", edf.recording_info.as_str()),
            ("edf.recording-local-time", edf.startdate.as_str()),
        ] {
            if !value.is_empty() {
                recording.add_source_key(source_key(namespace, value)?);
            }
        }
        draft.add_recording(recording);
        draft.add_clock(Clock::new(
            clock_id,
            ConceptId::new("edf:clock/recording-local-unknown-zone").unwrap(),
            None,
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap(),
            Rational::new(1, 1).unwrap(),
        ));
        draft.add_channel_basis(ChannelBasis::new(
            basis_id,
            channel_specs,
            ReferenceKind::Unknown,
        ));
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            ConceptId::new("abir:modality/eeg").unwrap(),
            atom_ids,
            Some(clock_id),
            Some(basis_id),
            None,
        ));
        mappings.push(exact(
            "recording.startdate-time".to_owned(),
            format!("recording:{recording_id}:source-key"),
        ));

        let semantic = draft
            .clone()
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        let interchange = interchange_content_id(&semantic)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?;
        let namespace = format!("adapter.{PROFILE}.binding.{interchange}");
        let source_content = payload_content_id(&entry.bytes);
        draft.add_source_capsule(SourceCapsule::new(
            source_key(&namespace, &entry.path)?,
            source_content,
            entry.media_type.as_deref(),
        ));
        let dataset = draft
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        mappings.push(exact(
            entry.path.clone(),
            format!("source-capsule:{source_content}"),
        ));
        payloads.push(PayloadObject {
            content_id: source_content,
            bytes: entry.bytes.clone(),
        });
        Ok(ParsedEdf {
            dataset,
            payloads,
            mappings,
            channels: signal_count,
            annotations: annotations.len() as u64,
            segments: segment_count,
        })
    }

    fn capsules<'a>(
        &self,
        dataset: &'a semantic_abir::AbirDataset,
    ) -> Result<Vec<&'a semantic_abir::SourceCapsule>, AdapterError> {
        let namespace = binding_namespace(&self.profile.id, dataset)?;
        Ok(dataset
            .source_capsules()
            .iter()
            .filter(|capsule| capsule.source().namespace() == namespace)
            .collect())
    }
}

impl Adapter for EdfAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = self.parse(entry, ValidationLimits::default())?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: vec![
                "recording wall-clock has no timezone in EDF and remains explicitly uncertain"
                    .to_owned(),
            ],
            required_resources: BTreeMap::from([
                ("channels".to_owned(), parsed.channels),
                ("annotations".to_owned(), parsed.annotations),
                ("time-segments".to_owned(), parsed.segments),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let parsed = self.parse(self.entry(source)?, limits)?;
        Ok(ImportOutcome {
            dataset: parsed.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ExactSemantic,
                entries: parsed.mappings,
                preserved_unknowns: 0,
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: parsed.payloads,
        })
    }

    fn plan_export(
        &self,
        dataset: &semantic_abir::AbirDataset,
    ) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings: capsules
                .iter()
                .map(|capsule| {
                    exact(
                        capsule.source().value().to_owned(),
                        capsule.source().value().to_owned(),
                    )
                })
                .collect(),
            requires_user_acceptance: false,
            unsupported: capsules.len() != 1,
            plan_id: String::new(),
        };
        plan.plan_id = plan_id(&plan);
        Ok(plan)
    }

    fn export(
        &self,
        dataset: &semantic_abir::AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        let expected = self.plan_export(dataset)?;
        if expected != *plan || plan_id(plan) != plan.plan_id {
            return Err(AdapterError::ExportPlanMismatch);
        }
        if !plan.accepts_without_loss() {
            return Err(AdapterError::UnsupportedMeaning(
                "dataset lacks one identity-bound EDF source capsule".to_owned(),
            ));
        }
        let capsule = self.capsules(dataset)?[0];
        let bytes = payloads.resolve(capsule.content_id())?;
        if payload_content_id(&bytes) != capsule.content_id() {
            return Err(AdapterError::MissingPayload(capsule.content_id()));
        }
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries: vec![ForeignEntry {
                    path: capsule.source().value().to_owned(),
                    media_type: capsule.media_type().map(str::to_owned),
                    bytes,
                }],
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: true,
                semantic_equivalence: true,
                output_content_ids: vec![capsule.content_id().to_string()],
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self
            .entry(source)
            .and_then(|entry| self.parse(entry, ValidationLimits::default()));
        ValidationArtifact {
            profile: self.profile.id.clone(),
            internal_valid: result.is_ok(),
            independent_validator: self.profile.required_validator.clone(),
            independent_valid: None,
            diagnostics: result
                .err()
                .map(|error| error.to_string())
                .into_iter()
                .collect(),
        }
    }
}

fn signal_headers(edf: &EdfFile) -> Result<Vec<SignalHeader>, AdapterError> {
    let n = edf.n_signals_total;
    let expected = 256_usize
        .checked_add(n.checked_mul(256).ok_or(AdapterError::SourceTooLarge)?)
        .ok_or(AdapterError::SourceTooLarge)?;
    if edf.raw_header.len() != expected {
        return Err(AdapterError::InvalidSource(
            "EDF retained header length is inconsistent".to_owned(),
        ));
    }
    let sh = &edf.raw_header[256..];
    const WIDTHS: [usize; 10] = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32];
    let mut offsets = [0_usize; 10];
    for index in 1..WIDTHS.len() {
        offsets[index] = offsets[index - 1] + WIDTHS[index - 1] * n;
    }
    let field = |kind: usize, channel: usize| {
        let start = offsets[kind] + channel * WIDTHS[kind];
        ascii(&sh[start..start + WIDTHS[kind]])
    };
    (0..n)
        .map(|channel| {
            Ok(SignalHeader {
                label: field(0, channel)?,
                unit: field(2, channel)?,
                physical_min: decimal(&field(3, channel)?)?,
                physical_max: decimal(&field(4, channel)?)?,
                digital_min: field(5, channel)?.parse().map_err(|_| {
                    AdapterError::InvalidSource("invalid EDF digital minimum".to_owned())
                })?,
                digital_max: field(6, channel)?.parse().map_err(|_| {
                    AdapterError::InvalidSource("invalid EDF digital maximum".to_owned())
                })?,
                samples_per_record: field(8, channel)?.parse().map_err(|_| {
                    AdapterError::InvalidSource("invalid EDF samples per record".to_owned())
                })?,
            })
        })
        .collect()
}

fn channel_values(
    edf: &EdfFile,
    source_index: usize,
    samples_per_record: usize,
) -> Result<Vec<i64>, AdapterError> {
    if let Some(position) = edf
        .eeg_indices
        .iter()
        .position(|index| *index == source_index)
    {
        return Ok(edf.signal[position].clone());
    }
    let raw = edf
        .non_eeg_data
        .iter()
        .find(|(index, _)| *index == source_index)
        .map(|(_, bytes)| bytes)
        .ok_or_else(|| AdapterError::InvalidSource("EDF channel payload is missing".to_owned()))?;
    let width = if edf.is_bdf { 3 } else { 2 };
    let expected = samples_per_record
        .checked_mul(edf.n_data_records)
        .and_then(|samples| samples.checked_mul(width))
        .ok_or(AdapterError::SourceTooLarge)?;
    if raw.len() != expected {
        return Err(AdapterError::InvalidSource(
            "EDF off-rate channel extent is inconsistent".to_owned(),
        ));
    }
    Ok(raw
        .chunks_exact(width)
        .map(|sample| {
            if width == 2 {
                i64::from(i16::from_le_bytes([sample[0], sample[1]]))
            } else {
                let raw = u32::from(sample[0])
                    | (u32::from(sample[1]) << 8)
                    | (u32::from(sample[2]) << 16);
                i64::from(if raw & 0x80_0000 != 0 {
                    (raw | 0xff00_0000) as i32
                } else {
                    raw as i32
                })
            }
        })
        .collect())
}

fn annotations(
    edf: &EdfFile,
    headers: &[SignalHeader],
) -> Result<(Vec<Rational>, Vec<Annotation>), AdapterError> {
    let mut onsets = vec![None; edf.n_data_records];
    let mut annotations = Vec::new();
    let width = if edf.is_bdf { 3 } else { 2 };
    for (channel, header) in headers.iter().enumerate() {
        if !is_annotation(&header.label) {
            continue;
        }
        let raw = edf
            .non_eeg_data
            .iter()
            .find(|(index, _)| *index == channel)
            .map(|(_, bytes)| bytes)
            .ok_or_else(|| {
                AdapterError::InvalidSource("EDF annotation payload is missing".to_owned())
            })?;
        let record_bytes = header
            .samples_per_record
            .checked_mul(width)
            .ok_or(AdapterError::SourceTooLarge)?;
        if raw.len() != record_bytes.saturating_mul(edf.n_data_records) {
            return Err(AdapterError::InvalidSource(
                "EDF annotation extent is inconsistent".to_owned(),
            ));
        }
        for (record, chunk) in raw.chunks_exact(record_bytes).enumerate() {
            let bytes = chunk.to_vec();
            for tal in bytes.split(|byte| *byte == 0).filter(|tal| !tal.is_empty()) {
                let Some(separator) = tal.iter().position(|byte| *byte == 0x14) else {
                    return Err(AdapterError::InvalidSource("malformed EDF+ TAL".to_owned()));
                };
                let timing = &tal[..separator];
                let mut timing_parts = timing.split(|byte| *byte == 0x15);
                let onset = decimal(&ascii(timing_parts.next().unwrap())?)?;
                let duration = timing_parts
                    .next()
                    .filter(|value| !value.is_empty())
                    .map(|value| decimal(&ascii(value)?))
                    .transpose()?;
                if timing_parts.next().is_some() {
                    return Err(AdapterError::InvalidSource(
                        "malformed EDF+ TAL timing".to_owned(),
                    ));
                }
                let texts = tal[separator + 1..]
                    .split(|byte| *byte == 0x14)
                    .filter(|text| !text.is_empty())
                    .map(utf8_text)
                    .collect::<Result<Vec<_>, _>>()?;
                if onsets[record].is_none() {
                    onsets[record] = Some(onset);
                }
                annotations.extend(texts.into_iter().map(|text| Annotation {
                    onset,
                    duration,
                    text,
                }));
            }
        }
    }
    Ok((onsets.into_iter().flatten().collect(), annotations))
}

fn calibration(header: &SignalHeader) -> Result<Calibration, AdapterError> {
    let digital_span = header
        .digital_max
        .checked_sub(header.digital_min)
        .ok_or(AdapterError::SourceTooLarge)?;
    if digital_span == 0 {
        return Err(AdapterError::InvalidSource(
            "EDF calibration has zero digital span".to_owned(),
        ));
    }
    let scale = divide_by(
        subtract(header.physical_max, header.physical_min)?,
        digital_span,
    )?;
    let offset = subtract(header.physical_min, multiply_by(scale, header.digital_min)?)?;
    Calibration::new(scale, offset, unit_concept(&header.unit)?)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn decimal(value: &str) -> Result<Rational, AdapterError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(AdapterError::InvalidSource("empty EDF decimal".to_owned()));
    }
    let (mantissa, exponent) = match value.find(['e', 'E']) {
        Some(index) => (
            &value[..index],
            value[index + 1..].parse::<i32>().map_err(|_| {
                AdapterError::InvalidSource("invalid EDF decimal exponent".to_owned())
            })?,
        ),
        None => (value, 0),
    };
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.strip_prefix(['-', '+']).unwrap_or(mantissa);
    let mut digits = String::new();
    let mut fractional = 0_i32;
    let mut seen_dot = false;
    for character in unsigned.chars() {
        if character == '.' && !seen_dot {
            seen_dot = true;
        } else if character.is_ascii_digit() {
            digits.push(character);
            fractional += i32::from(seen_dot);
        } else {
            return Err(AdapterError::InvalidSource(format!(
                "invalid EDF decimal {value:?}"
            )));
        }
    }
    if digits.is_empty() {
        return Err(AdapterError::InvalidSource(format!(
            "invalid EDF decimal {value:?}"
        )));
    }
    let mut numerator = digits
        .parse::<i128>()
        .map_err(|_| AdapterError::SourceTooLarge)?;
    if negative {
        numerator = numerator
            .checked_neg()
            .ok_or(AdapterError::SourceTooLarge)?;
    }
    let power = exponent - fractional;
    if power >= 0 {
        numerator = numerator
            .checked_mul(ten(power as u32)?)
            .ok_or(AdapterError::SourceTooLarge)?;
        Rational::new(numerator, 1).map_err(|error| AdapterError::InvalidSource(error.to_string()))
    } else {
        Rational::new(numerator, ten((-power) as u32)?)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))
    }
}

fn ten(power: u32) -> Result<i128, AdapterError> {
    (0..power).try_fold(1_i128, |value, _| {
        value.checked_mul(10).ok_or(AdapterError::SourceTooLarge)
    })
}

fn add(left: Rational, right: Rational) -> Result<Rational, AdapterError> {
    let (ln, ld) = left.parts();
    let (rn, rd) = right.parts();
    let numerator = ln
        .checked_mul(rd)
        .and_then(|value| {
            rn.checked_mul(ld)
                .and_then(|other| value.checked_add(other))
        })
        .ok_or(AdapterError::SourceTooLarge)?;
    let denominator = ld.checked_mul(rd).ok_or(AdapterError::SourceTooLarge)?;
    Rational::new(numerator, denominator)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn subtract(left: Rational, right: Rational) -> Result<Rational, AdapterError> {
    let (rn, rd) = right.parts();
    add(
        left,
        Rational::new(rn.checked_neg().ok_or(AdapterError::SourceTooLarge)?, rd)
            .map_err(|error| AdapterError::InvalidSource(error.to_string()))?,
    )
}

fn multiply_by(value: Rational, integer: i128) -> Result<Rational, AdapterError> {
    let (numerator, denominator) = value.parts();
    Rational::new(
        numerator
            .checked_mul(integer)
            .ok_or(AdapterError::SourceTooLarge)?,
        denominator,
    )
    .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn divide_by(value: Rational, integer: i128) -> Result<Rational, AdapterError> {
    if integer == 0 {
        return Err(AdapterError::InvalidSource(
            "division by zero in EDF exact number".to_owned(),
        ));
    }
    let (numerator, denominator) = value.parts();
    Rational::new(
        numerator,
        denominator
            .checked_mul(integer)
            .ok_or(AdapterError::SourceTooLarge)?,
    )
    .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn samples_per_second(samples: usize, duration: Rational) -> Result<Rational, AdapterError> {
    let samples = i128::try_from(samples).map_err(|_| AdapterError::SourceTooLarge)?;
    let (duration_numerator, duration_denominator) = duration.parts();
    if duration_numerator <= 0 {
        return Err(AdapterError::InvalidSource(
            "EDF record duration must be positive".to_owned(),
        ));
    }
    Rational::new(
        samples
            .checked_mul(duration_denominator)
            .ok_or(AdapterError::SourceTooLarge)?,
        duration_numerator,
    )
    .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn ascii(bytes: &[u8]) -> Result<String, AdapterError> {
    if bytes.iter().any(|byte| *byte > 0x7f) {
        return Err(AdapterError::InvalidSource(
            "EDF text is not ASCII/UTF-8".to_owned(),
        ));
    }
    Ok(std::str::from_utf8(bytes)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?
        .trim_matches(|character: char| character == '\0' || character == ' ')
        .to_owned())
}

fn utf8_text(bytes: &[u8]) -> Result<String, AdapterError> {
    Ok(std::str::from_utf8(bytes)
        .map_err(|error| AdapterError::InvalidSource(error.to_string()))?
        .trim_matches(|character: char| character == '\0' || character == ' ')
        .to_owned())
}

fn unit_concept(unit: &str) -> Result<ConceptId, AdapterError> {
    let normalized = match unit.trim() {
        "uV" | "µV" => "ucum:uV".to_owned(),
        "mV" => "ucum:mV".to_owned(),
        "V" => "ucum:V".to_owned(),
        other => format!("edf:unit/{}", blake3::hash(other.as_bytes()).to_hex()),
    };
    ConceptId::new(normalized).map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn channel_concept(label: &str, index: usize) -> Result<ConceptId, AdapterError> {
    ConceptId::new(format!(
        "edf:channel/{index}-{}",
        &blake3::hash(label.as_bytes()).to_hex()[..16]
    ))
    .map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn source_key(namespace: &str, value: &str) -> Result<SourceKey, AdapterError> {
    SourceKey::new(namespace, value).map_err(|error| AdapterError::InvalidSource(error.to_string()))
}

fn exact(source_path: String, target: String) -> MappingEntry {
    MappingEntry {
        source_path,
        target,
        disposition: MappingDisposition::Exact,
        reason: None,
    }
}

fn id<T>(seed: &blake3::Hash, domain: &[u8], index: u64) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lamquant.edf-semantic.v1\0");
    hasher.update(seed.as_bytes());
    hasher.update(domain);
    hasher.update(&index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    ObjectId::from_bytes(bytes)
}

fn is_annotation(label: &str) -> bool {
    label.to_ascii_lowercase().contains("annotation")
}
