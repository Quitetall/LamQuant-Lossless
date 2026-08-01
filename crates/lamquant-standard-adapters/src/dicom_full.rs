// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class DICOM PS3 waveform adapter (ADR 0143).
//!
//! DICOM's information model is the point: a waveform is not a loose array, it
//! belongs to a Series, inside a Study, for a Patient, acquired by a Device.
//! Dropping that hierarchy would turn identified clinical data into an
//! anonymous signal, so each level becomes its own ABIR catalog record joined
//! by typed `SourceRelationship` edges rather than by string convention.
//!
//! On top of the waveform itself this adapter promotes:
//!
//! * **annotations** -- the Waveform Annotation Sequence -- to `Event`s on the
//!   acquisition clock, since an annotation names a moment, not a row;
//! * **referenced media** and **reports** -- Referenced Image / Waveform /
//!   Structured Report sequences -- as named source keys and quarantined
//!   mapping entries. Those instances live in other files this adapter was not
//!   handed; inlining them would fabricate content;
//! * **private tags** -- every private element -- as source keys carrying their
//!   own group/element numbers, so vendor content is visible rather than
//!   silently discarded, and byte-exact in the source capsule regardless.
//!
//! Channel sensitivity and baseline become an ABIR `Calibration`, so the
//! samples stay the integers the file stored while the physical scale is
//! carried exactly beside them.

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use dicom_core::{header::HasLength, value::DataSetSequence, DataElement, PrimitiveValue, Tag, VR};
use dicom_object::{FileDicomObject, FileMetaTableBuilder, InMemDicomObject};
use semantic_abir::{
    interchange_content_id, payload_content_id as abir_payload_id, verify_payload_content,
    AbirDataset, Acquisition, AcquisitionTag, Atom, AtomTag, ByteOrder, Calibration, ChannelBasis,
    ChannelBasisTag, ChannelSpec, Clock, ClockTag, ConceptId, DatasetDraft, DatasetTag, Device,
    DeviceTag, ElementType, Event, EventTag, Layout, ObjectId, Patient, PatientTag,
    PayloadDescriptor, Presence, Rational, Recording, RecordingTag, ReferenceKind, Session,
    SessionTag, SignalBlock, SourceCapsule, SourceKey, SourceRelationship, Stream, StreamTag,
    TimeAxis, TimeSegment, ValidationLimits,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{binding_namespace, payload_content_id, plan_id, valid_relative_path};

const PROFILE: &str = "dicom.ps3.2026c";

pub struct DicomSemanticAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
}

type Dicom = FileDicomObject<InMemDicomObject>;

struct ChannelDefinition {
    label: String,
    sensitivity: Option<f64>,
    correction: Option<f64>,
    baseline: Option<f64>,
    unit: String,
}

struct Multiplex {
    channels: Vec<ChannelDefinition>,
    samples: u64,
    /// Sampling frequency as EXACT text, so the rational conversion never
    /// rounds through binary floating point.
    frequency: String,
    values: Vec<i64>,
}

struct Annotation {
    text: String,
    /// First and last referenced sample position, 1-based in DICOM.
    first_sample: Option<u64>,
    last_sample: Option<u64>,
}

struct ParsedDicom {
    patient_id: String,
    patient_name: String,
    study_uid: String,
    series_uid: String,
    modality: String,
    manufacturer: String,
    model: String,
    serial: String,
    multiplexes: Vec<Multiplex>,
    annotations: Vec<Annotation>,
    referenced_media: Vec<String>,
    reports: Vec<String>,
    private_tags: Vec<(String, String)>,
}

fn invalid(error: impl std::fmt::Display) -> AdapterError {
    AdapterError::InvalidSource(error.to_string())
}

fn concept(value: &str) -> Result<ConceptId, AdapterError> {
    ConceptId::new(value).map_err(invalid)
}

fn source_key(namespace: &str, value: &str) -> Result<SourceKey, AdapterError> {
    SourceKey::new(namespace, value).map_err(invalid)
}

fn exact(source_path: String, target: String) -> MappingEntry {
    MappingEntry {
        source_path,
        target,
        disposition: MappingDisposition::Exact,
        reason: None,
    }
}

fn quarantined(source_path: String, reason: &str) -> MappingEntry {
    MappingEntry {
        source_path,
        target: "abir.source-capsule".to_owned(),
        disposition: MappingDisposition::Quarantined,
        reason: Some(reason.to_owned()),
    }
}

fn id<T>(seed: &blake3::Hash, domain: &[u8], index: u64) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(seed.as_bytes());
    hasher.update(domain);
    hasher.update(&index.to_le_bytes());
    let digest = hasher.finalize();
    let mut raw = [0_u8; 16];
    raw.copy_from_slice(&digest.as_bytes()[..16]);
    ObjectId::from_bytes(raw)
}

/// Parse a DICOM decimal string into an exact rational.
fn decimal_rational(text: &str) -> Result<Rational, AdapterError> {
    let trimmed = text.trim();
    let (sign, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1_i128, rest),
        None => (1_i128, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    if whole.is_empty() && fraction.is_empty() {
        return Err(AdapterError::InvalidSource(format!(
            "DICOM decimal string is empty: {text:?}"
        )));
    }
    let mut numerator: i128 = 0;
    for character in whole.chars().chain(fraction.chars()) {
        let digit = character.to_digit(10).ok_or_else(|| {
            AdapterError::InvalidSource(format!("DICOM decimal string is not numeric: {text:?}"))
        })?;
        numerator = numerator
            .checked_mul(10)
            .and_then(|value| value.checked_add(i128::from(digit)))
            .ok_or_else(|| AdapterError::InvalidSource("DICOM decimal overflows".to_owned()))?;
    }
    let mut denominator: i128 = 1;
    for _ in 0..fraction.len() {
        denominator = denominator
            .checked_mul(10)
            .ok_or_else(|| AdapterError::InvalidSource("DICOM decimal overflows".to_owned()))?;
    }
    Rational::new(sign * numerator, denominator).map_err(invalid)
}

fn text_of(object: &InMemDicomObject, group: u16, element: u16) -> String {
    object
        .element_opt(dicom_core::Tag(group, element))
        .ok()
        .flatten()
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_default()
}

fn number_of(object: &InMemDicomObject, group: u16, element: u16) -> Option<f64> {
    object
        .element_opt(dicom_core::Tag(group, element))
        .ok()
        .flatten()
        .and_then(|value| value.to_float64().ok())
}

fn items_of(object: &InMemDicomObject, group: u16, element: u16) -> Vec<&InMemDicomObject> {
    object
        .element_opt(dicom_core::Tag(group, element))
        .ok()
        .flatten()
        .and_then(|value| value.items())
        .map(|items| items.iter().collect())
        .unwrap_or_default()
}

/// Collect the referenced SOP instances a sequence names.
fn referenced_instances(object: &InMemDicomObject, group: u16, element: u16) -> Vec<String> {
    items_of(object, group, element)
        .into_iter()
        .map(|item| {
            let class = text_of(item, 0x0008, 0x1150);
            let instance = text_of(item, 0x0008, 0x1155);
            format!("class={class};instance={instance}")
        })
        .filter(|value| value != "class=;instance=")
        .collect()
}

fn parse_channel(item: &InMemDicomObject, index: usize) -> ChannelDefinition {
    // The channel source is a code sequence; its meaning is the human label.
    let label = items_of(item, 0x003A, 0x0208)
        .first()
        .map(|code| text_of(code, 0x0008, 0x0104))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("channel{index}"));
    let unit = items_of(item, 0x003A, 0x0211)
        .first()
        .map(|code| text_of(code, 0x0008, 0x0104))
        .unwrap_or_default();
    ChannelDefinition {
        label,
        sensitivity: number_of(item, 0x003A, 0x0210),
        correction: number_of(item, 0x003A, 0x0212),
        baseline: number_of(item, 0x003A, 0x0213),
        unit,
    }
}

fn parse_dicom(bytes: &[u8]) -> Result<ParsedDicom, AdapterError> {
    let object: Dicom = dicom_object::from_reader(std::io::Cursor::new(bytes)).map_err(invalid)?;
    let root = object.into_inner();

    let mut multiplexes = Vec::new();
    for group in items_of(&root, 0x5400, 0x0100) {
        let channel_count = number_of(group, 0x003A, 0x0005).ok_or_else(|| {
            AdapterError::InvalidSource(
                "waveform group declares no NumberOfWaveformChannels".to_owned(),
            )
        })? as usize;
        let samples = number_of(group, 0x003A, 0x0010).ok_or_else(|| {
            AdapterError::InvalidSource(
                "waveform group declares no NumberOfWaveformSamples".to_owned(),
            )
        })? as u64;
        if channel_count == 0 || samples == 0 {
            return Err(AdapterError::InvalidSource(
                "waveform group declares an empty shape".to_owned(),
            ));
        }
        let frequency = text_of(group, 0x003A, 0x001A);
        if frequency.is_empty() {
            return Err(AdapterError::InvalidSource(
                "waveform group declares no SamplingFrequency".to_owned(),
            ));
        }
        let interpretation = text_of(group, 0x5400, 0x1006);
        if interpretation != "SS" && interpretation != "SB" && interpretation != "US" {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "waveform sample interpretation {interpretation:?} has no exact integer promotion"
            )));
        }
        let raw = root
            .element_opt(dicom_core::Tag(0x5400, 0x0100))
            .ok()
            .flatten()
            .and_then(|value| value.items())
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| std::ptr::eq(*item, group))
                    .and_then(|item| item.element_opt(dicom_core::Tag(0x5400, 0x1010)).ok())
                    .flatten()
            })
            .ok_or_else(|| {
                AdapterError::InvalidSource("waveform group carries no WaveformData".to_owned())
            })?;
        // WaveformData is a run of 16-bit words (VR OW) or bytes; the sample
        // interpretation -- not the VR -- says how to READ them. A signed
        // waveform is routinely carried in an unsigned VR, so the words are
        // reinterpreted rather than range-checked against the VR's type.
        let values: Vec<i64> = match interpretation.as_str() {
            "SB" => raw
                .to_bytes()
                .map_err(invalid)?
                .iter()
                .map(|value| i64::from(*value as i8))
                .collect(),
            "US" => raw
                .to_multi_int::<u16>()
                .map_err(invalid)?
                .into_iter()
                .map(i64::from)
                .collect(),
            _ => raw
                .to_multi_int::<u16>()
                .map_err(invalid)?
                .into_iter()
                .map(|word| i64::from(word as i16))
                .collect(),
        };
        let expected = (channel_count as u64)
            .checked_mul(samples)
            .ok_or(AdapterError::SourceTooLarge)?;
        if values.len() as u64 != expected {
            return Err(AdapterError::InvalidSource(format!(
                "waveform data holds {} values but its shape declares {expected}",
                values.len()
            )));
        }
        let channels = items_of(group, 0x003A, 0x0200)
            .into_iter()
            .enumerate()
            .map(|(index, item)| parse_channel(item, index))
            .collect::<Vec<_>>();
        if channels.len() != channel_count {
            return Err(AdapterError::InvalidSource(
                "channel definitions do not match the declared channel count".to_owned(),
            ));
        }
        multiplexes.push(Multiplex {
            channels,
            samples,
            frequency,
            values,
        });
    }
    if multiplexes.is_empty() {
        return Err(AdapterError::UnsupportedMeaning(
            "this profile covers DICOM waveform instances; the file carries none".to_owned(),
        ));
    }

    let annotations = items_of(&root, 0x0040, 0xB020)
        .into_iter()
        .map(|item| {
            let positions = item
                .element_opt(dicom_core::Tag(0x0040, 0xA132))
                .ok()
                .flatten()
                .and_then(|value| value.to_multi_int::<u32>().ok())
                .unwrap_or_default();
            // An annotation may carry free text, a coded concept, a NUMERIC
            // measurement, or some combination. Reading only the text would
            // silently drop the measurements -- which is exactly the quiet loss
            // this adapter exists to prevent -- so every form is read and NO
            // annotation is discarded for being uninterpretable.
            let name = items_of(item, 0x0040, 0xA043)
                .first()
                .map(|code| text_of(code, 0x0008, 0x0104))
                .unwrap_or_default();
            let text = text_of(item, 0x0040, 0xA160);
            let numeric = text_of(item, 0x0040, 0xA30A);
            let units = items_of(item, 0x0040, 0x08EA)
                .first()
                .map(|code| text_of(code, 0x0008, 0x0104))
                .unwrap_or_default();
            let rendered = match (name.as_str(), text.as_str(), numeric.as_str()) {
                (_, value, _) if !value.is_empty() => {
                    if name.is_empty() {
                        value.to_owned()
                    } else {
                        format!("{name}={value}")
                    }
                }
                (_, _, value) if !value.is_empty() => {
                    let measurement = if units.is_empty() {
                        value.to_owned()
                    } else {
                        format!("{value} {units}")
                    };
                    if name.is_empty() {
                        measurement
                    } else {
                        format!("{name}={measurement}")
                    }
                }
                (concept_name, _, _) if !concept_name.is_empty() => concept_name.to_owned(),
                _ => "uninterpreted-annotation".to_owned(),
            };
            Annotation {
                text: rendered,
                first_sample: positions.first().map(|value| u64::from(*value)),
                last_sample: positions.last().map(|value| u64::from(*value)),
            }
        })
        .collect::<Vec<_>>();

    let mut referenced_media = referenced_instances(&root, 0x0008, 0x1140);
    referenced_media.extend(referenced_instances(&root, 0x0008, 0x113A));
    let mut reports = referenced_instances(&root, 0x0008, 0x1110);
    reports.extend(referenced_instances(&root, 0x0040, 0xA375));

    // Every private element, named by its own numbers. A vendor extension the
    // adapter cannot interpret must still be visible.
    let mut private_tags = Vec::new();
    for element in root.iter() {
        let tag = element.header().tag;
        if tag.group() % 2 == 1 {
            private_tags.push((
                format!("{:04x},{:04x}", tag.group(), tag.element()),
                element
                    .value()
                    .to_str()
                    .map(|value| value.trim().to_owned())
                    .unwrap_or_else(|_| format!("{} bytes", element.header().length().0)),
            ));
        }
    }

    Ok(ParsedDicom {
        patient_id: text_of(&root, 0x0010, 0x0020),
        patient_name: text_of(&root, 0x0010, 0x0010),
        study_uid: text_of(&root, 0x0020, 0x000D),
        series_uid: text_of(&root, 0x0020, 0x000E),
        modality: text_of(&root, 0x0008, 0x0060),
        manufacturer: text_of(&root, 0x0008, 0x0070),
        model: text_of(&root, 0x0008, 0x1090),
        serial: text_of(&root, 0x0018, 0x1000),
        multiplexes,
        annotations,
        referenced_media,
        reports,
        private_tags,
    })
}

struct ParsedDataset {
    dataset: AbirDataset,
    payloads: Vec<PayloadObject>,
    mappings: Vec<MappingEntry>,
    channels: u64,
    annotations: u64,
    referenced: u64,
    reports: u64,
    private_tags: u64,
}

struct SemanticDicomSignal<'a> {
    atom_id: ObjectId<AtomTag>,
    descriptor: &'a PayloadDescriptor,
    calibration: Option<&'a Calibration>,
    label: &'a str,
}

struct SemanticDicomGroup<'a> {
    rate: Rational,
    samples: u64,
    signals: Vec<SemanticDicomSignal<'a>>,
}

struct SemanticDicom<'a> {
    patient_id: &'a str,
    patient_name: &'a str,
    study_uid: &'a str,
    series_uid: &'a str,
    modality: &'a str,
    manufacturer: &'a str,
    model: &'a str,
    serial: Option<&'a str>,
    groups: Vec<SemanticDicomGroup<'a>>,
    events: Vec<(&'a Event, &'a str)>,
}

impl DicomSemanticAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: AdapterProfile {
                id: ProfileId(PROFILE.to_owned()),
                standard: "DICOM".to_owned(),
                edition: "PS3 2026c".to_owned(),
                media_types: vec!["application/dicom".to_owned()],
                status: ProfileStatus::Semantic,
                required_validator: "pydicom".to_owned(),
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
                "DICOM semantic profile requires exactly one instance".to_owned(),
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
    ) -> Result<ParsedDataset, AdapterError> {
        let parsed = parse_dicom(&entry.bytes)?;
        let seed = blake3::hash(&entry.bytes);
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let clock_id = id::<ClockTag>(&seed, b"acquisition-clock", 0);
        let patient_id = id::<PatientTag>(&seed, b"patient", 0);
        let session_id = id::<SessionTag>(&seed, b"study", 0);
        let acquisition_id = id::<AcquisitionTag>(&seed, b"series", 0);
        let device_id = id::<DeviceTag>(&seed, b"device", 0);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();

        // The information model, as typed records joined by typed edges. A
        // waveform detached from its Study and Patient is a different -- and
        // clinically useless -- object.
        let mut patient = Patient::new(patient_id, concept("dicom:entity/patient")?);
        if !parsed.patient_id.is_empty() {
            patient = patient.with_source_key(source_key("dicom.patient-id", &parsed.patient_id)?);
        }
        if !parsed.patient_name.is_empty() {
            patient =
                patient.with_source_key(source_key("dicom.patient-name", &parsed.patient_name)?);
        }
        draft.add_patient(patient);
        let mut session = Session::new(session_id, concept("dicom:entity/study")?);
        if !parsed.study_uid.is_empty() {
            session =
                session.with_source_key(source_key("dicom.study-instance-uid", &parsed.study_uid)?);
        }
        draft.add_session(session);
        let mut acquisition = Acquisition::new(acquisition_id, concept("dicom:entity/series")?);
        if !parsed.series_uid.is_empty() {
            acquisition = acquisition
                .with_source_key(source_key("dicom.series-instance-uid", &parsed.series_uid)?);
        }
        if !parsed.modality.is_empty() {
            acquisition =
                acquisition.with_source_key(source_key("dicom.modality", &parsed.modality)?);
        }
        draft.add_acquisition(acquisition);
        let mut device = Device::new(device_id, concept("dicom:entity/equipment")?);
        for (namespace, value) in [
            ("dicom.manufacturer", parsed.manufacturer.as_str()),
            ("dicom.manufacturer-model", parsed.model.as_str()),
            ("dicom.device-serial-number", parsed.serial.as_str()),
        ] {
            if !value.is_empty() {
                device = device.with_source_key(source_key(namespace, value)?);
            }
        }
        draft.add_device(device);
        for relationship in [
            SourceRelationship::SessionPatient {
                session_id,
                patient_id,
            },
            SourceRelationship::AcquisitionSession {
                acquisition_id,
                session_id,
            },
            SourceRelationship::AcquisitionDevice {
                acquisition_id,
                device_id,
            },
            SourceRelationship::AcquisitionRecording {
                acquisition_id,
                recording_id,
            },
        ] {
            draft.add_source_relationship(relationship);
        }
        for (source_path, target, has_value) in [
            (
                "(0010,0020) PatientID",
                format!("patient:{patient_id}"),
                !parsed.patient_id.is_empty(),
            ),
            (
                "(0010,0010) PatientName",
                format!("patient:{patient_id}"),
                !parsed.patient_name.is_empty(),
            ),
            (
                "(0020,000D) StudyInstanceUID",
                format!("session:{session_id}"),
                !parsed.study_uid.is_empty(),
            ),
            (
                "(0020,000E) SeriesInstanceUID",
                format!("acquisition:{acquisition_id}"),
                !parsed.series_uid.is_empty(),
            ),
            (
                "(0008,0060) Modality",
                format!("acquisition:{acquisition_id}"),
                !parsed.modality.is_empty(),
            ),
            (
                "(0008,0070) Manufacturer",
                format!("device:{device_id}"),
                !parsed.manufacturer.is_empty(),
            ),
            (
                "(0008,1090) ManufacturerModelName",
                format!("device:{device_id}"),
                !parsed.model.is_empty(),
            ),
            (
                "(0018,1000) DeviceSerialNumber",
                format!("device:{device_id}"),
                !parsed.serial.is_empty(),
            ),
        ] {
            if has_value {
                mappings.push(exact(source_path.to_owned(), target));
            }
        }

        // One clock: every waveform group in an instance is sampled against the
        // same acquisition time base.
        draft.add_clock(Clock::new(
            clock_id,
            concept("dicom:clock/acquisition")?,
            None,
            Rational::new(0, 1).expect("zero is a rational"),
            Rational::new(1, 1).expect("unit rate is a rational"),
            Rational::new(0, 1).expect("zero is a rational"),
        ));

        let mut atom_ids = Vec::new();
        let mut channel_specs = Vec::new();
        let mut total_channels = 0_u64;
        for (group_index, multiplex) in parsed.multiplexes.iter().enumerate() {
            let rate = decimal_rational(&multiplex.frequency)?;
            for (channel_index, channel) in multiplex.channels.iter().enumerate() {
                let position = (group_index * 1024 + channel_index) as u64;
                let atom_id = id::<AtomTag>(&seed, b"channel", position);
                // The samples stay the integers the file stored; the physical
                // scale rides beside them as an exact calibration.
                let mut bytes = Vec::with_capacity(multiplex.samples as usize * 8);
                for sample in 0..multiplex.samples as usize {
                    let index = sample * multiplex.channels.len() + channel_index;
                    bytes.extend_from_slice(&multiplex.values[index].to_le_bytes());
                }
                let content_id = abir_payload_id(ElementType::I64, &bytes);
                let descriptor = PayloadDescriptor::new(
                    content_id,
                    bytes.len() as u64,
                    ElementType::I64,
                    ByteOrder::Little,
                    vec![1, multiplex.samples],
                    Layout::DenseRowMajor,
                    Some(concept("abir:encoding/raw")?),
                    None,
                );
                // Sensitivity and baseline are the physical scale the file
                // states; carrying them exactly is what lets the samples stay
                // the integers DICOM stored.
                let calibration = match channel.sensitivity {
                    Some(sensitivity) => {
                        let scale = sensitivity * channel.correction.unwrap_or(1.0);
                        Some(
                            Calibration::new(
                                decimal_rational(&format!("{scale:.9}"))?,
                                decimal_rational(&format!(
                                    "{:.9}",
                                    channel.baseline.unwrap_or(0.0)
                                ))?,
                                concept(&unit_concept(&channel.unit))?,
                            )
                            .map_err(invalid)?,
                        )
                    }
                    None => None,
                };
                draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                    atom_id,
                    Presence::Present,
                    Some(descriptor),
                    TimeAxis::Regular(
                        TimeSegment::new(
                            Rational::new(0, 1).expect("zero is a rational"),
                            rate,
                            multiplex.samples,
                        )
                        .map_err(invalid)?,
                    ),
                    calibration,
                )));
                payloads.push(PayloadObject { content_id, bytes });
                atom_ids.push(atom_id);
                channel_specs.push(
                    ChannelSpec::new(concept(&format!(
                        "dicom:channel/{group_index}-{channel_index}"
                    ))?)
                    .with_source_key(source_key("dicom.channel-source", &channel.label)?),
                );
                mappings.push(exact(
                    format!("(5400,0100)[{group_index}] channel {channel_index}"),
                    format!("atom:{atom_id}"),
                ));
                total_channels += 1;
            }
        }

        let basis_id = id::<ChannelBasisTag>(&seed, b"basis", 0);
        draft.add_channel_basis(ChannelBasis::new(
            basis_id,
            channel_specs,
            ReferenceKind::Unknown,
        ));

        let stream_id = id::<StreamTag>(&seed, b"stream", 0);
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            concept(if parsed.modality.eq_ignore_ascii_case("EEG") {
                "abir:modality/eeg"
            } else if parsed.modality.eq_ignore_ascii_case("ECG") {
                "abir:modality/ecg"
            } else {
                "abir:modality/unknown"
            })?,
            atom_ids,
            Some(clock_id),
            Some(basis_id),
            None,
        ));

        // Annotations mark moments, so they become events rather than rows.
        let sample_rate = decimal_rational(&parsed.multiplexes[0].frequency)?;
        for (index, annotation) in parsed.annotations.iter().enumerate() {
            let event_id = id::<EventTag>(&seed, b"annotation", index as u64);
            let (start, end) = match (annotation.first_sample, annotation.last_sample) {
                (Some(first), Some(last)) => (
                    sample_position_seconds(first, sample_rate)?,
                    sample_position_seconds(last, sample_rate)?,
                ),
                _ => (
                    Rational::new(0, 1).expect("zero is a rational"),
                    Rational::new(0, 1).expect("zero is a rational"),
                ),
            };
            draft.add_event(Event::new(
                event_id,
                concept("dicom:event/waveform-annotation")?,
                clock_id,
                start,
                end,
                Rational::new(0, 1).expect("zero is a rational"),
            ));
        }
        if !parsed.annotations.is_empty() {
            mappings.push(exact(
                "(0040,B020) WaveformAnnotationSequence".to_owned(),
                "events:dicom:event/waveform-annotation".to_owned(),
            ));
        }

        let mut recording = Recording::new(recording_id, vec![stream_id]);
        for (index, annotation) in parsed.annotations.iter().enumerate() {
            recording.add_source_key(source_key(
                &format!("dicom.annotation.{index}"),
                &annotation.text,
            )?);
        }
        for (index, reference) in parsed.referenced_media.iter().enumerate() {
            recording.add_source_key(source_key(
                &format!("dicom.referenced-media.{index}"),
                reference,
            )?);
            mappings.push(quarantined(
                format!("referenced media {index}"),
                "the referenced SOP instance lives in another file that was not part of this source object",
            ));
        }
        for (index, reference) in parsed.reports.iter().enumerate() {
            recording.add_source_key(source_key(&format!("dicom.report.{index}"), reference)?);
            mappings.push(quarantined(
                format!("referenced report {index}"),
                "the referenced report instance lives in another file that was not part of this source object",
            ));
        }
        for (tag, value) in &parsed.private_tags {
            recording.add_source_key(source_key(&format!("dicom.private.{tag}"), value)?);
        }
        if !parsed.private_tags.is_empty() {
            mappings.push(quarantined(
                "private elements".to_owned(),
                "vendor private elements are named and preserved byte-exact, but their meaning is not interpreted",
            ));
        }
        draft.add_recording(recording);

        let semantic = draft
            .clone()
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        let interchange = interchange_content_id(&semantic).map_err(invalid)?;
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
        Ok(ParsedDataset {
            dataset,
            payloads,
            mappings,
            channels: total_channels,
            annotations: parsed.annotations.len() as u64,
            referenced: parsed.referenced_media.len() as u64,
            reports: parsed.reports.len() as u64,
            private_tags: parsed.private_tags.len() as u64,
        })
    }

    fn capsules<'a>(
        &self,
        dataset: &'a AbirDataset,
    ) -> Result<Vec<&'a semantic_abir::SourceCapsule>, AdapterError> {
        let namespace = binding_namespace(&self.profile.id, dataset)?;
        Ok(dataset
            .source_capsules()
            .iter()
            .filter(|capsule| capsule.source().namespace() == namespace)
            .collect())
    }

    fn semantic_dicom<'a>(
        &self,
        dataset: &'a AbirDataset,
    ) -> Result<SemanticDicom<'a>, AdapterError> {
        if dataset.recordings().len() != 1
            || dataset.streams().len() != 1
            || dataset.clocks().len() != 1
            || dataset.channel_bases().len() != 1
            || dataset.patients().len() != 1
            || dataset.sessions().len() != 1
            || dataset.acquisitions().len() != 1
            || dataset.devices().len() != 1
            || !dataset.coordinate_frames().is_empty()
            || !dataset.policies().is_empty()
            || !dataset.proofs().is_empty()
            || !dataset.derivations().is_empty()
            || !dataset.fidelity().is_empty()
            || !dataset.observed_execution().is_empty()
            || !dataset.subjects().is_empty()
            || !dataset.sensors().is_empty()
            || !dataset.channels().is_empty()
            || !dataset.clock_relations().is_empty()
            || !dataset.frame_transforms().is_empty()
            || !dataset.concept_dictionaries().is_empty()
            || !dataset.derived_artifacts().is_empty()
        {
            return Err(unsupported(
                "DICOM semantic writeback requires one waveform information model without auxiliary ABIR catalogs",
            ));
        }
        let recording = &dataset.recordings()[0];
        let stream = &dataset.streams()[0];
        let clock = &dataset.clocks()[0];
        let basis = &dataset.channel_bases()[0];
        let patient = &dataset.patients()[0];
        let session = &dataset.sessions()[0];
        let acquisition = &dataset.acquisitions()[0];
        let device = &dataset.devices()[0];
        if recording.streams() != [stream.id()]
            || stream.recording_id() != recording.id()
            || stream.clock_id() != Some(clock.id())
            || stream.channel_basis_id() != Some(basis.id())
            || stream.policy_id().is_some()
            || stream.atoms().len() != basis.channels().len()
            || stream.atoms().len() != dataset.atoms().len()
            || stream.atoms().is_empty()
            || clock.kind().as_str() != "dicom:clock/acquisition"
            || clock.parent_id().is_some()
            || clock.offset() != Rational::new(0, 1).unwrap()
            || clock.rate() != Rational::new(1, 1).unwrap()
            || clock.uncertainty() != Rational::new(0, 1).unwrap()
            || patient.kind().as_str() != "dicom:entity/patient"
            || session.kind().as_str() != "dicom:entity/study"
            || acquisition.kind().as_str() != "dicom:entity/series"
            || device.kind().as_str() != "dicom:entity/equipment"
        {
            return Err(unsupported(
                "DICOM waveform stream or information-model identity is inconsistent",
            ));
        }
        let expected_relationships = BTreeSet::from([
            SourceRelationship::SessionPatient {
                session_id: session.id(),
                patient_id: patient.id(),
            },
            SourceRelationship::AcquisitionSession {
                acquisition_id: acquisition.id(),
                session_id: session.id(),
            },
            SourceRelationship::AcquisitionDevice {
                acquisition_id: acquisition.id(),
                device_id: device.id(),
            },
            SourceRelationship::AcquisitionRecording {
                acquisition_id: acquisition.id(),
                recording_id: recording.id(),
            },
        ]);
        if dataset
            .source_relationships()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            != expected_relationships
        {
            return Err(unsupported(
                "DICOM source relationships cannot be represented without loss",
            ));
        }
        let patient_id = required_source_value(patient.source_keys(), "dicom.patient-id")?;
        let patient_name = required_source_value(patient.source_keys(), "dicom.patient-name")?;
        let study_uid = required_source_value(session.source_keys(), "dicom.study-instance-uid")?;
        let series_uid =
            required_source_value(acquisition.source_keys(), "dicom.series-instance-uid")?;
        let modality = required_source_value(acquisition.source_keys(), "dicom.modality")?;
        let manufacturer = required_source_value(device.source_keys(), "dicom.manufacturer")?;
        let model = required_source_value(device.source_keys(), "dicom.manufacturer-model")?;
        let serial = optional_source_value(device.source_keys(), "dicom.device-serial-number")?;
        if !matches!(
            (modality, stream.modality().as_str()),
            ("ECG", "abir:modality/ecg") | ("EEG", "abir:modality/eeg")
        ) {
            return Err(unsupported(
                "DICOM modality disagrees with the ABIR waveform modality",
            ));
        }
        validate_dicom_uid(study_uid)?;
        validate_dicom_uid(series_uid)?;
        for (value, width) in [
            (patient_id, 64),
            (patient_name, 64),
            (manufacturer, 64),
            (model, 64),
        ] {
            validate_dicom_text(value, width)?;
        }
        if let Some(serial) = serial {
            validate_dicom_text(serial, 64)?;
        }
        reject_unknown_keys(
            patient.source_keys(),
            &["dicom.patient-id", "dicom.patient-name"],
        )?;
        reject_unknown_keys(session.source_keys(), &["dicom.study-instance-uid"])?;
        reject_unknown_keys(
            acquisition.source_keys(),
            &["dicom.series-instance-uid", "dicom.modality"],
        )?;
        reject_unknown_keys(
            device.source_keys(),
            &[
                "dicom.manufacturer",
                "dicom.manufacturer-model",
                "dicom.device-serial-number",
            ],
        )?;

        let mut grouped: BTreeMap<usize, Vec<(usize, SemanticDicomSignal<'a>, Rational, u64)>> =
            BTreeMap::new();
        for (atom_id, channel) in stream.atoms().iter().zip(basis.channels()) {
            let (group_index, channel_index) = dicom_channel_indices(channel.concept())?;
            let atom = dataset
                .atoms()
                .iter()
                .find(|atom| atom.id() == *atom_id)
                .ok_or_else(|| unsupported("DICOM stream references a missing signal atom"))?;
            let Atom::SignalBlock(block) = atom else {
                return Err(unsupported(
                    "DICOM waveform channels require SignalBlock atoms",
                ));
            };
            let descriptor = atom
                .payload()
                .ok_or_else(|| unsupported("DICOM waveform channel payload is missing"))?;
            let TimeAxis::Regular(segment) = block.time_axis() else {
                return Err(unsupported(
                    "DICOM waveform channels require regular timing",
                ));
            };
            if segment.start() != Rational::new(0, 1).unwrap()
                || descriptor.element() != ElementType::I64
                || descriptor.byte_order() != ByteOrder::Little
                || descriptor.layout() != &Layout::DenseRowMajor
                || descriptor.shape() != [1, segment.samples()]
                || descriptor.encoding().map(ConceptId::as_str) != Some("abir:encoding/raw")
            {
                return Err(unsupported(
                    "DICOM channels require zero-origin dense little-endian raw I64 signal blocks",
                ));
            }
            dicom_decimal(segment.rate())?;
            if let Some(calibration) = block.calibration() {
                dicom_calibration_decimal(calibration.scale())?;
                dicom_calibration_decimal(calibration.offset())?;
                dicom_unit(calibration.unit())?;
            }
            let label = required_source_value(channel.source_keys(), "dicom.channel-source")?;
            reject_unknown_keys(channel.source_keys(), &["dicom.channel-source"])?;
            if validate_dicom_text(label, 64).is_err() {
                return Err(unsupported("DICOM channel label is not representable"));
            }
            grouped.entry(group_index).or_default().push((
                channel_index,
                SemanticDicomSignal {
                    atom_id: *atom_id,
                    descriptor,
                    calibration: block.calibration(),
                    label,
                },
                segment.rate(),
                segment.samples(),
            ));
        }
        let mut groups = Vec::with_capacity(grouped.len());
        for (expected_group, (group_index, mut channels)) in grouped.into_iter().enumerate() {
            if group_index != expected_group {
                return Err(unsupported(
                    "DICOM waveform group indices are not contiguous",
                ));
            }
            channels.sort_by_key(|(channel, _, _, _)| *channel);
            let rate = channels[0].2;
            let samples = channels[0].3;
            if channels.iter().enumerate().any(|(expected, value)| {
                value.0 != expected || value.2 != rate || value.3 != samples
            }) {
                return Err(unsupported(
                    "DICOM channel indices, rates, or extents disagree within a multiplex group",
                ));
            }
            groups.push(SemanticDicomGroup {
                rate,
                samples,
                signals: channels
                    .into_iter()
                    .map(|(_, signal, _, _)| signal)
                    .collect(),
            });
        }

        let mut annotations = BTreeMap::new();
        for key in recording.source_keys() {
            let Some(index) = key.namespace().strip_prefix("dicom.annotation.") else {
                return Err(unsupported(&format!(
                    "DICOM recording metadata {} requires an exact source capsule",
                    key.namespace()
                )));
            };
            let index = index
                .parse::<usize>()
                .map_err(|_| unsupported("DICOM annotation index is malformed"))?;
            if annotations.insert(index, key.value()).is_some() {
                return Err(unsupported("DICOM annotation index is ambiguous"));
            }
        }
        if annotations.len() != dataset.events().len()
            || annotations.keys().copied().ne(0..annotations.len())
        {
            return Err(unsupported(
                "DICOM annotation text and event counts do not agree",
            ));
        }
        let mut events = Vec::with_capacity(dataset.events().len());
        for (index, event) in dataset.events().iter().enumerate() {
            if event.kind().as_str() != "dicom:event/waveform-annotation"
                || event.clock_id() != clock.id()
                || event.uncertainty() != Rational::new(0, 1).unwrap()
            {
                return Err(unsupported("DICOM annotation event is not representable"));
            }
            event_sample_position(event.start(), groups[0].rate)?;
            event_sample_position(event.end(), groups[0].rate)?;
            if !annotations[&index].is_ascii() || annotations[&index].contains('\0') {
                return Err(unsupported(
                    "DICOM annotation text requires an unsupported character set",
                ));
            }
            events.push((event, annotations[&index]));
        }
        Ok(SemanticDicom {
            patient_id,
            patient_name,
            study_uid,
            series_uid,
            modality,
            manufacturer,
            model,
            serial,
            groups,
            events,
        })
    }
}

/// DICOM sample positions are 1-based; position 1 is time zero.
fn sample_position_seconds(position: u64, rate: Rational) -> Result<Rational, AdapterError> {
    let index = position.saturating_sub(1);
    let (numerator, denominator) = rate.parts();
    if numerator <= 0 {
        return Err(AdapterError::InvalidSource(
            "DICOM sampling frequency must be positive".to_owned(),
        ));
    }
    Rational::new(
        i128::from(index as i64)
            .checked_mul(denominator)
            .ok_or(AdapterError::SourceTooLarge)?,
        numerator,
    )
    .map_err(invalid)
}

fn unit_concept(unit: &str) -> String {
    match unit.trim().to_ascii_lowercase().as_str() {
        "microvolt" | "uv" | "\u{b5}v" => "abir:unit/microvolt".to_owned(),
        "millivolt" | "mv" => "abir:unit/millivolt".to_owned(),
        "volt" | "v" => "abir:unit/volt".to_owned(),
        "" => "dicom:unit/unstated".to_owned(),
        other => format!("dicom:unit/{}", other.replace(' ', "-")),
    }
}

impl Adapter for DicomSemanticAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = parse_dicom(&entry.bytes)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: Vec::new(),
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                (
                    "channels".to_owned(),
                    parsed
                        .multiplexes
                        .iter()
                        .map(|group| group.channels.len() as u64)
                        .sum(),
                ),
                ("annotations".to_owned(), parsed.annotations.len() as u64),
                (
                    "referenced-media".to_owned(),
                    parsed.referenced_media.len() as u64,
                ),
                ("reports".to_owned(), parsed.reports.len() as u64),
                ("private-tags".to_owned(), parsed.private_tags.len() as u64),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = self.parse(entry, limits)?;
        Ok(ImportOutcome {
            dataset: parsed.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ProjectedSemantic,
                entries: parsed.mappings,
                preserved_unknowns: parsed
                    .referenced
                    .saturating_add(parsed.reports)
                    .saturating_add(1),
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: parsed.payloads,
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let semantic = if capsules.len() == 1 {
            None
        } else {
            self.semantic_dicom(dataset).ok()
        };
        let mappings = if let Some(semantic) = semantic.as_ref() {
            semantic
                .groups
                .iter()
                .enumerate()
                .flat_map(|(group, multiplex)| {
                    multiplex
                        .signals
                        .iter()
                        .enumerate()
                        .map(move |(channel, signal)| {
                            exact(
                                format!("atom:{}", signal.atom_id),
                                format!("(5400,0100)[{group}] channel {channel}"),
                            )
                        })
                })
                .collect()
        } else {
            capsules
                .iter()
                .map(|capsule| {
                    exact(
                        capsule.source().value().to_owned(),
                        capsule.source().value().to_owned(),
                    )
                })
                .collect()
        };
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings,
            requires_user_acceptance: false,
            unsupported: capsules.len() != 1 && semantic.is_none(),
            plan_id: String::new(),
        };
        plan.plan_id = plan_id(&plan);
        Ok(plan)
    }

    fn export(
        &self,
        dataset: &AbirDataset,
        plan: &ExportPlan,
        payloads: &dyn PayloadResolver,
    ) -> Result<(ForeignObject, FidelityReceipt), AdapterError> {
        let expected = self.plan_export(dataset)?;
        if expected != *plan || plan_id(plan) != plan.plan_id {
            return Err(AdapterError::ExportPlanMismatch);
        }
        if !plan.accepts_without_loss() {
            return Err(AdapterError::UnsupportedMeaning(
                "dataset lacks one exact DICOM source capsule and cannot be represented by semantic DICOM waveform writeback".to_owned(),
            ));
        }
        let capsules = self.capsules(dataset)?;
        if capsules.len() == 1 {
            let capsule = capsules[0];
            let bytes = payloads.resolve(capsule.content_id())?;
            if payload_content_id(&bytes) != capsule.content_id() {
                return Err(AdapterError::MissingPayload(capsule.content_id()));
            }
            parse_dicom(&bytes)?;
            return Ok((
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
            ));
        }
        let semantic = self.semantic_dicom(dataset)?;
        let bytes = write_semantic_dicom(dataset, &semantic, payloads)?;
        if u64::try_from(bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?
            > self.max_source_bytes
        {
            return Err(AdapterError::SourceTooLarge);
        }
        parse_dicom(&bytes)?;
        let content_id = payload_content_id(&bytes);
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries: vec![ForeignEntry {
                    path: "waveform.dcm".to_owned(),
                    media_type: Some("application/dicom".to_owned()),
                    bytes,
                }],
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: false,
                semantic_equivalence: true,
                output_content_ids: vec![content_id.to_string()],
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self
            .entry(source)
            .and_then(|entry| self.parse(entry, ValidationLimits::default()));
        let diagnostics = match &result {
            Ok(parsed) => vec![format!(
                "channels={} annotations={} referenced-media={} reports={} private-tags={}",
                parsed.channels,
                parsed.annotations,
                parsed.referenced,
                parsed.reports,
                parsed.private_tags
            )],
            Err(error) => vec![error.to_string()],
        };
        ValidationArtifact {
            profile: self.profile.id.clone(),
            internal_valid: result.is_ok(),
            independent_validator: self.profile.required_validator.clone(),
            independent_valid: None,
            diagnostics,
        }
    }
}

fn write_semantic_dicom(
    dataset: &AbirDataset,
    semantic: &SemanticDicom<'_>,
    payloads: &dyn PayloadResolver,
) -> Result<Vec<u8>, AdapterError> {
    const SOP_CLASS: &str = "1.2.840.10008.5.1.4.1.1.9.1.1";
    const TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.1";
    let sop_instance = format!("2.25.{}", u128::from_be_bytes(dataset.id().to_bytes()));
    let mut waveform_items = Vec::with_capacity(semantic.groups.len());
    for group in &semantic.groups {
        let channel_count =
            u16::try_from(group.signals.len()).map_err(|_| AdapterError::SourceTooLarge)?;
        let samples = u32::try_from(group.samples).map_err(|_| AdapterError::SourceTooLarge)?;
        let mut channel_items = Vec::with_capacity(group.signals.len());
        let mut channel_values = Vec::with_capacity(group.signals.len());
        for signal in &group.signals {
            let logical = payloads.resolve(signal.descriptor.content_id())?;
            verify_payload_content(signal.descriptor, &logical)
                .map_err(|_| AdapterError::MissingPayload(signal.descriptor.content_id()))?;
            let values = logical
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                .map(|value| {
                    i16::try_from(value).map_err(|_| {
                        unsupported(
                            "DICOM waveform payload contains a sample outside signed 16-bit range",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() as u64 != group.samples {
                return Err(AdapterError::SourceTooLarge);
            }
            channel_values.push(values);

            let source_code = coded_item(signal.label, "99LQ")?;
            let mut elements = vec![DataElement::new(
                Tag(0x003A, 0x0208),
                VR::SQ,
                DataSetSequence::from(vec![source_code]),
            )];
            if let Some(calibration) = signal.calibration {
                elements.push(DataElement::new(
                    Tag(0x003A, 0x0210),
                    VR::DS,
                    dicom_calibration_decimal(calibration.scale())?,
                ));
                elements.push(DataElement::new(Tag(0x003A, 0x0212), VR::DS, "1"));
                elements.push(DataElement::new(
                    Tag(0x003A, 0x0213),
                    VR::DS,
                    dicom_calibration_decimal(calibration.offset())?,
                ));
                elements.push(DataElement::new(
                    Tag(0x003A, 0x0211),
                    VR::SQ,
                    DataSetSequence::from(vec![coded_item(
                        dicom_unit(calibration.unit())?,
                        "UCUM",
                    )?]),
                ));
            }
            channel_items.push(InMemDicomObject::from_element_iter(elements));
        }
        let mut words = Vec::with_capacity(
            usize::try_from(group.samples)
                .ok()
                .and_then(|samples| samples.checked_mul(group.signals.len()))
                .ok_or(AdapterError::SourceTooLarge)?,
        );
        for sample in 0..usize::try_from(group.samples).map_err(|_| AdapterError::SourceTooLarge)? {
            for channel in &channel_values {
                words.push(channel[sample] as u16);
            }
        }
        waveform_items.push(InMemDicomObject::from_element_iter([
            DataElement::new(
                Tag(0x003A, 0x0005),
                VR::US,
                PrimitiveValue::from(channel_count),
            ),
            DataElement::new(Tag(0x003A, 0x0010), VR::UL, PrimitiveValue::from(samples)),
            DataElement::new(Tag(0x003A, 0x001A), VR::DS, dicom_decimal(group.rate)?),
            DataElement::new(
                Tag(0x003A, 0x0200),
                VR::SQ,
                DataSetSequence::from(channel_items),
            ),
            DataElement::new(Tag(0x5400, 0x1004), VR::US, PrimitiveValue::from(16_u16)),
            DataElement::new(Tag(0x5400, 0x1006), VR::CS, "SS"),
            DataElement::new(
                Tag(0x5400, 0x1010),
                VR::OW,
                PrimitiveValue::U16(words.into()),
            ),
        ]));
    }
    let mut root = InMemDicomObject::new_empty();
    for element in [
        DataElement::new(Tag(0x0008, 0x0016), VR::UI, SOP_CLASS),
        DataElement::new(Tag(0x0008, 0x0018), VR::UI, sop_instance.as_str()),
        DataElement::new(Tag(0x0008, 0x0060), VR::CS, semantic.modality),
        DataElement::new(Tag(0x0008, 0x0070), VR::LO, semantic.manufacturer),
        DataElement::new(Tag(0x0008, 0x1090), VR::LO, semantic.model),
        DataElement::new(Tag(0x0010, 0x0010), VR::PN, semantic.patient_name),
        DataElement::new(Tag(0x0010, 0x0020), VR::LO, semantic.patient_id),
        DataElement::new(Tag(0x0020, 0x000D), VR::UI, semantic.study_uid),
        DataElement::new(Tag(0x0020, 0x000E), VR::UI, semantic.series_uid),
    ] {
        root.put(element);
    }
    if let Some(serial) = semantic.serial {
        root.put(DataElement::new(Tag(0x0018, 0x1000), VR::LO, serial));
    }
    root.put(DataElement::new(
        Tag(0x5400, 0x0100),
        VR::SQ,
        DataSetSequence::from(waveform_items),
    ));
    if !semantic.events.is_empty() {
        let mut annotations = Vec::with_capacity(semantic.events.len());
        for (event, text) in &semantic.events {
            let positions = vec![
                event_sample_position(event.start(), semantic.groups[0].rate)?,
                event_sample_position(event.end(), semantic.groups[0].rate)?,
            ];
            annotations.push(InMemDicomObject::from_element_iter([
                DataElement::new(
                    Tag(0x0040, 0xA132),
                    VR::UL,
                    PrimitiveValue::U32(positions.into()),
                ),
                DataElement::new(Tag(0x0040, 0xA160), VR::UT, *text),
            ]));
        }
        root.put(DataElement::new(
            Tag(0x0040, 0xB020),
            VR::SQ,
            DataSetSequence::from(annotations),
        ));
    }
    let file = root
        .with_meta(FileMetaTableBuilder::new().transfer_syntax(TRANSFER_SYNTAX))
        .map_err(invalid)?;
    let temporary = tempfile::NamedTempFile::new().map_err(invalid)?;
    file.write_to_file(temporary.path()).map_err(invalid)?;
    std::fs::read(temporary.path()).map_err(invalid)
}

fn coded_item(meaning: &str, scheme: &str) -> Result<InMemDicomObject, AdapterError> {
    if meaning.is_empty() || meaning.chars().any(char::is_control) {
        return Err(unsupported("DICOM coded meaning is not representable"));
    }
    let digest = blake3::hash(meaning.as_bytes());
    let code = format!("LQ{}", &digest.to_hex()[..12]);
    Ok(InMemDicomObject::from_element_iter([
        DataElement::new(Tag(0x0008, 0x0100), VR::SH, code),
        DataElement::new(Tag(0x0008, 0x0102), VR::SH, scheme),
        DataElement::new(Tag(0x0008, 0x0104), VR::LO, meaning),
    ]))
}

fn required_source_value<'a>(
    keys: &'a [SourceKey],
    namespace: &str,
) -> Result<&'a str, AdapterError> {
    optional_source_value(keys, namespace)?
        .ok_or_else(|| unsupported(&format!("required DICOM source key {namespace} is missing")))
}

fn optional_source_value<'a>(
    keys: &'a [SourceKey],
    namespace: &str,
) -> Result<Option<&'a str>, AdapterError> {
    let mut matches = keys
        .iter()
        .filter(|key| key.namespace() == namespace)
        .map(SourceKey::value);
    let value = matches.next();
    if matches.next().is_some() {
        return Err(unsupported(&format!(
            "DICOM source key {namespace} is ambiguous"
        )));
    }
    Ok(value)
}

fn reject_unknown_keys(keys: &[SourceKey], allowed: &[&str]) -> Result<(), AdapterError> {
    if let Some(key) = keys.iter().find(|key| !allowed.contains(&key.namespace())) {
        return Err(unsupported(&format!(
            "DICOM source metadata {} has no semantic writer",
            key.namespace()
        )));
    }
    Ok(())
}

fn dicom_channel_indices(concept: &ConceptId) -> Result<(usize, usize), AdapterError> {
    let value = concept
        .as_str()
        .strip_prefix("dicom:channel/")
        .ok_or_else(|| unsupported("DICOM channel concept is not canonical"))?;
    let (group, channel) = value
        .split_once('-')
        .ok_or_else(|| unsupported("DICOM channel concept is malformed"))?;
    Ok((
        group
            .parse()
            .map_err(|_| unsupported("DICOM waveform group index is malformed"))?,
        channel
            .parse()
            .map_err(|_| unsupported("DICOM waveform channel index is malformed"))?,
    ))
}

fn event_sample_position(time: Rational, rate: Rational) -> Result<u32, AdapterError> {
    let (time_numerator, time_denominator) = time.parts();
    let (rate_numerator, rate_denominator) = rate.parts();
    let numerator = time_numerator
        .checked_mul(rate_numerator)
        .ok_or(AdapterError::SourceTooLarge)?;
    let denominator = time_denominator
        .checked_mul(rate_denominator)
        .ok_or(AdapterError::SourceTooLarge)?;
    if numerator < 0 || numerator % denominator != 0 {
        return Err(unsupported(
            "DICOM annotation time is not an exact sample position",
        ));
    }
    u32::try_from(
        numerator
            .checked_div(denominator)
            .and_then(|value| value.checked_add(1))
            .ok_or(AdapterError::SourceTooLarge)?,
    )
    .map_err(|_| AdapterError::SourceTooLarge)
}

fn dicom_unit(unit: &ConceptId) -> Result<&'static str, AdapterError> {
    match unit.as_str() {
        "abir:unit/microvolt" => Ok("microvolt"),
        "abir:unit/millivolt" => Ok("millivolt"),
        "abir:unit/volt" => Ok("volt"),
        "dicom:unit/unstated" => Err(unsupported(
            "DICOM calibrated data requires a stated measurement unit",
        )),
        _ => Err(unsupported("DICOM calibration unit has no exact writer")),
    }
}

fn validate_dicom_uid(value: &str) -> Result<(), AdapterError> {
    if value.is_empty()
        || value.len() > 64
        || value.starts_with('.')
        || value.ends_with('.')
        || value.split('.').any(|component| {
            component.is_empty()
                || !component.bytes().all(|byte| byte.is_ascii_digit())
                || (component.len() > 1 && component.starts_with('0'))
        })
    {
        return Err(unsupported("DICOM UID is not representable"));
    }
    Ok(())
}

fn validate_dicom_text(value: &str, max_bytes: usize) -> Result<(), AdapterError> {
    if value.is_empty()
        || value.len() > max_bytes
        || !value.is_ascii()
        || value.contains('\\')
        || value.chars().any(char::is_control)
    {
        return Err(unsupported("DICOM text value is not representable"));
    }
    Ok(())
}

fn dicom_calibration_decimal(value: Rational) -> Result<String, AdapterError> {
    let rendered = dicom_decimal(value)?;
    let parsed = rendered
        .parse::<f64>()
        .map_err(|_| unsupported("DICOM calibration decimal is malformed"))?;
    let reparsed = decimal_rational(&format!("{parsed:.9}"))?;
    if reparsed != value {
        return Err(unsupported(
            "DICOM calibration would change under the adapter precision contract",
        ));
    }
    Ok(rendered)
}

fn dicom_decimal(value: Rational) -> Result<String, AdapterError> {
    let (numerator, mut denominator) = value.parts();
    let mut twos = 0_u32;
    let mut fives = 0_u32;
    while denominator % 2 == 0 {
        denominator /= 2;
        twos += 1;
    }
    while denominator % 5 == 0 {
        denominator /= 5;
        fives += 1;
    }
    if denominator != 1 {
        return Err(unsupported(
            "DICOM exact number has no finite decimal representation",
        ));
    }
    let digits = twos.max(fives);
    let scaled = numerator
        .checked_mul(checked_power(2, digits - twos)?)
        .and_then(|value| value.checked_mul(checked_power(5, digits - fives).ok()?))
        .ok_or(AdapterError::SourceTooLarge)?;
    let negative = scaled < 0;
    let magnitude = scaled.unsigned_abs().to_string();
    let mut rendered = if digits == 0 {
        magnitude
    } else {
        let digits = usize::try_from(digits).map_err(|_| AdapterError::SourceTooLarge)?;
        if magnitude.len() <= digits {
            format!("0.{}{}", "0".repeat(digits - magnitude.len()), magnitude)
        } else {
            let split = magnitude.len() - digits;
            format!("{}.{}", &magnitude[..split], &magnitude[split..])
        }
    };
    if rendered.contains('.') {
        while rendered.ends_with('0') {
            rendered.pop();
        }
        if rendered.ends_with('.') {
            rendered.pop();
        }
    }
    if negative {
        rendered.insert(0, '-');
    }
    if rendered.len() > 16 {
        return Err(unsupported(
            "DICOM decimal exceeds its sixteen-character value limit",
        ));
    }
    Ok(rendered)
}

fn checked_power(base: i128, power: u32) -> Result<i128, AdapterError> {
    (0..power).try_fold(1_i128, |value, _| {
        value.checked_mul(base).ok_or(AdapterError::SourceTooLarge)
    })
}

fn unsupported(reason: &str) -> AdapterError {
    AdapterError::UnsupportedMeaning(reason.to_owned())
}
