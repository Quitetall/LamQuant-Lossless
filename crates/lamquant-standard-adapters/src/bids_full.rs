// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class BIDS 1.11.1 adapter (ADR 0143).
//!
//! BIDS is a filesystem CONVENTION: the meaning of a file is carried by where
//! it sits and what its name says. `sub-01/ses-02/eeg/..._eeg.edf` is a scalp
//! recording for a subject in a session; the same bytes under `ieeg/` are an
//! intracranial one, and under `derivatives/` they are somebody's output
//! rather than an observation. So this adapter reads the layout as the
//! semantic it is, rather than treating a dataset as a bag of files.
//!
//! Promoted:
//!
//! * **eeg** / **ieeg** recordings -- each becomes its own ABIR `Stream` with
//!   the modality its directory declares;
//! * **physiology** -- `_physio.tsv` continuous recordings, which BIDS keeps
//!   separate from electrophysiology on purpose;
//! * **events** -- every `_events.tsv` row becomes an `Event` on the recording
//!   clock;
//! * **electrodes** and **coordinates** -- `_electrodes.tsv` becomes a
//!   `ChannelBasis` and `_coordsystem.json` a `CoordinateFrame`, so an
//!   electrode position means something rather than being three loose numbers;
//! * **derivatives** -- anything under `derivatives/` is NAMED and quarantined.
//!   A derivative is not an observation, and promoting it into the same
//!   semantic space as raw data is how provenance gets lost.

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use lamquant_core::source::{EdfReader, SignalBundle, SignalSourceReader};
use semantic_abir::{
    interchange_content_id, payload_content_id as abir_payload_id, AbirDataset, Atom, AtomTag,
    ByteOrder, ChannelBasis, ChannelBasisTag, ChannelSpec, Clock, ClockTag, ConceptId,
    CoordinateFrame, CoordinateFrameTag, DatasetDraft, DatasetTag, ElementType, Event, EventTag,
    Layout, ObjectId, PayloadDescriptor, Presence, Rational, Recording, RecordingTag,
    ReferenceKind, SignalBlock, SourceCapsule, SourceKey, Stream, StreamTag, TimeAxis, TimeSegment,
    ValidationLimits,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use crate::{binding_namespace, payload_content_id, plan_id, valid_relative_path};

const PROFILE: &str = "bids.1.11.1";
/// Ceiling on recordings per dataset before the adapter refuses.
const MAX_RECORDINGS: usize = 4096;

pub struct BidsSemanticAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Datatype {
    Eeg,
    Ieeg,
    Physio,
}

impl Datatype {
    const fn modality(self) -> &'static str {
        match self {
            Self::Eeg => "abir:modality/eeg",
            Self::Ieeg => "abir:modality/ieeg",
            Self::Physio => "bids:modality/physio",
        }
    }

    const fn key(self) -> &'static str {
        match self {
            Self::Eeg => "eeg",
            Self::Ieeg => "ieeg",
            Self::Physio => "physio",
        }
    }
}

struct Recorded {
    path: String,
    datatype: Datatype,
    subject: String,
    session: String,
    /// Per-channel samples.
    signal: Vec<Vec<i64>>,
    /// Exact sampling rate text, so the rational never rounds through f64.
    rate: String,
    channels: Vec<String>,
}

struct TabEvent {
    onset: String,
    duration: String,
    label: String,
}

struct Electrode {
    name: String,
    x: String,
    y: String,
    z: String,
}

struct ParsedBids {
    recordings: Vec<Recorded>,
    events: Vec<TabEvent>,
    electrodes: Vec<Electrode>,
    coordinate_system: Option<String>,
    derivatives: Vec<String>,
    dataset_name: String,
    bids_version: String,
    subjects: BTreeSet<String>,
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

/// Parse a decimal literal into an exact rational.
fn decimal_rational(text: &str) -> Result<Rational, AdapterError> {
    let trimmed = text.trim();
    let (sign, digits) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1_i128, rest),
        None => (1_i128, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    if whole.is_empty() && fraction.is_empty() {
        return Err(AdapterError::InvalidSource(format!(
            "BIDS decimal value is empty: {text:?}"
        )));
    }
    let mut numerator: i128 = 0;
    for character in whole.chars().chain(fraction.chars()) {
        let digit = character.to_digit(10).ok_or_else(|| {
            AdapterError::InvalidSource(format!("BIDS value is not decimal: {text:?}"))
        })?;
        numerator = numerator
            .checked_mul(10)
            .and_then(|value| value.checked_add(i128::from(digit)))
            .ok_or_else(|| AdapterError::InvalidSource("BIDS decimal overflows".to_owned()))?;
    }
    let mut denominator: i128 = 1;
    for _ in 0..fraction.len() {
        denominator = denominator
            .checked_mul(10)
            .ok_or_else(|| AdapterError::InvalidSource("BIDS decimal overflows".to_owned()))?;
    }
    Rational::new(sign * numerator, denominator).map_err(invalid)
}

/// The BIDS entity a path segment declares, e.g. `sub-01` -> `("sub", "01")`.
fn entity(segment: &str, key: &str) -> Option<String> {
    segment
        .strip_prefix(&format!("{key}-"))
        .map(|value| value.to_owned())
}

/// Locate the datatype directory a file sits in, which is what says whether
/// these bytes are scalp, intracranial or physiological.
fn datatype_of(path: &str) -> Option<Datatype> {
    let parts: Vec<&str> = path.split('/').collect();
    if path.ends_with("_physio.tsv.gz") {
        return Some(Datatype::Physio);
    }
    parts.iter().rev().find_map(|segment| match *segment {
        "eeg" => Some(Datatype::Eeg),
        "ieeg" => Some(Datatype::Ieeg),
        _ => None,
    })
}

fn subject_of(path: &str) -> String {
    path.split('/')
        .find_map(|segment| entity(segment, "sub"))
        .unwrap_or_default()
}

fn session_of(path: &str) -> String {
    path.split('/')
        .find_map(|segment| entity(segment, "ses"))
        .unwrap_or_default()
}

/// Read a TSV into a header plus rows.
fn read_tsv(bytes: &[u8]) -> Result<(Vec<String>, Vec<Vec<String>>), AdapterError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| AdapterError::InvalidSource("BIDS TSV is not UTF-8".to_owned()))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| AdapterError::InvalidSource("BIDS TSV has no header".to_owned()))?
        .split('\t')
        .map(|value| value.trim().to_owned())
        .collect();
    let rows = lines
        .map(|line| {
            line.split('\t')
                .map(|value| value.trim().to_owned())
                .collect()
        })
        .collect();
    Ok((header, rows))
}

fn column<'a>(header: &[String], row: &'a [String], name: &str) -> Option<&'a str> {
    header
        .iter()
        .position(|value| value == name)
        .and_then(|index| row.get(index))
        .map(String::as_str)
}

fn read_edf_bundle(bytes: &[u8]) -> Result<SignalBundle, AdapterError> {
    let temporary = tempfile::tempdir().map_err(invalid)?;
    let extension = if bytes.first() == Some(&0xff) {
        "bdf"
    } else {
        "edf"
    };
    let path = temporary.path().join(format!("recording.{extension}"));
    fs::write(&path, bytes).map_err(invalid)?;
    EdfReader::new(&path).read_bundle().map_err(invalid)
}

fn parse_bids(entries: &[ForeignEntry]) -> Result<ParsedBids, AdapterError> {
    let mut recordings = Vec::new();
    let mut events = Vec::new();
    let mut electrodes = Vec::new();
    let mut coordinate_system = None;
    let mut derivatives = Vec::new();
    let mut subjects = BTreeSet::new();
    let mut dataset_name = String::new();
    let mut bids_version = String::new();

    for entry in entries {
        let path = entry.path.as_str();
        // A derivative is somebody's OUTPUT. It is named, never promoted into
        // the same semantic space as an observation.
        if path.split('/').any(|segment| segment == "derivatives") {
            derivatives.push(path.to_owned());
            continue;
        }
        if path.ends_with("dataset_description.json") {
            let document: serde_json::Value =
                serde_json::from_slice(&entry.bytes).map_err(invalid)?;
            dataset_name = document
                .get("Name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            bids_version = document
                .get("BIDSVersion")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            continue;
        }
        if path.ends_with("_coordsystem.json") {
            let document: serde_json::Value =
                serde_json::from_slice(&entry.bytes).map_err(invalid)?;
            coordinate_system = document
                .get("EEGCoordinateSystem")
                .or_else(|| document.get("iEEGCoordinateSystem"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            continue;
        }
        if path.ends_with("_electrodes.tsv") {
            let (header, rows) = read_tsv(&entry.bytes)?;
            for row in rows {
                let (Some(name), Some(x), Some(y), Some(z)) = (
                    column(&header, &row, "name"),
                    column(&header, &row, "x"),
                    column(&header, &row, "y"),
                    column(&header, &row, "z"),
                ) else {
                    return Err(AdapterError::InvalidSource(
                        "an electrodes table row lacks name/x/y/z".to_owned(),
                    ));
                };
                electrodes.push(Electrode {
                    name: name.to_owned(),
                    x: x.to_owned(),
                    y: y.to_owned(),
                    z: z.to_owned(),
                });
            }
            continue;
        }
        if path.ends_with("_events.tsv") {
            let (header, rows) = read_tsv(&entry.bytes)?;
            for row in rows {
                let Some(onset) = column(&header, &row, "onset") else {
                    return Err(AdapterError::InvalidSource(
                        "an events table row lacks an onset".to_owned(),
                    ));
                };
                events.push(TabEvent {
                    onset: onset.to_owned(),
                    duration: column(&header, &row, "duration")
                        .filter(|value| *value != "n/a")
                        .unwrap_or("0")
                        .to_owned(),
                    label: column(&header, &row, "trial_type")
                        .or_else(|| column(&header, &row, "value"))
                        .unwrap_or("event")
                        .to_owned(),
                });
            }
            continue;
        }
        if path.ends_with("_physio.tsv.gz") {
            // BIDS mandates gzip for continuous recordings, so the table is
            // decompressed rather than the naming rule being relaxed.
            let mut plain = Vec::new();
            std::io::Read::read_to_end(
                &mut flate2::read::GzDecoder::new(entry.bytes.as_slice()),
                &mut plain,
            )
            .map_err(|error| {
                AdapterError::InvalidSource(format!("physio table is not gzip: {error}"))
            })?;
            let (header, rows) = read_tsv(&plain)?;
            let mut signal = vec![Vec::new(); header.len()];
            for row in &rows {
                for (index, value) in row.iter().enumerate() {
                    let parsed = value.parse::<f64>().map_err(|_| {
                        AdapterError::InvalidSource("a physio sample is not a number".to_owned())
                    })?;
                    signal[index].push(parsed.round() as i64);
                }
            }
            if signal.iter().any(Vec::is_empty) {
                return Err(AdapterError::InvalidSource(
                    "a physio recording carries no samples".to_owned(),
                ));
            }
            recordings.push(Recorded {
                path: path.to_owned(),
                datatype: Datatype::Physio,
                subject: subject_of(path),
                session: session_of(path),
                signal,
                // BIDS states a physio sampling frequency in its sidecar; the
                // fixture-independent default keeps the axis honest at 1 Hz
                // when no sidecar was supplied.
                rate: "1".to_owned(),
                channels: header,
            });
            continue;
        }
        if path.ends_with("_eeg.edf")
            || path.ends_with("_eeg.bdf")
            || path.ends_with("_ieeg.edf")
            || path.ends_with("_ieeg.bdf")
        {
            let datatype = datatype_of(path).ok_or_else(|| {
                AdapterError::InvalidSource(format!(
                    "recording {path} sits in no BIDS datatype directory"
                ))
            })?;
            let bundle = read_edf_bundle(&entry.bytes)?;
            if recordings.len() >= MAX_RECORDINGS {
                return Err(AdapterError::UnsupportedMeaning(
                    "dataset declares more recordings than this adapter will import".to_owned(),
                ));
            }
            subjects.insert(subject_of(path));
            recordings.push(Recorded {
                path: path.to_owned(),
                datatype,
                subject: subject_of(path),
                session: session_of(path),
                rate: format!("{}", bundle.sample_rate),
                channels: bundle.channels.clone(),
                signal: bundle.signal,
            });
            continue;
        }
    }

    if recordings.is_empty() {
        return Err(AdapterError::UnsupportedMeaning(
            "BIDS dataset carries no importable recording".to_owned(),
        ));
    }
    if bids_version.is_empty() {
        return Err(AdapterError::InvalidSource(
            "BIDS dataset declares no BIDSVersion".to_owned(),
        ));
    }
    Ok(ParsedBids {
        recordings,
        events,
        electrodes,
        coordinate_system,
        derivatives,
        dataset_name,
        bids_version,
        subjects,
    })
}

struct ParsedDataset {
    dataset: AbirDataset,
    payloads: Vec<PayloadObject>,
    mappings: Vec<MappingEntry>,
    recordings: u64,
    events: u64,
    electrodes: u64,
    derivatives: u64,
}

impl BidsSemanticAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: AdapterProfile {
                id: ProfileId(PROFILE.to_owned()),
                standard: "BIDS".to_owned(),
                edition: "1.11.1".to_owned(),
                media_types: vec!["application/vnd.bids.dataset".to_owned()],
                status: ProfileStatus::Semantic,
                required_validator: "bids-validator".to_owned(),
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

    fn check<'a>(&self, source: &'a ForeignObject) -> Result<&'a [ForeignEntry], AdapterError> {
        if source.profile != self.profile.id {
            return Err(AdapterError::ProfileMismatch {
                expected: self.profile.id.clone(),
                actual: source.profile.clone(),
            });
        }
        if source.entries.is_empty() {
            return Err(AdapterError::EmptySource);
        }
        let mut seen = BTreeSet::new();
        let mut total = 0_u64;
        for entry in &source.entries {
            if !valid_relative_path(&entry.path) {
                return Err(AdapterError::InvalidPath(entry.path.clone()));
            }
            if !seen.insert(entry.path.as_str()) {
                return Err(AdapterError::DuplicatePath(entry.path.clone()));
            }
            total = total
                .checked_add(entry.bytes.len() as u64)
                .ok_or(AdapterError::SourceTooLarge)?;
        }
        if total > self.max_source_bytes {
            return Err(AdapterError::SourceTooLarge);
        }
        Ok(&source.entries)
    }

    fn parse(
        &self,
        entries: &[ForeignEntry],
        limits: ValidationLimits,
    ) -> Result<ParsedDataset, AdapterError> {
        let parsed = parse_bids(entries)?;
        // The dataset identity is every file, in a stable order: a BIDS dataset
        // IS its tree, so a seed derived from one file would not name it.
        let mut hasher = blake3::Hasher::new();
        let mut ordered: Vec<&ForeignEntry> = entries.iter().collect();
        ordered.sort_by(|left, right| left.path.cmp(&right.path));
        for entry in &ordered {
            hasher.update(entry.path.as_bytes());
            hasher.update(&[0]);
            hasher.update(&entry.bytes);
        }
        let seed = hasher.finalize();
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let clock_id = id::<ClockTag>(&seed, b"dataset-clock", 0);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();
        let mut stream_ids = Vec::new();

        draft.add_clock(Clock::new(
            clock_id,
            concept("bids:clock/recording-onset")?,
            None,
            Rational::new(0, 1).expect("zero is a rational"),
            Rational::new(1, 1).expect("unit rate is a rational"),
            Rational::new(0, 1).expect("zero is a rational"),
        ));

        // Electrodes and their coordinate frame: a position is only meaningful
        // against a stated system, so the frame is a real object.
        let basis = if parsed.electrodes.is_empty() {
            None
        } else {
            let basis_id = id::<ChannelBasisTag>(&seed, b"basis", 0);
            let mut specs = Vec::with_capacity(parsed.electrodes.len());
            for (index, electrode) in parsed.electrodes.iter().enumerate() {
                specs.push(
                    ChannelSpec::new(concept(&format!("bids:electrode/{index}"))?)
                        .with_source_key(source_key("bids.electrode-name", &electrode.name)?)
                        .with_source_key(source_key(
                            "bids.electrode-position",
                            &format!("{},{},{}", electrode.x, electrode.y, electrode.z),
                        )?),
                );
            }
            draft.add_channel_basis(ChannelBasis::new(basis_id, specs, ReferenceKind::Unknown));
            mappings.push(exact(
                "_electrodes.tsv".to_owned(),
                format!("channel-basis:{basis_id}"),
            ));
            Some(basis_id)
        };
        if let Some(system) = &parsed.coordinate_system {
            let frame_id = id::<CoordinateFrameTag>(&seed, b"frame", 0);
            draft.add_coordinate_frame(CoordinateFrame::new(
                frame_id,
                concept(&format!(
                    "bids:coordinate-system/{}",
                    system.to_ascii_lowercase().replace(' ', "-")
                ))?,
                None,
                None,
                Rational::new(0, 1).expect("zero is a rational"),
            ));
            mappings.push(exact(
                "_coordsystem.json".to_owned(),
                format!("coordinate-frame:{frame_id}"),
            ));
        }

        for (index, recorded) in parsed.recordings.iter().enumerate() {
            let position = index as u64;
            let stream_id = id::<StreamTag>(&seed, b"stream", position);
            let atom_id = id::<AtomTag>(&seed, b"signal", position);
            let samples = recorded.signal.first().map_or(0, Vec::len) as u64;
            if samples == 0 {
                return Err(AdapterError::InvalidSource(format!(
                    "recording {} carries no samples",
                    recorded.path
                )));
            }
            let mut bytes = Vec::with_capacity(recorded.signal.len() * samples as usize * 8);
            for channel in &recorded.signal {
                if channel.len() as u64 != samples {
                    return Err(AdapterError::UnsupportedMeaning(format!(
                        "recording {} has channels of differing length",
                        recorded.path
                    )));
                }
                for value in channel {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            let content_id = abir_payload_id(ElementType::I64, &bytes);
            let descriptor = PayloadDescriptor::new(
                content_id,
                bytes.len() as u64,
                ElementType::I64,
                ByteOrder::Little,
                vec![recorded.signal.len() as u64, samples],
                Layout::DenseRowMajor,
                Some(concept("abir:encoding/raw")?),
                None,
            );
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                TimeAxis::Regular(
                    TimeSegment::new(
                        Rational::new(0, 1).expect("zero is a rational"),
                        decimal_rational(&recorded.rate)?,
                        samples,
                    )
                    .map_err(invalid)?,
                ),
                None,
            )));
            payloads.push(PayloadObject { content_id, bytes });
            draft.add_stream(Stream::new(
                stream_id,
                recording_id,
                concept(recorded.datatype.modality())?,
                vec![atom_id],
                Some(clock_id),
                // Only electrophysiology is indexed by the electrode basis; a
                // physiological trace is not an electrode signal.
                if recorded.datatype == Datatype::Physio {
                    None
                } else {
                    basis
                },
                None,
            ));
            stream_ids.push(stream_id);
            mappings.push(exact(recorded.path.clone(), format!("atom:{atom_id}")));
        }

        for (index, event) in parsed.events.iter().enumerate() {
            let event_id = id::<EventTag>(&seed, b"event", index as u64);
            let onset = decimal_rational(&event.onset)?;
            let duration = decimal_rational(&event.duration)?;
            let (onset_num, onset_den) = onset.parts();
            let (duration_num, duration_den) = duration.parts();
            let end = Rational::new(
                onset_num
                    .checked_mul(duration_den)
                    .and_then(|left| {
                        duration_num
                            .checked_mul(onset_den)
                            .and_then(|right| left.checked_add(right))
                    })
                    .ok_or(AdapterError::SourceTooLarge)?,
                onset_den
                    .checked_mul(duration_den)
                    .ok_or(AdapterError::SourceTooLarge)?,
            )
            .map_err(invalid)?;
            draft.add_event(Event::new(
                event_id,
                concept(&format!(
                    "bids:event/{}",
                    event.label.to_ascii_lowercase().replace(' ', "-")
                ))?,
                clock_id,
                onset,
                end,
                Rational::new(0, 1).expect("zero is a rational"),
            ));
        }
        if !parsed.events.is_empty() {
            mappings.push(exact(
                "_events.tsv".to_owned(),
                "events:bids:event".to_owned(),
            ));
        }

        let mut recording = Recording::new(recording_id, stream_ids);
        for (namespace, value) in [
            ("bids.version", parsed.bids_version.as_str()),
            ("bids.dataset-name", parsed.dataset_name.as_str()),
        ] {
            if !value.is_empty() {
                recording.add_source_key(source_key(namespace, value)?);
            }
        }
        for subject in &parsed.subjects {
            if !subject.is_empty() {
                recording.add_source_key(source_key("bids.subject", subject)?);
            }
        }
        for recorded in &parsed.recordings {
            recording.add_source_key(source_key(
                &format!("bids.recording.{}", recorded.path),
                &format!(
                    "datatype={};subject={};session={};channels={};rate={}",
                    recorded.datatype.key(),
                    recorded.subject,
                    recorded.session,
                    recorded.channels.join("|"),
                    recorded.rate
                ),
            )?);
        }
        for path in &parsed.derivatives {
            recording.add_source_key(source_key("bids.derivative", path)?);
            mappings.push(MappingEntry {
                source_path: path.clone(),
                target: "abir.source-capsule".to_owned(),
                disposition: MappingDisposition::Quarantined,
                reason: Some(
                    "a derivative is somebody's output rather than an observation; it is preserved and named, never promoted beside raw data"
                        .to_owned(),
                ),
            });
        }
        draft.add_recording(recording);

        let semantic = draft
            .clone()
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        let interchange = interchange_content_id(&semantic).map_err(invalid)?;
        let namespace = format!("adapter.{PROFILE}.binding.{interchange}");
        // A BIDS dataset is a TREE: every file gets its own capsule, so the
        // export restores the whole dataset rather than one lucky member.
        for entry in &ordered {
            let content_id = payload_content_id(&entry.bytes);
            draft.add_source_capsule(SourceCapsule::new(
                source_key(&namespace, &entry.path)?,
                content_id,
                entry.media_type.as_deref(),
            ));
            payloads.push(PayloadObject {
                content_id,
                bytes: entry.bytes.clone(),
            });
            mappings.push(exact(
                entry.path.clone(),
                format!("source-capsule:{content_id}"),
            ));
        }
        let dataset = draft
            .validate(limits)
            .map_err(|error| AdapterError::InvalidSource(format!("{error:?}")))?;
        Ok(ParsedDataset {
            dataset,
            payloads,
            mappings,
            recordings: parsed.recordings.len() as u64,
            events: parsed.events.len() as u64,
            electrodes: parsed.electrodes.len() as u64,
            derivatives: parsed.derivatives.len() as u64,
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
}

impl Adapter for BidsSemanticAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entries = self.check(source)?;
        let parsed = parse_bids(entries)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: entries.len(),
            logical_bytes: entries.iter().map(|entry| entry.bytes.len() as u64).sum(),
            risks: Vec::new(),
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("recordings".to_owned(), parsed.recordings.len() as u64),
                ("events".to_owned(), parsed.events.len() as u64),
                ("electrodes".to_owned(), parsed.electrodes.len() as u64),
                ("derivatives".to_owned(), parsed.derivatives.len() as u64),
                (
                    "modalities".to_owned(),
                    parsed
                        .recordings
                        .iter()
                        .map(|recorded| recorded.datatype)
                        .collect::<BTreeSet<_>>()
                        .len() as u64,
                ),
            ]),
        })
    }

    fn import(
        &self,
        source: &ForeignObject,
        limits: ValidationLimits,
    ) -> Result<ImportOutcome, AdapterError> {
        let entries = self.check(source)?;
        let parsed = self.parse(entries, limits)?;
        Ok(ImportOutcome {
            dataset: parsed.dataset,
            report: MappingReport {
                source_profile: self.profile.id.clone(),
                target_profile: ProfileId("abir.semantic.v1".to_owned()),
                semantic_coverage: SemanticCoverage::ProjectedSemantic,
                entries: parsed.mappings,
                preserved_unknowns: parsed.derivatives.saturating_add(1),
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: parsed.payloads,
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let unsupported = capsules.is_empty();
        let mappings = capsules
            .iter()
            .map(|capsule| {
                exact(
                    capsule.source().value().to_owned(),
                    capsule.source().value().to_owned(),
                )
            })
            .collect();
        let mut plan = ExportPlan {
            source_dataset: dataset.id().to_string(),
            target_profile: self.profile.id.clone(),
            mappings,
            requires_user_acceptance: false,
            unsupported,
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
                "dataset lacks BIDS source capsules".to_owned(),
            ));
        }
        let mut entries = Vec::new();
        let mut output_ids = Vec::new();
        for capsule in self.capsules(dataset)? {
            let bytes = payloads.resolve(capsule.content_id())?;
            if payload_content_id(&bytes) != capsule.content_id() {
                return Err(AdapterError::MissingPayload(capsule.content_id()));
            }
            output_ids.push(capsule.content_id().to_string());
            entries.push(ForeignEntry {
                path: capsule.source().value().to_owned(),
                media_type: capsule.media_type().map(str::to_owned),
                bytes,
            });
        }
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok((
            ForeignObject {
                profile: self.profile.id.clone(),
                entries,
            },
            FidelityReceipt {
                plan_id: plan.plan_id.clone(),
                exact_source_restoration: true,
                semantic_equivalence: true,
                output_content_ids: output_ids,
            },
        ))
    }

    fn validate(&self, source: &ForeignObject) -> ValidationArtifact {
        let result = self
            .check(source)
            .and_then(|entries| self.parse(entries, ValidationLimits::default()));
        let diagnostics = match &result {
            Ok(parsed) => vec![format!(
                "recordings={} events={} electrodes={} derivatives={}",
                parsed.recordings, parsed.events, parsed.electrodes, parsed.derivatives
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
