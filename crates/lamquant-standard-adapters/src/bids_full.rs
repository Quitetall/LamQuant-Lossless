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
const MAX_EVENTS: usize = 262_144;
const MAX_ELECTRODES: usize = 262_144;
/// Reject implausibly wide rows before field vectors can dominate memory.
const MAX_TSV_COLUMNS: usize = 16_384;
/// Prevent tiny TSV cells from expanding into millions of heap allocations.
const MAX_TSV_CELLS: usize = 1_000_000;

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
    /// Per-channel samples in their exact promoted numeric domain.
    signal: RecordedSignal,
    /// Exact sampling rate text, so the rational never rounds through f64.
    rate: String,
    /// Exact onset relative to the recording clock.
    start: String,
    channels: Vec<String>,
}

enum RecordedSignal {
    Integer(Vec<Vec<i64>>),
    Real(Vec<Vec<f64>>),
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

struct PhysioSidecar {
    path: String,
    directory: String,
    entities: BTreeSet<String>,
    columns: Option<Vec<String>>,
    sampling_frequency: Option<String>,
    start_time: Option<String>,
}

struct PhysioMetadata {
    columns: Vec<String>,
    sampling_frequency: String,
    start_time: String,
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
    let (sign, unsigned) = match trimmed.strip_prefix('-') {
        Some(rest) => (-1_i128, rest),
        None => (1_i128, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    let (digits, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => {
            if exponent.contains(['e', 'E']) {
                return Err(AdapterError::InvalidSource(format!(
                    "BIDS value is not decimal: {text:?}"
                )));
            }
            let exponent = exponent.parse::<i32>().map_err(|_| {
                AdapterError::InvalidSource(format!("BIDS value has invalid exponent: {text:?}"))
            })?;
            (mantissa, exponent)
        }
        None => (unsigned, 0),
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
    if numerator == 0 {
        return Rational::new(0, 1).map_err(invalid);
    }
    let exponent_magnitude = exponent.unsigned_abs();
    if exponent_magnitude > 38 {
        return Err(AdapterError::InvalidSource(
            "BIDS decimal overflows".to_owned(),
        ));
    }
    if exponent >= 0 {
        for _ in 0..exponent_magnitude {
            numerator = numerator
                .checked_mul(10)
                .ok_or_else(|| AdapterError::InvalidSource("BIDS decimal overflows".to_owned()))?;
        }
    } else {
        for _ in 0..exponent_magnitude {
            denominator = denominator
                .checked_mul(10)
                .ok_or_else(|| AdapterError::InvalidSource("BIDS decimal overflows".to_owned()))?;
        }
    }
    Rational::new(sign * numerator, denominator).map_err(invalid)
}

fn path_directory(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(directory, _)| directory)
}

fn filename_entities(path: &str, suffix: &str) -> Result<BTreeSet<String>, AdapterError> {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let stem = filename
        .strip_suffix(suffix)
        .ok_or_else(|| AdapterError::InvalidSource(format!("{path} does not end in {suffix}")))?;
    let mut entities = BTreeSet::new();
    for token in stem.split('_').filter(|token| !token.is_empty()) {
        let Some((key, value)) = token.split_once('-') else {
            return Err(AdapterError::InvalidSource(format!(
                "BIDS sidecar entity {token:?} has no value"
            )));
        };
        if key.is_empty() || value.is_empty() || !entities.insert(token.to_owned()) {
            return Err(AdapterError::InvalidSource(format!(
                "BIDS sidecar has malformed or duplicate entity {token:?}"
            )));
        }
    }
    Ok(entities)
}

fn parse_physio_sidecar(entry: &ForeignEntry) -> Result<PhysioSidecar, AdapterError> {
    let document: serde_json::Value = serde_json::from_slice(&entry.bytes).map_err(invalid)?;
    let columns = document
        .get("Columns")
        .map(|value| {
            let columns = value
                .as_array()
                .ok_or_else(|| {
                    AdapterError::InvalidSource(format!(
                        "{} physio Columns is not an array",
                        entry.path
                    ))
                })?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|column| !column.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or_else(|| {
                            AdapterError::InvalidSource(format!(
                                "{} has a non-string or empty physio column",
                                entry.path
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if columns.is_empty() || columns.len() > MAX_TSV_COLUMNS {
                return Err(if columns.len() > MAX_TSV_COLUMNS {
                    AdapterError::SourceTooLarge
                } else {
                    AdapterError::InvalidSource(format!(
                        "{} declares an empty physio Columns array",
                        entry.path
                    ))
                });
            }
            let unique: BTreeSet<&str> = columns.iter().map(String::as_str).collect();
            if unique.len() != columns.len() {
                return Err(AdapterError::InvalidSource(format!(
                    "{} declares duplicate physio columns",
                    entry.path
                )));
            }
            Ok(columns)
        })
        .transpose()?;
    let number_text = |field: &str| -> Result<Option<String>, AdapterError> {
        document
            .get(field)
            .map(|value| {
                value.as_number().map(ToString::to_string).ok_or_else(|| {
                    AdapterError::InvalidSource(format!(
                        "{} declares non-numeric {field}",
                        entry.path
                    ))
                })
            })
            .transpose()
    };
    let sampling_frequency = number_text("SamplingFrequency")?;
    if let Some(value) = &sampling_frequency {
        if !decimal_rational(value)?.is_positive() {
            return Err(AdapterError::InvalidSource(format!(
                "{} SamplingFrequency is not positive",
                entry.path
            )));
        }
    }
    let start_time = number_text("StartTime")?;
    if let Some(value) = &start_time {
        decimal_rational(value)?;
    }
    Ok(PhysioSidecar {
        path: entry.path.clone(),
        directory: path_directory(&entry.path).to_owned(),
        entities: filename_entities(&entry.path, "_physio.json")?,
        columns,
        sampling_frequency,
        start_time,
    })
}

fn physio_sidecar_for(
    path: &str,
    sidecars: &[PhysioSidecar],
) -> Result<PhysioMetadata, AdapterError> {
    let directory = path_directory(path);
    let entities = filename_entities(path, "_physio.tsv.gz")?;
    let mut matches = sidecars
        .iter()
        .filter(|sidecar| {
            (sidecar.directory.is_empty()
                || directory == sidecar.directory
                || directory
                    .strip_prefix(&sidecar.directory)
                    .is_some_and(|rest| rest.starts_with('/')))
                && sidecar.entities.is_subset(&entities)
        })
        .map(|sidecar| {
            (
                sidecar
                    .directory
                    .split('/')
                    .filter(|part| !part.is_empty())
                    .count(),
                sidecar,
            )
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.path.cmp(&right.1.path))
    });
    if matches.is_empty() {
        return Err(AdapterError::InvalidSource(format!(
            "{path} has no applicable _physio.json sidecar"
        )));
    }
    if matches.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(AdapterError::InvalidSource(format!(
            "{path} has multiple applicable physio sidecars at one inheritance level"
        )));
    }
    let mut columns = None;
    let mut sampling_frequency = None;
    let mut start_time = None;
    for (_, sidecar) in matches {
        if let Some(value) = &sidecar.columns {
            columns = Some(value.clone());
        }
        if let Some(value) = &sidecar.sampling_frequency {
            sampling_frequency = Some(value.clone());
        }
        if let Some(value) = &sidecar.start_time {
            start_time = Some(value.clone());
        }
    }
    Ok(PhysioMetadata {
        columns: columns.ok_or_else(|| {
            AdapterError::InvalidSource(format!(
                "{path} effective physio metadata declares no Columns"
            ))
        })?,
        sampling_frequency: sampling_frequency.ok_or_else(|| {
            AdapterError::InvalidSource(format!(
                "{path} effective physio metadata declares no SamplingFrequency"
            ))
        })?,
        start_time: start_time.ok_or_else(|| {
            AdapterError::InvalidSource(format!(
                "{path} effective physio metadata declares no StartTime"
            ))
        })?,
    })
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
    let header_line = lines
        .next()
        .ok_or_else(|| AdapterError::InvalidSource("BIDS TSV has no header".to_owned()))?;
    let mut header = Vec::new();
    for value in header_line.split('\t') {
        if header.len() >= MAX_TSV_COLUMNS {
            return Err(AdapterError::SourceTooLarge);
        }
        header.push(value.trim().to_owned());
    }
    let mut rows = Vec::new();
    let mut cells = header.len();
    for line in lines {
        let mut row = Vec::new();
        for value in line.split('\t') {
            if row.len() >= MAX_TSV_COLUMNS || cells >= MAX_TSV_CELLS {
                return Err(AdapterError::SourceTooLarge);
            }
            row.push(value.trim().to_owned());
            cells += 1;
        }
        rows.push(row);
    }
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

fn read_gzip_bounded(bytes: &[u8], limit: u64) -> Result<Vec<u8>, AdapterError> {
    let read_limit = limit.checked_add(1).ok_or(AdapterError::SourceTooLarge)?;
    let mut plain = Vec::new();
    let decoder = flate2::read::GzDecoder::new(bytes);
    let mut limited = std::io::Read::take(decoder, read_limit);
    std::io::Read::read_to_end(&mut limited, &mut plain).map_err(|error| {
        AdapterError::InvalidSource(format!("physio table is not gzip: {error}"))
    })?;
    if plain.len() as u64 > limit {
        return Err(AdapterError::SourceTooLarge);
    }
    Ok(plain)
}

fn parse_bids(
    entries: &[ForeignEntry],
    max_decoded_bytes: u64,
) -> Result<ParsedBids, AdapterError> {
    let mut recordings = Vec::new();
    let mut events = Vec::new();
    let mut electrodes = Vec::new();
    let mut coordinate_system = None;
    let mut derivatives = Vec::new();
    let mut subjects = BTreeSet::new();
    let mut dataset_name = String::new();
    let mut bids_version = String::new();
    let mut retained_signal_bytes = 0_u64;
    let physio_sidecars = entries
        .iter()
        .filter(|entry| {
            !entry
                .path
                .split('/')
                .any(|segment| segment == "derivatives")
                && entry.path.ends_with("_physio.json")
        })
        .map(parse_physio_sidecar)
        .collect::<Result<Vec<_>, _>>()?;

    let mut ordered_entries = entries.iter().collect::<Vec<_>>();
    ordered_entries.sort_by(|left, right| left.path.cmp(&right.path));
    for entry in ordered_entries {
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
                if electrodes.len() >= MAX_ELECTRODES {
                    return Err(AdapterError::SourceTooLarge);
                }
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
                if events.len() >= MAX_EVENTS {
                    return Err(AdapterError::SourceTooLarge);
                }
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
            let remaining = max_decoded_bytes
                .checked_sub(retained_signal_bytes)
                .ok_or(AdapterError::SourceTooLarge)?;
            let plain = read_gzip_bounded(&entry.bytes, remaining)?;
            let text = std::str::from_utf8(&plain).map_err(|_| {
                AdapterError::InvalidSource("BIDS physio TSV is not UTF-8".to_owned())
            })?;
            let metadata = physio_sidecar_for(path, &physio_sidecars)?;
            let lines = text.lines().filter(|line| !line.trim().is_empty());
            let signal_budget = remaining
                .checked_sub(plain.len() as u64)
                .ok_or(AdapterError::SourceTooLarge)?;
            let max_values = signal_budget / core::mem::size_of::<i64>() as u64;
            if metadata.columns.len() as u64 > max_values {
                return Err(AdapterError::SourceTooLarge);
            }
            let mut signal = vec![Vec::new(); metadata.columns.len()];
            let mut values = 0_u64;
            for line in lines {
                let mut width = 0_usize;
                for (index, value) in line.split('\t').enumerate() {
                    if index >= metadata.columns.len() {
                        return Err(AdapterError::InvalidSource(
                            "a physio row does not match sidecar Columns".to_owned(),
                        ));
                    }
                    values = values.checked_add(1).ok_or(AdapterError::SourceTooLarge)?;
                    if values > max_values {
                        return Err(AdapterError::SourceTooLarge);
                    }
                    let parsed = value.parse::<f64>().map_err(|_| {
                        AdapterError::InvalidSource("a physio sample is not a number".to_owned())
                    })?;
                    if !parsed.is_finite() {
                        return Err(AdapterError::InvalidSource(
                            "a physio sample is not finite".to_owned(),
                        ));
                    }
                    signal[index].push(parsed);
                    width += 1;
                }
                if width != metadata.columns.len() {
                    return Err(AdapterError::InvalidSource(
                        "a physio row does not match sidecar Columns".to_owned(),
                    ));
                }
            }
            if signal.iter().any(Vec::is_empty) {
                return Err(AdapterError::InvalidSource(
                    "a physio recording carries no samples".to_owned(),
                ));
            }
            retained_signal_bytes = retained_signal_bytes
                .checked_add(
                    values
                        .checked_mul(core::mem::size_of::<f64>() as u64)
                        .ok_or(AdapterError::SourceTooLarge)?,
                )
                .ok_or(AdapterError::SourceTooLarge)?;
            recordings.push(Recorded {
                path: path.to_owned(),
                datatype: Datatype::Physio,
                subject: subject_of(path),
                session: session_of(path),
                signal: RecordedSignal::Real(signal),
                rate: metadata.sampling_frequency.clone(),
                start: metadata.start_time.clone(),
                channels: metadata.columns.clone(),
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
            let decoded_bytes = bundle
                .signal
                .iter()
                .try_fold(0_u64, |total, channel| {
                    total.checked_add(
                        (channel.len() as u64).checked_mul(core::mem::size_of::<i64>() as u64)?,
                    )
                })
                .ok_or(AdapterError::SourceTooLarge)?;
            retained_signal_bytes = retained_signal_bytes
                .checked_add(decoded_bytes)
                .ok_or(AdapterError::SourceTooLarge)?;
            if retained_signal_bytes > max_decoded_bytes {
                return Err(AdapterError::SourceTooLarge);
            }
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
                start: "0".to_owned(),
                channels: bundle.channels.clone(),
                signal: RecordedSignal::Integer(bundle.signal),
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
        let decoded_limit = self
            .max_source_bytes
            .saturating_mul(2)
            .min(limits.max_logical_payload_bytes);
        let parsed = parse_bids(entries, decoded_limit)?;
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
            let (channel_count, samples) = match &recorded.signal {
                RecordedSignal::Integer(signal) => {
                    (signal.len(), signal.first().map_or(0, Vec::len) as u64)
                }
                RecordedSignal::Real(signal) => {
                    (signal.len(), signal.first().map_or(0, Vec::len) as u64)
                }
            };
            if samples == 0 {
                return Err(AdapterError::InvalidSource(format!(
                    "recording {} carries no samples",
                    recorded.path
                )));
            }
            let mut bytes = Vec::with_capacity(channel_count * samples as usize * 8);
            let element = match &recorded.signal {
                RecordedSignal::Integer(signal) => {
                    for channel in signal {
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
                    ElementType::I64
                }
                RecordedSignal::Real(signal) => {
                    for channel in signal {
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
                    ElementType::F64
                }
            };
            let content_id = abir_payload_id(element, &bytes);
            let descriptor = PayloadDescriptor::new(
                content_id,
                bytes.len() as u64,
                element,
                ByteOrder::Little,
                vec![channel_count as u64, samples],
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
                        decimal_rational(&recorded.start)?,
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
        if !parsed.bids_version.is_empty() {
            mappings.push(exact(
                "dataset_description.json#BIDSVersion".to_owned(),
                format!("recording:{recording_id}"),
            ));
        }
        if !parsed.dataset_name.is_empty() {
            mappings.push(exact(
                "dataset_description.json#Name".to_owned(),
                format!("recording:{recording_id}"),
            ));
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
        let parsed = parse_bids(entries, self.max_source_bytes.saturating_mul(2))?;
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
