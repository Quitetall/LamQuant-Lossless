// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class NWB 2.10.0 adapter (ADR 0143).
//!
//! An NWB file is a typed HDF5 tree, and where a series LIVES is part of what
//! it means: `/acquisition` is recorded data, `/stimulus` is what was
//! presented, `/processing/behavior` is derived behavioural signal, and
//! anything else under `/processing` or `/scratch` is derived data. Flattening
//! them into one bag of arrays would throw that away, so each container becomes
//! its own ABIR `Stream` carrying its own modality concept.
//!
//! Beyond the series themselves this adapter promotes:
//!
//! * the extracellular-ephys **electrodes** table to a `ChannelBasis` plus an
//!   exact table atom, so channel identity is semantic rather than positional;
//! * every `/intervals` table to `Event`s on the session clock, because an
//!   epoch is a claim about time, not a spreadsheet;
//! * **external assets** -- HDF5 external links -- as named source keys and a
//!   quarantined mapping entry. Their bytes live in another file this adapter
//!   was not handed, so inlining them would fabricate content.
//!
//! A series with `starting_time` and a rate gets a regular time axis; one with
//! a `timestamps` dataset gets an explicit axis over those timestamps. Nothing
//! is given a rate it did not declare.

use abir_adapter::{
    Adapter, AdapterCapability, AdapterError, AdapterProfile, ExportPlan, FidelityReceipt,
    ForeignEntry, ForeignObject, ImportOutcome, InspectReport, MappingDisposition, MappingEntry,
    MappingReport, PayloadObject, PayloadResolver, ProfileId, ProfileStatus, SemanticCoverage,
    ValidationArtifact,
};
use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
use semantic_abir::{
    interchange_content_id, payload_content_id as abir_payload_id, AbirDataset, Atom, AtomTag,
    ByteOrder, ChannelBasis, ChannelBasisTag, ChannelSpec, Clock, ClockTag, ConceptId,
    DatasetDraft, DatasetTag, ElementType, Event, EventTag, Layout, ObjectId, PayloadDescriptor,
    Presence, Rational, Recording, RecordingTag, ReferenceKind, SignalBlock, SourceCapsule,
    SourceKey, Stream, StreamTag, TimeAxis, TimeSegment, ValidationLimits,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use crate::{binding_namespace, payload_content_id, plan_id, valid_relative_path};

const PROFILE: &str = "nwb.2.10.0";
/// Ceiling on how many series one file may declare before this adapter refuses.
/// A pathological tree would otherwise turn one import into unbounded work.
const MAX_SERIES: usize = 4096;

pub struct NwbAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
}

/// Where a series sits in the NWB tree, which is part of what it means.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Container {
    Acquisition,
    Stimulus,
    Behavior,
    Derived,
    Scratch,
}

impl Container {
    const fn key(self) -> &'static str {
        match self {
            Self::Acquisition => "acquisition",
            Self::Stimulus => "stimulus",
            Self::Behavior => "behavior",
            Self::Derived => "derived-data",
            Self::Scratch => "scratch",
        }
    }

    const fn modality(self) -> &'static str {
        match self {
            Self::Acquisition => "abir:modality/unknown",
            Self::Stimulus => "nwb:modality/stimulus",
            Self::Behavior => "nwb:modality/behavior",
            Self::Derived => "nwb:modality/derived",
            Self::Scratch => "nwb:modality/scratch",
        }
    }
}

/// One NWB TimeSeries, normalised.
struct Series {
    container: Container,
    path: String,
    neurodata_type: String,
    /// Row-major flattened samples, already widened to i64 or f64.
    values: SeriesValues,
    rows: u64,
    columns: u64,
    /// `Some((start, rate))` for a regular series; `None` when the file carries
    /// explicit timestamps instead.
    regular: Option<(f64, f64)>,
    timestamps: Vec<f64>,
    unit: String,
}

enum SeriesValues {
    Integer(Vec<i64>),
    Real(Vec<f64>),
}

impl SeriesValues {
    fn len(&self) -> usize {
        match self {
            Self::Integer(values) => values.len(),
            Self::Real(values) => values.len(),
        }
    }

    const fn element(&self) -> ElementType {
        match self {
            Self::Integer(_) => ElementType::I64,
            Self::Real(_) => ElementType::F64,
        }
    }

    /// Encode as [columns, rows] row-major.
    ///
    /// NWB writes `data[time][channel]`; ABIR declares the payload shape as
    /// `[channels, samples]`, so the values are genuinely transposed rather
    /// than relabelled -- declaring one layout while shipping another is how a
    /// channel mix-up becomes invisible.
    fn encode(&self, rows: usize, columns: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * columns * 8);
        for column in 0..columns {
            for row in 0..rows {
                let index = row * columns + column;
                match self {
                    Self::Integer(values) => bytes.extend_from_slice(&values[index].to_le_bytes()),
                    Self::Real(values) => bytes.extend_from_slice(&values[index].to_le_bytes()),
                }
            }
        }
        bytes
    }
}

/// One `/intervals` row: an epoch with a start and a stop on the session clock.
struct Interval {
    table: String,
    start: f64,
    stop: f64,
}

struct ParsedNwb {
    series: Vec<Series>,
    electrodes: Vec<String>,
    electrode_columns: Vec<String>,
    intervals: Vec<Interval>,
    external_assets: Vec<String>,
    session_description: String,
    identifier: String,
    nwb_version: String,
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

/// Seconds as an exact rational on a microsecond grid. NWB stores times as
/// f64 seconds; pinning them to a fixed grid keeps the promoted event
/// boundaries reproducible instead of platform-dependent.
fn seconds(value: f64) -> Result<Rational, AdapterError> {
    if !value.is_finite() {
        return Err(AdapterError::InvalidSource(
            "NWB time value is not finite".to_owned(),
        ));
    }
    let micros = (value * 1_000_000.0).round();
    if micros.abs() > 9.0e15 {
        return Err(AdapterError::InvalidSource(
            "NWB time value is out of range".to_owned(),
        ));
    }
    Rational::new(micros as i128, 1_000_000).map_err(invalid)
}

fn microsecond_ticks(value: f64) -> Result<i64, AdapterError> {
    if !value.is_finite() {
        return Err(AdapterError::InvalidSource(
            "NWB timestamp is not finite".to_owned(),
        ));
    }
    let micros = (value * 1_000_000.0).round();
    if micros.abs() > 9.0e15 {
        return Err(AdapterError::InvalidSource(
            "NWB timestamp is out of range".to_owned(),
        ));
    }
    Ok(micros as i64)
}

fn read_text_attr(location: &hdf5_metno::Group, name: &str) -> Option<String> {
    location
        .attr(name)
        .ok()
        .and_then(|attribute| {
            attribute
                .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                .ok()
        })
        .map(|value| value.as_str().to_owned())
}

fn read_string_dataset(dataset: &hdf5_metno::Dataset) -> Result<Vec<String>, AdapterError> {
    let values = dataset
        .read_raw::<hdf5_metno::types::VarLenUnicode>()
        .map_err(invalid)?;
    Ok(values
        .into_iter()
        .map(|value| value.as_str().to_owned())
        .collect())
}

fn read_numeric(dataset: &hdf5_metno::Dataset) -> Result<SeriesValues, AdapterError> {
    let descriptor = dataset.dtype().and_then(|dtype| dtype.to_descriptor());
    Ok(match descriptor.map_err(invalid)? {
        TypeDescriptor::Integer(IntSize::U1) => {
            SeriesValues::Integer(widen_int::<i8>(dataset, i64::from)?)
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            SeriesValues::Integer(widen_int::<i16>(dataset, i64::from)?)
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            SeriesValues::Integer(widen_int::<i32>(dataset, i64::from)?)
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            SeriesValues::Integer(widen_int::<i64>(dataset, |value| value)?)
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            SeriesValues::Integer(widen_int::<u8>(dataset, i64::from)?)
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            SeriesValues::Integer(widen_int::<u16>(dataset, i64::from)?)
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            SeriesValues::Integer(widen_int::<u32>(dataset, i64::from)?)
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            let raw = dataset.read_raw::<u64>().map_err(invalid)?;
            let mut values = Vec::with_capacity(raw.len());
            for value in raw {
                values.push(i64::try_from(value).map_err(|_| {
                    AdapterError::UnsupportedMeaning(
                        "NWB u64 sample exceeds the exact ABIR integer range".to_owned(),
                    )
                })?);
            }
            SeriesValues::Integer(values)
        }
        TypeDescriptor::Float(FloatSize::U4) => {
            SeriesValues::Real(widen_real::<f32>(dataset, f64::from)?)
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            SeriesValues::Real(widen_real::<f64>(dataset, |value| value)?)
        }
        other => {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "NWB dataset element type {other:?} has no exact ABIR promotion"
            )))
        }
    })
}

fn widen_int<T>(
    dataset: &hdf5_metno::Dataset,
    convert: fn(T) -> i64,
) -> Result<Vec<i64>, AdapterError>
where
    T: hdf5_metno::H5Type,
{
    Ok(dataset
        .read_raw::<T>()
        .map_err(invalid)?
        .into_iter()
        .map(convert)
        .collect())
}

fn widen_real<T>(
    dataset: &hdf5_metno::Dataset,
    convert: fn(T) -> f64,
) -> Result<Vec<f64>, AdapterError>
where
    T: hdf5_metno::H5Type,
{
    Ok(dataset
        .read_raw::<T>()
        .map_err(invalid)?
        .into_iter()
        .map(convert)
        .collect())
}

/// Read one TimeSeries group. Returns `None` when the group is not a series
/// (no `data` dataset), which is how the walk skips containers and tables.
fn read_series(
    group: &hdf5_metno::Group,
    container: Container,
    path: &str,
) -> Result<Option<Series>, AdapterError> {
    // An external asset is recognised BEFORE its data placeholder. NWB writes
    // an empty `data` dataset beside `external_file`, and treating that empty
    // array as a malformed series would refuse a perfectly valid file.
    if group.link_exists("external_file") {
        return Ok(None);
    }
    if !group.link_exists("data") {
        return Ok(None);
    }
    let data = group.dataset("data").map_err(invalid)?;
    let shape = data.shape();
    if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
        return Err(AdapterError::UnsupportedMeaning(format!(
            "NWB series {path} is not a nonempty rank-1 or rank-2 array"
        )));
    }
    let rows = shape[0] as u64;
    let columns = if shape.len() == 2 { shape[1] as u64 } else { 1 };
    let values = read_numeric(&data)?;
    if values.len() as u64 != rows.saturating_mul(columns) {
        return Err(AdapterError::InvalidSource(format!(
            "NWB series {path} read a different element count than its shape declares"
        )));
    }

    let mut regular = None;
    let mut timestamps = Vec::new();
    if group.link_exists("starting_time") {
        let starting = group.dataset("starting_time").map_err(invalid)?;
        let start = starting.read_scalar::<f64>().map_err(invalid)?;
        let rate = starting
            .attr("rate")
            .and_then(|attribute| attribute.read_scalar::<f64>())
            .map_err(invalid)?;
        if !start.is_finite() || !rate.is_finite() || rate <= 0.0 {
            return Err(AdapterError::InvalidSource(format!(
                "NWB series {path} declares a non-finite or non-positive rate"
            )));
        }
        regular = Some((start, rate));
    } else if group.link_exists("timestamps") {
        let stamps = group.dataset("timestamps").map_err(invalid)?;
        timestamps = match read_numeric(&stamps)? {
            SeriesValues::Real(values) => values,
            SeriesValues::Integer(values) => values.into_iter().map(|value| value as f64).collect(),
        };
        if timestamps.len() as u64 != rows {
            return Err(AdapterError::InvalidSource(format!(
                "NWB series {path} has one timestamp per row or none, not {}",
                timestamps.len()
            )));
        }
    } else {
        return Err(AdapterError::UnsupportedMeaning(format!(
            "NWB series {path} carries neither starting_time nor timestamps"
        )));
    }

    Ok(Some(Series {
        container,
        path: path.to_owned(),
        neurodata_type: read_text_attr(group, "neurodata_type")
            .unwrap_or_else(|| "TimeSeries".to_owned()),
        values,
        rows,
        columns,
        regular,
        timestamps,
        unit: data
            .attr("unit")
            .ok()
            .and_then(|attribute| {
                attribute
                    .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                    .ok()
            })
            .map(|value| value.as_str().to_owned())
            .unwrap_or_default(),
    }))
}

/// Walk one container group, collecting every series beneath it.
fn walk_container(
    parent: &hdf5_metno::Group,
    container: Container,
    prefix: &str,
    series: &mut Vec<Series>,
    external: &mut Vec<String>,
) -> Result<(), AdapterError> {
    for name in parent.member_names().map_err(invalid)? {
        let path = format!("{prefix}/{name}");
        let Ok(group) = parent.group(&name) else {
            continue;
        };
        // An external link resolves into another file this adapter was never
        // handed. Name it and move on; inlining it would fabricate content.
        if read_text_attr(&group, "neurodata_type").is_none() && !group.link_exists("data") {
            walk_container(&group, container, &path, series, external)?;
            continue;
        }
        match read_series(&group, container, &path)? {
            Some(found) => {
                if series.len() >= MAX_SERIES {
                    return Err(AdapterError::UnsupportedMeaning(
                        "NWB file declares more series than this adapter will import".to_owned(),
                    ));
                }
                series.push(found);
            }
            None => {
                if group.link_exists("external_file") {
                    external.push(path.clone());
                } else {
                    walk_container(&group, container, &path, series, external)?;
                }
            }
        }
    }
    Ok(())
}

fn parse_nwb(path: &std::path::Path) -> Result<ParsedNwb, AdapterError> {
    let file = hdf5_metno::File::open(path).map_err(invalid)?;
    let root = file.as_group().map_err(invalid)?;
    let nwb_version = root
        .attr("nwb_version")
        .and_then(|attribute| attribute.read_scalar::<hdf5_metno::types::VarLenUnicode>())
        .map(|value| value.as_str().to_owned())
        .map_err(|_| {
            AdapterError::InvalidSource("file declares no nwb_version attribute".to_owned())
        })?;
    if !nwb_version.starts_with("2.") {
        return Err(AdapterError::UnsupportedMeaning(format!(
            "this profile covers NWB 2.x; the file declares {nwb_version}"
        )));
    }

    let mut series = Vec::new();
    let mut external_assets = Vec::new();
    for (name, container) in [
        ("acquisition", Container::Acquisition),
        ("stimulus", Container::Stimulus),
        ("scratch", Container::Scratch),
    ] {
        if let Ok(group) = root.group(name) {
            walk_container(
                &group,
                container,
                &format!("/{name}"),
                &mut series,
                &mut external_assets,
            )?;
        }
    }
    if let Ok(processing) = root.group("processing") {
        for module in processing.member_names().map_err(invalid)? {
            let Ok(group) = processing.group(&module) else {
                continue;
            };
            let container = if module.eq_ignore_ascii_case("behavior") {
                Container::Behavior
            } else {
                Container::Derived
            };
            walk_container(
                &group,
                container,
                &format!("/processing/{module}"),
                &mut series,
                &mut external_assets,
            )?;
        }
    }
    if series.is_empty() {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB file carries no importable TimeSeries".to_owned(),
        ));
    }

    // The electrodes table: channel identity, not a spreadsheet.
    let mut electrodes = Vec::new();
    let mut electrode_columns = Vec::new();
    if let Ok(table) = root.group("general/extracellular_ephys/electrodes") {
        electrode_columns = table.member_names().map_err(invalid)?;
        let label_column = ["location", "group_name", "label"]
            .into_iter()
            .find(|name| table.link_exists(name));
        if let Some(column) = label_column {
            let dataset = table.dataset(column).map_err(invalid)?;
            electrodes = read_string_dataset(&dataset).unwrap_or_default();
        }
        if electrodes.is_empty() {
            if let Ok(dataset) = table.dataset("id") {
                electrodes = match read_numeric(&dataset)? {
                    SeriesValues::Integer(values) => {
                        values.into_iter().map(|value| value.to_string()).collect()
                    }
                    SeriesValues::Real(values) => {
                        values.into_iter().map(|value| value.to_string()).collect()
                    }
                };
            }
        }
    }

    // Interval tables: every row is a claim about time.
    let mut intervals = Vec::new();
    if let Ok(group) = root.group("intervals") {
        for name in group.member_names().map_err(invalid)? {
            let Ok(table) = group.group(&name) else {
                continue;
            };
            if !table.link_exists("start_time") || !table.link_exists("stop_time") {
                continue;
            }
            let starts = match read_numeric(&table.dataset("start_time").map_err(invalid)?)? {
                SeriesValues::Real(values) => values,
                SeriesValues::Integer(values) => {
                    values.into_iter().map(|value| value as f64).collect()
                }
            };
            let stops = match read_numeric(&table.dataset("stop_time").map_err(invalid)?)? {
                SeriesValues::Real(values) => values,
                SeriesValues::Integer(values) => {
                    values.into_iter().map(|value| value as f64).collect()
                }
            };
            if starts.len() != stops.len() {
                return Err(AdapterError::InvalidSource(format!(
                    "NWB interval table {name} has mismatched start and stop columns"
                )));
            }
            for (start, stop) in starts.into_iter().zip(stops) {
                intervals.push(Interval {
                    table: name.clone(),
                    start,
                    stop,
                });
            }
        }
    }

    Ok(ParsedNwb {
        series,
        electrodes,
        electrode_columns,
        intervals,
        external_assets,
        session_description: root
            .dataset("session_description")
            .ok()
            .and_then(|dataset| {
                dataset
                    .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                    .ok()
            })
            .map(|value| value.as_str().to_owned())
            .unwrap_or_default(),
        identifier: root
            .dataset("identifier")
            .ok()
            .and_then(|dataset| {
                dataset
                    .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                    .ok()
            })
            .map(|value| value.as_str().to_owned())
            .unwrap_or_default(),
        nwb_version,
    })
}

struct ParsedDataset {
    dataset: AbirDataset,
    payloads: Vec<PayloadObject>,
    mappings: Vec<MappingEntry>,
    series: u64,
    electrodes: u64,
    intervals: u64,
    external: u64,
}

impl NwbAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self {
            profile: AdapterProfile {
                id: ProfileId(PROFILE.to_owned()),
                standard: "NWB".to_owned(),
                edition: "2.10.0".to_owned(),
                media_types: vec!["application/x-nwb".to_owned()],
                status: ProfileStatus::Semantic,
                required_validator: "pynwb.validate".to_owned(),
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
                "NWB semantic profile requires exactly one file".to_owned(),
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

    fn read(&self, entry: &ForeignEntry) -> Result<ParsedNwb, AdapterError> {
        let temporary = tempfile::tempdir().map_err(invalid)?;
        let path = temporary.path().join("source.nwb");
        fs::write(&path, &entry.bytes).map_err(invalid)?;
        parse_nwb(&path)
    }

    fn parse(
        &self,
        entry: &ForeignEntry,
        limits: ValidationLimits,
    ) -> Result<ParsedDataset, AdapterError> {
        let parsed = self.read(entry)?;
        let seed = blake3::hash(&entry.bytes);
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let clock_id = id::<ClockTag>(&seed, b"session-clock", 0);
        let basis_id = id::<ChannelBasisTag>(&seed, b"electrode-basis", 0);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();

        // Every NWB time is stated against the session start time, so one
        // session clock is what the file actually claims.
        draft.add_clock(Clock::new(
            clock_id,
            concept("nwb:clock/session-start")?,
            None,
            Rational::new(0, 1).expect("zero is a rational"),
            Rational::new(1, 1).expect("unit rate is a rational"),
            Rational::new(0, 1).expect("zero is a rational"),
        ));

        let basis = if parsed.electrodes.is_empty() {
            None
        } else {
            let specs = parsed
                .electrodes
                .iter()
                .enumerate()
                .map(|(index, label)| {
                    let concept_id = concept(&format!("nwb:electrode/{index}"))?;
                    Ok(ChannelSpec::new(concept_id)
                        .with_source_key(source_key("nwb.electrode-label", label)?))
                })
                .collect::<Result<Vec<_>, AdapterError>>()?;
            draft.add_channel_basis(ChannelBasis::new(basis_id, specs, ReferenceKind::Unknown));
            mappings.push(exact(
                "/general/extracellular_ephys/electrodes".to_owned(),
                format!("channel-basis:{basis_id}"),
            ));
            Some(basis_id)
        };

        // One stream per container, so where a series lived stays meaningful.
        let mut by_container: BTreeMap<Container, Vec<ObjectId<AtomTag>>> = BTreeMap::new();
        let mut companions: Vec<(Container, ObjectId<AtomTag>)> = Vec::new();
        for (index, series) in parsed.series.iter().enumerate() {
            let position = index as u64;
            let atom_id = id::<AtomTag>(&seed, b"series", position);
            let bytes = series.values.encode(
                usize::try_from(series.rows).map_err(|_| AdapterError::SourceTooLarge)?,
                usize::try_from(series.columns).map_err(|_| AdapterError::SourceTooLarge)?,
            );
            let content_id = abir_payload_id(series.values.element(), &bytes);
            let descriptor = PayloadDescriptor::new(
                content_id,
                u64::try_from(bytes.len()).map_err(|_| AdapterError::SourceTooLarge)?,
                series.values.element(),
                ByteOrder::Little,
                vec![series.columns, series.rows],
                Layout::DenseRowMajor,
                Some(concept("abir:encoding/raw")?),
                None,
            );
            let time_axis = match series.regular {
                Some((start, rate)) => TimeAxis::Regular(
                    TimeSegment::new(seconds(start)?, seconds(rate)?, series.rows)
                        .map_err(invalid)?,
                ),
                None => {
                    // ABIR carries explicit timestamps as exact integer ticks,
                    // and the axis names a companion payload that must belong
                    // to a real atom -- a dangling reference is exactly what
                    // validation refuses.
                    let mut stamps = Vec::with_capacity(series.timestamps.len() * 8);
                    for value in &series.timestamps {
                        stamps.extend_from_slice(&microsecond_ticks(*value)?.to_le_bytes());
                    }
                    let stamp_id = abir_payload_id(ElementType::I64, &stamps);
                    let stamp_atom = id::<AtomTag>(&seed, b"timestamps", position);
                    draft.add_atom(Atom::Tensor(semantic_abir::Tensor::new(
                        stamp_atom,
                        Presence::Present,
                        Some(PayloadDescriptor::new(
                            stamp_id,
                            stamps.len() as u64,
                            ElementType::I64,
                            ByteOrder::Little,
                            vec![series.rows],
                            Layout::DenseRowMajor,
                            Some(concept("abir:encoding/raw")?),
                            None,
                        )),
                        vec![semantic_abir::SemanticAxis::new(
                            concept("abir:axis/sample")?,
                            series.rows,
                        )],
                    )));
                    payloads.push(PayloadObject {
                        content_id: stamp_id,
                        bytes: stamps,
                    });
                    companions.push((series.container, stamp_atom));
                    TimeAxis::Explicit {
                        timestamps: stamp_id,
                        count: series.rows,
                    }
                }
            };
            // The payload is [columns, rows]; a signal block's time axis is its
            // last dimension, which is exactly the row count.
            draft.add_atom(Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                time_axis,
                None,
            )));
            payloads.push(PayloadObject { content_id, bytes });
            by_container
                .entry(series.container)
                .or_default()
                .push(atom_id);
            mappings.push(exact(
                format!("{}/data", series.path),
                format!("atom:{atom_id}"),
            ));
        }

        for (container, atom) in companions {
            by_container.entry(container).or_default().push(atom);
        }
        let mut stream_ids = Vec::new();
        for (index, (container, atoms)) in by_container.iter().enumerate() {
            let stream_id = id::<StreamTag>(&seed, b"stream", index as u64);
            draft.add_stream(Stream::new(
                stream_id,
                recording_id,
                concept(container.modality())?,
                atoms.clone(),
                Some(clock_id),
                // Only recorded acquisition is indexed by the electrode basis;
                // a stimulus or behaviour series is not an electrode signal.
                if *container == Container::Acquisition {
                    basis
                } else {
                    None
                },
                None,
            ));
            stream_ids.push(stream_id);
        }

        for (index, interval) in parsed.intervals.iter().enumerate() {
            let event_id = id::<EventTag>(&seed, b"interval", index as u64);
            draft.add_event(Event::new(
                event_id,
                concept(&format!("nwb:interval/{}", interval.table))?,
                clock_id,
                seconds(interval.start)?,
                seconds(interval.stop)?,
                Rational::new(0, 1).expect("zero is a rational"),
            ));
        }
        if !parsed.intervals.is_empty() {
            mappings.push(exact(
                "/intervals".to_owned(),
                "events:nwb:interval".to_owned(),
            ));
        }

        // External assets: named, never inlined.
        for path in &parsed.external_assets {
            mappings.push(MappingEntry {
                source_path: path.clone(),
                target: "abir.source-capsule".to_owned(),
                disposition: MappingDisposition::Quarantined,
                reason: Some(
                    "the series references an external file that was not part of this source object"
                        .to_owned(),
                ),
            });
        }

        let mut recording = Recording::new(recording_id, stream_ids);
        for (namespace, value) in [
            ("nwb.version", parsed.nwb_version.as_str()),
            ("nwb.identifier", parsed.identifier.as_str()),
            (
                "nwb.session-description",
                parsed.session_description.as_str(),
            ),
        ] {
            if !value.is_empty() {
                recording.add_source_key(source_key(namespace, value)?);
            }
        }
        if !parsed.electrode_columns.is_empty() {
            recording.add_source_key(source_key(
                "nwb.electrode-columns",
                &parsed.electrode_columns.join("|"),
            )?);
        }
        for path in &parsed.external_assets {
            recording.add_source_key(source_key("nwb.external-asset", path)?);
        }
        for series in &parsed.series {
            recording.add_source_key(source_key(
                &format!("nwb.series.{}", series.path),
                &format!(
                    "container={};type={};unit={};rows={};columns={}",
                    series.container.key(),
                    series.neurodata_type,
                    series.unit,
                    series.rows,
                    series.columns
                ),
            )?);
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
            series: parsed.series.len() as u64,
            electrodes: parsed.electrodes.len() as u64,
            intervals: parsed.intervals.len() as u64,
            external: parsed.external_assets.len() as u64,
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

impl Adapter for NwbAdapter {
    fn profile(&self) -> &AdapterProfile {
        &self.profile
    }

    fn inspect(&self, source: &ForeignObject) -> Result<InspectReport, AdapterError> {
        let entry = self.entry(source)?;
        let parsed = self.read(entry)?;
        Ok(InspectReport {
            profile: self.profile.id.clone(),
            entry_count: 1,
            logical_bytes: entry.bytes.len() as u64,
            risks: Vec::new(),
            required_resources: BTreeMap::from([
                ("max-source-bytes".to_owned(), self.max_source_bytes),
                ("series".to_owned(), parsed.series.len() as u64),
                ("electrodes".to_owned(), parsed.electrodes.len() as u64),
                ("intervals".to_owned(), parsed.intervals.len() as u64),
                (
                    "external-assets".to_owned(),
                    parsed.external_assets.len() as u64,
                ),
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
                preserved_unknowns: parsed.external.saturating_add(1),
                sample_values_changed: false,
                timing_changed: false,
            },
            payloads: parsed.payloads,
        })
    }

    fn plan_export(&self, dataset: &AbirDataset) -> Result<ExportPlan, AdapterError> {
        let capsules = self.capsules(dataset)?;
        let unsupported = capsules.len() != 1;
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
                "dataset lacks one exact NWB source capsule".to_owned(),
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
        let diagnostics = match &result {
            Ok(parsed) => vec![format!(
                "series={} electrodes={} intervals={} external-assets={}",
                parsed.series, parsed.electrodes, parsed.intervals, parsed.external
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
