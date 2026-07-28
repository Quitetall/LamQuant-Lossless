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
//! * **external assets** named by NWB `external_file` datasets as source keys
//!   and quarantined mappings. Their bytes live outside the source object, so
//!   inlining them would fabricate content. HDF5 external links are rejected
//!   before resolution because their meaning depends on host filesystem state.
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
use std::sync::Arc;

use crate::{binding_namespace, payload_content_id, plan_id, valid_relative_path};

const PROFILE: &str = "nwb.2.10.0";
/// Ceiling on how many series one file may declare before this adapter refuses.
/// A pathological tree would otherwise turn one import into unbounded work.
const MAX_SERIES: usize = 4096;
const MAX_GROUPS: usize = 16_384;
const MAX_GROUP_DEPTH: usize = 64;
const MAX_INTERVALS: usize = 262_144;
const MAX_ELECTRODES: usize = 65_536;
const MAX_EXTERNAL_ASSETS: usize = 4096;
const MAX_EXTERNAL_ASSET_NAME_BYTES: usize = 4096;

pub struct NwbAdapter {
    profile: AdapterProfile,
    max_source_bytes: u64,
    max_decoded_bytes: u64,
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
    table: Arc<str>,
    start: f64,
    stop: f64,
}

struct ExternalAsset {
    series_path: String,
    file_index: usize,
    file_name: String,
    starting_frame: Option<i64>,
}

struct ParsedNwb {
    series: Vec<Series>,
    electrodes: Vec<String>,
    electrode_columns: Vec<String>,
    intervals: Vec<Interval>,
    external_assets: Vec<ExternalAsset>,
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
fn seconds(value: f64) -> Result<(Rational, bool), AdapterError> {
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
    let rational = Rational::new(micros as i128, 1_000_000).map_err(invalid)?;
    Ok((rational, micros / 1_000_000.0 != value))
}

fn microsecond_ticks(value: f64) -> Result<(i64, bool), AdapterError> {
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
    let ticks = micros as i64;
    Ok((ticks, ticks as f64 / 1_000_000.0 != value))
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

struct DecodeBudget {
    limit: u64,
    retained: u64,
}

impl DecodeBudget {
    const fn new(limit: u64) -> Self {
        Self { limit, retained: 0 }
    }

    fn reserve_retained(&mut self, bytes: u64) -> Result<(), AdapterError> {
        self.retained = self
            .retained
            .checked_add(bytes)
            .ok_or(AdapterError::SourceTooLarge)?;
        if self.retained > self.limit {
            return Err(AdapterError::SourceTooLarge);
        }
        Ok(())
    }

    fn reserve_numeric<T>(&mut self, dataset: &hdf5_metno::Dataset) -> Result<(), AdapterError> {
        let elements = u64::try_from(dataset.size()).map_err(|_| AdapterError::SourceTooLarge)?;
        let source_bytes = elements
            .checked_mul(core::mem::size_of::<T>() as u64)
            .ok_or(AdapterError::SourceTooLarge)?;
        let widened_bytes = elements
            .checked_mul(core::mem::size_of::<i64>() as u64)
            .ok_or(AdapterError::SourceTooLarge)?;
        let peak = self
            .retained
            .checked_add(source_bytes)
            .and_then(|bytes| bytes.checked_add(widened_bytes))
            .ok_or(AdapterError::SourceTooLarge)?;
        if peak > self.limit {
            return Err(AdapterError::SourceTooLarge);
        }
        self.retained = self
            .retained
            .checked_add(widened_bytes)
            .ok_or(AdapterError::SourceTooLarge)?;
        Ok(())
    }
}

fn read_numeric(
    dataset: &hdf5_metno::Dataset,
    budget: &mut DecodeBudget,
) -> Result<SeriesValues, AdapterError> {
    let descriptor = dataset.dtype().and_then(|dtype| dtype.to_descriptor());
    Ok(match descriptor.map_err(invalid)? {
        TypeDescriptor::Integer(IntSize::U1) => {
            SeriesValues::Integer(widen_int::<i8>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Integer(IntSize::U2) => {
            SeriesValues::Integer(widen_int::<i16>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            SeriesValues::Integer(widen_int::<i32>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            SeriesValues::Integer(widen_int::<i64>(dataset, |value| value, budget)?)
        }
        TypeDescriptor::Unsigned(IntSize::U1) => {
            SeriesValues::Integer(widen_int::<u8>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Unsigned(IntSize::U2) => {
            SeriesValues::Integer(widen_int::<u16>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            SeriesValues::Integer(widen_int::<u32>(dataset, i64::from, budget)?)
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            budget.reserve_numeric::<u64>(dataset)?;
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
            SeriesValues::Real(widen_real::<f32>(dataset, f64::from, budget)?)
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            SeriesValues::Real(widen_real::<f64>(dataset, |value| value, budget)?)
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
    budget: &mut DecodeBudget,
) -> Result<Vec<i64>, AdapterError>
where
    T: hdf5_metno::H5Type,
{
    budget.reserve_numeric::<T>(dataset)?;
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
    budget: &mut DecodeBudget,
) -> Result<Vec<f64>, AdapterError>
where
    T: hdf5_metno::H5Type,
{
    budget.reserve_numeric::<T>(dataset)?;
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
    budget: &mut DecodeBudget,
) -> Result<Option<Series>, AdapterError> {
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
    let values = read_numeric(&data, budget)?;
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
        timestamps = match read_numeric(&stamps, budget)? {
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
struct WalkState<'a> {
    series: &'a mut Vec<Series>,
    external: &'a mut Vec<ExternalAsset>,
    budget: &'a mut DecodeBudget,
    visited: &'a mut Vec<(u64, hdf5_metno::LocationToken)>,
}

fn member_links(
    group: &hdf5_metno::Group,
) -> Result<Vec<(String, hdf5_metno::LinkType)>, AdapterError> {
    group
        .iter_visit_default(Vec::new(), |_, name, info, members| {
            members.push((name.to_owned(), info.link_type));
            true
        })
        .map_err(invalid)
}

fn open_hard_group(
    parent: &hdf5_metno::Group,
    path: &str,
) -> Result<Option<hdf5_metno::Group>, AdapterError> {
    let mut group = parent.clone();
    for component in path.split('/').filter(|component| !component.is_empty()) {
        let link_type = member_links(&group)?
            .into_iter()
            .find(|(name, _)| name == component)
            .map(|(_, link_type)| link_type);
        match link_type {
            Some(hdf5_metno::LinkType::Hard) => {
                group = group.group(component).map_err(invalid)?;
            }
            Some(hdf5_metno::LinkType::Soft | hdf5_metno::LinkType::External) | None => {
                return Ok(None)
            }
        }
    }
    Ok(Some(group))
}

fn read_external_assets(
    group: &hdf5_metno::Group,
    series_path: &str,
    retained_count: usize,
    budget: &mut DecodeBudget,
) -> Result<Vec<ExternalAsset>, AdapterError> {
    let dataset = group.dataset("external_file").map_err(invalid)?;
    let count = dataset.size();
    if count == 0
        || retained_count
            .checked_add(count)
            .ok_or(AdapterError::SourceTooLarge)?
            > MAX_EXTERNAL_ASSETS
    {
        return Err(AdapterError::SourceTooLarge);
    }
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(invalid)?;
    let file_names = match descriptor {
        TypeDescriptor::VarLenUnicode => dataset
            .read_raw::<hdf5_metno::types::VarLenUnicode>()
            .map_err(invalid)?
            .into_iter()
            .map(|value| value.as_str().to_owned())
            .collect::<Vec<_>>(),
        TypeDescriptor::VarLenAscii => dataset
            .read_raw::<hdf5_metno::types::VarLenAscii>()
            .map_err(invalid)?
            .into_iter()
            .map(|value| value.as_str().to_owned())
            .collect::<Vec<_>>(),
        other => {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "NWB external_file uses unsupported string type {other:?}"
            )))
        }
    };
    let starting_frames = if let Ok(attribute) = dataset.attr("starting_frame") {
        let descriptor = attribute
            .dtype()
            .and_then(|dtype| dtype.to_descriptor())
            .map_err(invalid)?;
        if descriptor != TypeDescriptor::Integer(IntSize::U8) {
            return Err(AdapterError::UnsupportedMeaning(format!(
                "NWB ImageSeries {series_path} starting_frame attribute is not i64"
            )));
        }
        let values = attribute.read_raw::<i64>().map_err(invalid)?;
        if values.len() != file_names.len() {
            return Err(AdapterError::InvalidSource(format!(
                "NWB ImageSeries {series_path} has mismatched external_file and starting_frame lengths"
            )));
        }
        budget.reserve_retained(
            u64::try_from(values.len())
                .map_err(|_| AdapterError::SourceTooLarge)?
                .checked_mul(16)
                .ok_or(AdapterError::SourceTooLarge)?,
        )?;
        values.into_iter().map(Some).collect::<Vec<_>>()
    } else if group.link_exists("starting_frame") {
        match read_numeric(&group.dataset("starting_frame").map_err(invalid)?, budget)? {
            SeriesValues::Integer(values) if values.len() == file_names.len() => {
                values.into_iter().map(Some).collect::<Vec<_>>()
            }
            SeriesValues::Integer(_) => {
                return Err(AdapterError::InvalidSource(format!(
                    "NWB ImageSeries {series_path} has mismatched external_file and starting_frame lengths"
                )))
            }
            SeriesValues::Real(_) => {
                return Err(AdapterError::UnsupportedMeaning(format!(
                    "NWB ImageSeries {series_path} starting_frame is not integer"
                )))
            }
        }
    } else {
        vec![None; file_names.len()]
    };
    let retained_bytes = file_names
        .iter()
        .try_fold(0_u64, |total, file_name| {
            if file_name.is_empty() || file_name.len() > MAX_EXTERNAL_ASSET_NAME_BYTES {
                return None;
            }
            total.checked_add(
                (core::mem::size_of::<ExternalAsset>() + series_path.len() + file_name.len())
                    as u64,
            )
        })
        .ok_or(AdapterError::SourceTooLarge)?;
    budget.reserve_retained(retained_bytes)?;
    Ok(file_names
        .into_iter()
        .zip(starting_frames)
        .enumerate()
        .map(|(file_index, (file_name, starting_frame))| ExternalAsset {
            series_path: series_path.to_owned(),
            file_index,
            file_name,
            starting_frame,
        })
        .collect())
}

fn walk_container(
    parent: &hdf5_metno::Group,
    container: Container,
    prefix: &str,
    state: &mut WalkState<'_>,
    depth: usize,
) -> Result<(), AdapterError> {
    if depth > MAX_GROUP_DEPTH {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB group nesting exceeds the adapter depth limit".to_owned(),
        ));
    }
    for (name, link_type) in member_links(parent)? {
        let path = format!("{prefix}/{name}");
        if link_type != hdf5_metno::LinkType::Hard {
            continue;
        }
        let Ok(group) = parent.group(&name) else {
            continue;
        };
        mark_group(&group, state.visited)?;
        if group.link_exists("external_file") {
            let assets = read_external_assets(&group, &path, state.external.len(), state.budget)?;
            state.external.extend(assets);
            continue;
        }
        if read_text_attr(&group, "neurodata_type").is_none() && !group.link_exists("data") {
            walk_container(&group, container, &path, state, depth + 1)?;
            continue;
        }
        match read_series(&group, container, &path, state.budget)? {
            Some(found) => {
                if state.series.len() >= MAX_SERIES {
                    return Err(AdapterError::UnsupportedMeaning(
                        "NWB file declares more series than this adapter will import".to_owned(),
                    ));
                }
                state.series.push(found);
            }
            None => walk_container(&group, container, &path, state, depth + 1)?,
        }
    }
    Ok(())
}

fn reject_hdf5_external_links(
    parent: &hdf5_metno::Group,
    prefix: &str,
    visited: &mut Vec<(u64, hdf5_metno::LocationToken)>,
    depth: usize,
) -> Result<(), AdapterError> {
    if depth > MAX_GROUP_DEPTH {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB group nesting exceeds the adapter depth limit".to_owned(),
        ));
    }
    for (name, link_type) in member_links(parent)? {
        let path = format!("{prefix}/{name}");
        match link_type {
            hdf5_metno::LinkType::External => {
                return Err(AdapterError::UnsupportedMeaning(format!(
                    "NWB HDF5 external link is not self-contained: {path}"
                )))
            }
            hdf5_metno::LinkType::Soft => continue,
            hdf5_metno::LinkType::Hard => {}
        }
        let Ok(group) = parent.group(&name) else {
            continue;
        };
        if mark_external_scan_group(&group, visited)? {
            reject_hdf5_external_links(&group, &path, visited, depth + 1)?;
        }
    }
    Ok(())
}

fn mark_external_scan_group(
    group: &hdf5_metno::Group,
    visited: &mut Vec<(u64, hdf5_metno::LocationToken)>,
) -> Result<bool, AdapterError> {
    if visited.len() >= MAX_GROUPS {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB file declares more groups than this adapter will inspect".to_owned(),
        ));
    }
    let info = group.loc_info().map_err(invalid)?;
    if visited
        .iter()
        .any(|(file, token)| *file == info.fileno && *token == info.token)
    {
        return Ok(false);
    }
    visited.push((info.fileno, info.token));
    Ok(true)
}

fn mark_group(
    group: &hdf5_metno::Group,
    visited: &mut Vec<(u64, hdf5_metno::LocationToken)>,
) -> Result<(), AdapterError> {
    if visited.len() >= MAX_GROUPS {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB file declares more groups than this adapter will traverse".to_owned(),
        ));
    }
    let info = group.loc_info().map_err(invalid)?;
    if visited
        .iter()
        .any(|(file, token)| *file == info.fileno && *token == info.token)
    {
        return Err(AdapterError::UnsupportedMeaning(
            "NWB group graph contains a cycle or repeated hard link".to_owned(),
        ));
    }
    visited.push((info.fileno, info.token));
    Ok(())
}

fn parse_nwb(path: &std::path::Path, max_decoded_bytes: u64) -> Result<ParsedNwb, AdapterError> {
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

    let mut external_scan_visited = Vec::new();
    mark_external_scan_group(&root, &mut external_scan_visited)?;
    reject_hdf5_external_links(&root, "", &mut external_scan_visited, 0)?;

    let mut series = Vec::new();
    let mut external_assets = Vec::new();
    let mut budget = DecodeBudget::new(max_decoded_bytes);
    let mut visited = Vec::new();
    {
        let mut walk = WalkState {
            series: &mut series,
            external: &mut external_assets,
            budget: &mut budget,
            visited: &mut visited,
        };
        for (name, container) in [
            ("acquisition", Container::Acquisition),
            ("stimulus", Container::Stimulus),
            ("scratch", Container::Scratch),
        ] {
            if let Some(group) = open_hard_group(&root, name)? {
                mark_group(&group, walk.visited)?;
                walk_container(&group, container, &format!("/{name}"), &mut walk, 0)?;
            }
        }
        if let Some(processing) = open_hard_group(&root, "processing")? {
            mark_group(&processing, walk.visited)?;
            for (module, link_type) in member_links(&processing)? {
                if link_type != hdf5_metno::LinkType::Hard {
                    continue;
                }
                let Ok(group) = processing.group(&module) else {
                    continue;
                };
                mark_group(&group, walk.visited)?;
                let container = if module.eq_ignore_ascii_case("behavior") {
                    Container::Behavior
                } else {
                    Container::Derived
                };
                walk_container(
                    &group,
                    container,
                    &format!("/processing/{module}"),
                    &mut walk,
                    0,
                )?;
            }
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
    if let Some(table) = open_hard_group(&root, "general/extracellular_ephys/electrodes")? {
        electrode_columns = table.member_names().map_err(invalid)?;
        let dataset = table.dataset("id").map_err(|_| {
            AdapterError::UnsupportedMeaning(
                "NWB electrodes table lacks its required numeric id column".to_owned(),
            )
        })?;
        let rows = dataset.size();
        if rows > MAX_ELECTRODES {
            return Err(AdapterError::SourceTooLarge);
        }
        let derived_bytes = u64::try_from(rows)
            .map_err(|_| AdapterError::SourceTooLarge)?
            .checked_mul(512)
            .ok_or(AdapterError::SourceTooLarge)?;
        budget.reserve_retained(derived_bytes)?;
        electrodes = match read_numeric(&dataset, &mut budget)? {
            SeriesValues::Integer(values) => {
                values.into_iter().map(|value| value.to_string()).collect()
            }
            SeriesValues::Real(_) => {
                return Err(AdapterError::UnsupportedMeaning(
                    "NWB electrodes id column is not integer".to_owned(),
                ))
            }
        };
    }

    // Interval tables: every row is a claim about time.
    let mut intervals = Vec::new();
    if let Some(group) = open_hard_group(&root, "intervals")? {
        for (name, link_type) in member_links(&group)? {
            if link_type != hdf5_metno::LinkType::Hard {
                continue;
            }
            let Ok(table) = group.group(&name) else {
                continue;
            };
            if !table.link_exists("start_time") || !table.link_exists("stop_time") {
                continue;
            }
            let start_dataset = table.dataset("start_time").map_err(invalid)?;
            let stop_dataset = table.dataset("stop_time").map_err(invalid)?;
            let rows = start_dataset.size();
            let interval_count = intervals
                .len()
                .checked_add(rows)
                .ok_or(AdapterError::SourceTooLarge)?;
            if rows != stop_dataset.size() || interval_count > MAX_INTERVALS {
                return Err(AdapterError::SourceTooLarge);
            }
            let derived_bytes = u64::try_from(rows)
                .map_err(|_| AdapterError::SourceTooLarge)?
                .checked_mul(core::mem::size_of::<Interval>() as u64)
                .and_then(|bytes| bytes.checked_add(name.len() as u64 + 32))
                .ok_or(AdapterError::SourceTooLarge)?;
            budget.reserve_retained(derived_bytes)?;
            intervals
                .try_reserve_exact(rows)
                .map_err(|_| AdapterError::SourceTooLarge)?;
            let starts = match read_numeric(&start_dataset, &mut budget)? {
                SeriesValues::Real(values) => values,
                SeriesValues::Integer(_) => {
                    return Err(AdapterError::UnsupportedMeaning(format!(
                        "NWB interval table {name} start_time is not floating point"
                    )))
                }
            };
            let stops = match read_numeric(&stop_dataset, &mut budget)? {
                SeriesValues::Real(values) => values,
                SeriesValues::Integer(_) => {
                    return Err(AdapterError::UnsupportedMeaning(format!(
                        "NWB interval table {name} stop_time is not floating point"
                    )))
                }
            };
            if starts.len() != stops.len() {
                return Err(AdapterError::InvalidSource(format!(
                    "NWB interval table {name} has mismatched start and stop columns"
                )));
            }
            let table_name = Arc::<str>::from(name);
            for (start, stop) in starts.into_iter().zip(stops) {
                intervals.push(Interval {
                    table: Arc::clone(&table_name),
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
    timing_changed: bool,
}

impl NwbAdapter {
    pub fn new(max_source_bytes: u64) -> Self {
        Self::with_decoded_limit(max_source_bytes, max_source_bytes)
    }

    pub fn with_decoded_limit(max_source_bytes: u64, max_decoded_bytes: u64) -> Self {
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
            max_decoded_bytes,
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
        parse_nwb(&path, self.max_decoded_bytes)
    }

    fn parse(
        &self,
        entry: &ForeignEntry,
        limits: ValidationLimits,
    ) -> Result<ParsedDataset, AdapterError> {
        let decoded_limit = self.max_decoded_bytes.min(limits.max_logical_payload_bytes);
        let temporary = tempfile::tempdir().map_err(invalid)?;
        let path = temporary.path().join("source.nwb");
        fs::write(&path, &entry.bytes).map_err(invalid)?;
        let parsed = parse_nwb(&path, decoded_limit)?;
        let seed = blake3::hash(&entry.bytes);
        let dataset_id = id::<DatasetTag>(&seed, b"dataset", 0);
        let recording_id = id::<RecordingTag>(&seed, b"recording", 0);
        let clock_id = id::<ClockTag>(&seed, b"session-clock", 0);
        let basis_id = id::<ChannelBasisTag>(&seed, b"electrode-basis", 0);
        let mut draft = DatasetDraft::new(dataset_id);
        let mut payloads = Vec::new();
        let mut mappings = Vec::new();
        let mut timing_changed = false;

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
            let mut series_timing_changed = false;
            let time_axis = match series.regular {
                Some((start, rate)) => {
                    let (start, start_changed) = seconds(start)?;
                    let (rate, rate_changed) = seconds(rate)?;
                    series_timing_changed = start_changed || rate_changed;
                    TimeAxis::Regular(TimeSegment::new(start, rate, series.rows).map_err(invalid)?)
                }
                None => {
                    // ABIR carries explicit timestamps as exact integer ticks,
                    // and the axis names a companion payload that must belong
                    // to a real atom -- a dangling reference is exactly what
                    // validation refuses.
                    let mut stamps = Vec::with_capacity(series.timestamps.len() * 8);
                    for value in &series.timestamps {
                        let (ticks, changed) = microsecond_ticks(*value)?;
                        series_timing_changed |= changed;
                        stamps.extend_from_slice(&ticks.to_le_bytes());
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
            if series_timing_changed {
                timing_changed = true;
                mappings.push(MappingEntry {
                    source_path: format!("{}/data", series.path),
                    target: format!("atom:{atom_id}"),
                    disposition: MappingDisposition::Projected,
                    reason: Some(
                        "NWB series timing was projected to ABIR microsecond precision".to_owned(),
                    ),
                });
            } else {
                mappings.push(exact(
                    format!("{}/data", series.path),
                    format!("atom:{atom_id}"),
                ));
            }
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

        let mut interval_timing_changed = false;
        for (index, interval) in parsed.intervals.iter().enumerate() {
            let event_id = id::<EventTag>(&seed, b"interval", index as u64);
            let (start, start_changed) = seconds(interval.start)?;
            let (stop, stop_changed) = seconds(interval.stop)?;
            interval_timing_changed |= start_changed || stop_changed;
            draft.add_event(Event::new(
                event_id,
                concept(&format!("nwb:interval/{}", interval.table))?,
                clock_id,
                start,
                stop,
                Rational::new(0, 1).expect("zero is a rational"),
            ));
        }
        if !parsed.intervals.is_empty() {
            if interval_timing_changed {
                timing_changed = true;
                mappings.push(MappingEntry {
                    source_path: "/intervals".to_owned(),
                    target: "events:nwb:interval".to_owned(),
                    disposition: MappingDisposition::Projected,
                    reason: Some(
                        "NWB interval timing was projected to ABIR microsecond precision"
                            .to_owned(),
                    ),
                });
            } else {
                mappings.push(exact(
                    "/intervals".to_owned(),
                    "events:nwb:interval".to_owned(),
                ));
            }
        }

        // External assets: named, never inlined.
        for asset in &parsed.external_assets {
            mappings.push(MappingEntry {
                source_path: format!("{}/external_file[{}]", asset.series_path, asset.file_index),
                target: format!("external-asset:{}", asset.file_name),
                disposition: MappingDisposition::Quarantined,
                reason: Some(match asset.starting_frame {
                    Some(frame) => format!(
                        "external file was not supplied; ImageSeries starts it at frame {frame}"
                    ),
                    None => "external file was not supplied by this source object".to_owned(),
                }),
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
        for asset in &parsed.external_assets {
            recording.add_source_key(source_key("nwb.external-asset", &asset.file_name)?);
            recording.add_source_key(source_key(
                "nwb.external-asset-series",
                &format!(
                    "{}|file={}|starting_frame={}",
                    asset.series_path,
                    asset.file_name,
                    asset
                        .starting_frame
                        .map(|frame| frame.to_string())
                        .unwrap_or_else(|| "unknown".to_owned())
                ),
            )?);
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
            timing_changed,
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
                timing_changed: parsed.timing_changed,
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
