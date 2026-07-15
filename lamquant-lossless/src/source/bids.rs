//! BIDS dataset adapter into immutable ABIR2 recordings.
//!
//! The signal payload is delegated to the existing format readers. This layer
//! owns BIDS entity identity, inheritance-aware sidecar selection, canonical
//! event/table/coordinate projection, and byte-exact sidecar preservation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use abir::{
    Attachment, CoordinateFrame, CoordinatePoint, Event, Interval, LossReceipt, Property,
    PropertyBag, ProvenanceActivity, QualifiedName, Recording, SemanticDisposition, Table,
    TableColumn, Unit, Value, ValueType,
};

use crate::error::{LmlError, LmlResult};

use super::abir2::{
    recording_builder_from_signal_bundle_with_options, RecordingAdapterOptions, CLOCK_ID,
};
use super::{
    BrainVisionReader, CntReader, EdfReader, EeglabReader, SignalBundle, SignalSourceReader,
};

const BIDS_NAMESPACE: &str = "bids";
const MAX_SIDECAR_BYTES: u64 = 64 * 1024 * 1024;

/// Dataset-level reader for one BIDS signal file.
#[derive(Clone, Debug)]
pub struct BidsRecordingReader {
    signal_path: PathBuf,
}

impl BidsRecordingReader {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            signal_path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.signal_path
    }

    /// Read the signal plus its effective BIDS sidecars into one immutable
    /// source-agnostic recording graph.
    pub fn read_recording(&self) -> LmlResult<Recording> {
        let source_name = BidsName::parse(&self.signal_path)?;
        let root = find_bids_root(&self.signal_path)?;
        let sidecars = resolve_sidecars(&root, &self.signal_path, &source_name)?;
        let inherited_json = sidecars
            .iter()
            .filter(|sidecar| sidecar.kind == source_name.suffix && sidecar.extension == "json")
            .collect::<Vec<_>>();
        let extensions = merged_json_extension_properties(&inherited_json)?;

        let bundle = read_signal_bundle(&self.signal_path)?;
        let sample_rate = bundle.sample_rate;
        let channel_labels = bundle.channels.clone();
        let options = RecordingAdapterOptions {
            subject: source_name.entities.get("sub").cloned(),
            session: source_name.entities.get("ses").cloned(),
            run: source_name.entities.get("run").cloned(),
            declared_modality: Some(source_name.suffix.clone()),
        };
        let mut builder =
            recording_builder_from_signal_bundle_with_options(bundle, options, extensions)?;

        for (index, sidecar) in sidecars.iter().enumerate() {
            let id = format!("attachment:bids:{index:04}:{}", canonical_id(&sidecar.kind));
            builder
                .add_attachment(Attachment::new(
                    &id,
                    sidecar_media_type(&sidecar.extension),
                    Arc::from(sidecar.bytes.clone()),
                ))
                .map_err(graph_error)?;
            builder
                .add_loss_receipt(LossReceipt::new(
                    format!("receipt:bids:{index:04}:{}", canonical_id(&sidecar.kind)),
                    QualifiedName::new(BIDS_NAMESPACE, &sidecar.kind),
                    SemanticDisposition::Exact,
                    None,
                    "effective BIDS sidecar bytes preserved exactly as an indexed attachment",
                ))
                .map_err(graph_error)?;
        }

        if let Some(events) = sidecars.iter().find(|sidecar| sidecar.kind == "events") {
            add_events(&mut builder, events, sample_rate)?;
        }
        if let Some(channels) = sidecars.iter().find(|sidecar| sidecar.kind == "channels") {
            let table = parse_tsv(&channels.path, &channels.bytes)?;
            builder
                .add_table(table.to_abir("table:bids-channels"))
                .map_err(graph_error)?;
        }
        if let Some(electrodes) = sidecars.iter().find(|sidecar| sidecar.kind == "electrodes") {
            let table = parse_tsv(&electrodes.path, &electrodes.bytes)?;
            add_coordinates(
                &mut builder,
                &table,
                sidecars
                    .iter()
                    .find(|sidecar| sidecar.kind == "coordsystem"),
                &channel_labels,
                &source_name.suffix,
            )?;
            builder
                .add_table(table.to_abir("table:bids-electrodes"))
                .map_err(graph_error)?;
        }

        let mut provenance = ProvenanceActivity::new(
            "provenance:bids-adapter",
            QualifiedName::new(BIDS_NAMESPACE, "dataset-to-abir2"),
            concat!("lamquant-lml/", env!("CARGO_PKG_VERSION")),
        );
        for (index, sidecar) in sidecars.iter().enumerate() {
            provenance = provenance.with_input(format!(
                "attachment:bids:{index:04}:{}",
                canonical_id(&sidecar.kind)
            ));
        }
        builder.add_provenance(provenance).map_err(graph_error)?;
        builder.freeze().map_err(graph_error)
    }
}

#[derive(Clone, Debug)]
struct BidsName {
    entities: BTreeMap<String, String>,
    suffix: String,
}

impl BidsName {
    fn parse(path: &Path) -> LmlResult<Self> {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "BIDS signal path '{}' has no UTF-8 stem",
                    path.display()
                ))
            })?;
        let mut parts = stem.split('_').collect::<Vec<_>>();
        let suffix = parts
            .pop()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!("BIDS signal '{}' has no suffix", path.display()))
            })?;
        let mut entities = BTreeMap::new();
        for part in parts {
            let (key, value) = part.split_once('-').ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "BIDS entity '{part}' in '{}' is not key-value",
                    path.display()
                ))
            })?;
            if key.is_empty()
                || value.is_empty()
                || entities.insert(key.into(), value.into()).is_some()
            {
                return Err(LmlError::InvalidHeader(format!(
                    "BIDS entity '{part}' in '{}' is empty or duplicated",
                    path.display()
                )));
            }
        }
        if !entities.contains_key("sub") {
            return Err(LmlError::InvalidHeader(format!(
                "BIDS entity 'sub' is required in '{}'",
                path.display()
            )));
        }
        match suffix.to_ascii_lowercase().as_str() {
            "eeg" | "ieeg" => {}
            other => {
                return Err(LmlError::InvalidHeader(format!(
                    "BIDS modality suffix '{other}' is not supported by this adapter"
                )))
            }
        }
        Ok(Self {
            entities,
            suffix: suffix.to_ascii_lowercase(),
        })
    }

    fn parse_sidecar(path: &Path) -> Option<(Self, String)> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        if extension != "json" && extension != "tsv" {
            return None;
        }
        let stem = path.file_stem()?.to_str()?;
        let mut parts = stem.split('_').collect::<Vec<_>>();
        let suffix = parts.pop()?.to_owned();
        let mut entities = BTreeMap::new();
        for part in parts {
            let (key, value) = part.split_once('-')?;
            if key.is_empty()
                || value.is_empty()
                || entities.insert(key.into(), value.into()).is_some()
            {
                return None;
            }
        }
        Some((Self { entities, suffix }, extension))
    }
}

#[derive(Clone, Debug)]
struct ResolvedSidecar {
    kind: String,
    extension: String,
    path: PathBuf,
    bytes: Vec<u8>,
}

fn find_bids_root(signal_path: &Path) -> LmlResult<PathBuf> {
    let start = signal_path.parent().unwrap_or_else(|| Path::new("."));
    for directory in start.ancestors() {
        if directory.join("dataset_description.json").is_file() {
            return Ok(directory.to_path_buf());
        }
    }
    Err(LmlError::InvalidHeader(format!(
        "BIDS dataset_description.json not found above '{}'",
        signal_path.display()
    )))
}

fn resolve_sidecars(
    root: &Path,
    signal_path: &Path,
    source: &BidsName,
) -> LmlResult<Vec<ResolvedSidecar>> {
    let mut result = Vec::new();
    let description = root.join("dataset_description.json");
    result.push(read_sidecar("dataset-description", "json", description)?);
    for path in resolve_applicable_sidecars(root, signal_path, source, &source.suffix, "json")? {
        result.push(read_sidecar(&source.suffix, "json", path)?);
    }
    for (suffix, extension) in [
        ("events", "tsv"),
        ("channels", "tsv"),
        ("electrodes", "tsv"),
        ("coordsystem", "json"),
    ] {
        if let Some(path) = resolve_effective_sidecar(root, signal_path, source, suffix, extension)?
        {
            result.push(read_sidecar(suffix, extension, path)?);
        }
    }
    Ok(result)
}

fn resolve_effective_sidecar(
    root: &Path,
    signal_path: &Path,
    source: &BidsName,
    suffix: &str,
    extension: &str,
) -> LmlResult<Option<PathBuf>> {
    Ok(resolve_applicable_sidecars(root, signal_path, source, suffix, extension)?.pop())
}

fn resolve_applicable_sidecars(
    root: &Path,
    signal_path: &Path,
    source: &BidsName,
    suffix: &str,
    extension: &str,
) -> LmlResult<Vec<PathBuf>> {
    let signal_dir = signal_path.parent().unwrap_or_else(|| Path::new("."));
    let mut directories = Vec::new();
    for directory in signal_dir.ancestors() {
        directories.push(directory);
        if directory == root {
            break;
        }
    }
    if directories.last().copied() != Some(root) {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS signal '{}' is outside dataset root '{}'",
            signal_path.display(),
            root.display()
        )));
    }
    directories.reverse();
    let mut candidates = Vec::new();
    for (depth, directory) in directories.into_iter().enumerate() {
        let entries = std::fs::read_dir(directory).map_err(LmlError::Io)?;
        for entry in entries {
            let path = entry.map_err(LmlError::Io)?.path();
            if !path.is_file() {
                continue;
            }
            let Some((candidate, candidate_extension)) = BidsName::parse_sidecar(&path) else {
                continue;
            };
            if candidate.suffix != suffix || candidate_extension != extension {
                continue;
            }
            if !candidate.entities.iter().all(|(key, value)| {
                source
                    .entities
                    .get(key)
                    .is_some_and(|source_value| source_value == value)
            }) {
                continue;
            }
            candidates.push((candidate.entities.len(), depth, path));
        }
    }
    candidates.sort_by(|left, right| (left.0, left.1, &left.2).cmp(&(right.0, right.1, &right.2)));
    for pair in candidates.windows(2) {
        if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
            return Err(LmlError::InvalidHeader(format!(
                "ambiguous effective BIDS {suffix}.{extension} sidecars for '{}'",
                signal_path.display()
            )));
        }
    }
    Ok(candidates
        .into_iter()
        .map(|candidate| candidate.2)
        .collect())
}

fn read_sidecar(kind: &str, extension: &str, path: PathBuf) -> LmlResult<ResolvedSidecar> {
    let metadata = std::fs::metadata(&path).map_err(LmlError::Io)?;
    if metadata.len() > MAX_SIDECAR_BYTES {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS sidecar '{}' is {} bytes; maximum is {MAX_SIDECAR_BYTES}",
            path.display(),
            metadata.len()
        )));
    }
    Ok(ResolvedSidecar {
        kind: kind.to_owned(),
        extension: extension.to_owned(),
        bytes: std::fs::read(&path).map_err(LmlError::Io)?,
        path,
    })
}

fn read_signal_bundle(path: &Path) -> LmlResult<SignalBundle> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "edf" | "bdf" => EdfReader::new(path).read_bundle(),
        "vhdr" => BrainVisionReader::new(path).read_bundle(),
        "set" => EeglabReader::new(path).read_bundle(),
        "cnt" => CntReader::new(path).read_bundle(),
        "dcm" | "dicom" => {
            #[cfg(feature = "dicom")]
            {
                super::DicomWaveformReader::new(path).read_bundle()
            }
            #[cfg(not(feature = "dicom"))]
            {
                Err(LmlError::InvalidHeader(
                    "BIDS DICOM ingest requires the 'dicom' feature".into(),
                ))
            }
        }
        _ => Err(LmlError::InvalidHeader(format!(
            "unsupported BIDS signal extension '.{extension}'"
        ))),
    }
}

#[derive(Clone, Debug)]
struct TsvTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl TsvTable {
    fn column_index(&self, name: &str) -> Option<usize> {
        self.headers.iter().position(|header| header == name)
    }

    fn required_column(&self, name: &str, path: &Path) -> LmlResult<usize> {
        self.column_index(name).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "BIDS sidecar '{}' is missing required column '{name}'",
                path.display()
            ))
        })
    }

    fn to_abir(&self, id: &str) -> Table {
        let mut table = Table::new(id);
        for (column, header) in self.headers.iter().enumerate() {
            let values = self
                .rows
                .iter()
                .map(|row| Value::text(&row[column]))
                .collect::<Vec<_>>();
            table = table.with_column(TableColumn::new(
                QualifiedName::new(BIDS_NAMESPACE, header),
                ValueType::Text,
                values.into(),
            ));
        }
        table
    }
}

fn parse_tsv(path: &Path, bytes: &[u8]) -> LmlResult<TsvTable> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        LmlError::InvalidHeader(format!(
            "BIDS TSV '{}' is not UTF-8: {error}",
            path.display()
        ))
    })?;
    let mut lines = text.lines().filter(|line| !line.is_empty());
    let header_line = lines.next().ok_or_else(|| {
        LmlError::InvalidHeader(format!("BIDS TSV '{}' is empty", path.display()))
    })?;
    let headers = header_line
        .trim_end_matches('\r')
        .split('\t')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if headers.is_empty() || headers.iter().any(String::is_empty) {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS TSV '{}' has an empty header",
            path.display()
        )));
    }
    let mut unique = headers.clone();
    unique.sort();
    unique.dedup();
    if unique.len() != headers.len() {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS TSV '{}' has duplicate headers",
            path.display()
        )));
    }
    let mut rows = Vec::new();
    for (row_index, line) in lines.enumerate() {
        let row = line
            .trim_end_matches('\r')
            .split('\t')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if row.len() != headers.len() {
            return Err(LmlError::InvalidHeader(format!(
                "BIDS TSV '{}' row {} has {} fields; expected {}",
                path.display(),
                row_index + 2,
                row.len(),
                headers.len()
            )));
        }
        rows.push(row);
    }
    Ok(TsvTable { headers, rows })
}

fn add_events(
    builder: &mut abir::RecordingBuilder,
    sidecar: &ResolvedSidecar,
    sample_rate: f64,
) -> LmlResult<()> {
    let table = parse_tsv(&sidecar.path, &sidecar.bytes)?;
    let onset_column = table.required_column("onset", &sidecar.path)?;
    let duration_column = table.column_index("duration");
    let type_column = table.column_index("trial_type");
    for (index, row) in table.rows.iter().enumerate() {
        let onset = parse_seconds(&row[onset_column], "events.tsv onset", &sidecar.path)?;
        let (tick, onset_rounded) = seconds_to_tick(onset, sample_rate, &sidecar.path)?;
        let label = type_column
            .map(|column| row[column].as_str())
            .filter(|value| !value.is_empty() && *value != "n/a")
            .unwrap_or("event");
        let properties = table
            .headers
            .iter()
            .zip(row)
            .map(|(name, value)| {
                Property::new(QualifiedName::new(BIDS_NAMESPACE, name), Value::text(value))
            })
            .collect::<Vec<_>>();
        builder
            .add_event(
                Event::new(
                    format!("event:bids:{index:06}"),
                    CLOCK_ID,
                    tick,
                    QualifiedName::new(BIDS_NAMESPACE, label),
                )
                .with_properties(PropertyBag::new(properties)),
            )
            .map_err(graph_error)?;
        let duration = duration_column
            .map(|column| row[column].as_str())
            .filter(|value| !value.is_empty() && *value != "n/a")
            .map(|value| parse_seconds(value, "events.tsv duration", &sidecar.path))
            .transpose()?
            .unwrap_or(0.0);
        if duration < 0.0 {
            return Err(LmlError::InvalidHeader(format!(
                "BIDS events.tsv duration must be nonnegative in '{}'",
                sidecar.path.display()
            )));
        }
        let mut rounded = onset_rounded;
        if duration > 0.0 {
            let (end_tick, end_rounded) =
                seconds_to_tick(onset + duration, sample_rate, &sidecar.path)?;
            rounded |= end_rounded;
            builder
                .add_interval(Interval::new(
                    format!("interval:bids:{index:06}"),
                    CLOCK_ID,
                    tick,
                    end_tick,
                    QualifiedName::new(BIDS_NAMESPACE, label),
                ))
                .map_err(graph_error)?;
        }
        if rounded {
            builder
                .add_loss_receipt(LossReceipt::new(
                    format!("receipt:bids-event-time:{index:06}"),
                    QualifiedName::new(BIDS_NAMESPACE, "event-time"),
                    SemanticDisposition::Approximated,
                    None,
                    "BIDS seconds rounded to the nearest source-clock tick; original text retained",
                ))
                .map_err(graph_error)?;
        }
    }
    builder
        .add_table(table.to_abir("table:bids-events"))
        .map_err(graph_error)
}

fn parse_seconds(value: &str, field: &str, path: &Path) -> LmlResult<f64> {
    let number = value.parse::<f64>().map_err(|_| {
        LmlError::InvalidHeader(format!(
            "BIDS {field} '{value}' is not numeric in '{}'",
            path.display()
        ))
    })?;
    if !number.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS {field} '{value}' is not finite in '{}'",
            path.display()
        )));
    }
    Ok(number)
}

fn seconds_to_tick(seconds: f64, sample_rate: f64, path: &Path) -> LmlResult<(i64, bool)> {
    let scaled = seconds * sample_rate;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(LmlError::InvalidHeader(format!(
            "BIDS time {seconds} cannot be represented on source clock in '{}'",
            path.display()
        )));
    }
    let rounded = scaled.round();
    Ok((rounded as i64, (scaled - rounded).abs() > 1e-9))
}

fn add_coordinates(
    builder: &mut abir::RecordingBuilder,
    electrodes: &TsvTable,
    coordsystem: Option<&ResolvedSidecar>,
    channel_labels: &[String],
    modality: &str,
) -> LmlResult<()> {
    let Some(coordsystem) = coordsystem else {
        return Ok(());
    };
    let metadata: serde_json::Value =
        serde_json::from_slice(&coordsystem.bytes).map_err(|error| {
            LmlError::InvalidHeader(format!(
                "BIDS coordsystem JSON '{}' is invalid: {error}",
                coordsystem.path.display()
            ))
        })?;
    let object = metadata.as_object().ok_or_else(|| {
        LmlError::InvalidHeader(format!(
            "BIDS coordsystem JSON '{}' must be an object",
            coordsystem.path.display()
        ))
    })?;
    let prefix = if modality == "ieeg" { "iEEG" } else { "EEG" };
    let system = object
        .get(&format!("{prefix}CoordinateSystem"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "BIDS coordsystem '{}' lacks {prefix}CoordinateSystem",
                coordsystem.path.display()
            ))
        })?;
    let units = object
        .get(&format!("{prefix}CoordinateUnits"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "BIDS coordsystem '{}' lacks {prefix}CoordinateUnits",
                coordsystem.path.display()
            ))
        })?;
    let name_column = electrodes.required_column("name", &coordsystem.path)?;
    let x_column = electrodes.required_column("x", &coordsystem.path)?;
    let y_column = electrodes.required_column("y", &coordsystem.path)?;
    let z_column = electrodes.required_column("z", &coordsystem.path)?;
    builder
        .add_coordinate_frame(CoordinateFrame::new(
            "frame:bids-electrodes",
            3,
            QualifiedName::new(BIDS_NAMESPACE, system),
        ))
        .map_err(graph_error)?;
    for (row_index, row) in electrodes.rows.iter().enumerate() {
        if [x_column, y_column, z_column]
            .iter()
            .any(|column| row[*column] == "n/a" || row[*column].is_empty())
        {
            continue;
        }
        let values = [x_column, y_column, z_column]
            .map(|column| parse_seconds(&row[column], "electrode coordinate", &coordsystem.path))
            .into_iter()
            .collect::<LmlResult<Vec<_>>>()?;
        let label = &row[name_column];
        let channel_index = channel_labels
            .iter()
            .position(|candidate| candidate == label)
            .or_else(|| {
                channel_labels
                    .iter()
                    .position(|candidate| candidate.eq_ignore_ascii_case(label))
            })
            .ok_or_else(|| {
                LmlError::InvalidHeader(format!(
                    "BIDS electrode '{label}' has no matching signal channel"
                ))
            })?;
        builder
            .add_coordinate(CoordinatePoint::new(
                format!("coordinate:bids:{row_index:06}"),
                "frame:bids-electrodes",
                format!("signal:channel:{channel_index:06}"),
                Arc::from(values),
                Unit::ucum(units),
            ))
            .map_err(graph_error)?;
    }
    Ok(())
}

fn merged_json_extension_properties(sidecars: &[&ResolvedSidecar]) -> LmlResult<Vec<Property>> {
    let mut merged = BTreeMap::new();
    for sidecar in sidecars {
        let value: serde_json::Value = serde_json::from_slice(&sidecar.bytes).map_err(|error| {
            LmlError::InvalidHeader(format!(
                "BIDS JSON sidecar '{}' is invalid: {error}",
                sidecar.path.display()
            ))
        })?;
        let object = value.as_object().ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "BIDS JSON sidecar '{}' must be an object",
                sidecar.path.display()
            ))
        })?;
        for (key, value) in object {
            merged.insert(key.clone(), value.clone());
        }
    }
    Ok(merged
        .iter()
        .map(|(key, value)| {
            Property::new(
                QualifiedName::new(BIDS_NAMESPACE, key),
                json_to_abir_value(value),
            )
        })
        .collect())
}

fn json_to_abir_value(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(value) => Value::Bool(*value),
        serde_json::Value::Number(value) => value
            .as_u64()
            .map(Value::U64)
            .or_else(|| value.as_i64().map(Value::I64))
            .unwrap_or_else(|| Value::from(value.as_f64().unwrap_or(f64::NAN))),
        serde_json::Value::String(value) => Value::text(value),
        serde_json::Value::Array(values) => Value::list(
            values
                .iter()
                .map(json_to_abir_value)
                .collect::<Vec<_>>()
                .into(),
        ),
        serde_json::Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| key.as_str());
            Value::record(PropertyBag::new(
                entries
                    .into_iter()
                    .map(|(key, value)| {
                        Property::new(
                            QualifiedName::new(BIDS_NAMESPACE, key),
                            json_to_abir_value(value),
                        )
                    })
                    .collect(),
            ))
        }
    }
}

fn canonical_id(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn sidecar_media_type(extension: &str) -> &'static str {
    match extension {
        "json" => "application/json",
        "tsv" => "text/tab-separated-values",
        _ => "application/octet-stream",
    }
}

fn graph_error(error: impl std::fmt::Display) -> LmlError {
    LmlError::InvalidHeader(format!("BIDS ABIR2 adapter: {error}"))
}
