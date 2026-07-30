//! Current-generation LML archive envelope.
//!
//! The LML kernel still produces deterministic `LML1` packets. Public archive
//! artifacts are BCS2 codec bundles whose canonical ABIR root binds those
//! packets to signal semantics. Retired container magics are intentionally not
//! recognized here; callers needing them must use the supervised legacy
//! Adapter process.

use std::io::{Read, Write};
use std::path::Path;

use semantic_abir::{AbirDataset, Atom, PayloadAccess, TimeAxis};
use semantic_abir_bcs::ResourceBounds;

use crate::error::{LmlError, LmlResult};
use crate::source::{from_uniform_signal_view, SemanticRead, SourceMetadata};

/// Summary of one emitted BCS2 LML bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct ContainerStats {
    pub n_windows: usize,
    pub n_channels: usize,
    pub total_samples: usize,
    pub compressed_size: usize,
    pub raw_size: usize,
    pub cr: f64,
    pub duration_s: f64,
}

/// Explicit controls for semantic ABIR-to-LML encoding.
#[derive(Clone, Copy, Debug)]
pub struct LmlEncodeOptions {
    window_size: usize,
    lpc_mode: crate::lpc::LpcMode,
}

impl LmlEncodeOptions {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            lpc_mode: crate::lpc::LpcMode::default(),
        }
    }

    pub const fn with_lpc_mode(mut self, lpc_mode: crate::lpc::LpcMode) -> Self {
        self.lpc_mode = lpc_mode;
        self
    }

    pub const fn window_size(self) -> usize {
        self.window_size
    }

    pub const fn lpc_mode(self) -> crate::lpc::LpcMode {
        self.lpc_mode
    }
}

impl Default for LmlEncodeOptions {
    fn default() -> Self {
        Self::new(lamquant_abir_codec::MAX_PACKET_SAMPLES)
    }
}

/// Encoded BCS2 LML bytes with shape and compression evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLml {
    bytes: Vec<u8>,
    stats: ContainerStats,
}

impl EncodedLml {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub const fn stats(&self) -> &ContainerStats {
        &self.stats
    }
}

/// Semantic shape extracted from an authenticated BCS2 LML bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct ContainerHeader {
    pub n_channels: usize,
    pub total_samples: usize,
    pub n_windows: usize,
    pub window_size: usize,
    pub sample_rate_hz: f64,
    pub metadata: String,
}

/// Encode a canonical ABIR dataset as a BCS2 LML profile.
pub fn encode<A: PayloadAccess>(dataset: &AbirDataset, access: &A) -> LmlResult<Vec<u8>> {
    encode_with_options(dataset, access, LmlEncodeOptions::default()).map(EncodedLml::into_bytes)
}

/// Encode one validated source result, selecting its fused native cache when
/// available and otherwise resolving canonical ABIR payload leases.
pub fn encode_semantic_read(read: &SemanticRead) -> LmlResult<Vec<u8>> {
    encode_semantic_read_with_options(read, LmlEncodeOptions::default()).map(EncodedLml::into_bytes)
}

/// Encode one validated source result with explicit packet controls.
pub fn encode_semantic_read_with_options(
    read: &SemanticRead,
    options: LmlEncodeOptions,
) -> LmlResult<EncodedLml> {
    if let Some(signal) = read.native_i64() {
        let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
        encode_views_with_options(read.opened.dataset(), &views, options)
    } else {
        encode_with_options(read.opened.dataset(), read.opened.access(), options)
    }
}

/// Encode canonical ABIR semantics with explicit packet and predictor controls.
pub fn encode_with_options<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    options: LmlEncodeOptions,
) -> LmlResult<EncodedLml> {
    let (n_channels, total_samples) = semantic_signal_shape(dataset)?;
    let sample_rate = sample_rate(dataset)?;
    let packet_samples = options
        .window_size()
        .min(lamquant_abir_codec::MAX_PACKET_SAMPLES);
    if packet_samples == 0 {
        return Err(invalid("window size must be greater than zero"));
    }
    let bytes = lamquant_abir_codec::encode_lml_bundle_with_window_size_and_mode(
        dataset,
        access,
        packet_samples,
        options.lpc_mode(),
        ResourceBounds::default(),
    )
    .map_err(bundle_error)?;
    let stats = stats(
        n_channels,
        total_samples,
        packet_samples,
        bytes.len(),
        sample_rate,
    );
    Ok(EncodedLml { bytes, stats })
}

/// Encode borrowed native sample views after proving they close over ABIR.
///
/// This is the fused host path: ABIR remains the semantic seam while callers
/// avoid decoding a second sample matrix from an in-memory payload resolver.
pub fn encode_views_with_options(
    dataset: &AbirDataset,
    signal: &[&[i64]],
    options: LmlEncodeOptions,
) -> LmlResult<EncodedLml> {
    let (n_channels, total_samples) = semantic_signal_shape(dataset)?;
    let sample_rate = sample_rate(dataset)?;
    let packet_samples = options
        .window_size()
        .min(lamquant_abir_codec::MAX_PACKET_SAMPLES);
    if packet_samples == 0 {
        return Err(invalid("window size must be greater than zero"));
    }
    let bytes = lamquant_abir_codec::encode_lml_bundle_from_views_with_mode(
        dataset,
        signal,
        packet_samples,
        options.lpc_mode(),
        ResourceBounds::default(),
    )
    .map_err(bundle_error)?;
    let stats = stats(
        n_channels,
        total_samples,
        packet_samples,
        bytes.len(),
        sample_rate,
    );
    Ok(EncodedLml { bytes, stats })
}

/// Authenticate and decode a BCS2 LML profile.
pub fn open(data: &[u8]) -> LmlResult<lamquant_abir_codec::OpenedLmlBundle<'_>> {
    lamquant_abir_codec::open_lml_bundle(data, ResourceBounds::default()).map_err(bundle_error)
}

fn encode_uniform_signal(
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: crate::lpc::LpcMode,
) -> LmlResult<EncodedLml> {
    if noise_bits != 0 {
        return Err(LmlError::InvalidHeader(
            "the BCS2 LML profile is exact; use a registered lossy profile for noise_bits > 0"
                .into(),
        ));
    }
    validate_uniform_signal(signal, sample_rate)?;
    if window_size == 0 {
        return Err(invalid("window size must be greater than zero"));
    }
    let n_channels = signal.len();
    let total_samples = signal[0].len();
    let channels = (0..n_channels).map(|index| format!("ch{index}")).collect();
    let phys_min = signal
        .iter()
        .map(|channel| channel.iter().copied().min().unwrap_or(0) as f64)
        .collect();
    let phys_max = signal
        .iter()
        .map(|channel| channel.iter().copied().max().unwrap_or(0) as f64)
        .collect();
    let semantic = from_uniform_signal_view(
        signal,
        sample_rate,
        channels,
        phys_min,
        phys_max,
        total_samples as f64 / sample_rate,
        SourceMetadata {
            source_file: String::new(),
            format: "BCS2-LML".into(),
            patient_id: String::new(),
            recording_info: metadata_json.into(),
            startdate: String::new(),
            phys_dim: "digital".into(),
        },
        semantic_abir::ValidationLimits::default(),
    )?;
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    encode_views_with_options(
        semantic.opened.dataset(),
        &views,
        LmlEncodeOptions::new(window_size).with_lpc_mode(lpc_mode),
    )
}

pub fn write_file(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
) -> LmlResult<ContainerStats> {
    write_file_with_mode(
        path,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        crate::lpc::LpcMode::default(),
    )
}

pub fn write_file_with_mode(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: crate::lpc::LpcMode,
) -> LmlResult<ContainerStats> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    }
    let temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))
        .map_err(LmlError::Io)?;
    let encoded = encode_uniform_signal(
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
    )?;
    let mut file = temporary.reopen().map_err(LmlError::Io)?;
    file.write_all(encoded.as_bytes()).map_err(LmlError::Io)?;
    file.sync_all().map_err(LmlError::Io)?;
    drop(file);
    temporary
        .persist(path)
        .map_err(|error| LmlError::Io(error.error))?;
    Ok(encoded.stats)
}

pub fn read_bytes(data: &[u8]) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let opened = open(data)?;
    Ok((opened.signal().to_vec(), metadata(opened.dataset())))
}

pub fn read_from<R: Read>(source: &mut R) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let mut data = Vec::new();
    source.read_to_end(&mut data).map_err(LmlError::Io)?;
    read_bytes(&data)
}

/// Decode the current bundle directly into a caller-owned calibrated f32
/// matrix in channel-major order.
pub fn read_bytes_into_f32_calibrated(
    data: &[u8],
    out: &mut [f32],
    calibration: &[f32],
) -> LmlResult<ContainerHeader> {
    let opened = open(data)?;
    let header = header(&opened)?;
    let expected = header
        .n_channels
        .checked_mul(header.total_samples)
        .ok_or_else(|| invalid("decoded matrix size overflows usize"))?;
    if out.len() != expected {
        return Err(invalid(format!(
            "output buffer size mismatch: expected {expected} got {}",
            out.len()
        )));
    }
    if calibration.len() != header.n_channels.saturating_mul(4) {
        return Err(invalid(format!(
            "calibration length {} != n_channels*4 ({})",
            calibration.len(),
            header.n_channels.saturating_mul(4)
        )));
    }
    for (channel_index, channel) in opened.signal().iter().enumerate() {
        calibrate_row(
            channel,
            &mut out
                [channel_index * header.total_samples..(channel_index + 1) * header.total_samples],
            &calibration[channel_index * 4..channel_index * 4 + 4],
        );
    }
    Ok(header)
}

/// Decode selected channels directly into a caller-owned calibrated f32
/// matrix. `u16::MAX` denotes an absent channel and emits a zero row.
pub fn read_bytes_into_f32_calibrated_selected(
    data: &[u8],
    out: &mut [f32],
    calibration: &[f32],
    channel_mask: &[u16],
) -> LmlResult<ContainerHeader> {
    let opened = open(data)?;
    let header = header(&opened)?;
    let expected = channel_mask
        .len()
        .checked_mul(header.total_samples)
        .ok_or_else(|| invalid("selected decoded matrix size overflows usize"))?;
    if out.len() != expected {
        return Err(invalid(format!(
            "selected output buffer size mismatch: expected {expected} got {}",
            out.len()
        )));
    }
    if calibration.len() != channel_mask.len().saturating_mul(4) {
        return Err(invalid(format!(
            "selected calibration length {} != selected_channels*4 ({})",
            calibration.len(),
            channel_mask.len().saturating_mul(4)
        )));
    }

    for (selected_index, &source_index) in channel_mask.iter().enumerate() {
        let row = &mut out
            [selected_index * header.total_samples..(selected_index + 1) * header.total_samples];
        if source_index == u16::MAX {
            row.fill(0.0);
            continue;
        }
        let channel = opened.signal().get(source_index as usize).ok_or_else(|| {
            invalid(format!(
                "channel mask index {source_index} out of range (n_channels={})",
                header.n_channels
            ))
        })?;
        calibrate_row(
            channel,
            row,
            &calibration[selected_index * 4..selected_index * 4 + 4],
        );
    }
    Ok(header)
}

/// Read one authenticated packet window from the current ordered bundle.
pub fn read_window_from_bytes(
    data: &[u8],
    window_index: usize,
) -> LmlResult<(Vec<Vec<i64>>, ContainerHeader)> {
    let opened = open(data)?;
    let header = header(&opened)?;
    if window_index >= header.n_windows {
        return Err(invalid(format!(
            "window index {window_index} out of range (n_windows={})",
            header.n_windows
        )));
    }
    let start = opened.packet_sample_counts()[..window_index]
        .iter()
        .sum::<usize>();
    let end = start + opened.packet_sample_counts()[window_index];
    let window = opened
        .signal()
        .iter()
        .map(|channel| channel[start..end].to_vec())
        .collect();
    Ok((window, header))
}

pub fn parse_header(data: &[u8]) -> LmlResult<ContainerHeader> {
    let opened = open(data)?;
    header(&opened)
}

fn header(opened: &lamquant_abir_codec::OpenedLmlBundle<'_>) -> LmlResult<ContainerHeader> {
    let n_channels = opened.signal().len();
    let total_samples = opened.signal().first().map_or(0, Vec::len);
    let sample_rate_hz = sample_rate(opened.dataset())?;
    Ok(ContainerHeader {
        n_channels,
        total_samples,
        n_windows: opened.packet_sample_counts().len(),
        window_size: opened
            .packet_sample_counts()
            .iter()
            .copied()
            .max()
            .unwrap_or(0),
        sample_rate_hz,
        metadata: metadata(opened.dataset()),
    })
}

fn sample_rate(dataset: &AbirDataset) -> LmlResult<f64> {
    let stream = dataset
        .streams()
        .first()
        .ok_or_else(|| invalid("BCS2 LML dataset has no stream"))?;
    let atom_id = stream
        .atoms()
        .first()
        .ok_or_else(|| invalid("BCS2 LML stream has no signal atom"))?;
    let atom = dataset
        .atoms()
        .iter()
        .find(|atom| atom.id() == *atom_id)
        .ok_or_else(|| invalid("BCS2 LML stream atom is unresolved"))?;
    let Atom::SignalBlock(signal) = atom else {
        return Err(invalid("BCS2 LML stream atom is not a signal block"));
    };
    let TimeAxis::Regular(segment) = signal.time_axis() else {
        return Err(invalid("BCS2 LML profile requires regular time axes"));
    };
    let rate = segment.rate();
    let (numerator, denominator) = rate.parts();
    Ok(numerator as f64 / denominator as f64)
}

fn semantic_signal_shape(dataset: &AbirDataset) -> LmlResult<(usize, usize)> {
    let stream = dataset
        .streams()
        .first()
        .ok_or_else(|| invalid("BCS2 LML dataset has no stream"))?;
    let mut total_samples = None;
    for atom_id in stream.atoms() {
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or_else(|| invalid("BCS2 LML stream atom is unresolved"))?;
        let Atom::SignalBlock(_) = atom else {
            return Err(invalid("BCS2 LML stream atom is not a signal block"));
        };
        let samples = atom
            .payload()
            .ok_or_else(|| invalid("BCS2 LML signal has no payload"))?
            .shape()
            .last()
            .copied()
            .and_then(|samples| usize::try_from(samples).ok())
            .ok_or_else(|| invalid("BCS2 LML signal payload has invalid shape"))?;
        if samples == 0 {
            return Err(invalid("BCS2 LML signal payload is empty"));
        }
        if total_samples
            .replace(samples)
            .is_some_and(|old| old != samples)
        {
            return Err(invalid("BCS2 LML signal payloads are not uniform"));
        }
    }
    Ok((
        stream.atoms().len(),
        total_samples.ok_or_else(|| invalid("BCS2 LML stream has no signal atom"))?,
    ))
}

fn metadata(dataset: &AbirDataset) -> String {
    let Some(recording) = dataset.recordings().first() else {
        return "{}".into();
    };
    if let Some(value) = source_value(recording.source_keys(), "source.recording-info") {
        if !value.is_empty() {
            return value.into();
        }
    }

    let mut object = serde_json::Map::new();
    for (namespace, field) in [
        ("source.file", "source_file"),
        ("source.format", "format"),
        ("source.patient-id", "patient_id"),
        ("source.startdate", "startdate"),
    ] {
        if let Some(value) = source_value(recording.source_keys(), namespace) {
            object.insert(field.into(), serde_json::Value::String(value.into()));
        }
    }

    if let Some(stream) = dataset.streams().first() {
        object.insert(
            "n_channels".into(),
            serde_json::Value::from(stream.atoms().len() as u64),
        );
        if let Ok(rate) = sample_rate(dataset) {
            if let Some(rate) = serde_json::Number::from_f64(rate) {
                object.insert("sample_rate".into(), serde_json::Value::Number(rate));
            }
        }
        if let Some(basis_id) = stream.channel_basis_id() {
            if let Some(basis) = dataset
                .channel_bases()
                .iter()
                .find(|basis| basis.id() == basis_id)
            {
                let mut labels = Vec::with_capacity(basis.channels().len());
                let mut physical_min = Vec::with_capacity(basis.channels().len());
                let mut physical_max = Vec::with_capacity(basis.channels().len());
                let mut unit = None;
                for channel in basis.channels() {
                    labels.push(serde_json::Value::String(
                        source_value(channel.source_keys(), "source.channel-label")
                            .unwrap_or("")
                            .into(),
                    ));
                    physical_min.push(source_number(channel.source_keys(), "source.physical-min"));
                    physical_max.push(source_number(channel.source_keys(), "source.physical-max"));
                    unit = unit
                        .or_else(|| source_value(channel.source_keys(), "source.physical-unit"));
                }
                object.insert("channels".into(), serde_json::Value::Array(labels));
                object.insert("phys_min".into(), serde_json::Value::Array(physical_min));
                object.insert("phys_max".into(), serde_json::Value::Array(physical_max));
                if let Some(unit) = unit {
                    object.insert("phys_dim".into(), serde_json::Value::String(unit.into()));
                }
            }
        }
    }
    serde_json::to_string(&object)
        .expect("serde_json maps with String keys and Value entries are serializable")
}

fn source_value<'a>(keys: &'a [semantic_abir::SourceKey], namespace: &str) -> Option<&'a str> {
    keys.iter()
        .find(|key| key.namespace() == namespace)
        .map(semantic_abir::SourceKey::value)
}

fn source_number(keys: &[semantic_abir::SourceKey], namespace: &str) -> serde_json::Value {
    source_value(keys, namespace)
        .and_then(|value| value.parse::<f64>().ok())
        .and_then(serde_json::Number::from_f64)
        .map_or(serde_json::Value::Null, serde_json::Value::Number)
}

fn validate_uniform_signal(signal: &[Vec<i64>], sample_rate: f64) -> LmlResult<()> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid("sample rate must be finite and positive"));
    }
    let Some(samples) = signal.first().map(Vec::len) else {
        return Err(invalid("signal must contain at least one channel"));
    };
    if samples == 0 || signal.iter().any(|channel| channel.len() != samples) {
        return Err(invalid("signal channels must be non-empty and uniform"));
    }
    Ok(())
}

fn calibrate_row(input: &[i64], output: &mut [f32], calibration: &[f32]) {
    let digital_min = calibration[0];
    let digital_max = calibration[1];
    let physical_min = calibration[2];
    let physical_max = calibration[3];
    let digital_range = digital_max - digital_min;
    if digital_range == 0.0 {
        output.fill(0.0);
        return;
    }
    let scale = (physical_max - physical_min) / digital_range;
    let offset = physical_min - digital_min * scale;
    for (destination, &sample) in output.iter_mut().zip(input) {
        *destination = sample as f32 * scale + offset;
    }
}

fn stats(
    n_channels: usize,
    total_samples: usize,
    window_size: usize,
    compressed_size: usize,
    sample_rate: f64,
) -> ContainerStats {
    let raw_size = n_channels.saturating_mul(total_samples).saturating_mul(8);
    ContainerStats {
        n_windows: total_samples.div_ceil(window_size),
        n_channels,
        total_samples,
        compressed_size,
        raw_size,
        cr: raw_size as f64 / compressed_size.max(1) as f64,
        duration_s: total_samples as f64 / sample_rate,
    }
}

fn bundle_error(error: lamquant_abir_codec::LmlBundleError) -> LmlError {
    LmlError::InvalidHeader(format!("BCS2 LML bundle rejected: {error}"))
}

fn invalid(message: impl Into<String>) -> LmlError {
    LmlError::InvalidHeader(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcs2_round_trip_preserves_samples_and_metadata() {
        let signal = vec![vec![1, -2, 3, -4], vec![5, -6, 7, -8]];
        let encoded = encode_uniform_signal(
            &signal,
            250.0,
            4,
            0,
            "{\"source\":\"test\"}",
            crate::lpc::LpcMode::Fixed,
        )
        .unwrap();
        assert_eq!(&encoded.as_bytes()[..4], b"ABIR");
        assert_eq!(encoded.stats().n_channels, 2);
        assert_eq!(encoded.stats().n_windows, 1);
        let (decoded, metadata) = read_bytes(encoded.as_bytes()).unwrap();
        assert_eq!(decoded, signal);
        assert_eq!(metadata, "{\"source\":\"test\"}");
    }

    #[test]
    fn semantic_encode_options_honor_packet_size_and_round_trip() {
        let signal = vec![
            vec![1, -2, 3, -4, 5, -6, 7],
            vec![8, -9, 10, -11, 12, -13, 14],
        ];
        let semantic = from_uniform_signal_view(
            &signal,
            250.0,
            vec!["Fp1".into(), "Cz".into()],
            vec![-100.0, -200.0],
            vec![100.0, 200.0],
            7.0 / 250.0,
            SourceMetadata {
                source_file: "fixture.edf".into(),
                format: "EDF".into(),
                patient_id: String::new(),
                recording_info: "{\"source\":\"semantic-test\"}".into(),
                startdate: String::new(),
                phys_dim: "uV".into(),
            },
            semantic_abir::ValidationLimits::default(),
        )
        .unwrap();
        let expected_dataset_id = semantic.opened.dataset().id();

        let encoded = encode_with_options(
            semantic.opened.dataset(),
            semantic.opened.access(),
            LmlEncodeOptions::new(3).with_lpc_mode(crate::lpc::LpcMode::Fixed),
        )
        .unwrap();

        assert_eq!(encoded.stats().n_channels, 2);
        assert_eq!(encoded.stats().total_samples, 7);
        assert_eq!(encoded.stats().n_windows, 3);
        let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let fused = encode_views_with_options(
            semantic.opened.dataset(),
            &views,
            LmlEncodeOptions::new(3).with_lpc_mode(crate::lpc::LpcMode::Fixed),
        )
        .unwrap();
        assert_eq!(fused, encoded);
        let opened = open(encoded.as_bytes()).unwrap();
        assert_eq!(opened.packet_sample_counts(), &[3, 3, 1]);
        assert_eq!(opened.signal(), signal);
        assert_eq!(opened.dataset().id(), expected_dataset_id);
    }

    #[test]
    fn metadata_projection_is_derived_from_abir_when_legacy_blob_is_absent() {
        let signal = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let semantic = from_uniform_signal_view(
            &signal,
            500.0,
            vec!["Fp1".into(), "Cz".into()],
            vec![-100.0, -200.0],
            vec![100.0, 200.0],
            3.0 / 500.0,
            SourceMetadata {
                source_file: "fixture.edf".into(),
                format: "EDF".into(),
                patient_id: String::new(),
                recording_info: String::new(),
                startdate: "2026-07-30".into(),
                phys_dim: "uV".into(),
            },
            semantic_abir::ValidationLimits::default(),
        )
        .unwrap();
        let encoded = encode_semantic_read_with_options(
            &semantic,
            LmlEncodeOptions::new(3).with_lpc_mode(crate::lpc::LpcMode::Fixed),
        )
        .unwrap();
        let header = parse_header(encoded.as_bytes()).unwrap();
        let metadata: serde_json::Value = serde_json::from_str(&header.metadata).unwrap();
        assert_eq!(metadata["source_file"], "fixture.edf");
        assert_eq!(metadata["format"], "EDF");
        assert_eq!(metadata["sample_rate"], 500.0);
        assert_eq!(metadata["channels"], serde_json::json!(["Fp1", "Cz"]));
        assert_eq!(metadata["phys_min"], serde_json::json!([-100.0, -200.0]));
        assert_eq!(metadata["phys_dim"], "uV");
    }

    #[test]
    fn malformed_numeric_source_metadata_remains_explicitly_unknown() {
        let keys = [semantic_abir::SourceKey::new("source.physical-min", "not-a-number").unwrap()];

        assert_eq!(
            source_number(&keys, "source.physical-min"),
            serde_json::Value::Null
        );
    }

    #[test]
    fn bcs2_round_trip_preserves_multiple_windows_and_random_access() {
        let signal = vec![
            (0..10).map(i64::from).collect::<Vec<_>>(),
            (100..110).map(i64::from).collect::<Vec<_>>(),
        ];
        let encoded =
            encode_uniform_signal(&signal, 250.0, 4, 0, "{}", crate::lpc::LpcMode::Fixed).unwrap();
        assert_eq!(encoded.stats().n_windows, 3);
        let (decoded, _) = read_bytes(encoded.as_bytes()).unwrap();
        assert_eq!(decoded, signal);
        let (middle, header) = read_window_from_bytes(encoded.as_bytes(), 1).unwrap();
        assert_eq!(middle, vec![vec![4, 5, 6, 7], vec![104, 105, 106, 107]]);
        assert_eq!(header.n_windows, 3);
        assert_eq!(header.window_size, 4);
    }

    #[test]
    fn bcs2_packets_preserve_requested_lpc_mode_bytes() {
        let signal = vec![
            (0..96)
                .map(|sample| {
                    let sample = i64::from(sample);
                    sample * sample - 7 * sample + (sample % 5) * 31
                })
                .collect::<Vec<_>>(),
            (0..96)
                .map(|sample| {
                    let sample = i64::from(sample);
                    (sample % 11) * 101 - sample * 3
                })
                .collect::<Vec<_>>(),
        ];

        for mode in [
            crate::lpc::LpcMode::Fixed,
            crate::lpc::LpcMode::Adaptive { max_order: 16 },
        ] {
            let encoded = encode_uniform_signal(&signal, 250.0, 32, 0, "{}", mode).unwrap();
            let opened = open(encoded.as_bytes()).unwrap();
            let actual = opened.packets().collect::<Vec<_>>();
            let expected = signal[0]
                .chunks(32)
                .enumerate()
                .map(|(index, first_channel)| {
                    let start = index * 32;
                    let end = start + first_channel.len();
                    let window = signal
                        .iter()
                        .map(|channel| channel[start..end].to_vec())
                        .collect::<Vec<_>>();
                    crate::lml::compress_with_mode(&window, 0, mode).unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), expected.len());
            for (packet, expected_packet) in actual.iter().zip(&expected) {
                assert_eq!(*packet, expected_packet.as_slice());
            }
        }
    }

    #[test]
    fn retired_wire_is_not_accepted_in_process() {
        let error = read_bytes(b"BCS1retired").unwrap_err();
        assert!(error.to_string().contains("BCS2 LML bundle rejected"));
    }

    #[test]
    fn lossy_knob_requires_a_registered_profile() {
        let error =
            encode_uniform_signal(&[vec![1, 2]], 250.0, 2, 1, "{}", crate::lpc::LpcMode::Fixed)
                .unwrap_err();
        assert!(error.to_string().contains("registered lossy profile"));
    }
}
