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
use crate::source::{from_signal_bundle, from_uniform_signal_view, SignalBundle, SourceMetadata};

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
    lamquant_abir_codec::encode_lml_bundle(dataset, access, ResourceBounds::default())
        .map_err(bundle_error)
}

/// Authenticate and decode a BCS2 LML profile.
pub fn open(data: &[u8]) -> LmlResult<lamquant_abir_codec::OpenedLmlBundle<'_>> {
    lamquant_abir_codec::open_lml_bundle(data, ResourceBounds::default()).map_err(bundle_error)
}

/// Encode a validated source-neutral signal bundle as canonical ABIR + BCS2.
pub fn encode_signal_bundle(bundle: SignalBundle) -> LmlResult<Vec<u8>> {
    encode_signal_bundle_with_window_size(bundle, lamquant_abir_codec::MAX_PACKET_SAMPLES)
}

/// Encode a source-neutral signal bundle using bounded ordered LML1 packets.
pub fn encode_signal_bundle_with_window_size(
    bundle: SignalBundle,
    window_size: usize,
) -> LmlResult<Vec<u8>> {
    let semantic = from_signal_bundle(bundle, semantic_abir::ValidationLimits::default())?;
    lamquant_abir_codec::encode_lml_bundle_with_window_size(
        semantic.opened.dataset(),
        semantic.opened.access(),
        window_size,
        ResourceBounds::default(),
    )
    .map_err(bundle_error)
}

/// Convenience entry point for callers which already hold a uniform integer
/// matrix. New Adapter and processing seams should prefer [`encode`].
pub fn write_into<W: Write>(
    sink: &mut W,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: crate::lpc::LpcMode,
) -> LmlResult<ContainerStats> {
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
    let packet_samples = window_size.min(lamquant_abir_codec::MAX_PACKET_SAMPLES);
    let bytes = lamquant_abir_codec::encode_lml_bundle_from_signal_with_mode(
        semantic.opened.dataset(),
        signal,
        packet_samples,
        lpc_mode,
        ResourceBounds::default(),
    )
    .map_err(bundle_error)?;
    sink.write_all(&bytes).map_err(LmlError::Io)?;
    Ok(stats(
        n_channels,
        total_samples,
        packet_samples,
        bytes.len(),
        sample_rate,
    ))
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
    let mut file = temporary.reopen().map_err(LmlError::Io)?;
    let result = write_into(
        &mut file,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
    )?;
    file.sync_all().map_err(LmlError::Io)?;
    drop(file);
    temporary
        .persist(path)
        .map_err(|error| LmlError::Io(error.error))?;
    Ok(result)
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

pub fn read_file(path: &Path) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let data = std::fs::read(path).map_err(LmlError::Io)?;
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

fn metadata(dataset: &AbirDataset) -> String {
    dataset
        .recordings()
        .first()
        .and_then(|recording| {
            recording
                .source_keys()
                .iter()
                .find(|key| key.namespace() == "source.recording-info")
        })
        .map_or_else(|| "{}".into(), |key| key.value().into())
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
        let mut bytes = Vec::new();
        let result = write_into(
            &mut bytes,
            &signal,
            250.0,
            4,
            0,
            "{\"source\":\"test\"}",
            crate::lpc::LpcMode::Fixed,
        )
        .unwrap();
        assert_eq!(&bytes[..4], b"ABIR");
        assert_eq!(result.n_channels, 2);
        assert_eq!(result.n_windows, 1);
        let (decoded, metadata) = read_bytes(&bytes).unwrap();
        assert_eq!(decoded, signal);
        assert_eq!(metadata, "{\"source\":\"test\"}");
    }

    #[test]
    fn bcs2_round_trip_preserves_multiple_windows_and_random_access() {
        let signal = vec![
            (0..10).map(i64::from).collect::<Vec<_>>(),
            (100..110).map(i64::from).collect::<Vec<_>>(),
        ];
        let mut bytes = Vec::new();
        let result = write_into(
            &mut bytes,
            &signal,
            250.0,
            4,
            0,
            "{}",
            crate::lpc::LpcMode::Fixed,
        )
        .unwrap();
        assert_eq!(result.n_windows, 3);
        let (decoded, _) = read_bytes(&bytes).unwrap();
        assert_eq!(decoded, signal);
        let (middle, header) = read_window_from_bytes(&bytes, 1).unwrap();
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
            let mut bytes = Vec::new();
            write_into(&mut bytes, &signal, 250.0, 32, 0, "{}", mode).unwrap();
            let opened = open(&bytes).unwrap();
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
        let error = write_into(
            &mut Vec::new(),
            &[vec![1, 2]],
            250.0,
            2,
            1,
            "{}",
            crate::lpc::LpcMode::Fixed,
        )
        .unwrap_err();
        assert!(error.to_string().contains("registered lossy profile"));
    }
}
