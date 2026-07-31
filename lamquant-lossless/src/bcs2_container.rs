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
use crate::source::{from_uniform_signal_view, SourceMetadata};

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

/// Explicit packetization choices for one exact LML profile encoding.
#[derive(Clone, Copy, Debug)]
pub struct LmlEncodeOptions {
    pub window_size: usize,
    pub lpc_mode: crate::lpc::LpcMode,
}

impl Default for LmlEncodeOptions {
    fn default() -> Self {
        Self {
            window_size: lamquant_abir_codec::MAX_PACKET_SAMPLES,
            lpc_mode: crate::lpc::LpcMode::default(),
        }
    }
}

/// Encoded BCS2 LML bytes plus measurements derived from the same ABIR root.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedLml {
    bytes: Vec<u8>,
    stats: ContainerStats,
}

impl EncodedLml {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn stats(&self) -> &ContainerStats {
        &self.stats
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
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
    lamquant_abir_codec::encode_lml_bundle(dataset, access, ResourceBounds::default())
        .map_err(bundle_error)
}

/// Encode canonical ABIR with explicit exact-profile packetization.
pub fn encode_with_options<A: PayloadAccess>(
    dataset: &AbirDataset,
    access: &A,
    options: LmlEncodeOptions,
) -> LmlResult<EncodedLml> {
    if options.window_size == 0 {
        return Err(invalid("window size must be greater than zero"));
    }
    let packet_samples = options
        .window_size
        .min(lamquant_abir_codec::MAX_PACKET_SAMPLES);
    let bytes = lamquant_abir_codec::encode_lml_bundle_with_window_size_and_mode(
        dataset,
        access,
        packet_samples,
        options.lpc_mode,
        ResourceBounds::default(),
    )
    .map_err(bundle_error)?;
    finish_encoded(dataset, packet_samples, bytes)
}

/// Encode a native channel-major matrix whose exact closure is declared by
/// `dataset`. This is the fused host path: ABIR remains the semantic seam while
/// the kernel consumes its efficient native form without decoding payloads.
pub fn encode_from_signal_with_options(
    dataset: &AbirDataset,
    signal: &[Vec<i64>],
    options: LmlEncodeOptions,
) -> LmlResult<EncodedLml> {
    if options.window_size == 0 {
        return Err(invalid("window size must be greater than zero"));
    }
    let packet_samples = options
        .window_size
        .min(lamquant_abir_codec::MAX_PACKET_SAMPLES);
    let bytes = lamquant_abir_codec::encode_lml_bundle_from_signal_with_mode(
        dataset,
        signal,
        packet_samples,
        options.lpc_mode,
        ResourceBounds::default(),
    )
    .map_err(bundle_error)?;
    finish_encoded(dataset, packet_samples, bytes)
}

/// Every wire frozen by ADR 0071 and reaffirmed by ADR 0118. These are
/// decode-forever, but NOT here: ADR 0139/0143 put retired wires behind the
/// independent, supervised legacy Adapter process, and this crate's own
/// manifest records that the compatibility selector "must never restore a
/// main-graph legacy dependency". Recognising a magic costs eight bytes and no
/// dependency; decoding it would cost the dependency, so this only names them.
const FROZEN_LEGACY_MAGICS: &[(&[u8], &str)] = &[
    (b"LMLCRYPT", "LMLCRYPT"),
    (b"LMLFOOT1", "LMLFOOT1"),
    (b"LML1", "LML1"),
    (b"LMO1", "LMO1"),
    (b"LMA1", "LMA1"),
    (b"LMA2", "LMA2"),
    (b"LMQC", "LMQC"),
    (b"LFT2", "LFT2"),
];

/// Name the retired wire a byte slice starts with, if it is one.
fn frozen_legacy_magic(data: &[u8]) -> Option<&'static str> {
    FROZEN_LEGACY_MAGICS
        .iter()
        .find(|(magic, _)| data.starts_with(magic))
        .map(|(_, name)| *name)
}

/// Authenticate and decode a BCS2 LML profile.
pub fn open(data: &[u8]) -> LmlResult<lamquant_abir_codec::OpenedLmlBundle<'_>> {
    // Answer the retired wires by name before handing the bytes to the BCS2
    // opener. It rejects them on a leading-magic comparison, so without this a
    // Gen 1-7 file reports `Bcs2(BadMagic)` -- which reads as "corrupt" when
    // the file is fine and simply belongs to a reader that lives elsewhere.
    // Refusing is correct; refusing uninformatively is not.
    if let Some(name) = frozen_legacy_magic(data) {
        return Err(invalid(format!(
            "{name} is a retired wire and is not read by this binary. It remains \
             decode-forever through the independent legacy Adapter process \
             (capability `legacy.lml1.v1`); see ADR 0071 and ADR 0143."
        )));
    }
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
    encode_from_signal_with_options(
        semantic.opened.dataset(),
        signal,
        LmlEncodeOptions {
            window_size,
            lpc_mode,
        },
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
    let mut file = temporary.reopen().map_err(LmlError::Io)?;
    let encoded = encode_uniform_signal(
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
    )?;
    file.write_all(encoded.bytes()).map_err(LmlError::Io)?;
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

/// Project authenticated ABIR semantics into the compatibility header.
///
/// This is intentionally a scalar metadata view. Sample access remains behind
/// the already-validated opened bundle.
pub fn header(opened: &lamquant_abir_codec::OpenedLmlBundle<'_>) -> LmlResult<ContainerHeader> {
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

fn profile_shape(dataset: &AbirDataset) -> LmlResult<(usize, usize, f64)> {
    let stream = dataset
        .streams()
        .first()
        .ok_or_else(|| invalid("BCS2 LML dataset has no stream"))?;
    let n_channels = stream.atoms().len();
    if n_channels == 0 {
        return Err(invalid("BCS2 LML stream has no signal atoms"));
    }
    let mut total_samples = None;
    for atom_id in stream.atoms() {
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or_else(|| invalid("BCS2 LML stream atom is unresolved"))?;
        let descriptor = atom
            .payload()
            .ok_or_else(|| invalid("BCS2 LML stream atom has no payload"))?;
        let samples = descriptor
            .shape()
            .last()
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| invalid("BCS2 LML signal has no representable sample extent"))?;
        match total_samples {
            Some(expected) if expected != samples => {
                return Err(invalid("BCS2 LML signal channel extents differ"));
            }
            None => total_samples = Some(samples),
            _ => {}
        }
    }
    Ok((
        n_channels,
        total_samples.expect("non-empty stream establishes sample extent"),
        sample_rate(dataset)?,
    ))
}

fn finish_encoded(
    dataset: &AbirDataset,
    packet_samples: usize,
    bytes: Vec<u8>,
) -> LmlResult<EncodedLml> {
    let (n_channels, total_samples, sample_rate_hz) = profile_shape(dataset)?;
    let stats = stats(
        n_channels,
        total_samples,
        packet_samples,
        bytes.len(),
        sample_rate_hz,
    );
    Ok(EncodedLml { bytes, stats })
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
    // `Lml(inner)` already IS an `LmlError`, and it is the only variant that
    // carries a typed data-integrity verdict -- CrcMismatch, Truncated,
    // InvalidMagic. Wrapping it in InvalidHeader threw that verdict away and
    // reported bit-rot and truncation as "bad header", which is both wrong and
    // the least useful thing to tell someone holding a damaged archive.
    // Callers matching on LmlError::CrcMismatch could never see one through
    // this seam; the conformance suite pins exactly those variants.
    match error {
        lamquant_abir_codec::LmlBundleError::Lml(inner) => inner,
        other => LmlError::InvalidHeader(format!("BCS2 LML bundle rejected: {other}")),
    }
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
        let result = encoded.stats().clone();
        let bytes = encoded.into_bytes();
        assert_eq!(&bytes[..4], b"ABIR");
        assert_eq!(result.n_channels, 2);
        assert_eq!(result.n_windows, 1);
        let (decoded, metadata) = read_bytes(&bytes).unwrap();
        assert_eq!(decoded, signal);
        assert_eq!(metadata, "{\"source\":\"test\"}");
    }

    #[test]
    fn payload_and_native_abir_encoders_are_byte_identical() {
        let signal = vec![vec![1, -2, 3, -4], vec![5, -6, 7, -8]];
        let semantic = from_uniform_signal_view(
            &signal,
            250.0,
            vec!["ch0".into(), "ch1".into()],
            vec![-4.0, -8.0],
            vec![3.0, 7.0],
            4.0 / 250.0,
            SourceMetadata {
                source_file: String::new(),
                format: "BCS2-LML".into(),
                patient_id: String::new(),
                recording_info: "{\"source\":\"test\"}".into(),
                startdate: String::new(),
                phys_dim: "digital".into(),
            },
            semantic_abir::ValidationLimits::default(),
        )
        .unwrap();
        let payload_encoded = encode_with_options(
            semantic.opened.dataset(),
            semantic.opened.access(),
            LmlEncodeOptions {
                window_size: 3,
                lpc_mode: crate::lpc::LpcMode::Fixed,
            },
        )
        .unwrap();
        let native_encoded = encode_from_signal_with_options(
            semantic.opened.dataset(),
            &signal,
            LmlEncodeOptions {
                window_size: 3,
                lpc_mode: crate::lpc::LpcMode::Fixed,
            },
        )
        .unwrap();

        assert_eq!(payload_encoded, native_encoded);
    }

    #[test]
    fn bcs2_round_trip_preserves_multiple_windows_and_random_access() {
        let signal = vec![
            (0..10).map(i64::from).collect::<Vec<_>>(),
            (100..110).map(i64::from).collect::<Vec<_>>(),
        ];
        let encoded =
            encode_uniform_signal(&signal, 250.0, 4, 0, "{}", crate::lpc::LpcMode::Fixed).unwrap();
        let result = encoded.stats().clone();
        let bytes = encoded.into_bytes();
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
            let bytes = encode_uniform_signal(&signal, 250.0, 32, 0, "{}", mode)
                .unwrap()
                .into_bytes();
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
        let error =
            encode_uniform_signal(&[vec![1, 2]], 250.0, 2, 1, "{}", crate::lpc::LpcMode::Fixed)
                .unwrap_err();
        assert!(error.to_string().contains("registered lossy profile"));
    }
}
