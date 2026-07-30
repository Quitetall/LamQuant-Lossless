use lamquant_core::container::{self, EncodedLml, LmlEncodeOptions};
use lamquant_core::lpc::LpcMode;
use lamquant_core::source::{from_uniform_signal_view, SourceMetadata};

pub fn encode_uniform_signal(
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> EncodedLml {
    let n_channels = signal.len();
    let total_samples = signal.first().map_or(0, Vec::len);
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
    )
    .expect("uniform test signal lowers to ABIR");
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    container::encode_views_with_options(
        semantic.opened.dataset(),
        &views,
        LmlEncodeOptions::new(window_size).with_lpc_mode(lpc_mode),
    )
    .expect("ABIR test signal encodes")
}
