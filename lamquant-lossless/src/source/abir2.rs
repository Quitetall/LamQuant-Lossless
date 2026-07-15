//! Host adapters from the legacy reader seam into immutable ABIR2 recordings.
//!
//! Every file reader already lowers into [`SignalBundle`]. This module is the
//! one semantic bridge from that format-neutral carrier into ABIR2. Source
//! format details remain exact attachments with indexed receipts; trainers and
//! downstream projections consume only the canonical graph.

use std::collections::BTreeMap;
use std::sync::Arc;

use abir::{
    modality::infer_modality, name_for_tag, Attachment, ChannelDescriptor, Clock, ClockKind,
    LossReceipt, ModalityId, Property, PropertyBag, ProvenanceActivity, QualifiedName, Rational,
    Recording, RecordingBuilder, RecordingIdentity, SampleBuffer, SemanticDisposition,
    SignalSeries, SignalStream, Table, TableColumn, TimeAxis, Unit, Value, ValueType,
};

use crate::error::{LmlError, LmlResult};

use super::bundle::{SidecarBlob, SignalBundle};

pub(crate) const CLOCK_ID: &str = "clock:source";
const CALIBRATION_TABLE_ID: &str = "table:channel-calibration";
const SIDECAR_TABLE_ID: &str = "table:source-sidecars";
const SOURCE_NAMESPACE: &str = "lamquant.source";
const INTEROP_NAMESPACE: &str = "lamquant.interop";

/// Identity and modality facts supplied by a dataset-level adapter such as
/// BIDS or DICOM. Missing values fall back to the source bundle.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecordingAdapterOptions {
    /// Canonical dataset subject identifier.
    pub subject: Option<String>,
    /// Canonical dataset session identifier.
    pub session: Option<String>,
    /// Canonical dataset run identifier.
    pub run: Option<String>,
    /// Dataset-declared modality. Declared values override label inference.
    pub declared_modality: Option<String>,
}

/// Convert one validated, equal-rate [`SignalBundle`] into immutable ABIR2.
pub fn recording_from_signal_bundle(bundle: SignalBundle) -> LmlResult<Recording> {
    recording_from_signal_bundle_with_options(bundle, RecordingAdapterOptions::default())
}

/// Convert with dataset-level identity and declared-modality overrides.
pub fn recording_from_signal_bundle_with_options(
    bundle: SignalBundle,
    options: RecordingAdapterOptions,
) -> LmlResult<Recording> {
    recording_builder_from_signal_bundle_with_options(bundle, options, Vec::new())?
        .freeze()
        .map_err(graph_error)
}

/// Build the common source graph while leaving the final immutable transition
/// to a dataset adapter. BIDS/NWB use this seam to add dataset-level semantics
/// without reimplementing signal, calibration, sidecar, and provenance rules.
pub(crate) fn recording_builder_from_signal_bundle_with_options(
    bundle: SignalBundle,
    options: RecordingAdapterOptions,
    mut additional_extensions: Vec<Property>,
) -> LmlResult<RecordingBuilder> {
    bundle.validate()?;
    validate_adapter_input(&bundle, &options)?;
    let (sample_rate, rate_was_approximated) = rationalize_rate(bundle.sample_rate)?;
    let labels = bundle
        .channels
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let declared_modality = options
        .declared_modality
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    let (overall_tag, _) = infer_modality(&labels, None);
    let overall_modality = declared_modality
        .as_deref()
        .unwrap_or_else(|| modality_name(overall_tag));
    let identity_subject = options
        .subject
        .as_deref()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            (!bundle.metadata.patient_id.is_empty()).then_some(bundle.metadata.patient_id.as_str())
        })
        .unwrap_or("unknown");
    let identity = RecordingIdentity::new(
        identity_subject,
        options.session.as_deref(),
        options.run.as_deref(),
    );
    let mut builder = RecordingBuilder::new(identity);
    builder
        .add_clock(Clock::new(CLOCK_ID, ClockKind::Relative, sample_rate))
        .map_err(graph_error)?;

    let unit = source_unit(&bundle.metadata.phys_dim);
    let mut extensions = source_properties(&bundle, &options);
    extensions.append(&mut additional_extensions);
    let mut streams: BTreeMap<String, Vec<SignalSeries>> = BTreeMap::new();
    for (index, (label, samples)) in bundle.channels.iter().zip(bundle.signal).enumerate() {
        let modality = if declared_modality.is_some() {
            overall_modality
        } else {
            let (tag, _) = infer_modality(&[label.as_str()], None);
            let inferred = modality_name(tag);
            if inferred == "untyped" && overall_modality != "untyped" {
                overall_modality
            } else {
                inferred
            }
        };
        let modality_id = ModalityId::new(modality);
        let channel_id = format!("signal:channel:{index:06}");
        let series = SignalSeries::new(
            ChannelDescriptor::new(&channel_id, label, modality_id.clone(), unit.clone()),
            TimeAxis::uniform(CLOCK_ID, 0, sample_rate),
            SampleBuffer::from_i64(Arc::from(samples)),
        );
        streams.entry(modality.to_owned()).or_default().push(series);
    }
    for (modality, series) in streams {
        let mut stream = SignalStream::new(
            format!("signal:stream:{modality}"),
            ModalityId::new(&modality),
        );
        for item in series {
            stream = stream.with_series(item);
        }
        builder.add_signal_stream(stream).map_err(graph_error)?;
    }

    builder
        .add_table(channel_calibration_table(
            &bundle.channels,
            &bundle.phys_min,
            &bundle.phys_max,
            &bundle.metadata.phys_dim,
        ))
        .map_err(graph_error)?;

    let mut sidecar_ids = Vec::with_capacity(bundle.sidecar.len());
    let mut sidecar_keys = Vec::with_capacity(bundle.sidecar.len());
    let mut sidecar_aux = Vec::with_capacity(bundle.sidecar.len());
    for (index, sidecar) in bundle.sidecar.into_iter().enumerate() {
        let attachment_id = format!("attachment:source:{index:06}");
        let extension_name =
            QualifiedName::new(SOURCE_NAMESPACE, format!("sidecar_ref_{index:06}"));
        extensions.push(Property::new(
            extension_name.clone(),
            Value::text(&attachment_id),
        ));
        sidecar_ids.push(Value::text(&attachment_id));
        sidecar_keys.push(Value::text(&sidecar.key));
        sidecar_aux.push(sidecar.aux.map_or(Value::Null, Value::I64));
        builder
            .add_attachment(Attachment::new(
                &attachment_id,
                sidecar_media_type(&sidecar),
                Arc::from(sidecar.bytes),
            ))
            .map_err(graph_error)?;
        builder
            .add_loss_receipt(LossReceipt::new(
                format!("receipt:source-sidecar:{index:06}"),
                QualifiedName::new(INTEROP_NAMESPACE, "source-sidecar"),
                SemanticDisposition::PreservedAsExtension,
                Some(extension_name),
                "source-format bytes preserved exactly as an indexed ABIR2 attachment",
            ))
            .map_err(graph_error)?;
    }
    if !sidecar_ids.is_empty() {
        builder
            .add_table(
                Table::new(SIDECAR_TABLE_ID)
                    .with_column(TableColumn::new(
                        QualifiedName::new(SOURCE_NAMESPACE, "attachment_id"),
                        ValueType::Text,
                        sidecar_ids.into(),
                    ))
                    .with_column(TableColumn::new(
                        QualifiedName::new(SOURCE_NAMESPACE, "key"),
                        ValueType::Text,
                        sidecar_keys.into(),
                    ))
                    .with_column(TableColumn::new(
                        QualifiedName::new(SOURCE_NAMESPACE, "aux"),
                        ValueType::I64,
                        sidecar_aux.into(),
                    )),
            )
            .map_err(graph_error)?;
    }
    if rate_was_approximated {
        builder
            .add_loss_receipt(LossReceipt::new(
                "receipt:sample-rate-rationalization",
                QualifiedName::new(INTEROP_NAMESPACE, "sample-rate"),
                SemanticDisposition::Approximated,
                Some(QualifiedName::new(SOURCE_NAMESPACE, "sample_rate_f64_bits")),
                "time-axis rational rounded to at most one nanohertz; original f64 bits retained",
            ))
            .map_err(graph_error)?;
    }
    builder.set_extensions(PropertyBag::new(extensions));
    builder
        .add_provenance(ProvenanceActivity::new(
            "provenance:source-adapter",
            QualifiedName::new(INTEROP_NAMESPACE, "signal-bundle-to-abir2"),
            concat!("lamquant-lml/", env!("CARGO_PKG_VERSION")),
        ))
        .map_err(graph_error)?;
    Ok(builder)
}

fn channel_calibration_table(
    channels: &[String],
    phys_min: &[f64],
    phys_max: &[f64],
    unit: &str,
) -> Table {
    let channel_ids = (0..channels.len())
        .map(|index| Value::text(format!("signal:channel:{index:06}")))
        .collect::<Vec<_>>();
    let source_indices = (0..channels.len())
        .map(|index| Value::U64(index as u64))
        .collect::<Vec<_>>();
    let labels = channels.iter().map(Value::text).collect::<Vec<_>>();
    let minima = phys_min
        .iter()
        .copied()
        .map(Value::from)
        .collect::<Vec<_>>();
    let maxima = phys_max
        .iter()
        .copied()
        .map(Value::from)
        .collect::<Vec<_>>();
    let units = channels
        .iter()
        .map(|_| Value::text(unit))
        .collect::<Vec<_>>();
    Table::new(CALIBRATION_TABLE_ID)
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "channel_id"),
            ValueType::Text,
            channel_ids.into(),
        ))
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "source_index"),
            ValueType::U64,
            source_indices.into(),
        ))
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "label"),
            ValueType::Text,
            labels.into(),
        ))
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "phys_min"),
            ValueType::F64,
            minima.into(),
        ))
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "phys_max"),
            ValueType::F64,
            maxima.into(),
        ))
        .with_column(TableColumn::new(
            QualifiedName::new(SOURCE_NAMESPACE, "unit"),
            ValueType::Text,
            units.into(),
        ))
}

fn validate_adapter_input(
    bundle: &SignalBundle,
    options: &RecordingAdapterOptions,
) -> LmlResult<()> {
    if bundle.signal.is_empty() || bundle.signal[0].is_empty() {
        return Err(LmlError::InvalidHeader(
            "SignalBundle: ABIR2 source adapter requires non-empty signal data".into(),
        ));
    }
    for (index, (minimum, maximum)) in bundle
        .phys_min
        .iter()
        .zip(bundle.phys_max.iter())
        .enumerate()
    {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(LmlError::InvalidHeader(format!(
                "SignalBundle: invalid physical range for channel {index}: {minimum}..{maximum}"
            )));
        }
    }
    for (name, value) in [
        ("subject", options.subject.as_deref()),
        ("session", options.session.as_deref()),
        ("run", options.run.as_deref()),
        ("declared_modality", options.declared_modality.as_deref()),
    ] {
        if value.is_some_and(|item| item.trim().is_empty()) {
            return Err(LmlError::InvalidHeader(format!(
                "ABIR2 adapter option {name} must be non-empty when present"
            )));
        }
    }
    Ok(())
}

fn source_properties(bundle: &SignalBundle, options: &RecordingAdapterOptions) -> Vec<Property> {
    let values = [
        ("source_file", Value::text(&bundle.metadata.source_file)),
        ("format", Value::text(&bundle.metadata.format)),
        ("patient_id", Value::text(&bundle.metadata.patient_id)),
        (
            "recording_info",
            Value::text(&bundle.metadata.recording_info),
        ),
        ("startdate", Value::text(&bundle.metadata.startdate)),
        ("phys_dim", Value::text(&bundle.metadata.phys_dim)),
        ("duration_seconds", Value::from(bundle.duration_s)),
        (
            "sample_rate_f64_bits",
            Value::U64(bundle.sample_rate.to_bits()),
        ),
        (
            "declared_modality",
            options
                .declared_modality
                .as_deref()
                .map_or(Value::Null, Value::text),
        ),
    ];
    values
        .into_iter()
        .map(|(name, value)| Property::new(QualifiedName::new(SOURCE_NAMESPACE, name), value))
        .collect()
}

fn rationalize_rate(value: f64) -> LmlResult<(Rational, bool)> {
    const DENOMINATORS: [u64; 7] = [1, 10, 100, 1_000, 10_000, 1_000_000, 1_000_000_000];
    for denominator in DENOMINATORS {
        let scaled = value * denominator as f64;
        if scaled.is_finite() && scaled >= 1.0 && scaled <= u64::MAX as f64 {
            let numerator = scaled.round() as u64;
            if numerator as f64 / denominator as f64 == value {
                return Ok((Rational::new(numerator, denominator).unwrap(), false));
            }
        }
    }
    let denominator = 1_000_000_000_u64;
    let scaled = value * denominator as f64;
    if !scaled.is_finite() || scaled < 0.5 || scaled > u64::MAX as f64 {
        return Err(LmlError::InvalidHeader(format!(
            "SignalBundle: sample_rate {value} cannot be represented by ABIR2 Rational"
        )));
    }
    let numerator = scaled.round() as u64;
    Ok((Rational::new(numerator, denominator).unwrap(), true))
}

fn modality_name(tag: u8) -> &'static str {
    name_for_tag(tag).unwrap_or("untyped")
}

fn source_unit(value: &str) -> Unit {
    if value.is_empty() {
        Unit::new("source", "unspecified")
    } else {
        Unit::ucum(value)
    }
}

fn sidecar_media_type(sidecar: &SidecarBlob) -> &'static str {
    match sidecar.key.as_str() {
        "edf_meta" | "nwb_slots" | "bcs1_metadata_json" => "application/json",
        "raw_header" | "trailing_data" | "non_eeg_chunk" | "bcs1_header" => {
            "application/octet-stream"
        }
        "dicom_raw" => "application/dicom",
        "nwb_skeleton" => "application/x-hdf5",
        _ => "application/octet-stream",
    }
}

fn graph_error(error: impl std::fmt::Display) -> LmlError {
    LmlError::InvalidHeader(format!("ABIR2 source adapter: {error}"))
}
