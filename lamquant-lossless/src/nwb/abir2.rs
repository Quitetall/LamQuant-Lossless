//! Immutable ABIR2 projection for native NWB/HDF5 recordings.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use abir::{
    Attachment, ChannelDescriptor, Clock, ClockKind, LossReceipt, ModalityId, Property,
    PropertyBag, ProvenanceActivity, QualifiedName, Rational, Recording, RecordingBuilder,
    RecordingIdentity, SemanticDisposition, SignalSeries, SignalStream, Table, TableColumn, Tensor,
    Value, ValueType,
};
use hdf5_metno::types::{FloatSize, TypeDescriptor};
use hdf5_metno::{Dataset, File};
use serde::{Deserialize, Serialize};

use crate::error::{LmlError, LmlResult};
use crate::source::bundle::source_basename;

use super::{
    build_zeroed_skeleton, collect_datasets, flatten_slot, h5, read_int_signals_from_datasets,
};

mod buffers;
mod reconstruct;
mod timing;

pub use reconstruct::write_recording;

use buffers::{sample_buffer, tensor_buffer, FloatSignalData};
use timing::{
    data_unit, infer_nwb_modality, parent_path, read_first_string_dataset, read_timing, NwbTiming,
    CLOCK_ID as NWB_CLOCK_ID, CLOCK_TICKS_PER_SECOND as NWB_CLOCK_TICKS_PER_SECOND,
};

/// Read every integer dataset plus timed f32/f64 TimeSeries into an immutable
/// ABIR2 graph. Integer datasets that are not biosignals remain exact typed
/// tensors, so no HDF5 numeric lane disappears merely because it is metadata
/// rather than a waveform.
pub fn read_recording(path: &Path) -> LmlResult<Recording> {
    let file = h5(File::open(path), "open ABIR2 source")?;
    let mut datasets = Vec::new();
    collect_datasets(&file, &mut datasets)?;
    let signals = read_int_signals_from_datasets(&datasets)?;
    let float_signals = read_float_timeseries(&file, &datasets)?;
    if signals.is_empty() && float_signals.is_empty() {
        return Err(LmlError::InvalidHeader(
            "NWB contains no timed numeric datasets representable by ABIR2".into(),
        ));
    }

    let subject = read_first_string_dataset(&file, &["/general/subject/subject_id"])
        .unwrap_or_else(|| "unknown".into());
    let session = read_first_string_dataset(&file, &["/session_id"]);
    let mut builder =
        RecordingBuilder::new(RecordingIdentity::new(&subject, session.as_deref(), None));
    builder
        .add_clock(Clock::new(
            NWB_CLOCK_ID,
            ClockKind::Relative,
            Rational::new(NWB_CLOCK_TICKS_PER_SECOND, 1).unwrap(),
        ))
        .map_err(graph_error)?;

    let mut streams: BTreeMap<String, Vec<SignalSeries>> = BTreeMap::new();
    let dataset_count = signals.len() + float_signals.len();
    let mut slots = Vec::with_capacity(dataset_count);
    let mut paths = Vec::with_capacity(dataset_count);
    let mut numeric_kinds = Vec::with_capacity(dataset_count);
    let mut representations = Vec::with_capacity(dataset_count);
    let mut widths = Vec::with_capacity(dataset_count);
    let mut signedness = Vec::with_capacity(dataset_count);
    let mut shapes = Vec::with_capacity(dataset_count);
    let mut timed_dataset_count = 0_usize;

    for (dataset_index, signal) in signals.iter().enumerate() {
        let timing = read_timing(&file, &signal.h5_path)?;
        let mut node_ids = Vec::new();
        let representation;
        if let Some(timing) = timing {
            timed_dataset_count += 1;
            representation = "signal";
            let modality = infer_nwb_modality(&signal.h5_path);
            let unit = data_unit(&file, &signal.h5_path);
            let group_label = parent_path(&signal.h5_path)
                .rsplit('/')
                .next()
                .filter(|value| !value.is_empty())
                .unwrap_or("TimeSeries");
            let (time_axis, approximated) = timing.to_axis(signal.signal[0].len())?;
            if approximated {
                builder
                    .add_loss_receipt(LossReceipt::new(
                        format!("receipt:nwb-time:{dataset_index:06}"),
                        QualifiedName::new("nwb", "time-axis"),
                        SemanticDisposition::Approximated,
                        None,
                        "NWB seconds or rate rounded to bounded ABIR2 rational/nanosecond coordinates; source values remain in the skeleton",
                    ))
                    .map_err(graph_error)?;
            }
            for (channel_index, samples) in signal.signal.iter().enumerate() {
                let channel_id =
                    format!("signal:nwb:{dataset_index:06}:channel:{channel_index:06}");
                node_ids.push(channel_id.clone());
                streams
                    .entry(modality.clone())
                    .or_default()
                    .push(SignalSeries::new(
                        ChannelDescriptor::new(
                            &channel_id,
                            format!("{group_label}[{channel_index}]"),
                            ModalityId::new(&modality),
                            unit.clone(),
                        ),
                        time_axis.clone(),
                        sample_buffer(signal, samples),
                    ));
            }
        } else {
            representation = "tensor";
            let tensor_id = format!("tensor:nwb:{dataset_index:06}");
            node_ids.push(tensor_id.clone());
            checked_shape_len(&signal.orig_shape, &signal.h5_path)?;
            let flat = flatten_slot(&signal.signal, &signal.orig_shape, signal.time_major);
            builder
                .add_tensor(Tensor::new(
                    &tensor_id,
                    signal
                        .orig_shape
                        .iter()
                        .map(|&value| value as u64)
                        .collect::<Vec<_>>()
                        .into(),
                    tensor_buffer(signal, &flat),
                ))
                .map_err(graph_error)?;
        }
        slots.push(NwbGraphSlot {
            h5_path: signal.h5_path.clone(),
            numeric_kind: "integer".into(),
            int_bytes: signal.int_bytes,
            signed: signal.signed,
            orig_shape: signal.orig_shape.clone(),
            time_major: signal.time_major,
            representation: representation.into(),
            node_ids: node_ids.clone(),
        });
        paths.push(Value::text(&signal.h5_path));
        numeric_kinds.push(Value::text("integer"));
        representations.push(Value::text(representation));
        widths.push(Value::U64(signal.int_bytes.into()));
        signedness.push(Value::text(if signal.signed {
            "signed"
        } else {
            "unsigned"
        }));
        shapes.push(Value::text(
            serde_json::to_string(&signal.orig_shape).unwrap_or_else(|_| "[]".into()),
        ));
    }

    for (float_index, signal) in float_signals.iter().enumerate() {
        let dataset_index = signals.len() + float_index;
        timed_dataset_count += 1;
        let modality = infer_nwb_modality(&signal.h5_path);
        let unit = data_unit(&file, &signal.h5_path);
        let group_label = parent_path(&signal.h5_path)
            .rsplit('/')
            .next()
            .filter(|value| !value.is_empty())
            .unwrap_or("TimeSeries");
        let (time_axis, approximated) = signal.timing.to_axis(signal.data.sample_count())?;
        if approximated {
            builder
                .add_loss_receipt(LossReceipt::new(
                    format!("receipt:nwb-time:{dataset_index:06}"),
                    QualifiedName::new("nwb", "time-axis"),
                    SemanticDisposition::Approximated,
                    None,
                    "NWB seconds or rate rounded to bounded ABIR2 rational/nanosecond coordinates; source values remain in the skeleton",
                ))
                .map_err(graph_error)?;
        }
        let mut node_ids = Vec::with_capacity(signal.data.channel_count());
        for channel_index in 0..signal.data.channel_count() {
            let channel_id = format!("signal:nwb:{dataset_index:06}:channel:{channel_index:06}");
            node_ids.push(channel_id.clone());
            streams
                .entry(modality.clone())
                .or_default()
                .push(SignalSeries::new(
                    ChannelDescriptor::new(
                        &channel_id,
                        format!("{group_label}[{channel_index}]"),
                        ModalityId::new(&modality),
                        unit.clone(),
                    ),
                    time_axis.clone(),
                    signal.data.sample_buffer(channel_index),
                ));
        }
        slots.push(NwbGraphSlot {
            h5_path: signal.h5_path.clone(),
            numeric_kind: "float".into(),
            int_bytes: signal.float_bytes,
            signed: true,
            orig_shape: signal.orig_shape.clone(),
            time_major: signal.time_major,
            representation: "signal".into(),
            node_ids,
        });
        paths.push(Value::text(&signal.h5_path));
        numeric_kinds.push(Value::text("float"));
        representations.push(Value::text("signal"));
        widths.push(Value::U64(signal.float_bytes.into()));
        signedness.push(Value::text("not-applicable"));
        shapes.push(Value::text(
            serde_json::to_string(&signal.orig_shape).unwrap_or_else(|_| "[]".into()),
        ));
    }

    if timed_dataset_count == 0 {
        return Err(LmlError::InvalidHeader(
            "NWB contains numeric datasets but no timed numeric TimeSeries data".into(),
        ));
    }
    for (modality, series) in streams {
        let mut stream = SignalStream::new(
            format!("signal:stream:nwb:{modality}"),
            ModalityId::new(&modality),
        );
        for item in series {
            stream = stream.with_series(item);
        }
        builder.add_signal_stream(stream).map_err(graph_error)?;
    }
    builder
        .add_table(
            Table::new("table:nwb-numeric-datasets")
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "h5_path"),
                    ValueType::Text,
                    paths.into(),
                ))
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "numeric_kind"),
                    ValueType::Text,
                    numeric_kinds.into(),
                ))
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "representation"),
                    ValueType::Text,
                    representations.into(),
                ))
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "storage_bytes"),
                    ValueType::U64,
                    widths.into(),
                ))
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "signedness"),
                    ValueType::Text,
                    signedness.into(),
                ))
                .with_column(TableColumn::new(
                    QualifiedName::new("nwb", "orig_shape"),
                    ValueType::Text,
                    shapes.into(),
                )),
        )
        .map_err(graph_error)?;

    let float_skeleton_slots = float_signals
        .iter()
        .map(|signal| {
            Ok((
                signal.h5_path.clone(),
                signal.float_bytes,
                checked_shape_len(&signal.orig_shape, &signal.h5_path)?,
            ))
        })
        .collect::<LmlResult<Vec<_>>>()?;
    let skeleton = build_zeroed_skeleton(path, &signals, &float_skeleton_slots)?;
    let compressed_skeleton = zstd::stream::encode_all(skeleton.as_slice(), 9)
        .map_err(|error| LmlError::InvalidHeader(format!("NWB skeleton zstd: {error}")))?;
    let graph_slots = serde_json::to_vec(&slots)
        .map_err(|error| LmlError::InvalidHeader(format!("NWB graph slots encode: {error}")))?;
    builder
        .add_attachment(Attachment::new(
            "attachment:nwb:skeleton-zstd",
            "application/x-hdf5+zstd",
            Arc::from(compressed_skeleton),
        ))
        .map_err(graph_error)?;
    builder
        .add_attachment(Attachment::new(
            "attachment:nwb:graph-slots",
            "application/json",
            Arc::from(graph_slots),
        ))
        .map_err(graph_error)?;
    for (id, label, details) in [
        (
            "receipt:nwb:skeleton",
            "hdf5-skeleton",
            "all non-extracted datasets, attributes, groups, and references remain in a native HDF5 skeleton; original container byte layout is not claimed",
        ),
        (
            "receipt:nwb:graph-slots",
            "numeric-dataset-map",
            "every zeroed numeric dataset has an exact typed graph node and reconstruction map",
        ),
    ] {
        builder
            .add_loss_receipt(LossReceipt::new(
                id,
                QualifiedName::new("nwb", label),
                SemanticDisposition::Exact,
                None,
                details,
            ))
            .map_err(graph_error)?;
    }
    builder.set_extensions(PropertyBag::new(vec![
        Property::new(
            QualifiedName::new("lamquant.source", "source_file"),
            Value::text(source_basename(path)),
        ),
        Property::new(
            QualifiedName::new("lamquant.source", "format"),
            Value::text("NWB"),
        ),
    ]));
    builder
        .add_provenance(
            ProvenanceActivity::new(
                "provenance:nwb-adapter",
                QualifiedName::new("nwb", "hdf5-to-abir2"),
                concat!("lamquant-lml/", env!("CARGO_PKG_VERSION")),
            )
            .with_input("attachment:nwb:skeleton-zstd")
            .with_input("attachment:nwb:graph-slots"),
        )
        .map_err(graph_error)?;
    builder.freeze().map_err(graph_error)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct NwbGraphSlot {
    h5_path: String,
    numeric_kind: String,
    int_bytes: u8,
    signed: bool,
    orig_shape: Vec<usize>,
    time_major: bool,
    representation: String,
    node_ids: Vec<String>,
}

pub(super) fn checked_shape_len(shape: &[usize], h5_path: &str) -> LmlResult<usize> {
    shape
        .iter()
        .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "NWB graph slot '{}' shape overflows usize",
                h5_path
            ))
        })
}

struct H5FloatTimeSeries {
    h5_path: String,
    data: FloatSignalData,
    float_bytes: u8,
    time_major: bool,
    orig_shape: Vec<usize>,
    timing: NwbTiming,
}

fn read_float_timeseries(file: &File, datasets: &[Dataset]) -> LmlResult<Vec<H5FloatTimeSeries>> {
    let mut signals = Vec::new();
    for dataset in datasets {
        let h5_path = dataset.name();
        if !h5_path.ends_with("/data") {
            continue;
        }
        let descriptor = h5(
            h5(dataset.dtype(), "float dtype")?.to_descriptor(),
            "float descriptor",
        )?;
        let float_bytes = match descriptor {
            TypeDescriptor::Float(FloatSize::U4) => 4,
            TypeDescriptor::Float(FloatSize::U8) => 8,
            _ => continue,
        };
        let Some(timing) = read_timing(file, &h5_path)? else {
            continue;
        };
        let shape = dataset.shape();
        if shape.is_empty() || shape.len() > 2 || shape.contains(&0) {
            return Err(LmlError::InvalidHeader(format!(
                "NWB float TimeSeries data '{}' must be non-empty and one- or two-dimensional",
                h5_path
            )));
        }
        let (data, time_major) = match float_bytes {
            4 => {
                let (channels, time_major) = read_float_channels::<f32>(dataset, &shape)?;
                (FloatSignalData::F32(channels), time_major)
            }
            8 => {
                let (channels, time_major) = read_float_channels::<f64>(dataset, &shape)?;
                (FloatSignalData::F64(channels), time_major)
            }
            _ => unreachable!(),
        };
        signals.push(H5FloatTimeSeries {
            h5_path,
            data,
            float_bytes,
            time_major,
            orig_shape: shape,
            timing,
        });
    }
    Ok(signals)
}

fn read_float_channels<T>(dataset: &Dataset, shape: &[usize]) -> LmlResult<(Vec<Vec<T>>, bool)>
where
    T: hdf5_metno::H5Type + Copy,
{
    if shape.len() == 1 {
        let values = h5(dataset.read_1d::<T>(), "read float TimeSeries")?;
        Ok((vec![values.to_vec()], false))
    } else {
        let values = h5(dataset.read_2d::<T>(), "read float TimeSeries")?;
        let (time_count, channel_count) = (shape[0], shape[1]);
        let mut channels = (0..channel_count)
            .map(|_| Vec::with_capacity(time_count))
            .collect::<Vec<_>>();
        for time in 0..time_count {
            for (channel, samples) in channels.iter_mut().enumerate() {
                samples.push(values[[time, channel]]);
            }
        }
        Ok((channels, true))
    }
}

fn graph_error(error: impl std::fmt::Display) -> LmlError {
    LmlError::InvalidHeader(format!("NWB ABIR2 adapter: {error}"))
}
