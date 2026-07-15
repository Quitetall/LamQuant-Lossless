//! Atomic reconstruction of native NWB/HDF5 from an ABIR2 graph.

use std::path::Path;

use abir::{Recording, SignalSeries, SignalStream};
use hdf5_metno::File;

use crate::error::{LmlError, LmlResult};

use super::super::{h5, write_flat_i64};
use super::buffers::{sample_values_f32, sample_values_f64, sample_values_i64, tensor_values_i64};
use super::{checked_shape_len, NwbGraphSlot};

const MAX_GRAPH_SLOT_BYTES: usize = 64 * 1024 * 1024;
const MAX_GRAPH_SLOTS: usize = 1_000_000;

/// Reconstruct a native HDF5/NWB file from a recording produced by
/// [`read_recording`]. The output is data-identical for every supported numeric
/// dataset and retains the skeleton's groups, attributes, references, and
/// unprojected datasets. HDF5 physical byte layout is intentionally not part of
/// the contract.
pub fn write_recording(recording: &Recording, output: &Path) -> LmlResult<()> {
    let skeleton = recording
        .attachments()
        .iter()
        .find(|attachment| attachment.id() == "attachment:nwb:skeleton-zstd")
        .ok_or_else(|| LmlError::InvalidHeader("NWB recording lacks skeleton attachment".into()))?;
    let slot_map = recording
        .attachments()
        .iter()
        .find(|attachment| attachment.id() == "attachment:nwb:graph-slots")
        .ok_or_else(|| {
            LmlError::InvalidHeader("NWB recording lacks graph-slot attachment".into())
        })?;
    if slot_map.bytes().len() > MAX_GRAPH_SLOT_BYTES {
        return Err(LmlError::InvalidHeader(format!(
            "NWB graph-slot attachment is {} bytes; maximum is {MAX_GRAPH_SLOT_BYTES}",
            slot_map.bytes().len()
        )));
    }
    let slots: Vec<NwbGraphSlot> = serde_json::from_slice(slot_map.bytes())
        .map_err(|error| LmlError::InvalidHeader(format!("NWB graph slots decode: {error}")))?;
    if slots.len() > MAX_GRAPH_SLOTS {
        return Err(LmlError::InvalidHeader(format!(
            "NWB graph-slot count {} exceeds maximum {MAX_GRAPH_SLOTS}",
            slots.len()
        )));
    }
    if output.exists() {
        return Err(LmlError::InvalidHeader(format!(
            "NWB reconstruction destination '{}' already exists",
            output.display()
        )));
    }

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staging = tempfile::Builder::new()
        .prefix(".nwb-abir2-")
        .suffix(".partial")
        .tempfile_in(parent)
        .map_err(LmlError::Io)?;
    zstd::stream::copy_decode(skeleton.bytes(), staging.as_file_mut())
        .map_err(|error| LmlError::InvalidHeader(format!("NWB skeleton zstd decode: {error}")))?;
    staging.as_file().sync_all().map_err(LmlError::Io)?;

    {
        let file = h5(
            File::open_rw(staging.path()),
            "open_rw ABIR2 reconstruction",
        )?;
        for slot in slots {
            let expected = slot_element_count(&slot)?;
            let dataset = h5(file.dataset(&slot.h5_path), "reconstruction dataset")?;
            match (slot.numeric_kind.as_str(), slot.int_bytes) {
                ("integer", _) => {
                    let flat = integer_slot_values(recording, &slot)?;
                    validate_slot_count(&slot, flat.len(), expected)?;
                    write_flat_i64(&dataset, slot.int_bytes, slot.signed, &flat)?;
                }
                ("float", 4) => {
                    let channels = float_slot_channels(recording, &slot, sample_values_f32)?;
                    let flat = flatten_numeric(&channels, &slot.orig_shape, slot.time_major);
                    validate_slot_count(&slot, flat.len(), expected)?;
                    h5(dataset.write_raw(&flat), "write reconstructed f32 dataset")?;
                }
                ("float", 8) => {
                    let channels = float_slot_channels(recording, &slot, sample_values_f64)?;
                    let flat = flatten_numeric(&channels, &slot.orig_shape, slot.time_major);
                    validate_slot_count(&slot, flat.len(), expected)?;
                    h5(dataset.write_raw(&flat), "write reconstructed f64 dataset")?;
                }
                (kind, width) => {
                    return Err(LmlError::InvalidHeader(format!(
                        "NWB graph slot '{}' has unsupported numeric kind/width '{kind}/{width}'",
                        slot.h5_path
                    )))
                }
            }
        }
    }
    staging.as_file().sync_all().map_err(LmlError::Io)?;
    staging
        .persist_noclobber(output)
        .map_err(|error| LmlError::Io(error.error))?;
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(LmlError::Io)?;
    Ok(())
}

fn integer_slot_values(recording: &Recording, slot: &NwbGraphSlot) -> LmlResult<Vec<i64>> {
    match slot.representation.as_str() {
        "signal" => {
            let channels = slot
                .node_ids
                .iter()
                .map(|node_id| {
                    find_series(recording, slot, node_id)
                        .and_then(|series| sample_values_i64(series.samples()))
                })
                .collect::<LmlResult<Vec<_>>>()?;
            Ok(flatten_numeric(
                &channels,
                &slot.orig_shape,
                slot.time_major,
            ))
        }
        "tensor" => {
            if slot.node_ids.len() != 1 {
                return Err(LmlError::InvalidHeader(format!(
                    "NWB tensor slot '{}' must reference exactly one node",
                    slot.h5_path
                )));
            }
            let node_id = &slot.node_ids[0];
            let tensor = recording
                .tensors()
                .iter()
                .find(|tensor| tensor.id() == node_id)
                .ok_or_else(|| {
                    LmlError::InvalidHeader(format!(
                        "NWB graph slot '{}' references missing tensor '{node_id}'",
                        slot.h5_path
                    ))
                })?;
            tensor_values_i64(tensor.buffer())
        }
        other => Err(LmlError::InvalidHeader(format!(
            "NWB graph slot '{}' has unknown representation '{other}'",
            slot.h5_path
        ))),
    }
}

fn float_slot_channels<T>(
    recording: &Recording,
    slot: &NwbGraphSlot,
    convert: impl Fn(&abir::SampleBuffer) -> LmlResult<Vec<T>>,
) -> LmlResult<Vec<Vec<T>>> {
    if slot.representation != "signal" {
        return Err(LmlError::InvalidHeader(format!(
            "NWB float slot '{}' must reference signal nodes",
            slot.h5_path
        )));
    }
    slot.node_ids
        .iter()
        .map(|node_id| {
            find_series(recording, slot, node_id).and_then(|series| convert(series.samples()))
        })
        .collect()
}

fn find_series<'a>(
    recording: &'a Recording,
    slot: &NwbGraphSlot,
    node_id: &str,
) -> LmlResult<&'a SignalSeries> {
    recording
        .signal_streams()
        .iter()
        .flat_map(SignalStream::series)
        .find(|series| series.channel().id() == node_id)
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "NWB graph slot '{}' references missing channel '{node_id}'",
                slot.h5_path
            ))
        })
}

fn flatten_numeric<T: Copy>(channels: &[Vec<T>], shape: &[usize], time_major: bool) -> Vec<T> {
    let mut flat = Vec::new();
    if time_major {
        let (time_count, channel_count) = (shape[0], shape[1]);
        for time in 0..time_count {
            for channel in channels.iter().take(channel_count) {
                if let Some(&value) = channel.get(time) {
                    flat.push(value);
                }
            }
        }
    } else if let Some(channel) = channels.first() {
        flat.extend_from_slice(channel);
    }
    flat
}

fn slot_element_count(slot: &NwbGraphSlot) -> LmlResult<usize> {
    checked_shape_len(&slot.orig_shape, &slot.h5_path)
}

fn validate_slot_count(slot: &NwbGraphSlot, actual: usize, expected: usize) -> LmlResult<()> {
    if actual != expected {
        return Err(LmlError::InvalidHeader(format!(
            "NWB graph slot '{}' has {actual} values; expected {expected}",
            slot.h5_path
        )));
    }
    Ok(())
}
