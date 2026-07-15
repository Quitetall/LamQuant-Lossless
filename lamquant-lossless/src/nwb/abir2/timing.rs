//! NWB timing, identity, unit, and modality interpretation.

use abir::{Rational, TimeAxis, Unit};
use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
use hdf5_metno::File;

use crate::error::{LmlError, LmlResult};

use super::super::h5;

pub(super) const CLOCK_ID: &str = "clock:nwb-relative";
pub(super) const CLOCK_TICKS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Clone, Debug)]
pub(super) enum NwbTiming {
    Uniform { starting_time: f64, rate: f64 },
    Explicit(Vec<f64>),
}

impl NwbTiming {
    pub(super) fn to_axis(&self, sample_count: usize) -> LmlResult<(TimeAxis, bool)> {
        match self {
            Self::Uniform {
                starting_time,
                rate,
            } => {
                let (start_tick, start_approximated) = seconds_to_nwb_tick(*starting_time)?;
                let (rate, rate_approximated) = rationalize_nwb_rate(*rate)?;
                Ok((
                    TimeAxis::uniform(CLOCK_ID, start_tick, rate),
                    start_approximated || rate_approximated,
                ))
            }
            Self::Explicit(timestamps) => {
                if timestamps.len() != sample_count {
                    return Err(LmlError::InvalidHeader(format!(
                        "NWB timestamp count {} does not match sample count {sample_count}",
                        timestamps.len()
                    )));
                }
                let mut approximated = false;
                let ticks = timestamps
                    .iter()
                    .map(|&timestamp| {
                        let (tick, rounded) = seconds_to_nwb_tick(timestamp)?;
                        approximated |= rounded;
                        Ok(tick)
                    })
                    .collect::<LmlResult<Vec<_>>>()?;
                Ok((TimeAxis::explicit(CLOCK_ID, ticks.into()), approximated))
            }
        }
    }
}

pub(super) fn read_timing(file: &File, h5_path: &str) -> LmlResult<Option<NwbTiming>> {
    if !h5_path.ends_with("/data") {
        return Ok(None);
    }
    let parent = parent_path(h5_path);
    let group = h5(file.group(parent), "TimeSeries parent")?;
    let timestamps = group.dataset("timestamps").ok();
    let starting_time = group.dataset("starting_time").ok();
    if timestamps.is_some() && starting_time.is_some() {
        return Err(LmlError::InvalidHeader(format!(
            "NWB TimeSeries '{}' declares both timestamps and starting_time",
            parent
        )));
    }
    if let Some(timestamps) = timestamps {
        if timestamps.ndim() != 1 {
            return Err(LmlError::InvalidHeader(format!(
                "NWB timestamps for '{}' must be one-dimensional",
                parent
            )));
        }
        let values = h5(timestamps.read_1d::<f64>(), "read timestamps")?;
        let values = values.to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(LmlError::InvalidHeader(format!(
                "NWB timestamps for '{}' contain non-finite values",
                parent
            )));
        }
        return Ok(Some(NwbTiming::Explicit(values)));
    }
    if let Some(starting_time) = starting_time {
        let value = h5(starting_time.read_scalar::<f64>(), "read starting_time")?;
        let rate = h5(
            h5(starting_time.attr("rate"), "starting_time rate attribute")?.read_scalar::<f64>(),
            "read starting_time rate",
        )?;
        if !value.is_finite() || !rate.is_finite() || rate <= 0.0 {
            return Err(LmlError::InvalidHeader(format!(
                "NWB starting_time/rate for '{}' must be finite with rate > 0",
                parent
            )));
        }
        return Ok(Some(NwbTiming::Uniform {
            starting_time: value,
            rate,
        }));
    }
    if h5_path.starts_with("/acquisition/")
        || h5_path.starts_with("/processing/")
        || h5_path.starts_with("/stimulus/")
    {
        return Err(LmlError::InvalidHeader(format!(
            "NWB TimeSeries '{}' is missing timestamps or starting_time/rate",
            parent
        )));
    }
    Ok(None)
}

pub(super) fn parent_path(path: &str) -> &str {
    path.rsplit_once('/').map_or(
        "/",
        |(parent, _)| {
            if parent.is_empty() {
                "/"
            } else {
                parent
            }
        },
    )
}

pub(super) fn read_first_string_dataset(file: &File, paths: &[&str]) -> Option<String> {
    for path in paths {
        let Ok(dataset) = file.dataset(path) else {
            continue;
        };
        if let Ok(value) = dataset.read_scalar::<VarLenUnicode>() {
            let value = value.as_str().trim();
            if !value.is_empty() {
                return Some(value.into());
            }
        }
        if let Ok(value) = dataset.read_scalar::<VarLenAscii>() {
            let value = value.as_str().trim();
            if !value.is_empty() {
                return Some(value.into());
            }
        }
    }
    None
}

pub(super) fn data_unit(file: &File, path: &str) -> Unit {
    let Ok(dataset) = file.dataset(path) else {
        return Unit::new("source", "unspecified");
    };
    if let Ok(attribute) = dataset.attr("unit") {
        if let Ok(value) = attribute.read_scalar::<VarLenUnicode>() {
            return Unit::ucum(value.as_str());
        }
        if let Ok(value) = attribute.read_scalar::<VarLenAscii>() {
            return Unit::ucum(value.as_str());
        }
    }
    Unit::new("source", "unspecified")
}

pub(super) fn infer_nwb_modality(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    if lower.contains("ecg") || lower.contains("ekg") {
        "ecg"
    } else if lower.contains("emg") {
        "emg"
    } else if lower.contains("eog") {
        "eog"
    } else if lower.contains("eeg") || lower.contains("electricalseries") {
        "ieeg"
    } else {
        "electrophysiology"
    }
    .into()
}

fn seconds_to_nwb_tick(seconds: f64) -> LmlResult<(i64, bool)> {
    let scaled = seconds * CLOCK_TICKS_PER_SECOND as f64;
    if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(LmlError::InvalidHeader(format!(
            "NWB timestamp {seconds} cannot be represented as relative nanoseconds"
        )));
    }
    let rounded = scaled.round();
    Ok((rounded as i64, (scaled - rounded).abs() > 1e-6))
}

fn rationalize_nwb_rate(value: f64) -> LmlResult<(Rational, bool)> {
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
            "NWB rate {value} cannot be represented by ABIR2 Rational"
        )));
    }
    Ok((
        Rational::new(scaled.round() as u64, denominator).unwrap(),
        true,
    ))
}
