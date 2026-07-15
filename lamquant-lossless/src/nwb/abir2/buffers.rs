//! Exact native-width conversion at the NWB graph boundary.

use std::sync::Arc;

use abir::{SampleBuffer, TensorBuffer};

use crate::error::{LmlError, LmlResult};

use super::super::H5IntSignal;

pub(super) enum FloatSignalData {
    F32(Vec<Vec<f32>>),
    F64(Vec<Vec<f64>>),
}

impl FloatSignalData {
    pub(super) fn channel_count(&self) -> usize {
        match self {
            Self::F32(channels) => channels.len(),
            Self::F64(channels) => channels.len(),
        }
    }

    pub(super) fn sample_count(&self) -> usize {
        match self {
            Self::F32(channels) => channels.first().map_or(0, Vec::len),
            Self::F64(channels) => channels.first().map_or(0, Vec::len),
        }
    }

    pub(super) fn sample_buffer(&self, channel: usize) -> SampleBuffer {
        match self {
            Self::F32(channels) => SampleBuffer::from_f32(channels[channel].clone().into()),
            Self::F64(channels) => SampleBuffer::from_f64(channels[channel].clone().into()),
        }
    }
}

pub(super) fn sample_buffer(signal: &H5IntSignal, values: &[i64]) -> SampleBuffer {
    match (signal.int_bytes, signal.signed) {
        (1, true) => SampleBuffer::from_i8(
            values
                .iter()
                .map(|&value| value as i8)
                .collect::<Vec<_>>()
                .into(),
        ),
        (1, false) => SampleBuffer::from_u8(
            values
                .iter()
                .map(|&value| value as u8)
                .collect::<Vec<_>>()
                .into(),
        ),
        (2, true) => SampleBuffer::from_i16(
            values
                .iter()
                .map(|&value| value as i16)
                .collect::<Vec<_>>()
                .into(),
        ),
        (2, false) => SampleBuffer::from_u16(
            values
                .iter()
                .map(|&value| value as u16)
                .collect::<Vec<_>>()
                .into(),
        ),
        (4, true) => SampleBuffer::from_i32(
            values
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
                .into(),
        ),
        (4, false) => SampleBuffer::from_u32(
            values
                .iter()
                .map(|&value| value as u32)
                .collect::<Vec<_>>()
                .into(),
        ),
        _ => SampleBuffer::from_i64(Arc::from(values)),
    }
}

pub(super) fn tensor_buffer(signal: &H5IntSignal, values: &[i64]) -> TensorBuffer {
    match (signal.int_bytes, signal.signed) {
        (1, true) => TensorBuffer::from_i8(
            values
                .iter()
                .map(|&value| value as i8)
                .collect::<Vec<_>>()
                .into(),
        ),
        (1, false) => TensorBuffer::from_u8(
            values
                .iter()
                .map(|&value| value as u8)
                .collect::<Vec<_>>()
                .into(),
        ),
        (2, true) => TensorBuffer::from_i16(
            values
                .iter()
                .map(|&value| value as i16)
                .collect::<Vec<_>>()
                .into(),
        ),
        (2, false) => TensorBuffer::from_u16(
            values
                .iter()
                .map(|&value| value as u16)
                .collect::<Vec<_>>()
                .into(),
        ),
        (4, true) => TensorBuffer::from_i32(
            values
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
                .into(),
        ),
        (4, false) => TensorBuffer::from_u32(
            values
                .iter()
                .map(|&value| value as u32)
                .collect::<Vec<_>>()
                .into(),
        ),
        _ => TensorBuffer::from_i64(Arc::from(values)),
    }
}

pub(super) fn sample_values_i64(buffer: &SampleBuffer) -> LmlResult<Vec<i64>> {
    match buffer {
        SampleBuffer::I8(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        SampleBuffer::U8(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        SampleBuffer::I16(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        SampleBuffer::U16(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        SampleBuffer::I32(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        SampleBuffer::U32(values) => Ok(values.iter().map(|&value| i64::from(value)).collect()),
        SampleBuffer::I64(values) => Ok(values.to_vec()),
        SampleBuffer::F32(_) | SampleBuffer::F64(_) => Err(LmlError::InvalidHeader(
            "NWB integer graph slot unexpectedly references floating samples".into(),
        )),
    }
}

pub(super) fn tensor_values_i64(buffer: &TensorBuffer) -> LmlResult<Vec<i64>> {
    match buffer {
        TensorBuffer::I8(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        TensorBuffer::U8(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        TensorBuffer::I16(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        TensorBuffer::U16(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        TensorBuffer::I32(values) => Ok(values.iter().map(|&value| value.into()).collect()),
        TensorBuffer::U32(values) => Ok(values.iter().map(|&value| i64::from(value)).collect()),
        TensorBuffer::I64(values) => Ok(values.to_vec()),
        TensorBuffer::F32(_) | TensorBuffer::F64(_) => Err(LmlError::InvalidHeader(
            "NWB integer graph slot unexpectedly references floating tensor values".into(),
        )),
    }
}

pub(super) fn sample_values_f32(buffer: &SampleBuffer) -> LmlResult<Vec<f32>> {
    match buffer {
        SampleBuffer::F32(values) => Ok(values.to_vec()),
        _ => Err(LmlError::InvalidHeader(
            "NWB f32 graph slot references a non-f32 sample buffer".into(),
        )),
    }
}

pub(super) fn sample_values_f64(buffer: &SampleBuffer) -> LmlResult<Vec<f64>> {
    match buffer {
        SampleBuffer::F64(values) => Ok(values.to_vec()),
        _ => Err(LmlError::InvalidHeader(
            "NWB f64 graph slot references a non-f64 sample buffer".into(),
        )),
    }
}
