//! LamQuant Optimum v2: independent learned-lossless biosignal codec.
//!
//! This first vertical slice fixes the public `LMO1` v3 / `BGF1` seam and a
//! native exact raw mode. Learned graph/entropy modes add behind the same packet
//! contract only after actual-byte gates pass (ADR 0116).

use std::fmt;

#[doc(hidden)]
#[path = "universal.rs"]
pub mod fixed_universal_conformance;
pub mod model_pack;

pub const LMO_MAGIC: &[u8; 4] = b"LMO1";
pub const LMO_VERSION: u8 = 3;
pub const LMO_MODE_LOSSLESS: u8 = 0;
pub const LMO_BODY_OPTIMUM_V2: u8 = 3;
pub const LMO_V3_HEADER_LEN: usize = 7;

pub const BGF_MAGIC: &[u8; 4] = b"BGF1";
pub const BGF_VERSION: u8 = 1;
const BGF_HEADER_LEN: usize = 80;
const TILE_ENTRY_LEN: usize = 24;
const TILE_MODE_RAW_I32: u8 = 0;
const TILE_MODE_DELTA_VARINT: u8 = 1;
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES_PER_WINDOW: usize = 32_768;
const MAX_SAMPLE_VALUES: usize = 8_388_608;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeContext {
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub channel_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedWindow {
    pub samples: Vec<Vec<i64>>,
    pub context: EncodeContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimumV2Error {
    InvalidInput(String),
    InvalidPacket(String),
    Unsupported(String),
    Integrity(String),
}

impl fmt::Display for OptimumV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid Optimum-v2 input: {message}"),
            Self::InvalidPacket(message) => write!(f, "invalid Optimum-v2 packet: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported Optimum-v2 feature: {message}"),
            Self::Integrity(message) => write!(f, "Optimum-v2 integrity failure: {message}"),
        }
    }
}

impl std::error::Error for OptimumV2Error {}

#[derive(Debug, Default, Clone, Copy)]
pub struct OptimumV2Codec;

impl OptimumV2Codec {
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        context: &EncodeContext,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let (n_channels, n_samples) = validate_signal(signal, context)?;
        let labels = encode_labels(&context.channel_labels)?;
        let payload_len = n_channels
            .checked_mul(n_samples)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| OptimumV2Error::InvalidInput("raw payload size overflow".into()))?;
        let mut raw_payload = Vec::with_capacity(payload_len);
        for channel in signal {
            for &sample in channel {
                let sample = i32::try_from(sample).map_err(|_| {
                    OptimumV2Error::InvalidInput("raw mode supports signed i32 samples".into())
                })?;
                raw_payload.extend_from_slice(&sample.to_le_bytes());
            }
        }
        let delta_payload = encode_delta_varints(signal);
        let (tile_mode, payload) = if delta_payload.len() < raw_payload.len() {
            (TILE_MODE_DELTA_VARINT, delta_payload)
        } else {
            (TILE_MODE_RAW_I32, raw_payload.clone())
        };
        let decoded_crc = crc32c(&raw_payload);

        let mut directory = Vec::with_capacity(TILE_ENTRY_LEN);
        directory.push(tile_mode);
        directory.push(0);
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes()); // first_sample
        directory.extend_from_slice(
            &u32::try_from(n_samples)
                .map_err(|_| OptimumV2Error::InvalidInput("sample count exceeds u32".into()))?
                .to_le_bytes(),
        );
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| OptimumV2Error::InvalidInput("payload exceeds u32".into()))?
                .to_le_bytes(),
        );
        directory.extend_from_slice(&decoded_crc.to_le_bytes());

        let body_tail_len = labels
            .len()
            .checked_add(directory.len())
            .and_then(|value| value.checked_add(payload.len()))
            .ok_or_else(|| OptimumV2Error::InvalidInput("body size overflow".into()))?;
        let mut body_tail = Vec::with_capacity(body_tail_len);
        body_tail.extend_from_slice(&labels);
        body_tail.extend_from_slice(&directory);
        body_tail.extend_from_slice(&payload);
        let mut out = Vec::with_capacity(LMO_V3_HEADER_LEN + BGF_HEADER_LEN + body_tail.len());
        out.extend_from_slice(LMO_MAGIC);
        out.push(LMO_VERSION);
        out.push(LMO_MODE_LOSSLESS);
        out.push(LMO_BODY_OPTIMUM_V2);
        out.extend_from_slice(BGF_MAGIC);
        out.push(BGF_VERSION);
        out.push(0); // flags
        out.push(context.bit_depth);
        out.push(0); // reserved
        out.extend_from_slice(
            &u16::try_from(n_channels)
                .map_err(|_| OptimumV2Error::InvalidInput("channel count exceeds u16".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&1u16.to_le_bytes()); // tile_count
        out.extend_from_slice(
            &u32::try_from(n_samples)
                .map_err(|_| OptimumV2Error::InvalidInput("sample count exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&context.sample_rate_mhz.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // model_id = deterministic/native
        out.extend_from_slice(&[0u8; 32]); // model_sha256
        out.extend_from_slice(
            &u32::try_from(labels.len())
                .map_err(|_| OptimumV2Error::InvalidInput("labels exceed u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&0u32.to_le_bytes()); // graph_len
        out.extend_from_slice(
            &u32::try_from(directory.len())
                .map_err(|_| OptimumV2Error::InvalidInput("directory exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| OptimumV2Error::InvalidInput("payload exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&decoded_crc.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // packet CRC32C, filled below
        debug_assert_eq!(out.len(), LMO_V3_HEADER_LEN + BGF_HEADER_LEN);
        out.extend_from_slice(&body_tail);
        let packet_crc = crc32c(&out);
        let crc_offset = LMO_V3_HEADER_LEN + 76;
        out[crc_offset..crc_offset + 4].copy_from_slice(&packet_crc.to_le_bytes());
        Ok(out)
    }

    pub fn decode_window(&self, bytes: &[u8]) -> Result<DecodedWindow, OptimumV2Error> {
        if bytes.len() < LMO_V3_HEADER_LEN + BGF_HEADER_LEN {
            return Err(OptimumV2Error::InvalidPacket("header is truncated".into()));
        }
        if &bytes[0..4] != LMO_MAGIC {
            return Err(OptimumV2Error::InvalidPacket("LMO1 magic mismatch".into()));
        }
        if bytes[4] != LMO_VERSION {
            return Err(OptimumV2Error::Unsupported(format!(
                "LMO1 version {} is not v3",
                bytes[4]
            )));
        }
        if bytes[5] != LMO_MODE_LOSSLESS || bytes[6] != LMO_BODY_OPTIMUM_V2 {
            return Err(OptimumV2Error::Unsupported(
                "mode/body is not Optimum-v2 lossless".into(),
            ));
        }
        let header = LMO_V3_HEADER_LEN;
        if &bytes[header..header + 4] != BGF_MAGIC || bytes[header + 4] != BGF_VERSION {
            return Err(OptimumV2Error::InvalidPacket(
                "BGF1 magic/version mismatch".into(),
            ));
        }
        let packet_crc = read_u32(bytes, header + 76)?;
        if crc32c_zeroed_field(bytes, header + 76) != packet_crc {
            return Err(OptimumV2Error::Integrity(
                "BGF1 packet CRC32C mismatch".into(),
            ));
        }
        let flags = bytes[header + 5];
        let bit_depth = bytes[header + 6];
        let reserved = bytes[header + 7];
        if flags != 0 || reserved != 0 || !(1..=32).contains(&bit_depth) {
            return Err(OptimumV2Error::InvalidPacket(
                "invalid BGF1 flags/bit depth".into(),
            ));
        }
        let n_channels = read_u16(bytes, header + 8)? as usize;
        let tile_count = read_u16(bytes, header + 10)? as usize;
        let n_samples = read_u32(bytes, header + 12)? as usize;
        let sample_rate_mhz = read_u32(bytes, header + 16)?;
        let model_id = read_u32(bytes, header + 20)?;
        let model_hash = &bytes[header + 24..header + 56];
        let labels_len = read_u32(bytes, header + 56)? as usize;
        let graph_len = read_u32(bytes, header + 60)? as usize;
        let directory_len = read_u32(bytes, header + 64)? as usize;
        let payload_len = read_u32(bytes, header + 68)? as usize;
        let decoded_crc = read_u32(bytes, header + 72)?;

        if n_channels == 0
            || n_channels > MAX_CHANNELS
            || n_samples == 0
            || n_samples > MAX_SAMPLES_PER_WINDOW
            || !sample_count_within_bounds(n_channels, n_samples)
            || sample_rate_mhz == 0
        {
            return Err(OptimumV2Error::InvalidPacket(
                "dimensions/rate exceed limits".into(),
            ));
        }
        if model_id != 0 || model_hash.iter().any(|&byte| byte != 0) {
            return Err(OptimumV2Error::Unsupported(
                "unknown normative model".into(),
            ));
        }
        if graph_len != 0 || tile_count != 1 || directory_len != TILE_ENTRY_LEN {
            return Err(OptimumV2Error::Unsupported(
                "graph or tile layout is not implemented".into(),
            ));
        }
        let expected_raw_payload = n_channels
            .checked_mul(n_samples)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| OptimumV2Error::InvalidPacket("payload dimensions overflow".into()))?;
        let symbol_count = n_channels
            .checked_mul(n_samples)
            .ok_or_else(|| OptimumV2Error::InvalidPacket("symbol count overflow".into()))?;
        let tail_start = header + BGF_HEADER_LEN;
        let total_len = tail_start
            .checked_add(labels_len)
            .and_then(|value| value.checked_add(graph_len))
            .and_then(|value| value.checked_add(directory_len))
            .and_then(|value| value.checked_add(payload_len))
            .ok_or_else(|| OptimumV2Error::InvalidPacket("packet length overflow".into()))?;
        if total_len != bytes.len() {
            return Err(OptimumV2Error::InvalidPacket(
                "packet length mismatch".into(),
            ));
        }
        let labels_end = tail_start + labels_len;
        let labels = decode_labels(&bytes[tail_start..labels_end], n_channels)?;
        let directory_start = labels_end + graph_len;
        let directory = &bytes[directory_start..directory_start + directory_len];
        let tile_mode = directory[0];
        if !matches!(tile_mode, TILE_MODE_RAW_I32 | TILE_MODE_DELTA_VARINT)
            || directory[1] != 0
            || read_u16(directory, 2)? != 0
            || read_u32(directory, 4)? != 0
            || read_u32(directory, 8)? as usize != n_samples
            || read_u32(directory, 12)? != 0
            || read_u32(directory, 16)? as usize != payload_len
            || read_u32(directory, 20)? != decoded_crc
        {
            return Err(OptimumV2Error::InvalidPacket(
                "invalid native tile directory".into(),
            ));
        }
        let payload = &bytes[directory_start + directory_len..];
        let samples = match tile_mode {
            TILE_MODE_RAW_I32 => {
                if payload.len() != expected_raw_payload {
                    return Err(OptimumV2Error::InvalidPacket(
                        "raw payload length mismatch".into(),
                    ));
                }
                decode_raw_i32(payload, n_channels, n_samples)
            }
            TILE_MODE_DELTA_VARINT => {
                let max_payload = symbol_count.checked_mul(10).ok_or_else(|| {
                    OptimumV2Error::InvalidPacket("delta payload bound overflow".into())
                })?;
                if payload.len() < symbol_count || payload.len() > max_payload {
                    return Err(OptimumV2Error::InvalidPacket(
                        "delta payload length is outside canonical bounds".into(),
                    ));
                }
                decode_delta_varints(payload, n_channels, n_samples)?
            }
            _ => unreachable!(),
        };
        let min = -(1i64 << (bit_depth - 1));
        let max = (1i64 << (bit_depth - 1)) - 1;
        if samples
            .iter()
            .flatten()
            .any(|&sample| sample < min || sample > max)
        {
            return Err(OptimumV2Error::InvalidPacket(
                "decoded sample exceeds declared bit depth".into(),
            ));
        }
        if crc32c(&canonical_i32_bytes(&samples)?) != decoded_crc {
            return Err(OptimumV2Error::Integrity(
                "decoded-sample CRC mismatch".into(),
            ));
        }
        Ok(DecodedWindow {
            samples,
            context: EncodeContext {
                sample_rate_mhz,
                bit_depth,
                channel_labels: labels,
            },
        })
    }
}

fn canonical_i32_bytes(signal: &[Vec<i64>]) -> Result<Vec<u8>, OptimumV2Error> {
    let count = signal.iter().map(Vec::len).sum::<usize>();
    let mut bytes = Vec::with_capacity(count.saturating_mul(4));
    for channel in signal {
        for &sample in channel {
            let value = i32::try_from(sample)
                .map_err(|_| OptimumV2Error::InvalidPacket("decoded sample exceeds i32".into()))?;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn decode_raw_i32(payload: &[u8], n_channels: usize, n_samples: usize) -> Vec<Vec<i64>> {
    let mut samples = Vec::with_capacity(n_channels);
    let mut offset = 0usize;
    for _ in 0..n_channels {
        let mut channel = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let value = i32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            channel.push(i64::from(value));
            offset += 4;
        }
        samples.push(channel);
    }
    samples
}

fn encode_delta_varints(signal: &[Vec<i64>]) -> Vec<u8> {
    let mut out = Vec::new();
    for channel in signal {
        let mut previous = 0i64;
        for (index, &sample) in channel.iter().enumerate() {
            let delta = if index == 0 {
                sample
            } else {
                sample - previous
            };
            write_varint(zigzag(delta), &mut out);
            previous = sample;
        }
    }
    out
}

fn decode_delta_varints(
    payload: &[u8],
    n_channels: usize,
    n_samples: usize,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let mut offset = 0usize;
    let mut signal = Vec::with_capacity(n_channels);
    for _ in 0..n_channels {
        let mut channel = Vec::with_capacity(n_samples);
        let mut previous = 0i64;
        for index in 0..n_samples {
            let delta = unzigzag(read_varint(payload, &mut offset)?);
            let sample = if index == 0 {
                delta
            } else {
                previous.checked_add(delta).ok_or_else(|| {
                    OptimumV2Error::InvalidPacket("delta reconstruction overflow".into())
                })?
            };
            i32::try_from(sample)
                .map_err(|_| OptimumV2Error::InvalidPacket("delta sample exceeds i32".into()))?;
            channel.push(sample);
            previous = sample;
        }
        signal.push(channel);
    }
    if offset != payload.len() {
        return Err(OptimumV2Error::InvalidPacket(
            "delta payload has trailing bytes".into(),
        ));
    }
    Ok(signal)
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn write_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(payload: &[u8], offset: &mut usize) -> Result<u64, OptimumV2Error> {
    let start = *offset;
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *payload
            .get(*offset)
            .ok_or_else(|| OptimumV2Error::InvalidPacket("delta varint is truncated".into()))?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return Err(OptimumV2Error::InvalidPacket(
                "delta varint overflows u64".into(),
            ));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            if *offset - start > 1 && byte == 0 {
                return Err(OptimumV2Error::InvalidPacket(
                    "delta varint is not canonical".into(),
                ));
            }
            return Ok(value);
        }
    }
    Err(OptimumV2Error::InvalidPacket(
        "delta varint is too long".into(),
    ))
}

fn validate_signal(
    signal: &[Vec<i64>],
    context: &EncodeContext,
) -> Result<(usize, usize), OptimumV2Error> {
    if signal.is_empty() || signal.len() > MAX_CHANNELS {
        return Err(OptimumV2Error::InvalidInput(
            "channel count is outside supported range".into(),
        ));
    }
    let n_samples = signal[0].len();
    if n_samples == 0
        || n_samples > MAX_SAMPLES_PER_WINDOW
        || !sample_count_within_bounds(signal.len(), n_samples)
        || signal.iter().any(|channel| channel.len() != n_samples)
    {
        return Err(OptimumV2Error::InvalidInput(
            "signal must be rectangular and non-empty".into(),
        ));
    }
    if context.channel_labels.len() != signal.len()
        || !(1..=32).contains(&context.bit_depth)
        || context.sample_rate_mhz == 0
    {
        return Err(OptimumV2Error::InvalidInput(
            "context does not match signal".into(),
        ));
    }
    let min = -(1i64 << (context.bit_depth - 1));
    let max = (1i64 << (context.bit_depth - 1)) - 1;
    if signal
        .iter()
        .flatten()
        .any(|&sample| sample < min || sample > max)
    {
        return Err(OptimumV2Error::InvalidInput(
            "sample exceeds declared bit depth".into(),
        ));
    }
    Ok((signal.len(), n_samples))
}

fn sample_count_within_bounds(n_channels: usize, n_samples: usize) -> bool {
    matches!(
        n_channels.checked_mul(n_samples),
        Some(count) if count <= MAX_SAMPLE_VALUES
    )
}

fn encode_labels(labels: &[String]) -> Result<Vec<u8>, OptimumV2Error> {
    let mut out = Vec::new();
    for label in labels {
        let bytes = label.as_bytes();
        let len = u16::try_from(bytes.len())
            .map_err(|_| OptimumV2Error::InvalidInput("channel label is too long".into()))?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

fn decode_labels(bytes: &[u8], expected: usize) -> Result<Vec<String>, OptimumV2Error> {
    let mut labels = Vec::with_capacity(expected);
    let mut offset = 0usize;
    for _ in 0..expected {
        if offset + 2 > bytes.len() {
            return Err(OptimumV2Error::InvalidPacket(
                "channel labels are truncated".into(),
            ));
        }
        let len = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        if offset + len > bytes.len() {
            return Err(OptimumV2Error::InvalidPacket(
                "channel label bytes are truncated".into(),
            ));
        }
        let label = std::str::from_utf8(&bytes[offset..offset + len])
            .map_err(|_| OptimumV2Error::InvalidPacket("channel label is not UTF-8".into()))?;
        labels.push(label.to_owned());
        offset += len;
    }
    if offset != bytes.len() {
        return Err(OptimumV2Error::InvalidPacket(
            "channel label section has trailing bytes".into(),
        ));
    }
    Ok(labels)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, OptimumV2Error> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("u16 offset overflow".into()))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("truncated u16".into()))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, OptimumV2Error> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("u32 offset overflow".into()))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| OptimumV2Error::InvalidPacket("truncated u32".into()))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(!0u32, data) ^ !0u32
}

fn crc32c_update(mut state: u32, data: &[u8]) -> u32 {
    for &byte in data {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0x82F6_3B78 & (0u32.wrapping_sub(state & 1)));
        }
    }
    state
}

fn crc32c_zeroed_field(data: &[u8], offset: usize) -> u32 {
    let mut state = crc32c_update(!0u32, &data[..offset]);
    state = crc32c_update(state, &[0u8; 4]);
    state = crc32c_update(state, &data[offset + 4..]);
    state ^ !0u32
}

#[cfg(test)]
mod tests {
    use super::{
        crc32c, crc32c_zeroed_field, read_varint, EncodeContext, OptimumV2Codec, LMO_V3_HEADER_LEN,
    };

    #[test]
    fn crc32c_matches_castagnoli_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn noncanonical_varint_is_rejected() {
        let mut offset = 0;
        let error = read_varint(&[0x81, 0x00], &mut offset).expect_err("overlong one");
        assert!(error.to_string().contains("canonical"));
    }

    #[test]
    fn decoder_rejects_value_outside_declared_bit_depth() {
        let codec = OptimumV2Codec;
        let context = EncodeContext {
            sample_rate_mhz: 250_000,
            bit_depth: 16,
            channel_labels: vec!["Cz".into()],
        };
        let mut stream = codec.encode_window(&[vec![2]], &context).unwrap();
        stream[LMO_V3_HEADER_LEN + 6] = 2;
        let crc_offset = LMO_V3_HEADER_LEN + 76;
        let packet_crc = crc32c_zeroed_field(&stream, crc_offset);
        stream[crc_offset..crc_offset + 4].copy_from_slice(&packet_crc.to_le_bytes());
        let error = codec
            .decode_window(&stream)
            .expect_err("bit depth mismatch");
        assert!(error.to_string().contains("bit depth"));
    }
}
