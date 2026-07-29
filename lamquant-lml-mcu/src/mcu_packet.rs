//! Allocation-free MCU AOT invocation and fixed-LPC LML1 packet encoder.
//!
//! This is a physical execution ABI, not a storage format. Semantic identity
//! remains the ABIR dataset and compiled graph contract. The invocation bytes
//! carry one uniform, channel-major, little-endian `i64` window so firmware can
//! execute the production packet graph without pointer-bearing host values.

use crate::crc32::{crc32_update, CRC32_INIT};
use crate::lml::{compute_n_levels, BIAS_CTX};
use crate::lpc::FIXED_ORDER_SCHEDULE;

const INVOCATION_MAGIC: [u8; 4] = *b"LQSI";
const INVOCATION_VERSION: u8 = 1;
const ELEMENT_I64: u8 = 1;
const BYTE_ORDER_LITTLE: u8 = 1;
const LAYOUT_CHANNEL_MAJOR: u8 = 1;
const LML_MAGIC: [u8; 4] = *b"LML1";
const LML_HEADER_BYTES: usize = 22;
const Q_LPC: i64 = 27;
const MAX_CHANNELS: usize = 256;
const MAX_Q: u64 = 1_u64 << 40;
const MAX_LPC_ORDER: usize = 16;

/// Fixed header of the internal channel-major invocation ABI.
pub const UNIFORM_I64_INVOCATION_HEADER_BYTES: usize = 16;

/// Fail-closed errors from the allocation-free invocation path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McuPacketError {
    InvalidChannelCount,
    InvalidSampleCount,
    RaggedChannel {
        channel: usize,
        expected: usize,
        actual: usize,
    },
    SizeOverflow,
    InvocationBufferTooSmall {
        required: usize,
        actual: usize,
    },
    InvalidInvocation,
    WorkspaceTooSmall {
        required: usize,
        actual: usize,
    },
    OutputTooSmall {
        required: usize,
        actual: usize,
    },
    I64Min {
        index: usize,
    },
    ArithmeticOverflow,
}

#[derive(Clone, Copy)]
struct Invocation<'a> {
    channels: usize,
    samples: usize,
    payload: &'a [u8],
}

/// Deterministic predictor schedule accepted by the MCU packet kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McuLpcSchedule {
    Fixed,
    Adaptive { max_order: usize },
}

impl Invocation<'_> {
    fn sample(self, channel: usize, sample: usize) -> i64 {
        read_i64(self.payload, channel * self.samples + sample)
    }
}

/// Exact encoded size for one invocation buffer.
pub fn uniform_i64_invocation_len(
    channels: usize,
    samples: usize,
) -> Result<usize, McuPacketError> {
    validate_shape(channels, samples)?;
    channels
        .checked_mul(samples)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<i64>()))
        .and_then(|bytes| bytes.checked_add(UNIFORM_I64_INVOCATION_HEADER_BYTES))
        .ok_or(McuPacketError::SizeOverflow)
}

/// Serialize borrowed uniform channels into the internal MCU invocation ABI.
pub fn write_uniform_i64_invocation(
    channels: &[&[i64]],
    output: &mut [u8],
) -> Result<usize, McuPacketError> {
    let channel_count = channels.len();
    let samples = channels.first().map_or(0, |channel| channel.len());
    validate_shape(channel_count, samples)?;
    for (channel, values) in channels.iter().enumerate() {
        if values.len() != samples {
            return Err(McuPacketError::RaggedChannel {
                channel,
                expected: samples,
                actual: values.len(),
            });
        }
    }
    let required = uniform_i64_invocation_len(channel_count, samples)?;
    if output.len() < required {
        return Err(McuPacketError::InvocationBufferTooSmall {
            required,
            actual: output.len(),
        });
    }
    output[..4].copy_from_slice(&INVOCATION_MAGIC);
    output[4] = INVOCATION_VERSION;
    output[5] = ELEMENT_I64;
    output[6] = BYTE_ORDER_LITTLE;
    output[7] = LAYOUT_CHANNEL_MAJOR;
    output[8..10].copy_from_slice(&(channel_count as u16).to_le_bytes());
    output[10..12].copy_from_slice(&(samples as u16).to_le_bytes());
    output[12..16].copy_from_slice(&((required - 16) as u32).to_le_bytes());
    let mut cursor = UNIFORM_I64_INVOCATION_HEADER_BYTES;
    for channel in channels {
        for sample in *channel {
            output[cursor..cursor + 8].copy_from_slice(&sample.to_le_bytes());
            cursor += 8;
        }
    }
    Ok(required)
}

/// Scratch bytes required by [`compress_fixed_invocation_into`].
pub fn fixed_packet_workspace_len(samples: usize) -> Result<usize, McuPacketError> {
    if samples == 0 || samples > u16::MAX as usize {
        return Err(McuPacketError::InvalidSampleCount);
    }
    samples
        .checked_mul(core::mem::size_of::<i64>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(McuPacketError::SizeOverflow)
}

/// Encode one invocation as a baseline lossless LML1 packet.
///
/// Success performs no heap allocation. `workspace` owns two `i64`-sized byte
/// planes used for lifting and residuals; no alignment assumption is made.
/// Output may contain an unreachable partial packet after an error.
pub fn compress_fixed_invocation_into(
    invocation: &[u8],
    workspace: &mut [u8],
    output: &mut [u8],
) -> Result<usize, McuPacketError> {
    compress_invocation_into(invocation, McuLpcSchedule::Fixed, workspace, output)
}

/// Encode one invocation with an explicit deterministic predictor schedule.
///
/// `Anytime` graph configurations without a live deadline map to
/// `Adaptive`; deadline-bearing configurations are forbidden by the graph
/// schema. Predictor orders are clamped by the same N/8 and hard-16 ceilings
/// as the existing encoder.
pub fn compress_invocation_into(
    invocation: &[u8],
    schedule: McuLpcSchedule,
    workspace: &mut [u8],
    output: &mut [u8],
) -> Result<usize, McuPacketError> {
    if matches!(
        schedule,
        McuLpcSchedule::Adaptive {
            max_order: 0 | 65..
        }
    ) {
        return Err(McuPacketError::InvalidInvocation);
    }
    let invocation = parse_invocation(invocation)?;
    let required_workspace = fixed_packet_workspace_len(invocation.samples)?;
    if workspace.len() < required_workspace {
        return Err(McuPacketError::WorkspaceTooSmall {
            required: required_workspace,
            actual: workspace.len(),
        });
    }
    let plane_bytes = invocation
        .samples
        .checked_mul(8)
        .ok_or(McuPacketError::SizeOverflow)?;
    let (transformed, rest) = workspace.split_at_mut(plane_bytes);
    let temporary = &mut rest[..plane_bytes];
    let levels = compute_n_levels(invocation.samples);
    let (subbands, subband_count) = subband_ranges(invocation.samples, levels);
    let maximum_metadata_per_channel = subbands[..subband_count]
        .iter()
        .enumerate()
        .try_fold(0_usize, |total, (index, (_, len))| {
            total.checked_add(1 + maximum_order(schedule, index, *len) * 4)
        })
        .ok_or(McuPacketError::SizeOverflow)?;
    let maximum_metadata_len = maximum_metadata_per_channel
        .checked_mul(invocation.channels)
        .ok_or(McuPacketError::SizeOverflow)?;
    let prefix_len = lml_prefix_len(invocation.channels);
    let reserved_payload_start = prefix_len
        .checked_add(LML_HEADER_BYTES)
        .and_then(|value| value.checked_add(maximum_metadata_len))
        .ok_or(McuPacketError::SizeOverflow)?;
    if output.len() < reserved_payload_start {
        return Err(McuPacketError::OutputTooSmall {
            required: reserved_payload_start,
            actual: output.len(),
        });
    }

    write_lml_prefix(invocation.channels, output)?;
    output[prefix_len..prefix_len + 4].copy_from_slice(&LML_MAGIC);
    let mut metadata_cursor = prefix_len + LML_HEADER_BYTES;
    let mut payload_cursor = reserved_payload_start;
    for channel in 0..invocation.channels {
        transform_channel(invocation, channel, levels, transformed, temporary)?;
        for (subband_index, (start, len)) in subbands[..subband_count].iter().copied().enumerate() {
            let mut coefficients = [0_i32; MAX_LPC_ORDER];
            let order = analyze_into(
                &transformed[start * 8..(start + len) * 8],
                len,
                schedule,
                subband_index,
                &mut temporary[..len * 8],
                &mut coefficients,
            )?;
            output[metadata_cursor] = order as u8;
            metadata_cursor += 1;
            for coefficient in &coefficients[..order] {
                output[metadata_cursor..metadata_cursor + 4]
                    .copy_from_slice(&coefficient.to_le_bytes());
                metadata_cursor += 4;
            }
            let written =
                encode_rice_into(&temporary[..len * 8], len, &mut output[payload_cursor..])?;
            payload_cursor = payload_cursor
                .checked_add(written)
                .ok_or(McuPacketError::SizeOverflow)?;
        }
    }

    let metadata_start = prefix_len + LML_HEADER_BYTES;
    let metadata_len = metadata_cursor - metadata_start;
    let payload_len = payload_cursor - reserved_payload_start;
    let payload_start = metadata_cursor;
    output.copy_within(reserved_payload_start..payload_cursor, payload_start);
    payload_cursor = payload_start
        .checked_add(payload_len)
        .ok_or(McuPacketError::SizeOverflow)?;
    let metadata_len_u32 = u32::try_from(metadata_len).map_err(|_| McuPacketError::SizeOverflow)?;
    let payload_len_u32 = u32::try_from(payload_len).map_err(|_| McuPacketError::SizeOverflow)?;
    let mut header = [0_u8; 14];
    header[0..2].copy_from_slice(&(invocation.channels as u16).to_le_bytes());
    header[2..4].copy_from_slice(&(invocation.samples as u16).to_le_bytes());
    header[4] = levels;
    header[5] = 0;
    header[6..10].copy_from_slice(&metadata_len_u32.to_le_bytes());
    header[10..14].copy_from_slice(&payload_len_u32.to_le_bytes());
    let header_start = prefix_len + 4;
    output[header_start..header_start + 14].copy_from_slice(&header);
    let mut crc = CRC32_INIT;
    crc = crc32_update(crc, &header);
    crc = crc32_update(crc, &output[prefix_len + LML_HEADER_BYTES..payload_start]);
    crc = crc32_update(crc, &output[payload_start..payload_cursor]);
    output[header_start + 14..header_start + 18].copy_from_slice(&(crc ^ CRC32_INIT).to_le_bytes());
    Ok(payload_cursor)
}

fn validate_shape(channels: usize, samples: usize) -> Result<(), McuPacketError> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(McuPacketError::InvalidChannelCount);
    }
    if samples == 0 || samples > u16::MAX as usize {
        return Err(McuPacketError::InvalidSampleCount);
    }
    Ok(())
}

fn parse_invocation(bytes: &[u8]) -> Result<Invocation<'_>, McuPacketError> {
    if bytes.len() < UNIFORM_I64_INVOCATION_HEADER_BYTES
        || bytes[..4] != INVOCATION_MAGIC
        || bytes[4] != INVOCATION_VERSION
        || bytes[5] != ELEMENT_I64
        || bytes[6] != BYTE_ORDER_LITTLE
        || bytes[7] != LAYOUT_CHANNEL_MAJOR
    {
        return Err(McuPacketError::InvalidInvocation);
    }
    let channels = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let samples = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
    validate_shape(channels, samples)?;
    let declared = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let expected = uniform_i64_invocation_len(channels, samples)?;
    if declared != expected - UNIFORM_I64_INVOCATION_HEADER_BYTES || bytes.len() != expected {
        return Err(McuPacketError::InvalidInvocation);
    }
    Ok(Invocation {
        channels,
        samples,
        payload: &bytes[UNIFORM_I64_INVOCATION_HEADER_BYTES..],
    })
}

fn transform_channel(
    invocation: Invocation<'_>,
    channel: usize,
    levels: u8,
    transformed: &mut [u8],
    temporary: &mut [u8],
) -> Result<(), McuPacketError> {
    for sample in 0..invocation.samples {
        write_i64(transformed, sample, invocation.sample(channel, sample));
    }
    let mut current = invocation.samples;
    for _ in 0..levels {
        let detail_count = current / 2;
        let approx_count = current.div_ceil(2);
        for index in 0..approx_count {
            write_i64(temporary, index, read_i64(transformed, 2 * index));
        }
        for index in 0..detail_count {
            write_i64(
                temporary,
                approx_count + index,
                read_i64(transformed, 2 * index + 1),
            );
        }
        let bulk_end = if current % 2 == 0 {
            detail_count - 1
        } else {
            detail_count
        };
        for index in 0..bulk_end {
            let left = read_i64(temporary, index);
            let right = read_i64(temporary, index + 1);
            let prediction = left
                .checked_add(right)
                .ok_or(McuPacketError::ArithmeticOverflow)?
                >> 1;
            let detail_index = approx_count + index;
            let value = read_i64(temporary, detail_index)
                .checked_sub(prediction)
                .ok_or(McuPacketError::ArithmeticOverflow)?;
            write_i64(temporary, detail_index, value);
        }
        if current % 2 == 0 {
            let index = detail_count - 1;
            let detail_index = approx_count + index;
            let value = read_i64(temporary, detail_index)
                .checked_sub(read_i64(temporary, index))
                .ok_or(McuPacketError::ArithmeticOverflow)?;
            write_i64(temporary, detail_index, value);
        }
        let first_detail = read_i64(temporary, approx_count);
        let first = read_i64(temporary, 0)
            .checked_add(
                first_detail
                    .checked_add(1)
                    .ok_or(McuPacketError::ArithmeticOverflow)?
                    >> 1,
            )
            .ok_or(McuPacketError::ArithmeticOverflow)?;
        write_i64(temporary, 0, first);
        for index in 1..approx_count {
            let left_detail = read_i64(temporary, approx_count + index - 1);
            let update = if index < detail_count {
                left_detail
                    .checked_add(read_i64(temporary, approx_count + index))
                    .and_then(|value| value.checked_add(2))
                    .ok_or(McuPacketError::ArithmeticOverflow)?
                    >> 2
            } else {
                left_detail
                    .checked_add(1)
                    .ok_or(McuPacketError::ArithmeticOverflow)?
                    >> 1
            };
            let value = read_i64(temporary, index)
                .checked_add(update)
                .ok_or(McuPacketError::ArithmeticOverflow)?;
            write_i64(temporary, index, value);
        }
        transformed[..current * 8].copy_from_slice(&temporary[..current * 8]);
        current = approx_count;
    }
    Ok(())
}

fn subband_ranges(samples: usize, levels: u8) -> ([(usize, usize); 4], usize) {
    let mut details = [0_usize; 3];
    let mut approx = samples;
    for level in 0..levels as usize {
        details[level] = approx / 2;
        approx = approx.div_ceil(2);
    }
    let mut ranges = [(0_usize, 0_usize); 4];
    ranges[0] = (0, approx);
    let mut cursor = approx;
    for index in 0..levels as usize {
        let len = details[levels as usize - 1 - index];
        ranges[index + 1] = (cursor, len);
        cursor += len;
    }
    debug_assert_eq!(cursor, samples);
    (ranges, levels as usize + 1)
}

fn maximum_order(schedule: McuLpcSchedule, subband_index: usize, samples: usize) -> usize {
    match schedule {
        McuLpcSchedule::Fixed => FIXED_ORDER_SCHEDULE[subband_index],
        McuLpcSchedule::Adaptive { max_order } => max_order.min((samples / 8).min(MAX_LPC_ORDER)),
    }
}

fn analyze_into(
    subband: &[u8],
    samples: usize,
    schedule: McuLpcSchedule,
    subband_index: usize,
    residual: &mut [u8],
    coefficients: &mut [i32; MAX_LPC_ORDER],
) -> Result<usize, McuPacketError> {
    match schedule {
        McuLpcSchedule::Fixed => analyze_fixed_into(
            subband,
            samples,
            FIXED_ORDER_SCHEDULE[subband_index],
            residual,
            coefficients,
        ),
        McuLpcSchedule::Adaptive { max_order } => analyze_adaptive_into(
            subband,
            samples,
            maximum_order(schedule, subband_index, samples),
            max_order,
            residual,
            coefficients,
        ),
    }
}

fn analyze_fixed_into(
    subband: &[u8],
    samples: usize,
    order: usize,
    residual: &mut [u8],
    coefficients: &mut [i32; MAX_LPC_ORDER],
) -> Result<usize, McuPacketError> {
    coefficients.fill(0);
    if samples <= order || samples < 3 || order == 0 {
        residual[..samples * 8].copy_from_slice(&subband[..samples * 8]);
        bias_cancel(residual, samples)?;
        return Ok(order);
    }
    let segment = (samples / 2).clamp(1, 256);
    let mut autocorrelation = [0.0_f64; MAX_LPC_ORDER + 1];
    for lag in 0..=order {
        let mut sum = 0.0_f64;
        for index in 0..segment.saturating_sub(lag) {
            sum += read_i64(subband, index) as f64 * read_i64(subband, index + lag) as f64;
        }
        autocorrelation[lag] = sum;
    }
    if abs_f64(autocorrelation[0]) <= 1e-12 {
        residual[..samples * 8].copy_from_slice(&subband[..samples * 8]);
        bias_cancel(residual, samples)?;
        return Ok(order);
    }

    let mut a = [0.0_f64; MAX_LPC_ORDER];
    let mut next = [0.0_f64; MAX_LPC_ORDER];
    let mut error = autocorrelation[0];
    for m in 0..order {
        let mut lambda = autocorrelation[m + 1];
        for j in 0..m {
            lambda += a[j] * autocorrelation[m - j];
        }
        if abs_f64(error) <= 1e-12 {
            break;
        }
        let reflection = -lambda / error;
        next.fill(0.0);
        next[m] = reflection;
        for j in 0..m {
            next[j] = a[j] + reflection * a[m - 1 - j];
        }
        core::mem::swap(&mut a, &mut next);
        error *= 1.0 - reflection * reflection;
        if error <= 0.0 {
            error = 1e-10;
        }
    }
    let q27 = 1_i64 << Q_LPC;
    for index in 0..order {
        let value = -a[index] * q27 as f64;
        coefficients[index] = (if value >= 0.0 {
            value + 0.5
        } else {
            value - 0.5
        }) as i32;
    }
    for index in 0..samples {
        let mut prediction = 0_i64;
        for lag in 0..order.min(index) {
            let term = (coefficients[lag] as i64)
                .checked_mul(read_i64(subband, index - 1 - lag))
                .ok_or(McuPacketError::ArithmeticOverflow)?;
            prediction = prediction
                .checked_add(term)
                .ok_or(McuPacketError::ArithmeticOverflow)?;
        }
        let value = read_i64(subband, index)
            .checked_sub(prediction >> Q_LPC)
            .ok_or(McuPacketError::ArithmeticOverflow)?;
        write_i64(residual, index, value);
    }
    bias_cancel(residual, samples)?;
    Ok(order)
}

fn analyze_adaptive_into(
    subband: &[u8],
    samples: usize,
    scoped_max_order: usize,
    _requested_max_order: usize,
    residual: &mut [u8],
    coefficients: &mut [i32; MAX_LPC_ORDER],
) -> Result<usize, McuPacketError> {
    coefficients.fill(0);
    if scoped_max_order == 0 || samples < 3 {
        residual[..samples * 8].copy_from_slice(&subband[..samples * 8]);
        bias_cancel(residual, samples)?;
        return Ok(0);
    }
    let max_order = scoped_max_order
        .min(samples.saturating_sub(1))
        .min(samples / 4)
        .max(1);
    let segment = (samples / 2).clamp(1, 256);
    let mut autocorrelation = [0.0_f64; MAX_LPC_ORDER + 1];
    for lag in 0..=max_order {
        let mut sum = 0.0_f64;
        for index in 0..segment.saturating_sub(lag) {
            sum += read_i64(subband, index) as f64 * read_i64(subband, index + lag) as f64;
        }
        autocorrelation[lag] = sum;
    }
    if abs_f64(autocorrelation[0]) <= 1e-12 {
        residual[..samples * 8].copy_from_slice(&subband[..samples * 8]);
        bias_cancel(residual, samples)?;
        return Ok(0);
    }

    const ORDER_BIT_COST: f64 = 32.0 * core::f64::consts::LN_2;
    let n = samples as f64;
    let mut previous = [0.0_f64; MAX_LPC_ORDER];
    let mut current = [0.0_f64; MAX_LPC_ORDER];
    let mut best = [0.0_f64; MAX_LPC_ORDER];
    let mut error = autocorrelation[0];
    let mut best_cost = f64::INFINITY;
    let mut best_order = 0_usize;
    if autocorrelation[0] > 0.0 {
        let cost = 0.72 * n * libm::log(autocorrelation[0] / n);
        if cost < best_cost {
            best_cost = cost;
        }
    }
    for m in 0..max_order {
        let mut lambda = autocorrelation[m + 1];
        for j in 0..m {
            lambda += previous[j] * autocorrelation[m - j];
        }
        if abs_f64(error) <= 1e-12 {
            break;
        }
        let reflection = -lambda / error;
        current[m] = reflection;
        for j in 0..m {
            current[j] = previous[j] + reflection * previous[m - 1 - j];
        }
        let new_error = error * (1.0 - reflection * reflection);
        if new_error <= 0.0 || !new_error.is_finite() {
            break;
        }
        error = new_error;
        let order = m + 1;
        let cost = 0.72 * n * libm::log(error / n) + ORDER_BIT_COST * order as f64;
        if cost < best_cost {
            best_cost = cost;
            best[..order].copy_from_slice(&current[..order]);
            best_order = order;
        }
        previous[..order].copy_from_slice(&current[..order]);
    }
    if best_order == 0 {
        residual[..samples * 8].copy_from_slice(&subband[..samples * 8]);
        bias_cancel(residual, samples)?;
        return Ok(0);
    }
    let q27 = 1_i64 << Q_LPC;
    for index in 0..best_order {
        let value = -best[index] * q27 as f64;
        coefficients[index] = (if value >= 0.0 {
            value + 0.5
        } else {
            value - 0.5
        }) as i32;
    }
    for index in 0..samples {
        let mut prediction = 0_i64;
        for lag in 0..best_order.min(index) {
            let term = (coefficients[lag] as i64)
                .checked_mul(read_i64(subband, index - 1 - lag))
                .ok_or(McuPacketError::ArithmeticOverflow)?;
            prediction = prediction
                .checked_add(term)
                .ok_or(McuPacketError::ArithmeticOverflow)?;
        }
        let value = read_i64(subband, index)
            .checked_sub(prediction >> Q_LPC)
            .ok_or(McuPacketError::ArithmeticOverflow)?;
        write_i64(residual, index, value);
    }
    bias_cancel(residual, samples)?;
    Ok(best_order)
}

fn bias_cancel(data: &mut [u8], samples: usize) -> Result<(), McuPacketError> {
    let mut history = [0_i64; BIAS_CTX];
    let mut running_sum = 0_i64;
    for index in 0..samples {
        let bias = floor_div(running_sum, BIAS_CTX as i64);
        let value = read_i64(data, index);
        write_i64(
            data,
            index,
            value
                .checked_sub(bias)
                .ok_or(McuPacketError::ArithmeticOverflow)?,
        );
        let slot = index & (BIAS_CTX - 1);
        let old = history[slot];
        history[slot] = value;
        running_sum = running_sum
            .checked_add(value)
            .and_then(|sum| sum.checked_sub(old))
            .ok_or(McuPacketError::ArithmeticOverflow)?;
    }
    Ok(())
}

fn encode_rice_into(
    values: &[u8],
    count: usize,
    output: &mut [u8],
) -> Result<usize, McuPacketError> {
    let mut sum = 0_u128;
    let mut nonzero = 0_u128;
    for index in 0..count {
        let value = read_i64(values, index);
        if value == i64::MIN {
            return Err(McuPacketError::I64Min { index });
        }
        let encoded = zigzag(value);
        if encoded != 0 {
            sum += u128::from(encoded);
            nonzero += 1;
        }
    }
    let mut mean = sum as f64 / nonzero as f64;
    let mut k = 0_u8;
    while mean >= 2.0 && k < 31 {
        mean *= 0.5;
        k += 1;
    }
    let mut bits = 0_u128;
    for index in 0..count {
        let encoded = zigzag(read_i64(values, index));
        let quotient = encoded >> k;
        if quotient > MAX_Q {
            return Err(McuPacketError::ArithmeticOverflow);
        }
        bits = bits
            .checked_add(u128::from(quotient) + 1 + u128::from(k))
            .ok_or(McuPacketError::SizeOverflow)?;
    }
    let required = usize::try_from(bits.div_ceil(8))
        .map_err(|_| McuPacketError::SizeOverflow)?
        .checked_add(3)
        .ok_or(McuPacketError::SizeOverflow)?;
    if output.len() < required {
        return Err(McuPacketError::OutputTooSmall {
            required,
            actual: output.len(),
        });
    }
    output[0] = k;
    output[1..3].copy_from_slice(&(count as u16).to_le_bytes());
    let mut cursor = 3;
    let mut bit_buffer = 0_u64;
    let mut bit_count = 0_i32;
    let k_u64 = u64::from(k);
    let mask = if k == 0 { 0 } else { (1_u64 << k) - 1 };
    for index in 0..count {
        let encoded = zigzag(read_i64(values, index));
        let mut quotient = encoded >> k_u64;
        let remainder = encoded & mask;
        while quotient >= 56 {
            bit_buffer <<= 56;
            bit_count += 56;
            flush_full_bytes(output, &mut cursor, &mut bit_buffer, &mut bit_count);
            quotient -= 56;
        }
        let unary = quotient + 1;
        bit_buffer = (bit_buffer << unary) | 1;
        bit_count += unary as i32;
        flush_full_bytes(output, &mut cursor, &mut bit_buffer, &mut bit_count);
        if k > 0 {
            bit_buffer = (bit_buffer << k_u64) | remainder;
            bit_count += i32::from(k);
            flush_full_bytes(output, &mut cursor, &mut bit_buffer, &mut bit_count);
        }
    }
    if bit_count > 0 {
        output[cursor] = ((bit_buffer << (8 - bit_count) as u64) & 0xff) as u8;
        cursor += 1;
    }
    debug_assert_eq!(cursor, required);
    Ok(required)
}

fn flush_full_bytes(
    output: &mut [u8],
    cursor: &mut usize,
    bit_buffer: &mut u64,
    bit_count: &mut i32,
) {
    while *bit_count >= 8 {
        *bit_count -= 8;
        output[*cursor] = ((*bit_buffer >> *bit_count as u64) & 0xff) as u8;
        *cursor += 1;
    }
    *bit_buffer = if *bit_count > 0 {
        *bit_buffer & ((1_u64 << *bit_count as u64) - 1)
    } else {
        0
    };
}

fn lml_prefix_len(channels: usize) -> usize {
    b"LML | ".len() + decimal_digits(channels) + b"ch | lossless | CRC-32\n".len()
}

fn write_lml_prefix(channels: usize, output: &mut [u8]) -> Result<(), McuPacketError> {
    let required = lml_prefix_len(channels);
    if output.len() < required {
        return Err(McuPacketError::OutputTooSmall {
            required,
            actual: output.len(),
        });
    }
    let mut cursor = 0;
    cursor += copy_bytes(output, cursor, b"LML | ");
    cursor += write_decimal(channels, &mut output[cursor..]);
    cursor += copy_bytes(output, cursor, b"ch | lossless | CRC-32\n");
    debug_assert_eq!(cursor, required);
    Ok(())
}

fn decimal_digits(mut value: usize) -> usize {
    let mut digits = 1;
    while value >= 10 {
        digits += 1;
        value /= 10;
    }
    digits
}

fn write_decimal(mut value: usize, output: &mut [u8]) -> usize {
    let digits = decimal_digits(value);
    for index in (0..digits).rev() {
        output[index] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    digits
}

fn copy_bytes(output: &mut [u8], offset: usize, value: &[u8]) -> usize {
    output[offset..offset + value.len()].copy_from_slice(value);
    value.len()
}

fn floor_div(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    if (value ^ divisor) < 0 && quotient * divisor != value {
        quotient - 1
    } else {
        quotient
    }
}

fn abs_f64(value: f64) -> f64 {
    if value < 0.0 {
        -value
    } else {
        value
    }
}

fn zigzag(value: i64) -> u64 {
    ((value as u64) << 1) ^ ((value >> 63) as u64)
}

fn read_i64(bytes: &[u8], index: usize) -> i64 {
    let offset = index * 8;
    i64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn write_i64(bytes: &mut [u8], index: usize, value: i64) {
    let offset = index * 8;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
