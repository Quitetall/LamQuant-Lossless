//! Frozen shared-Q15 lattice and compact sparse graph used by MIX1.

use crate::OptimumV2Error;

pub(crate) const ORDER: usize = 16;
const Q_ONE: i128 = 1 << 15;
const PARENT_CAP: usize = 4;
const CANDIDATE_LIMIT: usize = 6;
const COEFFICIENT_RICE_K: u8 = 10;
const WEIGHT_RICE_K: u8 = 7;
const WEIGHT_MIN: i16 = -1024;
const WEIGHT_MAX: i16 = 1024;
const LOCAL_STEPS: [i16; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
const COORDINATE_PASS_LIMIT: usize = 2_049;
const LSQ_FRACTIONAL_BITS: u32 = 28;
const LSQ_FALLBACK_SWEEPS: usize = 4;

type SparseGraph = (Vec<Vec<usize>>, Vec<Vec<i16>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LatticeSide {
    pub(crate) coefficients: [i16; ORDER],
    pub(crate) parents: Vec<Vec<usize>>,
    pub(crate) weights_q8: Vec<Vec<i16>>,
}

pub(crate) fn fit_and_analyze(
    signal: &[Vec<i64>],
) -> Result<(LatticeSide, Vec<Vec<i64>>), OptimumV2Error> {
    let (channels, samples) = validate_signal(signal)?;
    if samples <= ORDER {
        return Err(input_error("MIX1 lattice requires more than 16 samples"));
    }
    let coefficients = fit_coefficients(signal)?;
    let innovations = analyze(signal, &coefficients)?;
    let (mut parents, mut weights_q8) = sparse_graph(&innovations)?;
    for channel in 0..channels {
        let mut pairs = parents[channel]
            .iter()
            .copied()
            .zip(weights_q8[channel].iter().copied())
            .collect::<Vec<_>>();
        pairs.sort_unstable();
        parents[channel] = pairs.iter().map(|pair| pair.0).collect();
        weights_q8[channel] = pairs.iter().map(|pair| pair.1).collect();
    }
    let side = LatticeSide {
        coefficients,
        parents,
        weights_q8,
    };
    let residuals = lift(&innovations, &side)?;
    Ok((side, residuals))
}

pub(crate) fn pack_side(side: &LatticeSide, score_shift: u8) -> Result<Vec<u8>, OptimumV2Error> {
    validate_side(side)?;
    if !(2..=8).contains(&score_shift) {
        return Err(input_error("MIX1 score shift must be in 2..=8"));
    }
    let mut writer = BitWriter::default();
    for &coefficient in &side.coefficients {
        writer.write_rice(zigzag(i64::from(coefficient)), COEFFICIENT_RICE_K)?;
    }
    for channel in 0..side.parents.len() {
        let parents = &side.parents[channel];
        let full_count = PARENT_CAP.min(channel);
        if parents.len() == full_count {
            writer.write_bit(1)?;
        } else if parents.len() < full_count {
            writer.write_bit(0)?;
            writer.write_bits(parents.len() as u64, 3)?;
        } else {
            return Err(input_error("MIX1 graph parent count is noncanonical"));
        }
        let combinations = combination(channel, parents.len())?;
        let width = bit_length(combinations - 1) as u8;
        writer.write_bits(colex_rank(parents)?, width)?;
        for &weight in &side.weights_q8[channel] {
            writer.write_rice(zigzag(i64::from(weight)), WEIGHT_RICE_K)?;
        }
    }
    let mut packed = Vec::from(&b"MIX1"[..]);
    packed.extend_from_slice(&[0xA7, score_shift]);
    packed.extend_from_slice(&writer.finish());
    Ok(packed)
}

pub(crate) fn parse_side(
    packed: &[u8],
    channels: usize,
    samples: usize,
) -> Result<(u8, LatticeSide), OptimumV2Error> {
    if packed.len() < 6 || &packed[..4] != b"MIX1" || packed[4] != 0xA7 {
        return Err(packet_error("MIX1 side identity is invalid"));
    }
    if !(1..=256).contains(&channels) || samples <= ORDER || samples > 32_768 {
        return Err(packet_error("MIX1 side dimensions are invalid"));
    }
    let score_shift = packed[5];
    if !(2..=8).contains(&score_shift) {
        return Err(packet_error("MIX1 score shift must be in 2..=8"));
    }
    let mut reader = BitReader::new(&packed[6..]);
    let mut coefficients = [0i16; ORDER];
    for coefficient in &mut coefficients {
        let value = reader.read_rice(COEFFICIENT_RICE_K, 2 * ((Q_ONE - 1) as u64))?;
        let signed = unzigzag(value)?;
        *coefficient = i16::try_from(signed)
            .map_err(|_| packet_error("MIX1 lattice coefficient exceeds i16"))?;
        if i128::from(*coefficient).unsigned_abs() >= Q_ONE as u128 {
            return Err(packet_error("MIX1 lattice coefficient is unstable"));
        }
    }
    let mut parents = Vec::with_capacity(channels);
    let mut weights_q8 = Vec::with_capacity(channels);
    for channel in 0..channels {
        let full_count = PARENT_CAP.min(channel);
        let count = if reader.read_bit()? == 1 {
            full_count
        } else {
            let count = reader.read_bits(3)? as usize;
            if count >= full_count {
                return Err(packet_error("MIX1 graph parent count is noncanonical"));
            }
            count
        };
        let combinations = combination(channel, count)?;
        let width = bit_length(combinations - 1) as u8;
        let rank = reader.read_bits(width)?;
        if rank >= combinations {
            return Err(packet_error("MIX1 graph parent rank is unused"));
        }
        let row = colex_unrank(rank, channel, count)?;
        let mut row_weights = Vec::with_capacity(count);
        for _ in 0..count {
            let value = reader.read_rice(WEIGHT_RICE_K, zigzag(i64::from(WEIGHT_MAX)))?;
            let weight = i16::try_from(unzigzag(value)?)
                .map_err(|_| packet_error("MIX1 graph weight exceeds i16"))?;
            if !(WEIGHT_MIN..=WEIGHT_MAX).contains(&weight) {
                return Err(packet_error("MIX1 graph weight exceeds Q8 bound"));
            }
            row_weights.push(weight);
        }
        parents.push(row);
        weights_q8.push(row_weights);
    }
    reader.finish_zero_padding()?;
    let side = LatticeSide {
        coefficients,
        parents,
        weights_q8,
    };
    validate_side(&side).map_err(as_packet_error)?;
    if pack_side(&side, score_shift).map_err(as_packet_error)? != packed {
        return Err(packet_error("MIX1 side information is noncanonical"));
    }
    Ok((score_shift, side))
}

pub(crate) fn graph_prediction(
    side: &LatticeSide,
    channel: usize,
    current_innovations: &[i64],
) -> Result<i64, OptimumV2Error> {
    let parents = side
        .parents
        .get(channel)
        .ok_or_else(|| input_error("MIX1 lattice channel is out of range"))?;
    let weights = &side.weights_q8[channel];
    let mut weighted = 0i128;
    for (&parent, &weight) in parents.iter().zip(weights) {
        let term = i128::from(weight)
            .checked_mul(i128::from(current_innovations[parent]))
            .ok_or_else(|| arithmetic_error("MIX1 graph product overflows i128"))?;
        weighted = weighted
            .checked_add(term)
            .ok_or_else(|| arithmetic_error("MIX1 graph sum overflows i128"))?;
    }
    i64::try_from(round_q8(weighted))
        .map_err(|_| arithmetic_error("MIX1 graph prediction exceeds i64"))
}

pub(crate) fn inverse_sample(
    innovation: i64,
    coefficients: &[i16; ORDER],
    previous_backward: &[i128],
) -> Result<i64, OptimumV2Error> {
    let mut forward = i128::from(innovation);
    for stage in (1..=ORDER).rev() {
        let product = i128::from(coefficients[stage - 1])
            .checked_mul(previous_backward[stage - 1])
            .ok_or_else(|| arithmetic_error("MIX1 lattice inverse product overflows i128"))?;
        forward = forward
            .checked_add(round_q15(product))
            .ok_or_else(|| arithmetic_error("MIX1 lattice inverse state overflows i128"))?;
    }
    i64::try_from(forward).map_err(|_| arithmetic_error("MIX1 lattice sample exceeds i64"))
}

pub(crate) fn analyze_sample(
    sample: i64,
    coefficients: &[i16; ORDER],
    previous_backward: &[i128],
    current_backward: &mut [i128],
) -> Result<i64, OptimumV2Error> {
    if previous_backward.len() != ORDER + 1 || current_backward.len() != ORDER + 1 {
        return Err(input_error("MIX1 lattice state width is invalid"));
    }
    current_backward[0] = i128::from(sample);
    let mut forward = i128::from(sample);
    for stage in 1..=ORDER {
        let previous_forward = forward;
        let forward_product = i128::from(coefficients[stage - 1])
            .checked_mul(previous_backward[stage - 1])
            .ok_or_else(|| arithmetic_error("MIX1 lattice analysis product overflows i128"))?;
        forward = previous_forward
            .checked_sub(round_q15(forward_product))
            .ok_or_else(|| arithmetic_error("MIX1 lattice analysis state overflows i128"))?;
        let backward_product = i128::from(coefficients[stage - 1])
            .checked_mul(previous_forward)
            .ok_or_else(|| arithmetic_error("MIX1 lattice backward product overflows i128"))?;
        current_backward[stage] = previous_backward[stage - 1]
            .checked_sub(round_q15(backward_product))
            .ok_or_else(|| arithmetic_error("MIX1 lattice backward state overflows i128"))?;
    }
    i64::try_from(forward).map_err(|_| arithmetic_error("MIX1 lattice innovation exceeds i64"))
}

fn fit_coefficients(signal: &[Vec<i64>]) -> Result<[i16; ORDER], OptimumV2Error> {
    let samples = signal[0].len();
    let mut correlations = [0i128; ORDER + 1];
    for lag in 0..=ORDER {
        let mut total = 0i128;
        for row in signal {
            for time in lag..samples {
                let product = i128::from(row[time])
                    .checked_mul(i128::from(row[time - lag]))
                    .ok_or_else(|| {
                        arithmetic_error("MIX1 autocorrelation product overflows i128")
                    })?;
                total = total
                    .checked_add(product)
                    .ok_or_else(|| arithmetic_error("MIX1 autocorrelation sum overflows i128"))?;
            }
        }
        correlations[lag] = total;
    }
    fit_reflections(&correlations)
}

fn fit_reflections(correlations: &[i128; ORDER + 1]) -> Result<[i16; ORDER], OptimumV2Error> {
    let mut error = correlations[0];
    if error < 0 {
        return Err(input_error("MIX1 autocorrelation energy is negative"));
    }
    if error == 0 {
        return Ok([0; ORDER]);
    }
    let mut direct: Vec<i128> = Vec::with_capacity(ORDER);
    let mut reflections = [0i16; ORDER];
    for stage in 1..=ORDER {
        let mut numerator = correlations[stage]
            .checked_mul(Q_ONE)
            .ok_or_else(|| arithmetic_error("MIX1 Levinson numerator overflows i128"))?;
        for index in 1..stage {
            let term = direct[index - 1]
                .checked_mul(correlations[stage - index])
                .ok_or_else(|| arithmetic_error("MIX1 Levinson numerator overflows i128"))?;
            numerator = numerator
                .checked_sub(term)
                .ok_or_else(|| arithmetic_error("MIX1 Levinson numerator overflows i128"))?;
        }
        let reflection = round_ratio_even(numerator, error)?.clamp(-Q_ONE + 1, Q_ONE - 1);
        let old = direct;
        direct = Vec::with_capacity(stage);
        for index in 0..stage - 1 {
            let product = reflection
                .checked_mul(old[stage - index - 2])
                .ok_or_else(|| arithmetic_error("MIX1 Levinson update overflows i128"))?;
            let correction = round_ratio_even(product, Q_ONE)?;
            direct.push(
                old[index]
                    .checked_sub(correction)
                    .ok_or_else(|| arithmetic_error("MIX1 Levinson update overflows i128"))?,
            );
        }
        direct.push(reflection);
        reflections[stage - 1] = reflection as i16;
        let stability = Q_ONE * Q_ONE - reflection * reflection;
        let energy_numerator = error
            .checked_mul(stability)
            .ok_or_else(|| arithmetic_error("MIX1 Levinson energy overflows i128"))?;
        error = 1.max(round_ratio_even(energy_numerator, Q_ONE * Q_ONE)?);
    }
    Ok(reflections)
}

fn analyze(
    signal: &[Vec<i64>],
    coefficients: &[i16; ORDER],
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = signal.len();
    let samples = signal[0].len();
    let mut previous_backward = vec![vec![0i128; ORDER + 1]; channels];
    let mut innovations = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        let mut current_backward = vec![vec![0i128; ORDER + 1]; channels];
        for channel in 0..channels {
            innovations[channel][time] = analyze_sample(
                signal[channel][time],
                coefficients,
                &previous_backward[channel],
                &mut current_backward[channel],
            )?;
        }
        previous_backward = current_backward;
    }
    Ok(innovations)
}

fn lift(innovations: &[Vec<i64>], side: &LatticeSide) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = innovations.len();
    let samples = innovations[0].len();
    let mut residuals = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            current[channel] = innovations[channel][time];
            residuals[channel][time] = innovations[channel][time]
                .checked_sub(graph_prediction(side, channel, &current)?)
                .ok_or_else(|| arithmetic_error("MIX1 lifting residual exceeds i64"))?;
        }
    }
    Ok(residuals)
}

fn sparse_graph(innovations: &[Vec<i64>]) -> Result<SparseGraph, OptimumV2Error> {
    let channels = innovations.len();
    let mut dots = vec![vec![0i128; channels]; channels];
    for left in 0..channels {
        for right in 0..=left {
            let mut value = 0i128;
            for (&a, &b) in innovations[left].iter().zip(&innovations[right]) {
                value = value
                    .checked_add(i128::from(a).checked_mul(i128::from(b)).ok_or_else(|| {
                        arithmetic_error("MIX1 sparse dot product overflows i128")
                    })?)
                    .ok_or_else(|| arithmetic_error("MIX1 sparse dot sum overflows i128"))?;
            }
            dots[left][right] = value;
            dots[right][left] = value;
        }
    }
    let mut parents = Vec::with_capacity(channels);
    let mut weights = Vec::with_capacity(channels);
    for channel in 0..channels {
        let snapshot = sparse_row(channel, innovations, &dots)?;
        parents.push(snapshot.0);
        weights.push(snapshot.1);
    }
    Ok((parents, weights))
}

fn sparse_row(
    channel: usize,
    innovations: &[Vec<i64>],
    dots: &[Vec<i128>],
) -> Result<(Vec<usize>, Vec<i16>), OptimumV2Error> {
    let target = &innovations[channel];
    let mut selected: Vec<usize> = Vec::new();
    let mut weights: Vec<i16> = Vec::new();
    let mut best_cost = sparse_cost(target);
    for _ in 0..PARENT_CAP {
        let current_residuals = sparse_residuals(target, innovations, &selected, &weights)?;
        let mut candidates = (0..channel)
            .filter(|parent| !selected.contains(parent))
            .map(|parent| {
                let correlation = innovations[parent]
                    .iter()
                    .zip(&current_residuals)
                    .try_fold(0i128, |sum, (&value, &residual)| {
                        sum.checked_add(i128::from(value).checked_mul(i128::from(residual))?)
                    })
                    .ok_or_else(|| arithmetic_error("MIX1 sparse correlation overflows i128"));
                correlation.map(|value| (parent, value.unsigned_abs()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        candidates.sort_by_key(|&(parent, magnitude)| (std::cmp::Reverse(magnitude), parent));
        candidates.truncate(CANDIDATE_LIMIT);
        let mut best: Option<(u64, usize, Vec<i16>)> = None;
        for (parent, _) in candidates {
            let mut candidate_parents = selected.clone();
            candidate_parents.push(parent);
            let initial = portable_lsq(&candidate_parents, channel, dots)?;
            let (cost, fitted) =
                coordinate_descent(target, innovations, &candidate_parents, &initial)?;
            let candidate = (cost, parent, fitted);
            if best.as_ref().map_or(true, |current| candidate < *current) {
                best = Some(candidate);
            }
        }
        let Some((cost, parent, fitted)) = best else {
            break;
        };
        if cost >= best_cost {
            break;
        }
        best_cost = cost;
        selected.push(parent);
        weights = fitted;
    }
    Ok((selected, weights))
}

fn portable_lsq(
    parents: &[usize],
    channel: usize,
    dots: &[Vec<i128>],
) -> Result<Vec<i16>, OptimumV2Error> {
    match portable_lsq_checked(parents, channel, dots) {
        Ok(weights) => Ok(weights),
        Err(_) => gauss_seidel_lsq(parents, channel, dots),
    }
}

fn portable_lsq_checked(
    parents: &[usize],
    channel: usize,
    dots: &[Vec<i128>],
) -> Result<Vec<i16>, OptimumV2Error> {
    let width = parents.len();
    let scale = 1i128 << LSQ_FRACTIONAL_BITS;
    let mut augmented = Vec::with_capacity(width);
    for &parent in parents {
        let row_scale = parents
            .iter()
            .map(|&other| dots[parent][other].unsigned_abs())
            .max()
            .unwrap_or(0);
        if row_scale == 0 {
            augmented.push(vec![0i128; width + 1]);
            continue;
        }
        let row_scale = i128::try_from(row_scale)
            .map_err(|_| arithmetic_error("MIX1 LSQ row scale exceeds i128"))?;
        let mut row = Vec::with_capacity(width + 1);
        for &other in parents {
            let numerator = dots[parent][other]
                .checked_mul(scale)
                .ok_or_else(|| arithmetic_error("MIX1 LSQ matrix overflows i128"))?;
            row.push(round_signed_ratio_even(numerator, row_scale)?);
        }
        let rhs = dots[parent][channel]
            .checked_mul(256)
            .and_then(|value| value.checked_mul(scale))
            .ok_or_else(|| arithmetic_error("MIX1 LSQ right side overflows i128"))?;
        row.push(round_signed_ratio_even(rhs, row_scale)?);
        augmented.push(row);
    }

    let mut pivot_row = 0usize;
    let mut pivot_columns = Vec::new();
    for column in 0..width {
        let pivot = (pivot_row..width)
            .max_by(|&left, &right| {
                let left_key = (
                    augmented[left][column].unsigned_abs(),
                    std::cmp::Reverse(left),
                );
                let right_key = (
                    augmented[right][column].unsigned_abs(),
                    std::cmp::Reverse(right),
                );
                left_key.cmp(&right_key)
            })
            .ok_or_else(|| arithmetic_error("MIX1 LSQ pivot set is empty"))?;
        if augmented[pivot][column] == 0 {
            continue;
        }
        augmented.swap(pivot_row, pivot);
        let divisor = augmented[pivot_row][column];
        for value in &mut augmented[pivot_row] {
            *value = round_signed_ratio_even(
                value
                    .checked_mul(scale)
                    .ok_or_else(|| arithmetic_error("MIX1 LSQ normalization overflows i128"))?,
                divisor,
            )?;
        }
        let pivot_values = augmented[pivot_row].clone();
        for (row_index, row) in augmented.iter_mut().enumerate().take(width) {
            if row_index == pivot_row {
                continue;
            }
            let factor = row[column];
            if factor == 0 {
                continue;
            }
            for index in 0..=width {
                let correction = round_signed_ratio_even(
                    factor
                        .checked_mul(pivot_values[index])
                        .ok_or_else(|| arithmetic_error("MIX1 LSQ elimination overflows i128"))?,
                    scale,
                )?;
                row[index] = row[index]
                    .checked_sub(correction)
                    .ok_or_else(|| arithmetic_error("MIX1 LSQ elimination overflows i128"))?;
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == width {
            break;
        }
    }
    let mut solution = vec![0i16; width];
    for (row, &column) in pivot_columns.iter().enumerate() {
        solution[column] = clamp_weight(round_signed_ratio_even(augmented[row][width], scale)?);
    }
    Ok(solution)
}

fn gauss_seidel_lsq(
    parents: &[usize],
    channel: usize,
    dots: &[Vec<i128>],
) -> Result<Vec<i16>, OptimumV2Error> {
    let mut weights = vec![0i16; parents.len()];
    for _ in 0..LSQ_FALLBACK_SWEEPS {
        for (index, &parent) in parents.iter().enumerate() {
            let denominator = dots[parent][parent];
            if denominator == 0 {
                weights[index] = 0;
                continue;
            }
            let mut numerator = dots[parent][channel]
                .checked_mul(256)
                .ok_or_else(|| arithmetic_error("MIX1 fallback LSQ numerator overflows i128"))?;
            for (other_index, &other_parent) in parents.iter().enumerate() {
                if other_index != index {
                    let term = dots[parent][other_parent]
                        .checked_mul(i128::from(weights[other_index]))
                        .ok_or_else(|| {
                            arithmetic_error("MIX1 fallback LSQ product overflows i128")
                        })?;
                    numerator = numerator.checked_sub(term).ok_or_else(|| {
                        arithmetic_error("MIX1 fallback LSQ numerator overflows i128")
                    })?;
                }
            }
            weights[index] = clamp_weight(round_ratio_even(numerator, denominator)?);
        }
    }
    Ok(weights)
}

fn coordinate_descent(
    target: &[i64],
    innovations: &[Vec<i64>],
    parents: &[usize],
    initial: &[i16],
) -> Result<(u64, Vec<i16>), OptimumV2Error> {
    let mut weights = initial.to_vec();
    let mut weighted = vec![0i128; target.len()];
    for (index, &parent) in parents.iter().enumerate() {
        for (time, &source) in innovations[parent].iter().enumerate() {
            weighted[time] = weighted[time]
                .checked_add(i128::from(weights[index]) * i128::from(source))
                .ok_or_else(|| arithmetic_error("MIX1 coordinate prediction overflows i128"))?;
        }
    }
    let mut best_cost = target
        .iter()
        .zip(&weighted)
        .try_fold(0u64, |sum, (&value, &q8)| {
            let residual = i128::from(value) - round_q8(q8);
            sum.checked_add(code_cost(residual)?)
        })
        .ok_or_else(|| arithmetic_error("MIX1 sparse cost overflows u64"))?;

    for step in LOCAL_STEPS {
        for _ in 0..COORDINATE_PASS_LIMIT {
            let mut changed = false;
            for index in 0..weights.len() {
                let current = weights[index];
                let candidates = [
                    (i32::from(current) - i32::from(step)).max(-1024) as i16,
                    current,
                    (i32::from(current) + i32::from(step)).min(1024) as i16,
                ];
                let mut best = (best_cost, current);
                for candidate in candidates {
                    let delta = i128::from(candidate) - i128::from(current);
                    let cost = if delta == 0 {
                        best_cost
                    } else {
                        target
                            .iter()
                            .enumerate()
                            .try_fold(0u64, |sum, (time, &value)| {
                                let prediction = weighted[time].checked_add(
                                    delta * i128::from(innovations[parents[index]][time]),
                                )?;
                                sum.checked_add(code_cost(
                                    i128::from(value) - round_q8(prediction),
                                )?)
                            })
                            .ok_or_else(|| arithmetic_error("MIX1 coordinate cost overflows"))?
                    };
                    if (cost, candidate) < best {
                        best = (cost, candidate);
                    }
                }
                if best.0 < best_cost || best.1 != current {
                    let delta = i128::from(best.1) - i128::from(current);
                    best_cost = best.0;
                    weights[index] = best.1;
                    for (time, value) in weighted.iter_mut().enumerate() {
                        *value = value
                            .checked_add(delta * i128::from(innovations[parents[index]][time]))
                            .ok_or_else(|| arithmetic_error("MIX1 coordinate update overflows"))?;
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    Ok((best_cost, weights))
}

fn sparse_residuals(
    target: &[i64],
    innovations: &[Vec<i64>],
    parents: &[usize],
    weights: &[i16],
) -> Result<Vec<i64>, OptimumV2Error> {
    let mut result = Vec::with_capacity(target.len());
    for (time, &value) in target.iter().enumerate() {
        let mut weighted = 0i128;
        for (index, &parent) in parents.iter().enumerate() {
            weighted = weighted
                .checked_add(i128::from(weights[index]) * i128::from(innovations[parent][time]))
                .ok_or_else(|| arithmetic_error("MIX1 sparse prediction overflows i128"))?;
        }
        result.push(
            i64::try_from(i128::from(value) - round_q8(weighted))
                .map_err(|_| arithmetic_error("MIX1 sparse residual exceeds i64"))?,
        );
    }
    Ok(result)
}

fn sparse_cost(values: &[i64]) -> u64 {
    values
        .iter()
        .map(|&value| code_cost(i128::from(value)).expect("i64 code cost is bounded"))
        .sum()
}

fn code_cost(value: i128) -> Option<u64> {
    let zigzag = if value >= 0 {
        u128::try_from(value).ok()?.checked_mul(2)?
    } else {
        value.unsigned_abs().checked_mul(2)?.checked_sub(1)?
    };
    Some(u64::from(
        u128::BITS - zigzag.saturating_add(1).leading_zeros(),
    ))
}

fn validate_signal(signal: &[Vec<i64>]) -> Result<(usize, usize), OptimumV2Error> {
    let channels = signal.len();
    if !(1..=256).contains(&channels) || signal[0].is_empty() {
        return Err(input_error("MIX1 lattice dimensions are invalid"));
    }
    let samples = signal[0].len();
    if samples > 32_768
        || channels
            .checked_mul(samples)
            .map_or(true, |values| values > 131_072)
        || signal.iter().any(|row| row.len() != samples)
        || signal
            .iter()
            .flatten()
            .any(|&value| i32::try_from(value).is_err())
    {
        return Err(input_error(
            "MIX1 lattice dimensions or samples are invalid",
        ));
    }
    Ok((channels, samples))
}

fn validate_side(side: &LatticeSide) -> Result<(), OptimumV2Error> {
    let channels = side.parents.len();
    if !(1..=256).contains(&channels) || side.weights_q8.len() != channels {
        return Err(input_error("MIX1 side graph dimensions are invalid"));
    }
    if side
        .coefficients
        .iter()
        .any(|&value| i128::from(value).unsigned_abs() >= Q_ONE as u128)
    {
        return Err(input_error("MIX1 lattice coefficient is unstable"));
    }
    for channel in 0..channels {
        let parents = &side.parents[channel];
        let weights = &side.weights_q8[channel];
        if parents.len() > PARENT_CAP.min(channel)
            || parents.len() != weights.len()
            || !parents.windows(2).all(|pair| pair[0] < pair[1])
            || parents.iter().any(|&parent| parent >= channel)
            || weights
                .iter()
                .any(|weight| !(WEIGHT_MIN..=WEIGHT_MAX).contains(weight))
        {
            return Err(input_error("MIX1 sparse graph is noncanonical"));
        }
    }
    Ok(())
}

pub(crate) fn round_q15(product: i128) -> i128 {
    let magnitude = (product.unsigned_abs() + (Q_ONE as u128 / 2)) / Q_ONE as u128;
    if product >= 0 {
        magnitude as i128
    } else {
        -(magnitude as i128)
    }
}

pub(crate) fn round_q8(value: i128) -> i128 {
    let magnitude = (value.unsigned_abs() + 128) / 256;
    if value >= 0 {
        magnitude as i128
    } else {
        -(magnitude as i128)
    }
}

fn round_ratio_even(numerator: i128, denominator: i128) -> Result<i128, OptimumV2Error> {
    if denominator <= 0 {
        return Err(arithmetic_error(
            "MIX1 rounding denominator must be positive",
        ));
    }
    let magnitude = numerator.unsigned_abs();
    let denominator = denominator as u128;
    let mut quotient = magnitude / denominator;
    let remainder = magnitude % denominator;
    let doubled = remainder
        .checked_mul(2)
        .ok_or_else(|| arithmetic_error("MIX1 rounding remainder overflows"))?;
    if doubled > denominator || (doubled == denominator && quotient % 2 == 1) {
        quotient += 1;
    }
    let quotient = i128::try_from(quotient)
        .map_err(|_| arithmetic_error("MIX1 rounded value exceeds i128"))?;
    Ok(if numerator >= 0 { quotient } else { -quotient })
}

fn round_signed_ratio_even(
    mut numerator: i128,
    mut denominator: i128,
) -> Result<i128, OptimumV2Error> {
    if denominator == 0 {
        return Err(arithmetic_error("MIX1 rounding denominator is zero"));
    }
    if denominator < 0 {
        numerator = numerator
            .checked_neg()
            .ok_or_else(|| arithmetic_error("MIX1 signed numerator cannot be negated"))?;
        denominator = -denominator;
    }
    round_ratio_even(numerator, denominator)
}

fn clamp_weight(value: i128) -> i16 {
    value.clamp(i128::from(WEIGHT_MIN), i128::from(WEIGHT_MAX)) as i16
}

fn zigzag(value: i64) -> u64 {
    if value >= 0 {
        value as u64 * 2
    } else {
        value.unsigned_abs() * 2 - 1
    }
}

fn unzigzag(value: u64) -> Result<i64, OptimumV2Error> {
    if value % 2 == 0 {
        i64::try_from(value / 2).map_err(|_| packet_error("MIX1 ZigZag value exceeds i64"))
    } else {
        let magnitude = i128::from(value / 2) + 1;
        i64::try_from(-magnitude).map_err(|_| packet_error("MIX1 ZigZag value exceeds i64"))
    }
}

fn colex_rank(parents: &[usize]) -> Result<u64, OptimumV2Error> {
    parents
        .iter()
        .enumerate()
        .try_fold(0u64, |sum, (index, &parent)| {
            sum.checked_add(combination(parent, index + 1).ok()?)
        })
        .ok_or_else(|| input_error("MIX1 colex rank overflows"))
}

fn colex_unrank(rank: u64, channels: usize, count: usize) -> Result<Vec<usize>, OptimumV2Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut remainder = rank;
    let mut parents = vec![0usize; count];
    let mut upper = channels - 1;
    for choose in (1..=count).rev() {
        let mut parent = upper;
        while parent >= choose - 1 && combination(parent, choose)? > remainder {
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
        if parent < choose - 1 {
            return Err(packet_error("MIX1 colex rank is invalid"));
        }
        parents[choose - 1] = parent;
        remainder -= combination(parent, choose)?;
        upper = parent.saturating_sub(1);
    }
    if remainder != 0 || colex_rank(&parents).map_err(as_packet_error)? != rank {
        return Err(packet_error("MIX1 colex rank is noncanonical"));
    }
    Ok(parents)
}

fn combination(n: usize, k: usize) -> Result<u64, OptimumV2Error> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for index in 1..=k {
        result = result
            .checked_mul((n - k + index) as u128)
            .ok_or_else(|| arithmetic_error("MIX1 combination overflows"))?
            / index as u128;
    }
    u64::try_from(result).map_err(|_| arithmetic_error("MIX1 combination exceeds u64"))
}

fn bit_length(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn write_bit(&mut self, value: u8) -> Result<(), OptimumV2Error> {
        if value > 1 {
            return Err(input_error("MIX1 compact side bit is not binary"));
        }
        self.current = (self.current << 1) | value;
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
        Ok(())
    }

    fn write_bits(&mut self, value: u64, width: u8) -> Result<(), OptimumV2Error> {
        if width < 64 && value >= (1u64 << width) {
            return Err(input_error("MIX1 compact side bit field exceeds width"));
        }
        for shift in (0..width).rev() {
            self.write_bit(((value >> shift) & 1) as u8)?;
        }
        Ok(())
    }

    fn write_rice(&mut self, value: u64, k: u8) -> Result<(), OptimumV2Error> {
        let quotient = value >> k;
        for _ in 0..quotient {
            self.write_bit(1)?;
        }
        self.write_bit(0)?;
        self.write_bits(value & ((1u64 << k) - 1), k)
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 {
            self.bytes.push(self.current << (8 - self.used));
        }
        self.bytes
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() * 8 - self.position
    }

    fn read_bit(&mut self) -> Result<u8, OptimumV2Error> {
        if self.position >= self.data.len() * 8 {
            return Err(packet_error("MIX1 compact side information is truncated"));
        }
        let byte = self.data[self.position / 8];
        let value = (byte >> (7 - self.position % 8)) & 1;
        self.position += 1;
        Ok(value)
    }

    fn read_bits(&mut self, width: u8) -> Result<u64, OptimumV2Error> {
        if usize::from(width) > self.remaining() {
            return Err(packet_error("MIX1 compact side bit field is truncated"));
        }
        let mut value = 0u64;
        for _ in 0..width {
            value = (value << 1) | u64::from(self.read_bit()?);
        }
        Ok(value)
    }

    fn read_rice(&mut self, k: u8, maximum: u64) -> Result<u64, OptimumV2Error> {
        let mut quotient = 0u64;
        let maximum_quotient = maximum >> k;
        while self.read_bit()? == 1 {
            quotient += 1;
            if quotient > maximum_quotient {
                return Err(packet_error("MIX1 compact Rice value exceeds bound"));
            }
        }
        let value = (quotient << k) | self.read_bits(k)?;
        if value > maximum {
            return Err(packet_error("MIX1 compact Rice value exceeds bound"));
        }
        Ok(value)
    }

    fn finish_zero_padding(&mut self) -> Result<(), OptimumV2Error> {
        if self.remaining() >= 8 {
            return Err(packet_error("MIX1 compact side has trailing bytes"));
        }
        while self.remaining() != 0 {
            if self.read_bit()? != 0 {
                return Err(packet_error("MIX1 compact side has nonzero padding"));
            }
        }
        Ok(())
    }
}

fn input_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

fn arithmetic_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn as_packet_error(error: OptimumV2Error) -> OptimumV2Error {
    packet_error(error.to_string())
}
