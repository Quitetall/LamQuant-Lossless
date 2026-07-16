//! Retired mode-`0x14` fixed-universal predictor conformance core.
//!
//! This module intentionally stops at residual generation. It does not select
//! an entropy coder or alter the existing BGF1 native packet path. It is kept
//! only to preserve the completed Python/Rust conformance evidence; standalone
//! BGF1 must not use this predictor or its graph law.

use crate::OptimumV2Error;

pub const FIXED_UNIVERSAL_MODE: u8 = 0x14;
pub const PMAX: usize = 7;
pub const WEIGHT_Q: u32 = 20;
pub const WEIGHT_ONE: i128 = 1i128 << WEIGHT_Q;
pub const WEIGHT_LIMIT: i128 = 16 * WEIGHT_ONE;
pub const COVARIANCE_Q: u32 = 36;
pub const COVARIANCE_ONE: i128 = 1i128 << COVARIANCE_Q;
pub const COVARIANCE_LIMIT: i128 = (1i128 << 42) - 1;
pub const GAIN_Q: u32 = 40;
pub const GAIN_ONE: i128 = 1i128 << GAIN_Q;
pub const GAIN_LIMIT: i128 = 8 * GAIN_ONE;
pub const FORGETTING_NUMERATOR: i128 = 253;
pub const FORGETTING_DENOMINATOR: i128 = 256;

const MAX_CHANNELS: usize = 256;

/// Compact causal one-parent graph used by fixed universal mode `0x14`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedUniversalGraph {
    parents: Vec<Option<u16>>,
}

impl FixedUniversalGraph {
    pub fn new(parents: Vec<Option<u16>>) -> Result<Self, OptimumV2Error> {
        validate_parents(&parents, InputKind::Caller)?;
        Ok(Self { parents })
    }

    pub fn parse(data: &[u8], channels: usize) -> Result<Self, OptimumV2Error> {
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(OptimumV2Error::InvalidPacket(
                "fixed universal graph channel count is outside 1..=256".into(),
            ));
        }
        let expected_len = channels
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                OptimumV2Error::InvalidPacket(
                    "fixed universal graph length calculation overflowed".into(),
                )
            })?;
        if data.len() != expected_len {
            return Err(OptimumV2Error::InvalidPacket(
                "fixed universal graph length does not match channels".into(),
            ));
        }
        if data[0] != FIXED_UNIVERSAL_MODE {
            return Err(OptimumV2Error::Unsupported(format!(
                "graph mode 0x{:02x} is not fixed universal mode 0x14",
                data[0]
            )));
        }
        let mut parents = Vec::with_capacity(channels);
        for channel in 0..channels {
            let offset = 1 + 2 * channel;
            let parent = u16::from_le_bytes([data[offset], data[offset + 1]]);
            parents.push((parent != u16::MAX).then_some(parent));
        }
        validate_parents(&parents, InputKind::Packet)?;
        Ok(Self { parents })
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 2 * self.parents.len());
        out.push(FIXED_UNIVERSAL_MODE);
        for parent in &self.parents {
            out.extend_from_slice(&parent.unwrap_or(u16::MAX).to_le_bytes());
        }
        out
    }

    pub fn parents(&self) -> &[Option<u16>] {
        &self.parents
    }

    pub fn channel_count(&self) -> usize {
        self.parents.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateTrace {
    pub z_q36: Vec<i128>,
    pub denominator_q36_raw2: i128,
    pub gain_q40: Vec<i128>,
    pub error_q20: i128,
    pub reset: bool,
}

/// One independently adapted fixed-point RLS prediction-order expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRlsExpert {
    width: usize,
    scale: i128,
    weights_q20: Vec<i128>,
    covariance_q36: Vec<Vec<i128>>,
    score_q20: i128,
    reset_count: u64,
}

impl FixedRlsExpert {
    pub fn new(width: usize, bit_depth: u8) -> Result<Self, OptimumV2Error> {
        if !(1..=2 * PMAX + 1).contains(&width) {
            return Err(OptimumV2Error::InvalidInput(
                "fixed RLS width is outside the normative bound".into(),
            ));
        }
        if !(1..=32).contains(&bit_depth) {
            return Err(OptimumV2Error::InvalidInput(
                "fixed RLS bit depth must be in 1..=32".into(),
            ));
        }
        let scale_shift = u32::from(bit_depth.saturating_sub(11));
        let scale = 1i128
            .checked_shl(scale_shift)
            .ok_or_else(|| arithmetic_error("fixed RLS scale exceeds signed i128"))?;
        Ok(Self {
            width,
            scale,
            weights_q20: vec![0; width],
            covariance_q36: identity_covariance(width),
            score_q20: 0,
            reset_count: 0,
        })
    }

    pub fn dot_q20(&self, features: &[i64]) -> Result<i128, OptimumV2Error> {
        self.validate_features(features)?;
        let mut total = 0i128;
        for (&weight, &feature) in self.weights_q20.iter().zip(features) {
            let product = checked_mul(weight, i128::from(feature), "fixed RLS prediction product")?;
            total = checked_add(total, product, "fixed RLS prediction sum")?;
        }
        Ok(total)
    }

    pub fn prediction(&self, features: &[i64]) -> Result<i64, OptimumV2Error> {
        let prediction =
            round_ratio_away(self.dot_q20(features)?, WEIGHT_ONE, "fixed RLS prediction")?;
        i64::try_from(prediction)
            .map_err(|_| arithmetic_error("fixed RLS prediction exceeds signed i64"))
    }

    pub fn observe(
        &mut self,
        features: &[i64],
        sample: i64,
    ) -> Result<UpdateTrace, OptimumV2Error> {
        let dot_q20 = self.dot_q20(features)?;
        self.observe_with_dot_q20(features, sample, dot_q20)
    }

    fn observe_with_dot_q20(
        &mut self,
        features: &[i64],
        sample: i64,
        dot_q20: i128,
    ) -> Result<UpdateTrace, OptimumV2Error> {
        self.validate_features(features)?;
        let sample_q20 = checked_mul(
            i128::from(sample),
            WEIGHT_ONE,
            "fixed RLS coefficient sample",
        )?;
        let error_q20 = checked_sub(sample_q20, dot_q20, "fixed RLS coefficient error")?;

        let mut z_q36 = Vec::with_capacity(self.width);
        for row in &self.covariance_q36 {
            let mut value = 0i128;
            for (&covariance, &feature) in row.iter().zip(features) {
                let product = checked_mul(
                    covariance,
                    i128::from(feature),
                    "fixed RLS covariance product",
                )?;
                value = checked_add(value, product, "fixed RLS covariance sum")?;
            }
            z_q36.push(value);
        }

        let scale_squared = checked_mul(self.scale, self.scale, "fixed RLS scale square")?;
        let regularizer = checked_mul(
            FORGETTING_NUMERATOR,
            COVARIANCE_ONE,
            "fixed RLS regularizer covariance",
        )?;
        let regularizer = checked_mul(regularizer, scale_squared, "fixed RLS regularizer scale")?
            / FORGETTING_DENOMINATOR;
        let mut denominator = regularizer;
        for (&feature, &value) in features.iter().zip(&z_q36) {
            let product = checked_mul(i128::from(feature), value, "fixed RLS denominator product")?;
            denominator = checked_add(denominator, product, "fixed RLS denominator sum")?;
        }

        let mut gain_q40 = Vec::new();
        let mut reset = denominator <= 0;
        if !reset {
            gain_q40.reserve(self.width);
            for &value in &z_q36 {
                let numerator = checked_mul(value, GAIN_ONE, "fixed RLS gain numerator")?;
                let item = round_ratio_away(numerator, denominator, "fixed RLS gain")?;
                gain_q40.push(item);
                if checked_abs(item, "fixed RLS gain magnitude")? > GAIN_LIMIT {
                    reset = true;
                }
            }
        }

        let mut next_weights = self.weights_q20.clone();
        let mut next_covariance = self.covariance_q36.clone();
        if !reset {
            let mut candidate_weights = Vec::with_capacity(self.width);
            for (&weight, &gain) in self.weights_q20.iter().zip(&gain_q40) {
                let numerator = checked_mul(gain, error_q20, "fixed RLS weight update product")?;
                let adjustment =
                    round_ratio_away(numerator, GAIN_ONE, "fixed RLS weight adjustment")?;
                let updated = checked_add(weight, adjustment, "fixed RLS weight update")?;
                candidate_weights.push(updated.clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT));
            }

            let mut candidate_covariance = vec![vec![0i128; self.width]; self.width];
            for row in 0..self.width {
                for column in row..self.width {
                    let left = checked_mul(
                        gain_q40[row],
                        z_q36[column],
                        "fixed RLS covariance left product",
                    )?;
                    let right = checked_mul(
                        gain_q40[column],
                        z_q36[row],
                        "fixed RLS covariance right product",
                    )?;
                    let symmetric_sum =
                        checked_add(left, right, "fixed RLS covariance symmetric sum")?;
                    let symmetric = round_ratio_away(
                        symmetric_sum,
                        2 * GAIN_ONE,
                        "fixed RLS covariance symmetric term",
                    )?;
                    let difference = checked_sub(
                        self.covariance_q36[row][column],
                        symmetric,
                        "fixed RLS covariance update difference",
                    )?;
                    let numerator = checked_mul(
                        difference,
                        FORGETTING_DENOMINATOR,
                        "fixed RLS covariance update numerator",
                    )?;
                    let updated = round_ratio_away(
                        numerator,
                        FORGETTING_NUMERATOR,
                        "fixed RLS covariance update",
                    )?;
                    candidate_covariance[row][column] = updated;
                    candidate_covariance[column][row] = updated;
                }
            }
            reset = (0..self.width).any(|index| candidate_covariance[index][index] <= 0);
            if !reset {
                'bounds: for row in &candidate_covariance {
                    for &value in row {
                        if checked_abs(value, "fixed RLS covariance magnitude")? > COVARIANCE_LIMIT
                        {
                            reset = true;
                            break 'bounds;
                        }
                    }
                }
            }
            if !reset {
                next_weights = candidate_weights;
                next_covariance = candidate_covariance;
            }
        }

        let next_reset_count = if reset {
            next_weights = vec![0; self.width];
            next_covariance = identity_covariance(self.width);
            self.reset_count.checked_add(1).ok_or_else(|| {
                arithmetic_error("fixed RLS reset count exceeds unsigned 64-bit range")
            })?
        } else {
            self.reset_count
        };
        let discounted_numerator = checked_mul(
            FORGETTING_NUMERATOR,
            self.score_q20,
            "fixed RLS score discount",
        )?;
        let discounted = round_ratio_away(
            discounted_numerator,
            FORGETTING_DENOMINATOR,
            "fixed RLS discounted score",
        )?;
        let next_score = checked_add(
            discounted,
            checked_abs(error_q20, "fixed RLS absolute coefficient error")?,
            "fixed RLS score",
        )?;

        self.weights_q20 = next_weights;
        self.covariance_q36 = next_covariance;
        self.reset_count = next_reset_count;
        self.score_q20 = next_score;
        Ok(UpdateTrace {
            z_q36,
            denominator_q36_raw2: denominator,
            gain_q40,
            error_q20,
            reset,
        })
    }

    pub fn weights_q20(&self) -> &[i128] {
        &self.weights_q20
    }

    pub fn covariance_q36(&self) -> &[Vec<i128>] {
        &self.covariance_q36
    }

    pub fn score_q20(&self) -> i128 {
        self.score_q20
    }

    pub fn reset_count(&self) -> u64 {
        self.reset_count
    }

    fn validate_features(&self, features: &[i64]) -> Result<(), OptimumV2Error> {
        if features.len() != self.width {
            return Err(OptimumV2Error::InvalidInput(
                "fixed RLS feature width mismatch".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PredictionTicket {
    feature_rows: Vec<Vec<i64>>,
    dots_q20: Vec<i128>,
    prediction: i64,
    current: Vec<i64>,
}

/// Adaptive predictor state shared by an encoder and decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniversalSession {
    graph: FixedUniversalGraph,
    history: Vec<Vec<i64>>,
    states: Vec<Vec<FixedRlsExpert>>,
    tickets: Vec<Option<PredictionTicket>>,
    row_samples: Vec<Option<i64>>,
    next_channel: usize,
    sample_min: i64,
    sample_max: i64,
}

impl UniversalSession {
    pub fn new(graph: FixedUniversalGraph, bit_depth: u8) -> Result<Self, OptimumV2Error> {
        if !(1..=32).contains(&bit_depth) {
            return Err(OptimumV2Error::InvalidInput(
                "universal session bit depth must be in 1..=32".into(),
            ));
        }
        let magnitude = 1i64 << (bit_depth - 1);
        let mut states = Vec::with_capacity(graph.channel_count());
        for parent in graph.parents() {
            let mut channel_states = Vec::with_capacity(PMAX + 1);
            for order in 0..=PMAX {
                let width = if parent.is_none() {
                    order + 1
                } else {
                    2 * order + 1
                };
                channel_states.push(FixedRlsExpert::new(width, bit_depth)?);
            }
            states.push(channel_states);
        }
        let channels = graph.channel_count();
        Ok(Self {
            graph,
            history: vec![vec![0; PMAX + 1]; channels],
            states,
            tickets: vec![None; channels],
            row_samples: vec![None; channels],
            next_channel: 0,
            sample_min: -magnitude,
            sample_max: magnitude - 1,
        })
    }

    pub fn prediction(&mut self, channel: usize, current: &[i64]) -> Result<i64, OptimumV2Error> {
        self.validate_channel_and_row(channel, current)?;
        if channel != self.next_channel {
            return Err(OptimumV2Error::InvalidInput(format!(
                "universal prediction expected channel {}, got {channel}",
                self.next_channel
            )));
        }
        if self.tickets[channel].is_some() {
            return Err(OptimumV2Error::InvalidInput(
                "universal prediction ticket is already open".into(),
            ));
        }
        let mut feature_rows = Vec::with_capacity(PMAX + 1);
        let mut dots_q20 = Vec::with_capacity(PMAX + 1);
        for order in 0..=PMAX {
            let features = self.features(channel, current, order);
            let dot = self.states[channel][order].dot_q20(&features)?;
            feature_rows.push(features);
            dots_q20.push(dot);
        }
        let mut selected = 0usize;
        for order in 1..=PMAX {
            let candidate = self.states[channel][order].score_q20();
            let best = self.states[channel][selected].score_q20();
            if candidate < best {
                selected = order;
            }
        }
        let prediction_q0 =
            round_ratio_away(dots_q20[selected], WEIGHT_ONE, "fixed universal prediction")?;
        let prediction = i64::try_from(prediction_q0)
            .map_err(|_| arithmetic_error("fixed universal prediction exceeds signed i64"))?;
        self.tickets[channel] = Some(PredictionTicket {
            feature_rows,
            dots_q20,
            prediction,
            current: current.to_vec(),
        });
        Ok(prediction)
    }

    pub fn observe(
        &mut self,
        channel: usize,
        current: &[i64],
        sample: i64,
        prediction: i64,
    ) -> Result<(), OptimumV2Error> {
        self.validate_channel_and_row(channel, current)?;
        if channel != self.next_channel {
            return Err(OptimumV2Error::InvalidInput(format!(
                "universal observation expected channel {}, got {channel}",
                self.next_channel
            )));
        }
        let ticket = self.tickets[channel].clone().ok_or_else(|| {
            OptimumV2Error::InvalidInput("universal observation has no prediction ticket".into())
        })?;
        if prediction != ticket.prediction {
            return Err(OptimumV2Error::InvalidInput(
                "universal observation prediction does not match ticket".into(),
            ));
        }
        if current != ticket.current {
            return Err(OptimumV2Error::InvalidInput(
                "universal observation row does not match prediction ticket".into(),
            ));
        }
        self.validate_sample(sample, "universal observed sample")?;
        let mut next_states = self.states[channel].clone();
        for ((expert, features), dot_q20) in next_states
            .iter_mut()
            .zip(&ticket.feature_rows)
            .zip(ticket.dots_q20)
        {
            expert.observe_with_dot_q20(features, sample, dot_q20)?;
        }
        let next_channel = self.next_channel.checked_add(1).ok_or_else(|| {
            OptimumV2Error::InvalidInput("universal completed-channel counter overflowed".into())
        })?;
        self.states[channel] = next_states;
        self.tickets[channel] = None;
        self.row_samples[channel] = Some(sample);
        self.next_channel = next_channel;
        Ok(())
    }

    pub fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        if current.len() != self.states.len() {
            return Err(OptimumV2Error::InvalidInput(
                "universal time row has the wrong channel count".into(),
            ));
        }
        if self.tickets.iter().any(Option::is_some) {
            return Err(OptimumV2Error::InvalidInput(
                "cannot finish universal time with an open ticket".into(),
            ));
        }
        if self.next_channel != self.states.len() {
            return Err(OptimumV2Error::InvalidInput(format!(
                "cannot finish universal time after completed {} of {} channels",
                self.next_channel,
                self.states.len()
            )));
        }
        let observed = self
            .row_samples
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                OptimumV2Error::InvalidInput(
                    "cannot finish universal time with an unobserved channel".into(),
                )
            })?;
        if observed != current {
            return Err(OptimumV2Error::InvalidInput(
                "universal completed row does not match observed samples".into(),
            ));
        }
        for (history, sample) in self.history.iter_mut().zip(observed) {
            history.copy_within(0..PMAX, 1);
            history[0] = sample;
        }
        self.row_samples.fill(None);
        self.next_channel = 0;
        Ok(())
    }

    pub fn reset_count(&self) -> Result<u64, OptimumV2Error> {
        self.states
            .iter()
            .flatten()
            .try_fold(0u64, |total, expert| {
                total.checked_add(expert.reset_count()).ok_or_else(|| {
                    arithmetic_error("universal reset count exceeds unsigned 64-bit range")
                })
            })
    }

    pub fn graph(&self) -> &FixedUniversalGraph {
        &self.graph
    }

    fn features(&self, channel: usize, current: &[i64], order: usize) -> Vec<i64> {
        match self.graph.parents[channel] {
            None => self.history[channel][..=order].to_vec(),
            Some(parent) => {
                let parent = usize::from(parent);
                let mut features = Vec::with_capacity(2 * order + 1);
                features.extend_from_slice(&self.history[channel][..order]);
                features.push(current[parent]);
                features.extend_from_slice(&self.history[parent][..order]);
                features
            }
        }
    }

    fn validate_channel_and_row(
        &self,
        channel: usize,
        current: &[i64],
    ) -> Result<(), OptimumV2Error> {
        if channel >= self.states.len() {
            return Err(OptimumV2Error::InvalidInput(
                "universal session channel is out of range".into(),
            ));
        }
        if current.len() != self.states.len() {
            return Err(OptimumV2Error::InvalidInput(
                "universal current row has the wrong channel count".into(),
            ));
        }
        for (index, &value) in current.iter().enumerate() {
            self.validate_sample(value, "universal current-row sample")?;
            if index < self.next_channel && self.row_samples[index] != Some(value) {
                return Err(OptimumV2Error::InvalidInput(
                    "universal current row disagrees with an observed prefix".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_sample(&self, sample: i64, label: &str) -> Result<(), OptimumV2Error> {
        if sample < self.sample_min || sample > self.sample_max {
            return Err(OptimumV2Error::InvalidInput(format!(
                "{label} exceeds declared bit depth"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum InputKind {
    Caller,
    Packet,
}

fn validate_parents(parents: &[Option<u16>], kind: InputKind) -> Result<(), OptimumV2Error> {
    let invalid_count = !(1..=MAX_CHANNELS).contains(&parents.len());
    let invalid_parent = parents
        .iter()
        .enumerate()
        .any(|(channel, parent)| parent.is_some_and(|parent| usize::from(parent) >= channel));
    if invalid_count || invalid_parent {
        let message = "fixed universal graph must have 1..=256 causal one-parent rows".into();
        return Err(match kind {
            InputKind::Caller => OptimumV2Error::InvalidInput(message),
            InputKind::Packet => OptimumV2Error::InvalidPacket(message),
        });
    }
    Ok(())
}

fn identity_covariance(width: usize) -> Vec<Vec<i128>> {
    (0..width)
        .map(|row| {
            (0..width)
                .map(|column| if row == column { COVARIANCE_ONE } else { 0 })
                .collect()
        })
        .collect()
}

fn round_ratio_away(
    numerator: i128,
    denominator: i128,
    label: &str,
) -> Result<i128, OptimumV2Error> {
    if denominator <= 0 {
        return Err(OptimumV2Error::InvalidInput(format!(
            "{label} denominator must be positive"
        )));
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let remainder_magnitude = checked_abs(remainder, label)?;
    let rounding_threshold = checked_add(
        denominator / 2,
        denominator % 2,
        "fixed RLS rounding threshold",
    )?;
    if remainder_magnitude < rounding_threshold {
        return Ok(quotient);
    }
    checked_add(quotient, if numerator >= 0 { 1 } else { -1 }, label)
}

fn checked_add(left: i128, right: i128, label: &str) -> Result<i128, OptimumV2Error> {
    left.checked_add(right)
        .ok_or_else(|| arithmetic_error(&format!("{label} exceeds signed i128")))
}

fn checked_sub(left: i128, right: i128, label: &str) -> Result<i128, OptimumV2Error> {
    left.checked_sub(right)
        .ok_or_else(|| arithmetic_error(&format!("{label} exceeds signed i128")))
}

fn checked_mul(left: i128, right: i128, label: &str) -> Result<i128, OptimumV2Error> {
    left.checked_mul(right)
        .ok_or_else(|| arithmetic_error(&format!("{label} exceeds signed i128")))
}

fn checked_abs(value: i128, label: &str) -> Result<i128, OptimumV2Error> {
    value
        .checked_abs()
        .ok_or_else(|| arithmetic_error(&format!("{label} exceeds signed i128")))
}

fn arithmetic_error(message: &str) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_is_atomic_when_a_later_expert_fails() {
        let graph = FixedUniversalGraph::new(vec![None]).unwrap();
        let mut session = UniversalSession::new(graph, 16).unwrap();
        session.states[0][1].score_q20 = i128::MAX;
        let current = [0i64];
        let prediction = session.prediction(0, &current).unwrap();
        let before = session.clone();

        assert!(session.observe(0, &current, 7, prediction).is_err());
        assert_eq!(session, before);
        assert!(session.observe(0, &current, 7, prediction).is_err());
        assert_eq!(session, before);
    }
}
