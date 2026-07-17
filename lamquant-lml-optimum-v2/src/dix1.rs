//! Wire-free DIX1 causal structural prototype.
//!
//! This module proves only the reversible derivation-incidence sample loop. It
//! deliberately allocates no packet tag and defines no normative CDF, rANS
//! stream, model pack, or release claim. Encoder and decoder run the same
//! fixed-integer temporal expert, exact current-row incidence path, temporal-
//! innovation incidence expert, and causal adaptive prediction blend.

use sha2::{Digest, Sha256};

use crate::derivation_incidence::{ChannelIdentity, DerivationIncidence, IncidenceChannel};
use crate::fixed_predictor::FixedRlsExpert;
use crate::OptimumV2Error;

const TEMPORAL_LAG_MS: [u32; 4] = [1, 4, 16, 64];
const MAX_SAMPLE_RATE_MHZ: u32 = 4_000_000;
const MAX_HISTORY: usize = 256;
const EXPERT_COUNT: usize = 4;
const DELTA_EXPERT: usize = 0;
const TEMPORAL_EXPERT: usize = 1;
const RAW_INCIDENCE_EXPERT: usize = 2;
const INNOVATION_INCIDENCE_EXPERT: usize = 3;
const SCORE_Q: u64 = 256;
const SCORE_DECAY_SHIFT: u32 = 5;
const MIX_WEIGHT_NUMERATOR: u64 = 1 << 20;
const INCIDENCE_PRIOR_COST: u64 = 64 * SCORE_Q;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dix1IncidenceMode {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChannelState {
    history: Vec<i64>,
    temporal: FixedRlsExpert,
    expert_scores: [u64; EXPERT_COUNT],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dix1Session {
    incidence: DerivationIncidence,
    incidence_mode: Dix1IncidenceMode,
    states: Vec<ChannelState>,
    sample_lags: [usize; TEMPORAL_LAG_MS.len()],
    sample_min: i64,
    sample_max: i64,
    rows: u64,
}

impl Dix1Session {
    pub fn new(
        identities: &[ChannelIdentity],
        bit_depth: u8,
        sample_rate_mhz: u32,
    ) -> Result<Self, OptimumV2Error> {
        Self::new_with_incidence_mode(
            identities,
            bit_depth,
            sample_rate_mhz,
            Dix1IncidenceMode::Enabled,
        )
    }

    #[doc(hidden)]
    pub fn new_with_incidence_mode(
        identities: &[ChannelIdentity],
        bit_depth: u8,
        sample_rate_mhz: u32,
        incidence_mode: Dix1IncidenceMode,
    ) -> Result<Self, OptimumV2Error> {
        if !(1..=32).contains(&bit_depth) {
            return Err(invalid("DIX1 bit depth must be in 1..=32"));
        }
        if !(1..=MAX_SAMPLE_RATE_MHZ).contains(&sample_rate_mhz) {
            return Err(invalid(
                "DIX1 sample rate is outside the bounded prototype range",
            ));
        }
        let incidence = DerivationIncidence::build(identities)?;
        let sample_lags = normalized_lags(sample_rate_mhz)?;
        let history_length = *sample_lags
            .iter()
            .max()
            .ok_or_else(|| invalid("DIX1 temporal lag set is empty"))?;
        if history_length == 0 || history_length > MAX_HISTORY {
            return Err(invalid(
                "DIX1 temporal history exceeds its bounded prototype",
            ));
        }
        let magnitude = 1i64 << (bit_depth - 1);
        let mut states = Vec::with_capacity(incidence.channel_count());
        for channel in incidence.channels() {
            let scores = if channel.supports().is_empty() {
                [0; EXPERT_COUNT]
            } else {
                [INCIDENCE_PRIOR_COST, INCIDENCE_PRIOR_COST, 0, 0]
            };
            states.push(ChannelState {
                history: vec![0; history_length],
                temporal: FixedRlsExpert::new(TEMPORAL_LAG_MS.len(), bit_depth)?,
                expert_scores: scores,
            });
        }
        Ok(Self {
            incidence,
            incidence_mode,
            states,
            sample_lags,
            sample_min: -magnitude,
            sample_max: magnitude - 1,
            rows: 0,
        })
    }

    pub fn incidence(&self) -> &DerivationIncidence {
        &self.incidence
    }

    pub fn incidence_mode(&self) -> Dix1IncidenceMode {
        self.incidence_mode
    }

    pub fn sample_lags(&self) -> [usize; TEMPORAL_LAG_MS.len()] {
        self.sample_lags
    }

    pub fn row_count(&self) -> u64 {
        self.rows
    }

    /// Transform one presented-order sample row into canonical DIX1 residuals.
    /// State is committed only after every channel has validated and updated.
    pub fn forward_row(&mut self, presented: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        let canonical = self.incidence.canonicalize_row(presented)?;
        for &sample in &canonical {
            self.validate_sample(sample)?;
        }
        let mut next = self.clone();
        let residuals = next.process_forward(&canonical)?;
        *self = next;
        Ok(residuals)
    }

    /// Invert one canonical residual row and return the original presentation
    /// order. The decoder evolves exactly the same state as `forward_row`.
    pub fn inverse_row(&mut self, residuals: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        if residuals.len() != self.incidence.channel_count() {
            return Err(invalid("DIX1 residual row has the wrong channel count"));
        }
        let mut next = self.clone();
        let canonical = next.process_inverse(residuals)?;
        let presented = next.incidence.restore_presented_row(&canonical)?;
        *self = next;
        Ok(presented)
    }

    /// Stable digest for cross-implementation state-trace tests. It is not a
    /// packet commitment and is intentionally absent from any wire grammar.
    pub fn state_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"LAMQUANT-DIX1-STATE-PROTOTYPE-V2");
        digest.update([match self.incidence_mode {
            Dix1IncidenceMode::Enabled => 1,
            Dix1IncidenceMode::Disabled => 0,
        }]);
        digest.update(self.rows.to_le_bytes());
        digest.update(self.sample_min.to_le_bytes());
        digest.update(self.sample_max.to_le_bytes());
        for lag in self.sample_lags {
            digest.update((lag as u64).to_le_bytes());
        }
        for channel in self.incidence.channels() {
            digest.update(channel.stable_id().to_le_bytes());
            digest.update((channel.normalized_label().len() as u64).to_le_bytes());
            digest.update(channel.normalized_label().as_bytes());
            for support in channel.supports() {
                digest.update((support.prior_channel as u64).to_le_bytes());
                digest.update(support.coefficient.to_le_bytes());
            }
            digest.update([0xff]);
        }
        for state in &self.states {
            for &sample in &state.history {
                digest.update(sample.to_le_bytes());
            }
            for &weight in state.temporal.weights_q20() {
                digest.update(weight.to_le_bytes());
            }
            for row in state.temporal.covariance_q36() {
                for &value in row {
                    digest.update(value.to_le_bytes());
                }
            }
            digest.update(state.temporal.score_q20().to_le_bytes());
            digest.update(state.temporal.reset_count().to_le_bytes());
            for score in state.expert_scores {
                digest.update(score.to_le_bytes());
            }
        }
        digest.finalize().into()
    }

    fn process_forward(&mut self, canonical: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        let mut reconstructed = Vec::with_capacity(canonical.len());
        let mut temporal_innovations = Vec::with_capacity(canonical.len());
        let mut residuals = Vec::with_capacity(canonical.len());
        for (channel, &sample) in canonical.iter().enumerate() {
            let ticket = self.prediction_ticket(channel, &reconstructed, &temporal_innovations)?;
            residuals.push(checked_sub(
                sample,
                ticket.blended,
                "DIX1 forward residual",
            )?);
            self.observe(channel, sample, &ticket)?;
            reconstructed.push(sample);
            temporal_innovations.push(checked_sub(
                sample,
                ticket.predictions[TEMPORAL_EXPERT],
                "DIX1 temporal innovation",
            )?);
        }
        self.finish_row(&reconstructed)?;
        Ok(residuals)
    }

    fn process_inverse(&mut self, residuals: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        let mut reconstructed = Vec::with_capacity(residuals.len());
        let mut temporal_innovations = Vec::with_capacity(residuals.len());
        for (channel, &residual) in residuals.iter().enumerate() {
            let ticket = self.prediction_ticket(channel, &reconstructed, &temporal_innovations)?;
            let sample = checked_add(ticket.blended, residual, "DIX1 inverse sample")?;
            self.validate_sample(sample)?;
            self.observe(channel, sample, &ticket)?;
            reconstructed.push(sample);
            temporal_innovations.push(checked_sub(
                sample,
                ticket.predictions[TEMPORAL_EXPERT],
                "DIX1 inverse temporal innovation",
            )?);
        }
        self.finish_row(&reconstructed)?;
        Ok(reconstructed)
    }

    fn prediction_ticket(
        &self,
        channel: usize,
        current: &[i64],
        current_innovations: &[i64],
    ) -> Result<PredictionTicket, OptimumV2Error> {
        if channel >= self.states.len()
            || current.len() != channel
            || current_innovations.len() != channel
        {
            return Err(invalid("DIX1 prediction prefix is not strictly causal"));
        }
        let state = &self.states[channel];
        let temporal_features: Vec<i64> = self
            .sample_lags
            .iter()
            .map(|&lag| state.history[lag - 1])
            .collect();
        let temporal = self.clip_prediction(state.temporal.prediction(&temporal_features)?);
        let delta = state.history[0];
        let incidence_channel = &self.incidence.channels()[channel];
        let use_incidence = self.incidence_mode == Dix1IncidenceMode::Enabled
            && !incidence_channel.supports().is_empty();
        let raw_incidence = if use_incidence {
            self.clip_prediction(signed_support_sum(
                incidence_channel,
                current,
                "DIX1 raw incidence",
            )?)
        } else {
            0
        };
        let innovation_adjustment = if use_incidence {
            signed_support_sum(
                incidence_channel,
                current_innovations,
                "DIX1 innovation incidence",
            )?
        } else {
            0
        };
        let innovation_incidence = self.clip_prediction(checked_add(
            temporal,
            innovation_adjustment,
            "DIX1 innovation prediction",
        )?);
        let mut predictions = [0; EXPERT_COUNT];
        predictions[DELTA_EXPERT] = delta;
        predictions[TEMPORAL_EXPERT] = temporal;
        predictions[RAW_INCIDENCE_EXPERT] = raw_incidence;
        predictions[INNOVATION_INCIDENCE_EXPERT] = innovation_incidence;
        let active = if use_incidence {
            [true; EXPERT_COUNT]
        } else {
            [true, true, false, false]
        };
        let blended = blend_predictions(
            predictions,
            state.expert_scores,
            active,
            self.sample_min,
            self.sample_max,
        )?;
        Ok(PredictionTicket {
            temporal_features,
            predictions,
            active,
            blended,
        })
    }

    fn observe(
        &mut self,
        channel: usize,
        sample: i64,
        ticket: &PredictionTicket,
    ) -> Result<(), OptimumV2Error> {
        self.validate_sample(sample)?;
        let state = self
            .states
            .get_mut(channel)
            .ok_or_else(|| invalid("DIX1 observed channel is out of range"))?;
        let mut temporal = state.temporal.clone();
        temporal.observe(&ticket.temporal_features, sample)?;
        let mut scores = state.expert_scores;
        for (expert, score) in scores.iter_mut().enumerate() {
            if !ticket.active[expert] {
                continue;
            }
            let residual = checked_sub(sample, ticket.predictions[expert], "DIX1 expert residual")?;
            let cost = signed_code_bits(residual);
            let decayed = *score - (*score >> SCORE_DECAY_SHIFT);
            *score = decayed
                .checked_add(
                    cost.checked_mul(SCORE_Q)
                        .ok_or_else(|| invalid("DIX1 expert cost overflowed"))?,
                )
                .ok_or_else(|| invalid("DIX1 expert score overflowed"))?;
        }
        state.temporal = temporal;
        state.expert_scores = scores;
        Ok(())
    }

    fn finish_row(&mut self, canonical: &[i64]) -> Result<(), OptimumV2Error> {
        if canonical.len() != self.states.len() {
            return Err(invalid("DIX1 completed row has the wrong channel count"));
        }
        for (state, &sample) in self.states.iter_mut().zip(canonical) {
            state.history.rotate_right(1);
            state.history[0] = sample;
        }
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| invalid("DIX1 row counter overflowed"))?;
        Ok(())
    }

    fn validate_sample(&self, sample: i64) -> Result<(), OptimumV2Error> {
        if !(self.sample_min..=self.sample_max).contains(&sample) {
            return Err(invalid("DIX1 sample exceeds the declared bit depth"));
        }
        Ok(())
    }

    fn clip_prediction(&self, prediction: i64) -> i64 {
        prediction.clamp(self.sample_min, self.sample_max)
    }
}

#[derive(Debug, Clone)]
struct PredictionTicket {
    temporal_features: Vec<i64>,
    predictions: [i64; EXPERT_COUNT],
    active: [bool; EXPERT_COUNT],
    blended: i64,
}

fn normalized_lags(sample_rate_mhz: u32) -> Result<[usize; TEMPORAL_LAG_MS.len()], OptimumV2Error> {
    let mut lags = [0; TEMPORAL_LAG_MS.len()];
    for (slot, milliseconds) in TEMPORAL_LAG_MS.into_iter().enumerate() {
        let numerator = u64::from(sample_rate_mhz)
            .checked_mul(u64::from(milliseconds))
            .and_then(|value| value.checked_add(500_000))
            .ok_or_else(|| invalid("DIX1 temporal lag calculation overflowed"))?;
        lags[slot] = usize::try_from((numerator / 1_000_000).max(1))
            .map_err(|_| invalid("DIX1 temporal lag exceeds usize"))?;
    }
    Ok(lags)
}

fn signed_support_sum(
    channel: &IncidenceChannel,
    prior_values: &[i64],
    label: &str,
) -> Result<i64, OptimumV2Error> {
    let mut total = 0i64;
    for support in channel.supports() {
        let value = *prior_values
            .get(support.prior_channel)
            .ok_or_else(|| invalid("DIX1 support references an unavailable channel"))?;
        let signed = if support.coefficient == 1 {
            value
        } else if support.coefficient == -1 {
            value
                .checked_neg()
                .ok_or_else(|| invalid(format!("{label} negation overflowed")))?
        } else {
            return Err(invalid(
                "DIX1 support coefficient is not signed unit incidence",
            ));
        };
        total = checked_add(total, signed, label)?;
    }
    Ok(total)
}

fn blend_predictions(
    predictions: [i64; EXPERT_COUNT],
    scores: [u64; EXPERT_COUNT],
    active: [bool; EXPERT_COUNT],
    sample_min: i64,
    sample_max: i64,
) -> Result<i64, OptimumV2Error> {
    let mut numerator = 0i128;
    let mut denominator = 0i128;
    for expert in 0..EXPERT_COUNT {
        if !active[expert] {
            continue;
        }
        let scaled_score = scores[expert] / SCORE_Q;
        let weight = MIX_WEIGHT_NUMERATOR / scaled_score.saturating_add(1);
        let weight = weight.max(1);
        numerator = numerator
            .checked_add(i128::from(predictions[expert]) * i128::from(weight))
            .ok_or_else(|| invalid("DIX1 mixture numerator overflowed"))?;
        denominator = denominator
            .checked_add(i128::from(weight))
            .ok_or_else(|| invalid("DIX1 mixture denominator overflowed"))?;
    }
    if denominator <= 0 {
        return Err(invalid("DIX1 mixture has no active expert"));
    }
    let magnitude = numerator.unsigned_abs();
    let rounded = (magnitude + (denominator as u128 / 2)) / denominator as u128;
    let rounded = i128::try_from(rounded)
        .map_err(|_| invalid("DIX1 mixture prediction exceeds signed i128"))?;
    let signed = if numerator < 0 { -rounded } else { rounded };
    let prediction =
        i64::try_from(signed).map_err(|_| invalid("DIX1 mixture prediction exceeds signed i64"))?;
    Ok(prediction.clamp(sample_min, sample_max))
}

fn signed_code_bits(value: i64) -> u64 {
    let magnitude = value.unsigned_abs();
    let zigzag = if value >= 0 {
        magnitude.saturating_mul(2)
    } else {
        magnitude.saturating_mul(2).saturating_sub(1)
    };
    u64::from(64 - zigzag.leading_zeros()) + 1
}

fn checked_add(left: i64, right: i64, label: &str) -> Result<i64, OptimumV2Error> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{label} overflowed signed i64")))
}

fn checked_sub(left: i64, right: i64, label: &str) -> Result<i64, OptimumV2Error> {
    left.checked_sub(right)
        .ok_or_else(|| invalid(format!("{label} overflowed signed i64")))
}

fn invalid(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}
