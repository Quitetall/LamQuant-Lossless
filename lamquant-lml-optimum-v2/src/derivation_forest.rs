//! Metadata-only causal channel forest for DIX2 TreeMED construction.
//!
//! Channels are directed electrode derivations. Each EEG channel selects at
//! most one earlier canonical channel sharing an endpoint. Shared endpoint
//! orientation fixes the sign: equal endpoint roles use `+1`, opposite roles
//! use `-1`. Non-reference electrodes outrank shared references; within that
//! class, the latest causal channel wins. AUX channels never enter the forest.

use crate::derivation_incidence::{
    ChannelIdentity, DerivationIncidence, IncidenceChannel, Partition,
};
use crate::OptimumV2Error;

const MAX_TREE_SUPPORTS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForestSupport {
    pub parent_channel: usize,
    pub coefficient: i8,
    pub shared_endpoint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForestChannel {
    canonical_index: usize,
    stable_id: u16,
    normalized_label: String,
    partition: Partition,
    supports: Vec<ForestSupport>,
}

impl ForestChannel {
    pub fn canonical_index(&self) -> usize {
        self.canonical_index
    }

    pub fn stable_id(&self) -> u16 {
        self.stable_id
    }

    pub fn normalized_label(&self) -> &str {
        &self.normalized_label
    }

    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn parent(&self) -> Option<&ForestSupport> {
        self.supports.first()
    }

    pub fn supports(&self) -> &[ForestSupport] {
        &self.supports
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationForest {
    incidence: DerivationIncidence,
    channels: Vec<ForestChannel>,
}

impl DerivationForest {
    pub fn build(identities: &[ChannelIdentity]) -> Result<Self, OptimumV2Error> {
        let incidence = DerivationIncidence::build(identities)?;
        let mut channels = Vec::with_capacity(incidence.channel_count());
        for channel in incidence.channels() {
            let supports = if channel.partition() == Partition::Eeg {
                select_supports(channel, incidence.channels())
            } else {
                Vec::new()
            };
            if supports
                .iter()
                .any(|support| support.parent_channel >= channel.canonical_index())
            {
                return Err(OptimumV2Error::InvalidInput(
                    "DIX2 forest parent is not strictly causal".into(),
                ));
            }
            channels.push(ForestChannel {
                canonical_index: channel.canonical_index(),
                stable_id: channel.stable_id(),
                normalized_label: channel.normalized_label().to_owned(),
                partition: channel.partition(),
                supports,
            });
        }
        Ok(Self {
            incidence,
            channels,
        })
    }

    pub fn channels(&self) -> &[ForestChannel] {
        &self.channels
    }

    /// Remove the causal TreeMED prediction from one canonical innovation row.
    pub fn forward_canonical_innovations(
        &self,
        innovations: &[i64],
    ) -> Result<Vec<i64>, OptimumV2Error> {
        if innovations.len() != self.channels.len() {
            return Err(invalid(
                "DIX2 innovation row has the wrong canonical channel count",
            ));
        }
        let mut residuals = Vec::with_capacity(innovations.len());
        for (channel, &innovation) in innovations.iter().enumerate() {
            let prediction = parent_prediction(&self.channels[channel], innovations)?;
            residuals.push(innovation.checked_sub(prediction).ok_or_else(|| {
                invalid("DIX2 tree residual overflowed signed 64-bit arithmetic")
            })?);
        }
        Ok(residuals)
    }

    /// Restore one canonical innovation row from causal TreeMED residuals.
    pub fn inverse_canonical_innovations(
        &self,
        residuals: &[i64],
    ) -> Result<Vec<i64>, OptimumV2Error> {
        if residuals.len() != self.channels.len() {
            return Err(invalid(
                "DIX2 residual row has the wrong canonical channel count",
            ));
        }
        let mut innovations = Vec::with_capacity(residuals.len());
        for (channel, &residual) in residuals.iter().enumerate() {
            let prediction = parent_prediction(&self.channels[channel], &innovations)?;
            innovations.push(residual.checked_add(prediction).ok_or_else(|| {
                invalid("DIX2 inverse innovation overflowed signed 64-bit arithmetic")
            })?);
        }
        Ok(innovations)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeInnovationSession {
    forest: DerivationForest,
    previous: Vec<i64>,
    row_count: u64,
}

impl TreeInnovationSession {
    pub fn new(identities: &[ChannelIdentity]) -> Result<Self, OptimumV2Error> {
        let forest = DerivationForest::build(identities)?;
        Ok(Self {
            previous: vec![0; forest.channels.len()],
            forest,
            row_count: 0,
        })
    }

    pub fn forest(&self) -> &DerivationForest {
        &self.forest
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn forward_row(&mut self, presented: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        let canonical = self.forest.incidence.canonicalize_row(presented)?;
        let mut innovations = Vec::with_capacity(canonical.len());
        for (channel, &sample) in canonical.iter().enumerate() {
            let innovation = sample.checked_sub(self.previous[channel]).ok_or_else(|| {
                invalid("DIX2 temporal innovation overflowed signed 64-bit arithmetic")
            })?;
            innovations.push(innovation);
        }
        let residuals = self.forest.forward_canonical_innovations(&innovations)?;
        let row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| invalid("DIX2 row counter overflowed"))?;
        self.previous = canonical;
        self.row_count = row_count;
        Ok(residuals)
    }

    pub fn inverse_row(&mut self, residuals: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
        if residuals.len() != self.forest.channels.len() {
            return Err(invalid("DIX2 residual row has the wrong channel count"));
        }
        let innovations = self.forest.inverse_canonical_innovations(residuals)?;
        let mut canonical = Vec::with_capacity(residuals.len());
        for (channel, &innovation) in innovations.iter().enumerate() {
            let sample = self.previous[channel]
                .checked_add(innovation)
                .ok_or_else(|| {
                    invalid("DIX2 inverse sample overflowed signed 64-bit arithmetic")
                })?;
            canonical.push(sample);
        }
        let presented = self.forest.incidence.restore_presented_row(&canonical)?;
        let row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| invalid("DIX2 row counter overflowed"))?;
        self.previous = canonical;
        self.row_count = row_count;
        Ok(presented)
    }
}

fn parent_prediction(channel: &ForestChannel, innovations: &[i64]) -> Result<i64, OptimumV2Error> {
    if channel.supports().is_empty() {
        return Ok(0);
    }
    let mut predictions = Vec::with_capacity(channel.supports().len());
    for support in channel.supports() {
        let innovation = innovations
            .get(support.parent_channel)
            .copied()
            .ok_or_else(|| invalid("DIX2 forest support is unavailable"))?;
        predictions.push(
            i64::from(support.coefficient)
                .checked_mul(innovation)
                .ok_or_else(|| {
                    invalid("DIX2 support innovation overflowed signed 64-bit arithmetic")
                })?,
        );
    }
    predictions.sort_unstable();
    Ok(predictions[(predictions.len() - 1) / 2])
}

fn select_supports(
    channel: &IncidenceChannel,
    channels: &[IncidenceChannel],
) -> Vec<ForestSupport> {
    let mut non_reference = Vec::new();
    let mut reference = Vec::new();
    for prior in channels[..channel.canonical_index()].iter().rev() {
        if prior.partition() != Partition::Eeg {
            continue;
        }
        let mut shared = shared_endpoints(channel, prior);
        shared.sort_unstable_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(&right.1)));
        let selected = shared
            .iter()
            .find(|(endpoint, _)| !endpoint.starts_with("R:"))
            .or_else(|| shared.first());
        if let Some(&(endpoint, coefficient)) = selected {
            let support = ForestSupport {
                parent_channel: prior.canonical_index(),
                coefficient,
                shared_endpoint: endpoint.to_owned(),
            };
            if endpoint.starts_with("R:") {
                reference.push(support);
            } else {
                non_reference.push(support);
            }
        }
    }
    non_reference.extend(reference);
    non_reference.truncate(MAX_TREE_SUPPORTS);
    non_reference
}

fn shared_endpoints<'a>(
    channel: &'a IncidenceChannel,
    prior: &'a IncidenceChannel,
) -> Vec<(&'a str, i8)> {
    let current = [
        channel.positive_endpoint().map(|endpoint| (endpoint, 1i8)),
        channel.negative_endpoint().map(|endpoint| (endpoint, -1i8)),
    ];
    let parent = [
        prior.positive_endpoint().map(|endpoint| (endpoint, 1i8)),
        prior.negative_endpoint().map(|endpoint| (endpoint, -1i8)),
    ];
    let mut shared = Vec::with_capacity(2);
    for (endpoint, current_sign) in current.into_iter().flatten() {
        for (prior_endpoint, prior_sign) in parent.into_iter().flatten() {
            if endpoint == prior_endpoint {
                shared.push((endpoint, current_sign * prior_sign));
            }
        }
    }
    shared
}

fn invalid(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}
