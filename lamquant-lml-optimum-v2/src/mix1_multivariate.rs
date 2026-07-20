//! Zero-side multi-parent fixed-RLS expert used by the MIX peer carrier.

use crate::fixed_predictor::FixedRlsExpert;
use crate::OptimumV2Error;

const MAX_ORDER: usize = 7;
const MAX_PARENTS: usize = 4;
const MAX_RLS_WIDTH: usize = 15;

#[derive(Debug, Clone)]
pub(crate) struct MultivariateSession {
    parents: Vec<Vec<usize>>,
    history: Vec<Vec<i64>>,
    states: Vec<Vec<FixedRlsExpert>>,
    row_samples: Vec<Option<i64>>,
    next_channel: usize,
    sample_min: i64,
    sample_max: i64,
    parent_history_depth: usize,
}

impl MultivariateSession {
    pub(crate) fn new(parents: &[Vec<usize>], bit_depth: u8) -> Result<Self, OptimumV2Error> {
        Self::new_with_parent_history(parents, bit_depth, 1)
    }

    pub(crate) fn new_with_parent_history(
        parents: &[Vec<usize>],
        bit_depth: u8,
        parent_history_depth: usize,
    ) -> Result<Self, OptimumV2Error> {
        if parents.is_empty()
            || parents.len() > 256
            || !(1..=32).contains(&bit_depth)
            || parent_history_depth > 4
            || parents.iter().enumerate().any(|(channel, row)| {
                row.len() > MAX_PARENTS
                    || row.iter().any(|&parent| parent >= channel)
                    || row.windows(2).any(|pair| pair[0] >= pair[1])
            })
        {
            return Err(input_error(
                "MIX1 multivariate graph or bit depth is invalid",
            ));
        }
        let mut states = Vec::with_capacity(parents.len());
        for row in parents {
            let parent_width = if row.is_empty() {
                0
            } else {
                (1 + parent_history_depth)
                    * row.len().min(MAX_RLS_WIDTH / (1 + parent_history_depth))
            };
            let maximum_order = if row.is_empty() {
                MAX_ORDER
            } else {
                MAX_ORDER.min(MAX_RLS_WIDTH - parent_width)
            };
            let mut experts = Vec::with_capacity(maximum_order + 1);
            for order in 0..=maximum_order {
                let width = if row.is_empty() {
                    order + 1
                } else {
                    order + parent_width
                };
                experts.push(FixedRlsExpert::new(width, bit_depth)?);
            }
            states.push(experts);
        }
        let magnitude = 1i64 << (bit_depth - 1);
        Ok(Self {
            parents: parents.to_vec(),
            history: vec![vec![0; MAX_ORDER + 1]; parents.len()],
            states,
            row_samples: vec![None; parents.len()],
            next_channel: 0,
            sample_min: -magnitude,
            sample_max: magnitude - 1,
            parent_history_depth,
        })
    }

    pub(crate) fn prediction(
        &self,
        channel: usize,
        current: &[i64],
    ) -> Result<i64, OptimumV2Error> {
        self.validate_row(channel, current)?;
        let selected = self.selected_order(channel);
        self.states[channel][selected].prediction(&self.features(channel, current, selected))
    }

    pub(crate) fn observe(
        &mut self,
        channel: usize,
        current: &[i64],
        sample: i64,
        prediction: i64,
    ) -> Result<(), OptimumV2Error> {
        self.validate_row(channel, current)?;
        if !(self.sample_min..=self.sample_max).contains(&sample) {
            return Err(input_error(
                "MIX1 multivariate observed sample exceeds bit depth",
            ));
        }
        if prediction != self.prediction(channel, current)? {
            return Err(input_error("MIX1 multivariate prediction ticket differs"));
        }
        let mut next = self.states[channel].clone();
        for (order, expert) in next.iter_mut().enumerate() {
            expert.observe(&self.features(channel, current, order), sample)?;
        }
        self.states[channel] = next;
        self.row_samples[channel] = Some(sample);
        self.next_channel += 1;
        Ok(())
    }

    pub(crate) fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        if self.next_channel != self.parents.len()
            || current.len() != self.parents.len()
            || self
                .row_samples
                .iter()
                .zip(current)
                .any(|(observed, sample)| *observed != Some(*sample))
        {
            return Err(input_error("MIX1 multivariate time row is incomplete"));
        }
        for (history, &sample) in self.history.iter_mut().zip(current) {
            history.copy_within(0..MAX_ORDER, 1);
            history[0] = sample;
        }
        self.row_samples.fill(None);
        self.next_channel = 0;
        Ok(())
    }

    fn selected_order(&self, channel: usize) -> usize {
        let mut selected = 0usize;
        for order in 1..self.states[channel].len() {
            if self.states[channel][order].score_q20() < self.states[channel][selected].score_q20()
            {
                selected = order;
            }
        }
        selected
    }

    fn features(&self, channel: usize, current: &[i64], order: usize) -> Vec<i64> {
        let parents = &self.parents[channel];
        if parents.is_empty() {
            return self.history[channel][..=order].to_vec();
        }
        let mut features = Vec::with_capacity(order + 2 * parents.len());
        features.extend_from_slice(&self.history[channel][..order]);
        let parent_cap = MAX_RLS_WIDTH / (1 + self.parent_history_depth);
        for &parent in parents.iter().take(parent_cap) {
            features.push(current[parent]);
            features.extend_from_slice(&self.history[parent][..self.parent_history_depth]);
        }
        features
    }

    fn validate_row(&self, channel: usize, current: &[i64]) -> Result<(), OptimumV2Error> {
        if channel != self.next_channel || current.len() != self.parents.len() {
            return Err(input_error("MIX1 multivariate channel sequence is invalid"));
        }
        for (index, &sample) in current.iter().enumerate() {
            if !(self.sample_min..=self.sample_max).contains(&sample)
                || (index < channel && self.row_samples[index] != Some(sample))
            {
                return Err(input_error("MIX1 multivariate current row is invalid"));
            }
        }
        Ok(())
    }
}

fn input_error(message: &str) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_enforces_graph_row_and_prediction_ticket_contracts() {
        assert!(MultivariateSession::new(&[vec![0]], 16).is_err());
        assert!(MultivariateSession::new(&[vec![], vec![0, 0]], 16).is_err());

        let mut session = MultivariateSession::new(&[vec![], vec![0]], 16).unwrap();
        let mut current = vec![0, 0];
        assert!(session.prediction(1, &current).is_err());

        let first = session.prediction(0, &current).unwrap();
        assert!(session.observe(0, &current, 7, first + 1).is_err());
        session.observe(0, &current, 7, first).unwrap();
        current[0] = 7;

        let second = session.prediction(1, &current).unwrap();
        session.observe(1, &current, -3, second).unwrap();
        assert!(session.finish_time(&current).is_err());
        current[1] = -3;
        session.finish_time(&current).unwrap();
        assert!(session.prediction(0, &current).is_ok());
    }
}
