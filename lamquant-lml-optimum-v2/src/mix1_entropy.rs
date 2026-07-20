//! Decoder-synchronous BM23 residual entropy used by the MIX1 carrier.

use std::collections::HashMap;

use crate::binary_rans::{BinaryRansDecoder, BinaryRansEncoder, CDF_TOTAL};
use crate::OptimumV2Error;

const RESCALE_AT: u32 = 4096;
const POSTERIOR_TOTAL: u64 = 1 << 24;
const MAX_EVENTS_PER_VALUE: usize = 129;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counts {
    zero: u32,
    one: u32,
}

impl Default for Counts {
    fn default() -> Self {
        Self { zero: 1, one: 1 }
    }
}

impl Counts {
    fn probability_one(self) -> u16 {
        let count = u64::from(self.zero) + u64::from(self.one);
        let frequency = (u64::from(self.one) * u64::from(CDF_TOTAL) + count / 2) / count;
        frequency.clamp(1, u64::from(CDF_TOTAL - 1)) as u16
    }

    fn observe(&mut self, symbol: u8) -> Result<(), OptimumV2Error> {
        match symbol {
            0 => self.zero = self.zero.checked_add(1).ok_or_else(count_overflow)?,
            1 => self.one = self.one.checked_add(1).ok_or_else(count_overflow)?,
            _ => return Err(input_error("MIX1 finite-state symbol must be binary")),
        }
        if self.zero + self.one >= RESCALE_AT {
            self.zero = self.zero.div_ceil(2);
            self.one = self.one.div_ceil(2);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BucketKey {
    channel: u16,
    relative_level: i8,
    previous_survives: bool,
    parent_survival: u8,
    previous2_survives: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SignKey {
    channel: u16,
    relative_bucket: i8,
    previous_sign: u8,
    parent_sign: u8,
}

#[derive(Debug, Clone)]
struct FiniteStateSession {
    scale_shift: u8,
    channel_context_mask: u8,
    scale: Vec<u64>,
    previous: Vec<i64>,
    previous2: Vec<i64>,
    bucket_counts: HashMap<BucketKey, Counts>,
    magnitude_counts: HashMap<(u16, i8), Counts>,
    sign_counts: HashMap<SignKey, Counts>,
    channel: Option<usize>,
    parents: Vec<i64>,
}

impl FiniteStateSession {
    fn new(
        channels: usize,
        scale_shift: u8,
        channel_context_mask: u8,
    ) -> Result<Self, OptimumV2Error> {
        if !(1..=256).contains(&channels) || !matches!(scale_shift, 2 | 3) {
            return Err(input_error("MIX1 finite-state shape is invalid"));
        }
        Ok(Self {
            scale_shift,
            channel_context_mask,
            scale: vec![1; channels],
            previous: vec![0; channels],
            previous2: vec![0; channels],
            bucket_counts: HashMap::new(),
            magnitude_counts: HashMap::new(),
            sign_counts: HashMap::new(),
            channel: None,
            parents: Vec::new(),
        })
    }

    fn begin_sample(&mut self, channel: usize, parents: &[i64]) -> Result<(), OptimumV2Error> {
        if self.channel.is_some() || channel >= self.scale.len() {
            return Err(input_error("MIX1 finite-state sample order is invalid"));
        }
        self.channel = Some(channel);
        self.parents.clear();
        self.parents.extend_from_slice(parents);
        Ok(())
    }

    fn channel(&self) -> Result<usize, OptimumV2Error> {
        self.channel
            .ok_or_else(|| input_error("MIX1 finite-state sample is not open"))
    }

    fn scale_log(&self) -> Result<i32, OptimumV2Error> {
        let scale = self.scale[self.channel()?];
        Ok((u64::BITS - 1 - scale.leading_zeros()) as i32)
    }

    fn parent_survival(&self, level: u8) -> u8 {
        if self.parents.is_empty() {
            return 0;
        }
        let mut any = false;
        let mut all = true;
        for &value in &self.parents {
            let survives = bit_length(value.unsigned_abs()) > u32::from(level);
            any |= survives;
            all &= survives;
        }
        match (any, all) {
            (false, _) => 1,
            (true, true) => 3,
            (true, false) => 2,
        }
    }

    fn parent_sign(&self) -> u8 {
        if self.parents.is_empty() {
            return 0;
        }
        let nonzero = self.parents.iter().copied().filter(|value| *value != 0);
        let values: Vec<i64> = nonzero.collect();
        if values.is_empty() {
            1
        } else if values.iter().all(|value| *value > 0) {
            2
        } else if values.iter().all(|value| *value < 0) {
            3
        } else {
            4
        }
    }

    fn bucket_key(&self, level: u8) -> Result<BucketKey, OptimumV2Error> {
        let channel = self.channel()?;
        Ok(BucketKey {
            channel: self.context_channel(channel, 1)?,
            relative_level: clamp_relative(i32::from(level) - self.scale_log()?, 6),
            previous_survives: bit_length(self.previous[channel].unsigned_abs()) > u32::from(level),
            parent_survival: self.parent_survival(level),
            previous2_survives: bit_length(self.previous2[channel].unsigned_abs())
                > u32::from(level),
        })
    }

    fn magnitude_key(&self, shift: u8) -> Result<(u16, i8), OptimumV2Error> {
        let channel = self.channel()?;
        Ok((
            self.context_channel(channel, 2)?,
            clamp_relative(i32::from(shift) - self.scale_log()?, 6),
        ))
    }

    fn sign_key(&self, bucket: u8) -> Result<SignKey, OptimumV2Error> {
        let channel = self.channel()?;
        Ok(SignKey {
            channel: self.context_channel(channel, 4)?,
            relative_bucket: clamp_relative(i32::from(bucket) - self.scale_log()?, 4),
            previous_sign: sign_category(self.previous[channel]),
            parent_sign: self.parent_sign(),
        })
    }

    fn context_channel(&self, channel: usize, flag: u8) -> Result<u16, OptimumV2Error> {
        if self.channel_context_mask & flag != 0 {
            u16::try_from(channel)
                .map_err(|_| input_error("MIX1 entropy channel context exceeds u16"))
        } else {
            Ok(256)
        }
    }

    fn bucket_probability(&mut self, level: u8) -> Result<u16, OptimumV2Error> {
        if level >= 64 {
            return Err(input_error("MIX1 unary level exceeds 63"));
        }
        let key = self.bucket_key(level)?;
        Ok(self.bucket_counts.entry(key).or_default().probability_one())
    }

    fn observe_bucket(&mut self, level: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        let key = self.bucket_key(level)?;
        self.bucket_counts.entry(key).or_default().observe(symbol)
    }

    fn magnitude_probability(&mut self, shift: u8) -> Result<u16, OptimumV2Error> {
        if shift >= 63 {
            return Err(input_error("MIX1 magnitude shift exceeds 62"));
        }
        let key = self.magnitude_key(shift)?;
        Ok(self
            .magnitude_counts
            .entry(key)
            .or_default()
            .probability_one())
    }

    fn observe_magnitude(&mut self, shift: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        let key = self.magnitude_key(shift)?;
        self.magnitude_counts
            .entry(key)
            .or_default()
            .observe(symbol)
    }

    fn sign_probability(&mut self, bucket: u8) -> Result<u16, OptimumV2Error> {
        if !(1..=64).contains(&bucket) {
            return Err(input_error("MIX1 sign bucket is outside 1..=64"));
        }
        let key = self.sign_key(bucket)?;
        Ok(self.sign_counts.entry(key).or_default().probability_one())
    }

    fn observe_sign(&mut self, bucket: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        let key = self.sign_key(bucket)?;
        self.sign_counts.entry(key).or_default().observe(symbol)
    }

    fn finish_sample(&mut self, residual: i64) -> Result<(), OptimumV2Error> {
        let channel = self.channel()?;
        let denominator = 1u128 << self.scale_shift;
        let numerator = (denominator - 1) * u128::from(self.scale[channel])
            + u128::from(residual.unsigned_abs())
            + denominator / 2;
        self.scale[channel] = u64::try_from(numerator / denominator)
            .map_err(|_| input_error("MIX1 finite-state scale exceeds u64"))?;
        self.channel = None;
        self.parents.clear();
        Ok(())
    }

    fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        if self.channel.is_some() || current.len() != self.scale.len() {
            return Err(input_error("MIX1 finite-state time row is invalid"));
        }
        self.previous2.clone_from(&self.previous);
        self.previous.clone_from_slice(current);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Bucket,
    Magnitude,
    Sign,
}

#[derive(Debug, Clone, Copy)]
struct PendingEvent {
    kind: EventKind,
    index: u8,
    probability2: u16,
    probability3: u16,
}

#[derive(Debug, Clone)]
struct BayesianMixture {
    expert2: FiniteStateSession,
    expert3: FiniteStateSession,
    weight2: u64,
    pending: Option<PendingEvent>,
}

impl BayesianMixture {
    fn new(channels: usize, channel_context_mask: u8) -> Result<Self, OptimumV2Error> {
        Ok(Self {
            expert2: FiniteStateSession::new(channels, 2, channel_context_mask)?,
            expert3: FiniteStateSession::new(channels, 3, channel_context_mask)?,
            weight2: POSTERIOR_TOTAL / 2,
            pending: None,
        })
    }

    fn begin_sample(&mut self, channel: usize, parents: &[i64]) -> Result<(), OptimumV2Error> {
        self.require_no_pending()?;
        self.expert2.begin_sample(channel, parents)?;
        self.expert3.begin_sample(channel, parents)
    }

    fn probability(
        &mut self,
        kind: EventKind,
        index: u8,
        probability2: u16,
        probability3: u16,
    ) -> Result<u16, OptimumV2Error> {
        self.require_no_pending()?;
        let weight3 = POSTERIOR_TOTAL - self.weight2;
        let numerator = self.weight2 * u64::from(probability2)
            + weight3 * u64::from(probability3)
            + POSTERIOR_TOTAL / 2;
        let mixed = (numerator / POSTERIOR_TOTAL).clamp(1, u64::from(CDF_TOTAL - 1)) as u16;
        self.pending = Some(PendingEvent {
            kind,
            index,
            probability2,
            probability3,
        });
        Ok(mixed)
    }

    fn bucket_probability(&mut self, level: u8) -> Result<u16, OptimumV2Error> {
        let p2 = self.expert2.bucket_probability(level)?;
        let p3 = self.expert3.bucket_probability(level)?;
        self.probability(EventKind::Bucket, level, p2, p3)
    }

    fn magnitude_probability(&mut self, shift: u8) -> Result<u16, OptimumV2Error> {
        let p2 = self.expert2.magnitude_probability(shift)?;
        let p3 = self.expert3.magnitude_probability(shift)?;
        self.probability(EventKind::Magnitude, shift, p2, p3)
    }

    fn sign_probability(&mut self, bucket: u8) -> Result<u16, OptimumV2Error> {
        let p2 = self.expert2.sign_probability(bucket)?;
        let p3 = self.expert3.sign_probability(bucket)?;
        self.probability(EventKind::Sign, bucket, p2, p3)
    }

    fn observe(&mut self, kind: EventKind, index: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        if symbol > 1 {
            return Err(input_error("MIX1 BM23 symbol must be binary"));
        }
        let pending = self
            .pending
            .ok_or_else(|| input_error("MIX1 BM23 observation has no pending event"))?;
        if pending.kind != kind || pending.index != index {
            return Err(input_error(
                "MIX1 BM23 observation does not match pending event",
            ));
        }
        let likelihood2 = if symbol == 1 {
            u64::from(pending.probability2)
        } else {
            u64::from(CDF_TOTAL - u32::from(pending.probability2))
        };
        let likelihood3 = if symbol == 1 {
            u64::from(pending.probability3)
        } else {
            u64::from(CDF_TOTAL - u32::from(pending.probability3))
        };
        let mass2 = self.weight2 * likelihood2;
        let mass3 = (POSTERIOR_TOTAL - self.weight2) * likelihood3;
        let denominator = mass2 + mass3;
        let updated = (mass2 * POSTERIOR_TOTAL + denominator / 2) / denominator;
        self.weight2 = updated.clamp(1, POSTERIOR_TOTAL - 1);
        match kind {
            EventKind::Bucket => {
                self.expert2.observe_bucket(index, symbol)?;
                self.expert3.observe_bucket(index, symbol)?;
            }
            EventKind::Magnitude => {
                self.expert2.observe_magnitude(index, symbol)?;
                self.expert3.observe_magnitude(index, symbol)?;
            }
            EventKind::Sign => {
                self.expert2.observe_sign(index, symbol)?;
                self.expert3.observe_sign(index, symbol)?;
            }
        }
        self.pending = None;
        Ok(())
    }

    fn finish_sample(&mut self, residual: i64) -> Result<(), OptimumV2Error> {
        self.require_no_pending()?;
        self.expert2.finish_sample(residual)?;
        self.expert3.finish_sample(residual)
    }

    fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        self.require_no_pending()?;
        self.expert2.finish_time(current)?;
        self.expert3.finish_time(current)
    }

    fn require_no_pending(&self) -> Result<(), OptimumV2Error> {
        if self.pending.is_some() {
            Err(input_error("MIX1 BM23 event is still pending"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ContextPending {
    kind: EventKind,
    index: u8,
    channel: usize,
    global_probability: u16,
    local_probability: u16,
}

#[derive(Debug, Clone)]
struct ChannelBucketMixture {
    global: BayesianMixture,
    local: BayesianMixture,
    global_weights: Vec<u64>,
    channel: Option<usize>,
    pending: Option<ContextPending>,
}

impl ChannelBucketMixture {
    fn new(channels: usize, channel_context_mask: u8) -> Result<Self, OptimumV2Error> {
        if !(1..=7).contains(&channel_context_mask) {
            return Err(input_error(
                "MIX1 hierarchical channel-context mask is invalid",
            ));
        }
        Ok(Self {
            global: BayesianMixture::new(channels, 0)?,
            local: BayesianMixture::new(channels, channel_context_mask)?,
            global_weights: vec![POSTERIOR_TOTAL / 2; channels],
            channel: None,
            pending: None,
        })
    }

    fn begin_sample(&mut self, channel: usize, parents: &[i64]) -> Result<(), OptimumV2Error> {
        if self.channel.is_some() || self.pending.is_some() || channel >= self.global_weights.len()
        {
            return Err(input_error(
                "MIX1 hierarchical context sample order is invalid",
            ));
        }
        self.global.begin_sample(channel, parents)?;
        self.local.begin_sample(channel, parents)?;
        self.channel = Some(channel);
        Ok(())
    }

    fn probability(
        &mut self,
        kind: EventKind,
        index: u8,
        global_probability: u16,
        local_probability: u16,
    ) -> Result<u16, OptimumV2Error> {
        if self.pending.is_some() {
            return Err(input_error(
                "MIX1 hierarchical context event is still pending",
            ));
        }
        let channel = self
            .channel
            .ok_or_else(|| input_error("MIX1 hierarchical context sample is not open"))?;
        let global_weight = self.global_weights[channel];
        let local_weight = POSTERIOR_TOTAL - global_weight;
        let numerator = global_weight * u64::from(global_probability)
            + local_weight * u64::from(local_probability)
            + POSTERIOR_TOTAL / 2;
        let mixed = (numerator / POSTERIOR_TOTAL).clamp(1, u64::from(CDF_TOTAL - 1)) as u16;
        self.pending = Some(ContextPending {
            kind,
            index,
            channel,
            global_probability,
            local_probability,
        });
        Ok(mixed)
    }

    fn bucket_probability(&mut self, level: u8) -> Result<u16, OptimumV2Error> {
        let global = self.global.bucket_probability(level)?;
        let local = self.local.bucket_probability(level)?;
        self.probability(EventKind::Bucket, level, global, local)
    }

    fn magnitude_probability(&mut self, shift: u8) -> Result<u16, OptimumV2Error> {
        let global = self.global.magnitude_probability(shift)?;
        let local = self.local.magnitude_probability(shift)?;
        self.probability(EventKind::Magnitude, shift, global, local)
    }

    fn sign_probability(&mut self, bucket: u8) -> Result<u16, OptimumV2Error> {
        let global = self.global.sign_probability(bucket)?;
        let local = self.local.sign_probability(bucket)?;
        self.probability(EventKind::Sign, bucket, global, local)
    }

    fn observe(&mut self, kind: EventKind, index: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        if symbol > 1 {
            return Err(input_error(
                "MIX1 hierarchical context symbol must be binary",
            ));
        }
        let pending = self
            .pending
            .ok_or_else(|| input_error("MIX1 hierarchical observation has no pending event"))?;
        if pending.kind != kind || pending.index != index || self.channel != Some(pending.channel) {
            return Err(input_error(
                "MIX1 hierarchical observation does not match pending event",
            ));
        }
        let likelihood = |probability: u16| {
            if symbol == 1 {
                u64::from(probability)
            } else {
                u64::from(CDF_TOTAL - u32::from(probability))
            }
        };
        let global_mass =
            self.global_weights[pending.channel] * likelihood(pending.global_probability);
        let local_mass = (POSTERIOR_TOTAL - self.global_weights[pending.channel])
            * likelihood(pending.local_probability);
        let denominator = global_mass + local_mass;
        self.global_weights[pending.channel] = ((global_mass * POSTERIOR_TOTAL + denominator / 2)
            / denominator)
            .clamp(1, POSTERIOR_TOTAL - 1);
        self.global.observe(kind, index, symbol)?;
        self.local.observe(kind, index, symbol)?;
        self.pending = None;
        Ok(())
    }

    fn finish_sample(&mut self, residual: i64) -> Result<(), OptimumV2Error> {
        if self.pending.is_some() || self.channel.is_none() {
            return Err(input_error(
                "MIX1 hierarchical context sample is incomplete",
            ));
        }
        self.global.finish_sample(residual)?;
        self.local.finish_sample(residual)?;
        self.channel = None;
        Ok(())
    }

    fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        if self.channel.is_some() || self.pending.is_some() {
            return Err(input_error(
                "MIX1 hierarchical context time row is incomplete",
            ));
        }
        self.global.finish_time(current)?;
        self.local.finish_time(current)
    }
}

#[derive(Debug, Clone)]
enum EntropySession {
    Base(Box<BayesianMixture>),
    Hierarchical(Box<ChannelBucketMixture>),
}

impl EntropySession {
    fn new(channels: usize, context: u8) -> Result<Self, OptimumV2Error> {
        match context {
            0 => Ok(Self::Base(Box::new(BayesianMixture::new(channels, 0)?))),
            1..=7 => Ok(Self::Hierarchical(Box::new(ChannelBucketMixture::new(
                channels, context,
            )?))),
            _ => Err(input_error("MIX1 entropy context identity is invalid")),
        }
    }

    fn begin_sample(&mut self, channel: usize, parents: &[i64]) -> Result<(), OptimumV2Error> {
        match self {
            Self::Base(session) => session.begin_sample(channel, parents),
            Self::Hierarchical(session) => session.begin_sample(channel, parents),
        }
    }

    fn bucket_probability(&mut self, level: u8) -> Result<u16, OptimumV2Error> {
        match self {
            Self::Base(session) => session.bucket_probability(level),
            Self::Hierarchical(session) => session.bucket_probability(level),
        }
    }

    fn magnitude_probability(&mut self, shift: u8) -> Result<u16, OptimumV2Error> {
        match self {
            Self::Base(session) => session.magnitude_probability(shift),
            Self::Hierarchical(session) => session.magnitude_probability(shift),
        }
    }

    fn sign_probability(&mut self, bucket: u8) -> Result<u16, OptimumV2Error> {
        match self {
            Self::Base(session) => session.sign_probability(bucket),
            Self::Hierarchical(session) => session.sign_probability(bucket),
        }
    }

    fn observe(&mut self, kind: EventKind, index: u8, symbol: u8) -> Result<(), OptimumV2Error> {
        match self {
            Self::Base(session) => session.observe(kind, index, symbol),
            Self::Hierarchical(session) => session.observe(kind, index, symbol),
        }
    }

    fn finish_sample(&mut self, residual: i64) -> Result<(), OptimumV2Error> {
        match self {
            Self::Base(session) => session.finish_sample(residual),
            Self::Hierarchical(session) => session.finish_sample(residual),
        }
    }

    fn finish_time(&mut self, current: &[i64]) -> Result<(), OptimumV2Error> {
        match self {
            Self::Base(session) => session.finish_time(current),
            Self::Hierarchical(session) => session.finish_time(current),
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn encode(
    residuals: &[Vec<i64>],
    parents: &[Vec<usize>],
) -> Result<(Vec<u8>, u32), OptimumV2Error> {
    encode_with_channel_context(residuals, parents, 0)
}

pub(crate) fn encode_hierarchical(
    residuals: &[Vec<i64>],
    parents: &[Vec<usize>],
) -> Result<(Vec<u8>, u32), OptimumV2Error> {
    encode_with_channel_context(residuals, parents, 1)
}

pub(crate) fn encode_channel_context(
    residuals: &[Vec<i64>],
    parents: &[Vec<usize>],
    channel_context_mask: u8,
) -> Result<(Vec<u8>, u32), OptimumV2Error> {
    if !(2..=7).contains(&channel_context_mask) {
        return Err(input_error(
            "MIX1 extended channel-context mask must be in 2..=7",
        ));
    }
    encode_with_channel_context(residuals, parents, channel_context_mask)
}

#[allow(clippy::needless_range_loop)]
fn encode_with_channel_context(
    residuals: &[Vec<i64>],
    parents: &[Vec<usize>],
    channel_context_mask: u8,
) -> Result<(Vec<u8>, u32), OptimumV2Error> {
    let (channels, samples, max_events) = shape(residuals, parents)?;
    let mut session = EntropySession::new(channels, channel_context_mask)?;
    let mut coder = BinaryRansEncoder::new(max_events)?;
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            let parent_values = parents[channel]
                .iter()
                .map(|&parent| current[parent])
                .collect::<Vec<_>>();
            session.begin_sample(channel, &parent_values)?;
            let residual = residuals[channel][time];
            let magnitude = residual.unsigned_abs();
            let bucket = bit_length(magnitude) as u8;
            for level in 0..bucket {
                let probability = session.bucket_probability(level)?;
                coder.push(1, probability)?;
                session.observe(EventKind::Bucket, level, 1)?;
            }
            if bucket < 64 {
                let probability = session.bucket_probability(bucket)?;
                coder.push(0, probability)?;
                session.observe(EventKind::Bucket, bucket, 0)?;
            }
            if bucket >= 2 {
                for shift in (0..=(bucket - 2)).rev() {
                    let symbol = ((magnitude >> shift) & 1) as u8;
                    let probability = session.magnitude_probability(shift)?;
                    coder.push(symbol, probability)?;
                    session.observe(EventKind::Magnitude, shift, symbol)?;
                }
            }
            if bucket != 0 {
                let symbol = u8::from(residual < 0);
                let probability = session.sign_probability(bucket)?;
                coder.push(symbol, probability)?;
                session.observe(EventKind::Sign, bucket, symbol)?;
            }
            session.finish_sample(residual)?;
            current[channel] = residual;
        }
        session.finish_time(&current)?;
    }
    let event_count = u32::try_from(coder.event_count())
        .map_err(|_| input_error("MIX1 event count exceeds u32"))?;
    Ok((coder.finish()?, event_count))
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn decode(
    payload: &[u8],
    event_count: u32,
    channels: usize,
    samples: usize,
    parents: &[Vec<usize>],
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    decode_with_channel_context(payload, event_count, channels, samples, parents, 0)
}

pub(crate) fn decode_hierarchical(
    payload: &[u8],
    event_count: u32,
    channels: usize,
    samples: usize,
    parents: &[Vec<usize>],
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    decode_with_channel_context(payload, event_count, channels, samples, parents, 1)
}

pub(crate) fn decode_channel_context(
    payload: &[u8],
    event_count: u32,
    channels: usize,
    samples: usize,
    parents: &[Vec<usize>],
    channel_context_mask: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if !(2..=7).contains(&channel_context_mask) {
        return Err(packet_error(
            "MIX1 extended channel-context mask must be in 2..=7",
        ));
    }
    decode_with_channel_context(
        payload,
        event_count,
        channels,
        samples,
        parents,
        channel_context_mask,
    )
}

#[allow(clippy::needless_range_loop)]
fn decode_with_channel_context(
    payload: &[u8],
    event_count: u32,
    channels: usize,
    samples: usize,
    parents: &[Vec<usize>],
    channel_context_mask: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let values = channels
        .checked_mul(samples)
        .ok_or_else(|| packet_error("MIX1 value count overflows"))?;
    let max_events = values
        .checked_mul(MAX_EVENTS_PER_VALUE)
        .ok_or_else(|| packet_error("MIX1 event bound overflows"))?;
    if parents.len() != channels {
        return Err(packet_error("MIX1 entropy graph channel count differs"));
    }
    let mut session =
        EntropySession::new(channels, channel_context_mask).map_err(as_packet_error)?;
    let mut coder = BinaryRansDecoder::new(payload, max_events)?;
    let mut residuals = vec![vec![0i64; samples]; channels];
    let mut events = 0u32;
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            let parent_values = parents[channel]
                .iter()
                .map(|&parent| current[parent])
                .collect::<Vec<_>>();
            session
                .begin_sample(channel, &parent_values)
                .map_err(as_packet_error)?;
            let mut bucket = 0u8;
            while bucket < 64 {
                let probability = session
                    .bucket_probability(bucket)
                    .map_err(as_packet_error)?;
                let symbol = coder.read(probability)?;
                session
                    .observe(EventKind::Bucket, bucket, symbol)
                    .map_err(as_packet_error)?;
                events = events
                    .checked_add(1)
                    .ok_or_else(|| packet_error("MIX1 event count overflows"))?;
                if symbol == 0 {
                    break;
                }
                bucket += 1;
            }
            let mut magnitude = if bucket == 0 { 0 } else { 1u64 << (bucket - 1) };
            if bucket >= 2 {
                for shift in (0..=(bucket - 2)).rev() {
                    let probability = session
                        .magnitude_probability(shift)
                        .map_err(as_packet_error)?;
                    let symbol = coder.read(probability)?;
                    session
                        .observe(EventKind::Magnitude, shift, symbol)
                        .map_err(as_packet_error)?;
                    magnitude |= u64::from(symbol) << shift;
                    events = events
                        .checked_add(1)
                        .ok_or_else(|| packet_error("MIX1 event count overflows"))?;
                }
            }
            let mut negative = false;
            if bucket != 0 {
                let probability = session.sign_probability(bucket).map_err(as_packet_error)?;
                let symbol = coder.read(probability)?;
                session
                    .observe(EventKind::Sign, bucket, symbol)
                    .map_err(as_packet_error)?;
                negative = symbol == 1;
                events = events
                    .checked_add(1)
                    .ok_or_else(|| packet_error("MIX1 event count overflows"))?;
            }
            let signed = if negative {
                -(i128::from(magnitude))
            } else {
                i128::from(magnitude)
            };
            let residual = i64::try_from(signed)
                .map_err(|_| packet_error("decoded MIX1 residual exceeds i64"))?;
            residuals[channel][time] = residual;
            session.finish_sample(residual).map_err(as_packet_error)?;
            current[channel] = residual;
        }
        session.finish_time(&current).map_err(as_packet_error)?;
    }
    if events != event_count {
        return Err(packet_error("decoded MIX1 event count differs from frame"));
    }
    coder.finish()?;
    Ok(residuals)
}

fn shape(
    residuals: &[Vec<i64>],
    parents: &[Vec<usize>],
) -> Result<(usize, usize, usize), OptimumV2Error> {
    let channels = residuals.len();
    if !(1..=256).contains(&channels) || parents.len() != channels || residuals[0].is_empty() {
        return Err(input_error("MIX1 entropy dimensions are invalid"));
    }
    let samples = residuals[0].len();
    if residuals.iter().any(|row| row.len() != samples) {
        return Err(input_error("MIX1 entropy residuals are not rectangular"));
    }
    for (channel, row) in parents.iter().enumerate() {
        if row.iter().any(|&parent| parent >= channel) {
            return Err(input_error("MIX1 entropy graph is not causal"));
        }
    }
    let values = channels
        .checked_mul(samples)
        .ok_or_else(|| input_error("MIX1 value count overflows"))?;
    let max_events = values
        .checked_mul(MAX_EVENTS_PER_VALUE)
        .ok_or_else(|| input_error("MIX1 event bound overflows"))?;
    Ok((channels, samples, max_events))
}

fn bit_length(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

fn sign_category(value: i64) -> u8 {
    if value > 0 {
        0
    } else if value < 0 {
        1
    } else {
        2
    }
}

fn clamp_relative(value: i32, bound: i32) -> i8 {
    value.clamp(-bound, bound) as i8
}

fn count_overflow() -> OptimumV2Error {
    input_error("MIX1 finite-state count overflowed")
}

fn input_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

fn as_packet_error(error: OptimumV2Error) -> OptimumV2Error {
    match error {
        OptimumV2Error::Integrity(message) => OptimumV2Error::Integrity(message),
        other => packet_error(other.to_string()),
    }
}
