//! Pack-free deterministic probability law for DIX1 residuals.
//!
//! Every binary context uses a Krichevsky-Trofimov count estimate. Counts are
//! keyed by canonical channel, event family, and magnitude-bit position; the
//! shared rANS core receives only the resulting decoder-synchronous CDF value.

use crate::binary_rans::{BinaryRansDecoder, BinaryRansEncoder, CDF_TOTAL};
use crate::OptimumV2Error;

const MAX_BIT_DEPTH: usize = 32;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    zeros: u32,
    ones: u32,
}

impl Counts {
    fn probability_one(self) -> u16 {
        let numerator = (2 * u64::from(self.ones) + 1) * u64::from(CDF_TOTAL);
        let denominator = 2 * (u64::from(self.zeros) + u64::from(self.ones) + 1);
        (numerator / denominator).clamp(1, u64::from(CDF_TOTAL - 1)) as u16
    }

    fn observe(&mut self, bit: u8) -> Result<(), OptimumV2Error> {
        let count = if bit == 0 {
            &mut self.zeros
        } else if bit == 1 {
            &mut self.ones
        } else {
            return Err(input_error("DIX1 entropy bit must be zero or one"));
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| input_error("DIX1 entropy context count overflowed"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChannelModel {
    nonzero: Counts,
    sign: Counts,
    exponent: [Counts; MAX_BIT_DEPTH],
    mantissa: [Counts; MAX_BIT_DEPTH],
}

impl Default for ChannelModel {
    fn default() -> Self {
        Self {
            nonzero: Counts::default(),
            sign: Counts::default(),
            exponent: [Counts::default(); MAX_BIT_DEPTH],
            mantissa: [Counts::default(); MAX_BIT_DEPTH],
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Family {
    Nonzero,
    Exponent,
    Sign,
    Mantissa,
}

impl ChannelModel {
    fn context(&self, family: Family, position: usize) -> Result<Counts, OptimumV2Error> {
        match family {
            Family::Nonzero => Ok(self.nonzero),
            Family::Sign => Ok(self.sign),
            Family::Exponent => self
                .exponent
                .get(position)
                .copied()
                .ok_or_else(|| input_error("DIX1 exponent context is out of range")),
            Family::Mantissa => self
                .mantissa
                .get(position)
                .copied()
                .ok_or_else(|| input_error("DIX1 mantissa context is out of range")),
        }
    }

    fn observe(&mut self, family: Family, position: usize, bit: u8) -> Result<(), OptimumV2Error> {
        let context = match family {
            Family::Nonzero => &mut self.nonzero,
            Family::Sign => &mut self.sign,
            Family::Exponent => self
                .exponent
                .get_mut(position)
                .ok_or_else(|| input_error("DIX1 exponent context is out of range"))?,
            Family::Mantissa => self
                .mantissa
                .get_mut(position)
                .ok_or_else(|| input_error("DIX1 mantissa context is out of range"))?,
        };
        context.observe(bit)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dix1EntropyEncoder {
    coder: BinaryRansEncoder,
    model: Vec<ChannelModel>,
    bit_depth: usize,
}

impl Dix1EntropyEncoder {
    pub(crate) fn new(
        channels: usize,
        values: usize,
        bit_depth: u8,
    ) -> Result<Self, OptimumV2Error> {
        let (bit_depth, max_events) =
            entropy_shape(channels, values, bit_depth, InputKind::Encoder)?;
        Ok(Self {
            coder: BinaryRansEncoder::new(max_events)?,
            model: vec![ChannelModel::default(); channels],
            bit_depth,
        })
    }

    #[cfg(test)]
    pub(crate) fn event_count(&self) -> usize {
        self.coder.event_count()
    }

    pub(crate) fn push_value(&mut self, channel: usize, value: i64) -> Result<(), OptimumV2Error> {
        let magnitude = value.unsigned_abs();
        self.push(channel, Family::Nonzero, 0, u8::from(magnitude != 0))?;
        if magnitude == 0 {
            return Ok(());
        }
        let exponent = (u64::BITS - 1 - magnitude.leading_zeros()) as usize;
        if exponent >= self.bit_depth {
            return Err(input_error(
                "DIX1 residual magnitude exceeds the declared bit-depth bound",
            ));
        }
        for position in 0..=exponent {
            self.push(
                channel,
                Family::Exponent,
                position,
                u8::from(position < exponent),
            )?;
        }
        self.push(channel, Family::Sign, 0, u8::from(value < 0))?;
        for position in (0..exponent).rev() {
            self.push(
                channel,
                Family::Mantissa,
                position,
                ((magnitude >> position) & 1) as u8,
            )?;
        }
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<Vec<u8>, OptimumV2Error> {
        self.coder.finish()
    }

    fn push(
        &mut self,
        channel: usize,
        family: Family,
        position: usize,
        bit: u8,
    ) -> Result<(), OptimumV2Error> {
        let model = self
            .model
            .get_mut(channel)
            .ok_or_else(|| input_error("DIX1 entropy channel is out of range"))?;
        let probability_one = model.context(family, position)?.probability_one();
        self.coder.push(bit, probability_one)?;
        model.observe(family, position, bit)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Dix1EntropyDecoder<'a> {
    coder: BinaryRansDecoder<'a>,
    model: Vec<ChannelModel>,
    bit_depth: usize,
}

impl<'a> Dix1EntropyDecoder<'a> {
    pub(crate) fn new(
        packed: &'a [u8],
        channels: usize,
        values: usize,
        bit_depth: u8,
    ) -> Result<Self, OptimumV2Error> {
        let (bit_depth, max_events) =
            entropy_shape(channels, values, bit_depth, InputKind::Decoder)?;
        Ok(Self {
            coder: BinaryRansDecoder::new(packed, max_events)?,
            model: vec![ChannelModel::default(); channels],
            bit_depth,
        })
    }

    pub(crate) fn event_count(&self) -> usize {
        self.coder.event_count()
    }

    pub(crate) fn read_value(&mut self, channel: usize) -> Result<i64, OptimumV2Error> {
        if self.read(channel, Family::Nonzero, 0)? == 0 {
            return Ok(0);
        }
        let mut exponent = 0usize;
        while self.read(channel, Family::Exponent, exponent)? == 1 {
            exponent = exponent
                .checked_add(1)
                .ok_or_else(|| packet_error("DIX1 residual exponent overflowed"))?;
            if exponent >= self.bit_depth {
                return Err(packet_error(
                    "DIX1 residual exponent exceeds the declared bit depth",
                ));
            }
        }
        let sign = self.read(channel, Family::Sign, 0)?;
        let mut magnitude = 1u64 << exponent;
        for position in (0..exponent).rev() {
            magnitude |= u64::from(self.read(channel, Family::Mantissa, position)?) << position;
        }
        if sign == 0 {
            i64::try_from(magnitude)
                .map_err(|_| packet_error("DIX1 positive residual exceeds signed i64"))
        } else {
            Ok(-(magnitude as i64))
        }
    }

    pub(crate) fn finish(&self) -> Result<(), OptimumV2Error> {
        self.coder.finish()
    }

    fn read(
        &mut self,
        channel: usize,
        family: Family,
        position: usize,
    ) -> Result<u8, OptimumV2Error> {
        let model = self
            .model
            .get_mut(channel)
            .ok_or_else(|| packet_error("DIX1 entropy channel is out of range"))?;
        let probability_one = model
            .context(family, position)
            .map_err(as_packet_error)?
            .probability_one();
        let bit = self.coder.read(probability_one)?;
        model
            .observe(family, position, bit)
            .map_err(as_packet_error)?;
        Ok(bit)
    }
}

#[derive(Clone, Copy)]
enum InputKind {
    Encoder,
    Decoder,
}

fn entropy_shape(
    channels: usize,
    values: usize,
    bit_depth: u8,
    kind: InputKind,
) -> Result<(usize, usize), OptimumV2Error> {
    let bit_depth = usize::from(bit_depth);
    let valid = channels > 0
        && values >= channels
        && values % channels == 0
        && (1..=MAX_BIT_DEPTH).contains(&bit_depth);
    let events_per_value = bit_depth
        .checked_mul(2)
        .and_then(|value| value.checked_add(1));
    let max_events = events_per_value.and_then(|bound| values.checked_mul(bound));
    if !valid || max_events.is_none() {
        let message = "DIX1 entropy dimensions or bit depth are invalid";
        return Err(match kind {
            InputKind::Encoder => input_error(message),
            InputKind::Decoder => packet_error(message),
        });
    }
    Ok((bit_depth, max_events.unwrap()))
}

fn as_packet_error(error: OptimumV2Error) -> OptimumV2Error {
    packet_error(error.to_string())
}

fn input_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

#[cfg(test)]
mod tests {
    use super::{Counts, Dix1EntropyDecoder, Dix1EntropyEncoder};

    #[test]
    fn kt_probability_trace_is_exact() {
        let mut counts = Counts::default();
        assert_eq!(counts.probability_one(), 16_384);
        counts.observe(1).expect("one");
        assert_eq!(counts.probability_one(), 24_576);
        counts.observe(1).expect("two");
        assert_eq!(counts.probability_one(), 27_306);
        counts.observe(0).expect("zero");
        assert_eq!(counts.probability_one(), 20_480);
    }

    #[test]
    fn signed_residuals_round_trip_at_the_32_bit_difference_bound() {
        let values = [0, 1, -1, 17, -91, i64::from(u32::MAX), -i64::from(u32::MAX)];
        let mut encoder = Dix1EntropyEncoder::new(1, values.len(), 32).expect("encoder");
        for &value in &values {
            encoder.push_value(0, value).expect("residual");
        }
        let event_count = encoder.event_count();
        let packed = encoder.finish().expect("stream");
        let mut decoder = Dix1EntropyDecoder::new(&packed, 1, values.len(), 32).expect("decoder");
        for &expected in &values {
            assert_eq!(decoder.read_value(0).expect("decoded residual"), expected);
        }
        assert_eq!(decoder.event_count(), event_count);
        decoder.finish().expect("canonical finish");
    }

    #[test]
    fn residual_exponents_and_stream_bounds_fail_closed() {
        let mut encoder = Dix1EntropyEncoder::new(1, 1, 8).expect("encoder");
        assert!(encoder.push_value(0, 256).is_err());
        assert!(Dix1EntropyEncoder::new(0, 1, 8).is_err());
        assert!(Dix1EntropyEncoder::new(1, 1, 0).is_err());
        assert!(Dix1EntropyDecoder::new(&[0; 3], 1, 1, 8).is_err());
    }
}
