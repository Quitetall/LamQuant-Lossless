//! Private deterministic binary rANS mechanics shared by construction codecs.
//!
//! Probability modelling remains codec-owned. This module accepts the exact
//! decoder-synchronous probability for each forward event, buffers it for
//! reverse rANS emission, and enforces one canonical byte stream.

use crate::OptimumV2Error;

const CDF_BITS: u32 = 15;
pub(crate) const CDF_TOTAL: u32 = 1 << CDF_BITS;
const RANS_L: u64 = 1 << 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BinaryEvent {
    bit: u8,
    probability_one: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinaryRansEncoder {
    events: Vec<BinaryEvent>,
    max_events: usize,
}

impl BinaryRansEncoder {
    pub(crate) fn new(max_events: usize) -> Result<Self, OptimumV2Error> {
        if max_events == 0 {
            return Err(input_error("binary rANS event bound must be nonzero"));
        }
        Ok(Self {
            events: Vec::new(),
            max_events,
        })
    }

    #[cfg(test)]
    pub(crate) fn event_count(&self) -> usize {
        self.events.len()
    }

    pub(crate) fn push(&mut self, bit: u8, probability_one: u16) -> Result<(), OptimumV2Error> {
        validate_probability(probability_one, InputKind::Encoder)?;
        if bit > 1 {
            return Err(input_error("binary rANS symbol must be zero or one"));
        }
        if self.events.len() >= self.max_events {
            return Err(input_error("binary rANS event buffer exceeds its bound"));
        }
        self.events.push(BinaryEvent {
            bit,
            probability_one,
        });
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<Vec<u8>, OptimumV2Error> {
        let mut state = RANS_L;
        let mut renormalized = Vec::new();
        for event in self.events.iter().rev() {
            let probability_one = u64::from(event.probability_one);
            let frequency_zero = u64::from(CDF_TOTAL) - probability_one;
            let (start, frequency) = if event.bit == 1 {
                (frequency_zero, probability_one)
            } else {
                (0, frequency_zero)
            };
            let maximum = ((RANS_L >> CDF_BITS) << 8) * frequency;
            while state >= maximum {
                renormalized.push(state as u8);
                state >>= 8;
            }
            state = ((state / frequency) << CDF_BITS) + (state % frequency) + start;
        }
        if !(RANS_L..(RANS_L << 8)).contains(&state) {
            return Err(input_error("binary rANS final state exceeds its u32 bound"));
        }
        let mut packed = Vec::with_capacity(4 + renormalized.len());
        packed.extend_from_slice(&(state as u32).to_le_bytes());
        packed.extend(renormalized.into_iter().rev());
        Ok(packed)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BinaryRansDecoder<'a> {
    packed: &'a [u8],
    state: u64,
    offset: usize,
    event_count: usize,
    max_events: usize,
}

impl<'a> BinaryRansDecoder<'a> {
    pub(crate) fn new(packed: &'a [u8], max_events: usize) -> Result<Self, OptimumV2Error> {
        if max_events == 0 {
            return Err(packet_error("binary rANS event bound must be nonzero"));
        }
        let state_bytes = packed
            .get(..4)
            .ok_or_else(|| packet_error("binary rANS stream is truncated"))?;
        let state = u64::from(u32::from_le_bytes(state_bytes.try_into().unwrap()));
        if !(RANS_L..(RANS_L << 8)).contains(&state) {
            return Err(packet_error("binary rANS initial state is invalid"));
        }
        Ok(Self {
            packed,
            state,
            offset: 4,
            event_count: 0,
            max_events,
        })
    }

    pub(crate) fn event_count(&self) -> usize {
        self.event_count
    }

    pub(crate) fn read(&mut self, probability_one: u16) -> Result<u8, OptimumV2Error> {
        validate_probability(probability_one, InputKind::Decoder)?;
        if self.event_count >= self.max_events {
            return Err(packet_error(
                "binary rANS decoded event count exceeds its bound",
            ));
        }
        let probability_one = u64::from(probability_one);
        let frequency_zero = u64::from(CDF_TOTAL) - probability_one;
        let cumulative = self.state & (u64::from(CDF_TOTAL) - 1);
        let (bit, start, frequency) = if cumulative < frequency_zero {
            (0, 0, frequency_zero)
        } else {
            (1, frequency_zero, probability_one)
        };
        self.state = frequency * (self.state >> CDF_BITS) + cumulative - start;
        while self.state < RANS_L {
            let byte = *self
                .packed
                .get(self.offset)
                .ok_or_else(|| packet_error("binary rANS renormalization is truncated"))?;
            self.state = (self.state << 8) | u64::from(byte);
            self.offset += 1;
        }
        self.event_count = self
            .event_count
            .checked_add(1)
            .ok_or_else(|| packet_error("binary rANS event count overflows"))?;
        Ok(bit)
    }

    pub(crate) fn finish(&self) -> Result<(), OptimumV2Error> {
        if self.offset != self.packed.len() || self.state != RANS_L {
            return Err(packet_error(
                "binary rANS stream is noncanonical or has trailing bytes",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum InputKind {
    Encoder,
    Decoder,
}

fn validate_probability(probability_one: u16, kind: InputKind) -> Result<(), OptimumV2Error> {
    if (1..CDF_TOTAL as u16).contains(&probability_one) {
        return Ok(());
    }
    let message = "binary rANS probability must be inside the open CDF interval";
    Err(match kind {
        InputKind::Encoder => input_error(message),
        InputKind::Decoder => packet_error(message),
    })
}

fn input_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

#[cfg(test)]
mod tests {
    use super::{BinaryRansDecoder, BinaryRansEncoder, CDF_TOTAL};

    #[test]
    fn explicit_probability_events_round_trip_canonically() {
        let probabilities = [1, 11_000, 16_384, 25_000, (CDF_TOTAL - 1) as u16];
        let bits = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0];
        let mut encoder = BinaryRansEncoder::new(bits.len()).expect("encoder");
        for (index, &bit) in bits.iter().enumerate() {
            encoder
                .push(bit, probabilities[index % probabilities.len()])
                .expect("bounded event");
        }
        let packed = encoder.finish().expect("canonical stream");
        let mut decoder = BinaryRansDecoder::new(&packed, bits.len()).expect("decoder");
        for (index, &expected) in bits.iter().enumerate() {
            assert_eq!(
                decoder
                    .read(probabilities[index % probabilities.len()])
                    .expect("decoded event"),
                expected
            );
        }
        decoder.finish().expect("canonical finish");

        let mut trailed = packed;
        trailed.push(0);
        let mut decoder = BinaryRansDecoder::new(&trailed, bits.len()).expect("trailed decoder");
        for index in 0..bits.len() {
            decoder
                .read(probabilities[index % probabilities.len()])
                .expect("decoded trailed event");
        }
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn invalid_probability_symbol_and_event_bounds_fail_closed() {
        let mut encoder = BinaryRansEncoder::new(1).expect("encoder");
        assert!(encoder.push(2, 16_384).is_err());
        assert!(encoder.push(0, 0).is_err());
        encoder.push(0, 16_384).expect("first event");
        assert!(encoder.push(1, 16_384).is_err());

        let packed = encoder.finish().expect("stream");
        let mut decoder = BinaryRansDecoder::new(&packed, 1).expect("decoder");
        assert!(decoder.read(0).is_err());
        assert_eq!(decoder.read(16_384).expect("first event"), 0);
        assert!(decoder.read(16_384).is_err());
    }
}
