//! Fixed-block engine for the construction-private DIX1 carrier.
//!
//! The packet-facing module owns framing and integrity. This module hides the
//! 128-row partition, per-block mode selection, compact directory, predictor
//! continuity, and block-local entropy resets behind one in-process seam.

use crate::derivation_incidence::ChannelIdentity;
use crate::dix1::{Dix1IncidenceMode, Dix1Session};
use crate::dix1_entropy::{Dix1EntropyDecoder, Dix1EntropyEncoder};
use crate::{
    canonical_i32_bytes, decode_delta_varints, decode_raw_i32, encode_delta_varints, OptimumV2Error,
};

pub(crate) const BLOCK_ROWS: usize = 128;
pub(crate) const DIRECTORY_ENTRY_LEN: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dix1CarrierMode {
    Raw,
    Delta,
    IncidenceRans,
    NoIncidenceRans,
}

impl Dix1CarrierMode {
    pub(crate) fn wire(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::Delta => 1,
            Self::IncidenceRans => 2,
            Self::NoIncidenceRans => 3,
        }
    }

    fn from_wire(value: u8) -> Result<Self, OptimumV2Error> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Delta),
            2 => Ok(Self::IncidenceRans),
            3 => Ok(Self::NoIncidenceRans),
            _ => Err(packet_error("DIX1 block mode is invalid")),
        }
    }

    fn is_entropy(self) -> bool {
        matches!(self, Self::IncidenceRans | Self::NoIncidenceRans)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CarrierProfile {
    Product,
    Native,
    ForcedRaw,
    ForcedDelta,
    ForcedIncidence,
    ForcedNoIncidence,
}

impl CarrierProfile {
    pub(crate) fn wire(self) -> u8 {
        match self {
            Self::Product => 0,
            Self::Native => 1,
            Self::ForcedRaw => 2,
            Self::ForcedDelta => 3,
            Self::ForcedIncidence => 4,
            Self::ForcedNoIncidence => 5,
        }
    }

    pub(crate) fn from_wire(value: u8) -> Result<Self, OptimumV2Error> {
        match value {
            0 => Ok(Self::Product),
            1 => Ok(Self::Native),
            2 => Ok(Self::ForcedRaw),
            3 => Ok(Self::ForcedDelta),
            4 => Ok(Self::ForcedIncidence),
            5 => Ok(Self::ForcedNoIncidence),
            _ => Err(packet_error("DIX1 carrier profile is invalid")),
        }
    }

    pub(crate) fn forced(mode: Dix1CarrierMode) -> Self {
        match mode {
            Dix1CarrierMode::Raw => Self::ForcedRaw,
            Dix1CarrierMode::Delta => Self::ForcedDelta,
            Dix1CarrierMode::IncidenceRans => Self::ForcedIncidence,
            Dix1CarrierMode::NoIncidenceRans => Self::ForcedNoIncidence,
        }
    }

    fn incidence_mode(self) -> Dix1IncidenceMode {
        if self == Self::ForcedNoIncidence {
            Dix1IncidenceMode::Disabled
        } else {
            Dix1IncidenceMode::Enabled
        }
    }

    fn permits(self, mode: Dix1CarrierMode) -> bool {
        match self {
            Self::Product => matches!(
                mode,
                Dix1CarrierMode::Raw | Dix1CarrierMode::Delta | Dix1CarrierMode::IncidenceRans
            ),
            Self::Native => matches!(mode, Dix1CarrierMode::Raw | Dix1CarrierMode::Delta),
            Self::ForcedRaw => mode == Dix1CarrierMode::Raw,
            Self::ForcedDelta => mode == Dix1CarrierMode::Delta,
            Self::ForcedIncidence => mode == Dix1CarrierMode::IncidenceRans,
            Self::ForcedNoIncidence => mode == Dix1CarrierMode::NoIncidenceRans,
        }
    }
}

pub(crate) struct BlockInput<'a> {
    pub(crate) canonical_identities: &'a [ChannelIdentity],
    pub(crate) canonical_signal: &'a [Vec<i64>],
    pub(crate) stable_signal: &'a [Vec<i64>],
    pub(crate) channels: usize,
    pub(crate) samples: usize,
    pub(crate) sample_rate_mhz: u32,
    pub(crate) bit_depth: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedBlocks {
    pub(crate) directory: Vec<u8>,
    pub(crate) payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedBlocks {
    pub(crate) stable_signal: Vec<Vec<i64>>,
    pub(crate) modes: Vec<Dix1CarrierMode>,
    pub(crate) event_count: u32,
}

#[derive(Debug)]
struct Candidate {
    mode: Dix1CarrierMode,
    payload: Vec<u8>,
}

pub(crate) fn block_count(samples: usize) -> usize {
    samples.div_ceil(BLOCK_ROWS)
}

pub(crate) fn encode(
    input: BlockInput<'_>,
    profile: CarrierProfile,
) -> Result<EncodedBlocks, OptimumV2Error> {
    let blocks = block_count(input.samples);
    let mut session = Dix1Session::new_with_incidence_mode(
        input.canonical_identities,
        input.bit_depth,
        input.sample_rate_mhz,
        profile.incidence_mode(),
    )?;
    let mut directory = Vec::with_capacity(blocks * DIRECTORY_ENTRY_LEN);
    let mut payload = Vec::new();

    for block in 0..blocks {
        let start = block * BLOCK_ROWS;
        let end = (start + BLOCK_ROWS).min(input.samples);
        let stable_block = slice_signal(input.stable_signal, start, end);
        let candidate = match profile {
            CarrierProfile::Product => {
                let raw = Candidate {
                    mode: Dix1CarrierMode::Raw,
                    payload: canonical_i32_bytes(&stable_block)?,
                };
                let delta = Candidate {
                    mode: Dix1CarrierMode::Delta,
                    payload: encode_delta_varints(&stable_block),
                };
                let incidence = encode_entropy_block(
                    &mut session,
                    &input,
                    start,
                    end,
                    Dix1CarrierMode::IncidenceRans,
                )?;
                choose_smallest([raw, delta, incidence])?
            }
            CarrierProfile::Native => {
                advance_input_block(&mut session, input.canonical_signal, start, end)?;
                choose_smallest([
                    Candidate {
                        mode: Dix1CarrierMode::Raw,
                        payload: canonical_i32_bytes(&stable_block)?,
                    },
                    Candidate {
                        mode: Dix1CarrierMode::Delta,
                        payload: encode_delta_varints(&stable_block),
                    },
                ])?
            }
            CarrierProfile::ForcedRaw => {
                advance_input_block(&mut session, input.canonical_signal, start, end)?;
                Candidate {
                    mode: Dix1CarrierMode::Raw,
                    payload: canonical_i32_bytes(&stable_block)?,
                }
            }
            CarrierProfile::ForcedDelta => {
                advance_input_block(&mut session, input.canonical_signal, start, end)?;
                Candidate {
                    mode: Dix1CarrierMode::Delta,
                    payload: encode_delta_varints(&stable_block),
                }
            }
            CarrierProfile::ForcedIncidence => encode_entropy_block(
                &mut session,
                &input,
                start,
                end,
                Dix1CarrierMode::IncidenceRans,
            )?,
            CarrierProfile::ForcedNoIncidence => encode_entropy_block(
                &mut session,
                &input,
                start,
                end,
                Dix1CarrierMode::NoIncidenceRans,
            )?,
        };
        let payload_len = u32::try_from(candidate.payload.len())
            .map_err(|_| input_error("DIX1 block payload length exceeds u32"))?;
        directory.push(candidate.mode.wire());
        directory.extend_from_slice(&payload_len.to_le_bytes());
        payload.extend_from_slice(&candidate.payload);
    }

    Ok(EncodedBlocks { directory, payload })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode(
    directory: &[u8],
    payload: &[u8],
    canonical_identities: &[ChannelIdentity],
    channels: usize,
    samples: usize,
    sample_rate_mhz: u32,
    bit_depth: u8,
    profile: CarrierProfile,
) -> Result<DecodedBlocks, OptimumV2Error> {
    let blocks = block_count(samples);
    let expected_directory_len = blocks
        .checked_mul(DIRECTORY_ENTRY_LEN)
        .ok_or_else(|| packet_error("DIX1 directory length overflows"))?;
    if directory.len() != expected_directory_len {
        return Err(packet_error("DIX1 compact directory length is invalid"));
    }

    let mut entries = Vec::with_capacity(blocks);
    let mut payload_sum = 0usize;
    for block in 0..blocks {
        let offset = block * DIRECTORY_ENTRY_LEN;
        let mode = Dix1CarrierMode::from_wire(directory[offset])?;
        if !profile.permits(mode) {
            return Err(packet_error(
                "DIX1 block mode is not permitted by the carrier profile",
            ));
        }
        let length = usize::try_from(u32::from_le_bytes(
            directory[offset + 1..offset + DIRECTORY_ENTRY_LEN]
                .try_into()
                .unwrap(),
        ))
        .map_err(|_| packet_error("DIX1 block payload length exceeds usize"))?;
        payload_sum = payload_sum
            .checked_add(length)
            .ok_or_else(|| packet_error("DIX1 block payload sum overflows"))?;
        entries.push((mode, length));
    }
    if payload_sum != payload.len() {
        return Err(packet_error(
            "DIX1 compact directory does not cover the payload exactly",
        ));
    }

    let mut session = Dix1Session::new_with_incidence_mode(
        canonical_identities,
        bit_depth,
        sample_rate_mhz,
        profile.incidence_mode(),
    )
    .map_err(as_packet_error)?;
    let mut stable_signal = (0..channels)
        .map(|_| Vec::with_capacity(samples))
        .collect::<Vec<_>>();
    let mut payload_offset = 0usize;
    let mut total_events = 0u32;
    let mut modes = Vec::with_capacity(blocks);

    for (block, (mode, length)) in entries.into_iter().enumerate() {
        let start = block * BLOCK_ROWS;
        let end = (start + BLOCK_ROWS).min(samples);
        let rows = end - start;
        let values = channels
            .checked_mul(rows)
            .ok_or_else(|| packet_error("DIX1 block value count overflows"))?;
        let payload_end = payload_offset
            .checked_add(length)
            .ok_or_else(|| packet_error("DIX1 block payload end overflows"))?;
        let block_payload = payload
            .get(payload_offset..payload_end)
            .ok_or_else(|| packet_error("DIX1 block payload is truncated"))?;

        let (block_signal, events) = match mode {
            Dix1CarrierMode::Raw => {
                let expected = values
                    .checked_mul(4)
                    .ok_or_else(|| packet_error("DIX1 raw block length overflows"))?;
                if block_payload.len() != expected {
                    return Err(packet_error("DIX1 raw block payload length is invalid"));
                }
                let signal = decode_raw_i32(block_payload, channels, rows);
                advance_decoded_block(&mut session, canonical_identities, &signal, rows)?;
                (signal, 0)
            }
            Dix1CarrierMode::Delta => {
                let maximum = values
                    .checked_mul(5)
                    .ok_or_else(|| packet_error("DIX1 delta block bound overflows"))?;
                if block_payload.len() < values || block_payload.len() > maximum {
                    return Err(packet_error(
                        "DIX1 delta block payload length is outside bounds",
                    ));
                }
                let signal = decode_delta_varints(block_payload, channels, rows)?;
                advance_decoded_block(&mut session, canonical_identities, &signal, rows)?;
                (signal, 0)
            }
            Dix1CarrierMode::IncidenceRans | Dix1CarrierMode::NoIncidenceRans => {
                decode_entropy_block(
                    block_payload,
                    &mut session,
                    canonical_identities,
                    channels,
                    rows,
                    bit_depth,
                )?
            }
        };
        for channel in 0..channels {
            stable_signal[channel].extend_from_slice(&block_signal[channel]);
        }
        total_events = total_events
            .checked_add(events)
            .ok_or_else(|| packet_error("DIX1 decoded event count exceeds u32"))?;
        modes.push(mode);
        payload_offset = payload_end;
    }

    if payload_offset != payload.len()
        || stable_signal.iter().any(|channel| channel.len() != samples)
    {
        return Err(packet_error("DIX1 block decode did not cover the window"));
    }
    Ok(DecodedBlocks {
        stable_signal,
        modes,
        event_count: total_events,
    })
}

fn choose_smallest<const N: usize>(
    candidates: [Candidate; N],
) -> Result<Candidate, OptimumV2Error> {
    candidates
        .into_iter()
        .min_by_key(|candidate| (candidate.payload.len(), candidate.mode))
        .ok_or_else(|| input_error("DIX1 candidate set is empty"))
}

fn encode_entropy_block(
    session: &mut Dix1Session,
    input: &BlockInput<'_>,
    start: usize,
    end: usize,
    mode: Dix1CarrierMode,
) -> Result<Candidate, OptimumV2Error> {
    debug_assert!(mode.is_entropy());
    let rows = end - start;
    let values = input
        .channels
        .checked_mul(rows)
        .ok_or_else(|| input_error("DIX1 block value count overflows"))?;
    let mut entropy = Dix1EntropyEncoder::new(input.channels, values, input.bit_depth)?;
    for sample in start..end {
        let row = input
            .canonical_signal
            .iter()
            .map(|channel| channel[sample])
            .collect::<Vec<_>>();
        let residuals = session.forward_row(&row)?;
        for (channel, residual) in residuals.into_iter().enumerate() {
            entropy.push_value(channel, residual)?;
        }
    }
    Ok(Candidate {
        mode,
        payload: entropy.finish()?,
    })
}

fn decode_entropy_block(
    payload: &[u8],
    session: &mut Dix1Session,
    canonical_identities: &[ChannelIdentity],
    channels: usize,
    rows: usize,
    bit_depth: u8,
) -> Result<(Vec<Vec<i64>>, u32), OptimumV2Error> {
    let values = channels
        .checked_mul(rows)
        .ok_or_else(|| packet_error("DIX1 entropy block value count overflows"))?;
    let mut entropy = Dix1EntropyDecoder::new(payload, channels, values, bit_depth)?;
    let mut stable_signal = (0..channels)
        .map(|_| Vec::with_capacity(rows))
        .collect::<Vec<_>>();
    for _ in 0..rows {
        let mut residuals = Vec::with_capacity(channels);
        for channel in 0..channels {
            residuals.push(entropy.read_value(channel)?);
        }
        let canonical_row = session.inverse_row(&residuals).map_err(as_packet_error)?;
        for (presented, &sample) in canonical_row.iter().enumerate() {
            let stable_id = usize::from(canonical_identities[presented].stable_id);
            stable_signal[stable_id].push(sample);
        }
    }
    entropy.finish()?;
    let events = u32::try_from(entropy.event_count())
        .map_err(|_| packet_error("DIX1 decoded block event count exceeds u32"))?;
    Ok((stable_signal, events))
}

fn advance_input_block(
    session: &mut Dix1Session,
    canonical_signal: &[Vec<i64>],
    start: usize,
    end: usize,
) -> Result<(), OptimumV2Error> {
    for sample in start..end {
        let row = canonical_signal
            .iter()
            .map(|channel| channel[sample])
            .collect::<Vec<_>>();
        session.forward_row(&row)?;
    }
    Ok(())
}

fn advance_decoded_block(
    session: &mut Dix1Session,
    canonical_identities: &[ChannelIdentity],
    stable_signal: &[Vec<i64>],
    rows: usize,
) -> Result<(), OptimumV2Error> {
    let first_channel = stable_signal
        .first()
        .ok_or_else(|| packet_error("DIX1 decoded block has no channels"))?;
    if first_channel.len() != rows || stable_signal.iter().any(|channel| channel.len() != rows) {
        return Err(packet_error("DIX1 decoded block shape is invalid"));
    }
    for (sample, _) in first_channel.iter().enumerate() {
        let row = canonical_identities
            .iter()
            .map(|identity| stable_signal[usize::from(identity.stable_id)][sample])
            .collect::<Vec<_>>();
        session.forward_row(&row).map_err(as_packet_error)?;
    }
    Ok(())
}

fn slice_signal(signal: &[Vec<i64>], start: usize, end: usize) -> Vec<Vec<i64>> {
    signal
        .iter()
        .map(|channel| channel[start..end].to_vec())
        .collect()
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
    use super::*;

    #[test]
    fn equal_length_candidate_ties_follow_wire_mode_order() {
        let candidate = choose_smallest([
            Candidate {
                mode: Dix1CarrierMode::IncidenceRans,
                payload: vec![3; 8],
            },
            Candidate {
                mode: Dix1CarrierMode::Delta,
                payload: vec![2; 8],
            },
            Candidate {
                mode: Dix1CarrierMode::Raw,
                payload: vec![1; 8],
            },
        ])
        .unwrap();
        assert_eq!(candidate.mode, Dix1CarrierMode::Raw);

        let candidate = choose_smallest([
            Candidate {
                mode: Dix1CarrierMode::IncidenceRans,
                payload: vec![3; 8],
            },
            Candidate {
                mode: Dix1CarrierMode::Delta,
                payload: vec![2; 8],
            },
        ])
        .unwrap();
        assert_eq!(candidate.mode, Dix1CarrierMode::Delta);
    }
}
