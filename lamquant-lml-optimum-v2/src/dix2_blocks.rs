//! Construction-private fixed-block carrier for DIX2 TreeMED.
//!
//! This seam deliberately stops below packet framing. It measures real rANS
//! bytes while DIX2 framing and the independent decoder remain unfrozen.

use crate::derivation_forest::DerivationForest;
use crate::derivation_incidence::{ChannelIdentity, DerivationIncidence};
use crate::dix1::{Dix1IncidenceMode, Dix1Session};
use crate::dix1_entropy::{Dix1EntropyDecoder, Dix1EntropyEncoder};
use crate::{
    canonical_i32_bytes, decode_delta_varints, decode_raw_i32, encode_delta_varints, OptimumV2Error,
};

const BLOCK_ROWS: usize = 128;
const DIRECTORY_ENTRY_LEN: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dix2CarrierMode {
    Raw,
    Delta,
    TemporalRans,
    TreeMedRans,
}

impl Dix2CarrierMode {
    fn wire(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::Delta => 1,
            Self::TemporalRans => 2,
            Self::TreeMedRans => 3,
        }
    }

    fn from_wire(value: u8) -> Result<Self, OptimumV2Error> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Delta),
            2 => Ok(Self::TemporalRans),
            3 => Ok(Self::TreeMedRans),
            _ => Err(packet_error("DIX2 block mode is invalid")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedDix2Blocks {
    pub directory: Vec<u8>,
    pub payload: Vec<u8>,
    pub modes: Vec<Dix2CarrierMode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedDix2Blocks {
    pub samples: Vec<Vec<i64>>,
    pub modes: Vec<Dix2CarrierMode>,
    pub event_count: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Dix2BlockCodec;

impl Dix2BlockCodec {
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<EncodedDix2Blocks, OptimumV2Error> {
        self.encode(signal, identities, sample_rate_mhz, bit_depth, None)
    }

    #[doc(hidden)]
    pub fn encode_forced(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        mode: Dix2CarrierMode,
    ) -> Result<EncodedDix2Blocks, OptimumV2Error> {
        self.encode(signal, identities, sample_rate_mhz, bit_depth, Some(mode))
    }

    pub fn decode_window(
        &self,
        encoded: &EncodedDix2Blocks,
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        samples: usize,
    ) -> Result<DecodedDix2Blocks, OptimumV2Error> {
        let prepared = PreparedIdentities::new(identities)?;
        validate_dimensions(prepared.channels, samples, bit_depth, sample_rate_mhz)?;
        let block_count = samples.div_ceil(BLOCK_ROWS);
        if encoded.directory.len() != block_count * DIRECTORY_ENTRY_LEN {
            return Err(packet_error("DIX2 compact directory length is invalid"));
        }
        let entries = parse_directory(&encoded.directory, encoded.payload.len())?;
        let directory_modes = entries.iter().map(|entry| entry.0).collect::<Vec<_>>();
        if encoded.modes != directory_modes {
            return Err(packet_error(
                "DIX2 mode summary disagrees with its directory",
            ));
        }

        let mut temporal = Dix1Session::new_with_incidence_mode(
            &prepared.canonical,
            bit_depth,
            sample_rate_mhz,
            Dix1IncidenceMode::Disabled,
        )
        .map_err(as_packet_error)?;
        let forest = DerivationForest::build(&prepared.canonical).map_err(as_packet_error)?;
        let tree_bit_depth = tree_bit_depth(bit_depth)?;
        let mut stable_signal = vec![Vec::with_capacity(samples); prepared.channels];
        let mut payload_offset = 0usize;
        let mut event_count = 0u32;

        for (block, (mode, length)) in entries.into_iter().enumerate() {
            let start = block * BLOCK_ROWS;
            let rows = (samples - start).min(BLOCK_ROWS);
            let end = payload_offset
                .checked_add(length)
                .ok_or_else(|| packet_error("DIX2 block payload end overflows"))?;
            let packed = encoded
                .payload
                .get(payload_offset..end)
                .ok_or_else(|| packet_error("DIX2 block payload is truncated"))?;
            let (block_signal, events) = match mode {
                Dix2CarrierMode::Raw => {
                    let values = prepared
                        .channels
                        .checked_mul(rows)
                        .ok_or_else(|| packet_error("DIX2 raw value count overflows"))?;
                    if packed.len() != values * 4 {
                        return Err(packet_error("DIX2 raw block length is invalid"));
                    }
                    let signal = decode_raw_i32(packed, prepared.channels, rows);
                    advance_native_block(
                        &mut temporal,
                        &forest,
                        &prepared.canonical,
                        &signal,
                        rows,
                    )?;
                    (signal, 0)
                }
                Dix2CarrierMode::Delta => {
                    let signal = decode_delta_varints(packed, prepared.channels, rows)?;
                    advance_native_block(
                        &mut temporal,
                        &forest,
                        &prepared.canonical,
                        &signal,
                        rows,
                    )?;
                    (signal, 0)
                }
                Dix2CarrierMode::TemporalRans => decode_entropy_block(
                    packed,
                    &mut temporal,
                    &forest,
                    &prepared.canonical,
                    rows,
                    bit_depth,
                    false,
                )?,
                Dix2CarrierMode::TreeMedRans => decode_entropy_block(
                    packed,
                    &mut temporal,
                    &forest,
                    &prepared.canonical,
                    rows,
                    tree_bit_depth,
                    true,
                )?,
            };
            for channel in 0..prepared.channels {
                stable_signal[channel].extend_from_slice(&block_signal[channel]);
            }
            event_count = event_count
                .checked_add(events)
                .ok_or_else(|| packet_error("DIX2 event count exceeds u32"))?;
            payload_offset = end;
        }
        if payload_offset != encoded.payload.len()
            || stable_signal.iter().any(|channel| channel.len() != samples)
        {
            return Err(packet_error("DIX2 block decode did not cover the window"));
        }
        Ok(DecodedDix2Blocks {
            samples: stable_signal,
            modes: encoded.modes.clone(),
            event_count,
        })
    }

    fn encode(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        forced: Option<Dix2CarrierMode>,
    ) -> Result<EncodedDix2Blocks, OptimumV2Error> {
        let prepared = PreparedInput::new(signal, identities, sample_rate_mhz, bit_depth)?;
        let mut temporal = Dix1Session::new_with_incidence_mode(
            &prepared.identities.canonical,
            bit_depth,
            sample_rate_mhz,
            Dix1IncidenceMode::Disabled,
        )?;
        let forest = DerivationForest::build(&prepared.identities.canonical)?;
        let tree_bit_depth = tree_bit_depth(bit_depth)?;
        let block_count = prepared.samples.div_ceil(BLOCK_ROWS);
        let mut directory = Vec::with_capacity(block_count * DIRECTORY_ENTRY_LEN);
        let mut payload = Vec::new();
        let mut modes = Vec::with_capacity(block_count);

        for block in 0..block_count {
            let start = block * BLOCK_ROWS;
            let end = (start + BLOCK_ROWS).min(prepared.samples);
            let stable_block = slice_signal(&prepared.stable_signal, start, end);
            let (temporal_rans, tree_rans) = encode_entropy_candidates(
                &mut temporal,
                &forest,
                &prepared.canonical_signal,
                start,
                end,
                bit_depth,
                tree_bit_depth,
            )?;
            let candidates = [
                Candidate {
                    mode: Dix2CarrierMode::Raw,
                    payload: canonical_i32_bytes(&stable_block)?,
                },
                Candidate {
                    mode: Dix2CarrierMode::Delta,
                    payload: encode_delta_varints(&stable_block),
                },
                Candidate {
                    mode: Dix2CarrierMode::TemporalRans,
                    payload: temporal_rans,
                },
                Candidate {
                    mode: Dix2CarrierMode::TreeMedRans,
                    payload: tree_rans,
                },
            ];
            let candidate = if let Some(mode) = forced {
                candidates
                    .into_iter()
                    .find(|candidate| candidate.mode == mode)
                    .ok_or_else(|| input_error("DIX2 forced candidate is unavailable"))?
            } else {
                candidates
                    .into_iter()
                    .min_by_key(|candidate| (candidate.payload.len(), candidate.mode))
                    .ok_or_else(|| input_error("DIX2 candidate set is empty"))?
            };
            let length = u32::try_from(candidate.payload.len())
                .map_err(|_| input_error("DIX2 block payload length exceeds u32"))?;
            directory.push(candidate.mode.wire());
            directory.extend_from_slice(&length.to_le_bytes());
            payload.extend_from_slice(&candidate.payload);
            modes.push(candidate.mode);
        }
        Ok(EncodedDix2Blocks {
            directory,
            payload,
            modes,
        })
    }
}

#[derive(Debug)]
struct Candidate {
    mode: Dix2CarrierMode,
    payload: Vec<u8>,
}

struct PreparedIdentities {
    canonical: Vec<ChannelIdentity>,
    channels: usize,
}

impl PreparedIdentities {
    fn new(identities: &[ChannelIdentity]) -> Result<Self, OptimumV2Error> {
        let channels = identities.len();
        if channels == 0 || channels > 64 {
            return Err(input_error("DIX2 channel count is outside bounds"));
        }
        let mut stable_ids = identities
            .iter()
            .map(|identity| usize::from(identity.stable_id))
            .collect::<Vec<_>>();
        stable_ids.sort_unstable();
        if stable_ids != (0..channels).collect::<Vec<_>>() {
            return Err(input_error("DIX2 stable IDs must be contiguous"));
        }
        let incidence = DerivationIncidence::build(identities)?;
        let canonical = incidence
            .channels()
            .iter()
            .map(|channel| identities[channel.presented_index()].clone())
            .collect();
        Ok(Self {
            canonical,
            channels,
        })
    }
}

struct PreparedInput {
    identities: PreparedIdentities,
    canonical_signal: Vec<Vec<i64>>,
    stable_signal: Vec<Vec<i64>>,
    samples: usize,
}

impl PreparedInput {
    fn new(
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Self, OptimumV2Error> {
        let prepared = PreparedIdentities::new(identities)?;
        let samples = signal.first().map_or(0, Vec::len);
        validate_dimensions(prepared.channels, samples, bit_depth, sample_rate_mhz)?;
        if signal.len() != prepared.channels
            || signal.iter().any(|channel| channel.len() != samples)
        {
            return Err(input_error("DIX2 signal shape is invalid"));
        }
        let minimum = -(1i64 << (bit_depth - 1));
        let maximum = (1i64 << (bit_depth - 1)) - 1;
        if signal
            .iter()
            .flatten()
            .any(|&sample| sample < minimum || sample > maximum)
        {
            return Err(input_error("DIX2 sample exceeds its declared bit depth"));
        }
        let incidence = DerivationIncidence::build(identities)?;
        let canonical_signal = incidence
            .channels()
            .iter()
            .map(|channel| signal[channel.presented_index()].clone())
            .collect();
        let mut stable_signal = vec![Vec::new(); prepared.channels];
        for (index, identity) in identities.iter().enumerate() {
            stable_signal[usize::from(identity.stable_id)] = signal[index].clone();
        }
        Ok(Self {
            identities: prepared,
            canonical_signal,
            stable_signal,
            samples,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_entropy_candidates(
    temporal: &mut Dix1Session,
    forest: &DerivationForest,
    canonical_signal: &[Vec<i64>],
    start: usize,
    end: usize,
    bit_depth: u8,
    tree_bit_depth: u8,
) -> Result<(Vec<u8>, Vec<u8>), OptimumV2Error> {
    let channels = canonical_signal.len();
    let values = channels
        .checked_mul(end - start)
        .ok_or_else(|| input_error("DIX2 block value count overflows"))?;
    let mut temporal_entropy = Dix1EntropyEncoder::new(channels, values, bit_depth)?;
    let mut tree_entropy = Dix1EntropyEncoder::new(channels, values, tree_bit_depth)?;
    for sample in start..end {
        let row = canonical_signal
            .iter()
            .map(|channel| channel[sample])
            .collect::<Vec<_>>();
        let innovations = temporal.forward_row(&row)?;
        let residuals = forest.forward_canonical_innovations(&innovations)?;
        for channel in 0..channels {
            temporal_entropy.push_value(channel, innovations[channel])?;
            tree_entropy.push_value(channel, residuals[channel])?;
        }
    }
    Ok((temporal_entropy.finish()?, tree_entropy.finish()?))
}

#[allow(clippy::too_many_arguments)]
fn decode_entropy_block(
    packed: &[u8],
    temporal: &mut Dix1Session,
    forest: &DerivationForest,
    canonical_identities: &[ChannelIdentity],
    rows: usize,
    entropy_bit_depth: u8,
    tree: bool,
) -> Result<(Vec<Vec<i64>>, u32), OptimumV2Error> {
    let channels = canonical_identities.len();
    let values = channels
        .checked_mul(rows)
        .ok_or_else(|| packet_error("DIX2 entropy value count overflows"))?;
    let mut entropy = Dix1EntropyDecoder::new(packed, channels, values, entropy_bit_depth)?;
    let mut stable_signal = vec![Vec::with_capacity(rows); channels];
    for _ in 0..rows {
        let mut coded = Vec::with_capacity(channels);
        for channel in 0..channels {
            coded.push(entropy.read_value(channel)?);
        }
        let innovations = if tree {
            forest.inverse_canonical_innovations(&coded)?
        } else {
            forest.forward_canonical_innovations(&coded)?;
            coded
        };
        let canonical = temporal
            .inverse_row(&innovations)
            .map_err(as_packet_error)?;
        for (channel, &sample) in canonical.iter().enumerate() {
            stable_signal[usize::from(canonical_identities[channel].stable_id)].push(sample);
        }
    }
    entropy.finish()?;
    let events = u32::try_from(entropy.event_count())
        .map_err(|_| packet_error("DIX2 entropy event count exceeds u32"))?;
    Ok((stable_signal, events))
}

fn advance_native_block(
    temporal: &mut Dix1Session,
    forest: &DerivationForest,
    canonical_identities: &[ChannelIdentity],
    stable_signal: &[Vec<i64>],
    rows: usize,
) -> Result<(), OptimumV2Error> {
    if stable_signal.len() != canonical_identities.len()
        || stable_signal.iter().any(|channel| channel.len() != rows)
    {
        return Err(packet_error("DIX2 native block shape is invalid"));
    }
    for (sample, _) in stable_signal[0].iter().enumerate() {
        let row = canonical_identities
            .iter()
            .map(|identity| stable_signal[usize::from(identity.stable_id)][sample])
            .collect::<Vec<_>>();
        let innovations = temporal.forward_row(&row).map_err(as_packet_error)?;
        forest
            .forward_canonical_innovations(&innovations)
            .map_err(as_packet_error)?;
    }
    Ok(())
}

fn parse_directory(
    directory: &[u8],
    payload_len: usize,
) -> Result<Vec<(Dix2CarrierMode, usize)>, OptimumV2Error> {
    let mut entries = Vec::with_capacity(directory.len() / DIRECTORY_ENTRY_LEN);
    let mut sum = 0usize;
    for entry in directory.chunks_exact(DIRECTORY_ENTRY_LEN) {
        let mode = Dix2CarrierMode::from_wire(entry[0])?;
        let length = usize::try_from(u32::from_le_bytes(entry[1..5].try_into().unwrap()))
            .map_err(|_| packet_error("DIX2 block length exceeds usize"))?;
        sum = sum
            .checked_add(length)
            .ok_or_else(|| packet_error("DIX2 payload sum overflows"))?;
        entries.push((mode, length));
    }
    if sum != payload_len {
        return Err(packet_error("DIX2 directory does not cover its payload"));
    }
    Ok(entries)
}

fn validate_dimensions(
    channels: usize,
    samples: usize,
    bit_depth: u8,
    sample_rate_mhz: u32,
) -> Result<(), OptimumV2Error> {
    let values = channels.checked_mul(samples);
    if channels == 0
        || channels > 64
        || samples == 0
        || samples > 32_768
        || values.map_or(true, |values| values > 131_072)
        || !(1..=32).contains(&bit_depth)
        || !(1..=4_000_000).contains(&sample_rate_mhz)
    {
        return Err(input_error(
            "DIX2 dimensions are outside construction bounds",
        ));
    }
    Ok(())
}

fn tree_bit_depth(bit_depth: u8) -> Result<u8, OptimumV2Error> {
    bit_depth
        .checked_add(1)
        .ok_or_else(|| input_error("DIX2 TreeMED guard bit overflows"))
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
