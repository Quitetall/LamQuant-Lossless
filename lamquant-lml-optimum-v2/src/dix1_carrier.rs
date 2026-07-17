//! Experimental self-decoding DIX1 construction carrier.
//!
//! `DIX1` packets use the full Optimum-v2 envelope for honest byte accounting,
//! but remain construction-private: they are not a released LMO mode or wire
//! compatibility promise. Identity, deterministic topology, directory, rANS,
//! framing, and both CRC fields are all present in the returned bytes.

use crate::derivation_incidence::{ChannelIdentity, DerivationIncidence};
use crate::dix1::{Dix1IncidenceMode, Dix1Session};
use crate::dix1_entropy::{Dix1EntropyDecoder, Dix1EntropyEncoder};
use crate::{
    canonical_i32_bytes, crc32c, crc32c_zeroed_field, decode_delta_varints, decode_raw_i32,
    encode_delta_varints, read_u16, read_u32, OptimumV2Error,
};

const CONSTRUCTION_FLAGS: u8 = 1;
const ENTROPY_PROFILE_NATIVE: u8 = 0;
const ENTROPY_PROFILE_KT_RANS_V1: u8 = 1;
const BODY_HEADER_LEN: usize = 80;
const DIRECTORY_LEN: usize = 24;
const PACKET_CRC_OFFSET: usize = 7 + 76;
const MAX_CHANNELS: usize = 64;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 131_072;
const MAX_LABEL_BYTES: usize = 255;
const MAX_SUPPORTS: usize = 4;
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const MAX_SAMPLE_RATE_MHZ: u32 = 4_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dix1CarrierMode {
    Raw,
    Delta,
    IncidenceRans,
    NoIncidenceRans,
}

impl Dix1CarrierMode {
    fn wire(self) -> u8 {
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
            _ => Err(packet_error("DIX1 construction mode is invalid")),
        }
    }

    fn is_entropy(self) -> bool {
        matches!(self, Self::IncidenceRans | Self::NoIncidenceRans)
    }

    fn incidence_mode(self) -> Option<Dix1IncidenceMode> {
        match self {
            Self::IncidenceRans => Some(Dix1IncidenceMode::Enabled),
            Self::NoIncidenceRans => Some(Dix1IncidenceMode::Disabled),
            Self::Raw | Self::Delta => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedDix1Window {
    pub samples: Vec<Vec<i64>>,
    pub identities: Vec<ChannelIdentity>,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub mode: Dix1CarrierMode,
    pub event_count: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Dix1ConstructionCodec;

impl Dix1ConstructionCodec {
    /// Encode the product construction arm. Native raw/delta wins every tie;
    /// DIX1 is selected only when its complete packet is strictly smaller.
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.smallest_packet(
            signal,
            identities,
            sample_rate_mhz,
            bit_depth,
            &[
                Dix1CarrierMode::Raw,
                Dix1CarrierMode::Delta,
                Dix1CarrierMode::IncidenceRans,
            ],
        )
    }

    /// Encode the matched native construction control with the same identity,
    /// topology, framing, and integrity overhead as the DIX1 arm.
    pub fn encode_native_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.smallest_packet(
            signal,
            identities,
            sample_rate_mhz,
            bit_depth,
            &[Dix1CarrierMode::Raw, Dix1CarrierMode::Delta],
        )
    }

    #[doc(hidden)]
    pub fn encode_forced(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        mode: Dix1CarrierMode,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let prepared = PreparedInput::new(signal, identities, sample_rate_mhz, bit_depth)?;
        self.encode_prepared(&prepared, mode)
    }

    pub fn decode_window(&self, packed: &[u8]) -> Result<DecodedDix1Window, OptimumV2Error> {
        if packed.len() < 7 + BODY_HEADER_LEN || packed.len() > MAX_PACKET_BYTES {
            return Err(packet_error("DIX1 packet length is outside bounds"));
        }
        if &packed[..7] != b"LMO1\x03\x00\x03" {
            return Err(packet_error("DIX1 LMO1 construction envelope is invalid"));
        }
        if &packed[7..11] != b"DIX1" || packed[11] != 1 {
            return Err(packet_error("DIX1 body magic/version is invalid"));
        }
        let flags = packed[12];
        let bit_depth = packed[13];
        let reserved = packed[14];
        let channels = usize::from(read_u16(packed, 15)?);
        let tile_count = read_u16(packed, 17)?;
        let samples = usize::try_from(read_u32(packed, 19)?)
            .map_err(|_| packet_error("DIX1 sample count exceeds usize"))?;
        let sample_rate_mhz = read_u32(packed, 23)?;
        let model_id = read_u32(packed, 27)?;
        let model_sha = packed
            .get(31..63)
            .ok_or_else(|| packet_error("DIX1 model digest is truncated"))?;
        let identity_len = usize::try_from(read_u32(packed, 63)?)
            .map_err(|_| packet_error("DIX1 identity length exceeds usize"))?;
        let topology_len = usize::try_from(read_u32(packed, 67)?)
            .map_err(|_| packet_error("DIX1 topology length exceeds usize"))?;
        let directory_len = usize::try_from(read_u32(packed, 71)?)
            .map_err(|_| packet_error("DIX1 directory length exceeds usize"))?;
        let payload_len = usize::try_from(read_u32(packed, 75)?)
            .map_err(|_| packet_error("DIX1 payload length exceeds usize"))?;
        let decoded_crc = read_u32(packed, 79)?;
        let packet_crc = read_u32(packed, 83)?;
        let values = validate_dimensions(
            channels,
            samples,
            bit_depth,
            sample_rate_mhz,
            InputKind::Packet,
        )?;
        if flags != CONSTRUCTION_FLAGS
            || reserved != 0
            || tile_count != 1
            || model_id != 0
            || model_sha != [0u8; 32]
            || directory_len != DIRECTORY_LEN
            || !(channels * 4..=channels * (3 + MAX_LABEL_BYTES)).contains(&identity_len)
            || !(channels..=channels * (1 + 2 * MAX_SUPPORTS)).contains(&topology_len)
        {
            return Err(packet_error("DIX1 construction header is invalid"));
        }
        let identity_start = 7 + BODY_HEADER_LEN;
        let identity_end = identity_start
            .checked_add(identity_len)
            .ok_or_else(|| packet_error("DIX1 identity end overflows"))?;
        let topology_end = identity_end
            .checked_add(topology_len)
            .ok_or_else(|| packet_error("DIX1 topology end overflows"))?;
        let directory_end = topology_end
            .checked_add(directory_len)
            .ok_or_else(|| packet_error("DIX1 directory end overflows"))?;
        let packet_end = directory_end
            .checked_add(payload_len)
            .ok_or_else(|| packet_error("DIX1 packet end overflows"))?;
        if packet_end != packed.len() {
            return Err(packet_error("DIX1 section lengths do not match the packet"));
        }
        if packet_crc != crc32c_zeroed_field(packed, PACKET_CRC_OFFSET) {
            return Err(OptimumV2Error::Integrity(
                "DIX1 packet CRC32C mismatch".into(),
            ));
        }

        let canonical_identities =
            decode_identities(&packed[identity_start..identity_end], channels)?;
        let incidence = DerivationIncidence::build(&canonical_identities)
            .map_err(|error| packet_error(error.to_string()))?;
        if encode_identities(&incidence, &canonical_identities)?
            != packed[identity_start..identity_end]
        {
            return Err(packet_error(
                "DIX1 identity section is not exact canonical order",
            ));
        }
        if encode_topology(&incidence)? != packed[identity_end..topology_end] {
            return Err(packet_error(
                "DIX1 topology does not match deterministic derivation incidence",
            ));
        }

        let directory = &packed[topology_end..directory_end];
        let mode = Dix1CarrierMode::from_wire(directory[0])?;
        let entropy_profile = directory[1];
        let directory_flags = read_u16(directory, 2)?;
        let first_sample = read_u32(directory, 4)?;
        let tile_samples = usize::try_from(read_u32(directory, 8)?)
            .map_err(|_| packet_error("DIX1 tile sample count exceeds usize"))?;
        let payload_offset = read_u32(directory, 12)?;
        let tile_payload_len = usize::try_from(read_u32(directory, 16)?)
            .map_err(|_| packet_error("DIX1 tile payload length exceeds usize"))?;
        let event_count = read_u32(directory, 20)?;
        let expected_profile = if mode.is_entropy() {
            ENTROPY_PROFILE_KT_RANS_V1
        } else {
            ENTROPY_PROFILE_NATIVE
        };
        if entropy_profile != expected_profile
            || directory_flags != 0
            || first_sample != 0
            || tile_samples != samples
            || payload_offset != 0
            || tile_payload_len != payload_len
        {
            return Err(packet_error("DIX1 tile directory is invalid"));
        }
        let max_events = values
            .checked_mul(2 * usize::from(bit_depth) + 1)
            .ok_or_else(|| packet_error("DIX1 maximum event count overflows"))?;
        if mode.is_entropy() {
            if payload_len < 4
                || usize::try_from(event_count).unwrap_or(usize::MAX) < values
                || usize::try_from(event_count).unwrap_or(usize::MAX) > max_events
            {
                return Err(packet_error("DIX1 entropy event count is outside bounds"));
            }
        } else if event_count != 0 {
            return Err(packet_error("DIX1 native mode carries entropy events"));
        }

        let payload = &packed[directory_end..];
        let stable_signal = match mode {
            Dix1CarrierMode::Raw => {
                if payload_len != values * 4 {
                    return Err(packet_error("DIX1 raw payload length is invalid"));
                }
                decode_raw_i32(payload, channels, samples)
            }
            Dix1CarrierMode::Delta => decode_delta_varints(payload, channels, samples)?,
            Dix1CarrierMode::IncidenceRans | Dix1CarrierMode::NoIncidenceRans => {
                decode_entropy_payload(
                    payload,
                    &canonical_identities,
                    channels,
                    samples,
                    values,
                    sample_rate_mhz,
                    bit_depth,
                    mode.incidence_mode().unwrap(),
                    event_count,
                )?
            }
        };
        if crc32c(&canonical_i32_bytes(&stable_signal)?) != decoded_crc {
            return Err(OptimumV2Error::Integrity(
                "DIX1 decoded-sample CRC32C mismatch".into(),
            ));
        }
        let mut stable_identities = canonical_identities.clone();
        stable_identities.sort_by_key(|identity| identity.stable_id);
        let canonical = self
            .encode_forced(
                &stable_signal,
                &stable_identities,
                sample_rate_mhz,
                bit_depth,
                mode,
            )
            .map_err(as_packet_error)?;
        if canonical != packed {
            return Err(packet_error("DIX1 carrier is not byte-canonical"));
        }
        Ok(DecodedDix1Window {
            samples: stable_signal,
            identities: stable_identities,
            sample_rate_mhz,
            bit_depth,
            mode,
            event_count,
        })
    }

    fn smallest_packet(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        modes: &[Dix1CarrierMode],
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let prepared = PreparedInput::new(signal, identities, sample_rate_mhz, bit_depth)?;
        let mut best: Option<(usize, Dix1CarrierMode, Vec<u8>)> = None;
        for &mode in modes {
            let packet = self.encode_prepared(&prepared, mode)?;
            let key = (packet.len(), mode);
            if best
                .as_ref()
                .map_or(true, |(length, best_mode, _)| key < (*length, *best_mode))
            {
                best = Some((packet.len(), mode, packet));
            }
        }
        best.map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("DIX1 construction mode set is empty"))
    }

    fn encode_prepared(
        &self,
        prepared: &PreparedInput,
        mode: Dix1CarrierMode,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let (payload, event_count) = match mode {
            Dix1CarrierMode::Raw => (canonical_i32_bytes(&prepared.stable_signal)?, 0),
            Dix1CarrierMode::Delta => (encode_delta_varints(&prepared.stable_signal), 0),
            Dix1CarrierMode::IncidenceRans | Dix1CarrierMode::NoIncidenceRans => {
                encode_entropy_payload(prepared, mode.incidence_mode().unwrap())?
            }
        };
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| input_error("DIX1 payload length exceeds u32"))?;
        let mut directory = Vec::with_capacity(DIRECTORY_LEN);
        directory.push(mode.wire());
        directory.push(if mode.is_entropy() {
            ENTROPY_PROFILE_KT_RANS_V1
        } else {
            ENTROPY_PROFILE_NATIVE
        });
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(&(prepared.samples as u32).to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(&payload_len.to_le_bytes());
        directory.extend_from_slice(&event_count.to_le_bytes());
        debug_assert_eq!(directory.len(), DIRECTORY_LEN);

        let decoded_crc = crc32c(&canonical_i32_bytes(&prepared.stable_signal)?);
        let mut packed = Vec::new();
        packed.extend_from_slice(b"LMO1\x03\x00\x03");
        packed.extend_from_slice(b"DIX1");
        packed.push(1);
        packed.push(CONSTRUCTION_FLAGS);
        packed.push(prepared.bit_depth);
        packed.push(0);
        packed.extend_from_slice(&(prepared.channels as u16).to_le_bytes());
        packed.extend_from_slice(&1u16.to_le_bytes());
        packed.extend_from_slice(&(prepared.samples as u32).to_le_bytes());
        packed.extend_from_slice(&prepared.sample_rate_mhz.to_le_bytes());
        packed.extend_from_slice(&0u32.to_le_bytes());
        packed.extend_from_slice(&[0u8; 32]);
        packed.extend_from_slice(&(prepared.identity_bytes.len() as u32).to_le_bytes());
        packed.extend_from_slice(&(prepared.topology_bytes.len() as u32).to_le_bytes());
        packed.extend_from_slice(&(DIRECTORY_LEN as u32).to_le_bytes());
        packed.extend_from_slice(&payload_len.to_le_bytes());
        packed.extend_from_slice(&decoded_crc.to_le_bytes());
        packed.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(packed.len(), 7 + BODY_HEADER_LEN);
        packed.extend_from_slice(&prepared.identity_bytes);
        packed.extend_from_slice(&prepared.topology_bytes);
        packed.extend_from_slice(&directory);
        packed.extend_from_slice(&payload);
        if packed.len() > MAX_PACKET_BYTES {
            return Err(input_error("DIX1 packet exceeds its 64 MiB bound"));
        }
        let packet_crc = crc32c_zeroed_field(&packed, PACKET_CRC_OFFSET);
        packed[PACKET_CRC_OFFSET..PACKET_CRC_OFFSET + 4].copy_from_slice(&packet_crc.to_le_bytes());
        Ok(packed)
    }
}

#[derive(Debug, Clone)]
struct PreparedInput {
    canonical_identities: Vec<ChannelIdentity>,
    canonical_signal: Vec<Vec<i64>>,
    stable_signal: Vec<Vec<i64>>,
    identity_bytes: Vec<u8>,
    topology_bytes: Vec<u8>,
    channels: usize,
    samples: usize,
    sample_rate_mhz: u32,
    bit_depth: u8,
}

impl PreparedInput {
    fn new(
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Self, OptimumV2Error> {
        let channels = identities.len();
        let samples = signal.first().map_or(0, Vec::len);
        validate_dimensions(
            channels,
            samples,
            bit_depth,
            sample_rate_mhz,
            InputKind::Caller,
        )?;
        if signal.len() != channels || signal.iter().any(|channel| channel.len() != samples) {
            return Err(input_error(
                "DIX1 signal shape does not match channel identities",
            ));
        }
        let minimum = -(1i64 << (bit_depth - 1));
        let maximum = (1i64 << (bit_depth - 1)) - 1;
        if signal
            .iter()
            .flatten()
            .any(|&sample| sample < minimum || sample > maximum)
        {
            return Err(input_error("DIX1 sample exceeds the declared bit depth"));
        }
        validate_stable_ids(identities, InputKind::Caller)?;
        let incidence = DerivationIncidence::build(identities)?;
        let canonical_identities = incidence
            .channels()
            .iter()
            .map(|channel| identities[channel.presented_index()].clone())
            .collect::<Vec<_>>();
        validate_wire_labels(&canonical_identities, InputKind::Caller)?;
        let canonical_signal = incidence
            .channels()
            .iter()
            .map(|channel| signal[channel.presented_index()].clone())
            .collect::<Vec<_>>();
        let mut stable_signal = vec![Vec::new(); channels];
        for (index, identity) in identities.iter().enumerate() {
            stable_signal[usize::from(identity.stable_id)] = signal[index].clone();
        }
        Ok(Self {
            identity_bytes: encode_identities(&incidence, identities)?,
            topology_bytes: encode_topology(&incidence)?,
            canonical_identities,
            canonical_signal,
            stable_signal,
            channels,
            samples,
            sample_rate_mhz,
            bit_depth,
        })
    }
}

fn encode_entropy_payload(
    prepared: &PreparedInput,
    incidence_mode: Dix1IncidenceMode,
) -> Result<(Vec<u8>, u32), OptimumV2Error> {
    let values = prepared
        .channels
        .checked_mul(prepared.samples)
        .ok_or_else(|| input_error("DIX1 value count overflows"))?;
    let mut session = Dix1Session::new_with_incidence_mode(
        &prepared.canonical_identities,
        prepared.bit_depth,
        prepared.sample_rate_mhz,
        incidence_mode,
    )?;
    let mut entropy = Dix1EntropyEncoder::new(prepared.channels, values, prepared.bit_depth)?;
    for sample in 0..prepared.samples {
        let row = prepared
            .canonical_signal
            .iter()
            .map(|channel| channel[sample])
            .collect::<Vec<_>>();
        let residuals = session.forward_row(&row)?;
        for (channel, residual) in residuals.into_iter().enumerate() {
            entropy.push_value(channel, residual)?;
        }
    }
    let event_count = u32::try_from(entropy.event_count())
        .map_err(|_| input_error("DIX1 entropy event count exceeds u32"))?;
    Ok((entropy.finish()?, event_count))
}

#[allow(clippy::too_many_arguments)]
fn decode_entropy_payload(
    payload: &[u8],
    canonical_identities: &[ChannelIdentity],
    channels: usize,
    samples: usize,
    values: usize,
    sample_rate_mhz: u32,
    bit_depth: u8,
    incidence_mode: Dix1IncidenceMode,
    expected_events: u32,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let mut session = Dix1Session::new_with_incidence_mode(
        canonical_identities,
        bit_depth,
        sample_rate_mhz,
        incidence_mode,
    )
    .map_err(as_packet_error)?;
    let mut entropy = Dix1EntropyDecoder::new(payload, channels, values, bit_depth)?;
    let mut stable_signal = (0..channels)
        .map(|_| Vec::with_capacity(samples))
        .collect::<Vec<_>>();
    for _ in 0..samples {
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
    if entropy.event_count() != usize::try_from(expected_events).unwrap_or(usize::MAX) {
        return Err(packet_error(
            "DIX1 decoded event count does not match the directory",
        ));
    }
    entropy.finish()?;
    Ok(stable_signal)
}

fn encode_identities(
    incidence: &DerivationIncidence,
    presented_identities: &[ChannelIdentity],
) -> Result<Vec<u8>, OptimumV2Error> {
    let mut packed = Vec::new();
    for channel in incidence.channels() {
        let identity = presented_identities
            .get(channel.presented_index())
            .ok_or_else(|| input_error("DIX1 identity presentation rank is out of range"))?;
        if identity.stable_id != channel.stable_id() {
            return Err(input_error(
                "DIX1 identity stable ID disagrees with canonical incidence",
            ));
        }
        let label = identity.label.as_bytes();
        if label.is_empty()
            || label.len() > MAX_LABEL_BYTES
            || !label.iter().all(|byte| (0x20..=0x7e).contains(byte))
        {
            return Err(input_error(
                "DIX1 construction labels must be printable normalized ASCII",
            ));
        }
        packed.extend_from_slice(&channel.stable_id().to_le_bytes());
        packed.push(label.len() as u8);
        packed.extend_from_slice(label);
    }
    Ok(packed)
}

fn decode_identities(
    packed: &[u8],
    channels: usize,
) -> Result<Vec<ChannelIdentity>, OptimumV2Error> {
    let mut identities = Vec::with_capacity(channels);
    let mut offset = 0usize;
    for _ in 0..channels {
        let stable_id = read_u16(packed, offset)?;
        offset = offset
            .checked_add(2)
            .ok_or_else(|| packet_error("DIX1 identity offset overflows"))?;
        let label_len = usize::from(
            *packed
                .get(offset)
                .ok_or_else(|| packet_error("DIX1 identity label length is truncated"))?,
        );
        offset = offset
            .checked_add(1)
            .ok_or_else(|| packet_error("DIX1 identity label offset overflows"))?;
        let end = offset
            .checked_add(label_len)
            .ok_or_else(|| packet_error("DIX1 identity label end overflows"))?;
        let label = packed
            .get(offset..end)
            .ok_or_else(|| packet_error("DIX1 identity label is truncated"))?;
        if label.is_empty() || !label.iter().all(|byte| (0x20..=0x7e).contains(byte)) {
            return Err(packet_error(
                "DIX1 construction label is not printable ASCII",
            ));
        }
        let label = std::str::from_utf8(label)
            .map_err(|_| packet_error("DIX1 identity label is not UTF-8"))?;
        identities.push(ChannelIdentity::new(stable_id, label));
        offset = end;
    }
    if offset != packed.len() {
        return Err(packet_error("DIX1 identity section has trailing bytes"));
    }
    validate_stable_ids(&identities, InputKind::Packet)?;
    validate_wire_labels(&identities, InputKind::Packet)?;
    Ok(identities)
}

fn encode_topology(incidence: &DerivationIncidence) -> Result<Vec<u8>, OptimumV2Error> {
    let mut packed = Vec::new();
    for channel in incidence.channels() {
        let supports = channel.supports();
        if supports.len() > MAX_SUPPORTS {
            return Err(input_error("DIX1 topology support count exceeds four"));
        }
        packed.push(supports.len() as u8);
        for support in supports {
            packed.push(
                u8::try_from(support.prior_channel)
                    .map_err(|_| input_error("DIX1 topology rank exceeds u8"))?,
            );
            packed.push(support.coefficient as u8);
        }
    }
    Ok(packed)
}

fn validate_dimensions(
    channels: usize,
    samples: usize,
    bit_depth: u8,
    sample_rate_mhz: u32,
    kind: InputKind,
) -> Result<usize, OptimumV2Error> {
    let values = channels.checked_mul(samples);
    let valid = (1..=MAX_CHANNELS).contains(&channels)
        && (1..=MAX_SAMPLES).contains(&samples)
        && (1..=32).contains(&bit_depth)
        && (1..=MAX_SAMPLE_RATE_MHZ).contains(&sample_rate_mhz)
        && values.is_some_and(|value| value <= MAX_VALUES);
    if valid {
        return Ok(values.unwrap());
    }
    let message = "DIX1 dimensions, bit depth, sample rate, or value count are outside bounds";
    Err(match kind {
        InputKind::Caller => input_error(message),
        InputKind::Packet => packet_error(message),
    })
}

fn validate_stable_ids(
    identities: &[ChannelIdentity],
    kind: InputKind,
) -> Result<(), OptimumV2Error> {
    let mut seen = vec![false; identities.len()];
    for identity in identities {
        let index = usize::from(identity.stable_id);
        if index >= identities.len() || seen[index] {
            let message = "DIX1 wire stable IDs must be exactly contiguous and unique";
            return Err(match kind {
                InputKind::Caller => input_error(message),
                InputKind::Packet => packet_error(message),
            });
        }
        seen[index] = true;
    }
    Ok(())
}

fn validate_wire_labels(
    identities: &[ChannelIdentity],
    kind: InputKind,
) -> Result<(), OptimumV2Error> {
    if identities.iter().all(|identity| {
        !identity.label.is_empty()
            && identity.label.len() <= MAX_LABEL_BYTES
            && identity
                .label
                .bytes()
                .all(|byte| (0x20..=0x7e).contains(&byte))
    }) {
        return Ok(());
    }
    let message = "DIX1 construction labels must be printable ASCII";
    Err(match kind {
        InputKind::Caller => input_error(message),
        InputKind::Packet => packet_error(message),
    })
}

#[derive(Clone, Copy)]
enum InputKind {
    Caller,
    Packet,
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
