//! Experimental self-decoding DIX2 TreeMED construction carrier.
//!
//! `DIX2` is additive and construction-private. It does not change the DIX1
//! packet law or allocate a released LMO mode.

use crate::derivation_forest::DerivationForest;
use crate::derivation_incidence::{ChannelIdentity, DerivationIncidence};
pub use crate::dix2_blocks::Dix2CarrierMode;
use crate::dix2_blocks::{Dix2BlockCodec, EncodedDix2Blocks};
use crate::OptimumV2Error;
use crate::{canonical_i32_bytes, crc32c, crc32c_zeroed_field, read_u16, read_u32};

const CONSTRUCTION_FLAGS: u8 = 1;
const BODY_VERSION: u8 = 1;
const BODY_HEADER_LEN: usize = 80;
const PACKET_CRC_OFFSET: usize = 7 + 76;
const DIRECTORY_ENTRY_LEN: usize = 5;
const MAX_CHANNELS: usize = 64;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 131_072;
const MAX_LABEL_BYTES: usize = 255;
const MAX_SUPPORTS: usize = 3;
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const MAX_SAMPLE_RATE_MHZ: u32 = 4_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedDix2Window {
    pub samples: Vec<Vec<i64>>,
    pub identities: Vec<ChannelIdentity>,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub tile_modes: Vec<Dix2CarrierMode>,
    pub event_count: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Dix2ConstructionCodec;

impl Dix2ConstructionCodec {
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_with_profile(
            signal,
            identities,
            sample_rate_mhz,
            bit_depth,
            CarrierProfile::Product,
        )
    }

    pub fn encode_native_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_with_profile(
            signal,
            identities,
            sample_rate_mhz,
            bit_depth,
            CarrierProfile::Native,
        )
    }

    #[doc(hidden)]
    pub fn encode_forced(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        mode: Dix2CarrierMode,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_with_profile(
            signal,
            identities,
            sample_rate_mhz,
            bit_depth,
            CarrierProfile::forced(mode),
        )
    }

    pub fn decode_window(&self, packed: &[u8]) -> Result<DecodedDix2Window, OptimumV2Error> {
        if packed.len() < 7 + BODY_HEADER_LEN || packed.len() > MAX_PACKET_BYTES {
            return Err(packet_error("DIX2 packet length is outside bounds"));
        }
        if &packed[..7] != b"LMO1\x03\x00\x03" {
            return Err(packet_error("DIX2 LMO1 construction envelope is invalid"));
        }
        if &packed[7..11] != b"DIX2" || packed[11] != BODY_VERSION {
            return Err(packet_error("DIX2 body magic/version is invalid"));
        }
        let flags = packed[12];
        let bit_depth = packed[13];
        let profile = CarrierProfile::from_wire(packed[14])?;
        let channels = usize::from(read_u16(packed, 15)?);
        let tile_count = usize::from(read_u16(packed, 17)?);
        let samples = usize::try_from(read_u32(packed, 19)?)
            .map_err(|_| packet_error("DIX2 sample count exceeds usize"))?;
        let sample_rate_mhz = read_u32(packed, 23)?;
        let model_id = read_u32(packed, 27)?;
        let model_sha = packed
            .get(31..63)
            .ok_or_else(|| packet_error("DIX2 model digest is truncated"))?;
        let identity_len = usize::try_from(read_u32(packed, 63)?)
            .map_err(|_| packet_error("DIX2 identity length exceeds usize"))?;
        let topology_len = usize::try_from(read_u32(packed, 67)?)
            .map_err(|_| packet_error("DIX2 topology length exceeds usize"))?;
        let directory_len = usize::try_from(read_u32(packed, 71)?)
            .map_err(|_| packet_error("DIX2 directory length exceeds usize"))?;
        let payload_len = usize::try_from(read_u32(packed, 75)?)
            .map_err(|_| packet_error("DIX2 payload length exceeds usize"))?;
        let decoded_crc = read_u32(packed, 79)?;
        let packet_crc = read_u32(packed, 83)?;
        validate_dimensions(
            channels,
            samples,
            bit_depth,
            sample_rate_mhz,
            InputKind::Packet,
        )?;
        let expected_tiles = samples.div_ceil(128);
        let expected_directory_len = expected_tiles
            .checked_mul(DIRECTORY_ENTRY_LEN)
            .ok_or_else(|| packet_error("DIX2 directory length overflows"))?;
        if flags != CONSTRUCTION_FLAGS
            || tile_count != expected_tiles
            || model_id != 0
            || model_sha != [0u8; 32]
            || directory_len != expected_directory_len
            || !(channels * 4..=channels * (3 + MAX_LABEL_BYTES)).contains(&identity_len)
            || !(channels..=channels * (1 + 2 * MAX_SUPPORTS)).contains(&topology_len)
        {
            return Err(packet_error("DIX2 construction header is invalid"));
        }
        let identity_start = 7 + BODY_HEADER_LEN;
        let identity_end = identity_start
            .checked_add(identity_len)
            .ok_or_else(|| packet_error("DIX2 identity end overflows"))?;
        let topology_end = identity_end
            .checked_add(topology_len)
            .ok_or_else(|| packet_error("DIX2 topology end overflows"))?;
        let directory_end = topology_end
            .checked_add(directory_len)
            .ok_or_else(|| packet_error("DIX2 directory end overflows"))?;
        let packet_end = directory_end
            .checked_add(payload_len)
            .ok_or_else(|| packet_error("DIX2 packet end overflows"))?;
        if packet_end != packed.len() {
            return Err(packet_error("DIX2 section lengths do not match the packet"));
        }
        if packet_crc != crc32c_zeroed_field(packed, PACKET_CRC_OFFSET) {
            return Err(OptimumV2Error::Integrity(
                "DIX2 packet CRC32C mismatch".into(),
            ));
        }

        let canonical_identities =
            decode_identities(&packed[identity_start..identity_end], channels)?;
        if encode_identities(&canonical_identities)? != packed[identity_start..identity_end] {
            return Err(packet_error(
                "DIX2 identity section is not exact canonical order",
            ));
        }
        if encode_topology(&canonical_identities)? != packed[identity_end..topology_end] {
            return Err(packet_error(
                "DIX2 topology does not match its deterministic forest",
            ));
        }
        let modes = decode_modes(&packed[topology_end..directory_end], profile)?;
        let encoded = EncodedDix2Blocks {
            directory: packed[topology_end..directory_end].to_vec(),
            payload: packed[directory_end..].to_vec(),
            modes,
        };
        let blocks = Dix2BlockCodec.decode_window(
            &encoded,
            &canonical_identities,
            sample_rate_mhz,
            bit_depth,
            samples,
        )?;
        if crc32c(&canonical_i32_bytes(&blocks.samples)?) != decoded_crc {
            return Err(OptimumV2Error::Integrity(
                "DIX2 decoded-sample CRC32C mismatch".into(),
            ));
        }
        let mut stable_identities = canonical_identities.clone();
        stable_identities.sort_by_key(|identity| identity.stable_id);
        let canonical = self
            .encode_with_profile(
                &blocks.samples,
                &stable_identities,
                sample_rate_mhz,
                bit_depth,
                profile,
            )
            .map_err(as_packet_error)?;
        if canonical != packed {
            return Err(packet_error("DIX2 carrier is not byte-canonical"));
        }
        Ok(DecodedDix2Window {
            samples: blocks.samples,
            identities: stable_identities,
            sample_rate_mhz,
            bit_depth,
            tile_modes: blocks.modes,
            event_count: blocks.event_count,
        })
    }

    fn encode_with_profile(
        &self,
        signal: &[Vec<i64>],
        identities: &[ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        profile: CarrierProfile,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let prepared = PreparedInput::new(signal, identities, sample_rate_mhz, bit_depth)?;
        let blocks = match profile {
            CarrierProfile::Product => Dix2BlockCodec.encode_window(
                &prepared.stable_signal,
                &prepared.stable_identities,
                sample_rate_mhz,
                bit_depth,
            )?,
            CarrierProfile::Native => Dix2BlockCodec.encode_native_window(
                &prepared.stable_signal,
                &prepared.stable_identities,
                sample_rate_mhz,
                bit_depth,
            )?,
            CarrierProfile::Forced(mode) => Dix2BlockCodec.encode_forced(
                &prepared.stable_signal,
                &prepared.stable_identities,
                sample_rate_mhz,
                bit_depth,
                mode,
            )?,
        };
        let tile_count = u16::try_from(prepared.samples.div_ceil(128))
            .map_err(|_| input_error("DIX2 block count exceeds u16"))?;
        let directory_len = u32::try_from(blocks.directory.len())
            .map_err(|_| input_error("DIX2 directory length exceeds u32"))?;
        let payload_len = u32::try_from(blocks.payload.len())
            .map_err(|_| input_error("DIX2 payload length exceeds u32"))?;
        let decoded_crc = crc32c(&canonical_i32_bytes(&prepared.stable_signal)?);
        let mut packed = Vec::new();
        packed.extend_from_slice(b"LMO1\x03\x00\x03");
        packed.extend_from_slice(b"DIX2");
        packed.push(BODY_VERSION);
        packed.push(CONSTRUCTION_FLAGS);
        packed.push(bit_depth);
        packed.push(profile.wire());
        packed.extend_from_slice(&(prepared.channels as u16).to_le_bytes());
        packed.extend_from_slice(&tile_count.to_le_bytes());
        packed.extend_from_slice(&(prepared.samples as u32).to_le_bytes());
        packed.extend_from_slice(&sample_rate_mhz.to_le_bytes());
        packed.extend_from_slice(&0u32.to_le_bytes());
        packed.extend_from_slice(&[0u8; 32]);
        packed.extend_from_slice(&(prepared.identity_bytes.len() as u32).to_le_bytes());
        packed.extend_from_slice(&(prepared.topology_bytes.len() as u32).to_le_bytes());
        packed.extend_from_slice(&directory_len.to_le_bytes());
        packed.extend_from_slice(&payload_len.to_le_bytes());
        packed.extend_from_slice(&decoded_crc.to_le_bytes());
        packed.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(packed.len(), 7 + BODY_HEADER_LEN);
        packed.extend_from_slice(&prepared.identity_bytes);
        packed.extend_from_slice(&prepared.topology_bytes);
        packed.extend_from_slice(&blocks.directory);
        packed.extend_from_slice(&blocks.payload);
        if packed.len() > MAX_PACKET_BYTES {
            return Err(input_error("DIX2 packet exceeds its 64 MiB bound"));
        }
        let packet_crc = crc32c_zeroed_field(&packed, PACKET_CRC_OFFSET);
        packed[PACKET_CRC_OFFSET..PACKET_CRC_OFFSET + 4].copy_from_slice(&packet_crc.to_le_bytes());
        Ok(packed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CarrierProfile {
    Product,
    Native,
    Forced(Dix2CarrierMode),
}

impl CarrierProfile {
    fn forced(mode: Dix2CarrierMode) -> Self {
        Self::Forced(mode)
    }

    fn wire(self) -> u8 {
        match self {
            Self::Product => 0,
            Self::Native => 1,
            Self::Forced(Dix2CarrierMode::Raw) => 2,
            Self::Forced(Dix2CarrierMode::Delta) => 3,
            Self::Forced(Dix2CarrierMode::TemporalRans) => 4,
            Self::Forced(Dix2CarrierMode::TreeMedRans) => 5,
        }
    }

    fn from_wire(value: u8) -> Result<Self, OptimumV2Error> {
        match value {
            0 => Ok(Self::Product),
            1 => Ok(Self::Native),
            2 => Ok(Self::Forced(Dix2CarrierMode::Raw)),
            3 => Ok(Self::Forced(Dix2CarrierMode::Delta)),
            4 => Ok(Self::Forced(Dix2CarrierMode::TemporalRans)),
            5 => Ok(Self::Forced(Dix2CarrierMode::TreeMedRans)),
            _ => Err(packet_error("DIX2 carrier profile is invalid")),
        }
    }

    fn permits(self, mode: Dix2CarrierMode) -> bool {
        match self {
            Self::Product => true,
            Self::Native => matches!(mode, Dix2CarrierMode::Raw | Dix2CarrierMode::Delta),
            Self::Forced(forced) => mode == forced,
        }
    }
}

struct PreparedInput {
    stable_identities: Vec<ChannelIdentity>,
    stable_signal: Vec<Vec<i64>>,
    identity_bytes: Vec<u8>,
    topology_bytes: Vec<u8>,
    channels: usize,
    samples: usize,
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
            return Err(input_error("DIX2 signal shape does not match identities"));
        }
        validate_stable_ids(identities, InputKind::Caller)?;
        let minimum = -(1i64 << (bit_depth - 1));
        let maximum = (1i64 << (bit_depth - 1)) - 1;
        if signal
            .iter()
            .flatten()
            .any(|&sample| sample < minimum || sample > maximum)
        {
            return Err(input_error("DIX2 sample exceeds the declared bit depth"));
        }
        let mut stable_identities = vec![ChannelIdentity::new(0, "placeholder"); channels];
        let mut stable_signal = vec![Vec::new(); channels];
        for (presented, identity) in identities.iter().enumerate() {
            let stable = usize::from(identity.stable_id);
            stable_identities[stable] = identity.clone();
            stable_signal[stable] = signal[presented].clone();
        }
        let identity_bytes = encode_identities(&stable_identities)?;
        let canonical_identities = decode_identities(&identity_bytes, channels)?;
        let topology_bytes = encode_topology(&canonical_identities)?;
        Ok(Self {
            stable_identities,
            stable_signal,
            identity_bytes,
            topology_bytes,
            channels,
            samples,
        })
    }
}

fn encode_identities(identities: &[ChannelIdentity]) -> Result<Vec<u8>, OptimumV2Error> {
    let incidence = DerivationIncidence::build(identities)?;
    let mut packed = Vec::new();
    for channel in incidence.channels() {
        let identity = &identities[channel.presented_index()];
        let label = identity.label.as_bytes();
        if label.is_empty()
            || label.len() > MAX_LABEL_BYTES
            || !label.iter().all(|byte| (0x20..=0x7e).contains(byte))
        {
            return Err(input_error(
                "DIX2 construction labels must be printable ASCII",
            ));
        }
        packed.extend_from_slice(&identity.stable_id.to_le_bytes());
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
            .ok_or_else(|| packet_error("DIX2 identity offset overflows"))?;
        let label_len = usize::from(
            *packed
                .get(offset)
                .ok_or_else(|| packet_error("DIX2 identity label length is truncated"))?,
        );
        offset = offset
            .checked_add(1)
            .ok_or_else(|| packet_error("DIX2 identity label offset overflows"))?;
        let end = offset
            .checked_add(label_len)
            .ok_or_else(|| packet_error("DIX2 identity label end overflows"))?;
        let label = packed
            .get(offset..end)
            .ok_or_else(|| packet_error("DIX2 identity label is truncated"))?;
        if label.is_empty() || !label.iter().all(|byte| (0x20..=0x7e).contains(byte)) {
            return Err(packet_error("DIX2 identity label is not printable ASCII"));
        }
        let label = std::str::from_utf8(label)
            .map_err(|_| packet_error("DIX2 identity label is not UTF-8"))?;
        identities.push(ChannelIdentity::new(stable_id, label));
        offset = end;
    }
    if offset != packed.len() {
        return Err(packet_error("DIX2 identity section has trailing bytes"));
    }
    validate_stable_ids(&identities, InputKind::Packet)?;
    DerivationIncidence::build(&identities).map_err(as_packet_error)?;
    Ok(identities)
}

fn encode_topology(identities: &[ChannelIdentity]) -> Result<Vec<u8>, OptimumV2Error> {
    let forest = DerivationForest::build(identities)?;
    let mut packed = Vec::new();
    for channel in forest.channels() {
        if channel.supports().len() > MAX_SUPPORTS {
            return Err(input_error("DIX2 topology exceeds three supports"));
        }
        packed.push(channel.supports().len() as u8);
        for support in channel.supports() {
            packed.push(
                u8::try_from(support.parent_channel)
                    .map_err(|_| input_error("DIX2 topology rank exceeds u8"))?,
            );
            packed.push(support.coefficient as u8);
        }
    }
    Ok(packed)
}

fn decode_modes(
    directory: &[u8],
    profile: CarrierProfile,
) -> Result<Vec<Dix2CarrierMode>, OptimumV2Error> {
    let mut modes = Vec::with_capacity(directory.len() / DIRECTORY_ENTRY_LEN);
    for entry in directory.chunks_exact(DIRECTORY_ENTRY_LEN) {
        let mode = Dix2CarrierMode::from_wire(entry[0])?;
        if !profile.permits(mode) {
            return Err(packet_error(
                "DIX2 block mode is not permitted by its carrier profile",
            ));
        }
        modes.push(mode);
    }
    Ok(modes)
}

fn validate_stable_ids(
    identities: &[ChannelIdentity],
    kind: InputKind,
) -> Result<(), OptimumV2Error> {
    let mut stable = identities
        .iter()
        .map(|identity| usize::from(identity.stable_id))
        .collect::<Vec<_>>();
    stable.sort_unstable();
    if stable != (0..identities.len()).collect::<Vec<_>>() {
        return Err(kind.error("DIX2 stable IDs must be contiguous and unique"));
    }
    Ok(())
}

fn validate_dimensions(
    channels: usize,
    samples: usize,
    bit_depth: u8,
    sample_rate_mhz: u32,
    kind: InputKind,
) -> Result<(), OptimumV2Error> {
    let values = channels.checked_mul(samples);
    if channels == 0
        || channels > MAX_CHANNELS
        || samples == 0
        || samples > MAX_SAMPLES
        || values.map_or(true, |values| values > MAX_VALUES)
        || !(1..=32).contains(&bit_depth)
        || !(1..=MAX_SAMPLE_RATE_MHZ).contains(&sample_rate_mhz)
    {
        return Err(kind.error("DIX2 dimensions are outside construction bounds"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum InputKind {
    Caller,
    Packet,
}

impl InputKind {
    fn error(self, message: impl Into<String>) -> OptimumV2Error {
        match self {
            Self::Caller => input_error(message),
            Self::Packet => packet_error(message),
        }
    }
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
