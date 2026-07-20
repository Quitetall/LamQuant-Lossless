//! Self-decoding BGF1 learned carrier (construction flags = 1, modes 2/3).

use std::cmp::Ordering;
use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::bgf1_model_pack::Bgf1ModelPack;
use crate::model_pack::Tensor;
use crate::OptimumV2Error;

const BGF_CONSTRUCTION_FLAGS: u8 = 1;
const TILE_MODE_PRIOR_RANS_NO_FLOW: u8 = 2;
const TILE_MODE_PRIOR_RANS_FLOW: u8 = 3;
const ENTROPY_PROFILE_BINARY_RANS_V1: u8 = 1;
const MAX_LEARNED_CHANNELS: usize = 64;
const MAX_SAMPLES_PER_WINDOW: usize = 32_768;
const MAX_LEARNED_VALUES: usize = 131_072;
const MAX_BUFFERED_EVENTS: usize = 2_000_000;
const MAX_PACKET_BYTES: usize = 64 * 1024 * 1024;
const BGF_HEADER_LEN: usize = 80;
const DIRECTORY_LEN: usize = 24;
const PACKET_CRC_OFFSET: usize = 7 + 76;
const STATE_DIM: usize = 16;
const GRAPH_MAX_DEGREE: usize = 4;
const CDF_BITS: u32 = 15;
const CDF_TOTAL: u32 = 1 << CDF_BITS;
const RANS_L: u64 = 1 << 23;
const ADAPT_SHIFT: u32 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bgf1ChannelIdentity {
    pub stable_id: u16,
    pub exact_label: String,
}

impl Bgf1ChannelIdentity {
    pub fn new(stable_id: u16, exact_label: impl Into<String>) -> Self {
        Self {
            stable_id,
            exact_label: exact_label.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bgf1LearnedMode {
    NoFlow,
    Flow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBgf1Window {
    pub samples: Vec<Vec<i64>>,
    pub identities: Vec<Bgf1ChannelIdentity>,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub mode: Bgf1LearnedMode,
    pub event_count: u32,
}

impl Bgf1LearnedMode {
    pub fn wire_mode(self) -> u8 {
        match self {
            Self::NoFlow => TILE_MODE_PRIOR_RANS_NO_FLOW,
            Self::Flow => TILE_MODE_PRIOR_RANS_FLOW,
        }
    }
}

#[derive(Debug, Clone)]
struct CouplingWeights {
    predict_second: Vec<Vec<i8>>,
    update_first: Vec<Vec<i8>>,
}

#[derive(Debug, Clone)]
struct PriorWeights {
    graph: Vec<Vec<i8>>,
    scale_bias: Vec<i8>,
    temporal: Vec<Vec<i8>>,
}

#[derive(Debug, Clone)]
struct EntropyWeights {
    exponent_logits: Vec<Vec<i8>>,
    mantissa_logits: Vec<Vec<i8>>,
    sign_logits: Vec<i8>,
    token_magnitude_bias: Vec<i8>,
}

#[derive(Debug, Clone)]
struct Bgf1Model {
    model_id: i32,
    full_pack_sha256: [u8; 32],
    coupling: CouplingWeights,
    prior: PriorWeights,
    entropy: EntropyWeights,
}

#[derive(Debug, Clone)]
pub struct Bgf1LearnedCodec {
    model: Bgf1Model,
}

impl Bgf1LearnedCodec {
    pub fn from_lqw1(bytes: &[u8]) -> Result<Self, OptimumV2Error> {
        let pack = Bgf1ModelPack::decode(bytes)?;
        let tensor = |name: &str| -> Result<&Tensor, OptimumV2Error> {
            pack.tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| {
                    OptimumV2Error::InvalidPacket(format!("BGF1 LQW1 tensor {name} is missing"))
                })
        };
        let coupling = CouplingWeights {
            predict_second: i8_matrix(tensor("coupling.predict_second")?, 2, 256)?,
            update_first: i8_matrix(tensor("coupling.update_first")?, 2, 256)?,
        };
        let prior = PriorWeights {
            graph: i8_matrix(tensor("prior.graph")?, 256, 4)?,
            scale_bias: i8_vector(tensor("prior.scale_bias")?, 256)?,
            temporal: i8_matrix(tensor("prior.temporal")?, 256, 16)?,
        };
        let entropy = EntropyWeights {
            exponent_logits: i8_matrix(tensor("entropy.exponent_logits")?, 16, 16)?,
            mantissa_logits: i8_matrix(tensor("entropy.mantissa_logits")?, 16, 16)?,
            sign_logits: i8_vector(tensor("entropy.sign_logits")?, 256)?,
            token_magnitude_bias: i8_vector(tensor("entropy.token_magnitude_bias")?, 256)?,
        };
        Ok(Self {
            model: Bgf1Model {
                model_id: pack.model_id,
                full_pack_sha256: Sha256::digest(bytes).into(),
                coupling,
                prior,
                entropy,
            },
        })
    }

    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        identities: &[Bgf1ChannelIdentity],
        sample_rate_mhz: u32,
        bit_depth: u8,
        mode: Bgf1LearnedMode,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let n_samples = validate_input(signal, identities, sample_rate_mhz, bit_depth)?;
        let graph = DerivationGraph::build(identities)?;
        let identity_bytes = graph.encode_identities();
        let graph_bytes = graph.encode_graph();
        let mut prior = GraphSsmPrior::new(&graph, &self.model.prior, bit_depth);
        let mut entropy = BinaryRansEncoder::new(&self.model.entropy);

        for sample_index in 0..n_samples {
            let canonical_raw = graph.canonicalize_row(signal, sample_index);
            let context = prior.begin_row()?;
            let innovations = prior.innovations(&canonical_raw, &context)?;
            let coded = match mode {
                Bgf1LearnedMode::NoFlow => innovations.clone(),
                Bgf1LearnedMode::Flow => {
                    coupling_forward(&innovations, &graph, &self.model.coupling)?
                }
            };
            for ((&value, &scale_bucket), &token_bucket) in coded
                .iter()
                .zip(&context.scale_buckets)
                .zip(&context.token_buckets)
            {
                entropy.push_value(value, scale_bucket, token_bucket)?;
                if entropy.event_count() > MAX_BUFFERED_EVENTS {
                    return Err(OptimumV2Error::InvalidInput(
                        "BGF1 entropy event buffer exceeds its resource bound".into(),
                    ));
                }
            }
            prior.observe_row(&canonical_raw, &innovations, &context)?;
        }
        let payload = entropy.finish()?;
        let event_count = u32::try_from(entropy.event_count())
            .map_err(|_| OptimumV2Error::InvalidInput("BGF1 event count exceeds u32".into()))?;
        if event_count == 0 {
            return Err(OptimumV2Error::InvalidInput(
                "BGF1 learned stream has no entropy events".into(),
            ));
        }

        let mut directory = Vec::with_capacity(DIRECTORY_LEN);
        directory.push(mode.wire_mode());
        directory.push(ENTROPY_PROFILE_BINARY_RANS_V1);
        directory.extend_from_slice(&0u16.to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(&(n_samples as u32).to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        directory.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| OptimumV2Error::InvalidInput("BGF1 payload exceeds u32".into()))?
                .to_le_bytes(),
        );
        directory.extend_from_slice(&event_count.to_le_bytes());

        let decoded_crc = crc32c(&stable_raw_bytes(signal, identities)?);
        let mut out = Vec::new();
        out.extend_from_slice(b"LMO1");
        out.extend_from_slice(&[3, 0, 3]);
        out.extend_from_slice(b"BGF1");
        out.push(1);
        out.push(BGF_CONSTRUCTION_FLAGS);
        out.push(bit_depth);
        out.push(0);
        out.extend_from_slice(&(signal.len() as u16).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&(n_samples as u32).to_le_bytes());
        out.extend_from_slice(&sample_rate_mhz.to_le_bytes());
        out.extend_from_slice(&(self.model.model_id as u32).to_le_bytes());
        out.extend_from_slice(&self.model.full_pack_sha256);
        out.extend_from_slice(
            &u32::try_from(identity_bytes.len())
                .map_err(|_| OptimumV2Error::InvalidInput("BGF1 identities exceed u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&(graph_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(DIRECTORY_LEN as u32).to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(payload.len())
                .map_err(|_| OptimumV2Error::InvalidInput("BGF1 payload exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(&decoded_crc.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(out.len(), 7 + BGF_HEADER_LEN);
        out.extend_from_slice(&identity_bytes);
        out.extend_from_slice(&graph_bytes);
        out.extend_from_slice(&directory);
        out.extend_from_slice(&payload);
        if out.len() > MAX_PACKET_BYTES {
            return Err(OptimumV2Error::InvalidInput(
                "BGF1 packet exceeds 64 MiB".into(),
            ));
        }
        let packet_crc = crc32c_zeroed_field(&out, PACKET_CRC_OFFSET);
        out[PACKET_CRC_OFFSET..PACKET_CRC_OFFSET + 4].copy_from_slice(&packet_crc.to_le_bytes());
        Ok(out)
    }

    pub fn decode_window(&self, packed: &[u8]) -> Result<DecodedBgf1Window, OptimumV2Error> {
        if packed.len() < 7 + BGF_HEADER_LEN || packed.len() > MAX_PACKET_BYTES {
            return Err(packet_error("BGF1 packet length is outside bounds"));
        }
        if &packed[..7] != b"LMO1\x03\x00\x03" {
            return Err(packet_error("BGF1 LMO1 envelope is invalid"));
        }
        if &packed[7..11] != b"BGF1" || packed[11] != 1 {
            return Err(packet_error("BGF1 magic/version is invalid"));
        }
        let flags = packed[12];
        let bit_depth = packed[13];
        let reserved = packed[14];
        if flags != BGF_CONSTRUCTION_FLAGS || reserved != 0 {
            return Err(packet_error("BGF1 construction flags/reserved are invalid"));
        }
        let n_channels = read_u16(packed, 15)? as usize;
        let tile_count = read_u16(packed, 17)?;
        let n_samples = read_u32(packed, 19)? as usize;
        let sample_rate_mhz = read_u32(packed, 23)?;
        let model_id = read_u32(packed, 27)?;
        let model_sha256: [u8; 32] = packed[31..63].try_into().unwrap();
        let identity_length = read_u32(packed, 63)? as usize;
        let graph_length = read_u32(packed, 67)? as usize;
        let directory_length = read_u32(packed, 71)? as usize;
        let payload_length = read_u32(packed, 75)? as usize;
        let decoded_crc = read_u32(packed, 79)?;
        let packet_crc = read_u32(packed, 83)?;
        let symbol_count = n_channels
            .checked_mul(n_samples)
            .ok_or_else(|| packet_error("BGF1 symbol count overflows"))?;
        if !(1..=32).contains(&bit_depth)
            || !(1..=MAX_LEARNED_CHANNELS).contains(&n_channels)
            || tile_count != 1
            || !(1..=MAX_SAMPLES_PER_WINDOW).contains(&n_samples)
            || symbol_count > MAX_LEARNED_VALUES
            || sample_rate_mhz == 0
        {
            return Err(packet_error("BGF1 dimensions/rate are outside bounds"));
        }
        if model_id != self.model.model_id as u32 || model_sha256 != self.model.full_pack_sha256 {
            return Err(packet_error("BGF1 model ID/SHA-256 binding is invalid"));
        }
        if graph_length != GRAPH_MAX_DEGREE * n_channels
            || directory_length != DIRECTORY_LEN
            || payload_length < 4
        {
            return Err(packet_error("BGF1 learned section lengths are invalid"));
        }
        let total = (7 + BGF_HEADER_LEN)
            .checked_add(identity_length)
            .and_then(|value| value.checked_add(graph_length))
            .and_then(|value| value.checked_add(directory_length))
            .and_then(|value| value.checked_add(payload_length))
            .ok_or_else(|| packet_error("BGF1 packet section length overflows"))?;
        if total != packed.len() {
            return Err(packet_error("BGF1 packet section lengths do not match"));
        }
        if packet_crc != crc32c_zeroed_field(packed, PACKET_CRC_OFFSET) {
            return Err(OptimumV2Error::Integrity(
                "BGF1 packet CRC32C mismatch".into(),
            ));
        }

        let identity_start = 7 + BGF_HEADER_LEN;
        let identity_end = identity_start + identity_length;
        let graph_end = identity_end + graph_length;
        let directory_end = graph_end + directory_length;
        let canonical_identities =
            decode_identities(&packed[identity_start..identity_end], n_channels)?;
        let graph = DerivationGraph::build(&canonical_identities)
            .map_err(|error| packet_error(&error.to_string()))?;
        if graph
            .channels
            .iter()
            .zip(&canonical_identities)
            .any(|(channel, identity)| {
                channel.stable_id != identity.stable_id
                    || channel.exact_label != identity.exact_label
            })
        {
            return Err(packet_error(
                "BGF1 identity section is not in canonical derivation order",
            ));
        }
        if packed[identity_end..graph_end] != graph.encode_graph() {
            return Err(packet_error(
                "BGF1 graph does not match deterministic reconstruction",
            ));
        }

        let directory = &packed[graph_end..directory_end];
        let mode = match directory[0] {
            TILE_MODE_PRIOR_RANS_NO_FLOW => Bgf1LearnedMode::NoFlow,
            TILE_MODE_PRIOR_RANS_FLOW => Bgf1LearnedMode::Flow,
            _ => return Err(packet_error("BGF1 learned tile mode is invalid")),
        };
        let entropy_profile = directory[1];
        let directory_flags = read_u16(directory, 2)?;
        let first_sample = read_u32(directory, 4)?;
        let tile_samples = read_u32(directory, 8)? as usize;
        let payload_offset = read_u32(directory, 12)?;
        let tile_payload_length = read_u32(directory, 16)? as usize;
        let event_count = read_u32(directory, 20)?;
        if entropy_profile != ENTROPY_PROFILE_BINARY_RANS_V1
            || directory_flags != 0
            || first_sample != 0
            || tile_samples != n_samples
            || payload_offset != 0
            || tile_payload_length != payload_length
        {
            return Err(packet_error("BGF1 learned tile directory is invalid"));
        }
        let maximum_events = symbol_count.saturating_mul(129).min(MAX_BUFFERED_EVENTS);
        if (event_count as usize) < symbol_count || event_count as usize > maximum_events {
            return Err(packet_error("BGF1 learned event count is outside bounds"));
        }

        let mut decoder = BinaryRansDecoder::new(&packed[directory_end..], &self.model.entropy)?;
        let mut prior = GraphSsmPrior::new(&graph, &self.model.prior, bit_depth);
        let mut canonical_rows = Vec::with_capacity(n_samples);
        let minimum = -(1_i64 << (bit_depth - 1));
        let maximum = (1_i64 << (bit_depth - 1)) - 1;
        for _ in 0..n_samples {
            let context = prior.begin_row().map_err(decode_arithmetic)?;
            let mut coded = Vec::with_capacity(n_channels);
            for (&scale_bucket, &token_bucket) in
                context.scale_buckets.iter().zip(&context.token_buckets)
            {
                coded.push(decoder.read_value(scale_bucket, token_bucket)?);
                if decoder.event_count() > MAX_BUFFERED_EVENTS {
                    return Err(packet_error("BGF1 decoded event count exceeds its bound"));
                }
            }
            let innovations = match mode {
                Bgf1LearnedMode::NoFlow => coded,
                Bgf1LearnedMode::Flow => coupling_inverse(&coded, &graph, &self.model.coupling)
                    .map_err(decode_arithmetic)?,
            };
            let canonical_raw = context
                .predictions
                .iter()
                .zip(&innovations)
                .map(|(&prediction, &innovation)| {
                    prediction
                        .checked_add(innovation)
                        .ok_or_else(|| packet_error("BGF1 reconstructed sample exceeds i64"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if canonical_raw
                .iter()
                .any(|&value| value < minimum || value > maximum)
            {
                return Err(packet_error(
                    "BGF1 reconstructed sample exceeds declared bit depth",
                ));
            }
            prior
                .observe_row(&canonical_raw, &innovations, &context)
                .map_err(decode_arithmetic)?;
            canonical_rows.push(canonical_raw);
        }
        decoder.finish()?;
        if decoder.event_count() != event_count as usize {
            return Err(packet_error(
                "BGF1 learned event count does not match decoded grammar",
            ));
        }

        let mut identities = graph
            .channels
            .iter()
            .map(|channel| Bgf1ChannelIdentity::new(channel.stable_id, &channel.exact_label))
            .collect::<Vec<_>>();
        identities.sort_by_key(|identity| identity.stable_id);
        let mut samples = vec![Vec::with_capacity(n_samples); n_channels];
        for row in &canonical_rows {
            for (rank, &value) in row.iter().enumerate() {
                samples[graph.channels[rank].stable_id as usize].push(value);
            }
        }
        if crc32c(&stable_raw_bytes(&samples, &identities).map_err(decode_arithmetic)?)
            != decoded_crc
        {
            return Err(OptimumV2Error::Integrity(
                "BGF1 decoded-sample CRC32C mismatch".into(),
            ));
        }
        let canonical = self
            .encode_window(&samples, &identities, sample_rate_mhz, bit_depth, mode)
            .map_err(decode_arithmetic)?;
        if canonical != packed {
            return Err(packet_error("BGF1 learned carrier is not byte-canonical"));
        }
        Ok(DecodedBgf1Window {
            samples,
            identities,
            sample_rate_mhz,
            bit_depth,
            mode,
            event_count,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Partition {
    Eeg,
    Aux,
}

impl Partition {
    fn order(self) -> u8 {
        match self {
            Self::Eeg => 0,
            Self::Aux => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Eeg => "EEG",
            Self::Aux => "AUX",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DerivationKind {
    Monopolar,
    Referential,
    Bipolar,
    Aux,
}

impl DerivationKind {
    fn order(self) -> u8 {
        match self {
            Self::Monopolar => 0,
            Self::Referential => 1,
            Self::Bipolar => 2,
            Self::Aux => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Monopolar => "monopolar",
            Self::Referential => "referential",
            Self::Bipolar => "bipolar",
            Self::Aux => "aux",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DerivationToken {
    partition: Partition,
    kind: DerivationKind,
    positive: String,
    negative: Option<String>,
    normalized_label: String,
}

#[derive(Debug, Clone)]
struct GraphChannel {
    stable_id: u16,
    exact_label: String,
    partition: Partition,
    token_bucket: u8,
    neighbors: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DerivationGraph {
    channels: Vec<GraphChannel>,
    canonical_to_presented: Vec<usize>,
    eeg_ranks: Vec<usize>,
    aux_ranks: Vec<usize>,
}

impl DerivationGraph {
    fn build(identities: &[Bgf1ChannelIdentity]) -> Result<Self, OptimumV2Error> {
        let mut presented = identities
            .iter()
            .enumerate()
            .map(|(index, identity)| {
                Ok((
                    index,
                    identity.stable_id,
                    identity.exact_label.clone(),
                    parse_derivation(&identity.exact_label)?,
                ))
            })
            .collect::<Result<Vec<_>, OptimumV2Error>>()?;
        presented.sort_by(|left, right| {
            compare_tokens(&left.3, &right.3).then_with(|| left.1.cmp(&right.1))
        });
        let canonical_to_presented: Vec<usize> =
            presented.iter().map(|channel| channel.0).collect();
        let mut channels = presented
            .into_iter()
            .map(|(_, stable_id, exact_label, token)| GraphChannel {
                stable_id,
                exact_label,
                partition: token.partition,
                token_bucket: token_bucket(&token),
                neighbors: Vec::new(),
            })
            .collect::<Vec<_>>();
        let tokens = canonical_to_presented
            .iter()
            .map(|&index| parse_derivation(&identities[index].exact_label))
            .collect::<Result<Vec<_>, _>>()?;
        let eeg_ranks = channels
            .iter()
            .enumerate()
            .filter_map(|(rank, channel)| (channel.partition == Partition::Eeg).then_some(rank))
            .collect::<Vec<_>>();
        let aux_ranks = channels
            .iter()
            .enumerate()
            .filter_map(|(rank, channel)| (channel.partition == Partition::Aux).then_some(rank))
            .collect::<Vec<_>>();
        for &rank in &eeg_ranks {
            let mut candidates = eeg_ranks
                .iter()
                .copied()
                .filter(|&candidate| candidate != rank)
                .collect::<Vec<_>>();
            candidates.sort_by_key(|&candidate| {
                neighbor_key(&tokens[rank], &tokens[candidate], candidate)
            });
            channels[rank].neighbors = candidates.into_iter().take(GRAPH_MAX_DEGREE).collect();
        }
        Ok(Self {
            channels,
            canonical_to_presented,
            eeg_ranks,
            aux_ranks,
        })
    }

    fn encode_identities(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for channel in &self.channels {
            out.extend_from_slice(&channel.stable_id.to_le_bytes());
            out.push(channel.exact_label.len() as u8);
            out.extend_from_slice(channel.exact_label.as_bytes());
        }
        out
    }

    fn encode_graph(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.channels.len() * GRAPH_MAX_DEGREE);
        for channel in &self.channels {
            out.extend(channel.neighbors.iter().map(|&rank| rank as u8));
            out.resize(out.len() + GRAPH_MAX_DEGREE - channel.neighbors.len(), 0xff);
        }
        out
    }

    fn canonicalize_row(&self, signal: &[Vec<i64>], sample_index: usize) -> Vec<i64> {
        self.canonical_to_presented
            .iter()
            .map(|&presented| signal[presented][sample_index])
            .collect()
    }
}

fn validate_input(
    signal: &[Vec<i64>],
    identities: &[Bgf1ChannelIdentity],
    sample_rate_mhz: u32,
    bit_depth: u8,
) -> Result<usize, OptimumV2Error> {
    if signal.is_empty() || signal.len() > MAX_LEARNED_CHANNELS {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 learned channel count is outside 1..64".into(),
        ));
    }
    if identities.len() != signal.len() || sample_rate_mhz == 0 || !(1..=32).contains(&bit_depth) {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 identities, rate, or bit depth are invalid".into(),
        ));
    }
    let mut stable_ids = identities
        .iter()
        .map(|identity| identity.stable_id as usize)
        .collect::<Vec<_>>();
    stable_ids.sort_unstable();
    if stable_ids != (0..identities.len()).collect::<Vec<_>>() {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 stable IDs must be exactly contiguous manifest ordinals".into(),
        ));
    }
    for identity in identities {
        let bytes = identity.exact_label.as_bytes();
        if bytes.is_empty()
            || bytes.len() > u8::MAX as usize
            || bytes.iter().any(|&byte| !(0x20..=0x7e).contains(&byte))
        {
            return Err(OptimumV2Error::InvalidInput(
                "BGF1 construction label is outside printable ASCII".into(),
            ));
        }
    }
    let n_samples = signal[0].len();
    if n_samples == 0
        || n_samples > MAX_SAMPLES_PER_WINDOW
        || signal.iter().any(|channel| channel.len() != n_samples)
        || !matches!(
            signal.len().checked_mul(n_samples),
            Some(count) if count <= MAX_LEARNED_VALUES
        )
    {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 signal dimensions exceed the construction profile".into(),
        ));
    }
    let minimum = -(1_i64 << (bit_depth - 1));
    let maximum = (1_i64 << (bit_depth - 1)) - 1;
    if signal
        .iter()
        .flatten()
        .any(|&sample| sample < minimum || sample > maximum)
    {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 sample exceeds declared bit depth".into(),
        ));
    }
    Ok(n_samples)
}

fn normalized_label(label: &str) -> String {
    let mut normalized = label
        .split_ascii_whitespace()
        .map(str::to_ascii_uppercase)
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(without_prefix) = normalized.strip_prefix("EEG ") {
        normalized = without_prefix.trim().to_owned();
    }
    while normalized.ends_with('.') {
        normalized.pop();
    }
    normalized
}

const AUX_PREFIXES: [&str; 17] = [
    "ECG", "EKG", "RESP", "SPO2", "SP02", "PULSE", "HEART", "HR", "EMG", "EOG", "MARK", "MK",
    "TRIGGER", "EVENT", "TEMP", "CO2", "AIRFLOW",
];

fn is_known_auxiliary(normalized: &str) -> bool {
    AUX_PREFIXES.iter().any(|prefix| {
        normalized == *prefix
            || normalized.starts_with(&format!("{prefix} "))
            || normalized.starts_with(&format!("{prefix}-"))
    })
}

fn parse_derivation(label: &str) -> Result<DerivationToken, OptimumV2Error> {
    let normalized = normalized_label(label);
    if normalized.is_empty() {
        return Err(OptimumV2Error::InvalidInput(
            "BGF1 label becomes empty after normalization".into(),
        ));
    }
    if is_known_auxiliary(&normalized) {
        return Ok(DerivationToken {
            partition: Partition::Aux,
            kind: DerivationKind::Aux,
            positive: normalized.clone(),
            negative: None,
            normalized_label: normalized,
        });
    }
    let pieces = normalized.split('-').collect::<Vec<_>>();
    if pieces.len() == 1 {
        if let Some(positive) = electrode(pieces[0]) {
            return Ok(DerivationToken {
                partition: Partition::Eeg,
                kind: DerivationKind::Monopolar,
                positive: positive.token,
                negative: None,
                normalized_label: normalized,
            });
        }
    }
    if pieces.len() == 2 {
        if let Some(positive) = electrode(pieces[0]) {
            let negative = pieces[1].trim().trim_end_matches('.').to_ascii_uppercase();
            if matches!(negative.as_str(), "REF" | "LE" | "AR" | "AVG" | "CZREF") {
                return Ok(DerivationToken {
                    partition: Partition::Eeg,
                    kind: DerivationKind::Referential,
                    positive: positive.token,
                    negative: Some(negative),
                    normalized_label: normalized,
                });
            }
            if electrode(&negative).is_some() {
                return Ok(DerivationToken {
                    partition: Partition::Eeg,
                    kind: DerivationKind::Bipolar,
                    positive: positive.token,
                    negative: Some(negative),
                    normalized_label: normalized,
                });
            }
        }
    }
    Ok(DerivationToken {
        partition: Partition::Aux,
        kind: DerivationKind::Aux,
        positive: normalized.clone(),
        negative: None,
        normalized_label: normalized,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ElectrodeKey {
    anterior_posterior: u8,
    family: u8,
    pair: u8,
    side: u8,
    token: String,
}

fn electrode(value: &str) -> Option<ElectrodeKey> {
    const FAMILIES: [(&str, u8, u8); 15] = [
        ("FP", 0, 0),
        ("AF", 1, 1),
        ("FC", 3, 3),
        ("FT", 3, 4),
        ("TP", 5, 8),
        ("CP", 5, 7),
        ("PO", 7, 10),
        ("F", 2, 2),
        ("T", 4, 6),
        ("C", 4, 5),
        ("P", 6, 9),
        ("O", 8, 11),
        ("I", 9, 12),
        ("A", 10, 13),
        ("M", 10, 14),
    ];
    let normalized = value
        .trim()
        .to_ascii_uppercase()
        .trim_end_matches('.')
        .to_owned();
    for (family, ap, family_order) in FAMILIES {
        let Some(suffix) = normalized.strip_prefix(family) else {
            continue;
        };
        let (pair, side) = if suffix == "Z" {
            (0, 0)
        } else if (suffix.len() == 1 || suffix.len() == 2)
            && suffix.as_bytes()[0].is_ascii_digit()
            && suffix.as_bytes()[0] != b'0'
            && suffix.bytes().all(|byte| byte.is_ascii_digit())
        {
            let number: u8 = suffix.parse().ok()?;
            (number.div_ceil(2), if number % 2 == 1 { 0 } else { 1 })
        } else {
            continue;
        };
        return Some(ElectrodeKey {
            anterior_posterior: ap,
            family: family_order,
            pair,
            side,
            token: normalized,
        });
    }
    None
}

fn electrode_sort_key(value: Option<&str>) -> ElectrodeKey {
    match value {
        None => ElectrodeKey {
            anterior_posterior: 0,
            family: 0,
            pair: 0,
            side: 0,
            token: String::new(),
        },
        Some(value) => electrode(value).unwrap_or_else(|| ElectrodeKey {
            anterior_posterior: 99,
            family: 99,
            pair: 99,
            side: 99,
            token: value.to_owned(),
        }),
    }
}

fn compare_tokens(left: &DerivationToken, right: &DerivationToken) -> Ordering {
    left.partition
        .order()
        .cmp(&right.partition.order())
        .then_with(|| left.kind.order().cmp(&right.kind.order()))
        .then_with(|| {
            electrode_sort_key(Some(&left.positive)).cmp(&electrode_sort_key(Some(&right.positive)))
        })
        .then_with(|| {
            electrode_sort_key(left.negative.as_deref())
                .cmp(&electrode_sort_key(right.negative.as_deref()))
        })
        .then_with(|| left.normalized_label.cmp(&right.normalized_label))
}

fn token_bucket(token: &DerivationToken) -> u8 {
    let mut packed = Vec::from(&b"BGF1-TOKEN-BUCKET-V1"[..]);
    for value in [
        token.partition.name(),
        token.kind.name(),
        &token.positive,
        token.negative.as_deref().unwrap_or(""),
        &token.normalized_label,
    ] {
        packed.extend_from_slice(&(value.len() as u16).to_le_bytes());
        packed.extend_from_slice(value.as_bytes());
    }
    let semantic = Sha256::digest(&packed)[0] & 0x7f;
    match token.partition {
        Partition::Eeg => semantic,
        Partition::Aux => semantic | 0x80,
    }
}

fn token_endpoints(token: &DerivationToken) -> Vec<&str> {
    let mut endpoints = vec![token.positive.as_str()];
    if token.negative.as_deref().and_then(electrode).is_some() {
        endpoints.push(token.negative.as_deref().unwrap());
    }
    endpoints
}

fn neighbor_key(
    source: &DerivationToken,
    candidate: &DerivationToken,
    candidate_rank: usize,
) -> (u8, u8, u8, u8, u8, usize) {
    let source_endpoints = token_endpoints(source);
    let candidate_endpoints = token_endpoints(candidate);
    let shared = source_endpoints
        .iter()
        .any(|endpoint| candidate_endpoints.contains(endpoint));
    let left = electrode(&source.positive).unwrap();
    let right = electrode(&candidate.positive).unwrap();
    (
        if shared { 0 } else { 1 },
        left.anterior_posterior.abs_diff(right.anterior_posterior),
        left.pair.abs_diff(right.pair),
        left.side.abs_diff(right.side),
        left.family.abs_diff(right.family),
        candidate_rank,
    )
}

#[derive(Debug, Clone)]
struct PriorRowContext {
    predictions: Vec<i64>,
    scale_buckets: Vec<u8>,
    token_buckets: Vec<u8>,
}

struct GraphSsmPrior<'a> {
    graph: &'a DerivationGraph,
    weights: &'a PriorWeights,
    bit_depth: u8,
    history: Vec<[i64; STATE_DIM]>,
    scale_q4: Vec<i64>,
}

impl<'a> GraphSsmPrior<'a> {
    fn new(graph: &'a DerivationGraph, weights: &'a PriorWeights, bit_depth: u8) -> Self {
        Self {
            graph,
            weights,
            bit_depth,
            history: vec![[0; STATE_DIM]; graph.channels.len()],
            scale_q4: vec![0; graph.channels.len()],
        }
    }

    fn begin_row(&self) -> Result<PriorRowContext, OptimumV2Error> {
        Ok(PriorRowContext {
            predictions: self.predict_row()?,
            scale_buckets: (0..self.graph.channels.len())
                .map(|rank| self.scale_bucket(rank))
                .collect(),
            token_buckets: self
                .graph
                .channels
                .iter()
                .map(|channel| channel.token_bucket)
                .collect(),
        })
    }

    fn predict_row(&self) -> Result<Vec<i64>, OptimumV2Error> {
        let minimum = -(1_i64 << (self.bit_depth - 1));
        let maximum = (1_i64 << (self.bit_depth - 1)) - 1;
        self.graph
            .channels
            .iter()
            .enumerate()
            .map(|(rank, channel)| {
                let bucket = channel.token_bucket as usize;
                let mut accumulator = 0_i64;
                for (&weight, &value) in self.weights.temporal[bucket]
                    .iter()
                    .zip(&self.history[rank])
                {
                    accumulator = accumulator
                        .checked_add(
                            i64::from(weight)
                                .checked_mul(value)
                                .ok_or_else(prior_overflow)?,
                        )
                        .ok_or_else(prior_overflow)?;
                }
                let source_previous = self.history[rank][0];
                for (slot, &neighbor) in channel.neighbors.iter().enumerate() {
                    let difference = self.history[neighbor][0]
                        .checked_sub(source_previous)
                        .ok_or_else(prior_overflow)?;
                    accumulator = accumulator
                        .checked_add(
                            i64::from(self.weights.graph[bucket][slot])
                                .checked_mul(difference)
                                .ok_or_else(prior_overflow)?,
                        )
                        .ok_or_else(prior_overflow)?;
                }
                Ok(round_q6(accumulator).clamp(minimum, maximum))
            })
            .collect()
    }

    fn scale_bucket(&self, rank: usize) -> u8 {
        let rounded = ((self.scale_q4[rank] + 8) >> 4) as u64;
        let magnitude_bucket = if rounded == 0 {
            0
        } else {
            u64::BITS - rounded.leading_zeros()
        } as i32;
        let token = self.graph.channels[rank].token_bucket as usize;
        (magnitude_bucket + i32::from(self.weights.scale_bias[token] >> 4)).clamp(0, 15) as u8
    }

    fn innovations(
        &self,
        raw: &[i64],
        context: &PriorRowContext,
    ) -> Result<Vec<i64>, OptimumV2Error> {
        raw.iter()
            .zip(&context.predictions)
            .map(|(&value, &prediction)| value.checked_sub(prediction).ok_or_else(prior_overflow))
            .collect()
    }

    fn observe_row(
        &mut self,
        raw: &[i64],
        innovations: &[i64],
        _context: &PriorRowContext,
    ) -> Result<(), OptimumV2Error> {
        for (rank, &value) in raw.iter().enumerate() {
            self.history[rank].rotate_right(1);
            self.history[rank][0] = value;
        }
        for (rank, &innovation) in innovations.iter().enumerate() {
            let target = i64::try_from(innovation.unsigned_abs())
                .ok()
                .and_then(|value| value.checked_shl(4))
                .ok_or_else(prior_overflow)?;
            let difference = target
                .checked_sub(self.scale_q4[rank])
                .ok_or_else(prior_overflow)?;
            let step = if difference == 0 {
                0
            } else {
                let magnitude = i64::try_from((difference.unsigned_abs() + 8) >> 4)
                    .map_err(|_| prior_overflow())?
                    .max(1);
                if difference > 0 {
                    magnitude
                } else {
                    -magnitude
                }
            };
            self.scale_q4[rank] = self.scale_q4[rank]
                .checked_add(step)
                .ok_or_else(prior_overflow)?;
        }
        Ok(())
    }
}

fn prior_overflow() -> OptimumV2Error {
    OptimumV2Error::InvalidInput("BGF1 prior arithmetic exceeds signed i64".into())
}

fn round_q6(value: i64) -> i64 {
    let rounded = (value.unsigned_abs() + 32) >> 6;
    if value >= 0 {
        rounded as i64
    } else {
        -(rounded as i64)
    }
}

fn partition_pairs(partition: &[usize], layer: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut index = layer & 1;
    while index + 1 < partition.len() {
        pairs.push((partition[index], partition[index + 1]));
        index += 2;
    }
    pairs
}

fn coupling_forward(
    innovations: &[i64],
    graph: &DerivationGraph,
    weights: &CouplingWeights,
) -> Result<Vec<i64>, OptimumV2Error> {
    let mut work = innovations.to_vec();
    for layer in 0..2 {
        for partition in [&graph.eeg_ranks, &graph.aux_ranks] {
            for (first, second) in partition_pairs(partition, layer) {
                let first_bucket = graph.channels[first].token_bucket as usize;
                let second_bucket = graph.channels[second].token_bucket as usize;
                let predicted_second = round_q6(
                    i64::from(weights.predict_second[layer][second_bucket])
                        .checked_mul(work[first])
                        .ok_or_else(coupling_overflow)?,
                );
                let transformed_second = work[second]
                    .checked_sub(predicted_second)
                    .ok_or_else(coupling_overflow)?;
                let updated_first = round_q6(
                    i64::from(weights.update_first[layer][first_bucket])
                        .checked_mul(transformed_second)
                        .ok_or_else(coupling_overflow)?,
                );
                work[first] = work[first]
                    .checked_sub(updated_first)
                    .ok_or_else(coupling_overflow)?;
                work[second] = transformed_second;
            }
        }
    }
    Ok(work)
}

fn coupling_inverse(
    transformed: &[i64],
    graph: &DerivationGraph,
    weights: &CouplingWeights,
) -> Result<Vec<i64>, OptimumV2Error> {
    let mut work = transformed.to_vec();
    for layer in (0..2).rev() {
        for partition in [&graph.eeg_ranks, &graph.aux_ranks] {
            for (first, second) in partition_pairs(partition, layer).into_iter().rev() {
                let first_bucket = graph.channels[first].token_bucket as usize;
                let second_bucket = graph.channels[second].token_bucket as usize;
                let updated_first = round_q6(
                    i64::from(weights.update_first[layer][first_bucket])
                        .checked_mul(work[second])
                        .ok_or_else(coupling_overflow)?,
                );
                let original_first = work[first]
                    .checked_add(updated_first)
                    .ok_or_else(coupling_overflow)?;
                let predicted_second = round_q6(
                    i64::from(weights.predict_second[layer][second_bucket])
                        .checked_mul(original_first)
                        .ok_or_else(coupling_overflow)?,
                );
                work[first] = original_first;
                work[second] = work[second]
                    .checked_add(predicted_second)
                    .ok_or_else(coupling_overflow)?;
            }
        }
    }
    Ok(work)
}

fn coupling_overflow() -> OptimumV2Error {
    OptimumV2Error::InvalidInput("BGF1 coupling arithmetic exceeds signed i64".into())
}

#[derive(Debug, Clone, Copy)]
struct BinaryEvent {
    bit: u8,
    probability_one: u16,
}

struct AdaptiveModel<'a> {
    weights: &'a EntropyWeights,
    nonzero: HashMap<(u8, u8), u16>,
    exponent: HashMap<(u8, u8, u8), u16>,
    mantissa: HashMap<(u8, u8, u8), u16>,
    sign: HashMap<u8, u16>,
}

#[derive(Debug, Clone, Copy)]
enum ContextFamily {
    Nonzero,
    Exponent,
    Mantissa,
    Sign,
}

impl<'a> AdaptiveModel<'a> {
    fn new(weights: &'a EntropyWeights) -> Self {
        Self {
            weights,
            nonzero: HashMap::new(),
            exponent: HashMap::new(),
            mantissa: HashMap::new(),
            sign: HashMap::new(),
        }
    }

    fn probability(&mut self, family: ContextFamily, scale: u8, token: u8, position: u8) -> u16 {
        let index = position.min(15);
        match family {
            ContextFamily::Nonzero => *self.nonzero.entry((token, scale)).or_insert_with(|| {
                logit_probability(
                    i32::from(self.weights.exponent_logits[scale as usize][0])
                        + i32::from(self.weights.token_magnitude_bias[token as usize]),
                )
            }),
            ContextFamily::Exponent => {
                *self
                    .exponent
                    .entry((token, scale, index))
                    .or_insert_with(|| {
                        logit_probability(
                            i32::from(self.weights.exponent_logits[scale as usize][index as usize])
                                + i32::from(self.weights.token_magnitude_bias[token as usize]),
                        )
                    })
            }
            ContextFamily::Mantissa => {
                *self
                    .mantissa
                    .entry((token, scale, index))
                    .or_insert_with(|| {
                        logit_probability(
                            i32::from(self.weights.mantissa_logits[scale as usize][index as usize])
                                + i32::from(self.weights.token_magnitude_bias[token as usize]),
                        )
                    })
            }
            ContextFamily::Sign => *self.sign.entry(token).or_insert_with(|| {
                logit_probability(i32::from(self.weights.sign_logits[token as usize]))
            }),
        }
    }

    fn observe(&mut self, family: ContextFamily, scale: u8, token: u8, position: u8, bit: u8) {
        let probability = self.probability(family, scale, token, position);
        let updated = adapt_probability(probability, bit);
        let index = position.min(15);
        match family {
            ContextFamily::Nonzero => {
                self.nonzero.insert((token, scale), updated);
            }
            ContextFamily::Exponent => {
                self.exponent.insert((token, scale, index), updated);
            }
            ContextFamily::Mantissa => {
                self.mantissa.insert((token, scale, index), updated);
            }
            ContextFamily::Sign => {
                self.sign.insert(token, updated);
            }
        }
    }
}

struct BinaryRansEncoder<'a> {
    model: AdaptiveModel<'a>,
    events: Vec<BinaryEvent>,
}

impl<'a> BinaryRansEncoder<'a> {
    fn new(weights: &'a EntropyWeights) -> Self {
        Self {
            model: AdaptiveModel::new(weights),
            events: Vec::new(),
        }
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }

    fn push(&mut self, family: ContextFamily, scale: u8, token: u8, position: u8, bit: u8) {
        let probability_one = self.model.probability(family, scale, token, position);
        self.events.push(BinaryEvent {
            bit,
            probability_one,
        });
        self.model.observe(family, scale, token, position, bit);
    }

    fn push_value(&mut self, value: i64, scale: u8, token: u8) -> Result<(), OptimumV2Error> {
        let magnitude = value.unsigned_abs();
        self.push(
            ContextFamily::Nonzero,
            scale,
            token,
            0,
            u8::from(magnitude != 0),
        );
        if magnitude == 0 {
            return Ok(());
        }
        let exponent = (u64::BITS - 1 - magnitude.leading_zeros()) as u8;
        for level in 0..=exponent {
            self.push(
                ContextFamily::Exponent,
                scale,
                token,
                level,
                u8::from(level < exponent),
            );
        }
        self.push(ContextFamily::Sign, scale, token, 0, u8::from(value < 0));
        for position in (0..exponent).rev() {
            self.push(
                ContextFamily::Mantissa,
                scale,
                token,
                position,
                ((magnitude >> position) & 1) as u8,
            );
        }
        Ok(())
    }

    fn finish(&self) -> Result<Vec<u8>, OptimumV2Error> {
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
            return Err(OptimumV2Error::InvalidInput(
                "BGF1 rANS final state is outside u32".into(),
            ));
        }
        let mut out = Vec::with_capacity(4 + renormalized.len());
        out.extend_from_slice(&(state as u32).to_le_bytes());
        out.extend(renormalized.into_iter().rev());
        Ok(out)
    }
}

struct BinaryRansDecoder<'a> {
    packed: &'a [u8],
    state: u64,
    offset: usize,
    model: AdaptiveModel<'a>,
    event_count: usize,
}

impl<'a> BinaryRansDecoder<'a> {
    fn new(packed: &'a [u8], weights: &'a EntropyWeights) -> Result<Self, OptimumV2Error> {
        if packed.len() < 4 {
            return Err(packet_error("BGF1 rANS stream is truncated"));
        }
        let state = u32::from_le_bytes(packed[..4].try_into().unwrap()) as u64;
        if !(RANS_L..(RANS_L << 8)).contains(&state) {
            return Err(packet_error("BGF1 rANS initial state is invalid"));
        }
        Ok(Self {
            packed,
            state,
            offset: 4,
            model: AdaptiveModel::new(weights),
            event_count: 0,
        })
    }

    fn event_count(&self) -> usize {
        self.event_count
    }

    fn read(
        &mut self,
        family: ContextFamily,
        scale: u8,
        token: u8,
        position: u8,
    ) -> Result<u8, OptimumV2Error> {
        let probability_one = u64::from(self.model.probability(family, scale, token, position));
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
                .ok_or_else(|| packet_error("BGF1 rANS renormalization is truncated"))?;
            self.state = (self.state << 8) | u64::from(byte);
            self.offset += 1;
        }
        self.model.observe(family, scale, token, position, bit);
        self.event_count = self
            .event_count
            .checked_add(1)
            .ok_or_else(|| packet_error("BGF1 event count overflows"))?;
        Ok(bit)
    }

    fn read_value(&mut self, scale: u8, token: u8) -> Result<i64, OptimumV2Error> {
        if self.read(ContextFamily::Nonzero, scale, token, 0)? == 0 {
            return Ok(0);
        }
        let mut exponent = 0_u8;
        while self.read(ContextFamily::Exponent, scale, token, exponent)? == 1 {
            exponent = exponent
                .checked_add(1)
                .ok_or_else(|| packet_error("BGF1 signed magnitude exponent exceeds i64"))?;
            if exponent > 63 {
                return Err(packet_error("BGF1 signed magnitude exponent exceeds i64"));
            }
        }
        let sign = self.read(ContextFamily::Sign, scale, token, 0)?;
        let mut magnitude = 1_u64 << exponent;
        for position in (0..exponent).rev() {
            magnitude |=
                u64::from(self.read(ContextFamily::Mantissa, scale, token, position)?) << position;
        }
        if magnitude > (1_u64 << 63) || (magnitude == (1_u64 << 63) && sign == 0) {
            return Err(packet_error("BGF1 decoded magnitude exceeds signed i64"));
        }
        if sign == 0 {
            Ok(magnitude as i64)
        } else if magnitude == (1_u64 << 63) {
            Ok(i64::MIN)
        } else {
            Ok(-(magnitude as i64))
        }
    }

    fn finish(&self) -> Result<(), OptimumV2Error> {
        if self.offset != self.packed.len() || self.state != RANS_L {
            return Err(packet_error(
                "BGF1 rANS stream is noncanonical or has trailing bytes",
            ));
        }
        Ok(())
    }
}

fn logit_probability(logit: i32) -> u16 {
    ((CDF_TOTAL as i32 / 2) + logit.clamp(-128, 127) * 120).clamp(1, CDF_TOTAL as i32 - 1) as u16
}

fn adapt_probability(probability_one: u16, bit: u8) -> u16 {
    let probability = u32::from(probability_one);
    if bit == 1 {
        (probability + ((CDF_TOTAL - probability) >> ADAPT_SHIFT).max(1)).min(CDF_TOTAL - 1) as u16
    } else {
        (probability - (probability >> ADAPT_SHIFT).max(1)).max(1) as u16
    }
}

fn stable_raw_bytes(
    signal: &[Vec<i64>],
    identities: &[Bgf1ChannelIdentity],
) -> Result<Vec<u8>, OptimumV2Error> {
    let mut order = (0..identities.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| identities[index].stable_id);
    let mut out = Vec::with_capacity(signal.len() * signal[0].len() * 4);
    for index in order {
        for &sample in &signal[index] {
            let sample = i32::try_from(sample).map_err(|_| {
                OptimumV2Error::InvalidInput("BGF1 sample exceeds signed i32".into())
            })?;
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Ok(out)
}

fn decode_identities(
    packed: &[u8],
    n_channels: usize,
) -> Result<Vec<Bgf1ChannelIdentity>, OptimumV2Error> {
    let mut identities = Vec::with_capacity(n_channels);
    let mut offset = 0usize;
    for _ in 0..n_channels {
        if offset + 3 > packed.len() {
            return Err(packet_error("BGF1 identity section is truncated"));
        }
        let stable_id = u16::from_le_bytes(packed[offset..offset + 2].try_into().unwrap());
        let label_length = packed[offset + 2] as usize;
        offset += 3;
        let end = offset
            .checked_add(label_length)
            .ok_or_else(|| packet_error("BGF1 identity length overflows"))?;
        if label_length == 0 || end > packed.len() {
            return Err(packet_error("BGF1 identity label is truncated"));
        }
        let label = &packed[offset..end];
        if label.iter().any(|&byte| !(0x20..=0x7e).contains(&byte)) {
            return Err(packet_error(
                "BGF1 identity label is outside printable ASCII",
            ));
        }
        identities.push(Bgf1ChannelIdentity::new(
            stable_id,
            std::str::from_utf8(label).unwrap(),
        ));
        offset = end;
    }
    if offset != packed.len() {
        return Err(packet_error("BGF1 identity section has trailing bytes"));
    }
    let mut stable_ids = identities
        .iter()
        .map(|identity| identity.stable_id as usize)
        .collect::<Vec<_>>();
    stable_ids.sort_unstable();
    if stable_ids != (0..n_channels).collect::<Vec<_>>() {
        return Err(packet_error(
            "BGF1 stable IDs are not contiguous manifest ordinals",
        ));
    }
    Ok(identities)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, OptimumV2Error> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| packet_error("BGF1 u16 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| packet_error("BGF1 u16 is truncated"))?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, OptimumV2Error> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| packet_error("BGF1 u32 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| packet_error("BGF1 u32 is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn packet_error(message: &str) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

fn decode_arithmetic(error: OptimumV2Error) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(error.to_string())
}

fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(!0, data) ^ !0
}

fn crc32c_update(mut state: u32, data: &[u8]) -> u32 {
    for &byte in data {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(state & 1));
        }
    }
    state
}

fn crc32c_zeroed_field(data: &[u8], offset: usize) -> u32 {
    let mut state = crc32c_update(!0, &data[..offset]);
    state = crc32c_update(state, &[0; 4]);
    state = crc32c_update(state, &data[offset + 4..]);
    state ^ !0
}

fn i8_vector(tensor: &Tensor, length: usize) -> Result<Vec<i8>, OptimumV2Error> {
    if tensor.data.len() != length {
        return Err(OptimumV2Error::InvalidPacket(
            "BGF1 signed-int8 vector has the wrong length".into(),
        ));
    }
    Ok(tensor.data.iter().map(|&byte| byte as i8).collect())
}

fn i8_matrix(tensor: &Tensor, rows: usize, columns: usize) -> Result<Vec<Vec<i8>>, OptimumV2Error> {
    let values = i8_vector(
        tensor,
        rows.checked_mul(columns)
            .ok_or_else(|| OptimumV2Error::InvalidPacket("BGF1 matrix size overflow".into()))?,
    )?;
    Ok(values
        .chunks_exact(columns)
        .map(|row| row.to_vec())
        .collect())
}
