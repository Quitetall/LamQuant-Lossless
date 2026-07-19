//! MIX1: causal fixed-universal/lattice expert selection with BM23 entropy.
//!
//! This module is byte-conformant with the independently implemented Python
//! development carrier. It remains an opened-construction carrier until the
//! source-frozen peer gates in ADR 0116 pass.

use crate::fixed_predictor::{FixedUniversalGraph, UniversalSession};
use crate::mix1_entropy;
use crate::mix1_lattice::{self, LatticeSide, ORDER};
use crate::mix1_multivariate::MultivariateSession;
use crate::{canonical_i32_bytes, crc32c, OptimumV2Error};

const HEADER_LEN: usize = 72;
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 131_072;
const MAX_EVENTS_PER_VALUE: usize = 129;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mix1Decoded {
    pub samples: Vec<Vec<i64>>,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub score_shift: u8,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Mix1Codec;

impl Mix1Codec {
    pub fn encode_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        Ok(self
            .encode_score_family(signal, sample_rate_mhz, bit_depth, &[score_shift])?
            .pop()
            .expect("one requested MIX1 score shift")
            .1)
    }

    pub fn encode_score_family(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shifts: &[u8],
    ) -> Result<Vec<(u8, Vec<u8>)>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(score_shifts)?;
        let universal = universal_residuals(signal, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(signal)?;
        let decoded_crc = crc32c(&canonical_i32_bytes(signal)?);
        score_shifts
            .iter()
            .map(|&score_shift| {
                let selected = select_residuals(&universal, &lattice, score_shift)?;
                let (payload, event_count) = mix1_entropy::encode(&selected, &side.parents)?;
                let graph = mix1_lattice::pack_side(&side, score_shift)?;
                let packet = pack_frame(Frame {
                    bit_depth,
                    sample_rate_mhz,
                    channels,
                    samples,
                    event_count,
                    graph,
                    payload,
                    decoded_crc,
                })?;
                Ok((score_shift, packet))
            })
            .collect()
    }

    pub fn encode_best_score_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_score_family(signal, sample_rate_mhz, bit_depth, &[2, 3, 4, 5, 6, 7, 8])?
            .into_iter()
            .min_by_key(|(score_shift, packet)| (packet.len(), *score_shift))
            .map(|(_, packet)| packet)
            .ok_or_else(|| input_error("MIX1 score family is empty"))
    }

    pub fn encode_multivariate_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_peer_family(signal, sample_rate_mhz, bit_depth, &[score_shift], &[false])?
            .pop()
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX1 multivariate family is empty"))
    }

    fn encode_peer_family(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shifts: &[u8],
        hierarchical_modes: &[bool],
    ) -> Result<Vec<(bool, u8, Vec<u8>)>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(score_shifts)?;
        if hierarchical_modes.is_empty()
            || hierarchical_modes
                .iter()
                .enumerate()
                .any(|(index, mode)| hierarchical_modes[..index].contains(mode))
        {
            return Err(input_error(
                "MIX peer entropy modes must be a nonempty unique list",
            ));
        }
        let universal = universal_residuals(signal, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(signal)?;
        let multivariate = multivariate_residuals(signal, &side.parents, bit_depth)?;
        let decoded_crc = crc32c(&canonical_i32_bytes(signal)?);
        let mut packets = Vec::with_capacity(score_shifts.len() * hierarchical_modes.len());
        for &score_shift in score_shifts {
            let selected =
                select_three_residuals(&universal, &lattice, &multivariate, score_shift)?;
            for &hierarchical in hierarchical_modes {
                let (payload, event_count) = if hierarchical {
                    mix1_entropy::encode_hierarchical(&selected, &side.parents)?
                } else {
                    mix1_entropy::encode(&selected, &side.parents)?
                };
                let mut graph = mix1_lattice::pack_side(&side, score_shift)?;
                if hierarchical {
                    graph[..4].copy_from_slice(b"MCH1");
                } else {
                    graph[..4].copy_from_slice(b"MMV1");
                }
                let packet = pack_frame(Frame {
                    bit_depth,
                    sample_rate_mhz,
                    channels,
                    samples,
                    event_count,
                    graph,
                    payload,
                    decoded_crc,
                })?;
                packets.push((hierarchical, score_shift, packet));
            }
        }
        Ok(packets)
    }

    pub fn encode_hierarchical_multivariate_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_peer_family(signal, sample_rate_mhz, bit_depth, &[score_shift], &[true])?
            .pop()
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX peer hierarchical family is empty"))
    }

    pub fn encode_best_peer_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let incumbent = self.encode_best_score_window(signal, sample_rate_mhz, bit_depth)?;
        let candidate = self
            .encode_peer_family(
                signal,
                sample_rate_mhz,
                bit_depth,
                &[2, 3, 4, 5, 6, 7, 8],
                &[false, true],
            )?
            .into_iter()
            .min_by_key(|(hierarchical, score_shift, packet)| {
                (packet.len(), *hierarchical, *score_shift)
            })
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX peer family is empty"))?;
        if candidate.len() < incumbent.len() {
            Ok(candidate)
        } else {
            Ok(incumbent)
        }
    }

    pub fn decode_window(&self, packet: &[u8]) -> Result<Mix1Decoded, OptimumV2Error> {
        let frame = unpack_frame(packet)?;
        let magic = frame.graph.get(..4);
        let hierarchical = magic == Some(&b"MCH1"[..]);
        let multivariate = magic == Some(&b"MMV1"[..]) || hierarchical;
        let mut graph = frame.graph.clone();
        if multivariate {
            graph[..4].copy_from_slice(b"MIX1");
        }
        let (score_shift, side) = mix1_lattice::parse_side(&graph, frame.channels, frame.samples)?;
        let residuals = if hierarchical {
            mix1_entropy::decode_hierarchical(
                &frame.payload,
                frame.event_count,
                frame.channels,
                frame.samples,
                &side.parents,
            )?
        } else {
            mix1_entropy::decode(
                &frame.payload,
                frame.event_count,
                frame.channels,
                frame.samples,
                &side.parents,
            )?
        };
        let samples = if multivariate {
            decode_multivariate_samples(
                &residuals,
                score_shift,
                &side,
                &side.parents,
                frame.bit_depth,
            )?
        } else {
            decode_samples(&residuals, score_shift, &side, frame.bit_depth)?
        };
        if crc32c(&canonical_i32_bytes(&samples).map_err(as_packet_error)?) != frame.decoded_crc {
            return Err(OptimumV2Error::Integrity(
                "MIX1 decoded sample CRC32C mismatch".into(),
            ));
        }
        Ok(Mix1Decoded {
            samples,
            sample_rate_mhz: frame.sample_rate_mhz,
            bit_depth: frame.bit_depth,
            score_shift,
        })
    }
}

#[derive(Debug)]
struct Frame {
    bit_depth: u8,
    sample_rate_mhz: u32,
    channels: usize,
    samples: usize,
    event_count: u32,
    graph: Vec<u8>,
    payload: Vec<u8>,
    decoded_crc: u32,
}

fn pack_frame(frame: Frame) -> Result<Vec<u8>, OptimumV2Error> {
    validate_frame(&frame, InputKind::Caller)?;
    let graph_len = u32::try_from(frame.graph.len())
        .map_err(|_| input_error("MIX1 graph length exceeds u32"))?;
    let payload_len = u32::try_from(frame.payload.len())
        .map_err(|_| input_error("MIX1 payload length exceeds u32"))?;
    let mut packet = Vec::with_capacity(HEADER_LEN + frame.graph.len() + frame.payload.len());
    packet.extend_from_slice(b"OV2P");
    packet.extend_from_slice(&[2, 0, frame.bit_depth, 2]);
    packet.extend_from_slice(&frame.sample_rate_mhz.to_le_bytes());
    packet.extend_from_slice(
        &u32::try_from(frame.channels)
            .map_err(|_| input_error("MIX1 channel count exceeds u32"))?
            .to_le_bytes(),
    );
    packet.extend_from_slice(
        &u32::try_from(frame.samples)
            .map_err(|_| input_error("MIX1 sample count exceeds u32"))?
            .to_le_bytes(),
    );
    packet.extend_from_slice(&frame.event_count.to_le_bytes());
    packet.extend_from_slice(&graph_len.to_le_bytes());
    packet.extend_from_slice(&payload_len.to_le_bytes());
    packet.extend_from_slice(&frame.decoded_crc.to_le_bytes());
    packet.extend_from_slice(&[0u8; 32]);
    packet.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(packet.len(), HEADER_LEN);
    packet.extend_from_slice(&frame.graph);
    packet.extend_from_slice(&frame.payload);
    let packet_crc = crc32c(&packet);
    packet[68..72].copy_from_slice(&packet_crc.to_le_bytes());
    Ok(packet)
}

fn unpack_frame(packet: &[u8]) -> Result<Frame, OptimumV2Error> {
    if packet.len() < HEADER_LEN {
        return Err(packet_error("OV2P header is truncated"));
    }
    if &packet[..4] != b"OV2P" || packet[4] != 2 {
        return Err(packet_error("OV2P magic or version is invalid"));
    }
    if packet[5] != 0 || packet[7] != 2 || packet[36..68].iter().any(|&byte| byte != 0) {
        return Err(packet_error("MIX1 frame identity is invalid"));
    }
    let graph_len = read_u32(packet, 24)? as usize;
    let payload_len = read_u32(packet, 28)? as usize;
    let expected_len = HEADER_LEN
        .checked_add(graph_len)
        .and_then(|value| value.checked_add(payload_len))
        .ok_or_else(|| packet_error("OV2P section lengths overflow"))?;
    if expected_len != packet.len() {
        return Err(packet_error("OV2P section lengths do not match packet"));
    }
    let packet_crc = read_u32(packet, 68)?;
    let mut zeroed = packet.to_vec();
    zeroed[68..72].fill(0);
    if crc32c(&zeroed) != packet_crc {
        return Err(OptimumV2Error::Integrity("OV2P packet CRC mismatch".into()));
    }
    let graph_end = HEADER_LEN + graph_len;
    let frame = Frame {
        bit_depth: packet[6],
        sample_rate_mhz: read_u32(packet, 8)?,
        channels: read_u32(packet, 12)? as usize,
        samples: read_u32(packet, 16)? as usize,
        event_count: read_u32(packet, 20)?,
        graph: packet[HEADER_LEN..graph_end].to_vec(),
        payload: packet[graph_end..].to_vec(),
        decoded_crc: read_u32(packet, 32)?,
    };
    validate_frame(&frame, InputKind::Packet)?;
    Ok(frame)
}

fn validate_frame(frame: &Frame, kind: InputKind) -> Result<(), OptimumV2Error> {
    let values = frame.channels.checked_mul(frame.samples);
    let maximum_events = values.and_then(|count| count.checked_mul(MAX_EVENTS_PER_VALUE));
    let valid = (1..=MAX_CHANNELS).contains(&frame.channels)
        && (1..=MAX_SAMPLES).contains(&frame.samples)
        && values.is_some_and(|count| count <= MAX_VALUES)
        && (1..=32).contains(&frame.bit_depth)
        && frame.sample_rate_mhz != 0
        && values.is_some_and(|count| frame.event_count as usize >= count)
        && maximum_events.is_some_and(|count| frame.event_count as usize <= count)
        && frame.payload.len() >= 4;
    if valid {
        Ok(())
    } else {
        Err(match kind {
            InputKind::Caller => input_error("OV2P dimensions or counts exceed bounds"),
            InputKind::Packet => packet_error("OV2P dimensions or counts exceed bounds"),
        })
    }
}

fn universal_residuals(
    signal: &[Vec<i64>],
    bit_depth: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = signal.len();
    let samples = signal[0].len();
    let graph = FixedUniversalGraph::new(
        (0..channels)
            .map(|channel| {
                if channel == 0 {
                    Ok(None)
                } else {
                    Ok(Some(u16::try_from(channel - 1).map_err(|_| {
                        input_error("MIX1 universal parent exceeds u16")
                    })?))
                }
            })
            .collect::<Result<Vec<_>, OptimumV2Error>>()?,
    )?;
    let mut session = UniversalSession::new(graph, bit_depth)?;
    let mut residuals = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            let prediction = session.prediction(channel, &current)?;
            let sample = signal[channel][time];
            residuals[channel][time] = sample
                .checked_sub(prediction)
                .ok_or_else(|| arithmetic_error("MIX1 universal residual exceeds i64"))?;
            session.observe(channel, &current, sample, prediction)?;
            current[channel] = sample;
        }
        session.finish_time(&current)?;
    }
    Ok(residuals)
}

fn multivariate_residuals(
    signal: &[Vec<i64>],
    parents: &[Vec<usize>],
    bit_depth: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = signal.len();
    let samples = signal[0].len();
    let mut session = MultivariateSession::new(parents, bit_depth)?;
    let mut residuals = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            let prediction = session.prediction(channel, &current)?;
            let sample = signal[channel][time];
            residuals[channel][time] = sample
                .checked_sub(prediction)
                .ok_or_else(|| arithmetic_error("MIX1 multivariate residual exceeds i64"))?;
            session.observe(channel, &current, sample, prediction)?;
            current[channel] = sample;
        }
        session.finish_time(&current)?;
    }
    Ok(residuals)
}

fn validate_score_shifts(score_shifts: &[u8]) -> Result<(), OptimumV2Error> {
    if score_shifts.is_empty()
        || score_shifts.iter().any(|shift| !(2..=8).contains(shift))
        || score_shifts
            .iter()
            .enumerate()
            .any(|(index, shift)| score_shifts[..index].contains(shift))
    {
        return Err(input_error(
            "MIX1 score shifts must be a nonempty unique list in 2..=8",
        ));
    }
    Ok(())
}

fn select_residuals(
    universal: &[Vec<i64>],
    lattice: &[Vec<i64>],
    score_shift: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if universal.len() != lattice.len()
        || universal.is_empty()
        || universal[0].is_empty()
        || universal
            .iter()
            .zip(lattice)
            .any(|(left, right)| left.len() != universal[0].len() || right.len() != left.len())
    {
        return Err(input_error("MIX1 expert residual dimensions differ"));
    }
    let channels = universal.len();
    let samples = universal[0].len();
    let mut selector = Selector::new(channels, score_shift)?;
    let mut selected = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        for channel in 0..channels {
            selected[channel][time] = if selector.universal(channel)? {
                universal[channel][time]
            } else {
                lattice[channel][time]
            };
            selector.observe(channel, universal[channel][time], lattice[channel][time])?;
        }
    }
    Ok(selected)
}

fn select_three_residuals(
    universal: &[Vec<i64>],
    lattice: &[Vec<i64>],
    multivariate: &[Vec<i64>],
    score_shift: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if universal.len() != lattice.len()
        || universal.len() != multivariate.len()
        || universal.is_empty()
        || universal[0].is_empty()
        || universal
            .iter()
            .zip(lattice)
            .zip(multivariate)
            .any(|((left, middle), right)| {
                left.len() != universal[0].len()
                    || middle.len() != left.len()
                    || right.len() != left.len()
            })
    {
        return Err(input_error("MIX1 three-expert residual dimensions differ"));
    }
    let channels = universal.len();
    let samples = universal[0].len();
    let mut selector = TripleSelector::new(channels, score_shift)?;
    let mut selected = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        for channel in 0..channels {
            selected[channel][time] = match selector.choice(channel)? {
                ExpertChoice::Universal => universal[channel][time],
                ExpertChoice::Lattice => lattice[channel][time],
                ExpertChoice::Multivariate => multivariate[channel][time],
            };
            selector.observe(
                channel,
                universal[channel][time],
                lattice[channel][time],
                multivariate[channel][time],
            )?;
        }
    }
    Ok(selected)
}

fn decode_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    bit_depth: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = residuals.len();
    if channels == 0
        || side.parents.len() != channels
        || residuals[0].is_empty()
        || residuals.iter().any(|row| row.len() != residuals[0].len())
    {
        return Err(packet_error("MIX1 residual dimensions are invalid"));
    }
    let samples = residuals[0].len();
    let graph = FixedUniversalGraph::new(
        (0..channels)
            .map(|channel| {
                if channel == 0 {
                    Ok(None)
                } else {
                    Ok(Some(u16::try_from(channel - 1).map_err(|_| {
                        packet_error("MIX1 universal parent exceeds u16")
                    })?))
                }
            })
            .collect::<Result<Vec<_>, OptimumV2Error>>()?,
    )
    .map_err(as_packet_error)?;
    let mut universal = UniversalSession::new(graph, bit_depth).map_err(as_packet_error)?;
    let mut selector = Selector::new(channels, score_shift).map_err(as_packet_error)?;
    let mut previous_backward = vec![vec![0i128; ORDER + 1]; channels];
    let mut reconstructed = vec![vec![0i64; samples]; channels];
    let magnitude = 1i64 << (bit_depth - 1);
    let minimum = -magnitude;
    let maximum = magnitude - 1;

    for time in 0..samples {
        let mut current_samples = vec![0i64; channels];
        let mut current_innovations = vec![0i64; channels];
        let mut current_backward = vec![vec![0i128; ORDER + 1]; channels];
        for channel in 0..channels {
            let prediction = universal
                .prediction(channel, &current_samples)
                .map_err(as_packet_error)?;
            let graph_prediction =
                mix1_lattice::graph_prediction(side, channel, &current_innovations)
                    .map_err(as_packet_error)?;
            let choose_universal = selector.universal(channel).map_err(as_packet_error)?;
            let coded = residuals[channel][time];
            let sample = if choose_universal {
                prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 universal reconstruction exceeds i64"))?
            } else {
                let innovation = coded
                    .checked_add(graph_prediction)
                    .ok_or_else(|| packet_error("MIX1 lattice innovation exceeds i64"))?;
                mix1_lattice::inverse_sample(
                    innovation,
                    &side.coefficients,
                    &previous_backward[channel],
                )
                .map_err(as_packet_error)?
            };
            if !(minimum..=maximum).contains(&sample) {
                return Err(packet_error(
                    "decoded MIX1 sample exceeds declared bit depth",
                ));
            }
            let innovation = mix1_lattice::analyze_sample(
                sample,
                &side.coefficients,
                &previous_backward[channel],
                &mut current_backward[channel],
            )
            .map_err(as_packet_error)?;
            let lattice_residual = innovation
                .checked_sub(graph_prediction)
                .ok_or_else(|| packet_error("MIX1 lattice residual exceeds i64"))?;
            let universal_residual = sample
                .checked_sub(prediction)
                .ok_or_else(|| packet_error("MIX1 universal residual exceeds i64"))?;
            let selected = if choose_universal {
                universal_residual
            } else {
                lattice_residual
            };
            if selected != coded {
                return Err(packet_error(
                    "decoded MIX1 selector residual is inconsistent",
                ));
            }
            universal
                .observe(channel, &current_samples, sample, prediction)
                .map_err(as_packet_error)?;
            selector
                .observe(channel, universal_residual, lattice_residual)
                .map_err(as_packet_error)?;
            reconstructed[channel][time] = sample;
            current_samples[channel] = sample;
            current_innovations[channel] = innovation;
        }
        universal
            .finish_time(&current_samples)
            .map_err(as_packet_error)?;
        previous_backward = current_backward;
    }
    Ok(reconstructed)
}

fn decode_multivariate_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = residuals.len();
    if channels == 0
        || side.parents.len() != channels
        || residuals[0].is_empty()
        || residuals.iter().any(|row| row.len() != residuals[0].len())
    {
        return Err(packet_error(
            "MIX1 multivariate residual dimensions are invalid",
        ));
    }
    let samples = residuals[0].len();
    let graph = FixedUniversalGraph::new(
        (0..channels)
            .map(|channel| {
                if channel == 0 {
                    Ok(None)
                } else {
                    Ok(Some(u16::try_from(channel - 1).map_err(|_| {
                        packet_error("MIX1 universal parent exceeds u16")
                    })?))
                }
            })
            .collect::<Result<Vec<_>, OptimumV2Error>>()?,
    )
    .map_err(as_packet_error)?;
    let mut universal = UniversalSession::new(graph, bit_depth).map_err(as_packet_error)?;
    let mut multivariate =
        MultivariateSession::new(multivariate_parents, bit_depth).map_err(as_packet_error)?;
    let mut selector = TripleSelector::new(channels, score_shift).map_err(as_packet_error)?;
    let mut previous_backward = vec![vec![0i128; ORDER + 1]; channels];
    let mut reconstructed = vec![vec![0i64; samples]; channels];
    let magnitude = 1i64 << (bit_depth - 1);
    let minimum = -magnitude;
    let maximum = magnitude - 1;

    for time in 0..samples {
        let mut current_samples = vec![0i64; channels];
        let mut current_innovations = vec![0i64; channels];
        let mut current_backward = vec![vec![0i128; ORDER + 1]; channels];
        for channel in 0..channels {
            let universal_prediction = universal
                .prediction(channel, &current_samples)
                .map_err(as_packet_error)?;
            let multivariate_prediction = multivariate
                .prediction(channel, &current_samples)
                .map_err(as_packet_error)?;
            let graph_prediction =
                mix1_lattice::graph_prediction(side, channel, &current_innovations)
                    .map_err(as_packet_error)?;
            let choice = selector.choice(channel).map_err(as_packet_error)?;
            let coded = residuals[channel][time];
            let sample = match choice {
                ExpertChoice::Universal => universal_prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 universal reconstruction exceeds i64"))?,
                ExpertChoice::Multivariate => multivariate_prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 multivariate reconstruction exceeds i64"))?,
                ExpertChoice::Lattice => {
                    let innovation = coded
                        .checked_add(graph_prediction)
                        .ok_or_else(|| packet_error("MIX1 lattice innovation exceeds i64"))?;
                    mix1_lattice::inverse_sample(
                        innovation,
                        &side.coefficients,
                        &previous_backward[channel],
                    )
                    .map_err(as_packet_error)?
                }
            };
            if !(minimum..=maximum).contains(&sample) {
                return Err(packet_error(
                    "decoded MIX1 multivariate sample exceeds declared bit depth",
                ));
            }
            let innovation = mix1_lattice::analyze_sample(
                sample,
                &side.coefficients,
                &previous_backward[channel],
                &mut current_backward[channel],
            )
            .map_err(as_packet_error)?;
            let lattice_residual = innovation
                .checked_sub(graph_prediction)
                .ok_or_else(|| packet_error("MIX1 lattice residual exceeds i64"))?;
            let universal_residual = sample
                .checked_sub(universal_prediction)
                .ok_or_else(|| packet_error("MIX1 universal residual exceeds i64"))?;
            let multivariate_residual = sample
                .checked_sub(multivariate_prediction)
                .ok_or_else(|| packet_error("MIX1 multivariate residual exceeds i64"))?;
            let selected = match choice {
                ExpertChoice::Universal => universal_residual,
                ExpertChoice::Lattice => lattice_residual,
                ExpertChoice::Multivariate => multivariate_residual,
            };
            if selected != coded {
                return Err(packet_error(
                    "decoded MIX1 multivariate selector residual is inconsistent",
                ));
            }
            universal
                .observe(channel, &current_samples, sample, universal_prediction)
                .map_err(as_packet_error)?;
            multivariate
                .observe(channel, &current_samples, sample, multivariate_prediction)
                .map_err(as_packet_error)?;
            selector
                .observe(
                    channel,
                    universal_residual,
                    lattice_residual,
                    multivariate_residual,
                )
                .map_err(as_packet_error)?;
            reconstructed[channel][time] = sample;
            current_samples[channel] = sample;
            current_innovations[channel] = innovation;
        }
        universal
            .finish_time(&current_samples)
            .map_err(as_packet_error)?;
        multivariate
            .finish_time(&current_samples)
            .map_err(as_packet_error)?;
        previous_backward = current_backward;
    }
    Ok(reconstructed)
}

#[derive(Debug, Clone)]
struct Selector {
    score_shift: u8,
    universal_scores: Vec<u128>,
    lattice_scores: Vec<u128>,
}

impl Selector {
    fn new(channels: usize, score_shift: u8) -> Result<Self, OptimumV2Error> {
        if channels == 0 || !(2..=8).contains(&score_shift) {
            return Err(input_error("MIX1 selector shape is invalid"));
        }
        Ok(Self {
            score_shift,
            universal_scores: vec![0; channels],
            lattice_scores: vec![0; channels],
        })
    }

    fn universal(&self, channel: usize) -> Result<bool, OptimumV2Error> {
        let universal = self
            .universal_scores
            .get(channel)
            .ok_or_else(|| input_error("MIX1 selector channel is out of range"))?;
        Ok(*universal <= self.lattice_scores[channel])
    }

    fn observe(
        &mut self,
        channel: usize,
        universal: i64,
        lattice: i64,
    ) -> Result<(), OptimumV2Error> {
        if channel >= self.universal_scores.len() {
            return Err(input_error("MIX1 selector channel is out of range"));
        }
        let denominator = 1u128 << self.score_shift;
        self.universal_scores[channel] = update_score(
            self.universal_scores[channel],
            universal.unsigned_abs(),
            denominator,
        )?;
        self.lattice_scores[channel] = update_score(
            self.lattice_scores[channel],
            lattice.unsigned_abs(),
            denominator,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertChoice {
    Universal,
    Lattice,
    Multivariate,
}

#[derive(Debug, Clone)]
struct TripleSelector {
    score_shift: u8,
    universal_scores: Vec<u128>,
    lattice_scores: Vec<u128>,
    multivariate_scores: Vec<u128>,
}

impl TripleSelector {
    fn new(channels: usize, score_shift: u8) -> Result<Self, OptimumV2Error> {
        if channels == 0 || !(2..=8).contains(&score_shift) {
            return Err(input_error("MIX1 three-expert selector shape is invalid"));
        }
        Ok(Self {
            score_shift,
            universal_scores: vec![0; channels],
            lattice_scores: vec![0; channels],
            multivariate_scores: vec![0; channels],
        })
    }

    fn choice(&self, channel: usize) -> Result<ExpertChoice, OptimumV2Error> {
        let universal = *self
            .universal_scores
            .get(channel)
            .ok_or_else(|| input_error("MIX1 three-expert selector channel is out of range"))?;
        let lattice = self.lattice_scores[channel];
        let multivariate = self.multivariate_scores[channel];
        let mut choice = ExpertChoice::Universal;
        let mut best = universal;
        if lattice < best {
            choice = ExpertChoice::Lattice;
            best = lattice;
        }
        if multivariate < best {
            choice = ExpertChoice::Multivariate;
        }
        Ok(choice)
    }

    fn observe(
        &mut self,
        channel: usize,
        universal: i64,
        lattice: i64,
        multivariate: i64,
    ) -> Result<(), OptimumV2Error> {
        if channel >= self.universal_scores.len() {
            return Err(input_error(
                "MIX1 three-expert selector channel is out of range",
            ));
        }
        let denominator = 1u128 << self.score_shift;
        self.universal_scores[channel] = update_score(
            self.universal_scores[channel],
            universal.unsigned_abs(),
            denominator,
        )?;
        self.lattice_scores[channel] = update_score(
            self.lattice_scores[channel],
            lattice.unsigned_abs(),
            denominator,
        )?;
        self.multivariate_scores[channel] = update_score(
            self.multivariate_scores[channel],
            multivariate.unsigned_abs(),
            denominator,
        )?;
        Ok(())
    }
}

fn update_score(score: u128, magnitude: u64, denominator: u128) -> Result<u128, OptimumV2Error> {
    (denominator - 1)
        .checked_mul(score)
        .and_then(|value| value.checked_add(denominator / 2))
        .map(|value| value / denominator)
        .and_then(|value| value.checked_add(u128::from(magnitude)))
        .ok_or_else(|| arithmetic_error("MIX1 selector score overflows u128"))
}

fn validate_signal(
    signal: &[Vec<i64>],
    sample_rate_mhz: u32,
    bit_depth: u8,
) -> Result<(usize, usize), OptimumV2Error> {
    let channels = signal.len();
    if !(1..=MAX_CHANNELS).contains(&channels) || signal[0].is_empty() {
        return Err(input_error("MIX1 signal dimensions are invalid"));
    }
    let samples = signal[0].len();
    let values = channels.checked_mul(samples);
    if samples > MAX_SAMPLES
        || values.map_or(true, |count| count > MAX_VALUES)
        || signal.iter().any(|row| row.len() != samples)
        || sample_rate_mhz == 0
        || !(1..=32).contains(&bit_depth)
    {
        return Err(input_error("MIX1 signal dimensions or context are invalid"));
    }
    let magnitude = 1i64 << (bit_depth - 1);
    let range = -magnitude..=magnitude - 1;
    if signal
        .iter()
        .flatten()
        .any(|&sample| !range.contains(&sample) || i32::try_from(sample).is_err())
    {
        return Err(input_error("MIX1 samples exceed bit depth or signed i32"));
    }
    Ok((channels, samples))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, OptimumV2Error> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| packet_error("OV2P integer field is truncated"))?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

#[derive(Clone, Copy)]
enum InputKind {
    Caller,
    Packet,
}

fn input_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn packet_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidPacket(message.into())
}

fn arithmetic_error(message: impl Into<String>) -> OptimumV2Error {
    OptimumV2Error::InvalidInput(message.into())
}

fn as_packet_error(error: OptimumV2Error) -> OptimumV2Error {
    match error {
        OptimumV2Error::Integrity(message) => OptimumV2Error::Integrity(message),
        other => packet_error(other.to_string()),
    }
}
