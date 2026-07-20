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
const COMPACT_HEADER_LEN: usize = 40;
const ULTRA_COMPACT_HEADER_LEN: usize = 24;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mix1EntropyProfile {
    pub score_shift: u8,
    pub channel_context_mask: u8,
    pub history_context: u8,
    pub scale_profile: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mix1TunedProfile {
    pub entropy: Mix1EntropyProfile,
    pub parent_history_depth: u8,
    pub parent_penalty: u64,
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

    pub fn encode_channel_context_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
        channel_context_mask: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_channel_context_family(
            signal,
            sample_rate_mhz,
            bit_depth,
            &[score_shift],
            &[channel_context_mask],
        )?
        .pop()
        .map(|(_, _, packet)| packet)
        .ok_or_else(|| input_error("MIX peer channel-context family is empty"))
    }

    fn encode_channel_context_family(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shifts: &[u8],
        channel_context_masks: &[u8],
    ) -> Result<Vec<(u8, u8, Vec<u8>)>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(score_shifts)?;
        if channel_context_masks.is_empty()
            || channel_context_masks
                .iter()
                .any(|mask| !(2..=7).contains(mask))
            || channel_context_masks
                .iter()
                .enumerate()
                .any(|(index, mask)| channel_context_masks[..index].contains(mask))
        {
            return Err(input_error(
                "MIX peer channel-context masks must be a nonempty unique list in 2..=7",
            ));
        }
        let universal = universal_residuals(signal, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(signal)?;
        let multivariate = multivariate_residuals(signal, &side.parents, bit_depth)?;
        let decoded_crc = crc32c(&canonical_i32_bytes(signal)?);
        let mut packets = Vec::with_capacity(score_shifts.len() * channel_context_masks.len());
        for &score_shift in score_shifts {
            let selected =
                select_three_residuals(&universal, &lattice, &multivariate, score_shift)?;
            for &channel_context_mask in channel_context_masks {
                let (payload, event_count) = mix1_entropy::encode_channel_context(
                    &selected,
                    &side.parents,
                    channel_context_mask,
                )?;
                let mut graph = mix1_lattice::pack_side(&side, score_shift)?;
                graph[..4].copy_from_slice(b"MCX1");
                graph.insert(6, channel_context_mask);
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
                packets.push((channel_context_mask, score_shift, packet));
            }
        }
        Ok(packets)
    }

    pub fn encode_common_mode_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
        channel_context_mask: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_common_mode_family(
            signal,
            sample_rate_mhz,
            bit_depth,
            &[score_shift],
            &[channel_context_mask],
        )?
        .pop()
        .map(|(_, _, packet)| packet)
        .ok_or_else(|| input_error("MIX peer common-mode family is empty"))
    }

    pub fn encode_compact_common_profile_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        profile: Mix1EntropyProfile,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let Mix1EntropyProfile {
            score_shift,
            channel_context_mask,
            history_context,
            scale_profile,
        } = profile;
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(&[score_shift])?;
        if !(2..=7).contains(&channel_context_mask)
            || !valid_profile_history(history_context)
            || scale_profile > 6
        {
            return Err(input_error(
                "MIX peer compact common-mode entropy profile is invalid",
            ));
        }
        let universal = universal_residuals(signal, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(signal)?;
        let multivariate = multivariate_residuals(signal, &side.parents, bit_depth)?;
        let common_mode = common_mode_residuals(signal)?;
        let selected = select_four_residuals(
            &universal,
            &lattice,
            &multivariate,
            &common_mode,
            score_shift,
        )?;
        let (payload, event_count) = mix1_entropy::encode_profile_channel_context(
            &selected,
            &side.parents,
            channel_context_mask,
            history_context,
            scale_profile,
        )?;
        let (mut graph, coefficient_rice_k, weight_rice_k) =
            mix1_lattice::pack_side_adaptive(&side, score_shift)?;
        graph[..4].copy_from_slice(b"BQX1");
        let tail = graph.split_off(6);
        graph.extend_from_slice(&[
            channel_context_mask,
            history_context,
            scale_profile,
            coefficient_rice_k,
            weight_rice_k,
        ]);
        graph.extend_from_slice(&tail);
        pack_frame_ultracompact(Frame {
            bit_depth,
            sample_rate_mhz,
            channels,
            samples,
            event_count,
            graph,
            payload,
            decoded_crc: crc32c(&canonical_i32_bytes(signal)?),
        })
    }

    fn encode_common_mode_family(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shifts: &[u8],
        channel_context_masks: &[u8],
    ) -> Result<Vec<(u8, u8, Vec<u8>)>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(score_shifts)?;
        if channel_context_masks.is_empty()
            || channel_context_masks
                .iter()
                .any(|mask| !(2..=7).contains(mask))
            || channel_context_masks
                .iter()
                .enumerate()
                .any(|(index, mask)| channel_context_masks[..index].contains(mask))
        {
            return Err(input_error(
                "MIX peer common-mode masks must be a nonempty unique list in 2..=7",
            ));
        }
        let universal = universal_residuals(signal, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(signal)?;
        let multivariate = multivariate_residuals(signal, &side.parents, bit_depth)?;
        let common_mode = common_mode_residuals(signal)?;
        let decoded_crc = crc32c(&canonical_i32_bytes(signal)?);
        let mut packets = Vec::with_capacity(score_shifts.len() * channel_context_masks.len());
        for &score_shift in score_shifts {
            let selected = select_four_residuals(
                &universal,
                &lattice,
                &multivariate,
                &common_mode,
                score_shift,
            )?;
            for &channel_context_mask in channel_context_masks {
                let (payload, event_count) = mix1_entropy::encode_channel_context(
                    &selected,
                    &side.parents,
                    channel_context_mask,
                )?;
                let mut graph = mix1_lattice::pack_side(&side, score_shift)?;
                graph[..4].copy_from_slice(b"MQX1");
                graph.insert(6, channel_context_mask);
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
                packets.push((channel_context_mask, score_shift, packet));
            }
        }
        Ok(packets)
    }

    pub fn encode_permuted_common_mode_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shift: u8,
        channel_context_mask: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_permuted_common_mode_family(
            signal,
            sample_rate_mhz,
            bit_depth,
            &[score_shift],
            &[channel_context_mask],
        )?
        .pop()
        .map(|(_, _, packet)| packet)
        .ok_or_else(|| input_error("MIX peer permuted common-mode family is empty"))
    }

    fn encode_permuted_common_mode_family(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        score_shifts: &[u8],
        channel_context_masks: &[u8],
    ) -> Result<Vec<(u8, u8, Vec<u8>)>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(score_shifts)?;
        if channel_context_masks.is_empty()
            || channel_context_masks
                .iter()
                .any(|mask| !(2..=7).contains(mask))
            || channel_context_masks
                .iter()
                .enumerate()
                .any(|(index, mask)| channel_context_masks[..index].contains(mask))
        {
            return Err(input_error(
                "MIX peer permuted common-mode masks must be a nonempty unique list in 2..=7",
            ));
        }
        let permutation = fit_channel_permutation(signal)?;
        let permuted = permutation
            .iter()
            .map(|&channel| signal[channel].clone())
            .collect::<Vec<_>>();
        let universal = universal_residuals(&permuted, bit_depth)?;
        let (side, lattice) = mix1_lattice::fit_and_analyze(&permuted)?;
        let multivariate = multivariate_residuals(&permuted, &side.parents, bit_depth)?;
        let common_mode = common_mode_residuals(&permuted)?;
        let decoded_crc = crc32c(&canonical_i32_bytes(signal)?);
        let mut packets = Vec::with_capacity(score_shifts.len() * channel_context_masks.len());
        for &score_shift in score_shifts {
            let selected = select_four_residuals(
                &universal,
                &lattice,
                &multivariate,
                &common_mode,
                score_shift,
            )?;
            for &channel_context_mask in channel_context_masks {
                let (payload, event_count) = mix1_entropy::encode_channel_context(
                    &selected,
                    &side.parents,
                    channel_context_mask,
                )?;
                let mut graph = mix1_lattice::pack_side(&side, score_shift)?;
                graph[..4].copy_from_slice(b"MPX1");
                let tail = graph.split_off(6);
                graph.push(channel_context_mask);
                for &channel in &permutation {
                    graph.push(
                        u8::try_from(channel)
                            .map_err(|_| input_error("MIX peer permutation channel exceeds u8"))?,
                    );
                }
                graph.extend_from_slice(&tail);
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
                packets.push((channel_context_mask, score_shift, packet));
            }
        }
        Ok(packets)
    }

    pub fn encode_tuned_permuted_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        profile: Mix1TunedProfile,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let Mix1TunedProfile {
            entropy:
                Mix1EntropyProfile {
                    score_shift,
                    channel_context_mask,
                    history_context,
                    scale_profile,
                },
            parent_history_depth,
            parent_penalty,
        } = profile;
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        validate_score_shifts(&[score_shift])?;
        if !(2..=7).contains(&channel_context_mask)
            || scale_profile > 6
            || parent_history_depth > 4
            || !valid_profile_history(history_context)
        {
            return Err(input_error("MIX peer tuned entropy profile is invalid"));
        }
        let permutation = fit_channel_permutation(signal)?;
        let permuted = permutation
            .iter()
            .map(|&channel| signal[channel].clone())
            .collect::<Vec<_>>();
        let universal = universal_residuals(&permuted, bit_depth)?;
        let (side, lattice) =
            mix1_lattice::fit_and_analyze_with_parent_penalty(&permuted, parent_penalty)?;
        let multivariate = multivariate_residuals_with_parent_history(
            &permuted,
            &side.parents,
            bit_depth,
            usize::from(parent_history_depth),
        )?;
        let common_mode = common_mode_residuals(&permuted)?;
        let selected = select_four_residuals(
            &universal,
            &lattice,
            &multivariate,
            &common_mode,
            score_shift,
        )?;
        let (payload, event_count) = mix1_entropy::encode_profile_channel_context(
            &selected,
            &side.parents,
            channel_context_mask,
            history_context,
            scale_profile,
        )?;
        let (mut graph, coefficient_rice_k, weight_rice_k) =
            mix1_lattice::pack_side_adaptive(&side, score_shift)?;
        graph[..4].copy_from_slice(b"APX1");
        let tail = graph.split_off(6);
        graph.extend_from_slice(&[
            channel_context_mask,
            history_context,
            scale_profile,
            parent_history_depth,
            coefficient_rice_k,
            weight_rice_k,
        ]);
        graph.extend_from_slice(&pack_permutation_indices(&permutation)?);
        graph.extend_from_slice(&tail);
        pack_frame_ultracompact(Frame {
            bit_depth,
            sample_rate_mhz,
            channels,
            samples,
            event_count,
            graph,
            payload,
            decoded_crc: crc32c(&canonical_i32_bytes(signal)?),
        })
    }

    pub fn encode_best_peer_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let mut candidates =
            vec![self.encode_baseline_peer_window(signal, sample_rate_mhz, bit_depth)?];
        candidates.push(self.encode_tuned_permuted_window(
            signal,
            sample_rate_mhz,
            bit_depth,
            Mix1TunedProfile {
                entropy: Mix1EntropyProfile {
                    score_shift: 8,
                    channel_context_mask: 7,
                    history_context: 52,
                    scale_profile: 4,
                },
                parent_history_depth: 2,
                parent_penalty: 6,
            },
        )?);
        candidates.push(self.encode_tuned_permuted_window(
            signal,
            sample_rate_mhz,
            bit_depth,
            Mix1TunedProfile {
                entropy: Mix1EntropyProfile {
                    score_shift: 11,
                    channel_context_mask: 3,
                    history_context: 84,
                    scale_profile: 6,
                },
                parent_history_depth: 1,
                parent_penalty: 32,
            },
        )?);
        candidates.push(self.encode_tuned_permuted_window(
            signal,
            sample_rate_mhz,
            bit_depth,
            Mix1TunedProfile {
                entropy: Mix1EntropyProfile {
                    score_shift: 8,
                    channel_context_mask: 3,
                    history_context: 84,
                    scale_profile: 6,
                },
                parent_history_depth: 1,
                parent_penalty: 32,
            },
        )?);
        candidates.push(self.encode_compact_common_profile_window(
            signal,
            sample_rate_mhz,
            bit_depth,
            Mix1EntropyProfile {
                score_shift: 10,
                channel_context_mask: 3,
                history_context: 84,
                scale_profile: 4,
            },
        )?);
        candidates
            .into_iter()
            .enumerate()
            .min_by_key(|(priority, packet)| (packet.len(), *priority))
            .map(|(_, packet)| packet)
            .ok_or_else(|| input_error("MIX peer portfolio is empty"))
    }

    fn encode_baseline_peer_window(
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
        let extended = self
            .encode_channel_context_family(signal, sample_rate_mhz, bit_depth, &[8], &[5])?
            .into_iter()
            .min_by_key(|(channel_context_mask, score_shift, packet)| {
                (packet.len(), *channel_context_mask, *score_shift)
            })
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX peer channel-context family is empty"))?;
        let common_mode = self
            .encode_common_mode_family(signal, sample_rate_mhz, bit_depth, &[5, 6, 8], &[3])?
            .into_iter()
            .min_by_key(|(channel_context_mask, score_shift, packet)| {
                (packet.len(), *channel_context_mask, *score_shift)
            })
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX peer common-mode family is empty"))?;
        let permuted = self
            .encode_permuted_common_mode_family(
                signal,
                sample_rate_mhz,
                bit_depth,
                &[3, 5, 6, 7, 8],
                &[3, 4, 5, 7],
            )?
            .into_iter()
            .min_by_key(|(channel_context_mask, score_shift, packet)| {
                (packet.len(), *channel_context_mask, *score_shift)
            })
            .map(|(_, _, packet)| packet)
            .ok_or_else(|| input_error("MIX peer permuted common-mode family is empty"))?;
        let mut best = incumbent;
        if candidate.len() < best.len() {
            best = candidate;
        }
        if extended.len() < best.len() {
            best = extended;
        }
        if common_mode.len() < best.len() {
            best = common_mode;
        }
        if permuted.len() < best.len() {
            best = permuted;
        }
        Ok(best)
    }

    pub fn decode_window(&self, packet: &[u8]) -> Result<Mix1Decoded, OptimumV2Error> {
        let frame = unpack_frame(packet)?;
        let magic = frame.graph.get(..4);
        let hierarchical = magic == Some(&b"MCH1"[..]);
        let channel_context = magic == Some(&b"MCX1"[..]);
        let tuned_permuted = magic == Some(&b"APX1"[..]);
        let compact_common_profile = magic == Some(&b"BQX1"[..]);
        let permuted_mode = magic == Some(&b"MPX1"[..]) || tuned_permuted;
        let common_mode = magic == Some(&b"MQX1"[..])
            || magic == Some(&b"MPX1"[..])
            || tuned_permuted
            || compact_common_profile;
        let multivariate =
            magic == Some(&b"MMV1"[..]) || hierarchical || channel_context || common_mode;
        let mut graph = frame.graph.clone();

        let tuned_profile = if tuned_permuted {
            let history = *graph
                .get(7)
                .ok_or_else(|| packet_error("MIX peer tuned history context is truncated"))?;
            let scale = *graph
                .get(8)
                .ok_or_else(|| packet_error("MIX peer tuned scale profile is truncated"))?;
            let parent_history_depth = *graph
                .get(9)
                .ok_or_else(|| packet_error("MIX peer parent history depth is truncated"))?;
            let coefficient_rice_k = *graph
                .get(10)
                .ok_or_else(|| packet_error("MIX peer coefficient Rice parameter is truncated"))?;
            let weight_rice_k = *graph
                .get(11)
                .ok_or_else(|| packet_error("MIX peer weight Rice parameter is truncated"))?;
            if scale > 6
                || !valid_profile_history(history)
                || parent_history_depth > 4
                || coefficient_rice_k > 15
                || weight_rice_k > 10
            {
                return Err(packet_error("MIX peer tuned entropy profile is invalid"));
            }
            Some((
                history,
                scale,
                parent_history_depth,
                coefficient_rice_k,
                weight_rice_k,
            ))
        } else {
            None
        };
        let compact_profile = if compact_common_profile {
            let history = *graph
                .get(7)
                .ok_or_else(|| packet_error("MIX peer compact history context is truncated"))?;
            let scale = *graph
                .get(8)
                .ok_or_else(|| packet_error("MIX peer compact scale profile is truncated"))?;
            let coefficient_rice_k = *graph
                .get(9)
                .ok_or_else(|| packet_error("MIX peer coefficient Rice parameter is truncated"))?;
            let weight_rice_k = *graph
                .get(10)
                .ok_or_else(|| packet_error("MIX peer weight Rice parameter is truncated"))?;
            if !valid_profile_history(history)
                || scale > 6
                || coefficient_rice_k > 15
                || weight_rice_k > 10
            {
                return Err(packet_error(
                    "MIX peer compact common-mode profile is invalid",
                ));
            }
            Some((history, scale, coefficient_rice_k, weight_rice_k))
        } else {
            None
        };

        let permutation = if tuned_permuted {
            let start = 12usize;
            let end = start
                .checked_add(packed_permutation_len(frame.channels)?)
                .ok_or_else(|| packet_error("MIX peer permutation length overflows"))?;
            if graph.len() < end {
                return Err(packet_error("MIX peer permutation is truncated"));
            }
            Some(unpack_permutation_indices(
                &graph.drain(start..end).collect::<Vec<_>>(),
                frame.channels,
            )?)
        } else if permuted_mode {
            let end = 7usize
                .checked_add(frame.channels)
                .ok_or_else(|| packet_error("MIX peer permutation length overflows"))?;
            if graph.len() < end {
                return Err(packet_error("MIX peer permutation is truncated"));
            }
            let permutation = graph.drain(7..end).map(usize::from).collect::<Vec<_>>();
            if permutation.iter().enumerate().any(|(index, channel)| {
                *channel >= frame.channels || permutation[..index].contains(channel)
            }) {
                return Err(packet_error("MIX peer permutation is invalid"));
            }
            Some(permutation)
        } else {
            None
        };

        if tuned_permuted {
            graph.drain(7..12);
        } else if compact_common_profile {
            graph.drain(7..11);
        }
        let channel_context_mask = if channel_context || common_mode {
            if graph.len() < 7 {
                return Err(packet_error("MIX peer channel context is truncated"));
            }
            let mask = graph.remove(6);
            if !(2..=7).contains(&mask) {
                return Err(packet_error(
                    "MIX peer channel-context mask must be in 2..=7",
                ));
            }
            Some(mask)
        } else {
            None
        };
        if multivariate {
            graph[..4].copy_from_slice(b"MIX1");
        }
        let (score_shift, side) =
            if let Some((_, _, _, coefficient_rice_k, weight_rice_k)) = tuned_profile {
                mix1_lattice::parse_side_adaptive(
                    &graph,
                    frame.channels,
                    frame.samples,
                    coefficient_rice_k,
                    weight_rice_k,
                )?
            } else if let Some((_, _, coefficient_rice_k, weight_rice_k)) = compact_profile {
                mix1_lattice::parse_side_adaptive(
                    &graph,
                    frame.channels,
                    frame.samples,
                    coefficient_rice_k,
                    weight_rice_k,
                )?
            } else {
                mix1_lattice::parse_side(&graph, frame.channels, frame.samples)?
            };

        let residuals = if let Some(mask) = channel_context_mask {
            if let Some((history, scale, _, _)) = compact_profile {
                mix1_entropy::decode_profile_channel_context(
                    &frame.payload,
                    frame.event_count,
                    (frame.channels, frame.samples),
                    &side.parents,
                    mask,
                    history,
                    scale,
                )?
            } else if let Some((history, scale, _, _, _)) = tuned_profile {
                mix1_entropy::decode_profile_channel_context(
                    &frame.payload,
                    frame.event_count,
                    (frame.channels, frame.samples),
                    &side.parents,
                    mask,
                    history,
                    scale,
                )?
            } else {
                mix1_entropy::decode_channel_context(
                    &frame.payload,
                    frame.event_count,
                    frame.channels,
                    frame.samples,
                    &side.parents,
                    mask,
                )?
            }
        } else if hierarchical {
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

        let mut samples = if common_mode {
            if let Some((_, _, parent_history_depth, _, _)) = tuned_profile {
                decode_common_mode_samples_with_parent_history(
                    &residuals,
                    score_shift,
                    &side,
                    &side.parents,
                    frame.bit_depth,
                    usize::from(parent_history_depth),
                )?
            } else {
                decode_common_mode_samples(
                    &residuals,
                    score_shift,
                    &side,
                    &side.parents,
                    frame.bit_depth,
                )?
            }
        } else if multivariate {
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
        if let Some(permutation) = permutation {
            samples = unpermute_signal(&samples, &permutation)?;
            if fit_channel_permutation(&samples).map_err(as_packet_error)? != permutation {
                return Err(packet_error("MIX peer permutation is noncanonical"));
            }
        }
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

fn pack_frame_ultracompact(frame: Frame) -> Result<Vec<u8>, OptimumV2Error> {
    validate_frame(&frame, InputKind::Caller)?;
    let channels = if frame.channels == 256 {
        0
    } else {
        u8::try_from(frame.channels)
            .map_err(|_| input_error("MIX1 channel count exceeds compact u8"))?
    };
    let samples = u16::try_from(frame.samples)
        .map_err(|_| input_error("MIX1 sample count exceeds compact u16"))?;
    let graph_len = u16::try_from(frame.graph.len())
        .map_err(|_| input_error("MIX1 graph length exceeds compact u16"))?;
    let mut packet =
        Vec::with_capacity(ULTRA_COMPACT_HEADER_LEN + frame.graph.len() + frame.payload.len());
    packet.extend_from_slice(b"OV2P");
    packet.extend_from_slice(&[4, frame.bit_depth, channels, 2]);
    packet.extend_from_slice(&frame.sample_rate_mhz.to_le_bytes());
    packet.extend_from_slice(&samples.to_le_bytes());
    packet.extend_from_slice(&graph_len.to_le_bytes());
    packet.extend_from_slice(&frame.decoded_crc.to_le_bytes());
    packet.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(packet.len(), ULTRA_COMPACT_HEADER_LEN);
    packet.extend_from_slice(&frame.graph);
    packet.extend_from_slice(&frame.payload);
    let packet_crc = crc32c(&packet);
    packet[20..24].copy_from_slice(&packet_crc.to_le_bytes());
    Ok(packet)
}

fn unpack_frame(packet: &[u8]) -> Result<Frame, OptimumV2Error> {
    if packet.len() < ULTRA_COMPACT_HEADER_LEN {
        return Err(packet_error("OV2P header is truncated"));
    }
    if &packet[..4] != b"OV2P" || !matches!(packet[4], 2..=4) {
        return Err(packet_error("OV2P magic or version is invalid"));
    }
    let version = packet[4];
    let header_len = match version {
        2 => HEADER_LEN,
        3 => COMPACT_HEADER_LEN,
        4 => ULTRA_COMPACT_HEADER_LEN,
        _ => unreachable!(),
    };
    if packet.len() < header_len
        || packet[7] != 2
        || version < 4 && packet[5] != 0
        || version == 2 && packet[36..68].iter().any(|&byte| byte != 0)
    {
        return Err(packet_error("MIX1 frame identity is invalid"));
    }
    let graph_len = if version == 4 {
        usize::from(u16::from_le_bytes(packet[14..16].try_into().unwrap()))
    } else {
        read_u32(packet, 24)? as usize
    };
    let graph_end = header_len
        .checked_add(graph_len)
        .ok_or_else(|| packet_error("OV2P graph length overflows"))?;
    if graph_end > packet.len() {
        return Err(packet_error("OV2P graph length exceeds packet"));
    }
    if version < 4 {
        let payload_len = read_u32(packet, 28)? as usize;
        let expected_len = graph_end
            .checked_add(payload_len)
            .ok_or_else(|| packet_error("OV2P payload length overflows"))?;
        if expected_len != packet.len() {
            return Err(packet_error("OV2P section lengths do not match packet"));
        }
    }
    let crc_offset = header_len - 4;
    let packet_crc = read_u32(packet, crc_offset)?;
    let mut zeroed = packet.to_vec();
    zeroed[crc_offset..header_len].fill(0);
    if crc32c(&zeroed) != packet_crc {
        return Err(OptimumV2Error::Integrity("OV2P packet CRC mismatch".into()));
    }
    let (bit_depth, channels, samples, event_count, decoded_crc) = if version == 4 {
        let encoded_channels = usize::from(packet[6]);
        (
            packet[5],
            if encoded_channels == 0 {
                256
            } else {
                encoded_channels
            },
            usize::from(u16::from_le_bytes(packet[12..14].try_into().unwrap())),
            0,
            read_u32(packet, 16)?,
        )
    } else {
        (
            packet[6],
            read_u32(packet, 12)? as usize,
            read_u32(packet, 16)? as usize,
            read_u32(packet, 20)?,
            read_u32(packet, 32)?,
        )
    };
    let frame = Frame {
        bit_depth,
        sample_rate_mhz: read_u32(packet, 8)?,
        channels,
        samples,
        event_count,
        graph: packet[header_len..graph_end].to_vec(),
        payload: packet[graph_end..].to_vec(),
        decoded_crc,
    };
    validate_frame(&frame, InputKind::Packet)?;
    Ok(frame)
}

fn validate_frame(frame: &Frame, kind: InputKind) -> Result<(), OptimumV2Error> {
    let values = frame.channels.checked_mul(frame.samples);
    let maximum_events = values.and_then(|count| count.checked_mul(MAX_EVENTS_PER_VALUE));
    let event_count_valid = frame.event_count == 0 && matches!(kind, InputKind::Packet)
        || values.is_some_and(|count| frame.event_count as usize >= count)
            && maximum_events.is_some_and(|count| frame.event_count as usize <= count);
    let valid = (1..=MAX_CHANNELS).contains(&frame.channels)
        && (1..=MAX_SAMPLES).contains(&frame.samples)
        && values.is_some_and(|count| count <= MAX_VALUES)
        && (1..=32).contains(&frame.bit_depth)
        && frame.sample_rate_mhz != 0
        && event_count_valid
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
    multivariate_residuals_with_parent_history(signal, parents, bit_depth, 1)
}

fn multivariate_residuals_with_parent_history(
    signal: &[Vec<i64>],
    parents: &[Vec<usize>],
    bit_depth: u8,
    parent_history_depth: usize,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = signal.len();
    let samples = signal[0].len();
    let mut session =
        MultivariateSession::new_with_parent_history(parents, bit_depth, parent_history_depth)?;
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

fn common_mode_residuals(signal: &[Vec<i64>]) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if signal.is_empty()
        || signal[0].is_empty()
        || signal.iter().any(|row| row.len() != signal[0].len())
    {
        return Err(input_error(
            "MIX1 common-mode signal dimensions are invalid",
        ));
    }
    let channels = signal.len();
    let samples = signal[0].len();
    let mut residuals = vec![vec![0i64; samples]; channels];
    let mut previous = vec![0i64; channels];
    for time in 0..samples {
        let mut current = vec![0i64; channels];
        for channel in 0..channels {
            let prediction = common_mode_prediction(channel, &current, &previous)?;
            let sample = signal[channel][time];
            residuals[channel][time] = sample
                .checked_sub(prediction)
                .ok_or_else(|| arithmetic_error("MIX1 common-mode residual exceeds i64"))?;
            current[channel] = sample;
        }
        previous = current;
    }
    Ok(residuals)
}

fn common_mode_prediction(
    channel: usize,
    current: &[i64],
    previous: &[i64],
) -> Result<i64, OptimumV2Error> {
    if current.len() != previous.len() || channel >= current.len() {
        return Err(input_error("MIX1 common-mode row dimensions are invalid"));
    }
    if channel == 0 {
        return Ok(previous[0]);
    }
    let mut deltas = Vec::with_capacity(channel);
    for parent in 0..channel {
        deltas.push(
            current[parent]
                .checked_sub(previous[parent])
                .ok_or_else(|| arithmetic_error("MIX1 common-mode delta exceeds i64"))?,
        );
    }
    deltas.sort_unstable();
    let middle = deltas.len() / 2;
    let common_delta = if deltas.len() % 2 == 1 {
        i128::from(deltas[middle])
    } else {
        let pair_sum = i128::from(deltas[middle - 1]) + i128::from(deltas[middle]);
        if pair_sum >= 0 {
            (pair_sum + 1) / 2
        } else {
            -((-pair_sum + 1) / 2)
        }
    };
    let prediction = i128::from(previous[channel])
        .checked_add(common_delta)
        .ok_or_else(|| arithmetic_error("MIX1 common-mode prediction exceeds i128"))?;
    i64::try_from(prediction)
        .map_err(|_| arithmetic_error("MIX1 common-mode prediction exceeds i64"))
}

fn packed_permutation_len(channels: usize) -> Result<usize, OptimumV2Error> {
    if !(1..=MAX_CHANNELS).contains(&channels) {
        return Err(packet_error(
            "MIX peer permutation channel count is invalid",
        ));
    }
    let bits = (1..=channels)
        .map(|remaining| usize::BITS as usize - (remaining - 1).leading_zeros() as usize)
        .sum::<usize>();
    Ok(bits.div_ceil(8))
}

fn pack_permutation_indices(permutation: &[usize]) -> Result<Vec<u8>, OptimumV2Error> {
    if permutation.is_empty()
        || permutation.len() > MAX_CHANNELS
        || permutation.iter().enumerate().any(|(index, channel)| {
            *channel >= permutation.len() || permutation[..index].contains(channel)
        })
    {
        return Err(input_error("MIX peer permutation is invalid"));
    }
    let mut remaining = (0..permutation.len()).collect::<Vec<_>>();
    let mut writer = PermutationBitWriter::default();
    for &channel in permutation {
        let index = remaining
            .iter()
            .position(|candidate| *candidate == channel)
            .expect("validated permutation channel remains");
        let width = usize::BITS as u8 - (remaining.len() - 1).leading_zeros() as u8;
        writer.write(index, width)?;
        remaining.remove(index);
    }
    let packed = writer.finish();
    debug_assert_eq!(packed.len(), packed_permutation_len(permutation.len())?);
    Ok(packed)
}

fn unpack_permutation_indices(
    packed: &[u8],
    channels: usize,
) -> Result<Vec<usize>, OptimumV2Error> {
    if packed.len() != packed_permutation_len(channels)? {
        return Err(packet_error("MIX peer packed permutation length differs"));
    }
    let mut remaining = (0..channels).collect::<Vec<_>>();
    let mut reader = PermutationBitReader::new(packed);
    let mut permutation = Vec::with_capacity(channels);
    while !remaining.is_empty() {
        let width = usize::BITS as u8 - (remaining.len() - 1).leading_zeros() as u8;
        let index = reader.read(width)?;
        if index >= remaining.len() {
            return Err(packet_error("MIX peer packed permutation index is unused"));
        }
        permutation.push(remaining.remove(index));
    }
    reader.finish()?;
    Ok(permutation)
}

#[derive(Default)]
struct PermutationBitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl PermutationBitWriter {
    fn write(&mut self, value: usize, width: u8) -> Result<(), OptimumV2Error> {
        if width < usize::BITS as u8 && value >= 1usize << width {
            return Err(input_error("MIX peer permutation index exceeds width"));
        }
        for shift in (0..width).rev() {
            self.current = (self.current << 1) | ((value >> shift) & 1) as u8;
            self.used += 1;
            if self.used == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.used = 0;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 {
            self.bytes.push(self.current << (8 - self.used));
        }
        self.bytes
    }
}

struct PermutationBitReader<'a> {
    packed: &'a [u8],
    position: usize,
}

impl<'a> PermutationBitReader<'a> {
    fn new(packed: &'a [u8]) -> Self {
        Self {
            packed,
            position: 0,
        }
    }

    fn read(&mut self, width: u8) -> Result<usize, OptimumV2Error> {
        if self.position + usize::from(width) > self.packed.len() * 8 {
            return Err(packet_error("MIX peer packed permutation is truncated"));
        }
        let mut value = 0usize;
        for _ in 0..width {
            value = (value << 1)
                | usize::from((self.packed[self.position / 8] >> (7 - self.position % 8)) & 1);
            self.position += 1;
        }
        Ok(value)
    }

    fn finish(&mut self) -> Result<(), OptimumV2Error> {
        while self.position < self.packed.len() * 8 {
            if self.read(1)? != 0 {
                return Err(packet_error(
                    "MIX peer packed permutation has nonzero padding",
                ));
            }
        }
        Ok(())
    }
}

fn fit_channel_permutation(signal: &[Vec<i64>]) -> Result<Vec<usize>, OptimumV2Error> {
    if signal.is_empty()
        || signal[0].is_empty()
        || signal.iter().any(|row| row.len() != signal[0].len())
        || signal.len() > 256
    {
        return Err(input_error(
            "MIX peer permutation signal dimensions are invalid",
        ));
    }
    let channels = signal.len();
    let start = (0..channels)
        .min_by_key(|&channel| (delta_energy(&signal[channel]), channel))
        .expect("validated nonempty signal");
    let mut permutation = Vec::with_capacity(channels);
    let mut used = vec![false; channels];
    permutation.push(start);
    used[start] = true;
    while permutation.len() < channels {
        let previous = *permutation.last().expect("permutation has a start");
        let next = (0..channels)
            .filter(|&channel| !used[channel])
            .min_by_key(|&channel| (delta_distance(&signal[previous], &signal[channel]), channel))
            .expect("unused permutation channel remains");
        used[next] = true;
        permutation.push(next);
    }
    Ok(permutation)
}

fn delta_energy(row: &[i64]) -> u128 {
    let mut previous = 0i64;
    let mut total = 0u128;
    for &sample in row {
        total += i128::from(sample)
            .checked_sub(i128::from(previous))
            .expect("i64 difference fits i128")
            .unsigned_abs();
        previous = sample;
    }
    total
}

fn delta_distance(left: &[i64], right: &[i64]) -> u128 {
    let mut previous_left = 0i64;
    let mut previous_right = 0i64;
    let mut total = 0u128;
    for (&left_sample, &right_sample) in left.iter().zip(right) {
        let left_delta = i128::from(left_sample) - i128::from(previous_left);
        let right_delta = i128::from(right_sample) - i128::from(previous_right);
        total += (left_delta - right_delta).unsigned_abs();
        previous_left = left_sample;
        previous_right = right_sample;
    }
    total
}

fn unpermute_signal(
    permuted: &[Vec<i64>],
    permutation: &[usize],
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if permuted.len() != permutation.len()
        || permutation.iter().enumerate().any(|(index, channel)| {
            *channel >= permutation.len() || permutation[..index].contains(channel)
        })
    {
        return Err(packet_error("MIX peer permutation is invalid"));
    }
    let mut signal = vec![Vec::new(); permutation.len()];
    for (row, &channel) in permuted.iter().zip(permutation) {
        signal[channel] = row.clone();
    }
    Ok(signal)
}

fn validate_score_shifts(score_shifts: &[u8]) -> Result<(), OptimumV2Error> {
    if score_shifts.is_empty()
        || score_shifts.iter().any(|shift| !(2..=12).contains(shift))
        || score_shifts
            .iter()
            .enumerate()
            .any(|(index, shift)| score_shifts[..index].contains(shift))
    {
        return Err(input_error(
            "MIX1 score shifts must be a nonempty unique list in 2..=12",
        ));
    }
    Ok(())
}

fn valid_profile_history(history_context: u8) -> bool {
    history_context & 0x0f == 4 && (1..=7).contains(&(history_context >> 4))
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

fn select_four_residuals(
    universal: &[Vec<i64>],
    lattice: &[Vec<i64>],
    multivariate: &[Vec<i64>],
    common_mode: &[Vec<i64>],
    score_shift: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    if universal.len() != lattice.len()
        || universal.len() != multivariate.len()
        || universal.len() != common_mode.len()
        || universal.is_empty()
        || universal[0].is_empty()
        || universal
            .iter()
            .zip(lattice)
            .zip(multivariate)
            .zip(common_mode)
            .any(|(((left, middle), right), fourth)| {
                left.len() != universal[0].len()
                    || middle.len() != left.len()
                    || right.len() != left.len()
                    || fourth.len() != left.len()
            })
    {
        return Err(input_error("MIX1 four-expert residual dimensions differ"));
    }
    let channels = universal.len();
    let samples = universal[0].len();
    let mut selector = QuadSelector::new(channels, score_shift)?;
    let mut selected = vec![vec![0i64; samples]; channels];
    for time in 0..samples {
        for channel in 0..channels {
            selected[channel][time] = match selector.choice(channel)? {
                QuadExpertChoice::Universal => universal[channel][time],
                QuadExpertChoice::Lattice => lattice[channel][time],
                QuadExpertChoice::Multivariate => multivariate[channel][time],
                QuadExpertChoice::CommonMode => common_mode[channel][time],
            };
            selector.observe(
                channel,
                universal[channel][time],
                lattice[channel][time],
                multivariate[channel][time],
                common_mode[channel][time],
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

fn decode_common_mode_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    decode_four_mode_samples(
        residuals,
        score_shift,
        side,
        multivariate_parents,
        bit_depth,
        1,
    )
}

fn decode_common_mode_samples_with_parent_history(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
    parent_history_depth: usize,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    decode_four_mode_samples(
        residuals,
        score_shift,
        side,
        multivariate_parents,
        bit_depth,
        parent_history_depth,
    )
}

fn decode_four_mode_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
    parent_history_depth: usize,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let channels = residuals.len();
    if channels == 0
        || side.parents.len() != channels
        || residuals[0].is_empty()
        || residuals.iter().any(|row| row.len() != residuals[0].len())
    {
        return Err(packet_error(
            "MIX1 four-mode residual dimensions are invalid",
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
    let mut multivariate = MultivariateSession::new_with_parent_history(
        multivariate_parents,
        bit_depth,
        parent_history_depth,
    )
    .map_err(as_packet_error)?;
    let mut selector = QuadSelector::new(channels, score_shift).map_err(as_packet_error)?;
    let mut previous_backward = vec![vec![0i128; ORDER + 1]; channels];
    let mut previous_samples = vec![0i64; channels];
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
            let fourth_prediction =
                common_mode_prediction(channel, &current_samples, &previous_samples)
                    .map_err(as_packet_error)?;
            let graph_prediction =
                mix1_lattice::graph_prediction(side, channel, &current_innovations)
                    .map_err(as_packet_error)?;
            let choice = selector.choice(channel).map_err(as_packet_error)?;
            let coded = residuals[channel][time];
            let sample = match choice {
                QuadExpertChoice::Universal => universal_prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 universal reconstruction exceeds i64"))?,
                QuadExpertChoice::Multivariate => multivariate_prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 multivariate reconstruction exceeds i64"))?,
                QuadExpertChoice::CommonMode => fourth_prediction
                    .checked_add(coded)
                    .ok_or_else(|| packet_error("MIX1 fourth-mode reconstruction exceeds i64"))?,
                QuadExpertChoice::Lattice => {
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
                    "decoded MIX1 four-mode sample exceeds declared bit depth",
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
            let fourth_residual = sample
                .checked_sub(fourth_prediction)
                .ok_or_else(|| packet_error("MIX1 fourth-mode residual exceeds i64"))?;
            let selected = match choice {
                QuadExpertChoice::Universal => universal_residual,
                QuadExpertChoice::Lattice => lattice_residual,
                QuadExpertChoice::Multivariate => multivariate_residual,
                QuadExpertChoice::CommonMode => fourth_residual,
            };
            if selected != coded {
                return Err(packet_error(
                    "decoded MIX1 four-mode selector residual is inconsistent",
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
                    fourth_residual,
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
        previous_samples = current_samples;
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
        if channels == 0 || !(2..=12).contains(&score_shift) {
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
        if channels == 0 || !(2..=12).contains(&score_shift) {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuadExpertChoice {
    Universal,
    Lattice,
    Multivariate,
    CommonMode,
}

#[derive(Debug, Clone)]
struct QuadSelector {
    score_shift: u8,
    universal_scores: Vec<u128>,
    lattice_scores: Vec<u128>,
    multivariate_scores: Vec<u128>,
    common_mode_scores: Vec<u128>,
}

impl QuadSelector {
    fn new(channels: usize, score_shift: u8) -> Result<Self, OptimumV2Error> {
        if channels == 0 || !(2..=12).contains(&score_shift) {
            return Err(input_error("MIX1 four-expert selector shape is invalid"));
        }
        Ok(Self {
            score_shift,
            universal_scores: vec![0; channels],
            lattice_scores: vec![0; channels],
            multivariate_scores: vec![0; channels],
            common_mode_scores: vec![0; channels],
        })
    }

    fn choice(&self, channel: usize) -> Result<QuadExpertChoice, OptimumV2Error> {
        let universal = *self
            .universal_scores
            .get(channel)
            .ok_or_else(|| input_error("MIX1 four-expert selector channel is out of range"))?;
        let lattice = self.lattice_scores[channel];
        let multivariate = self.multivariate_scores[channel];
        let common_mode = self.common_mode_scores[channel];
        let mut choice = QuadExpertChoice::Universal;
        let mut best = universal;
        if lattice < best {
            choice = QuadExpertChoice::Lattice;
            best = lattice;
        }
        if multivariate < best {
            choice = QuadExpertChoice::Multivariate;
            best = multivariate;
        }
        if common_mode < best {
            choice = QuadExpertChoice::CommonMode;
        }
        Ok(choice)
    }

    fn observe(
        &mut self,
        channel: usize,
        universal: i64,
        lattice: i64,
        multivariate: i64,
        common_mode: i64,
    ) -> Result<(), OptimumV2Error> {
        if channel >= self.universal_scores.len() {
            return Err(input_error(
                "MIX1 four-expert selector channel is out of range",
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
        self.common_mode_scores[channel] = update_score(
            self.common_mode_scores[channel],
            common_mode.unsigned_abs(),
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
