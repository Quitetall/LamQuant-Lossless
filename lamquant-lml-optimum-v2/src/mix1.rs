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
use lamquant_lml_optimum::{Codec as LegacyCodec, LmoCodec, Mode as LegacyMode};
use std::collections::HashMap;

const HEADER_LEN: usize = 72;
const COMPACT_HEADER_LEN: usize = 40;
const ULTRA_COMPACT_HEADER_LEN: usize = 24;
const MAX_CHANNELS: usize = 256;
const MAX_SAMPLES: usize = 32_768;
const MAX_VALUES: usize = 131_072;
const MAX_EVENTS_PER_VALUE: usize = 129;
const WPX1_BLOCK_SIZES: [usize; 2] = [256, 512];

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

    pub fn encode_wavelet_override_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        block_size: usize,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        if !WPX1_BLOCK_SIZES.contains(&block_size) {
            return Err(input_error("WPX1 block size must be 256 or 512"));
        }
        let profile = Mix1TunedProfile {
            entropy: Mix1EntropyProfile {
                score_shift: 8,
                channel_context_mask: 7,
                history_context: 52,
                scale_profile: 4,
            },
            parent_history_depth: 2,
            parent_penalty: 6,
        };
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        let permutation = fit_channel_permutation(signal)?;
        let permuted = permutation
            .iter()
            .map(|&channel| signal[channel].clone())
            .collect::<Vec<_>>();
        let universal = universal_residuals(&permuted, bit_depth)?;
        let (side, lattice) =
            mix1_lattice::fit_and_analyze_with_parent_penalty(&permuted, profile.parent_penalty)?;
        let multivariate = multivariate_residuals_with_parent_history(
            &permuted,
            &side.parents,
            bit_depth,
            usize::from(profile.parent_history_depth),
        )?;
        let common_mode = common_mode_residuals(&permuted)?;
        let selected = select_four_residuals(
            &universal,
            &lattice,
            &multivariate,
            &common_mode,
            profile.entropy.score_shift,
        )?;
        let blocks_per_channel = samples / block_size;
        let map_bits = channels
            .checked_mul(blocks_per_channel)
            .ok_or_else(|| input_error("WPX1 block map size overflows"))?;
        let mut candidates = Vec::new();
        let local_parents = vec![Vec::new()];
        for channel in 0..channels {
            for block in 0..blocks_per_channel {
                let start = block * block_size;
                let end = start + block_size;
                let wavelet = wavelet53_forward(&permuted[channel][start..end])?;
                let residual_cost = mix1_entropy::encode_profile_channel_context(
                    &[selected[channel][start..end].to_vec()],
                    &local_parents,
                    profile.entropy.channel_context_mask,
                    profile.entropy.history_context,
                    profile.entropy.scale_profile,
                )?
                .0
                .len();
                let wavelet_cost = mix1_entropy::encode_profile_channel_context(
                    std::slice::from_ref(&wavelet),
                    &local_parents,
                    profile.entropy.channel_context_mask,
                    profile.entropy.history_context,
                    profile.entropy.scale_profile,
                )?
                .0
                .len();
                if wavelet_cost < residual_cost {
                    candidates.push((residual_cost - wavelet_cost, channel, block, wavelet));
                }
            }
        }
        candidates.sort_by_key(|(savings, channel, block, _)| {
            (core::cmp::Reverse(*savings), *channel, *block)
        });

        let mut block_map = vec![0u8; map_bits.div_ceil(8)];
        let mut coded = selected.clone();
        let (mut payload, mut event_count) = mix1_entropy::encode_profile_channel_context(
            &selected,
            &side.parents,
            profile.entropy.channel_context_mask,
            profile.entropy.history_context,
            profile.entropy.scale_profile,
        )?;
        for (_, channel, block, wavelet) in candidates {
            let start = block * block_size;
            let end = start + block_size;
            coded[channel][start..end].copy_from_slice(&wavelet);
            let trial = mix1_entropy::encode_profile_channel_context(
                &coded,
                &side.parents,
                profile.entropy.channel_context_mask,
                profile.entropy.history_context,
                profile.entropy.scale_profile,
            )?;
            if trial.0.len() < payload.len() {
                set_block_map(&mut block_map, channel * blocks_per_channel + block);
                payload = trial.0;
                event_count = trial.1;
            } else {
                coded[channel][start..end].copy_from_slice(&selected[channel][start..end]);
            }
        }
        let (mut graph, coefficient_rice_k, weight_rice_k) =
            mix1_lattice::pack_side_adaptive(&side, profile.entropy.score_shift)?;
        graph[..4].copy_from_slice(b"WPX1");
        let tail = graph.split_off(6);
        graph.extend_from_slice(&[
            profile.entropy.channel_context_mask,
            profile.entropy.history_context,
            profile.entropy.scale_profile,
            profile.parent_history_depth,
            coefficient_rice_k,
            weight_rice_k,
            block_size.trailing_zeros() as u8,
        ]);
        graph.extend_from_slice(&pack_permutation_indices(&permutation)?);
        graph.extend_from_slice(&block_map);
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

    pub fn encode_wavelet_split_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
        block_size: usize,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        if !WPX1_BLOCK_SIZES.contains(&block_size) {
            return Err(input_error("WSX1 block size must be 256 or 512"));
        }
        let profile = Mix1TunedProfile {
            entropy: Mix1EntropyProfile {
                score_shift: 8,
                channel_context_mask: 7,
                history_context: 52,
                scale_profile: 4,
            },
            parent_history_depth: 2,
            parent_penalty: 6,
        };
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        let permutation = fit_channel_permutation(signal)?;
        let permuted = permutation
            .iter()
            .map(|&channel| signal[channel].clone())
            .collect::<Vec<_>>();
        let universal = universal_residuals(&permuted, bit_depth)?;
        let (side, lattice) =
            mix1_lattice::fit_and_analyze_with_parent_penalty(&permuted, profile.parent_penalty)?;
        let multivariate = multivariate_residuals_with_parent_history(
            &permuted,
            &side.parents,
            bit_depth,
            usize::from(profile.parent_history_depth),
        )?;
        let common_mode = common_mode_residuals(&permuted)?;
        let selected = select_four_residuals(
            &universal,
            &lattice,
            &multivariate,
            &common_mode,
            profile.entropy.score_shift,
        )?;
        let blocks_per_channel = samples / block_size;
        let map_bits = channels
            .checked_mul(blocks_per_channel)
            .ok_or_else(|| input_error("WSX1 block map size overflows"))?;
        let local_parents = vec![Vec::new()];
        let mut candidates = Vec::new();
        for channel in 0..channels {
            for block in 0..blocks_per_channel {
                let start = block * block_size;
                let end = start + block_size;
                let wavelet = wavelet53_forward(&permuted[channel][start..end])?;
                let residual_cost = mix1_entropy::encode_profile_channel_context(
                    &[selected[channel][start..end].to_vec()],
                    &local_parents,
                    profile.entropy.channel_context_mask,
                    profile.entropy.history_context,
                    profile.entropy.scale_profile,
                )?
                .0
                .len();
                let wavelet_payload = mix1_entropy::encode_profile_channel_context(
                    &[wavelet],
                    &local_parents,
                    profile.entropy.channel_context_mask,
                    profile.entropy.history_context,
                    profile.entropy.scale_profile,
                )?
                .0;
                if wavelet_payload.len() + 2 < residual_cost {
                    candidates.push((
                        residual_cost - wavelet_payload.len() - 2,
                        channel,
                        block,
                        wavelet_payload,
                    ));
                }
            }
        }
        candidates.sort_by_key(|(savings, channel, block, _)| {
            (core::cmp::Reverse(*savings), *channel, *block)
        });

        let mut block_map = vec![0u8; map_bits.div_ceil(8)];
        let mut coded = selected.clone();
        let (mut main_payload, mut event_count) = mix1_entropy::encode_profile_channel_context(
            &coded,
            &side.parents,
            profile.entropy.channel_context_mask,
            profile.entropy.history_context,
            profile.entropy.scale_profile,
        )?;
        let mut block_payloads = vec![None; map_bits];
        let mut total_payload_len = 4usize
            .checked_add(main_payload.len())
            .ok_or_else(|| input_error("WSX1 payload length overflows"))?;
        for (_, channel, block, block_payload) in candidates {
            let start = block * block_size;
            let end = start + block_size;
            coded[channel][start..end].fill(0);
            let trial = mix1_entropy::encode_profile_channel_context(
                &coded,
                &side.parents,
                profile.entropy.channel_context_mask,
                profile.entropy.history_context,
                profile.entropy.scale_profile,
            )?;
            let trial_total = total_payload_len
                .checked_sub(main_payload.len())
                .and_then(|length| length.checked_add(trial.0.len()))
                .and_then(|length| length.checked_add(2 + block_payload.len()))
                .ok_or_else(|| input_error("WSX1 trial payload length overflows"))?;
            if trial_total < total_payload_len {
                let bit = channel * blocks_per_channel + block;
                set_block_map(&mut block_map, bit);
                block_payloads[bit] = Some(block_payload);
                main_payload = trial.0;
                event_count = trial.1;
                total_payload_len = trial_total;
            } else {
                coded[channel][start..end].copy_from_slice(&selected[channel][start..end]);
            }
        }
        let mut payload = Vec::with_capacity(total_payload_len);
        payload.extend_from_slice(
            &u32::try_from(main_payload.len())
                .map_err(|_| input_error("WSX1 main payload exceeds u32"))?
                .to_le_bytes(),
        );
        payload.extend_from_slice(&main_payload);
        for block_payload in block_payloads.into_iter().flatten() {
            payload.extend_from_slice(
                &u16::try_from(block_payload.len())
                    .map_err(|_| input_error("WSX1 block payload exceeds u16"))?
                    .to_le_bytes(),
            );
            payload.extend_from_slice(&block_payload);
        }
        debug_assert_eq!(payload.len(), total_payload_len);

        let (mut graph, coefficient_rice_k, weight_rice_k) =
            mix1_lattice::pack_side_adaptive(&side, profile.entropy.score_shift)?;
        graph[..4].copy_from_slice(b"WSX1");
        let tail = graph.split_off(6);
        graph.extend_from_slice(&[
            profile.entropy.channel_context_mask,
            profile.entropy.history_context,
            profile.entropy.scale_profile,
            profile.parent_history_depth,
            coefficient_rice_k,
            weight_rice_k,
            block_size.trailing_zeros() as u8,
        ]);
        graph.extend_from_slice(&pack_permutation_indices(&permutation)?);
        graph.extend_from_slice(&block_map);
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

    pub fn encode_legacy_optimum_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        let payload = LmoCodec
            .encode(signal, LegacyMode::Lossless)
            .map_err(|error| input_error(format!("LPX1 nested LMO encode failed: {error}")))?;
        let mut graph = Vec::with_capacity(6);
        graph.extend_from_slice(b"LPX1");
        graph.extend_from_slice(&[0xa7, 1]);
        let values = channels
            .checked_mul(samples)
            .ok_or_else(|| input_error("LPX1 value count overflows"))?;
        pack_frame_ultracompact(Frame {
            bit_depth,
            sample_rate_mhz,
            channels,
            samples,
            event_count: u32::try_from(values)
                .map_err(|_| input_error("LPX1 value count exceeds u32"))?,
            graph,
            payload,
            decoded_crc: crc32c(&canonical_i32_bytes(signal)?),
        })
    }

    pub fn encode_bitplane_layer_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        if bit_depth < 2 {
            return Err(input_error("BLX1 requires bit depth of at least two"));
        }
        let upper = signal
            .iter()
            .map(|channel| channel.iter().map(|sample| sample >> 1).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let nested = self.encode_best_peer_window_without_bitplane(
            &upper,
            sample_rate_mhz,
            bit_depth.saturating_sub(1),
        )?;
        let nested_len = u32::try_from(nested.len())
            .map_err(|_| input_error("BLX1 nested packet exceeds u32"))?;
        let modes = fit_low_bit_modes(signal)?;
        let low_bits = encode_low_bit_payload(signal, &modes)?;
        let mut payload = Vec::with_capacity(4 + nested.len() + low_bits.len());
        payload.extend_from_slice(&nested_len.to_le_bytes());
        payload.extend_from_slice(&nested);
        payload.extend_from_slice(&low_bits);
        let mut graph = b"BLX1\xa7\x02\x01".to_vec();
        graph.extend_from_slice(&pack_low_bit_modes(&modes)?);
        pack_frame_ultracompact(Frame {
            bit_depth,
            sample_rate_mhz,
            channels,
            samples,
            event_count: u32::try_from(channels * samples)
                .map_err(|_| input_error("BLX1 value count exceeds u32"))?,
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
        let mut best =
            self.encode_best_peer_window_without_bitplane(signal, sample_rate_mhz, bit_depth)?;
        if bit_depth >= 2 {
            let bitplane = self.encode_bitplane_layer_window(signal, sample_rate_mhz, bit_depth)?;
            if bitplane.len() < best.len() {
                best = bitplane;
            }
        }
        Ok(best)
    }

    fn encode_best_peer_window_without_bitplane(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let mut best =
            self.encode_best_peer_window_without_alias(signal, sample_rate_mhz, bit_depth)?;
        if let Some(alias) =
            self.encode_alias_window_optional(signal, sample_rate_mhz, bit_depth)?
        {
            if alias.len() < best.len() {
                best = alias;
            }
        }
        Ok(best)
    }

    pub fn encode_alias_window(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        self.encode_alias_window_optional(signal, sample_rate_mhz, bit_depth)?
            .ok_or_else(|| input_error("ALX1 requires at least one exact channel alias"))
    }

    fn encode_alias_window_optional(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Option<Vec<u8>>, OptimumV2Error> {
        let (channels, samples) = validate_signal(signal, sample_rate_mhz, bit_depth)?;
        let (representatives, aliases) = fit_channel_aliases(signal)?;
        if representatives.len() == channels {
            return Ok(None);
        }
        let unique = representatives
            .iter()
            .map(|&index| signal[index].clone())
            .collect::<Vec<_>>();
        let nested =
            self.encode_best_peer_window_without_legacy(&unique, sample_rate_mhz, bit_depth)?;
        let unique_count = u8::try_from(unique.len())
            .map_err(|_| input_error("ALX1 unique channel count exceeds u8"))?;
        let mut graph = Vec::with_capacity(7 + channels);
        graph.extend_from_slice(b"ALX1");
        graph.extend_from_slice(&[0xa7, 1, unique_count]);
        graph.extend_from_slice(&aliases);
        let values = channels
            .checked_mul(samples)
            .ok_or_else(|| input_error("ALX1 value count overflows"))?;
        Ok(Some(pack_frame_ultracompact(Frame {
            bit_depth,
            sample_rate_mhz,
            channels,
            samples,
            event_count: u32::try_from(values)
                .map_err(|_| input_error("ALX1 value count exceeds u32"))?,
            graph,
            payload: nested,
            decoded_crc: crc32c(&canonical_i32_bytes(signal)?),
        })?))
    }

    pub fn encode_best_peer_window_without_alias(
        &self,
        signal: &[Vec<i64>],
        sample_rate_mhz: u32,
        bit_depth: u8,
    ) -> Result<Vec<u8>, OptimumV2Error> {
        let incumbent =
            self.encode_best_peer_window_without_legacy(signal, sample_rate_mhz, bit_depth)?;
        let legacy = self.encode_legacy_optimum_window(signal, sample_rate_mhz, bit_depth)?;
        if legacy.len() < incumbent.len() && legacy_peer_is_independently_decodable(&legacy)? {
            Ok(legacy)
        } else {
            Ok(incumbent)
        }
    }

    fn encode_best_peer_window_without_legacy(
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
        if magic == Some(&b"ALX1"[..]) {
            return decode_alias_frame(frame);
        }
        if magic == Some(&b"BLX1"[..]) {
            return decode_bitplane_layer_frame(frame);
        }
        if magic == Some(&b"LPX1"[..]) {
            return decode_legacy_optimum_frame(frame);
        }
        let split_wavelet_mode = magic == Some(&b"WSX1"[..]);
        let wavelet_override_mode = magic == Some(&b"WPX1"[..]) || split_wavelet_mode;
        let hierarchical = magic == Some(&b"MCH1"[..]);
        let channel_context = magic == Some(&b"MCX1"[..]);
        let tuned_permuted = magic == Some(&b"APX1"[..]) || wavelet_override_mode;
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

        let wavelet_override = if wavelet_override_mode {
            let block_log2 = *graph
                .get(12)
                .ok_or_else(|| packet_error("WPX1 block size is truncated"))?;
            let block_size = 1usize
                .checked_shl(u32::from(block_log2))
                .ok_or_else(|| packet_error("WPX1 block size overflows"))?;
            if !WPX1_BLOCK_SIZES.contains(&block_size) {
                return Err(packet_error("WPX1 block size is invalid"));
            }
            let blocks_per_channel = frame.samples / block_size;
            let map_bits = frame
                .channels
                .checked_mul(blocks_per_channel)
                .ok_or_else(|| packet_error("WPX1 block map size overflows"))?;
            let permutation_len = packed_permutation_len(frame.channels)?;
            let map_start = 13usize
                .checked_add(permutation_len)
                .ok_or_else(|| packet_error("WPX1 block map offset overflows"))?;
            let map_end = map_start
                .checked_add(map_bits.div_ceil(8))
                .ok_or_else(|| packet_error("WPX1 block map length overflows"))?;
            let block_map = graph
                .get(map_start..map_end)
                .ok_or_else(|| packet_error("WPX1 block map is truncated"))?
                .to_vec();
            validate_block_map_padding(&block_map, map_bits)?;
            Some(WaveletOverride {
                block_size,
                blocks_per_channel,
                block_map,
            })
        } else {
            None
        };

        let permutation = if wavelet_override_mode {
            let start = 13usize;
            let end = start
                .checked_add(packed_permutation_len(frame.channels)?)
                .ok_or_else(|| packet_error("MIX peer permutation length overflows"))?;
            if graph.len() < end {
                return Err(packet_error("MIX peer permutation is truncated"));
            }
            let permutation = unpack_permutation_indices(&graph[start..end], frame.channels)?;
            let map_end = end
                .checked_add(
                    wavelet_override
                        .as_ref()
                        .expect("WPX1 metadata parsed")
                        .block_map
                        .len(),
                )
                .ok_or_else(|| packet_error("WPX1 block map end overflows"))?;
            graph.drain(12..map_end);
            Some(permutation)
        } else if tuned_permuted {
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

        let residuals = if split_wavelet_mode {
            let (history, scale, _, _, _) =
                tuned_profile.ok_or_else(|| packet_error("WSX1 tuned profile is missing"))?;
            decode_wavelet_split_payload(
                &frame.payload,
                (frame.channels, frame.samples),
                &side.parents,
                channel_context_mask
                    .ok_or_else(|| packet_error("WSX1 channel context is missing"))?,
                history,
                scale,
                wavelet_override
                    .as_ref()
                    .ok_or_else(|| packet_error("WSX1 block metadata is missing"))?,
            )?
        } else if let Some(mask) = channel_context_mask {
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
                if let Some(wavelet_override) = &wavelet_override {
                    decode_wavelet_override_samples(
                        &residuals,
                        score_shift,
                        &side,
                        &side.parents,
                        frame.bit_depth,
                        usize::from(parent_history_depth),
                        wavelet_override,
                    )?
                } else {
                    decode_common_mode_samples_with_parent_history(
                        &residuals,
                        score_shift,
                        &side,
                        &side.parents,
                        frame.bit_depth,
                        usize::from(parent_history_depth),
                    )?
                }
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

fn fit_low_bit_modes(signal: &[Vec<i64>]) -> Result<Vec<u8>, OptimumV2Error> {
    if signal.is_empty() || signal.iter().any(Vec::is_empty) {
        return Err(input_error("BLX1 low-bit signal is empty"));
    }
    signal
        .iter()
        .map(|samples| {
            let first = (samples[0] & 1) as u8;
            if samples.iter().all(|sample| (*sample & 1) as u8 == first) {
                return Ok(first);
            }
            let raw_bytes = samples.len().div_ceil(8);
            let mut run_bytes = 1usize;
            let mut current = first;
            let mut run = 1usize;
            for sample in &samples[1..] {
                let bit = (*sample & 1) as u8;
                if bit == current {
                    run += 1;
                } else {
                    run_bytes += uleb128_len(run);
                    current = bit;
                    run = 1;
                }
            }
            run_bytes += uleb128_len(run);
            Ok(if raw_bytes <= run_bytes { 2 } else { 3 })
        })
        .collect()
}

fn uleb128_len(mut value: usize) -> usize {
    debug_assert!(value != 0);
    let mut bytes = 1usize;
    while value >= 0x80 {
        bytes += 1;
        value >>= 7;
    }
    bytes
}

fn pack_low_bit_modes(modes: &[u8]) -> Result<Vec<u8>, OptimumV2Error> {
    if modes.is_empty() || modes.len() > MAX_CHANNELS || modes.iter().any(|&mode| mode > 3) {
        return Err(input_error("BLX1 low-bit modes are invalid"));
    }
    let mut packed = vec![0u8; (modes.len() * 2).div_ceil(8)];
    for (index, &mode) in modes.iter().enumerate() {
        packed[index / 4] |= mode << (6 - (index % 4) * 2);
    }
    Ok(packed)
}

fn unpack_low_bit_modes(packed: &[u8], channels: usize) -> Result<Vec<u8>, OptimumV2Error> {
    let expected = (channels * 2).div_ceil(8);
    if packed.len() != expected {
        return Err(packet_error("BLX1 low-bit mode-map length differs"));
    }
    if channels % 4 != 0 {
        let padding_bits = (4 - channels % 4) * 2;
        let padding_mask = (1u8 << padding_bits) - 1;
        if packed.last().is_some_and(|byte| byte & padding_mask != 0) {
            return Err(packet_error("BLX1 low-bit mode map has nonzero padding"));
        }
    }
    Ok((0..channels)
        .map(|index| (packed[index / 4] >> (6 - (index % 4) * 2)) & 3)
        .collect())
}

fn encode_low_bit_payload(signal: &[Vec<i64>], modes: &[u8]) -> Result<Vec<u8>, OptimumV2Error> {
    if signal.len() != modes.len() {
        return Err(input_error("BLX1 low-bit dimensions differ"));
    }
    let mut encoded = Vec::new();
    for (channel, &mode) in signal.iter().zip(modes) {
        match mode {
            0 | 1 => {
                if channel.iter().any(|sample| (*sample & 1) as u8 != mode) {
                    return Err(input_error("BLX1 constant low-bit mode disagrees"));
                }
            }
            2 => {
                for chunk in channel.chunks(8) {
                    let mut packed = 0u8;
                    for (index, sample) in chunk.iter().enumerate() {
                        packed |= ((*sample & 1) as u8) << (7 - index);
                    }
                    encoded.push(packed);
                }
            }
            3 => {
                let first = (channel
                    .first()
                    .ok_or_else(|| input_error("BLX1 low-bit channel is empty"))?
                    & 1) as u8;
                encoded.push(first);
                let mut current = first;
                let mut run = 1usize;
                for sample in &channel[1..] {
                    let bit = (*sample & 1) as u8;
                    if bit == current {
                        run = run
                            .checked_add(1)
                            .ok_or_else(|| input_error("BLX1 low-bit run length overflows"))?;
                    } else {
                        encode_uleb128(run, &mut encoded);
                        current = bit;
                        run = 1;
                    }
                }
                encode_uleb128(run, &mut encoded);
            }
            _ => return Err(input_error("BLX1 low-bit mode is invalid")),
        }
    }
    Ok(encoded)
}

fn encode_uleb128(mut value: usize, output: &mut Vec<u8>) {
    debug_assert!(value != 0);
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn decode_low_bit_run_channel(
    bytes: &[u8],
    expected: usize,
) -> Result<(Vec<u8>, usize), OptimumV2Error> {
    let (&first, mut tail) = bytes
        .split_first()
        .ok_or_else(|| packet_error("BLX1 low-bit run channel is truncated"))?;
    if first > 1 {
        return Err(packet_error("BLX1 first low bit is invalid"));
    }
    let mut decoded = Vec::with_capacity(expected);
    let mut bit = first;
    while decoded.len() < expected {
        let original = tail;
        let mut value = 0usize;
        let mut shift = 0u32;
        loop {
            let (&byte, rest) = tail
                .split_first()
                .ok_or_else(|| packet_error("BLX1 low-bit run is truncated"))?;
            tail = rest;
            if shift >= usize::BITS || usize::from(byte & 0x7f) > (usize::MAX >> shift) {
                return Err(packet_error("BLX1 low-bit run length overflows"));
            }
            value |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        if value == 0 {
            return Err(packet_error("BLX1 low-bit run is empty"));
        }
        let consumed = original.len() - tail.len();
        let mut canonical = Vec::new();
        encode_uleb128(value, &mut canonical);
        if canonical.as_slice() != &original[..consumed] {
            return Err(packet_error("BLX1 low-bit run is noncanonical"));
        }
        let end = decoded
            .len()
            .checked_add(value)
            .ok_or_else(|| packet_error("BLX1 low-bit run end overflows"))?;
        if end > expected {
            return Err(packet_error("BLX1 low-bit run exceeds signal shape"));
        }
        decoded.resize(end, bit);
        bit ^= 1;
    }
    Ok((decoded, bytes.len() - tail.len()))
}

fn decode_low_bit_payload(
    bytes: &[u8],
    modes: &[u8],
    samples: usize,
) -> Result<Vec<Vec<u8>>, OptimumV2Error> {
    let mut tail = bytes;
    let mut decoded = Vec::with_capacity(modes.len());
    for &mode in modes {
        match mode {
            0 | 1 => decoded.push(vec![mode; samples]),
            2 => {
                let packed_len = samples.div_ceil(8);
                let (packed, rest) = tail
                    .split_at_checked(packed_len)
                    .ok_or_else(|| packet_error("BLX1 raw low-bit channel is truncated"))?;
                if samples % 8 != 0 {
                    let padding = 8 - samples % 8;
                    if packed
                        .last()
                        .is_some_and(|byte| byte & ((1 << padding) - 1) != 0)
                    {
                        return Err(packet_error("BLX1 raw low-bit channel has nonzero padding"));
                    }
                }
                decoded.push(
                    (0..samples)
                        .map(|index| (packed[index / 8] >> (7 - index % 8)) & 1)
                        .collect(),
                );
                tail = rest;
            }
            3 => {
                let (channel, consumed) = decode_low_bit_run_channel(tail, samples)?;
                decoded.push(channel);
                tail = &tail[consumed..];
            }
            _ => return Err(packet_error("BLX1 low-bit mode is invalid")),
        }
    }
    if !tail.is_empty() {
        return Err(packet_error("BLX1 low-bit payload has trailing bytes"));
    }
    Ok(decoded)
}

fn decode_bitplane_layer_frame(frame: Frame) -> Result<Mix1Decoded, OptimumV2Error> {
    let mode_len = (frame.channels * 2).div_ceil(8);
    if frame.graph.len() != 7 + mode_len
        || frame.graph.get(..7) != Some(&b"BLX1\xa7\x02\x01"[..])
        || frame.bit_depth < 2
    {
        return Err(packet_error("BLX1 side data is invalid or noncanonical"));
    }
    let modes = unpack_low_bit_modes(&frame.graph[7..], frame.channels)?;
    let nested_len = usize::try_from(read_u32(&frame.payload, 0)?)
        .map_err(|_| packet_error("BLX1 nested packet length exceeds usize"))?;
    let nested_end = 4usize
        .checked_add(nested_len)
        .ok_or_else(|| packet_error("BLX1 nested packet end overflows"))?;
    let nested_packet = frame
        .payload
        .get(4..nested_end)
        .ok_or_else(|| packet_error("BLX1 nested packet is truncated"))?;
    let nested_frame = unpack_frame(nested_packet)?;
    if nested_frame.graph.get(..4) == Some(&b"BLX1"[..]) {
        return Err(packet_error("BLX1 nesting depth exceeds one"));
    }
    let nested = Mix1Codec.decode_window(nested_packet)?;
    if nested.samples.len() != frame.channels
        || nested
            .samples
            .iter()
            .any(|channel| channel.len() != frame.samples)
        || nested.sample_rate_mhz != frame.sample_rate_mhz
        || nested.bit_depth != frame.bit_depth - 1
    {
        return Err(packet_error(
            "BLX1 nested peer dimensions or metadata disagree",
        ));
    }
    let low_bits = decode_low_bit_payload(
        frame
            .payload
            .get(nested_end..)
            .ok_or_else(|| packet_error("BLX1 low-bit stream is truncated"))?,
        &modes,
        frame.samples,
    )?;
    let magnitude = 1i64 << (frame.bit_depth - 1);
    let minimum = -magnitude;
    let maximum = magnitude - 1;
    let score_shift = nested.score_shift;
    let mut samples = nested.samples;
    for channel in 0..frame.channels {
        for time in 0..frame.samples {
            let sample = &mut samples[channel][time];
            *sample = sample
                .checked_mul(2)
                .and_then(|upper| upper.checked_add(i64::from(low_bits[channel][time])))
                .ok_or_else(|| packet_error("BLX1 reconstructed sample exceeds i64"))?;
            if !(minimum..=maximum).contains(sample) {
                return Err(packet_error(
                    "BLX1 reconstructed sample exceeds declared bit depth",
                ));
            }
        }
    }
    if fit_low_bit_modes(&samples).map_err(as_packet_error)? != modes {
        return Err(packet_error("BLX1 low-bit mode map is noncanonical"));
    }
    if crc32c(&canonical_i32_bytes(&samples).map_err(as_packet_error)?) != frame.decoded_crc {
        return Err(OptimumV2Error::Integrity(
            "BLX1 decoded sample CRC32C mismatch".into(),
        ));
    }
    Ok(Mix1Decoded {
        samples,
        sample_rate_mhz: frame.sample_rate_mhz,
        bit_depth: frame.bit_depth,
        score_shift,
    })
}

fn fit_channel_aliases(signal: &[Vec<i64>]) -> Result<(Vec<usize>, Vec<u8>), OptimumV2Error> {
    let mut lookup: HashMap<&[i64], u8> = HashMap::new();
    let mut representatives = Vec::new();
    let mut aliases = Vec::with_capacity(signal.len());
    for (channel_index, channel) in signal.iter().enumerate() {
        if let Some(&index) = lookup.get(channel.as_slice()) {
            aliases.push(index);
            continue;
        }
        let index = u8::try_from(representatives.len())
            .map_err(|_| input_error("ALX1 unique channel index exceeds u8"))?;
        lookup.insert(channel.as_slice(), index);
        representatives.push(channel_index);
        aliases.push(index);
    }
    Ok((representatives, aliases))
}

fn decode_legacy_optimum_frame(frame: Frame) -> Result<Mix1Decoded, OptimumV2Error> {
    if frame.graph != b"LPX1\xa7\x01" {
        return Err(packet_error("LPX1 side data is invalid or noncanonical"));
    }
    let samples = LmoCodec
        .decode(&frame.payload)
        .map_err(|error| packet_error(format!("LPX1 nested LMO decode failed: {error}")))?;
    let (channels, sample_count) =
        validate_signal(&samples, frame.sample_rate_mhz, frame.bit_depth)
            .map_err(as_packet_error)?;
    if channels != frame.channels || sample_count != frame.samples {
        return Err(packet_error("LPX1 nested LMO dimensions disagree"));
    }
    let canonical = LmoCodec
        .encode(&samples, LegacyMode::Lossless)
        .map_err(|error| packet_error(format!("LPX1 nested LMO re-encode failed: {error}")))?;
    if canonical != frame.payload {
        return Err(packet_error("LPX1 nested LMO packet is noncanonical"));
    }
    if crc32c(&canonical_i32_bytes(&samples).map_err(as_packet_error)?) != frame.decoded_crc {
        return Err(OptimumV2Error::Integrity(
            "LPX1 decoded sample CRC32C mismatch".into(),
        ));
    }
    Ok(Mix1Decoded {
        samples,
        sample_rate_mhz: frame.sample_rate_mhz,
        bit_depth: frame.bit_depth,
        score_shift: 0,
    })
}

fn legacy_peer_is_independently_decodable(packet: &[u8]) -> Result<bool, OptimumV2Error> {
    let frame = unpack_frame(packet)?;
    let payload = frame.payload;
    if payload.get(..7) != Some(&b"LMO1\x02\x00\x02"[..]) {
        return Ok(false);
    }
    let body = &payload[7..];
    let base = if body.first() == Some(&0xfe) {
        if body.len() < 12 || body[1] == 0 || body[1] > 16 {
            return Ok(false);
        }
        let upper_len = usize::try_from(read_u32(body, 8)?)
            .map_err(|_| packet_error("LPX1 upper body length exceeds usize"))?;
        let upper_end = 12usize
            .checked_add(upper_len)
            .ok_or_else(|| packet_error("LPX1 upper body end overflows"))?;
        match body.get(12..upper_end) {
            Some(base) => base,
            None => return Ok(false),
        }
    } else {
        body
    };
    if base.len() < 5 || base[0] != 3 {
        return Ok(false);
    }
    let channels = usize::from(u16::from_le_bytes([base[2], base[3]]));
    let mut position = 4usize;
    for _ in 0..channels {
        let Some(&references) = base.get(position) else {
            return Ok(false);
        };
        position = match position
            .checked_add(1)
            .and_then(|value| value.checked_add(usize::from(references) * 6))
        {
            Some(position) if position <= base.len() => position,
            _ => return Ok(false),
        };
    }
    Ok(base
        .get(position)
        .is_some_and(|mode| (1..=4).contains(mode)))
}

fn decode_alias_frame(frame: Frame) -> Result<Mix1Decoded, OptimumV2Error> {
    let expected_graph_len = 7usize
        .checked_add(frame.channels)
        .ok_or_else(|| packet_error("ALX1 map length overflows"))?;
    if frame.graph.len() != expected_graph_len || frame.graph[4] != 0xa7 || frame.graph[5] != 1 {
        return Err(packet_error("ALX1 side data is invalid or noncanonical"));
    }
    let unique_count = usize::from(frame.graph[6]);
    if unique_count == 0 || unique_count >= frame.channels {
        return Err(packet_error("ALX1 unique channel count is invalid"));
    }
    let aliases = &frame.graph[7..];
    let mut seen = vec![false; unique_count];
    let mut next = 0usize;
    for &alias in aliases {
        let alias = usize::from(alias);
        if alias >= unique_count {
            return Err(packet_error("ALX1 alias index is out of range"));
        }
        if !seen[alias] {
            if alias != next {
                return Err(packet_error(
                    "ALX1 representatives are not first-occurrence ordered",
                ));
            }
            seen[alias] = true;
            next += 1;
        }
    }
    if next != unique_count {
        return Err(packet_error("ALX1 contains an unused representative"));
    }

    let nested_frame = unpack_frame(&frame.payload)?;
    if nested_frame.graph.get(..4) == Some(&b"ALX1"[..]) {
        return Err(packet_error("ALX1 nesting depth exceeds one"));
    }
    let nested = Mix1Codec.decode_window(&frame.payload)?;
    if nested.samples.len() != unique_count
        || nested
            .samples
            .iter()
            .any(|channel| channel.len() != frame.samples)
        || nested.sample_rate_mhz != frame.sample_rate_mhz
        || nested.bit_depth != frame.bit_depth
    {
        return Err(packet_error(
            "ALX1 nested peer dimensions or metadata disagree",
        ));
    }
    let samples = aliases
        .iter()
        .map(|&alias| nested.samples[usize::from(alias)].clone())
        .collect::<Vec<_>>();
    let (canonical_representatives, canonical_aliases) =
        fit_channel_aliases(&samples).map_err(as_packet_error)?;
    if canonical_aliases != aliases
        || canonical_representatives.len() != nested.samples.len()
        || canonical_representatives
            .iter()
            .zip(&nested.samples)
            .any(|(&representative, unique)| {
                samples[representative].as_slice() != unique.as_slice()
            })
    {
        return Err(packet_error("ALX1 alias partition is noncanonical"));
    }
    if crc32c(&canonical_i32_bytes(&samples).map_err(as_packet_error)?) != frame.decoded_crc {
        return Err(OptimumV2Error::Integrity(
            "ALX1 decoded sample CRC32C mismatch".into(),
        ));
    }
    Ok(Mix1Decoded {
        samples,
        sample_rate_mhz: frame.sample_rate_mhz,
        bit_depth: frame.bit_depth,
        score_shift: nested.score_shift,
    })
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

#[derive(Debug, Clone)]
struct WaveletOverride {
    block_size: usize,
    blocks_per_channel: usize,
    block_map: Vec<u8>,
}

impl WaveletOverride {
    fn selected(&self, channel: usize, time: usize) -> bool {
        let block = time / self.block_size;
        block < self.blocks_per_channel
            && block_map_is_set(&self.block_map, channel * self.blocks_per_channel + block)
    }
}

fn set_block_map(block_map: &mut [u8], bit: usize) {
    block_map[bit / 8] |= 1 << (7 - bit % 8);
}

fn block_map_is_set(block_map: &[u8], bit: usize) -> bool {
    block_map[bit / 8] & (1 << (7 - bit % 8)) != 0
}

fn validate_block_map_padding(block_map: &[u8], bits: usize) -> Result<(), OptimumV2Error> {
    if block_map.len() != bits.div_ceil(8) {
        return Err(packet_error("WPX1 block map length differs"));
    }
    if bits % 8 != 0 {
        let padding_mask = (1u8 << (8 - bits % 8)) - 1;
        if block_map
            .last()
            .is_some_and(|last| last & padding_mask != 0)
        {
            return Err(packet_error("WPX1 block map has nonzero padding"));
        }
    }
    Ok(())
}

fn wavelet53_forward(block: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
    if !WPX1_BLOCK_SIZES.contains(&block.len()) {
        return Err(input_error("WPX1 wavelet block length is invalid"));
    }
    let mut approximation = block.to_vec();
    let mut details = Vec::new();
    while approximation.len() >= 8 {
        let half = approximation.len() / 2;
        let mut detail = Vec::with_capacity(half);
        for index in 0..half {
            let left = i128::from(approximation[index * 2]);
            let right = i128::from(
                approximation
                    .get(index * 2 + 2)
                    .copied()
                    .unwrap_or(approximation[index * 2]),
            );
            let prediction = (left + right).div_euclid(2);
            let value = i128::from(approximation[index * 2 + 1]) - prediction;
            detail.push(
                i64::try_from(value)
                    .map_err(|_| arithmetic_error("WPX1 forward wavelet detail exceeds i64"))?,
            );
        }
        let mut next = Vec::with_capacity(half);
        for index in 0..half {
            let left_detail = i128::from(if index == 0 {
                detail[0]
            } else {
                detail[index - 1]
            });
            let right_detail = i128::from(detail[index]);
            let update = (left_detail + right_detail + 2).div_euclid(4);
            let value = i128::from(approximation[index * 2]) + update;
            next.push(
                i64::try_from(value).map_err(|_| {
                    arithmetic_error("WPX1 forward wavelet approximation exceeds i64")
                })?,
            );
        }
        details.push(detail);
        approximation = next;
    }
    let mut transformed = Vec::with_capacity(block.len());
    transformed.extend_from_slice(&approximation);
    for detail in details.iter().rev() {
        transformed.extend_from_slice(detail);
    }
    debug_assert_eq!(transformed.len(), block.len());
    Ok(transformed)
}

fn wavelet53_inverse(coefficients: &[i64]) -> Result<Vec<i64>, OptimumV2Error> {
    if !WPX1_BLOCK_SIZES.contains(&coefficients.len()) {
        return Err(packet_error("WPX1 wavelet coefficient length is invalid"));
    }
    let levels = coefficients.len().trailing_zeros() as usize - 2;
    let coarsest = coefficients.len() >> levels;
    let mut approximation = coefficients[..coarsest].to_vec();
    let mut offset = coarsest;
    for _ in 0..levels {
        let detail_len = approximation.len();
        let end = offset
            .checked_add(detail_len)
            .ok_or_else(|| packet_error("WPX1 wavelet detail offset overflows"))?;
        let detail = coefficients
            .get(offset..end)
            .ok_or_else(|| packet_error("WPX1 wavelet detail is truncated"))?;
        let mut even = Vec::with_capacity(detail_len);
        for index in 0..detail_len {
            let left_detail = i128::from(if index == 0 {
                detail[0]
            } else {
                detail[index - 1]
            });
            let right_detail = i128::from(detail[index]);
            let update = (left_detail + right_detail + 2).div_euclid(4);
            let value = i128::from(approximation[index]) - update;
            even.push(
                i64::try_from(value)
                    .map_err(|_| packet_error("WPX1 inverse wavelet even sample exceeds i64"))?,
            );
        }
        let mut restored = Vec::with_capacity(detail_len * 2);
        for index in 0..detail_len {
            let left = i128::from(even[index]);
            let right = i128::from(even.get(index + 1).copied().unwrap_or(even[index]));
            let prediction = (left + right).div_euclid(2);
            let odd = i128::from(detail[index]) + prediction;
            restored.push(even[index]);
            restored.push(
                i64::try_from(odd)
                    .map_err(|_| packet_error("WPX1 inverse wavelet odd sample exceeds i64"))?,
            );
        }
        approximation = restored;
        offset = end;
    }
    if offset != coefficients.len() || approximation.len() != coefficients.len() {
        return Err(packet_error("WPX1 wavelet layout is noncanonical"));
    }
    Ok(approximation)
}

fn decode_wavelet_split_payload(
    payload: &[u8],
    dimensions: (usize, usize),
    parents: &[Vec<usize>],
    channel_context_mask: u8,
    history_context: u8,
    scale_profile: u8,
    wavelet_override: &WaveletOverride,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    let main_len = usize::try_from(read_u32(payload, 0)?)
        .map_err(|_| packet_error("WSX1 main payload length exceeds usize"))?;
    let main_end = 4usize
        .checked_add(main_len)
        .ok_or_else(|| packet_error("WSX1 main payload end overflows"))?;
    let main_payload = payload
        .get(4..main_end)
        .ok_or_else(|| packet_error("WSX1 main payload is truncated"))?;
    let mut residuals = mix1_entropy::decode_profile_channel_context(
        main_payload,
        0,
        dimensions,
        parents,
        channel_context_mask,
        history_context,
        scale_profile,
    )?;
    if wavelet_override.blocks_per_channel != dimensions.1 / wavelet_override.block_size {
        return Err(packet_error("WSX1 block geometry differs"));
    }
    let local_parents = vec![Vec::new()];
    let mut cursor = main_end;
    for (channel, residual_channel) in residuals.iter_mut().enumerate().take(dimensions.0) {
        for block in 0..wavelet_override.blocks_per_channel {
            let bit = channel * wavelet_override.blocks_per_channel + block;
            if !block_map_is_set(&wavelet_override.block_map, bit) {
                continue;
            }
            let start = block * wavelet_override.block_size;
            let end = start + wavelet_override.block_size;
            if residual_channel[start..end].iter().any(|&value| value != 0) {
                return Err(packet_error(
                    "WSX1 selected block has nonzero main-stream data",
                ));
            }
            let length_end = cursor
                .checked_add(2)
                .ok_or_else(|| packet_error("WSX1 block length offset overflows"))?;
            let length_bytes = payload
                .get(cursor..length_end)
                .ok_or_else(|| packet_error("WSX1 block length is truncated"))?;
            let block_len = usize::from(u16::from_le_bytes(
                length_bytes.try_into().expect("two bytes"),
            ));
            let block_end = length_end
                .checked_add(block_len)
                .ok_or_else(|| packet_error("WSX1 block payload end overflows"))?;
            let block_payload = payload
                .get(length_end..block_end)
                .ok_or_else(|| packet_error("WSX1 block payload is truncated"))?;
            let block_coefficients = mix1_entropy::decode_profile_channel_context(
                block_payload,
                0,
                (1, wavelet_override.block_size),
                &local_parents,
                channel_context_mask,
                history_context,
                scale_profile,
            )?;
            residual_channel[start..end].copy_from_slice(&block_coefficients[0]);
            cursor = block_end;
        }
    }
    if cursor != payload.len() {
        return Err(packet_error("WSX1 payload has trailing bytes"));
    }
    Ok(residuals)
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
        None,
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
        None,
    )
}

fn decode_wavelet_override_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
    parent_history_depth: usize,
    wavelet_override: &WaveletOverride,
) -> Result<Vec<Vec<i64>>, OptimumV2Error> {
    decode_four_mode_samples(
        residuals,
        score_shift,
        side,
        multivariate_parents,
        bit_depth,
        parent_history_depth,
        Some(wavelet_override),
    )
}

fn decode_four_mode_samples(
    residuals: &[Vec<i64>],
    score_shift: u8,
    side: &LatticeSide,
    multivariate_parents: &[Vec<usize>],
    bit_depth: u8,
    parent_history_depth: usize,
    wavelet_override: Option<&WaveletOverride>,
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
    let override_samples = if let Some(wavelet_override) = wavelet_override {
        if wavelet_override.blocks_per_channel != samples / wavelet_override.block_size {
            return Err(packet_error("WPX1 block geometry differs"));
        }
        let mut override_samples = residuals.to_vec();
        for channel in 0..channels {
            for block in 0..wavelet_override.blocks_per_channel {
                let bit = channel * wavelet_override.blocks_per_channel + block;
                if !block_map_is_set(&wavelet_override.block_map, bit) {
                    continue;
                }
                let start = block * wavelet_override.block_size;
                let end = start + wavelet_override.block_size;
                let restored = wavelet53_inverse(&residuals[channel][start..end])?;
                override_samples[channel][start..end].copy_from_slice(&restored);
            }
        }
        Some(override_samples)
    } else {
        None
    };
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
            let overridden = wavelet_override
                .is_some_and(|wavelet_override| wavelet_override.selected(channel, time));
            let sample = if overridden {
                override_samples
                    .as_ref()
                    .expect("WPX1 override samples reconstructed")[channel][time]
            } else {
                match choice {
                    QuadExpertChoice::Universal => universal_prediction
                        .checked_add(coded)
                        .ok_or_else(|| packet_error("MIX1 universal reconstruction exceeds i64"))?,
                    QuadExpertChoice::Multivariate => {
                        multivariate_prediction.checked_add(coded).ok_or_else(|| {
                            packet_error("MIX1 multivariate reconstruction exceeds i64")
                        })?
                    }
                    QuadExpertChoice::CommonMode => {
                        fourth_prediction.checked_add(coded).ok_or_else(|| {
                            packet_error("MIX1 fourth-mode reconstruction exceeds i64")
                        })?
                    }
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
            if !overridden && selected != coded {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_bit_modes_choose_constant_raw_and_run_encodings() {
        let constant_zero = vec![0; 32];
        let constant_one = vec![1; 32];
        let raw = (0..32)
            .map(|index| i64::from(matches!(index % 5, 1 | 2)))
            .collect::<Vec<_>>();
        let runs = (0..32)
            .map(|index| i64::from(index >= 16))
            .collect::<Vec<_>>();

        assert_eq!(
            fit_low_bit_modes(&[constant_zero, constant_one, raw, runs]).unwrap(),
            vec![0, 1, 2, 3]
        );
    }
}
