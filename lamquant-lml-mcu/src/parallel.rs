//! Host-only Rayon execution profile for the LML codec.
//!
//! Packet validation, preparation, wire assembly, and decode plans remain
//! crate-private. This module changes only how independent channels execute.

use alloc::vec::Vec;

use rayon::prelude::*;

use crate::error::LmlResult;
use crate::lml::{
    self, encode_one_channel, finalize_channels, parse_lml_channels, prepare_encode,
    synthesize_channel_signal, validate_and_levels, DecodePlan,
};
use crate::lpc::LpcMode;

/// Parallel LML encode. Output is byte-identical to
/// [`crate::lml::compress_with_mode`] for clock-free modes.
pub fn compress_with_mode_parallel(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let prep = prepare_encode(signal, noise_bits)?;
    let per_channel = (0..prep.n_ch)
        .into_par_iter()
        .map(|ch| {
            encode_one_channel(
                &prep.signal[ch],
                prep.n_levels,
                mode,
                prep.flags.0,
                prep.flags.1,
                prep.flags.2,
            )
        })
        .collect::<LmlResult<Vec<_>>>()?;
    let (lpc_meta, payload, wins) = finalize_channels(&per_channel);
    Ok(lml::assemble_lml_packet(
        prep.n_ch,
        prep.t,
        prep.n_levels,
        noise_bits,
        wins,
        &lpc_meta,
        &payload,
    ))
}

/// Parallel zero-copy encode for pre-sliced channel windows.
///
/// Live `LpcMode::Anytime` deadlines remain serial-only because independent
/// workers would sample the wall clock at different instants. Clock-free modes
/// are byte-identical to [`crate::lml::compress_with_mode_views`].
pub fn compress_with_mode_parallel_views(
    windows: &[&[i64]],
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let n_ch = windows.len();
    let t = windows.first().map(|window| window.len()).unwrap_or(0);
    let shape = validate_and_levels(n_ch, t, noise_bits)?;
    let per_channel = if noise_bits == 0 {
        windows
            .par_iter()
            .map(|&channel| {
                encode_one_channel(
                    channel,
                    shape.n_levels,
                    mode,
                    shape.flags.0,
                    shape.flags.1,
                    shape.flags.2,
                )
            })
            .collect::<LmlResult<Vec<_>>>()?
    } else {
        let shifted: Vec<Vec<i64>> = windows
            .iter()
            .map(|window| window.iter().map(|&value| value >> noise_bits).collect())
            .collect();
        shifted
            .par_iter()
            .map(|channel| {
                encode_one_channel(
                    channel,
                    shape.n_levels,
                    mode,
                    shape.flags.0,
                    shape.flags.1,
                    shape.flags.2,
                )
            })
            .collect::<LmlResult<Vec<_>>>()?
    };
    let (lpc_meta, payload, wins) = finalize_channels(&per_channel);
    Ok(lml::assemble_lml_packet(
        shape.n_ch,
        shape.t,
        shape.n_levels,
        noise_bits,
        wins,
        &lpc_meta,
        &payload,
    ))
}

/// Parallel LML decode: cursor-bound parsing stays serial; channel synthesis
/// and inverse lifting execute independently through Rayon.
pub fn decompress_parallel(data: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    match parse_lml_channels(data)? {
        DecodePlan::Done(signal) => Ok(signal),
        DecodePlan::Synthesize {
            n_levels,
            noise_bits,
            channels,
        } => {
            let mut signal = channels
                .into_par_iter()
                .map(|subbands| synthesize_channel_signal(subbands, n_levels))
                .collect::<LmlResult<Vec<_>>>()?;
            if noise_bits > 0 {
                for channel in &mut signal {
                    for value in channel {
                        *value <<= noise_bits;
                    }
                }
            }
            Ok(signal)
        }
    }
}
