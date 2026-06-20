//! The Desktop fast path: rayon-parallel encode/decode orchestrators built over
//! the MCU tier's codec primitives (ADR 0058 carve-full).
//!
//! These are byte-identical to the serial MCU path (`lamquant_lml_mcu::lml::
//! compress_with_mode` / `decompress`) **by construction** — they call the exact
//! same primitives (`prepare_encode`, `encode_one_channel`, `finalize_channels`,
//! `assemble_lml_packet` / `parse_lml_channels`, `synthesize_channel_signal`);
//! only the per-channel loop runs across rayon workers instead of serially. The
//! `byte_equal_backends` gate (now in this crate's tests) locks the invariant.

use rayon::prelude::*;

use lamquant_lml_mcu::error::LmlResult;
use lamquant_lml_mcu::lml::{
    self, encode_one_channel, finalize_channels, parse_lml_channels, prepare_encode,
    synthesize_channel_signal, DecodePlan,
};
use lamquant_lml_mcu::lpc::LpcMode;

/// Parallel LML encode (rayon per-channel). Byte-identical output to
/// [`lamquant_lml_mcu::lml::compress_with_mode`].
pub fn compress_with_mode_parallel(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let prep = prepare_encode(signal, noise_bits)?;
    // Parallel per-channel encode. `into_par_iter().map(...).collect()` preserves
    // input order, so the concatenated bytes match the serial path exactly.
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

/// Parallel LML decode: serial parse (cursor-bound) + rayon per-channel synth.
/// Byte-identical output to [`lamquant_lml_mcu::lml::decompress`].
pub fn decompress_parallel(data: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    match parse_lml_channels(data)? {
        DecodePlan::Done(signal) => Ok(signal),
        DecodePlan::Synthesize {
            n_levels,
            noise_bits,
            channels,
        } => {
            let mut signal: Vec<Vec<i64>> = channels
                .into_par_iter()
                .map(|subs| synthesize_channel_signal(subs, n_levels))
                .collect::<LmlResult<Vec<_>>>()?;
            if noise_bits > 0 {
                for ch in signal.iter_mut() {
                    for v in ch.iter_mut() {
                        *v <<= noise_bits;
                    }
                }
            }
            Ok(signal)
        }
    }
}
