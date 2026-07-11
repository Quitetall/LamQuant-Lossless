//! The Desktop fast path: rayon-parallel encode/decode orchestrators built over
//! the MCU tier's codec primitives (ADR 0058 carve-full).
//!
//! These are byte-identical to the serial MCU path (`lamquant_lml_mcu::lml::
//! compress_with_mode` / `decompress`) **by construction** — they call the exact
//! same primitives (`prepare_encode`, `encode_one_channel`, `finalize_channels`,
//! `assemble_lml_packet` / `parse_lml_channels`, `synthesize_channel_signal`);
//! only the per-channel loop runs across rayon workers instead of serially. The
//! `byte_equal_backends` gate (now in this crate's tests) locks the invariant.
//!
//! Exception: `LpcMode::Anytime{deadline: Some(_)}` (a live wall-clock
//! deadline) is NOT covered by the "by construction" claim above — see the
//! task #32 caveat on [`compress_with_mode_parallel_views`] for why, and
//! `abir_container::write_abir` (in `lamquant-lossless`) for the caller-side
//! routing that works around it today.

use rayon::prelude::*;

use lamquant_lml_mcu::error::LmlResult;
use lamquant_lml_mcu::lml::{
    self, encode_one_channel, finalize_channels, parse_lml_channels, prepare_encode,
    synthesize_channel_signal, validate_and_levels, DecodePlan,
};
use lamquant_lml_mcu::lpc::LpcMode;

/// Assemble one LML packet at a fixed `n_levels` via rayon-parallel per-channel encode. Byte-identical
/// to the MCU serial `encode_channels_core` at the same `n_levels` (same primitives, order-preserving
/// `par_iter`). The keep-best over `{full, skip}` layered on top ([`keep_best_levels_parallel`]) mirrors
/// the MCU tier's `encode_maybe_skip`, so both tiers pick the same packet with the transform-skip flag
/// ON or OFF.
fn assemble_at_levels_parallel(
    channels: &[&[i64]],
    n_ch: usize,
    t: usize,
    n_levels: u8,
    noise_bits: u8,
    flags: (bool, bool, bool),
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let per_channel = channels
        .par_iter()
        .map(|&ch| encode_one_channel(ch, n_levels, mode, flags.0, flags.1, flags.2))
        .collect::<LmlResult<Vec<_>>>()?;
    let (lpc_meta, payload, wins) = finalize_channels(&per_channel);
    Ok(lml::assemble_lml_packet(n_ch, t, n_levels, noise_bits, wins, &lpc_meta, &payload))
}

/// Adaptive transform-skip keep-best (parallel mirror of `lml::encode_maybe_skip`): encode at
/// `full_levels`, and when the flag is on and the transform is in use, ALSO at `n_levels = 0`, keeping
/// the smaller. Deterministic length compare ⇒ byte-identical to the serial MCU path either way.
fn keep_best_levels_parallel(
    channels: &[&[i64]],
    n_ch: usize,
    t: usize,
    full_levels: u8,
    noise_bits: u8,
    flags: (bool, bool, bool),
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let full = assemble_at_levels_parallel(channels, n_ch, t, full_levels, noise_bits, flags, mode)?;
    if lml::transform_skip_enabled() && full_levels > 0 {
        let skip = assemble_at_levels_parallel(channels, n_ch, t, 0, noise_bits, flags, mode)?;
        if skip.len() < full.len() {
            return Ok(skip);
        }
    }
    Ok(full)
}

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
    let views: Vec<&[i64]> = prep.signal.iter().map(|v| v.as_slice()).collect();
    keep_best_levels_parallel(
        &views,
        prep.n_ch,
        prep.t,
        prep.n_levels,
        noise_bits,
        prep.flags,
        mode,
    )
}

/// Parallel zero-copy LML encode: `windows` are already-sliced `&[i64]`
/// views (e.g. `abir::Abir::window_views` — this crate doesn't
/// depend on `abir` directly, so that's plain text, not a doc
/// link) — no per-window `Vec<Vec<i64>>` materialization (ADR 0069 L6.3). Mirrors
/// [`compress_with_mode_parallel`]'s rayon-per-channel split over the SAME
/// primitives (`validate_and_levels`, `encode_one_channel`,
/// `finalize_channels`, `assemble_lml_packet`), so it is byte-identical to
/// [`lamquant_lml_mcu::lml::compress_with_mode_views`] (the serial views
/// entry point) and to [`lamquant_lml_mcu::lml::compress_with_mode`] for the
/// same logical input — only the per-channel loop runs across rayon workers
/// instead of serially. Locked by the `views == vecs` extension of
/// `byte_equal_backends.rs`.
///
/// **Caveat (task #32):** the byte-identity claim above holds for every
/// CLOCK-FREE `mode` (`Fixed`, `Adaptive`, `Anytime{deadline: None}`) —
/// which is everything `byte_equal_backends.rs`'s `GOLDEN_VECTORS` exercise
/// today. It does NOT hold for `LpcMode::Anytime{deadline: Some(_)}` (a
/// LIVE wall-clock deadline): `encode_one_channel`'s inner
/// `analyze_anytime_host` re-reads `Instant::now()` per subband, and this
/// function's rayon workers each sample that clock at their own
/// independent schedule time — a different "time remains" decision per
/// subband than the serial caller's monotonic single-thread read, and
/// potentially different run-to-run on this SAME function. Callers with a
/// live deadline must NOT rely on this function agreeing byte-for-byte
/// with the serial path; `abir_container::write_abir`'s `ComputeBackend::
/// Desktop` arm accounts for this by routing `Anytime{deadline: Some(_)}`
/// to the serial `compress_with_mode_views` instead of calling this
/// function at all. The full fix — thread an explicit per-channel
/// `time_remaining` signal through this kernel so it matches serial even
/// WITH a live deadline — is the tracked follow-up, deliberately not done
/// here (minimal safe close, not the kernel refactor).
///
/// `noise_bits == 0` (hot, lossless — the only mode `write_abir` dispatches
/// today): the rayon closure borrows directly from `windows`, so this is
/// TRUE zero-copy. `noise_bits > 0` (cold): pre-shift each channel into an
/// owned `Vec<Vec<i64>>` (`v >> noise_bits`, an unavoidable copy — the shift
/// produces new values) and rayon-map over THOSE borrows, still passing the
/// *original* `noise_bits` to `assemble_lml_packet` so the wire header
/// matches what the decoder needs to left-shift back (same reasoning as
/// `compress_with_mode_views`'s cold path — do NOT delegate to
/// `compress_with_mode_parallel(&shifted, 0, mode)`, which would write a
/// wrong `noise_bits=0` header field).
pub fn compress_with_mode_parallel_views(
    windows: &[&[i64]],
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let n_ch = windows.len();
    let t = windows.first().map(|w| w.len()).unwrap_or(0);
    let shape = validate_and_levels(n_ch, t, noise_bits)?;
    if noise_bits == 0 {
        keep_best_levels_parallel(
            windows,
            shape.n_ch,
            shape.t,
            shape.n_levels,
            noise_bits,
            shape.flags,
            mode,
        )
    } else {
        let shifted: Vec<Vec<i64>> = windows
            .iter()
            .map(|w| w.iter().map(|&v| v >> noise_bits).collect())
            .collect();
        let shifted_views: Vec<&[i64]> = shifted.iter().map(|v| v.as_slice()).collect();
        keep_best_levels_parallel(
            &shifted_views,
            shape.n_ch,
            shape.t,
            shape.n_levels,
            noise_bits,
            shape.flags,
            mode,
        )
    }
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
