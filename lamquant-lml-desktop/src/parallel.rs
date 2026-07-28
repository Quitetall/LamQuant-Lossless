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
//! host caller-side dispatch for the routing that works around it today.

use rayon::prelude::*;

use lamquant_lml_mcu::error::LmlResult;
use lamquant_lml_mcu::lml::{
    self, channel_payload_limit, encode_one_channel, encode_one_channel_bounded, finalize_channels,
    parse_lml_channels, prepare_encode, synthesize_channel_signal, validate_and_levels, DecodePlan,
    EncodeFeatures, EncodeShape,
};
use lamquant_lml_mcu::lpc::LpcMode;

#[derive(Clone, Copy)]
struct ParallelEncodePolicy {
    flags: (bool, bool, bool),
    max_packet_bytes: Option<usize>,
}

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
    policy: ParallelEncodePolicy,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let per_channel_limit = channel_payload_limit(policy.max_packet_bytes, n_ch, t, n_levels)?;
    let per_channel = channels
        .par_iter()
        .map(|&ch| {
            if policy.max_packet_bytes.is_some() {
                encode_one_channel_bounded(
                    ch,
                    n_levels,
                    mode,
                    policy.flags.0,
                    policy.flags.1,
                    policy.flags.2,
                    per_channel_limit,
                )
            } else {
                encode_one_channel(
                    ch,
                    n_levels,
                    mode,
                    policy.flags.0,
                    policy.flags.1,
                    policy.flags.2,
                )
            }
        })
        .collect::<LmlResult<Vec<_>>>()?;
    let (lpc_meta, payload, wins) = finalize_channels(&per_channel);
    let packet = lml::assemble_lml_packet(n_ch, t, n_levels, noise_bits, wins, &lpc_meta, &payload);
    if let Some(limit) = policy.max_packet_bytes {
        if packet.len() > limit {
            return Err(lamquant_lml_mcu::error::LmlError::InvalidHeader(format!(
                "encoded packet requires {} bytes, limit is {}",
                packet.len(),
                limit
            )));
        }
    }
    Ok(packet)
}

/// Adaptive transform-skip keep-best (parallel mirror of `lml::encode_maybe_skip`): encode at
/// `full_levels`, and when the flag is on and the transform is in use, ALSO at `n_levels = 0`, keeping
/// the smaller. Deterministic length compare ⇒ byte-identical to the serial MCU path either way.
///
/// MUST MIRROR: `lamquant_lml_mcu::lml::encode_maybe_skip` is the serial twin. A new candidate depth or
/// selection rule added there MUST be added here in lockstep, or `byte_equal_backends` diverges.
fn keep_best_levels_parallel(
    channels: &[&[i64]],
    shape: EncodeShape,
    noise_bits: u8,
    policy: ParallelEncodePolicy,
    try_transform_skip: bool,
    mode: LpcMode,
) -> LmlResult<Vec<u8>> {
    let full = assemble_at_levels_parallel(
        channels,
        shape.n_ch,
        shape.t,
        shape.n_levels,
        noise_bits,
        policy,
        mode,
    )?;
    if try_transform_skip && shape.n_levels > 0 {
        let skip = assemble_at_levels_parallel(
            channels, shape.n_ch, shape.t, 0, noise_bits, policy, mode,
        )?;
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
        EncodeShape {
            n_ch: prep.n_ch,
            t: prep.t,
            n_levels: prep.n_levels,
            flags: prep.flags,
        },
        noise_bits,
        ParallelEncodePolicy {
            flags: prep.flags,
            max_packet_bytes: None,
        },
        lml::transform_skip_enabled(),
        mode,
    )
}

/// Parallel zero-copy LML encode: `windows` are already-sliced `&[i64]`
/// views (no per-window `Vec<Vec<i64>>` materialization). Mirrors
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
/// with the serial path; host dispatch accounts for this by routing
/// `Anytime{deadline: Some(_)}` to the serial
/// `compress_with_mode_views` instead of calling this function at all. The full
/// fix — thread an explicit per-channel
/// `time_remaining` signal through this kernel so it matches serial even
/// WITH a live deadline — is the tracked follow-up, deliberately not done
/// here (minimal safe close, not the kernel refactor).
///
/// `noise_bits == 0` (hot, lossless): the rayon closure borrows directly from
/// `windows`, so this is
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
    compress_validated_views(
        windows,
        noise_bits,
        mode,
        shape,
        ParallelEncodePolicy {
            flags: shape.flags,
            max_packet_bytes: None,
        },
        lml::transform_skip_enabled(),
    )
}

/// Parallel zero-copy compression with explicit experimental choices.
///
/// Packet selection does not consult process environment variables.
pub fn compress_with_mode_parallel_views_explicit(
    windows: &[&[i64]],
    noise_bits: u8,
    mode: LpcMode,
    features: EncodeFeatures,
) -> LmlResult<Vec<u8>> {
    let n_ch = windows.len();
    let t = windows.first().map(|w| w.len()).unwrap_or(0);
    let shape = validate_and_levels(n_ch, t, noise_bits)?;
    let (flags, try_transform_skip, max_packet_bytes) = features.resolve()?;
    compress_validated_views(
        windows,
        noise_bits,
        mode,
        shape,
        ParallelEncodePolicy {
            flags,
            max_packet_bytes,
        },
        try_transform_skip,
    )
}

fn compress_validated_views(
    windows: &[&[i64]],
    noise_bits: u8,
    mode: LpcMode,
    shape: EncodeShape,
    policy: ParallelEncodePolicy,
    try_transform_skip: bool,
) -> LmlResult<Vec<u8>> {
    if noise_bits == 0 {
        keep_best_levels_parallel(windows, shape, noise_bits, policy, try_transform_skip, mode)
    } else {
        let shifted: Vec<Vec<i64>> = windows
            .iter()
            .map(|w| w.iter().map(|&v| v >> noise_bits).collect())
            .collect();
        let shifted_views: Vec<&[i64]> = shifted.iter().map(|v| v.as_slice()).collect();
        keep_best_levels_parallel(
            &shifted_views,
            shape,
            noise_bits,
            policy,
            try_transform_skip,
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
