//! LML lossless codec — compress/decompress per-window packets.
//!
//! Wire format (LML1, 22-byte header + ASCII prefix):
//!   "LML | {ch}ch | lossless | CRC-32\n"   (human-readable prefix)
//!   [0:4]   'LML1' magic
//!   [4:6]   n_channels (u16 LE)
//!   [6:8]   T samples (u16 LE)
//!   [8]     n_levels (u8)
//!   [9]     flags (u8)
//!   [10:14] LPC meta length (u32 LE)
//!   [14:18] payload length (u32 LE)
//!   [18:22] CRC-32 (u32 LE)
//!   [22..]  LPC meta + Golomb-Rice payload

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::bit_pack;
use crate::crc32::{crc32_update, CRC32_INIT};
use crate::error::{LmlError, LmlResult};
use crate::golomb;
use crate::lifting;
use crate::lpc;
use crate::quant;
use crate::zrle;

/// ADR 0023 Track B1: `flags` bit 0 indicates whether the payload uses
/// the per-subband codec-tag framing. When unset (legacy), every
/// subband is Golomb-Rice; payload bytes are the concatenation of
/// `golomb::encode_dense` outputs.
///
/// When set, every subband's payload is prefixed with a 1-byte tag:
///   * `0x00` → Golomb-Rice (legacy framing within the subband)
///   * `0x01` → BitPack (`bit_pack::encode_dense` framing)
///
/// Encoder sets this bit only when at least one subband strictly
/// reduces payload size by switching to BitPack. Packets whose
/// subbands all favour Golomb-Rice emit byte-equal output to the
/// pre-B1 codec — the `byte_equal_backends` gate stays green on every
/// vector where bit-pack never wins.
pub(crate) const FLAG_BIT_PER_SUBBAND_TAG: u8 = 1 << 0;

/// Per-subband codec tag emitted when `FLAG_BIT_PER_SUBBAND_TAG` is set.
pub(crate) const SUBBAND_TAG_GOLOMB: u8 = 0x00;
pub(crate) const SUBBAND_TAG_BIT_PACK: u8 = 0x01;

/// Track 2 (ADR 0051): flags bit 1 marks a non-lossless "track-2" packet
/// (near-lossless / rate-controlled lossy). Old readers already fail-closed
/// on this bit, so they reject a track-2 packet rather than mis-decode it.
/// When set, the lpc_meta region begins with a 1-byte `mode_id` + mode
/// parameters (see `MODE_*`), and the wavelet/subband layout is replaced by
/// the mode-specific payload. Mutually exclusive with `noise_bits` (the
/// bits-2..7 field is unused — read as 0 — when this flag is set).
pub(crate) const FLAG_BIT_TRACK2_MODE: u8 = 1 << 1;

/// Track-2 mode ids (first byte of lpc_meta when `FLAG_BIT_TRACK2_MODE`).
/// `0x01` = bounded-MAE closed-loop DPCM (sample-domain, guarantees
/// `max|orig − recon| ≤ δ`); δ follows as a u64 LE. `0x02` = target-BPS
/// coefficient-domain (wavelet subbands deadzone-quantized, rate-controlled).
pub(crate) const MODE_BOUNDED_MAE: u8 = 0x01;

/// Track-2 mode `0x02`: target-BPS rate-controlled lossy. Sub-header is
/// `[0x02][n_sub:u8][q_s:u32 LE × n_sub]`, then per channel per subband
/// `[order:u8][coeffs:i32 × order]`; payload = per channel per subband
/// `golomb(residual)`. Uses the real `n_levels` (wavelet path). See
/// [`compress_target_bps`].
pub(crate) const MODE_TARGET_BPS: u8 = 0x02;

/// Track-2 per-subband payload entropy coder tags (ADR 0051 P3). Each track-2
/// subband payload is prefixed with one of these so the decoder dispatches and
/// the encoder can keep whichever is smaller per subband.
const PAYLOAD_CODER_GOLOMB: u8 = 0x00;
const PAYLOAD_CODER_ZRLE: u8 = 0x01;
/// Empirical-categorical range coder (P3.5, order-0). Opt-in
/// `experimental_arithmetic` build only; firmware / default builds fail closed.
const PAYLOAD_CODER_ARITHMETIC: u8 = 0x02;
/// Context-adaptive empirical-categorical range coder (P3.5, order-1 on the
/// previous coefficient's magnitude bucket). Same opt-in / fail-closed rules.
const PAYLOAD_CODER_ARITH_CTX: u8 = 0x03;

/// Encode one track-2 subband residual, keeping the smaller of Golomb-Rice and
/// zero-run-length (P3). Output is `[tag][coded]`. zrle wins on the zero-heavy
/// heavily-quantized low-BPS streams (Golomb's 1-bit/symbol floor); Golomb wins
/// on dense streams. Never worse than min(golomb, zrle) + 1 tag byte.
fn encode_subband_payload(values: &[i64]) -> LmlResult<Vec<u8>> {
    let mut best_tag = PAYLOAD_CODER_GOLOMB;
    let mut best = golomb::encode_dense(values)?;
    let z = zrle::encode_dense(values)?;
    if z.len() < best.len() {
        best_tag = PAYLOAD_CODER_ZRLE;
        best = z;
    }
    // P3.5: empirical-categorical range coder (opt-in build). Falls back
    // silently if its alphabet is too wide (Err) or it doesn't win.
    #[cfg(feature = "experimental_arithmetic")]
    {
        if let Ok(a) = crate::arith_cat::encode_dense(values) {
            if a.len() < best.len() {
                best_tag = PAYLOAD_CODER_ARITHMETIC;
                best = a;
            }
        }
        if let Ok(a) = crate::arith_cat::encode_dense_ctx(values) {
            if a.len() < best.len() {
                best_tag = PAYLOAD_CODER_ARITH_CTX;
                best = a;
            }
        }
    }
    let mut out = Vec::with_capacity(1 + best.len());
    out.push(best_tag);
    out.extend_from_slice(&best);
    Ok(out)
}

/// Decode one track-2 subband payload written by [`encode_subband_payload`],
/// starting at `offset`. Returns `(values, bytes_consumed_from_offset)`.
fn decode_subband_payload(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    if offset >= data.len() {
        return Err(LmlError::Truncated {
            expected: offset + 1,
            actual: data.len(),
            context: "track-2 payload coder tag",
        });
    }
    let tag = data[offset];
    let (vals, consumed) = match tag {
        PAYLOAD_CODER_GOLOMB => golomb::decode_dense(data, offset + 1)?,
        PAYLOAD_CODER_ZRLE => zrle::decode_dense(data, offset + 1)?,
        PAYLOAD_CODER_ARITHMETIC => {
            #[cfg(feature = "experimental_arithmetic")]
            {
                crate::arith_cat::decode_dense(data, offset + 1)?
            }
            #[cfg(not(feature = "experimental_arithmetic"))]
            {
                return Err(LmlError::InvalidHeader(
                    "payload coder 0x02 (arithmetic) requires an experimental_arithmetic \
                     build; this reader fails closed"
                        .into(),
                ));
            }
        }
        PAYLOAD_CODER_ARITH_CTX => {
            #[cfg(feature = "experimental_arithmetic")]
            {
                crate::arith_cat::decode_dense_ctx(data, offset + 1)?
            }
            #[cfg(not(feature = "experimental_arithmetic"))]
            {
                return Err(LmlError::InvalidHeader(
                    "payload coder 0x03 (arith-ctx) requires an experimental_arithmetic \
                     build; this reader fails closed"
                        .into(),
                ));
            }
        }
        other => {
            return Err(LmlError::InvalidHeader(format!(
                "unknown track-2 payload coder tag 0x{:02X}",
                other
            )))
        }
    };
    Ok((vals, consumed + 1))
}
/// ADR 0023 Track B5: Witten-Neal-Cleary arithmetic coding with a
/// static Laplace probability model. Decoder support is always
/// compiled in on host builds; firmware (no_std) decoder fails
/// closed on tag 0x02 with `unknown per-subband codec tag`. Encoder
/// considers arithmetic a candidate only when
/// `LAMQUANT_TRY_ARITHMETIC=1`, so default-built archives never
/// carry this tag and existing readers are unaffected.
#[cfg(feature = "experimental_arithmetic")]
pub(crate) const SUBBAND_TAG_ARITHMETIC: u8 = 0x02;

#[cfg(feature = "experimental_bit_pack")]
fn experimental_bit_pack_enabled() -> bool {
    std::env::var("LAMQUANT_TRY_BIT_PACK")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// ADR 0023 Track B5+: read at compress entry to decide whether the
/// arithmetic-coder candidate is added to the per-subband selection.
/// One-shot per compress call so the env-var check doesn't taint
/// the hot path.
#[cfg(feature = "experimental_arithmetic")]
fn experimental_arithmetic_enabled() -> bool {
    std::env::var("LAMQUANT_TRY_ARITHMETIC")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// ADR 0023 Track B6: raise the per-subband Burg / AIC ceiling from
/// `LPC_ORDER_HARD_CAP` to `EXTENDED_LPC_ORDER_HARD_CAP` so the
/// adaptive search may pick orders > 16 when the residual savings
/// outweigh the extra ~4 bytes per coefficient of meta. Default
/// off — encoder behaves identically to pre-B6 unless
/// `LAMQUANT_TRY_EXTENDED_LPC=1`. AIC self-throttles when the
/// extra coefficients don't pay off, so the worst case is "same
/// order picked as before".
#[cfg(feature = "host")]
fn experimental_extended_lpc_enabled() -> bool {
    std::env::var("LAMQUANT_TRY_EXTENDED_LPC")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Track B6 ceiling. Bumped from 16 → 32 so the AIC walk can
/// explore deeper for highly autocorrelated single-channel sources
/// like Bonn EEG. Larger orders mean more `i32` coefficients in
/// `lpc_meta` (~4 bytes each), which AIC must justify against the
/// residual-byte savings before picking. Wire format already
/// stores `chosen_order: u8`, so 32 fits without a schema change.
pub(crate) const EXTENDED_LPC_ORDER_HARD_CAP: usize = 32;

/// Compute per-subband sample counts for a packet with window length
/// `t` decomposed at `n_levels` lifting depth. Mirrors the subband
/// shapes produced by `lifting::forward_*` (approx = ceil(n/2),
/// detail = floor(n/2)). Subband order matches the encoder's `subs`
/// vec: `[a_top, d_top, d_top-1, ..., d_1]` for n_levels >= 1; a
/// single full-length subband for n_levels == 0.
///
/// Track B1 decoder uses these counts to drive `bit_pack::decode_dense`,
/// which needs the sample count up-front (it isn't carried inline in
/// the bit-packed payload).
fn subband_lengths(t: usize, n_levels: u8) -> Vec<usize> {
    if n_levels == 0 {
        return vec![t];
    }
    // Step the L1, L2, … decomposition: at each level the current
    // approx-of-the-previous is halved (ceil) into a new approx +
    // floor-half into a new detail. Collect details from top down.
    let mut details = Vec::with_capacity(n_levels as usize);
    let mut approx = t;
    for _ in 0..n_levels {
        let next_approx = (approx + 1) / 2;
        let detail = approx / 2;
        details.push(detail);
        approx = next_approx;
    }
    // Final approx (highest-level) sits first; then each level's detail
    // from top → bottom (encoder pushes a_top, d_top, then earlier
    // details in lifting order).
    let mut out = Vec::with_capacity(n_levels as usize + 1);
    out.push(approx);
    for d in details.iter().rev() {
        out.push(*d);
    }
    out
}

/// Warn-once latch for legacy (pre-a81cd04) CRC-scope acceptance.
///
/// Set the first time a packet is accepted via the payload-only legacy
/// CRC scope (see [`verify_packet_crc`]). On `std` builds this also emits
/// a one-shot `eprintln!` so the operator can observe that an archive
/// predates the 2026-05-11 header-CRC change. Public so callers (tools,
/// tests) can query whether any legacy packet was read this run.
///
/// Latched false→true on the first legacy accept and **never cleared** in
/// production — that monotonicity is what makes the warning fire exactly
/// once per process. Tests reset it explicitly to observe their own runs.
pub static SAW_LEGACY_CRC: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Outcome of the per-window CRC verification — which scope matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrcScope {
    /// Modern (a81cd04+) scope: `crc32(header[4..18] || lpc_meta || payload)`.
    /// This is the common fast path; the legacy recompute never runs here.
    Modern,
    /// Legacy (pre-a81cd04) scope: `crc32(lpc_meta || payload)` — the
    /// header variable-fields were NOT covered. The packet's data is
    /// valid; only the older CRC convention was used. A warn-once is
    /// emitted and [`SAW_LEGACY_CRC`] is latched.
    Legacy,
}

/// Verify a per-window LML1 packet CRC with back-compat fallback.
///
/// ROOT CAUSE: commit a81cd04 (2026-05-11) widened the per-window CRC
/// scope from `crc32(lpc_meta || payload)` to
/// `crc32(header[4..18] || lpc_meta || payload)` on BOTH encode and
/// decode with no version gate, so every file written *before* that
/// commit fails CRC under the current reader even though its bytes are
/// intact. The LML1 packet header has no version field to branch on, so
/// we branch on the CRC itself:
///
///   1. Compute the **modern** scope. If it matches → `Ok(Modern)`. This
///      is the common path and costs exactly what it did before — the
///      legacy recompute below never runs for a modern file.
///   2. On mismatch, recompute the **legacy** scope (payload-only, i.e.
///      omit the `data[4..18]` prefix). If it matches → `Ok(Legacy)`:
///      the packet is a genuine pre-a81cd04 packet whose data is valid;
///      latch [`SAW_LEGACY_CRC`] + warn-once so the read is observable.
///   3. If both miss → `Err(CrcMismatch)`: genuine corruption.
///
/// `header_var` is `&data[4..18]` (the 14 variable-header bytes) and
/// `payload_data` is `&data[HEADER_SIZE..HEADER_SIZE + lpc_len + sub_len]`
/// (== `lpc_meta || payload`, contiguous). The encoder is unchanged: new
/// files keep the stronger modern scope. This is decode-side only.
#[inline]
fn verify_packet_crc(
    header_var: &[u8],
    payload_data: &[u8],
    crc_exp: u32,
) -> LmlResult<CrcScope> {
    // (1) Modern scope — the fast path. Identical cost to the old code.
    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, header_var);
    crc_state = crc32_update(crc_state, payload_data);
    let crc_modern = crc_state ^ CRC32_INIT;
    if crc_modern == crc_exp {
        return Ok(CrcScope::Modern);
    }

    // (2) Legacy scope — payload-only (no header[4..18] prefix). Only
    // runs when the modern scope already missed, so the common path
    // stays byte-for-byte the same cost.
    let mut crc_legacy_state = CRC32_INIT;
    crc_legacy_state = crc32_update(crc_legacy_state, payload_data);
    let crc_legacy = crc_legacy_state ^ CRC32_INIT;
    if crc_legacy == crc_exp {
        // Latch + warn-once: the read is observable, but the data is good.
        if !SAW_LEGACY_CRC.swap(true, core::sync::atomic::Ordering::Relaxed) {
            #[cfg(feature = "std")]
            eprintln!(
                "warning: LML packet accepted via legacy (pre-2026-05-11) \
                 payload-only CRC scope. Archive predates commit a81cd04; \
                 data is valid. Re-encode to upgrade to the header+payload CRC."
            );
        }
        return Ok(CrcScope::Legacy);
    }

    // (3) Both scopes miss → genuine corruption. Report `crc_modern` as
    // `actual`: the modern header+payload scope is the authoritative one
    // for any file this codec writes, so it is the canonical "what we
    // computed" value. Do NOT switch this to `crc_legacy` — that would
    // make a corrupt MODERN file (the common case) report a meaningless
    // payload-only CRC. The human-readable Display string spells out that
    // neither scope matched, so the diagnostic is unambiguous either way.
    Err(LmlError::CrcMismatch {
        expected: crc_exp,
        actual: crc_modern,
    })
}

/// LML1 packet magic. Single source of truth — used by `lml.rs` here,
/// the container-format wrapper in `container.rs`, and the streaming
/// parser in `stream.rs`. Any drift between these breaks decode.
pub const MAGIC: &[u8; 4] = b"LML1";
const HEADER_SIZE: usize = 22;
const BIAS_CTX: usize = 32;

/// Maximum bytes a decoder will allocate for the reconstructed signal.
///
/// Audit-2026-05-11 Fix-C4: without this cap an attacker can craft a
/// 22-byte LML1 packet with `n_ch=1024, t=65535` and a valid CRC over
/// `header_var || empty payload` → decoder attempts `vec![vec![0i64;
/// 65535]; 1024]` = 512 MB malloc before any payload-shape sanity check.
///
/// Sized to fit 1024-channel × 65535-sample windows (× 8 B / i64 = 537 MB)
/// with margin — the worst realistic case is a high-channel-count
/// research rig at 128 kHz sample rate. 1 GB matches what the host CLI
/// can reasonably allocate before the OS pushes back; the firmware never
/// uses this path (firmware encodes windows directly without going
/// through the decoder allocator).
const MAX_DECODE_BYTES: u64 = 1024 * 1024 * 1024;

/// Hard ceiling on the LPC order any subband can request, regardless of
/// what the principled length-based rule below would yield. Caps Levinson
/// O(p²) inner work and bounds the per-subband lpc_meta size at 64 bytes.
const LPC_ORDER_HARD_CAP: usize = 16;

/// Burg-style principled ceiling for the AIC search: `max_order ≈ N/8`,
/// clamped to `[0, LPC_ORDER_HARD_CAP]` and `t/4` (the autocorr stability
/// bound that `analyze_adaptive` itself enforces).
///
/// The N/8 rule comes from the AR-modelling literature (Berryman, Marple,
/// Kay): for a length-N segment, orders much beyond N/8 start fitting
/// noise rather than signal, and the AIC penalty becomes the only thing
/// pulling the chosen order back down. Capping at N/8 keeps the AIC walk
/// inside the regime where each extra coefficient still has meaningful
/// signal to explain — no discrete sample-rate bins, no per-subband
/// hardcoded schedule. A 250 Hz × 10 s window has subband lengths
/// 313/313/625/1250, giving ceilings 16/16/16/16 (all hard-capped); a
/// 32-sample tiny window gives ceiling 4. Principled and smooth.
#[inline]
fn lpc_max_order(subband_len: usize) -> usize {
    (subband_len / 8).min(LPC_ORDER_HARD_CAP)
}

/// ADR 0023 Track B6 — extended ceiling for the experimental
/// long-context predictor path. Same Burg N/8 rule applied against
/// the higher `EXTENDED_LPC_ORDER_HARD_CAP`. AIC byte-cost search
/// inside `lpc::analyze_with_mode` picks the actual order.
#[inline]
#[cfg(feature = "host")]
fn lpc_max_order_extended(subband_len: usize) -> usize {
    (subband_len / 8).min(EXTENDED_LPC_ORDER_HARD_CAP)
}

/// Clamp a mode's `max_order` against the per-subband Burg ceiling so
/// `analyze_with_mode` never asks Levinson for more orders than the
/// subband can support. `Fixed` mode is unaffected — its schedule is
/// already small and the inner `analyze` does its own size check.
#[inline]
fn scope_lpc_mode(mode: lpc::LpcMode, ceiling: usize) -> lpc::LpcMode {
    match mode {
        lpc::LpcMode::Fixed => mode,
        lpc::LpcMode::Adaptive { max_order } => lpc::LpcMode::Adaptive {
            max_order: max_order.min(ceiling),
        },
        #[cfg(feature = "std")]
        lpc::LpcMode::Anytime {
            max_order,
            deadline,
        } => lpc::LpcMode::Anytime {
            max_order: max_order.min(ceiling),
            deadline,
        },
        #[cfg(not(feature = "std"))]
        lpc::LpcMode::Anytime { max_order } => lpc::LpcMode::Anytime {
            max_order: max_order.min(ceiling),
        },
    }
}

/// Compress [n_ch][T] signal → LML1 packet bytes (default LPC mode).
///
/// Thin wrapper around [`compress_with_mode`] using
/// [`lpc::LpcMode::default()`] — `Anytime { max_order: 16, deadline:
/// None }`, which behaves identically to pure adaptive when no clock
/// pressure is signalled.
pub fn compress(signal: &[Vec<i64>], noise_bits: u8) -> LmlResult<Vec<u8>> {
    compress_with_mode(signal, noise_bits, lpc::LpcMode::default())
}

/// Compress [n_ch][T] signal → LML1 packet bytes with explicit LPC mode.
///
/// `mode` controls the speed / CR trade-off the encoder makes per
/// subband. See [`lpc::LpcMode`] for semantics:
/// * `Fixed` — fastest, slightly worse CR
/// * `Adaptive` — best CR, variable CPU
/// * `Anytime` — fixed first then adaptive if deadline allows (default)
///
/// Returns `Err(LmlError::InvalidHeader)` on out-of-range dimensions —
/// FFI/WASM consumers cannot recover from a process-wide panic, so the
/// previous `assert!`-based validation was hostile to those callers.
/// Audit-2026-05-11 Fix-C3.
pub fn compress_with_mode(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: lpc::LpcMode,
) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };

    // Validate u16 header field limits — return Err instead of panicking.
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!(
            "n_ch={} out of range 1..=1024",
            n_ch
        )));
    }
    if t == 0 || t > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!(
            "T={} out of range 1..={}",
            t,
            u16::MAX
        )));
    }
    // noise_bits is stored as a u6 in flag bits 2-7 (max 63 on the wire),
    // but the practical / Python clamp is 32. Keep both implementations aligned.
    if noise_bits > 32 {
        return Err(LmlError::InvalidHeader(format!(
            "noise_bits={} exceeds max 32",
            noise_bits
        )));
    }

    // Noise stripping: shift in-place if needed, otherwise borrow
    let owned_signal: Vec<Vec<i64>>;
    let signal_ref: &[Vec<i64>] = if noise_bits > 0 {
        owned_signal = signal
            .iter()
            .map(|ch| ch.iter().map(|&v| v >> noise_bits).collect())
            .collect();
        &owned_signal
    } else {
        signal
    };

    let n_levels = compute_n_levels(t);

    // Pre-allocate output buffers
    let mut lpc_meta = Vec::with_capacity(n_ch * 4 * 40);
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let mut per_channel_results: Vec<ChannelEncodeOutput> = Vec::with_capacity(n_ch);
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    let mut payload: Vec<u8> = Vec::with_capacity(n_ch * t * 4);
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let mut total_n_subbands: usize = 0;
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let mut total_bp_savings: usize = 0;

    // ADR 0023 Track B5+: env-var-gated experimental arithmetic coder
    // candidate. Cargo-feature-gated too: when `experimental_arithmetic`
    // isn't compiled in, the candidate is permanently off regardless
    // of the env var.
    let try_arithmetic = {
        #[cfg(feature = "experimental_arithmetic")]
        {
            experimental_arithmetic_enabled()
        }
        #[cfg(not(feature = "experimental_arithmetic"))]
        {
            false
        }
    };
    let try_bit_pack = {
        #[cfg(feature = "experimental_bit_pack")]
        {
            experimental_bit_pack_enabled()
        }
        #[cfg(not(feature = "experimental_bit_pack"))]
        {
            false
        }
    };
    // ADR 0023 Track B6: env-var-gated extended LPC order cap.
    let try_extended_lpc = {
        #[cfg(feature = "host")]
        {
            experimental_extended_lpc_enabled()
        }
        #[cfg(not(feature = "host"))]
        {
            false
        }
    };
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    {
        for ch_idx in 0..n_ch {
            let ch_out = encode_one_channel(
                &signal_ref[ch_idx],
                n_levels,
                mode,
                try_arithmetic,
                try_bit_pack,
                try_extended_lpc,
            )?;
            lpc_meta.extend_from_slice(&ch_out.meta);
            total_n_subbands += ch_out.n_subbands;
            total_bp_savings += ch_out.bp_savings;
            per_channel_results.push(ch_out);
        }
    }
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    {
        for ch_idx in 0..n_ch {
            let ch_out = encode_one_channel(
                &signal_ref[ch_idx],
                n_levels,
                mode,
                try_arithmetic,
                try_bit_pack,
                try_extended_lpc,
            )?;
            lpc_meta.extend_from_slice(&ch_out.meta);
            payload.extend_from_slice(&ch_out.payload);
        }
    }

    // ADR 0023 Track B1: switch to tagged framing only when the
    // bit-pack savings exceed the 1-byte-per-subband tag overhead.
    // Strict `>` so a break-even doesn't churn the wire format.
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let any_bit_pack_wins_global = total_bp_savings > total_n_subbands;
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    let any_bit_pack_wins_global = false;
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let payload = assemble_payload(&per_channel_results, any_bit_pack_wins_global);

    // Build the 14-byte variable header region (everything between magic
    // and the CRC field) so we can stream it through CRC and serialise it
    // to the output. CRC must cover this region — without it, a one-byte
    // flip in n_ch / t / n_levels / flags / lpc_len / sub_len escapes
    // detection (the test `test_exits_nonzero_on_corrupted_lml` regressed
    // on this). Magic is constant so it does not need CRC; the CRC field
    // itself is excluded to avoid self-reference.
    let mut flags: u8 = (noise_bits & 0x3F) << 2;
    if any_bit_pack_wins_global {
        flags |= FLAG_BIT_PER_SUBBAND_TAG;
    }
    let mut header_var = [0u8; 14];
    header_var[0..2].copy_from_slice(&(n_ch as u16).to_le_bytes());
    header_var[2..4].copy_from_slice(&(t as u16).to_le_bytes());
    header_var[4] = n_levels;
    header_var[5] = flags;
    header_var[6..10].copy_from_slice(&(lpc_meta.len() as u32).to_le_bytes());
    header_var[10..14].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    // Streaming CRC over: header_var || lpc_meta || payload.
    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, &header_var);
    crc_state = crc32_update(crc_state, &lpc_meta);
    crc_state = crc32_update(crc_state, &payload);
    let crc = crc_state ^ CRC32_INIT;

    // Assemble
    let mode = if noise_bits == 0 { "lossless" } else { "noise" };
    let prefix = format!("LML | {}ch | {} | CRC-32\n", n_ch, mode);

    let total = prefix.len() + HEADER_SIZE + lpc_meta.len() + payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&header_var);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&lpc_meta);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Compress [n_ch][T] → LML1 **bounded-MAE near-lossless** packet (ADR 0051
/// track 2, mode `MODE_BOUNDED_MAE`).
///
/// Guarantees `max|orig − recon| ≤ δ` per sample. Uses closed-loop
/// sample-domain DPCM ([`lpc::analyze_closed_loop_bounded`]): a per-channel
/// LPC predictor runs over *reconstructed* samples and the prediction
/// residual is uniform-quantized with step `q = 2δ+1`. The decoder replays
/// the identical loop, so the bound is structural (independent of the signal
/// or coeffs). `δ = 0` is exact lossless. The wavelet/subband path is
/// bypassed entirely.
///
/// Wire: per-window LML1 with `flags = FLAG_BIT_TRACK2_MODE`, `n_levels = 0`.
/// `lpc_meta = [MODE_BOUNDED_MAE][δ:u64 LE]` then per channel
/// `[order:u8][coeffs:i32 LE × order]`; `payload` = per-channel
/// `golomb::encode_dense(indices)`. Decoded by [`decompress`] /
/// [`decompress_parallel`] (firmware-decodable — integer-only).
pub fn compress_bounded_mae(
    signal: &[Vec<i64>],
    delta: u64,
    mode: lpc::LpcMode,
) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!(
            "n_ch={} out of range 1..=1024",
            n_ch
        )));
    }
    if t == 0 || t > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!(
            "T={} out of range 1..={}",
            t,
            u16::MAX
        )));
    }
    // All channels must share the window length (header carries one T).
    for (c, ch) in signal.iter().enumerate() {
        if ch.len() != t {
            return Err(LmlError::InvalidHeader(format!(
                "ragged channels: ch {} has {} samples, expected {}",
                c,
                ch.len(),
                t
            )));
        }
    }
    // q = 2δ+1 must not overflow i64. δ is an error budget in raw sample
    // units — anything beyond i32 range is already absurd for real signals;
    // bound it well below the overflow point and fail-closed otherwise.
    if delta > (i64::MAX as u64 - 1) / 2 {
        return Err(LmlError::InvalidHeader(format!(
            "delta={} exceeds max bounded-MAE budget",
            delta
        )));
    }
    let q = 2 * delta as i64 + 1;

    let scoped = scope_lpc_mode(mode, lpc_max_order(t));

    // lpc_meta begins with the mode sub-header, then per-channel predictor.
    let mut lpc_meta = Vec::with_capacity(9 + n_ch * (1 + 16 * 4));
    lpc_meta.push(MODE_BOUNDED_MAE);
    lpc_meta.extend_from_slice(&delta.to_le_bytes());

    let mut payload: Vec<u8> = Vec::with_capacity(n_ch * t);
    for ch in signal.iter() {
        // Predictor coeffs from open-loop analysis of the raw channel — these
        // are just predictor taps; the bound holds for any of them. sb_idx=0
        // only matters for Fixed mode (gives a small order).
        let (coeffs, _residual, order) =
            lpc::analyze_with_mode(ch, 0, scoped, BIAS_CTX, None);
        let indices = lpc::analyze_closed_loop_bounded(ch, &coeffs, order, q);

        // order must fit u8 (scope_lpc_mode caps at lpc_max_order ≤ 16).
        debug_assert!(order <= u8::MAX as usize);
        lpc_meta.push(order as u8);
        for c in &coeffs {
            lpc_meta.extend_from_slice(&c.to_le_bytes());
        }

        payload.extend_from_slice(&encode_subband_payload(&indices)?);
    }

    let flags: u8 = FLAG_BIT_TRACK2_MODE;
    let n_levels: u8 = 0; // unused in track-2 bounded mode
    let mut header_var = [0u8; 14];
    header_var[0..2].copy_from_slice(&(n_ch as u16).to_le_bytes());
    header_var[2..4].copy_from_slice(&(t as u16).to_le_bytes());
    header_var[4] = n_levels;
    header_var[5] = flags;
    header_var[6..10].copy_from_slice(&(lpc_meta.len() as u32).to_le_bytes());
    header_var[10..14].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, &header_var);
    crc_state = crc32_update(crc_state, &lpc_meta);
    crc_state = crc32_update(crc_state, &payload);
    let crc = crc_state ^ CRC32_INIT;

    let prefix = format!("LML | {}ch | near-lossless | CRC-32\n", n_ch);
    let total = prefix.len() + HEADER_SIZE + lpc_meta.len() + payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&header_var);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&lpc_meta);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Forward lifting of one channel into its ordered subbands for the given
/// `n_levels` (`[approx, detail_top, ..., detail_1]`). Mirrors
/// `encode_one_channel`'s split + `quant::inverse_for_levels`.
fn forward_subbands(signal_ch: &[i64], n_levels: u8) -> Vec<Vec<i64>> {
    match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(signal_ch);
            vec![a3, d3, d2, d1]
        }
        2 => {
            let (l1a, l1d) = lifting::forward(signal_ch);
            let (l2a, l2d) = lifting::forward(&l1a);
            vec![l2a, l2d, l1d]
        }
        1 => {
            let (a, d) = lifting::forward(signal_ch);
            vec![a, d]
        }
        _ => vec![signal_ch.to_vec()],
    }
}

/// Compress [n_ch][T] → LML1 **target-BPS rate-controlled lossy** packet
/// (ADR 0051 track 2, mode `MODE_TARGET_BPS`).
///
/// Minimizes distortion subject to a bits-per-sample ceiling — the H.BWC
/// WP1..WP8 competition tier. Each channel is lifting-transformed; each
/// subband is deadzone-quantized with a per-subband step `q_s` weighted by the
/// subband's synthesis gain (finer steps where errors amplify most), the
/// quantized indices are LPC-coded, and the residual Golomb-coded. A single
/// global `scale` is binary-searched so the packet lands at/under `target_bps`
/// (the finest quantization meeting the budget). Host-side search; the decode
/// is integer-only and firmware-capable.
pub fn compress_target_bps(
    signal: &[Vec<i64>],
    target_bps: f64,
    mode: lpc::LpcMode,
) -> LmlResult<Vec<u8>> {
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("n_ch={} out of range 1..=1024", n_ch)));
    }
    if t == 0 || t > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!("T={} out of range 1..={}", t, u16::MAX)));
    }
    for (c, ch) in signal.iter().enumerate() {
        if ch.len() != t {
            return Err(LmlError::InvalidHeader(format!(
                "ragged channels: ch {} has {} samples, expected {}",
                c, ch.len(), t
            )));
        }
    }
    if !(target_bps.is_finite() && target_bps > 0.0) {
        return Err(LmlError::InvalidHeader(format!(
            "target_bps must be finite > 0, got {}",
            target_bps
        )));
    }

    let n_levels = compute_n_levels(t);
    let sub_lens = subband_lengths(t, n_levels);
    let n_sub = sub_lens.len();
    let gains = quant::synthesis_gains(n_levels, &sub_lens);

    // Transform every channel ONCE; the rate search re-quantizes these.
    let chan_subs: Vec<Vec<Vec<i64>>> =
        signal.iter().map(|ch| forward_subbands(ch, n_levels)).collect();

    let nm = (n_ch * t) as f64;
    // Fixed wire overhead outside meta_body+payload: 22-byte header + mode_id
    // + n_sub byte + q_s table.
    let fixed_overhead = HEADER_SIZE + 1 + 1 + 4 * n_sub;

    // Encode all channels at a quantizer scale → (meta_body, payload, q_s).
    let encode_at = |scale: f64| -> LmlResult<(Vec<u8>, Vec<u8>, Vec<i64>)> {
        let qs = quant::steps_for_scale(scale, &gains);
        let mut meta = Vec::new();
        let mut payload = Vec::new();
        for subs in &chan_subs {
            for (sb_idx, sub) in subs.iter().enumerate() {
                let idx = quant::quantize(sub, qs[sb_idx]);
                let scoped = scope_lpc_mode(mode, lpc_max_order(sub.len()));
                let (coeffs, residual, order) =
                    lpc::analyze_with_mode(&idx, sb_idx, scoped, BIAS_CTX, None);
                meta.push(order as u8);
                for &c in &coeffs {
                    meta.extend_from_slice(&c.to_le_bytes());
                }
                payload.extend_from_slice(&encode_subband_payload(&residual)?);
            }
        }
        Ok((meta, payload, qs))
    };
    let bps_of = |meta: &[u8], payload: &[u8]| -> f64 {
        (fixed_overhead + meta.len() + payload.len()) as f64 * 8.0 / nm
    };

    // Binary-search the smallest scale whose packet fits the BPS ceiling
    // (smallest scale = finest quantization = best quality within budget).
    let (m0, p0, _) = encode_at(0.0)?;
    let best_scale = if bps_of(&m0, &p0) <= target_bps {
        0.0
    } else {
        let mut hi = 1.0f64;
        let mut iters = 0;
        loop {
            let (m, p, _) = encode_at(hi)?;
            if bps_of(&m, &p) <= target_bps || hi >= 1.0e7 {
                break;
            }
            hi *= 2.0;
            iters += 1;
            if iters > 40 {
                break;
            }
        }
        let mut lo = 0.0f64;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let (m, p, _) = encode_at(mid)?;
            if bps_of(&m, &p) <= target_bps {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    };

    let (meta_body, payload, qs) = encode_at(best_scale)?;

    // lpc_meta = [MODE_TARGET_BPS][n_sub][q_s × n_sub] + per-ch per-sub meta.
    let mut lpc_meta = Vec::with_capacity(2 + 4 * n_sub + meta_body.len());
    lpc_meta.push(MODE_TARGET_BPS);
    lpc_meta.push(n_sub as u8);
    for &q in &qs {
        lpc_meta.extend_from_slice(&(q as u32).to_le_bytes());
    }
    lpc_meta.extend_from_slice(&meta_body);

    let flags: u8 = FLAG_BIT_TRACK2_MODE;
    let mut header_var = [0u8; 14];
    header_var[0..2].copy_from_slice(&(n_ch as u16).to_le_bytes());
    header_var[2..4].copy_from_slice(&(t as u16).to_le_bytes());
    header_var[4] = n_levels;
    header_var[5] = flags;
    header_var[6..10].copy_from_slice(&(lpc_meta.len() as u32).to_le_bytes());
    header_var[10..14].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, &header_var);
    crc_state = crc32_update(crc_state, &lpc_meta);
    crc_state = crc32_update(crc_state, &payload);
    let crc = crc_state ^ CRC32_INIT;

    let prefix = format!("LML | {}ch | lossy-bps | CRC-32\n", n_ch);
    let total = prefix.len() + HEADER_SIZE + lpc_meta.len() + payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&header_var);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&lpc_meta);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Decode a track-2 packet (lpc_meta begins with a `mode_id`). Shared by the
/// serial and parallel decoders. `lpc_data` and `payload` are the already
/// CRC-verified, bounds-checked regions. Integer-only ⇒ firmware-decodable.
fn decode_track2(
    n_ch: usize,
    t: usize,
    n_levels: u8,
    lpc_data: &[u8],
    payload: &[u8],
) -> LmlResult<Vec<Vec<i64>>> {
    if lpc_data.is_empty() {
        return Err(LmlError::Truncated {
            expected: 1,
            actual: 0,
            context: "track-2 mode id",
        });
    }
    let mode_id = lpc_data[0];
    match mode_id {
        MODE_BOUNDED_MAE => {
            if lpc_data.len() < 9 {
                return Err(LmlError::Truncated {
                    expected: 9,
                    actual: lpc_data.len(),
                    context: "track-2 bounded-MAE sub-header",
                });
            }
            let delta = u64::from_le_bytes([
                lpc_data[1], lpc_data[2], lpc_data[3], lpc_data[4], lpc_data[5], lpc_data[6],
                lpc_data[7], lpc_data[8],
            ]);
            if delta > (i64::MAX as u64 - 1) / 2 {
                return Err(LmlError::InvalidHeader(format!(
                    "track-2 delta={} exceeds max budget",
                    delta
                )));
            }
            let q = 2 * delta as i64 + 1;
            let mut lpc_pos = 9usize;
            let mut sub_pos = 0usize;
            let mut signal = vec![vec![0i64; t]; n_ch];
            for ch in signal.iter_mut() {
                if lpc_pos >= lpc_data.len() {
                    return Err(LmlError::Truncated {
                        expected: lpc_pos + 1,
                        actual: lpc_data.len(),
                        context: "track-2 LPC order",
                    });
                }
                let order = lpc_data[lpc_pos] as usize;
                lpc_pos += 1;
                let mut coeffs = Vec::with_capacity(order);
                for _ in 0..order {
                    if lpc_pos + 4 > lpc_data.len() {
                        return Err(LmlError::Truncated {
                            expected: lpc_pos + 4,
                            actual: lpc_data.len(),
                            context: "track-2 LPC coefficients",
                        });
                    }
                    coeffs.push(i32::from_le_bytes([
                        lpc_data[lpc_pos],
                        lpc_data[lpc_pos + 1],
                        lpc_data[lpc_pos + 2],
                        lpc_data[lpc_pos + 3],
                    ]));
                    lpc_pos += 4;
                }
                if sub_pos > payload.len() {
                    return Err(LmlError::InvalidHeader(format!(
                        "track-2 payload cursor {sub_pos} past end {}",
                        payload.len()
                    )));
                }
                let (indices, consumed) = decode_subband_payload(&payload[sub_pos..], 0)?;
                sub_pos += consumed;
                if indices.len() != t {
                    return Err(LmlError::InvalidHeader(format!(
                        "track-2 channel decoded {} samples, expected {}",
                        indices.len(),
                        t
                    )));
                }
                *ch = lpc::synthesize_closed_loop_bounded(&indices, &coeffs, order, q);
            }
            Ok(signal)
        }
        MODE_TARGET_BPS => {
            // [0x02][n_sub:u8][q_s:u32 × n_sub] then per-ch per-sub [order][coeffs];
            // payload = per-ch per-sub golomb(residual). Decode: golomb → LPC
            // synth → dequant (×q_s) → inverse lifting.
            if lpc_data.len() < 2 {
                return Err(LmlError::Truncated {
                    expected: 2,
                    actual: lpc_data.len(),
                    context: "track-2 target-BPS sub-header",
                });
            }
            let n_sub = lpc_data[1] as usize;
            let expected_sub = if n_levels == 0 { 1 } else { n_levels as usize + 1 };
            if n_sub != expected_sub {
                return Err(LmlError::InvalidHeader(format!(
                    "track-2 target-BPS n_sub={} != expected {} for n_levels={}",
                    n_sub, expected_sub, n_levels
                )));
            }
            let mut lpc_pos = 2usize;
            let mut qs = Vec::with_capacity(n_sub);
            for _ in 0..n_sub {
                if lpc_pos + 4 > lpc_data.len() {
                    return Err(LmlError::Truncated {
                        expected: lpc_pos + 4,
                        actual: lpc_data.len(),
                        context: "track-2 quantizer steps",
                    });
                }
                let q = u32::from_le_bytes([
                    lpc_data[lpc_pos],
                    lpc_data[lpc_pos + 1],
                    lpc_data[lpc_pos + 2],
                    lpc_data[lpc_pos + 3],
                ]) as i64;
                if q < 1 {
                    return Err(LmlError::InvalidHeader("track-2 quantizer step < 1".into()));
                }
                qs.push(q);
                lpc_pos += 4;
            }
            let sub_lens = subband_lengths(t, n_levels);
            let mut sub_pos = 0usize;
            let mut signal = vec![vec![0i64; t]; n_ch];
            for ch in signal.iter_mut() {
                let mut subs: Vec<Vec<i64>> = Vec::with_capacity(n_sub);
                for sb_idx in 0..n_sub {
                    if lpc_pos >= lpc_data.len() {
                        return Err(LmlError::Truncated {
                            expected: lpc_pos + 1,
                            actual: lpc_data.len(),
                            context: "track-2 LPC order",
                        });
                    }
                    let order = lpc_data[lpc_pos] as usize;
                    lpc_pos += 1;
                    let mut coeffs = Vec::with_capacity(order);
                    for _ in 0..order {
                        if lpc_pos + 4 > lpc_data.len() {
                            return Err(LmlError::Truncated {
                                expected: lpc_pos + 4,
                                actual: lpc_data.len(),
                                context: "track-2 LPC coefficients",
                            });
                        }
                        coeffs.push(i32::from_le_bytes([
                            lpc_data[lpc_pos],
                            lpc_data[lpc_pos + 1],
                            lpc_data[lpc_pos + 2],
                            lpc_data[lpc_pos + 3],
                        ]));
                        lpc_pos += 4;
                    }
                    if sub_pos > payload.len() {
                        return Err(LmlError::InvalidHeader(format!(
                            "track-2 payload cursor {sub_pos} past end {}",
                            payload.len()
                        )));
                    }
                    let (residual, consumed) = decode_subband_payload(&payload[sub_pos..], 0)?;
                    sub_pos += consumed;
                    let want = sub_lens.get(sb_idx).copied().unwrap_or(0);
                    if residual.len() != want {
                        return Err(LmlError::InvalidHeader(format!(
                            "track-2 subband {} decoded {} samples, expected {}",
                            sb_idx,
                            residual.len(),
                            want
                        )));
                    }
                    let idx = lpc::synthesize(&residual, &coeffs, order, BIAS_CTX);
                    subs.push(quant::dequantize(&idx, qs[sb_idx]));
                }
                let mut recon = quant::inverse_for_levels(n_levels, &subs);
                recon.resize(t, 0); // forward/inverse preserve length; guard anyway
                *ch = recon;
            }
            Ok(signal)
        }
        other => Err(LmlError::InvalidHeader(format!(
            "unknown track-2 mode id 0x{:02X}",
            other
        ))),
    }
}

/// Convenience wrapper: panics on invalid input. Use only in tests where
/// invalid input is intentional, never in production code.
#[cfg(test)]
pub(crate) fn compress_or_panic(signal: &[Vec<i64>], noise_bits: u8) -> Vec<u8> {
    compress(signal, noise_bits).expect("compress: invalid input")
}

/// Pick the lifting-DWT depth for a window of `t` samples.
///
/// Starts at 3 levels (the encoder's nominal choice) and steps down
/// until each level still has at least 4 samples on its smallest
/// subband (Burg's `4 * 2^n_levels` rule). Both serial and parallel
/// encoders MUST agree on this number — otherwise the same input
/// produces different `n_levels` in the header and the byte-equal
/// invariant breaks.
fn compute_n_levels(t: usize) -> u8 {
    let mut n_levels: u8 = 3;
    while (t as u64) < 4 * (1u64 << n_levels) && n_levels > 0 {
        n_levels -= 1;
    }
    n_levels
}

/// Encode a single channel into its `(lpc_meta_bytes, payload_bytes)`
/// pair. Pure function — no shared state — which is why the host
/// parallel variant can dispatch this across rayon workers and still
/// produce byte-identical output to the serial path.
///
/// The serial `compress_with_mode` calls this in a `for` loop; the
/// host-only `compress_with_mode_parallel` calls it via
/// `par_iter().map(...).collect()`, which preserves input order, so
/// the concatenated bytes match the serial path exactly. The
/// `tests/byte_equal_backends.rs` gate locks that invariant.
/// Per-subband encoding result for the experimental selection
/// machinery. Holds the always-available Golomb bytes plus the
/// winning encoder's bytes when a non-Golomb encoder strictly beats
/// Golomb on size.
#[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
struct SubbandResult {
    golomb_bytes: alloc::vec::Vec<u8>,
    winner_tag: u8,
    winner_bytes: Option<alloc::vec::Vec<u8>>,
}

/// Per-channel encode result — feature-gated for shape. When neither
/// experimental feature is compiled, this collapses to
/// `(meta, payload)` (same layout as pre-B1, no per-subband
/// allocations) so rayon's `collect` on `Vec<ChannelEncodeOutput>`
/// moves only ~48 bytes of header per channel — matching the pre-B1
/// throughput on multi-channel desktop-parallel encode.
#[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
struct ChannelEncodeOutput {
    meta: Vec<u8>,
    subbands: Vec<SubbandResult>,
    n_subbands: usize,
    bp_savings: usize,
}

#[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
struct ChannelEncodeOutput {
    meta: Vec<u8>,
    payload: Vec<u8>,
}

/// ADR 0023 Track B1+: lazy payload assembly. Compiled only when at
/// least one experimental codec is enabled; the default-feature
/// build inlines payload concatenation in `compress_with_mode`
/// directly to dodge the per-channel-output-Vec overhead that
/// regressed desktop-parallel encode by ~28 % vs pre-B1.
#[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
fn assemble_payload(per_channel: &[ChannelEncodeOutput], use_tagged: bool) -> Vec<u8> {
    // Pre-size: legacy = sum(golomb_bytes), tagged = sum(winner_bytes or
    // golomb_bytes) + 1B/subband for tags.
    let mut total: usize = 0;
    for ch in per_channel {
        for sb in &ch.subbands {
            if use_tagged {
                let chosen_len = sb
                    .winner_bytes
                    .as_ref()
                    .map(|b| b.len())
                    .unwrap_or(sb.golomb_bytes.len());
                total += 1 + chosen_len;
            } else {
                total += sb.golomb_bytes.len();
            }
        }
    }
    let mut payload = Vec::with_capacity(total);
    for ch in per_channel {
        for sb in &ch.subbands {
            if use_tagged {
                payload.push(sb.winner_tag);
                if let Some(bytes) = &sb.winner_bytes {
                    payload.extend_from_slice(bytes);
                } else {
                    payload.extend_from_slice(&sb.golomb_bytes);
                }
            } else {
                payload.extend_from_slice(&sb.golomb_bytes);
            }
        }
    }
    payload
}

fn encode_one_channel(
    signal_ch: &[i64],
    n_levels: u8,
    mode: lpc::LpcMode,
    try_arithmetic: bool,
    try_bit_pack: bool,
    try_extended_lpc: bool,
) -> LmlResult<ChannelEncodeOutput> {
    let subbands: Vec<Vec<i64>> = match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(signal_ch);
            vec![a3, d3, d2, d1]
        }
        2 => {
            let (l1a, l1d) = lifting::forward(signal_ch);
            let (l2a, l2d) = lifting::forward(&l1a);
            vec![l2a, l2d, l1d]
        }
        1 => {
            let (a, d) = lifting::forward(signal_ch);
            vec![a, d]
        }
        _ => vec![signal_ch.to_vec()],
    };

    let mut local_meta = Vec::with_capacity(subbands.len() * 40);
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let mut sb_results: Vec<SubbandResult> = Vec::with_capacity(subbands.len());
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    let mut local_payload: Vec<u8> = Vec::with_capacity(signal_ch.len() * 4);
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let mut bp_savings: usize = 0;
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let n_subbands = subbands.len();

    for (sb_idx, sub) in subbands.iter().enumerate() {
        // Per-subband AIC ceiling (Burg's N/8 rule). The mode wrapper
        // clamps internally — for `Fixed` we ignore this and use the
        // legacy `[3,3,6,8]` schedule via `fixed_order_for_subband`.
        //
        // Track B6 (opt-in): swap the ceiling from 16 → 32 so the
        // AIC walk explores deeper. Default off → byte-equal to
        // the pre-B6 path.
        #[cfg(feature = "host")]
        let burg_ceiling = if try_extended_lpc {
            lpc_max_order_extended(sub.len())
        } else {
            lpc_max_order(sub.len())
        };
        #[cfg(not(feature = "host"))]
        let burg_ceiling = {
            let _ = try_extended_lpc;
            lpc_max_order(sub.len())
        };

        // Substitute the user-requested max_order into the mode's
        // adaptive ceiling. The wrapper itself decides whether
        // adaptive runs (e.g. Anytime checks the deadline first).
        let scoped_mode = scope_lpc_mode(mode, burg_ceiling);

        let (coeffs, residual, chosen_order) = lpc::analyze_with_mode(
            sub,
            sb_idx,
            scoped_mode,
            BIAS_CTX,
            /* time_remaining = */ None,
        );

        local_meta.push(chosen_order as u8);
        for &c in &coeffs {
            local_meta.extend_from_slice(&c.to_le_bytes());
        }

        let golomb_bytes = golomb::encode_dense(&residual)?;

        // ADR 0023 Track B1/B5+: experimental per-subband candidate
        // selection. Cfg-gated so default builds (no experimental
        // feature compiled) write the Golomb bytes directly into a
        // flat per-channel payload — same hot-path memory layout as
        // the pre-B1 codec; no SubbandResult struct, no per-subband
        // Vec, no max_abs scan.
        #[cfg(not(any(
            feature = "experimental_bit_pack",
            feature = "experimental_arithmetic"
        )))]
        {
            let _ = try_arithmetic;
            let _ = try_bit_pack;
            local_payload.extend_from_slice(&golomb_bytes);
            continue;
        }
        #[cfg(any(
            feature = "experimental_bit_pack",
            feature = "experimental_arithmetic"
        ))]
        let (winner_tag, winner_bytes, savings): (u8, Option<Vec<u8>>, usize) =
            if try_arithmetic || try_bit_pack {
                // Cheap pre-check before the full bit-pack sizing
                // pass: scan residuals once for their max absolute
                // value. If the extreme requires > ~12 bits after
                // zigzag, neither bit-pack nor the Laplace-modelled
                // arithmetic coder beats Golomb-Rice; skip the
                // sizing pass entirely so wide-residual subbands
                // (clinical 32-channel data) don't pay the probe
                // cost.
                let max_abs: i64 = residual
                    .iter()
                    .map(|&v| if v == i64::MIN { i64::MAX } else { v.abs() })
                    .fold(0i64, |a, b| a.max(b));
                #[cfg(feature = "experimental_bit_pack")]
                let bp_size = if try_bit_pack && max_abs <= (1i64 << 12) {
                    bit_pack::encoded_byte_len(&residual).unwrap_or(usize::MAX)
                } else {
                    usize::MAX
                };
                #[cfg(not(feature = "experimental_bit_pack"))]
                let bp_size = usize::MAX;

                #[cfg(feature = "experimental_arithmetic")]
                let ac_bytes_opt: Option<Vec<u8>> =
                    if try_arithmetic && max_abs <= (1i64 << 12) {
                        Some(crate::arithmetic::encode_dense(&residual))
                    } else {
                        None
                    };
                #[cfg(not(feature = "experimental_arithmetic"))]
                let ac_bytes_opt: Option<Vec<u8>> = None;
                let ac_size = ac_bytes_opt.as_ref().map(|b| b.len()).unwrap_or(usize::MAX);

                let golomb_size = golomb_bytes.len();
                if ac_size < golomb_size && ac_size < bp_size {
                    #[cfg(feature = "experimental_arithmetic")]
                    {
                        let ac_bytes = ac_bytes_opt
                            .expect("ac_bytes present when ac_size < ∞");
                        (SUBBAND_TAG_ARITHMETIC, Some(ac_bytes), golomb_size - ac_size)
                    }
                    #[cfg(not(feature = "experimental_arithmetic"))]
                    {
                        unreachable!("arithmetic never picked without feature");
                    }
                } else if bp_size < golomb_size {
                    #[cfg(feature = "experimental_bit_pack")]
                    {
                        let bp_bytes = bit_pack::encode_dense(&residual).map_err(|e| {
                            LmlError::InvalidHeader(alloc::format!(
                                "bit_pack::encode_dense failed: {}",
                                e
                            ))
                        })?;
                        (SUBBAND_TAG_BIT_PACK, Some(bp_bytes), golomb_size - bp_size)
                    }
                    #[cfg(not(feature = "experimental_bit_pack"))]
                    {
                        unreachable!("bit_pack never picked without feature");
                    }
                } else {
                    (SUBBAND_TAG_GOLOMB, None, 0)
                }
            } else {
                // Experimental feature compiled but env vars not set:
                // skip the per-subband selection at runtime. Same
                // bytes as default build.
                (SUBBAND_TAG_GOLOMB, None, 0)
            };
        #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
        {
            bp_savings += savings;
            sb_results.push(SubbandResult {
                golomb_bytes,
                winner_tag,
                winner_bytes,
            });
        }
    }

    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    return Ok(ChannelEncodeOutput {
        meta: local_meta,
        subbands: sb_results,
        n_subbands,
        bp_savings,
    });
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    Ok(ChannelEncodeOutput {
        meta: local_meta,
        payload: local_payload,
    })
}

/// Host-only parallel compress: same wire-format output as
/// [`compress_with_mode`], computed via rayon across channels.
///
/// Channels are independent through every stage (lifting + LPC +
/// golomb), so this is an embarrassingly-parallel transformation.
/// `par_iter().map().collect()` preserves the input order, so the
/// concatenated `lpc_meta + payload` is byte-equal to the serial
/// path. The `byte_equal_backends.rs` conformance gate enforces it.
///
/// Returns identical bytes to `compress_with_mode`; the only
/// difference is wall-clock time on multi-channel inputs (typical
/// clinical EEG = 8-24 channels = 4-12× speedup on a 16-core host).
#[cfg(feature = "host")]
pub fn compress_with_mode_parallel(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: lpc::LpcMode,
) -> LmlResult<Vec<u8>> {
    use rayon::prelude::*;

    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!(
            "n_ch={} out of range 1..=1024",
            n_ch
        )));
    }
    if t == 0 || t > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!(
            "T={} out of range 1..={}",
            t,
            u16::MAX
        )));
    }
    if noise_bits > 32 {
        return Err(LmlError::InvalidHeader(format!(
            "noise_bits={} exceeds max 32",
            noise_bits
        )));
    }

    let owned_signal: Vec<Vec<i64>>;
    let signal_ref: &[Vec<i64>] = if noise_bits > 0 {
        owned_signal = signal
            .iter()
            .map(|ch| ch.iter().map(|&v| v >> noise_bits).collect())
            .collect();
        &owned_signal
    } else {
        signal
    };

    let n_levels = compute_n_levels(t);

    // Parallel per-channel encode. Output order is preserved because
    // `par_iter().map(...).collect::<Result<Vec<_>>>()` keeps the
    // input order. `collect::<Result<...>>()` short-circuits on the
    // first Err -- rayon stops scheduling new work on the channels
    // that haven't started yet, propagating the error up.
    #[cfg(feature = "experimental_arithmetic")]
    let try_arithmetic = experimental_arithmetic_enabled();
    #[cfg(not(feature = "experimental_arithmetic"))]
    let try_arithmetic = false;
    #[cfg(feature = "experimental_bit_pack")]
    let try_bit_pack = experimental_bit_pack_enabled();
    #[cfg(not(feature = "experimental_bit_pack"))]
    let try_bit_pack = false;
    let try_extended_lpc = experimental_extended_lpc_enabled();
    let per_channel: Vec<ChannelEncodeOutput> = (0..n_ch)
        .into_par_iter()
        .map(|ch_idx| {
            encode_one_channel(
                &signal_ref[ch_idx],
                n_levels,
                mode,
                try_arithmetic,
                try_bit_pack,
                try_extended_lpc,
            )
        })
        .collect::<LmlResult<Vec<_>>>()?;

    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let total_n_subbands: usize = per_channel.iter().map(|c| c.n_subbands).sum();
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let total_bp_savings: usize = per_channel.iter().map(|c| c.bp_savings).sum();
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let any_bit_pack_wins_global = total_bp_savings > total_n_subbands;
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    let any_bit_pack_wins_global = false;
    let lpc_meta_total: usize = per_channel.iter().map(|c| c.meta.len()).sum();

    let mut lpc_meta = Vec::with_capacity(lpc_meta_total);
    for c in &per_channel {
        lpc_meta.extend_from_slice(&c.meta);
    }
    #[cfg(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic"))]
    let payload = assemble_payload(&per_channel, any_bit_pack_wins_global);
    #[cfg(not(any(feature = "experimental_bit_pack", feature = "experimental_arithmetic")))]
    let payload: Vec<u8> = {
        let total: usize = per_channel.iter().map(|c| c.payload.len()).sum();
        let mut p = Vec::with_capacity(total);
        for c in &per_channel {
            p.extend_from_slice(&c.payload);
        }
        p
    };

    let mut flags: u8 = (noise_bits & 0x3F) << 2;
    if any_bit_pack_wins_global {
        flags |= FLAG_BIT_PER_SUBBAND_TAG;
    }
    let mut header_var = [0u8; 14];
    header_var[0..2].copy_from_slice(&(n_ch as u16).to_le_bytes());
    header_var[2..4].copy_from_slice(&(t as u16).to_le_bytes());
    header_var[4] = n_levels;
    header_var[5] = flags;
    header_var[6..10].copy_from_slice(&(lpc_meta.len() as u32).to_le_bytes());
    header_var[10..14].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    let mut crc_state = CRC32_INIT;
    crc_state = crc32_update(crc_state, &header_var);
    crc_state = crc32_update(crc_state, &lpc_meta);
    crc_state = crc32_update(crc_state, &payload);
    let crc = crc_state ^ CRC32_INIT;

    let mode_label = if noise_bits == 0 { "lossless" } else { "noise" };
    let prefix = format!("LML | {}ch | {} | CRC-32\n", n_ch, mode_label);

    let total = prefix.len() + HEADER_SIZE + lpc_meta.len() + payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&header_var);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&lpc_meta);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Compress a signal into any [`std::io::Write`] sink.
///
/// Phase 0.4 wrapper around [`compress_with_mode`]: today this buffers
/// the full LML1 packet in `Vec<u8>` and `write_all`s it. Phase 0.5
/// will rewrite `compress_with_mode` itself to stream into the sink so
/// peak allocation drops to one window. The signature is the future
/// shape — callers can adopt it now without another rewrite later.
///
/// Bible R33 — the `LmlSink` trait's `before_window` hook (in
/// `crate::io`) is the future attachment point for backpressure once
/// per-window streaming lands; for now sinks just receive the full
/// packet.
///
/// Returns the number of bytes written.
#[cfg(feature = "std")]
pub fn compress_into<W: std::io::Write>(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: lpc::LpcMode,
    sink: &mut W,
) -> LmlResult<usize> {
    let bytes = compress_with_mode(signal, noise_bits, mode)?;
    sink.write_all(&bytes).map_err(LmlError::Io)?;
    Ok(bytes.len())
}

/// Decompress an LML1 packet from any [`std::io::Read`] source.
///
/// Phase 0.4 wrapper around [`decompress`]: today this `read_to_end`s
/// the source into `Vec<u8>` and decompresses. Partial-read sources
/// (one-byte-at-a-time pipes, `ByteAtATime` test adapter) work
/// correctly because `read_to_end` loops internally. Phase 0.5 will
/// teach the decoder to consume bytes incrementally so streaming
/// sources don't need to fit in memory.
///
/// Callers who already have `&[u8]` should keep using [`decompress`]
/// — this wrapper exists for the `LmlSource` / stdin / S3 path.
#[cfg(feature = "std")]
pub fn decompress_from<R: std::io::Read>(src: &mut R) -> LmlResult<Vec<Vec<i64>>> {
    let mut buf = Vec::new();
    src.read_to_end(&mut buf).map_err(LmlError::Io)?;
    decompress(&buf)
}

/// Decompress LML1 packet → [n_ch][T].
pub fn decompress(data: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    let offset = find_magic_offset(data)?;
    let data = &data[offset..];

    if data.len() < HEADER_SIZE {
        return Err(LmlError::Truncated {
            expected: HEADER_SIZE,
            actual: data.len(),
            context: "packet header",
        });
    }

    let magic = &data[0..4];
    if magic != MAGIC {
        // No backwards compatibility — production accepts only LML1.
        // Future versions are explicit; legacy iterations (LMQ4/LMQ5/LML )
        // route to LmlError::InvalidMagic and must be decoded out-of-band.
        if magic[0..3] == *b"LML" && magic[3].is_ascii_digit() && magic[3] != b'1' {
            return Err(LmlError::UnsupportedVersion(magic[3]));
        }
        let mut m = [0u8; 4];
        m.copy_from_slice(magic);
        return Err(LmlError::InvalidMagic(m));
    }

    let n_ch = u16::from_le_bytes([data[4], data[5]]) as usize;
    let t = u16::from_le_bytes([data[6], data[7]]) as usize;
    let n_levels = data[8];
    let flags = data[9];
    let lpc_len = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
    let sub_len = u32::from_le_bytes([data[14], data[15], data[16], data[17]]) as usize;
    let crc_exp = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);

    // ADR 0023 Track B1: bit 0 (`FLAG_BIT_PER_SUBBAND_TAG`) is a feature
    // flag. ADR 0051 track 2: bit 1 (`FLAG_BIT_TRACK2_MODE`) marks a
    // near-lossless / lossy packet — dispatched after CRC verify below. The
    // two are mutually exclusive (track-2 has no wavelet subbands), so a
    // packet asserting both is malformed → fail-closed.
    if (flags & FLAG_BIT_TRACK2_MODE != 0) && (flags & FLAG_BIT_PER_SUBBAND_TAG != 0) {
        return Err(LmlError::InvalidHeader(format!(
            "LML1 track-2 + per-subband-tag both set (flags=0x{:02X})",
            flags
        )));
    }
    let use_per_subband_tag = (flags & FLAG_BIT_PER_SUBBAND_TAG) != 0;
    let noise_bits = (flags >> 2) & 0x3F;

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    if t == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    // n_levels is the lifting-DWT depth; only 0..=3 are defined (the
    // synthesis match below handles 3/2/1 and 0). A crafted CRC-valid
    // packet with n_levels>=4 would otherwise decode n_levels+1 subbands
    // and fall through the `_` arm to return ONLY the first subband —
    // wrong-length output presented as Ok instead of an error. Reject it.
    // (The serial `decompress` and parallel `decompress_parallel` paths
    // share this identical guard.)
    if n_levels > 3 {
        return Err(LmlError::InvalidHeader(format!(
            "n_levels must be 0..=3, got {}",
            n_levels
        )));
    }
    if HEADER_SIZE + lpc_len + sub_len > data.len() {
        return Err(LmlError::Truncated {
            expected: HEADER_SIZE + lpc_len + sub_len,
            actual: data.len(),
            context: "payload",
        });
    }

    // CRC covers the variable-fields header (bytes 4..18) plus
    // lpc_meta + payload. Magic (bytes 0..4) is constant so excluded;
    // the CRC field itself (bytes 18..22) is excluded to avoid self-
    // reference. This matches the encoder side in `compress`.
    //
    // `verify_packet_crc` first checks the modern (a81cd04+) scope — the
    // common fast path, byte-for-byte the old cost. Only on mismatch does
    // it recompute the legacy payload-only scope to accept pre-a81cd04
    // packets (warn-once via SAW_LEGACY_CRC). Both miss → CrcMismatch.
    let payload_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len + sub_len];
    verify_packet_crc(&data[4..18], payload_data, crc_exp)?;

    // ADR 0051 track 2: near-lossless / lossy packet. CRC + size bounds are
    // verified above; dispatch to the integer-only mode decoder.
    if flags & FLAG_BIT_TRACK2_MODE != 0 {
        let lpc_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len];
        let payload = &data[HEADER_SIZE + lpc_len..HEADER_SIZE + lpc_len + sub_len];
        return decode_track2(n_ch, t, n_levels, lpc_data, payload);
    }

    let lpc_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len];
    let mut lpc_pos = 0usize;
    let mut sub_pos = HEADER_SIZE + lpc_len;
    // n_levels can be 0..3 (Python adaptively reduces for short windows).
    // Subbands: n_levels=3 → 4 (approx3 + detail3,2,1),
    //           n_levels=2 → 3 (approx2 + detail2,1),
    //           n_levels=1 → 2 (approx1 + detail1),
    //           n_levels=0 → 1 (raw signal, no DWT)
    let n_sub = if n_levels == 0 {
        1
    } else {
        (n_levels as usize) + 1
    };

    // Audit-2026-05-11 Fix-C4: bound output allocation against
    // MAX_DECODE_BYTES so an attacker-controlled `n_ch * t` cannot
    // trigger a multi-hundred-MB malloc on a tiny packet. CRC having
    // passed earlier only proves the bytes the attacker supplied are
    // self-consistent; it says nothing about the implied decoded size.
    let total_bytes = (n_ch as u64)
        .checked_mul(t as u64)
        .and_then(|p| p.checked_mul(8))
        .ok_or_else(|| {
            LmlError::InvalidHeader(alloc::format!("n_ch={} × t={} × 8 overflows u64", n_ch, t))
        })?;
    if total_bytes > MAX_DECODE_BYTES {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "decoded size {total_bytes} bytes > MAX_DECODE_BYTES {MAX_DECODE_BYTES}"
        )));
    }

    let mut signal = vec![vec![0i64; t]; n_ch];
    // ADR 0023 Track B1: only computed (and required) on packets
    // that opt in via FLAG_BIT_PER_SUBBAND_TAG. Bit-pack decode
    // needs the sample count per subband out-of-band.
    let sub_lens: Vec<usize> = if use_per_subband_tag {
        subband_lengths(t, n_levels)
    } else {
        Vec::new()
    };

    for ch in 0..n_ch {
        let mut subs: Vec<Vec<i64>> = Vec::with_capacity(n_sub);

        for sb_idx in 0..n_sub {
            if lpc_pos >= lpc_data.len() {
                return Err(LmlError::Truncated {
                    expected: lpc_pos + 1,
                    actual: lpc_data.len(),
                    context: "LPC metadata",
                });
            }
            let order = lpc_data[lpc_pos] as usize;
            lpc_pos += 1;

            let mut coeffs = Vec::with_capacity(order);
            for _ in 0..order {
                if lpc_pos + 4 > lpc_data.len() {
                    return Err(LmlError::Truncated {
                        expected: lpc_pos + 4,
                        actual: lpc_data.len(),
                        context: "LPC coefficients",
                    });
                }
                coeffs.push(i32::from_le_bytes([
                    lpc_data[lpc_pos],
                    lpc_data[lpc_pos + 1],
                    lpc_data[lpc_pos + 2],
                    lpc_data[lpc_pos + 3],
                ]));
                lpc_pos += 4;
            }

            // Audit-2026-05-11 Fix-C5: bound golomb decoder to the
            // declared subband region. Without this bound the decoder
            // can read past `sub_len` into the next subband (or off the
            // packet entirely) and produce internally-consistent garbage
            // that decodes as `Ok` — a CRC-recompute attack would let
            // a crafted packet smuggle decoded values across subband
            // boundaries.
            let payload_end = HEADER_SIZE + lpc_len + sub_len;
            if sub_pos > payload_end {
                return Err(LmlError::InvalidHeader(alloc::format!(
                    "subband cursor {sub_pos} past declared payload end {payload_end}"
                )));
            }
            // ADR 0023 Track B1: dispatch on per-subband tag when the
            // header opted into the new framing. Legacy framing (no
            // tag) is the pre-B1 path — byte-equal to old archives.
            let decoded = if use_per_subband_tag {
                if sub_pos >= payload_end {
                    return Err(LmlError::Truncated {
                        expected: sub_pos + 1,
                        actual: payload_end,
                        context: "per-subband codec tag",
                    });
                }
                let tag = data[sub_pos];
                sub_pos += 1;
                let sub_slice = &data[sub_pos..payload_end];
                match tag {
                    SUBBAND_TAG_GOLOMB => {
                        let (d, consumed) = golomb::decode_dense(sub_slice, 0)?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(alloc::format!(
                                "golomb decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    SUBBAND_TAG_BIT_PACK => {
                        let n_samples = sub_lens
                            .get(sb_idx)
                            .copied()
                            .ok_or_else(|| LmlError::InvalidHeader(alloc::format!(
                                "no subband length for index {} (n_sub={})", sb_idx, n_sub
                            )))?;
                        let (d, consumed) = bit_pack::decode_dense(sub_slice, n_samples)
                            .map_err(|e| LmlError::InvalidHeader(alloc::format!(
                                "bit_pack::decode_dense failed: {}", e
                            )))?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(alloc::format!(
                                "bit_pack decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    #[cfg(feature = "experimental_arithmetic")]
                    SUBBAND_TAG_ARITHMETIC => {
                        let n_samples = sub_lens
                            .get(sb_idx)
                            .copied()
                            .ok_or_else(|| LmlError::InvalidHeader(alloc::format!(
                                "no subband length for index {} (n_sub={})", sb_idx, n_sub
                            )))?;
                        let (d, consumed) = crate::arithmetic::decode_dense(sub_slice, n_samples)
                            .map_err(|e| LmlError::InvalidHeader(alloc::format!(
                                "arithmetic::decode_dense failed: {}", e
                            )))?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(alloc::format!(
                                "arithmetic decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    _ => {
                        return Err(LmlError::InvalidHeader(alloc::format!(
                            "unknown per-subband codec tag 0x{:02X}", tag
                        )));
                    }
                }
            } else {
                let sub_slice = &data[sub_pos..payload_end];
                let (d, consumed) = golomb::decode_dense(sub_slice, 0)?;
                sub_pos += consumed;
                if sub_pos > payload_end {
                    return Err(LmlError::InvalidHeader(alloc::format!(
                        "golomb decoder consumed {consumed} bytes, past declared payload end"
                    )));
                }
                d
            };

            // Always run synthesize — even order=0 needs bias_restore
            subs.push(lpc::synthesize(&decoded, &coeffs, order, BIAS_CTX));
        }

        signal[ch] = match n_levels {
            3 => lifting::inverse_3level(&subs[0], &subs[1], &subs[2], &subs[3]),
            2 => {
                // 2-level: subs = [l2_approx, l2_detail, l1_detail]
                let l1_approx = lifting::inverse(&subs[0], &subs[1]);
                lifting::inverse(&l1_approx, &subs[2])
            }
            1 => {
                // 1-level: subs = [approx, detail]
                lifting::inverse(&subs[0], &subs[1])
            }
            _ => {
                // 0 levels: raw signal, single subband.
                // Audit-2026-05-11 Fix-#42: previous `unwrap_or_default()`
                // silently returned an empty Vec on a corrupt manifest
                // where `n_sub` claimed 1 subband but decode produced
                // zero — that fed a zero-filled signal downstream
                // without diagnostic. Return Err so the caller learns.
                subs.into_iter().next().ok_or_else(|| {
                    LmlError::InvalidHeader("n_levels=0 but no subband decoded".into())
                })?
            }
        };
    }

    if noise_bits > 0 {
        for ch in signal.iter_mut() {
            for v in ch.iter_mut() {
                *v <<= noise_bits;
            }
        }
    }

    Ok(signal)
}

/// Synthesize one channel's signal from its per-subband
/// `(coeffs, residual)` pairs. Pure function -- no shared state.
///
/// Mirrors the body of the inner per-channel loop in `decompress`
/// (lpc::synthesize each subband + lifting::inverse_Nlevel + the
/// n_levels=0 edge case). Extracted so the parallel decoder can
/// dispatch this work across rayon workers while the byte-equal
/// invariant is preserved by construction.
#[cfg(feature = "host")]
fn synthesize_channel_signal(
    per_subband: Vec<(Vec<i32>, Vec<i64>)>,
    n_levels: u8,
) -> LmlResult<Vec<i64>> {
    let subs: Vec<Vec<i64>> = per_subband
        .into_iter()
        .map(|(coeffs, residual)| {
            let order = coeffs.len();
            lpc::synthesize(&residual, &coeffs, order, BIAS_CTX)
        })
        .collect();
    Ok(match n_levels {
        3 => lifting::inverse_3level(&subs[0], &subs[1], &subs[2], &subs[3]),
        2 => {
            let l1_approx = lifting::inverse(&subs[0], &subs[1]);
            lifting::inverse(&l1_approx, &subs[2])
        }
        1 => lifting::inverse(&subs[0], &subs[1]),
        _ => subs
            .into_iter()
            .next()
            .ok_or_else(|| LmlError::InvalidHeader("n_levels=0 but no subband decoded".into()))?,
    })
}

/// Host-only parallel decompress. Byte-equal output to the serial
/// `decompress` path; the parallelism happens in the heaviest step
/// (LPC synthesize + lifting inverse) per channel via rayon.
///
/// Two phases:
///
///   1. **Serial parse** -- walk the input cursor through `lpc_data`
///      + golomb-encoded payload, decoding coeffs + residuals per
///      `(channel, subband)`. Has to be sequential because each
///      subband's byte length is determined by golomb decode (no
///      upfront index).
///   2. **Parallel synth + lift** -- for each channel, dispatch
///      `synthesize_channel_signal(subs, n_levels)` across rayon
///      workers. Output order preserved by
///      `par_iter().map(...).collect()`.
///
/// The byte-equal cross-backend gate (`tests/byte_equal_backends.rs`)
/// proves the parallel output matches the serial path for every
/// fixture.
#[cfg(feature = "host")]
pub fn decompress_parallel(data: &[u8]) -> LmlResult<Vec<Vec<i64>>> {
    use rayon::prelude::*;

    let offset = find_magic_offset(data)?;
    let data = &data[offset..];

    if data.len() < HEADER_SIZE {
        return Err(LmlError::Truncated {
            expected: HEADER_SIZE,
            actual: data.len(),
            context: "packet header",
        });
    }

    let magic = &data[0..4];
    if magic != MAGIC {
        if magic[0..3] == *b"LML" && magic[3].is_ascii_digit() && magic[3] != b'1' {
            return Err(LmlError::UnsupportedVersion(magic[3]));
        }
        let mut m = [0u8; 4];
        m.copy_from_slice(magic);
        return Err(LmlError::InvalidMagic(m));
    }

    let n_ch = u16::from_le_bytes([data[4], data[5]]) as usize;
    let t = u16::from_le_bytes([data[6], data[7]]) as usize;
    let n_levels = data[8];
    let flags = data[9];
    let lpc_len = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
    let sub_len = u32::from_le_bytes([data[14], data[15], data[16], data[17]]) as usize;
    let crc_exp = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);

    // ADR 0023 Track B1: bit 0 (`FLAG_BIT_PER_SUBBAND_TAG`) is now
    // a defined feature flag. Bit 1 stays reserved.
    // ADR 0051 track 2: bit 1 is the track-2 mode flag (dispatched after
    // CRC verify below); mutually exclusive with the per-subband-tag framing.
    if (flags & FLAG_BIT_TRACK2_MODE != 0) && (flags & FLAG_BIT_PER_SUBBAND_TAG != 0) {
        return Err(LmlError::InvalidHeader(format!(
            "LML1 track-2 + per-subband-tag both set (flags=0x{:02X})",
            flags
        )));
    }
    let use_per_subband_tag = (flags & FLAG_BIT_PER_SUBBAND_TAG) != 0;
    let noise_bits = (flags >> 2) & 0x3F;

    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    if t == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    // n_levels is the lifting-DWT depth; only 0..=3 are defined (the
    // synthesis match below handles 3/2/1 and 0). A crafted CRC-valid
    // packet with n_levels>=4 would otherwise decode n_levels+1 subbands
    // and fall through the `_` arm to return ONLY the first subband —
    // wrong-length output presented as Ok instead of an error. Reject it.
    // (The serial `decompress` and parallel `decompress_parallel` paths
    // share this identical guard.)
    if n_levels > 3 {
        return Err(LmlError::InvalidHeader(format!(
            "n_levels must be 0..=3, got {}",
            n_levels
        )));
    }
    if HEADER_SIZE + lpc_len + sub_len > data.len() {
        return Err(LmlError::Truncated {
            expected: HEADER_SIZE + lpc_len + sub_len,
            actual: data.len(),
            context: "payload",
        });
    }

    // Same modern-then-legacy CRC verification as the sequential
    // `decompress` path (see `verify_packet_crc`). Decode-side only;
    // the encoder still writes the modern header+payload scope.
    let payload_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len + sub_len];
    verify_packet_crc(&data[4..18], payload_data, crc_exp)?;

    // ADR 0051 track 2: near-lossless / lossy packet. CRC + size bounds are
    // verified above; dispatch to the integer-only mode decoder.
    if flags & FLAG_BIT_TRACK2_MODE != 0 {
        let lpc_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len];
        let payload = &data[HEADER_SIZE + lpc_len..HEADER_SIZE + lpc_len + sub_len];
        return decode_track2(n_ch, t, n_levels, lpc_data, payload);
    }

    let lpc_data = &data[HEADER_SIZE..HEADER_SIZE + lpc_len];
    let mut lpc_pos = 0usize;
    let mut sub_pos = HEADER_SIZE + lpc_len;
    let n_sub = if n_levels == 0 {
        1
    } else {
        (n_levels as usize) + 1
    };

    let total_bytes = (n_ch as u64)
        .checked_mul(t as u64)
        .and_then(|p| p.checked_mul(8))
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!("n_ch={} × t={} × 8 overflows u64", n_ch, t))
        })?;
    if total_bytes > MAX_DECODE_BYTES {
        return Err(LmlError::InvalidHeader(format!(
            "decoded size {total_bytes} bytes > MAX_DECODE_BYTES {MAX_DECODE_BYTES}"
        )));
    }

    // Phase 1: sequential parse. Each entry of `per_channel` is
    // a Vec<(coeffs, residual)> -- one per subband for that channel.
    let mut per_channel: Vec<Vec<(Vec<i32>, Vec<i64>)>> = Vec::with_capacity(n_ch);
    // ADR 0023 Track B1: only needed on packets that opted in.
    let sub_lens: Vec<usize> = if use_per_subband_tag {
        subband_lengths(t, n_levels)
    } else {
        Vec::new()
    };
    for _ch in 0..n_ch {
        let mut subs: Vec<(Vec<i32>, Vec<i64>)> = Vec::with_capacity(n_sub);
        for sb_idx in 0..n_sub {
            if lpc_pos >= lpc_data.len() {
                return Err(LmlError::Truncated {
                    expected: lpc_pos + 1,
                    actual: lpc_data.len(),
                    context: "LPC metadata",
                });
            }
            let order = lpc_data[lpc_pos] as usize;
            lpc_pos += 1;
            let mut coeffs = Vec::with_capacity(order);
            for _ in 0..order {
                if lpc_pos + 4 > lpc_data.len() {
                    return Err(LmlError::Truncated {
                        expected: lpc_pos + 4,
                        actual: lpc_data.len(),
                        context: "LPC coefficients",
                    });
                }
                coeffs.push(i32::from_le_bytes([
                    lpc_data[lpc_pos],
                    lpc_data[lpc_pos + 1],
                    lpc_data[lpc_pos + 2],
                    lpc_data[lpc_pos + 3],
                ]));
                lpc_pos += 4;
            }
            let payload_end = HEADER_SIZE + lpc_len + sub_len;
            if sub_pos > payload_end {
                return Err(LmlError::InvalidHeader(format!(
                    "subband cursor {sub_pos} past declared payload end {payload_end}"
                )));
            }
            // ADR 0023 Track B1: per-subband tag dispatch when the
            // packet opted in. Legacy framing is the pre-B1 path.
            let decoded = if use_per_subband_tag {
                if sub_pos >= payload_end {
                    return Err(LmlError::Truncated {
                        expected: sub_pos + 1,
                        actual: payload_end,
                        context: "per-subband codec tag",
                    });
                }
                let tag = data[sub_pos];
                sub_pos += 1;
                let sub_slice = &data[sub_pos..payload_end];
                match tag {
                    SUBBAND_TAG_GOLOMB => {
                        let (d, consumed) = golomb::decode_dense(sub_slice, 0)?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(format!(
                                "golomb decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    SUBBAND_TAG_BIT_PACK => {
                        let n_samples = sub_lens.get(sb_idx).copied().ok_or_else(|| {
                            LmlError::InvalidHeader(format!(
                                "no subband length for index {} (n_sub={})", sb_idx, n_sub
                            ))
                        })?;
                        let (d, consumed) =
                            bit_pack::decode_dense(sub_slice, n_samples).map_err(|e| {
                                LmlError::InvalidHeader(format!(
                                    "bit_pack::decode_dense failed: {}", e
                                ))
                            })?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(format!(
                                "bit_pack decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    #[cfg(feature = "experimental_arithmetic")]
                    SUBBAND_TAG_ARITHMETIC => {
                        let n_samples = sub_lens.get(sb_idx).copied().ok_or_else(|| {
                            LmlError::InvalidHeader(format!(
                                "no subband length for index {} (n_sub={})", sb_idx, n_sub
                            ))
                        })?;
                        let (d, consumed) =
                            crate::arithmetic::decode_dense(sub_slice, n_samples).map_err(|e| {
                                LmlError::InvalidHeader(format!(
                                    "arithmetic::decode_dense failed: {}", e
                                ))
                            })?;
                        sub_pos += consumed;
                        if sub_pos > payload_end {
                            return Err(LmlError::InvalidHeader(format!(
                                "arithmetic decoder consumed {consumed} bytes, past declared payload end"
                            )));
                        }
                        d
                    }
                    _ => {
                        return Err(LmlError::InvalidHeader(format!(
                            "unknown per-subband codec tag 0x{:02X}", tag
                        )));
                    }
                }
            } else {
                let sub_slice = &data[sub_pos..payload_end];
                let (d, consumed) = golomb::decode_dense(sub_slice, 0)?;
                sub_pos += consumed;
                if sub_pos > payload_end {
                    return Err(LmlError::InvalidHeader(format!(
                        "golomb decoder consumed {consumed} bytes, past declared payload end"
                    )));
                }
                d
            };
            subs.push((coeffs, decoded));
        }
        per_channel.push(subs);
    }

    // Phase 2: parallel synth + lifting inverse. Order-preserving
    // collect keeps the channel-wise output stable.
    let signal: Result<Vec<Vec<i64>>, LmlError> = per_channel
        .into_par_iter()
        .map(|subs| synthesize_channel_signal(subs, n_levels))
        .collect();
    let mut signal = signal?;

    if noise_bits > 0 {
        for ch in signal.iter_mut() {
            for v in ch.iter_mut() {
                *v <<= noise_bits;
            }
        }
    }

    Ok(signal)
}

fn find_magic_offset(data: &[u8]) -> LmlResult<usize> {
    if data.len() < 4 {
        return Err(LmlError::Truncated {
            expected: 4,
            actual: data.len(),
            context: "magic bytes",
        });
    }
    if &data[0..4] == MAGIC {
        return Ok(0);
    }
    for i in 0..data.len().min(128) {
        if data[i] == b'\n'
            && i + 4 < data.len()
            && &data[i + 1..i + 4] == b"LML"
            && data[..i].iter().all(|&b| (0x20..=0x7E).contains(&b))
        {
            return Ok(i + 1);
        }
    }
    let mut m = [0u8; 4];
    m.copy_from_slice(&data[0..4]);
    Err(LmlError::InvalidMagic(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_21ch() {
        let signal: Vec<Vec<i64>> = (0..21)
            .map(|ch| {
                (0..2500)
                    .map(|i| ((i * (ch + 1) * 47) % 10000 - 5000) as i64)
                    .collect()
            })
            .collect();
        let c = compress(&signal, 0).unwrap();
        let r = decompress(&c).unwrap();
        assert_eq!(signal, r);
    }

    #[test]
    fn roundtrip_short() {
        let s = vec![vec![100i64, -200, 300, -400]];
        assert_eq!(s, decompress(&compress(&s, 0).unwrap()).unwrap());
    }

    #[test]
    fn roundtrip_noise() {
        let s: Vec<Vec<i64>> = vec![(0..2500)
            .map(|i| ((i * 137) % 50000 - 25000) as i64)
            .collect()];
        let r = decompress(&compress(&s, 4).unwrap()).unwrap();
        let expected: Vec<i64> = s[0].iter().map(|&v| (v >> 4) << 4).collect();
        assert_eq!(expected, r[0]);
    }

    #[test]
    fn compress_into_writes_to_vec_sink() {
        let signal = vec![vec![1i64, 2, 3, 4, 5]];
        let direct = compress(&signal, 0).unwrap();
        let mut sink: Vec<u8> = Vec::new();
        let written = compress_into(&signal, 0, lpc::LpcMode::default(), &mut sink).unwrap();
        assert_eq!(written, direct.len(), "byte count must match direct call");
        assert_eq!(sink, direct, "sink output must be byte-identical");
    }

    #[test]
    fn decompress_from_reads_from_cursor() {
        let signal: Vec<Vec<i64>> = (0..4)
            .map(|ch| (0..256).map(|i| ((i * (ch + 1)) % 1000) as i64).collect())
            .collect();
        let bytes = compress(&signal, 0).unwrap();
        let mut cursor = std::io::Cursor::new(&bytes);
        let recovered = decompress_from(&mut cursor).unwrap();
        assert_eq!(signal, recovered);
    }

    #[test]
    fn decompress_from_handles_partial_reads() {
        // The `io::ByteAtATime` adapter forces one-byte-per-read; the
        // generic decompress_from must still recover the full signal.
        let signal = vec![vec![42i64; 128]];
        let bytes = compress(&signal, 0).unwrap();
        let mut src = crate::io::tests::ByteAtATime::new(&bytes);
        let recovered = decompress_from(&mut src).unwrap();
        assert_eq!(signal, recovered);
    }

    #[test]
    fn compress_into_then_decompress_from_roundtrip() {
        let signal: Vec<Vec<i64>> = (0..3)
            .map(|ch| {
                (0..512)
                    .map(|i| ((i * 7 + ch * 11) % 5000) as i64)
                    .collect()
            })
            .collect();
        let mut sink: Vec<u8> = Vec::new();
        compress_into(&signal, 0, lpc::LpcMode::default(), &mut sink).unwrap();
        let mut cursor = std::io::Cursor::new(&sink);
        let recovered = decompress_from(&mut cursor).unwrap();
        assert_eq!(signal, recovered);
    }

    #[test]
    fn crc_catches_corruption() {
        let s = vec![vec![1i64; 100]];
        let mut c = compress(&s, 0).unwrap();
        *c.last_mut().unwrap() ^= 0xFF;
        assert!(decompress(&c).is_err());
    }

    /// Regression: CRC must cover the variable-fields header (n_ch, t,
    /// n_levels, flags, lpc_len, sub_len). A pre-existing scoping bug
    /// only CRC'd `lpc_meta + payload`, so flipping a header byte would
    /// silently produce a mangled but "valid"-looking signal. Surfaced
    /// when the LPC algorithm change shifted file sizes enough to move
    /// `tests/integration/test_cli_inspect.py::test_exits_nonzero_on_
    /// corrupted_lml`'s flip target from the magic byte (lucky) to the
    /// `t` byte (silently absorbed).
    #[test]
    fn crc_catches_header_corruption() {
        let s = vec![vec![100i64; 100]];
        let baseline = compress(&s, 0).unwrap();

        // Locate the LML1 magic so we can flip header bytes downstream.
        let mag = baseline
            .windows(4)
            .position(|w| w == MAGIC)
            .expect("magic must be present");

        // Try flipping every variable-header byte (offsets 4..18 from magic).
        // CRC field itself (18..22) is excluded from coverage, so it stays
        // self-protected by the CRC algebra; everything else must trip the
        // CRC check.
        for off in 4..18u32 {
            let mut c = baseline.clone();
            c[mag + off as usize] ^= 0x01;
            let err = decompress(&c)
                .err()
                .unwrap_or_else(|| panic!("flip at header offset {off} not detected"));
            // Either CRC mismatch (when the corrupted byte still produces
            // a parseable header) or InvalidHeader (when the flip lands on
            // a sanity-checked field). Both are valid detection paths.
            match err {
                LmlError::CrcMismatch { .. } | LmlError::InvalidHeader(_) => {}
                LmlError::Truncated { .. } => {}
                other => panic!("flip at offset {off} produced unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_future_version() {
        let s = vec![vec![1i64; 100]];
        let mut c = compress(&s, 0).unwrap();
        let pos = c.windows(4).position(|w| w == MAGIC).unwrap();
        c[pos + 3] = b'2';
        let err = decompress(&c).unwrap_err();
        assert!(matches!(err, LmlError::UnsupportedVersion(_)), "{}", err);
    }

    /// Audit-2026-05-11 Fix-C4: decompress rejects packets whose declared
    /// dimensions would trigger a >MAX_DECODE_BYTES allocation, even when
    /// the CRC is valid.
    ///
    /// After bumping MAX_DECODE_BYTES to 1 GiB to support 1024-channel ×
    /// u16::MAX-sample windows (= 537 MB, well within cap), the largest
    /// attacker-controllable allocation inside the legitimate header
    /// range cannot trip MAX_DECODE_BYTES anymore — defense-in-depth
    /// only. The first guard now in scope is the `n_ch > 1024` cap; the
    /// MAX_DECODE_BYTES guard catches any future header-cap relaxation.
    /// Test exercises the n_ch > 1024 rejection path.
    #[test]
    fn decompress_rejects_oversized_decoded_alloc() {
        let mut header_var = [0u8; 14];
        header_var[0..2].copy_from_slice(&2048u16.to_le_bytes()); // n_ch > 1024 cap
        header_var[2..4].copy_from_slice(&65535u16.to_le_bytes()); // t
        header_var[4] = 0;
        header_var[5] = 0;
        header_var[6..10].copy_from_slice(&0u32.to_le_bytes());
        header_var[10..14].copy_from_slice(&0u32.to_le_bytes());

        let mut crc_state = CRC32_INIT;
        crc_state = crc32_update(crc_state, &header_var);
        let crc = crc_state ^ CRC32_INIT;

        let mut pkt = Vec::with_capacity(22);
        pkt.extend_from_slice(MAGIC);
        pkt.extend_from_slice(&header_var);
        pkt.extend_from_slice(&crc.to_le_bytes());
        assert_eq!(pkt.len(), HEADER_SIZE);

        let err = decompress(&pkt).expect_err("must reject oversized n_ch");
        match err {
            LmlError::InvalidHeader(msg) => {
                assert!(
                    msg.contains("channel count"),
                    "expected n_ch > 1024 rejection, got: {msg}"
                );
            }
            other => panic!("expected InvalidHeader, got {other:?}"),
        }
    }

    /// Audit-2026-05-11 Fix-C3: compress returns Err instead of panicking
    /// on out-of-range header dimensions. FFI/WASM consumers must be able
    /// to recover from invalid input without process abort.
    #[test]
    fn compress_returns_err_on_invalid_dimensions() {
        // Empty signal — n_ch == 0.
        let r = compress(&[], 0);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");

        // Empty channel — t == 0.
        let empty_ch: Vec<Vec<i64>> = vec![vec![]];
        let r = compress(&empty_ch, 0);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");

        // T > u16::MAX.
        let oversize_t: Vec<Vec<i64>> = vec![vec![0i64; 70_000]];
        let r = compress(&oversize_t, 0);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");

        // n_ch > 1024.
        let oversize_ch: Vec<Vec<i64>> = (0..1100).map(|_| vec![1i64; 10]).collect();
        let r = compress(&oversize_ch, 0);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");

        // noise_bits > 32.
        let signal: Vec<Vec<i64>> = vec![vec![1i64; 100]];
        let r = compress(&signal, 33);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "{r:?}");

        // Valid case still works.
        let r = compress(&signal, 0);
        assert!(r.is_ok(), "valid input rejected: {r:?}");
    }

    #[test]
    fn decompress_rejects_out_of_range_n_levels() {
        // A packet whose header claims n_levels >= 4 must be REJECTED, not
        // silently decoded to just the first subband (the old `_`-arm bug).
        // Build a real packet, then flip the n_levels byte (header offset 8).
        // The n_levels guard runs before the CRC check, so the mutated byte
        // is rejected regardless of the now-stale CRC.
        let signal: Vec<Vec<i64>> = vec![vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; 2];
        let mut packet = compress(&signal, 0).expect("compress");
        assert!(decompress(&packet).is_ok(), "unmutated packet should decode");
        // The packet may carry a prefix before MAGIC (decompress locates it
        // via find_magic_offset), so n_levels is at magic_pos + 8, not byte 8.
        let magic_pos = packet
            .windows(4)
            .position(|w| w == MAGIC)
            .expect("packet contains MAGIC");
        packet[magic_pos + 8] = 4; // n_levels = 4, outside the defined 0..=3 range
        let r = decompress(&packet);
        assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "serial: {r:?}");
        #[cfg(feature = "host")]
        {
            let r = decompress_parallel(&packet);
            assert!(matches!(r, Err(LmlError::InvalidHeader(_))), "parallel: {r:?}");
        }
    }
}

#[test]
fn roundtrip_small_windows() {
    // Test various small window sizes that can occur as the last window
    for n in [3, 5, 7, 8, 10, 15, 20, 24, 30, 50, 100] {
        let signal: Vec<Vec<i64>> = vec![(0..n).map(|i| ((i as i64 * 137) % 100 - 50)).collect()];
        let compressed = compress(&signal, 0).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            signal[0].len(),
            decompressed[0].len(),
            "Length mismatch at n={}: expected {} got {}",
            n,
            signal[0].len(),
            decompressed[0].len()
        );
        assert_eq!(
            signal, decompressed,
            "Value mismatch at n={}:\n  input:  {:?}\n  output: {:?}",
            n, &signal[0], &decompressed[0]
        );
    }
}

#[test]
fn stress_boundary_values() {
    // int16 boundaries
    let signals = vec![
        vec![vec![-32768i64; 100]], // all min
        vec![vec![32767i64; 100]],  // all max
        vec![vec![0i64; 100]],      // all zero
        vec![(0..100)
            .map(|i| if i % 2 == 0 { -32768 } else { 32767 })
            .collect()], // alternating min/max
        vec![(0..100).map(|i| if i == 50 { 32767 } else { 0 }).collect()], // impulse
        vec![(0..100).map(|i| -32768 + (i as i64 * 655)).collect()], // full-range ramp
        // int24 BDF boundaries
        vec![vec![-8388608i64; 100]],
        vec![vec![8388607i64; 100]],
        vec![(0..100)
            .map(|i| if i % 2 == 0 { -8388608 } else { 8388607 })
            .collect()],
    ];
    for (idx, signal) in signals.iter().enumerate() {
        let compressed = compress(signal, 0).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            signal,
            &decompressed,
            "Boundary test {} failed:\n  in:  {:?}\n  out: {:?}",
            idx,
            &signal[0][..10.min(signal[0].len())],
            &decompressed[0][..10.min(decompressed[0].len())]
        );
    }
}

#[test]
fn stress_tiny_windows() {
    // Every possible tiny size through 3-level DWT
    for n in 1..=32 {
        for n_ch in [1, 2, 21, 64] {
            let signal: Vec<Vec<i64>> = (0..n_ch)
                .map(|ch| {
                    (0..n)
                        .map(|i| ((i as i64 * (ch + 1) as i64 * 37) % 1000 - 500))
                        .collect()
                })
                .collect();
            let compressed = compress(&signal, 0).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                signal, decompressed,
                "Tiny window n={} ch={} failed",
                n, n_ch
            );
        }
    }
}

#[test]
fn stress_large_values_lpc() {
    // Large values that could cause LPC overflow
    let signal = vec![(0..2500)
        .map(|i| {
            let v = ((i as f64 * 0.01).sin() * 2_000_000_000.0) as i64;
            v
        })
        .collect::<Vec<i64>>()];
    let compressed = compress(&signal, 0).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(signal, decompressed, "Large value LPC test failed");
}

#[test]
fn stress_channel_counts() {
    for n_ch in [1, 2, 3, 16, 64, 128, 256] {
        let signal: Vec<Vec<i64>> = (0..n_ch)
            .map(|ch| {
                (0..500)
                    .map(|i| ((i * (ch + 1) * 17) % 10000 - 5000) as i64)
                    .collect()
            })
            .collect();
        let compressed = compress(&signal, 0).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(signal, decompressed, "Channel count {} failed", n_ch);
    }
}

#[test]
fn stress_adversarial_patterns() {
    let patterns: Vec<(&str, Vec<i64>)> = vec![
        ("dc_offset", vec![16384i64; 2500]),
        (
            "square_wave",
            (0..2500)
                .map(|i| if i % 100 < 50 { -10000 } else { 10000 })
                .collect(),
        ),
        (
            "sawtooth",
            (0..2500).map(|i| (i % 500) as i64 * 40 - 10000).collect(),
        ),
        (
            "spike_train",
            (0..2500)
                .map(|i| if i % 250 == 0 { 30000 } else { 0 })
                .collect(),
        ),
        (
            "white_noise_sim",
            (0..2500)
                .map(|i| ((i as i64 * 48271) % 65536 - 32768))
                .collect(),
        ),
        (
            "near_overflow",
            (0..2500).map(|i| 32767 - (i as i64 % 3)).collect(),
        ),
        (
            "near_underflow",
            (0..2500).map(|i| -32768 + (i as i64 % 3)).collect(),
        ),
    ];
    for (name, data) in &patterns {
        let signal = vec![data.clone()];
        let compressed = compress(&signal, 0).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            &signal, &decompressed,
            "Adversarial pattern '{}' failed",
            name
        );
    }
}
