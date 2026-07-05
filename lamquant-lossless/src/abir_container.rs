//! ABIR container writer (ADR 0069 L6.2, zero-copy kernel L6.3, BCS1 wire L9).
//!
//! A clean, faithful clone of the legacy v1 `encode_into` writer
//! (`lamquant-lml-legacy::container::encode_into`), sourced from
//! `abir::Abir` instead of `&[Vec<i64>]`. Through L8 this was
//! BYTE-IDENTICAL to the legacy writer by construction — proven by extending
//! the L1 differential oracle (`tests/oracle_diff.rs`).
//!
//! **L9 (ADR 0069/0071) — the ONE deliberate byte change.** `write_abir` no
//! longer reproduces the legacy 32-byte `LML1` header. It now emits the new
//! `BCS1` 40-byte TYPED header (born-typed modality + codec descriptor + mode
//! + tier — see `abir::bcs1`), wrapping the exact same
//! byte-unchanged tail: JSON metadata → window index → per-window `LML1`
//! payloads → `LMLFOOT1` footer. `lamquant_lml_legacy::container::write_into`
//! (the retiring v1 writer, oracle-only) is UNCHANGED and still emits `LML1`
//! — the two writers are now intentionally divergent formats, not two
//! implementations of the same byte-identical wire. The oracle
//! (`tests/oracle_diff.rs`) was restructured accordingly: it proves
//! `decode(write_abir(x)) == x` (round-trip, independent of any golden)
//! instead of `write_abir(x) == write_into(x)`.
//!
//! **The lossless dispatch is zero-copy (L6.3).** The per-window signal is
//! built via [`abir::Abir::window_views`] (a `Cow`-backed O(1)
//! sub-slice on the `I64` lane); the `&[i64]` borrows off those `Cow`s are
//! handed straight to `lml::compress_with_mode_views` /
//! `compress_with_mode_parallel_views` — no `.to_vec()`, no per-window
//! `Vec<Vec<i64>>` materialization. That was the L6.2-era memcpy; L6.3 kills
//! it on the hot (lossless, `noise_bits == 0`) path. The cold bounded-MAE /
//! target-BPS arms still materialize an owned `Vec<Vec<i64>>` from the same
//! `Cow`s — those kernels (`compress_bounded_mae`/`compress_target_bps`)
//! take `&[Vec<i64>]` wholesale and are RD-search paths, not the hot loop;
//! a view-taking variant for them is a tracked fast-follow, not L6.3. Every
//! byte AFTER the header (metadata, window index, per-window payloads,
//! footer) is produced by logic cloned verbatim from the legacy writer, so
//! that tail is bit-for-bit unchanged; the header itself is the L9 BCS1
//! typed header described above, not a clone of the legacy 32-byte one.
//!
//! **Self-containment (L6.2):** this module does NOT call
//! `lamquant_lml_legacy::container::encode_into`. The small write-only helpers
//! (`metadata_with_codec_mode`, `lossless_mode_for_lpc_mode`) are cloned here
//! verbatim; the header itself is built from `abir::Bcs1Header`
//! (L9) rather than cloned legacy header-writing code. `write_abir` itself
//! mirrors `encode_into`'s signature/body directly (it IS the encode-loop level, not the
//! `ContainerStats`-computing `write_into` wrapper, so no `CountingWriter` is
//! needed here). The offset-table / footer serializer (`OffsetEntry` /
//! `OffsetTable::write_into`) is REUSED from `crate::offset_table` (re-exported
//! from `lamquant-lml-legacy`) rather than re-cloned: it's a nontrivial CRC'd
//! binary seek-table serializer, not a "small helper", and duplicating it here
//! would itself be a wire-format-fork risk. **L8 (cutover):** `OffsetTable::write_into`
//! and `ContainerStats` were moved off the `legacy-encode` gate onto
//! `legacy-decode` (crates/lamquant-lml-legacy) precisely so this module can
//! reuse them WITHOUT linking the retiring `encode_into`/`write_into` v1
//! writer — `legacy-encode` is no longer part of the live `archive` feature
//! (see `Cargo.toml`; it now lives only under `oracle`, which the
//! differential-oracle test links directly and independently).
//!
//! **L8 — full `container` facade.** Below `write_abir`/`write_abir_to_vec`,
//! this module also exposes `write_into`/`write_file`/`write_file_with_mode`/
//! `write_file_bounded_mae`/`write_file_target_bps` shims with the EXACT
//! legacy signatures (each builds an `Abir` from the caller's `&[Vec<i64>]`
//! and calls `write_abir`), plus a re-export of the legacy crate's frozen
//! READ side (`parse_header`, `ContainerHeader`, `read_window_from_bytes`,
//! `read_bytes_into_f32_calibrated`) and shared types (`ContainerStats`,
//! `OffsetEntry`, `OffsetTable`). `lamquant_core::container` (`lib.rs`) is
//! aliased to `abir_container` at the cutover, so every existing
//! `container::*` call site keeps compiling unchanged while the write half
//! now goes through `write_abir`.
//!
//! **L9 — `read_file`/`read_bytes`/`read_from` become a magic DISPATCHER,
//! not a plain re-export.** Since `write_abir` now emits `BCS1`, the
//! facade's read side can no longer be a bare `pub use` of the legacy reader
//! (which rejects `BCS1` at its very first guard — `data[0..3] != b"LML"`).
//! `read_bytes` peeks `data[0..4]`: `b"BCS1"` routes to the new
//! [`bcs1_read_bytes`]; every other magic (the seven legacy ones —
//! `LML1`/`LMO1`/`LMA1`/`LMA2`/`LMQC`/`LMLCRYPT`, plus anything too short to
//! tell) falls through to the FROZEN `lamquant_lml_legacy::container::
//! read_bytes` — decode-forever for every file already on disk. `read_file`
//! and `read_from` are thin wrappers over it (`read_from` buffers its
//! `std::io::Read` source, same shape as the legacy `read_from` it replaces
//! — needed because `codec_stages::DecompressStage`, production code in this
//! crate, calls it directly). The other three read paths (`parse_header`,
//! `read_window_from_bytes`, `read_bytes_into_f32_calibrated`) were pure
//! legacy re-exports through L9; **task #34 turns them into the same magic
//! DISPATCHER shape** (`BCS1` → the `bcs1_*` bodies below, else the FROZEN
//! legacy function) so the `lamquant-py` training dataloader — which reaches
//! the wire through exactly these three (`container_metadata`,
//! `container_read_phys_f32`, `container_read_window_np`) — stops silently
//! dropping a BCS1-re-archived corpus (a BCS1 buffer used to hit the legacy
//! `parse_header` magic guard → `PyValueError` → `except: return None`).

use crate::deployment::LosslessMode;
use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::lpc::LpcMode;
use abir::{
    Abir, Bcs1Header, CodecDescriptor, Column, Modality, BCS1_HEADER_LEN, BCS1_MAGIC,
    BCS1_VERSION_MAJOR,
};
use std::path::Path;

// Re-exported at `abir_container::{OffsetEntry, OffsetTable}` (this `pub use`
// both imports them for `write_abir`'s own use below AND makes them resolve
// as `container::{OffsetEntry, OffsetTable}` post-cutover, per the L8 facade
// contract above).
pub use crate::offset_table::{OffsetEntry, OffsetTable};

// The FROZEN v1 reader + shared write-result/parse types, re-exported
// verbatim from `lamquant-lml-legacy` (always available under
// `legacy-decode`, which `archive` keeps on). These are unchanged by the
// cutover — only the WRITE half moves to `write_abir` below. NOTE:
// `read_file`/`read_bytes`/`read_from` are intentionally NOT re-exported here
// anymore — L9 replaces them with the dispatcher functions of the same name
// defined below (see module docs "L9 — read_file/read_bytes become a magic
// DISPATCHER"). `read_from` joins that trio (beyond L9's originally-scoped
// read_file/read_bytes) because it is exercised by PRODUCTION code in this
// same crate — `codec_stages::DecompressStage` (the `pass.rs`/`pipeline.rs`
// Reversible-Pass Stage machinery) calls `container::read_from` directly, so
// leaving it legacy-only would silently break that pipeline's decode half
// the moment `write_abir`/BCS1 became the live encode path. The fix is the
// same buffer-then-dispatch shape as the legacy `read_from` itself (see
// below) — no new design surface, so it stays in L9 rather than becoming its
// own step.
pub use lamquant_lml_legacy::container::{ContainerHeader, ContainerStats};
// The FROZEN legacy read bodies this module dispatches AROUND, each aliased
// `legacy_*` so the dispatcher of the same public name below can call it
// without self-shadowing. `read_bytes` was aliased at L9; task #34 adds the
// other three (`parse_header`/`read_bytes_into_f32_calibrated`/
// `read_window_from_bytes`) so those dispatchers can keep the legacy path
// byte-for-byte frozen while the `BCS1` branch routes to the `bcs1_*` bodies.
// `read_file` still needs no alias — its dispatcher re-reads the file and
// calls this module's `read_bytes`, where the routing already happens.
use lamquant_lml_legacy::container::{
    parse_header as legacy_parse_header, read_bytes as legacy_read_bytes,
    read_bytes_into_f32_calibrated as legacy_read_bytes_into_f32_calibrated,
    read_window_from_bytes as legacy_read_window_from_bytes,
};

// Cloned verbatim from `lamquant-lml-legacy::container` (VERSION_MAJOR/MINOR,
// FLAG_HAS_FOOTER) — these are wire-format constants, not implementation
// details, so pinning them here independently is intentional and correct:
// both writers must agree on the v1 wire, not share a symbol.
const VERSION_MAJOR: u8 = 1;
const VERSION_MINOR: u8 = 0;

/// Header flag bit 0 — when set, the file carries an `LMLFOOT1` seek table at
/// EOF. See `lamquant-lml-legacy::container::FLAG_HAS_FOOTER` for the full
/// rationale; cloned here so this module is self-contained.
const FLAG_HAS_FOOTER: u8 = 0b0000_0001;

/// Cloned verbatim from `lamquant-lml-legacy::container::lossless_mode_for_lpc_mode`.
fn lossless_mode_for_lpc_mode(mode: LpcMode) -> LosslessMode {
    match mode {
        LpcMode::Fixed => LosslessMode::Mcu,
        LpcMode::Adaptive { .. } | LpcMode::Anytime { .. } => LosslessMode::Basestation,
    }
}

/// Cloned verbatim from `lamquant-lml-legacy::container::metadata_with_codec_mode`.
fn metadata_with_codec_mode(
    metadata_json: &str,
    lpc_mode: LpcMode,
    delta: Option<u64>,
    target_bps: Option<f64>,
) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(metadata_json) else {
        return metadata_json.to_owned();
    };
    let Some(object) = value.as_object_mut() else {
        return metadata_json.to_owned();
    };
    if let Some(tb) = target_bps {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("target_bps_lossy".to_owned()),
        );
        object.insert("target_bps".to_owned(), serde_json::json!(tb));
    } else if let Some(d) = delta {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("bounded_mae_near_lossless".to_owned()),
        );
        object.insert("max_error".to_owned(), serde_json::json!(d));
    } else {
        object.insert(
            "codec_mode".to_owned(),
            serde_json::Value::String("lossless".to_owned()),
        );
        object.insert(
            "lossless_mode".to_owned(),
            serde_json::Value::String(lossless_mode_for_lpc_mode(lpc_mode).as_str().to_owned()),
        );
    }
    object.insert(
        "lpc_mode".to_owned(),
        serde_json::Value::String(format!("{lpc_mode:?}")),
    );
    serde_json::to_string(&value).unwrap_or_else(|_| metadata_json.to_owned())
}

/// Write a complete LML v1 container from an [`Abir`] into a fresh `Vec<u8>`.
///
/// Generic over the modality marker `M` (ADR 0069 S3a) — the encoder egress
/// is modality-blind by design (see `abir::modality`), so this
/// accepts `&Abir<M>` for ANY `M: Modality`, `Untyped` (the default,
/// pre-S3a behavior) included. Purely a forward-compat signature widening:
/// no byte of the emitted wire format depends on `M`.
#[allow(clippy::too_many_arguments)]
pub fn write_abir_to_vec<M: Modality>(
    abir: &Abir<M>,
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
    delta: Option<u64>,
    target_bps: Option<f64>,
) -> LmlResult<Vec<u8>> {
    let mut buf = Vec::new();
    write_abir(
        &mut buf,
        abir,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
        delta,
        target_bps,
    )?;
    Ok(buf)
}

/// Write a complete LML v1 container from an [`Abir`] into any
/// [`std::io::Write`] sink. Byte-identical to
/// `lamquant_lml_legacy::container::write_into`/`encode_into` for the same
/// inputs (ADR 0069 L6.2 — see module docs for what's cloned vs reused).
/// Mirrors `encode_into`'s signature and body directly (this IS the
/// encode-loop level, not the `ContainerStats`-computing `write_into`
/// wrapper).
///
/// `n_ch`/`total_samples` are derived from `abir`
/// (`n_channels()`/`n_samples`); `sample_rate` is taken as an explicit
/// parameter (mirrors the legacy writer, which never reads a sample rate off
/// the signal currency itself).
///
/// Generic over the modality marker `M` (ADR 0069 S3a) — see
/// [`write_abir_to_vec`] for why: the encoder is modality-blind, this is a
/// forward-compat widening, not a behavior change.
#[allow(clippy::too_many_arguments)]
pub fn write_abir<M: Modality, W: std::io::Write + ?Sized>(
    sink: &mut W,
    abir: &Abir<M>,
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
    // ADR 0051 track 2: Some(δ) → per-window bounded-MAE near-lossless
    // (mutually exclusive with noise_bits); None → standard path.
    delta: Option<u64>,
    // ADR 0051 track 2 P2: Some(bps) → per-window target-BPS lossy (takes
    // precedence over delta/noise_bits). None → not target-BPS.
    target_bps: Option<f64>,
) -> LmlResult<usize /* n_windows */> {
    let n_ch = abir.n_channels();
    if n_ch == 0 {
        return Err(LmlError::InvalidHeader("0 channels".into()));
    }
    let total_samples = abir.n_samples;
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("0 samples".into()));
    }

    // Max-precision invariant (2026-07-02): the integer LML codec must never
    // SILENTLY lose precision. An `F32` (float) column would truncate toward
    // zero when `window_views` widens it to i64 — a silent, un-reviewed loss.
    // The data-driven descriptor path already refuses F32 (G5, `descriptor.rs`);
    // this guards the codec boundary itself so ANY caller that hand-builds an
    // F32-column Abir fails LOUD here instead. Float channels must route to the
    // lossless zstd/LMA path, not the integer codec.
    if let Some(ch) = abir
        .channels
        .iter()
        .position(|c| matches!(c.data, Column::F32(_)))
    {
        return Err(LmlError::InvalidHeader(format!(
            "channel {ch} is an F32 (float) column — the integer LML codec would \
             silently truncate it; route float channels to the zstd/LMA path"
        )));
    }

    // Audit-2026-05-11 Fix-#17: reject NaN/Inf sample_rate before letting them
    // cast silently to 0 or usize::MAX.
    if !sample_rate.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "sample_rate {sample_rate} is not finite"
        )));
    }
    if sample_rate <= 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "sample_rate {sample_rate} must be positive"
        )));
    }

    let actual_window_f = window_size as f64 * sample_rate / 250.0;
    if !actual_window_f.is_finite() || actual_window_f < 0.0 {
        return Err(LmlError::InvalidHeader(format!(
            "computed actual_window {actual_window_f} is not a valid window size"
        )));
    }
    // Cap at u16::MAX (the wire-format field width) instead of erroring.
    let actual_window = (actual_window_f as usize).min(u16::MAX as usize);
    if actual_window == 0 {
        return Err(LmlError::InvalidHeader("computed window size is 0".into()));
    }

    // Audit-2026-05-11 Fix-#18: checked_add so a multi-TB signal (total_samples
    // near usize::MAX) cannot overflow silently.
    let n_windows_raw = total_samples
        .checked_add(actual_window)
        .and_then(|s| s.checked_sub(1))
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "n_windows calculation overflow: total_samples={} actual_window={}",
                total_samples, actual_window
            ))
        })?
        / actual_window;
    let n_windows = n_windows_raw.max(1);
    let metadata_json = metadata_with_codec_mode(metadata_json, lpc_mode, delta, target_bps);
    let meta_bytes = metadata_json.as_bytes();
    let sample_rate_mhz = (sample_rate * 1000.0) as u32;

    // Validate u16 header fields won't truncate.
    if n_windows > u16::MAX as usize {
        return Err(LmlError::InvalidHeader(format!(
            "too many windows ({}) — max {} for LML header",
            n_windows,
            u16::MAX
        )));
    }
    debug_assert!(actual_window <= u16::MAX as usize);

    // Compress each window. Dispatch through the process-wide `ComputeBackend`
    // selector, same as the legacy writer, for byte-identical output regardless
    // of backend (only wall-clock differs).
    let backend = crate::backend::global_backend();
    let mut window_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * actual_window;
        let end = (start + actual_window).min(total_samples);
        // ADR 0069 L6.3: the zero-copy kernel dispatch. `Abir::window_views`
        // returns `Vec<Cow<'_, [i64]>>` — `Cow::Borrowed` on an all-`I64`
        // `Abir` (the whole point). We take `&[i64]` borrows off the `Cow`s
        // and hand THOSE straight to `compress_with_mode_views` /
        // `compress_with_mode_parallel_views` — no `.to_vec()`, no per-window
        // `Vec<Vec<i64>>` materialization (that was the L6.2-era memcpy this
        // step kills). The bounded-MAE / target-BPS arms below are cold RD
        // paths untouched by this step — they still materialize (tracked
        // fast-follow, not this commit).
        let cows = abir.window_views(start, end);
        let views: Vec<&[i64]> = cows.iter().map(|c| c.as_ref()).collect();
        let compressed = if let Some(tb) = target_bps {
            let window: Vec<Vec<i64>> = cows.iter().map(|c| c.to_vec()).collect();
            lml::compress_target_bps(&window, tb, lpc_mode)?
        } else if let Some(d) = delta {
            let window: Vec<Vec<i64>> = cows.iter().map(|c| c.to_vec()).collect();
            lml::compress_bounded_mae(&window, d, lpc_mode)?
        } else {
            match backend {
                crate::backend::ComputeBackend::Firmware => {
                    lml::compress_with_mode_views(&views, noise_bits, lpc_mode)?
                }
                crate::backend::ComputeBackend::Desktop => {
                    // task #32 (SAFE close, option b — see docs/proposals or
                    // the task ledger for the full writeup). `LpcMode::
                    // Anytime{deadline: Some(_)}` is a LIVE wall-clock
                    // deadline: `analyze_anytime_host` (lamquant-lml-mcu/
                    // src/lpc.rs) re-reads `Instant::now()` PER SUBBAND. The
                    // serial encoder crosses that deadline at a monotonic,
                    // single-thread channel/subband index; the rayon-
                    // parallel encoder has each worker sample the clock at
                    // its own independent schedule time, so the "time still
                    // remains" decision (adaptive vs fixed-order fallback)
                    // can differ PER SUBBAND between the two — and can even
                    // differ run-to-run on the SAME backend. That would make
                    // Desktop and Firmware emit DIFFERENT `.lml` bytes for
                    // the identical logical input, violating the
                    // byte-equal cross-backend invariant
                    // (`byte_equal_backends.rs`). It is latent today (no
                    // in-repo caller passes `Some(deadline)`; every golden
                    // vector is clock-free `None`/`Fixed`/`Adaptive`), so
                    // this is a hardening close, not a regression fix.
                    //
                    // Route a live-deadline Anytime mode to the SAME serial
                    // path Firmware uses, so both backends make the
                    // identical time-remaining decision at the identical
                    // (single-threaded) point in the loop and agree
                    // byte-for-byte. Every clock-free mode (`Anytime{
                    // deadline: None}`, `Fixed`, `Adaptive`, and the
                    // bounded-MAE / target-BPS arms above) is UNTOUCHED and
                    // keeps the parallel path. Threading an explicit
                    // per-channel `time_remaining` signal through the
                    // parallel kernel (so it can match serial exactly even
                    // WITH a live deadline, instead of falling back to
                    // running serially) is the tracked follow-up — out of
                    // scope for this minimal safe close.
                    if matches!(lpc_mode, LpcMode::Anytime { deadline: Some(_), .. }) {
                        lml::compress_with_mode_views(&views, noise_bits, lpc_mode)?
                    } else {
                        crate::compress_with_mode_parallel_views(&views, noise_bits, lpc_mode)?
                    }
                }
            }
        };
        window_payloads.push(compressed);
    }

    // BCS1 40-byte typed header (ADR 0069/0071 L9 — the ONE deliberate byte
    // change). `mode` mirrors `metadata_with_codec_mode`'s own precedence
    // (target_bps wins over delta wins over lossless); `tier` is the
    // DESCRIPTIVE, non-gating deployment stamp (`lossless_mode_for_lpc_mode`);
    // `decode_capability` is the actual GATE — this writer only ever emits
    // `CodecDescriptor::Lml53` payloads, so it is unconditionally the integer
    // floor (0).
    let mode: u8 = if target_bps.is_some() {
        2
    } else if delta.is_some() {
        1
    } else {
        0
    };
    let tier: u8 = match lossless_mode_for_lpc_mode(lpc_mode) {
        LosslessMode::Mcu => 0,
        LosslessMode::Basestation => 1,
    };
    let bcs1_header = Bcs1Header {
        version_major: VERSION_MAJOR,
        version_minor: VERSION_MINOR,
        modality_tag: abir.prov.tag,
        modality_source: abir.prov.source.to_u8(),
        codec_descriptor: CodecDescriptor::Lml53.to_u8(),
        mode,
        tier,
        decode_capability: 0,
        n_channels: n_ch as u16,
        n_windows: n_windows as u16,
        total_samples: total_samples as u32,
        window_size: actual_window as u16,
        sample_rate_mhz,
        bit_depth: 16,
        flags: FLAG_HAS_FOOTER,
        metadata_length: meta_bytes.len() as u32,
    };
    sink.write_all(&bcs1_header.to_bytes()).map_err(LmlError::Io)?;

    // Metadata JSON
    sink.write_all(meta_bytes).map_err(LmlError::Io)?;

    // Window index (u32 offsets) — validate total payload fits in u32
    let total_payload: u64 = window_payloads.iter().map(|p| p.len() as u64 + 4).sum();
    if total_payload > u32::MAX as u64 {
        return Err(LmlError::InvalidHeader(format!(
            "Total payload {} bytes exceeds u32 max for window index",
            total_payload
        )));
    }
    let mut offset = 0u32;
    for payload in &window_payloads {
        sink.write_all(&offset.to_le_bytes())
            .map_err(LmlError::Io)?;
        offset += (payload.len() as u32) + 4;
    }

    // CRITICAL ARITHMETIC (ADR 0069/0071 L9): the header is now the fixed
    // 40-byte BCS1 header, NOT the legacy 32-byte one — `BCS1_HEADER_LEN`
    // (=40), not a bare `32u64`. Metadata is `meta_bytes.len()`; window index
    // is `n_windows * 4`. The first payload's length prefix begins
    // immediately after. Missing this shifts the `LMLFOOT1` footer's absolute
    // offsets by 8 bytes off the true payload positions (the `OffsetTable`
    // stores ABSOLUTE offsets, so this is the only place the new header size
    // must be threaded through).
    let first_payload_abs = BCS1_HEADER_LEN as u64 + meta_bytes.len() as u64 + n_windows as u64 * 4;

    // Window payloads (length-prefixed) — track absolute offsets so the
    // LMLFOOT1 seek table at EOF carries O(log n) random-access entries.
    let mut offset_entries: Vec<OffsetEntry> = Vec::with_capacity(n_windows);
    let mut payload_abs = first_payload_abs;
    for (w, payload) in window_payloads.iter().enumerate() {
        let payload_len_with_prefix = 4u32 + payload.len() as u32;
        offset_entries.push(OffsetEntry {
            abs_offset: payload_abs,
            payload_len: payload_len_with_prefix,
            first_sample_idx: (w * actual_window) as u32,
        });
        sink.write_all(&(payload.len() as u32).to_le_bytes())
            .map_err(LmlError::Io)?;
        sink.write_all(payload).map_err(LmlError::Io)?;
        payload_abs += payload_len_with_prefix as u64;
    }

    // Append LMLFOOT1 seek table + 32-byte footer at EOF. REUSED from
    // `crate::offset_table` (see module docs) — old readers stop after the
    // last window payload and silently skip these; new readers parse them for
    // random access.
    let table = OffsetTable::new(offset_entries);
    table.write_into(sink, payload_abs)?;

    sink.flush().map_err(LmlError::Io)?;
    Ok(n_windows)
}

// ─────────────────────── L9: the BCS1 read dispatch ───────────────────────
//
// `write_abir` (above) now emits `BCS1`. The FROZEN legacy reader
// (`lamquant_lml_legacy::container::{read_file,read_bytes}`) rejects it
// outright — `parse_header`'s very first guard is `data[0..3] != b"LML"`,
// and `B`,`C`,`S` differ from `L`,`M`,`L` at byte 0, so a `BCS1` stream hits
// `LmlError::InvalidMagic` immediately, before any BCS1-specific field is
// even read. That is the correct, safe failure mode for the legacy path —
// it must never be taught to understand the new header. Decoding `BCS1` is
// `bcs1_read_bytes`/`bcs1_read_file` below; `read_bytes`/`read_file` (which
// shadow the legacy re-exports of the same name — see the module docs "L9 —
// read_file/read_bytes become a magic DISPATCHER") peek the leading 4 bytes
// and route to whichever reader understands them.

/// Fail-closed decode gate on a parsed BCS1 header (task #36): this build
/// decodes only `version_major` ≤ 1, `decode_capability` 0 (the integer floor),
/// and `codec_descriptor` `Lml53`. GOLDEN-NEUTRAL — v1 / cap0 / Lml53 headers
/// (all the L9 writer emits) pass unchanged; a future/incompatible header
/// fails closed (`UnsupportedVersion` / `InvalidHeader`) instead of being
/// mis-decoded. NB `Bcs1Header::parse` stays permissive (any version/cap parses
/// into the struct) so inspection tools can still read a future header; the gate
/// lives here on the DECODE path.
fn bcs1_gate_decodable(header: &Bcs1Header) -> LmlResult<()> {
    if header.version_major > BCS1_VERSION_MAJOR {
        return Err(LmlError::UnsupportedVersion(header.version_major));
    }
    if header.decode_capability > 0 {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 decode_capability {} exceeds this reader's max (0 = integer floor)",
            header.decode_capability
        )));
    }
    if header.codec_descriptor != CodecDescriptor::Lml53.to_u8() {
        return Err(LmlError::InvalidHeader(format!(
            "BCS1 codec_descriptor {} not wired to a decoder in this build \
             (only CODEC_LML_53=0 decodes today; LMO/LMQ descriptors are deferred)",
            header.codec_descriptor
        )));
    }
    Ok(())
}

/// Decode a `BCS1` container from in-memory bytes into `(signal,
/// metadata_json)` — the BCS1 counterpart of
/// `lamquant_lml_legacy::container::read_bytes`. Parses the 40-byte typed
/// header, then walks the SAME metadata → window-index → per-window-payload
/// layout the legacy reader does (only the header shape + its length
/// changed), dispatching the payload decode on `codec_descriptor`.
///
/// Only `CodecDescriptor::Lml53` (=0) is wired to an actual decoder today —
/// the LMO/LMQ descriptors are parseable (the header round-trips cleanly)
/// but deliberately NOT decodable yet (ADR 0069 L9 minimal scope; see
/// `abir::bcs1` module docs). An unrecognized or not-yet-wired
/// descriptor is a clean `LmlError::InvalidHeader`, never a silent
/// mis-decode or a panic.
pub fn bcs1_read_bytes(data: &[u8]) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let header = Bcs1Header::parse(data)
        .map_err(|e| LmlError::InvalidHeader(format!("BCS1 header: {e}")))?;

    let n_ch = header.n_channels as usize;
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    let total_samples = header.total_samples as usize;
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    let n_windows = header.n_windows as usize;
    if n_windows == 0 {
        return Err(LmlError::InvalidHeader("zero windows".into()));
    }
    let window_size = header.window_size as usize;
    // Bound the signal allocation against the data-implied size (mirrors the
    // legacy `read_bytes` guard) — total_samples can never legitimately
    // exceed n_windows * window_size, so reject before allocating.
    let max_samples = (n_windows as u64)
        .checked_mul(window_size as u64)
        .ok_or_else(|| LmlError::InvalidHeader("n_windows * window_size overflows u64".into()))?;
    if total_samples as u64 > max_samples {
        return Err(LmlError::InvalidHeader(format!(
            "total_samples {total_samples} exceeds n_windows*window_size {max_samples}"
        )));
    }

    let meta_len = header.metadata_length as usize;
    let mut pos = BCS1_HEADER_LEN;
    if pos + meta_len > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + meta_len,
            actual: data.len(),
            context: "metadata",
        });
    }
    let metadata = std::str::from_utf8(&data[pos..pos + meta_len])
        .map_err(|e| LmlError::InvalidHeader(format!("metadata is not valid UTF-8: {e}")))?
        .to_string();
    pos += meta_len;

    // Skip the window-length index (n_windows × u32 LE offsets) — this
    // decoder walks payloads sequentially, same as the legacy reader.
    // Random access via the index / LMLFOOT1 footer for the BCS1 path is a
    // tracked fast-follow (L9 is the wire + the sequential decode proof;
    // `stream::LmlReader`-style seek access is out of this minimal scope).
    pos += n_windows * 4;

    bcs1_gate_decodable(&header)?;

    // Decompress windows — identical loop to
    // `lamquant_lml_legacy::container::read_bytes`'s tail (the per-window
    // `LML1` packet format is byte-unchanged by L9).
    let mut signal = vec![vec![0i64; total_samples]; n_ch];
    for w in 0..n_windows {
        if pos + 4 > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: data.len(),
                context: "window length",
            });
        }
        let payload_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + payload_len > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + payload_len,
                actual: data.len(),
                context: "window payload",
            });
        }
        let window = lml::decompress(&data[pos..pos + payload_len])?;
        pos += payload_len;

        if window.len() != n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "window {w}: decoded channel count {} != header n_ch {}",
                window.len(),
                n_ch
            )));
        }

        let start = w * window_size;
        for ch in 0..n_ch {
            let end = (start + window[ch].len()).min(total_samples);
            let copy_len = end - start;
            signal[ch][start..start + copy_len].copy_from_slice(&window[ch][..copy_len]);
        }
    }

    Ok((signal, metadata))
}

/// Decode a `BCS1` container file — thin wrapper over [`bcs1_read_bytes`],
/// mirroring the legacy `read_file`/`read_bytes` split.
pub fn bcs1_read_file(path: &Path) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let data = std::fs::read(path).map_err(LmlError::Io)?;
    bcs1_read_bytes(&data)
}

/// Read a container from in-memory bytes — the facade DISPATCHER (ADR 0069
/// L9). Peeks `data[0..4]`: `BCS1_MAGIC` routes to [`bcs1_read_bytes`];
/// every other leading 4 bytes (the seven legacy magics, or a buffer too
/// short to tell) falls through to the FROZEN
/// `lamquant_lml_legacy::container::read_bytes`, which owns its own
/// truncation/magic error handling for that case — this function does not
/// duplicate it.
pub fn read_bytes(data: &[u8]) -> LmlResult<(Vec<Vec<i64>>, String)> {
    if data.len() >= 4 && &data[0..4] == BCS1_MAGIC {
        bcs1_read_bytes(data)
    } else {
        legacy_read_bytes(data)
    }
}

/// Read a container from a file — the facade DISPATCHER (ADR 0069 L9).
/// Same routing as [`read_bytes`]; reads the file once and dispatches on the
/// same 4-byte peek so a `BCS1` file never round-trips through the legacy
/// reader's own (separate) file-read.
pub fn read_file(path: &Path) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let data = std::fs::read(path).map_err(LmlError::Io)?;
    read_bytes(&data)
}

/// Read a container from any [`std::io::Read`] source — the facade
/// DISPATCHER (ADR 0069 L9). Buffers the whole stream (identical shape to
/// the legacy `read_from`'s own `read_to_end` behavior — no new design
/// surface) then routes through this module's [`read_bytes`] dispatcher.
/// Exists alongside `read_file`/`read_bytes` (rather than staying a bare
/// legacy re-export) because `codec_stages::DecompressStage` — production
/// code in this crate — calls `container::read_from` directly; see the
/// module-level import comment above for why.
pub fn read_from<R: std::io::Read>(src: &mut R) -> LmlResult<(Vec<Vec<i64>>, String)> {
    let mut data = Vec::new();
    src.read_to_end(&mut data).map_err(LmlError::Io)?;
    read_bytes(&data)
}

// ───────────────── #34: BCS1 metadata / calibrated-f32 / window reads ─────────────────
//
// The three dataloader read paths (`parse_header`, `read_bytes_into_f32_calibrated`,
// `read_window_from_bytes`) become magic DISPATCHERS here, exactly like the L9
// `read_bytes`/`read_file`/`read_from` trio above. Everything at offset >= 40 in a BCS1
// container (metadata JSON → `n_windows × u32` window-length index → per-window
// `[u32 len][LML1 payload]` blocks → `LMLFOOT1`) is byte-identical to the legacy layout,
// so once `bcs1_parse_header` produces a `ContainerHeader` with the right `payload_start`
// (= `BCS1_HEADER_LEN + metadata_length + n_windows*4`), the calibrated-f32 and
// random-access loops are the legacy loops verbatim. The legacy branch stays the FROZEN
// `legacy_*` body untouched (existing `.lma` corpora are all LML1); only the BCS1 branch
// is new.

/// Parse a `BCS1` container header into the shared [`ContainerHeader`] shape — the BCS1
/// counterpart of the legacy `parse_header`. Fail-closed on a non-`Lml53`
/// `codec_descriptor` (mirrors [`bcs1_read_bytes`]): the header round-trips but the body
/// is not decodable in this build, so callers get `InvalidHeader`, never a silent
/// mis-decode.
pub fn bcs1_parse_header(data: &[u8]) -> LmlResult<ContainerHeader> {
    let header = Bcs1Header::parse(data)
        .map_err(|e| LmlError::InvalidHeader(format!("BCS1 header: {e}")))?;

    let n_ch = header.n_channels as usize;
    if n_ch == 0 || n_ch > 1024 {
        return Err(LmlError::InvalidHeader(format!("channel count: {}", n_ch)));
    }
    let total_samples = header.total_samples as usize;
    if total_samples == 0 {
        return Err(LmlError::InvalidHeader("zero samples".into()));
    }
    let n_windows = header.n_windows as usize;
    if n_windows == 0 {
        return Err(LmlError::InvalidHeader("zero windows".into()));
    }
    let window_size = header.window_size as usize;
    // Cross-field consistency (MiMo #34 review): total_samples can never
    // legitimately exceed n_windows * window_size. Guard it here — the
    // calibrated-f32 and window read paths parse through this fn, so without
    // it they'd lack the protection the whole-decode `bcs1_read_bytes` already
    // has (a crafted header would allocate an oversized output and silently
    // zero-fill the tail into garbage training data).
    let max_samples = (n_windows as u64)
        .checked_mul(window_size as u64)
        .ok_or_else(|| LmlError::InvalidHeader("n_windows * window_size overflows u64".into()))?;
    if total_samples as u64 > max_samples {
        return Err(LmlError::InvalidHeader(format!(
            "total_samples {total_samples} exceeds n_windows*window_size {max_samples}"
        )));
    }

    bcs1_gate_decodable(&header)?;

    let meta_len = header.metadata_length as usize;
    let hdr_end = BCS1_HEADER_LEN;
    // checked_* on the offset arithmetic (MiMo #34 review): `metadata_length`
    // is an attacker-controllable u32. On the 64-bit host these can't actually
    // wrap usize, but this matches the discipline in
    // `bcs1_read_window_from_bytes` and keeps the truncation guard sound if
    // abir_container ever moves to a 32-bit (no_std) target.
    let meta_end = hdr_end
        .checked_add(meta_len)
        .ok_or_else(|| LmlError::InvalidHeader("metadata offset overflow".into()))?;
    if meta_end > data.len() {
        return Err(LmlError::Truncated {
            expected: meta_end,
            actual: data.len(),
            context: "metadata",
        });
    }
    let metadata = std::str::from_utf8(&data[hdr_end..meta_end])
        .map_err(|e| LmlError::InvalidHeader(format!("metadata is not valid UTF-8: {e}")))?
        .to_string();

    // Window-length index (n_windows × u32) sits immediately after the metadata; the
    // first payload's length prefix begins right after it. Identical relative layout to
    // the legacy reader — only the fixed header length (40 vs 32/20/18) differs.
    let payload_start = meta_end
        .checked_add(n_windows * 4)
        .ok_or_else(|| LmlError::InvalidHeader("payload_start overflow".into()))?;

    Ok(ContainerHeader {
        n_ch,
        n_windows,
        total_samples,
        window_size,
        metadata,
        payload_start,
    })
}

/// Calibrated full-signal f32 decode of a `BCS1` container — the BCS1 counterpart of the
/// legacy `read_bytes_into_f32_calibrated` (the memory-bounded workhorse behind
/// `container_read_phys_f32`). Same per-channel digital→physical affine and same
/// sequential window walk (one decoded i64 window transient + the caller's f32 output);
/// only the header parse differs.
pub fn bcs1_read_bytes_into_f32_calibrated(
    data: &[u8],
    out: &mut [f32],
    calib: &[f32],
) -> LmlResult<ContainerHeader> {
    let header = bcs1_parse_header(data)?;
    let n_ch = header.n_ch;
    let total = header.total_samples;
    if out.len() != n_ch * total {
        return Err(LmlError::InvalidHeader(format!(
            "output buffer size mismatch: expected {} got {}",
            n_ch * total,
            out.len()
        )));
    }
    if calib.len() != n_ch * 4 {
        return Err(LmlError::InvalidHeader(format!(
            "calib length {} != n_ch*4 ({})",
            calib.len(),
            n_ch * 4
        )));
    }

    // Per-channel scale + offset so we do one mul + one add per sample (identical to the
    // legacy body — the degenerate `dig_range == 0` row emits zero).
    let mut scale = vec![0.0f32; n_ch];
    let mut offset = vec![0.0f32; n_ch];
    for ch in 0..n_ch {
        let dig_min = calib[ch * 4];
        let dig_max = calib[ch * 4 + 1];
        let phys_min = calib[ch * 4 + 2];
        let phys_max = calib[ch * 4 + 3];
        let dig_range = dig_max - dig_min;
        if dig_range == 0.0 {
            scale[ch] = 0.0;
            offset[ch] = 0.0;
        } else {
            scale[ch] = (phys_max - phys_min) / dig_range;
            offset[ch] = phys_min - dig_min * scale[ch];
        }
    }

    let window_size = header.window_size;
    let mut pos = header.payload_start;
    for w in 0..header.n_windows {
        if pos + 4 > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + 4,
                actual: data.len(),
                context: "window length",
            });
        }
        let payload_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + payload_len > data.len() {
            return Err(LmlError::Truncated {
                expected: pos + payload_len,
                actual: data.len(),
                context: "window payload",
            });
        }
        let window = lml::decompress(&data[pos..pos + payload_len])?;
        pos += payload_len;

        // Refuse to silently zero-fill a header/window channel-count disagreement (would
        // produce garbage training data without a warning) — matches the legacy body.
        if window.len() != n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "window {w}: decoded channel count {} != header n_ch {}",
                window.len(),
                n_ch
            )));
        }

        let start = w * window_size;
        for ch in 0..n_ch {
            let src = &window[ch];
            let copy_len = (src.len()).min(total.saturating_sub(start));
            let dst_off = ch * total + start;
            let s = scale[ch];
            let o = offset[ch];
            for i in 0..copy_len {
                out[dst_off + i] = src[i] as f32 * s + o;
            }
        }
    }

    Ok(header)
}

/// Random-access read of one window from a `BCS1` container — the BCS1 counterpart of the
/// legacy `read_window_from_bytes`. Uses the same in-header window-length index (the u32
/// relative offsets `write_abir` writes at `payload_start - n_windows*4`), decompressing
/// ONLY the requested window.
pub fn bcs1_read_window_from_bytes(
    data: &[u8],
    window_idx: usize,
) -> LmlResult<(Vec<Vec<i64>>, ContainerHeader)> {
    let header = bcs1_parse_header(data)?;
    if window_idx >= header.n_windows {
        return Err(LmlError::InvalidHeader(format!(
            "window_idx {} out of range (n_windows {})",
            window_idx, header.n_windows
        )));
    }
    let index_base = header
        .payload_start
        .checked_sub(header.n_windows * 4)
        .ok_or_else(|| LmlError::InvalidHeader("payload_start underflow".into()))?;
    let entry_pos = index_base + window_idx * 4;
    if entry_pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: entry_pos + 4,
            actual: data.len(),
            context: "window index entry",
        });
    }
    let rel_off = u32::from_le_bytes([
        data[entry_pos],
        data[entry_pos + 1],
        data[entry_pos + 2],
        data[entry_pos + 3],
    ]) as usize;
    let block_pos = header
        .payload_start
        .checked_add(rel_off)
        .ok_or_else(|| LmlError::InvalidHeader("window block position overflow".into()))?;
    if block_pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: block_pos + 4,
            actual: data.len(),
            context: "window length",
        });
    }
    let payload_len = u32::from_le_bytes([
        data[block_pos],
        data[block_pos + 1],
        data[block_pos + 2],
        data[block_pos + 3],
    ]) as usize;
    let payload_start = block_pos + 4;
    if payload_start + payload_len > data.len() {
        return Err(LmlError::Truncated {
            expected: payload_start + payload_len,
            actual: data.len(),
            context: "window payload",
        });
    }
    let window = lml::decompress(&data[payload_start..payload_start + payload_len])?;
    if window.len() != header.n_ch {
        return Err(LmlError::InvalidHeader(format!(
            "window {window_idx}: decoded channel count {} != header n_ch {}",
            window.len(),
            header.n_ch
        )));
    }
    Ok((window, header))
}

/// Parse a container header from in-memory bytes — the facade DISPATCHER (#34). Peeks
/// `data[0..4]`: `BCS1_MAGIC` → [`bcs1_parse_header`]; every other leading 4 bytes → the
/// FROZEN legacy `parse_header` (the 32/20/18-byte auto-detect), which owns its own
/// magic/truncation errors. Every caller of `container::parse_header` (the dataloader's
/// `container_metadata`, `cmd_recover`, `cmd_info`) transparently gains BCS1 support.
pub fn parse_header(data: &[u8]) -> LmlResult<ContainerHeader> {
    if data.len() >= 4 && &data[0..4] == BCS1_MAGIC {
        bcs1_parse_header(data)
    } else {
        legacy_parse_header(data)
    }
}

/// Calibrated full-signal f32 decode — the facade DISPATCHER (#34). Same 4-byte routing
/// as [`parse_header`]: `BCS1` → [`bcs1_read_bytes_into_f32_calibrated`], else the FROZEN
/// legacy body. This is the path `lamquant-py`'s `container_read_phys_f32` calls; before
/// #34 a BCS1 buffer hit the legacy magic guard and was silently dropped from training.
pub fn read_bytes_into_f32_calibrated(
    data: &[u8],
    out: &mut [f32],
    calib: &[f32],
) -> LmlResult<ContainerHeader> {
    if data.len() >= 4 && &data[0..4] == BCS1_MAGIC {
        bcs1_read_bytes_into_f32_calibrated(data, out, calib)
    } else {
        legacy_read_bytes_into_f32_calibrated(data, out, calib)
    }
}

/// Selected-channel calibrated f32 decode (ADR 0075 A2). Decodes only the channels in
/// `channel_mask` into `out` (`[n_sel * total]`, in selected order) — skipping the full
/// `[n_ch, total]` f32 array [`read_bytes_into_f32_calibrated`] materialises AND the
/// downstream channel-select copy. `calib` is `[n_sel * 4]` in selected order;
/// `channel_mask[sel]` is the SOURCE channel index. Reuses the FROZEN dispatched
/// per-window reader [`read_window_from_bytes`] (correct for BCS1 + legacy), so no wire
/// walk is reimplemented; the transient is one decoded window (`[n_ch, ws]` i64), freed
/// each iteration. The channel selection is purely in the write step, so a selected row
/// is bit-identical to the corresponding row of the full decode.
pub fn read_bytes_into_f32_calibrated_selected(
    data: &[u8],
    out: &mut [f32],
    calib: &[f32],
    channel_mask: &[u16],
) -> LmlResult<ContainerHeader> {
    let header = parse_header(data)?;
    let n_ch = header.n_ch;
    let total = header.total_samples;
    let n_sel = channel_mask.len();
    if out.len() != n_sel * total {
        return Err(LmlError::InvalidHeader(format!(
            "selected output buffer size mismatch: expected {} got {}",
            n_sel * total,
            out.len()
        )));
    }
    if calib.len() != n_sel * 4 {
        return Err(LmlError::InvalidHeader(format!(
            "selected calib length {} != n_sel*4 ({})",
            calib.len(),
            n_sel * 4
        )));
    }
    // `MISSING` (u16::MAX) marks a target the caller couldn't resolve (e.g. an absent
    // optional electrode) — that output row is left zero-filled, matching
    // `extract_channel_data`'s zero-fill. EEG channel counts are « u16::MAX, so a real
    // index never collides with the sentinel.
    const MISSING: u16 = u16::MAX;
    for &src_ch in channel_mask {
        if src_ch != MISSING && src_ch as usize >= n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "channel_mask index {src_ch} out of range (n_ch={n_ch})"
            )));
        }
    }
    // Zero the MISSING rows up front (the window loop never touches them) so the output
    // is fully defined regardless of the caller's buffer init.
    for sel in 0..n_sel {
        if channel_mask[sel] == MISSING {
            for v in &mut out[sel * total..(sel + 1) * total] {
                *v = 0.0;
            }
        }
    }
    // Per-selected-channel scale + offset — same formula as the full path.
    let mut scale = vec![0.0f32; n_sel];
    let mut offset = vec![0.0f32; n_sel];
    for s in 0..n_sel {
        let dig_min = calib[s * 4];
        let dig_max = calib[s * 4 + 1];
        let phys_min = calib[s * 4 + 2];
        let phys_max = calib[s * 4 + 3];
        let dig_range = dig_max - dig_min;
        if dig_range == 0.0 {
            scale[s] = 0.0;
            offset[s] = 0.0;
        } else {
            scale[s] = (phys_max - phys_min) / dig_range;
            offset[s] = phys_min - dig_min * scale[s];
        }
    }
    let window_size = header.window_size;
    for w in 0..header.n_windows {
        let (window, _wh) = read_window_from_bytes(data, w)?;
        if window.len() != n_ch {
            return Err(LmlError::InvalidHeader(format!(
                "window {w}: decoded channel count {} != header n_ch {}",
                window.len(),
                n_ch
            )));
        }
        let start = w * window_size;
        for sel in 0..n_sel {
            if channel_mask[sel] == MISSING {
                continue; // zero-filled up front
            }
            let src = &window[channel_mask[sel] as usize];
            let copy_len = src.len().min(total.saturating_sub(start));
            let dst_off = sel * total + start;
            let s = scale[sel];
            let o = offset[sel];
            for i in 0..copy_len {
                out[dst_off + i] = src[i] as f32 * s + o;
            }
        }
    }
    Ok(header)
}

/// Random-access single-window read — the facade DISPATCHER (#34). Same 4-byte routing:
/// `BCS1` → [`bcs1_read_window_from_bytes`], else the FROZEN legacy body. Backs
/// `lamquant-py`'s `container_read_window_np`.
pub fn read_window_from_bytes(
    data: &[u8],
    window_idx: usize,
) -> LmlResult<(Vec<Vec<i64>>, ContainerHeader)> {
    if data.len() >= 4 && &data[0..4] == BCS1_MAGIC {
        bcs1_read_window_from_bytes(data, window_idx)
    } else {
        legacy_read_window_from_bytes(data, window_idx)
    }
}

#[cfg(test)]
mod bcs1_read_tests {
    use super::*;

    fn two_channel_signal() -> Vec<Vec<i64>> {
        vec![
            (0..600i64).map(|t| ((t * 37) % 4001) - 2000).collect(),
            (0..600i64).map(|t| ((t * 53) % 3001) - 1500).collect(),
        ]
    }

    /// A2 (ADR 0075) — a selected-channel decode is bit-identical to the corresponding
    /// rows of the full decode (reordering + subset + distinct per-channel calib), and
    /// out-of-range mask indices error.
    #[test]
    fn selected_channel_matches_full() {
        let n_ch = 5usize;
        let t = 600usize;
        let sig: Vec<Vec<i64>> = (0..n_ch)
            .map(|c| (0..t).map(|i| ((i as i64 * 37 + c as i64 * 101) % 4001) - 2000).collect())
            .collect();
        let abir = Abir::from_channels_i64(sig, 250.0);
        let buf = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");

        // Distinct per-channel calib exercises the calib indexing.
        let mut calib_full = vec![0.0f32; n_ch * 4];
        for c in 0..n_ch {
            calib_full[c * 4] = 0.0; // dig_min
            calib_full[c * 4 + 1] = 100.0; // dig_max
            calib_full[c * 4 + 2] = c as f32; // phys_min
            calib_full[c * 4 + 3] = c as f32 + 50.0; // phys_max
        }
        let total = parse_header(&buf).unwrap().total_samples;
        let mut full = vec![0.0f32; n_ch * total];
        read_bytes_into_f32_calibrated(&buf, &mut full, &calib_full).unwrap();

        // Reordering + subset {3, 1, 4}.
        let mask: Vec<u16> = vec![3, 1, 4];
        let mut calib_sel = vec![0.0f32; mask.len() * 4];
        for (s, &src) in mask.iter().enumerate() {
            let src = src as usize;
            calib_sel[s * 4..s * 4 + 4].copy_from_slice(&calib_full[src * 4..src * 4 + 4]);
        }
        let mut sel = vec![0.0f32; mask.len() * total];
        read_bytes_into_f32_calibrated_selected(&buf, &mut sel, &calib_sel, &mask).unwrap();

        for (s, &src) in mask.iter().enumerate() {
            let src = src as usize;
            for i in 0..total {
                assert_eq!(
                    sel[s * total + i],
                    full[src * total + i],
                    "selected row {s} (src ch {src}) sample {i} must match the full decode"
                );
            }
        }

        // A MISSING sentinel (u16::MAX) zero-fills that output row; present rows still
        // match. Init the buffer to a non-zero value to prove the zero-fill actually runs.
        let mask_m: Vec<u16> = vec![2, u16::MAX, 0];
        let mut calib_m = vec![0.0f32; mask_m.len() * 4];
        calib_m[0..4].copy_from_slice(&calib_full[2 * 4..2 * 4 + 4]);
        calib_m[8..12].copy_from_slice(&calib_full[0..4]); // MISSING row's calib is unused
        let mut selm = vec![7.0f32; mask_m.len() * total];
        read_bytes_into_f32_calibrated_selected(&buf, &mut selm, &calib_m, &mask_m).unwrap();
        for i in 0..total {
            assert_eq!(selm[i], full[2 * total + i], "present row 0 (ch2) sample {i}");
            assert_eq!(selm[total + i], 0.0, "MISSING row must be zero-filled (was 7.0)");
            assert_eq!(selm[2 * total + i], full[i], "present row 2 (ch0) sample {i}");
        }

        // Out-of-range mask index errors (n_ch=5, index 9 invalid; u16::MAX is allowed).
        let mut bad = vec![0.0f32; total];
        assert!(
            read_bytes_into_f32_calibrated_selected(&buf, &mut bad, &[0.0; 4], &[9u16]).is_err(),
            "out-of-range channel_mask must error"
        );
    }

    #[test]
    fn dispatcher_routes_bcs1_and_round_trips() {
        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal.clone(), 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        assert_eq!(&bytes[0..4], BCS1_MAGIC, "write_abir must emit BCS1");

        let (rec, _meta) = read_bytes(&bytes).expect("dispatcher read_bytes");
        assert_eq!(rec, signal, "dispatched BCS1 decode must round-trip exactly");

        let (rec2, _meta2) = bcs1_read_bytes(&bytes).expect("bcs1_read_bytes directly");
        assert_eq!(rec2, signal, "bcs1_read_bytes must round-trip exactly");

        // `read_from` (the `std::io::Read`-generic entry point production
        // code like `codec_stages::DecompressStage` calls) must dispatch
        // identically to `read_bytes`.
        let (rec3, _meta3) =
            read_from(&mut std::io::Cursor::new(&bytes)).expect("dispatcher read_from");
        assert_eq!(rec3, signal, "dispatched BCS1 read_from must round-trip exactly");
    }

    #[test]
    fn dispatcher_falls_through_to_legacy_for_lml1() {
        // A legacy LML1 buffer (built via the frozen legacy writer under
        // `oracle`/`legacy-encode` isn't linked here, so hand-craft the
        // smallest possible rejection case instead: legacy read_bytes must
        // still own the LML1 path, i.e. the dispatcher must NOT try to BCS1
        // -parse an LML1 buffer.) Feed a truncated LML1 buffer to prove it's
        // legacy_read_bytes (which returns Truncated for <18 bytes), not the
        // BCS1 parser (which would report a magic mismatch instead — a
        // different error shape — if it were wrongly invoked).
        let short_lml1 = b"LML1\x01\x00\x00\x00";
        let err = read_bytes(short_lml1).expect_err("too-short buffer must Err");
        assert!(
            matches!(err, LmlError::Truncated { .. }),
            "non-BCS1 magic must fall through to the legacy reader's own \
             Truncated error, not a BCS1 InvalidMagic: {err:?}"
        );
    }

    #[test]
    fn footer_offsets_point_at_real_window_payloads_in_bcs1_output() {
        // THE direct proof of the `:382` `first_payload_abs` arithmetic fix
        // (32 -> BCS1_HEADER_LEN=40): `first_payload_abs` seeds
        // `payload_abs`, which becomes every `OffsetEntry::abs_offset`
        // written into the `LMLFOOT1` footer. `bcs1_read_bytes`'s own
        // round-trip tests decode SEQUENTIALLY (never touch the footer), so
        // they would NOT have caught an 8-byte drift here — this test reads
        // the footer's offsets back and confirms each one points at a REAL
        // `[u32 len][packet]` block inside the actual BCS1 file bytes.
        // Mirrors `lamquant_lml_legacy::container::tests::
        // footer_offsets_point_at_real_window_payloads` for the legacy
        // 32-byte header.
        let signal: Vec<Vec<i64>> = vec![
            (0..384i64).map(|t| ((t * 17) % 2001) - 1000).collect(),
            (0..384i64).map(|t| ((t * 29) % 1501) - 750).collect(),
        ];
        let abir = Abir::from_channels_i64(signal, 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        assert_eq!(&bytes[0..4], BCS1_MAGIC);

        let table = OffsetTable::read_from_buffer(&bytes)
            .expect("footer parse must not error")
            .expect("BCS1 output must carry an LMLFOOT1 footer (FLAG_HAS_FOOTER is set)");
        assert!(table.len() >= 1, "at least one window");
        for e in table.entries() {
            let off = e.abs_offset as usize;
            assert!(
                off + 4 <= bytes.len(),
                "abs_offset {off} lands past EOF (len={}) — the header-size base is wrong",
                bytes.len()
            );
            let prefix =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            assert_eq!(
                prefix + 4,
                e.payload_len,
                "length-prefix at abs_offset={off} doesn't match the footer entry — \
                 first_payload_abs is off (the exact :382 8-byte-drift bug this test targets)"
            );
        }
    }

    #[test]
    fn legacy_parse_header_rejects_real_bcs1_bytes_cleanly() {
        // The exact confirmation ADR 0069/0071 L9 Step 3 calls for: a REAL
        // `write_abir` (BCS1) output fed straight into the FROZEN legacy
        // `lamquant_lml_legacy::container::parse_header` must be rejected at
        // its very first guard (`data[0..3] != b"LML"` — `B`,`C`,`S` != `L`,
        // `M`,`L`) BEFORE the probe field (`data[4..6]`) is ever read, so
        // there is no risk of the legacy reader mis-interpreting a BCS1
        // stream as some exotic legacy header variant.
        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        assert_eq!(&bytes[0..4], BCS1_MAGIC);

        // `ContainerHeader` (the Ok side) isn't `Debug`, so `expect_err` won't
        // compile — match explicitly instead. NOTE (#34): the wire-safety
        // property under test is that the FROZEN LEGACY parser rejects BCS1, so
        // this calls `legacy_parse_header` explicitly — the module's own
        // `parse_header` is now a dispatcher that ACCEPTS BCS1 (proven in
        // `dispatcher_parse_header_accepts_bcs1` below).
        let err = match legacy_parse_header(&bytes) {
            Err(e) => e,
            Ok(_) => panic!("legacy parse_header must reject real BCS1 bytes, got Ok"),
        };
        assert!(
            matches!(err, LmlError::InvalidMagic([b'B', b'C', b'S', b'1'])),
            "expected InvalidMagic([B,C,S,1]) from the FIRST guard, got: {err:?}"
        );
    }

    /// #34: the DISPATCHER `parse_header` (unlike the frozen legacy one) parses a
    /// real `write_abir` BCS1 container, returning the correct dims + metadata.
    #[test]
    fn dispatcher_parse_header_accepts_bcs1() {
        let signal = two_channel_signal();
        let n_samples = signal[0].len();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{\"k\":1}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        assert_eq!(&bytes[0..4], BCS1_MAGIC);

        let hdr = parse_header(&bytes).expect("dispatcher parse_header must accept BCS1");
        assert_eq!(hdr.n_ch, 2);
        assert_eq!(hdr.total_samples, n_samples);
        // `write_abir` augments the metadata via `metadata_with_codec_mode`
        // (injects codec_mode/lossless_mode/lpc_mode), so the stored JSON is a
        // superset of the input — assert the caller's key survives round-trip.
        assert!(
            hdr.metadata.contains("\"k\":1"),
            "original metadata key must survive BCS1 round-trip, got: {}",
            hdr.metadata
        );
    }

    /// #34: the calibrated-f32 dispatcher decodes a BCS1 container. With an identity
    /// calibration (dig==phys range) the f32 output equals the original i64 samples,
    /// so this doubles as a value round-trip of the sequential BCS1 decode loop.
    #[test]
    fn dispatcher_read_f32_calibrated_round_trips_bcs1() {
        let signal = two_channel_signal();
        let n_ch = signal.len();
        let n_samples = signal[0].len();
        let expected = signal.clone();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");

        // Identity affine per channel: dig_min/max = phys_min/max = full i16 range.
        let mut calib = vec![0.0f32; n_ch * 4];
        for ch in 0..n_ch {
            calib[ch * 4] = -32768.0;
            calib[ch * 4 + 1] = 32767.0;
            calib[ch * 4 + 2] = -32768.0;
            calib[ch * 4 + 3] = 32767.0;
        }
        let mut out = vec![0.0f32; n_ch * n_samples];
        let hdr = read_bytes_into_f32_calibrated(&bytes, &mut out, &calib)
            .expect("BCS1 calibrated f32 decode");
        assert_eq!(hdr.n_ch, n_ch);
        for ch in 0..n_ch {
            for i in 0..n_samples {
                assert_eq!(
                    out[ch * n_samples + i],
                    expected[ch][i] as f32,
                    "ch {ch} sample {i} mismatch"
                );
            }
        }
    }

    /// #34: the random-access dispatcher reads every window of a BCS1 container and
    /// reconstructs the full signal window-by-window (proves the in-header window
    /// index + relative-offset math is byte-correct for BCS1).
    #[test]
    fn dispatcher_read_window_reconstructs_bcs1() {
        let signal = two_channel_signal();
        let n_ch = signal.len();
        let n_samples = signal[0].len();
        let expected = signal.clone();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");

        let (_first, hdr) =
            read_window_from_bytes(&bytes, 0).expect("BCS1 window 0 must decode");
        let mut recon = vec![vec![0i64; n_samples]; n_ch];
        for w in 0..hdr.n_windows {
            let (window, _h) =
                read_window_from_bytes(&bytes, w).expect("BCS1 window decode");
            let start = w * hdr.window_size;
            for ch in 0..n_ch {
                for (i, &v) in window[ch].iter().enumerate() {
                    if start + i < n_samples {
                        recon[ch][start + i] = v;
                    }
                }
            }
        }
        assert_eq!(recon, expected, "window-by-window BCS1 reconstruction mismatch");

        // Out-of-range index is a clean error, not a panic.
        assert!(read_window_from_bytes(&bytes, hdr.n_windows).is_err());
    }

    /// #34 (MiMo review): a crafted BCS1 header whose total_samples exceeds
    /// n_windows*window_size is rejected by `bcs1_parse_header`, not silently
    /// decoded into a zero-padded oversized buffer.
    #[test]
    fn bcs1_parse_header_rejects_inconsistent_total_samples() {
        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let mut bytes =
            write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
                .expect("write_abir_to_vec");
        // total_samples is the u32 at BCS1 header bytes 16..20 — inflate it far
        // past n_windows*window_size.
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        // `ContainerHeader` (Ok side) isn't `Debug`, so match instead of expect_err.
        let err = match parse_header(&bytes) {
            Err(e) => e,
            Ok(_) => panic!("inconsistent total_samples must be rejected, got Ok"),
        };
        assert!(matches!(err, LmlError::InvalidHeader(_)), "got: {err:?}");
    }

    #[test]
    fn bcs1_read_bytes_rejects_unwired_codec_descriptor() {
        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let mut bytes = write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        bytes[8] = CodecDescriptor::Lmo97.to_u8(); // codec_descriptor byte
        let err = bcs1_read_bytes(&bytes).expect_err("unwired descriptor must Err, not decode");
        assert!(
            matches!(err, LmlError::InvalidHeader(_)),
            "unwired codec_descriptor must be a clean InvalidHeader: {err:?}"
        );
    }

    /// Max-precision invariant: write_abir must REFUSE an F32 (float) column
    /// rather than silently truncate it through the integer codec.
    #[test]
    fn write_abir_refuses_f32_columns() {
        let ch = abir::Channel {
            label: alloc::sync::Arc::from("Fp1"),
            data: Column::F32(alloc::sync::Arc::from(vec![1.5f32, -2.5, 3.5].as_slice())),
            phys_min: 0.0,
            phys_max: 0.0,
        };
        let a = Abir::from_parts(vec![ch], 250.0, 3);
        let err = write_abir_to_vec(&a, 250.0, 128, 0, "{}", LpcMode::default(), None, None)
            .expect_err("F32 column must be refused, not silently truncated");
        assert!(
            matches!(err, LmlError::InvalidHeader(ref m) if m.contains("F32") || m.contains("float")),
            "expected an F32-refusal InvalidHeader, got: {err:?}"
        );
    }

    /// #36: decode fails closed on a future header version (golden-neutral —
    /// version_major=1 still decodes; a v2 header is rejected).
    #[test]
    fn bcs1_decode_rejects_future_version() {
        let abir = Abir::from_channels_i64(two_channel_signal(), 250.0);
        let mut bytes =
            write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None).unwrap();
        bytes[4] = 2; // version_major = 2 (a future header layout)
        assert!(matches!(bcs1_read_bytes(&bytes), Err(LmlError::UnsupportedVersion(2))));
        assert!(matches!(parse_header(&bytes), Err(LmlError::UnsupportedVersion(2))));
    }

    /// #36: decode fails closed on a decode_capability above this reader's max
    /// (0 = integer floor). Golden-neutral — cap0 files still decode.
    #[test]
    fn bcs1_decode_rejects_unsupported_capability() {
        let abir = Abir::from_channels_i64(two_channel_signal(), 250.0);
        let mut bytes =
            write_abir_to_vec(&abir, 250.0, 128, 0, "{}", LpcMode::default(), None, None).unwrap();
        bytes[11] = 1; // decode_capability = 1 (> reader max 0)
        assert!(matches!(bcs1_read_bytes(&bytes), Err(LmlError::InvalidHeader(_))));
        assert!(matches!(parse_header(&bytes), Err(LmlError::InvalidHeader(_))));
    }
}

// ─────────────────────── L8: the `container::*` write shims ───────────────────────
//
// Everything below is new at the cutover. Each shim has the EXACT legacy
// `lamquant_lml_legacy::container` signature (same params, same
// `LmlResult<ContainerStats>` return) so the ~20 existing call sites
// (bin/lml.rs, lma.rs, async_io.rs, codec_stages.rs, range.rs, lamquant-py)
// keep compiling unchanged after `lib.rs` re-points `container` at this
// module. Each builds an `Abir` from the caller's `&[Vec<i64>]` (one bulk
// `.to_vec()`, not zero-copy — callers that already hold an `Abir`, e.g. the
// L7 `lower_to_abir` readers, should call `write_abir` directly instead) and
// delegates the actual encode to `write_abir` above.

/// Internal: byte-counting writer, mirroring the pattern in the legacy
/// `write_into` (`lamquant-lml-legacy::container::CountingWriter`) so
/// `ContainerStats::compressed_size` is exact without a `fs::metadata()`
/// round-trip. A second, independent copy — kept local so this module stays
/// self-contained (see module docs); it is NOT the retiring writer's
/// `CountingWriter`, which stays `legacy-encode`-gated in the legacy crate.
struct CountingWriter<'w, W: std::io::Write + ?Sized> {
    inner: &'w mut W,
    count: u64,
}

impl<'w, W: std::io::Write + ?Sized> std::io::Write for CountingWriter<'w, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Internal: build the same `ContainerStats` shape the legacy writer
/// returns, from a signal + the encoder's own `n_windows`/`compressed_size`
/// results. Cloned verbatim from `lamquant-lml-legacy::container`'s
/// `write_into`/`write_file_bounded_mae`/`write_file_target_bps` tail
/// (identical arithmetic — `raw_size` is `n_channels * total_samples * 2`,
/// matching the legacy 16-bit-sample convention, not `size_of::<i64>()`).
fn stats_from(
    n_windows: usize,
    signal: &[Vec<i64>],
    compressed_size: usize,
    sample_rate: f64,
) -> ContainerStats {
    let n_channels = signal.len();
    let total_samples = signal.first().map(|ch| ch.len()).unwrap_or(0);
    let raw_size = n_channels * total_samples * 2;
    ContainerStats {
        n_windows,
        n_channels,
        total_samples,
        compressed_size,
        raw_size,
        cr: if compressed_size > 0 {
            raw_size as f64 / compressed_size as f64
        } else {
            0.0
        },
        duration_s: if sample_rate > 0.0 {
            total_samples as f64 / sample_rate
        } else {
            0.0
        },
    }
}

/// Write a complete LML v1 container into any [`std::io::Write`] sink.
/// Same signature/semantics as the legacy
/// `lamquant_lml_legacy::container::write_into`; sourced from `signal` via
/// an `Abir::from_channels_i64` build, encoded through [`write_abir`].
pub fn write_into<W: std::io::Write>(
    sink: &mut W,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    write_into_with_modality(
        sink,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
        None,
    )
}

/// Like [`write_into`] but stamps an authoritative `Manual` modality TAG on the
/// BCS1 header (ADR 0074 Track I) — the born-typed ingest path. `modality_tag =
/// Some(t)` sets header byte 6 to `t` (source = `Manual`); `None` reproduces
/// today's born-`Untyped` bytes EXACTLY (the wrapper above). Byte-neutral apart
/// from the one provenance byte — proven by `modality_provenance_snapshot` — so it
/// applies uniformly to a single `.lml` and to every `.lma` archive entry.
#[allow(clippy::too_many_arguments)]
pub fn write_into_with_modality<W: std::io::Write>(
    sink: &mut W,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
    modality_tag: Option<u8>,
) -> LmlResult<ContainerStats> {
    let abir = Abir::from_channels_i64(signal.to_vec(), sample_rate);
    let abir = match modality_tag {
        Some(tag) => abir.with_manual_modality_tag(tag),
        None => abir,
    };
    let mut cw = CountingWriter {
        inner: sink,
        count: 0,
    };
    let n_windows = write_abir(
        &mut cw,
        &abir,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
        None,
        None,
    )?;
    let compressed_size = cw.count as usize;
    Ok(stats_from(n_windows, signal, compressed_size, sample_rate))
}

/// Write a complete LML v1 container file (default LPC mode). Thin wrapper
/// over [`write_file_with_mode`] — mirrors the legacy shim exactly.
pub fn write_file(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
) -> LmlResult<ContainerStats> {
    write_file_with_mode(
        path,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        LpcMode::default(),
    )
}

/// Write a complete LML v1 container file with explicit LPC mode.
pub fn write_file_with_mode(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    noise_bits: u8,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let stats = write_into(
        &mut f,
        signal,
        sample_rate,
        window_size,
        noise_bits,
        metadata_json,
        lpc_mode,
    )?;
    // Durability — flush the kernel page cache to disk before returning
    // (mirrors the legacy writer's `f.sync_all()`).
    f.sync_all().map_err(LmlError::Io)?;
    Ok(stats)
}

/// Write a complete LML container file in **bounded-MAE near-lossless**
/// mode (ADR 0051 track 2): every window is encoded with a guaranteed
/// `max|orig-recon| <= delta`.
#[allow(clippy::too_many_arguments)]
pub fn write_file_bounded_mae(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    delta: u64,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let abir = Abir::from_channels_i64(signal.to_vec(), sample_rate);
    let mut cw = CountingWriter {
        inner: &mut f,
        count: 0,
    };
    let n_windows = write_abir(
        &mut cw,
        &abir,
        sample_rate,
        window_size,
        0, // noise_bits unused in bounded mode
        metadata_json,
        lpc_mode,
        Some(delta),
        None,
    )?;
    let compressed_size = cw.count as usize;
    f.sync_all().map_err(LmlError::Io)?;
    Ok(stats_from(n_windows, signal, compressed_size, sample_rate))
}

/// Write a complete LML container file in **target-BPS rate-controlled
/// lossy** mode (ADR 0051 track 2 P2): each window is rate-controlled to
/// `target_bps`.
#[allow(clippy::too_many_arguments)]
pub fn write_file_target_bps(
    path: &Path,
    signal: &[Vec<i64>],
    sample_rate: f64,
    window_size: usize,
    target_bps: f64,
    metadata_json: &str,
    lpc_mode: LpcMode,
) -> LmlResult<ContainerStats> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).map_err(LmlError::Io)?;
    let mut f = std::fs::File::create(path).map_err(LmlError::Io)?;
    let abir = Abir::from_channels_i64(signal.to_vec(), sample_rate);
    let mut cw = CountingWriter {
        inner: &mut f,
        count: 0,
    };
    let n_windows = write_abir(
        &mut cw,
        &abir,
        sample_rate,
        window_size,
        0,
        metadata_json,
        lpc_mode,
        None,
        Some(target_bps),
    )?;
    let compressed_size = cw.count as usize;
    f.sync_all().map_err(LmlError::Io)?;
    Ok(stats_from(n_windows, signal, compressed_size, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{set_global_backend, ComputeBackend};
    use std::sync::{Mutex, OnceLock};

    /// Guards mutation of the process-wide `GLOBAL_BACKEND` selector so
    /// this test's Firmware/Desktop toggling has deterministic
    /// before/after state regardless of what else runs concurrently in
    /// this test binary (mirrors the `env_lock()` pattern in
    /// `tests/config_save.rs`). Note this is belt-and-braces, not a
    /// correctness requirement for OTHER tests: every `LpcMode` other
    /// than a live-deadline `Anytime` is backend-invariant by
    /// construction (the entire point of `ComputeBackend` — see
    /// `byte_equal_backends.rs`), so a concurrently-running test
    /// observing a transiently-different backend mid-encode would still
    /// get byte-identical output for its own (non-live-deadline) mode.
    fn backend_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn two_channel_signal() -> Vec<Vec<i64>> {
        vec![
            (0..600i64).map(|t| ((t * 37) % 4001) - 2000).collect(),
            (0..600i64).map(|t| ((t * 53) % 3001) - 1500).collect(),
        ]
    }

    /// Task #32 fix B (SAFE close, option b): a LIVE `Anytime` deadline
    /// must now produce identical `.lml` bytes on the `Firmware` and
    /// `Desktop` backend dispatch, because the `ComputeBackend::Desktop`
    /// arm above routes `Anytime{deadline: Some(_)}` through the same
    /// serial `compress_with_mode_views` kernel Firmware uses, instead of
    /// the rayon-parallel one.
    ///
    /// The deadline here is already in the past, which makes this test
    /// 100% deterministic: `analyze_anytime_host` reads `time_remains =
    /// Instant::now() < deadline`, and once `deadline` is behind "now",
    /// EVERY subsequent read of `Instant::now()` (on any thread, at any
    /// later point) is guaranteed to keep evaluating `false` — so both
    /// backends take the "budget exhausted -> Fixed schedule" branch on
    /// every subband, with no dependency on real-time scheduling.
    ///
    /// This intentionally does NOT reproduce the actual pre-fix race
    /// (which needs a deadline that expires MID-encode, at a point that
    /// differs between a monotonic serial loop and independently-
    /// scheduled rayon workers) — that is inherently non-deterministic
    /// without mocking `Instant::now()`, which the task #32 safe close
    /// explicitly defers (see the `ComputeBackend::Desktop` arm's doc
    /// comment above). This test instead locks the FIXED behavior: the
    /// routing added for task #32 must keep producing byte-identical
    /// output across backends for a live deadline, today and going
    /// forward.
    #[test]
    fn write_abir_anytime_live_deadline_matches_across_backends() {
        let _guard = backend_lock().lock().unwrap_or_else(|e| e.into_inner());

        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let past_deadline = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mode = LpcMode::Anytime {
            max_order: 16,
            deadline: Some(past_deadline),
        };

        set_global_backend(ComputeBackend::Firmware);
        let firmware = write_abir_to_vec(&abir, 250.0, 250, 0, "{}", mode, None, None)
            .expect("firmware encode");

        set_global_backend(ComputeBackend::Desktop);
        let desktop = write_abir_to_vec(&abir, 250.0, 250, 0, "{}", mode, None, None)
            .expect("desktop encode");

        set_global_backend(ComputeBackend::default());

        assert_eq!(
            firmware, desktop,
            "task #32: a live-deadline Anytime input must route Desktop through \
             the same serial path as Firmware, so both backends agree byte-for-byte"
        );
    }

    /// Sibling sanity check: the clock-free `Anytime{deadline: None}`
    /// path is UNCHANGED by the task #32 routing fix — `write_abir`'s
    /// `ComputeBackend::Desktop` arm still takes the parallel path for
    /// it. `byte_equal_backends.rs` already locks this at the lower
    /// (`compress_with_mode_parallel_views`) kernel level; this test
    /// confirms `write_abir`'s own dispatch didn't accidentally start
    /// special-casing `None` too (the `matches!` guard is specifically
    /// `deadline: Some(_)`).
    #[test]
    fn write_abir_anytime_no_deadline_still_matches_across_backends() {
        let _guard = backend_lock().lock().unwrap_or_else(|e| e.into_inner());

        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);
        let mode = LpcMode::Anytime {
            max_order: 16,
            deadline: None,
        };

        set_global_backend(ComputeBackend::Firmware);
        let firmware = write_abir_to_vec(&abir, 250.0, 250, 0, "{}", mode, None, None)
            .expect("firmware encode");

        set_global_backend(ComputeBackend::Desktop);
        let desktop = write_abir_to_vec(&abir, 250.0, 250, 0, "{}", mode, None, None)
            .expect("desktop encode");

        set_global_backend(ComputeBackend::default());

        assert_eq!(firmware, desktop);
    }

    /// ADR 0069/0071 L9 — the BCS1 header itself must be backend-invariant,
    /// not just the payload. Every header field (`n_channels`, `n_windows`,
    /// `total_samples`, `window_size`, `sample_rate_mhz`, `modality_tag`/
    /// `modality_source`, `mode`, `tier`, `decode_capability`,
    /// `metadata_length`) is computed from `abir`/`sample_rate`/
    /// `window_size`/`metadata_json`/`lpc_mode`/`delta`/`target_bps` BEFORE
    /// the `backend` match arm ever runs — `backend` only selects which
    /// per-window compression kernel runs. This test asserts FULL byte
    /// equality (header AND payload) across Firmware/Desktop for a spread of
    /// codec modes wider than the two Anytime-only tests above (which were
    /// written for the pre-existing task #32 edge case, not for L9): Fixed,
    /// Adaptive, BoundedMae, and TargetBps.
    #[test]
    fn write_abir_bcs1_output_matches_across_backends_for_multiple_codec_modes() {
        let _guard = backend_lock().lock().unwrap_or_else(|e| e.into_inner());

        let signal = two_channel_signal();
        let abir = Abir::from_channels_i64(signal, 250.0);

        struct Case {
            name: &'static str,
            lpc: LpcMode,
            delta: Option<u64>,
            target_bps: Option<f64>,
        }
        let cases = [
            Case { name: "fixed", lpc: LpcMode::Fixed, delta: None, target_bps: None },
            Case {
                name: "adaptive16",
                lpc: LpcMode::Adaptive { max_order: 16 },
                delta: None,
                target_bps: None,
            },
            Case {
                name: "bounded_mae_d8",
                lpc: LpcMode::default(),
                delta: Some(8),
                target_bps: None,
            },
            Case {
                name: "target_bps_4",
                lpc: LpcMode::default(),
                delta: None,
                target_bps: Some(4.0),
            },
        ];

        for case in cases {
            set_global_backend(ComputeBackend::Firmware);
            let firmware = write_abir_to_vec(
                &abir, 250.0, 250, 0, "{}", case.lpc, case.delta, case.target_bps,
            )
            .unwrap_or_else(|e| panic!("firmware encode ({}): {e:?}", case.name));

            set_global_backend(ComputeBackend::Desktop);
            let desktop = write_abir_to_vec(
                &abir, 250.0, 250, 0, "{}", case.lpc, case.delta, case.target_bps,
            )
            .unwrap_or_else(|e| panic!("desktop encode ({}): {e:?}", case.name));

            set_global_backend(ComputeBackend::default());

            assert_eq!(&firmware[0..4], BCS1_MAGIC, "case {}: firmware output must be BCS1", case.name);
            assert_eq!(
                firmware, desktop,
                "case {}: BCS1 output (header + payload) diverged across backends",
                case.name
            );
            // The header specifically (first BCS1_HEADER_LEN bytes) — a
            // narrower, explicit assertion so a future regression that only
            // touches the header (not the payload) still fails loudly here
            // rather than only showing up as "some byte differs".
            assert_eq!(
                &firmware[..BCS1_HEADER_LEN],
                &desktop[..BCS1_HEADER_LEN],
                "case {}: BCS1 HEADER diverged across backends",
                case.name
            );
        }
    }
}
