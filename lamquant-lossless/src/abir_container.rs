//! ABIR container writer (ADR 0069 L6.2, zero-copy kernel L6.3).
//!
//! A clean, faithful clone of the legacy v1 `encode_into` writer
//! (`lamquant-lml-legacy::container::encode_into`), sourced from
//! `lamquant_abir::Abir` instead of `&[Vec<i64>]`. BYTE-IDENTICAL to the legacy
//! writer by construction — proven by extending the L1 differential oracle
//! (`tests/oracle_diff.rs`).
//!
//! **The lossless dispatch is zero-copy (L6.3).** The per-window signal is
//! built via [`lamquant_abir::Abir::window_views`] (a `Cow`-backed O(1)
//! sub-slice on the `I64` lane); the `&[i64]` borrows off those `Cow`s are
//! handed straight to `lml::compress_with_mode_views` /
//! `compress_with_mode_parallel_views` — no `.to_vec()`, no per-window
//! `Vec<Vec<i64>>` materialization. That was the L6.2-era memcpy; L6.3 kills
//! it on the hot (lossless, `noise_bits == 0`) path. The cold bounded-MAE /
//! target-BPS arms still materialize an owned `Vec<Vec<i64>>` from the same
//! `Cow`s — those kernels (`compress_bounded_mae`/`compress_target_bps`)
//! take `&[Vec<i64>]` wholesale and are RD-search paths, not the hot loop;
//! a view-taking variant for them is a tracked fast-follow, not L6.3. Every
//! other byte (header, metadata, offset table, footer) is produced by logic
//! cloned verbatim from the legacy writer, so the output is bit-for-bit
//! unchanged.
//!
//! **Self-containment (L6.2):** this module does NOT call
//! `lamquant_lml_legacy::container::encode_into`. The small write-only helpers
//! (`metadata_with_codec_mode`, `lossless_mode_for_lpc_mode`, the 32-byte-header
//! constants) are cloned here verbatim. `write_abir` itself mirrors
//! `encode_into`'s signature/body directly (it IS the encode-loop level, not the
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
//! READ side (`read_file`, `read_bytes`, `read_from`, `parse_header`,
//! `ContainerHeader`, `read_window_from_bytes`,
//! `read_bytes_into_f32_calibrated`) and shared types (`ContainerStats`,
//! `OffsetEntry`, `OffsetTable`). `lamquant_core::container` (`lib.rs`) is
//! aliased to `abir_container` at the cutover, so every existing
//! `container::*` call site keeps compiling unchanged while the write half
//! now goes through `write_abir`.

use crate::deployment::LosslessMode;
use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::lpc::LpcMode;
use lamquant_abir::{Abir, Modality};
use std::path::Path;

// Re-exported at `abir_container::{OffsetEntry, OffsetTable}` (this `pub use`
// both imports them for `write_abir`'s own use below AND makes them resolve
// as `container::{OffsetEntry, OffsetTable}` post-cutover, per the L8 facade
// contract above).
pub use crate::offset_table::{OffsetEntry, OffsetTable};

// The FROZEN v1 reader + shared write-result/parse types, re-exported
// verbatim from `lamquant-lml-legacy` (always available under
// `legacy-decode`, which `archive` keeps on). These are unchanged by the
// cutover — only the WRITE half moves to `write_abir` below.
pub use lamquant_lml_legacy::container::{
    parse_header, read_bytes, read_bytes_into_f32_calibrated, read_file, read_from,
    read_window_from_bytes, ContainerHeader, ContainerStats,
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
/// is modality-blind by design (see `lamquant_abir::modality`), so this
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
                    crate::compress_with_mode_parallel_views(&views, noise_bits, lpc_mode)?
                }
            }
        };
        window_payloads.push(compressed);
    }

    // 32-byte header (spec Section 2.1)
    sink.write_all(lml::MAGIC).map_err(LmlError::Io)?;
    sink.write_all(&[VERSION_MAJOR]).map_err(LmlError::Io)?; // byte 4
    sink.write_all(&[VERSION_MINOR]).map_err(LmlError::Io)?; // byte 5
    sink.write_all(&(n_ch as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 6-7
    sink.write_all(&(n_windows as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 8-9
    sink.write_all(&(total_samples as u32).to_le_bytes())
        .map_err(LmlError::Io)?; // 10-13
    sink.write_all(&(actual_window as u16).to_le_bytes())
        .map_err(LmlError::Io)?; // 14-15
    sink.write_all(&sample_rate_mhz.to_le_bytes())
        .map_err(LmlError::Io)?; // 16-19
    sink.write_all(&[16u8]).map_err(LmlError::Io)?; // 20: bit_depth (default 16)
    sink.write_all(&[FLAG_HAS_FOOTER]).map_err(LmlError::Io)?; // 21: flags
    sink.write_all(&(meta_bytes.len() as u32).to_le_bytes())
        .map_err(LmlError::Io)?; // 22-25
    sink.write_all(&[0u8; 2]).map_err(LmlError::Io)?; // 26-27: reserved_0
    sink.write_all(&[0u8; 4]).map_err(LmlError::Io)?; // 28-31: reserved_1

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

    // The header is fixed 32 bytes; metadata is `meta_bytes.len()`; window index
    // is `n_windows * 4`. The first payload's length prefix begins immediately
    // after.
    let first_payload_abs = 32u64 + meta_bytes.len() as u64 + n_windows as u64 * 4;

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
    let abir = Abir::from_channels_i64(signal.to_vec(), sample_rate);
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
