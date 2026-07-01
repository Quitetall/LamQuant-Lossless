//! ABIR container writer (ADR 0069 L6.2).
//!
//! A clean, faithful clone of the legacy v1 `encode_into` writer
//! (`lamquant-lml-legacy::container::encode_into`), sourced from
//! `lamquant_abir::Abir` instead of `&[Vec<i64>]`. BYTE-IDENTICAL to the legacy
//! writer by construction — proven by extending the L1 differential oracle
//! (`tests/oracle_diff.rs`).
//!
//! **The ONE change** vs the legacy encode loop: the per-window signal is built
//! via [`lamquant_abir::Abir::window_views`] (a `Cow`-backed O(1) sub-slice on the
//! `I64` lane) instead of `signal.iter().map(|ch| ch[start..end].to_vec())`. The
//! views are still immediately materialized into an owned `Vec<Vec<i64>>` here —
//! the kernels (`lml::compress_*`) take `&[Vec<i64>]` wholesale, so a real
//! zero-copy kernel is a later step (L6.3). Materialize, don't optimize: every
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
//! from `lamquant-lml-legacy`, gated by the `legacy-encode` feature that
//! `archive` already turns on) rather than re-cloned: it's a nontrivial CRC'd
//! binary seek-table serializer, not a "small helper", and duplicating it here
//! would itself be a wire-format-fork risk. **L8 note:** full legacy
//! independence needs `OffsetTable` relocated to a non-legacy-gated crate (or
//! copied here) — tracked, not resolved by L6.2.

use crate::deployment::LosslessMode;
use crate::error::{LmlError, LmlResult};
use crate::lml;
use crate::lpc::LpcMode;
use crate::offset_table::{OffsetEntry, OffsetTable};
use lamquant_abir::Abir;

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
#[allow(clippy::too_many_arguments)]
pub fn write_abir_to_vec(
    abir: &Abir,
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
#[allow(clippy::too_many_arguments)]
pub fn write_abir<W: std::io::Write + ?Sized>(
    sink: &mut W,
    abir: &Abir,
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
        // ADR 0069 L6.2: the ONE change vs the legacy loop — build the window
        // from `Abir::window_views` (Cow-backed O(1) sub-slice on the I64 lane)
        // instead of `signal.iter().map(|ch| ch[start..end].to_vec())`. Still
        // materializes an owned `Vec<Vec<i64>>` here; the kernels below take
        // `&[Vec<i64>]` wholesale. Zero-copy kernel dispatch is L6.3.
        let window: Vec<Vec<i64>> = abir
            .window_views(start, end)
            .iter()
            .map(|c| c.to_vec())
            .collect();
        let compressed = if let Some(tb) = target_bps {
            lml::compress_target_bps(&window, tb, lpc_mode)?
        } else if let Some(d) = delta {
            lml::compress_bounded_mae(&window, d, lpc_mode)?
        } else {
            match backend {
                crate::backend::ComputeBackend::Firmware => {
                    lml::compress_with_mode(&window, noise_bits, lpc_mode)?
                }
                crate::backend::ComputeBackend::Desktop => {
                    crate::compress_with_mode_parallel(&window, noise_bits, lpc_mode)?
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
