//! EDF/EDF+/BDF reader — safe, hardened for clinical use.
//!
//! Supports the complete EDF family:
//!   - EDF (1992): int16, continuous
//!   - EDF+ continuous (EDF+C): int16, TAL annotations
//!   - EDF+ discontinuous (EDF+D): int16, TAL annotations
//!   - BDF (BioSemi): int24, continuous
//!
//! No unsafe code. Correct on all architectures.

use crate::error::{LmlError, LmlResult};
use crate::source::ascii::{parse_float, parse_i64, parse_usize};
use crate::source::bitstream::read_i24_le;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct EdfFile {
    pub signal: Vec<Vec<i64>>,
    pub channels: Vec<String>,
    pub sample_rate: f64,
    pub n_channels: usize,
    pub total_samples: usize,
    pub duration_s: f64,
    pub source_file: String,
    pub patient_id: String,
    // Full header preservation for bit-exact roundtrip
    pub raw_header: Vec<u8>, // main (256) + signal headers (n_signals * 256)
    pub non_eeg_data: Vec<(usize, Vec<u8>)>, // (channel_index, raw_bytes) for non-EEG channels
    pub n_signals_total: usize,
    pub n_data_records: usize,
    pub record_duration: f64,
    pub all_labels: Vec<String>,
    pub all_ns_per_rec: Vec<usize>,
    pub eeg_indices: Vec<usize>,
    pub recording_info: String,
    pub startdate: String,
    pub format: String, // "EDF", "EDF+C", "EDF+D", "BDF"
    pub phys_min: Vec<f64>,
    pub phys_max: Vec<f64>,
    pub dig_min: Vec<i32>,
    pub dig_max: Vec<i32>,
    pub phys_dim: String,
    pub trailing_data: Vec<u8>, // bytes beyond last complete record (partial records)
    pub is_bdf: bool,           // true for BDF (24-bit), false for EDF (16-bit)
}

pub fn read_edf(path: &Path) -> LmlResult<EdfFile> {
    let data = fs::read(path).map_err(LmlError::Io)?;
    if data.len() < 256 {
        return Err(LmlError::Truncated {
            expected: 256,
            actual: data.len(),
            context: "EDF header",
        });
    }

    // ── Check for Unicode BOM (common corruption) ──
    if data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF {
        return Err(LmlError::InvalidHeader(
            "File starts with UTF-8 BOM (0xEF BB BF). Not a valid EDF/BDF — strip the BOM.".into(),
        ));
    }
    if data[0] == 0xFE && data[1] == 0xFF {
        return Err(LmlError::InvalidHeader(
            "File starts with UTF-16 BE BOM (0xFE FF). Not a valid EDF/BDF.".into(),
        ));
    }

    // ── Detect format: EDF vs BDF ──
    // ADR 0021: pre-fix code did `&data[1..4] == b"\xFE"` which
    // compared a 3-byte slice against a 1-byte literal -- the
    // equality is always false and EVERY file with byte 0xFF at
    // position 0 was accepted as BDF, including UTF-16 LE BOM
    // files and arbitrary corrupt input. Fix: require bytes 1-7
    // to spell "BIOSEMI" per the BDF spec.
    let is_bdf = data[0] == 0xFF;
    if is_bdf {
        if data.len() < 8 || &data[1..8] != b"BIOSEMI" {
            let preview = &data[..data.len().min(8)];
            return Err(LmlError::InvalidHeader(format!(
                "File starts with 0xFF but bytes 1-7 are not 'BIOSEMI' (got {:02X?}); \
                 not a valid BDF",
                preview
            )));
        }
    }
    let bytes_per_sample: usize = if is_bdf { 3 } else { 2 };

    // ── Main header (256 bytes) ──
    let n_data_records_raw: i64 = parse_i64(&data[236..244])?;
    if n_data_records_raw <= 0 {
        return Err(LmlError::InvalidHeader(format!(
            "File declares {} data records. Cannot encode.",
            n_data_records_raw
        )));
    }
    let n_data_records = n_data_records_raw as usize;

    let dur_record: f64 = parse_float(&data[244..252])?;
    if dur_record <= 0.0 || !dur_record.is_finite() {
        return Err(LmlError::InvalidHeader(format!(
            "EDF record duration {} is invalid (must be > 0)",
            dur_record
        )));
    }

    let n_signals: usize = parse_usize(&data[252..256])?;
    if n_signals == 0 {
        return Err(LmlError::InvalidHeader("0 EDF signals".into()));
    }

    // ── Signal headers ──
    let sig_hdr_off = 256usize;
    let sig_hdr_size = 256 * n_signals;
    if data.len() < sig_hdr_off + sig_hdr_size {
        return Err(LmlError::Truncated {
            expected: sig_hdr_off + sig_hdr_size,
            actual: data.len(),
            context: "EDF signal headers",
        });
    }
    let sh = &data[sig_hdr_off..sig_hdr_off + sig_hdr_size];

    // Field widths and cumulative offsets
    const WIDTHS: [usize; 10] = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32];
    let mut offsets = [0usize; 10];
    for i in 1..10 {
        offsets[i] = offsets[i - 1] + WIDTHS[i - 1] * n_signals;
    }

    let field = |fi: usize, si: usize| -> &[u8] {
        let o = offsets[fi] + si * WIDTHS[fi];
        &sh[o..o + WIDTHS[fi]]
    };
    let field_str = |fi: usize, si: usize| -> String {
        String::from_utf8_lossy(field(fi, si)).trim().to_string()
    };
    let field_int = |fi: usize, si: usize| -> LmlResult<usize> {
        let s = field_str(fi, si);
        s.parse().map_err(|e| {
            LmlError::InvalidHeader(format!("EDF signal {} field {}: '{}' — {}", si, fi, s, e))
        })
    };

    let labels: Vec<String> = (0..n_signals).map(|i| field_str(0, i)).collect();
    let ns_per_rec: Vec<usize> = (0..n_signals)
        .map(|i| field_int(8, i))
        .collect::<LmlResult<Vec<_>>>()?;

    // Validate: no channel has 0 samples per record
    for (i, &ns) in ns_per_rec.iter().enumerate() {
        if ns == 0 {
            return Err(LmlError::InvalidHeader(format!(
                "Channel {} ('{}') has 0 samples per record",
                i, labels[i]
            )));
        }
    }

    // Filter: EEG channels = non-annotation channels at mode sample rate
    let eeg_idx_all: Vec<usize> = (0..n_signals)
        .filter(|&i| !labels[i].to_lowercase().contains("annotation"))
        .collect();
    if eeg_idx_all.is_empty() {
        return Err(LmlError::InvalidHeader(
            "no EEG channels (all annotations)".into(),
        ));
    }

    // Pick the rate group with the most TOTAL DATA, not the most channels.
    // PSG files commonly carry 3 high-rate EEG channels alongside 4 low-rate
    // markers — counting channels would route the EEG to the zstd fallback
    // (CR ≈ 1.2) instead of the DWT+LPC pipeline (CR ≈ 2.5). Weight each
    // rate by (channels × samples_per_record) to follow the data volume.
    //
    // Audit-2026-05-11 Fix-#29: explicit Err on empty/zero-mode rather
    // than relying on upstream guards. `sr_weights` is non-empty here
    // (eeg_idx_all checked earlier) but `mode_ns == 0` would still slip
    // through and divide-by-zero downstream. Both branches return
    // InvalidHeader with diagnostic context.
    let mut sr_weights: HashMap<usize, usize> = HashMap::new();
    for &i in &eeg_idx_all {
        // Weight by samples-per-record (channels at the same rate accumulate).
        *sr_weights.entry(ns_per_rec[i]).or_insert(0) += ns_per_rec[i];
    }
    let mode_ns = match sr_weights.iter().max_by_key(|&(_, w)| w) {
        Some((&n, _)) if n > 0 => n,
        Some((&n, _)) => {
            return Err(LmlError::InvalidHeader(format!(
                "EDF mode sample rate is {n} (must be > 0)"
            )));
        }
        None => {
            return Err(LmlError::InvalidHeader(
                "EDF has no countable sample rates".into(),
            ));
        }
    };

    // Off-rate channels go through the secondary (zstd) lossless path inside
    // non_eeg_data — bit-exact roundtrip is preserved, they just don't enter
    // the DWT+LPC+Golomb-Rice pipeline (which assumes uniform sample rate).
    let off_rate: Vec<(&str, usize)> = eeg_idx_all
        .iter()
        .filter(|&&i| ns_per_rec[i] != mode_ns)
        .map(|&i| (labels[i].as_str(), ns_per_rec[i]))
        .collect();
    if !off_rate.is_empty() {
        let names: Vec<String> = off_rate
            .iter()
            .map(|(n, ns)| format!("{} ({} samp/rec)", n, ns))
            .collect();
        eprintln!(
            "Note: {} channel(s) at non-mode sample rate stored via secondary lossless path \
             (zstd) instead of DWT pipeline; bit-exact roundtrip preserved: {}",
            off_rate.len(),
            names.join(", ")
        );
    }

    let eeg_idx: Vec<usize> = eeg_idx_all
        .into_iter()
        .filter(|&i| ns_per_rec[i] == mode_ns)
        .collect();
    let n_ch = eeg_idx.len();
    let sr = mode_ns as f64 / dur_record;

    // ── Data block ──
    //
    // Audit-2026-05-11 Fix-C6: use checked_add so a malicious EDF header
    // with many huge `ns_per_rec` values cannot overflow `usize` on
    // 32-bit targets (max ~4 GB; 1024 channels × 8-digit ASCII = ~10^11).
    let total_per_rec: usize = ns_per_rec
        .iter()
        .try_fold(0usize, |acc, &n| acc.checked_add(n))
        .ok_or_else(|| LmlError::InvalidHeader("total_per_rec sum overflows usize".into()))?;
    let data_start = sig_hdr_off + sig_hdr_size;
    if data_start > data.len() {
        return Err(LmlError::Truncated {
            expected: data_start,
            actual: data.len(),
            context: "data block start",
        });
    }
    let available_bytes = data.len() - data_start;
    let bytes_per_rec = total_per_rec * bytes_per_sample;
    if bytes_per_rec == 0 {
        return Err(LmlError::InvalidHeader("computed record size is 0".into()));
    }
    let usable_records = (available_bytes / bytes_per_rec).min(n_data_records);
    if usable_records == 0 {
        return Err(LmlError::InvalidHeader(
            "EDF data block empty or truncated".into(),
        ));
    }

    if usable_records < n_data_records {
        eprintln!(
            "WARNING: EDF truncated — header declares {} records, \
                  only {} complete records available ({} dropped)",
            n_data_records,
            usable_records,
            n_data_records - usable_records
        );
    }

    // Capture trailing bytes beyond the last complete record.
    // ADR 0021: checked_mul on usable_records * bytes_per_rec so
    // a hostile header (huge usable_records × bytes_per_rec) can't
    // wrap usize on 32-bit MCU targets.
    let usable_data_bytes = usable_records
        .checked_mul(bytes_per_rec)
        .ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "EDF header arithmetic overflow: usable_records ({}) * bytes_per_rec ({})",
                usable_records, bytes_per_rec
            ))
        })?;
    let trailing_data: Vec<u8> = if available_bytes > usable_data_bytes {
        data[data_start + usable_data_bytes..].to_vec()
    } else {
        Vec::new()
    };

    // ADR 0021: checked_mul on mode_ns * usable_records too --
    // the subsequent `vec![vec![0i64; total_samples]; n_ch]`
    // allocation depends on this product being right-sized.
    let total_samples = mode_ns.checked_mul(usable_records).ok_or_else(|| {
        LmlError::InvalidHeader(format!(
            "EDF header arithmetic overflow: mode_ns ({}) * usable_records ({})",
            mode_ns, usable_records
        ))
    })?;
    if total_samples < 4 {
        return Err(LmlError::InvalidHeader(format!(
            "EDF too short: {} samples",
            total_samples
        )));
    }

    // Per-signal sample offsets within each record.
    //
    // Audit-2026-05-11 Fix-C6: defensive bounds check. By construction the
    // running `pos` accumulator never exceeds `total_per_rec` (since
    // `total_per_rec = sum(ns_per_rec)`), but we make it explicit + assert
    // post-hoc so any future refactor that breaks the invariant fails
    // loudly. Use `checked_add` for usize overflow safety on 32-bit
    // targets where many large `ns_per_rec` values could overflow.
    let mut rec_offsets = Vec::with_capacity(n_signals);
    let mut pos = 0usize;
    for i in 0..n_signals {
        rec_offsets.push(pos);
        pos = pos.checked_add(ns_per_rec[i]).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "channel {} ns_per_rec accumulator overflow (pos={}, ns={})",
                i, pos, ns_per_rec[i]
            ))
        })?;
    }
    // Per-channel post-condition: every (offset, ns) pair must fit inside
    // one record. Cheap defense against a future bug where rec_offsets is
    // built from a different source than `ns_per_rec.iter().sum()`.
    for ch in 0..n_signals {
        let end = rec_offsets[ch].checked_add(ns_per_rec[ch]).ok_or_else(|| {
            LmlError::InvalidHeader(format!(
                "channel {} rec_offset {} + ns_per_rec {} overflows usize",
                ch, rec_offsets[ch], ns_per_rec[ch]
            ))
        })?;
        if end > total_per_rec {
            return Err(LmlError::InvalidHeader(format!(
                "channel {} rec_offset {} + ns_per_rec {} = {} exceeds total_per_rec {}",
                ch, rec_offsets[ch], ns_per_rec[ch], end, total_per_rec
            )));
        }
    }

    // ── Deinterleave: gather EEG channels from interleaved records ──
    let data_slice = &data[data_start..data_start + usable_data_bytes];
    let mut signal = vec![vec![0i64; total_samples]; n_ch];

    if is_bdf {
        // BDF: 3 bytes per sample, int24 little-endian with sign extension
        for (j, &ch) in eeg_idx.iter().enumerate() {
            let o = rec_offsets[ch];
            let ns = ns_per_rec[ch];
            let out = &mut signal[j];

            for r in 0..usable_records {
                let out_base = r * ns;
                for s in 0..ns {
                    let byte_off = (r * total_per_rec + o + s) * 3;
                    let val = read_i24_le(data_slice, byte_off);
                    out[out_base + s] = val as i64;
                }
            }
        }
    } else {
        // EDF: 2 bytes per sample, int16 little-endian
        for (j, &ch) in eeg_idx.iter().enumerate() {
            let o = rec_offsets[ch];
            let ns = ns_per_rec[ch];
            let out = &mut signal[j];

            for r in 0..usable_records {
                let out_base = r * ns;
                for s in 0..ns {
                    let byte_off = (r * total_per_rec + o + s) * 2;
                    let val = i16::from_le_bytes([data_slice[byte_off], data_slice[byte_off + 1]]);
                    out[out_base + s] = val as i64;
                }
            }
        }
    }

    // Audit-2026-05-11 Fix-#21: warn when EDF header fields contain
    // non-printable-ASCII bytes (typically Latin-1 patient names or
    // CRLF litter). `from_utf8_lossy` silently replaces invalid bytes
    // with U+FFFD; the warning surfaces the corruption so downstream
    // consumers can at least know provenance is uncertain.
    let warn_if_non_ascii = |field: &[u8], name: &str| {
        if field
            .iter()
            .any(|&b| !(0x20..=0x7E).contains(&b) && b != 0x20)
        {
            eprintln!(
                "WARNING: EDF header field '{name}' contains non-ASCII bytes \
                 (Latin-1 or binary); will be UTF-8-lossy-decoded with U+FFFD \
                 substitution. Patient/recording metadata may be corrupted."
            );
        }
    };
    warn_if_non_ascii(&data[8..88], "patient_id");
    warn_if_non_ascii(&data[88..168], "recording_info");
    let patient_id = String::from_utf8_lossy(&data[8..88]).trim().to_string();
    let recording_info = String::from_utf8_lossy(&data[88..168]).trim().to_string();
    let startdate = format!(
        "{} {}",
        String::from_utf8_lossy(&data[168..176]).trim(),
        String::from_utf8_lossy(&data[176..184]).trim()
    );
    let ch_names: Vec<String> = eeg_idx.iter().map(|&i| labels[i].clone()).collect();

    // Detect format
    let reserved = String::from_utf8_lossy(&data[192..236]).trim().to_string();
    let format = if is_bdf {
        "BDF"
    } else if reserved.contains("EDF+D") {
        "EDF+D"
    } else if reserved.contains("EDF+C") {
        "EDF+C"
    } else {
        "EDF"
    }
    .to_string();

    // Calibration for EEG channels — Audit-2026-05-11 Fix-#22:
    // propagate parse errors. Previously `unwrap_or(-32768.0)` silently
    // substituted dummy bounds when phys_min/max failed to parse,
    // producing wildly miscalibrated physical units (μV ↔ μV swap, or
    // dimensionless raw samples treated as calibrated voltage). For
    // clinical EEG this is a **safety-critical** silent failure: a
    // patient's seizure detection could fire on the wrong amplitude
    // threshold. Return InvalidHeader so the caller learns the file is
    // malformed instead of receiving garbage.
    let parse_f64 = |fi: usize, label: &str, ch: usize| -> LmlResult<f64> {
        let s = field_str(fi, ch);
        let val: f64 = s.parse().map_err(|e| {
            LmlError::InvalidHeader(format!("EDF channel {ch} {label}: '{s}' — {e}"))
        })?;
        // ADR 0021: f64::from_str accepts "NaN", "Infinity",
        // "+Inf", "-Inf" as valid float syntax. Non-finite scale
        // factors propagate into LML metadata as garbage μV
        // calibration that downstream clinical tooling silently
        // accepts. Hard-reject at the parse boundary.
        if !val.is_finite() {
            return Err(LmlError::InvalidHeader(format!(
                "EDF channel {ch} {label}: '{s}' parsed as non-finite ({:?}); \
                 NaN/Inf scale factors are rejected",
                val
            )));
        }
        Ok(val)
    };
    let parse_i32 = |fi: usize, label: &str, ch: usize| -> LmlResult<i32> {
        let s = field_str(fi, ch);
        s.parse()
            .map_err(|e| LmlError::InvalidHeader(format!("EDF channel {ch} {label}: '{s}' — {e}")))
    };
    let phys_min: Vec<f64> = eeg_idx
        .iter()
        .map(|&i| parse_f64(3, "phys_min", i))
        .collect::<LmlResult<Vec<_>>>()?;
    let phys_max: Vec<f64> = eeg_idx
        .iter()
        .map(|&i| parse_f64(4, "phys_max", i))
        .collect::<LmlResult<Vec<_>>>()?;
    let dig_min: Vec<i32> = eeg_idx
        .iter()
        .map(|&i| parse_i32(5, "dig_min", i))
        .collect::<LmlResult<Vec<_>>>()?;
    let dig_max: Vec<i32> = eeg_idx
        .iter()
        .map(|&i| parse_i32(6, "dig_max", i))
        .collect::<LmlResult<Vec<_>>>()?;
    let phys_dim = if !eeg_idx.is_empty() {
        field_str(2, eeg_idx[0])
    } else {
        "uV".into()
    };

    // Raw header bytes for bit-exact preservation
    let raw_header = data[0..sig_hdr_off + sig_hdr_size].to_vec();

    // Non-EEG channel raw data (annotation channels, EMG, ECG at different rates, etc.).
    // ADR 0021: pre-fix `if end <= data.len()` silently skipped
    // records whose declared layout extended past the file -- the
    // resulting ch_bytes had wrong length and the EDF reader
    // claimed success despite truncating data. Now hard-Err so
    // the caller knows the file is malformed.
    let ann_idx: Vec<usize> = (0..n_signals).filter(|i| !eeg_idx.contains(i)).collect();
    let mut non_eeg_data = Vec::new();
    for &ch in &ann_idx {
        let o = rec_offsets[ch];
        let ns = ns_per_rec[ch];
        let chunk_bytes = ns * bytes_per_sample;
        let mut ch_bytes = Vec::with_capacity(usable_records * chunk_bytes);
        for r in 0..usable_records {
            let start = data_start + (r * total_per_rec + o) * bytes_per_sample;
            let end = start + chunk_bytes;
            if end > data.len() {
                return Err(LmlError::InvalidHeader(format!(
                    "EDF non-EEG channel {} (label {:?}): record {} extends past end of file \
                     (record_end={}, file_size={}); data is truncated",
                    ch,
                    labels.get(ch).map(|s| s.as_str()).unwrap_or("?"),
                    r,
                    end,
                    data.len()
                )));
            }
            ch_bytes.extend_from_slice(&data[start..end]);
        }
        non_eeg_data.push((ch, ch_bytes));
    }

    Ok(EdfFile {
        signal,
        channels: ch_names,
        sample_rate: sr,
        n_channels: n_ch,
        total_samples,
        duration_s: total_samples as f64 / sr,
        source_file: path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        patient_id,
        raw_header,
        non_eeg_data,
        n_signals_total: n_signals,
        n_data_records: usable_records,
        record_duration: dur_record,
        all_labels: labels,
        all_ns_per_rec: ns_per_rec,
        eeg_indices: eeg_idx,
        recording_info,
        startdate,
        format,
        phys_min,
        phys_max,
        dig_min,
        dig_max,
        phys_dim,
        trailing_data,
        is_bdf,
    })
}

// Parse helpers (`parse_usize`, `parse_i64`, `parse_float`) and the
// 24-bit read (`read_i24_le`) moved to `crate::source::ascii` and
// `crate::source::bitstream` in Phase 0.2 so BrainVision/CNT/raw
// readers can reuse them. EDF behaviour unchanged.
