//! Synthesize a minimal valid 1-channel EDF from `i16` samples.
//!
//! Layout invariants (consumed by `lma::re_emit_synthetic`):
//!   - Header length = `SYNTH_EDF_HEADER_LEN` bytes (main 256 + signal 256).
//!   - Sample data is little-endian `i16`, immediately after header.
//!   - Exactly one signal, one record, `samples_per_record = samples.len()`.
//! Bumping any of these is a wire-format change — extract code that
//! reads back from the manifest's `synthetic_from` template depends
//! on the constants below being stable. Add new layouts as a new
//! `SyntheticFormat` variant, never by mutating this one.
//!
//! The EDF format has rigid ASCII-padded header fields. This helper
//! produces the smallest spec-compliant header so the LML codec can
//! consume the result via its existing EDF reader path. The synthesized
//! file is purely an internal intermediate — it's never written to
//! disk; the caller feeds the bytes directly into `encode_edf_to_lml`.
//!
//! Header layout (per EDF spec):
//!   - 256 bytes main header
//!   - 256 bytes per signal header
//!   - n_records * samples_per_record * 2 bytes of int16 LE data
//!
//! All ASCII fields are right-padded with spaces to their fixed widths.
//! Numeric ranges (`phys_min/max`, `dig_min/max`) span the full i16
//! domain so any value the parser observed survives the EDF round-trip
//! without clipping.

/// Build a minimal valid 1-channel EDF byte buffer. Single record,
/// `samples_per_record = samples.len()`, `record_duration = samples
/// .len() / sample_rate` seconds. `sample_rate` is informational only
/// — the codec doesn't depend on it for correctness.
///
/// Returns the in-memory byte buffer ready to be passed to the LML
/// encoder (which expects an on-disk file). Callers in `pack_archive`
/// must spill it to a temp file colocated with the output to avoid
/// the /tmp tmpfs ENOSPC trap from commit 5769562.
/// Header length of the synthesised single-channel EDF. Re-emit
/// path slices past this offset to recover the i16 sample data, so
/// keep in sync with any header layout change here.
pub const SYNTH_EDF_HEADER_LEN: usize = 256 + 256;

pub fn synth_single_channel_edf(samples: &[i16], sample_rate: f64) -> Vec<u8> {
    let n_samples = samples.len();
    let header_bytes = SYNTH_EDF_HEADER_LEN; // main + 1 signal
    let mut out = Vec::with_capacity(header_bytes + n_samples * 2);

    // ─── Main header (256 bytes) ──────────────────────────────────
    push_padded(&mut out, "0", 8); // version
    push_padded(&mut out, "X X X X", 80); // patient_id
    push_padded(&mut out, "Startdate X X X X", 80); // recording_id
    push_padded(&mut out, "01.01.01", 8); // startdate dd.mm.yy
    push_padded(&mut out, "00.00.00", 8); // starttime hh.mm.ss
    push_padded(&mut out, &header_bytes.to_string(), 8);
    push_padded(&mut out, "", 44); // reserved
    push_padded(&mut out, "1", 8); // n_records
    let record_dur = if sample_rate > 0.0 {
        (n_samples as f64 / sample_rate).max(1e-6)
    } else {
        1.0
    };
    push_padded(&mut out, &format!("{:.6}", record_dur), 8);
    push_padded(&mut out, "1", 4); // n_signals
    debug_assert_eq!(out.len(), 256, "EDF main header must be exactly 256 bytes");

    // ─── Signal header (256 bytes for the single signal) ──────────
    push_padded(&mut out, "EEG ch0", 16); // label
    push_padded(&mut out, "", 80); // transducer
    push_padded(&mut out, "uV", 8); // phys_dim
    // Use the full i16 domain so the parser's observed range is always
    // representable. Pre-fix versions used ±2048 (12-bit) which clipped
    // S-set Bonn data (range ±1885 — within ±2048 by luck, not design).
    push_padded(&mut out, "-32768", 8);
    push_padded(&mut out, "32767", 8);
    push_padded(&mut out, "-32768", 8);
    push_padded(&mut out, "32767", 8);
    push_padded(&mut out, "", 80); // prefiltering
    push_padded(&mut out, &n_samples.to_string(), 8);
    push_padded(&mut out, "", 32); // reserved
    debug_assert_eq!(out.len(), 512, "EDF signal header must bring total to 512");

    // ─── Sample data (little-endian i16) ──────────────────────────
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn push_padded(out: &mut Vec<u8>, value: &str, width: usize) {
    let bytes = value.as_bytes();
    if bytes.len() >= width {
        // Truncate — caller's responsibility to keep values short.
        out.extend_from_slice(&bytes[..width]);
    } else {
        out.extend_from_slice(bytes);
        for _ in bytes.len()..width {
            out.push(b' ');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_exactly_512() {
        let samples = vec![0i16; 256];
        let edf = synth_single_channel_edf(&samples, 256.0);
        assert_eq!(edf.len(), 512 + 256 * 2);
        assert_eq!(&edf[..8], b"0       ");
    }

    #[test]
    fn data_is_le_int16() {
        let samples: Vec<i16> = vec![0x1234, -1, 0x7FFF, i16::MIN];
        let edf = synth_single_channel_edf(&samples, 256.0);
        assert_eq!(&edf[512..514], &[0x34, 0x12]);
        assert_eq!(&edf[514..516], &[0xFF, 0xFF]);
        assert_eq!(&edf[516..518], &[0xFF, 0x7F]);
        assert_eq!(&edf[518..520], &[0x00, 0x80]);
    }

    #[test]
    fn n_signals_field_is_one() {
        let samples = vec![0i16; 256];
        let edf = synth_single_channel_edf(&samples, 256.0);
        // n_signals is the last 4 bytes of the main header.
        assert_eq!(&edf[252..256], b"1   ");
    }

    #[test]
    fn record_duration_reflects_sample_rate() {
        // 4097 samples @ 173.61 Hz (Bonn). Duration = 23.60... seconds.
        let samples = vec![0i16; 4097];
        let edf = synth_single_channel_edf(&samples, 173.61);
        // record_duration field is offset 244..252 in the main header.
        let dur_field = std::str::from_utf8(&edf[244..252]).unwrap();
        let dur: f64 = dur_field.trim().parse().unwrap();
        assert!((dur - 4097.0 / 173.61).abs() < 1e-3);
    }

    #[test]
    fn fallback_record_dur_when_sr_zero() {
        let samples = vec![0i16; 100];
        let edf = synth_single_channel_edf(&samples, 0.0);
        let dur_field = std::str::from_utf8(&edf[244..252]).unwrap();
        let dur: f64 = dur_field.trim().parse().unwrap();
        assert!((dur - 1.0).abs() < 1e-6);
    }
}
