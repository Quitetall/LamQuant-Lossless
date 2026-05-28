//! rANS (range Asymmetric Numeral Systems) entropy coding.
//!
//! Port of `lamquant_codec/ops/rans.py` — byte-identical output.
//!
//! Reference: Duda 2014, "Asymmetric numeral systems: entropy coding
//! combining speed of Huffman coding with compression rate of arithmetic coding"
//!
//! State invariant: RANS_L <= state < RANS_L * 256 after each symbol.
//!
//! ADR 0021: encoder + decoder return `Result<_, RansError>`. Old
//! `debug_assert!` (release-stripped) silent-failure paths replaced
//! with typed errors at the trust boundary. The PyO3 wrappers in
//! `lib.rs` surface these as `PyValueError`.

use alloc::vec::Vec;

use crate::codec_errors::RansError;

/// Upper bound on the n_symbols a caller can request the decoder to
/// produce. Bounds `Vec::with_capacity(n_symbols)` against attacker-
/// supplied size so an adversarial Python client can't force an OOM
/// abort by claiming billions of output symbols. EEG windows are
/// bounded at ~2500 samples × ~32 channels × ~4 subbands ≈ 320 K
/// max realistic n_symbols. 2^20 = 1 M is a generous defensive
/// ceiling; legitimate calls stay well below it.
pub const MAX_RANS_SYMBOLS: usize = 1 << 20;

/// Encode symbols given frequency and cumulative-start tables.
///
/// Processes symbols in reverse (rANS = LIFO). Output is a byte vector
/// whose format is identical to the Python `_encode_rans_jit` implementation.
///
/// Audit-2026-05-11 HIGH-G3:
/// - Fix-#14: assert non-negative symbol + bounds against `freq.len()`
///   before indexing. Previously a negative `sym` cast to `usize` would
///   wrap to a huge value and panic on `freq[sym as usize]` with no
///   diagnostic message.
/// - Fix-#15: assert `freq[sym] > 0` so the `state / f` below cannot
///   divide by zero. A symbol with `freq=0` is corrupt by construction
///   (it cannot have been emitted by the encoder side).
pub fn encode(
    symbols: &[i64],
    freq: &[i32],
    start: &[i32],
    m: u64,
) -> Result<Vec<u8>, RansError> {
    if m == 0 {
        return Err(RansError::ZeroDivisor);
    }
    // V4 Pro review of K1: `256 * m` overflows u64 when m >
    // u64::MAX / 256. Caller-supplied m crosses the FFI boundary
    // unbounded; cap explicitly so the subsequent renormalize
    // arithmetic stays correct.
    let rans_l = 256u64.checked_mul(m).ok_or(RansError::HeaderOverflow {
        claimed: m as usize,
        max_allowed: (u64::MAX / 256) as usize,
    })?;
    let rl_div_m: u64 = rans_l / m;

    let mut state: u64 = rans_l;
    // Worst case: ~4 bytes per symbol + 4 flush bytes.
    let mut out = Vec::with_capacity(symbols.len() * 4 + 16);

    // Encode in reverse.
    for &sym in symbols.iter().rev() {
        // ADR 0021: replace `debug_assert!` (release-stripped) with
        // runtime check. Symbol indexing must be sound at every
        // build profile, not just dev.
        if sym < 0 || (sym as usize) >= freq.len() {
            return Err(RansError::HeaderOverflow {
                claimed: sym as usize,
                max_allowed: freq.len(),
            });
        }
        let f = freq[sym as usize] as u64;
        if f == 0 {
            return Err(RansError::ZeroDivisor);
        }
        let s = start[sym as usize] as u64;
        let threshold = (rl_div_m * f) << 8;

        while state >= threshold {
            out.push((state & 0xFF) as u8);
            state >>= 8;
        }
        state = (state / f) * m + (state % f) + s;
    }

    // Flush final 32 bits of state.
    for _ in 0..4 {
        out.push((state & 0xFF) as u8);
        state >>= 8;
    }

    Ok(out)
}

/// Decode `n_symbols` from a rANS byte stream.
///
/// Reads bytes from the end of `data` (rANS is LIFO).
/// `cum2sym` is the inverse cumulative table: cum2sym[slot] -> symbol.
pub fn decode(
    data: &[u8],
    freq: &[i32],
    start: &[i32],
    cum2sym: &[i32],
    m: u64,
    n_symbols: usize,
) -> Result<Vec<i64>, RansError> {
    // ADR 0021: hard-bound every adversarial / pathological input
    // at entry. Old code used `debug_assert!` (release-stripped)
    // for cum2sym length + silent `break` on truncated state/zero-
    // freq symbols. Each of those is now a typed Err.
    if m == 0 {
        return Err(RansError::ZeroDivisor);
    }
    if (cum2sym.len() as u64) < m {
        return Err(RansError::HeaderOverflow {
            claimed: m as usize,
            max_allowed: cum2sym.len(),
        });
    }
    // Bound `Vec::with_capacity(n_symbols)` against attacker-
    // supplied size. PyO3 callers pass n_symbols through from a
    // Python header field; an adversarial value (billions of
    // symbols) used to OOM-abort here.
    if n_symbols > MAX_RANS_SYMBOLS {
        return Err(RansError::HeaderOverflow {
            claimed: n_symbols,
            max_allowed: MAX_RANS_SYMBOLS,
        });
    }
    // Same 256 * m guard as encoder (V4 Pro review fix).
    let rans_l = 256u64.checked_mul(m).ok_or(RansError::HeaderOverflow {
        claimed: m as usize,
        max_allowed: (u64::MAX / 256) as usize,
    })?;
    if data.len() < 4 {
        // Empty stream is a valid degenerate input -- caller may
        // be probing. Non-zero n_symbols on a stream too short to
        // reconstruct state is a truncation error.
        if n_symbols == 0 {
            return Ok(Vec::new());
        }
        return Err(RansError::TruncatedStream {
            at_byte: data.len(),
            bytes_needed: 4 - data.len(),
        });
    }
    let mut byte_idx: isize = data.len() as isize - 1;

    // Reconstruct initial state from final 4 bytes.
    let mut state: u64 = 0;
    for _ in 0..4 {
        if byte_idx >= 0 {
            state = (state << 8) | data[byte_idx as usize] as u64;
            byte_idx -= 1;
        }
    }

    let mut out = Vec::with_capacity(n_symbols);
    for _ in 0..n_symbols {
        let slot = (state % m) as usize;
        // Bounds checked by the cum2sym.len() >= m guard above;
        // the index can never reach cum2sym.len() because slot <
        // m <= cum2sym.len(). debug_assert documents the
        // invariant for future maintainers.
        debug_assert!(slot < cum2sym.len());
        let sym = cum2sym[slot];
        if sym < 0 || (sym as usize) >= freq.len() {
            // Adversarial cum2sym entry pointing outside freq
            // table -- old code silently broke out of the loop.
            return Err(RansError::HeaderOverflow {
                claimed: sym as usize,
                max_allowed: freq.len(),
            });
        }
        let f = freq[sym as usize] as u64;
        if f == 0 {
            return Err(RansError::ZeroDivisor);
        }
        let s_val = start[sym as usize] as u64;
        // ADR 0021 fix: `state - s_val` would underflow on
        // adversarial input. checked_sub surfaces it as
        // `StateUnderflow` so callers don't trip an unsigned
        // wrap-around.
        let numer = f.checked_mul(state / m).ok_or(RansError::StateUnderflow {
            state: state as u32,
            s_val: s_val as u32,
        })?;
        let with_remainder = numer + (state % m);
        state = with_remainder
            .checked_sub(s_val)
            .ok_or(RansError::StateUnderflow {
                state: with_remainder as u32,
                s_val: s_val as u32,
            })?;

        while state < rans_l && byte_idx >= 0 {
            state = (state << 8) | data[byte_idx as usize] as u64;
            byte_idx -= 1;
        }
        out.push(sym as i64);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build freq table and cumulative start from raw counts.
    fn build_tables(counts: &[i32]) -> (Vec<i32>, Vec<i32>, i64) {
        let m: i64 = counts.iter().map(|&c| c as i64).sum();
        let mut start = vec![0i32; counts.len()];
        for i in 1..counts.len() {
            start[i] = start[i - 1] + counts[i - 1];
        }
        (counts.to_vec(), start, m)
    }

    fn build_cum2sym(freq: &[i32], start: &[i32], m: usize) -> Vec<i32> {
        let mut cum2sym = vec![0i32; m];
        for (s, (&st, &f)) in start.iter().zip(freq.iter()).enumerate() {
            for j in st..(st + f) {
                cum2sym[j as usize] = s as i32;
            }
        }
        cum2sym
    }

    #[test]
    fn roundtrip_simple() {
        let freq = vec![1024i32, 2048, 1024];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let cum2sym = build_cum2sym(&freq, &start, m as usize);

        let symbols: Vec<i64> = vec![0, 1, 1, 2, 0, 1, 2, 1, 0, 0];
        let encoded = encode(&symbols, &freq, &start, m_u).expect("encodes ok");
        let decoded =
            decode(&encoded, &freq, &start, &cum2sym, m_u, symbols.len()).expect("decodes ok");
        assert_eq!(symbols, decoded);
    }

    #[test]
    fn roundtrip_uniform() {
        // All symbols equally likely.
        let n_sym = 8;
        let freq: Vec<i32> = vec![512; n_sym];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let cum2sym = build_cum2sym(&freq, &start, m as usize);

        let symbols: Vec<i64> = (0..200).map(|i| (i % n_sym as i64)).collect();
        let encoded = encode(&symbols, &freq, &start, m_u).expect("encodes ok");
        let decoded =
            decode(&encoded, &freq, &start, &cum2sym, m_u, symbols.len()).expect("decodes ok");
        assert_eq!(symbols, decoded);
    }

    /// Audit-2026-05-11 Fix-#48 + ADR 0021: truncated stream now
    /// distinguishes "n_symbols=0 → empty Ok" from "n_symbols>0 +
    /// stream too short → TruncatedStream Err". The old behaviour
    /// of returning empty for any truncated stream silently hid
    /// the difference.
    #[test]
    fn decode_handles_truncated_stream() {
        let freq = vec![512i32; 4];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let cum2sym = build_cum2sym(&freq, &start, m as usize);

        // n_symbols=0 on truncated stream is a no-op (probe).
        for short_len in 0..4 {
            let data = vec![0u8; short_len];
            let out = decode(&data, &freq, &start, &cum2sym, m_u, 0);
            assert!(matches!(out, Ok(v) if v.is_empty()), "n=0 short={short_len}");
        }
        // n_symbols>0 on truncated stream is an error.
        for short_len in 0..4 {
            let data = vec![0u8; short_len];
            match decode(&data, &freq, &start, &cum2sym, m_u, 10) {
                Err(RansError::TruncatedStream { .. }) => {}
                other => panic!("len={short_len}: expected TruncatedStream, got {:?}", other),
            }
        }
    }

    #[test]
    fn roundtrip_skewed() {
        // Highly skewed distribution.
        let freq = vec![3800i32, 100, 50, 30, 10, 6, 2, 2];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let cum2sym = build_cum2sym(&freq, &start, m as usize);

        let symbols: Vec<i64> = vec![0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0];
        let encoded = encode(&symbols, &freq, &start, m_u).expect("encodes ok");
        let decoded =
            decode(&encoded, &freq, &start, &cum2sym, m_u, symbols.len()).expect("decodes ok");
        assert_eq!(symbols, decoded);
    }

    // ─── ADR 0021 adversarial fixtures (K1) ────────────────────────

    #[test]
    fn encoder_rejects_m_zero() {
        // m=0 would divide-by-zero on threshold computation. Old
        // code silently UB'd; now hard-Err.
        match encode(&[], &[], &[], 0) {
            Err(RansError::ZeroDivisor) => {}
            other => panic!("expected ZeroDivisor, got {:?}", other),
        }
    }

    #[test]
    fn encoder_rejects_oversize_symbol() {
        // symbol index >= freq.len() used to debug_assert + index-
        // panic in release. Now: HeaderOverflow.
        let freq = vec![512i32; 4];
        let (freq, start, m) = build_tables(&freq);
        match encode(&[99], &freq, &start, m as u64) {
            Err(RansError::HeaderOverflow {
                claimed: 99,
                max_allowed: 4,
            }) => {}
            other => panic!("expected HeaderOverflow, got {:?}", other),
        }
    }

    #[test]
    fn encoder_rejects_zero_freq_symbol() {
        // Freq=0 for a referenced symbol used to debug_assert +
        // div-by-zero panic in release. Now: ZeroDivisor.
        let freq = vec![512i32, 0, 512];
        let (freq, start, m) = build_tables(&freq);
        match encode(&[1], &freq, &start, m as u64) {
            Err(RansError::ZeroDivisor) => {}
            other => panic!("expected ZeroDivisor, got {:?}", other),
        }
    }

    #[test]
    fn decoder_rejects_oversize_n_symbols() {
        // Attacker-supplied n_symbols used to drive
        // Vec::with_capacity(n_symbols) → OOM abort. Now: bounded
        // at MAX_RANS_SYMBOLS via HeaderOverflow.
        let freq = vec![512i32; 4];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let cum2sym = build_cum2sym(&freq, &start, m as usize);
        let data = vec![0u8; 10];
        match decode(&data, &freq, &start, &cum2sym, m_u, MAX_RANS_SYMBOLS + 1) {
            Err(RansError::HeaderOverflow { claimed, max_allowed })
                if claimed == MAX_RANS_SYMBOLS + 1 && max_allowed == MAX_RANS_SYMBOLS => {}
            other => panic!("expected HeaderOverflow, got {:?}", other),
        }
    }

    #[test]
    fn rejects_m_overflow() {
        // V4 Pro K1 review nit: `256 * m` would overflow u64 for
        // m > u64::MAX / 256. Now both encoder + decoder guard.
        let huge_m = (u64::MAX / 256) + 1; // first value that overflows
        match encode(&[], &[], &[], huge_m) {
            Err(RansError::HeaderOverflow { .. }) => {}
            other => panic!("encode: expected HeaderOverflow, got {:?}", other),
        }
        let dummy = vec![0u8; 10];
        match decode(&dummy, &[], &[], &[], huge_m, 0) {
            Err(RansError::HeaderOverflow { .. }) => {}
            other => panic!("decode: expected HeaderOverflow, got {:?}", other),
        }
    }

    #[test]
    fn decoder_rejects_cum2sym_too_small() {
        // cum2sym shorter than m used to debug_assert (release-
        // stripped). Now: HeaderOverflow.
        let freq = vec![512i32; 4];
        let (freq, start, m) = build_tables(&freq);
        let m_u = m as u64;
        let short_cum2sym = vec![0i32; (m as usize) - 1]; // ONE SHORT
        let data = vec![0u8; 10];
        match decode(&data, &freq, &start, &short_cum2sym, m_u, 5) {
            Err(RansError::HeaderOverflow { .. }) => {}
            other => panic!("expected HeaderOverflow, got {:?}", other),
        }
    }
}
