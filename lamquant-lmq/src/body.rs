//! ADR 0074 Track N — the FSQ→rANS neural **body** codec (the bytes past the
//! outer BCS2 LMQ packet).
//!
//! Backend-independent: it operates on already-produced FSQ tokens (unsigned
//! symbols in `[0, alphabet)`) + the per-timestep schedule, entropy-codes the
//! tokens with the byte-exact [`lamquant_lml_mcu::rans`] coder, and frames them
//! self-describingly so [`decode_body`] is standalone. rANS carries **no model in
//! the stream**, so the frequency `counts` travel IN-BAND.
//!
//! Body layout (little-endian):
//! ```text
//! version:u8 | n_symbols:u32 | alphabet:u16 | counts:[u32; alphabet]
//!            | sched_len:u32 | schedule:[u8; sched_len]
//!            | rans_len:u32  | rans:[u8; rans_len]
//! ```

use alloc::vec;
use alloc::vec::Vec;

use lamquant_lml_mcu::rans;

/// Body-format version (the first byte of the LMQ token body).
pub const LMQ_BODY_VERSION: u8 = 1;

/// Failure encoding/decoding a neural body.
#[derive(Debug, PartialEq, Eq)]
pub enum BodyError {
    /// Ran out of bytes parsing the body.
    Truncated,
    /// Unknown body-format version.
    BadVersion(u8),
    /// Empty / zero-sum frequency model — no valid rANS alphabet.
    EmptyModel,
    /// Malformed frequency model from an untrusted body: a negative count, or a
    /// total that exceeds [`MAX_MODEL_TOTAL`] (would drive the `cum2sym`
    /// allocation huge / overflow the cumulative `i32`). Fail-closed.
    BadModel,
    /// The entropy coder rejected the stream/model.
    Rans,
}

/// Upper bound on the rANS total (Σ counts). The real FSQ model normalizes to
/// ~4096; `1 << 20` is a generous ceiling that (a) bounds the `cum2sym`
/// allocation against a crafted body and (b) keeps the cumulative sum well under
/// `i32::MAX`, so the `start` table can never wrap.
pub const MAX_MODEL_TOTAL: u64 = 1 << 20;

/// Frame FSQ `tokens` (symbols in `[0, counts.len())`), the per-timestep
/// `schedule`, and the frequency `counts` (the rANS model) into a self-describing
/// body. `decode_body(encode_body(..)) == (tokens, schedule)`.
pub fn encode_body(tokens: &[i64], schedule: &[u8], counts: &[i32]) -> Result<Vec<u8>, BodyError> {
    let (freq, start, m) = build_tables(counts)?;
    let rans_bytes = rans::encode(tokens, &freq, &start, m).map_err(|_| BodyError::Rans)?;

    // Fixed prefix is 15 bytes (version 1 + n_symbols 4 + alphabet 2 + sched_len
    // 4 + rans_len 4) + the counts + schedule + rANS.
    let mut out = Vec::with_capacity(15 + counts.len() * 4 + schedule.len() + rans_bytes.len());
    out.push(LMQ_BODY_VERSION);
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    out.extend_from_slice(&(counts.len() as u16).to_le_bytes());
    for &c in counts {
        out.extend_from_slice(&(c as u32).to_le_bytes());
    }
    out.extend_from_slice(&(schedule.len() as u32).to_le_bytes());
    out.extend_from_slice(schedule);
    out.extend_from_slice(&(rans_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&rans_bytes);
    Ok(out)
}

/// Parse a neural body back into `(tokens, schedule, alphabet)`. Standalone: the
/// model is rebuilt from the in-band `counts` (`alphabet == counts.len()`, which
/// the shell needs to reconstruct `NeuralTokens`). Every length field is
/// bounds-checked (no panic on a crafted body).
pub fn decode_body(buf: &[u8]) -> Result<(Vec<i64>, Vec<u8>, u16), BodyError> {
    let mut off = 0usize;
    let version = *buf.get(off).ok_or(BodyError::Truncated)?;
    off += 1;
    if version != LMQ_BODY_VERSION {
        return Err(BodyError::BadVersion(version));
    }
    let n_symbols = read_u32(buf, &mut off)? as usize;
    let alphabet = read_u16(buf, &mut off)? as usize;
    let mut counts = vec![0i32; alphabet];
    for c in counts.iter_mut() {
        *c = read_u32(buf, &mut off)? as i32;
    }
    let sched_len = read_u32(buf, &mut off)? as usize;
    let schedule = read_bytes(buf, &mut off, sched_len)?.to_vec();
    let rans_len = read_u32(buf, &mut off)? as usize;
    let rans_data = read_bytes(buf, &mut off, rans_len)?;

    let (freq, start, m) = build_tables(&counts)?;
    let cum2sym = build_cum2sym(&freq, &start, m);
    let tokens = rans::decode(rans_data, &freq, &start, &cum2sym, m, n_symbols)
        .map_err(|_| BodyError::Rans)?;
    Ok((tokens, schedule, alphabet as u16))
}

// ── Model tables (the caller-supplied rANS model; `rans`'s own helpers are
//    test-only, so we replicate the cumulative-frequency build here). ──────────

/// `counts` (per-symbol frequency) → `(freq, start, m)`: `freq == counts`,
/// `start[i] = Σ counts[0..i]`, `m = Σ counts`.
fn build_tables(counts: &[i32]) -> Result<(Vec<i32>, Vec<i32>, u64), BodyError> {
    if counts.is_empty() {
        return Err(BodyError::EmptyModel);
    }
    let freq = counts.to_vec();
    let mut start = vec![0i32; counts.len()];
    let mut acc = 0i64;
    for (i, &c) in counts.iter().enumerate() {
        // Untrusted on decode: reject a negative count (from a u32 > i32::MAX
        // cast) — it would make `start`/`cum2sym` slots negative and panic.
        if c < 0 {
            return Err(BodyError::BadModel);
        }
        // Cap BEFORE the `as i32` so the cumulative `start` can never wrap, and
        // the eventual `cum2sym` allocation stays bounded (MAX_MODEL_TOTAL <<
        // i32::MAX).
        if acc > MAX_MODEL_TOTAL as i64 {
            return Err(BodyError::BadModel);
        }
        start[i] = acc as i32;
        acc += c as i64;
    }
    if acc == 0 {
        return Err(BodyError::EmptyModel);
    }
    if acc as u64 > MAX_MODEL_TOTAL {
        return Err(BodyError::BadModel);
    }
    Ok((freq, start, acc as u64))
}

/// Inverse table `cum2sym[slot] -> symbol` for rANS decode.
fn build_cum2sym(freq: &[i32], start: &[i32], m: u64) -> Vec<i32> {
    let mut c2s = vec![0i32; m as usize];
    for (sym, (&s, &f)) in start.iter().zip(freq.iter()).enumerate() {
        for slot in s..(s + f) {
            c2s[slot as usize] = sym as i32;
        }
    }
    c2s
}

fn read_u32(buf: &[u8], off: &mut usize) -> Result<u32, BodyError> {
    let end = off.checked_add(4).ok_or(BodyError::Truncated)?;
    let s = buf.get(*off..end).ok_or(BodyError::Truncated)?;
    *off = end;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_u16(buf: &[u8], off: &mut usize) -> Result<u16, BodyError> {
    let end = off.checked_add(2).ok_or(BodyError::Truncated)?;
    let s = buf.get(*off..end).ok_or(BodyError::Truncated)?;
    *off = end;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn read_bytes<'a>(buf: &'a [u8], off: &mut usize, len: usize) -> Result<&'a [u8], BodyError> {
    let end = off.checked_add(len).ok_or(BodyError::Truncated)?;
    let s = buf.get(*off..end).ok_or(BodyError::Truncated)?;
    *off = end;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_roundtrips_tokens_and_schedule() {
        // A 5-symbol alphabet (FSQ L=5), ten tokens, a per-timestep level schedule.
        let tokens: Vec<i64> = vec![0, 1, 4, 2, 3, 3, 1, 0, 4, 2];
        let schedule: Vec<u8> = vec![5, 5, 5, 3, 3, 2, 2, 5, 5, 3];
        let counts: Vec<i32> = vec![3, 3, 3, 3, 4]; // per-symbol freq, Σ = 16
        let body = encode_body(&tokens, &schedule, &counts).unwrap();
        assert_eq!(body[0], LMQ_BODY_VERSION);
        let (dt, ds, alpha) = decode_body(&body).unwrap();
        assert_eq!(dt, tokens, "tokens must round-trip through rANS");
        assert_eq!(ds, schedule, "schedule must round-trip verbatim");
        assert_eq!(alpha, 5, "alphabet (= counts.len()) must round-trip");
    }

    #[test]
    fn decode_body_rejects_truncation_and_bad_version() {
        let body = encode_body(&[0, 1, 2], &[3, 3, 3], &[2, 2, 2]).unwrap();
        assert_eq!(
            decode_body(&body[..body.len() - 2]),
            Err(BodyError::Truncated)
        );
        let mut bad = body.clone();
        bad[0] = 0xFF;
        assert_eq!(decode_body(&bad), Err(BodyError::BadVersion(0xFF)));
    }

    #[test]
    fn empty_model_is_rejected() {
        assert_eq!(encode_body(&[0], &[], &[]), Err(BodyError::EmptyModel));
    }

    #[test]
    fn crafted_body_model_is_fail_closed_not_oom_or_panic() {
        // Hand-craft a body with a 1-symbol alphabet whose count is a huge u32
        // (0xFFFF_FFFF → -1 as i32, AND drives the total absurd). decode_body must
        // reject it via BadModel — never allocate ~16 GB for cum2sym, never panic.
        let mut body = Vec::new();
        body.push(LMQ_BODY_VERSION);
        body.extend_from_slice(&1u32.to_le_bytes()); // n_symbols
        body.extend_from_slice(&1u16.to_le_bytes()); // alphabet = 1
        body.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // count = -1 as i32
        body.extend_from_slice(&0u32.to_le_bytes()); // sched_len
        body.extend_from_slice(&0u32.to_le_bytes()); // rans_len
        assert_eq!(decode_body(&body), Err(BodyError::BadModel));

        // A count just over the cap (positive) is also rejected before allocating.
        let mut big = Vec::new();
        big.push(LMQ_BODY_VERSION);
        big.extend_from_slice(&1u32.to_le_bytes());
        big.extend_from_slice(&1u16.to_le_bytes());
        big.extend_from_slice(&((MAX_MODEL_TOTAL as u32) + 1).to_le_bytes());
        big.extend_from_slice(&0u32.to_le_bytes());
        big.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode_body(&big), Err(BodyError::BadModel));
    }
}
