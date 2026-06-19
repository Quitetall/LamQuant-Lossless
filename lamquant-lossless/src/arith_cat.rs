//! Track 2 P3.5 (ADR 0051): empirical-categorical range coding.
//!
//! The existing `arithmetic.rs` fits a STATIC Laplace — which can't model the
//! huge zero-spike that deadzone quantization creates, so it lost to Golomb in
//! testing. This coder instead histograms the ACTUAL quantized symbols, ships
//! the (fixed-point) empirical PMF, and range-codes i.i.d. against it. It
//! captures the zero-spike + the real tail exactly, so it has no Golomb-style
//! 1-bit/symbol floor and approaches the block's order-0 entropy. This is the
//! same family (categorical arithmetic coding) Fraunhofer HHI's CABAC is built
//! on; context modelling is a later layer on top.
//!
//! Wire layout (per subband, when this codec wins keep-best selection):
//! ```text
//!   [4B min:i32 LE]      // alphabet base (smallest symbol)
//!   [4B k:u32 LE]        // alphabet size (max-min+1)
//!   [4B n:u32 LE]        // symbol count
//!   [k×4B freqs:u32 LE]  // fixed-point PMF, each >=1, sum == 2^PRECISION
//!   [4B body_len:u32 LE]
//!   [body bytes]         // constriction range-coder stream
//! ```
//! Host-only (constriction dep); firmware fails closed on the payload tag.

use alloc::vec::Vec;

use constriction::stream::{
    model::DefaultContiguousCategoricalEntropyModel,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
    Decode, Encode,
};

use crate::error::{LmlError, LmlResult};

/// constriction `Default*` models use 24-bit precision; the fixed-point PMF
/// must sum to exactly `2^24`.
const PRECISION: u32 = 24;
const TOTAL: u64 = 1u64 << PRECISION;

/// Leaky-quantize raw counts to fixed-point frequencies: every entry >= 1
/// ("leaky" — even absent interior symbols stay representable) and the sum is
/// exactly `TOTAL`. Deterministic integer math ⇒ identical model on both sides.
fn leaky_quantize(counts: &[u64]) -> Vec<u32> {
    let k = counts.len();
    let n: u64 = counts.iter().sum::<u64>().max(1);
    let mut freqs: Vec<u32> = counts
        .iter()
        .map(|&c| (((c as u128 * TOTAL as u128) / n as u128) as u64).max(1) as u32)
        .collect();
    // Reconcile the sum to exactly TOTAL by adjusting the largest-count symbol
    // (it has the most slack to absorb the correction without underflow).
    let sum: u64 = freqs.iter().map(|&f| f as u64).sum();
    let big = (0..k).max_by_key(|&i| counts[i]).unwrap_or(0);
    if sum < TOTAL {
        freqs[big] += (TOTAL - sum) as u32;
    } else if sum > TOTAL {
        let mut excess = sum - TOTAL;
        // Remove from the largest, keeping it >= 1; spill to others if needed.
        for &i in &order_by_freq_desc(&freqs) {
            if excess == 0 {
                break;
            }
            let removable = (freqs[i] as u64).saturating_sub(1);
            let take = removable.min(excess);
            freqs[i] -= take as u32;
            excess -= take;
        }
        let _ = big;
    }
    debug_assert_eq!(freqs.iter().map(|&f| f as u64).sum::<u64>(), TOTAL);
    freqs
}

fn order_by_freq_desc(freqs: &[u32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..freqs.len()).collect();
    idx.sort_by(|&a, &b| freqs[b].cmp(&freqs[a]));
    idx
}

/// Encode a residual block with an empirical-categorical range coder.
/// Returns the self-delimiting payload (header + body).
pub fn encode_dense(values: &[i64]) -> LmlResult<Vec<u8>> {
    let n = values.len();
    if n == 0 {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&0i32.to_le_bytes()); // min
        out.extend_from_slice(&0u32.to_le_bytes()); // k
        out.extend_from_slice(&0u32.to_le_bytes()); // n
        out.extend_from_slice(&0u32.to_le_bytes()); // body_len
        return Ok(out);
    }
    let mn = *values.iter().min().unwrap();
    let mx = *values.iter().max().unwrap();
    // Alphabet size; bail to caller (keep-best) if absurdly wide — the freq
    // table would dwarf any coding gain.
    let k_u128 = (mx as i128 - mn as i128 + 1) as u128;
    if k_u128 > 1_000_000 {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "arith_cat alphabet too wide ({k_u128})"
        )));
    }
    let k = k_u128 as usize;

    let mut counts = alloc::vec![0u64; k];
    for &v in values {
        counts[(v - mn) as usize] += 1;
    }
    let freqs = leaky_quantize(&counts);

    let model = DefaultContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities(
        freqs.iter().copied(),
        false,
    )
    .map_err(|_| LmlError::InvalidHeader("arith_cat model build failed".into()))?;

    let mut encoder = DefaultRangeEncoder::new();
    let symbols = values.iter().map(|&v| (v - mn) as usize);
    encoder
        .encode_iid_symbols(symbols, &model)
        .map_err(|_| LmlError::InvalidHeader("arith_cat encode failed".into()))?;
    let words: Vec<u32> = encoder
        .into_compressed()
        .map_err(|_| LmlError::InvalidHeader("arith_cat compressed failed".into()))?;
    let mut body = Vec::with_capacity(words.len() * 4);
    for w in &words {
        body.extend_from_slice(&w.to_le_bytes());
    }

    let mut out = Vec::with_capacity(16 + 4 * k + body.len());
    out.extend_from_slice(&(mn as i32).to_le_bytes());
    out.extend_from_slice(&(k as u32).to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for &f in &freqs {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

fn rd_u32(data: &[u8], pos: usize) -> LmlResult<u32> {
    if pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + 4,
            actual: data.len(),
            context: "arith_cat header",
        });
    }
    Ok(u32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

/// Decode a block written by [`encode_dense`] starting at `offset`. Returns
/// `(values, bytes_consumed_from_offset)`.
pub fn decode_dense(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    let mn = rd_u32(data, offset)? as i32 as i64;
    let k = rd_u32(data, offset + 4)? as usize;
    let n = rd_u32(data, offset + 8)? as usize;
    if k == 0 || n == 0 {
        // empty block: header is 16 bytes (no freqs, no body)
        return Ok((Vec::new(), 16));
    }
    let mut pos = offset + 12;
    let mut freqs = Vec::with_capacity(k);
    for _ in 0..k {
        freqs.push(rd_u32(data, pos)?);
        pos += 4;
    }
    let body_len = rd_u32(data, pos)? as usize;
    pos += 4;
    if pos + body_len > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + body_len,
            actual: data.len(),
            context: "arith_cat body",
        });
    }
    if body_len % 4 != 0 {
        return Err(LmlError::InvalidHeader("arith_cat body not word-aligned".into()));
    }
    let mut words = Vec::with_capacity(body_len / 4);
    for w in 0..body_len / 4 {
        let b = pos + w * 4;
        words.push(u32::from_le_bytes([data[b], data[b + 1], data[b + 2], data[b + 3]]));
    }

    let model = DefaultContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities(
        freqs.iter().copied(),
        false,
    )
    .map_err(|_| LmlError::InvalidHeader("arith_cat model rebuild failed".into()))?;
    let mut decoder = DefaultRangeDecoder::from_compressed(words)
        .map_err(|_| LmlError::InvalidHeader("arith_cat decoder init failed".into()))?;
    let mut out = Vec::with_capacity(n);
    for sym in decoder.decode_iid_symbols(n, &model) {
        let s = sym.map_err(|_| LmlError::InvalidHeader("arith_cat decode failed".into()))?;
        out.push(mn + s as i64);
    }
    Ok((out, (pos + body_len) - offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn roundtrip(v: &[i64]) {
        let enc = encode_dense(v).unwrap();
        let (dec, consumed) = decode_dense(&enc, 0).unwrap();
        assert_eq!(dec, v, "roundtrip mismatch");
        assert_eq!(consumed, enc.len(), "consumed != encoded length");
    }

    #[test]
    fn roundtrip_shapes() {
        roundtrip(&[]);
        roundtrip(&[0]);
        roundtrip(&[7, 7, 7, 7]);
        roundtrip(&[3, -2, 0, 5, -9, 0, 0, 1]);
        let mut sparse = vec![0i64; 2000];
        sparse[5] = 11;
        sparse[1999] = -4;
        roundtrip(&sparse);
        let mixed: Vec<i64> = (0..1000).map(|i| ((i * 31) % 17) as i64 - 8).collect();
        roundtrip(&mixed);
    }

    #[test]
    fn beats_golomb_on_skewed() {
        // Skewed (non-geometric) distribution with a zero spike: empirical
        // categorical should beat Golomb, which assumes a geometric source.
        let mut v = vec![0i64; 4000];
        for i in 0..200 {
            v[i * 20] = if i % 3 == 0 { 1 } else { -1 };
        }
        let a = encode_dense(&v).unwrap().len();
        let g = crate::golomb::encode_dense(&v).unwrap().len();
        assert!(a < g, "arith_cat {} should beat golomb {} on skewed", a, g);
    }
}
