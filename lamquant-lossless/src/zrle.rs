//! Track 2 P3 (ADR 0051): zero-run-length entropy coding for the heavily
//! quantized, zero-heavy subband residuals produced by the target-BPS lossy
//! mode.
//!
//! Golomb-Rice has a hard ~1 bit/symbol floor: every zero costs at least the
//! unary terminator, so a 90%-zero stream still spends ~N bits even though its
//! entropy is well under 1 bit/symbol. That floor is exactly what caps the
//! target-BPS encoder at ~1–1.5 BPS (measured). This coder collapses runs of
//! zeros to a count, so a near-all-zero subband costs only the run codes.
//!
//! Encoding: split the value stream into (run-of-zeros-before, nonzero-value)
//! pairs; Golomb-code the run lengths and the nonzero values as two separate
//! dense streams. Trailing zeros are implicit (the decoder pads to `n`).
//! Format: `[n:u32 LE][golomb(runs)][golomb(vals)]`. Integer-only ⇒
//! firmware-decodable; reuses the existing Golomb coder for the two substreams.

use alloc::vec::Vec;

use crate::error::{LmlError, LmlResult};
use crate::golomb;

/// Encode a value stream with zero-run-length + Golomb substreams.
pub fn encode_dense(values: &[i64]) -> LmlResult<Vec<u8>> {
    let mut runs: Vec<i64> = Vec::new();
    let mut vals: Vec<i64> = Vec::new();
    let mut zc: i64 = 0;
    for &v in values {
        if v == 0 {
            zc += 1;
        } else {
            runs.push(zc);
            vals.push(v);
            zc = 0;
        }
    }
    // Trailing `zc` zeros are implicit — recovered by padding to `n` on decode.
    let mut out = Vec::with_capacity(8 + values.len());
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    out.extend_from_slice(&golomb::encode_dense(&runs)?);
    out.extend_from_slice(&golomb::encode_dense(&vals)?);
    Ok(out)
}

/// Decode a zero-run-length stream written by [`encode_dense`], starting at
/// `offset`. Returns `(values, bytes_consumed_from_offset)`.
pub fn decode_dense(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    if offset + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: offset + 4,
            actual: data.len(),
            context: "zrle header",
        });
    }
    let n = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as usize;
    let mut pos = offset + 4;
    let (runs, c1) = golomb::decode_dense(data, pos)?;
    pos += c1;
    let (vals, c2) = golomb::decode_dense(data, pos)?;
    pos += c2;
    if runs.len() != vals.len() {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "zrle runs/vals length mismatch: {} != {}",
            runs.len(),
            vals.len()
        )));
    }
    let mut out: Vec<i64> = Vec::with_capacity(n);
    for (&r, &v) in runs.iter().zip(vals.iter()) {
        if r < 0 || out.len() + r as usize + 1 > n {
            return Err(LmlError::InvalidHeader(alloc::format!(
                "zrle run overflow: run={} pos={} n={}",
                r,
                out.len(),
                n
            )));
        }
        for _ in 0..r {
            out.push(0);
        }
        out.push(v);
    }
    if out.len() > n {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "zrle decoded {} > n {}",
            out.len(),
            n
        )));
    }
    out.resize(n, 0); // implicit trailing zeros
    Ok((out, pos - offset))
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
        roundtrip(&[0, 0, 0, 0, 0]);
        roundtrip(&[5, -3, 7]);
        roundtrip(&[0, 0, 9, 0, 0, 0, -4, 0]);
        roundtrip(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
        let mut big = vec![0i64; 1000];
        big[10] = 42;
        big[900] = -17;
        roundtrip(&big);
    }

    #[test]
    fn beats_golomb_on_sparse() {
        // ~99% zeros: zrle must be much smaller than plain Golomb.
        let mut v = vec![0i64; 2000];
        v[100] = 5;
        v[1500] = -3;
        let z = encode_dense(&v).unwrap().len();
        let g = golomb::encode_dense(&v).unwrap().len();
        assert!(z < g, "zrle {} should beat golomb {} on sparse", z, g);
    }
}
