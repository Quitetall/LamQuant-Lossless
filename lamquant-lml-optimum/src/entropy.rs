//! Keep-best residual entropy coder (ADR 0054): single-k Golomb-Rice vs
//! **block-adaptive** Golomb (a fresh optimal Rice k per 256-sample block). The
//! RLS-predictor residuals are bursty/non-stationary (adaptation lag), so a
//! per-block k wins big there (−16% on the hard `ma` case); on stationary
//! residuals single-k wins. Keep-smallest with a mode byte ⇒ never worse.
//!
//! Reuses `golomb::encode_dense` (which already picks the optimal k for whatever
//! it's handed) per block — so "block-adaptive Golomb" is just per-chunk Golomb.
//!
//! Wire: `[mode u8]` then `0` = `golomb::encode_dense(res)` | `1` =
//! `[n u32][concatenated golomb-encoded 256-sample chunks]`.

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::golomb;

#[cfg(feature = "encode")]
const BLOCK: usize = 256;
const MODE_SINGLE: u8 = 0;
const MODE_BLOCK: u8 = 1;

/// Encode a residual, keeping the smaller of single-k and block-adaptive Golomb.
#[cfg(feature = "encode")]
pub fn encode(res: &[i64]) -> LmlResult<Vec<u8>> {
    let single = golomb::encode_dense(res)?;
    let mut block = Vec::with_capacity(single.len());
    block.extend_from_slice(&(res.len() as u32).to_le_bytes());
    for chunk in res.chunks(BLOCK) {
        block.extend_from_slice(&golomb::encode_dense(chunk)?);
    }
    if block.len() < single.len() {
        let mut out = Vec::with_capacity(1 + block.len());
        out.push(MODE_BLOCK);
        out.extend_from_slice(&block);
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(1 + single.len());
        out.push(MODE_SINGLE);
        out.extend_from_slice(&single);
        Ok(out)
    }
}

/// Decode a slice produced by [`encode`] (the caller passes the exact byte range).
pub fn decode(data: &[u8]) -> LmlResult<Vec<i64>> {
    if data.is_empty() {
        return Err(LmlError::Truncated { expected: 1, actual: 0, context: "entropy mode" });
    }
    match data[0] {
        MODE_SINGLE => Ok(golomb::decode_dense(data, 1)?.0),
        MODE_BLOCK => {
            if data.len() < 5 {
                return Err(LmlError::Truncated { expected: 5, actual: data.len(), context: "entropy block n" });
            }
            let n = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
            let mut out = Vec::with_capacity(n);
            let mut pos = 5usize;
            while out.len() < n {
                let (v, consumed) = golomb::decode_dense(data, pos)?;
                if v.is_empty() && consumed == 0 {
                    return Err(LmlError::InvalidHeader("entropy block stalled".into()));
                }
                out.extend_from_slice(&v);
                pos += consumed;
            }
            out.truncate(n);
            Ok(out)
        }
        other => Err(LmlError::InvalidHeader(alloc::format!("entropy unknown mode 0x{other:02X}"))),
    }
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn roundtrip_various() {
        let mut st = 0x1234_5678u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for n in [0usize, 1, 255, 256, 257, 1000, 5000] {
            // non-stationary: scale jumps between blocks
            let v: Vec<i64> = (0..n)
                .map(|i| {
                    let scale = if (i / 300) % 2 == 0 { 2 } else { 200 };
                    ((next() % (scale * 2 + 1)) as i64) - scale as i64
                })
                .collect();
            let enc = encode(&v).unwrap();
            assert_eq!(decode(&enc).unwrap(), v, "n={n}");
        }
    }
}
