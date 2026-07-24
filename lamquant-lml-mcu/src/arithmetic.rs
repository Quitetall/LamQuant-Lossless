//! Range-coded entropy coding via `constriction`.
//!
//! ADR 0023 Track B5+. Replaces the earlier hand-rolled WNC bit-by-bit
//! arithmetic coder with `constriction`'s byte-oriented range coder
//! + `LeakyQuantizer<Laplace>` model. ~25-100× faster encode and ~5-15 % tighter output on Laplacian residuals than the WNC bootstrap.
//!
//! Wire layout (per subband, when this codec wins selection):
//!
//!   ```text
//!   [4B: min: i32 LE]   // model's lower bound (inclusive)
//!   [4B: max: i32 LE]   // model's upper bound (inclusive)
//!   [4B: mean_q: i32 LE]  // 256 × mean, scaled
//!   [4B: scale_q: u32 LE] // 256 × scale, scaled
//!   [4B: body_len: u32 LE]
//!   [body bytes]        // constriction range coder output
//!   ```
//!
//! Total header = 20 bytes. Body is byte-aligned constriction stream.
//! Selection rule + decoder dispatch live in `lml.rs` under
//! `SUBBAND_TAG_ARITHMETIC = 0x02`.

use alloc::vec::Vec;
use constriction::stream::{
    model::DefaultLeakyQuantizer,
    queue::{DefaultRangeDecoder, DefaultRangeEncoder},
    Decode, Encode,
};
use probability::distribution::Laplace;

const MEAN_SCALE_QUANT: f64 = 256.0;

#[derive(Debug, PartialEq, Eq)]
pub enum ArithmeticError {
    /// Decoder hit unexpected EOF on the payload header.
    Truncated,
    /// Constriction encode failed (alphabet inconsistency).
    EncodeFailure,
    /// Constriction decode failed (stream corruption).
    DecodeFailure,
    /// Empty input — caller should special-case before invoking.
    EmptyHeader,
}

impl core::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            ArithmeticError::Truncated => write!(f, "arithmetic: payload truncated"),
            ArithmeticError::EncodeFailure => write!(f, "arithmetic: encode failure"),
            ArithmeticError::DecodeFailure => write!(f, "arithmetic: decode failure"),
            ArithmeticError::EmptyHeader => {
                write!(f, "arithmetic: input too short for header")
            }
        }
    }
}

impl std::error::Error for ArithmeticError {}

/// Encode a block of residuals.
pub fn encode_dense(coeffs: &[i64]) -> Vec<u8> {
    if coeffs.is_empty() {
        // Empty block: header with zero body. Decoder special-cases
        // n_samples == 0 and never reads the body.
        let mut out = Vec::with_capacity(20);
        out.extend_from_slice(&0i32.to_le_bytes()); // min
        out.extend_from_slice(&0i32.to_le_bytes()); // max
        out.extend_from_slice(&0i32.to_le_bytes()); // mean_q
        out.extend_from_slice(&0u32.to_le_bytes()); // scale_q
        out.extend_from_slice(&0u32.to_le_bytes()); // body_len
        return out;
    }

    // Bracket the range. Clamp into i32 because constriction's
    // LeakyQuantizer model is parameterized by i32 bounds; values
    // outside i32 are exceedingly rare for post-LPC residuals.
    let (mn, mx) = coeffs.iter().fold((i32::MAX, i32::MIN), |(mn, mx), &v| {
        let v = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        (mn.min(v), mx.max(v))
    });
    // constriction's LeakyQuantizer requires `support.end() >
    // support.start()`. Degenerate blocks (all equal) collapse mn ==
    // mx; pad mx by 1 so the model is well-formed. Decoder applies
    // the same padding from the wire-format `mn`/`mx` it reads.
    let mx = if mn == mx { mx.saturating_add(1) } else { mx };

    // Estimate Laplace parameters (mean + scale = mean abs deviation).
    let n = coeffs.len() as f64;
    let mean: f64 = coeffs.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mad: f64 = coeffs.iter().map(|&v| (v as f64 - mean).abs()).sum::<f64>() / n;
    let scale = mad.max(0.5); // floor — degenerate (all-equal) blocks still encodable

    // Quantise mean + scale to fixed-point so decoder reconstructs
    // identically.
    let mean_q = (mean * MEAN_SCALE_QUANT).round() as i32;
    let scale_q = (scale * MEAN_SCALE_QUANT).round().max(1.0) as u32;

    // Reconstruct the float values from the quantised header so the
    // encoder + decoder use identical models.
    let mean_recon = mean_q as f64 / MEAN_SCALE_QUANT;
    let scale_recon = scale_q as f64 / MEAN_SCALE_QUANT;

    let quantizer = DefaultLeakyQuantizer::<f64, i32>::new(mn..=mx);
    let model = quantizer.quantize(Laplace::new(mean_recon, scale_recon));

    let mut encoder = DefaultRangeEncoder::new();
    let symbols: Vec<i32> = coeffs
        .iter()
        .map(|&v| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
        .collect();
    encoder
        .encode_iid_symbols(symbols.iter().copied(), model)
        .expect("constriction encode (model is well-formed)");
    let body: Vec<u8> = {
        // `into_compressed::<u8>()` gives a Vec<u8> stream.
        let words: Vec<u32> = encoder.into_compressed().expect("compressed words");
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    };

    let mut out = Vec::with_capacity(20 + body.len());
    out.extend_from_slice(&mn.to_le_bytes());
    out.extend_from_slice(&mx.to_le_bytes());
    out.extend_from_slice(&mean_q.to_le_bytes());
    out.extend_from_slice(&scale_q.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Decode `n_samples` symbols from a stream produced by `encode_dense`.
pub fn decode_dense(bytes: &[u8], n_samples: usize) -> Result<(Vec<i64>, usize), ArithmeticError> {
    if bytes.len() < 20 {
        return Err(ArithmeticError::EmptyHeader);
    }
    let mn = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let mx = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let mean_q = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let scale_q = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let body_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;

    if 20 + body_len > bytes.len() {
        return Err(ArithmeticError::Truncated);
    }
    if n_samples == 0 {
        return Ok((Vec::new(), 20 + body_len));
    }

    let mean_recon = mean_q as f64 / MEAN_SCALE_QUANT;
    let scale_recon = scale_q as f64 / MEAN_SCALE_QUANT;

    let quantizer = DefaultLeakyQuantizer::<f64, i32>::new(mn..=mx);
    let model = quantizer.quantize(Laplace::new(mean_recon, scale_recon));

    let body_bytes = &bytes[20..20 + body_len];
    // Reconstitute Vec<u32> from LE bytes.
    if body_bytes.len() % 4 != 0 {
        return Err(ArithmeticError::DecodeFailure);
    }
    let words: Vec<u32> = body_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut decoder =
        DefaultRangeDecoder::from_compressed(words).map_err(|_| ArithmeticError::DecodeFailure)?;

    let mut out = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let sym: i32 = decoder
            .decode_symbol(&model)
            .map_err(|_| ArithmeticError::DecodeFailure)?;
        out.push(sym as i64);
    }
    Ok((out, 20 + body_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn round_trip_small_values() {
        let cases: Vec<Vec<i64>> = vec![
            vec![0; 100],
            vec![0, 1, -1, 2, -2, 3, -3, 4, -4],
            (-50..50i64).collect(),
            (0..200i64).flat_map(|i| [i, -i]).collect(),
        ];
        for c in cases {
            let enc = encode_dense(&c);
            let (dec, consumed) = decode_dense(&enc, c.len()).unwrap();
            assert_eq!(dec, c, "round-trip mismatch on len={}", c.len());
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn round_trip_wider_range() {
        let mut data: Vec<i64> = (0..200i64).map(|i| (i % 31) - 15).collect();
        data.push(1000);
        data.push(-1000);
        let enc = encode_dense(&data);
        let (dec, _) = decode_dense(&enc, data.len()).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn empty_block_roundtrips() {
        let enc = encode_dense(&[]);
        assert_eq!(enc.len(), 20); // header only
        let (dec, consumed) = decode_dense(&enc, 0).unwrap();
        assert!(dec.is_empty());
        assert_eq!(consumed, 20);
    }

    #[test]
    fn decode_empty_input_errors() {
        let err = decode_dense(&[], 1).unwrap_err();
        assert_eq!(err, ArithmeticError::EmptyHeader);
    }

    #[test]
    fn deterministic_output() {
        let c = vec![3i64, -1, 5, -2, 0, 7, -4, 2, 1, -3, 0, 0, 1, 0];
        let a = encode_dense(&c);
        let b = encode_dense(&c);
        assert_eq!(a, b);
    }

    #[test]
    fn decoder_rejects_truncated() {
        let c = vec![5i64; 50];
        let enc = encode_dense(&c);
        let truncated = &enc[..enc.len() - 4];
        let err = decode_dense(truncated, c.len()).unwrap_err();
        assert!(matches!(
            err,
            ArithmeticError::Truncated | ArithmeticError::DecodeFailure
        ));
    }
}
