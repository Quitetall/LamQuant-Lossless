//! Fixed-width bit-packing for LPC residual blocks.
//!
//! ADR 0023 Track B1: When a subband's residuals have a small effective
//! range (post-LPC), a fixed N-bit-per-symbol stream is shorter than
//! Golomb-Rice + adaptive-k. Examples:
//!
//!   * Bonn EEG (1ch, 12-bit ADC) post-LPC residuals often live in ±8.
//!     Zigzag → max 16, packed at 5 bits each. Golomb-Rice with the
//!     same data spends 6-8 bits per symbol due to per-block-k overhead.
//!
//!   * CHB-MIT clinical seizure channels with mostly-flat sections —
//!     same story: a narrow zigzag range fits in 4-6 bits.
//!
//! Wire layout (per subband, when selected):
//!
//!   ```text
//!   [1B: n_bits (1..=63)]   // bit width per symbol
//!   [⌈n_samples × n_bits / 8⌉ B: big-endian bit-packed zigzag values]
//!   ```
//!
//! Decoder is told `n_samples` out-of-band (the existing per-subband
//! length already crosses the wire via LPC meta). Encoder picks
//! `n_bits = ceil(log2(max_zigzag_value + 1))`, clamped to ≥ 1.
//!
//! Determinism is load-bearing: both `ComputeBackend::Firmware` and
//! `Desktop` must produce identical bytes for the same input so the
//! `byte_equal_backends` gate stays green. All operations are scalar
//! integer math + a left-to-right bit writer; no floating point, no
//! reordering of operations across the residual block.

use alloc::vec::Vec;

/// Maximum bits per packed residual. zigzag(i64::MIN) yields a u64
/// up to 2^64 - 1, which needs 64 bits. We cap at 63 because the
/// LPC residual ceiling (`golomb::MAX_Q = 2^40`) is way smaller and
/// 63 fits in a `u8` width-field with one bit reserved for a future
/// extension. Any value requiring ≥ 64 bits is rejected.
pub const MAX_BIT_PACK_BITS: u8 = 63;

/// Minimum bits per packed residual. Forced to 1 even when every
/// sample is zero — the alternative is special-casing a zero-length
/// payload which adds branchy decoder code for negligible savings.
pub const MIN_BIT_PACK_BITS: u8 = 1;

#[derive(Debug, PartialEq, Eq)]
pub enum BitPackError {
    /// All residuals were zigzag-encoded but the max needed > 63 bits.
    /// Encoder gives up and the caller falls back to Golomb-Rice.
    Overrange,
    /// Decoder hit unexpected EOF on the payload (shorter than the
    /// declared sample count × bit width).
    Truncated,
    /// Decoder saw a `n_bits` byte outside the inclusive range
    /// `[MIN_BIT_PACK_BITS, MAX_BIT_PACK_BITS]`. Manifest corruption.
    InvalidBitWidth(u8),
}

impl core::fmt::Display for BitPackError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            BitPackError::Overrange => write!(f, "bit-pack: residual range exceeds 63 bits"),
            BitPackError::Truncated => write!(f, "bit-pack: payload truncated"),
            BitPackError::InvalidBitWidth(n) => {
                write!(f, "bit-pack: invalid n_bits {} (must be 1..=63)", n)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BitPackError {}

/// Zigzag-encode a signed value into a non-negative `u64`. Same
/// transform `golomb` uses — kept duplicated to avoid the bit-pack
/// module taking a `pub use` from `golomb` and locking the two
/// together at the API level.
#[inline(always)]
fn zigzag_encode(v: i64) -> u64 {
    ((v as u64) << 1) ^ ((v >> 63) as u64)
}

#[inline(always)]
fn zigzag_decode(v: u64) -> i64 {
    let v = v as i64;
    (v >> 1) ^ -(v & 1)
}

/// Compute the minimum `n_bits` to represent every value in `coeffs`
/// after zigzag. Returns `Err(Overrange)` if any value needs ≥ 64
/// bits. `MIN_BIT_PACK_BITS` floor applied. Pure function — used by
/// the encoder's selection logic (compare against Golomb size before
/// committing to either path).
pub fn min_bits_for(coeffs: &[i64]) -> Result<u8, BitPackError> {
    let mut max_z: u64 = 0;
    for &v in coeffs {
        let z = zigzag_encode(v);
        if z > max_z {
            max_z = z;
        }
    }
    // Need ceil(log2(max_z + 1)) bits to represent values 0..=max_z.
    // u64::ilog2(max_z) + 1 = ceil(log2(max_z + 1)) for max_z >= 1.
    // max_z = 0 → 1 bit (MIN_BIT_PACK_BITS floor).
    let n_bits = if max_z == 0 {
        MIN_BIT_PACK_BITS
    } else {
        let b = (64 - max_z.leading_zeros()) as u8;
        if b > MAX_BIT_PACK_BITS {
            return Err(BitPackError::Overrange);
        }
        b
    };
    Ok(n_bits.max(MIN_BIT_PACK_BITS))
}

/// Exact byte count an `encode_dense` of `coeffs` produces. Lets the
/// caller size-compare against Golomb-Rice WITHOUT actually emitting
/// either stream. Useful for fast selection in the per-subband
/// dispatch.
pub fn encoded_byte_len(coeffs: &[i64]) -> Result<usize, BitPackError> {
    let n_bits = min_bits_for(coeffs)? as usize;
    let payload_bits = coeffs.len() * n_bits;
    let payload_bytes = (payload_bits + 7) / 8;
    Ok(1 + payload_bytes) // 1-byte header + payload
}

/// Pack `coeffs` as `n_bits`-each zigzag values, big-endian bit-stream.
/// Output: `[n_bits as u8] [packed bytes ...]`. Last byte is zero-
/// padded on the right.
pub fn encode_dense(coeffs: &[i64]) -> Result<Vec<u8>, BitPackError> {
    let n_bits = min_bits_for(coeffs)?;
    let payload_bits = coeffs.len() * n_bits as usize;
    let payload_bytes = (payload_bits + 7) / 8;
    let mut out = Vec::with_capacity(1 + payload_bytes);
    out.push(n_bits);
    out.resize(1 + payload_bytes, 0);

    // Big-endian bit stream: high bits first. `bit_pos` is the global
    // bit offset from the start of the payload section.
    let mut bit_pos: usize = 0;
    let payload_start = 1;
    for &v in coeffs {
        let z = zigzag_encode(v);
        // Walk the value high bit → low bit so reading high-bit-first
        // recovers it identically. n_bits ≤ 63 so the shift never
        // touches the sign bit.
        for i in (0..n_bits as usize).rev() {
            let bit = ((z >> i) & 1) as u8;
            if bit != 0 {
                let byte_idx = payload_start + (bit_pos >> 3);
                let in_byte_bit = 7 - (bit_pos & 7);
                out[byte_idx] |= 1 << in_byte_bit;
            }
            bit_pos += 1;
        }
    }
    Ok(out)
}

/// Unpack the inverse of `encode_dense`. `n_samples` is told out-of-
/// band; the function reads `1 + ceil(n_samples × n_bits / 8)` bytes
/// from `bytes` starting at offset 0 and returns the decoded `Vec<i64>`
/// + how many bytes were consumed.
pub fn decode_dense(
    bytes: &[u8],
    n_samples: usize,
) -> Result<(Vec<i64>, usize), BitPackError> {
    if bytes.is_empty() {
        return Err(BitPackError::Truncated);
    }
    let n_bits = bytes[0];
    if !(MIN_BIT_PACK_BITS..=MAX_BIT_PACK_BITS).contains(&n_bits) {
        return Err(BitPackError::InvalidBitWidth(n_bits));
    }
    let payload_bits = n_samples * n_bits as usize;
    let payload_bytes = (payload_bits + 7) / 8;
    if bytes.len() < 1 + payload_bytes {
        return Err(BitPackError::Truncated);
    }

    let mut out = Vec::with_capacity(n_samples);
    let payload_start = 1;
    let mut bit_pos: usize = 0;
    for _ in 0..n_samples {
        let mut z: u64 = 0;
        for _ in 0..n_bits as usize {
            let byte_idx = payload_start + (bit_pos >> 3);
            let in_byte_bit = 7 - (bit_pos & 7);
            let bit = ((bytes[byte_idx] >> in_byte_bit) & 1) as u64;
            z = (z << 1) | bit;
            bit_pos += 1;
        }
        out.push(zigzag_decode(z));
    }
    Ok((out, 1 + payload_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn min_bits_basic_ranges() {
        // Zero-only block clamps to MIN_BIT_PACK_BITS (1).
        assert_eq!(min_bits_for(&[0, 0, 0]).unwrap(), 1);
        // Range [-1, 1] → zigzag {0, 1, 2} → max=2 → 2 bits.
        assert_eq!(min_bits_for(&[-1, 0, 1]).unwrap(), 2);
        // Range [-4, 3] → zigzag(-4)=7, zigzag(3)=6 → max=7 → 3 bits.
        assert_eq!(min_bits_for(&[-4, 3]).unwrap(), 3);
        // Range [-5, 4] → zigzag(-5)=9, zigzag(4)=8 → max=9 → 4 bits.
        assert_eq!(min_bits_for(&[-5, 4]).unwrap(), 4);
        // i32 max → fits in ≤ 33 bits (zigzag adds 1).
        let v: i64 = i32::MAX as i64;
        let z = (v as u64) << 1;
        let need = 64 - z.leading_zeros() as u8;
        assert_eq!(min_bits_for(&[v]).unwrap(), need);
    }

    #[test]
    fn overrange_when_zigzag_overflows_63_bits() {
        // i64::MIN zigzags to u64::MAX → 64 bits needed → reject.
        let err = min_bits_for(&[i64::MIN]).unwrap_err();
        assert_eq!(err, BitPackError::Overrange);
    }

    #[test]
    fn encoded_byte_len_matches_actual() {
        let cases: &[&[i64]] = &[
            &[0],
            &[0, 0, 0, 0],
            &[-1, 0, 1, -2, 2],
            &[100, -50, 200, -75, 80],
            &[1024, -1024],
        ];
        for c in cases {
            let computed = encoded_byte_len(c).unwrap();
            let actual = encode_dense(c).unwrap().len();
            assert_eq!(computed, actual, "mismatch on {:?}", c);
        }
    }

    #[test]
    fn round_trip_typical_residuals() {
        let cases: Vec<Vec<i64>> = vec![
            vec![0; 64],
            vec![1, -1, 2, -2, 3, -3, 4, -4],
            (-128..128i64).collect(),
            (0..100i64).flat_map(|i| [i, -i]).collect(),
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
            vec![i32::MAX as i64; 8], // wide range — 32-bit zigzag
        ];
        for c in cases {
            let enc = encode_dense(&c).unwrap();
            let (dec, consumed) = decode_dense(&enc, c.len()).unwrap();
            assert_eq!(dec, c, "round-trip mismatch on len={}", c.len());
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn decode_rejects_truncated_payload() {
        let c = vec![5i64; 100];
        let enc = encode_dense(&c).unwrap();
        // Strip the last byte → payload short by one byte.
        let truncated = &enc[..enc.len() - 1];
        let err = decode_dense(truncated, c.len()).unwrap_err();
        assert_eq!(err, BitPackError::Truncated);
    }

    #[test]
    fn decode_rejects_invalid_bit_width() {
        // First byte = 0 → invalid n_bits.
        let bytes = [0u8, 0, 0];
        assert_eq!(
            decode_dense(&bytes, 1).unwrap_err(),
            BitPackError::InvalidBitWidth(0)
        );
        // First byte = 64 → > MAX_BIT_PACK_BITS.
        let bytes = [64u8, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            decode_dense(&bytes, 1).unwrap_err(),
            BitPackError::InvalidBitWidth(64)
        );
    }

    #[test]
    fn empty_input_zero_byte_payload() {
        let enc = encode_dense(&[]).unwrap();
        // 1-byte header (n_bits=1 floor) + 0 payload bytes.
        assert_eq!(enc.len(), 1);
        assert_eq!(enc[0], 1);
        let (dec, consumed) = decode_dense(&enc, 0).unwrap();
        assert!(dec.is_empty());
        assert_eq!(consumed, 1);
    }

    #[test]
    fn determinism_run_twice_same_bytes() {
        let c = vec![13i64, -7, 42, -100, 33, -22, 11, -5, 0, 1];
        let a = encode_dense(&c).unwrap();
        let b = encode_dense(&c).unwrap();
        assert_eq!(a, b);
    }
}
