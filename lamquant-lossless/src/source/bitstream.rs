//! Bit-level decode helpers for binary signal formats.
//!
//! Today: 24-bit little-endian sign-extended integers (BDF/BioSemi
//! sample format, future BrainVision INT_24, custom raw with int24 ADC
//! output). Phase 4 readers reuse this primitive.

/// Read a 24-bit little-endian signed integer at `offset` in `data` and
/// sign-extend to `i32` via two's complement.
///
/// Panics if `offset + 3 > data.len()` — callers are expected to bound
/// the offset against a per-channel record stride (see Audit-2026-05-11
/// fix C6 in `edf.rs`). The bounds check stays at the caller because
/// the caller knows the *meaning* of the failure (channel index,
/// record number) and can produce a useful error message; this helper
/// is unconditionally a hot-path single-pass three-byte read.
#[inline(always)]
pub fn read_i24_le(data: &[u8], offset: usize) -> i32 {
    let b0 = data[offset] as i32;
    let b1 = data[offset + 1] as i32;
    let b2 = data[offset + 2] as i32;
    let val = b0 | (b1 << 8) | (b2 << 16);
    // Two's-complement sign extension: high bit of b2 set → negative.
    if val >= 0x800000 {
        val - 0x1000000
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_i24_le_positive() {
        assert_eq!(read_i24_le(&[0x01, 0x00, 0x00], 0), 1);
        assert_eq!(read_i24_le(&[0xFF, 0x7F, 0x00], 0), 0x7FFF);
        assert_eq!(read_i24_le(&[0x00, 0x00, 0x40], 0), 0x400000);
    }

    #[test]
    fn read_i24_le_max_positive() {
        // 0x7FFFFF is the largest positive i24.
        assert_eq!(read_i24_le(&[0xFF, 0xFF, 0x7F], 0), 0x7FFFFF);
    }

    #[test]
    fn read_i24_le_negative() {
        // 0xFFFFFF == -1 in i24.
        assert_eq!(read_i24_le(&[0xFF, 0xFF, 0xFF], 0), -1);
        // 0x800000 == -8388608 (min i24).
        assert_eq!(read_i24_le(&[0x00, 0x00, 0x80], 0), -8388608);
    }

    #[test]
    fn read_i24_le_with_offset() {
        // Read at offset 2 of a 5-byte buffer.
        let data = [0xAA, 0xBB, 0x01, 0x02, 0x03];
        assert_eq!(read_i24_le(&data, 2), 0x030201);
    }

    #[test]
    #[should_panic]
    fn read_i24_le_oob_panics() {
        // Documented panic — callers must bound the offset.
        let _ = read_i24_le(&[0x01, 0x02], 0);
    }
}
