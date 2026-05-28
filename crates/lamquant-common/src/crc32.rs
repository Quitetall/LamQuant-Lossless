//! CRC-32 (ISO 3309 / zlib) — slice-by-4 for throughput.
//!
//! Processes 4 bytes per iteration using 4 lookup tables (4 KB total).
//! ~3-4× faster than byte-at-a-time on modern CPUs.
//! Falls back to byte-at-a-time for the tail (0-3 bytes).

const POLY: u32 = 0xEDB8_8320;

const fn make_table() -> [[u32; 256]; 4] {
    let mut tables = [[0u32; 256]; 4];
    // Table 0: standard byte-at-a-time
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        tables[0][i as usize] = crc;
        i += 1;
    }
    // Tables 1-3: extended for slice-by-4
    let mut t = 1;
    while t < 4 {
        i = 0;
        while i < 256 {
            let prev = tables[t - 1][i as usize];
            tables[t][i as usize] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
            i += 1;
        }
        t += 1;
    }
    tables
}

static TABLE: [[u32; 256]; 4] = make_table();

/// Compute CRC-32 over a byte slice. Identical to `zlib.crc32`.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0xFFFF_FFFF, data) ^ 0xFFFF_FFFF
}

/// Streaming CRC-32: feed multiple slices without concatenating.
///
/// Usage:
///     let mut state = CRC32_INIT;
///     state = crc32_update(state, &slice_a);
///     state = crc32_update(state, &slice_b);
///     let crc = state ^ CRC32_INIT;
pub const CRC32_INIT: u32 = 0xFFFF_FFFF;

#[inline]
pub fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    let mut i = 0;
    let len = data.len();

    // Slice-by-4: process 4 bytes per iteration
    while i + 4 <= len {
        crc ^= (data[i] as u32)
            | ((data[i + 1] as u32) << 8)
            | ((data[i + 2] as u32) << 16)
            | ((data[i + 3] as u32) << 24);
        crc = TABLE[3][(crc & 0xFF) as usize]
            ^ TABLE[2][((crc >> 8) & 0xFF) as usize]
            ^ TABLE[1][((crc >> 16) & 0xFF) as usize]
            ^ TABLE[0][((crc >> 24) & 0xFF) as usize];
        i += 4;
    }

    // Tail: byte-at-a-time
    while i < len {
        crc = (crc >> 8) ^ TABLE[0][((crc ^ data[i] as u32) & 0xFF) as usize];
        i += 1;
    }

    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"hello"), 0x3610_a686);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = crc32(data);
        let mut state = CRC32_INIT;
        state = crc32_update(state, &data[..10]);
        state = crc32_update(state, &data[10..]);
        let streaming = state ^ CRC32_INIT;
        assert_eq!(oneshot, streaming);
    }
}
