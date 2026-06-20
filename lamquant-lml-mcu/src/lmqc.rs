//! LMQC — channel-agnostic neural codec container (`.lmq`).
//!
//! The channel-agnostic neural decoder cannot infer the channel count / montage
//! from the fixed `[32,79]` latent, so the wire MUST carry it. This container
//! stores, per recording: a MONTAGE block (electrode coords `[N,3]` f32, meters,
//! `NaN` = unknown, plus channel-name strings) and a versioned opaque PAYLOAD
//! (the encoded latent), tagged by `payload_kind` so the entropy backend can be
//! upgraded without changing the montage capability or the reader contract.
//!
//! This is the canonical wire-format implementation (Rust); the deprecated
//! Python `lamquant_codec.fileformat` `.lmq` (write-path deleted) is superseded.
//! `no_std` + `alloc`: pure byte framing; file I/O lives in the PyO3 layer.
//!
//! Layout (little-endian):
//! ```text
//!   magic            4   b"LMQC"
//!   version          1   LMQC_VERSION
//!   flags            1   bit0=names_present, bit1=coords_present
//!   payload_kind     1   0=fp16-latent, 1=fsq-tokens(reserved)
//!   reserved         1   0
//!   n_channels       2   N (electrode count; independent of latent_c)
//!   latent_c         2   latent feature dim (32)
//!   latent_t         2   latent time frames (79)
//!   sample_rate      2   decoded fs (250)
//!   window_samples   4   decoded output length (2500)
//!   [coords]         N*3*4  f32          (iff coords flag)
//!   [names]          4 (len) + utf8      (iff names flag; names '\n'-joined)
//!   payload          4 (len) + bytes
//!   crc32            4   CRC-32 ISO 3309 over all preceding bytes
//! ```
extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::crc32::crc32;

pub const LMQC_MAGIC: &[u8; 4] = b"LMQC";
pub const LMQC_VERSION: u8 = 1;
pub const HEADER_SIZE: usize = 20;

pub const FLAG_NAMES: u8 = 1 << 0;
pub const FLAG_COORDS: u8 = 1 << 1;

pub const PAYLOAD_FP16_LATENT: u8 = 0;
pub const PAYLOAD_FSQ_TOKENS: u8 = 1;

#[derive(Debug, PartialEq)]
pub enum LmqcError {
    TooShort,
    BadMagic,
    BadVersion(u8),
    CrcMismatch,
    Truncated,
    BadCoordsLen,
    BadNamesLen,
    BadUtf8,
}

/// Decoded container.
#[derive(Debug, Clone, PartialEq)]
pub struct LmqcContainer {
    pub version: u8,
    pub n_channels: u16,
    pub latent_c: u16,
    pub latent_t: u16,
    pub sample_rate: u16,
    pub window_samples: u32,
    pub payload_kind: u8,
    pub coords: Option<Vec<f32>>,    // flat, len = 3*N (row-major [N,3])
    pub channels: Option<Vec<String>>,
    pub payload: Vec<u8>,
}

/// Frame an LMQC container. `coords` (if present) must have len `3*n_channels`;
/// `names` (if present) must have len `n_channels`.
pub fn encode_lmqc(
    n_channels: u16,
    latent_c: u16,
    latent_t: u16,
    sample_rate: u16,
    window_samples: u32,
    payload_kind: u8,
    coords: Option<&[f32]>,
    names: Option<&[String]>,
    payload: &[u8],
) -> Result<Vec<u8>, LmqcError> {
    let n = n_channels as usize;
    if let Some(c) = coords {
        if c.len() != 3 * n {
            return Err(LmqcError::BadCoordsLen);
        }
    }
    if let Some(nm) = names {
        if nm.len() != n {
            return Err(LmqcError::BadNamesLen);
        }
    }
    // Names are '\n'-joined on the wire; a name containing '\n' would split
    // into the wrong count on decode. Reject it (no real EEG label has one).
    if let Some(nm) = names {
        if nm.iter().any(|s| s.contains('\n')) {
            return Err(LmqcError::BadNamesLen);
        }
    }
    let mut flags = 0u8;
    if coords.is_some() {
        flags |= FLAG_COORDS;
    }
    if names.is_some() {
        flags |= FLAG_NAMES;
    }

    let mut buf = Vec::with_capacity(HEADER_SIZE + payload.len() + 64);
    buf.extend_from_slice(LMQC_MAGIC);
    buf.push(LMQC_VERSION);
    buf.push(flags);
    buf.push(payload_kind);
    buf.push(0u8); // reserved
    buf.extend_from_slice(&n_channels.to_le_bytes());
    buf.extend_from_slice(&latent_c.to_le_bytes());
    buf.extend_from_slice(&latent_t.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&window_samples.to_le_bytes());

    if let Some(c) = coords {
        for v in c {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    if let Some(nm) = names {
        let joined = nm.join("\n");
        let bytes = joined.as_bytes();
        let len: u32 = bytes.len().try_into().map_err(|_| LmqcError::BadNamesLen)?;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(bytes);
    }
    let plen: u32 = payload.len().try_into().map_err(|_| LmqcError::Truncated)?;
    buf.extend_from_slice(&plen.to_le_bytes());
    buf.extend_from_slice(payload);

    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse + validate (magic, version, CRC, bounds) an LMQC container.
pub fn decode_lmqc(buf: &[u8]) -> Result<LmqcContainer, LmqcError> {
    if buf.len() < HEADER_SIZE + 4 {
        return Err(LmqcError::TooShort);
    }
    let stored_crc = rd_u32(buf, buf.len() - 4);
    if crc32(&buf[..buf.len() - 4]) != stored_crc {
        return Err(LmqcError::CrcMismatch);
    }
    if &buf[0..4] != LMQC_MAGIC {
        return Err(LmqcError::BadMagic);
    }
    let version = buf[4];
    if version != LMQC_VERSION {
        return Err(LmqcError::BadVersion(version));
    }
    let flags = buf[5];
    let payload_kind = buf[6];
    let n_channels = rd_u16(buf, 8);
    let latent_c = rd_u16(buf, 10);
    let latent_t = rd_u16(buf, 12);
    let sample_rate = rd_u16(buf, 14);
    let window_samples = rd_u32(buf, 16);
    let n = n_channels as usize;
    let end = buf.len() - 4; // exclude crc

    // All offset arithmetic is checked: a crafted length field (near u32::MAX)
    // must NOT wrap `off + len` past `end` and bypass the bound on 32-bit
    // (firmware) targets. `bounded(off, len)` returns the validated end offset.
    let bounded = |off: usize, len: usize| -> Result<usize, LmqcError> {
        match off.checked_add(len) {
            Some(v) if v <= end => Ok(v),
            _ => Err(LmqcError::Truncated),
        }
    };

    let mut off = HEADER_SIZE;
    let mut coords = None;
    if flags & FLAG_COORDS != 0 {
        let nbytes = n.checked_mul(12).ok_or(LmqcError::Truncated)?; // 3 * f32
        let stop = bounded(off, nbytes)?;
        let mut c = Vec::with_capacity(n * 3);
        for i in 0..n * 3 {
            c.push(f32::from_le_bytes([
                buf[off + i * 4],
                buf[off + i * 4 + 1],
                buf[off + i * 4 + 2],
                buf[off + i * 4 + 3],
            ]));
        }
        coords = Some(c);
        off = stop;
    }
    let mut channels = None;
    if flags & FLAG_NAMES != 0 {
        let after_len = bounded(off, 4)?;
        let nlen = rd_u32(buf, off) as usize;
        off = after_len;
        let stop = bounded(off, nlen)?;
        let s = core::str::from_utf8(&buf[off..stop]).map_err(|_| LmqcError::BadUtf8)?;
        let names: Vec<String> = if nlen == 0 {
            Vec::new()
        } else {
            s.split('\n').map(|x| x.to_string()).collect()
        };
        // The names count MUST match n_channels (a crafted file could carry
        // fewer names than the header claims, silently mismatching the montage).
        if names.len() != n {
            return Err(LmqcError::BadNamesLen);
        }
        channels = Some(names);
        off = stop;
    }
    let after_len = bounded(off, 4)?;
    let plen = rd_u32(buf, off) as usize;
    off = after_len;
    let stop = bounded(off, plen)?;
    let payload = buf[off..stop].to_vec();

    Ok(LmqcContainer {
        version,
        n_channels,
        latent_c,
        latent_t,
        sample_rate,
        window_samples,
        payload_kind,
        coords,
        channels,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn names21() -> Vec<String> {
        (0..21).map(|i| alloc::format!("EEG E{}-REF", i)).collect()
    }

    #[test]
    fn roundtrip_coords_names_payload() {
        let coords: Vec<f32> = (0..63).map(|i| i as f32 * 0.01).collect();
        let names = names21();
        let payload: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let buf = encode_lmqc(21, 32, 79, 250, 2500, PAYLOAD_FP16_LATENT,
                              Some(&coords), Some(&names), &payload).unwrap();
        let d = decode_lmqc(&buf).unwrap();
        assert_eq!(d.n_channels, 21);
        assert_eq!((d.latent_c, d.latent_t), (32, 79));
        assert_eq!(d.sample_rate, 250);
        assert_eq!(d.window_samples, 2500);
        assert_eq!(d.coords.as_ref().unwrap(), &coords);
        assert_eq!(d.channels.as_ref().unwrap(), &names);
        assert_eq!(d.payload, payload);
    }

    #[test]
    fn nan_coords_preserved() {
        let mut coords = vec![0.05f32; 24];
        coords[0] = f32::NAN;
        let buf = encode_lmqc(8, 32, 79, 250, 2500, 0, Some(&coords), None, &[1, 2, 3]).unwrap();
        let d = decode_lmqc(&buf).unwrap();
        assert!(d.coords.as_ref().unwrap()[0].is_nan());
        assert_eq!(d.coords.as_ref().unwrap()[1], 0.05);
        assert!(d.channels.is_none());
    }

    #[test]
    fn crc_detects_corruption() {
        let mut buf = encode_lmqc(2, 32, 79, 250, 2500, 0, None, None, &[9, 9, 9]).unwrap();
        let i = buf.len() - 6;
        buf[i] ^= 0xFF;
        assert_eq!(decode_lmqc(&buf), Err(LmqcError::CrcMismatch));
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = encode_lmqc(1, 32, 79, 250, 2500, 0, None, None, &[0]).unwrap();
        buf[0] = b'X';
        let crc = crc32(&buf[..buf.len() - 4]);
        let n = buf.len();
        buf[n - 4..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(decode_lmqc(&buf), Err(LmqcError::BadMagic));
    }

    #[test]
    fn bad_coords_len_rejected() {
        let coords = vec![0.0f32; 5]; // not 3*2
        assert_eq!(
            encode_lmqc(2, 32, 79, 250, 2500, 0, Some(&coords), None, &[]),
            Err(LmqcError::BadCoordsLen)
        );
    }

    #[test]
    fn too_short_rejected() {
        assert_eq!(decode_lmqc(&[1, 2, 3]), Err(LmqcError::TooShort));
    }

    #[test]
    fn newline_in_name_rejected() {
        let names: Vec<String> = vec!["A".to_string(), "B\nC".to_string()];
        assert_eq!(
            encode_lmqc(2, 32, 79, 250, 2500, 0, None, Some(&names), &[]),
            Err(LmqcError::BadNamesLen)
        );
    }

    #[test]
    fn names_only_and_coords_only_roundtrip() {
        let names = names21();
        let b1 = encode_lmqc(21, 32, 79, 250, 2500, 0, None, Some(&names), &[7]).unwrap();
        let d1 = decode_lmqc(&b1).unwrap();
        assert_eq!(d1.channels.as_ref().unwrap(), &names);
        assert!(d1.coords.is_none());

        let coords: Vec<f32> = vec![0.1; 63];
        let b2 = encode_lmqc(21, 32, 79, 250, 2500, 0, Some(&coords), None, &[7]).unwrap();
        let d2 = decode_lmqc(&b2).unwrap();
        assert_eq!(d2.coords.as_ref().unwrap(), &coords);
        assert!(d2.channels.is_none());
    }

    #[test]
    fn crafted_names_count_mismatch_rejected() {
        // Header claims 21 channels but the names blob holds only 1 → reject.
        let one = vec!["solo".to_string()];
        let mut buf = encode_lmqc(1, 32, 79, 250, 2500, 0, None, Some(&one), &[]).unwrap();
        // patch n_channels (offset 8, u16 LE) from 1 → 21, fix CRC.
        buf[8] = 21;
        buf[9] = 0;
        let crc = crc32(&buf[..buf.len() - 4]);
        let n = buf.len();
        buf[n - 4..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(decode_lmqc(&buf), Err(LmqcError::BadNamesLen));
    }

    #[test]
    fn crafted_huge_len_does_not_panic() {
        // A names length near u32::MAX must be rejected, not wrap+OOB.
        let mut buf = encode_lmqc(1, 32, 79, 250, 2500, 0, None, Some(&vec!["x".to_string()]), &[]).unwrap();
        // names-len prefix sits right after the 20-byte header → bytes [20..24].
        buf[20..24].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        let crc = crc32(&buf[..buf.len() - 4]);
        let n = buf.len();
        buf[n - 4..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(decode_lmqc(&buf), Err(LmqcError::Truncated));
    }
}
