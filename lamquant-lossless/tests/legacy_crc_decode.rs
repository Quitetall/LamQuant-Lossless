//! LML legacy (pre-a81cd04) CRC back-compat decode gate.
//!
//! ROOT CAUSE: commit a81cd04 (2026-05-11, "fix(lml): CRC covers packet
//! header to detect single-byte header corruption") widened the LML1
//! per-window CRC-32 scope from `crc32(lpc_meta || payload)` (legacy,
//! payload-only) to `crc32(header[4..18] || lpc_meta || payload)`
//! (modern) on BOTH encode and decode — with no version gate and no
//! back-compat read path. The LML1 packet header carries no version
//! field, so every file written before a81cd04 fails CRC under the
//! current reader even though its bytes are perfectly intact.
//!
//! The fix (decode-side only, in `lml::verify_packet_crc`): on a CRC
//! miss against the modern scope, recompute the legacy payload-only
//! scope; if THAT matches, the packet is a valid pre-a81cd04 packet —
//! accept it and latch [`lamquant_core::lml::SAW_LEGACY_CRC`] + warn
//! once. If both scopes miss, it's genuine corruption → `CrcMismatch`.
//!
//! ## Fixtures
//!
//! `tests/fixtures/legacy_payload_crc.lml` is a REAL pre-a81cd04 LML
//! container, carved verbatim out of `/tmp/forensic/corrupt_mental.lma`
//! (the physionet `mental_arithmetic` corpus, `Subject33_2.edf`, the
//! smallest entry) via `lml cat`. Every one of its per-window packets
//! uses the legacy payload-only CRC scope, so it exercises the fallback
//! on real wire bytes, not a synthetic re-encode.
//!
//! These tests pin:
//!   1. POSITIVE — the legacy container decodes successfully via the
//!      new fallback, `SAW_LEGACY_CRC` latches, and the recovered
//!      samples match a frozen sha256 (futureproof golden: a fixed
//!      hash, not an implementation-derived expected value).
//!   2. NEGATIVE — flipping one payload byte of a legacy packet makes
//!      BOTH scopes miss, so `lml::decompress` returns `CrcMismatch`
//!      (the fallback does not weaken genuine corruption detection).

use lamquant_core::container;
use lamquant_core::error::LmlError;
use lamquant_core::lml;
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/legacy_payload_crc.lml"
);

/// sha256 over the decoded samples (channel-major, each i64 little-endian).
/// Frozen golden — regenerate ONLY if the fixture itself is replaced, and
/// explain why in the commit message.
const DECODED_SAMPLES_SHA256: &str =
    "2dd003d15e50b8d5e7d927923ea1b9e7a6d5b07e6cbce319153ce6d8926377c0";

/// Stable byte serialization of the decoded signal: channel-major,
/// every sample as 8 little-endian bytes. Pinned via sha256 so the test
/// asserts against a fixed value, not anything derived from the decoder
/// under test (futureproof-tests memory: pin against frozen goldens).
fn samples_sha256(signal: &[Vec<i64>]) -> String {
    let mut hasher = Sha256::new();
    for ch in signal {
        for &s in ch {
            hasher.update(s.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Locate the first per-window LML1 packet inside a container buffer and
/// return `(packet_start, packet_len)`. Mirrors the container framing in
/// `container.rs::parse_header` + `window_for_sample`:
///
///   - 32-byte header (current `write_file`): `data[20] ∈ {16,24,32}`,
///     `n_ch = u16 @6`, `n_windows = u16 @8`, `meta_len = u32 @22`.
///   - `payload_start = 32 + meta_len + n_windows*4`.
///   - The `n_windows × u32` index at `32 + meta_len` holds each
///     window's relative offset from `payload_start` to its
///     `[u32 len LE][packet bytes]` block. Window 0's rel-off is the
///     first u32 in the index.
///
/// We resolve window 0's block, read its length prefix, and return the
/// packet slice so the negative test can corrupt a real legacy packet.
fn first_window_packet(buf: &[u8]) -> (usize, usize) {
    assert!(&buf[0..4] == lml::MAGIC, "container starts with LML1 magic");
    // The fixture is written by the current 32-byte-header container.
    assert!(
        buf.len() >= 32 && matches!(buf[20], 16 | 24 | 32),
        "expected 32-byte container header (data[20] ∈ {{16,24,32}})"
    );
    let n_windows = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    assert!(n_windows >= 1, "container has at least one window");
    let meta_len = u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]) as usize;
    let hdr_end = 32usize;
    let index_start = hdr_end + meta_len;
    let payload_start = index_start + n_windows * 4;
    // Window 0 relative offset (first u32 in the index).
    let rel_off = u32::from_le_bytes([
        buf[index_start],
        buf[index_start + 1],
        buf[index_start + 2],
        buf[index_start + 3],
    ]) as usize;
    let block_pos = payload_start + rel_off;
    let len = u32::from_le_bytes([
        buf[block_pos],
        buf[block_pos + 1],
        buf[block_pos + 2],
        buf[block_pos + 3],
    ]) as usize;
    let packet_start = block_pos + 4;
    assert!(
        packet_start + len <= buf.len(),
        "window packet length {len} overruns buffer"
    );
    // The per-window packet begins with the encoder's ASCII prefix
    // ("LML | 21ch | lossless | CRC-32\n"); the binary `LML1` magic
    // follows it. `lml::decompress` skips the prefix via its internal
    // `find_magic_offset`, so we only assert the magic is present
    // somewhere in the carved slice, not that it sits at byte 0.
    let pkt = &buf[packet_start..packet_start + len];
    assert!(
        pkt.windows(4).any(|w| w == lml::MAGIC),
        "first window packet contains LML1 magic after its ASCII prefix"
    );
    (packet_start, len)
}

#[test]
fn legacy_container_decodes_via_fallback() {
    // Reset the latch so this test observes its own effect regardless of
    // test ordering (integration tests in one binary share the static).
    lml::SAW_LEGACY_CRC.store(false, Ordering::Relaxed);

    let (signal, _metadata) = container::read_file(std::path::Path::new(FIXTURE))
        .expect("legacy container must decode via the payload-only CRC fallback");

    assert!(!signal.is_empty(), "decoded signal has channels");
    assert!(!signal[0].is_empty(), "decoded signal has samples");

    // The decode must have taken the legacy path (this is a pre-a81cd04
    // artifact). If this assert fails, either the fixture was regenerated
    // with a modern encoder or the fallback never fired.
    assert!(
        lml::SAW_LEGACY_CRC.load(Ordering::Relaxed),
        "SAW_LEGACY_CRC must latch: the fixture is a legacy payload-only-CRC file"
    );

    let got = samples_sha256(&signal);
    assert_eq!(
        got, DECODED_SAMPLES_SHA256,
        "decoded samples drifted from the frozen golden (sha256). \
         If the fixture changed intentionally, update DECODED_SAMPLES_SHA256 \
         and say why in the commit message."
    );
}

#[test]
fn flipped_payload_byte_still_rejected() {
    // A genuinely corrupt packet must fail BOTH scopes. We carve the
    // first legacy window packet, flip a byte well inside its payload
    // (past the 22-byte header so we corrupt data the CRC covers under
    // BOTH the modern and legacy scopes), and assert decompress rejects.
    let buf = std::fs::read(FIXTURE).expect("read fixture");
    let (start, len) = first_window_packet(&buf);
    let mut packet = buf[start..start + len].to_vec();

    // Sanity: the carved bytes are themselves a valid legacy packet.
    lml::SAW_LEGACY_CRC.store(false, Ordering::Relaxed);
    lml::decompress(&packet).expect("carved packet decodes cleanly before corruption");

    // Flip one byte deep inside the CRC-covered region. The packet opens
    // with an ASCII prefix, then the 22-byte binary header at the LML1
    // magic, then lpc_meta+payload. Locate the magic and flip a byte well
    // past header end so it's squarely in lpc_meta/payload — covered by
    // BOTH the modern and legacy CRC scopes, so a flip there must trip the
    // genuine-corruption path (both scopes miss).
    let magic_off = packet
        .windows(4)
        .position(|w| w == lml::MAGIC)
        .expect("packet has LML1 magic");
    // 22-byte header + 64 bytes into the payload region.
    let flip_at = magic_off + 22 + 64;
    assert!(flip_at < packet.len(), "packet large enough to corrupt payload");
    packet[flip_at] ^= 0x01;

    match lml::decompress(&packet) {
        Err(LmlError::CrcMismatch { .. }) => { /* expected: both scopes miss */ }
        Err(other) => panic!(
            "corrupt payload produced {other:?}; expected CrcMismatch (both CRC scopes must miss)"
        ),
        Ok(_) => panic!(
            "corrupt payload decoded successfully — the legacy fallback must NOT accept \
             a packet whose payload-only CRC also fails"
        ),
    }
}
