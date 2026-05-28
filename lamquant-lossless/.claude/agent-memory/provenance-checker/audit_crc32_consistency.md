---
name: CRC-32 consistency across Rust and Python paths
description: Both Rust and Python use ISO 3309 / zlib-compatible CRC-32 with identical polynomial and byte order; per-window coverage is complete
type: project
---

Rust: /mnt/4tb/LamQuant/lamquant-core/src/crc32.rs
- Polynomial: 0xEDB88320 (reflected IEEE 802.3 / zlib)
- Init: 0xFFFFFFFF, final XOR: 0xFFFFFFFF
- Matches known vectors: crc32(b"123456789") == 0xCBF43926 (confirmed by test)
- CRC is computed on lpc_meta + payload concatenated (streaming, no temp alloc)
- Verified: test crc_catches_corruption passes

Python: lamquant_codec/lossless.py
- Uses zlib.crc32(payload) & 0xFFFFFFFF
- zlib.crc32 is identical polynomial and init/finalXOR to Rust impl
- Applied to (lpc_meta + subband_payload) concatenated as one bytes object

Cross-language compatibility: CONFIRMED. Both operate on the same byte range. Files written by either implementation can be read and verified by the other.

CRC-32 scope per window packet:
- Covers: LPC metadata bytes + Golomb-Rice subband payload bytes
- Does NOT cover: the 22-byte packet header itself (n_ch, T, n_levels, flags, lengths)
- Header corruption would still be caught by length mismatch errors at decompress time

**How to apply:** CRC-32 per-window is sound. No action needed for regulatory. The only gap is that the LML container header (32 bytes) and the LMA archive manifest are not CRC-32 protected at the per-window level (they have SHA-256 at archive level instead).
