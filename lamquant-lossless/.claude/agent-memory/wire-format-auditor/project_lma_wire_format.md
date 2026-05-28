---
name: LMA/LML Wire Format Audit — 2026-04-24 (full audit)
description: Canonical wire format details, known issues, critical bias-cancel numba bug, and file locations for all LML/LMA codec formats
type: project
---

Full audit completed 2026-04-24 across all 10 Rust and Python codec files.

## LML Per-Window Packet (22-byte header)
Layout: magic(4) + n_ch(2) + T(2) + n_levels(1) + flags(1) + lpc_len(4) + sub_len(4) + crc32(4)
Canonical files: lamquant-core/src/lml.rs (HEADER_SIZE=22, MAGIC=b"LML1") and lamquant_codec/lossless.py (hdr_size=22, format '<4sHHBBIII')
All fields CONSISTENT across Rust, Python ref path, and Python fused/numba path.

## LML Container File (32-byte header)
Layout: magic(4) + ver_major(1) + ver_minor(1) + n_ch(2) + n_win(2) + total(4) + ws(2) + sr_mhz(4) + bit_depth(1) + flags(1) + meta_len(4) + reserved(6)
Canonical files: lamquant-core/src/container.rs and lamquant_codec/edf_to_lml.py (struct '<4sBBHHIHIBBI2x4x' = 32 bytes)
All fields CONSISTENT. Python _read_header peek[] offset arithmetic verified correct for 32-byte format.

## LMA Archive (16-byte header)
Layout: magic(4) + version_u32_LE(4) + n_entries_u32_LE(4) + manifest_len_u32_LE(4) | zstd_manifest | payloads | sha256(32)
Canonical files: lamquant-core/src/lma.rs (LMA_MAGIC=b"LMA1", LMA_VERSION=1) and lamquant_codec/lma.py (same)
CONSISTENT. Manifest always zstd (hardcoded both sides). Payload section offset = 16 + manifest_len (consistent).

## Subband Order
Both emit [l3_approx, l3_detail, l2_detail, l1_detail] per channel.
LPC order schedule both use [1, 1, 2, 3] for [approx, d3, d2, d1].

## Golomb-Rice Format
Both: [k:u8][n_total:u16LE][bitstream]. Zigzag: (v<<1)^(v>>63). CONSISTENT.

## LPC
Q-factor: Q27 everywhere (lpc.rs:6, lpc.py:29, constants.py:28). Coefficients: 1 byte order + i32 LE per coeff. CONSISTENT.

## Bias Cancellation Context Length
ctx_len = 32 everywhere (lml.rs:23, constants.py:26). CONSISTENT.

## DWT Le Gall 5/3
Predict step: d[i] -= (a[i]+a[i+1])>>1, boundary: d[last] -= a[last-1] for even-length.
Update step: a[0] += (d[0]+1)>>1, interior: a[i] += (d[i-1]+d[i]+2)>>2, right edge: a[i] += (d[i-1]+1)>>1.
CONSISTENT across Rust lifting.rs and Python lifting.py (all variants).

## CRITICAL BUG: Bias Cancel Floor Division in Numba JIT
File: lamquant_codec/ops/bias.py, lines 23, 34 (cancel), 38, 50 (restore)

Problem: Numba @njit uses C-style truncated division for `//`, NOT Python floor division.
For negative running_sum values, Python `//` rounds toward -inf but numba `//` truncates toward 0.
Example: -1 // 32 = -1 (Python) vs 0 (numba).

Impact: Packets compressed with the numba path cannot be correctly decompressed by Rust (or Python ref path) when running_sum goes negative during bias cancellation. The CRC still passes (it covers bias-cancelled data), so corruption is SILENT. Affects production encode path whenever numba is installed.

Rust floor_div (lpc.rs:109-111) correctly matches Python // semantics. Python non-numba fallback (bias.py:59-90) is also correct.

Fix: In both _cancel_jit_inner and _restore_jit_inner, replace `bias = running_sum // ctx_len` with explicit floor-div:
  d = running_sum // ctx_len  # numba truncation
  bias = d - numba.int64(1) if (running_sum < 0) & (d * ctx_len != running_sum) else d

**Why:** Cross-language compatibility is production-critical. A silent CRC-passing corruption that fires on negative residual DC drift is a latent data integrity failure.
**How to apply:** Fix bias.py before any cross-language (Python→Rust or Rust→Python) decompression is used in production.

## Known Non-Breaking Asymmetries
- method strings: Rust writes "secondary" for zstd payloads, Python writes "secondary". Both accept "zstd" as legacy alias.
- LMA manifest compressor: Python tries registered compressors in sequence on read; Rust always calls zstd::decode_all. Latent incompatibility if Python ever uses non-zstd secondary compressor (it hardcodes zstd for manifests but the code path is confusing).
- KLT flag (bit 0 of flags): Python can set it; Rust ignores it on decode. Not a production issue since KLT is not used by default.
