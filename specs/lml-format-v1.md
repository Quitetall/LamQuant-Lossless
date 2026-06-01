# LML Format Specification v1.0

**Status:** FROZEN. This specification defines the LML v1 wire format. Files encoded to this specification will be decodable by all future LML readers indefinitely.

**Date:** 2026-04-21 (rev 1: ctx 16→32, division semantics clarified)

**Authors:** OpenHuman Technologies

---

## 1. Introduction

LML (LamQuant Lossless) is a lossless compression format for multi-channel electrophysiological signals. It guarantees bit-exact reconstruction of the original digital signal values.

This document is the authoritative specification. If code and spec disagree, the spec wins and the code must be fixed. This spec is written to be implementable by any engineer without access to the reference code.

### 1.1 Design Principles

- **Self-describing:** Every file contains enough information to decode without external context.
- **Integrity at every level:** CRC-32 per window, SHA-256 per file, magic bytes at every boundary.
- **Forward-compatible:** Reserved fields and extensible metadata ensure v1 files remain readable as the format evolves.
- **Platform-independent:** All integers are little-endian. No floating-point in the wire format. No alignment requirements.

### 1.2 Conventions

Throughout this specification:
- **MUST**, **MUST NOT**, **SHOULD**, **MAY** per RFC 2119.
- All multi-byte integers are unsigned little-endian unless stated otherwise.
- `u8`, `u16`, `u32` = unsigned 8/16/32-bit. `i16`, `i32`, `i64` = signed (two's complement).
- `>>` = arithmetic right shift (sign-extending for signed types).
- Bit numbering within bytes: MSB-first (bit 7 is most significant).
- Byte order: always little-endian (least significant byte first).

---

## 2. File Container

An LML file has the extension `.lml` and consists of:

```
[Container Header]  [Metadata]  [Window Index]  [Window Payloads]  [LMLFOOT1 Footer]
```

The trailing `LMLFOOT1` footer (Section 2.8) is additive: current encoders write it
on every file, but a reader that stops after the last window payload still decodes
the full signal correctly. Its presence is signalled by header flag bit 0
(Section 2.2).

### 2.1 Container Header (32 bytes)

```
Offset  Size  Type    Field               Notes
------  ----  ------  ------------------  ---------------------------------
0       4     bytes   magic               'LML1' (0x4C 0x4D 0x4C 0x31)
4       1     u8      version_major       1
5       1     u8      version_minor       0
6       2     u16     n_channels          1 to 1024
8       2     u16     n_windows           1 to 65535
10      4     u32     total_samples       per channel
14      2     u16     window_size         samples per window
16      4     u32     sample_rate_mhz     sample rate × 1000 (e.g., 250 Hz = 250000)
20      1     u8      bit_depth           original sample bit depth (16, 24, 32)
21      1     u8      flags               see Section 2.2
22      4     u32     metadata_length     bytes of metadata JSON
26      2     u16     reserved_0          MUST be 0. Decoders MUST ignore.
28      4     u32     reserved_1          MUST be 0. Decoders MUST ignore.
```

**Total: 32 bytes.**

### 2.2 Container Flags

```
Bit 0:    has_footer (0 = no, 1 = LMLFOOT1 seek-table footer present at EOF)
Bits 1-7: reserved (MUST be 0, decoders MUST ignore)
```

Bit 0 (`FLAG_HAS_FOOTER`) is set by current encoders on **every** new file: the
`LMLFOOT1` seek-table footer (Section 2.8) is written unconditionally. The footer
is purely additive — it lives after the last window payload, so a reader that does
not understand the flag stops after the final window and silently ignores the
trailing bytes. There is no separate "has_seek_table" flag; the seek table is the
footer.

Decoders MUST ignore unknown flag bits. This allows older decoders to read newer
files that set new flags, as long as the core format is unchanged.

### 2.3 Metadata Block

Immediately after the 32-byte header:

```
Offset     Size              Content
------     ----              -------
32         metadata_length   UTF-8 JSON metadata
```

The metadata is a JSON object. Decoders MUST tolerate unknown keys. Required keys for v1.0:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `format_version` | string | YES | `"1.0"` |
| `encoder` | string | YES | Encoder identifier (e.g., `"lamquant-core 0.2.0"`) |
| `encoded_at` | string | YES | ISO 8601 UTC timestamp |
| `sample_rate` | number | YES | Sample rate in Hz |
| `n_channels` | integer | YES | Channel count |
| `channels` | string[] | SHOULD | Channel labels in order |
| `bit_depth` | integer | SHOULD | Original sample bit depth |
| `signal_sha256` | string | SHOULD | SHA-256 hex of original signal (channel-major i64 LE) |
| `source_file` | string | MAY | Original source filename |
| `patient_id` | string | MAY | Patient identifier |
| `phys_min` | number[] | MAY | Physical minimum per channel |
| `phys_max` | number[] | MAY | Physical maximum per channel |
| `dig_min` | integer[] | MAY | Digital minimum per channel |
| `dig_max` | integer[] | MAY | Digital maximum per channel |
| `annotations` | object[] | MAY | Clinical annotations (see Section 2.4) |

### 2.4 Annotations

Each annotation in the `annotations` array:

```json
{"onset": 10.5, "duration": 5.0, "description": "seizure"}
```

`onset` is seconds from recording start. `duration` is seconds (0 if instantaneous). `description` is free text.

### 2.5 Window Index

```
Offset              Size           Content
------              ----           -------
32 + M              4 * n_windows  u32 LE byte offsets into payload section
```

Each offset is relative to the start of the window payload section (the first byte after the index). Offset 0 means the first window payload.

### 2.6 Window Payloads

Each window is length-prefixed:

```
[4 bytes: payload_length u32 LE]  [payload_length bytes: compressed packet]
```

The compressed packet format is defined in Section 3.

### 2.7 Reserved Space and Extension

The 6 reserved bytes in the header (`reserved_0` at bytes 26-27, `reserved_1` at
bytes 28-31) and the 7 reserved flag bits (bits 1-7) provide room for future
features without a major version bump. Possible uses:
- Encryption indicator
- Custom block types
- Aggregate file-level checksums

The random-access seek table is already realised as the `LMLFOOT1` footer
(Section 2.8) and is no longer a reserved-space candidate.

**Encoders MUST write the reserved header bytes as zero. Decoders MUST ignore
these bytes.** This ensures that files with new features added via reserved bytes
remain readable by older decoders (they simply skip the unknown parts).

### 2.8 LMLFOOT1 Seek Footer

When header flag bit 0 (`has_footer`) is set, the file ends with a per-window
offset table immediately followed by a fixed 32-byte footer. Current encoders
always emit this footer; it enables `O(log n)` random access to any window without
scanning the window-length index from the start. The footer and table sit *after*
the last window payload, so a reader that ignores flag bit 0 stops after the final
window and never observes them.

**Footer (fixed 32 bytes, at `file_end - 32 .. file_end`):**

```
Offset  Size  Type    Field          Notes
------  ----  ------  -------------  --------------------------------------------
0       8     bytes   magic          'LMLFOOT1' (0x4C 4D 4C 46 4F 4F 54 31)
8       4     u32     footer_len     always 32
12      4     u32     n_windows      number of offset-table entries
16      4     u32     crc32          CRC-32 over (table bytes || footer bytes 0..16)
20      4     u32     table_start    absolute file offset where the table begins
24      8     bytes   reserved       8 × 0x00
```

**Offset table (`n_windows` × 16 bytes, at `table_start .. file_end - 32`):**

```
Offset  Size  Type   Field             Notes
------  ----  -----  ----------------  ----------------------------------------
0       8     u64    abs_offset        absolute file offset of the window's
                                       4-byte length prefix
8       4     u32    payload_len       length-prefix (4) + payload bytes
12      4     u32    first_sample_idx  sample index of this window's first sample
```

The footer's `crc32` covers the concatenation of all table-entry bytes plus the
leading 16 bytes of the footer (magic + footer_len + n_windows). A mismatch means
the footer is corrupt: conformant readers MUST fall back to the slow path (scanning
the Section 2.5 window index) rather than trusting the table. Readers MUST clamp
`n_windows` to a sane ceiling before allocating the table (the reference reader caps
at 2²⁰ = 1,048,576 windows) so an adversarial footer cannot trigger an unbounded
allocation. `table_start` MUST fit in `u32`, bounding v1 footer files to 4 GiB.

---

## 3. Per-Window Compressed Packet

### 3.1 ASCII Prefix

Every packet begins with a human-readable ASCII line terminated by `\n` (0x0A):

```
LML | {n_ch}ch | {mode} | CRC-32\n
```

Where `{mode}` is `lossless` when noise_bits=0, or `noise_bits={N}` otherwise.

This prefix is informational only. Decoders locate the binary header by scanning for `\n` followed by `LML` (3 bytes). The prefix MUST consist entirely of printable ASCII (0x20-0x7E) before the `\n`.

### 3.2 Packet Header (22 bytes)

```
Offset  Size  Type   Field                Notes
------  ----  ----   ------------------   ---------------------------------
0       4     bytes  magic                'LML1' (0x4C 0x4D 0x4C 0x31)
4       2     u16    n_channels           1 to 1024
6       2     u16    n_samples            per channel, this window
8       1     u8     n_levels             lifting DWT depth (typically 3)
9       1     u8     flags                see below
10      4     u32    lpc_meta_length      bytes
14      4     u32    payload_length       bytes
18      4     u32    crc32                CRC-32 of (lpc_meta + payload)
```

**Packet flags byte:**

```
Bit 0:     reserved (MUST be 0)
Bit 1:     reserved (MUST be 0)
Bits 2-7:  noise_bits (0-63; 0 = fully lossless)
```

### 3.3 CRC-32

CRC-32 per ISO 3309 / ITU-T V.42. Polynomial: 0xEDB88320 (reflected). Same algorithm as `zlib.crc32()`.

**Scope:** There are TWO CRC scopes in the wild. Both are computed with the
identical CRC-32 algorithm above; they differ only in which bytes are fed in.

| Scope | Cutover | CRC input bytes |
|-------|---------|-----------------|
| **Modern** (current encoder) | commit a81cd04, **2026-05-11** onward | `header[4..18]` (the 14 variable-header bytes: `n_channels`, `n_samples`, `n_levels`, `flags`, `lpc_meta_length`, `payload_length`) **then** `lpc_meta` **then** `payload` |
| **Legacy** (pre-a81cd04) | all files encoded before 2026-05-11 | `lpc_meta` **then** `payload` only — the 22-byte header is NOT covered |

The legacy scope cannot detect a single-byte corruption of the variable
header (e.g. a flipped `n_samples`) because those bytes are outside the CRC.
The modern scope was widened to close that gap. **The magic, the magic-byte
version digit, and `crc32` field itself (header bytes 0–3 and 18–21) are
excluded from both scopes** — only `header[4..18]` is added by the modern
scope.

#### 3.3.1 Back-compat fallback contract (decode side only)

The per-window packet header (Section 3.2) carries **no format-version
field**, so a decoder cannot tell a legacy packet from a modern one by
inspection — it must branch on the CRC itself. A conformant decoder MUST:

1. Compute the **modern** scope. If it equals the stored `crc32` → accept;
   this is the common fast path.
2. On mismatch ONLY, recompute the **legacy** (payload-only) scope. If THAT
   equals the stored `crc32`, the packet is a valid pre-a81cd04 packet whose
   data is intact (only the older CRC convention was used) → accept it, and
   SHOULD surface that a legacy-scope packet was read (the reference reader
   latches `lml::SAW_LEGACY_CRC` and warns once per process).
3. If BOTH scopes miss → genuine corruption → reject with a CRC error.

This fallback is **decode-side only**. Encoders MUST always write the modern
scope — they MUST NOT emit legacy-scope CRCs for new files. The fallback
never weakens corruption detection: a flipped byte inside `lpc_meta`/`payload`
fails both scopes (it is covered by both), so it is still rejected. Only a
flipped byte inside `header[4..18]` is asymmetric — it fails the modern scope
but, on a legacy file, was never CRC-covered to begin with, exactly as it was
not before a81cd04.

Decoders MUST verify the CRC-32 (per the contract above) and reject packets
where neither scope matches.

### 3.4 LPC Metadata

Serialized per-channel, per-subband. For n_levels=3, there are 4 subbands per channel in this order:

1. L3 approximation (order 1)
2. L3 detail (order 1)
3. L2 detail (order 2)
4. L1 detail (order 3)

For each subband:

```
[1 byte]         order (u8, 0-16)
[4*order bytes]  coefficients (i32 LE, Q27 fixed-point)
```

When order=0, no coefficient bytes follow. The reference encoder caps the LPC
order at 16 (`LPC_ORDER_HARD_CAP`); the `Fixed` mode uses a per-subband schedule
that tops out at order 8, while `Adaptive`/`Anytime` modes search up to 16. The
order field is a `u8`, so decoders MUST handle any declared order and validate that
`4*order` coefficient bytes are actually present.

### 3.5 Golomb-Rice Payload

Serialized per-channel, per-subband (same order as LPC metadata).

For each subband:

```
[1 byte]   k (u8, Golomb-Rice parameter, 0-31)
[2 bytes]  n_total (u16 LE, number of encoded values)
[variable] bitstream
```

---

## 4. Compression Pipeline

### 4.1 Overview

```
Input: signal[C][T] as i64
  → noise bit strip (signal >>= noise_bits)
  → per-channel 3-level Le Gall 5/3 integer lifting DWT
  → per-subband LPC analysis (Levinson-Durbin → Q27 prediction → residual)
  → per-subband bias cancellation (running mean, ctx=32, floor division)
  → per-subband Golomb-Rice coding (zigzag → adaptive k → unary + binary)
Output: compressed bytes
```

### 4.2 Integer Lifting DWT (Le Gall 5/3)

All arithmetic is integer. No floating-point. Bit-exact across all platforms.

**Forward transform** on signal x[0..N-1]:

Predict step (writes odd indices):
```
For n = 0 to floor(N/2)-2:
    x[2n+1] -= (x[2n] + x[2n+2]) >> 1

If N is even:
    x[N-1] -= x[N-2]
If N is odd:
    x[N-2] -= (x[N-3] + x[N-1]) >> 1
```

Update step (writes even indices):
```
x[0] += (x[1] + 1) >> 1

For n = 1 to ceil(N/2)-1:
    If 2n+1 < N:
        x[2n] += (x[2n-1] + x[2n+1] + 2) >> 2
    Else:
        x[2n] += (x[2n-1] + 1) >> 1
```

De-interleave: approx = x[0], x[2], x[4], ... and detail = x[1], x[3], x[5], ...

Applied 3 times in sequence: L1 → L2 → L3.

**Inverse transform** is the exact reverse: undo update, then undo predict, then interleave.

### 4.3 LPC Analysis

1. Compute biased autocorrelation R[0..order] on the first min(256, floor(T/2)) samples.
2. Levinson-Durbin recursion → float64 LP coefficients a[0..order-1].
3. If R[0] ≤ 1e-12 (zero-energy signal): coefficients = all zeros.
4. Quantize: `coeffs_q27[i] = round(-a[i] × 2^27)` as i32.
5. Integer forward prediction:
```
For n = 0 to T-1:
    pred = sum(coeffs_q27[k] × signal[n-1-k] for k in 0..order where n-1-k ≥ 0)
    residual[n] = signal[n] - (pred >> 27)
```

### 4.4 Bias Cancellation

Running mean subtraction with circular buffer of length 32:

```
buf[0..31] = 0
running_sum = 0
For i = 0 to T-1:
    bias = floor(running_sum / 32)     (floor division, toward negative infinity)
    val = residual[i]
    residual[i] = residual[i] - bias
    old = buf[i & 0x1F]
    buf[i & 0x1F] = val
    running_sum = running_sum + val - old
```

**IMPORTANT:** The division MUST use floor semantics (round toward negative infinity),
not truncation (toward zero). For positive running_sum these are identical. For negative
running_sum: `floor(-997 / 32) = -32`, NOT `-31`. Using truncation will produce wrong
output that passes CRC but is not bit-identical to the reference encoder.

### 4.5 Zigzag Encoding

Maps signed integers to unsigned for Golomb-Rice:

```
zigzag(v) = (v << 1) XOR (v >> 63)
```

0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...

### 4.6 Golomb-Rice Entropy Coding

**k parameter selection:** k = floor(log2(mean of nonzero zigzag values)). If all values are zero, k = 0.

**Encoding** each zigzag value v:
1. Quotient: q = v >> k
2. Remainder: r = v & ((1 << k) - 1)
3. Write q zeros followed by a single 1 bit (unary code)
4. Write k bits of r (MSB-first, binary)

**Bitstream flushing:** Final byte is left-aligned (unused bits are zero-padded on the right).

---

## 5. Decompression

Exact inverse of Section 4, in reverse order:

1. Golomb-Rice decode → zigzag decode → residuals
2. Bias restoration (running mean **addition**, same algorithm with += instead of -=)
3. LPC synthesis (IIR feedback: `signal[n] = restored[n] + (pred >> 27)`)
4. Inverse lifting DWT (3 levels)
5. Noise bit restoration: `signal <<= noise_bits`

### 5.1 Bias Restoration

```
buf[0..31] = 0
running_sum = 0
For i = 0 to T-1:
    bias = floor(running_sum / 32)     (floor division, toward negative infinity)
    data[i] = data[i] + bias
    old = buf[i & 0x1F]
    buf[i & 0x1F] = data[i]
    running_sum = running_sum + data[i] - old
```

### 5.2 LPC Synthesis

```
For n = 0 to T-1:
    pred = sum(coeffs_q27[k] × signal[n-1-k] for k in 0..min(order, n)-1)
    signal[n] = restored[n] + (pred >> 27)
```

---

## 6. Versioning

### 6.1 Version Scheme

Major.Minor (e.g., 1.0, 1.1, 2.0).

- **Major version** (in magic byte): Breaking format change. Old decoders cannot read new files. New decoders SHOULD read old files.
- **Minor version** (in header byte): Additive features. Old decoders can still read new files by ignoring unknown flags, metadata keys, and reserved fields.

### 6.2 Magic Byte Structure

Bytes 0-3: `LML` + ASCII digit (version major). `LML1` = version 1.x, `LML2` = version 2.x, etc.

### 6.3 Reader Behavior

| Magic | Action |
|-------|--------|
| `LML1` | Decode normally |
| `LML2` through `LML9` | Reject: "LML version N newer than reader. Update." |
| `LML` + non-digit | Reject: "Invalid LML magic. File corrupt." |
| Anything else | Reject: "Not an LML file." |

### 6.4 Forward Compatibility Rules

A v1.0 decoder encountering a v1.x file (x > 0) MUST:
1. Read the 32-byte header normally.
2. Ignore unknown flag bits.
3. Ignore reserved header bytes.
4. Skip unknown metadata JSON keys.
5. Decode all windows using the v1 pipeline.
6. Ignore any trailing data after the last window.

This guarantees that features added in v1.1, v1.2, etc. do not break v1.0 decoders.

### 6.5 Backward Compatibility Commitment

All files encoded with LML v1.x MUST be decodable by all future LML readers. This commitment is permanent and irrevocable. Clinical data compressed today will be readable in 2056.

---

## 7. Integrity

### 7.1 Per-Window CRC-32

Every window packet has a CRC-32 (Section 3.3). The current ("modern")
encoder covers `header[4..18] || lpc_meta || payload`; files written before
commit a81cd04 (2026-05-11) cover `lpc_meta || payload` only. Decoders MUST
verify the CRC using the two-scope fallback contract in Section 3.3.1 and
reject only when BOTH scopes miss.

**Hardening gap (known).** The per-window packet header (Section 3.2) has no
format-version field, so the modern vs legacy CRC scope is NOT self-describing
— a reader must branch on the CRC value itself. The a81cd04 cutover changed
the scope on both encode and decode with no version gate, which silently broke
back-compat for every pre-cutover file until the Section 3.3.1 fallback was
added. See Appendix F for the planned remediation.

### 7.2 Per-File SHA-256

Encoders SHOULD compute SHA-256 of the original signal bytes (channel-major, i64 LE, before noise-bit stripping) and store it in the metadata as `signal_sha256`. After encoding, encoders SHOULD decode the file and verify the hash matches.

### 7.3 Decoder Safety Requirements

A conformant decoder MUST:
- Reject files with invalid magic
- Reject packets with CRC-32 mismatch
- Reject truncated files (declared sizes exceed actual data)
- Bound memory allocation (no unbounded malloc from header values)
- Never crash, hang, or execute arbitrary code on any input
- Return a structured error for all failure modes

---

## 8. Provenance

Every LML file MUST contain the following in metadata, establishing an unbroken chain of provenance:

| Field | Purpose |
|-------|---------|
| `encoder` | Software and version that produced this file |
| `encoded_at` | Timestamp of encoding (ISO 8601 UTC) |
| `format_version` | Spec version (`"1.0"`) |
| `sample_rate` | Prevents misinterpretation of timing |
| `n_channels` | Prevents misinterpretation of data layout |

These fields enable any future reader to understand how, when, and why the file was created — even without access to the original software.

---

## Appendix A. Test Vectors

Canonical test: 21 channels × 2500 samples, values in [-5000, 5000]:

```python
import numpy as np
signal = np.random.default_rng(42).integers(-5000, 5000, (21, 2500), dtype=np.int64)
```

A conformant encoder+decoder MUST roundtrip this signal bit-exactly.

## Appendix B. Reference Implementations

| Language | Location | LOC | Status |
|----------|----------|-----|--------|
| Rust | `lamquant-lossless` crate (lib target `lamquant_core`) | ~2200 | Canonical |
| Python | `lamquant_codec/` | ~1500 | Production |

Both produce byte-identical output for the same input.

## Appendix C. Format Limits

| Parameter | Min | Max | Wire Type |
|-----------|-----|-----|-----------|
| n_channels | 1 | 1024 | u16 |
| n_samples/window | 1 | 65535 | u16 |
| n_windows/file | 1 | 65535 | u16 |
| total_samples/file | 1 | 4,294,967,295 | u32 |
| sample_rate_mhz | 1 | 4,294,967,295 | u32 |
| noise_bits | 0 | 63 | 6 bits |
| LPC order | 0 | 16 | u8 |
| Golomb-Rice k | 0 | 31 | u8 |
| metadata_length | 0 | 16,777,216 | u32 |
| payload_length | 0 | 268,435,456 | u32 |

## Appendix D. Byte Order Reference

This format uses **little-endian** byte order exclusively. Example: the value 0x1234 is stored as bytes [0x34, 0x12]. This matches x86, ARM (default), and RISC-V (default) native byte order. Big-endian systems MUST byte-swap when reading/writing.

## Appendix E. Change Policy

- **v1.0.x** (patch): Spec clarifications only. No format changes.
- **v1.x.0** (minor): New optional features via reserved fields/flags. v1.0 decoders still work.
- **v2.0.0** (major): Format changes that break v1 decoders. Minimum 2 years notice. v2 decoders MUST read v1.

This specification is designed to last 30+ years. Changes are made reluctantly.

## Appendix F. Future Work — Per-Packet Format Version (NOT YET IMPLEMENTED)

**Status: PROPOSED. Not part of v1.0. Do not implement against this section
yet — it is recorded here so the gap is tracked, not closed.**

The root cause of the a81cd04 (2026-05-11) CRC-scope back-compat break
(Sections 3.3.1, 7.1) is that the per-window packet header (Section 3.2)
carries **no format-version field**. Old and new packets are therefore not
self-describing: a reader cannot tell which CRC scope a packet used without
trial-recomputing both. The Section 3.3.1 fallback patches the symptom for
this one cutover, but the underlying ambiguity remains for any future
scope/layout change.

The intended permanent fix is a real **per-packet format-version field** so
that old and new packets self-describe and a reader can dispatch the correct
CRC scope (and any future layout variation) deterministically, with no
CRC-guessing fallback. Candidate encoding, to be finalised when this is
scheduled:

- Repurpose one of the two reserved bits in the packet flags byte (Section
  3.2, bits 0–1 currently "MUST be 0"), OR a small `u8` in a minor-version
  packet layout, to carry a packet-format generation number.
- Generation 0 = legacy payload-only CRC scope; generation 1 = a81cd04
  header+payload scope; future generations as needed.
- Once present, the Section 3.3.1 "recompute both scopes" fallback becomes a
  pure read-compat shim for generation-0 files that lack the field, and new
  writers stamp the generation explicitly.

This is a minor (v1.x) additive change under Appendix E — old decoders ignore
the reserved bit, new decoders honour it — and MUST preserve the Section 6.5
backward-compatibility commitment.
