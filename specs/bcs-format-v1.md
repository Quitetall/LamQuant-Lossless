# BCS Format Specification v1.0

**Status:** FROZEN. This specification defines the BCS1 wire format. Files encoded to this specification will be decodable by all future BCS readers indefinitely.

**Date:** 2026-07-02

**Authors:** OpenHuman Technologies

**Related decisions:** ADR 0069 (ABIR — the modality-typed IR), ADR 0071 (BCS — the Biosignal Compression Standard).

---

## 1. Introduction

BCS (Biosignal Compression Standard) is a **neutral, descriptor-driven container** for compressed multi-channel biosignal recordings. A BCS file wraps a compressed body — today the LML integer lossless stream, in future an LMO (lossy) or LMQ (neural) body — behind a single fixed 40-byte typed header. The header carries the recording's **born-typed modality** (EEG, iEEG, ECG, …), its provenance, and a **codec descriptor** that names the body format; **decode dispatches on the descriptor byte, not on the magic**.

This document is the authoritative specification. If code and spec disagree, the spec wins and the code must be fixed. It is written to be implementable by any engineer without access to the reference code.

### 1.1 Design Principles

- **One header, many bodies.** The 40-byte header is codec-neutral. The `codec_descriptor` field (Section 3) selects the body decoder. A single reader framework decodes every BCS body it has a decoder for, and fails closed — never silently mis-decodes — for a body it does not.
- **Born-typed.** Modality is a required header field, assigned at capture/lowering from channel labels + format metadata, with the assignment method (`modality_source`) recorded. A file is never silently "untyped by default".
- **Byte-stable body.** Everything at offset ≥ 40 (metadata, window index, payloads, footer) is byte-for-byte the corresponding LML1 layout (Section 5) when `codec_descriptor = CODEC_LML_53`. BCS1 is a header re-frame of a proven body, not a new body.
- **Legacy decode-forever.** The seven pre-BCS magics remain decodable indefinitely (Section 6). A BCS reader dispatches on the leading four bytes.
- **Platform-independent.** All multi-byte integers are unsigned little-endian. No floating-point in the header. No alignment requirements.

### 1.2 Conventions

- **MUST**, **MUST NOT**, **SHOULD**, **MAY** per RFC 2119.
- All multi-byte integers are unsigned little-endian unless stated otherwise.
- `u8`, `u16`, `u32` = unsigned 8/16/32-bit. `i16`, `i32`, `i64` = signed (two's complement).
- Byte order: always little-endian.

---

## 2. File Container

A BCS file has the extension `.lml` (a BCS-framed LML body) or `.lmq` (a BCS-framed neural body, future) and consists of:

```
[BCS1 Header (40 bytes)]  [Metadata]  [Window Index]  [Window Payloads]  [LMLFOOT1 Footer]
```

The trailing `LMLFOOT1` footer is additive: current encoders write it on every file (header `flags` bit 0 set), but a reader that stops after the last window payload still decodes the full signal correctly. The footer provides O(log n) random access (Section 5.4).

Everything from offset 40 onward is **identical to the LML1 v1 container body** (see `lml-format-v1.md` §2.3–2.8) when `codec_descriptor = CODEC_LML_53`. The only deliberate change from LML1 is the header: a fixed 40-byte BCS1 header replaces the 32-byte LML1 header.

---

## 3. BCS1 Header (40 bytes)

```
Offset  Size  Type    Field               Notes
------  ----  ------  ------------------  --------------------------------------------
0       4     bytes   magic               'BCS1' (0x42 0x43 0x53 0x31)
4       1     u8      version_major       1
5       1     u8      version_minor       0
6       1     u8      modality_tag        born-typed modality (Section 4.1); 255 = untyped
7       1     u8      modality_source     how modality_tag was decided (Section 4.2)
8       1     u8      codec_descriptor    body format selector (Section 4.3)
9       1     u8      mode                0 = Lossless, 1 = BoundedMae, 2 = TargetBps
10      1     u8      tier                deployment stamp, DESCRIPTIVE only (Section 4.4)
11      1     u8      decode_capability   minimum reader capability required (Section 4.5)
12      2     u16     n_channels          1 to 1024
14      2     u16     n_windows           1 to 65535
16      4     u32     total_samples       per channel
20      2     u16     window_size         samples per window
22      4     u32     sample_rate_mhz     sample rate × 1000 (milli-Hz), e.g. 250000 = 250 Hz
26      1     u8      bit_depth           source ADC bit depth (e.g. 16)
27      1     u8      flags               bit 0 = LMLFOOT1 footer present
28      4     u32     metadata_length     length of the Metadata section in bytes
32      8     bytes   reserved            MUST be written as zero; readers MUST ignore
```

Total: **40 bytes**. A reader MUST reject a file shorter than 40 bytes (`Truncated`), and MUST reject a file whose first four bytes are not `BCS1` when dispatched to the BCS1 parser (`InvalidMagic`). Bytes 32..40 are reserved for a future header extension and MUST be zero in v1.

`magic` = `BCS1` is a fresh four-byte magic distinct from every legacy magic (Section 6), so a BCS1 file and a legacy file are told apart from byte 0. In particular, the legacy `LML1` parser rejects a BCS1 file at its first guard (`B` ≠ `L`), and vice versa; there is no probe-field ambiguity between the formats.

---

## 4. Header Field Semantics

### 4.1 modality_tag (offset 6)

The born-typed modality of the recording (for a multi-modality/PSG file, the majority modality; per-channel typing is a metadata concern). Values:

```
0   eeg        scalp electroencephalography
1   ieeg       intracranial EEG
2   ecog       electrocorticography
3   seeg       stereo-EEG
4   ecg        electrocardiography
5   emg        electromyography
6   eog        electrooculography
7   resp       respiration
8   accel      accelerometry
9   other      not covered by a dedicated tag
255 untyped    modality could not be determined
```

Tags 10..254 are reserved. A reader encountering an unknown tag SHOULD treat it as `untyped` (it is provenance, not a decode gate).

### 4.2 modality_source (offset 7)

How `modality_tag` was decided — recorded so an audit can explain why a recording is typed the way it is:

```
0   channel_label     inferred from channel labels (e.g. 10-20 montage → eeg)
1   format_declared   declared by the source format's own metadata
2   manual            explicitly set by the operator / caller
```

### 4.3 codec_descriptor (offset 8) — the body selector

Names the compressed body format. **Decode dispatches on this byte.** Legacy values 0–2 retain their numerical LMO `transform_id` mapping; later values are independent BCS body identifiers:

```
0          CODEC_LML_53      LML integer 5/3 lifting stream (lossless floor)
1          CODEC_LMO_97      LMO-native 9/7 float PCRD body (lossy)
2          CODEC_LMO_LOSSLESS Optimum lossless body (cross-channel + LML)
3          CODEC_OPTIMUM_V2   Optimum v2 LMO1-v3/BGF1 learned-lossless body
0x10..0xFF (reserved)        LMQ / neural descriptor family
```

An implementation that supports descriptor 3 MUST dispatch it only to an `LMO1` version-3 `BGF1` decoder and MUST NOT interpret it as legacy LMO version-2 transform ID 3 (bounded MV-RLS). Baseline readers may refuse it fail-closed.

**Conformance:** a baseline BCS1 v1 encoder MUST emit `codec_descriptor = 0` (`CODEC_LML_53`) — it is the only body every v1 decoder is required to decode. A decoder that does not implement a given descriptor's body MUST fail closed with a clear error (`InvalidHeader` / "descriptor not wired"), and MUST NOT attempt to decode the body as if it were `CODEC_LML_53`. Descriptors 1–3 and the 0x10+ range parse cleanly; readers without those body decoders refuse them.

### 4.4 tier (offset 10) — descriptive, non-gating

The deployment tier the file was produced under: `0` = Mcu, `1` = Basestation. This is **provenance only**. A reader MUST NOT refuse to decode a file based on `tier`.

### 4.5 decode_capability (offset 11) — the decode gate

The minimum reader capability required to decode the body: `0` = the integer floor (every BCS reader, MCU included, decodes it). This is the ONLY value a v1 (`CODEC_LML_53`) encoder emits. A reader MUST refuse (fail closed) a file whose `decode_capability` exceeds what it implements. Unlike `tier`, this byte IS a gate.

---

## 5. Body (offset 40 onward)

For `codec_descriptor = CODEC_LML_53`, the body is byte-identical to the LML1 v1 container body. Summarized here; the normative reference is `lml-format-v1.md`.

### 5.1 Metadata (offset 40)

`metadata_length` bytes of UTF-8 JSON, byte-unchanged from the caller-supplied metadata (BCS1 does not re-encode it). Carries channel labels, per-channel calibration (`dig_min`/`dig_max`/`phys_min`/`phys_max`), `sample_rate`, `codec_mode`/`lpc_mode`, and other side-channel fields. A reader MUST validate it is valid UTF-8; it MAY tolerate metadata that is not a JSON object (treating it as opaque).

### 5.2 Window Index

`n_windows` × `u32` little-endian **relative** offsets. Entry `w` is the byte offset, relative to the first payload block (`payload_start = 40 + metadata_length + n_windows*4`), of window `w`'s `[u32 length][payload]` block. Identical to the LML1 window index.

### 5.3 Window Payloads

`n_windows` blocks, each `[u32 payload_length][payload bytes]`. Each payload is an LML1 packet (Section 5 of `lml-format-v1.md`) decoded by the same integer 5/3 + LPC + entropy pipeline. The last window MAY be shorter than `window_size` when `total_samples` is not an exact multiple.

### 5.4 LMLFOOT1 Footer

Present when `flags` bit 0 is set. An `LMLFOOT1`/`LFT2` seek table at EOF carrying **absolute** byte offsets (from the start of the file — note this base is the 40-byte BCS1 header, not the 32-byte LML1 header) for O(log n) random access. Additive: a reader that stops after the last payload decodes correctly without it.

---

## 6. Legacy Compatibility (decode-forever)

A BCS reader dispatches on the leading four bytes. The seven pre-BCS magics MUST remain decodable indefinitely via the frozen legacy path:

```
Magic       Meaning
--------    ------------------------------------------
LML1        LML v1 lossless container
LMO1        LMO (Optimum) container
LMA1/LMA2   LMA archive (v1 / v2)
LMQC        LMQC neural container
LMLCRYPT    encrypted LML container
```

Footer magics `LMLFOOT1` and `LFT2` are the legacy/current seek-table markers. A reader that sees `BCS1` at byte 0 uses the BCS1 parser (this document); any other leading four bytes route to the legacy decoder for that magic. The legacy `LML1` parser MUST reject a `BCS1` file cleanly (`InvalidMagic`), and the BCS1 parser MUST reject a legacy file cleanly — the two never cross-decode.

---

## 7. Versioning

`version_major`/`version_minor` (offsets 4–5) version the **header**, independent of the inner body's own version. A v1 reader:

- MUST decode `version_major = 1`.
- SHOULD reject `version_major > 1` with `UnsupportedVersion` (a future header layout).
- MUST ignore the reserved bytes (32..40) so a v1.x header extension that uses them stays readable by a v1.0 reader that does not understand the extension (subject to the extension being defined additively).

---

## 8. Conformance

A conforming BCS1 **encoder** MUST:
- Emit a 40-byte header with `magic = BCS1`, `version_major = 1`, `codec_descriptor = 0`, `decode_capability = 0`, reserved bytes zero.
- Emit a body byte-identical to the LML1 v1 container body for the same signal + metadata + codec mode.

A conforming BCS1 **decoder** MUST:
- Dispatch on the leading four bytes; decode `BCS1` per this document and the seven legacy magics per their specs.
- Fail closed (never mis-decode) on an unknown `codec_descriptor`, an unsupported `decode_capability`, an unsupported `version_major`, a truncated file, or invalid UTF-8 metadata.
- Reconstruct the exact digital sample values for `codec_descriptor = CODEC_LML_53` (lossless).

The reference conformance vectors live under `specs/conformance/`; the `ECS-Bench-v1` harness (`evaluation/openecs/`) runs a decoder against them.
