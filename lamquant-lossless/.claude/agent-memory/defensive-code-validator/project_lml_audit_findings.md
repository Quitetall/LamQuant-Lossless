---
name: LML codec audit critical findings
description: Critical and high findings from FDA 510(k) defensive audit of lossless LML codec (lml.rs, golomb.rs, lpc.rs, lifting.rs, container.rs, edf.rs)
type: project
---

First comprehensive defensive audit conducted 2026-04-24.

Key findings:
- CRITICAL: lml.rs sample count T stored as u16, silent truncation at 65536+ samples
- CRITICAL: golomb.rs n_total stored as u16, silently wraps at 65536+ coefficients
- CRITICAL: edf.rs uses unsafe from_raw_parts with potential alignment violation on odd-aligned data
- CRITICAL: edf.rs uses unsafe str::from_utf8_unchecked on untrusted EDF header bytes
- HIGH: lml.rs CRC-32 does NOT protect header fields, only payload
- HIGH: Python bias_cancel uses // (floor division for negatives), Rust uses floor_div — verified match but ctx_len=0 panics
- HIGH: container.rs window index u32 offset can overflow for large files
- HIGH: zigzag_encode(i64::MIN) causes signed overflow in Rust debug mode
- HIGH: golomb.rs compute_k sum can overflow u64 with large coefficient sets
- No BDF int24 support despite format field advertising "BDF"

**Why:** FDA 510(k) clinical-grade codec — every bug is potential patient data loss.
**How to apply:** Reference these findings when reviewing any codec changes or cross-language parity work.
