---
name: LML/LMA pipeline provenance audit - first run
description: Full chain-of-custody audit of EDF→LML→LMA pipeline covering SHA-256, CRC-32, metadata preservation, and silent-drop risks
type: project
---

First full provenance audit conducted 2026-04-24 targeting FDA 510(k) readiness.

Files audited:
- /mnt/4tb/LamQuant/lamquant-core/src/edf.rs
- /mnt/4tb/LamQuant/lamquant_codec/edf_to_lml.py
- /mnt/4tb/LamQuant/lamquant-core/src/lma.rs
- /mnt/4tb/LamQuant/lamquant-core/src/bin/lml.rs
- /mnt/4tb/LamQuant/lamquant-core/src/container.rs
- /mnt/4tb/LamQuant/lamquant-core/src/lml.rs
- /mnt/4tb/LamQuant/lamquant-core/src/crc32.rs
- /mnt/4tb/LamQuant/lamquant_codec/lossless.py
- /mnt/4tb/LamQuant/lamquant_codec/lma.py

**Why:** 510(k) submission requires airtight chain of custody from original EDF through archive and back. Any silent data drop is a regulatory disqualification.

**How to apply:** Use this as baseline when re-auditing after changes. Focus on the gaps identified below before any regulatory submission.

Key findings:
1. CRITICAL: Rust lma.rs encode_edf_to_lml writes signal_sha256:"" — no hash stored for LML entries
2. HIGH: Rust edf.rs has unaligned unsafe transmute of raw EDF data bytes to i16 slice
3. HIGH: mode_ns filtering silently drops non-mode-frequency EEG channels (documented but not flagged to user)
4. HIGH: Python read_edf_digital drops trailing partial record bytes WITHIN the data read; only partial truncation captured
5. MEDIUM: Rust EDF reader misses trailing_data field entirely — no capture of post-last-record bytes
6. MEDIUM: Rust encode path missing 9 metadata fields present in Python path
7. MEDIUM: BDF int24 sign extension in decode_lml_to_edf uses wrong formula for negative values
8. LOW: CRC-32 computed on concatenated lpc_meta+payload in Rust vs separate in Python (compatible but verify)
9. LOW: Rust container.rs window index is not validated on read (skipped, trusting the payload length prefix)
