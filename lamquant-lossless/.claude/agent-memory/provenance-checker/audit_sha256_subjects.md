---
name: SHA-256 hash subjects diverge across pipeline paths
description: Critical gap — Rust LMA path stores empty signal_sha256; Python edf_to_lml and CLI hash signal bytes; hash subjects also differ between CLI and Python
type: project
---

Three paths produce different SHA-256 semantics:

1. Python edf_to_lml.py convert_edf_to_lml():
   - hash subject: signal_int.tobytes() (numpy row-major bytes of int64 signal)
   - stored as: metadata['signal_sha256']
   - verified: roundtrip decompress + recompute + compare → delete file on mismatch
   - STATUS: CORRECT and verified

2. Rust lml.rs CLI encode_one():
   - hash subject: channel-major iteration, each sample.to_le_bytes() (i64 LE)
   - stored as: "signal_sha256":"<hex>" in metadata JSON
   - verified: optional --verify or --cross-validate flags
   - STATUS: CORRECT but verify is NOT mandatory (opt-in flags)
   - NOTE: numpy .tobytes() on [C,T] int64 array is also row-major i64 LE → COMPATIBLE

3. Rust lma.rs encode_edf_to_lml():
   - hash stored as: "signal_sha256":""   (EMPTY STRING)
   - no verification step after encode
   - STATUS: CRITICAL GAP — this is the LMA archive path. Every EDF compressed
     through lml archive or lml extract goes through this path with no signal hash.

The SHA-256 in the LMA manifest (per entry) is the hash of the ORIGINAL EDF FILE bytes,
not the signal. That hash IS verified at extract time when --verify is passed. However
the EDF file hash is not recorded in the LML metadata blob, only in the archive manifest.
If the LML is extracted and used standalone, there is no file-level or signal-level hash.

**How to apply:** Before 510(k) submission, encode_edf_to_lml() in lma.rs must compute
signal SHA-256 and insert it into the metadata string (analogous to lml.rs CLI path).
The verify step (decompress + recheck hash) should also be mandatory in the LMA path,
not just in the CLI path.
