# Verification

> Integrity and provenance. How LamQuant proves every byte of every
> file survived encode → store → decode. For the encode-side
> `--cross-validate` and `--verify` flags that exercise this chain
> right after encode, see [Compression](./01-compression.md).

LML and LMA both ship layered integrity checks: CRC-32 per window,
SHA-256 per entry, archive-wide SHA-256, optional HMAC signature,
tamper-evident audit log. `lml verify` auto-dispatches between LML and
LMA based on the magic bytes so `lml verify foo.lma` and
`lml verify foo.lml` both just work.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| LML CRC-32 per window | `lml verify` (on `.lml`) | shipped | v1.0 | Single-step structural check |
| LMA archive-wide SHA-256 | `lml verify` / `lml verify-archive` | shipped | v1.0 | 32-byte trailer on every `.lma` |
| Per-entry SHA-256 | `lml verify` (on `.lma`) | shipped | v1.0 | Manifest carries one SHA per entry |
| Auditable explain mode | `lml verify --explain` | shipped | v1.2 | 5-section per-step readout |
| Magic-byte auto-dispatch | `lml verify` / `lml info` | shipped | v1.1 | First 4 bytes pick LML vs LMA path |
| LMA archive verify | `lml verify-archive` | shipped | v1.0 | LMA-specific verifier |
| Manifest schema check | `lml verify-manifest` | shipped | v1.0 | Existence + size + SHA-256 of every listed file |
| Roundtrip paranoid | `lml roundtrip` | shipped | v1.0 | 4-slice SHA comparison: signal, header, non_eeg, trailing |
| HMAC sign | `lml sign` | shipped | v1.0 | Detached 32-byte tag sidecar; see [Cryptography](./06-cryptography.md) |
| HMAC verify | `lml verify-signature` | shipped | v1.0 | See [Cryptography](./06-cryptography.md) |
| Audit log (SHA chain) | `lml audit-log append` / `verify` | shipped | v1.0 (Phase 7.6) | Tamper-evident JSONL |
| Conformance test vectors | `specs/conformance/` | shipped | v1.0 | 13 vectors (Group E) |
| Volume per-volume SHA-256 | `lml volume-split` (built in) | shipped | v1.2 (V) | Each volume independently SHA-checked at assemble time |

For HMAC and AES integrity (cryptographic auth tag), see
[Cryptography](./06-cryptography.md). For the SBOM that proves the
running binary matches a known source tree, see
[Build / Release](./10-build-release.md).

## Commands

### `lml verify`

Verify LML/LMA file integrity. Auto-dispatches on the file's magic
bytes:

- `LML1` → CRC-32 over every window
- `LMA1` → archive verifier (delegates to `lml verify-archive`)

So `lml verify foo.lma` works without thinking about the file type;
operators ran into the awkward mixed-directory case in v1.0 (`for f in
*.l??; do lml verify "$f"; done` failing for half the files) and the
magic-byte dispatch was added to fix it in v1.1.

Synopsis:
```
lml verify [-r] [--explain] <INPUT>
```

Examples:
```
# Single file (works for both .lml and .lma)
lml verify recording.lml
lml verify recording.lma

# Recursive sweep of a mixed directory
lml verify -r /backup/eeg/

# Auditable readout on an LMA
lml verify recording.lma --explain
```

### `lml verify --explain`

Auditable per-step verification chain on an LMA. Five sections:

1. **Size** — file length + magic header
2. **Archive SHA-256** — over content + footer, compared to trailer
3. **Manifest** — decompress + parse, byte counts, entry count
4. **Per-entry SHA-256** — hash each entry, compare to manifest
5. **Summary** — OK n/n entries, cumulative elapsed time

No-op on `.lml` inputs (the LML verify path is a single CRC-32 sweep,
nothing to enumerate).

Example:
```
lml verify recording.lma --explain
```

Output is human-readable and meant for the "no black box" contract —
operators should be able to reproduce the verification by hand from the
spec ([`../lml-format-v1.md`](../lml-format-v1.md)). For machine
parsing, prefer the compact default output (`OK n/n entries`) or
`--emit-json-events`.

### `lml verify-archive`

LMA-specific verifier. Same checks as `lml verify` on an `.lma` (and
the same `--explain` flag), but skips the magic-byte dispatch — useful
when you already know the file is an archive and want an explicit
error if it isn't.

Synopsis:
```
lml verify-archive [--explain] <INPUT>
```

### `lml verify-manifest`

Verify a `manifest.lml.json` produced alongside batch encodes. Checks
existence, size, and SHA-256 of every listed file.

Synopsis:
```
lml verify-manifest <MANIFEST>
```

### `lml roundtrip`

Paranoid full-roundtrip bit-exact verification on EDF/BDF files. For
each input file: encodes to a tempfile `.lml`, then compares FOUR
SHA-256 slices between original and recovered:

1. signal samples (channel data)
2. raw_header (full EDF header bytes)
3. non_eeg channel data (annotations + non-EEG signals)
4. trailing_data (partial-record bytes at file tail)

All four must match exactly. Failure modes are reported per-file as
structured JSON. Designed for clinical-grade verification where any
drift is unacceptable.

Synopsis:
```
lml roundtrip [-r] [-o <REPORT>] [--fail-fast] [-j <N>] <INPUT>
```

Examples:
```
# Single file, report to stdout
lml roundtrip recording.edf

# Recursive sweep, JSON report to disk
lml roundtrip -r /data/tueg/ -o roundtrip.json

# Bail early on a known-clean corpus
lml roundtrip -r /data/ --fail-fast
```

### `lml audit-log`

Append-only JSONL with a SHA-chain link between consecutive entries.
Tampering with any historical record breaks the chain.

Synopsis:
```
lml audit-log append --log <PATH> --op <OP> --msg <MSG>
lml audit-log verify --log <PATH>
```

Examples:
```
# Append an audit record
lml audit-log append --log audit.jsonl --op decrypt --msg "operator alice"

# Verify chain integrity
lml audit-log verify --log audit.jsonl
```

See [Cryptography](./06-cryptography.md) for the broader signing /
encryption surface.

## Integrity chain

How an LMA archive proves its integrity, from inside out:

```
       ┌─────────────────────────────────────────────┐
       │  per-window CRC-32 (inside each .lml entry) │ ← lml verify foo.lml
       ├─────────────────────────────────────────────┤
       │  per-entry SHA-256 (in archive manifest)    │ ← lml verify foo.lma
       ├─────────────────────────────────────────────┤
       │  archive-wide SHA-256 (32-byte trailer)     │ ← lml verify-archive
       ├─────────────────────────────────────────────┤
       │  optional HMAC-SHA-256 detached signature   │ ← lml verify-signature
       ├─────────────────────────────────────────────┤
       │  per-volume SHA-256 (volume-split byte-stream sets) │ ← lml volume-assemble
       └─────────────────────────────────────────────┘
```

Every layer is independently verifiable. The CRC-32 covers transient
corruption (bit flip on disk). The SHA-256 covers intentional
tampering (manifest substitution). The HMAC covers authenticated
provenance (tag + secret key). The audit-log SHA chain covers
operational history.

## Conformance test vectors

`specs/conformance/` ships 13 versioned test vectors (Group E) for
third-party LML readers. Each vector includes:

- Source bytes (synthetic or real EEG fixture)
- Expected `.lml` output (byte-exact)
- Per-window CRC-32 expectations
- Decode comparison SHA-256

CI's `conformance` job runs the suite on every PR. Documented in
`specs/conformance/README.md`. The Python verifier (`verify.py`) is
structural-only — the Rust crate is the authoritative decoder. A
clean-room pure-Python decoder is a deferred follow-up.

## Tamper-evident audit log (Phase 7.6)

`audit.jsonl` is append-only JSON-lines. Each line has:

```json
{
  "ts_ms": 1715000000000,
  "op": "decrypt",
  "msg": "operator alice",
  "prev_sha": "...",
  "sha": "<sha256(prev_sha || ts || op || msg)>"
}
```

`lml audit-log verify` walks the chain and reports the first index
where `sha` doesn't match the recomputed value. Useful for clinical
audits and chain-of-custody.

## Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--explain` | bool | false | Print per-step readout instead of summary (LMA verify only) |
| `-r`, `--recursive` | bool | false | Walk subdirectories |
| `--fail-fast` | bool | false | Abort batch on first mismatch (`roundtrip`) |
| `-o`, `--report <PATH>` | path | (stdout) | JSON report path (`roundtrip`) |
| `-j`, `--parallel <N>` | usize | 0 | Worker count for `roundtrip` (0 = rayon default) |

## Error cases

| Trigger | Error |
|---|---|
| LML CRC-32 mismatch | "CRC-32 mismatch at window N" |
| LMA archive-SHA mismatch | "archive sha256 mismatch" |
| Per-entry SHA mismatch | "entry <path>: sha256 mismatch" |
| Manifest decompress fails | "manifest: zstd decompress failed" |
| Manifest > 256 MB decompressed | "manifest exceeds MAX_MANIFEST_SIZE" (zstd-bomb guard) |
| Entry claims `original_size > 16 GiB` | "MAX_ENTRY_ORIGINAL_SIZE exceeded" |
| `roundtrip` non_eeg SHA mismatch | per-file JSON record with slice + bytes diff |
| `audit-log verify` chain break | reports the first index whose `sha` doesn't recompute |
| `verify-manifest` file missing | per-file FAIL record |
| Magic-byte dispatch sees unknown magic | "unknown file type: not LML1 or LMA1" |

## Related

- **Other buckets**:
  - [Compression](./01-compression.md) — `--verify` / `--cross-validate` on encode
  - [Decompression](./02-decompression.md) — extract-time SHA check via `--verify` default
  - [Cryptography](./06-cryptography.md) — HMAC sign / verify, AES auth tag
  - [Archive Ops](./04-archive-ops.md) — `volume-assemble` reuses the per-volume SHA
  - [Browse / Inspect](./05-browse-inspect.md) — `lml ls --long` exposes SHA-256 per entry
  - [Build / Release](./10-build-release.md) — conformance suite + fuzz harness
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:339` — `Verify` subcommand
  - `lamquant-core/src/bin/lml.rs:540` — `VerifyArchive` subcommand
  - `lamquant-core/src/bin/lml.rs:362` — `Roundtrip` subcommand
  - `lamquant-core/src/bin/lml.rs:792` — `AuditLog` subcommand
  - `lamquant-core/src/lma.rs:1620` — archive verifier entry
  - `lamquant-core/src/container.rs` — LML CRC-32 per window
  - `lamquant-core/src/security.rs` — `AuditLog::append` / `verify`
  - `specs/conformance/` — 13 versioned test vectors
- **Tests**:
  - `tests/integration/test_verify_explain.py`
  - `tests/integration/test_magic_byte_dispatch.py`
  - `tests/integration/test_roundtrip_*.py`
  - `tests/integration/test_audit_log.py`
- **Commits**:
  - `eebc580` — `lml verify --explain` (v1.2 X)
  - `6a3fd43` — magic-byte auto-dispatch for `info` / `verify` (v1.1 P.2)
  - `661d7bc` — publishable conformance suite (E)
- **Cross-cutting docs**:
  - [`../lml-format-v1.md`](../lml-format-v1.md) — frozen wire format with integrity field layout
  - [`../FAQ.md`](../FAQ.md) — "why does `roundtrip` fail on non_eeg?"
