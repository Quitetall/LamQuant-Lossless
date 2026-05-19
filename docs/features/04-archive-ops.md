# Archive Ops

> Lifecycle commands that mutate an existing `.lma` or `.lml`:
> split, concat, append, recompress, strip-pii, set-metadata, and the
> external byte-stream volume splitter. For create-side `lml encode`,
> see [Compression](./01-compression.md). For read-side inspection
> without mutation, see [Browse / Inspect](./05-browse-inspect.md).

These commands all share two safety properties:

1. **Atomic via WAL** — `<archive>.lma.new` tempfile + rename + fsync,
   so an interrupted operation leaves either the old archive intact or
   the new archive complete. Never a half-updated file on disk.
2. **Idempotent where possible** — `append` is a no-op if the same SHA-256
   is already in the archive at the same path.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Append entry | `lml append` | shipped | v1.0 (3.8) | WAL + atomic; `.lma.bak` retained unless `--no-bak` |
| Strip PII | `lml strip-pii` | shipped | v1.0 (3.9) | Whitelist default; `--in-place` atomic |
| Recompress | `lml recompress` | shipped | v1.0 (3.10) | New `--noise-bits` / `--window-size` / `--lpc-mode` |
| Set metadata | `lml set-metadata` | shipped | v1.0 (3.11) | `--set` / `--remove` / `--sidecar`, atomic `--in-place` |
| Split LML | `lml split` | shipped | v1.0 (3.5) | N equal-sample chunks; requires LMLFOOT1 seek table |
| Concat LMLs | `lml concat` | shipped | v1.0 (3.6) | Order-invariant if split markers present; SHA-equal |
| Archive directory | `lml archive` | shipped | v1.0 | Bundle a directory into `.lma` (alt to `encode`) |
| Volume-split byte-stream | `lml volume-split` | shipped | v1.2 (V) | `<archive>.001`, `.002`, …; up to 999 volumes |
| Volume-assemble byte-stream | `lml volume-assemble` | shipped | v1.2 (V) | Auto-detect `<base>.NNN` glob; gap detection |
| Recover damaged LML | `lml recover` | shipped | v1.0 | Salvage valid windows from corruption |
| List archive (deprecated) | `lml list-archive` | shipped | v1.0 | One-line stderr deprecation; use `lml ls --tree` |

## Commands

### `lml split`

Split a single LML into N equal-sample-count chunks. Each chunk is
re-encoded with the source's sample rate, window size, and a copy of
the source metadata; the metadata gains `split_chunk_idx` and
`split_n_chunks` so a future `lml concat` detects provenance and
preserves ordering. Last chunk absorbs any remainder.

Requires the source carry an LMLFOOT1 seek table (legacy `.lml`
without footer errors explicitly).

Synopsis:
```
lml split <INPUT> -o <DIR> [--chunks N] [--force]
```

Example:
```
# Split into 4 chunks
lml split recording.lml -o chunks/ --chunks 4
# Produces: chunks/recording.part-01-of-04.lml … part-04-of-04.lml
```

### `lml concat`

Concatenate sibling LMLs (same `n_channels`, `sample_rate`,
`window_size`) into one LML. If every input carries the
`split_chunk_idx` + `split_n_chunks` metadata, inputs are sorted by
chunk index and validated for completeness (0..N-1, no gaps or dups).
Otherwise inputs are concatenated in lexicographic filename order.

Bible R32: byte-identical output regardless of argv order.

Synopsis:
```
lml concat <INPUT...> -o <OUTPUT> [--force]
```

Example:
```
# Order-invariant; chunk markers drive ordering
lml concat chunks/*.lml -o combined.lml
```

### `lml append`

Append a file to an existing LMA archive without rewriting the
pre-existing payload bytes. Same-directory tempfile (WAL) + atomic
rename + fsync, so the old archive is either fully replaced or fully
retained — never half-updated. Old archive retained at `<archive>.lma.bak`
unless `--no-bak`.

EDF/BDF goes through LML compression; other types go through zstd;
already-compressed types fall through to store (same `choose_method`
cascade as `encode`).

**Idempotent** on identical SHA-256 (no-op). Errors on duplicate path
with different content unless `--force`.

Synopsis:
```
lml append <ARCHIVE> <FILE> [--as <PATH>] [--zstd-level N] [--force] [--no-bak]
```

Examples:
```
# Add a new EDF
lml append recording.lma extra.edf

# Pin the in-archive path
lml append recording.lma notes.txt --as docs/notes.txt

# Same SHA → no-op (idempotent)
lml append recording.lma extra.edf   # second time is fine

# Different content at same path → refuse unless --force
lml append recording.lma extra.edf --force
```

### `lml strip-pii`

Strip patient PII from an LML. Masks the EDF header's `patient_id`
(bytes 8..88) and `recording_id` (bytes 88..168) with spaces. By
default also masks `start_date` and `start_time`; pass `--keep-dates`
to retain.

Re-encodes the container against the original signal so byte-exact
decompression still works (signal SHA-256 is invariant).

Default output is to a separate file. `--in-place` atomically swaps
via same-dir tempfile + rename + fsync.

Synopsis:
```
lml strip-pii <INPUT> [-o <OUTPUT> | --in-place] [--keep-dates] [--force]
```

Examples:
```
# Default: write masked copy
lml strip-pii recording.lml -o redacted.lml

# In-place atomic swap, keep dates
lml strip-pii recording.lml --in-place --keep-dates
```

### `lml recompress`

Recompress an LML with new compression parameters. Decodes the signal
from the source and re-encodes against the same metadata using new
`--noise-bits` / `--window-size` / `--lpc-mode`. Combined with
`--noise-bits > 0` this is the recommended path to convert an existing
lossless LML into a lossy one without re-running the EDF reader.

Synopsis:
```
lml recompress <INPUT> [-o <OUTPUT> | --in-place]
                       [--noise-bits N] [--window-size N] [--lpc-mode MODE]
                       [--force]
```

Examples:
```
# Strip 2 LSBs of ADC noise from an existing lossless LML
lml recompress recording.lml -o lossy.lml --noise-bits 2

# In-place re-encode with the adaptive LPC mode
lml recompress recording.lml --in-place --lpc-mode adaptive
```

### `lml set-metadata`

Update an LML's metadata JSON via key/value edits and/or a sidecar
JSON file. The signal payload is never touched; only the metadata
blob in the container header is rewritten.

Edit order: sidecar overlay (top-level merge) → `--set` values →
`--remove` keys. `--set` values are parsed as JSON literals where
syntactically valid (numbers, true/false/null, "...", [...], {...});
otherwise stored as strings verbatim.

Synopsis:
```
lml set-metadata <INPUT> [-o <OUTPUT> | --in-place]
                          [--sidecar <PATH>]
                          [--set KEY=VALUE]... [--remove KEY]...
                          [--force]
```

Examples:
```
# Set a single key
lml set-metadata recording.lml -o updated.lml --set operator=alice

# Merge a sidecar JSON and remove a key
lml set-metadata recording.lml --in-place \
  --sidecar extra.json --remove legacy_field

# Set a JSON literal (parsed as an integer)
lml set-metadata recording.lml --in-place --set retries=3
```

### `lml volume-split`

Split an `.lma` archive into fixed-size volumes for cloud / email /
removable-media uploads. Produces `<archive>.001`, `<archive>.002`,
... each ≤ `--size` bytes. **Pure byte-stream split** — no LMA
wire-format change, no manifest mutation. The volumes are an external
container; reassembling them gives the original `.lma` byte-identical.

Up to 999 volumes (3-digit suffix). Deletes the original `.lma` by
default; pass `--keep` to retain it alongside.

Synopsis:
```
lml volume-split <INPUT> --size <SIZE> [--keep] [--force]
```

Examples:
```
# 100 MB volumes
lml volume-split recording.lma --size 100M

# Keep the original
lml volume-split recording.lma --size 100M --keep

# 4 GiB chunks (for DVD-R / cloud size limits)
lml volume-split big.lma --size 4G
```

Size suffix: `K` / `M` / `G` (1024-based). Bare numbers = bytes.

### `lml volume-assemble`

Reassemble volume files into a single `.lma`. Auto-detects the volume
set by globbing `<base>.NNN` for sequential 3-digit suffixes. The
volume set must be complete — gaps in the numbering → typed error.

Multi-volume archives can also be **read transparently** without
explicit assembly: the LMA reader auto-resolves a single archive name
to `<parent_dir>/<stem>.lma.[0-9][0-9][0-9]` if no monolithic
`.lma` exists. Tools like `lml ls foo.lma` work directly on the
volume set.

Synopsis:
```
lml volume-assemble <ANY_VOLUME> -o <OUTPUT> [--force]
```

Example:
```
# Reassemble all foo.lma.* into foo.lma
lml volume-assemble foo.lma.001 -o foo.lma

# Force overwrite an existing target
lml volume-assemble foo.lma.001 -o foo.lma --force
```

### `lml archive`

Bundle a directory tree into a single `.lma`. Alternative entry point
to `lml encode --lma` for the case where you've already organised
files on disk and don't need the encoder to choose per-recording
boundaries.

Synopsis:
```
lml archive <INPUT_DIR> [-o <OUTPUT>] [--zstd-level N]
```

Example:
```
lml archive /data/study-7/ -o study-7.lma --zstd-level 12
```

### `lml recover`

Best-effort recovery of valid windows from a damaged LML. Walks the
file, validates each window's CRC-32, writes the surviving windows to
a new LML.

Synopsis:
```
lml recover <INPUT> <OUTPUT>
```

## Flags

| Flag | Type | Default | Subcommand(s) | Description |
|---|---|---|---|---|
| `--force` | bool | false | most | Overwrite existing output |
| `--in-place` | bool | false | `recompress` / `set-metadata` / `strip-pii` | Atomic swap via same-dir tempfile + rename |
| `--chunks <N>` | u32 | 2 | `split` | Output chunk count (≥ 2) |
| `--as <PATH>` | string | (basename) | `append` | Entry path inside the archive |
| `--zstd-level <N>` | i32 | 9 | `archive` / `append` | Zstd level (1–22) for non-LML entries |
| `--no-bak` | bool | false | `append` | Discard `.lma.bak` after rename |
| `--noise-bits <N>` | u8 | 0 | `recompress` | Strip N LSBs |
| `--window-size <N>` | usize | 2500 | `recompress` | Samples per compression window |
| `--lpc-mode <MODE>` | enum | `fixed` | `recompress` | `fixed` / `adaptive` / `anytime` |
| `--sidecar <PATH>` | path | (none) | `set-metadata` | JSON merged top-level into metadata |
| `--set KEY=VALUE` | repeatable | — | `set-metadata` | Set one or more keys |
| `--remove KEY` | repeatable | — | `set-metadata` | Remove one or more top-level keys |
| `--keep-dates` | bool | false | `strip-pii` | Skip `start_date` / `start_time` masking |
| `--size <SIZE>` | string | (required) | `volume-split` | Volume size (`100M`, `4G`, ...) |
| `--keep` | bool | false | `volume-split` | Keep original `.lma` alongside volumes |

## Wire-format / on-disk shape

`append` writes `<archive>.lma.new`, fsyncs, renames into place,
fsyncs the parent directory. Atomic by POSIX rename semantics on local
filesystems. Old archive at `<archive>.lma.bak` survives unless
`--no-bak`.

`volume-split` is a pure byte-stream split: concatenating the volumes
back together is byte-equal to the source `.lma`. Each volume is a
prefix of the next, no per-volume header. Volume integrity comes from
the archive-wide SHA-256 (preserved across the split) — assembling all
volumes and re-checking the trailer SHA confirms the set is complete
and untampered.

The multi-volume *reader* (used by `lml ls`, `lml extract`, etc.)
recognises `<stem>.lma.001` and stitches the volumes on the fly via the
auto-resolve glob `<parent_dir>/<stem>.lma.[0-9][0-9][0-9]`. Bare
`lml ls foo.lma` works whether `foo.lma` is monolithic or a volume
set.

## Error cases

| Trigger | Error |
|---|---|
| `split` on a `.lml` without LMLFOOT1 footer | "split requires LMLFOOT1 seek table" |
| `concat` mismatched channel count / sample rate | "incompatible LML inputs" |
| `concat` with markers showing a gap | "split sequence has gap at chunk_idx N" |
| `append` duplicate path, different SHA, no `--force` | "entry already in archive with different content" |
| `recompress --in-place` and `-o` both set | refuses |
| `strip-pii --in-place` and `-o` both set | refuses |
| `set-metadata --set KEY` with no `=` | "missing `=` in --set" |
| `volume-split --size 0` or unparseable size | "invalid size" |
| `volume-assemble` set with missing volume | "missing volume <stem>.lma.NNN" |
| `volume-split` target file exists | refuses without `--force` |

## Related

- **Other buckets**:
  - [Compression](./01-compression.md) — `lml encode --lma`; the create-side path
  - [Decompression](./02-decompression.md) — `lml extract` reads what these commands produce
  - [Verification](./03-verification.md) — per-volume + archive-wide SHA-256
  - [Browse / Inspect](./05-browse-inspect.md) — see what's inside before / after mutation
  - [Cryptography](./06-cryptography.md) — sign-after-mutate workflows
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:613` — `Split`
  - `lamquant-core/src/bin/lml.rs:634` — `Concat`
  - `lamquant-core/src/bin/lml.rs:913` — `Append` (WAL + atomic)
  - `lamquant-core/src/bin/lml.rs:888` — `StripPii`
  - `lamquant-core/src/bin/lml.rs:821` — `Recompress`
  - `lamquant-core/src/bin/lml.rs:854` — `SetMetadata`
  - `lamquant-core/src/bin/lml.rs:489` — `VolumeSplit`
  - `lamquant-core/src/bin/lml.rs:514` — `VolumeAssemble`
- **Tests**:
  - `tests/integration/test_split_concat.py`
  - `tests/integration/test_append_idempotent.py`
  - `tests/integration/test_strip_pii.py`
  - `tests/integration/test_recompress.py`
  - `tests/integration/test_set_metadata.py`
  - `tests/integration/test_volume_split.py`
- **Commits**:
  - `81a46aa` — `lml split` (3.5)
  - `e27bb71` — `lml concat` (3.6)
  - `8f81dd3` — `lml append` WAL + atomic (3.8)
  - `4907fdd` — `lml strip-pii` (3.9)
  - `2900263` — `lml recompress` (3.10)
  - `228e4d6` — `lml set-metadata` (3.11)
  - `046d393` — `volume-split` / `volume-assemble` (v1.2 V)
- **Cross-cutting docs**:
  - [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) — auto-generated full flag listing
  - [`../lml-format-v1.md`](../lml-format-v1.md) — wire format invariants these commands preserve
