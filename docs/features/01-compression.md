# Compression

> Everything that controls how bytes go INTO an archive. For taking
> bytes back OUT, see [Decompression](./02-decompression.md).

LamQuant's compression entry point is `lml encode`. The default mode
emits **one `.lma` archive per recording** that bundles the LML-encoded
signal, the original source file, and every stem-matched sibling
sidecar. No byte is silently dropped — bare `.lml` output exists but
requires explicit opt-in.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Default per-recording `.lma` | `lml encode` (no flag) | shipped | v1.0 | LML signal + source bytes + every sibling in one archive |
| Legacy corpus-wide `.lma` | `--lma` | shipped | v1.0 | Packs every output into one big archive at `-o` |
| Bare `.lml` opt-out | `--no-bundle` / `--bare-lml` / `--mirror` | shipped | v1.0 | Loud 20-line stderr warning per invocation |
| Silence data-loss warning | `--i-understand-data-loss` | shipped | v1.0 | Required co-flag for `--no-bundle` |
| Per-file cascade | (automatic) | shipped | v1.0 | LML → zstd-9 → store per extension via `choose_method` |
| Include glob filter | `--include` | shipped | v1.2 | Multiple `--include` flags; tar/gitignore-style |
| Exclude glob filter | `--exclude` | shipped | v1.2 | Per-file stderr notice; pair with `--i-understand-data-loss` |
| LPC mode | `--lpc-mode` | shipped | v1.2 | `adaptive` / `fixed` / `anytime` (default) |
| Noise-bit stripping | `--noise-bits` | shipped | v1.0 | 0 = lossless (default); strips N LSBs |
| Compression window size | `--window-size` | shipped | v1.0 | Default 2500 samples (10 s @ 250 Hz ref) |
| Recursion | `-r` / `--recursive` | shipped | v1.0 | Walks subdirectories |
| Skip-existing | `--skip-existing` | shipped | v1.0 | Resume interrupted batch runs |
| Sibling-file detection | (automatic) | shipped | v1.0 | `find_sidecars` collects stem-prefix-matched siblings |
| Dry-run | `--dry-run` | shipped | v1.0 | Estimate sizes/time without writing |
| Cross-validate | `--cross-validate` | shipped | v1.0 | Decode output and compare SHA-256 against source |
| Verify after encode | `--verify` | shipped | v1.0 | Roundtrip check |
| Fail-fast batch | `--fail-fast` | shipped | v1.0 | Abort on first failure |
| Continue-on-error | `--continue-on-error` | shipped | v1.0 | Explicit alias for default behavior |
| Parallelism | `-j` / `--threads` | shipped | v1.0 | 0 = auto |

For external byte-stream volume splitting (cloud / email / removable
media), see [Archive Ops](./04-archive-ops.md). Volume splitting
happens *after* encode — it is not an `encode` flag.

## Commands

### `lml encode`

Encode EDF/BDF/BrainVision/CNT/DICOM/EEGLAB/raw to per-recording
`.lma`. Default mode produces one `.lma` per input recording that
bundles the LML-compressed signal plus the original source bytes plus
every sibling annotation file via the LML → zstd → store cascade. No
input byte can be silently dropped.

Synopsis:
```
lml encode [OPTIONS] <INPUT> [-o <OUTPUT>]
```

Examples:
```
# Default: per-recording .lma with full sibling preservation
lml encode recording.edf -o out/recording.lma

# Recursive batch encode of a corpus
lml encode -r /data/tueg/ -o /backup/tueg/

# Bare .lml output (with loud data-loss warning)
lml encode recording.edf -o out/ --no-bundle --i-understand-data-loss

# Cross-validate: encode then decode-compare against source
lml encode recording.edf -o out/recording.lma --verify --cross-validate

# Include/exclude filters
lml encode -r /data/ -o out/ --include '*.edf' --exclude '**/qc/*.edf' \
  --i-understand-data-loss

# Lossy: strip 2 LSBs of ADC noise
lml encode recording.edf -o out/ --noise-bits 2
```

## Output modes

| Mode | Trigger | What you get |
|---|---|---|
| **Default per-recording `.lma`** | `lml encode foo.edf` | One `.lma` archive containing `foo.lml` + the original source bytes + every stem-matched sibling. Travels as one unit; impossible to leave sidecars behind. |
| **Corpus-wide `.lma`** | `--lma` | Legacy: pack every encode output into one big `.lma` at the `-o` path. Mutually exclusive with `--no-bundle`. |
| **Bare LML + mirror sidecars** | `--no-bundle` / `--bare-lml` / `--mirror` | Bare `.lml` per recording PLUS a mirror copy of every sibling in the output directory. Sidecars preserved but not compressed; the operator is responsible for keeping them next to the `.lml` when moving files. Prints loud data-loss warning unless paired with `--i-understand-data-loss`. |

## Per-file cascade

For every file collected into an archive, the encoder picks a method
from `choose_method(path)` (see `lamquant-core/src/lma.rs:138`):

| File extension | Method | Rationale |
|---|---|---|
| `.edf`, `.bdf` | `Method::Lml` | Domain-specific lossless codec (~2.3:1 CR on clinical EEG) |
| `.lml`, `.lmq`, `.lma`, `.gz`, `.zst`, `.zip`, `.7z`, `.png`, `.jpg`, `.jpeg`, `.mp4`, `.avi` | `Method::Store` | Already compressed; recompressing would only waste cycles |
| everything else | `Method::Zstd` | Sidecar text (`.tse`, `.csv_bi`, `.lbl_bi`, `_summary.txt`, `.json`) compresses well at zstd-9 |

Corrupted / incompressible files fall through to store as a last resort.
The cascade decision is policy, not a fallback (see `feedback_lma_zstd_policy`):
zstd compression of non-EDF entries is the intended behavior, not a
silent codec failure.

## Supported source formats

| Format | Extensions | Notes |
|---|---|---|
| EDF / EDF+C / EDF+D | `.edf` | Foundation; TAL annotations preserved |
| BDF / BDF+ (24-bit) | `.bdf` | TAL in non-EEG channel slots |
| BrainVision | `.vhdr` + `.vmrk` + `.eeg` | All three files preserved; filename anchors from `DataFile=` / `MarkerFile=` |
| NeuroScan CNT | `.cnt` | 900-byte SETUP + 75-byte ELECTLOC × N + int16 multiplexed |
| EEGLAB | `.set` + `.fdt` | Lossless f32→i64 bit-cast default; opt-in `--lossy-int16`; full MAT v5 struct preserved |
| DICOM Waveform | `.dcm` | Behind `--features dicom`; 12-Lead + General ECG SOP classes |
| Custom raw + sidecar | `.raw` + `.json` | Any dtype / orientation / channel layout / phys range |

Every reader stashes the original source bytes as `SidecarBlob`s; the
encoder writes them as byte-exact preservation copies alongside the
`.lml`. See `tests/integration/test_full_parity_*.py` for the per-format
contract.

## Sibling-file detection

`find_sidecars(edf_path)` (in `lamquant-core/src/bin/lml.rs:1065`)
collects every file in the same directory whose basename starts with
the EDF's stem followed by `.` or `_`. Stem-collision disambiguation:
if `recording.edf` and `recording_extra.edf` both exist, files matching
the longer stem belong to that other EDF and are not misattributed.

Dropped sidecars in the default mode → loud stderr warning. Dropped
sidecars in `--no-bundle` mode → mirrored next to the `.lml`. Either way,
the encoder never silently drops a sibling.

## Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output <PATH>` | path | (derived) | Output `.lma` / `.lml` file or directory |
| `--verify` | bool | false | Decode + compare immediately after encode |
| `--cross-validate` | bool | false | Decode output and compare SHA-256 against original |
| `--noise-bits <N>` | u8 | 0 | Strip N LSBs (0 = lossless) |
| `--window-size <N>` | usize | 2500 | Samples per compression window |
| `-j`, `--threads <N>` | usize | 0 | Parallel worker count (0 = auto = CPU count) |
| `-r`, `--recursive` | bool | false | Walk subdirectories |
| `--skip-existing` | bool | false | Skip files whose output already exists |
| `--include <GLOB>` | repeatable | (all) | Tar-style include filter (`*.edf`, `**/sub-*/eeg/*`) |
| `--exclude <GLOB>` | repeatable | (none) | Exclude filter applied after `--include`; per-file stderr notice |
| `--dry-run` | bool | false | Estimate sizes/time without writing |
| `--lma` | bool | false | Legacy corpus-wide single archive |
| `--no-bundle` / `--bare-lml` / `--mirror` | bool | false | Bare `.lml` + mirror sidecars (data-loss warning) |
| `--i-understand-data-loss` | bool | false | Silence the `--no-bundle` warning |
| `--lpc-mode <MODE>` | enum | `anytime` | `adaptive` / `fixed` / `anytime` |
| `--fail-fast` | bool | false | Abort batch on first failure |
| `--continue-on-error` | bool | true (implicit) | Explicit no-op alias for the default behavior |

## Wire format (LMA1 envelope)

Per-recording `.lma` archive layout (see `lamquant-core/src/lma.rs:1-16`):

```
[4 bytes]   Magic: b"LMA1"
[4 bytes]   Version: u32 LE (1)
[4 bytes]   Number of entries: u32 LE
[4 bytes]   Manifest length: u32 LE (after zstd compression)
[variable]  Manifest: zstd-compressed JSON
[variable]  Entry payloads (concatenated)
[32 bytes]  Archive SHA-256 (of everything before this)
```

The manifest records, per entry: path, `original_size`, `compressed_size`,
method (`lml` / `secondary` / `store`), per-entry SHA-256, offset, mtime,
sub-second mtime nanos, Unix mode. See [Verification](./03-verification.md)
for the integrity chain and [Browse / Inspect](./05-browse-inspect.md)
for the read-side surface.

Caps: `MAX_ENTRY_ORIGINAL_SIZE = 16 GiB`, `MAX_ENTRY_DECOMPRESS_SIZE = 4 GiB`,
`MAX_MANIFEST_SIZE = 256 MB` (decompressed). All three are zstd-bomb guards.

## Error cases

| Trigger | Behavior |
|---|---|
| `--no-bundle` without `--i-understand-data-loss` | 20-line stderr warning paragraph |
| `--lma` with `--no-bundle` | Clap conflict; refuses to parse |
| `--include` pattern matches zero files | Refusal — empty output is a footgun |
| Sidecar scan directory unreadable | `Warning: sidecar scan could not read directory ...` on stderr; bundle proceeds without those siblings |
| `--exclude` drops a file | One-line stderr notice per file (loud) |
| Source format reader fails mid-batch | Per-file failure logged; batch continues unless `--fail-fast` |
| Output path exists | `encode` silently overwrites (legacy behavior); other writers refuse without `--force` |

## Related

- **Other buckets**:
  - [Decompression](./02-decompression.md) — `lml decode` / `lml extract` reverse path
  - [Verification](./03-verification.md) — `--verify` and `--cross-validate` integrity chain
  - [Archive Ops](./04-archive-ops.md) — `volume-split` / `append` / `recompress` lifecycle
  - [Browse / Inspect](./05-browse-inspect.md) — read the archive you just made
  - [CLI UX](./11-cli-ux.md) — `--quiet` / `-v` / `--color` / `--emit-json-events`
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:192` — `Encode` subcommand definition
  - `lamquant-core/src/bin/lml.rs:1065` — `find_sidecars`
  - `lamquant-core/src/lma.rs:138` — `choose_method`
  - `lamquant-core/src/lml.rs` — `compress_into<W: Write>` core codec
- **Tests**:
  - `tests/integration/test_full_parity_*.py` — per-format byte-exact contract
  - `tests/integration/test_sidecar_preservation.py` — `find_sidecars`
  - `tests/integration/test_data_loss_footguns.py` — `--no-bundle` warning + `--i-understand-data-loss`
- **Commits**:
  - `fbd8155` — `--include` / `--exclude` (v1.2)
  - `834a6c2` — data-loss warning + `--i-understand-data-loss` co-flag
  - `8179ae3` — universal per-file cascade through `pack_archive`
- **Cross-cutting docs**:
  - [`../FEATURES.md`](../FEATURES.md) §1 (core lossless codec)
  - [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) (auto-generated full flag listing)
  - [`../lml-format-v1.md`](../lml-format-v1.md) (frozen wire-format spec)
