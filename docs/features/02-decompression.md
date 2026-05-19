# Decompression

> Everything that takes bytes OUT of an `.lml` / `.lma`. For the encode
> side, see [Compression](./01-compression.md). For integrity and SHA
> checks during extract, see [Verification](./03-verification.md).

The decode surface is three subcommands: `lml decode` (LML → raw int32
or reconstructed EDF), `lml extract` (full `.lma` → directory), and
`lml extract-entry` (single archive entry). Partial-decode flags
(`--channels`, `--time-range`) let you avoid materialising data you
don't need.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Decode to raw int32 LE | `lml decode` | shipped | v1.0 | Channel-major int32 LE binary; default output |
| Reconstruct byte-exact EDF | `lml decode --to-edf` | shipped | v1.0 | Required for `lml roundtrip` paranoid check |
| Extract whole archive | `lml extract` | shipped | v1.0 | All entries → directory; mtime + mode preserved |
| Extract single entry | `lml extract-entry` | shipped | v1.0 (3.7) | Exact path or suffix match; SHA-verified |
| Partial channel decode | `--channels 0,2,4` | shipped | v1.0 (3.2) | Requires LMLFOOT1 seek table |
| Partial time decode | `--time-range START:END` | shipped | v1.0 (3.3) | Inclusive-start, exclusive-end sample index |
| Streaming decode | (automatic) | shipped | v1.0 (3.4) | Window-by-window; no whole-file load |
| Cross-format reverse path | `lml extract` | shipped | v1.0 | Recovers byte-equal `.edf` / `.bv` / `.cnt` / `.set` |
| Strict decode | (default) | shipped | v1.0 | `lma::read_entry_decoded` errors on codec failure |
| Best-effort fallback | `lmafs --allow-raw-fallback` | shipped | v1.4 | FUSE-only; raw LML bytes when codec refuses |
| SHA-256 extraction verify | (default on) | shipped | v1.0 | `--no-verify` opts out on trusted archives |
| Path-traversal protection | (automatic) | shipped | v1.0 | Rejects `..`, absolute, Windows device names, NUL |
| Stdin signal piping | `lml decode -` | shipped | v1.0 (1.5b) | Refuses TTY |
| Stdout signal piping | `-o -` | shipped | v1.0 (1.5b) | Refuses TTY / `--to-edf` / batch |
| Recursion | `-r` / `--recursive` | shipped | v1.0 | Walks subdirectories |
| Skip-existing | `--skip-existing` | shipped | v1.0 | Resume interrupted runs |
| Fail-fast batch | `--fail-fast` | shipped | v1.0 | Abort on first failure |
| Continue-on-error | `--continue-on-error` | shipped | v1.0 | Default; explicit alias |

## Commands

### `lml decode`

Decode an `.lml` file to raw int32 LE signal samples (channel-major,
sort+dedup'd in channel order). With `--to-edf` reconstructs the
full byte-identical EDF/BDF file (header + data records + trailing
bytes). Required by `lml roundtrip` for the paranoid bit-exact check.

Synopsis:
```
lml decode [OPTIONS] <INPUT> [-o <OUTPUT>]
```

Examples:
```
# Raw int32 LE to file
lml decode recording.lml -o samples.bin

# Reconstruct byte-identical EDF
lml decode recording.lml -o recording.edf --to-edf

# Partial decode: only channels 0, 2, 4 (sort+dedup'd)
lml decode recording.lml --channels 0,2,4 -o ch024.bin

# Partial decode: first 5000 samples
lml decode recording.lml --time-range 0:5000 -o head.bin

# Stream to stdout (refuses TTY)
lml decode recording.lml -o - | xxd | head

# Stream from stdin
cat recording.lml | lml decode - -o samples.bin
```

### `lml extract`

Extract an `.lma` archive into a directory. Recovers every byte of
every entry: signal `.lml`, original source-format bytes (`.edf` /
`.vhdr+.vmrk+.eeg` / `.dcm` / `.set+.fdt` / `.cnt` / `.raw+sidecar`),
and every sibling annotation. SHA-256 verified per entry by default.

Synopsis:
```
lml extract [OPTIONS] <INPUT> [-o <OUTPUT_DIR>]
```

Examples:
```
# Standard extract
lml extract recording.lma -o restored/

# Skip the per-entry SHA-256 check (faster, only for trusted archives)
lml extract recording.lma -o restored/ --no-verify
```

mtime + Unix mode are preserved on extracted entries.
Path-traversal protection rejects `..` / absolute / Windows device-name
entries before writing.

### `lml extract-entry`

Pull a single entry out of an `.lma` archive without unpacking the
rest. Entry is matched by exact path first, then by suffix (passing
the bare basename works when the manifest path is `chb06/chb06_01.edf`).
LML entries are reconstructed to the original EDF/BDF; zstd entries
are decompressed; store entries are copied verbatim. SHA-256 of the
reconstructed payload is verified against the manifest.

Synopsis:
```
lml extract-entry <ARCHIVE> <ENTRY> -o <OUTPUT>
```

Examples:
```
# Pull the original EDF back from the archive
lml extract-entry recording.lma recording.edf -o recording.edf

# Suffix match works
lml extract-entry recording.lma chb06_01.edf -o out.edf

# Pull a sidecar annotation
lml extract-entry recording.lma recording.tse -o recording.tse
```

## Cross-format reverse path

`lml extract` recovers the original source file byte-equal for every
supported format because the encoder stashes the source bytes as a
preservation copy inside the `.lma`. No format-specific `--to-X` flag
is required.

| Source format | Recovered file(s) |
|---|---|
| EDF / EDF+ | `recording.edf` (byte-identical to input) |
| BDF / BDF+ | `recording.bdf` |
| BrainVision | `recording.vhdr` + `recording.vmrk` + `recording.eeg` |
| NeuroScan CNT | `recording.cnt` (900-byte SETUP + ELECTLOC tables + multiplexed int16) |
| EEGLAB | `recording.set` (full MAT v5 struct) + `recording.fdt` |
| DICOM | `recording.dcm` |
| Custom raw + sidecar | `recording.raw` + `recording.json` |

Format-aware reverse paths via the signal-only LML (`lml decode
--to-eeglab` / `--to-cnt`) are deferred — `lml extract` already
satisfies the use case from any modern `.lma`. See [`../FEATURES.md`](../FEATURES.md)
§3 ("EEGLAB writer reverse path", planned).

## Partial-decode constraints

`--channels` and `--time-range` both require an LMLFOOT1 seek table in
the source `.lml`. Files written before the footer was added decode
sequentially and the partial flags error explicitly rather than
falling back to a full scan.

- `--channels 0,2,4` — parser sort+dedups internally. Output is
  channel-major in sort+dedup'd index order (`[ch0, ch2, ch4]`).
  Refused alongside `--to-edf` (EDF reconstruction needs every
  channel slot).
- `--time-range START:END` — inclusive-start, exclusive-end sample
  indices. Past-EOF `END` clamps; past-EOF `START` errors. Refused
  alongside `--to-edf`.

## Strict vs best-effort decode

Two extract paths with different policies:

- **Strict** (default everywhere): `lma::read_entry_decoded` is the
  canonical reader. On codec failure (CRC mismatch, version skew,
  malformed metadata) it returns an error. `lml extract` and
  `lml extract-entry` both use this path.
- **Best-effort**: lmafs FUSE mounts can pass `--allow-raw-fallback`
  to return the raw stored LML bytes when the codec refuses. Intended
  for forensic triage of pre-v1.1 archives only — applications that
  expect real EDF will silently mis-interpret the result, so it's off
  by default. See [OS Integration](./07-os-integration.md).

## Stdin/stdout piping

The decode side supports stdin/stdout for shell composition (Refactor
R.3). Three rails:

| Mode | Trigger | Behavior |
|---|---|---|
| stdin in | `lml decode -` | Reads `.lml` from stdin into a tempfile shim, decodes |
| stdout out | `-o -` | Writes channel-major int32 LE to stdout |
| Both | `lml decode - -o -` | Pure pipe |

Stdout-to-TTY is refused (would corrupt the terminal). `--to-edf`
to stdout is refused. Batch mode (recursive directory input) cannot
go to stdout — the output of N files would interleave.

## Path-traversal protection

`path_is_unsafe(path)` in `lamquant-core/src/lma.rs:38` rejects any
entry path matching:

- Unix absolute (`/etc/passwd`)
- Windows backslash root (`\Windows\System32`)
- UNC (`\\server\share`)
- Parent traversal (`..` exact, `../`, `..\\`)
- Windows drive letters (`C:\…`)
- Embedded NUL bytes
- Windows Alternate Data Streams (`file:stream`)
- Reserved DOS device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)

Caught at archive write time (refuse to include) and again at extract
time (refuse to materialise). Defense-in-depth — neither half is
sufficient on its own.

## Flags

### `lml decode`

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output <PATH>` | path | (stdout) | Output file or `-` for stdout |
| `-r`, `--recursive` | bool | false | Walk subdirectories |
| `--skip-existing` | bool | false | Skip files whose output already exists |
| `-j`, `--threads <N>` | usize | 0 | Parallel worker count (0 = auto) |
| `--to-edf` | bool | false | Reconstruct full byte-identical EDF/BDF |
| `--channels <CSV>` | string | (all) | Comma-separated zero-based indices (requires LMLFOOT1) |
| `--time-range <START:END>` | string | (all) | Sample range (requires LMLFOOT1) |
| `--fail-fast` | bool | false | Abort batch on first failure |
| `--continue-on-error` | bool | true (implicit) | Explicit alias |

### `lml extract`

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output <DIR>` | path | (derived) | Output directory |
| `--verify` | bool | **true** | Verify per-entry SHA-256 against manifest |
| `--no-verify` | bool | false | Skip the SHA pass (trusted archives only) |

### `lml extract-entry`

| Flag | Type | Default | Description |
|---|---|---|---|
| `<ARCHIVE>` | positional | — | Input `.lma` file |
| `<ENTRY>` | positional | — | Entry path (exact or suffix match) |
| `-o`, `--output <PATH>` | path | (required) | Output file |
| `--force` | bool | false | Overwrite existing output |

## Error cases

| Trigger | Error |
|---|---|
| `--channels` / `--time-range` on file without LMLFOOT1 | Explicit "no seek table" error; no fallback |
| `--time-range START:END` with `START > total_samples` | Past-EOF start; refuses |
| `--channels` with `--to-edf` | Conflicting modes; refuses |
| Extract entry path is unsafe (`..`, `/`, etc.) | Refuse at extract time even if archive snuck it in |
| SHA-256 mismatch with `--verify` on | Per-entry error; extract aborts that file |
| Strict-decode codec failure | Error returned to caller (extract / extract-entry) |
| Stdout to TTY | Refuse |
| `lml decode --to-edf -o -` | Refuse (EDF needs seekable output) |
| Recursive decode to stdout | Refuse (output would interleave) |

## Related

- **Other buckets**:
  - [Compression](./01-compression.md) — `lml encode`, the reverse path
  - [Verification](./03-verification.md) — SHA chain consulted during extract
  - [Archive Ops](./04-archive-ops.md) — `volume-assemble` to reassemble before extract
  - [Browse / Inspect](./05-browse-inspect.md) — `lml ls` / `lml cat` for read-only inspection
  - [OS Integration](./07-os-integration.md) — lmafs FUSE mount, `--allow-raw-fallback`
  - [Export](./08-export.md) — different output formats (CSV, NPY, MAT, BIDS) from decoded signal
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:288` — `Decode` subcommand
  - `lamquant-core/src/bin/lml.rs:452` — `Extract` subcommand
  - `lamquant-core/src/bin/lml.rs:802` — `ExtractEntry` subcommand
  - `lamquant-core/src/lma.rs` — `read_entry`, `read_entry_decoded`, `extract_entry`
  - `lamquant-core/src/lma.rs:38` — `path_is_unsafe`
  - `lamquant-core/src/stream.rs` — `LmlReader::next_window`, `seek_to_window`, `windows_for_range`
- **Tests**:
  - `tests/integration/test_partial_decode.py`
  - `tests/integration/test_full_parity_*.py` — per-format extract roundtrip
  - `tests/integration/test_lma_browse.py`
- **Commits**:
  - `8b8bd1a` — `--channels` + `--time-range` partial decode (3.2 + 3.3)
  - `6e85fd5` — true streaming decode (3.4)
  - `0ee5b44` — `lml extract-entry` (3.7)
  - `f584432` — lmafs strict-decode default + `--allow-raw-fallback` (v1.4)
- **Cross-cutting docs**:
  - [`../FAQ.md`](../FAQ.md) — partial-decode edge cases, stdout-to-TTY guard
  - [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) — auto-generated full flag listing
