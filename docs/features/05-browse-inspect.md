# Browse / Inspect

> Read-only surface: list, cat, info. Composes with `less`, `grep`,
> `fzf`, jq, etc. For mutating archive lifecycle, see
> [Archive Ops](./04-archive-ops.md). For double-click-to-open and
> file-manager browse, see [OS Integration](./07-os-integration.md).

The browse surface is intentionally small: `lml ls` lists, `lml cat`
prints one entry, `lml info` dumps metadata. All three auto-dispatch
between LML and LMA on the magic bytes so a single mental model works
for either format.

## At a glance

| Feature | Command / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Flat list | `lml ls` | shipped | v1.0 | One path per line; tar tf style |
| Tree view | `lml ls --tree` | shipped | v1.0 | Sizes + method + sha256 prefix |
| Machine-readable list | `lml ls --long` | shipped | v1.4 | TAB-separated; `#lml-ls schema=1` marker line |
| Single-entry cat | `lml cat` | shipped | v1.0 | Composes with `less` / `grep` / `fzf` |
| LML inspect | `lml info` | shipped | v1.0 | Header + duration + flag decode + JSON metadata |
| LMA auto-dispatch | `lml info` (on `.lma`) | shipped | v1.1 | Routes to `ls --tree` |
| Per-channel stats | `lml stats` | shipped | v1.0 | Compression ratio + entropy |
| LML-to-LML diff | `lml diff` | shipped | v1.0 | Sample-by-sample comparison |
| Deprecated list-archive | `lml list-archive` | shipped | v1.0 | Stderr deprecation; use `lml ls --tree` |

## Commands

### `lml ls`

Browse `.lma` archive contents. Default output is flat, one path per
line — pipes cleanly to `grep`, `awk`, `fzf`.

Synopsis:
```
lml ls <INPUT>
lml ls <INPUT> --tree
lml ls <INPUT> --long
```

Examples:
```
# One path per line (tar tf style)
lml ls recording.lma

# Tree view with sizes, compression method, sha256 prefix
lml ls recording.lma --tree

# Machine-readable for plugins (Ark, file managers, scripts)
lml ls recording.lma --long
```

`--tree` output:
```
recording.lma (3.3 KB, 8 entries, 1.97x CR)
├── recording.lml             1.8 KB  lml      sha256:7f4a...
├── recording.edf.zst         312 B   zstd     sha256:a2cd...
├── recording.tse              58 B   store    sha256:bc8e...
└── recording_summary.txt     144 B   store    sha256:de77...
```

`--long` output (TAB-separated, schema-versioned):
```
#lml-ls schema=1
1843  1843  lml       7f4a9b...  recording.lml
312   312   secondary a2cd47...  recording.edf.zst
58    58    store     bc8e0f...  recording.tse
144   144   store     de77c1...  recording_summary.txt
```

Columns (`--long`): `original_size\tcompressed_size\tmethod\tsha256\tpath`.
The `#lml-ls schema=1` first line is a versioned marker so the Ark
Kerfuffle plugin (and any future file-manager integration) can detect
ABI drift before parsing. See [OS Integration](./07-os-integration.md).

### `lml cat`

Extract a single entry to stdout. Composes with `less` / `grep` /
`fzf` / shell preview panes. Path-traversal protected: the entry must
appear verbatim in the manifest.

Synopsis:
```
lml cat <INPUT> <ENTRY>
```

Examples:
```
# Show a sidecar
lml cat recording.lma recording.tse

# Page through it
lml cat recording.lma recording_summary.txt | less

# Search seizure events
lml cat recording.lma recording.csv_bi | grep seiz

# Quick preview in fzf-driven file pickers
lml ls recording.lma | fzf --preview 'lml cat recording.lma {}'
```

For binary entries (`.lml`, `.edf`, `.dcm`, etc.) `lml cat` writes
raw bytes — pipe to `xxd` or `file` if you want a human view, or use
`lml extract-entry` to write to disk.

### `lml info`

Show LML file information. Default output: header (channels,
window_size, sample_rate, bit_depth, total_samples, n_windows), flag
byte decode, seek table count if present, JSON metadata.

Auto-dispatches on magic bytes: `LMA1` → delegates to `lml ls --tree`.
So `lml info foo.lma` shows the archive contents tree without you
needing to remember which subcommand to type.

Synopsis:
```
lml info <INPUT>
```

Examples:
```
# LML metadata dump
lml info recording.lml

# Auto-routes to ls --tree
lml info recording.lma
```

### `lml stats`

Show per-channel signal statistics for LML files: compression ratio,
per-channel entropy, byte savings per channel.

Synopsis:
```
lml stats [-r] <INPUT>
```

### `lml diff`

Compare two LML files sample-by-sample. Reports first-mismatch
indices, max absolute difference, and per-channel SHA-256.

Synopsis:
```
lml diff <A> <B>
```

### `lml list-archive` (deprecated)

Deprecated alias for `lml ls --tree`. Prints a one-line stderr
deprecation notice; same behavior otherwise. Slated for removal in
v2.0.

## Magic-byte auto-dispatch

Both `lml info` and `lml verify` look at the first 4 bytes of the
input:

| First 4 bytes | Routed to |
|---|---|
| `LML1` | LML codec path (info: header dump; verify: CRC-32 sweep) |
| `LMA1` | LMA archive path (info: `ls --tree`; verify: archive verifier) |

Mixed-directory walks (`-r`) call the right path per file. Operators
no longer need to remember the file's type to inspect it.

## Composing browse with shell tools

`lml ls` flat output pipes into the standard Unix toolbox:

```sh
# Find every entry larger than 1 MB across a corpus
for lma in /data/eeg/*.lma; do
  lml ls --long "$lma" \
    | awk -F'\t' 'NR>1 && $2 > 1048576 {print FILENAME"\t"$5}'
done

# fzf-driven entry picker with live preview
lml ls big.lma \
  | fzf --preview 'lml cat big.lma {}' --bind 'enter:execute(lml extract-entry big.lma {} -o -)+abort'

# Filter the manifest by extension
lml ls --long recording.lma | awk -F'\t' '$5 ~ /\.tse$/'
```

`lml cat | jq` works on `.json` entries.

## Flags

| Flag | Type | Default | Subcommand | Description |
|---|---|---|---|---|
| `--tree` | bool | false | `ls` | Tree-style view with sizes + method + sha256 prefix |
| `--long` | bool | false | `ls` | TAB-separated machine-readable (`#lml-ls schema=1` header) |
| `-r`, `--recursive` | bool | false | `stats` | Walk subdirectories |

`--tree` and `--long` are mutually exclusive (clap-enforced).

## Error cases

| Trigger | Error |
|---|---|
| `lml cat` with entry not in manifest | "entry '<path>' not found in archive" |
| `lml ls` on a file with neither LML1 nor LMA1 magic | "unknown file type" |
| `lml info` on truncated LML | "container header truncated" |
| `lml ls --tree --long` (both set) | clap conflict |
| `lml cat` entry path includes `..` | refuse (manifest-side rejection at write time means this shouldn't occur, but defense-in-depth) |

## Related

- **Other buckets**:
  - [Decompression](./02-decompression.md) — `lml extract-entry` writes what `lml cat` streams
  - [Verification](./03-verification.md) — `lml verify` uses the same magic-byte dispatch
  - [OS Integration](./07-os-integration.md) — `lml ls --long` is the Ark Kerfuffle plugin's wire format
  - [CLI UX](./11-cli-ux.md) — `--emit-json-events` provides a richer machine format
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:571` — `Ls` subcommand
  - `lamquant-core/src/bin/lml.rs:599` — `Cat` subcommand
  - `lamquant-core/src/bin/lml.rs:332` — `Info` subcommand
  - `lamquant-core/src/bin/lml.rs:384` — `Stats` subcommand
  - `lamquant-core/src/bin/lml.rs:403` — `Diff` subcommand
- **Tests**:
  - `tests/integration/test_lma_browse.py`
  - `tests/integration/test_magic_byte_dispatch.py`
  - `tests/integration/test_ls_long_format.py`
- **Commits**:
  - `9524287` — `lml ls --tree` + `lml cat` + auto-dispatch (8.15)
  - `d809549` — `lml ls --long` versioned wire format (v1.4)
  - `40f9780` — `list-archive` deprecated alias
- **Cross-cutting docs**:
  - [`../INSTALL_FILE_MANAGER_INTEGRATION.md`](../INSTALL_FILE_MANAGER_INTEGRATION.md) — uses `ls --long` for plugin integration
  - [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) — auto-generated full flag listing
