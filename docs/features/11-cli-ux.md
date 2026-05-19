# CLI UX

> User-facing CLI ergonomics: colors, verbosity, completions, man
> pages, force-overwrite semantics, JSON event stream, stdin/stdout
> piping. Cross-cutting flags that apply across every subcommand.

These flags don't ship features so much as they shape how every other
feature behaves. They live as **global** clap flags on the root CLI,
so `lml encode --quiet` and `lml verify --quiet` both work.

## At a glance

| Feature | Flag | Status | First shipped | Notes |
|---|---|---|---|---|
| Color control | `--color {auto,always,never}` | shipped | v1.0 (1.3) | Respects `NO_COLOR` + `CLICOLOR_FORCE` |
| Quiet log | `-q` / `--quiet` | shipped | v1.0 (1.2) | Equivalent to `RUST_LOG=off` |
| Verbose log | `-v` / `-vv` / `-vvv` | shipped | v1.0 (1.2) | INFO / DEBUG / TRACE |
| Force-overwrite | `--force` | shipped | v1.0 (1.4) | Default refuse-clobber on most writers |
| Continue-on-error | `--continue-on-error` | shipped | v1.0 (1.7) | Explicit alias for the default behavior |
| Fail-fast | `--fail-fast` | shipped | v1.0 | Mutually exclusive with `--continue-on-error` |
| Shell completions | `lml completions <SHELL>` | shipped | v1.0 (1.8) | clap_complete: bash / zsh / fish / pwsh / elvish |
| Man page | `lml manpage` | shipped | v1.0 (1.9) | clap_mangen-rendered roff |
| Self-test | `lml self-test` | shipped | v1.0 (1.10) | Synth → encode → decode → verify, no fs dep |
| Stdin pipe (encode) | `lml encode -` | shipped | v1.0 (1.5a) | Tempfile shim; refuses TTY |
| Stdout pipe (decode) | `-o -` | shipped | v1.0 (1.5b) | Refuses TTY / `--to-edf` / batch |
| Output templating | `-o '{stem}.lml'` | shipped | v1.0 (1.6) | `paths::expand_template` |
| JSON event stream | `--emit-json-events` | shipped | v1.0 | OpEvent JSON lines on stdout (Tauri GUI / Python TUI consume) |
| Backend selector | `--backend {desktop,firmware}` | shipped | v1.0 | Host rayon+AVX2 vs scalar reference |
| Help text | `--help` / `-h` | shipped | v1.0 | Per-subcommand EXAMPLES blocks |
| Version | `--version` | shipped | v1.0 | clap auto-injected |

## Global flags

These appear on every subcommand because they're declared on the
top-level `Cli` struct as `global = true`:

```
$ lml --help
...
GLOBAL OPTIONS:
  --emit-json-events       Emit one JSON line per OpEvent to stdout
  -q, --quiet              Suppress all log output
  -v, --verbose...         Increase log verbosity
  --color <CHOICE>         auto / always / never
  --backend <CHOICE>       desktop / firmware
```

### `--color`

Three values:

- **`auto`** (default) — Enable colour when stderr is a TTY; respect
  `NO_COLOR` and `CLICOLOR_FORCE`
- **`always`** — Force ANSI escapes even when stderr isn't a TTY
- **`never`** — No ANSI escapes

Standard env-var overrides:

- `NO_COLOR=anything` → disabled, even when `--color=always`
  (the [no-color.org](https://no-color.org/) standard)
- `CLICOLOR_FORCE=1` → forced when `--color=auto` and stderr isn't a TTY

The resolver lives in `ColorChoice::resolve()` (lml.rs:132). Bible R32:
single deterministic point of truth — no scattered "is colour on?"
checks.

### `--quiet` / `-v`

Filter precedence (highest first):

1. `RUST_LOG` env var (standard tracing-subscriber escape hatch)
2. `--quiet` / `-v` CLI flags
3. Fallback: `lamquant_core=warn,lml=info`

| Flag | Filter |
|---|---|
| `--quiet` | `off` (nothing routes to stderr) |
| (none) | `lamquant_core=warn,lml=info` |
| `-v` | `lamquant_core=info,lml=info` |
| `-vv` | `lamquant_core=debug,lml=debug` |
| `-vvv` | `lamquant_core=trace,lml=trace` |

Output writer is **stderr**, not stdout — stdout is reserved for
signal data (decode), JSON events (`--emit-json-events`), and command
output (cat / ls / etc.).

`--quiet` and `-v` are mutually exclusive (clap conflict).

### `--force`

Default behavior is **refuse-clobber on most write commands**:

- `append` / `recompress` / `set-metadata` / `strip-pii` / `extract-entry` /
  `volume-split` / `volume-assemble` / `encrypt` / `decrypt` / `sign` /
  `split` / `concat` / `fetch`

Pass `--force` to opt into overwrite. The legacy `encode` / `decode` /
`export` paths keep their **silent-overwrite default** to preserve
existing CI scripts; they have `--skip-existing` instead.

If you want fail-on-exist semantics for `encode` / `decode` / `export`
in a CI pipeline, the operator should check `test ! -e <output>`
before invocation.

### `--emit-json-events`

When set, every subcommand emits one JSON line per `OpEvent` on
stdout instead of pretty progress. All status output goes to stderr;
stdout is reserved for the event stream.

Used by the Tauri GUI and Python TUI to consume the same wire format
(`specs/op-events.schema.json`). Events include:

| Event | Fields |
|---|---|
| `Started` | `ts_ms`, `op_id`, `total` (optional) |
| `Progress` | `ts_ms`, `current`, `total`, `message` |
| `Log` | `ts_ms`, `message` |
| `FileDone` | `ts_ms`, `path`, `success`, `ms`, `cr`, `bytes_in`, `bytes_out`, `samples`, ... |
| `Done` | `ts_ms`, `message` |
| `Error` | `ts_ms`, `message` |

Example consumer:
```sh
lml encode -r /data/ -o out/ --emit-json-events 2>/dev/null \
  | jq -c 'select(.type == "FileDone" and .success == false)'
```

### `--backend`

Selects the compute path:

- **`desktop`** (default on host) — rayon per-channel + AVX2 inner
  loops. Production perf path.
- **`firmware`** — Reference scalar path. Slower; matches MCU build
  behaviour byte-for-byte. Use for debugging or conformance checks.

Output bytes are identical across backends (locked by
`tests/byte_equal_backends.rs`); only wall-clock differs.

## Commands

### `lml completions`

Generate shell completion script for the given shell. Pipe to the
shell's completion directory.

Synopsis:
```
lml completions <SHELL>
```

Where `<SHELL>` is one of: `bash`, `zsh`, `fish`, `powershell`,
`elvish`.

Examples:
```
lml completions bash       > ~/.local/share/bash-completion/completions/lml
lml completions zsh        > "${fpath[1]}/_lml"
lml completions fish       > ~/.config/fish/completions/lml.fish
lml completions powershell | Out-String | Invoke-Expression
lml completions elvish     > ~/.config/elvish/lib/lml.elv
```

After install, `lml <TAB>` completes subcommands; `lml encode --<TAB>`
completes flags.

### `lml manpage`

Emit a roff-format man page to stdout via `clap_mangen`. Pipe to your
system's man path.

Synopsis:
```
lml manpage > ~/.local/share/man/man1/lml.1
mandb
```

### `lml self-test`

Built-in end-to-end smoke test: synthesize a deterministic signal,
encode it, decode it, verify byte-exact roundtrip + CRC-32 footer
match. No filesystem dependency (uses an in-memory tempfile shim).
Exits 0 on success, 1 on any mismatch.

Synopsis:
```
lml self-test
```

Useful for confirming a fresh install works before pointing it at
real data. Diagnostic for "is my install broken?" without needing a
test EEG file handy.

## Stdin / stdout piping

Three rails (Refactor R.3):

| Mode | Trigger | Behavior |
|---|---|---|
| Stdin encode | `lml encode -` | Reads source bytes from stdin into a tempfile shim, encodes |
| Stdout decode | `lml decode … -o -` | Writes channel-major int32 LE to stdout |
| Cross-pipe | `lml encode - -o - \| lml decode -` | Memory-only roundtrip |

Guards:

- Stdin must NOT be a TTY (would pretty-print binary garbage and
  pollute the user's terminal)
- Stdout to TTY is refused on `decode -o -`
- `lml decode --to-edf -o -` is refused (EDF reconstruction needs
  seekable output)
- Batch mode (recursive directory input) cannot decode to stdout —
  outputs from N files would interleave

## Output templating

`-o '{stem}.lml'` runs through `paths::expand_template` with the
following tokens:

| Token | Expands to |
|---|---|
| `{stem}` | Input filename stem (e.g. `recording` for `recording.edf`) |
| `{name}` | Input filename (e.g. `recording.edf`) |
| `{ext}` | Input extension (e.g. `edf`) |
| `{parent}` | Input parent directory |

Example:
```sh
lml encode -r /data/study-7/ -o '{parent}/encoded/{stem}.lma'
# /data/study-7/sub-01/recording.edf -> /data/study-7/sub-01/encoded/recording.lma
```

## Help text + EXAMPLES blocks

Per-subcommand `--help` carries an EXAMPLES section for the high-traffic
commands (`encode`, `decode`, `extract`, `ls`, `cat`, `verify-archive`,
`volume-split`, `volume-assemble`). Each shows 2-4 common forms with
exact CLI strings.

Top-level `lml --help` carries a COMMON WORKFLOWS footer:

```
COMMON WORKFLOWS:
  lml encode foo.edf -o out/foo.lma       # encode (default = per-recording LMA)
  lml extract foo.lma -o restored/        # unpack everything byte-for-byte
  lml ls foo.lma --tree                   # browse archive contents
  lml cat foo.lma recording.tse           # extract single entry to stdout
  lml verify foo.lma                      # archive integrity (auto-dispatches)
  lml info foo.lma                        # tree listing (auto-dispatches)
  lml roundtrip foo.edf                   # paranoid bit-exact verification
  lml encode -r dir/ -o out/              # batch encode (per-recording LMAs)
```

## Interactive mode

Running `lml` with no subcommand drops into an interactive TUI
(`tui::run_interactive`). Pre-existing UX for picking input files +
operation without remembering CLI syntax. Documented in
[`../cli_guide.md`](../cli_guide.md).

## Flags

All globally-scoped flags (work on every subcommand):

| Flag | Type | Default | Description |
|---|---|---|---|
| `--emit-json-events` | bool | false | Emit OpEvent JSON lines on stdout |
| `-q`, `--quiet` | bool | false | Suppress all log output (mutually excl. with `-v`) |
| `-v`, `--verbose` | count | 0 | `-v` INFO, `-vv` DEBUG, `-vvv` TRACE |
| `--color <CHOICE>` | enum | `auto` | `auto` / `always` / `never` |
| `--backend <CHOICE>` | enum | (lib default) | `desktop` / `firmware` |

Per-subcommand UX flags:

| Flag | Subcommand(s) | Description |
|---|---|---|
| `--force` | most writers | Overwrite existing output |
| `--fail-fast` | encode / decode / roundtrip | Abort batch on first failure |
| `--continue-on-error` | encode / decode | Explicit alias for default behavior |
| `--skip-existing` | encode / decode | Skip outputs that already exist |

## Environment variables

| Var | Purpose |
|---|---|
| `RUST_LOG` | Override the `--quiet` / `-v` filter directly |
| `NO_COLOR` | Force colour off regardless of `--color` |
| `CLICOLOR_FORCE` | Force colour on when stderr isn't a TTY (`--color=auto` only) |
| `LAMQUANT_KEY` | 32-byte hex AES key (see [Cryptography](./06-cryptography.md)) |
| `LAMQUANT_PASSWORD` | Password for `--password` mode |

## Error cases

| Trigger | Behavior |
|---|---|
| `--quiet` and `-v` together | clap conflict; refuses to parse |
| `--color always` with `NO_COLOR` set | colour stays off (no-color.org wins) |
| `lml encode -` with stdin attached to a TTY | refuses ("won't read binary from a TTY") |
| `lml decode … -o -` with stdout to a TTY | refuses |
| `lml decode --to-edf -o -` | refuses |
| Recursive decode to stdout | refuses (outputs would interleave) |
| `lml completions foo` with unknown shell | clap-rejects the value |

## Related

- **Other buckets**:
  - All other buckets — every command in every bucket inherits the
    global flags
  - [Operational](./09-operational.md) — `--emit-json-events` for daemonised use
  - [Browse / Inspect](./05-browse-inspect.md) — `lml cat` is a UX-friendly entry point
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:20-90` — root `Cli` struct (global flags)
  - `lamquant-core/src/bin/lml.rs:132` — `ColorChoice::resolve`
  - `lamquant-core/src/bin/lml.rs:1327` — `init_tracing` (verbosity)
  - `lamquant-core/src/bin/lml.rs:959` — `Completions`
  - `lamquant-core/src/bin/lml.rs:949` — `SelfTest`
  - `lamquant-core/src/bin/lml.rs:942` — `Manpage`
  - `lamquant-core/src/paths.rs` — `expand_template` (output templating)
  - `crates/lamquant-ops/` — `OpEvent` JSON wire format
- **Tests**:
  - `tests/integration/test_color_env_overrides.py`
  - `tests/integration/test_completions_generation.py`
  - `tests/integration/test_self_test.py`
  - `tests/integration/test_output_templating.py`
  - `tests/integration/test_emit_json_events.py`
- **Commits**:
  - Phase 1.2 — `--quiet` / `-v`
  - Phase 1.3 — `--color`
  - Phase 1.4 — `--force` overwrite
  - Phase 1.5 — stdin / stdout piping
  - Phase 1.6 — output templating
  - Phase 1.7 — `--continue-on-error`
  - Phase 1.8 — shell completions
  - Phase 1.9 — man page
  - Phase 1.10 — self-test
- **Cross-cutting docs**:
  - [`../CLI_REFERENCE.md`](../CLI_REFERENCE.md) — auto-generated full flag listing
  - [`../cli_guide.md`](../cli_guide.md) — interactive TUI walkthrough
  - [`../FAQ.md`](../FAQ.md) — common UX gotchas
