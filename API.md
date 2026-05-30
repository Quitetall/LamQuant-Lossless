# LamQuant-Lossless — API reference

The codec's supported developer surfaces: the `lml` CLI, the `.lml`/`.lma` wire
formats, the Rust library crates, and the Python (PyO3) bindings. **This repo is
authoritative for the LML/LMA format and encode contract** — other repos invoke it,
never redefine it.

A surface not listed here is unsupported. Byte-exact wire spec: [`specs/lml-format-v1.md`](specs/lml-format-v1.md).

---

## 1. CLI — `lml`

`lml` is the core codec binary (single binary, clap subcommands). The `lamquant`
AIO wrapper (TUI + dispatch) lives in the Codec/hub repo and `exec`s `lml`;
`lamquant <sub>` forwards to `lml <sub>`. Inside this repo, `lml` is the surface.

### Global flags (before the subcommand)

| Flag | Effect |
|---|---|
| `--backend desktop\|firmware` | Compute path. `desktop` (default) = rayon + AVX2. `firmware` = scalar serial, matches the MCU build **byte-for-byte**. Output bytes identical across both (locked by `tests/byte_equal_backends.rs`); only wall-clock differs. |
| `--json` | One JSON `OpEvent` per line to stdout (`specs/op-events.schema.json`); status → stderr. The GUI/TUI wire format. |
| `-v` / `-vv` / `-vvv` | INFO / DEBUG / TRACE. `RUST_LOG` overrides. |
| `--quiet` | Errors only. Mutually exclusive with `-v`. |
| `--color auto\|always\|never` | ANSI control; respects `NO_COLOR` / `CLICOLOR_FORCE`. |

### Encode / decode

| Subcommand | Input → Output | Notes |
|---|---|---|
| `encode <in> <out>` | EDF/BDF/BrainVision/CNT/DICOM/EEGLAB/raw → **one `.lma` per recording** | Default bundles compressed `.lml` + original source bytes + every sibling annotation (`.tse`, `.csv_bi`, `.lbl_bi`, `_summary.txt`…) via the **LML→zstd→store cascade**. No byte silently dropped: every input file → exactly one LMA entry. Flags: `--noise-bits N` (strip N LSBs, 0=lossless), `--window N`, `--threads N`, `--recursive`, `--skip-existing`, `--include`/`--exclude` (glob), `--verify`, `--cross-validate`. |
| `encode … --no-bundle`/`--bare-lml` | → bare `.lml` (signal only, siblings dropped) | Prints a 20-line loud stderr warning every call; suppress only with `--i-understand-data-loss`. **Footgun** — use `archive` for datasets. |
| `decode <in> <out>` | `.lml` → raw signal (`i32` LE binary) | |
| `export <in>` | `.lml` → CSV / NPY / raw | |

### Archive (LMA) — the dataset surface

| Subcommand | Input → Output | Notes |
|---|---|---|
| `archive <dir> <out.lma>` | a directory → **ONE `.lma`** | The correct way to package a corpus: walks the dir, EDF→LML, other files→zstd, incompressible→stored, copies failures — one archive, no byte dropped. **One corpus = one LMA.** |
| `list-archive <a.lma>` | → entry listing | |
| `extract <a.lma> <dir>` | `.lma` → restored tree | |
| `extract-entry <a.lma> <name>` | → single entry (payload verified vs manifest) | Pure-LMA training reads per-recording LML entries this way. |
| `append <a.lma> <file>` | add entry (refuses path-collision unless `--force`) | |
| `recompress <a.lma>` | re-pack entries (`--in-place` atomic or `--output`) | |
| `verify-archive <a.lma>` | integrity check all entries | |

### Volumes (split oversized archives)

| Subcommand | Role |
|---|---|
| `volume-split` / `volume-assemble` | Split a `.lma` into size-bounded volumes / reassemble. |
| `split` / `concat` | Split/join `.lml` by window range (argv-order-independent, byte-identical). |

### Inspect

| Subcommand | Role |
|---|---|
| `info <in>` | LML header / metadata. |
| `stats <in>` | Per-channel signal statistics. |
| `diff <a> <b>` | Sample-by-sample compare two `.lml`. |
| `ls <a.lma>` / `cat <a.lma> <entry>` | Browse archive entries. |
| `metrics <in>` | Compression metrics. |

### Verify / integrity / recover

| Subcommand | Role |
|---|---|
| `verify <in>` | Decode-and-check the file round-trips. |
| `roundtrip <in>` | Strict encode→decode→compare; any drift fails. |
| `verify-manifest <manifest.lml.json>` | Check listed files: existence, size, SHA-256. |
| `recover <damaged.lml>` | Salvage valid windows past corruption (LMLFOOT1 seek table). |
| `self-test` | Built-in conformance vectors. |

### Security

| Subcommand | Role |
|---|---|
| `encrypt` / `decrypt` | AEAD (magic/version/auth-tag checked; fails closed on mismatch). |
| `sign` / `verify-signature` | HMAC `.hmac` sidecar. |
| `strip-pii` | Atomic-swap scrub of identifying header fields. |
| `audit-log` | Append/verify tamper-evident audit log. |
| `set-metadata` | Atomic metadata update. |

### Ops / misc

`bench` (encode/decode speed) · `watch` · `fetch` · `notify` · `pccp` (model-version pin / integrity verify — neural-codec governance, runtime-only here) · `manpage` · `completions <shell>`.

---

## 2. Wire formats — `.lml` and `.lma`

**Authoritative byte spec:** [`specs/lml-format-v1.md`](specs/lml-format-v1.md). Conformance vectors under [`specs/conformance/`](specs/conformance/).

- **`.lml`** = one recording, losslessly compressed. Pipeline: integer Le Gall 5/3 DWT (3 levels) → LPC (orders 1–32, anytime-adaptive, bias-cancelled) → Golomb-Rice → container. Integer-only (`i32`/`i64`); platform-independent. Magic `LML1`. Per-window CRC-32 + `LMLFOOT1` footer seek table.
- **`.lma`** = an archive (zip-like) of many entries: inner `.lml` for recordings, zstd for other files, stored/copied for the incompressible or the un-encodable — a **cascade** that never drops a byte. Magic `LMA1`. One corpus → one `.lma`.

**Backend contract:** `Firmware` (no_std scalar) and `Desktop` (rayon+AVX2) emit **byte-identical** output. Any optimization that breaks byte-equality is a wire-format change — needs a new golden table (`LAMQUANT_REGEN_GOLDENS=1`), not a quiet bump. Gated by `tests/byte_equal_backends.rs`.

---

## 3. Rust library

Workspace crates:

| Crate | Role |
|---|---|
| `lamquant-lossless` | The codec: `lml`, `container`, `lma`, `edf`, entropy coders (`golomb`, `rans`, `arithmetic`), `lifting`, `lpc`, two `backend`s, `ffi`, `wasm`, PyO3 bindings. |
| `lamquant-common` | Shared primitives re-exported by the codec: `crc32`, `ingest`, `paths`, `source/*` readers, pipeline traits. |
| `lamquant-firmware` | `no_std` MCU build (RP2350 / Hazard3 RISC-V). |
| `lamquant-weights` | Lossless LUTs / static tables. |
| `crates/lmafs` | LMA filesystem helpers. |
| `crates/lamquant-lsl` | LSL bridge primitives. |

Public API: `lamquant_lossless::{lml, container, lma, edf, backend, golomb, rans, …}`. The codec re-exports `lamquant_common::{crc32, ingest, paths}`.

---

## 4. Python bindings (PyO3)

Module `lamquant_codec` (built from `lamquant-lossless`). Exposes encode/decode/container ops + entropy primitives. Used by BLUT's training pipeline to read LML entries out of an LMA (pure-LMA training). Neural ships its own bindings separately — the PyO3 surface here is lossless-only.

---

## Contracts

- **`encode` = per-recording, `archive` = per-corpus.** Use `archive` for a dataset. `encode --bare-lml` is a loud-warned footgun.
- **One corpus = one `.lma`.**
- **Byte-identical across backends** or it's a versioned format change.
- **No byte silently dropped** in either cascade (encode-bundle or archive).

See also: [README](README.md) · meta index [`../API.md`](../API.md).
