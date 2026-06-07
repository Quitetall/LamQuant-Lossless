<h1 align="center">LamQuant Lossless</h1>

<p align="center">
  <a href="https://github.com/Quitetall/LamQuant-Lossless/actions/workflows/ci.yml"><img src="https://github.com/Quitetall/LamQuant-Lossless/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://doi.org/10.5281/zenodo.20484969"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20484969.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/tests-418%20passing-brightgreen" alt="Tests: 418 passing">
  <img src="https://img.shields.io/badge/rust-1.81%2B-dea584" alt="Rust 1.81+">
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="License GPL-3.0"></a>
  <img src="https://img.shields.io/badge/MCU-RP2350%20Hazard3-red" alt="RP2350 Hazard3">
  <a href="docs/features/"><img src="https://img.shields.io/badge/docs-feature%20catalogue-blueviolet" alt="Feature catalogue"></a>
</p>

<p align="center">
  <strong>Bit-exact lossless EEG compression codec — real-time, integer-only, datacenter to microcontroller.</strong>
</p>

<p align="center">
  <em>≈2.3× lossless on clinical EEG, byte-exact EDF/BDF roundtrip, and the same encoder runs on a 520 KB-SRAM RISC-V MCU.</em>
</p>

---

## 30-second quickstart

LamQuant Lossless is a **Rust-first** codec. One install, one binary (`lml`).

```bash
git clone https://github.com/Quitetall/LamQuant-Lossless.git && cd LamQuant-Lossless
cargo install --path lamquant-lossless --bin lml

lml encode recording.edf            # → recording.lma  (≈2.3× smaller, SHA-256 per entry)
lml ls recording.lma                # browse archive entries
lml verify-archive recording.lma    # end-to-end integrity check
```

Full CLI surface — wire formats, every subcommand and flag, Rust crates + Python bindings — is in **[API.md](API.md)** (authoritative for the LML/LMA format contract). For task-oriented walkthroughs see the **[feature catalogue](docs/features/)**.

---

## Table of contents

- [Why LamQuant Lossless?](#why-lamquant-lossless)
- [What is it?](#what-is-it)
- [Install](#install)
- [First steps](#first-steps)
- [The codec (LML)](#the-codec-lml)
- [Headline numbers](#headline-numbers)
- [Quality + verification](#quality--verification)
- [Project structure](#project-structure)
- [Ecosystem](#ecosystem)
- [Documentation](#documentation)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Cite](#cite)

---

## Why LamQuant Lossless?

Clinical EEG is stored as raw EDF/BDF, which general-purpose compressors (gzip, zstd, xz) handle poorly because they don't exploit EEG-specific structure — inter-channel correlation, sub-band sparsity, and slow bias drift. LamQuant Lossless is purpose-built for that signal: it shrinks EEG roughly **3× past zstd** with a **bit-exact** roundtrip, so not a single header byte, annotation, or timestamp is lost.

Crucially, the codec is **integer-only** (`i32` / `i64`, no floating-point hardware required) and **platform-independent**, so the *same* encoder that runs in a datacenter also runs on a 520 KB-SRAM RP2350 Hazard3 RISC-V microcontroller for on-device telemetry — at **119× real-time**.

---

## What is it?

**Two extensions, one lossless pipeline.**

### `.lml` — single-recording mode

One recording, losslessly compressed: integer Le Gall 5/3 DWT → LPC → bias cancellation → Golomb-Rice → container (magic `LML1`, per-window CRC-32, `LMLFOOT1` footer seek table). `lml encode --bare-lml` emits a bare `.lml` (signal only). It prints a loud data-loss warning unless paired with `--i-understand-data-loss` — move a `.lml` away from its sibling annotation files and that metadata is gone.

### `.lma` — directory-archive mode (default)

`lml encode <dir>` packs an entire recording (or `lml archive <dir>` an entire corpus) into one self-contained `.lma` (magic `LMA1`) via a cascade that **never drops a byte**:

| If the file is… | Stored as |
|---|---|
| EEG (`.edf` / `.bdf` / `.vhdr` / `.cnt` / `.set` / `.dcm`) | LML (lossless codec) |
| Anything else that compresses (`.tse`, `.csv_bi`, `_summary.txt`, …) | zstd-9 |
| Already compressed (`.gz` / `.zip` / `.jpg` / `.mp4` / …) | stored verbatim |

Result: **one corpus = one `.lma`**, SHA-256 per entry, mtime preserved, byte-exact extract on every entry, directory structure intact. Archives are browsable as a read-only mount via the FUSE filesystem (`lmafs foo.lma /mnt/foo`).

---

## Install

LamQuant Lossless is Rust-first — the headline install is a single cargo command, no Python environment required. Requires **Rust 1.81+**.

### Rust CLI (recommended)

```bash
git clone https://github.com/Quitetall/LamQuant-Lossless.git && cd LamQuant-Lossless
cargo install --path lamquant-lossless --bin lml
```

Installs the **`lml`** binary (full codec CLI: encode / decode / archive / extract / verify / split / encrypt / …) to `~/.cargo/bin/`. The `lamquant` AIO TUI wrapper that `exec`s `lml` lives in the meta-repo; inside this repo `lml` is the surface.

### Python bindings (optional, research-side)

The Rust crate is the production codec. A PyO3 wrapper (`lamquant_codec`) is provided for research and training workflows; it is **not** published to PyPI.

```bash
cd reference_implementations/python_codec && pip install .
```

```python
import lamquant_codec as lq
lq.compress("in.edf", "out.lml")
samples = lq.decompress("out.lml")
```

### Firmware (RP2350 / no_std)

The codec core is `no_std` + alloc-free on the hot path. Build the bare core that the MCU uses:

```bash
cargo build -p lamquant-lossless --no-default-features   # no_std + alloc core
```

Full RP2350 bare-metal bring-up (linker scripts, XIP, `riscv32imac-unknown-none-elf`) lives in the dedicated **[LamQuant-Firmware](#ecosystem)** repo.

---

## First steps

```bash
# 1. Archive a recording + every sibling annotation into one .lma — nothing dropped
lml encode study/recording.edf            # → recording.lma
#    See: docs/features/01-compression.md

# 2. Browse the archive (entries with sizes + sha256 prefixes)
lml ls --tree recording.lma
#    See: docs/features/05-browse-inspect.md

# 3. Restore the tree to disk (byte-exact, per-entry SHA-256 verified)
lml extract recording.lma restored/
#    See: docs/features/02-decompression.md

# 4. Prove bit-exactness: strict encode → decode → compare in one shot
lml encode recording.edf --verify         # non-zero exit on any drift

# 5. Verify integrity end-to-end (SHA-256 per entry + archive-wide)
lml verify-archive recording.lma
#    See: docs/features/03-verification.md
```

For the full subcommand surface (`split` / `concat` / `append` / `recompress` / `volume-split` / `encrypt` / `sign` / `strip-pii` / `export` / `recover` …) see **[API.md](API.md)** and the **[feature catalogue](docs/features/)** — 11 buckets sorted by what you want to do.

---

## The codec (LML)

```
EDF/BDF input
  → integer Le Gall 5/3 lifting DWT (3 levels)
  → LPC predictor (orders 1–32, anytime-adaptive, bias-cancelled)
  → Golomb-Rice entropy coding (adaptive k)
  → LML1 container (per-window CRC-32, LMLFOOT1 seek table)
  → LMA archive wrapper for batch / sidecar metadata
```

Every stage is integer-only (`i32` / `i64`); the wire format is platform-independent. Two backends — `Firmware` (no_std, scalar) and `Desktop` (rayon + AVX2) — produce **byte-identical** output, locked by [`tests/byte_equal_backends.rs`](lamquant-lossless/tests/byte_equal_backends.rs). Any optimization that breaks byte-equality is a versioned wire-format change, not a quiet bump.

EDF, EDF+C, EDF+D, and BDF (24-bit) are all supported. Detailed, implementable byte spec: **[specs/lml-format-v1.md](specs/lml-format-v1.md)** (FROZEN).

---

## Headline numbers

From the IEEE JBHI manuscript; full per-corpus evidence JSON lives in the separate **[Eagle](https://github.com/Quitetall/Eagle)** validation repo.

| Metric | Value |
|---|---|
| TUEG v2.0.2 (1.76 TB, 70,831 EDF files) | **2.287 : 1** compression ratio |
| CHB-MIT (686 files, 45.76 GB) | **2.7229 : 1** — 15.9% over Chen et al. |
| RP2350 Hazard3 (RISC-V, Verilator-measured) | 0.627 Msa/s, **119× real-time**, CPI 1.071 |
| Bit-exact roundtrip | **0 failures** over 88,147 encode/decode ops across 13 corpora |

Shannon-entropy ceilings, baseline comparators (gzip/zstd/FLAC), and the Verilator RTL provenance are documented in the reproducibility package — see [Reproducibility](#reproducibility).

---

## Quality + verification

Developed to clinical-grade reliability standards.

| Lane | Coverage |
|---|---|
| Rust unit tests | **418 passing** (`cargo test --workspace --lib`) |
| Wire-format byte-equality | `Firmware` ↔ `Desktop` byte-identical, gated by `tests/byte_equal_backends.rs` |
| Roundtrip verification | 88,147 encode/decode ops × 13 corpora × **0 failures** |
| EDF-reader cross-validation | pyedflib + MNE parity harness |
| Fuzz harness | 6 targets (`cd lamquant-lossless && cargo +nightly fuzz run <target>`) |
| Conformance vectors | publishable test vectors + verifier for third-party LML readers ([`specs/conformance/`](specs/conformance/)) |
| Paper-claim verification | every numeric claim cross-checked against evidence JSON (60/0 PASS at submission) |

```bash
cargo test --workspace --lib                  # 418 unit tests
cargo test --test byte_equal_backends         # Firmware vs Desktop byte-equality gate
cargo build -p lamquant-lossless --no-default-features   # no_std core build
```

---

## Project structure

<details>
<summary>Workspace layout (click to expand)</summary>

```
lamquant-lossless/        Codec library (lamquant_core) + CLI binary
  src/lml.rs                LML codec
  src/lma.rs                LMA archive format
  src/container.rs          LML file container
  src/golomb.rs             Golomb-Rice entropy coder
  src/lifting.rs            Le Gall 5/3 integer DWT
  src/lpc.rs                LPC + bias cancellation
  src/edf.rs                EDF/BDF reader
  src/backend.rs            Firmware (no_std) + Desktop (rayon) paths
  src/bin/lml.rs            CLI binary
  fuzz/                     6 libFuzzer targets

crates/
  lamquant-common/          Shared primitives (CRC-32, EDF reader, ingest, paths)
  lmafs/                    FUSE filesystem — read-only mount of .lma

reference_implementations/
  python_codec/             Python wheel `lamquant-codec` (PyO3 bindings)
  c_firmware/               C parity reference for the firmware build

specs/                     Wire-format spec + conformance vectors
docs/                      Design docs, feature catalogue, paper source
```

</details>

| Crate | Purpose |
|---|---|
| `lamquant-lossless/` | Codec library (`lamquant_core`) + `lml` CLI binary |
| `crates/lamquant-common/` | Shared primitives (CRC-32, EDF reader, LMA helpers, ingest) re-exported by the codec |
| `crates/lmafs/` | FUSE filesystem over `.lma` archives |
| `reference_implementations/python_codec/` | Python wheel `lamquant-codec` (PyO3 bindings) |
| `reference_implementations/c_firmware/` | C parity reference for the firmware build |

---

## Ecosystem

This repository is the public, lossless-codec slice of a multi-product decomposition of LamQuant. It is **authoritative for the LML/LMA format and encode contract** — other products invoke it, never redefine it.

| Public | Private (for now) |
|---|---|
| **LamQuant-Lossless** (this repo) | LamQuant (monorepo, source of truth) |
| Eagle (validation + evidence) | LamQuant-Neural (SNN/TNN models) |
| LamQuant-Firmware (RP2350 bare-metal) | LamQuant-Codec (turnkey integration) |
| BLUT (training orchestrator) | |
| LamQuant-Vision (LSL + visualization) | |

---

## Documentation

| Document | Content |
|---|---|
| [API.md](API.md) | `lml` CLI, `.lml`/`.lma` wire formats, Rust crates + Python bindings — the format contract |
| [Feature catalogue](docs/features/) | 11-bucket guide sorted by what you want to do |
| [specs/lml-format-v1.md](specs/lml-format-v1.md) | Frozen LML wire format — implementable without reference code |
| [specs/conformance/](specs/conformance/) | Publishable test vectors + verifier for third-party readers |
| [docs/design/architecture.md](docs/design/architecture.md) | System architecture |
| [docs/design/mathematics.md](docs/design/mathematics.md) | Mathematical foundations (DWT / LPC / Golomb-Rice) |
| [docs/design/hardware.md](docs/design/hardware.md) | Hardware reference (RP2350 Hazard3) |
| [docs/design/validation.md](docs/design/validation.md) | Validation strategy |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Reproducibility

Paper bench numbers, the Verilator RTL harness, MNE/pyedflib cross-validation, and per-corpus evidence JSON live in the separate **[Eagle](https://github.com/Quitetall/Eagle)** repository.

The IEEE JBHI manuscript source is at [`docs/paper/lamquant_lossless.tex`](docs/paper/lamquant_lossless.tex); the supplementary reproducibility package is described in [`docs/paper/SUPPLEMENTARY_README.md`](docs/paper/SUPPLEMENTARY_README.md). Rebuild the submission bundle with:

```bash
bash tools/build_submission.sh   # → outputs/submission/manuscript.pdf
```

---

## License

**GNU General Public License v3** (see [`LICENSE.md`](LICENSE.md)). The patent grant under GPLv3 §11 covers **US Patent Pending #64/032,641** (commercial implementation rights only; academic and derivative research are unaffected).

---

## Cite

Archived at Zenodo: [`10.5281/zenodo.20484969`](https://doi.org/10.5281/zenodo.20484969) (see [`CITATION.cff`](CITATION.cff)).

```bibtex
@article{lam2026lamquant,
  title   = {LamQuant Lossless: A Real-Time, Bit-Exact, Wirelessly-Deployable EEG Compression Algorithm},
  author  = {Lam, Brian},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Submitted}
}
```

---

<p align="center">
  <sub>OpenHuman Technologies — open-source brain-computer interfaces for everyone</sub>
</p>
