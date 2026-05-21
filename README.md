<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Clinical-Grade EEG Compression Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/Quitetall/LamQuant/releases/latest"><img src="https://img.shields.io/github/v/release/Quitetall/LamQuant?label=release&color=blue" alt="Latest release"></a>
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/tests-1271%20passing-brightgreen" alt="Tests: 1271 passing">
  <img src="https://img.shields.io/badge/rust-1.70%2B-dea584" alt="Rust 1.70+ (production codec)">
  <img src="https://img.shields.io/badge/python-3.10%2B%20%28training%20only%29-3776ab" alt="Python 3.10+ for training only">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/MCU-RP2350%20Hazard3-red" alt="RP2350">
  <a href="docs/features/"><img src="https://img.shields.io/badge/docs-feature%20catalogue-blueviolet" alt="Feature catalogue"></a>
</p>

<p align="center">
  <strong>Clinical-grade lossless + neural EEG compression codec targeting FDA 510(k) clearance</strong>
</p>

<p align="center">
  <em>EEG storage shrunk ~3× lossless vs zstd, with bit-exact roundtrip + random-access seek tables.</em>
</p>

---

## 30-second quickstart

LamQuant is a **Rust-first** codec. One install, three binaries.

```bash
# Clone + install via cargo (one command, builds lamquant + lml)
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant
cargo install --path lamquant-core --features host

# Launch the TUI (file picker, encode, browse, verify)
lamquant

# Or drive it from the shell — same operations, no TUI
lml encode recording.edf        # → recording.lma (~3× smaller, SHA-256 per entry)
lml ls --tree recording.lma     # browse contents
lml verify recording.lma        # end-to-end integrity check
```

**The three binaries:**

| Command                | What it does                                                                |
|------------------------|-----------------------------------------------------------------------------|
| `lamquant`             | Open the TUI home screen (file picker, op router)                           |
| `lamquant <op> [args]` | Run a TUI operation from the shell (passthrough to `lml`)                   |
| `lml [args]`           | Lossless codec — full CLI when args given; open TUI on lossless screen with no args (post-v1.4 wiring) |
| `lmq [args]`           | Neural codec — planned; binary lands when LMQ training passes ship gate     |

Want it in KDE Ark like a `.zip`? See **[OS Integration](docs/features/07-os-integration.md)**.
Want the full CLI tour? See the **[feature catalogue](docs/features/)**.

---

## Table of contents

- [Why LamQuant?](#why-lamquant)
- [What is LamQuant?](#what-is-lamquant)
- [Install](#install)
- [First 5 minutes](#first-5-minutes)
- [Lossless Codec (LML)](#lossless-codec-lml)
- [Neural Codec (LMQ)](#neural-codec-lmq)
- [Architecture](#architecture)
- [Quality + Safety](#quality--safety)
- [Project Structure](#project-structure)
- [What's New in v1.4](#whats-new-in-v14)
- [Documentation](#documentation)
- [Community](#community)
- [License](#license)

---

## Demo

<p align="center">
  <img src="assets/lma-ark-demo.gif" alt="Double-click .lma in Dolphin → opens in KDE Ark with entries and sizes populated" width="900">
  <br/>
  <sub><em>Double-click a `.lma` archive in Dolphin → opens in KDE Ark, entries listed with sizes. v1.4 Kerfuffle plugin, no FUSE mount required.</em></sub>
</p>

---

## Why LamQuant?

Hospital EEG storage is dominated by raw EDF/BDF that compresses poorly with general-purpose tools (gzip, zstd) because those tools don't exploit EEG-specific structure — inter-channel correlation, sub-band sparsity, low-amplitude bias drift. A regional hospital with 18 TB of clinical EEG cold-stores at ~$720/yr; the same data through LamQuant lossless lands at ~$200/yr **and** every transfer moves 3.6× less data. Same encoder runs on a 520 KB SRAM RP2350 for on-device telemetry.

Read the full pitch + three concrete user shapes (storage operator / telemetry pipeline / research archive): **[docs/PROBLEM.md](docs/PROBLEM.md)**.

---

## What is LamQuant?

**Two codecs, two extensions, one pipeline.**

### LML — the lossless codec

LamQuant Lossless. Le Gall 5/3 integer lifting DWT → LPC(2) → bias cancellation → Golomb-Rice entropy. Bit-exact EDF/BDF roundtrip — every header byte, annotation, timestamp, and trailing record preserved. ~2.3× compression on clinical EEG. Verified on 76,254 files across 7 datasets with zero failures.

LML is the **only** lossless codec in the project. The `.lml` and `.lma` extensions below are two ways to *apply* it:

**`.lml` files** — single-file mode. `lml encode --bare-lml <dir>` walks the directory and turns each EEG file (`.edf` / `.bdf` / `.vhdr` / `.cnt` / `.set` / `.dcm`) into a `.lml`. Sidecars + metadata (`.tse` / `.csv_bi` / `_summary.txt` / etc.) are **copied as-is** alongside, no compression. Directory structure unchanged. Useful when downstream tooling reads `.lml` directly and storage cost of uncompressed sidecars is acceptable. **Prints a loud data-loss warning** unless paired with `--i-understand-data-loss` — if you move the `.lml` away from its sidecar siblings, the metadata is gone.

**`.lma` archives** — **directory-archive mode (default)**. `lml encode <dir>` packs the *entire directory* into one `.lma` per recording. Every file goes through the cascade:

  | If file is… | Stored as |
  |---|---|
  | EEG (`.edf` / `.bdf` / `.vhdr` / `.cnt` / `.set` / `.dcm`) | LML (lossless codec) |
  | Anything else that compresses | zstd-9 |
  | Already compressed (`.gz` / `.zip` / `.jpg` / `.mp4` / …) | Stored verbatim |

  Result: **one self-contained `.lma` per recording**. SHA-256 per entry, mtime preserved, byte-exact extract on every entry. Directory structure preserved. **No byte ever lost.** Operators who delete originals after encoding recover everything via `lml extract`. Archives browsable via FUSE (`lmafs foo.lma /mnt/foo`) or natively in KDE Ark (v1.4 Kerfuffle plugin, no mount step).

### LMQ — the neural codec (in training)

Separate codec for high-compression-ratio neural reconstruction. Ternary encoder (W2A16, 2.6M params) + Vocos iSTFT decoder (4M–845M params) + Adaptive SNAC FSQ + rANS entropy. Targets 50–100× compression at R > 0.90. **Status: training. Ship gate is R > 0.90 on full TUEG validation sweep.**

> Not a cleared medical device. FDA 510(k) submission in preparation. See [docs/SAFETY.md](docs/SAFETY.md).

---

## Install

LamQuant is Rust-first. The headline install is **one cargo command** — no Python environment, no PyPI, no system package manager.

### Rust CLI (recommended)

```bash
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant
cargo install --path lamquant-core --features host
```

Builds + installs two binaries to `~/.cargo/bin/`:

  - **`lamquant`** — top-level launcher. No args → TUI home. Pass args → CLI passthrough.
  - **`lml`** — lossless-codec CLI. Full subcommand surface (encode / extract / verify / split / append / …).

A third binary `lmq` (neural codec CLI) lands when the LMQ training run clears its R > 0.90 ship gate. Until then, neural encoding is exercised through `python -m ai_models.experiment_runner` (training-only).

### Desktop GUI (Tauri)

```bash
cd gui && npm install && npm run tauri build
```

Or grab the pre-built bundle from the release page:

  - **linux**: `lamquant-gui_*.AppImage` / `lamquant-gui_*.deb`
  - **macos**: `lamquant-gui_*.dmg`
  - **windows**: `lamquant-gui_*.msi`

GUI shares the op-event contract, config, and history with the TUI/CLI — same `lamquant.toml`, same operations.

### Python bindings (optional, training-side only)

The Rust crate is the production codec. A Python wrapper lives in `lamquant_codec/` for research workflows that pre-date the Rust port + for the neural-codec training stack. It is **not** the recommended entry point and is currently **not published to PyPI**.

```bash
# From the cloned repo (PEP 668 systems require a venv):
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Pinned PyPI publishing tracked in [#TODO-pypi](https://github.com/Quitetall/LamQuant/issues) — until then, source-install is the only Python path.

---

## First 5 minutes

```bash
# 1. Encode an EDF + every sibling annotation into one .lma archive
lml encode recording.edf
# → recording.lma (one archive, ~3× smaller, every byte preserved)
#   See: docs/features/01-compression.md

# 2. Browse the archive (tree view with sizes + sha256 prefixes)
lml ls --tree recording.lma
#   See: docs/features/05-browse-inspect.md

# 3. Open it like a zip in your file manager
xdg-mime default org.kde.ark.desktop application/x-lma    # one-time setup
# Now double-click recording.lma in Dolphin → opens in Ark
#   See: docs/features/07-os-integration.md

# 4. Extract archive contents back to disk
#    Note: today the archive stores the LML-encoded EEG (recording.lml),
#    not the original .edf. Extract gives you the encoded form back.
lml extract recording.lma -o restored/
ls restored/                                  # → recording.lml + bundled sidecars

# 5. Decode the .lml back to the byte-equal original .edf
lml decode --to-edf restored/recording.lml -o restored/recording.edf
diff recording.edf restored/recording.edf     # silent — byte-equal
#   See: docs/features/02-decompression.md

# 6. Verify integrity end-to-end (SHA-256 per entry + archive-wide)
lml verify recording.lma --explain
#   See: docs/features/03-verification.md
```

> **Note**: a future encoder refactor will store the entry as `recording.edf` with `Method::Lml` so `lml extract` auto-decodes back to `.edf` in one step. Until then, `extract → decode --to-edf` is the documented two-step path.

For the full subcommand surface (split, concat, append, encrypt, multi-volume, …) browse the **[feature catalogue](docs/features/)** — 11 buckets sorted by what you want to do.

---

## Lossless Codec (LML)

```
signal[Nch][2500]
  → 3-level Le Gall 5/3 integer lifting DWT
  → per-subband LPC prediction (order 2)
  → bias cancellation (running mean, ctx=32, floor division)
  → Golomb-Rice entropy coding (adaptive k)
  → LML1 per-window packets, CRC-32 per window, zstd-9 metadata
```

Preserves every byte of the original EDF file: headers, annotations, timestamps, trailing data. EDF, EDF+C, EDF+D, and BDF (24-bit) all supported.

### Performance

| Metric | Value |
|--------|-------|
| Compression ratio | **2.3 : 1** on TUEG clinical EEG |
| Rust encode | ~200 MB/s (parallel with rayon) |
| Roundtrip verification | 76,254 files, 7 datasets, 0 failures |
| Wire format | Integer-only, bit-exact on x86/ARM/RISC-V |

### Competitive analysis

| Codec | CR | Type |
|-------|-----|------|
| gzip -6 | 1.61 : 1 | General |
| zstd -3 | 1.64 : 1 | General |
| FLAC | ~2.0 : 1 | Audio lossless |
| **LML** | **2.3 : 1** | **EEG-specific** |
| Shannon limit | ~2.4 : 1 | Theoretical |

Methodology + machine-pin + jitter notes: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Neural Codec (LMQ)

The ternary encoder targets the RP2350 Hazard3 RISC-V MCU (150 MHz, 520 KB SRAM, 4 MB flash). Weights ship to flash and run via XIP; activation memory is the real constraint. Decoders reconstruct the signal at the base station.

| Component | Params | Precision | Where |
|-----------|--------|-----------|-------|
| Encoder (TernaryMobileNetV5) | 2.6M | W2A16 | RP2350 (flash via XIP) |
| Decoder (Vocos ConvNeXt) | 4M – 845M | FP32 | Base station GPU |

Architecture: depthwise separable convolutions, FocalNet-style gating, ternary weights (XNOR + popcount, ~1.2 cyc/MAC on Hazard3 Zbb). Training uses SOAP optimizer (+0.0135 R over AdamW), WSD infinite LR schedule, and gradient checkpointing for large decoder tiers.

**Ship gate**: R > 0.90 on full TUEG validation sweep (held-out Siena). Current best on dev split is below threshold; sweep ongoing.

### Training data

- 14,779 hours of clinical EEG
- 5,320,265 windows (10 s each) across 11,580 patients
- 7 datasets: TUEG, CHB-MIT, PhysioNet Sleep-EDF, Siena Scalp EEG, EEGMMIDB, Mental Arithmetic, CAP Sleep
- Patient-level train/val split, Siena held out as validation-only

---

## Architecture

```
       Source files               Lossless pipeline             On-disk
                                                                output
  ┌─────────────────┐
  │  recording.edf  │──┐
  │  recording.tse  │  │       ┌──────────────────┐         ┌────────────┐
  │  summary.txt    │  ├──────▶│  EDF / BV / CNT  │         │            │
  │  sibling.csv    │  │       │  / SET / DICOM   │────┐    │ recording. │
  └─────────────────┘  │       │  / raw+sidecar   │    │    │   lma      │
                       │       │      reader      │    │    │            │
                       │       └──────────────────┘    │    │ ┌────────┐ │
                       │                ▼              ▼    │ │ .lml   │ │
                       │       ┌──────────────────┐ ┌────┐  │ │ (LML  )│ │
                       │       │  LML codec       │ │zstd│  │ ├────────┤ │
                       │       │  DWT→LPC→bias→GR │ │ -9 │  │ │ .edf   │ │
                       │       └──────────────────┘ └────┘  │ │ (zstd) │ │
                       │                ▼              ▼    │ ├────────┤ │
                       │       ┌──────────────────────────┐ │ │ .tse   │ │
                       └──────▶│  LMA cascade: LML→zstd   │ │ │ (zstd) │ │
                               │  →store, SHA-256/entry,  │─┤ ├────────┤ │
                               │  mtime preserved         │ │ │ ...    │ │
                               └──────────────────────────┘ │ └────────┘ │
                                                            └────────────┘
                                                                  │
                          ┌───────────────────────────────────────┤
                          ▼                ▼                ▼     ▼
                   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
                   │ lml extract│  │ lml verify │  │  lmafs FUSE│  │ KDE Ark    │
                   │ (byte-exact│  │ --explain  │  │  mount     │  │ Kerfuffle  │
                   │  restore)  │  │            │  │            │  │ plugin     │
                   └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

Detailed wire format: **[docs/lml-format-v1.md](docs/lml-format-v1.md)** (frozen, implementable without reference code).

---

## Quality + Safety

LamQuant is developed to FDA-grade reliability standards.

| Lane | Coverage |
|---|---|
| Integration tests | **1271 passing**, gated pre-push |
| Rust unit tests (lamquant-core) | 277 passing |
| Rust unit tests (lmafs FUSE) | 7 passing |
| Roundtrip verification | 76,254 EDFs × 7 datasets × 0 failures |
| Cross-language wire format | Python ↔ Rust, 22 format items verified |
| Adversarial suite | 18 tests: BDF, boundary values, trailing data, unicode filenames, cross-decoder, negative-mean signals |
| Fuzz harness | 3 targets (`cargo +nightly fuzz`) |
| Conformance vectors | 13 publishable test vectors for third-party LML readers |

Hardening invariants:

- Zero `unsafe` code in Rust
- Path-traversal protection on archive extract + plugin list-line parse
- u16/u32 overflow guards on all wire-format fields
- Manifest decompression bomb guard (16 MB compressed cap, 256 MB decompressed cap)
- Per-entry `original_size` cap (16 GB) — refuses zstd-bomb pattern (small compressed → multi-GB decompressed → OOM)
- Full JSON escaping (backslashes, quotes, control characters)

State map of every encoder/decoder/archive/crypto/FUSE/TUI state mapped to its named test: **[docs/TEST_COVERAGE_STATE_MAP.md](docs/TEST_COVERAGE_STATE_MAP.md)**.

---

## Project Structure

<details>
<summary>Top-of-tree layout (click to expand)</summary>

```
lamquant-core/          Rust library + CLI binary
  src/lml.rs              LML codec
  src/lma.rs              LMA archive format
  src/container.rs        LML file container
  src/golomb.rs           Golomb-Rice encoder/decoder
  src/lifting.rs          Le Gall 5/3 integer DWT
  src/lpc.rs              LPC + bias cancellation
  src/edf.rs              EDF/BDF reader
  src/bin/lml.rs          CLI binary

lamquant_codec/         Python codec (numba JIT + pure Python fallback)
  ops/                    DSP primitives (single source of truth)
  lossless.py             LML compress/decompress
  edf_to_lml.py           EDF → LML converter + container I/O
  lma.py                  LMA archive reader/writer
  batch.py                Batch compress/decompress/validate

crates/
  lmafs/                  FUSE filesystem (read-only mount of .lma)
  lamquant-ops/           Op-event runner shared with TUI/GUI

installer/
  ark-plugin/             KDE Ark Kerfuffle CliInterface plugin (C++/Qt6)
  install-mime.sh         MIME registration for .lma
  install-ark-plugin.sh   Build + install the Ark plugin
  lma-open                Double-click handler (FUSE mount → xdg-open)

ai_models/              Neural codec training
  architectures/          Model definitions (encoder, decoder, SNN, blocks)
  student/                Joint training configs and sweep tooling
  dataset_sim/            Manifest building and preprocessing
  experiment_runner.py    Unified runner (run/ab/sweep/leaderboard)

firmware/               RP2350/ESP32 embedded targets
docs/                   Specifications and design docs
  features/               11-bucket feature catalogue (NEW v1.4)
```

</details>

---

## What's New in v1.4

Tag: **[`lossless-codec-v1.4`](https://github.com/Quitetall/LamQuant/releases/tag/lossless-codec-v1.4)** (2026-05-19)

- **KDE Ark plugin** — Kerfuffle CliInterface plugin lets Ark open `.lma` natively. No FUSE mount step. Entries listed with size + compressed size. ([install guide](docs/INSTALL_FILE_MANAGER_INTEGRATION.md))
- **`lml ls --long`** — versioned (`#lml-ls schema=1`) TAB-separated wire format for OS plugin shell-outs.
- **`lmafs` default-strict decode** — codec failure returns `EIO` instead of silently corrupted bytes. Opt back into the old fallback via `--allow-raw-fallback`.
- **lma core hardening** — refuses zstd-bomb archives via `original_size` cap (16 GB); strict 5-column TSV parse with path-traversal defense at plugin layer.
- **Installer hardening** — distro-agnostic preflight (pacman/dpkg/rpm), Plasma session env, ABI drift sentinel.
- **11-bucket feature catalogue** under [`docs/features/`](docs/features/).

Full changelog: **[CHANGELOG.md](CHANGELOG.md)**.

---

## Documentation

### Get started

| Document | Content |
|----------|---------|
| [Feature catalogue](docs/features/) | 11-bucket guide sorted by what you want to do |
| [Problem statement](docs/PROBLEM.md) | Who LamQuant is for, three user shapes, vs gzip/bzip2/zstd/xz/FLAC |
| [File-manager integration](docs/INSTALL_FILE_MANAGER_INTEGRATION.md) | KDE Ark plugin + FUSE mount + double-click flow |
| [FAQ + troubleshooting](docs/FAQ.md) | Common errors, recovery, partial-decode recipes |

### Reference

| Document | Content |
|----------|---------|
| [CLI reference](docs/CLI_REFERENCE.md) | Every subcommand + every flag (auto-generated from `lml --help`) |
| [Feature inventory](docs/FEATURES.md) | Value-ordered inventory: shipped / deferred / out-of-scope |
| [Feature matrix](docs/CLI_FEATURE_MATRIX.md) | Chronological audit trail by phase, with commit refs |
| [Benchmarks](docs/BENCHMARKS.md) | Published numbers + methodology + machine-pin + jitter notes |

### Internals

| Document | Content |
|----------|---------|
| [System spec](docs/SPEC.md) | Full system architecture, memory map, timing |
| [Test coverage state map](docs/TEST_COVERAGE_STATE_MAP.md) | Every encoder/decoder/archive/crypto/FUSE/TUI state → named test |
| [Stress testing](docs/STRESS_TESTING.md) | 1M-file harness + >100 GB fixture recipe |
| [Safety](docs/SAFETY.md) | Risk analysis, regulatory status |

### Spec (third-party implementers)

| Document | Content |
|----------|---------|
| [LML format spec](docs/lml-format-v1.md) | Frozen wire format — implementable without reference code |
| [Conformance suite](specs/conformance/README.md) | 13 publishable test vectors for third-party LML readers |

---

## Community

- **Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md). Looking for a first issue? Try the [good-first-issue label on GitHub](https://github.com/Quitetall/LamQuant/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- **Security**: report vulnerabilities per [SECURITY.md](SECURITY.md) — do NOT open a public issue for security bugs.
- **Discussion**: open a [GitHub issue](https://github.com/Quitetall/LamQuant/issues) for bugs, feature requests, or questions.
- **License compliance**: AGPL-3.0 means network use counts as distribution. If you ship a service built on LamQuant, the source has to be available to the people using it.

---

## License

**Code**: [AGPL-3.0](LICENSE.md) | **Weights**: CC BY-NC 4.0 | **Spec**: CC BY 4.0

---

<p align="center">
  <sub>OpenHuman Technologies — open-source brain-computer interfaces for everyone</sub>
</p>
