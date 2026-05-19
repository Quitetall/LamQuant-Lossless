<p align="center">
  <img src="assets/banner.svg" alt="LamQuant — Clinical-Grade EEG Compression Codec" width="100%">
</p>

<p align="center">
  <a href="https://github.com/quitetall/lamquant/actions/workflows/ci.yml"><img src="https://github.com/quitetall/lamquant/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/rust-1.70%2B-dea584" alt="Rust 1.70+">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License AGPL-3.0">
  <img src="https://img.shields.io/badge/MCU-RP2350%20Hazard3-red" alt="RP2350">
</p>

<p align="center">
  <strong>Clinical-grade lossless + neural EEG compression codec targeting FDA 510(k) clearance</strong>
</p>

<p align="center">
  <em>EEG storage shrunk ~3× lossless vs zstd, with bit-exact roundtrip + random-access seek tables.</em><br/>
  <em>For who, when, and why: <a href="docs/PROBLEM.md">docs/PROBLEM.md</a>. For numbers: <a href="docs/BENCHMARKS.md">docs/BENCHMARKS.md</a>.</em>
</p>

---

## What is LamQuant?

LamQuant compresses EEG recordings for clinical storage and on-device transmission. Two codecs, one pipeline:

**LML (Lossless)** -- bit-exact EDF/BDF roundtrip. Le Gall 5/3 integer lifting DWT -> LPC(2) -> bias cancellation -> Golomb-Rice entropy coding. ~2.3x compression ratio on clinical EEG. Verified on 76,254 files across 7 datasets with zero failures.

**LMA (Archive)** -- the **default output unit**. One `.lma` archive per recording bundles the `.lml` signal + the original source-format bytes (`.edf` / `.vhdr+.vmrk+.eeg` / `.dcm` / `.set+.fdt` / `.cnt` / `.raw+sidecar`) + every sibling annotation file (TUH `.tse`, `.csv_bi`, `.lbl_bi`, `_summary.txt`, etc.) via the LML → zstd → store cascade. SHA-256 per entry, mtime preserved, byte-exact extract on every format. **No byte ever lost.** Operators who delete originals after encoding recover everything via `lml extract`.

**LMQ (Neural)** -- ternary encoder (W2A16, 2.6M params) + Vocos iSTFT decoder (up to 845M params) + Adaptive SNAC FSQ + rANS entropy coding. Targets 50-100x compression at R > 0.90. Currently training.

Bare `.lml` (signal-only, no archive envelope) is available via `--no-bundle` / `--bare-lml`, but prints a loud data-loss warning every invocation that has to be paired with `--i-understand-data-loss` to silence. The encoder will not let you drop sibling files quietly.

> Not a cleared medical device. FDA 510(k) submission in preparation. See [docs/SAFETY.md](docs/SAFETY.md).

---

## Install

```bash
pip install lamquant-codec              # lossless codec (numpy only, no GPU needed)
pip install lamquant-codec[fast]        # + numba JIT (3-4x faster)
pip install lamquant-codec[neural]      # + torch (neural codec + training)
pip install lamquant-codec[all]         # everything
```

Or from source:

```bash
git clone https://github.com/Quitetall/LamQuant.git && cd LamQuant
pip install -e ".[all]"
```

Rust CLI (no dependencies):

```bash
# Installs both `lml` (codec CLI) and `lamquant` (top-level launcher)
cargo install --path lamquant-core
```

`lamquant` is the recommended entrypoint — it auto-routes to the TUI
(`lamquant --tui`), the desktop GUI (`lamquant --gui` if installed),
or passes subcommands through to `lml` (`lamquant encode <file>`).

Desktop GUI (Tauri):

```bash
# From source
cd gui && npm install && npm run tauri build

# Or download the pre-built bundle for your platform from the
# v1.0+ release page:
#   linux:   lamquant-gui_*.AppImage  /  lamquant-gui_*.deb
#   macos:   lamquant-gui_*.dmg
#   windows: lamquant-gui_*.msi
```

The GUI shares its op-event contract, config, and history with the
TUI/CLI — same lamquant.toml, same operations.

---

## CLI Usage

### Python CLI (`lamquant` / `oh`)

```bash
# Compress a directory of EEG files (lossless, no model needed)
lamquant compress data/ -o compressed/ --mode lossless -r

# Neural compression (requires trained checkpoint)
lamquant compress data/ -o compressed/ --mode neural -c weights/student_subband_gold.ckpt -r

# Decompress back to .npy
lamquant decompress compressed/ -o decoded/ -r

# Validate quality against originals (HTML report)
lamquant validate compressed/ -r --reference data/ --level C --report-html quality.html

# Inspect a file's metadata
lamquant info recording.lml --json

# First-run wizard
lamquant init
```

### Rust CLI (`lml`)

```bash
# Encode a directory of EDF files to LML
lml encode <dir>

# Bundle into a single .lma archive
lml archive <dir> -o archive.lma

# Extract archive bit-exactly
lml extract archive.lma -o <dir>

# Verify archive integrity
lml verify-archive archive.lma

# List archive contents
lml list-archive archive.lma

# Benchmark encode/decode speed
lml bench <file.edf>
```

---

## Lossless Codec (LML)

```
signal[Nch][2500]
  -> 3-level Le Gall 5/3 integer lifting DWT
  -> per-subband LPC prediction (order 2)
  -> bias cancellation (running mean, ctx=32, floor division)
  -> Golomb-Rice entropy coding (adaptive k)
  -> LML1 per-window packets, CRC-32 per window, zstd-9 metadata
```

Preserves every byte of the original EDF file: headers, annotations, timestamps, trailing data. EDF, EDF+C, EDF+D, and BDF (24-bit) are all supported.

### Performance

| Metric | Value |
|--------|-------|
| Compression ratio | **2.3 : 1** on TUEG clinical EEG |
| Rust encode | ~200 MB/s (parallel with rayon) |
| Roundtrip verification | 76,254 files, 7 datasets, 0 failures |
| Wire format | Integer-only, bit-exact on x86/ARM/RISC-V |

### Competitive Analysis

| Codec | CR | Type |
|-------|-----|------|
| gzip -6 | 1.61 : 1 | General |
| zstd -3 | 1.64 : 1 | General |
| FLAC | ~2.0 : 1 | Audio lossless |
| **LML** | **2.3 : 1** | **EEG-specific** |
| Shannon limit | ~2.4 : 1 | Theoretical |

---

## Neural Codec (LMQ)

The ternary encoder targets the RP2350 Hazard3 RISC-V MCU (150 MHz, 520 KB SRAM, 4 MB flash). Weights ship to flash and run via XIP; activation memory is the real constraint. Decoders reconstruct the signal at the base station.

| Component | Params | Precision | Where |
|-----------|--------|-----------|-------|
| Encoder (TernaryMobileNetV5) | 2.6M | W2A16 | RP2350 (flash via XIP) |
| Decoder (Vocos ConvNeXt) | 4M -- 845M | FP32 | Base station GPU |

Architecture: depthwise separable convolutions, FocalNet-style gating, ternary weights (XNOR + popcount, ~1.2 cyc/MAC on Hazard3 Zbb). Training uses SOAP optimizer (+0.0135 R over AdamW), WSD infinite LR schedule, and gradient checkpointing for large decoder tiers.

### Training Data

- 14,779 hours of clinical EEG
- 5,320,265 windows (10 s each) across 11,580 patients
- 7 datasets: TUEG, CHB-MIT, PhysioNet Sleep-EDF, Siena Scalp EEG, EEGMMIDB, Mental Arithmetic, CAP Sleep
- Patient-level train/val split, Siena held out as validation-only

---

## Hardening

LamQuant is developed to FDA-grade reliability standards:

- Zero `unsafe` code in Rust
- BOM detection with clear error messages
- Path traversal protection on archive extraction
- u16/u32 overflow guards on all wire format fields
- Manifest decompression bomb guard (256 MB cap)
- Full JSON escaping (backslashes, quotes, control characters)
- Cross-language wire format verified (Python <-> Rust) on 22 format items
- Adversarial test suite: 18 tests covering BDF, boundary values, trailing data, unicode filenames, cross-decoder, and negative-mean signals

---

## Project Structure

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
  edf_to_lml.py           EDF -> LML converter + container I/O
  lma.py                  LMA archive reader/writer
  batch.py                Batch compress/decompress/validate

ai_models/              Training infrastructure
  architectures/          Model definitions (encoder, decoder, SNN, blocks)
  student/                Joint training configs and sweep tooling
  dataset_sim/            Manifest building and preprocessing
  experiment_runner.py    Unified runner (run/ab/sweep/leaderboard)

firmware/               RP2350/ESP32 embedded targets
installer/              Cross-platform installer
docs/                   Specifications and design docs
```

---

## Testing

```bash
# Python unit + integration tests
pytest tests/ -q

# Rust tests
cargo test --manifest-path lamquant-core/Cargo.toml

# Lossless roundtrip verification (FDA-grade)
python scripts/verify_roundtrip.py /path/to/edfs/ --recursive --workers 4

# Fuzzing
cd lamquant-core && cargo +nightly fuzz run decompress
```

---

## Training

```bash
# Single experiment
python -m ai_models.experiment_runner run --tier 3 --preset fast

# Named recipe
python -m ai_models.experiment_runner recipe snac_balanced --tier 3

# Parameter sweep
python -m ai_models.experiment_runner sweep --tier 3 --grid '{"prd_weight": [0.05, 0.1, 0.2]}'

# Leaderboard
python -m ai_models.experiment_runner leaderboard
```

---

## Documentation

| Document | Content |
|----------|---------|
| [CLI Reference](docs/CLI_REFERENCE.md) | Every subcommand + every flag + examples (auto-generated from `lml --help`) |
| [Feature Inventory](docs/FEATURES.md) | Value-ordered inventory by category: shipped / deferred / left / out-of-scope |
| [Feature Matrix](docs/CLI_FEATURE_MATRIX.md) | Chronological audit trail (104 rows by phase, with commit refs) |
| [Problem Statement](docs/PROBLEM.md) | Who LamQuant is for, three user shapes, comparison vs gzip/bzip2/zstd/xz/FLAC |
| [Benchmarks](docs/BENCHMARKS.md) | Published numbers + methodology (canonical machine pin, ±15% jitter, FLAC multi-channel notes) |
| [LML Format Spec](docs/lml-format-v1.md) | Frozen wire format -- implementable without reference code |
| [Conformance Suite](specs/conformance/README.md) | 13 publishable test vectors for third-party LML readers |
| [System Spec](docs/SPEC.md) | Full system architecture, memory map, timing |
| [FAQ + Troubleshooting](docs/FAQ.md) | Common errors, recovery, partial-decode recipes |
| [Stress Testing](docs/STRESS_TESTING.md) | 1M-file harness + >100 GB fixture recipe |
| [Safety](docs/SAFETY.md) | Risk analysis, regulatory status |

---

## License

**Code**: [AGPL-3.0](LICENSE.md) | **Weights**: CC BY-NC 4.0 | **Spec**: CC BY 4.0

---

<p align="center">
  <sub>OpenHuman Technologies -- open-source brain-computer interfaces for everyone</sub>
</p>
