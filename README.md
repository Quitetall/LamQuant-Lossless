# LamQuant Lossless

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20436114.svg)](https://doi.org/10.5281/zenodo.20436114)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Bit-exact lossless EEG compression codec. Real-time, integer-only, deployable on x86 / ARM / RISC-V without floating-point hardware. Paper subject of *IEEE Transactions on Biomedical Circuits and Systems* (2026 submission).

**Cite:** archived at Zenodo — [`10.5281/zenodo.20436114`](https://doi.org/10.5281/zenodo.20436114) (see [`CITATION.cff`](CITATION.cff)).

**Headline numbers (paper, full evidence in [Eagle](https://github.com/Quitetall/Eagle)):**
- TUEG v2.0.2 (1.76 TB, 70,831 EDF files): **2.287:1** compression ratio
- CHB-MIT: **2.7229:1** (15.9% improvement over Chen et al.)
- RP2350 Hazard3 (RISC-V, Verilator-measured): 0.627 Msa/s, **119× real-time**, CPI 1.071
- Bit-exact roundtrip verified on 88,147 encode/decode operations across 13 corpora — zero failures

## Pipeline

```
EDF input → integer Le Gall 5/3 DWT (3 levels) → LPC predictor (orders 1-32,
anytime adaptive w/ bias cancellation) → Golomb-Rice entropy → LML container
(LMA archive wrapper for batch / sidecar metadata)
```

All stages are integer-only (`i32` / `i64`). Wire format is platform-independent. Two backends (`Firmware` no_std + `Desktop` with rayon) produce byte-identical output, gated by `tests/byte_equal_backends.rs`.

## Workspace layout

| Crate | Purpose |
|---|---|
| `crates/lamquant-common/` | Shared primitives (CRC-32, EDF reader, LMA archive, path helpers, ingest) — both lossless and (private) neural codecs depend on this |
| `lamquant-lossless/` | Codec library + CLI binaries (`lml`, `lamquant`) |
| `lamquant-firmware/` | RP2350 bare-metal bin (riscv32imac-unknown-none-elf, no_std + alloc-free hot path) |
| `lamquant-weights/` | LUTs as data (alloc-free) |
| `crates/lamquant-ops/` | Op runner / launcher / transport (shared by CLI, TUI, Python) |
| `crates/lmafs/` | FUSE filesystem over `.lma` archives |
| `crates/lamquant-history/` | TUI session history reader/writer |
| `crates/lamquant-ipc-types/` | MCU↔host IPC protocol (postcard-encoded envelopes) |
| `crates/lamquant-lsl/` | Lab Streaming Layer integration (opt-in, requires liblsl) |
| `reference_implementations/python_codec/` | Python wheel `lamquant-codec` (PyO3 bindings) |
| `reference_implementations/c_firmware/` | C parity reference for firmware build |

## Quick start

### CLI

```bash
cargo install --path lamquant-lossless --bin lml
lml encode in.edf out.lml
lml decode out.lml restored.edf
diff in.edf restored.edf  # byte-identical
```

### Python

```bash
cd reference_implementations/python_codec && pip install .
```
```python
import lamquant_codec as lq
lq.compress("in.edf", "out.lml")
samples = lq.decompress("out.lml")
```

### Firmware

```bash
cargo build -p lamquant-firmware --target riscv32imac-unknown-none-elf --profile firmware
```

## Tests

```bash
cargo test --workspace --lib                                       # 418 unit tests
cargo test --test byte_equal_backends                              # wire-format byte-equality gate
cargo build -p lamquant-firmware --no-default-features             # firmware no_std build
```

## Reproducibility

Paper bench numbers, Verilator RTL harness, MNE/pyedflib cross-validation, and per-corpus evidence JSON live in the separate **Eagle** repository: https://github.com/Quitetall/Eagle

The IEEE TBioCAS manuscript source lives at `docs/paper/lamquant_lossless.tex`. Rebuild the submission bundle with:

```bash
bash tools/build_submission.sh   # → outputs/submission/manuscript.pdf (12 pp, 0 warnings)
```

## Architecture

This repository is one repo in an 8-product Unix decomposition of LamQuant:

| Public | Private (for now) |
|---|---|
| **LamQuant-Lossless** (this repo) | LamQuant (monorepo source of truth) |
| Eagle (validation) | LamQuant-Neural (SNN/TNN models) |
| LamQuant-Firmware (planned formal split) | LamQuant-Codec (turnkey integration) |
| BLUT (training orchestrator) | |
| LamQuant-Vision (LSL + viz) | |

## License

GNU GENERAL PUBLIC LICENSE v3 (see `LICENSE.md`). Patent grant per GPLv3 §11 covers `US Patent Pending #64/032,641`.

## Cite

```bibtex
@article{lam2026lamquant,
  title   = {LamQuant Lossless: A Real-Time, Bit-Exact, Wirelessly-Deployable EEG Compression Algorithm},
  author  = {Lam, Brian},
  journal = {IEEE Transactions on Biomedical Circuits and Systems},
  year    = {2026},
  note    = {Submitted}
}
```
