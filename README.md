# lml

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20484969.svg)](https://doi.org/10.5281/zenodo.20484969)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

`lml` is the CLI + TUI for LamQuant lossless EEG compression. Bare `lml` opens the TUI; `lml <subcommand>` runs scriptable CLI workflows. Default compression is MCU-safe, integer-only, and bit-exact. Basestation lossless mode is opt-in behind experimental Cargo features.

**Cite:** archived at Zenodo — [`10.5281/zenodo.20484969`](https://doi.org/10.5281/zenodo.20484969) (see [`CITATION.cff`](CITATION.cff)).

**API reference:** [API.md](API.md) — the `lml` CLI, `.lml`/`.lma` wire formats, Rust crates + Python bindings. Authoritative for the LML/LMA format contract.

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

## Lossless deployment modes

```bash
lml encode in.edf -o out.lma
lml encode in.edf -o out.lma --lossless-mode mcu
cargo build -p lml --features experimental_basestation
lml encode in.edf -o out.lma --lossless-mode basestation
```

`mcu` is default and uses the bounded integer profile. `basestation` remains bit-exact but may use host-only, computationally expensive compression paths as they land. Near-lossless `--max-error` and lossy H.BWC `--target-bps` are separate modes, not lossless deployment modes.

Host export supports HDF5 with `--features hdf5` and NWB-like HDF5 `ElectricalSeries` with `--features nwb`:

```bash
lml export out.lma --format hdf5 -o out.h5 --lossless-mode mcu
lml export out.lma --format nwb -o out.nwb --lossless-mode mcu
```

## Workspace layout

| Crate | Purpose |
|---|---|
| `crates/lamquant-common/` | Shared primitives (CRC-32, EDF reader, LMA archive, path helpers, ingest) — both lossless and (private) neural codecs depend on this |
| `lamquant-lossless/` | `lml` crate: codec library + CLI/TUI binary |
| `crates/lmafs/` | FUSE filesystem over `.lma` archives |
| `reference_implementations/python_codec/` | Python wheel `lamquant-codec` (PyO3 bindings) |

## Quick start

### CLI

```bash
cargo install --path lamquant-lossless --bin lml
lml encode in.edf -o out.lma
lml extract out.lma -o restored/
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

The IEEE TBME manuscript source lives at `docs/paper/lamquant_lossless.tex`. Rebuild the submission bundle with:

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

GNU AFFERO GENERAL PUBLIC LICENSE v3 or later (see `LICENSE.md`). The repository carries the AGPLv3 network-source obligation and patent grant terms for `US Patent Pending #64/032,641`.

## Cite

```bibtex
@article{lam2026lamquant,
  title   = {LamQuant Lossless: A Real-Time, Bit-Exact, Wirelessly-Deployable EEG Compression Algorithm},
  author  = {Lam, Brian},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2026},
  note    = {Submitted}
}
```
