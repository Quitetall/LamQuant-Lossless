# lamquant-lml

Lossless (and near-lossless / rate-controlled lossy) codec for EEG and other
biomedical waveforms, in pure Rust. Ships the `lml` command-line tool + an
interactive TUI, and a library (`lamquant_core`) usable on desktop **and**
bare-metal MCUs (`no_std` + `alloc`).

## What it does

- **Lossless**: integer Le Gall 5/3 lifting DWT → per-subband LPC → Golomb-Rice,
  bit-exact round-trip of EDF/BDF (and BrainVision/CNT/EEGLAB/DICOM via ingest).
  Cross-backend byte-identical (desktop AVX2 path == firmware scalar path),
  enforced by a golden-vector conformance gate.
- **Near-lossless** (`--max-error δ`): closed-loop DPCM with a guaranteed
  per-sample bound `max|orig − recon| ≤ δ`.
- **Rate-controlled lossy** (`--target-bps X`): transform-domain quantization +
  synthesis-gain-weighted bit allocation to a bits-per-sample ceiling.
- **Archives** (`.lma`): pack a whole corpus into one file (rayon-parallel,
  SHA-256 verified, streaming v2 format).
- **Containers**: EDF/BDF in; NWB/HDF5 + CSV/NPY/MAT export (feature-gated).

## Deployment tiers

The container self-describes its codec mode + tier (`lossless` / near-lossless /
target-BPS; `mcu` / `basestation`), so a decoder reconstructs whatever an
encoder produced. The lossless + near-lossless + target-BPS decode paths are
integer-only and compile for `no_std` MCU targets (e.g. RP2350 RISC-V); the
crate's lossless encoder has run bit-exact on real silicon.

## Build

```
cargo build                      # full host (CLI + TUI + EDF + LMA)
cargo build --no-default-features --target riscv32imac-unknown-none-elf  # firmware
cargo test
```

Feature flags gate every heavy capability (`host`, `hdf5`, `parquet`,
`experimental_arithmetic`, …) so the firmware build stays small.

## License

AGPL-3.0-or-later. A commercial license is available — see the repository.
