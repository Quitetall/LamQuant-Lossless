# lamquant-lossless

Lossless EEG/biosignal codec in pure Rust. Integer Le Gall 5/3 lifting, per-subband LPC, adaptive Golomb-Rice entropy coding, bit-exact EDF/BDF round-trip. The same library crate (`lamquant_core`) builds for desktop (AVX2 SIMD parallel path) and bare-metal MCU (`no_std` + `alloc`; RP2350, ESP32-P4, STM32N6). Measured at **116× real-time on RP2350 RISC-V silicon** @ 150 MHz (86.3 ms/window, 0.60 Msa/s, 21 ch × 250 Hz); Verilator RTL sim: 119× / 83.7 ms.

Ships the `lml` CLI and TUI (`cargo build` default). Crate: `lamquant-lml`. Library: `lamquant_core`.

## Codec modes

| Mode | Flag | Reconstruction guarantee |
|------|------|--------------------------|
| Lossless | *(default)* | Integer-domain bit-exact; cross-backend byte-identical (enforced by golden-vector gate) |
| Near-lossless | `--max-error δ` | Closed-loop DPCM; `max\|orig − recon\| ≤ δ` per sample |
| Rate-controlled | `--target-bps X` | Transform-domain quantization with synthesis-gain-weighted bit allocation |

`.lma` archives pack a whole corpus into one file.

Ingest: EDF/BDF native. BrainVision / CNT / EEGLAB / DICOM via feature flag.
Export: NWB/HDF5, CSV, NPY, MAT (all feature-gated).

## Build

```bash
cargo build                                                               # CLI + TUI + EDF + LMA
cargo build --no-default-features --target riscv32imac-unknown-none-elf  # MCU firmware (no_std)
cargo test
```

| Feature | Adds |
|---------|------|
| `archive` | `.lma` file I/O |
| `cli` | `lml` binary |
| `tui` | ratatui TUI |
| `security` | AES-256-GCM encryption + Argon2id key derivation |
| `hdf5` / `parquet` / `s3` | export backends |
| `experimental_arithmetic` | range coding path (not yet stable) |
| `host` | all of the above |

## License

AGPL-3.0-or-later. Commercial license available.
