# lamquant-lml

Lossless EEG/biosignal codec in pure Rust. Integer Le Gall 5/3 lifting → per-subband LPC → Golomb-Rice, bit-exact EDF/BDF round-trip. The same library crate (`lamquant_core`) builds for desktop (AVX2 parallel path) and bare-metal MCU (`no_std` + `alloc` — runs on RP2350 RISC-V silicon).

Ships the `lml` CLI and an interactive TUI (`cargo build` default).

## Codec modes

| Mode | Flag | Guarantee |
|---|---|---|
| Lossless | *(default)* | byte-exact reconstruction; cross-backend byte-identical (enforced by golden-vector gate) |
| Near-lossless | `--max-error δ` | closed-loop DPCM; `max\|orig − recon\| ≤ δ` per sample |
| Rate-controlled | `--target-bps X` | transform-domain quantization with synthesis-gain-weighted bit allocation |

`.lma` archives pack a whole corpus into one file — rayon-parallel encode, SHA-256 per entry, streaming footer (no 2× disk blowup).

Ingest: EDF/BDF native; BrainVision/CNT/EEGLAB/DICOM via feature flag. Export: NWB/HDF5, CSV, NPY, MAT (all feature-gated).

## Build

```
cargo build                                                               # CLI + TUI + EDF + LMA
cargo build --no-default-features --target riscv32imac-unknown-none-elf  # MCU firmware
cargo test
```

Feature flags: `archive` (file I/O), `cli`, `tui`, `security` (AES-GCM + Argon2), `hdf5`, `parquet`, `s3`, `experimental_arithmetic`. `host` enables all of the above.

## License

AGPL-3.0-or-later. Commercial license available — see the repository.
