# lamquant-py

Python bindings for the LamQuant LML lossless EEG codec, built with PyO3 + maturin.

This crate is a thin cdylib wrapper over `lamquant-lml` (package alias `lamquant-core`). It exposes the same Golomb, rANS, LML packet, container, LMA archive, and LMQC APIs that the training dataloader and notebooks use.

## Build

```bash
# From codec-lossless/lamquant-py/
maturin develop --release
# Or from meta-repo root:
maturin develop --release -m codec-lossless/lamquant-py/Cargo.toml
```

## Usage

```python
import lamquant_core

# Compress/decompress LML packet
data = lamquant_core.lml_compress([[1, 2, 3], [4, 5, 6]], noise_bits=0)
signal = lamquant_core.lml_decompress(data)

# Read a file from an LMA archive without unpacking
entry = lamquant_core.lma_read_entry("/data/corpus.lma", "recording.lml")
signal, meta = lamquant_core.container_read_bytes(entry)

# Fast window-level random access
window, meta, n_windows = lamquant_core.container_read_window_np(entry, window_idx=0)
```

## Cross-language parity tests

Tests that verify Python-visible behaviour matches the Rust source of truth live here (not in `lamquant-lml`). Run with:

```bash
maturin develop && pytest tests/
```
