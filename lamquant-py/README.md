# lamquant-py

PyO3 + maturin bindings for `lamquant-lml`. Exposes Golomb, rANS, LML packet, container, LMA archive, and LMQC APIs as a native Python extension (`import lamquant_core`).

## Build

```bash
maturin develop --release -m codec-lossless/lamquant-py/Cargo.toml
```

## Usage

```python
import lamquant_core

# Round-trip a packet
data = lamquant_core.lml_compress([[1, 2, 3], [4, 5, 6]], noise_bits=0)
signal = lamquant_core.lml_decompress(data)

# Read one entry from an LMA archive without extracting
raw = lamquant_core.lma_read_entry("/data/corpus.lma", "recording.lml")
signal, meta = lamquant_core.container_read_bytes(raw)

# Random-access a single window (peak RSS = one window, not full file)
window, meta, n_windows = lamquant_core.container_read_window_np(raw, window_idx=0)
```

Python-to-Rust parity tests live here (`tests/`), not in `lamquant-lml`. Rust is the source of truth.
