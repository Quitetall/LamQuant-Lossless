# lamquant-py

PyO3 + maturin bindings for `lamquant-lml`.

## ABIR dataset-view API

`lamquant_core.AbirDatasetView` is the high-level Python entrypoint for authenticated ABIR round-trip.

```python
import lamquant_core

signal = ...  # np.ndarray[int64], shape [channels, samples], C-contiguous
view = lamquant_core.AbirDatasetView.from_numpy(
    signal,
    sample_rate_hz=256.0,
    window_size=128,
    metadata_json='{"subject":"fixture"}',
    channel_names=["C3", "C4"],
)
```

One authenticated decode:
- `open_abir(bytes_or_bytearray)` parses and authenticates BCS2 LML bytes into a verified `AbirDatasetView`.
- `AbirDatasetView.from_lml(bytes_or_bytearray)` is the same authenticated entrypoint as a class method.

Zero-copy Python views are intentionally stable:
- `numpy_view()` returns a read-only NumPy int64 array with stable backing pointer across repeated calls.
- `payload_pointer()` returns that same pointer as an integer.
- `lml_bytes()` returns the retained authenticated BCS2 byte object with the same Python object identity on each call.

Canonical identity and JSON are explicit:
- `content_id` exposes canonical dataset identity.
- `canonical_json()` returns canonicalized ABIR JSON for exact identity comparison.
- `numpy_view()` values remain equal to the authenticated decoded samples, and `lml_bytes()` preserves the exact opened or newly encoded BCS2 bytes.

Round-trip helpers:
- `view.shape`, `view.n_channels`, `view.total_samples`, `view.n_windows`, `view.packet_sample_counts`, and `view.sample_rate_hz` expose parsed dataset metadata.
- `metadata_json` must encode a JSON object; malformed JSON and non-object roots fail closed.

Build:

```bash
maturin develop --release -m lamquant-py/Cargo.toml
```

## Compatibility and low-level/legacy APIs

For compatibility and lower-level control, this crate also exposes packet and container primitives plus LMA helpers:

- `lml_compress`, `lml_decompress`
- `container_read_bytes`, `container_read_window_np`
- `lma_read_entry`
- `lmqc_*` helpers

Use these only when low-level format operations are required. ABIR dataset-view paths are above. Rust remains source of truth for canonical codec behavior.
