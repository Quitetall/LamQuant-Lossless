"""Single-source import of the Rust PyO3 wheel `lamquant_core`.

Every cross-language test that talks to Rust via the in-process wheel should
import from here:

    from tests.helpers.rust_bindings import (
        rust_compress, rust_decompress, HAS_RUST, requires_rust,
    )

This avoids per-file try/except blocks and gives one place to drop in a
`pytest.skip` reason when the wheel hasn't been built.
"""
from __future__ import annotations

import pytest

try:
    import lamquant_core as _rust
    HAS_RUST = True
    rust_compress = _rust.lml_compress      # (List[List[int]], noise_bits) -> bytes
    rust_decompress = _rust.lml_decompress  # (bytes) -> List[List[int]]
    rust_container_write = _rust.container_write
    rust_container_read = _rust.container_read
    rust_golomb_encode = _rust.golomb_encode_dense
    rust_golomb_decode = _rust.golomb_decode_dense
    rust_rans_encode = _rust.rans_encode
    rust_rans_decode = _rust.rans_decode
except ImportError:
    HAS_RUST = False
    _rust = None
    rust_compress = None
    rust_decompress = None
    rust_container_write = None
    rust_container_read = None
    rust_golomb_encode = None
    rust_golomb_decode = None
    rust_rans_encode = None
    rust_rans_decode = None


# Decorator-style skip for tests that require the Rust wheel.
# Use as: `pytestmark = requires_rust` at file level, or `@requires_rust`
# on individual test functions/classes.
requires_rust = pytest.mark.skipif(
    not HAS_RUST,
    reason="lamquant_core PyO3 wheel not installed (run `maturin develop --features python`)",
)


def to_rust_signal(sig) -> list[list[int]]:
    """Convert a numpy 2D array to the Vec<Vec<i64>> form Rust expects."""
    return [row.tolist() for row in sig]
