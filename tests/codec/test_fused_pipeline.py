"""Lossless-owner parity contract for the fused compressor."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFusedCompress:
    """fused_compress must be byte-identical to _compress_bytes."""

    def test_byte_identical(self, rng):
        from lamquant_codec.lossless import _compress_bytes_ref
        from lamquant_codec.ops.fused_lml import fused_compress, HAS_NUMBA
        if not HAS_NUMBA:
            pytest.skip("numba not available")

        for n_ch, T in [(1, 250), (4, 1250), (21, 2500)]:
            sig = rng.integers(-5000, 5000, (n_ch, T)).astype(np.float64)
            ref = _compress_bytes_ref(sig)
            fused = fused_compress(sig)
            assert ref == fused, f"Mismatch at {n_ch}ch x {T}"
