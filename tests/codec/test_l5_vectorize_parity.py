"""Lossless-owner vectorized entropy and warmup parity contracts."""

import numpy as np
import pytest


# ============================================================
# JIT'd entropy coders — bit-exact vs pure-Python references.
# encode_dense/decode_dense and rans encode/decode are JIT'd hot
# loops in ops/golomb.py and ops/rans.py.
# ============================================================

@pytest.mark.parametrize('n', [10, 100, 1250, 5000])
@pytest.mark.parametrize('vmax', [10, 1000, 100000])
@pytest.mark.parametrize('seed', [0, 42])
def test_golomb_dense_jit_byte_exact(n, vmax, seed):
    from lamquant_codec.ops.golomb import (
        encode_dense, decode_dense,
        _encode_dense_pyref, _decode_dense_pyref,
    )
    rng = np.random.default_rng(seed)
    data = rng.integers(-vmax, vmax, n, dtype=np.int64)
    ref_bytes = _encode_dense_pyref(data)
    new_bytes = encode_dense(data)
    assert ref_bytes == new_bytes, 'JIT encode_dense not byte-identical'
    ref_dec, _ = _decode_dense_pyref(ref_bytes)
    new_dec, _ = decode_dense(new_bytes)
    assert np.array_equal(ref_dec, new_dec), \
        f"n={n} vmax={vmax} seed={seed}: GR decode mismatch between ref and JIT"
    assert np.array_equal(new_dec, data), \
        f"n={n} vmax={vmax} seed={seed}: GR roundtrip broken"


def test_golomb_dense_jit_fuzz():
    """Fuzz: 1000 random int64 sequences."""
    from lamquant_codec.ops.golomb import (
        encode_dense, decode_dense,
        _encode_dense_pyref, _decode_dense_pyref,
    )
    rng = np.random.default_rng(20260422)
    for _ in range(1000):
        n = int(rng.integers(1, 5000))
        vmax = int(rng.integers(2, 1_000_000))
        data = rng.integers(-vmax, vmax, n, dtype=np.int64)
        ref_b = _encode_dense_pyref(data)
        new_b = encode_dense(data)
        if ref_b != new_b:
            pytest.fail(f'encode mismatch at n={n}, vmax={vmax}')
        new_d, _ = decode_dense(new_b)
        if not np.array_equal(new_d, data):
            pytest.fail(f'decode mismatch at n={n}, vmax={vmax}')


@pytest.mark.parametrize('n', [10, 100, 2528, 5000])
@pytest.mark.parametrize('L', [4, 8, 16, 32])
@pytest.mark.parametrize('seed', [0, 42])
def test_rans_jit_byte_exact(n, L, seed):
    from lamquant_codec.ops.rans import (
        compute_freq, encode_with_freq, decode,
        _encode_with_freq_pyref, _decode_pyref,
    )
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, L, n, dtype=np.int64)
    freq = compute_freq(syms, n_sym=L)
    ref_bytes = _encode_with_freq_pyref(syms, freq)
    new_bytes = encode_with_freq(syms, freq)
    assert ref_bytes == new_bytes, 'JIT rANS encode not byte-identical'
    ref_dec = _decode_pyref(ref_bytes, freq, n)
    new_dec = decode(new_bytes, freq, n)
    assert np.array_equal(ref_dec, new_dec), \
        f"n={n} L={L} seed={seed}: rANS decode mismatch between ref and JIT"
    assert np.array_equal(new_dec, syms), \
        f"n={n} L={L} seed={seed}: rANS roundtrip broken"


def test_rans_jit_fuzz():
    """Fuzz: 1000 random rANS streams."""
    from lamquant_codec.ops.rans import (
        compute_freq, encode_with_freq, decode,
        _encode_with_freq_pyref, _decode_pyref,
    )
    rng = np.random.default_rng(20260423)
    for _ in range(1000):
        n = int(rng.integers(1, 5000))
        L = int(rng.integers(2, 64))
        syms = rng.integers(0, L, n, dtype=np.int64)
        freq = compute_freq(syms, n_sym=L)
        ref_b = _encode_with_freq_pyref(syms, freq)
        new_b = encode_with_freq(syms, freq)
        if ref_b != new_b:
            pytest.fail(f'rANS encode mismatch at n={n}, L={L}')
        new_d = decode(new_b, freq, n)
        if not np.array_equal(new_d, syms):
            pytest.fail(f'rANS decode mismatch at n={n}, L={L}')


# ============================================================
# warm_jit() integration — verify it's a no-op safe to call repeatedly.
# ============================================================

def test_warm_jit_idempotent():
    """warm_jit() must be safe to call multiple times without error."""
    from lamquant_codec import warm_jit
    warm_jit()
    warm_jit()  # second call should be a near-no-op (cache hit)
