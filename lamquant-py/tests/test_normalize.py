"""S7b (ADR 0069): the `normalize_eeg_f32` pyfunction is bit-exact to the Python
`decode_lma_signal` DSP tail (resample→250 → 0.5 Hz zero-phase HP → Q31 → f32).

Run after building the extension (`maturin develop`); skips when unbuilt.
"""
from math import gcd

import pytest

lc = pytest.importorskip("lamquant_core")
np = pytest.importorskip("numpy")
signal = pytest.importorskip("scipy.signal")


def _py_norm(data_f32, orig_sr):
    """Faithful replica of decode_lma_signal's tail (post channel-select)."""
    data = data_f32
    if abs(orig_sr - 250.0) > 0.5:
        data = data.astype(np.float64)  # S7b f64 fix
        up, down = 250, int(orig_sr)
        g = gcd(up, down)
        up, down = up // g, down // g
        data = signal.resample_poly(data, up, down, axis=1)
    sos = signal.butter(2, 0.5, btype="high", fs=250.0, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=1)
    max_abs = float(np.max(np.abs(data)))
    if max_abs < 1e-12:
        return None
    gain = 0.72 / max_abs
    q31 = (data * gain * 2147483647.0).astype(np.int32)
    return (q31.astype(np.float32) / 2147483647.0) * 1000.0


def _synth(n_ch, t):
    return np.array(
        [[float(((c * 37 + tt * 5) % 4001) - 2000) for tt in range(t)] for c in range(n_ch)],
        dtype=np.float32,
    )


@pytest.mark.parametrize("orig_sr,t", [(250.0, 300), (200.0, 240), (500.0, 600), (1000.0, 1200)])
def test_normalize_eeg_f32_bit_exact_to_python(orig_sr, t):
    x = _synth(21, t)
    py = _py_norm(x.copy(), orig_sr)
    rust = lc.normalize_eeg_f32(x, orig_sr)
    assert rust is not None and py is not None
    assert rust.shape == py.shape
    assert rust.dtype == np.float32
    # bit-exact: Rust f64 DSP == scipy f64 DSP, and the f32 round-trip matches numpy.
    assert np.array_equal(rust, py), f"max|Δ|={np.max(np.abs(rust.astype(np.float64) - py.astype(np.float64)))}"


def test_flat_signal_returns_none():
    flat = np.zeros((21, 512), dtype=np.float32)
    assert lc.normalize_eeg_f32(flat, 250.0) is None


def test_fft_branch_rate_raises_for_python_fallback():
    # 257 Hz: gcd(250,257)=1 → down=257 > 256 → FFT branch (not ported to Rust).
    with pytest.raises(NotImplementedError):
        lc.normalize_eeg_f32(_synth(21, 128), 257.0)
