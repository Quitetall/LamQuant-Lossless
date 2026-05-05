"""Canonical signal generators for test fixtures.

Provides deterministic, labeled signal generators for adversarial, boundary,
and synthetic EEG test data. Uses np.random.default_rng (local state, not
the deprecated np.random.seed global state).
"""
import numpy as np


def synth_signal(n_ch: int, T: int, *, seed: int = 0,
                 amp: int = 8000, dtype=np.int64) -> np.ndarray:
    """Reproducible integer-domain signal of shape (n_ch, T).

    Used by all cross-language and roundtrip tests where the input must be
    integer-valued (LML wire format encodes int samples). Replaces inline
    `synth_signal` definitions and ad-hoc `np.random.seed(42); randn(...)`
    patterns scattered across the test suite.

    Args:
        n_ch: number of channels.
        T: samples per channel.
        seed: RNG seed (independent per call — uses default_rng).
        amp: maximum absolute value (default 8000, well within int16).
        dtype: output dtype (default int64; pass np.int16/int32 if needed).

    Returns:
        ndarray of shape (n_ch, T) with values in [-amp, amp).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(-amp, amp, size=(n_ch, T), dtype=dtype)


def make_synthetic_eeg(n_channels=21, n_samples=2500, seed=42):
    """Generate synthetic EEG-like signal with realistic spectral content.

    Band-limited, autocorrelated signal that compresses well with the
    lossless codec (CR >= 3). Used as fallback when real q31_events data
    is unavailable.

    Returns float32 array of shape (n_channels, n_samples).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 250.0
    seg = np.zeros((n_channels, n_samples), dtype=np.float32)
    for c in range(n_channels):
        seg[c] = (40 * np.sin(2 * np.pi * 10 * t + c * 0.1) +
                  30 * np.sin(2 * np.pi * 3 * t + c * 0.2) +
                  15 * np.sin(2 * np.pi * 6 * t + c * 0.3) +
                  rng.standard_normal(n_samples) * 2).astype(np.float32)
    return seg


def adversarial_signals(lengths=None):
    """Yield (label, signal) for canonical adversarial inputs.

    These cover all known edge cases for the lifting DWT + LPC + entropy
    coding pipeline: DC, Dirac, alternating max, boundary values, etc.
    """
    if lengths is None:
        lengths = [8, 100, 313, 625, 2500, 2501]

    for n in lengths:
        yield f'zeros_{n}', np.zeros((21, n), dtype=np.float64)
        yield f'ones_{n}', np.ones((21, n), dtype=np.float64)

        sig = np.full((21, n), 32767.0)
        yield f'max_positive_{n}', sig

        sig = np.full((21, n), -32768.0)
        yield f'max_negative_{n}', sig

        sig = np.zeros((21, n), dtype=np.float64)
        sig[:, ::2] = 32767
        sig[:, 1::2] = -32768
        yield f'alternating_max_{n}', sig

        sig = np.zeros((21, n), dtype=np.float64)
        sig[0, n // 2] = 32767
        yield f'single_spike_{n}', sig

        rng = np.random.default_rng(42)
        yield f'gaussian_{n}', rng.standard_normal((21, n)) * 5000

        sig = np.zeros((21, n), dtype=np.float64)
        for ch in range(21):
            sig[ch] = np.linspace(-10000, 10000, n)
        yield f'linear_ramp_{n}', sig

        yield f'dc_offset_{n}', np.full((21, n), 15000.0)

        sig = np.zeros((21, n), dtype=np.float64)
        sig[:, :n//2] = -10000
        sig[:, n//2:] = 10000
        yield f'step_function_{n}', sig
