"""Pure quantization helpers — shared by both C and Rust emitters.

All functions take numpy arrays and return numpy arrays. No I/O, no globals,
no dependencies on torch (model is unpacked into ndarrays before calling).
"""
from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────────────
# Q-format scaling
# ────────────────────────────────────────────────────────────────────


def to_q15(values: np.ndarray) -> np.ndarray:
    """Float [-1, +1] → int16 Q15 [-32767, +32767]."""
    flat = np.asarray(values, dtype=np.float64).flatten()
    clipped = np.clip(flat, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def to_q7(values: np.ndarray) -> np.ndarray:
    """Float [-1, +1] → int8 Q7 [-127, +127]."""
    flat = np.asarray(values, dtype=np.float64).flatten()
    clipped = np.clip(flat, -1.0, 1.0)
    return np.round(clipped * 127.0).astype(np.int8)


# ────────────────────────────────────────────────────────────────────
# LSQ ternary quantization
# ────────────────────────────────────────────────────────────────────


def lsq_ternarize(weight: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """LSQ quantize a float weight tensor to ternary {-1, 0, +1}.

    Args:
        weight: float weight tensor of any shape.
        alpha:  float LSQ alpha — scalar or per-output-channel array.
    Returns:
        int8 array of same shape as `weight`, values in {-1, 0, +1}.
    """
    w = np.asarray(weight, dtype=np.float64)
    a = np.abs(np.asarray(alpha, dtype=np.float64))
    # Broadcast: alpha is per-output-channel for [out, in, k] conv weights.
    # Match training-time shape: alpha is [out, 1, 1] → broadcasts naturally.
    if a.ndim == 0:
        a = a.reshape(1)
    a_safe = a + 1e-8
    # LSQ forward: round(clamp(w / alpha, -1, 1))
    return np.round(np.clip(w / a_safe.reshape(*a.shape, *([1] * (w.ndim - a.ndim))), -1.0, 1.0)).astype(np.int8)


def clamp_int8_weight(weight: np.ndarray) -> np.ndarray:
    """Non-ternary: clamp + round to int8 in [-1, +1] then scale to int8 range."""
    w = np.asarray(weight, dtype=np.float64)
    return np.round(np.clip(w, -1.0, 1.0)).astype(np.int8)


# ────────────────────────────────────────────────────────────────────
# Ternary 2-bit packing (NativeTernary)
# ────────────────────────────────────────────────────────────────────


def pack_ternary_2bit(values: np.ndarray) -> np.ndarray:
    """Pack ternary weights {-1, 0, +1} into 2-bit-per-weight bytes.

    Encoding (per 2-bit slot):
        00 = 0
        01 = +1
        10 = -1
        11 = ERROR (reserved, never generated; firmware decodes as 0)

    Order: weight[i] occupies bits `(i % 4) * 2` of byte `i // 4`. So byte 0
    holds weights 0..3 with weight 0 in the LOW bits.

    Returns:
        uint8 array of length ceil(len(values) / 4).

    Raises:
        ValueError: if any input value is outside {-1, 0, +1}.
    """
    flat = np.asarray(values, dtype=np.int8).flatten()
    invalid = np.sum((flat != -1) & (flat != 0) & (flat != 1))
    if invalid > 0:
        raise ValueError(
            f"pack_ternary_2bit: {invalid} non-ternary values in input "
            "(expected {-1, 0, +1})"
        )

    n = len(flat)
    packed = np.zeros((n + 3) // 4, dtype=np.uint8)
    for i in range(n):
        v = int(flat[i])
        bits = 0b00 if v == 0 else (0b01 if v == 1 else 0b10)
        packed[i // 4] |= bits << ((i % 4) * 2)
    return packed


def validate_native_ternary(packed: np.ndarray) -> None:
    """Verify no byte contains the reserved 0b11 pattern in any 2-bit slot.

    Raises:
        ValueError: with offset of the offending byte/slot.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    for i, b in enumerate(arr):
        for slot in range(4):
            if ((int(b) >> (slot * 2)) & 0b11) == 0b11:
                raise ValueError(
                    f"NativeTernary violation: byte {i} slot {slot}: "
                    f"0x{int(b):02X} contains 0b11 (reserved error pattern)"
                )


# ────────────────────────────────────────────────────────────────────
# Cayley rotation
# ────────────────────────────────────────────────────────────────────


def cayley_rotation_q15(rotation_a: np.ndarray) -> np.ndarray:
    """Compute the Cayley rotation Q = (I-A)(I+A)^{-1} from a generator A.

    A is enforced skew-symmetric (A := A - A.T). Q is then a proper rotation
    matrix. Output is row-major Q15 (int16 array of length dim*dim).
    """
    a = np.asarray(rotation_a, dtype=np.float64)
    a_skew = a - a.T
    eye = np.eye(a_skew.shape[0])
    q = np.linalg.solve(eye + a_skew, eye - a_skew)
    return to_q15(q.flatten())
