"""
L2/L5 — Ternary weight packing: firmware 2-bit encoding + Q31 alpha + rANS tables.

The firmware encodes ternary weights as 2-bit values packed 4-per-byte:
  00 = 0, 01 = +1, 10 = -1, 11 = reserved
A packing bug silently corrupts every weight the firmware uses.

Also validates Q31 alpha scaling precision (6+ decimal digits), alpha
overflow bounds, and rANS frequency table invariants (sum + monotonicity).
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

# The same LUT the C firmware uses (ternary_mac.c line 18)
TERNARY_LUT = [0, 1, -1, 0]


def pack_ternary_weights(w_np):
    """Reimplement the packing logic from export_firmware.py lines 110-118."""
    packed = []
    for i in range(0, len(w_np), 4):
        byte = 0
        for j in range(4):
            if i + j < len(w_np):
                val = w_np[i + j]
                bits = 0b00 if val == 0 else (0b01 if val == 1 else 0b10)
                byte |= (bits << (2 * j))
        packed.append(byte)
    return packed


def unpack_ternary_weights(packed, total_weights):
    """Unpack using the same LUT as C firmware."""
    result = []
    for i in range(total_weights):
        byte_idx = i // 4
        bit_shift = (i % 4) * 2
        bits = (packed[byte_idx] >> bit_shift) & 0x03
        result.append(TERNARY_LUT[bits])
    return result


@pytest.mark.l2
@pytest.mark.l5
class TestTernaryPacking:
    def test_roundtrip_all_zeros(self):
        w = np.zeros(16, dtype=np.int8)
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, len(w))
        assert unpacked == [0] * 16

    def test_roundtrip_all_plus_one(self):
        w = np.ones(16, dtype=np.int8)
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, len(w))
        assert unpacked == [1] * 16

    def test_roundtrip_all_minus_one(self):
        w = np.full(16, -1, dtype=np.int8)
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, len(w))
        assert unpacked == [-1] * 16

    def test_roundtrip_mixed_pattern(self):
        w = np.array([1, -1, 0, 1, -1, 0, 0, 1], dtype=np.int8)
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, len(w))
        np.testing.assert_array_equal(unpacked, w)

    def test_roundtrip_non_multiple_of_4(self):
        """The remainder path (export_firmware.py lines 111-118)."""
        for length in [1, 2, 3, 5, 7, 13, 15]:
            np.random.seed(length)
            w = np.random.choice([-1, 0, 1], size=length).astype(np.int8)
            packed = pack_ternary_weights(w)
            unpacked = unpack_ternary_weights(packed, length)
            np.testing.assert_array_equal(unpacked, w,
                                          err_msg=f"Failed for length {length}")

    def test_packing_byte_values(self):
        """Verify exact byte values for the KAT vector from ternary_mac.c."""
        # From ternary_mac.c line 112: weights [1, -1, 0, 1]
        # Expected byte: 0x01 | (0x02 << 2) | (0x00 << 4) | (0x01 << 6) = 0x49
        w = np.array([1, -1, 0, 1], dtype=np.int8)
        packed = pack_ternary_weights(w)
        assert packed[0] == 0x49

    def test_large_random_roundtrip(self):
        """Round-trip 10k weights (typical layer size)."""
        np.random.seed(0)
        w = np.random.choice([-1, 0, 1], size=10000).astype(np.int8)
        packed = pack_ternary_weights(w)
        unpacked = unpack_ternary_weights(packed, len(w))
        np.testing.assert_array_equal(unpacked, w)


@pytest.mark.l5
class TestQ31Scaling:
    def test_alpha_q31_no_overflow(self):
        """Alpha values should be small enough that alpha * Q31_MAX fits int32."""
        Q31_MAX = 2147483647
        # Realistic alpha range from training
        for alpha_val in [0.001, 0.01, 0.1, 0.5, 0.99]:
            result = int(alpha_val * Q31_MAX)
            assert -(2**31) <= result < 2**31, \
                f"Alpha {alpha_val} overflows int32: {result}"

    def test_alpha_q31_precision(self):
        """Q31 conversion should preserve at least 6 decimal digits."""
        Q31_MAX = 2147483647
        alpha = 0.123456
        q31_val = int(alpha * Q31_MAX)
        recovered = q31_val / Q31_MAX
        assert abs(recovered - alpha) < 1e-6


@pytest.mark.l2
class TestFSQRansTable:
    def test_frequency_table_sums_to_total(self):
        """Build a synthetic frequency table and verify invariants."""
        num_levels = 16
        total_freq = 4096
        # Simulate a distribution
        np.random.seed(42)
        counts = np.random.randint(1, 1000, size=num_levels)
        total = counts.sum()

        freq = np.maximum(1, (counts / total * total_freq).astype(np.int32))
        diff = total_freq - freq.sum()
        freq[np.argmax(freq)] += diff

        assert freq.sum() == total_freq
        assert np.all(freq >= 1)

    def test_cumulative_starts_monotonic(self):
        """Cumulative start table must be strictly monotonic."""
        num_levels = 16
        total_freq = 4096
        np.random.seed(42)
        counts = np.random.randint(1, 1000, size=num_levels)
        total = counts.sum()

        freq = np.maximum(1, (counts / total * total_freq).astype(np.int32))
        diff = total_freq - freq.sum()
        freq[np.argmax(freq)] += diff

        start = np.zeros(num_levels, dtype=np.int32)
        for i in range(1, num_levels):
            start[i] = start[i - 1] + freq[i - 1]

        # Starts must be monotonically increasing
        assert np.all(np.diff(start) > 0)
        # Last start + last freq = total
        assert start[-1] + freq[-1] == total_freq
