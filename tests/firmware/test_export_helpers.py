"""Unit tests for firmware/export/{crc,quantize,fsq} — Phase 1 quick win.

Pure-numpy helpers used by the firmware codegen pipeline. Shared between
the C and Rust emitters. No torch dependency for crc/quantize tests; the
FsqCalibration dataclass is tested without driving a model (the
`calibrate` function needs a real `.encode` so it's covered by integration
tests that ship a calibrated checkpoint).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from c_firmware.export.crc import crc32_of, crc32_of_file
from c_firmware.export.fsq import FsqCalibration, calibrate
from c_firmware.export.quantize import (
    cayley_rotation_q15,
    clamp_int8_weight,
    lsq_ternarize,
    pack_ternary_2bit,
    to_q7,
    to_q15,
    validate_native_ternary,
)

pytestmark = pytest.mark.l1


# ---------------------------------------------------------------------------
# crc.py
# ---------------------------------------------------------------------------
class TestCrc:
    def test_empty_iter_zero(self):
        assert crc32_of([]) == 0

    def test_single_buffer_matches_zlib(self):
        buf = b"hello world"
        assert crc32_of([buf]) == (zlib.crc32(buf) & 0xFFFFFFFF)

    def test_multi_buffer_matches_concat(self):
        bufs = [b"abc", b"def", b"ghi"]
        assert crc32_of(bufs) == (zlib.crc32(b"abcdefghi") & 0xFFFFFFFF)

    def test_output_is_uint32(self):
        out = crc32_of([b"x"])
        assert 0 <= out <= 0xFFFFFFFF

    def test_file_crc_matches_bytes(self, tmp_path: Path):
        data = b"firmware payload\x00\xff"
        p = tmp_path / "blob.bin"
        p.write_bytes(data)
        assert crc32_of_file(p) == (zlib.crc32(data) & 0xFFFFFFFF)

    def test_file_crc_empty(self, tmp_path: Path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert crc32_of_file(p) == 0


# ---------------------------------------------------------------------------
# quantize.py — to_q15 / to_q7
# ---------------------------------------------------------------------------
class TestToQ15:
    def test_one_maps_to_32767(self):
        out = to_q15(np.array([1.0]))
        assert out[0] == 32767

    def test_minus_one_maps_to_minus_32767(self):
        out = to_q15(np.array([-1.0]))
        assert out[0] == -32767

    def test_zero_maps_to_zero(self):
        assert to_q15(np.array([0.0]))[0] == 0

    def test_clips_above_one(self):
        assert to_q15(np.array([2.0]))[0] == 32767
        assert to_q15(np.array([-2.0]))[0] == -32767

    def test_returns_int16(self):
        assert to_q15(np.array([0.5])).dtype == np.int16

    def test_flattens(self):
        out = to_q15(np.array([[0.5, -0.5], [0.25, 0.0]]))
        assert out.ndim == 1
        assert len(out) == 4


class TestToQ7:
    def test_one_maps_to_127(self):
        assert to_q7(np.array([1.0]))[0] == 127

    def test_minus_one_maps_to_minus_127(self):
        assert to_q7(np.array([-1.0]))[0] == -127

    def test_returns_int8(self):
        assert to_q7(np.array([0.5])).dtype == np.int8

    def test_clip_behaviour(self):
        assert to_q7(np.array([5.0]))[0] == 127


# ---------------------------------------------------------------------------
# quantize.py — LSQ ternarize
# ---------------------------------------------------------------------------
class TestLsqTernarize:
    def test_scalar_alpha(self):
        w = np.array([0.5, -0.3, 0.0, 0.8, -1.2])
        out = lsq_ternarize(w, np.array(0.5))
        # w / 0.5 → [1, -0.6, 0, 1.6, -2.4] → clip [-1,1] → round
        assert out.tolist() == [1, -1, 0, 1, -1]

    def test_per_channel_alpha(self):
        w = np.array([[0.1, 0.5, 0.9], [0.4, 0.8, 1.5]])
        alpha = np.array([0.2, 0.5])  # per-row
        out = lsq_ternarize(w, alpha)
        assert out.shape == w.shape
        assert out.dtype == np.int8
        assert set(out.flatten().tolist()).issubset({-1, 0, 1})

    def test_returns_only_ternary_values(self):
        w = np.random.randn(50)
        out = lsq_ternarize(w, np.array(0.3))
        assert set(out.tolist()).issubset({-1, 0, 1})

    def test_alpha_zero_doesnt_div_by_zero(self):
        out = lsq_ternarize(np.array([1.0]), np.array(0.0))
        assert out[0] in (-1, 0, 1)


# ---------------------------------------------------------------------------
# quantize.py — clamp_int8_weight
# ---------------------------------------------------------------------------
class TestClampInt8Weight:
    def test_clamps_to_pm1(self):
        # Note: numpy rounds half-to-even, so 0.5 → 0 not 1.
        out = clamp_int8_weight(np.array([2.5, -2.5, 0.7, -0.7, 0.0]))
        assert out.tolist() == [1, -1, 1, -1, 0]

    def test_returns_int8(self):
        assert clamp_int8_weight(np.array([0.0])).dtype == np.int8


# ---------------------------------------------------------------------------
# quantize.py — pack_ternary_2bit + validate_native_ternary
# ---------------------------------------------------------------------------
class TestPackTernary2Bit:
    def test_basic_pack_4_weights(self):
        # weights {0, 1, -1, 0} → bit slots 00, 01, 10, 00
        # byte = (00<<6) | (10<<4) | (01<<2) | (00<<0) = 0b00100100 = 0x24
        packed = pack_ternary_2bit(np.array([0, 1, -1, 0], dtype=np.int8))
        assert len(packed) == 1
        assert int(packed[0]) == 0x24

    def test_pad_for_partial_byte(self):
        # 3 weights → still 1 byte
        packed = pack_ternary_2bit(np.array([1, 1, 1], dtype=np.int8))
        assert len(packed) == 1

    def test_8_weights_two_bytes(self):
        packed = pack_ternary_2bit(np.zeros(8, dtype=np.int8))
        assert len(packed) == 2
        assert (packed == 0).all()

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="non-ternary"):
            pack_ternary_2bit(np.array([2, 0, -1]))

    def test_returns_uint8(self):
        assert pack_ternary_2bit(np.array([0, 0])).dtype == np.uint8


class TestValidateNativeTernary:
    def test_accepts_valid(self):
        # All-zero is valid
        validate_native_ternary(np.array([0x00, 0x55], dtype=np.uint8))

    def test_rejects_0b11_pattern(self):
        # Byte 0x03 = slot0 has 0b11 (reserved)
        with pytest.raises(ValueError, match="NativeTernary violation"):
            validate_native_ternary(np.array([0x03], dtype=np.uint8))

    def test_rejects_0b11_in_higher_slot(self):
        # Byte 0xC0 = slot3 has 0b11
        with pytest.raises(ValueError, match="slot 3"):
            validate_native_ternary(np.array([0xC0], dtype=np.uint8))

    def test_round_trip_pack_then_validate(self):
        w = np.random.choice([-1, 0, 1], size=20).astype(np.int8)
        packed = pack_ternary_2bit(w)
        validate_native_ternary(packed)  # should not raise


# ---------------------------------------------------------------------------
# quantize.py — cayley rotation
# ---------------------------------------------------------------------------
class TestCayleyRotation:
    def test_returns_int16(self):
        a = np.random.randn(4, 4)
        out = cayley_rotation_q15(a)
        assert out.dtype == np.int16

    def test_output_length_dim_squared(self):
        a = np.random.randn(4, 4)
        out = cayley_rotation_q15(a)
        assert len(out) == 16

    def test_zero_input_is_identity_q15(self):
        a = np.zeros((3, 3))
        out = cayley_rotation_q15(a)
        # I = (I-0)(I+0)^-1 = I → diagonal of Q15-encoded I should be 32767
        assert out[0] == 32767  # I[0,0]
        assert out[4] == 32767  # I[1,1]
        assert out[8] == 32767  # I[2,2]
        assert out[1] == 0      # I[0,1]


# ---------------------------------------------------------------------------
# fsq.py — FsqCalibration dataclass + calibrate
# ---------------------------------------------------------------------------
class TestFsqCalibration:
    def test_dataclass_frozen(self):
        c = FsqCalibration(num_levels=4, total_freq=4096,
                            freq=[1024, 1024, 1024, 1024],
                            start=[0, 1024, 2048, 3072],
                            vmin_q31=-1000, vmax_q31=1000,
                            inv_range_q31=12345, entropy_bps=2.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            c.num_levels = 99

    def test_calibrate_with_stub_model(self):
        import torch

        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def encode(self, x, quantize=False):
                # Return [B, 32, 79] random latent
                return torch.randn(x.shape[0], 32, 79)

        m = StubModel()
        cal = calibrate(m, n_samples=3, input_shape=(2, 21, 250),
                        num_levels=8, total_freq=2048)
        assert cal.num_levels == 8
        assert cal.total_freq == 2048
        assert sum(cal.freq) == 2048
        assert len(cal.freq) == 8
        assert len(cal.start) == 8
        assert cal.start[0] == 0
        assert cal.entropy_bps > 0

    def test_calibrate_entropy_bounded(self):
        import torch

        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def encode(self, x, quantize=False):
                return torch.randn(x.shape[0], 32, 79)

        cal = calibrate(StubModel(), n_samples=3, input_shape=(2, 21, 250),
                        num_levels=16, total_freq=4096)
        assert 0 < cal.entropy_bps <= np.log2(16) + 0.01
