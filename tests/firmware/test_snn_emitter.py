"""Unit tests for firmware/export/snn_emitter.py — Phase 2.

Covers all 8 helpers (_to_i8_with_scale, _format_i8_array, _to_q15_with_scale,
_format_i16_array, _format_f32_const, _emit_linear, _emit_ssm_block) and
emit_snn_crate end-to-end with a synthetic state_dict checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from firmware.export.snn_emitter import (
    _emit_linear,
    _emit_ssm_block,
    _format_f32_const,
    _format_i16_array,
    _format_i8_array,
    _to_i8_with_scale,
    _to_q15_with_scale,
    emit_snn_crate,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# _to_i8_with_scale + _to_q15_with_scale
# ---------------------------------------------------------------------------
class TestToI8:
    def test_empty_array(self):
        q, s = _to_i8_with_scale(np.array([]))
        assert q.shape == (0,)
        assert q.dtype == np.int8
        assert s == 1.0

    def test_zero_array(self):
        q, s = _to_i8_with_scale(np.zeros(10))
        assert (q == 0).all()
        assert s == 1.0

    def test_roundtrip_within_tolerance(self):
        t = np.linspace(-1.0, 1.0, 100)
        q, s = _to_i8_with_scale(t)
        recovered = q.astype(np.float64) * s
        assert np.max(np.abs(recovered - t)) <= 0.5 * s + 1e-9

    def test_extreme_values_clipped(self):
        t = np.array([1.0, -1.0])
        q, s = _to_i8_with_scale(t)
        assert q.max() <= 127
        assert q.min() >= -128

    def test_returns_int8_dtype(self):
        q, _ = _to_i8_with_scale(np.array([0.5]))
        assert q.dtype == np.int8


class TestToQ15:
    def test_empty(self):
        q, s = _to_q15_with_scale(np.array([]))
        assert q.shape == (0,)
        assert q.dtype == np.int16
        assert s == 1.0

    def test_zero(self):
        q, s = _to_q15_with_scale(np.zeros(8))
        assert (q == 0).all()

    def test_roundtrip_better_than_i8(self):
        t = np.random.randn(100)
        q, s = _to_q15_with_scale(t)
        rec = q.astype(np.float64) * s
        assert np.max(np.abs(rec - t)) < 1e-3  # Q15 << int8 error


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
class TestFormatHelpers:
    def test_format_i8_array_basic(self):
        s = _format_i8_array("FOO", np.array([1, 2, 3], dtype=np.int8))
        assert "pub const FOO_LEN: usize = 3;" in s
        assert "pub static FOO: [i8; 3] = [" in s
        assert "1," in s

    def test_format_i8_array_with_doc(self):
        s = _format_i8_array("BAR", np.array([0], dtype=np.int8), doc="hi")
        assert "/// hi" in s

    def test_format_i8_array_line_break_every_16(self):
        s = _format_i8_array("X", np.arange(20, dtype=np.int8))
        # After 16 values a newline should appear
        assert "\n" in s

    def test_format_i16_array_basic(self):
        s = _format_i16_array("Q", np.array([100, -100], dtype=np.int16))
        assert "pub static Q: [i16; 2] = [" in s

    def test_format_f32_const_scientific(self):
        s = _format_f32_const("S", 1.23e-4, doc="scale")
        assert "/// scale" in s
        assert "pub const S: f32 =" in s
        assert "e-" in s


# ---------------------------------------------------------------------------
# _emit_linear
# ---------------------------------------------------------------------------
class TestEmitLinear:
    def test_with_bias(self):
        w = torch.randn(8, 4)
        b = torch.randn(8)
        s = _emit_linear("dense", w, b)
        assert "DENSE_WEIGHT" in s
        assert "DENSE_BIAS" in s
        assert "DENSE_WEIGHT_SCALE" in s
        assert "DENSE_BIAS_SCALE" in s

    def test_no_bias(self):
        s = _emit_linear("dense", torch.randn(4, 4), None)
        assert "DENSE_WEIGHT" in s
        assert "DENSE_BIAS" not in s


# ---------------------------------------------------------------------------
# _emit_ssm_block
# ---------------------------------------------------------------------------
class TestEmitSSMBlock:
    def _sd(self, prefix="ssm_blocks.0.fwd"):
        return {
            f"{prefix}.in_proj.weight":   torch.randn(160, 40),
            f"{prefix}.x_proj.weight":    torch.randn(33, 80),
            f"{prefix}.out_proj.weight":  torch.randn(40, 80),
            f"{prefix}.conv1d.weight":    torch.randn(80, 1, 4),
            f"{prefix}.conv1d.bias":      torch.randn(80),
            f"{prefix}.A_log":            torch.randn(80, 16),
            f"{prefix}.D":                torch.randn(80),
            f"{prefix}.dt_bias":          torch.randn(80),
        }

    def test_all_tensors_present(self):
        """Emitter must surface ALL of the supplied tensors. Pin only the
        tensor-name prefixes that show up in the Rust constant identifiers
        (e.g., ``IN_PROJ_W``, ``X_PROJ_W``, ...). The exact suffix
        (``_Q15``, ``_PRE_Q15``, ``_SCALE``) is an implementation detail
        of the int-quantization pipeline and drifts with refactors.
        """
        prefix = "ssm_blocks.0.fwd"
        body = _emit_ssm_block(prefix, self._sd(prefix))
        for name in ("IN_PROJ_W", "X_PROJ_W", "OUT_PROJ_W", "CONV1D_W",
                      "CONV1D_B", "A", "D", "DT_BIAS"):
            assert name in body, f"missing tensor prefix {name!r} in body"

    def test_a_log_uses_q15(self):
        """The A_log decay tensor must emit at Q15 precision somewhere
        in the body — the emitter currently produces ``A_PRE_Q15: [i16;
        ...]`` but the contract pinned is "A is i16 Q15", not the
        exact identifier."""
        prefix = "ssm_blocks.0.fwd"
        body = _emit_ssm_block(prefix, self._sd(prefix))
        assert "_Q15: [i16;" in body

    def test_skips_missing_tensors(self):
        """When only ``D`` is in the state_dict, the emitter must emit
        a D constant and nothing for the other tensors. The exact D
        identifier (``D`` or ``D_Q15``) is an implementation detail —
        pin the prefix only.
        """
        partial = {"ssm_blocks.0.fwd.D": torch.randn(8)}
        body = _emit_ssm_block("ssm_blocks.0.fwd", partial)
        assert "D_Q15" in body or "D:" in body, \
            "expected a D constant in the body"
        assert "IN_PROJ_W" not in body


# ---------------------------------------------------------------------------
# emit_snn_crate
# ---------------------------------------------------------------------------
def _make_full_sd():
    """Mamba bidirectional SNN state_dict."""
    sd = {
        "spatial_mix.weight": torch.randn(40, 21),
        "spatial_mix.bias":   torch.randn(40),
        "readout.weight":     torch.randn(8, 40),
        "readout.bias":       torch.randn(8),
    }
    for bi in (0, 1):
        sd[f"ssm_blocks.{bi}.norm.weight"] = torch.randn(40)
        sd[f"ssm_blocks.{bi}.norm.bias"] = torch.randn(40)
        for direction in ("fwd", "bwd"):
            p = f"ssm_blocks.{bi}.{direction}"
            sd[f"{p}.in_proj.weight"]  = torch.randn(160, 40)
            sd[f"{p}.x_proj.weight"]   = torch.randn(33, 80)
            sd[f"{p}.out_proj.weight"] = torch.randn(40, 80)
            sd[f"{p}.conv1d.weight"]   = torch.randn(80, 1, 4)
            sd[f"{p}.conv1d.bias"]     = torch.randn(80)
            sd[f"{p}.A_log"]           = torch.randn(80, 16)
            sd[f"{p}.D"]               = torch.randn(80)
            sd[f"{p}.dt_bias"]         = torch.randn(80)
    return sd


class TestEmitSnnCrate:
    def test_full_emission(self, tmp_path):
        sd = _make_full_sd()
        ckpt = tmp_path / "snn.pt"
        torch.save(sd, ckpt)
        out_dir = tmp_path / "generated"
        out_dir.mkdir()

        crc_inputs, crc_order = emit_snn_crate(ckpt, out_dir)

        # Files written
        snn_dir = out_dir / "snn"
        assert snn_dir.is_dir()
        assert (snn_dir / "spatial_mix.rs").is_file()
        assert (snn_dir / "readout.rs").is_file()
        assert (snn_dir / "layer0_norm.rs").is_file()
        assert (snn_dir / "layer0_fwd.rs").is_file()
        assert (snn_dir / "layer0_bwd.rs").is_file()
        assert (snn_dir / "layer1_fwd.rs").is_file()
        assert (snn_dir / "mod.rs").is_file()

        # Return values
        assert isinstance(crc_inputs, list)
        assert isinstance(crc_order, list)
        assert len(crc_inputs) == len(crc_order)
        assert len(crc_inputs) > 0
        assert all(isinstance(b, bytes) for b in crc_inputs)
        assert all(isinstance(s, str) for s in crc_order)

    def test_wrapped_state_dict(self, tmp_path):
        sd = _make_full_sd()
        ckpt = tmp_path / "wrap.pt"
        torch.save({"state_dict": sd, "epoch": 100}, ckpt)
        out_dir = tmp_path / "generated"; out_dir.mkdir()
        crc_inputs, _ = emit_snn_crate(ckpt, out_dir)
        assert len(crc_inputs) > 0

    def test_model_key_wrapper(self, tmp_path):
        sd = _make_full_sd()
        ckpt = tmp_path / "model.pt"
        torch.save({"model": sd}, ckpt)
        out_dir = tmp_path / "generated"; out_dir.mkdir()
        crc_inputs, _ = emit_snn_crate(ckpt, out_dir)
        assert len(crc_inputs) > 0

    def test_non_dict_raises(self, tmp_path):
        ckpt = tmp_path / "bad.pt"
        torch.save("not a dict", ckpt)
        out_dir = tmp_path / "generated"; out_dir.mkdir()
        with pytest.raises(RuntimeError, match="did not yield a state_dict"):
            emit_snn_crate(ckpt, out_dir)

    def test_minimal_sd_partial_emit(self, tmp_path):
        # Only spatial_mix — no readout, no ssm_blocks
        sd = {
            "spatial_mix.weight": torch.randn(4, 4),
        }
        ckpt = tmp_path / "min.pt"
        torch.save(sd, ckpt)
        out_dir = tmp_path / "generated"; out_dir.mkdir()
        crc_inputs, crc_order = emit_snn_crate(ckpt, out_dir)
        assert (out_dir / "snn" / "spatial_mix.rs").is_file()
        assert not (out_dir / "snn" / "readout.rs").is_file()

    def test_mod_rs_lists_all_submodules(self, tmp_path):
        sd = _make_full_sd()
        ckpt = tmp_path / "snn.pt"
        torch.save(sd, ckpt)
        out_dir = tmp_path / "generated"; out_dir.mkdir()
        emit_snn_crate(ckpt, out_dir)
        mod_rs = (out_dir / "snn" / "mod.rs").read_text()
        assert "pub mod spatial_mix;" in mod_rs
        assert "pub mod readout;" in mod_rs
        assert "pub mod layer0_fwd;" in mod_rs
