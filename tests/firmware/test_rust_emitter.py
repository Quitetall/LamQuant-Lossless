"""Unit tests for firmware/export/rust_emitter.py — Phase 2."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from firmware.export.checkpoint import LoadedCheckpoint
from firmware.export.fsq import FsqCalibration
from firmware.export.rust_emitter import (
    EmitContext,
    Int8LayerData,
    RustEmitter,
    TernaryLayerData,
    _find_conv,
    _find_norm,
    _module_at,
    extract_int8_layer,
    extract_ternary_layer,
)
from firmware.export.schema import ArchSpec, ResolvedLayer, load_schema

pytestmark = pytest.mark.l2


_REPO = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO / "firmware" / "export_schema.toml"


# ---------------------------------------------------------------------------
# Stub model classes
# ---------------------------------------------------------------------------
class _TernaryConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, with_norm=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k))
        self.lsq_alpha = nn.Parameter(torch.tensor([0.1] * out_ch).reshape(out_ch, 1, 1))
        if with_norm:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)


class _Int8Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k, with_qs=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k) * 0.1)
        if with_qs:
            self.quant_scale = nn.Parameter(torch.tensor([1 / 127.0] * out_ch))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.premix = _TernaryConv(21, 21, 1, with_norm=False)
        self.focal2 = nn.Module()
        self.focal2.conv = _TernaryConv(21, 8, 3, with_norm=True)
        self.bneck_v = _Int8Conv(8, 32, 1)


# ---------------------------------------------------------------------------
# _module_at / _find_conv / _find_norm
# ---------------------------------------------------------------------------
class TestModuleHelpers:
    def test_module_at_top_level(self):
        m = _Model()
        assert _module_at(m, "premix") is m.premix

    def test_module_at_dotted(self):
        m = _Model()
        assert _module_at(m, "focal2.conv") is m.focal2.conv

    def test_module_at_missing_returns_none(self):
        m = _Model()
        assert _module_at(m, "does.not.exist") is None

    def test_find_conv_via_layer_name(self):
        m = _Model()
        c = _find_conv(m, "premix")
        assert c is m.premix

    def test_find_conv_via_dot_conv(self):
        m = _Model()
        c = _find_conv(m, "focal2")
        assert c is m.focal2.conv

    def test_find_conv_returns_none_if_missing(self):
        m = _Model()
        assert _find_conv(m, "bogus") is None

    def test_find_norm_via_dot_norm(self):
        m = _Model()
        n = _find_norm(m, "focal2.conv")
        assert isinstance(n, nn.GroupNorm)

    def test_find_norm_returns_none_when_missing(self):
        m = _Model()
        assert _find_norm(m, "premix") is None


# ---------------------------------------------------------------------------
# extract_ternary_layer + extract_int8_layer
# ---------------------------------------------------------------------------
def _rl(name, in_c, out_c, k=3, weight_kind="ternary_2bit_packed",
         has_norm=False, has_alphas=True, groups=None):
    return ResolvedLayer(
        name=name, in_channels=in_c, out_channels=out_c, kernel_size=k,
        stride=1, weight_kind=weight_kind, has_alphas=has_alphas,
        has_norm=has_norm, has_shortcut=False, groups=groups,
        has_quant_scale=False,
    )


class TestExtractTernary:
    def test_basic_extraction(self):
        m = _Model()
        data = extract_ternary_layer(m, _rl("premix", 21, 21, k=1))
        assert isinstance(data, TernaryLayerData)
        assert isinstance(data.packed, list)
        assert all(0 <= b <= 255 for b in data.packed)
        assert data.n_weights == 21 * 21 * 1
        assert data.norm_weight_q7 is None

    def test_with_norm(self):
        m = _Model()
        # Spec name = focal2.conv → _find_conv resolves to focal2.conv,
        # _find_norm looks for focal2.conv.norm (present on _TernaryConv).
        data = extract_ternary_layer(m,
            _rl("focal2.conv", 21, 8, k=3, has_norm=True))
        assert data.norm_weight_q7 is not None
        assert data.norm_bias_q15 is not None
        assert len(data.norm_weight_q7) == 8

    def test_missing_layer_raises(self):
        m = _Model()
        with pytest.raises(KeyError):
            extract_ternary_layer(m, _rl("missing", 1, 1))

    def test_no_alpha_raises(self):
        m = _Model()
        # bneck_v is Int8Conv (no lsq_alpha) but extracted as ternary spec
        with pytest.raises(ValueError, match="no lsq_alpha"):
            extract_ternary_layer(m, _rl("bneck_v", 8, 32, k=1))

    def test_packed_bytes_round_trips_to_bytes(self):
        m = _Model()
        data = extract_ternary_layer(m, _rl("premix", 21, 21, k=1))
        assert isinstance(data.packed_bytes, bytes)
        assert len(data.packed_bytes) == len(data.packed)


class TestExtractInt8:
    def test_basic_extraction(self):
        m = _Model()
        data = extract_int8_layer(m, _rl("bneck_v", 8, 32, k=1,
                                          weight_kind="int8",
                                          has_alphas=False))
        assert isinstance(data, Int8LayerData)
        assert len(data.weights) == 8 * 32 * 1
        # Default scales: 1/127 → 32767/127 ≈ 258
        assert len(data.scales_q15) == 32

    def test_with_quant_scale(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.bneck_v = _Int8Conv(8, 32, 1, with_qs=True)
        data = extract_int8_layer(
            _M(), _rl("bneck_v", 8, 32, k=1, weight_kind="int8",
                       has_alphas=False))
        assert len(data.scales_q15) == 32

    def test_missing_raises(self):
        m = _Model()
        with pytest.raises(KeyError):
            extract_int8_layer(m, _rl("nope", 1, 1, weight_kind="int8"))


# ---------------------------------------------------------------------------
# RustEmitter helpers (no full emit yet)
# ---------------------------------------------------------------------------
def _make_ckpt(tmp_path):
    p = tmp_path / "fake.ckpt"
    p.write_bytes(b"fake-bytes")
    return LoadedCheckpoint(
        path=p, sha256="ab" * 32, state_dict={},
        arch_name="subband_v1", grade="gold",
    )


def _make_emitter(tmp_path):
    if not _SCHEMA_PATH.is_file():
        pytest.skip("schema not present")
    schema = load_schema(_SCHEMA_PATH)
    ckpt = _make_ckpt(tmp_path)
    arch_name = next(iter(schema.architectures))
    crate_root = tmp_path / "crate"
    crate_root.mkdir()
    return RustEmitter(schema, ckpt, crate_root, arch_name=arch_name), schema, arch_name


class TestRustEmitterHelpers:
    def test_construction(self, tmp_path):
        em, _, arch = _make_emitter(tmp_path)
        assert em.arch_name == arch

    def test_git_commit_returns_string(self, tmp_path):
        em, _, _ = _make_emitter(tmp_path)
        commit = em._git_commit()
        assert isinstance(commit, str)

    def test_git_commit_handles_no_git(self, tmp_path):
        em, _, _ = _make_emitter(tmp_path)
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            assert em._git_commit() == "unknown"

    def test_common_ctx_keys(self, tmp_path):
        em, schema, arch = _make_emitter(tmp_path)
        ctx = EmitContext(schema=schema, arch_name=arch,
                           ckpt=em.ckpt, crate_root=em.crate_root,
                           git_commit="abc", timestamp="t",
                           export_timestamp_unix=0)
        c = em._common_ctx(ctx)
        assert c["arch_name"] == arch
        assert c["schema_version"] == schema.schema_version
        assert c["ckpt_sha256"] == em.ckpt.sha256


# ---------------------------------------------------------------------------
# Full emit() end-to-end with synthetic model + FSQ cal + SNN ckpt
# ---------------------------------------------------------------------------
def _build_full_model_from_schema(schema, arch_name):
    """Build an nn.Module that exposes every conv layer the schema declares."""
    arch = schema.get_arch(arch_name)
    m = nn.Module()
    for spec in schema.resolved_layers(arch_name):
        # Build a stub matching the resolved shape
        if spec.weight_kind == "ternary_2bit_packed":
            layer = _TernaryConv(spec.in_channels, spec.out_channels,
                                  spec.kernel_size,
                                  with_norm=spec.has_norm)
        elif spec.weight_kind == "int8":
            layer = _Int8Conv(spec.in_channels, spec.out_channels,
                               spec.kernel_size,
                               with_qs=spec.has_quant_scale)
        else:
            continue
        # Place under m using the schema's dotted name
        parts = spec.name.split(".")
        cur = m
        for p in parts[:-1]:
            if not hasattr(cur, p):
                setattr(cur, p, nn.Module())
            cur = getattr(cur, p)
        # The conv module wraps the actual weight; place at either `.conv`
        # or as the top-level layer attribute.
        setattr(cur, parts[-1], layer)
    return m


class TestEmitEndToEnd:
    def test_emit_writes_all_files(self, tmp_path):
        if not _SCHEMA_PATH.is_file():
            pytest.skip("schema not present")
        schema = load_schema(_SCHEMA_PATH)
        arch_name = next(iter(schema.architectures))

        # Build LoadedCheckpoint with rotation param in state_dict
        rot_param = schema.rotation.source_param
        sd = {rot_param: torch.randn(schema.rotation.dim,
                                      schema.rotation.dim)}
        ckpt_path = tmp_path / "ck.ckpt"
        ckpt_path.write_bytes(b"fake")
        ckpt = LoadedCheckpoint(path=ckpt_path, sha256="cd" * 32,
                                 state_dict=sd, arch_name=arch_name,
                                 grade="gold")

        crate_root = tmp_path / "crate"
        crate_root.mkdir()
        em = RustEmitter(schema, ckpt, crate_root, arch_name=arch_name)
        model = _build_full_model_from_schema(schema, arch_name)

        # FSQ calibration stub
        fsq_cal = FsqCalibration(
            num_levels=16, total_freq=4096,
            freq=[256] * 16, start=[i * 256 for i in range(16)],
            vmin_q31=-1000, vmax_q31=1000,
            inv_range_q31=12345, entropy_bps=4.0,
        )

        out = em.emit(model, fsq_cal=fsq_cal, snn_ckpt=None)
        assert out == crate_root
        gen = crate_root / "src" / "generated"
        assert gen.is_dir()
        assert (gen / "fsq.rs").is_file()
        assert (gen / "toeplitz.rs").is_file()
        assert (gen / "crc.rs").is_file()
        assert (gen / "mod.rs").is_file()
        assert (gen / "rotation.rs").is_file()
        assert (crate_root / "src" / "metadata.rs").is_file()
        assert (crate_root / ".exportlock.json").is_file()

    def test_emit_without_fsq_warns(self, tmp_path, capsys):
        if not _SCHEMA_PATH.is_file():
            pytest.skip("schema not present")
        schema = load_schema(_SCHEMA_PATH)
        arch_name = next(iter(schema.architectures))
        ckpt_path = tmp_path / "ck.ckpt"
        ckpt_path.write_bytes(b"fake")
        ckpt = LoadedCheckpoint(path=ckpt_path, sha256="cd" * 32,
                                 state_dict={}, arch_name=arch_name,
                                 grade="gold")
        crate_root = tmp_path / "crate"
        crate_root.mkdir()
        em = RustEmitter(schema, ckpt, crate_root, arch_name=arch_name)
        model = _build_full_model_from_schema(schema, arch_name)

        em.emit(model, fsq_cal=None, snn_ckpt=None)
        err = capsys.readouterr().err
        assert "No FSQ calibration" in err

    def test_emit_skips_missing_layers(self, tmp_path, capsys):
        if not _SCHEMA_PATH.is_file():
            pytest.skip("schema not present")
        schema = load_schema(_SCHEMA_PATH)
        arch_name = next(iter(schema.architectures))
        ckpt_path = tmp_path / "ck.ckpt"
        ckpt_path.write_bytes(b"fake")
        ckpt = LoadedCheckpoint(path=ckpt_path, sha256="cd" * 32,
                                 state_dict={}, arch_name=arch_name,
                                 grade="gold")
        crate_root = tmp_path / "crate"
        crate_root.mkdir()
        em = RustEmitter(schema, ckpt, crate_root, arch_name=arch_name)

        # Empty model — every layer is missing
        empty_model = nn.Module()
        em.emit(empty_model, fsq_cal=None, snn_ckpt=None)
        err = capsys.readouterr().err
        assert "[skip]" in err
