"""Unit tests for firmware/export/schema.py — Phase 1 quick win.

Tests the typed dataclasses (ArchSpec, LayerSpec, ResolvedLayer, FsqSpec,
RotationSpec, ToeplitzSpec, SnnTensorSpec, SnnSpec, Schema) and the
`load_schema` / `validate_schema` loaders against the real
firmware/export_schema.toml that ships in-tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from firmware.export.schema import (
    ArchSpec,
    FsqSpec,
    LayerSpec,
    ResolvedLayer,
    RotationSpec,
    Schema,
    SnnSpec,
    SnnTensorSpec,
    ToeplitzSpec,
    _resolve_dim,
    load_schema,
    validate_schema,
)

pytestmark = pytest.mark.l1


_REPO = Path(__file__).resolve().parents[2]
_REAL_SCHEMA = _REPO / "firmware" / "export_schema.toml"


# ---------------------------------------------------------------------------
# _resolve_dim
# ---------------------------------------------------------------------------
def _arch(width=128):
    return ArchSpec(
        name="x", display_name="X", encoder_class="C",
        encoder_width=width, n_focal_blocks=3,
        latent_dims=32, latent_timesteps=79,
        checkpoint_globs=[],
    )


class TestResolveDim:
    def test_int_passthrough(self):
        assert _resolve_dim(64, _arch()) == 64

    def test_arch_width_substitutes(self):
        assert _resolve_dim("_arch_width", _arch(width=216)) == 216

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown dimension placeholder"):
            _resolve_dim("bogus", _arch())


# ---------------------------------------------------------------------------
# Dataclasses are frozen
# ---------------------------------------------------------------------------
class TestDataclassFrozen:
    def test_arch_spec_frozen(self):
        a = _arch()
        with pytest.raises(Exception):
            a.encoder_width = 999

    def test_layer_spec_frozen(self):
        ls = LayerSpec(name="l", in_channels=4, out_channels="_arch_width",
                       kernel_size=3, stride=1, weight_kind="int8",
                       has_alphas=False, has_norm=False, has_shortcut=False)
        with pytest.raises(Exception):
            ls.kernel_size = 99


# ---------------------------------------------------------------------------
# LayerSpec.resolve + ResolvedLayer.n_weights / n_packed_bytes
# ---------------------------------------------------------------------------
class TestLayerResolve:
    def test_resolve_replaces_placeholders(self):
        ls = LayerSpec(name="conv1", in_channels=21,
                       out_channels="_arch_width",
                       kernel_size=7, stride=2,
                       weight_kind="ternary_2bit_packed",
                       has_alphas=True, has_norm=True, has_shortcut=False)
        rl = ls.resolve(_arch(width=216))
        assert isinstance(rl, ResolvedLayer)
        assert rl.in_channels == 21
        assert rl.out_channels == 216
        assert rl.kernel_size == 7

    def test_resolve_with_groups(self):
        ls = LayerSpec(name="dw", in_channels="_arch_width",
                       out_channels="_arch_width", kernel_size=3, stride=1,
                       weight_kind="ternary_2bit_packed", has_alphas=True,
                       has_norm=True, has_shortcut=False,
                       groups="_arch_width")
        rl = ls.resolve(_arch(width=128))
        assert rl.groups == 128

    def test_resolve_no_groups(self):
        ls = LayerSpec(name="pw", in_channels=32, out_channels=64,
                       kernel_size=1, stride=1, weight_kind="int8",
                       has_alphas=False, has_norm=False, has_shortcut=False)
        rl = ls.resolve(_arch())
        assert rl.groups is None


class TestResolvedLayerWeights:
    def test_n_weights_dense(self):
        rl = ResolvedLayer(name="x", in_channels=4, out_channels=8,
                           kernel_size=3, stride=1, weight_kind="int8",
                           has_alphas=False, has_norm=False, has_shortcut=False,
                           groups=None, has_quant_scale=False)
        assert rl.n_weights == 4 * 8 * 3

    def test_n_weights_depthwise(self):
        rl = ResolvedLayer(name="x", in_channels=16, out_channels=16,
                           kernel_size=7, stride=1, weight_kind="int8",
                           has_alphas=False, has_norm=False, has_shortcut=False,
                           groups=16, has_quant_scale=False)
        # depthwise: out_channels * kernel_size
        assert rl.n_weights == 16 * 7

    def test_n_packed_bytes_ternary(self):
        rl = ResolvedLayer(name="x", in_channels=4, out_channels=8,
                           kernel_size=3, stride=1,
                           weight_kind="ternary_2bit_packed",
                           has_alphas=False, has_norm=False, has_shortcut=False,
                           groups=None, has_quant_scale=False)
        # n = 96 → bytes = (96+3)/4 = 24
        assert rl.n_packed_bytes == 24

    def test_n_packed_bytes_int8_zero(self):
        rl = ResolvedLayer(name="x", in_channels=4, out_channels=8,
                           kernel_size=3, stride=1, weight_kind="int8",
                           has_alphas=False, has_norm=False, has_shortcut=False,
                           groups=None, has_quant_scale=False)
        assert rl.n_packed_bytes == 0


# ---------------------------------------------------------------------------
# load_schema — against the real ship-tree file
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _REAL_SCHEMA.is_file(),
                     reason="firmware/export_schema.toml missing")
class TestLoadSchemaReal:
    def test_loads_without_error(self):
        s = load_schema(_REAL_SCHEMA)
        assert isinstance(s, Schema)
        assert s.schema_version == "1.0"

    def test_has_at_least_one_arch(self):
        s = load_schema(_REAL_SCHEMA)
        assert len(s.architectures) >= 1

    def test_has_at_least_one_layer(self):
        s = load_schema(_REAL_SCHEMA)
        assert len(s.layers) >= 1

    def test_get_arch_known(self):
        s = load_schema(_REAL_SCHEMA)
        first = next(iter(s.architectures))
        assert s.get_arch(first).name == first

    def test_get_arch_unknown_raises(self):
        s = load_schema(_REAL_SCHEMA)
        with pytest.raises(KeyError, match="Unknown architecture"):
            s.get_arch("no_such_arch")

    def test_resolved_layers_returns_list(self):
        s = load_schema(_REAL_SCHEMA)
        first = next(iter(s.architectures))
        rl = s.resolved_layers(first)
        assert isinstance(rl, list)
        assert all(isinstance(x, ResolvedLayer) for x in rl)
        assert len(rl) == len(s.layers)


# ---------------------------------------------------------------------------
# load_schema — error paths
# ---------------------------------------------------------------------------
class TestLoadSchemaErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_schema(tmp_path / "nope.toml")

    def test_wrong_version_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text('[meta]\nschema_version = "2.0"\n')
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            load_schema(p)

    def test_missing_version_raises(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text('[meta]\n')
        with pytest.raises(ValueError, match="Unsupported"):
            load_schema(p)


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _REAL_SCHEMA.is_file(),
                     reason="firmware/export_schema.toml missing")
class TestValidateSchema:
    def test_real_schema_passes(self, capsys):
        validate_schema(_REAL_SCHEMA)
        out = capsys.readouterr().out
        assert "[OK]" in out
        assert "valid" in out
