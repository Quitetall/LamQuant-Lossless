"""Unit tests for ai_models.decoder.vocos_decoder — Phase 2 fill-in.

Covers the helper modules (SubPixelShuffle1d, GRN, SpatialCoherenceLayer,
Snake1d, SEBlock, InceptionDWConv1d, ConvNeXtBlock, ISTFTHead),
loss helpers (band_weighted_reconstruction_loss, per_band_prd,
anti_wrapping_phase_loss), and end-to-end VocosDecoder forward for the
two smallest tiers (Tier 1 dim=32, Tier 2 dim=64).
"""
from __future__ import annotations

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ai_models.decoder.vocos_decoder import (
    ConvNeXtBlock,
    GRN,
    InceptionDWConv1d,
    ISTFTHead,
    SEBlock,
    Snake1d,
    SpatialCoherenceLayer,
    SubPixelShuffle1d,
    VocosDecoder,
    anti_wrapping_phase_loss,
    band_weighted_reconstruction_loss,
    per_band_prd,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# SubPixelShuffle1d
# ---------------------------------------------------------------------------
class TestSubPixelShuffle1d:
    def test_doubles_time_halves_channels(self):
        m = SubPixelShuffle1d(2)
        x = torch.randn(2, 8, 10)
        y = m(x)
        assert y.shape == (2, 4, 20)

    def test_factor_four(self):
        m = SubPixelShuffle1d(4)
        x = torch.randn(1, 16, 5)
        y = m(x)
        assert y.shape == (1, 4, 20)

    def test_non_divisible_channels_raises(self):
        m = SubPixelShuffle1d(3)
        x = torch.randn(1, 5, 4)
        with pytest.raises(AssertionError):
            m(x)

    def test_preserves_total_elements(self):
        m = SubPixelShuffle1d(2)
        x = torch.randn(3, 8, 7)
        y = m(x)
        assert y.numel() == x.numel()


# ---------------------------------------------------------------------------
# GRN
# ---------------------------------------------------------------------------
class TestGRN:
    def test_zero_init_returns_input_residual_only(self):
        # gamma, beta init to 0 → output = x (residual)
        m = GRN(dim=8)
        x = torch.randn(2, 8, 16)
        y = m(x)
        assert torch.allclose(y, x, atol=1e-6)

    def test_preserves_shape(self):
        m = GRN(dim=32)
        x = torch.randn(4, 32, 100)
        assert m(x).shape == x.shape

    def test_gamma_nonzero_alters_output(self):
        m = GRN(dim=8)
        m.gamma.data.fill_(1.0)
        x = torch.randn(2, 8, 16)
        y = m(x)
        assert not torch.allclose(y, x)


# ---------------------------------------------------------------------------
# band_weighted_reconstruction_loss
# ---------------------------------------------------------------------------
class TestBandWeightedLoss:
    def test_identical_signals_zero(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 256)
        loss = band_weighted_reconstruction_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_nonidentical(self):
        x = torch.randn(2, 4, 256)
        y = torch.randn(2, 4, 256)
        assert band_weighted_reconstruction_loss(x, y).item() > 0

    def test_returns_scalar(self):
        x = torch.randn(2, 4, 256)
        y = torch.randn(2, 4, 256)
        out = band_weighted_reconstruction_loss(x, y)
        assert out.ndim == 0

    def test_gradient_flows(self):
        x = torch.randn(2, 4, 256, requires_grad=True)
        y = torch.randn(2, 4, 256)
        loss = band_weighted_reconstruction_loss(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# per_band_prd
# ---------------------------------------------------------------------------
class TestPerBandPRD:
    def test_returns_5_bands(self):
        x = torch.randn(1, 4, 256)
        y = torch.randn(1, 4, 256)
        d = per_band_prd(x, y)
        assert set(d.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_identical_returns_zero(self):
        x = torch.randn(1, 4, 256)
        d = per_band_prd(x, x.clone())
        for k, v in d.items():
            assert v == pytest.approx(0.0, abs=1e-6)

    def test_all_values_finite(self):
        x = torch.randn(1, 4, 1024)
        y = torch.randn(1, 4, 1024)
        d = per_band_prd(x, y)
        for v in d.values():
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# SpatialCoherenceLayer
# ---------------------------------------------------------------------------
class TestSpatialCoherenceLayer:
    def test_zero_gate_is_identity(self):
        m = SpatialCoherenceLayer(n_channels=21, dim=16)
        x = torch.randn(2, 16, 32)
        y = m(x)
        # gate init to zero → output = x
        assert torch.allclose(y, x, atol=1e-6)

    def test_nonzero_gate_alters_output(self):
        m = SpatialCoherenceLayer(n_channels=21, dim=16)
        m.gate.data.fill_(1.0)
        x = torch.randn(2, 16, 32)
        y = m(x)
        assert not torch.allclose(y, x)

    def test_preserves_shape(self):
        m = SpatialCoherenceLayer(n_channels=21, dim=8)
        x = torch.randn(1, 8, 50)
        assert m(x).shape == x.shape


# ---------------------------------------------------------------------------
# Snake1d
# ---------------------------------------------------------------------------
class TestSnake1d:
    def test_basic_forward_preserves_shape(self):
        m = Snake1d(channels=8)
        x = torch.randn(2, 8, 16)
        assert m(x).shape == x.shape

    def test_zero_input_returns_zero(self):
        m = Snake1d(channels=4)
        x = torch.zeros(1, 4, 8)
        y = m(x)
        assert torch.allclose(y, x, atol=1e-7)

    def test_gradient_flows(self):
        m = Snake1d(channels=4)
        x = torch.randn(1, 4, 16, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# SEBlock
# ---------------------------------------------------------------------------
class TestSEBlock:
    def test_zero_init_half_scaling(self):
        # Final layer zero-init → sigmoid(0) = 0.5 → x * 0.5
        m = SEBlock(dim=8)
        x = torch.randn(2, 8, 16)
        y = m(x)
        assert torch.allclose(y, x * 0.5, atol=1e-6)

    def test_preserves_shape(self):
        m = SEBlock(dim=16, reduction=4)
        x = torch.randn(3, 16, 32)
        assert m(x).shape == x.shape


# ---------------------------------------------------------------------------
# InceptionDWConv1d
# ---------------------------------------------------------------------------
class TestInceptionDWConv1d:
    def test_preserves_shape(self):
        m = InceptionDWConv1d(channels=16, short_kernel=3, band_kernel=11,
                              branch_ratio=0.25)
        x = torch.randn(2, 16, 32)
        assert m(x).shape == x.shape

    def test_split_sizes_correct(self):
        m = InceptionDWConv1d(channels=16, branch_ratio=0.125)
        # gc = 16 * 0.125 = 2 → split_sizes = (2, 2, 12)
        assert m.split_sizes == (2, 2, 12)

    def test_identity_branch_passes_through(self):
        m = InceptionDWConv1d(channels=8, branch_ratio=0.125)
        # Zero out both conv branches so output channels 2: → identity
        for conv in (m.dwconv_short, m.dwconv_long):
            conv.weight.data.zero_()
            conv.bias.data.zero_()
        x = torch.randn(1, 8, 16)
        y = m(x)
        # Identity branch is channels 2: (gc=1, gc=1, identity=6)
        assert torch.allclose(y[:, 2:, :], x[:, 2:, :], atol=1e-6)


# ---------------------------------------------------------------------------
# ConvNeXtBlock — exercise major flag combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {},
    {"use_grn": True},
    {"use_snake": True},
    {"use_se": True},
    {"use_inception": True},
    {"layer_scale_init": 0.5},
    {"dilation": 2},
    {"use_snake": True, "use_grn": True},
])
class TestConvNeXtBlock:
    def test_forward_preserves_shape(self, kw):
        m = ConvNeXtBlock(dim=8, intermediate_dim=16, **kw)
        x = torch.randn(2, 8, 32)
        y = m(x)
        assert y.shape == x.shape

    def test_residual_when_pwconv2_zero(self, kw):
        m = ConvNeXtBlock(dim=8, intermediate_dim=16, **kw)
        m.pwconv2.weight.data.zero_()
        m.pwconv2.bias.data.zero_()
        x = torch.randn(2, 8, 32)
        y = m(x)
        # With pwconv2 → 0, residual branch is + layer_scale * 0 = + 0
        # so y == residual == x (modulo SE if enabled)
        if not kw.get("use_se"):
            assert torch.allclose(y, x, atol=1e-5)


# ---------------------------------------------------------------------------
# anti_wrapping_phase_loss
# ---------------------------------------------------------------------------
class TestAntiWrappingPhaseLoss:
    def test_identical_zero(self):
        x = torch.randn(1, 4, 128)
        loss = anti_wrapping_phase_loss(x, x.clone(), n_fft=32, hop_length=8)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self):
        x = torch.randn(1, 4, 256)
        y = torch.randn(1, 4, 256)
        loss = anti_wrapping_phase_loss(x, y, n_fft=64, hop_length=8)
        assert loss.item() > 0

    def test_returns_scalar(self):
        x = torch.randn(1, 4, 128)
        out = anti_wrapping_phase_loss(x, x.clone() + 0.01,
                                        n_fft=32, hop_length=8)
        assert out.ndim == 0


# ---------------------------------------------------------------------------
# ISTFTHead
# ---------------------------------------------------------------------------
class TestISTFTHead:
    def test_output_shape(self):
        m = ISTFTHead(dim=32, n_channels=21, n_fft=16, hop_length=8,
                      output_length=2500)
        x = torch.randn(2, 32, 312)
        y = m(x)
        assert y.shape == (2, 21, 2500)

    def test_explicit_output_length(self):
        m = ISTFTHead(dim=16, n_channels=4, n_fft=8, hop_length=4,
                      output_length=128)
        x = torch.randn(1, 16, 32)
        y = m(x)
        assert y.shape == (1, 4, 128)

    def test_finite_output(self):
        m = ISTFTHead(dim=16, n_channels=2, n_fft=8, hop_length=4,
                      output_length=64)
        x = torch.randn(1, 16, 16)
        y = m(x)
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# VocosDecoder — end-to-end on small tiers (Tier 1, Tier 2)
# ---------------------------------------------------------------------------
class TestVocosDecoderTiers:
    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="tier must be one of"):
            VocosDecoder(tier=99)

    def test_tier1_forward_shape(self):
        m = VocosDecoder(tier=1)
        x = torch.randn(2, 32, 79)
        y = m(x)
        assert y.shape == (2, 21, 313)

    def test_tier2_forward_shape(self):
        m = VocosDecoder(tier=2)
        x = torch.randn(1, 32, 79)
        y = m(x)
        assert y.shape == (1, 21, 313)

    def test_param_count_positive(self):
        m = VocosDecoder(tier=1)
        assert m.param_count() > 0

    def test_repr_contains_tier(self):
        m = VocosDecoder(tier=1)
        s = repr(m)
        assert "tier=1" in s
        assert "params=" in s

    def test_tier_3_construction_only(self):
        # Tier 3+ is too slow on CPU; just construct + check param count.
        m = VocosDecoder(tier=3)
        assert m.dim == 256
        assert m.output_mode == "istft"
        assert m.param_count() > 0

    def test_spatial_gate_present_on_tier5(self):
        m = VocosDecoder(tier=5)
        assert m.spatial_gate is not None
        assert m.spatial_proj is not None

    def test_spatial_gate_absent_on_tier1(self):
        m = VocosDecoder(tier=1)
        assert m.spatial_gate is None
        assert m.spatial_proj is None

    def test_detail_conditioning_path_constructs(self):
        m = VocosDecoder(tier=1, detail_conditioning=True)
        # FiLM modules created
        assert len(m.film_scale) == m.n_blocks
        assert len(m.film_shift) == m.n_blocks

    def test_detail_conditioning_forward(self):
        m = VocosDecoder(tier=1, detail_conditioning=True)
        x = torch.randn(1, 32, 79)
        details = {
            "l1_detail": torch.randn(1, 21, 1250),
            "l2_detail": torch.randn(1, 21, 625),
            "l3_detail": torch.randn(1, 21, 312),
        }
        y = m(x, details=details)
        assert y.shape == (1, 21, 313)

    def test_gradient_checkpointing_path(self):
        m = VocosDecoder(tier=1, gradient_checkpointing=True)
        m.train()
        x = torch.randn(1, 32, 79, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# VocosDecoder.from_pretrained
# ---------------------------------------------------------------------------
class TestFromPretrained:
    def test_loads_raw_state_dict(self, tmp_path):
        m = VocosDecoder(tier=1)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save(m.state_dict(), ckpt_path)
        loaded = VocosDecoder.from_pretrained(str(ckpt_path), tier=1)
        assert isinstance(loaded, VocosDecoder)
        assert not loaded.training  # eval mode

    def test_loads_wrapped_state_dict(self, tmp_path):
        m = VocosDecoder(tier=1)
        ckpt_path = tmp_path / "ckpt_wrapped.pt"
        torch.save({"state_dict": m.state_dict()}, ckpt_path)
        loaded = VocosDecoder.from_pretrained(str(ckpt_path), tier=1)
        assert isinstance(loaded, VocosDecoder)
