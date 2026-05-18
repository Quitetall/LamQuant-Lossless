"""Unit tests for ai_models.decoder.run_decoder_tier helpers — Phase 3.

Covers the three pure-torch metric helpers (`pearson_r_loss`, `pearson_r_batch`,
`prd_batch`). `main()` is gated by `--tier` argparse + frozen-student checkpoint
+ CUDA paths and is exercised separately via subprocess smoke (see Phase 5).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Module loader: run_decoder_tier.py pollutes sys.path on import and pulls in
# heavy dependencies (lamquant_codec, auraloss, raw_window_dataset). We stub
# the imports needed for module load, then exec the source. The metric helpers
# don't touch any of the stubbed modules.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(__file__).resolve().parents[2] / "ai_models" / "decoder" / "run_decoder_tier.py"


def _stub(name: str, **attrs) -> types.ModuleType:
    """Insert a stub module into sys.modules and return it."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUBBED = ("vocos_decoder", "discriminator", "lamquant_codec",
             "lamquant_codec.models", "lamquant_codec.models.encoder",
             "raw_window_dataset", "data_types", "auraloss", "auraloss.freq",
             "ai_models.decoder.run_decoder_tier_under_test")


@pytest.fixture(scope="module")
def rdt():
    """Load run_decoder_tier module with heavy deps stubbed out.

    Stubs are cleaned up at fixture finalisation so they don't bleed
    into later test modules (e.g. the real `data_types` is loaded by
    integration tests that import the joint codec).
    """
    pre_loaded = {n: sys.modules.get(n) for n in _STUBBED}

    # Stub heavy modules referenced at import-time
    if "vocos_decoder" not in sys.modules:
        m = _stub("vocos_decoder",
                  VocosDecoder=object,
                  anti_wrapping_phase_loss=lambda *a, **k: torch.tensor(0.0))
        sys.modules["vocos_decoder"] = m
    if "discriminator" not in sys.modules:
        _stub("discriminator", EEGDiscriminator=object)
    if "lamquant_codec" not in sys.modules:
        pkg = _stub("lamquant_codec")
        models = _stub("lamquant_codec.models")
        encoder_mod = _stub(
            "lamquant_codec.models.encoder",
            TernaryMobileNetV5_Subband=object,
            TernaryMobileNetV5_Subband_V2=object,
        )
        pkg.models = models  # type: ignore[attr-defined]
        models.encoder = encoder_mod  # type: ignore[attr-defined]
    if "raw_window_dataset" not in sys.modules:
        _stub("raw_window_dataset", RawWindowDataset=object)
    if "data_types" not in sys.modules:
        _stub("data_types", DatasetManifest=object, Split=object)
    if "auraloss" not in sys.modules:
        pkg = _stub("auraloss")
        freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
        pkg.freq = freq  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(
        "ai_models.decoder.run_decoder_tier_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    try:
        yield module
    finally:
        for name, prev in pre_loaded.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ---------------------------------------------------------------------------
# pearson_r_loss
# ---------------------------------------------------------------------------
class TestPearsonRLoss:
    def test_identical_signals_loss_zero(self, rdt):
        x = torch.randn(4, 21, 2500)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert torch.isfinite(loss)
        assert loss.item() < 1e-5

    def test_negated_signal_loss_two(self, rdt):
        x = torch.randn(2, 4, 100)
        loss = rdt.pearson_r_loss(x, -x)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_uncorrelated_loss_near_one(self, rdt):
        torch.manual_seed(42)
        a = torch.randn(8, 8, 1024)
        b = torch.randn(8, 8, 1024)
        loss = rdt.pearson_r_loss(a, b)
        # Random gaussians: |r| < 0.1 → loss in [0.9, 1.1]
        assert 0.5 < loss.item() < 1.5

    def test_gradient_flows(self, rdt):
        pred = torch.randn(3, 4, 16, requires_grad=True)
        target = torch.randn(3, 4, 16)
        loss = rdt.pearson_r_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0  # non-trivial gradient

    def test_returns_tensor(self, rdt):
        out = rdt.pearson_r_loss(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_2d_input_works(self, rdt):
        # Helper reshapes to (B, -1), so 2D works.
        x = torch.randn(5, 200)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert loss.item() < 1e-5

    def test_4d_input_works(self, rdt):
        x = torch.randn(3, 4, 8, 16)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert loss.item() < 1e-5

    def test_eps_protects_zero_variance(self, rdt):
        # All-constant target → tc = 0, denom would be 0 without eps.
        pred = torch.randn(2, 4, 16)
        target = torch.zeros(2, 4, 16)
        loss = rdt.pearson_r_loss(pred, target)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# pearson_r_batch (scalar float)
# ---------------------------------------------------------------------------
class TestPearsonRBatch:
    def test_identical_returns_one(self, rdt):
        x = torch.randn(4, 21, 2500)
        r = rdt.pearson_r_batch(x, x.clone())
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_negated_returns_minus_one(self, rdt):
        x = torch.randn(2, 4, 100)
        r = rdt.pearson_r_batch(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_returns_python_float(self, rdt):
        r = rdt.pearson_r_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(r, float)

    def test_consistent_with_pearson_r_loss(self, rdt):
        torch.manual_seed(0)
        pred = torch.randn(4, 8, 64)
        target = torch.randn(4, 8, 64)
        r = rdt.pearson_r_batch(pred, target)
        loss = rdt.pearson_r_loss(pred, target).item()
        assert loss == pytest.approx(1.0 - r, abs=1e-5)

    def test_3d_input(self, rdt):
        r = rdt.pearson_r_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert -1.0 <= r <= 1.0

    def test_random_uncorrelated_near_zero(self, rdt):
        torch.manual_seed(7)
        a = torch.randn(32, 21, 2500)
        b = torch.randn(32, 21, 2500)
        r = rdt.pearson_r_batch(a, b)
        assert abs(r) < 0.05  # n large → r → 0


# ---------------------------------------------------------------------------
# prd_batch (percent root-mean-square diff)
# ---------------------------------------------------------------------------
class TestPrdBatch:
    def test_identical_returns_zero(self, rdt):
        x = torch.randn(4, 21, 2500)
        p = rdt.prd_batch(x, x.clone())
        assert p == pytest.approx(0.0, abs=1e-5)

    def test_returns_python_float(self, rdt):
        p = rdt.prd_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(p, float)

    def test_nonneg(self, rdt):
        torch.manual_seed(1)
        for _ in range(5):
            p = rdt.prd_batch(torch.randn(4, 4, 100), torch.randn(4, 4, 100))
            assert p >= 0.0

    def test_known_value_pure_signal_vs_doubled(self, rdt):
        # pred = 2*target → diff = target → numerator = ||target||,
        # denom = ||target||, prd = 100%.
        target = torch.randn(4, 4, 100)
        pred = 2 * target
        p = rdt.prd_batch(pred, target)
        assert p == pytest.approx(100.0, rel=1e-4)

    def test_eps_protects_zero_target(self, rdt):
        pred = torch.randn(2, 4, 16)
        target = torch.zeros(2, 4, 16)
        p = rdt.prd_batch(pred, target)
        # Denom clamped at 1e-8, so result is a finite (very large) value.
        assert torch.isfinite(torch.tensor(p))

    def test_scale_invariance_in_ratio(self, rdt):
        # PRD invariant under target scaling when noise scales linearly.
        torch.manual_seed(3)
        target = torch.randn(4, 4, 100)
        noise = 0.1 * torch.randn_like(target)
        p1 = rdt.prd_batch(target + noise, target)
        p2 = rdt.prd_batch(10 * (target + noise), 10 * target)
        assert p1 == pytest.approx(p2, rel=1e-4)


# ---------------------------------------------------------------------------
# Module-level smoke: ensure constants/helpers don't raise on weird shapes.
# ---------------------------------------------------------------------------
class TestEdgeShapes:
    def test_batch_size_one(self, rdt):
        x = torch.randn(1, 4, 16)
        assert rdt.pearson_r_loss(x, x.clone()).item() < 1e-5
        assert rdt.pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)
        assert rdt.prd_batch(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_dtype_float32(self, rdt):
        x = torch.randn(2, 4, 16, dtype=torch.float32)
        assert rdt.pearson_r_loss(x, x.clone()).dtype == torch.float32

    def test_dtype_float64(self, rdt):
        x = torch.randn(2, 4, 16, dtype=torch.float64)
        out = rdt.pearson_r_loss(x, x.clone())
        assert out.dtype == torch.float64
