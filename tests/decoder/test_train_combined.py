"""Unit tests for ai_models/decoder/train_combined.py — Phase 3."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[2]
                / "ai_models" / "decoder" / "train_combined.py")


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_STUBBED = ("vocos_decoder", "train_teacher", "train_student_subband",
             "streaming_dataset", "raw_window_dataset",
             "auraloss", "auraloss.freq",
             "lamquant_codec", "lamquant_codec.models",
             "lamquant_codec.models.encoder",
             "ai_models.decoder.train_combined_under_test")


@pytest.fixture(scope="module")
def tc():
    pre = {n: sys.modules.get(n) for n in _STUBBED}
    sys.modules["vocos_decoder"] = _stub("vocos_decoder", VocosDecoder=object)
    sys.modules["train_teacher"] = _stub("train_teacher", L3Teacher=object)
    sys.modules["train_student_subband"] = _stub(
        "train_student_subband",
        pearson_r_loss=lambda p, t: torch.tensor(0.0))
    sys.modules["streaming_dataset"] = _stub(
        "streaming_dataset", PrecomputedL3Dataset=object)
    sys.modules["raw_window_dataset"] = _stub(
        "raw_window_dataset", RawWindowDataset=object)
    pkg = _stub("auraloss")
    freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
    pkg.freq = freq  # type: ignore[attr-defined]
    sys.modules["auraloss"] = pkg
    sys.modules["auraloss.freq"] = freq
    cpkg = _stub("lamquant_codec")
    cmodels = _stub("lamquant_codec.models")
    cenc = _stub("lamquant_codec.models.encoder",
                  TernaryMobileNetV5_Subband=object)
    cpkg.models = cmodels  # type: ignore[attr-defined]
    cmodels.encoder = cenc  # type: ignore[attr-defined]
    sys.modules["lamquant_codec"] = cpkg
    sys.modules["lamquant_codec.models"] = cmodels
    sys.modules["lamquant_codec.models.encoder"] = cenc

    name = "ai_models.decoder.train_combined_under_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        for n, prev in pre.items():
            if prev is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev


# ---------------------------------------------------------------------------
# pearson_r_batch + prd_batch
# ---------------------------------------------------------------------------
class TestMetricsHelpers:
    def test_pearson_identical(self, tc):
        x = torch.randn(4, 21, 313)
        assert tc.pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)

    def test_pearson_negated(self, tc):
        x = torch.randn(2, 4, 100)
        assert tc.pearson_r_batch(x, -x) == pytest.approx(-1.0, abs=1e-5)

    def test_prd_identical_zero(self, tc):
        x = torch.randn(4, 21, 313)
        assert tc.prd_batch(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_prd_doubled_target_is_100(self, tc):
        target = torch.randn(4, 4, 100)
        pred = 2 * target
        assert tc.prd_batch(pred, target) == pytest.approx(100.0, rel=1e-4)

    def test_prd_eps_protects_zero_target(self, tc):
        # All-zero target → denom clamped at 1e-8
        out = tc.prd_batch(torch.randn(2, 4, 16), torch.zeros(2, 4, 16))
        assert torch.isfinite(torch.tensor(out))

    def test_returns_python_float(self, tc):
        assert isinstance(tc.pearson_r_batch(torch.randn(2, 4, 8),
                                              torch.randn(2, 4, 8)), float)
        assert isinstance(tc.prd_batch(torch.randn(2, 4, 8),
                                        torch.randn(2, 4, 8)), float)


# ---------------------------------------------------------------------------
# validate_teacher
# ---------------------------------------------------------------------------
class TestValidateTeacher:
    def test_returns_two_floats(self, tc):
        class _Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(313, 313)
            def forward(self, x):
                return self.lin(x)

        teacher = _Teacher()
        # val_loader yields tuples (x_l3, ?, ?)
        x = torch.randn(2, 21, 313)
        val_loader = [(x, None, None), (x, None, None)]
        r, prd = tc.validate_teacher(teacher, val_loader, torch.device("cpu"))
        assert isinstance(r, float)
        assert isinstance(prd, float)

    def test_empty_loader_returns_zero(self, tc):
        teacher = torch.nn.Module()
        r, prd = tc.validate_teacher(teacher, [], torch.device("cpu"))
        assert r == 0.0
        assert prd == 0.0
