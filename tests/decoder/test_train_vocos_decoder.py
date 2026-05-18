"""Unit tests for ai_models/decoder/train_vocos_decoder.py — Phase 3.

Covers `pearson_r_batch` + `discover_student_checkpoint`. The main()
training loop needs real data + GPU + heavy deps (auraloss,
streaming_dataset, perceptual_losses); covered indirectly via the
integration suite.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[2]
                / "ai_models" / "decoder" / "train_vocos_decoder.py")


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_STUBBED = ("vocos_decoder", "train_student_subband", "auraloss",
             "auraloss.freq", "streaming_dataset", "flow_postfilter",
             "perceptual_losses", "lamquant_codec",
             "lamquant_codec.models", "lamquant_codec.models.encoder",
             "ai_models.decoder.train_vocos_decoder_under_test")


@pytest.fixture(scope="module")
def tvd():
    """Load train_vocos_decoder with heavy deps stubbed out.

    Snapshot/restore sys.modules to avoid leaking stubs into other tests.
    """
    pre = {n: sys.modules.get(n) for n in _STUBBED}

    if "vocos_decoder" not in sys.modules:
        sys.modules["vocos_decoder"] = _stub("vocos_decoder",
                                              VocosDecoder=object)
    if "train_student_subband" not in sys.modules:
        sys.modules["train_student_subband"] = _stub(
            "train_student_subband",
            pearson_r_loss=lambda p, t: torch.tensor(0.0))
    if "auraloss" not in sys.modules:
        pkg = _stub("auraloss")
        freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
        pkg.freq = freq  # type: ignore[attr-defined]
        sys.modules["auraloss"] = pkg
        sys.modules["auraloss.freq"] = freq
    if "streaming_dataset" not in sys.modules:
        sys.modules["streaming_dataset"] = _stub(
            "streaming_dataset", PrecomputedL3Dataset=object)
    if "flow_postfilter" not in sys.modules:
        sys.modules["flow_postfilter"] = _stub(
            "flow_postfilter", CFMPostfilter=object)
    if "perceptual_losses" not in sys.modules:
        sys.modules["perceptual_losses"] = _stub(
            "perceptual_losses", MultiTeacherPerceptualLoss=object)
    if "lamquant_codec" not in sys.modules:
        pkg = _stub("lamquant_codec")
        models = _stub("lamquant_codec.models")
        enc = _stub("lamquant_codec.models.encoder",
                     TernaryMobileNetV5_Subband=object)
        pkg.models = models  # type: ignore[attr-defined]
        models.encoder = enc  # type: ignore[attr-defined]
        sys.modules["lamquant_codec"] = pkg
        sys.modules["lamquant_codec.models"] = models
        sys.modules["lamquant_codec.models.encoder"] = enc

    name = "ai_models.decoder.train_vocos_decoder_under_test"
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
# pearson_r_batch
# ---------------------------------------------------------------------------
class TestPearsonRBatch:
    def test_identical_returns_one(self, tvd):
        x = torch.randn(4, 21, 313)
        r = tvd.pearson_r_batch(x, x.clone())
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_negated_returns_minus_one(self, tvd):
        x = torch.randn(2, 4, 100)
        r = tvd.pearson_r_batch(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_returns_python_float(self, tvd):
        r = tvd.pearson_r_batch(torch.randn(2, 4, 16),
                                 torch.randn(2, 4, 16))
        assert isinstance(r, float)

    def test_random_uncorrelated_near_zero(self, tvd):
        torch.manual_seed(0)
        r = tvd.pearson_r_batch(torch.randn(32, 21, 313),
                                 torch.randn(32, 21, 313))
        assert abs(r) < 0.05


# ---------------------------------------------------------------------------
# discover_student_checkpoint
# ---------------------------------------------------------------------------
class TestDiscoverCheckpoint:
    def test_explicit_path_returned(self, tvd, tmp_path):
        p = tmp_path / "explicit.ckpt"
        p.write_text("x")
        assert tvd.discover_student_checkpoint(str(p)) == str(p)

    def test_explicit_missing_exits_1(self, tvd, tmp_path):
        with pytest.raises(SystemExit) as e:
            tvd.discover_student_checkpoint(str(tmp_path / "nope.ckpt"))
        assert e.value.code == 1

    def test_no_args_no_candidates_exits(self, tvd, monkeypatch, tmp_path):
        # Override ROOT_DIR so no candidate paths exist
        monkeypatch.setattr(tvd, "ROOT_DIR", str(tmp_path))
        with pytest.raises(SystemExit) as e:
            tvd.discover_student_checkpoint()
        assert e.value.code == 1

    def test_finds_first_candidate(self, tvd, monkeypatch, tmp_path):
        monkeypatch.setattr(tvd, "ROOT_DIR", str(tmp_path))
        # Create one candidate
        (tmp_path / "ai_models" / "student").mkdir(parents=True)
        ckpt = tmp_path / "ai_models" / "student" / "student_hardened.ckpt"
        ckpt.write_text("x")
        result = tvd.discover_student_checkpoint()
        assert os.path.basename(result) == "student_hardened.ckpt"
