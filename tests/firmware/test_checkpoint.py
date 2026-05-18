"""Unit tests for firmware/export/checkpoint.py — Phase 1.

Covers detect_arch, _grade_of, find_checkpoint, sha256_of,
load_checkpoint (auto-detect + explicit-path paths).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from firmware.export.checkpoint import (
    LoadedCheckpoint,
    _grade_of,
    detect_arch,
    find_checkpoint,
    load_checkpoint,
    sha256_of,
)
from firmware.export.schema import ArchSpec

pytestmark = pytest.mark.l1


# ---------------------------------------------------------------------------
# Helpers — fake ArchSpec dict
# ---------------------------------------------------------------------------
def _archs():
    return {
        "subband_v1": ArchSpec(name="subband_v1", display_name="V1",
                                encoder_class="C", encoder_width=128,
                                n_focal_blocks=3, latent_dims=32,
                                latent_timesteps=79,
                                checkpoint_globs=["weights/subband_*.ckpt"]),
        "subband_v2": ArchSpec(name="subband_v2", display_name="V2",
                                encoder_class="C", encoder_width=216,
                                n_focal_blocks=4, latent_dims=32,
                                latent_timesteps=79,
                                checkpoint_globs=["weights/subband_v2_*.ckpt"]),
        "legacy_v7_0": ArchSpec(name="legacy_v7_0", display_name="L",
                                 encoder_class="C", encoder_width=128,
                                 n_focal_blocks=3, latent_dims=32,
                                 latent_timesteps=79,
                                 checkpoint_globs=["weights/legacy_*.ckpt"]),
    }


# ---------------------------------------------------------------------------
# LoadedCheckpoint dataclass
# ---------------------------------------------------------------------------
class TestLoadedCheckpoint:
    def test_short_sha(self):
        c = LoadedCheckpoint(path=Path("/x"), sha256="0123456789abcdef" * 4,
                              state_dict={}, arch_name="subband_v1",
                              grade="gold")
        assert c.short_sha() == "0123456789ab"

    def test_frozen(self):
        c = LoadedCheckpoint(path=Path("/x"), sha256="a", state_dict={},
                              arch_name="x", grade="y")
        with pytest.raises(Exception):
            c.grade = "z"


# ---------------------------------------------------------------------------
# detect_arch
# ---------------------------------------------------------------------------
class TestDetectArch:
    def test_v2_via_focal2_dw(self):
        sd = {"focal2.dw.weight": torch.zeros(1)}
        assert detect_arch(sd, _archs()) == "subband_v2"

    def test_v1_via_rotation_A_premix(self):
        sd = {"rotation_A": torch.zeros(1), "premix.weight": torch.zeros(1)}
        assert detect_arch(sd, _archs()) == "subband_v1"

    def test_legacy_fallback(self):
        sd = {"some_legacy_key": torch.zeros(1)}
        assert detect_arch(sd, _archs()) == "legacy_v7_0"

    def test_no_match_raises(self):
        # Empty arch dict + non-matching keys → raise
        sd = {"foo": torch.zeros(1)}
        with pytest.raises(ValueError, match="Could not detect"):
            detect_arch(sd, {})


# ---------------------------------------------------------------------------
# _grade_of
# ---------------------------------------------------------------------------
class TestGradeOf:
    def test_gold(self):
        assert _grade_of(Path("/a/subband_gold.ckpt")) == "gold"

    def test_std(self):
        assert _grade_of(Path("/a/subband_std.ckpt")) == "std"

    def test_fast(self):
        assert _grade_of(Path("/a/subband_fast.ckpt")) == "fast"

    def test_canonical(self):
        assert _grade_of(Path("/a/subband.ckpt")) == "canonical"

    def test_dev_for_ai_models_path(self):
        assert _grade_of(Path("/x/ai_models/y/foo.ckpt")) == "dev"

    def test_legacy_hardened(self):
        assert _grade_of(Path("/a/hardened_v7.ckpt")) == "legacy"

    def test_untagged(self):
        assert _grade_of(Path("/a/something_else.ckpt")) == "untagged"


# ---------------------------------------------------------------------------
# find_checkpoint
# ---------------------------------------------------------------------------
class TestFindCheckpoint:
    def test_explicit_path_returned(self, tmp_path):
        p = tmp_path / "x.ckpt"
        p.write_bytes(b"x")
        assert find_checkpoint(tmp_path, [], explicit=p) == p

    def test_explicit_relative_resolves(self, tmp_path):
        p = tmp_path / "x.ckpt"
        p.write_bytes(b"x")
        out = find_checkpoint(tmp_path, [], explicit=Path("x.ckpt"))
        assert out.resolve() == p.resolve()

    def test_explicit_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path, [], explicit=tmp_path / "nope.ckpt")

    def test_no_matches_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No checkpoint matched"):
            find_checkpoint(tmp_path, ["weights/*.ckpt"])

    def test_picks_gold_over_fast(self, tmp_path):
        # Create three candidates; gold should win
        wdir = tmp_path / "weights"
        wdir.mkdir()
        gold = wdir / "subband_gold.ckpt"
        fast = wdir / "subband_fast.ckpt"
        std = wdir / "subband_std.ckpt"
        for p in (fast, std, gold):
            p.write_bytes(b"x")
        out = find_checkpoint(tmp_path, ["weights/subband_*.ckpt"])
        assert out.name == "subband_gold.ckpt"


# ---------------------------------------------------------------------------
# sha256_of
# ---------------------------------------------------------------------------
class TestSha256Of:
    def test_matches_hashlib(self, tmp_path):
        data = b"hello, firmware"
        p = tmp_path / "f.bin"
        p.write_bytes(data)
        assert sha256_of(p) == hashlib.sha256(data).hexdigest()

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert sha256_of(p) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------
class TestLoadCheckpoint:
    def _make_ckpt(self, tmp_path, sd_keys, name="subband_gold.ckpt"):
        wdir = tmp_path / "weights"
        wdir.mkdir(exist_ok=True)
        p = wdir / name
        sd = {k: torch.zeros(1) for k in sd_keys}
        torch.save(sd, p)
        return p

    def test_explicit_path_auto_detect(self, tmp_path):
        p = self._make_ckpt(tmp_path, ["focal2.dw.weight"])
        loaded = load_checkpoint(tmp_path, arch_name=None,
                                  known_archs=_archs(), explicit_path=p)
        assert loaded.arch_name == "subband_v2"
        assert loaded.grade == "gold"
        assert loaded.path.name == "subband_gold.ckpt"

    def test_explicit_relative_path(self, tmp_path):
        self._make_ckpt(tmp_path, ["rotation_A", "premix.weight"])
        loaded = load_checkpoint(
            tmp_path, arch_name=None, known_archs=_archs(),
            explicit_path=Path("weights/subband_gold.ckpt"))
        assert loaded.arch_name == "subband_v1"

    def test_explicit_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path, arch_name=None, known_archs=_archs(),
                            explicit_path=tmp_path / "no.ckpt")

    def test_arch_name_search(self, tmp_path):
        self._make_ckpt(tmp_path, ["rotation_A", "premix.weight"])
        loaded = load_checkpoint(tmp_path, arch_name="subband_v1",
                                  known_archs=_archs())
        assert loaded.arch_name == "subband_v1"

    def test_missing_both_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Either"):
            load_checkpoint(tmp_path, arch_name=None, known_archs=_archs())

    def test_wrapped_state_dict_unwrapped(self, tmp_path):
        wdir = tmp_path / "weights"
        wdir.mkdir()
        p = wdir / "subband_gold.ckpt"
        torch.save({"state_dict": {"focal2.dw.weight": torch.zeros(1)},
                    "epoch": 100}, p)
        loaded = load_checkpoint(tmp_path, arch_name=None,
                                  known_archs=_archs(), explicit_path=p)
        # Top-level "epoch" key dropped; state_dict unwrapped
        assert "focal2.dw.weight" in loaded.state_dict
        assert "epoch" not in loaded.state_dict
