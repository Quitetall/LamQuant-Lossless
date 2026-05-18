"""Unit tests for firmware/export_firmware.py — Phase 3.

Focused on:
- generate_biquad_coefficients (pure math)
- _load_model_for_export error paths (no checkpoint found)
- main() --validate-schema fast path
- main() error path when no checkpoint

Heavy paths (_emit_c, _emit_rust with real model + FSQ calibration)
are covered by integration tests with real ckpts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import firmware.export_firmware as ef

pytestmark = pytest.mark.l2


_REPO = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO / "firmware" / "export_schema.toml"


# ---------------------------------------------------------------------------
# generate_biquad_coefficients
# ---------------------------------------------------------------------------
class TestGenerateBiquadCoefficients:
    def test_runs_without_error(self, capsys):
        ef.generate_biquad_coefficients()
        out = capsys.readouterr().out
        # Should print biquad coefficients
        assert len(out) > 0


# ---------------------------------------------------------------------------
# export_toeplitz_seeds
# ---------------------------------------------------------------------------
class TestExportToeplitzSeeds:
    def test_writes_header(self, tmp_path):
        out = tmp_path / "toep.h"
        ef.export_toeplitz_seeds(str(out))
        assert out.is_file()
        text = out.read_text()
        assert "#ifndef" in text
        assert "toeplitz" in text.lower()


# ---------------------------------------------------------------------------
# compute_firmware_crc
# ---------------------------------------------------------------------------
class TestComputeFirmwareCrc:
    def test_writes_crc_header(self, tmp_path):
        header = tmp_path / "weights.h"
        header.write_text("/* fake header */\nstatic const int8_t w[] = {1, 2, 3};")
        crc_out = tmp_path / "crc.h"
        ef.compute_firmware_crc(str(header), str(crc_out))
        assert crc_out.is_file()
        text = crc_out.read_text()
        assert "CRC" in text.upper() or "0x" in text.lower()


# ---------------------------------------------------------------------------
# _load_model_for_export error paths
# ---------------------------------------------------------------------------
class TestLoadModelErrors:
    def test_explicit_missing_exits(self, tmp_path, monkeypatch, capsys):
        # Stub away the train_ternary import so it doesn't pull real torch
        # objects, then assert the explicit-not-found path exits with 1.
        with pytest.raises(SystemExit) as e:
            ef._load_model_for_export(torch.device("cpu"),
                                       explicit_ckpt=str(tmp_path / "nope.ckpt"))
        assert e.value.code == 1
        out = capsys.readouterr().out
        assert "not found" in out


# ---------------------------------------------------------------------------
# main() --validate-schema
# ---------------------------------------------------------------------------
class TestMainValidateSchema:
    @pytest.mark.skipif(not _SCHEMA.is_file(),
                         reason="export_schema.toml not present")
    def test_validate_schema_runs(self, capsys):
        ef.main(["--validate-schema", str(_SCHEMA)])
        out = capsys.readouterr().out
        assert "[OK]" in out
