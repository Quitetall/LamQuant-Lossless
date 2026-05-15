"""CLI integration tests — read-only inspection commands.

Pins exit-code, stdout patterns, and behavioural contracts for:
  lml info
  lml stats
  lml verify
  lml verify-archive
  lml verify-manifest
  lml list-archive

Strategy: subprocess.run, assert returncode + stdout regex/substring +
no spurious stderr on success path.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture
def tiny_edf(tmp_path: Path) -> Path:
    """A minimal valid EDF file: 4 channels × 1 second @ 250 Hz."""
    path = tmp_path / "tiny.edf"
    create_edf(str(path), n_channels=4, n_records=1, sample_rate=250)
    return path


@pytest.fixture
def encoded_lml(tmp_path: Path, tiny_edf: Path, lml_cli_binary: Path) -> Path:
    """An LML file encoded from tiny_edf via the binary.

    Uses `--no-bundle` so the encoder emits a bare `.lml` rather than
    the default per-EDF `.lma` archive — these inspection tests want
    to point `lml info` / `lml verify` / etc. at the raw `.lml`. The
    sidecar-preservation contract lives in
    `test_sidecar_preservation.py`.
    """
    out_dir = tmp_path / "lml"
    out_dir.mkdir()
    result = subprocess.run(
        [str(lml_cli_binary), "encode", str(tiny_edf),
         "-o", str(out_dir), "--no-bundle"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"setup encode failed: stderr={result.stderr[:200]}"
    )
    lml_files = list(out_dir.glob("*.lml"))
    assert len(lml_files) == 1, f"expected 1 .lml file, got {lml_files}"
    return lml_files[0]


@pytest.fixture
def archived(tmp_path: Path, tiny_edf: Path, lml_cli_binary: Path) -> Path:
    """A 2-file .lma archive (tiny_edf + a CSV sidecar)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.edf").write_bytes(tiny_edf.read_bytes())
    (src / "notes.csv").write_text("a,b\n1,2\n")
    archive = tmp_path / "out.lma"
    result = subprocess.run(
        [str(lml_cli_binary), "archive", str(src), "-o", str(archive)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"setup archive failed: {result.stderr[:200]}"
    return archive


def _run(binary: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ============================================================
# 1. lml info
# ============================================================


class TestLmlInfo:

    def test_exits_zero_on_valid_lml(self, encoded_lml, lml_cli_binary):
        r = _run(lml_cli_binary, "info", str(encoded_lml))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"

    def test_stdout_includes_channel_count(self, encoded_lml, lml_cli_binary):
        r = _run(lml_cli_binary, "info", str(encoded_lml))
        assert r.returncode == 0
        # 4 channels in tiny_edf — info must surface this.
        assert "4" in r.stdout, f"channel count not in stdout: {r.stdout[:300]}"


# ============================================================
# 2. lml stats
# ============================================================


class TestLmlStats:

    def test_exits_zero_on_valid_lml(self, encoded_lml, lml_cli_binary):
        r = _run(lml_cli_binary, "stats", str(encoded_lml))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"


# ============================================================
# 3. lml verify
# ============================================================


class TestLmlVerify:

    def test_exits_zero_on_valid_lml(self, encoded_lml, lml_cli_binary):
        r = _run(lml_cli_binary, "verify", str(encoded_lml))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"

    def test_exits_nonzero_on_corrupted_lml(self, encoded_lml, lml_cli_binary, tmp_path):
        # Flip a byte inside an LML1 packet's CRC-covered region.
        # Per-packet CRC covers `header_var[4..18] || lpc_meta || payload`
        # (see lml.rs:compress); container JSON metadata + offset table
        # are explicitly out-of-CRC-contract for `verify`. The old
        # `len//2` midpoint heuristic landed inside the JSON metadata
        # blob on the 4-ch × 1-rec fixture and was silently passing
        # whenever the metadata grew past file-midpoint (which it does
        # for tiny fixtures). Anchor on the first LML1 magic byte and
        # flip 25 bytes past it (3 bytes into lpc_meta, always
        # CRC-protected).
        corrupted = tmp_path / "corrupted.lml"
        bytes_orig = bytearray(encoded_lml.read_bytes())
        first_packet = bytes_orig.find(b"LML1", 32)
        assert first_packet >= 0, "fixture missing LML1 packet magic"
        flip_idx = first_packet + 25
        assert flip_idx < len(bytes_orig), "fixture too small for CRC-covered flip"
        bytes_orig[flip_idx] ^= 0x01
        corrupted.write_bytes(bytes(bytes_orig))

        r = _run(lml_cli_binary, "verify", str(corrupted))
        assert r.returncode != 0, (
            f"verify should detect single-byte corruption in CRC-covered region — "
            f"returncode={r.returncode}, stdout={r.stdout[:200]}"
        )


# ============================================================
# 4. lml list-archive
# ============================================================


class TestLmlListArchive:

    def test_exits_zero_on_valid_archive(self, archived, lml_cli_binary):
        r = _run(lml_cli_binary, "list-archive", str(archived))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"

    def test_lists_every_archived_file(self, archived, lml_cli_binary):
        r = _run(lml_cli_binary, "list-archive", str(archived))
        assert r.returncode == 0
        # Both archived files must appear in output.
        assert "data.edf" in r.stdout, f"data.edf missing: {r.stdout[:300]}"
        assert "notes.csv" in r.stdout, f"notes.csv missing: {r.stdout[:300]}"


# ============================================================
# 5. lml verify-archive
# ============================================================


class TestLmlVerifyArchive:

    def test_exits_zero_on_valid_archive(self, archived, lml_cli_binary):
        r = _run(lml_cli_binary, "verify-archive", str(archived))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"

    def test_exits_nonzero_on_corrupted_archive(self, archived, lml_cli_binary, tmp_path):
        corrupted = tmp_path / "corrupted.lma"
        bytes_orig = bytearray(archived.read_bytes())
        # Flip far enough into the file to land in the manifest/payload region.
        flip_idx = max(64, len(bytes_orig) // 2)
        bytes_orig[flip_idx] ^= 0x10
        corrupted.write_bytes(bytes(bytes_orig))

        r = _run(lml_cli_binary, "verify-archive", str(corrupted))
        assert r.returncode != 0, (
            f"verify-archive should detect corruption — "
            f"returncode={r.returncode}"
        )
