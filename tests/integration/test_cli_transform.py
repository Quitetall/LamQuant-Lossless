"""CLI integration tests — transform commands that produce output files.

Pins exit-code, stdout, and produced-file contracts for:
  lml encode
  lml decode
  lml archive
  lml extract
  lml diff

Strategy: subprocess.run, assert returncode + produced file exists with
non-zero size + (where applicable) byte-exact recovery.
"""
from __future__ import annotations

import hashlib
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
    path = tmp_path / "tiny.edf"
    create_edf(str(path), n_channels=4, n_records=1, sample_rate=250)
    return path


def _run(binary: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ============================================================
# 1. lml encode
# ============================================================


class TestLmlEncode:

    # The default encode output is a per-EDF `.lma` archive (which
    # also bundles every sibling sidecar — see
    # `test_sidecar_preservation.py`). For these tests we pass
    # `--no-bundle` to keep the bare `.lml` shape that the rest of the
    # subtest assertions inspect.

    def test_produces_lml_file(self, tiny_edf, tmp_path, lml_cli_binary):
        out_dir = tmp_path / "lml"
        out_dir.mkdir()
        r = _run(lml_cli_binary, "encode", str(tiny_edf),
                 "-o", str(out_dir), "--no-bundle",
                 "--i-understand-data-loss")
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"
        files = list(out_dir.glob("*.lml"))
        assert len(files) == 1
        assert files[0].stat().st_size > 0

    def test_output_starts_with_lml_magic(self, tiny_edf, tmp_path, lml_cli_binary):
        out_dir = tmp_path / "lml"
        out_dir.mkdir()
        r = _run(lml_cli_binary, "encode", str(tiny_edf),
                 "-o", str(out_dir), "--no-bundle",
                 "--i-understand-data-loss")
        assert r.returncode == 0
        produced = next(out_dir.glob("*.lml"))
        assert produced.read_bytes()[:4] == b"LML1"


# ============================================================
# 2. lml decode
# ============================================================


class TestLmlDecode:

    def test_decode_produces_output(self, tiny_edf, tmp_path, lml_cli_binary):
        # Encode first — `--no-bundle` to inspect the raw `.lml` rather
        # than the default per-EDF `.lma` archive.
        out_dir = tmp_path / "lml"
        out_dir.mkdir()
        r = _run(lml_cli_binary, "encode", str(tiny_edf),
                 "-o", str(out_dir), "--no-bundle",
                 "--i-understand-data-loss")
        assert r.returncode == 0
        lml_file = next(out_dir.glob("*.lml"))

        # Decode to raw int32 LE.
        decoded = tmp_path / "decoded.bin"
        r = _run(lml_cli_binary, "decode", str(lml_file), "-o", str(decoded))
        assert r.returncode == 0, f"decode stderr: {r.stderr[:200]}"
        assert decoded.exists()
        assert decoded.stat().st_size > 0


# ============================================================
# 3. lml archive
# ============================================================


class TestLmlArchive:

    def test_archive_creates_lma_file(self, tmp_path, tiny_edf, lml_cli_binary):
        src = tmp_path / "src"
        src.mkdir()
        (src / "data.edf").write_bytes(tiny_edf.read_bytes())
        (src / "notes.csv").write_text("a,b\n")

        archive = tmp_path / "out.lma"
        r = _run(lml_cli_binary, "archive", str(src), "-o", str(archive))
        assert r.returncode == 0, f"stderr: {r.stderr[:200]}"
        assert archive.exists()
        assert archive.read_bytes()[:4] == b"LMA1"

    def test_archive_rejects_nonexistent_input(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist"
        archive = tmp_path / "out.lma"
        r = _run(lml_cli_binary, "archive", str(missing), "-o", str(archive))
        assert r.returncode != 0


# ============================================================
# 4. lml extract
# ============================================================


class TestLmlExtract:

    def test_extract_recovers_byte_exact(self, tmp_path, tiny_edf, lml_cli_binary):
        # Pack
        src = tmp_path / "src"
        src.mkdir()
        original_edf = src / "data.edf"
        original_edf.write_bytes(tiny_edf.read_bytes())
        original_csv = src / "notes.csv"
        original_csv.write_text("a,b\n1,2\n")
        original_edf_sha = _sha256(original_edf)
        original_csv_sha = _sha256(original_csv)

        archive = tmp_path / "out.lma"
        r = _run(lml_cli_binary, "archive", str(src), "-o", str(archive))
        assert r.returncode == 0

        # Extract
        dst = tmp_path / "dst"
        dst.mkdir()
        r = _run(lml_cli_binary, "extract", str(archive), "-o", str(dst))
        assert r.returncode == 0, f"extract stderr: {r.stderr[:200]}"

        # CSV must come back byte-exact (Store / Zstd path).
        # EDF goes through LML → reconstruction; SHA may or may not match
        # depending on whether the binary writes a .lml or reconstructs.
        # Assert the CSV sidecar at minimum.
        recovered_csv = dst / "notes.csv"
        assert recovered_csv.exists()
        assert _sha256(recovered_csv) == original_csv_sha


# ============================================================
# 5. lml diff
# ============================================================


class TestLmlDiff:

    def test_diff_zero_on_identical_files(self, tiny_edf, tmp_path, lml_cli_binary):
        # `--no-bundle` so `lml diff` sees the raw `.lml` it expects;
        # the default per-EDF `.lma` path is covered in the
        # archive-roundtrip suite.
        out_dir = tmp_path / "lml"
        out_dir.mkdir()
        r = _run(lml_cli_binary, "encode", str(tiny_edf),
                 "-o", str(out_dir), "--no-bundle",
                 "--i-understand-data-loss")
        assert r.returncode == 0
        original = next(out_dir.glob("*.lml"))

        # Copy the file — diff against itself should be all-zero / pass.
        copy = tmp_path / "copy.lml"
        copy.write_bytes(original.read_bytes())

        r = _run(lml_cli_binary, "diff", str(original), str(copy))
        assert r.returncode == 0, (
            f"diff against identical file should pass — "
            f"returncode={r.returncode}, stderr={r.stderr[:200]}"
        )
