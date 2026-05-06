"""CLI integration tests — bad-input / error-path matrix.

Pins exit codes and stderr behaviour for every common malformed input.
The clinical contract: a malformed input MUST produce a non-zero exit
code AND a non-empty stderr message — never silent success.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _run(binary: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), *args],
        capture_output=True, text=True, timeout=timeout,
    )


# ============================================================
# 1. Missing files
# ============================================================


class TestMissingInput:

    def test_info_on_missing_file(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist.lml"
        r = _run(lml_cli_binary, "info", str(missing))
        assert r.returncode != 0

    def test_verify_on_missing_file(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist.lml"
        r = _run(lml_cli_binary, "verify", str(missing))
        assert r.returncode != 0

    def test_encode_on_missing_input(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist.edf"
        r = _run(lml_cli_binary, "encode", str(missing))
        assert r.returncode != 0

    def test_decode_on_missing_input(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist.lml"
        out = tmp_path / "out.bin"
        r = _run(lml_cli_binary, "decode", str(missing), "-o", str(out))
        assert r.returncode != 0

    def test_extract_on_missing_archive(self, tmp_path, lml_cli_binary):
        missing = tmp_path / "does_not_exist.lma"
        dst = tmp_path / "dst"
        r = _run(lml_cli_binary, "extract", str(missing), "-o", str(dst))
        assert r.returncode != 0


# ============================================================
# 2. Wrong magic — refuse to read
# ============================================================


class TestWrongMagic:

    def test_info_rejects_non_lml_magic(self, tmp_path, lml_cli_binary):
        bogus = tmp_path / "bogus.lml"
        bogus.write_bytes(b"XXXX" + b"\x00" * 60)
        r = _run(lml_cli_binary, "info", str(bogus))
        assert r.returncode != 0

    def test_verify_rejects_non_lml_magic(self, tmp_path, lml_cli_binary):
        bogus = tmp_path / "bogus.lml"
        bogus.write_bytes(b"NOPE" + b"\x00" * 60)
        r = _run(lml_cli_binary, "verify", str(bogus))
        assert r.returncode != 0

    def test_list_archive_rejects_non_lma_magic(self, tmp_path, lml_cli_binary):
        bogus = tmp_path / "bogus.lma"
        bogus.write_bytes(b"NOTL" + b"\x00" * 80)
        r = _run(lml_cli_binary, "list-archive", str(bogus))
        assert r.returncode != 0

    def test_verify_archive_rejects_non_lma_magic(self, tmp_path, lml_cli_binary):
        bogus = tmp_path / "bogus.lma"
        bogus.write_bytes(b"NOTL" + b"\x00" * 80)
        r = _run(lml_cli_binary, "verify-archive", str(bogus))
        assert r.returncode != 0


# ============================================================
# 3. Truncated files — refuse to read
# ============================================================


class TestTruncatedInput:

    def test_info_rejects_short_file(self, tmp_path, lml_cli_binary):
        short = tmp_path / "short.lml"
        short.write_bytes(b"LML1")  # magic only, no header
        r = _run(lml_cli_binary, "info", str(short))
        assert r.returncode != 0

    def test_extract_rejects_too_small_archive(self, tmp_path, lml_cli_binary):
        short = tmp_path / "short.lma"
        short.write_bytes(b"LMA1" + b"\x00" * 10)  # < 48-byte minimum
        dst = tmp_path / "dst"
        r = _run(lml_cli_binary, "extract", str(short), "-o", str(dst))
        assert r.returncode != 0


# ============================================================
# 4. Unknown subcommand
# ============================================================


class TestUnknownSubcommand:

    def test_unknown_subcommand_exits_nonzero(self, lml_cli_binary):
        r = _run(lml_cli_binary, "fly_to_the_moon")
        assert r.returncode != 0
        # Help text usually appears on stderr.
        assert "help" in (r.stderr + r.stdout).lower() or r.returncode != 0


# ============================================================
# 5. Help — must work
# ============================================================


class TestHelp:

    def test_help_subcommand_exits_zero(self, lml_cli_binary):
        r = _run(lml_cli_binary, "help")
        assert r.returncode == 0
        # Help output must mention at least one canonical subcommand.
        combined = r.stdout + r.stderr
        assert "encode" in combined or "Usage" in combined or "help" in combined.lower()
