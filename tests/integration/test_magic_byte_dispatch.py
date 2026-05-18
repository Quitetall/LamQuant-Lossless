"""`lml info` / `lml verify` magic-byte auto-dispatch.

v1.1 P.2 — both subcommands previously errored "Not LML" on `.lma`
input. Now they read the first 4 bytes, route to the archive
inspector (`lml ls --tree`) or archive verifier (`lml verify-archive`)
on `LMA1` magic, and stay on the LML path on `LML1` magic.

Same ergonomics as `tar` / `7z` / `unzip`: one user-typed verb,
content-aware dispatch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


def _build_lma(tmp_path: Path, lml_cli_binary: Path) -> tuple[Path, Path]:
    """Encode a small EDF into a per-recording `.lma`. Returns the
    archive path + the staging-resolved inner `.lml`."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)

    out = tmp_path / "out"
    out.mkdir()
    lma = out / "rec.lma"
    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf), "-o", str(lma)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]
    return lma, edf


def test_info_on_lma_auto_dispatches_to_ls_tree(tmp_path, lml_cli_binary):
    """`lml info foo.lma` routes to the archive inspector with a
    one-line stderr note."""
    lma, _edf = _build_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "info", str(lma)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr[:300]
    # Stderr names the dispatch.
    assert "LMA archive" in r.stderr, r.stderr[:300]
    assert "ls --tree" in r.stderr or "archive inspector" in r.stderr, r.stderr[:300]
    # Stdout has the tree-style listing (matches `lml ls --tree` shape).
    assert "rec.lma" in r.stdout, r.stdout[:400]
    assert "├──" in r.stdout or "└──" in r.stdout, r.stdout[:400]


def test_verify_on_lma_auto_dispatches_to_verify_archive(tmp_path, lml_cli_binary):
    """`lml verify foo.lma` routes to `cmd_verify_archive` and
    succeeds on a freshly-encoded archive."""
    lma, _edf = _build_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "verify", str(lma)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    assert "LMA archive" in r.stderr, r.stderr[:300]
    assert "verify-archive" in r.stderr or "archive verifier" in r.stderr


def test_verify_mixed_dir_walks_both_lml_and_lma(tmp_path, lml_cli_binary):
    """`lml verify <dir>` on a directory containing both `.lml`
    (bare-LML mode) and `.lma` (default mode) verifies every entry."""
    src = tmp_path / "src"
    src.mkdir()
    create_edf(str(src / "a.edf"), n_channels=4, n_records=2, sample_rate=250)
    create_edf(str(src / "b.edf"), n_channels=4, n_records=2, sample_rate=250)

    out = tmp_path / "out"
    out.mkdir()
    # `a.edf` -> bare LML
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(src / "a.edf"),
            "-o",
            str(out / "a.lml"),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]
    # `b.edf` -> default `.lma`
    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(src / "b.edf"), "-o", str(out / "b.lma")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]

    r = subprocess.run(
        [str(lml_cli_binary), "verify", str(out), "-r"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"verify failed: {r.stderr[:400]}\nstdout: {r.stdout[:400]}"
    # Mixed corpus walks: stdout names both formats.
    assert "a.lml" in r.stdout, r.stdout[:400]
    assert "b.lma" in r.stdout, r.stdout[:400]


def test_info_on_bad_magic_errors_with_useful_message(tmp_path, lml_cli_binary):
    """`lml info` on a file with neither `LML1` nor `LMA1` magic
    fails non-zero and names BOTH expected magics in the error."""
    bad = tmp_path / "garbage.lml"
    bad.write_bytes(b"NOTAMAGIC" + b"\x00" * 32)
    r = subprocess.run(
        [str(lml_cli_binary), "info", str(bad)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode != 0
    combined = r.stderr + r.stdout
    assert "LML1" in combined or "Not LML" in combined
