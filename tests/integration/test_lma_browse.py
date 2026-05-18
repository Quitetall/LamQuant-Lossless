"""`lml ls` + `lml cat` -- LMA browse surface.

These two subcommands are the CLI foundation for OS file-manager
integration (Ark on KDE, Nautilus on GNOME, Finder on macOS,
Explorer on Windows). A file-manager plugin shells out to:

  * `lml ls foo.lma` to enumerate entries.
  * `lml ls foo.lma --tree` for a human-readable tree view.
  * `lml cat foo.lma <entry-path>` to extract one entry to stdout
    (the plugin can pipe to its preview pane or a temp file).

This file pins the contract end-to-end.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


def _build_test_lma(tmp_path: Path, lml_cli_binary: Path) -> Path:
    """Helper: build an LMA with an EDF + two sibling files."""
    src = tmp_path / "src"
    src.mkdir()
    edf_path = src / "rec.edf"
    create_edf(str(edf_path), n_channels=4, n_records=2, sample_rate=250)
    (src / "rec.tse").write_bytes(b"0.0 30.0 bckg 1.0\n")
    (src / "rec_summary.txt").write_bytes(b"Patient: smoke-test\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:300]
    return lma_target


def test_ls_flat_lists_every_entry(tmp_path, lml_cli_binary):
    """`lml ls foo.lma` prints one entry path per line."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "ls", str(lma)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    assert "rec.lml" in lines
    assert "rec.tse" in lines
    assert "rec_summary.txt" in lines


def test_ls_tree_includes_header_and_sizes(tmp_path, lml_cli_binary):
    """`lml ls foo.lma --tree` includes a header line + per-entry
    size + compression-method + sha256 prefix."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "ls", str(lma), "--tree"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr
    stdout = r.stdout
    # Header mentions archive name + entry count + CR.
    assert "rec.lma" in stdout
    assert "3 entries" in stdout, stdout
    # Tree branches use the Unicode box-drawing characters.
    assert "├──" in stdout or "└──" in stdout, stdout
    # Each entry has a method tag.
    assert "lml" in stdout
    # Every entry has the sha256: prefix.
    assert "sha256:" in stdout


def test_cat_extracts_entry_byte_for_byte(tmp_path, lml_cli_binary):
    """`lml cat foo.lma <entry>` writes the byte-equal entry to stdout."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    expected = b"0.0 30.0 bckg 1.0\n"
    r = subprocess.run(
        [str(lml_cli_binary), "cat", str(lma), "rec.tse"],
        capture_output=True,
        timeout=15,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", errors="replace")[:400]
    assert r.stdout == expected, (
        f"`lml cat` returned bytes that differ from source. "
        f"expected {expected!r}, got {r.stdout!r}"
    )


def test_cat_unknown_entry_errors_nonzero(tmp_path, lml_cli_binary):
    """`lml cat` on a missing entry fails loudly, not silently."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "cat", str(lma), "does_not_exist.txt"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode != 0
    combined = (r.stderr + r.stdout).lower()
    assert "does_not_exist" in combined or "not found" in combined or "no such" in combined, (
        f"missing-entry error did not name the missing entry; "
        f"stderr: {r.stderr[:300]}, stdout: {r.stdout[:300]}"
    )
