"""`lml encode --include` / `--exclude` glob filters (v1.2 I).

User feedback that drove v1.0 footgun work: silent file drops are
unacceptable. v1.2 I lets the operator filter the batch encode set
with tar-style glob patterns, but every excluded file is named on
stderr -- loss stays loud.

Cases:

  1. `--exclude '*.tmp'` strips temp files; non-tmp encoded.
  2. `--include 'rec_*.edf'` keeps only matching prefix; others
     excluded with stderr notice each.
  3. `--include` + `--exclude` together (AND semantics; exclude
     wins on overlap).
  4. All files filtered → encoder refuses with explicit error.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


def _make_corpus(src_dir: Path, stems: list[str]) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        create_edf(
            str(src_dir / f"{stem}.edf"),
            n_channels=4, n_records=2, sample_rate=250,
        )


def test_exclude_glob_drops_matching_files_and_names_them(
    tmp_path, lml_cli_binary
):
    """`--exclude '*excluded*'` skips matching EDFs; stderr names each."""
    src = tmp_path / "src"
    _make_corpus(src, ["rec_001", "rec_002", "tmp_excluded", "extra_excluded"])

    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(src),
            "-o",
            str(out),
            "-r",
            "--exclude",
            "*excluded*",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[:600]

    # Excluded files named explicitly on stderr.
    assert "excluded: " in r.stderr, r.stderr[:600]
    assert "tmp_excluded.edf" in r.stderr, r.stderr[:600]
    assert "extra_excluded.edf" in r.stderr, r.stderr[:600]

    # Output dir has the included files only.
    lma_files = sorted(p.name for p in out.rglob("*.lma"))
    assert "rec_001.lma" in lma_files, lma_files
    assert "rec_002.lma" in lma_files, lma_files
    assert all("excluded" not in n for n in lma_files), lma_files


def test_include_glob_keeps_only_matching_files(tmp_path, lml_cli_binary):
    """`--include 'keep_*.edf'` -- only matching files encoded."""
    src = tmp_path / "src"
    _make_corpus(src, ["keep_001", "keep_002", "skip_001", "skip_002"])

    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(src),
            "-o",
            str(out),
            "-r",
            "--include",
            "keep_*.edf",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[:600]

    # skip_* excluded with stderr notice.
    assert "skip_001.edf" in r.stderr, r.stderr[:600]
    assert "skip_002.edf" in r.stderr, r.stderr[:600]

    # Only keep_* encoded.
    lma_files = sorted(p.name for p in out.rglob("*.lma"))
    assert "keep_001.lma" in lma_files, lma_files
    assert "keep_002.lma" in lma_files, lma_files
    assert "skip_001.lma" not in lma_files
    assert "skip_002.lma" not in lma_files


def test_include_and_exclude_combined_uses_and_semantics(tmp_path, lml_cli_binary):
    """`--include 'rec_*.edf' --exclude '*_skip*'` -- both apply;
    exclude wins on overlap."""
    src = tmp_path / "src"
    _make_corpus(
        src,
        ["rec_keep_001", "rec_keep_002", "rec_skip_001", "other_001"],
    )

    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(src),
            "-o",
            str(out),
            "-r",
            "--include",
            "rec_*.edf",
            "--exclude",
            "*_skip_*",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[:600]

    lma_files = sorted(p.name for p in out.rglob("*.lma"))
    assert "rec_keep_001.lma" in lma_files
    assert "rec_keep_002.lma" in lma_files
    # rec_skip_* excluded by exclude pattern (overrides include).
    assert "rec_skip_001.lma" not in lma_files
    # other_* never matched include pattern.
    assert "other_001.lma" not in lma_files


def test_all_excluded_errors_with_explicit_message(tmp_path, lml_cli_binary):
    """Empty filter result -> typed error, not silent success."""
    src = tmp_path / "src"
    _make_corpus(src, ["rec_001", "rec_002"])

    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(src),
            "-o",
            str(out),
            "-r",
            "--exclude",
            "*.edf",  # nukes everything
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode != 0, "encoder should refuse on empty filter result"
    combined = r.stderr + r.stdout
    assert "filter" in combined.lower() or "excluded" in combined.lower(), combined[:600]
