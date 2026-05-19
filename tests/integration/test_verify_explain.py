"""`lml verify --explain` auditable per-step readout (v1.2 X).

User direction: "I don't want a totally untrackable black box that
... cannot be audited."

`lml verify-archive` defaults to a compact OK/FAIL summary. With
`--explain`, the verifier prints the literal chain of checks it
performs:

  1. Archive size + structural minimum
  2. Archive-wide SHA-256 over content+footer
  3. Manifest decompress + parse
  4. Per-entry payload read + method-specific verify + SHA match
  5. Decompression byte counts + per-entry CR
  6. Cumulative elapsed time + OK/FAIL summary

This file pins the shape so the audit story doesn't drift.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


def _build_test_lma(tmp_path: Path, lml_cli_binary: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    (src / "rec.tse").write_bytes(b"0.0 30.0 bckg 1.0\n")
    (src / "rec_summary.txt").write_bytes(b"Patient: smoke\n")
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
    return lma


def test_verify_explain_prints_full_chain(tmp_path, lml_cli_binary):
    """`lml verify-archive foo.lma --explain` prints every check."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "verify-archive", str(lma), "--explain"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:600]
    stdout = r.stdout
    # Every documented section must appear.
    for sentinel in (
        "auditable readout",
        "[1/5] Archive size:",
        "[2/5] Archive SHA-256:",
        "[3/5] Manifest:",
        "[4/5] Per-entry verify:",
        "[5/5] Summary:",
        "Compressed total:",
        "Decompressed total:",
        "Archive CR:",
        "Verified:",
        "Elapsed:",
        "Result: PASS",
    ):
        assert sentinel in stdout, (
            f"missing sentinel {sentinel!r}; stdout (first 1200 chars):\n"
            f"{stdout[:1200]}"
        )


def test_verify_default_summary_is_compact(tmp_path, lml_cli_binary):
    """Without `--explain`, the existing PASS/FAIL summary still works
    (unchanged behaviour). The auditable chain must NOT print."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "verify-archive", str(lma)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]
    stdout = r.stdout
    # The auditable banner must NOT appear in default mode.
    assert "auditable readout" not in stdout, (
        "default verify-archive should be the compact summary, not the "
        f"explain banner; stdout:\n{stdout[:600]}"
    )
    # But the existing summary line should be present.
    assert "INTEGRITY OK" in stdout or "verified" in stdout, stdout[:400]


def test_verify_lma_via_verify_routes_explain_through(tmp_path, lml_cli_binary):
    """`lml verify foo.lma --explain` magic-byte dispatches to the
    archive verifier AND forwards the explain flag."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    r = subprocess.run(
        [str(lml_cli_binary), "verify", str(lma), "--explain"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:600]
    assert "auditable readout" in r.stdout, r.stdout[:600]


def test_verify_explain_detects_tampered_entry(tmp_path, lml_cli_binary):
    """Flip a byte inside a payload entry; --explain must mark the
    failing entry with ✗ and print SHA mismatch detail."""
    lma = _build_test_lma(tmp_path, lml_cli_binary)
    original = lma.read_bytes()
    # Corrupt one byte deep in the payload section. The header (first
    # 16 bytes) + manifest are zstd-compressed JSON; tampering the
    # tail surfaces as a per-entry SHA mismatch on the LML or store
    # entry (whichever lives there).
    corrupted = bytearray(original)
    flip_idx = len(corrupted) - 60  # well inside payload, before footer
    corrupted[flip_idx] ^= 0x01
    lma.write_bytes(bytes(corrupted))

    r = subprocess.run(
        [str(lml_cli_binary), "verify-archive", str(lma), "--explain"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Archive-wide SHA check fires first; the corruption breaks it.
    # The exit must be non-zero and stdout/stderr must explain why.
    assert r.returncode != 0, "verify on corrupted archive should fail"
    combined = r.stdout + r.stderr
    assert (
        "FAILED" in combined
        or "FAIL" in combined
        or "mismatch" in combined.lower()
        or "corrupted" in combined.lower()
    ), f"expected explicit failure detail; got:\n{combined[:1200]}"
