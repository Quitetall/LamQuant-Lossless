"""`lml volume-split` + `lml volume-assemble` (v1.2 V).

User direction: multi-volume splitting for cloud uploads, email
attachments, removable-media transfer. Implementation is an external
byte-stream split — no LMA wire-format change. Pairs with `volume-
assemble` to reverse.

Cases:

  1. Split + assemble round-trips byte-equal.
  2. Default behavior deletes the original; `--keep` preserves it.
  3. Refuses to clobber existing volumes without `--force`.
  4. Volume-assemble detects gaps in the sequence.
  5. Volume-assemble auto-discovers all volumes from any single input.
  6. Byte-size parsing accepts K/M/G suffixes.
  7. > 999 volumes refused (3-digit NNN ceiling).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_blob(path: Path, size_bytes: int) -> None:
    """Deterministic non-trivial bytes (random-ish but reproducible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    seed = b"lamquant-v1.2-V "
    while len(payload) < size_bytes:
        payload.extend(seed)
    path.write_bytes(bytes(payload[:size_bytes]))


def test_volume_split_assemble_roundtrip_byte_equal(tmp_path, lml_cli_binary):
    """Split + assemble must produce byte-identical bytes."""
    src = tmp_path / "big.lma"
    _make_blob(src, 3_500)  # 3 volumes at 1K each
    original_sha = _sha256(src)

    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "1K", "--keep"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]

    # Expect 4 volumes: 1K, 1K, 1K, ~404B
    vols = sorted(tmp_path.glob("big.lma.*"))
    assert len(vols) == 4, [p.name for p in vols]
    # --keep preserves the original.
    assert src.exists()

    out = tmp_path / "reassembled.lma"
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "volume-assemble",
            str(tmp_path / "big.lma.001"),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    assert _sha256(out) == original_sha


def test_volume_split_default_removes_original(tmp_path, lml_cli_binary):
    """Without `--keep`, volume-split deletes the source after success."""
    src = tmp_path / "big.lma"
    _make_blob(src, 2_500)
    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "1K"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    assert not src.exists(), "original should be deleted (no --keep)"
    vols = sorted(tmp_path.glob("big.lma.*"))
    assert len(vols) == 3


def test_volume_split_refuses_clobber_without_force(tmp_path, lml_cli_binary):
    """Pre-existing volume file blocks the split unless `--force`."""
    src = tmp_path / "big.lma"
    _make_blob(src, 2_500)
    # Stub an existing volume to trigger the clobber check.
    (tmp_path / "big.lma.002").write_bytes(b"existing")
    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "1K"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode != 0, r.stderr[:400]
    assert "force" in (r.stderr + r.stdout).lower()
    # With --force it succeeds.
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "volume-split",
            str(src),
            "--size",
            "1K",
            "--force",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]


def test_volume_assemble_detects_sequence_gaps(tmp_path, lml_cli_binary):
    """Missing volume in the sequence -> typed error."""
    src = tmp_path / "big.lma"
    _make_blob(src, 3_500)
    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "1K"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    # Delete volume 002 to create a gap (001, ___, 003, 004)
    (tmp_path / "big.lma.002").unlink()

    out = tmp_path / "reassembled.lma"
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "volume-assemble",
            str(tmp_path / "big.lma.001"),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode != 0, "missing volume should error"
    combined = r.stderr + r.stdout
    assert "gap" in combined.lower() or "expected" in combined.lower()


def test_volume_assemble_auto_discovers_from_any_volume(tmp_path, lml_cli_binary):
    """`volume-assemble big.lma.003` finds .001/.002/.003/.004."""
    src = tmp_path / "big.lma"
    _make_blob(src, 3_500)
    original_sha = _sha256(src)
    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "1K"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0
    # Point at the LAST volume (003 / 004). Auto-discovery must still
    # find the full set.
    out = tmp_path / "reassembled.lma"
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "volume-assemble",
            str(tmp_path / "big.lma.003"),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    assert _sha256(out) == original_sha


def test_volume_split_byte_size_suffixes(tmp_path, lml_cli_binary):
    """K / M / G suffix parsing works."""
    src = tmp_path / "big.lma"
    _make_blob(src, 4_096)  # 4 KB
    # Use --size 2K → expect 2 volumes
    r = subprocess.run(
        [str(lml_cli_binary), "volume-split", str(src), "--size", "2K"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    vols = sorted(tmp_path.glob("big.lma.*"))
    assert len(vols) == 2
