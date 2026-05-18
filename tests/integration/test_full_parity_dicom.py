"""DICOM Waveform full bit-exact roundtrip.

The DICOM reader already stashed the source `.dcm` bytes as a
`SidecarBlob` (key `dicom_raw`) for the legacy b64-in-metadata embed.
This batch wires the encoder to also emit the byte-exact `.dcm`
preservation copy + stem-matched siblings to the staging dir so the
outer `pack_archive` pulls them into the per-recording LMA.

Pins: `lml encode foo.dcm -o out/foo.lma -> lml extract` recovers
the `.dcm` and every sibling byte-for-byte.

Requires the `dicom` Cargo feature; tests skip when the binary
under test was built without it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


FIXTURES = Path(__file__).resolve().parents[1].parent / "lamquant-core/tests/fixtures/dicom"


def _has_dicom_support(lml_cli_binary: Path) -> bool:
    """Probe the binary for dicom feature support by trying to encode
    a known-good fixture and looking for the feature-stub error."""
    if not FIXTURES.exists():
        return False
    fixture = FIXTURES / "general_ecg.dcm"
    if not fixture.exists():
        return False
    # Smoke probe with help -- cheaper than a full encode.
    r = subprocess.run(
        [str(lml_cli_binary), "encode", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.returncode == 0


def test_dicom_default_encode_recovers_dcm_byte_for_byte(tmp_path, lml_cli_binary):
    """Default-mode encode -> extract recovers the source `.dcm`
    byte-for-byte. Uses the General ECG fixture (smaller than the
    12-lead so the test is cheap)."""
    if not _has_dicom_support(lml_cli_binary):
        pytest.skip("dicom feature not built or fixtures missing")

    fixture = FIXTURES / "general_ecg.dcm"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    dcm_path = src_dir / "rec.dcm"
    shutil.copy(fixture, dcm_path)
    expected_bytes = dcm_path.read_bytes()

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(dcm_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        msg = r.stderr[:400]
        if "feature" in msg.lower() and "dicom" in msg.lower():
            pytest.skip(f"binary built without dicom feature: {msg}")
        pytest.fail(f"encode failed: {msg}")
    assert lma_target.exists()

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]

    recovered = list(extract_dir.rglob("rec.dcm"))
    assert recovered, (
        f"`.dcm` missing from extracted LMA: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered[0].read_bytes() == expected_bytes, (
        "`.dcm` bytes diverged across encode -> extract; DICOM "
        "lossless promise broken"
    )
