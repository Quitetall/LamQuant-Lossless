"""Custom raw + sidecar full bit-exact roundtrip.

The Raw reader now stashes the literal `.raw` payload bytes as a
`SidecarBlob` (key `raw_payload_raw`) plus filename anchors so the
encoder writes byte-exact preservation copies. Combined with the
sidecar JSON preservation and stem-matched sibling copies, every
file in the source directory survives encode -> extract.
"""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _synth_raw_pair(
    src_dir: Path,
    stem: str = "rec",
    n_channels: int = 4,
    n_samples: int = 250,
    sample_rate_hz: float = 250.0,
) -> tuple[Path, Path, bytes, bytes]:
    """Build a synthetic .raw + sidecar JSON pair. Returns paths +
    bytes for both."""
    # int16 multiplexed -- sample0_ch0, sample0_ch1, ...
    samples = []
    for s in range(n_samples):
        for ch in range(n_channels):
            samples.append((ch * 100 + s) % (1 << 15))
    raw_bytes = struct.pack(f"<{n_channels * n_samples}h", *samples)
    raw_path = src_dir / f"{stem}.raw"
    raw_path.write_bytes(raw_bytes)

    sidecar = {
        "n_channels": n_channels,
        "sample_rate": sample_rate_hz,
        "dtype": "int16",
        "orientation": "multiplexed",
        "channels": [f"Ch{i}" for i in range(n_channels)],
        "phys_min": [-32768.0] * n_channels,
        "phys_max": [32767.0] * n_channels,
        "phys_dim": "uV",
    }
    sidecar_text = json.dumps(sidecar, indent=2)
    sidecar_bytes = sidecar_text.encode("utf-8")
    sidecar_path = src_dir / f"{stem}.json"
    sidecar_path.write_text(sidecar_text)

    return raw_path, sidecar_path, raw_bytes, sidecar_bytes


def test_raw_default_encode_recovers_raw_and_sidecar_byte_for_byte(
    tmp_path, lml_cli_binary
):
    """Default-mode encode -> extract recovers the `.raw` payload AND
    the sidecar JSON byte-for-byte."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    raw_path, sidecar_path, raw_bytes, sidecar_bytes = _synth_raw_pair(src_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(raw_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]
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

    recovered_raw = list(extract_dir.rglob("rec.raw"))
    assert recovered_raw, (
        f"`.raw` missing from extracted LMA: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered_raw[0].read_bytes() == raw_bytes, (
        "`.raw` payload bytes diverged across encode -> extract"
    )

    # Sidecar JSON should also be present (under whatever name the
    # locator picked -- `.json` first).
    recovered_sidecar = list(extract_dir.rglob("rec.json"))
    assert recovered_sidecar, (
        f"sidecar JSON missing from extracted LMA: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered_sidecar[0].read_bytes() == sidecar_bytes


def test_raw_default_encode_bundles_extra_siblings(tmp_path, lml_cli_binary):
    """Stem-matched annotation siblings next to the `.raw` survive
    encode -> extract."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    raw_path, sidecar_path, raw_bytes, sidecar_bytes = _synth_raw_pair(src_dir)

    siblings = {
        "rec_events.csv": b"time,event\n0.0,start\n",
        "rec_notes.txt": b"Investigator notes line.\n",
    }
    for name, payload in siblings.items():
        (src_dir / name).write_bytes(payload)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(raw_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]

    cases = {
        "rec.raw": raw_bytes,
        "rec.json": sidecar_bytes,
        **siblings,
    }
    for fname, expected in cases.items():
        recovered = list(extract_dir.rglob(fname))
        assert recovered, (
            f"{fname!r} missing from extracted LMA: "
            f"{[p.name for p in extract_dir.rglob('*')]}"
        )
        assert recovered[0].read_bytes() == expected
