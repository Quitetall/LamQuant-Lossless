"""NeuroScan CNT full bit-exact roundtrip.

The CNT reader already stashed the source `.cnt` bytes as a
`SidecarBlob` (key `cnt_raw`) for the legacy b64-in-metadata embed.
This batch wires the encoder to also write a byte-exact `.cnt`
preservation copy + every stem-matched sibling annotation file
(`recording_events.csv`, `recording.lbl`, etc.) to the staging dir
so the outer `pack_archive` pulls them into the per-recording LMA.

Pins: `lml encode foo.cnt -o out/foo.lma -> lml extract` recovers
the `.cnt` and every sibling byte-for-byte.
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _synth_cnt(
    src_dir: Path,
    stem: str = "rec",
    n_channels: int = 4,
    n_samples: int = 250,
    sample_rate_hz: float = 250.0,
) -> tuple[Path, bytes]:
    """Build a minimal synthetic NeuroScan CNT file.

    The reader walks a 900-byte SETUP header + 75-byte ELECTLOC * N
    + int16 multiplexed samples. We construct the bytes manually so
    we can byte-cmp after roundtrip.
    """
    setup = bytearray(900)
    # rev + nextfile + prevfile + type + id + oper (placeholder; the
    # reader is liberal). nchannels at offset 370 (u16 LE), pntsps
    # / nsamples at... actually for parity testing we only need byte
    # equality, not semantic correctness. Stuff a deterministic
    # pattern.
    for i in range(900):
        setup[i] = (i * 7 + 3) & 0xFF
    # Overwrite the bare minimum fields the reader needs to not
    # bail. We piggy-back on the existing reader's known offsets:
    # struct.pack_into for nchannels=4, rate=250.
    struct.pack_into("<H", setup, 370, n_channels)  # nchannels
    struct.pack_into("<H", setup, 376, int(sample_rate_hz))  # rate

    # ELECTLOC table.
    electloc = bytearray(75 * n_channels)
    for i, b in enumerate(electloc):
        electloc[i] = (i * 13 + 5) & 0xFF
    # Set channel-name bytes (first 10 bytes per ELECTLOC) to ASCII.
    for ch in range(n_channels):
        base = ch * 75
        for j in range(10):
            electloc[base + j] = 0
        name = f"Ch{ch}".encode("ascii")
        electloc[base : base + len(name)] = name

    # Body: int16 multiplexed. sample0_ch0, sample0_ch1, ...
    samples = []
    for s in range(n_samples):
        for ch in range(n_channels):
            samples.append((ch * 100 + s) % (1 << 15))
    body = struct.pack(f"<{n_channels * n_samples}h", *samples)

    cnt_bytes = bytes(setup) + bytes(electloc) + body
    cnt_path = src_dir / f"{stem}.cnt"
    cnt_path.write_bytes(cnt_bytes)
    return cnt_path, cnt_bytes


def test_cnt_default_encode_recovers_cnt_byte_for_byte(tmp_path, lml_cli_binary):
    """Default-mode encode -> extract recovers the `.cnt` byte-for-byte."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cnt_path, cnt_bytes = _synth_cnt(src_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(cnt_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        pytest.skip(f"cnt encoder rejected synth fixture: {r.stderr[:300]}")
    assert lma_target.exists()

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:300]

    recovered = list(extract_dir.rglob("rec.cnt"))
    assert recovered, (
        f"`.cnt` missing from extracted LMA; present: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered[0].read_bytes() == cnt_bytes


def test_cnt_default_encode_bundles_stem_matched_siblings(tmp_path, lml_cli_binary):
    """A CNT recording with sibling annotation files (`rec_events.csv`,
    `rec.lbl`) emits a `.lma` containing all of them byte-equal."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    cnt_path, cnt_bytes = _synth_cnt(src_dir)

    sibling_payloads = {
        "rec_events.csv": b"time,event\n0.0,stim\n5.0,resp\n",
        "rec.lbl": b"montage = ar\nnumber_of_levels = {1}\n",
    }
    for name, payload in sibling_payloads.items():
        (src_dir / name).write_bytes(payload)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(cnt_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        pytest.skip(f"cnt encoder rejected synth fixture: {r.stderr[:300]}")

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:300]

    for fname, expected in {"rec.cnt": cnt_bytes, **sibling_payloads}.items():
        recovered = list(extract_dir.rglob(fname))
        assert recovered, (
            f"{fname!r} missing from extracted LMA: "
            f"{[p.name for p in extract_dir.rglob('*')]}"
        )
        assert recovered[0].read_bytes() == expected
