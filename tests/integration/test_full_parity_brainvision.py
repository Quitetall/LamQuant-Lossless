"""BrainVision full bit-exact roundtrip — `.vhdr` + `.vmrk` + `.eeg`.

The BrainVision reader already stashed the original `.vhdr` and `.vmrk`
bytes as `SidecarBlob`s for the b64-in-metadata embed. The `.eeg`
payload was loaded for signal decode but never preserved as bytes;
encoder relied on the b64 metadata for source reconstruction.

This batch flips the architecture: the reader now stashes the `.eeg`
bytes too (plus filename anchors from `DataFile=` / `MarkerFile=` in
the .vhdr) and the encoder writes byte-exact preservation copies to
`lml_path`'s parent dir. The outer `pack_archive` pulls them into the
per-recording `.lma` (default) or they sit as siblings of the bare
`.lml` under `--no-bundle`.

This file pins the contract end-to-end: `lml encode foo.vhdr ->
lml extract` recovers `.vhdr`, `.vmrk`, and `.eeg` byte-for-byte.
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _synth_brainvision_triple(
    src_dir: Path,
    stem: str = "rec",
    n_channels: int = 4,
    n_samples: int = 250,
    sample_rate_hz: float = 250.0,
) -> tuple[Path, Path, Path, bytes, bytes, bytes]:
    """Build a synthetic BrainVision `.vhdr` + `.vmrk` + `.eeg` set.

    Returns (vhdr_path, vmrk_path, eeg_path, vhdr_bytes, vmrk_bytes,
    eeg_bytes). The signals are int16, multiplexed, channel-major,
    deterministic ramp so byte-cmp on the .eeg is meaningful.
    """
    sample_interval_us = 1_000_000.0 / sample_rate_hz

    # `.eeg`: int16 multiplexed. sample0_ch0, sample0_ch1, ...
    samples = []
    for s in range(n_samples):
        for ch in range(n_channels):
            samples.append((ch * 100 + s) % (1 << 15))
    eeg_bytes = struct.pack(f"<{n_channels * n_samples}h", *samples)
    eeg_path = src_dir / f"{stem}.eeg"
    eeg_path.write_bytes(eeg_bytes)

    # `.vmrk`: minimal markers file.
    vmrk_text = (
        "BrainVision Data Exchange Marker File, Version 1.0\n"
        "\n"
        "[Common Infos]\n"
        f"DataFile={stem}.eeg\n"
        "\n"
        "[Marker Infos]\n"
        "Mk1=New Segment,,1,0,0,20251018120000\n"
    )
    vmrk_bytes = vmrk_text.encode("utf-8")
    vmrk_path = src_dir / f"{stem}.vmrk"
    vmrk_path.write_bytes(vmrk_bytes)

    # `.vhdr`: header.
    channels_section = "\n".join(
        f"Ch{i + 1}=ch{i},,1.0,uV" for i in range(n_channels)
    )
    vhdr_text = (
        "BrainVision Data Exchange Header File Version 1.0\n"
        "\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={stem}.eeg\n"
        f"MarkerFile={stem}.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_channels}\n"
        f"SamplingInterval={sample_interval_us}\n"
        "\n"
        "[Binary Infos]\n"
        "BinaryFormat=INT_16\n"
        "\n"
        "[Channel Infos]\n"
        f"{channels_section}\n"
    )
    vhdr_bytes = vhdr_text.encode("utf-8")
    vhdr_path = src_dir / f"{stem}.vhdr"
    vhdr_path.write_bytes(vhdr_bytes)

    return vhdr_path, vmrk_path, eeg_path, vhdr_bytes, vmrk_bytes, eeg_bytes


def test_brainvision_default_recovers_vhdr_vmrk_eeg_byte_for_byte(
    tmp_path, lml_cli_binary
):
    """Default-mode `lml encode foo.vhdr -o out/foo.lma` bundles
    `.vhdr` + `.vmrk` + `.eeg` into the LMA. `lml extract` recovers
    all three byte-for-byte."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    vhdr_path, vmrk_path, eeg_path, vhdr_bytes, vmrk_bytes, eeg_bytes = (
        _synth_brainvision_triple(src_dir, stem="rec")
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(vhdr_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:600]}"
    assert lma_target.exists()

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"extract failed: {r.stderr[:500]}"

    cases = [
        ("rec.vhdr", vhdr_bytes),
        ("rec.vmrk", vmrk_bytes),
        ("rec.eeg", eeg_bytes),
    ]
    for fname, expected in cases:
        recovered = list(extract_dir.rglob(fname))
        assert recovered, (
            f"{fname!r} missing from extracted LMA; present: "
            f"{[p.name for p in extract_dir.rglob('*')]}"
        )
        assert recovered[0].read_bytes() == expected, (
            f"{fname!r} bytes diverged across encode -> extract; "
            f"BrainVision lossless promise broken"
        )


def test_brainvision_no_bundle_mirrors_all_three_files(tmp_path, lml_cli_binary):
    """`--no-bundle --i-understand-data-loss` keeps the bare `.lml`
    plus all three source files sitting next to it as siblings."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    vhdr_path, _vmrk_path, _eeg_path, vhdr_bytes, vmrk_bytes, eeg_bytes = (
        _synth_brainvision_triple(src_dir, stem="rec")
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(vhdr_path),
            "-o",
            str(out_dir),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:600]}"

    for fname, expected in [
        ("rec.vhdr", vhdr_bytes),
        ("rec.vmrk", vmrk_bytes),
        ("rec.eeg", eeg_bytes),
    ]:
        mirrored = list(out_dir.rglob(fname))
        assert mirrored, (
            f"{fname!r} missing from --no-bundle output dir: "
            f"{[p.name for p in out_dir.rglob('*')]}"
        )
        assert mirrored[0].read_bytes() == expected


def test_brainvision_in_place_encode_does_not_touch_source(tmp_path, lml_cli_binary):
    """Encoding in-place (`-o <source-dir>/rec.lml`) must not rewrite
    the source `.vhdr` / `.vmrk` / `.eeg` files. Same canonical-parent
    short-circuit as the EEGLAB encoder."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    vhdr_path, vmrk_path, eeg_path, vhdr_bytes, vmrk_bytes, eeg_bytes = (
        _synth_brainvision_triple(src_dir, stem="rec")
    )

    mtimes_before = {
        p: os.stat(p).st_mtime_ns for p in [vhdr_path, vmrk_path, eeg_path]
    }

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(vhdr_path),
            "-o",
            str(src_dir / "rec.lml"),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:600]}"

    assert vhdr_path.read_bytes() == vhdr_bytes
    assert vmrk_path.read_bytes() == vmrk_bytes
    assert eeg_path.read_bytes() == eeg_bytes

    for p, before in mtimes_before.items():
        after = os.stat(p).st_mtime_ns
        assert before == after, (
            f"encoder rewrote source {p.name!r} despite same-dir guard"
        )
