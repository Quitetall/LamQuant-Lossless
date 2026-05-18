"""EEGLAB full bit-exact roundtrip — `.set` MAT v5 + `.fdt` payload.

Before this batch, `EeglabReader::read_bundle` extracted only the
fields the codec needed (nbchan, pnts, srate, channel labels) from
the sibling `<name>.lml-meta.json` sidecar and dropped every other
`EEG.*` field in the original MAT v5 struct:

  * `EEG.event`            (clinical event markers)
  * `EEG.urevent`           (original-time event markers)
  * `EEG.chanlocs.{theta,radius,X,Y,Z,sph_theta,sph_phi,sph_radius}`
                            (electrode geometry)
  * `EEG.icaweights` / `EEG.icasphere` / `EEG.icachansind`
                            (ICA decomposition)
  * `EEG.reject` / `EEG.stats`
  * `EEG.history`           (analysis provenance)
  * `EEG.etc` / `EEG.comments` / `EEG.ref` / `EEG.datasubject`
  * Anything else stored inside the MAT v5 struct

The signal was lossless (f32 bit-cast), but every other byte was lost
the moment the operator deleted the source `.set`. That violates the
"no data ever lost" invariant.

The fix:

  * `EeglabReader::read_bundle` now stashes the original `.set` and
    `.fdt` bytes as opaque `SidecarBlob`s (keys `set_raw` / `fdt_raw`).
  * `encode_one_eeglab` writes those blobs alongside the `.lml` so
    the outer `pack_archive` pulls them into the per-recording `.lma`
    (default) OR they sit as siblings of the bare `.lml` under
    `--no-bundle`.
  * `lml extract` recovers both files byte-for-byte.

This file pins the contract end-to-end. If any byte of the `.set` or
`.fdt` diverges across encode → extract, the EEGLAB lossless promise
has regressed.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _synth_eeglab_triple(
    src_dir: Path,
    stem: str,
    n_channels: int = 4,
    n_samples: int = 100,
    sample_rate: float = 250.0,
    channels: list[str] | None = None,
) -> tuple[Path, Path, Path, bytes, bytes]:
    """Build a synthetic `.set` + `.fdt` + sidecar JSON triple.

    `.set` is a 2 KB opaque blob (the encoder doesn't parse it; we
    only need to assert that whatever the reader stashed lands
    byte-for-byte on the other side). `.fdt` is channel-major float32.
    Returns (set_path, fdt_path, json_path, set_bytes, fdt_bytes).
    """
    if channels is None:
        channels = [f"Ch{i}" for i in range(n_channels)]

    # `.set`: opaque payload. Stuff a deterministic byte pattern so
    # any single-byte drift is detectable on byte-cmp.
    set_path = src_dir / f"{stem}.set"
    set_bytes = b"MAT5OPAQUE" + bytes(range(256)) * 8 + b"\x00END\x00"
    set_path.write_bytes(set_bytes)

    # `.fdt`: channel-major float32 ramp signal. Bit-pattern is
    # deterministic so post-roundtrip we can byte-cmp it directly.
    samples = []
    for ch in range(n_channels):
        for s in range(n_samples):
            samples.append(float(ch * 1000 + s))
    fdt_path = src_dir / f"{stem}.fdt"
    fdt_bytes = struct.pack(f"<{n_channels * n_samples}f", *samples)
    fdt_path.write_bytes(fdt_bytes)

    # Metadata sidecar (the codec contract).
    json_path = src_dir / f"{stem}.lml-meta.json"
    meta = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "channels": channels,
        "phys_dim": "uV",
    }
    json_path.write_text(json.dumps(meta, indent=2))

    return set_path, fdt_path, json_path, set_bytes, fdt_bytes


def test_eeglab_default_encode_recovers_set_and_fdt_byte_for_byte(
    tmp_path, lml_cli_binary
):
    """`lml encode foo.set -o out/foo.lma` (default mode) bundles the
    full `.set` MAT v5 + `.fdt` bytes into the archive. `lml extract`
    recovers both byte-for-byte. The MAT struct's hidden fields
    (events, urevents, chanlocs xyz, ICA matrices, history) survive
    intact because we preserve the raw bytes, not a parsed subset."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    set_path, fdt_path, _json_path, set_bytes, fdt_bytes = _synth_eeglab_triple(
        src_dir, stem="rec"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(set_path), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:500]}"
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

    recovered_set = list(extract_dir.rglob("rec.set"))
    assert recovered_set, (
        f"`.set` missing from extracted archive: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered_set[0].read_bytes() == set_bytes, (
        "`.set` MAT v5 bytes diverged across encode -> extract; "
        "the EEGLAB lossless invariant is broken"
    )

    recovered_fdt = list(extract_dir.rglob("rec.fdt"))
    assert recovered_fdt, (
        f"`.fdt` missing from extracted archive: "
        f"{[p.name for p in extract_dir.rglob('*')]}"
    )
    assert recovered_fdt[0].read_bytes() == fdt_bytes, (
        "`.fdt` float32 bytes diverged across encode -> extract"
    )


def test_eeglab_no_bundle_mirror_copies_set_and_fdt_to_output_dir(
    tmp_path, lml_cli_binary
):
    """`--no-bundle --i-understand-data-loss` mode: bare `.lml` plus
    sibling `.set` + `.fdt` in the output dir, byte-for-byte equal.
    Operator who keeps the whole output dir together still has full
    parity; only an operator who moves the `.lml` away from its
    siblings will lose them, and that's the contract we wave about
    in the warning paragraph."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    set_path, _fdt_path, _json_path, set_bytes, fdt_bytes = _synth_eeglab_triple(
        src_dir, stem="rec"
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(set_path),
            "-o",
            str(out_dir),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:500]}"

    mirrored_set = list(out_dir.rglob("rec.set"))
    assert mirrored_set, (
        f"`.set` missing from --no-bundle output dir: "
        f"{[p.name for p in out_dir.rglob('*')]}"
    )
    assert mirrored_set[0].read_bytes() == set_bytes

    mirrored_fdt = list(out_dir.rglob("rec.fdt"))
    assert mirrored_fdt, (
        f"`.fdt` missing from --no-bundle output dir: "
        f"{[p.name for p in out_dir.rglob('*')]}"
    )
    assert mirrored_fdt[0].read_bytes() == fdt_bytes


def test_eeglab_encode_does_not_overwrite_source_when_out_is_source_dir(
    tmp_path, lml_cli_binary
):
    """When the user points `-o` back at the source directory (e.g.
    in-place encode), we must NOT overwrite the source `.set` /
    `.fdt` with the preservation copy -- they're the same file. The
    encoder canonical-compares parents and short-circuits the copy."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    set_path, fdt_path, _json_path, set_bytes, fdt_bytes = _synth_eeglab_triple(
        src_dir, stem="rec"
    )

    # Capture mtimes pre-encode.
    set_mtime_before = os.stat(set_path).st_mtime_ns
    fdt_mtime_before = os.stat(fdt_path).st_mtime_ns

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(set_path),
            "-o",
            str(src_dir / "rec.lml"),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"encode failed: {r.stderr[:500]}"

    # Source bytes still match originals.
    assert set_path.read_bytes() == set_bytes
    assert fdt_path.read_bytes() == fdt_bytes

    # And ideally the mtime hasn't changed either (no needless rewrite).
    set_mtime_after = os.stat(set_path).st_mtime_ns
    fdt_mtime_after = os.stat(fdt_path).st_mtime_ns
    assert set_mtime_before == set_mtime_after, (
        "encoder rewrote source `.set` despite same-dir guard"
    )
    assert fdt_mtime_before == fdt_mtime_after, (
        "encoder rewrote source `.fdt` despite same-dir guard"
    )
