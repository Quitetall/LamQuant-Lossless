"""Lossless sidecar preservation — the contract that `lml encode` must
never silently drop label sidecars next to an input EDF.

Background — design contract
============================

LamQuant ships a *lossless* compression toolchain. "Lossless" here is
not just bit-exact signal recovery; it includes every artifact the
clinician or training pipeline depends on. For the major TUH EEG
datasets (TUSZ, TUEV, TUSL, TUAR, TUEP, TUAB) the event labels and
annotations live in *sibling sidecar files* next to each `.edf`:

    sub-001_ses-01_run-001.edf
    sub-001_ses-01_run-001.tse           # TUSZ seizure intervals
    sub-001_ses-01_run-001.tse_bi
    sub-001_ses-01_run-001.csv           # TUEV 6-class events
    sub-001_ses-01_run-001.csv_bi
    sub-001_ses-01_run-001.lbl           # TUSL slowing intervals
    sub-001_ses-01_run-001.lbl_bi
    sub-001_ses-01_run-001_summary.txt

If the encoder converts the `.edf` and silently drops the sidecars on
the floor, every label-dependent training pipeline downstream becomes
unrecoverable — the signal survives, the supervision does not. This
is exactly what the `feedback_end_to_end_only.md` memory entry warns
against: *only compress→store→decompress→evaluate matters*.

The contract
============

For any `lml encode <input.edf> -o <output_dir>`:

  * **Default mode** must produce a `.lma` archive that bundles the
    EDF *and* every sibling sidecar (zstd-9 per `lma.rs`). Round-
    tripping the `.lma` must recover both the bit-exact EDF and every
    sidecar byte-for-byte.

  * **Opt-in `--no-bundle` mode** must produce a `.lml` of the signal
    *and* copy every sibling sidecar to the matching location in the
    output directory. The sidecars are not compressed, but they are
    preserved next to the `.lml` exactly as they sat next to the
    `.edf`.

In both cases, a sidecar that exists on input must exist on output.
There is no "silently lossy" mode.

This file pins that contract.

Strategy
========

Build a synthetic EDF with seven differently-extensioned sidecars next
to it, invoke `lml encode`, then verify each sidecar's bytes survive
to the output side. The test is parameterized over `--no-bundle` so
both modes are pinned in one suite.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


# Every TUH-family sidecar pattern the encoder must handle. The keys are
# the file extensions (or full suffix in the case of summary files);
# the values are representative payloads — small enough to keep the
# test cheap, real enough that bytewise equality is meaningful.
SIDECARS: dict[str, bytes] = {
    ".tse": b"0.0000 12.3456 bckg 1.0000\n12.3456 30.0000 seiz 1.0000\n",
    ".tse_bi": b"0.0000 30.0000 bckg 1.0000\n",
    ".csv": b"channel,start_time,stop_time,label,confidence\nFP1-F7,0.0,4.5,seiz,0.92\n",
    ".csv_bi": b"channel,start_time,stop_time,label,confidence\nALL,0.0,30.0,bckg,1.0\n",
    ".lbl": b"montage = ar\nnumber_of_levels = {2}\nlevel = 0\n",
    ".lbl_bi": b"montage = ar_bi\nnumber_of_levels = {2}\nlevel = 0\n",
    "_summary.txt": b"Patient: test\nRecording duration: 30.0 s\nChannels: 21\n",
}


@pytest.fixture
def edf_with_sidecars(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """An EDF surrounded by realistic TUH-style sidecar files.

    Returns the path to the EDF and a {filename: bytes} dict for the
    sidecars, so the test can assert byte-for-byte equality against
    the originals after the encode/decode round trip.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    edf_path = src_dir / "recording.edf"
    create_edf(str(edf_path), n_channels=4, n_records=2, sample_rate=250)

    written: dict[str, bytes] = {}
    for suffix, payload in SIDECARS.items():
        sidecar = src_dir / f"recording{suffix}"
        sidecar.write_bytes(payload)
        written[sidecar.name] = payload

    return edf_path, written


def _encode(edf_path: Path, out_dir: Path, lml_binary: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run `lml encode` and bail loudly if it crashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(lml_binary), "encode", str(edf_path), "-o", str(out_dir), *extra],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"encode failed: rc={result.returncode}\n"
        f"stderr: {result.stderr[:400]}\nstdout: {result.stdout[:400]}"
    )
    return result


class TestSidecarPreservationDefault:
    """Default encode mode must bundle sidecars into a `.lma`."""

    def test_default_emits_lma_not_lml(
        self, edf_with_sidecars, tmp_path, lml_cli_binary
    ):
        edf_path, _ = edf_with_sidecars
        out_dir = tmp_path / "out_default"
        _encode(edf_path, out_dir, lml_cli_binary)

        # Default lossless output is the LMA archive — never a bare
        # `.lml`, which would be a silent drop of every sidecar.
        lma_files = list(out_dir.rglob("*.lma"))
        lml_files = list(out_dir.rglob("*.lml"))
        assert len(lma_files) == 1, (
            f"default encode must emit exactly one .lma archive; "
            f"got {lma_files} (and bare .lml: {lml_files})"
        )
        assert not any(
            f.parent == out_dir and f.suffix == ".lml" for f in lml_files
        ), f"bare .lml in default output dir is the silently-lossy bug: {lml_files}"

    def test_lma_bundles_every_sidecar(
        self, edf_with_sidecars, tmp_path, lml_cli_binary
    ):
        edf_path, originals = edf_with_sidecars
        out_dir = tmp_path / "out_default"
        _encode(edf_path, out_dir, lml_cli_binary)

        archives = list(out_dir.rglob("*.lma"))
        assert len(archives) == 1, archives
        archive = archives[0]

        # `lml list-archive` must enumerate every original sidecar.
        listing = subprocess.run(
            [str(lml_cli_binary), "list-archive", str(archive)],
            capture_output=True, text=True, timeout=15,
        )
        assert listing.returncode == 0, listing.stderr[:200]
        for sidecar_name in originals:
            assert sidecar_name in listing.stdout, (
                f"sidecar {sidecar_name!r} missing from archive listing — "
                f"this is the silent-data-loss bug. listing:\n{listing.stdout[:600]}"
            )

    def test_extract_recovers_every_sidecar_byte_for_byte(
        self, edf_with_sidecars, tmp_path, lml_cli_binary
    ):
        edf_path, originals = edf_with_sidecars
        out_dir = tmp_path / "out_default"
        _encode(edf_path, out_dir, lml_cli_binary)
        archive = next(out_dir.rglob("*.lma"))

        extracted = tmp_path / "extracted"
        extracted.mkdir()
        result = subprocess.run(
            [str(lml_cli_binary), "extract", str(archive), "-o", str(extracted)],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr[:300]

        for sidecar_name, expected_bytes in originals.items():
            recovered = list(extracted.rglob(sidecar_name))
            assert recovered, (
                f"extracted archive missing sidecar {sidecar_name!r}; "
                f"present: {[p.name for p in extracted.rglob('*')]}"
            )
            assert recovered[0].read_bytes() == expected_bytes, (
                f"sidecar {sidecar_name!r} bytes diverged after roundtrip"
            )


class TestSidecarPreservationNoBundle:
    """`--no-bundle` mode keeps the legacy `.lml` signal output but is
    required to copy sidecars to the output dir at mirror locations —
    so the operator who deliberately picks the smaller artifact still
    does not lose labels.
    """

    def test_no_bundle_emits_lml(self, edf_with_sidecars, tmp_path, lml_cli_binary):
        edf_path, _ = edf_with_sidecars
        out_dir = tmp_path / "out_no_bundle"
        _encode(edf_path, out_dir, lml_cli_binary, "--no-bundle")

        lml_files = list(out_dir.rglob("*.lml"))
        assert len(lml_files) == 1, (
            f"--no-bundle must emit exactly one .lml; got {lml_files}"
        )

    def test_no_bundle_copies_every_sidecar_to_output(
        self, edf_with_sidecars, tmp_path, lml_cli_binary
    ):
        edf_path, originals = edf_with_sidecars
        out_dir = tmp_path / "out_no_bundle"
        _encode(edf_path, out_dir, lml_cli_binary, "--no-bundle")

        for sidecar_name, expected_bytes in originals.items():
            mirrored = list(out_dir.rglob(sidecar_name))
            assert mirrored, (
                f"--no-bundle dropped sidecar {sidecar_name!r}; this is the "
                f"silent-data-loss bug. Output dir contents: "
                f"{[p.name for p in out_dir.rglob('*')]}"
            )
            assert mirrored[0].read_bytes() == expected_bytes, (
                f"--no-bundle copied {sidecar_name!r} but bytes diverged"
            )


class TestSidecarLossWarnings:
    """If someone *forces* the silently-lossy historical behaviour
    (whatever shape it takes — perhaps a future `--no-sidecars` flag),
    the encoder must warn loudly on stderr when sidecars exist on the
    input side. We do not ship a silent drop.
    """

    def test_warning_emitted_when_sidecars_skipped(
        self, edf_with_sidecars, tmp_path, lml_cli_binary
    ):
        edf_path, originals = edf_with_sidecars
        out_dir = tmp_path / "out_warn"
        # This subtest tolerates either of:
        #   (a) the encoder rejects the request entirely (no
        #       silently-lossy flag exists), or
        #   (b) the encoder accepts it but emits a per-sidecar
        #       warning on stderr.
        # What it forbids: silent success with sidecars dropped and
        # no operator-visible signal.
        result = subprocess.run(
            [str(lml_cli_binary), "encode", str(edf_path),
             "-o", str(out_dir), "--no-bundle", "--drop-sidecars"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            # (a) — encoder refused. Acceptable.
            return
        # (b) — accepted. Every sidecar must be named in stderr.
        for sidecar_name in originals:
            assert sidecar_name in result.stderr, (
                f"--drop-sidecars succeeded silently for {sidecar_name!r}; "
                f"stderr did not mention the loss. Operator must see it."
            )
