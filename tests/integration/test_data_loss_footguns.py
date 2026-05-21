"""Data-loss footgun guards on `lml encode`.

Pins the user-facing contracts that no data is ever lost by accident:

  * Default mode (no `--no-bundle` / `--bare-lml` flag) emits a per-
    recording `.lma` archive that bundles the `.lml` plus every sibling
    sidecar via the LML -> zstd -> store cascade. There is no silent
    drop possible.

  * `--no-bundle` / `--bare-lml` prints a loud 20-line warning to
    stderr on every invocation. The warning is silenced only when
    paired with the explicit `--i-understand-data-loss` co-flag.

  * The cascade itself (covered structurally by lma.rs tests) is
    LML for EDF/BDF -> zstd for everything else -> 1:1 store for any
    file that compresses negatively or fails to encode. Every file in
    scope appears in the LMA as exactly one entry.

This file is the user-facing safety net. If any of these assertions
break, the encoder has regressed in a way that can lose end-user
data; ship-blocker by design.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


# ----------------------------------------------------------------------
# Defaults always produce LMA, never bare LML
# ----------------------------------------------------------------------


def test_default_encode_emits_lma_not_bare_lml(tmp_path, lml_cli_binary):
    """`lml encode foo.edf -o out/foo.lma` (no flags) produces an LMA."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert lma_target.exists(), f"expected .lma at {lma_target}, got {list(out_dir.iterdir())}"


# ----------------------------------------------------------------------
# --no-bundle / --bare-lml print loud warning
# ----------------------------------------------------------------------


WARNING_SENTINELS = [
    "ERROR: --no-bundle / --bare-lml refuses to run without",
    "--i-understand-data-loss",
]


def test_no_bundle_prints_loud_warning(tmp_path, lml_cli_binary):
    """`--no-bundle` without the co-flag is a hard refuse (rc=2) with the
    full ERROR paragraph on stderr. Tier 3 audit (O2): the previous
    warn-and-proceed shape was silently survivable under `2>/dev/null`.
    """
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf), "-o", str(out_dir), "--no-bundle"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 2, (
        f"--no-bundle without the co-flag must exit 2; got "
        f"rc={r.returncode}\nstderr:\n{r.stderr[:1200]}"
    )
    for sentinel in WARNING_SENTINELS:
        assert sentinel in r.stderr, (
            f"missing sentinel {sentinel!r}; stderr:\n{r.stderr[:1200]}"
        )


def test_bare_lml_alias_also_warns(tmp_path, lml_cli_binary):
    """`--bare-lml` is an alias for `--no-bundle`; same hard refuse fires."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf), "-o", str(out_dir), "--bare-lml"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 2, (
        f"--bare-lml without the co-flag must exit 2; got "
        f"rc={r.returncode}\nstderr:\n{r.stderr[:1200]}"
    )
    for sentinel in WARNING_SENTINELS:
        assert sentinel in r.stderr, (
            f"missing sentinel {sentinel!r}; stderr:\n{r.stderr[:1200]}"
        )


def test_i_understand_data_loss_silences_warning(tmp_path, lml_cli_binary):
    """`--no-bundle --i-understand-data-loss` runs without the warning."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(edf),
            "-o",
            str(out_dir),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    for sentinel in WARNING_SENTINELS:
        assert sentinel not in r.stderr, (
            f"`--i-understand-data-loss` failed to silence warning "
            f"{sentinel!r}; stderr:\n{r.stderr[:1200]}"
        )


def test_i_understand_data_loss_alone_is_a_noop(tmp_path, lml_cli_binary):
    """`--i-understand-data-loss` without `--no-bundle` does nothing
    surprising -- default mode still produces the LMA, no warning."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(edf),
            "-o",
            str(lma_target),
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert lma_target.exists()
    for sentinel in WARNING_SENTINELS:
        assert sentinel not in r.stderr, (
            f"unexpected data-loss warning in default-mode run: "
            f"{sentinel!r}; stderr:\n{r.stderr[:600]}"
        )


# ----------------------------------------------------------------------
# Cascade: every file in scope ends up in the LMA, byte-recoverable
# ----------------------------------------------------------------------


def test_default_cascade_bundles_edf_plus_sidecars_byte_recoverable(
    tmp_path, lml_cli_binary
):
    """Default-mode encode of an EDF with mixed-extension siblings
    produces an LMA whose `lml extract` recovers every byte of every
    sibling. Pins the LML -> zstd -> store cascade end-to-end."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)

    # One sibling per cascade tier:
    #   * TSE -> text -> zstd entry
    #   * binary blob -> negative compression -> store entry
    #   * summary -> text -> zstd entry
    siblings = {
        "rec.tse": b"0.0 30.0 bckg 1.0\n",
        "rec.bin": os.urandom(4096),  # random bytes -> zstd-negative -> store
        "rec_summary.txt": b"Patient: test\nDuration: 30.0\n",
    }
    for name, payload in siblings.items():
        (src / name).write_bytes(payload)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    lma_target = out_dir / "rec.lma"

    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf), "-o", str(lma_target)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert lma_target.exists()

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [str(lml_cli_binary), "extract", str(lma_target), "-o", str(extract_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr

    for name, expected in siblings.items():
        recovered = list(extract_dir.rglob(name))
        assert recovered, (
            f"sibling {name!r} not recovered from LMA; "
            f"present: {[p.name for p in extract_dir.rglob('*')]}"
        )
        assert recovered[0].read_bytes() == expected, (
            f"sibling {name!r} bytes diverged after roundtrip"
        )
