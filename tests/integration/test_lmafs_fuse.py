"""lmafs FUSE filesystem — mount + browse + cat round-trip.

v1.1 Group U. Pins the contract that `lmafs foo.lma /mnt/foo`
exposes every archive entry as a regular file in the mountpoint,
with byte-exact content recovery via `cat`. This is the path that
makes Dolphin / Nautilus / Thunar etc. open `.lma` archives as
directories.

Skips automatically when:
  * `fusermount` is not on PATH (no FUSE userspace).
  * /dev/fuse is not accessible.
  * The lmafs binary hasn't been built yet.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


def _have_fuse() -> bool:
    if not shutil.which("fusermount") and not shutil.which("fusermount3"):
        return False
    # Probe /dev/fuse readability; CI runners often lack it.
    return os.path.exists("/dev/fuse")


def _lmafs_binary() -> Path | None:
    candidates = [
        Path("/mnt/4tb/LamQuant/target/release/lmafs"),
        Path("/mnt/4tb/LamQuant/target/debug/lmafs"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


@pytest.fixture
def lmafs_bin():
    binary = _lmafs_binary()
    if binary is None:
        pytest.skip("lmafs binary not built; run `cargo build --release -p lmafs`")
    return binary


@pytest.fixture
def fuse_available():
    if not _have_fuse():
        pytest.skip("FUSE not available on this host (fusermount missing or /dev/fuse inaccessible)")


def _unmount(mountpoint: Path):
    """Force-unmount the FUSE mount. Tolerant of repeated unmounts."""
    for cmd in (["fusermount", "-u", str(mountpoint)], ["fusermount3", "-u", str(mountpoint)]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, capture_output=True, timeout=10)
            return


def test_lmafs_mounts_and_lists_entries(
    tmp_path, lml_cli_binary, lmafs_bin, fuse_available
):
    """`lmafs foo.lma /mnt/foo` mounts; `ls /mnt/foo` lists every entry."""
    # Build a synth archive with deterministic entries.
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    (src / "rec.tse").write_bytes(b"0.0 30.0 bckg 1.0\n")
    (src / "rec_summary.txt").write_bytes(b"Patient: test\n")

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

    mount = tmp_path / "mnt"
    mount.mkdir()

    # Foreground mount in a background subprocess; we kill it after.
    proc = subprocess.Popen(
        [str(lmafs_bin), "--foreground", str(lma), str(mount)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 5s for the mount to appear.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                # Mountpoint shows the FS once mounted.
                entries = sorted(p.name for p in mount.iterdir())
            except (FileNotFoundError, PermissionError):
                entries = []
            if entries:
                break
            time.sleep(0.1)
        assert entries, (
            f"lmafs did not surface entries within 5s. Process stderr:\n"
            f"{proc.stderr.read(2000).decode('utf-8', errors='replace') if proc.stderr else ''}"
        )

        # Every archive entry must appear.
        for expected in ("rec.lml", "rec.tse", "rec_summary.txt"):
            assert expected in entries, (
                f"{expected!r} missing from mount; got {entries}"
            )

        # Byte-exact recovery via `cat` (filesystem read path).
        assert (mount / "rec.tse").read_bytes() == b"0.0 30.0 bckg 1.0\n"
        assert (mount / "rec_summary.txt").read_bytes() == b"Patient: test\n"
    finally:
        _unmount(mount)
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
