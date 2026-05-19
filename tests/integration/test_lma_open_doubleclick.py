"""`lma-open` double-click handler (v1.3).

User direction: "I want it so that when you double-click a .lma file,
it opens up a directory inside where you can view each file."

`installer/lma-open` is the .desktop Exec target. It:
  1. Refuses non-LMA inputs (magic-byte check).
  2. Computes a deterministic mountpoint under
     `~/.cache/lamquant/mounts/<archive>_<sha16>`.
  3. Mounts via `lmafs --foreground` in the background (nohup + setsid).
  4. Polls up to 5 s for the mount to appear.
  5. xdg-opens the mountpoint → default file manager opens it as a dir.

This test exercises the mount + ls + cat path without invoking
xdg-open (uses `XDG_OPEN_BIN=true` to no-op the file-manager launch).
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
    return os.path.exists("/dev/fuse")


def _lmafs_binary() -> Path | None:
    for c in (
        Path("/mnt/4tb/LamQuant/target/release/lmafs"),
        Path("/mnt/4tb/LamQuant/target/debug/lmafs"),
    ):
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def _lma_open_script() -> Path:
    return Path("/mnt/4tb/LamQuant/installer/lma-open")


def _unmount_all(prefix: Path):
    if not prefix.exists():
        return
    for mp in prefix.iterdir():
        if mp.is_dir():
            for cmd in (
                ["fusermount", "-u", str(mp)],
                ["fusermount3", "-u", str(mp)],
            ):
                if shutil.which(cmd[0]):
                    subprocess.run(cmd, capture_output=True, timeout=10)
                    break


def test_lma_open_mounts_and_lists_archive(tmp_path, lml_cli_binary):
    """`lma-open foo.lma` mounts the archive; entries appear under
    `~/.cache/lamquant/mounts/foo.lma_<hash>/`."""
    if not _have_fuse():
        pytest.skip("FUSE not available")
    if _lmafs_binary() is None:
        pytest.skip("lmafs binary not built")
    lma_open = _lma_open_script()
    if not lma_open.exists():
        pytest.skip(f"lma-open helper missing at {lma_open}")

    # Defensive: clear stale mounts from previous test / smoke-test runs.
    mount_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "lamquant/mounts"
    _unmount_all(mount_root)
    if mount_root.exists():
        for mp in list(mount_root.iterdir()):
            try:
                mp.rmdir()
            except OSError:
                pass

    # Build a synth archive in tmp_path with two entries.
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    (src / "rec.tse").write_bytes(b"0.0 30.0 bckg 1.0\n")
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

    # Spawn lma-open with xdg-open stubbed to /bin/true so the test
    # doesn't try to actually open a GUI file manager.
    env = {**os.environ, "PATH": f"/usr/bin:/bin:/tmp/lma-open-stub"}
    stub_dir = Path("/tmp/lma-open-stub")
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "xdg-open").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "xdg-open").chmod(0o755)

    proc = subprocess.run(
        [str(lma_open), str(lma)],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    try:
        assert proc.returncode == 0, (
            f"lma-open exit {proc.returncode}; stderr:\n{proc.stderr[:600]}"
        )

        # Compute the deterministic mountpoint path the script uses --
        # sha256(realpath(archive))[:16] -- and look up that exact
        # directory. Glob would pick up stale mounts from earlier runs.
        import hashlib

        archive_real = str(lma.resolve())
        archive_hash = hashlib.sha256(archive_real.encode()).hexdigest()[:16]
        mount_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "lamquant/mounts"
        mountpoint = mount_root / f"rec.lma_{archive_hash}"
        assert mountpoint.exists(), (
            f"expected mountpoint at {mountpoint}; existing dirs: "
            f"{list(mount_root.iterdir()) if mount_root.exists() else 'no root'}"
        )

        # Confirm the mount surfaces entries.
        deadline = time.monotonic() + 5.0
        names: list[str] = []
        while time.monotonic() < deadline:
            try:
                names = sorted(p.name for p in mountpoint.iterdir())
            except (FileNotFoundError, PermissionError):
                names = []
            if names:
                break
            time.sleep(0.1)
        assert "rec.lml" in names, f"rec.lml missing from {mountpoint}; got {names}"
        assert "rec.tse" in names, f"rec.tse missing from {mountpoint}; got {names}"
        # Byte-equal sibling content.
        assert (mountpoint / "rec.tse").read_bytes() == b"0.0 30.0 bckg 1.0\n"
    finally:
        _unmount_all(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "lamquant/mounts")


def test_lma_open_refuses_non_lma_input(tmp_path):
    """`lma-open` magic-byte rejection -- non-LMA file errors before
    invoking lmafs."""
    if not _have_fuse():
        pytest.skip("FUSE not available")
    lma_open = _lma_open_script()
    if not lma_open.exists():
        pytest.skip(f"lma-open helper missing at {lma_open}")

    not_an_lma = tmp_path / "fake.lma"
    not_an_lma.write_bytes(b"NOT-LMA1" + b"\x00" * 32)
    r = subprocess.run(
        [str(lma_open), str(not_an_lma)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode != 0
    combined = r.stderr + r.stdout
    assert "LMA" in combined or "lma" in combined or "magic" in combined.lower()
