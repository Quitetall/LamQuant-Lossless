"""v1.2 T.4 — state-map gap-fill tests.

Each test plugs a hole identified in `docs/TEST_COVERAGE_STATE_MAP.md`
§8 ("TOP 10 gaps"). One file holds them all so the gap-fill batch is
easy to audit and easy to extend.

Covered here:

  * `lml extract --no-verify` fast path
  * `lmafs read` at offset > 0 (partial slice through FUSE)
  * `lmafs` refuses non-LMA inputs with a typed error
  * encrypt → sign → verify-signature → decrypt chain composition
  * strip-pii in-place preserves the signal SHA-256

Remaining state-map gaps (CNT/DICOM/Raw `--no-bundle` mirror tests,
mount-while-encoder-writes race) are tracked in
TEST_COVERAGE_STATE_MAP.md for future fill.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.helpers.edf_factory import create_edf

pytestmark = pytest.mark.l3


# ─── Helpers ────────────────────────────────────────────────────────


def _build_lma_with_siblings(tmp_path: Path, lml_cli_binary: Path) -> Path:
    """Encode a synth EDF plus two sibling annotation files into a
    per-recording LMA. Returns the archive path."""
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    (src / "rec.tse").write_bytes(b"0.0 30.0 bckg 1.0\n")
    (src / "rec_summary.txt").write_bytes(b"Patient: gap-fill smoke\n")
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
    return lma


def _lmafs_binary() -> Path | None:
    for c in (
        Path("/mnt/4tb/LamQuant/target/release/lmafs"),
        Path("/mnt/4tb/LamQuant/target/debug/lmafs"),
    ):
        if c.exists() and os.access(c, os.X_OK):
            return c
    return None


def _have_fuse() -> bool:
    if not shutil.which("fusermount") and not shutil.which("fusermount3"):
        return False
    return os.path.exists("/dev/fuse")


def _unmount(mountpoint: Path):
    for cmd in (
        ["fusermount", "-u", str(mountpoint)],
        ["fusermount3", "-u", str(mountpoint)],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, capture_output=True, timeout=10)
            return


# ─── extract --no-verify ────────────────────────────────────────────


def test_extract_no_verify_fast_path_recovers_entries(tmp_path, lml_cli_binary):
    """`lml extract --no-verify` skips the per-entry SHA pass; entries
    still recover byte-equal because the extract logic is the same.

    The flag is for trusted archives where extract speed matters; we
    pin the contract that it doesn't silently corrupt anything."""
    lma = _build_lma_with_siblings(tmp_path, lml_cli_binary)
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "extract",
            str(lma),
            "-o",
            str(extract_dir),
            "--no-verify",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]
    # Sibling bytes match originals.
    recovered_tse = next(extract_dir.rglob("rec.tse"), None)
    recovered_summary = next(extract_dir.rglob("rec_summary.txt"), None)
    assert recovered_tse is not None
    assert recovered_summary is not None
    assert recovered_tse.read_bytes() == b"0.0 30.0 bckg 1.0\n"
    assert recovered_summary.read_bytes() == b"Patient: gap-fill smoke\n"


# ─── lmafs partial-offset read ──────────────────────────────────────


def test_lmafs_read_at_offset_returns_correct_slice(tmp_path, lml_cli_binary):
    """FUSE `read` with offset > 0 returns the correct slice via
    standard POSIX `pread`. Pins the partial-read state we marked
    GAP in state-map §5."""
    if not _have_fuse():
        pytest.skip("FUSE not available")
    lmafs_bin = _lmafs_binary()
    if lmafs_bin is None:
        pytest.skip("lmafs binary not built")

    lma = _build_lma_with_siblings(tmp_path, lml_cli_binary)
    mount = tmp_path / "mnt"
    mount.mkdir()
    proc = subprocess.Popen(
        [str(lmafs_bin), "--foreground", str(lma), str(mount)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                names = [p.name for p in mount.iterdir()]
            except (FileNotFoundError, PermissionError):
                names = []
            if names:
                break
            time.sleep(0.1)
        assert names, "lmafs failed to surface entries"

        # rec.tse is 18 bytes: "0.0 30.0 bckg 1.0\n"
        # offset 5, size 8 → bytes[5:13] = "0.0 bckg"
        target = mount / "rec.tse"
        with target.open("rb") as f:
            f.seek(5)
            partial = f.read(8)
        assert partial == b"0.0 bckg", repr(partial)

        # offset >= size → empty slice
        with target.open("rb") as f:
            f.seek(1000)  # > 18
            past_eof = f.read(8)
        assert past_eof == b"", repr(past_eof)

        # size > remaining → clamp to remaining bytes
        with target.open("rb") as f:
            f.seek(10)
            tail = f.read(1000)
        # File is 18 bytes; from offset 10 -> 8 bytes left
        assert len(tail) == 8, len(tail)
    finally:
        _unmount(mount)
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_lmafs_refuses_non_lma_input(tmp_path, lml_cli_binary):
    """`lmafs` on a non-LMA file errors before mounting; doesn't leave
    a stale mountpoint."""
    if not _have_fuse():
        pytest.skip("FUSE not available")
    lmafs_bin = _lmafs_binary()
    if lmafs_bin is None:
        pytest.skip("lmafs binary not built")

    not_an_lma = tmp_path / "garbage.lma"
    not_an_lma.write_bytes(b"GARBAGE not LMA1 magic" + b"\x00" * 32)
    mount = tmp_path / "mnt"
    mount.mkdir()
    r = subprocess.run(
        [str(lmafs_bin), "--foreground", str(not_an_lma), str(mount)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode != 0
    combined = r.stderr + r.stdout
    assert (
        "lma" in combined.lower()
        or "magic" in combined.lower()
        or "invalid" in combined.lower()
        or "manifest" in combined.lower()
    ), f"expected non-LMA refusal detail; got:\n{combined[:400]}"


# ─── encrypt → sign → verify-signature → decrypt chain ─────────────


def test_encrypt_sign_verify_decrypt_chain(tmp_path, lml_cli_binary):
    """Compose AES-GCM encryption + HMAC signing. The chain is
    encrypt → sign → verify-signature → decrypt. Every step must
    leave the cryptographic state intact for the next.

    Tests state-map §7 cross-state-combos top entry."""
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"clinical summary line\n")

    # 32-byte hex key shared between AES-GCM and HMAC for both ops.
    env = {**os.environ, "LAMQUANT_KEY": "a" * 64}

    # 1. Encrypt
    ct = tmp_path / "ct.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert r.returncode == 0, r.stderr[:300]

    # 2. Sign the ciphertext
    r = subprocess.run(
        [str(lml_cli_binary), "sign", str(ct)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert r.returncode == 0, r.stderr[:300]
    sig = tmp_path / "ct.bin.hmac"
    assert sig.exists()

    # 3. Verify signature
    r = subprocess.run(
        [str(lml_cli_binary), "verify-signature", str(ct)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert r.returncode == 0, r.stderr[:300]
    assert "OK" in r.stdout or "verified" in r.stdout.lower()

    # 4. Decrypt back to plaintext
    out = tmp_path / "out.txt"
    r = subprocess.run(
        [str(lml_cli_binary), "decrypt", str(ct), "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert r.returncode == 0, r.stderr[:300]
    assert out.read_bytes() == plain.read_bytes()


# ─── strip-pii in-place preserves signal SHA-256 ────────────────────


def test_strip_pii_in_place_preserves_signal_sha(tmp_path, lml_cli_binary):
    """Stripping PII from a `.lml` must not touch the signal payload.

    Hash the signal-only bytes via `lml decode -o -` (raw int32 LE
    stdout) before and after `strip-pii --in-place`; they must match.
    """
    src = tmp_path / "src"
    src.mkdir()
    edf = src / "rec.edf"
    create_edf(str(edf), n_channels=4, n_records=2, sample_rate=250)
    out = tmp_path / "out"
    out.mkdir()
    lml = out / "rec.lml"
    # Use --no-bundle to get a bare .lml file (strip-pii operates on
    # the LML container, not the LMA archive).
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "encode",
            str(edf),
            "-o",
            str(lml),
            "--no-bundle",
            "--i-understand-data-loss",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[:400]

    def signal_sha(lml_path: Path) -> str:
        r = subprocess.run(
            [str(lml_cli_binary), "decode", str(lml_path), "-o", "-"],
            capture_output=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stderr.decode()[:200]
        return hashlib.sha256(r.stdout).hexdigest()

    sha_before = signal_sha(lml)
    r = subprocess.run(
        [str(lml_cli_binary), "strip-pii", str(lml), "--in-place"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr[:400]
    sha_after = signal_sha(lml)
    assert sha_before == sha_after, (
        f"strip-pii modified the signal! before={sha_before} after={sha_after}"
    )
