"""`lml encrypt --password` Argon2id KDF (v1.2 P).

The v1 path reads a 64-char hex key from `LAMQUANT_KEY` env. v1.2
adds an Argon2id KDF so operators can use a normal password
without having to manage hex strings. Salt + Argon2 params live in
a `<output>.lmcrypt.header` sidecar so decrypt re-derives the same
key.

Cases:

  1. Round-trip with same password recovers the plaintext byte-equal.
  2. Wrong password produces auth-tag-mismatch error (non-zero exit).
  3. `--password` writes the sidecar header alongside the ciphertext.
  4. Missing sidecar on decrypt errors out explicitly.
  5. Empty password refused.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.l3


def _env_with(password: str) -> dict[str, str]:
    env = os.environ.copy()
    env["LAMQUANT_PASSWORD"] = password
    # Make sure no stale hex key shadows the password path.
    env.pop("LAMQUANT_KEY", None)
    return env


def test_password_roundtrip_byte_equal(tmp_path, lml_cli_binary):
    """Encrypt with password → decrypt with same password → plaintext."""
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"clinical EEG metadata: subject 042, 2026-05-19\n")

    ct = tmp_path / "ciphertext.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct), "--password"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("correct horse battery staple"),
    )
    assert r.returncode == 0, r.stderr[:400]
    assert ct.exists()

    restored = tmp_path / "restored.txt"
    r = subprocess.run(
        [str(lml_cli_binary), "decrypt", str(ct), "-o", str(restored), "--password"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("correct horse battery staple"),
    )
    assert r.returncode == 0, r.stderr[:400]
    assert restored.read_bytes() == plain.read_bytes()


def test_wrong_password_errors_auth_tag_mismatch(tmp_path, lml_cli_binary):
    """Decrypt with the wrong password surfaces the GCM auth-tag mismatch."""
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"secret")
    ct = tmp_path / "ct.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct), "--password"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("hunter2"),
    )
    assert r.returncode == 0, r.stderr[:400]

    out = tmp_path / "out.txt"
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "decrypt",
            str(ct),
            "-o",
            str(out),
            "--password",
            "--force",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("WRONG-password"),
    )
    assert r.returncode != 0, "wrong password must fail decrypt"
    combined = r.stderr + r.stdout
    assert "auth fail" in combined.lower() or "auth-tag" in combined.lower() or "aead" in combined.lower(), (
        f"expected auth-tag mismatch detail; got:\n{combined[:400]}"
    )


def test_password_writes_sidecar_with_salt_and_params(tmp_path, lml_cli_binary):
    """Sidecar header is created alongside the ciphertext."""
    plain = tmp_path / "p"
    plain.write_bytes(b"x")
    ct = tmp_path / "ct.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct), "--password"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("pw"),
    )
    assert r.returncode == 0, r.stderr[:400]

    sidecar = tmp_path / "ct.lmcrypt.header"
    assert sidecar.exists(), "missing .lmcrypt.header sidecar"
    bytes_ = sidecar.read_bytes()
    # Sidecar layout: 32 bytes total
    #   [0..4]   magic "LMHP"
    #   [4..8]   version u32 LE = 1
    #   [8..24]  16-byte salt
    #   [24..28] m_kib u32 LE  (default 65536)
    #   [28..30] t_cost u16 LE (default 3)
    #   [30]     p_cost u8     (default 1)
    #   [31]     reserved
    assert len(bytes_) == 32, len(bytes_)
    assert bytes_[:4] == b"LMHP"
    assert bytes_[4:8] == (1).to_bytes(4, "little")
    assert int.from_bytes(bytes_[24:28], "little") == 65536
    assert int.from_bytes(bytes_[28:30], "little") == 3
    assert bytes_[30] == 1


def test_decrypt_missing_sidecar_errors_clearly(tmp_path, lml_cli_binary):
    """Decrypt without the sidecar header → explicit error."""
    plain = tmp_path / "p"
    plain.write_bytes(b"x")
    ct = tmp_path / "ct.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct), "--password"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("pw"),
    )
    assert r.returncode == 0, r.stderr[:400]
    # Nuke the sidecar; decrypt now has no salt to re-derive the key.
    (tmp_path / "ct.lmcrypt.header").unlink()

    out = tmp_path / "out"
    r = subprocess.run(
        [
            str(lml_cli_binary),
            "decrypt",
            str(ct),
            "-o",
            str(out),
            "--password",
            "--force",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_env_with("pw"),
    )
    assert r.returncode != 0
    combined = r.stderr + r.stdout
    assert "sidecar" in combined.lower() or "lmcrypt.header" in combined.lower(), (
        f"expected sidecar-missing error; got:\n{combined[:400]}"
    )


def test_empty_password_refused(tmp_path, lml_cli_binary):
    """Empty password from env is refused (defends against typo loops)."""
    plain = tmp_path / "p"
    plain.write_bytes(b"x")
    ct = tmp_path / "ct.bin"
    r = subprocess.run(
        [str(lml_cli_binary), "encrypt", str(plain), "-o", str(ct), "--password"],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "LAMQUANT_PASSWORD": "", "LAMQUANT_KEY": ""},
    )
    # Either non-zero exit OR the binary fell back to interactive
    # prompting and hit no-tty. Both are correct refusals; what we
    # forbid is a silent encrypt with an empty key.
    if r.returncode == 0:
        # Read back the ciphertext bytes to ensure SOMETHING was
        # protected by a non-empty key.
        assert False, (
            "encrypt with empty LAMQUANT_PASSWORD should not succeed; "
            f"stdout: {r.stdout[:300]}"
        )
    combined = r.stderr + r.stdout
    assert "password" in combined.lower() or "empty" in combined.lower() or "tty" in combined.lower()
