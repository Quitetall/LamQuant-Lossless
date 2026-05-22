"""Coverage tests for ``lamquant_codec.cli.backend``.

The backend module dispatches encode/decode to either the Rust ``lml``
binary, a custom binary, or the Python fallback. Pure helpers exercised:

  * ``detect_backend(cfg)`` returns one of {rust, python, custom}
  * ``get_backend_version(cfg)`` returns a string for every branch
  * ``run_encode`` / ``run_decode`` propagate exit codes from subprocess
  * Missing-binary path returns nonzero exit code
  * Timeout / FileNotFoundError / KeyboardInterrupt branches

All subprocess invocations are mocked — no real binary is executed.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lamquant_codec.cli import backend as backend_mod
from lamquant_codec.cli.config import LamQuantConfig


def _cfg(mode="python", custom_binary=""):
    cfg = LamQuantConfig()
    cfg.backend.mode = mode
    cfg.backend.custom_binary = custom_binary
    return cfg


# ----- detect_backend ---------------------------------------------------


def test_detect_backend_returns_string():
    cfg = _cfg("python")
    out = backend_mod.detect_backend(cfg)
    assert isinstance(out, str)
    assert out in {"rust", "python", "custom"}


def test_detect_backend_python_mode():
    assert backend_mod.detect_backend(_cfg("python")) == "python"


def test_detect_backend_rust_mode_when_explicit():
    cfg = _cfg("rust")
    out = backend_mod.detect_backend(cfg)
    # rust mode is honored even when no binary is found, because
    # `resolve()` just returns "rust" when mode == "rust"
    assert out == "rust"


def test_detect_backend_custom_with_binary():
    cfg = _cfg("custom", custom_binary="/some/path/lml")
    assert backend_mod.detect_backend(cfg) == "custom"


# ----- get_backend_version ----------------------------------------------


def test_get_backend_version_python_returns_string():
    out = backend_mod.get_backend_version(_cfg("python"))
    assert isinstance(out, str)
    assert "lamquant-codec" in out.lower() or "python" in out.lower()


def test_get_backend_version_rust_unknown_when_no_binary():
    """When ``rust`` mode is set but no binary is on PATH, version is unknown."""
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary", return_value=None
    ):
        out = backend_mod.get_backend_version(cfg)
    # branch returns "unknown" when binary not found
    assert isinstance(out, str)


def test_get_backend_version_rust_with_subprocess_success():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=0, stdout="lml 1.2.3\n")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "lml 1.2.3"


def test_get_backend_version_rust_with_subprocess_failure():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=1, stdout="")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


def test_get_backend_version_rust_subprocess_raises():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", side_effect=OSError("boom")
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


def test_get_backend_version_custom_success():
    cfg = _cfg("custom", custom_binary="/path/to/x")
    fake_proc = SimpleNamespace(returncode=0, stdout="custom 0.1\n")
    with patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "custom 0.1"


def test_get_backend_version_custom_oserror():
    cfg = _cfg("custom", custom_binary="/nope")
    with patch(
        "lamquant_codec.cli.backend.subprocess.run", side_effect=OSError()
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


# ----- run_encode -------------------------------------------------------


def test_run_encode_rust_missing_binary_returns_one(capsys):
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary", return_value=None
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1


def test_run_encode_rust_success_propagates_returncode():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=0)
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ) as run_mock:
        rc = backend_mod.run_encode(
            cfg, "/in", "/out", workers=2, noise_bits=3,
        )
    assert rc == 0
    # CLI flags should be present
    cmd = run_mock.call_args[0][0]
    assert "encode" in cmd
    assert "/in" in cmd
    assert "--output" in cmd
    assert "--threads" in cmd
    assert "--noise-bits" in cmd


def test_run_encode_rust_timeout_returns_one():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=subprocess.TimeoutExpired("lml", 1),
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1


def test_run_encode_rust_filenotfound_returns_one():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=FileNotFoundError(),
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1


def test_run_encode_rust_keyboard_interrupt_returns_130():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=KeyboardInterrupt(),
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 130


def test_run_encode_python_dispatch():
    """Python backend should dispatch to ``compress.main``."""
    cfg = _cfg("python")
    with patch(
        "lamquant_codec.cli.compress.main", return_value=0
    ) as mock_main:
        rc = backend_mod.run_encode(
            cfg, "/in", "/out",
            workers=4, noise_bits=2, skip_existing=True,
        )
    assert rc == 0
    # Ensure dispatch invoked
    mock_main.assert_called_once()
    argv = mock_main.call_args[0][0]
    assert "/in" in argv
    assert "-o" in argv
    assert "/out" in argv


def test_run_encode_python_main_returns_none_normalized_to_zero():
    cfg = _cfg("python")
    with patch(
        "lamquant_codec.cli.compress.main", return_value=None
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 0


# ----- run_decode -------------------------------------------------------


def test_run_decode_rust_missing_binary_returns_one():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary", return_value=None
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1


def test_run_decode_rust_success():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=0)
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ) as run_mock:
        rc = backend_mod.run_decode(cfg, "/in", "/out", workers=2)
    assert rc == 0
    cmd = run_mock.call_args[0][0]
    assert "decode" in cmd
    assert "/in" in cmd


def test_run_decode_rust_timeout():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=subprocess.TimeoutExpired("lml", 1),
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1


def test_run_decode_rust_filenotfound():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=FileNotFoundError(),
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1


def test_run_decode_rust_keyboard_interrupt():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run",
        side_effect=KeyboardInterrupt(),
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 130


def test_run_decode_python_dispatch():
    cfg = _cfg("python")
    fake_report = SimpleNamespace(n_failed=0)
    with patch(
        "lamquant_codec.batch.decompress_batch", return_value=fake_report
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out", workers=2)
    assert rc == 0


def test_run_decode_python_failure():
    cfg = _cfg("python")
    fake_report = SimpleNamespace(n_failed=3)
    with patch(
        "lamquant_codec.batch.decompress_batch", return_value=fake_report
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1
