"""Compression backend dispatcher.

Routes encode/decode operations to the configured backend:
  - rust:   Rust `lml` binary (198 MB/s, default when available)
  - python: Python lamquant_codec (15 MB/s, always available)
  - custom: user-provided binary (must follow the backend contract)

Backend contract:
  <binary> encode <input> --output <output> [--recursive] [--verify]
           [--skip-existing] [--threads N] [--noise-bits N]
  <binary> decode <input> --output <output> [--recursive]
           [--skip-existing] [--threads N]

Output: LML v1 files with CRC-32 per window, SHA-256 in metadata.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def detect_backend(config) -> str:
    """Detect and return the active backend name: 'rust', 'python', or 'custom'."""
    return config.backend.resolve()


def get_backend_version(config) -> str:
    """Get the version string of the active backend."""
    backend = config.backend.resolve()
    if backend == "rust":
        binary = _resolve_binary(config)
        if binary:
            try:
                r = subprocess.run([binary, "--version"],
                                   capture_output=True, text=True, timeout=5)
                return r.stdout.strip() if r.returncode == 0 else "unknown"
            except Exception:
                return "unknown"
    elif backend == "custom":
        binary = config.backend.custom_binary
        try:
            r = subprocess.run([binary, "--version"],
                               capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    else:
        from lamquant_codec import __version__
        return f"lamquant-codec {__version__} (Python)"
    return "unknown"


def run_encode(config, input_path: str, output_path: str, *,
               recursive: bool = True, verify: bool = True,
               skip_existing: bool = True, workers: int = 0,
               noise_bits: int = 0) -> int:
    """Dispatch encode to the active backend. Returns exit code."""
    backend = config.backend.resolve()

    if backend in ("rust", "custom"):
        binary = _resolve_binary(config) if backend == "rust" else config.backend.custom_binary
        if not binary:
            print(f"error: Rust binary not found. Install with: cargo build --release",
                  file=sys.stderr)
            print(f"       Or set backend.mode = 'python' in lamquant.toml", file=sys.stderr)
            return 1

        cmd = [binary, "encode", input_path, "--output", output_path]
        if recursive:
            cmd.append("--recursive")
        if verify:
            cmd.append("--verify")
        if skip_existing:
            cmd.append("--skip-existing")
        if workers > 0:
            cmd.extend(["--threads", str(workers)])
        if noise_bits > 0:
            cmd.extend(["--noise-bits", str(noise_bits)])

        try:
            result = subprocess.run(cmd, timeout=7200)
            return result.returncode
        except subprocess.TimeoutExpired:
            print(f"error: backend timed out after 2 hours", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print(f"error: backend binary not found: {binary}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130

    else:
        # Python backend
        from lamquant_codec.cli.compress import main as compress_main
        args = [input_path, "-o", output_path]
        if workers > 0:
            args.extend(["-j", str(workers)])
        if noise_bits > 0:
            args.extend(["--noise-bits", str(noise_bits)])
        if skip_existing:
            args.append("--skip-existing")
        return compress_main(args) or 0


def run_decode(config, input_path: str, output_path: str, *,
               recursive: bool = True, skip_existing: bool = True,
               workers: int = 0) -> int:
    """Dispatch decode to the active backend. Returns exit code."""
    backend = config.backend.resolve()

    if backend in ("rust", "custom"):
        binary = _resolve_binary(config) if backend == "rust" else config.backend.custom_binary
        if not binary:
            print(f"error: backend binary not found", file=sys.stderr)
            return 1

        cmd = [binary, "decode", input_path, "--output", output_path]
        if recursive:
            cmd.append("--recursive")
        if skip_existing:
            cmd.append("--skip-existing")
        if workers > 0:
            cmd.extend(["--threads", str(workers)])

        try:
            result = subprocess.run(cmd, timeout=7200)
            return result.returncode
        except subprocess.TimeoutExpired:
            print(f"error: backend timed out after 2 hours", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print(f"error: backend binary not found: {binary}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130

    else:
        # Python backend
        from lamquant_codec.batch import decompress_batch
        report = decompress_batch(
            inputs=[input_path], output_dir=output_path,
            recursive=recursive, skip_existing=skip_existing,
            workers=workers or None, quiet=False,
        )
        return 1 if report.n_failed > 0 else 0


def _resolve_binary(config) -> Optional[str]:
    """Resolve the Rust binary path from config."""
    from lamquant_codec.cli.config import _find_rust_binary
    return _find_rust_binary(config.backend.rust_binary)
