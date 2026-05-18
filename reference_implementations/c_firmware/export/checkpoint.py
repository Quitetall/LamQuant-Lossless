"""Checkpoint loading + architecture variant detection.

Searches a list of glob patterns from the schema, picks the highest-grade
hit (gold > std > fast > untagged > legacy). Detects which architecture
class to instantiate via key-presence heuristics.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .schema import ArchSpec


@dataclass(frozen=True)
class LoadedCheckpoint:
    """One loaded checkpoint with its provenance."""

    path: Path
    sha256: str             # hex digest of the .ckpt file bytes
    state_dict: dict[str, Any]
    arch_name: str          # subband_v1 / subband_v2 / legacy_v7_0
    grade: str              # gold / std / fast / canonical / dev / legacy

    def short_sha(self) -> str:
        return self.sha256[:12]


# ────────────────────────────────────────────────────────────────────
# Variant detection
# ────────────────────────────────────────────────────────────────────


def detect_arch(state_dict: dict[str, Any], known_archs: dict[str, ArchSpec]) -> str:
    """Return the schema arch_name that matches this checkpoint's keys.

    Heuristics (mirrors export_firmware.py):
      - has 'focal2.dw.weight'  → subband_v2 (depthwise-separable focal)
      - has 'rotation_A' + 'premix.weight' → subband_v1
      - else → legacy_v7_0
    """
    has = lambda k: k in state_dict  # noqa: E731
    if has("focal2.dw.weight") and "subband_v2" in known_archs:
        return "subband_v2"
    if has("rotation_A") and has("premix.weight") and "subband_v1" in known_archs:
        return "subband_v1"
    if "legacy_v7_0" in known_archs:
        return "legacy_v7_0"
    raise ValueError(
        "Could not detect architecture from checkpoint keys; "
        f"sample keys: {list(state_dict.keys())[:5]}"
    )


# ────────────────────────────────────────────────────────────────────
# File discovery
# ────────────────────────────────────────────────────────────────────


def _grade_of(path: Path) -> str:
    name = path.stem.lower()
    for tag in ("gold", "std", "fast"):
        if tag in name:
            return tag
    if "subband" in name and not any(t in name for t in ("gold", "std", "fast")):
        return "canonical"
    if "ai_models" in str(path).lower():
        return "dev"
    if "hardened" in name:
        return "legacy"
    return "untagged"


def find_checkpoint(
    repo_root: Path,
    arch_globs: list[str],
    explicit: Path | None = None,
) -> Path:
    """Find a checkpoint file. If `explicit` is given, use it and skip search."""
    if explicit is not None:
        p = Path(explicit)
        if not p.is_absolute():
            p = repo_root / p
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    candidates: list[Path] = []
    for glob_pattern in arch_globs:
        candidates.extend(sorted(repo_root.glob(glob_pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matched any of: {arch_globs} (root={repo_root})"
        )

    # Prefer gold > std > fast > others.
    grade_order = {"gold": 0, "std": 1, "fast": 2, "canonical": 3, "untagged": 4, "dev": 5, "legacy": 6}
    candidates.sort(key=lambda p: grade_order.get(_grade_of(p), 99))
    return candidates[0]


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of the file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────────────
# Top-level loader
# ────────────────────────────────────────────────────────────────────


def load_checkpoint(
    repo_root: Path,
    arch_name: str | None,
    known_archs: dict[str, ArchSpec],
    explicit_path: Path | None = None,
    device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """High-level: pick checkpoint, load, detect arch, return typed result.

    If `arch_name` is None, auto-detects from the checkpoint contents.
    Otherwise uses the named arch's `checkpoint_globs` for file search.
    """
    if explicit_path is not None and arch_name is None:
        # Need to load first, then auto-detect.
        path = Path(explicit_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    else:
        if arch_name is None:
            raise ValueError(
                "Either --checkpoint or --arch must be given to load_checkpoint."
            )
        arch_spec = known_archs[arch_name]
        path = find_checkpoint(repo_root, arch_spec.checkpoint_globs, explicit_path)

    state_dict = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if arch_name is None:
        arch_name = detect_arch(state_dict, known_archs)

    return LoadedCheckpoint(
        path=path,
        sha256=sha256_of(path),
        state_dict=state_dict,
        arch_name=arch_name,
        grade=_grade_of(path),
    )
