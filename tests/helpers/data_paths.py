"""Centralised resolver for test data, model checkpoints, and binaries.

Replaces scattered `Path(__file__).parent / '../weights/...'` chains and
ad-hoc skip logic. Every external dependency a test might need has a
single function here that returns the path or `None`.

Convention: returning `None` means "not available — caller should skip".
Callers should use the matching pytest fixture in `tests/conftest.py`
which translates `None` into `pytest.skip(...)` with a clear reason.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def q31_events_dir() -> Path | None:
    """Real EEG q31 dataset directory. None if not present."""
    p = REPO_ROOT / "ai_models" / "dataset_sim" / "q31_events"
    return p if p.exists() else None


def student_checkpoint() -> Path | None:
    """Trained student-subband checkpoint. None if not present."""
    p = REPO_ROOT / "weights" / "student_subband.ckpt"
    return p if p.exists() else None


def manifest_v3() -> Path | None:
    """Dataset manifest v3 JSON. None if not present."""
    p = REPO_ROOT / "ai_models" / "dataset_sim" / "manifest_v3.json"
    return p if p.exists() else None


def lml_cli_binary() -> Path | None:
    """The `lml` Rust CLI binary. Tries env var, then release, then debug."""
    env = os.environ.get("LML_BINARY")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for variant in ("release", "debug"):
        p = REPO_ROOT / "target" / variant / "lml"
        if p.exists():
            return p
    on_path = shutil.which("lml")
    return Path(on_path) if on_path else None


def canonical_split_config() -> Path | None:
    """Legacy data_pipeline_v2 split config. None if not present.

    Used only by tests/training/test_l2_data_split.py. Marked for migration
    to manifest_v3 — see the file's docstring.
    """
    p = REPO_ROOT / "legacy" / "data_pipeline_v2" / "canonical_split.json"
    return p if p.exists() else None


def validation_manifest() -> Path | None:
    """Dataset_sim validation manifest. None if not present."""
    p = REPO_ROOT / "ai_models" / "dataset_sim" / "validation_manifest.json"
    return p if p.exists() else None


def availability_report() -> dict[str, bool]:
    """Snapshot of every external dependency. Used by pytest_sessionstart."""
    return {
        "q31_events": q31_events_dir() is not None,
        "student_checkpoint": student_checkpoint() is not None,
        "manifest_v3": manifest_v3() is not None,
        "lml_cli_binary": lml_cli_binary() is not None,
        "canonical_split_config": canonical_split_config() is not None,
        "validation_manifest": validation_manifest() is not None,
    }
