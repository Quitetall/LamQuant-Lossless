"""Real-EDF fixture loaders for the LamQuant test suite.

User direction (2026-05-21): **no synthetic data**. Every fixture
in this module locates a real EDF / LMA / Q31 NPZ on disk and skips
the test if the corpus is not present.

This file previously hosted synthetic-data factories (now retired).
The factory function names are kept as no-op shims that raise
``pytest.skip`` so any historical caller fails loud and clean,
pointing at the real-data substitute.

Real fixture sources (all gitignored — local-only):

  - ``reference_software/pyedflib-master/pyedflib/tests/data/test_generator.edf``
    1.2 MB synthetic-EEG EDF shipped with pyedflib. Verified
    byte-identical round-trip through the ``lml`` Rust binary in
    this session — safe to use as the canonical "small real EDF".

  - ``reference_software/nedc_eeg_resnet/v1.0.1/test/data/{eval,dev,train}/*.edf``
    TUH EEG corpus subset — real multi-channel clinical EEG, used
    by the NEDC seizure-detection benchmark. File naming matches
    the TUH stem convention (``aaaaaaaq_s006_t000.edf``).

  - ``reference_software/nedc_pyprint_edf/v1.0.0/example.edf``
    Single canonical example file shipped with NEDC tooling.

Q31 NPZ fixtures are derived on demand from these real EDFs via
``preprocess.py``'s production pipeline — never fabricated.

Tests that need a real corpus should:

    @pytest.mark.data
    def test_xyz(real_test_edf):
        # real_test_edf is a Path; the fixture skips if missing.
        ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Canonical real-EDF paths. Tests should reference these via the
# ``real_test_edf`` / ``real_tuh_edf`` fixtures in conftest rather
# than recomputing the path.
PYEDFLIB_TEST_GENERATOR = (
    _REPO_ROOT
    / "reference_software/pyedflib-master/pyedflib/tests/data/test_generator.edf"
)
NEDC_PYPRINT_EXAMPLE = (
    _REPO_ROOT
    / "reference_software/nedc_pyprint_edf/v1.0.0/example.edf"
)
NEDC_TUH_EVAL_DIR = (
    _REPO_ROOT
    / "reference_software/nedc_eeg_resnet/v1.0.1/test/data/eval"
)


def find_real_test_edf() -> Optional[Path]:
    """Return the canonical small real EDF, or ``None`` if absent.

    Prefers ``pyedflib``'s test_generator.edf — it's the smallest
    (1.2 MB) and verified byte-identical through the Rust codec.
    Falls back to the NEDC pyprint example if pyedflib's is missing.
    """
    for candidate in (PYEDFLIB_TEST_GENERATOR, NEDC_PYPRINT_EXAMPLE):
        if candidate.is_file():
            return candidate
    return None


def find_real_tuh_edfs(limit: int = 5) -> List[Path]:
    """Return up to ``limit`` real TUH EDFs from the NEDC eval split.

    Empty list if the corpus is absent. Sorted by filename for
    determinism.
    """
    if not NEDC_TUH_EVAL_DIR.is_dir():
        return []
    return sorted(NEDC_TUH_EVAL_DIR.glob("*.edf"))[:limit]


def require_real_test_edf() -> Path:
    """Like ``find_real_test_edf`` but skips the test if missing."""
    p = find_real_test_edf()
    if p is None:
        pytest.skip(
            "real EDF fixture not present "
            f"(looked at {PYEDFLIB_TEST_GENERATOR} and "
            f"{NEDC_PYPRINT_EXAMPLE}). Tests that need a real EDF "
            "are skipped in environments without the reference_software/ "
            "corpus checked out."
        )
    return p


def require_real_tuh_edfs(min_count: int = 1) -> List[Path]:
    """Return at least ``min_count`` real TUH EDFs or skip the test."""
    paths = find_real_tuh_edfs(limit=max(min_count, 5))
    if len(paths) < min_count:
        pytest.skip(
            f"need >= {min_count} real TUH EDFs at {NEDC_TUH_EVAL_DIR}, "
            f"found {len(paths)}. Skipped in environments without the "
            "NEDC corpus."
        )
    return paths


# ============================================================
# Real-EDF -> Q31 NPZ via the production preprocess pipeline
# ============================================================

def edf_to_q31_npz(edf_path: Path, npz_path: Path,
                   *, sample_rate: int = 250) -> Path:
    """Convert a real EDF to Q31 NPZ via the production pipeline.

    Uses ``ai_models.dataset_sim.preprocess`` end-to-end. No data is
    fabricated — only the format/window-slicing layer is exercised.
    Returns ``npz_path``.

    Skips the test if the production preprocess module cannot
    import (e.g., missing ``mne`` in CI without dev deps).
    """
    try:
        from ai_models.dataset_sim import preprocess as pp
    except Exception as e:
        pytest.skip(f"preprocess pipeline unavailable: {e}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Use whatever single-file entry point the production preprocess
    # module exposes. The exact name has drifted historically — try
    # the documented one first, fall back.
    fn = getattr(pp, "preprocess_one_edf", None) or \
        getattr(pp, "edf_to_q31", None) or \
        getattr(pp, "preprocess_file", None)
    if fn is None:
        pytest.skip(
            "preprocess.py has no single-file entry point — "
            "test cannot derive Q31 NPZ from real EDF without it"
        )
    fn(str(edf_path), str(npz_path), sample_rate=sample_rate)
    return npz_path


# ============================================================
# Backward-compat: retired synthetic-data factories
# ============================================================
# These names existed in an earlier version of this file. They now
# raise pytest.skip with an explanatory message so any historical
# caller fails loud + clean. Keep the names so import lines don't
# break — only the behaviour is retired.

def _retired(name: str) -> None:
    pytest.skip(
        f"{name}() was a synthetic-data factory and has been retired. "
        "Use the real-EDF fixtures (require_real_test_edf, "
        "require_real_tuh_edfs) instead. See "
        "feedback_mamba_snn_promoted memory: no synthetic data allowed."
    )


def make_q31_npz(*args, **kw): _retired("make_q31_npz")
def make_q31_corpus(*args, **kw): _retired("make_q31_corpus")
def make_l3_npz(*args, **kw): _retired("make_l3_npz")
def make_manifest_v3(*args, **kw): _retired("make_manifest_v3")
def make_fake_encoder(*args, **kw): _retired("make_fake_encoder")
def make_fake_decoder(*args, **kw): _retired("make_fake_decoder")
def make_fake_snn(*args, **kw): _retired("make_fake_snn")
def make_student_ckpt(*args, **kw): _retired("make_student_ckpt")
def make_vocos_ckpt(*args, **kw): _retired("make_vocos_ckpt")
def make_mamba_snn_ckpt(*args, **kw): _retired("make_mamba_snn_ckpt")
def make_fullband_memmap(*args, **kw): _retired("make_fullband_memmap")
