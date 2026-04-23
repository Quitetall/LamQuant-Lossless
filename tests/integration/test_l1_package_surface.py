"""tests/test_package_surface.py — pin the public API of `ai_models`.

Consumers must be able to import everything via `from ai_models import X`
without sys.path hacks or knowing where the symbol actually lives.
These tests fail loudly if a refactor accidentally hides a symbol or
moves a heavy import into the cheap-import path.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent


# ============================================================
# Public surface — every name a caller is allowed to import
# ============================================================
#
# Adding a symbol to ai_models's public API:
#   1. Add it to ai_models/__init__.py (eager re-export or _LAZY entry)
#   2. Add it to this list
# Removing a symbol:
#   1. Remove it from ai_models/__init__.py (and any consumers)
#   2. Remove it from this list
#
# CI gates the contract: this list is the API. Anything else is internal.

EAGER_SYMBOLS = [
    # Typed contracts
    'Split', 'Dataset', 'VALIDATION_ONLY_DATASETS',
    'parse_npz_filename', 'ParsedFilename',
    'FileEntry', 'DatasetEntry', 'DatasetManifest', 'MANIFEST_VERSION',
    'EEGWindow', 'L3Window', 'TrainingBatch',
    # Reporting
    'EpochReport', 'RunSummary', 'TrainingLogger',
    # Metrics (numpy + scipy, no torch)
    'EEG_BANDS',
    'prd_numpy', 'pearson_r_numpy',
    'per_band_prd', 'per_band_r',
    'lqs_compliance', 'lqs_pretty',
    # Experiment log (refactor #77)
    'ExperimentRecord', 'log_experiment', 'list_experiments',
    'best_by', 'find_by_run_id', 'compare', 'summary_table',
    'set_log_path', 'get_log_path',
]

LAZY_SYMBOLS = [
    # Datasets
    'PrecomputedL3Dataset', 'StreamingQ31Dataset', 'HybridQ31Dataset',
    'MemmapTeacherDataset',
    # Codec
    'JointCodec', 'build_default_joint', 'read_ckpt_provenance',
    # Checkpoint manager
    'CheckpointManager', 'GuardConfig', 'TrainingHaltException',
    'make_param_groups',
    # Differentiable metrics (need torch)
    'prd_torch', 'asymmetric_eeg_loss', 'band_aware_asymmetric_loss',
    # Training config
    'TrainingConfig', 'CONFIGS',
]


@pytest.mark.parametrize('name', EAGER_SYMBOLS)
def test_eager_symbol_importable(name):
    import ai_models
    assert hasattr(ai_models, name), (
        f'ai_models.{name} missing — eager re-export was dropped from __init__.py'
    )


@pytest.mark.parametrize('name', LAZY_SYMBOLS)
def test_lazy_symbol_importable(name):
    import ai_models
    obj = getattr(ai_models, name)   # triggers __getattr__ → import
    assert obj is not None


def test_eager_import_does_not_pull_torch():
    """`import ai_models` must NOT load torch — that's the whole point of
    the lazy split. CLI tools that only need types stay sub-100 ms.

    Run in a subprocess so the cold-import test doesn't pollute the
    in-process sys.modules table (other tests in the suite hold live
    references to ai_models submodules; deleting from sys.modules here
    would leave them with stale handles).
    """
    import subprocess
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; import ai_models; '
         'sys.exit(0 if "torch" not in sys.modules else 1)'],
        capture_output=True, cwd=str(_REPO),
    )
    assert result.returncode == 0, (
        f'ai_models eager import pulled torch — a heavy module leaked '
        f'into the cheap path. Move it to the _LAZY map. '
        f'stderr: {result.stderr.decode()[:300]}'
    )


def test_eager_import_is_fast():
    """Sanity check: eager `import ai_models` completes under 1 second
    in a fresh subprocess (we observe ~50 ms in practice)."""
    import subprocess
    result = subprocess.run(
        [sys.executable, '-c',
         'import time; t0=time.perf_counter(); import ai_models; '
         'print(f"{(time.perf_counter()-t0)*1000:.1f}")'],
        capture_output=True, cwd=str(_REPO), text=True,
    )
    assert result.returncode == 0, result.stderr
    elapsed_ms = float(result.stdout.strip())
    assert elapsed_ms < 1000, (
        f'import ai_models took {elapsed_ms:.0f} ms — a heavy import '
        f'probably leaked into the eager path'
    )


def test_dir_lists_lazy_symbols():
    """tab-completion on `ai_models.<TAB>` must show lazy names too."""
    import ai_models
    listed = set(dir(ai_models))
    for name in LAZY_SYMBOLS:
        assert name in listed, f'dir(ai_models) missing {name}'


def test_unknown_attribute_raises_attribute_error():
    """Asking for a non-existent attribute raises AttributeError, not
    ImportError, ModuleNotFoundError, or anything weirder."""
    import ai_models
    with pytest.raises(AttributeError, match='no attribute'):
        _ = ai_models.DefinitelyNotARealSymbol_29384


def test_no_consumer_uses_sys_path_insert_for_ai_models():
    """Soft check: count consumers that still use the legacy
    sys.path.insert pattern. This number should only go DOWN over time —
    hardened to a strict assertion once we migrate the remaining scripts.

    Counts only ACTUAL code lines, not docstring examples (the new
    `ai_models/__init__.py` shows the deprecated pattern in its docstring
    to teach the right alternative).
    """
    import ast
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent

    def has_real_sys_path_insert(path: Path) -> bool:
        """True iff the file has an actual sys.path.insert(...ai_models...)
        call (not a docstring example, not a comment)."""
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match `sys.path.insert(...)`
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == 'insert'
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == 'path'
                    and isinstance(f.value.value, ast.Name)
                    and f.value.value.id == 'sys'):
                # Check args contain 'ai_models'
                src = ast.unparse(node)
                if 'ai_models' in src:
                    return True
        return False

    offenders = []
    for f in repo.glob('ai_models/**/*.py'):
        if any(p in str(f) for p in ('__pycache__', '/legacy/')):
            continue
        if has_real_sys_path_insert(f):
            offenders.append(str(f.relative_to(repo)))

    # Threshold = current count. Tightening this number is the migration tax;
    # the migration plan is to ratchet this DOWN to 0 as we touch each script.
    THRESHOLD = 28
    assert len(offenders) <= THRESHOLD, (
        f'sys.path.insert(ai_models...) count grew above {THRESHOLD}: '
        f'{len(offenders)} offenders. Migrate to `from ai_models import X`. '
        f'Files: {offenders[:5]}{"..." if len(offenders) > 5 else ""}'
    )
