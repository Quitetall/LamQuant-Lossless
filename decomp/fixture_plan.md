# Test Fixture Split Plan for LamQuant Monorepo Decomposition

**Prepared**: 2026-05-28  
**Scope**: 8-way Unix-philosophy split into Lossless, Neural (private), Eagle (public), and Common  
**Methodology**: Static analysis of `conftest.py`, `tests/fixtures/`, `tests/helpers/`, and per-subdir conftests

---

## Executive Summary

The test infrastructure is **moderately coupled**. Three critical fixture/helper sharing points exist:

1. **`tests/helpers/signals.py`** — used by LOSSLESS (codec tests), NEURAL (SNN), and EAGLE (benchmarks)
2. **Real-EDF fixtures** (`require_real_test_edf`, `require_real_tuh_edfs`, `edf_to_q31_npz`) — used across all repos
3. **sys.path manipulation** in root conftest.py — adds both LOSSLESS (`lamquant_codec`) and NEURAL (`ai_models`) paths unconditionally

**Recommendation**: Move `tests/helpers/` and core fixture loaders to a shared Python package `lamquant_common_testdata` vendored in Lossless repo, installed as a dependency by Neural and Eagle. This avoids duplication drift and keeps public (Eagle) tests runnable standalone.

---

## Root conftest.py Inventory

**File**: `/mnt/4tb/LamQuant/tests/conftest.py` (244 lines)

### 1. sys.path Manipulation (lines 22–35) — CROSS-REPO ISSUE

```python
_REPO_ROOT = Path(__file__).parent.parent
for _rel in (
    "ai_models/student",       # NEURAL
    "ai_models/oracle",        # NEURAL
    "ai_models/snn",           # NEURAL
    "ai_models/dataset_sim",   # NEURAL
    "ai_models/validation",    # NEURAL (+ Eagle uses it)
    "ai_models",               # NEURAL
    "firmware",                # LOSSLESS
    str(_REPO_ROOT),           # REPO_ROOT (needed by imports)
):
    _p = str((_REPO_ROOT / _rel).resolve()) if _rel != str(_REPO_ROOT) else str(_REPO_ROOT)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
```

**Problem**: After split, each repo will register **only its own paths**:
- **Lossless repo** needs: `firmware/`, `lamquant_codec/` (as installed package)
- **Neural repo** needs: `ai_models/` (entire subtree), `lamquant_codec/` (imported)
- **Eagle repo** needs: both, OR neither (if installed as packages)

**Decision**: Each repo's conftest will have its own `sys.path` setup. Lossless will add `firmware/`. Neural will add `ai_models/*`. Eagle will assume both are installed.

**Migration Impact**: MEDIUM — each conftest needs per-repo customization.

---

### 2. Session-Scoped Resource Fixtures (lines 38–143) — SHARED SKIPPABLE FIXTURES

| Fixture | Scope | Returns | Consumers | Destination | Notes |
|---------|-------|---------|-----------|-------------|-------|
| `test_data_dir` | session | tempdir Path | root conftest only | **Common** | Generic temp dir; keep in Common |
| `real_test_edf` | session | Path \| skip | codec, dataset_sim, firmware | **Common** | Wraps `require_real_test_edf()` from fixtures; mark @data |
| `real_tuh_edfs` | session | [Path] \| skip | training, experiments | **Common** | Wraps `require_real_tuh_edfs()` from fixtures; mark @data |
| `real_q31_from_edf` | function | Path (npz) | none observed | **Common** | Unused in current test suite; keep for future |
| `root_dir` | session | Path | training/conftest, student | **Common** | Returns repo root; needed by per-dir conftests |
| `q31_events_dir` | session | Path \| skip | none observed | **Common** | Marked @data; used by training setup |
| `manifest_v3_path` | session | Path \| skip | none observed | **Common** | Marked @data; used by dataset_sim |
| `student_checkpoint_path` | session | Path \| skip | none observed | **Common** | Marked @checkpoint; training-specific |
| `lml_cli_binary` | session | Path \| skip | integration, benchmarks | **Common** | Marked @rust; critical for lossless CLI tests |
| `rust_wheel` | session | module \| skip | codec cross_lang, container | **Common** | Marked @rust; imports `lamquant_core` PyO3 |

**All are skip-on-missing**, so each repo can run independently. None depend on repo-internal modules (they use path-based lookups via `tests.helpers.data_paths`).

**Destination**: Move to **`lamquant_common_testdata` Python package** in Lossless repo. Install via `pip install -e ../lamquant-common/` in Neural and Eagle.

---

### 3. Cross-Cutting Test Data Fixtures (lines 157–197) — SHARED BY MULTIPLE REPOS

| Fixture | Scope | Returns | Definition | Consumers | Destination |
|---------|-------|---------|-----------|-----------|-------------|
| `sample_seizure_mask` | function | np.ndarray [2500] | `np.random.seed(42); binomial(...)` | root conftest only (via `sample_npz_file`) | **Common** |
| `ternary_model` | function | TernaryMobileNetV5 | Instantiates encoder; calls `torch.manual_seed(0)` | firmware `test_l3_export.py` (LOSSLESS), training `test_lsq_quantization.py` (NEURAL) | **Common** |
| `tmp_header` | function | Path | `tmp_path / "focal_net_weights.h"` | firmware `test_l3_export.py` (LOSSLESS) only | **Common** |
| `sample_npz_file` | session | Path (npz) | Depends: `test_data_dir`, `sample_eeg_q31`, `sample_seizure_mask` | none observed in grep; defined but unused | **Common** |

**Issues**:
- `ternary_model` is used by both LOSSLESS (firmware) and NEURAL (training) — must be in Common
- `sample_eeg_q31` is defined in `tests/codec/conftest.py` but needed by root fixtures — circular dependency

**Resolution**: Move to Common. `sample_eeg_q31` stays in codec/conftest because codec is the only consumer (other than `sample_npz_file`, which is currently unused).

---

### 4. Pytest Hooks (lines 200–243)

#### `pytest_sessionstart` (lines 216–243) — DATA AVAILABILITY REPORTING

Prints a banner of resource checks at session start. Each repo will have a modified version checking only its own resources:

- **Lossless**: q31_events, Rust CLI binary, manifest_v3 (shared)
- **Neural**: q31_events, student checkpoint, manifest_v3 (shared)
- **Eagle**: None (benchmarks use synthetic or pre-loaded data)

---

### 5. Markers (lines 210–213) — DECLARED IN pyproject.toml

Per line 210–213 comment: markers are declared in `pyproject.toml` under `[tool.pytest.ini_options].markers`, not inline. This is the **single source of truth**.

From `pyproject.toml`:
```python
markers = [
    "l1: KAT — fixed input, fixed output (codec/ only)",
    "l2: property/invariant",
    "l3: metamorphic / regression",
    "l4: fuzz",
    "l5: cross-implementation parity (Py↔Rust↔C)",
    "l7: adversarial / boundary",
    "slow: model loading or heavy computation (>5s)",
    "perf: performance regression sentinel (nightly only)",
    "data: requires real EEG (q31_events / manifest_v3)",
    "checkpoint: requires trained weights on disk",
    "rust: requires lamquant_core PyO3 wheel",
    "c_host: compiles and runs C code on the host",
    "doctest: doctest collection target",
    "cross_lang: Python↔Rust drift sentinel",
]
```

**Post-split**: Each repo's `pyproject.toml` will register **only the markers it uses**:
- **Lossless**: `l1`, `l2`, `l3`, `l4`, `l5`, `l7`, `slow`, `data`, `rust`, `c_host`, `cross_lang`
- **Neural**: `l2`, `l3`, `slow`, `data`, `checkpoint`
- **Eagle**: `slow`, `perf`, `data`
- **Common**: (no markers registered; fixtures don't require them)

---

## Per-Subdir conftest.py Inventory

### `tests/codec/conftest.py` (26 lines)

**Fixtures**:
- `sample_eeg_q31` [function scope]: Q31-format EEG [21, 2500]
- `sample_eeg_float` [function scope]: float32 EEG [21, 2500]

**Consumers**: codec tests only (`test_batch_coverage.py`, `test_l2_preprocessing.py`, etc.) + root conftest (via `sample_npz_file`)

**Destination**: **Stays in Lossless repo** (`tests/codec/conftest.py`). No changes needed.

**Note**: Neither fixture imports from `ai_models`, so no cross-repo coupling.

---

### `tests/training/conftest.py` (57 lines)

**Fixtures**:
- `random_eeg_batch` [function]: torch tensor [B=2, C=21, T=2500]
- `canonical_split_config` [function]: legacy JSON (skips if absent); depends on `root_dir`
- `validation_manifest` [function]: v2 manifest JSON; depends on `root_dir`

**Consumers**: training tests only + must depend on `root_dir` from root conftest

**Destination**: **Moves to Neural repo** (`tests/training/conftest.py`). Must import `root_dir` from Common conftest.

**Migration**: Lossless doesn't have training tests, so no conflict.

---

## tests/fixtures/ — File-by-File Split

### `tests/fixtures/__init__.py` (24 lines)

**Exports**:
```python
from .synthetic import (
    PYEDFLIB_TEST_GENERATOR,   # Path constant
    NEDC_PYPRINT_EXAMPLE,      # Path constant
    NEDC_TUH_EVAL_DIR,         # Path constant
    edf_to_q31_npz,            # Function
    find_real_test_edf,        # Function
    find_real_tuh_edfs,        # Function
    require_real_test_edf,     # Function
    require_real_tuh_edfs,     # Function
)
```

**Usage**: Imported by root conftest.py (lines 54, 61, 71)

**Destination**: **Move to Common package** (`lamquant_common_testdata/fixtures/__init__.py`)

---

### `tests/fixtures/synthetic.py` (179 lines)

**What it provides**:

| Item | Type | Consumers | Repo Bucket |
|------|------|-----------|-------------|
| `PYEDFLIB_TEST_GENERATOR` (const) | Path | root conftest → all repos | **MULTI-REPO** |
| `NEDC_PYPRINT_EXAMPLE` (const) | Path | root conftest → all repos | **MULTI-REPO** |
| `NEDC_TUH_EVAL_DIR` (const) | Path | root conftest, dataset_sim | **LOSSLESS + NEURAL** |
| `find_real_test_edf()` | Function | root conftest → all repos | **MULTI-REPO** |
| `find_real_tuh_edfs(limit)` | Function | root conftest, dataset_sim | **LOSSLESS + NEURAL** |
| `require_real_test_edf()` | Function | root conftest → all repos (via fixture) | **MULTI-REPO** |
| `require_real_tuh_edfs(min_count)` | Function | root conftest → all repos (via fixture) | **MULTI-REPO** |
| `edf_to_q31_npz(edf_path, npz_path, sample_rate)` | Function | root conftest → all repos (via fixture) | **MULTI-REPO** |
| Retired synthetic factories (lines 159–177) | No-op shims | none | **DEPRECATED** |

**Critical coupling**: `edf_to_q31_npz()` imports from `ai_models.dataset_sim.preprocess` (line 132). After split:
- Lossless repo won't have `ai_models` installed by default
- This function will `pytest.skip()` when the import fails (by design, line 133–134)
- **This is acceptable** — the fixture already gracefully skips when preprocess is unavailable

**Destination**: **Move to Common package** (`lamquant_common_testdata/fixtures/synthetic.py`). All three repos will `pip install` the Common package.

---

## tests/helpers/ — File-by-File Split

### `tests/helpers/__init__.py` (11 lines)

**Content**: Module docstring only; no actual code.

**Destination**: Move to **Common package** (`lamquant_common_testdata/helpers/__init__.py`)

---

### `tests/helpers/data_paths.py` (79 lines)

**What it provides**:

| Function | Returns | Used By | Repo Bucket |
|----------|---------|---------|-------------|
| `q31_events_dir()` | Path \| None | root fixture → all repos (via q31_events_dir fixture) | **MULTI-REPO** |
| `student_checkpoint()` | Path \| None | root fixture (student_checkpoint_path) | **NEURAL** |
| `manifest_v3()` | Path \| None | root fixture (manifest_v3_path) | **NEURAL + EAGLE** |
| `lml_cli_binary()` | Path \| None | root fixture (lml_cli_binary) | **LOSSLESS** |
| `canonical_split_config()` | Path \| None | training/conftest | **NEURAL** |
| `validation_manifest()` | Path \| None | training/conftest | **NEURAL** |
| `availability_report()` | dict | pytest_sessionstart | **MULTI-REPO** |

**Critical detail**: `lml_cli_binary()` checks `LML_BINARY` env var and searches `target/{release,debug}/lml`. After split:
- Lossless repo will have `target/` directory (Rust build artifacts)
- Neural repo won't — it imports Lossless as a package
- **This is acceptable**: The function returns `None` if not found; fixture skips gracefully

**Cross-repo imports**: None. Uses only pathlib and os.

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/data_paths.py`)

---

### `tests/helpers/asserts.py` (162 lines)

**What it provides**:

| Function | Purpose | Consumers | Repo Bucket |
|----------|---------|-----------|-------------|
| `assert_raises_lml(expected_class, fn, *args, **kwargs)` | Type-safe exception assertion | codec tests: `test_errors.py` (LOSSLESS) | **LOSSLESS** |
| `assert_bytes_equal(actual, expected, *, context)` | Byte-exact comparison w/ diagnostic | codec cross_lang tests (LOSSLESS) | **LOSSLESS** |
| `assert_array_equal_strict(actual, expected, ...)` | Strict dtype+shape+value check | codec cross_lang tests (LOSSLESS) | **LOSSLESS** |

**Cross-repo imports**: `from lamquant_codec.errors import ...` (lines 32–36)

**Problem**: After split, Neural repo won't have `lamquant_codec` code imported locally. The asserts module imports from it. However, this is used **only by Lossless tests**, so it can stay in Lossless.

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/asserts.py`), but make the import of `lamquant_codec.errors` conditional (try/except with graceful skip if import fails, since Neural won't use it).

---

### `tests/helpers/edf_factory.py` (127 lines)

**What it provides**:

| Function | Purpose | Consumers | Repo Bucket |
|----------|---------|-----------|-------------|
| `create_edf(path, n_channels=21, ...)` | Synthetic EDF/BDF file generator | integration tests (ALL 9 test files in `tests/integration/`) | **LOSSLESS + NEURAL + EAGLE** |

**Consumer list**:
- `test_cli_transform.py` (LOSSLESS — codec CLI)
- `test_cli_inspect.py` (LOSSLESS)
- `test_lma_browse.py` (LOSSLESS)
- `test_lmafs_fuse.py` (LOSSLESS)
- `test_lma_open_doubleclick.py` (LOSSLESS)
- `test_magic_byte_dispatch.py` (LOSSLESS)
- `test_data_loss_footguns.py` (LOSSLESS)
- `test_sidecar_preservation.py` (LOSSLESS)
- `test_verify_explain.py` (LOSSLESS)
- `test_include_exclude_globs.py` (LOSSLESS)
- `test_v12_gap_fills.py` (LOSSLESS)
- **Total**: 11 files, all in LOSSLESS integration tests

**Cross-repo imports**: Only `os`, `struct`, `numpy` — no internal modules.

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/edf_factory.py`)

---

### `tests/helpers/roundtrip.py` (83 lines)

**What it provides**:

| Function | Purpose | Consumers | Repo Bucket |
|----------|---------|-----------|-------------|
| `assert_lml_roundtrip(signal, *, n_levels=3, label="")` | Compress→decompress→compare | (none observed in grep) | **UNUSED** |
| `assert_lml_compression_valid(signal, *, n_levels=3)` | Verify LML1 structure | (none observed in grep) | **UNUSED** |

**Cross-repo imports** (lines 16–17):
```python
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))
from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
```

**Problem**: Hardcodes path to `reference_implementations/python_codec/lamquant_codec`. After split:
- Lossless repo will have this directory (or install lamquant_codec as a package)
- Neural repo won't

**Current usage**: Not used by any test in the current suite (grep found none). **Safe to deprecate or move to Lossless-only**.

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/roundtrip.py`), but fix the import:
```python
# Instead of hardcoding path, just import:
try:
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
except ImportError:
    _compress_bytes = None
    _decompress_bytes = None
```

Then in the functions, check if they're None and skip. This way Neural repo won't break if it imports the module.

---

### `tests/helpers/rust_bindings.py` (53 lines)

**What it provides**:

| Item | Purpose | Consumers | Repo Bucket |
|------|---------|-----------|-------------|
| `HAS_RUST` (bool) | Global flag: is lamquant_core installed? | root fixture (rust_wheel), codec tests | **LOSSLESS** |
| `rust_compress`, `rust_decompress`, etc. (functions) | Rust codec wrappers | codec cross_lang tests | **LOSSLESS** |
| `requires_rust` (pytest mark) | Decorator/mark for skip-when-missing | codec tests | **LOSSLESS** |
| `to_rust_signal(sig)` | Convert numpy to Vec<Vec<i64>> | codec cross_lang tests | **LOSSLESS** |

**Cross-repo imports** (lines 17–38):
```python
try:
    import lamquant_core as _rust
    HAS_RUST = True
    ...
except ImportError:
    HAS_RUST = False
```

**Used by**: codec cross_lang tests (`test_container_cross_lang.py`, `test_l5_cross_lang.py`), all LOSSLESS

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/rust_bindings.py`). The try/except handles missing lamquant_core gracefully.

---

### `tests/helpers/signals.py` (95 lines)

**What it provides**:

| Function | Purpose | Consumers | Repo Bucket |
|----------|---------|-----------|-------------|
| `synth_signal(n_ch, T, *, seed, amp, dtype)` | Deterministic integer signal [n_ch, T] | codec tests: `test_l5_cross_lang.py`, cross_lang tests | **LOSSLESS** |
| `make_synthetic_eeg(n_channels, n_samples, seed)` | Synthetic EEG-like signal | integration test: `test_perf_sentinels.py` | **LOSSLESS** |
| `adversarial_signals(lengths)` | Generator: canonical adversarial cases | (none observed in grep) | **UNUSED** |

**Current consumers** (from grep):
- `codec/test_l5_cross_lang.py` (LOSSLESS)
- `codec/cross_lang/test_container_cross_lang.py` (LOSSLESS)
- `integration/test_perf_sentinels.py` (LOSSLESS)

**Cross-repo imports**: Only numpy. No internal dependencies.

**Destination**: **Move to Common package** (`lamquant_common_testdata/helpers/signals.py`). Only Lossless uses it, but it has no internal deps, so no harm in making it public.

---

## Cross-Repo Coupling — Resolution Required

### 1. **Real-EDF Fixtures Are Used By All Repos**

**Coupling**: `require_real_test_edf()`, `require_real_tuh_edfs()`, `edf_to_q31_npz()` are defined in `tests/fixtures/synthetic.py` and called by root conftest.py fixtures. The root fixtures are then used by:
- Lossless: codec, container, c_host, firmware, edf_reader tests
- Neural: training, dataset_sim, snn, student tests
- Eagle: validation, audits, benchmarks tests

**Resolution**: Move the fixture loaders to Common package (`lamquant_common_testdata`). Each repo's root conftest will import them and re-export:

```python
# In Lossless repo's tests/conftest.py (post-split)
from lamquant_common_testdata.fixtures import require_real_test_edf, require_real_tuh_edfs, edf_to_q31_npz
```

All three repos will then `pip install -e ../lamquant-common/` to get the package.

---

### 2. **sys.path Manipulation — Each Repo Must Customize**

**Coupling**: Current root conftest adds BOTH `ai_models/*` (NEURAL) and `firmware/` (LOSSLESS) to sys.path.

**Resolution**: Each repo's post-split conftest will add **only its own paths**:

#### Lossless repo — `tests/conftest.py`
```python
_REPO_ROOT = Path(__file__).parent.parent
_p = str((_REPO_ROOT / "firmware").resolve())
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)
```

#### Neural repo — `tests/conftest.py`
```python
_REPO_ROOT = Path(__file__).parent.parent
for _rel in ("ai_models/student", "ai_models/oracle", "ai_models/snn", "ai_models/dataset_sim", "ai_models/validation", "ai_models"):
    _p = str((_REPO_ROOT / _rel).resolve())
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
```

#### Eagle repo — `tests/conftest.py`
```python
# No custom sys.path needed; both Lossless and Neural are installed as packages
```

---

### 3. **ternary_model Used By Both Lossless (firmware) and Neural (training)**

**Coupling**: `ternary_model` fixture (root conftest, line 165) instantiates `TernaryMobileNetV5` from `lamquant_codec.models.encoder`. Used by:
- Lossless: `tests/firmware/test_l3_export.py`
- Neural: `tests/training/test_lsq_quantization.py`

**Resolution**: Keep in Common conftest, since both repos will have `lamquant_codec` installed (as a package in Lossless's repo, vendored/installed in Neural).

---

### 4. **benchmark_c_parity.py (EAGLE) Imports from lamquant_codec.models.encoder (LOSSLESS)**

**Coupling**: `/mnt/4tb/LamQuant/tests/benchmarks/benchmark_c_parity.py` (lines 35–36):
```python
from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from lamquant_codec.models.blocks import TernaryConv1d
```

This is EAGLE code, but it imports LOSSLESS module code. After split, Eagle repo will have Lossless installed as a dependency.

**Decision**: **LEAVE UNCHANGED**. Eagle's test suite will install Lossless as a public package; the imports will resolve correctly. No migration required.

---

## Per-Repo conftest.py Drafts (Post-Split)

### Lossless Repo conftest.py (post-split)

**File**: `lamquant-lossless/tests/conftest.py`

```python
"""
Pytest configuration for LamQuant Lossless (codec, container, firmware, c_host, edf_reader).

Fixtures defined here:
  - Session-scoped resource checks (real_test_edf, real_tuh_edfs, lml_cli_binary, rust_wheel)
  - Cross-cutting test data (ternary_model, tmp_header, sample_seizure_mask, sample_npz_file)
  
These fixtures are defined in the `lamquant_common_testdata` package, which is
installed from the sibling Lossless repo. This allows Neural and Eagle tests
to run standalone without needing private sibling repos checked out.

Additional fixtures from tests/codec/conftest.py:
  - sample_eeg_q31, sample_eeg_float (codec-specific)
"""

import os
import sys
import warnings
import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Expose firmware module for tests that import by bare name.
_REPO_ROOT = Path(__file__).parent.parent
_p = str((_REPO_ROOT / "firmware").resolve())
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def root_dir():
    """Return root project directory."""
    return Path(__file__).parent.parent


# Import shared fixtures from common package
from lamquant_common_testdata.conftest import (
    real_test_edf,
    real_tuh_edfs,
    real_q31_from_edf,
    q31_events_dir,
    manifest_v3_path,
    student_checkpoint_path,
    lml_cli_binary,
    rust_wheel,
    sample_seizure_mask,
    ternary_model,
    tmp_header,
    sample_npz_file,
    pytest_sessionstart,
)


def pytest_sessionstart(session):
    """Report data availability at session start (overridden from Common)."""
    repo = Path(__file__).parent.parent
    checks = {
        "q31_events (real EEG)": repo / "ai_models" / "dataset_sim" / "q31_events",  # Not in Lossless, but may be cached
        "Rust CLI binary": repo / "target" / "release" / "lml",
    }
    print("\n" + "=" * 60)
    print("LOSSLESS TEST DATA AVAILABILITY")
    print("=" * 60)
    all_present = True
    for name, path in checks.items():
        present = path.exists()
        if not present:
            all_present = False
        status = "PRESENT" if present else "MISSING — tests will use fallbacks"
        print(f"  {name}: {status}")
    if not all_present:
        print("  NOTE: Some tests will run with synthetic data or skip.")
    print("=" * 60 + "\n")
```

**Diffs from current root conftest**:
- Line 22–26: Remove `ai_models/*` paths; keep only `firmware/`
- Line 46–143: Replace inline fixture definitions with imports from `lamquant_common_testdata.conftest`
- Line 216–243: Replace pytest_sessionstart with Lossless-specific version (checks only `target/release/lml`)
- Markers: Inherit from `pyproject.toml` (no change)

---

### Neural Repo conftest.py (post-split)

**File**: `lamquant-neural/tests/conftest.py`

```python
"""
Pytest configuration for LamQuant Neural (training, snn, student, dataset_sim, architectures, etc.).

Fixtures defined here:
  - Session-scoped resource checks (real_test_edf, real_tuh_edfs, student_checkpoint_path)
  - Cross-cutting test data (ternary_model)
  
These are defined in the `lamquant_common_testdata` package, installed from
the public Lossless repo. Neural repo can run its full test suite standalone.
"""

import os
import sys
import warnings
import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Expose ai_models module for tests that import by bare name.
_REPO_ROOT = Path(__file__).parent.parent
for _rel in (
    "ai_models/student",
    "ai_models/oracle",
    "ai_models/snn",
    "ai_models/dataset_sim",
    "ai_models/validation",
    "ai_models",
):
    _p = str((_REPO_ROOT / _rel).resolve())
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def root_dir():
    """Return root project directory."""
    return Path(__file__).parent.parent


# Import shared fixtures from common package
from lamquant_common_testdata.conftest import (
    real_test_edf,
    real_tuh_edfs,
    real_q31_from_edf,
    q31_events_dir,
    manifest_v3_path,
    student_checkpoint_path,
    ternary_model,
    sample_seizure_mask,
    sample_npz_file,
)


def pytest_sessionstart(session):
    """Report data availability at session start (Neural-specific)."""
    repo = Path(__file__).parent.parent
    checks = {
        "q31_events (real EEG)": repo / "ai_models" / "dataset_sim" / "q31_events",
        "student checkpoint": repo / "weights" / "student_subband.ckpt",
        "manifest_v3": repo / "ai_models" / "dataset_sim" / "manifest_v3.json",
    }
    print("\n" + "=" * 60)
    print("NEURAL TEST DATA AVAILABILITY")
    print("=" * 60)
    all_present = True
    for name, path in checks.items():
        present = path.exists()
        if not present:
            all_present = False
        status = "PRESENT" if present else "MISSING — tests will use fallbacks"
        print(f"  {name}: {status}")
    if not all_present:
        print("  NOTE: Some tests will run with synthetic data or skip.")
    print("=" * 60 + "\n")
```

**Diffs from current root conftest**:
- Line 24–32: Keep only `ai_models/*` paths; remove `firmware/`
- Line 46–143: Replace inline fixture definitions with imports
- Line 216–243: Replace pytest_sessionstart (check student checkpoint, manifest_v3, q31_events)

**Additional changes**:
- `tests/training/conftest.py` stays as-is (it already imports `root_dir` from root conftest, which will be re-exported)

---

### Eagle Repo conftest.py (post-split)

**File**: `lamquant-eagle/tests/conftest.py`

```python
"""
Pytest configuration for LamQuant Eagle (public benchmark suite).

All fixtures are imported from the `lamquant_common_testdata` package,
which is installed from the public Lossless repo.

Eagle repo is a pure consumer; no sys.path manipulation needed.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def root_dir():
    """Return root project directory."""
    return Path(__file__).parent.parent


# Import shared fixtures from common package
from lamquant_common_testdata.conftest import (
    real_test_edf,
    real_tuh_edfs,
    manifest_v3_path,
    lml_cli_binary,
    ternary_model,
    tmp_header,
    sample_seizure_mask,
    sample_npz_file,
)
```

**Diffs from current root conftest**:
- No sys.path manipulation (all dependencies are installed packages)
- Import only fixtures Eagle uses (benchmarks, validation, audits)
- Minimal pytest_sessionstart (Eagle doesn't need resource checks; it generates synthetic data)

---

## lamquant_common_testdata Package Structure

To be created in Lossless repo at: `tests/common_testdata/` (Python package)

**Files to move/copy**:
```
lamquant_common_testdata/
  __init__.py
  conftest.py              ← Core fixture definitions
  fixtures/
    __init__.py
    synthetic.py           ← Real-EDF loaders
  helpers/
    __init__.py
    data_paths.py          ← Resource path resolvers
    asserts.py             ← Codec assertion helpers
    edf_factory.py         ← EDF generation
    roundtrip.py           ← Roundtrip helpers (updated)
    rust_bindings.py       ← Rust PyO3 bindings wrapper
    signals.py             ← Signal generators
```

**Entry point** (`lamquant_common_testdata/__init__.py`):
```python
"""Shared test infrastructure for all LamQuant repos.

Provides:
  - Fixture definitions (session-scoped resource checks)
  - Real-EDF loaders and Q31 converters
  - Helper functions for assertions, signal generation, EDF creation
  - Rust bindings wrapper
"""

from .fixtures import (
    PYEDFLIB_TEST_GENERATOR,
    NEDC_PYPRINT_EXAMPLE,
    NEDC_TUH_EVAL_DIR,
    edf_to_q31_npz,
    find_real_test_edf,
    find_real_tuh_edfs,
    require_real_test_edf,
    require_real_tuh_edfs,
)

from .helpers.asserts import (
    assert_raises_lml,
    assert_bytes_equal,
    assert_array_equal_strict,
)

from .helpers.edf_factory import create_edf
from .helpers.signals import synth_signal, make_synthetic_eeg, adversarial_signals
from .helpers.rust_bindings import (
    HAS_RUST,
    requires_rust,
    rust_compress,
    rust_decompress,
    to_rust_signal,
)

__all__ = [
    # Fixtures
    'PYEDFLIB_TEST_GENERATOR',
    'NEDC_PYPRINT_EXAMPLE',
    'NEDC_TUH_EVAL_DIR',
    'edf_to_q31_npz',
    'find_real_test_edf',
    'find_real_tuh_edfs',
    'require_real_test_edf',
    'require_real_tuh_edfs',
    # Helpers
    'assert_raises_lml',
    'assert_bytes_equal',
    'assert_array_equal_strict',
    'create_edf',
    'synth_signal',
    'make_synthetic_eeg',
    'adversarial_signals',
    'HAS_RUST',
    'requires_rust',
    'rust_compress',
    'rust_decompress',
    'to_rust_signal',
]
```

**Installation**: In Neural and Eagle repo's top-level `pyproject.toml`:
```toml
[project]
dependencies = [
    "lamquant-lossless @ file://../lamquant-lossless",  # Or published to PyPI
]
```

Or (for development):
```bash
cd ../lamquant-lossless
pip install -e .
```

---

## benchmark_c_parity Decision

**File**: `tests/benchmarks/benchmark_c_parity.py` (Eagle repo)

**Current state**: Imports from `lamquant_codec.models.encoder` and `lamquant_codec.models.blocks` (LOSSLESS code)

**Options**:
1. ✅ **Leave as-is** — Eagle installs Lossless as a public dependency. Imports resolve correctly.
2. ❌ Skip when Lossless unavailable — Adds conditional logic to a high-value benchmark
3. ❌ Move to Neural — Wrong repo (benchmark is for public use)
4. ❌ Split into two benches — Adds complexity for no gain

**Decision**: **Option 1 (Leave as-is)**. The benchmark **must** run with actual Lossless encoder code to be meaningful. Eagle's `pyproject.toml` will list Lossless as a required dependency.

**No changes to benchmark code required.**

---

## Recommended Approach for Shared Fixtures

**Chosen Strategy**: **(c) Shared Python package in Lossless repo**

**Justification**:
- ✅ **No duplication** — single source of truth for all helpers
- ✅ **No drift risk** — common functions stay synchronized
- ✅ **Lossless doesn't bloat** — just a `tests/common_testdata/` package
- ✅ **No circular dependencies** — Lossless is independent; Neural/Eagle depend on it
- ✅ **Easy to distribute** — publish to PyPI later if needed
- ✅ **Minimal overhead** — just `pip install -e ../lamquant-lossless` in CI

**Rejected alternatives**:
- (a) Duplication: Would require per-repo sync rules; maintenance nightmare
- (b) Separate crate (Rust): Overkill for Python fixtures; added complexity
- (d) Other: None viable

---

## Migration Steps (Executable)

These steps assume you have three destination repos checked out:
- `/path/to/lamquant-lossless` (Lossless, public)
- `/path/to/lamquant-neural` (Neural, private)
- `/path/to/lamquant-eagle` (Eagle, public)

### Phase 1: Create Common Package in Lossless

```bash
cd /path/to/lamquant-lossless

# Create the common package directory
mkdir -p tests/common_testdata/fixtures tests/common_testdata/helpers

# Copy fixture files
cp /mnt/4tb/LamQuant/tests/fixtures/__init__.py tests/common_testdata/fixtures/
cp /mnt/4tb/LamQuant/tests/fixtures/synthetic.py tests/common_testdata/fixtures/

# Copy helper files
cp /mnt/4tb/LamQuant/tests/helpers/__init__.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/data_paths.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/asserts.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/edf_factory.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/roundtrip.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/rust_bindings.py tests/common_testdata/helpers/
cp /mnt/4tb/LamQuant/tests/helpers/signals.py tests/common_testdata/helpers/

# Create __init__.py files
touch tests/common_testdata/__init__.py

# Update package __init__.py with re-exports (see draft above)
# (Edit tests/common_testdata/__init__.py)

# Update imports in helpers (e.g., roundtrip.py) to be conditional
# (Edit tests/common_testdata/helpers/roundtrip.py — see note below)

# Update fixtures/__init__.py to use lamquant_common_testdata name
# (Edit tests/common_testdata/fixtures/__init__.py)

# Create conftest.py in tests/common_testdata/ with all fixture definitions
# (See draft in "Per-Repo conftest.py Drafts" section)
cp /mnt/4tb/LamQuant/tests/conftest.py tests/common_testdata/conftest.py
# Edit: keep only the core fixture definitions; remove sys.path manipulation

# Add to lossless pyproject.toml:
# [project]
# name = "lamquant-lossless"
# ...
# [project.optional-dependencies]
# test = ["lamquant-common-testdata @ file://."]

# Test that Lossless tests still pass
pytest tests/codec/ -x --tb=short

# Verify the common package can be imported
python -c "from lamquant_common_testdata.conftest import real_test_edf; print('OK')"
```

**Verification gate**: `pytest tests/codec/ tests/firmware/ -x` should pass

---

### Phase 2: Update Lossless Root conftest.py

```bash
cd /path/to/lamquant-lossless

# Backup original
cp tests/conftest.py tests/conftest.py.backup

# Create new conftest.py (see draft above)
# Edit tests/conftest.py:
#   1. Keep test_data_dir and root_dir (they're needed by per-dir conftests)
#   2. Remove all fixture definitions; import from lamquant_common_testdata.conftest
#   3. Keep sys.path manipulation for firmware/ only
#   4. Update pytest_sessionstart to Lossless-specific version

# (Manually edit or use a template — the draft is above)

# Test
pytest tests/codec/ -x --tb=short
pytest tests/firmware/ -x --tb=short
```

**Verification gate**: `pytest tests/codec/ tests/firmware/ tests/c_host/ -x` should pass

---

### Phase 3: Migrate Lossless codec/conftest.py (No Changes)

```bash
cd /path/to/lamquant-lossless

# Copy (no changes needed; it's already self-contained)
cp /mnt/4tb/LamQuant/tests/codec/conftest.py tests/codec/conftest.py

# Verify
pytest tests/codec/ -x --tb=short
```

**Verification gate**: All codec tests should still pass

---

### Phase 4: Set Up Neural Repo

```bash
cd /path/to/lamquant-neural

# Update pyproject.toml to depend on Lossless
# [project]
# dependencies = [
#     ...,
#     "lamquant-lossless @ file://../lamquant-lossless",
# ]

# Install Lossless in editable mode
pip install -e ../lamquant-lossless

# Create new root conftest.py (see draft above)
# Edit tests/conftest.py:
#   1. Keep test_data_dir and root_dir
#   2. Import fixtures from lamquant_common_testdata.conftest
#   3. Keep sys.path manipulation for ai_models/* only
#   4. Update pytest_sessionstart to Neural-specific version

# Copy training/conftest.py (no changes)
cp /mnt/4tb/LamQuant/tests/training/conftest.py tests/training/conftest.py

# Test
pytest tests/training/ -x --tb=short
pytest tests/snn/ -x --tb=short
```

**Verification gate**: `pytest tests/training/ tests/dataset_sim/ tests/snn/ -x` should pass

---

### Phase 5: Set Up Eagle Repo

```bash
cd /path/to/lamquant-eagle

# Update pyproject.toml to depend on Lossless
# [project]
# dependencies = [
#     ...,
#     "lamquant-lossless @ file://../lamquant-lossless",
# ]

# Install Lossless in editable mode
pip install -e ../lamquant-lossless

# Create minimal conftest.py (see draft above)
# Edit tests/conftest.py:
#   1. Keep test_data_dir and root_dir only
#   2. Import minimal fixtures from lamquant_common_testdata.conftest
#   3. No sys.path manipulation
#   4. Minimal pytest_sessionstart (or skip it)

# Test
pytest tests/benchmarks/ -x --tb=short
pytest tests/validation/ -x --tb=short
```

**Verification gate**: `pytest tests/benchmarks/ tests/validation/ tests/audits/ -x` should pass

---

### Phase 6: Smoke Test All Repos Together

```bash
cd /path/to/lamquant-lossless && pytest tests/ -x --tb=short
cd /path/to/lamquant-neural && pytest tests/ -x --tb=short
cd /path/to/lamquant-eagle && pytest tests/ -x --tb=short
```

**Expected result**: All three repos pass independently.

---

### Phase 7: Test Standalone Execution (Remove Siblings)

```bash
# In a fresh directory:
mkdir /tmp/eagle-isolated
cd /tmp/eagle-isolated

# Clone Eagle only
git clone <eagle-repo> .

# Install Lossless from PyPI (or manually built wheel)
pip install lamquant-lossless

# Run tests
pytest tests/ -x --tb=short
```

**Expected result**: Eagle tests pass without Neural sibling checked out.

---

## Import Path Adjustments Required

### In `lamquant_common_testdata/helpers/roundtrip.py`

**Current** (lines 16–17):
```python
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))
from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
```

**Updated**:
```python
try:
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
except ImportError:
    _compress_bytes = None
    _decompress_bytes = None

def assert_lml_roundtrip(signal, *, n_levels=3, label=""):
    if _compress_bytes is None or _decompress_bytes is None:
        pytest.skip("lamquant_codec.lossless not available")
    # ... rest of function
```

This allows the module to be imported by Neural/Eagle without breaking (they just skip if codec unavailable).

---

## Pytest Markers After Split

Each repo's `pyproject.toml` will register **only its markers**:

### Lossless `pyproject.toml`
```toml
[tool.pytest.ini_options]
markers = [
    "l1: KAT — fixed input, fixed output (codec/ only)",
    "l2: property/invariant",
    "l3: metamorphic / regression",
    "l4: fuzz",
    "l5: cross-implementation parity (Py↔Rust↔C)",
    "l7: adversarial / boundary",
    "slow: model loading or heavy computation (>5s)",
    "data: requires real EEG (q31_events / manifest_v3)",
    "rust: requires lamquant_core PyO3 wheel",
    "c_host: compiles and runs C code on the host",
    "cross_lang: Python↔Rust drift sentinel",
]
```

### Neural `pyproject.toml`
```toml
[tool.pytest.ini_options]
markers = [
    "l2: property/invariant",
    "l3: metamorphic / regression",
    "slow: model loading or heavy computation (>5s)",
    "data: requires real EEG (q31_events / manifest_v3)",
    "checkpoint: requires trained weights on disk",
]
```

### Eagle `pyproject.toml`
```toml
[tool.pytest.ini_options]
markers = [
    "slow: model loading or heavy computation (>5s)",
    "perf: performance regression sentinel (nightly only)",
    "data: requires real EEG (q31_events / manifest_v3)",
]
```

---

## Summary Table: Post-Split Fixture Locations

| Fixture / Helper | Current Location | Post-Split Location | Used By | Notes |
|------------------|------------------|-------------------|---------|-------|
| `test_data_dir` | root conftest | Each repo's conftest | All | Generic tempdir |
| `root_dir` | root conftest | Each repo's conftest | All | Needed by per-dir conftests |
| `real_test_edf` | root conftest (wraps synthetic.py) | Common (`lamquant_common_testdata`) | All | Multi-repo; skip-safe |
| `real_tuh_edfs` | root conftest (wraps synthetic.py) | Common | All | Multi-repo; skip-safe |
| `real_q31_from_edf` | root conftest (wraps synthetic.py) | Common | All | Multi-repo; skip-safe |
| `q31_events_dir` | root conftest (wraps data_paths.py) | Common | All | Multi-repo; skip-safe |
| `manifest_v3_path` | root conftest (wraps data_paths.py) | Common | All | Multi-repo; skip-safe |
| `student_checkpoint_path` | root conftest (wraps data_paths.py) | Common | Neural only | Single-repo; can stay |
| `lml_cli_binary` | root conftest (wraps data_paths.py) | Common | Lossless only | Single-repo; can stay |
| `rust_wheel` | root conftest (wraps rust_bindings.py) | Common | Lossless only | Single-repo; can stay |
| `sample_seizure_mask` | root conftest | Common | All (via sample_npz_file) | Multi-repo |
| `ternary_model` | root conftest | Common | Lossless + Neural | Multi-repo; imports lamquant_codec |
| `tmp_header` | root conftest | Common | Lossless only | Can stay |
| `sample_npz_file` | root conftest | Common | None observed | Unused; can stay |
| `sample_eeg_q31` | codec/conftest | Lossless codec/conftest | Lossless only | No cross-repo deps |
| `sample_eeg_float` | codec/conftest | Lossless codec/conftest | Lossless only | No cross-repo deps |
| `random_eeg_batch` | training/conftest | Neural training/conftest | Neural only | No cross-repo deps |
| `canonical_split_config` | training/conftest | Neural training/conftest | Neural only | No cross-repo deps |
| `validation_manifest` | training/conftest | Neural training/conftest | Neural only | No cross-repo deps |
| **Helpers** | | | | |
| `assert_raises_lml` | helpers/asserts.py | Common | Lossless only | Imports lamquant_codec |
| `assert_bytes_equal` | helpers/asserts.py | Common | Lossless only | Imports lamquant_codec |
| `assert_array_equal_strict` | helpers/asserts.py | Common | Lossless only | Imports lamquant_codec |
| `create_edf` | helpers/edf_factory.py | Common | Lossless only | No internal deps |
| `assert_lml_roundtrip` | helpers/roundtrip.py | Common | Unused | Conditional import |
| `assert_lml_compression_valid` | helpers/roundtrip.py | Common | Unused | Conditional import |
| `synth_signal` | helpers/signals.py | Common | Lossless only | No internal deps |
| `make_synthetic_eeg` | helpers/signals.py | Common | Lossless only | No internal deps |
| `adversarial_signals` | helpers/signals.py | Common | Unused | No internal deps |
| `HAS_RUST` | helpers/rust_bindings.py | Common | Lossless only | Conditional import of lamquant_core |
| `requires_rust` | helpers/rust_bindings.py | Common | Lossless only | Conditional import of lamquant_core |
| `rust_compress` | helpers/rust_bindings.py | Common | Lossless only | Conditional import of lamquant_core |
| `to_rust_signal` | helpers/rust_bindings.py | Common | Lossless only | Conditional import of lamquant_core |

---

## Risk Assessment & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Roundtrip helper breaks Neural import | Low | Conditional import + pytest.skip (done in draft) |
| Asserts helper breaks Neural import | Low | Conditional import + pytest.skip for lamquant_codec |
| Circular dependency (Common ← Lossless → Common) | Very Low | Common has no internal deps; Lossless imports Common |
| Drift in shared fixtures | Low | Single source in Common; synchronized across repos |
| Eagle tests fail to import Lossless | Medium | Set up dependency in pyproject.toml; test in Phase 7 |
| Neural tests fail without sibling (Lossless) | Low | Lossless is a public dependency; always available |
| sys.path issues in post-split repos | Medium | Test each repo independently after Phase 2–5 |

---

## Conclusion

The fixture split is **achievable with low risk** using a shared Common package in the Lossless repo. Key decisions:

1. ✅ Move all `tests/fixtures/` and `tests/helpers/` to `lamquant_common_testdata` package
2. ✅ Each repo's conftest imports from Common, with repo-specific sys.path and pytest_sessionstart
3. ✅ benchmark_c_parity stays in Eagle; Lossless is a declared dependency
4. ✅ Make roundtrip.py and asserts.py imports conditional to avoid breakage in Neural/Eagle
5. ✅ No fixtures are duplicated; drift risk is minimal

**Timeline**: ~2–3 days for full migration and testing, assuming parallel per-repo setup.
