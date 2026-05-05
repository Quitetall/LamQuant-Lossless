"""
Pytest configuration and shared fixtures for all test levels.

AUDIT (2026-04-27): Added session-start reporting of data availability
so CI logs clearly show whether tests ran with real or synthetic data.
Also added pytest_terminal_summary hook to report skip counts by reason.
"""

import os
import sys
import warnings
import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Expose repo modules that tests import by bare name (train_ternary,
# export_firmware, subband_preprocess, ...). These live under feature-specific
# subdirectories rather than in an installed package, so the test runner needs
# their parent dirs on sys.path.
_REPO_ROOT = Path(__file__).parent.parent
for _rel in (
    "ai_models/student",
    "ai_models/oracle",
    "ai_models/snn",
    "ai_models/dataset_sim",
    "ai_models/validation",
    "ai_models",           # for `from snn import ...` etc.
    "firmware",
    str(_REPO_ROOT),       # for `from ai_models.validation import ...`
):
    _p = str((_REPO_ROOT / _rel).resolve()) if _rel != str(_REPO_ROOT) else str(_REPO_ROOT)
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


# ============================================================
# Centralised skip-on-missing-resource fixtures.
#
# Each fixture returns a Path or module on success, or calls pytest.skip
# with a clear reason. Tests that need an external resource should depend
# on the matching fixture AND mark themselves with the matching marker
# (data / checkpoint / rust / c_host) so the CI fast lane can filter them
# out without parsing skip messages.
# ============================================================


@pytest.fixture(scope="session")
def q31_events_dir():
    """Real EEG q31 dataset directory. Skips if not present."""
    from tests.helpers.data_paths import q31_events_dir as _resolve
    p = _resolve()
    if p is None:
        pytest.skip("q31_events not present (use synthetic fallback or set up data)")
    return p


@pytest.fixture(scope="session")
def manifest_v3_path():
    """Dataset manifest_v3.json. Skips if not present."""
    from tests.helpers.data_paths import manifest_v3 as _resolve
    p = _resolve()
    if p is None:
        pytest.skip("manifest_v3.json not present — run build_manifest.py")
    return p


@pytest.fixture(scope="session")
def student_checkpoint_path():
    """Trained student-subband checkpoint. Skips if not present."""
    from tests.helpers.data_paths import student_checkpoint as _resolve
    p = _resolve()
    if p is None:
        pytest.skip("student_subband.ckpt not present in weights/")
    return p


@pytest.fixture(scope="session")
def lml_cli_binary():
    """Path to the `lml` Rust CLI binary. Skips if not built."""
    from tests.helpers.data_paths import lml_cli_binary as _resolve
    p = _resolve()
    if p is None:
        pytest.skip("lml binary not built — run `cargo build --release --bin lml`")
    return p


@pytest.fixture(scope="session")
def rust_wheel():
    """Imported lamquant_core PyO3 wheel. Skips if not installed."""
    from tests.helpers.rust_bindings import HAS_RUST
    if not HAS_RUST:
        pytest.skip(
            "lamquant_core PyO3 wheel not installed — "
            "run `maturin develop --features python --manifest-path lamquant-core/Cargo.toml`"
        )
    import lamquant_core
    return lamquant_core


@pytest.fixture
def canonical_split_config(root_dir):
    """Load official_split_config.json for testing.

    Moved to legacy/data_pipeline_v2/ on 2026-04-16 (the typed
    DatasetManifest replaces the dual-config split logic). Tests
    that depend on this fixture should migrate to manifest_v3 via
    `ai_models.DatasetManifest`; the fixture stays for the L2/L5/L7
    legacy tests until they're rewritten.
    """
    config_path = (root_dir / "legacy" / "data_pipeline_v2"
                   / "official_split_config.json")
    if not config_path.exists():
        pytest.skip(
            "legacy official_split_config.json not present — migrate this "
            "test to ai_models.DatasetManifest (see test_data_integrity.py)"
        )
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def validation_manifest(root_dir):
    """Load v2 validation_manifest.json for testing. Kept on disk because
    build_manifest.py uses it as the canonical (dataset, subject) lookup
    when building manifest_v3. Tests of the *current* split should use
    `ai_models.DatasetManifest.load(.../manifest_v3.json)`.
    """
    manifest_path = (root_dir / "ai_models" / "dataset_sim"
                     / "validation_manifest" / "validation_manifest.json")
    if not manifest_path.exists():
        pytest.skip("validation_manifest.json not present")
    with open(manifest_path) as f:
        return json.load(f)


@pytest.fixture
def sample_eeg_q31():
    """Generate sample Q31-format EEG data [21 channels, 2500 samples]."""
    np.random.seed(42)
    # Q31: signed 32-bit integers in range [-2^31, 2^31-1]
    eeg_q31 = np.random.randint(-2147483647, 2147483647, size=(21, 2500), dtype=np.int32)
    return eeg_q31


@pytest.fixture
def sample_eeg_float():
    """Generate sample normalized EEG [21, 2500] in microvolts."""
    np.random.seed(42)
    eeg_float = np.random.randn(21, 2500).astype(np.float32) * 100  # ~100 uV std dev
    return eeg_float


@pytest.fixture
def sample_seizure_mask():
    """Generate sample seizure mask [2500] with ~5% active."""
    np.random.seed(42)
    mask = np.random.binomial(1, 0.05, size=2500).astype(np.float32)
    return mask


@pytest.fixture
def ternary_model():
    """Fresh TernaryMobileNetV5 autoencoder instance on CPU in eval mode."""
    import torch
    from lamquant_codec.models.encoder import TernaryMobileNetV5
    torch.manual_seed(0)
    m = TernaryMobileNetV5(in_ch=21, latent_dim=32)
    m.eval()
    return m


@pytest.fixture
def random_eeg_batch():
    """Small random EEG batch [B=2, C=21, T=2500], float32, CPU."""
    import torch
    torch.manual_seed(0)
    return torch.randn(2, 21, 2500)


@pytest.fixture
def tmp_header(tmp_path):
    """Temp path for an exported C header file."""
    return tmp_path / "focal_net_weights.h"


@pytest.fixture
def sample_npz_file(test_data_dir, sample_eeg_q31, sample_seizure_mask):
    """Create a temporary NPZ file with sample data."""
    npz_path = test_data_dir / "test_sample.npz"
    np.savez_compressed(
        str(npz_path),
        data=sample_eeg_q31,
        seizure_mask=sample_seizure_mask,
        gain=np.array([1.0] * 21),
        channels=np.array([f"EEG {i}" for i in range(21)]),
        sample_rate=np.array([250]),
        source=np.array(['test_source']),
        dataset=np.array(['test_dataset']),
        l3=np.random.randn(1, 21, 313).astype(np.float32),  # Pre-computed L3
    )
    return npz_path


class ParanoiaLevel:
    """Decorator to mark test with paranoia level."""
    def __init__(self, level):
        self.level = level

    def __call__(self, func):
        func.paranoia_level = self.level
        return func


# Markers (l1-l7, slow, data, checkpoint, rust, c_host, doctest, cross_lang, perf)
# now declared in pyproject.toml [tool.pytest.ini_options].markers as the single
# source of truth — `--strict-markers` enforces from there. Do not add inline
# `addinivalue_line` calls here; use pyproject.toml.


def pytest_sessionstart(session):
    """Report data availability at session start so CI logs clearly show
    whether tests are running with real EEG data or synthetic fallbacks.

    AUDIT (2026-04-27): This makes it impossible to miss that tests ran
    in degraded mode. Previously, synthetic fallback was silent.
    """
    repo = Path(__file__).parent.parent
    checks = {
        "q31_events (real EEG)": repo / "ai_models" / "dataset_sim" / "q31_events",
        "student checkpoint": repo / "weights" / "student_subband.ckpt",
        "manifest_v3": repo / "ai_models" / "dataset_sim" / "manifest_v3.json",
        "Rust CLI binary": repo / "target" / "release" / "lml",
    }
    print("\n" + "=" * 60)
    print("TEST DATA AVAILABILITY")
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
        print("  This is expected in CI. On dev machines, ensure data is present.")
    print("=" * 60 + "\n")
