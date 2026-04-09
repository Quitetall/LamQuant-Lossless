"""
Pytest configuration and shared fixtures for all test levels.
"""

import os
import sys
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
for _rel in ("ai_models/student", "firmware", "ai_models/dataset_sim"):
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


@pytest.fixture
def canonical_split_config(root_dir):
    """Load official_split_config.json for testing."""
    config_path = root_dir / "ai_models" / "dataset_sim" / "official_split_config.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def validation_manifest(root_dir):
    """Load validation_manifest.json for testing."""
    manifest_path = root_dir / "ai_models" / "dataset_sim" / "validation_manifest" / "validation_manifest.json"
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
    from train_ternary import TernaryMobileNetV5
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


# Fixture markers for running by level
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "l1: KAT test (known good inputs)"
    )
    config.addinivalue_line(
        "markers", "l2: Property-based test (invariants)"
    )
    config.addinivalue_line(
        "markers", "l3: Metamorphic test (relationships)"
    )
    config.addinivalue_line(
        "markers", "l4: Fuzz test (random inputs)"
    )
    config.addinivalue_line(
        "markers", "l5: Cross-implementation test"
    )
    config.addinivalue_line(
        "markers", "l6: Statistical test (distributions)"
    )
    config.addinivalue_line(
        "markers", "l7: Adversarial test (edge cases)"
    )
