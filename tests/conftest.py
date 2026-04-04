"""Shared fixtures for LamQuant Gen 6 test suite."""
import sys
import os
import pytest
import torch
import numpy as np

# Add project paths so tests can import ai_models and firmware modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for subdir in ['ai_models', 'ai_models/student', 'ai_models/dataset_sim',
               'firmware']:
    path = os.path.join(ROOT_DIR, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def ternary_model():
    """A freshly-initialized TernaryMobileNetV5 on CPU (no checkpoint)."""
    from train_ternary import TernaryMobileNetV5
    model = TernaryMobileNetV5(in_ch=21, latent_dim=32)
    model.eval()
    return model


@pytest.fixture
def random_eeg_batch():
    """Synthetic EEG batch: [B=2, 21ch, 2500 samples] normalised to ~±50 uV."""
    torch.manual_seed(42)
    return torch.randn(2, 21, 2500) * 20.0


@pytest.fixture
def tmp_header(tmp_path):
    """Return a path inside a temp dir for writing C header files."""
    return tmp_path / "test_weights.h"
