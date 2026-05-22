"""Real-fixture loaders for the LamQuant test suite.

User direction (2026-05-21): no synthetic data — every test that needs
EEG bytes reads from the real fixtures committed to ``reference_software/``
(via a sibling git-ignored copy) or skips cleanly when absent.

Public API (registered as pytest fixtures via ``tests/conftest.py``):

  - ``real_test_edf``     — small canonical real EDF (skips if absent)
  - ``real_tuh_edfs``     — list of real TUH-style EDFs (skips if absent)
  - ``real_q31_from_edf`` — tmp_path NPZ derived from a real EDF via
                             the production preprocess pipeline
"""

from .synthetic import (  # noqa: F401
    PYEDFLIB_TEST_GENERATOR,
    NEDC_PYPRINT_EXAMPLE,
    NEDC_TUH_EVAL_DIR,
    edf_to_q31_npz,
    find_real_test_edf,
    find_real_tuh_edfs,
    require_real_test_edf,
    require_real_tuh_edfs,
)
