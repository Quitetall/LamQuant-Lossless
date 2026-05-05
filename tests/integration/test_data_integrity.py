"""tests/test_data_integrity.py — contracts on the typed data pipeline.

These tests exist because the OLD untyped pipeline shipped four
independent split bugs (TUH holdouts empty, validation_only datasets
orphaned, window-level masks ignored, TUEV parser misidentifying
subjects). Each bug was the same root cause: implicit contracts at
component boundaries.

Tests are organised by the contract they enforce:

  Manifest schema       — DatasetManifest round-trips, validate() runs
  Split semantics       — subject-disjoint, validation_only routing
  Parser correctness    — every NPZ on disk parses; TUEV → tuev_NNN
  Runtime safety        — TrainingBatch.assert_no_leakage catches leakage
  Build determinism     — build_manifest produces identical output twice
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'ai_models'))

from ai_models.data_types import (
    Dataset, Split, VALIDATION_ONLY_DATASETS,
    FileEntry, DatasetEntry, DatasetManifest, MANIFEST_VERSION,
    TrainingBatch, parse_npz_filename, ParsedFilename,
)

# Module-level @data marker — every test in this file requires the
# manifest or q31_events. CI fast lane filters with `-m "not data"`
# to skip cleanly.
pytestmark = pytest.mark.data

# Q31 dir kept as a module-level path for the @skipif markers below;
# the manifest itself is resolved via the centralised fixture.
Q31_DIR = _REPO / 'ai_models' / 'dataset_sim' / 'q31_events'


@pytest.fixture(scope='module')
def manifest(manifest_v3_path):
    """Load DatasetManifest from manifest_v3.json. Skips via the
    `manifest_v3_path` session fixture if the file is missing.
    """
    return DatasetManifest.load(manifest_v3_path)


# ============================================================
# Parser — handles every NPZ on disk, produces stable patient_ids
# ============================================================

def test_parser_chbmit_basic():
    p = parse_npz_filename('chbmit_chb01_03_q31.npz')
    assert p.dataset == Dataset.CHBMIT
    assert p.patient_id == 'chb01'
    assert p.session_id == '03'
    assert p.event_type == ''
    assert p.segment is None


def test_parser_chbmit_split_recording_normalises_to_subject():
    """chb17a/chb17b/chb17c are sub-recordings of subject 17 — must
    collapse to chb17 to preserve subject-disjoint holdout."""
    for letter in 'abc':
        p = parse_npz_filename(f'chbmit_chb17{letter}_03_q31.npz')
        assert p.patient_id == 'chb17', f'chb17{letter} did not normalise'
        assert letter in p.session_id   # recording letter preserved in session


def test_parser_chbmit_plus_session_suffix():
    p = parse_npz_filename('chbmit_chb02_16+_q31.npz')
    assert p.dataset == Dataset.CHBMIT
    assert p.patient_id == 'chb02'
    assert '+' in p.session_id


def test_parser_tuev_extracts_event_and_prefixes_patient():
    """Bug 4 permanent fix: tuh_spsw_046_a_1.npz parses correctly."""
    p = parse_npz_filename('tuh_spsw_046_a_1_q31.npz')
    assert p.dataset == Dataset.TUH_EVENTS
    assert p.patient_id == 'tuev_046'
    assert p.session_id == 'a'
    assert p.event_type == 'spsw'
    assert p.segment == 1


def test_parser_tuev_no_segment_suffix():
    """Some TUEV files end with trailing underscore (no segment number)."""
    p = parse_npz_filename('tuh_spsw_058_a__q31.npz')
    assert p.dataset == Dataset.TUH_EVENTS
    assert p.patient_id == 'tuev_058'
    assert p.event_type == 'spsw'
    assert p.segment is None


def test_parser_tuev_patient_no_collision_with_tuh_seizure():
    """TUH_SEIZURE patient '046' and TUEV patient '046' must not collide."""
    tuev = parse_npz_filename('tuh_spsw_046_a_1_q31.npz')
    # No TUH_SEIZURE file actually has just '046' as patient id
    # (real TUH ids are alphabetic codes like 'aaaaaaac'), but check the
    # design invariant: TUEV patient_ids are prefixed.
    assert tuev.patient_id.startswith('tuev_'), (
        'TUEV patients must have tuev_ prefix to prevent collision'
    )


def test_parser_tuh_seizure():
    p = parse_npz_filename('tuh_aaaaaaac_s002_t000_q31.npz')
    # Defaults to TUEG — builder overrides per annotations via v2 lookup.
    assert p.dataset == Dataset.TUEG
    assert p.patient_id == 'aaaaaaac'
    assert p.session_id == 's002_t000'
    assert p.event_type == ''


def test_parser_tuep_distinct_from_tuh():
    """TUEP files (tuep_ prefix) parse to TUH_EPILEPSY, NOT TUH_SEIZURE."""
    p = parse_npz_filename('tuep_aaaaaanr_s001_t001_q31.npz')
    assert p.dataset == Dataset.TUH_EPILEPSY, (
        'tuep_ prefix must map to TUH_EPILEPSY, not TUH_SEIZURE — '
        'else preprocessed TUEP files silently mix into seizure dataset'
    )
    assert p.patient_id == 'aaaaaanr'
    assert p.session_id == 's001_t001'


def test_parser_rejects_unknown():
    with pytest.raises(ValueError):
        parse_npz_filename('mystery_file_q31.npz')


@pytest.mark.skipif(not Q31_DIR.exists(), reason='q31_events not present')
def test_parser_handles_every_on_disk_file():
    """Every NPZ in q31_events/ must parse without error."""
    npz = sorted(Q31_DIR.glob('*.npz'))
    if not npz:
        pytest.skip('q31_events is empty')
    failures = []
    for f in npz:
        try:
            parse_npz_filename(f.name)
        except ValueError as e:
            failures.append((f.name, str(e)))
    if failures:
        sample = failures[:5]
        pytest.fail(
            f'{len(failures)}/{len(npz)} NPZs failed to parse. '
            f'First 5: {sample}'
        )


# ============================================================
# Manifest schema + round-trip
# ============================================================

def test_manifest_round_trip(tmp_path):
    """save() then load() must produce an equivalent manifest."""
    fe = FileEntry(
        path='/tmp/fake_q31.npz',
        dataset=Dataset.CHBMIT, patient_id='chb01', session_id='03',
        split=Split.TRAIN, n_windows=360,
    )
    de = DatasetEntry(name='chbmit', files=[fe])
    de.recompute_aggregates()
    m = DatasetManifest(seed=42, val_fraction=0.05, datasets={'chbmit': de})

    out = tmp_path / 'm.json'
    m.save(out)
    m2 = DatasetManifest.load(out)

    assert m2.version == m.version
    assert m2.seed == m.seed
    assert 'chbmit' in m2.datasets
    e2 = m2.datasets['chbmit']
    assert len(e2.files) == 1
    assert e2.files[0].dataset == Dataset.CHBMIT
    assert e2.files[0].split == Split.TRAIN
    assert e2.files[0].patient_id == 'chb01'


def test_manifest_load_raises_on_invalid_split(tmp_path):
    """validate() must catch a bad split and refuse to load."""
    raw = {
        'version': MANIFEST_VERSION, 'seed': 42,
        'datasets': {
            'chbmit': {
                'name': 'chbmit', 'n_files': 2, 'n_windows': 100,
                'n_patients': 1, 'n_train_files': 1, 'n_val_files': 1,
                'n_train_windows': 50, 'n_val_windows': 50,
                'holdout_patients': [], 'validation_only': False,
                'files': [
                    {'path': '/a.npz', 'dataset': 'chbmit', 'patient_id': 'chb01',
                     'session_id': '01', 'split': 'train', 'n_windows': 50,
                     'event_type': '', 'segment': None, 'has_seizure': False,
                     'sample_rate': 250, 'n_channels': 21},
                    {'path': '/b.npz', 'dataset': 'chbmit', 'patient_id': 'chb01',
                     'session_id': '02', 'split': 'val', 'n_windows': 50,
                     'event_type': '', 'segment': None, 'has_seizure': False,
                     'sample_rate': 250, 'n_channels': 21},
                ],
            },
        },
    }
    out = tmp_path / 'bad.json'
    out.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='LEAKAGE'):
        DatasetManifest.load(out)


# ============================================================
# Manifest content (depends on the on-disk manifest_v3)
# ============================================================

def test_manifest_no_leakage(manifest):
    """Subject-disjoint guarantee: no patient appears in both train and val."""
    issues = manifest.validate()
    assert issues == [], f'Manifest validation failed: {issues}'


def test_manifest_validation_only_datasets_never_in_train(manifest):
    """siena/eegmmidb/mental never appear in TRAIN if present."""
    train = manifest.get_files(Split.TRAIN)
    train_strs = [str(p) for p in train]
    for ds in VALIDATION_ONLY_DATASETS:
        bad = [p for p in train_strs if f'/{ds.value}_' in p or p.startswith(f'{ds.value}_')]
        assert not bad, (
            f'{len(bad)} files from {ds.value} appear in TRAIN — '
            f'validation_only datasets must never be in training.'
        )


def test_manifest_holdout_patients_actually_exist(manifest):
    """Every holdout_patients entry must correspond to a real patient."""
    for name, entry in manifest.datasets.items():
        actual = {f.patient_id for f in entry.files}
        phantoms = set(entry.holdout_patients) - actual
        assert not phantoms, (
            f'{name}: holdout_patients lists unknown patients {sorted(phantoms)[:5]}'
        )


def test_manifest_chbmit_keeps_chb21_24(manifest):
    """The canonical CHB-MIT baseline holdout set must be preserved."""
    chb = manifest.datasets.get('chbmit')
    if chb is None:
        pytest.skip('chbmit not in manifest')
    assert set(chb.holdout_patients) >= {'chb21', 'chb22', 'chb23', 'chb24'}, (
        'chbmit lost the canonical chb21-24 baseline holdout — every '
        'published comparison against this dataset assumes those four.'
    )


def test_manifest_val_fraction_reasonable(manifest):
    """Val should be 3-15% of total windows (5% target ± reasonable slack)."""
    if manifest.total_windows == 0:
        pytest.skip('empty manifest')
    val_pct = manifest.val_windows / manifest.total_windows
    assert 0.03 <= val_pct <= 0.15, (
        f'val fraction {val_pct:.2%} outside [3%, 15%] — '
        f'either too small to estimate R or too large (overfitting val)'
    )


def test_manifest_get_files_idempotent(manifest):
    """get_files(TRAIN) and get_files(VAL) must be disjoint."""
    train = set(map(str, manifest.get_files(Split.TRAIN)))
    val = set(map(str, manifest.get_files(Split.VAL)))
    overlap = train & val
    assert not overlap, f'TRAIN ∩ VAL = {len(overlap)} files (should be 0)'


def test_manifest_get_files_with_dataset_filter(manifest):
    """get_files(VAL, datasets=[CHBMIT]) returns only chbmit files."""
    chbmit_val = manifest.get_files(Split.VAL, datasets=[Dataset.CHBMIT])
    for p in chbmit_val:
        assert 'chbmit' in str(p), f'expected chbmit file, got {p}'


# ============================================================
# Runtime safety — TrainingBatch.assert_no_leakage
# ============================================================

def test_batch_leakage_assertion_fires_on_leak():
    """A val sample inside a train batch must trigger AssertionError."""
    import torch
    b = TrainingBatch(
        l3_approx=torch.zeros(4, 21, 313),
        datasets=['chbmit'] * 4,
        patient_ids=['chb01', 'chb02', 'chb21', 'chb03'],   # chb21 is val
        splits=['train', 'train', 'val', 'train'],
    )
    with pytest.raises(AssertionError, match='leakage'):
        b.assert_no_leakage(Split.TRAIN)


def test_batch_leakage_assertion_passes_on_clean_batch():
    import torch
    b = TrainingBatch(
        l3_approx=torch.zeros(3, 21, 313),
        datasets=['chbmit'] * 3,
        patient_ids=['chb01', 'chb02', 'chb03'],
        splits=['train', 'train', 'train'],
    )
    b.assert_no_leakage(Split.TRAIN)   # must not raise


def test_batch_subject_disjoint_assertion():
    """assert_subject_disjoint_from catches patient overlap between batches."""
    import torch
    a = TrainingBatch(
        l3_approx=torch.zeros(2, 21, 313),
        datasets=['chbmit', 'chbmit'],
        patient_ids=['chb01', 'chb02'],
        splits=['train', 'train'],
    )
    b = TrainingBatch(
        l3_approx=torch.zeros(2, 21, 313),
        datasets=['chbmit', 'chbmit'],
        patient_ids=['chb02', 'chb03'],
        splits=['train', 'train'],
    )
    with pytest.raises(AssertionError, match='overlap'):
        a.assert_subject_disjoint_from(b)


# ============================================================
# Build determinism
# ============================================================

@pytest.mark.skipif(not Q31_DIR.exists(), reason='q31_events not present')
def test_build_manifest_deterministic(tmp_path):
    """Two consecutive builds with the same seed must produce identical splits."""
    out1 = tmp_path / 'a.json'
    out2 = tmp_path / 'b.json'
    script = _REPO / 'ai_models' / 'dataset_sim' / 'build_manifest.py'

    for out in (out1, out2):
        result = subprocess.run(
            [sys.executable, str(script), '--output', str(out)],
            capture_output=True, text=True, cwd=str(_REPO),
        )
        if result.returncode != 0:
            pytest.fail(f'build_manifest failed:\n{result.stderr}')

    a = json.loads(out1.read_text())
    b = json.loads(out2.read_text())
    # `created` timestamp will differ; everything else must match.
    a.pop('created', None)
    b.pop('created', None)
    assert a == b, 'build_manifest is not deterministic'


# ============================================================
# Manifest hash — checkpoint provenance (refactor #72)
# ============================================================
# These tests pin the contract that "what manifest produced this
# checkpoint" is a deterministic, recoverable question — not guesswork
# from filenames or git timestamps.

def _tiny_manifest() -> 'DatasetManifest':
    fe = FileEntry(
        path='/tmp/fake_q31.npz',
        dataset=Dataset.CHBMIT, patient_id='chb01', session_id='03',
        split=Split.TRAIN, n_windows=360,
    )
    de = DatasetEntry(name='chbmit', files=[fe]); de.recompute_aggregates()
    m = DatasetManifest(seed=42, val_fraction=0.05, datasets={'chbmit': de})
    m.recompute_aggregates()
    return m


def test_manifest_hash_is_deterministic():
    """Two calls to .hash() on the same manifest produce the same digest."""
    m = _tiny_manifest()
    assert m.hash() == m.hash()


def test_manifest_hash_ignores_created_timestamp():
    """The `created` field changes every save but must NOT affect the hash."""
    m1 = _tiny_manifest()
    m2 = _tiny_manifest()
    m1.created = '2026-01-01T00:00:00+00:00'
    m2.created = '2026-12-31T23:59:59+00:00'
    assert m1.hash() == m2.hash(), (
        'manifest hash leaked the created timestamp — every save would '
        'invalidate every checkpoint provenance check'
    )


def test_manifest_hash_changes_when_split_changes():
    """A different split (different patient → different val fraction) must
    produce a different hash. Otherwise the hash is meaningless."""
    m1 = _tiny_manifest()
    m2 = _tiny_manifest()
    m2.datasets['chbmit'].files[0].split = Split.VAL
    m2.datasets['chbmit'].recompute_aggregates()
    m2.recompute_aggregates()
    assert m1.hash() != m2.hash()


def test_manifest_hash_format():
    """Hash format is 'sha256:<64 hex chars>' for unambiguous identification."""
    h = _tiny_manifest().hash()
    assert h.startswith('sha256:')
    assert len(h) == len('sha256:') + 64
    assert all(c in '0123456789abcdef' for c in h.removeprefix('sha256:'))


def test_manifest_diff_finds_split_changes():
    m1 = _tiny_manifest()
    m2 = _tiny_manifest()
    m2.datasets['chbmit'].files[0].split = Split.VAL
    m2.datasets['chbmit'].recompute_aggregates()
    m2.recompute_aggregates()
    diff = m1.diff(m2)
    # train_files / train_windows differ, val_files / val_windows differ
    assert 'train_files' in diff or 'val_files' in diff


def test_checkpoint_payload_carries_provenance(tmp_path):
    """CheckpointManager._save_atomic embeds the provenance dict."""
    import torch
    sys.path.insert(0, str(_REPO / 'ai_models' / 'student'))
    from checkpoint_manager import CheckpointManager, GuardConfig

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    m = _Stub()
    prov = {
        'manifest_hash': 'sha256:deadbeef',
        'manifest_path': '/tmp/fake_manifest.json',
        'manifest_version': '3.0.0',
        'run_id': 'test_run_123',
    }
    cm = CheckpointManager(m, tmp_path / 'ckpt.pt',
                            smoke_input=None, provenance=prov)
    cm.on_validation(epoch=1, val_r=0.5)
    saved = torch.load(tmp_path / 'ckpt.pt', map_location='cpu',
                        weights_only=False)
    assert isinstance(saved, dict) and 'state_dict' in saved
    for k, v in prov.items():
        assert saved[k] == v, f'provenance key {k} not round-tripped'
    assert 'best_val_r' in saved
    assert 'saved_at' in saved


def test_checkpoint_loaders_handle_both_formats(tmp_path):
    """Old (raw state_dict) and new (wrapped) checkpoints both load."""
    import torch
    sys.path.insert(0, str(_REPO / 'ai_models' / 'student'))
    from joint_codec import _ckpt_payload, read_ckpt_provenance

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    m1 = _Stub()
    # Old format: raw state_dict
    raw_path = tmp_path / 'old.ckpt'
    torch.save(m1.state_dict(), raw_path)
    # New format: wrapped with provenance
    new_path = tmp_path / 'new.ckpt'
    torch.save(_ckpt_payload(m1, {'manifest_hash': 'sha256:abc'}), new_path)

    # read_ckpt_provenance returns {} for old, prov dict for new
    assert read_ckpt_provenance(raw_path) == {}
    new_prov = read_ckpt_provenance(new_path)
    assert new_prov['manifest_hash'] == 'sha256:abc'
    assert 'saved_at' in new_prov

    # Both loadable into a fresh model via the standard hand-rolled idiom
    for path in (raw_path, new_path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        m2 = _Stub()
        m2.load_state_dict(state)
