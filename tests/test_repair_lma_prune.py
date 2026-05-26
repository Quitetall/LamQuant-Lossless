"""Unit tests for scripts/repair_lma_prune.py.

Synthetic file trees per corpus exercise the TUH allowlist:
* TUSZ — `.edf` + `.tse` + `.tse_bi` survive; `_labels.npz`, `_seiz.csv`,
  `meta.json`, `.lml` get pruned.
* TUEP — additionally keeps `.csv` (TUH ships them natively).
* TUEV — keeps `.lbl` + `.lbl_bi`.
* TUAB — minimal: `.edf` + `.txt`.
* TUEG — `.edf` + `.tse` + `.tse_bi` only.
* Root docs (`AAREADME.txt`, `RECORDS`, `DOCS/*`, …) survive all corpora.
* Empty subdirs get cleaned up after pruning.
* Double-extension allowlist (`.tse_bi`, `.csv_bi`, `.lbl_bi`) works.

Run:
    python3 -m pytest tests/test_repair_lma_prune.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ─── Module loader ────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "repair_lma_prune.py"


@pytest.fixture(scope="session")
def prune_module():
    """Import the script as a module so we can call its functions."""
    spec = importlib.util.spec_from_file_location("repair_lma_prune", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── Test tree builder ────────────────────────────────────────────────


def _make_tree(root: Path, files: list[str]) -> None:
    """Create files (touch) under `root`, making parent dirs as needed."""
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")  # 1 byte sentinel


# ─── Per-corpus scenarios ─────────────────────────────────────────────


def _common_root_docs() -> list[str]:
    return [
        "AAREADME.txt",
        "AAREADME.txt,v",
        "ANNOTATORS",
        "RECORDS",
        "RECORDS-WITH-SEIZURES",
        "SUBJECT-INFO",
        "DOCS/01_tcp_ar_montage.txt",
        "DOCS/nedc_ann_eeg_tools_map_v01.txt",
        "shoeb-icml-2010.pdf",
    ]


def test_tusz_keeps_edf_tse_tse_bi_drops_npz_meta_lml(tmp_path, prune_module):
    """TUSZ allowlist: `.edf`, `.tse`, `.tse_bi`."""
    corpus = tmp_path / "tusz_v2.0.6"
    files = _common_root_docs() + [
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.edf",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.tse",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.tse_bi",
        # things that must die:
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000_labels.npz",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000_seiz.csv",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.lml",
        "edf/dev/meta.json",
    ]
    _make_tree(corpus, files)

    audit = prune_module.prune_corpus_tree(corpus, "tusz")
    survivors = sorted(p.relative_to(corpus).as_posix()
                       for p in corpus.rglob("*") if p.is_file())

    # All root docs survive.
    for doc in _common_root_docs():
        assert doc in survivors, f"root doc {doc} should survive"

    # Real EDF + TUH sidecars survive.
    expected = {
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.edf",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.tse",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.tse_bi",
    }
    assert expected.issubset(set(survivors))

    # Forbidden files removed.
    forbidden = {
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000_labels.npz",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000_seiz.csv",
        "edf/dev/aaaa/s001/01_tcp_ar/aaaa_s001_t000.lml",
        "edf/dev/meta.json",
    }
    assert forbidden.isdisjoint(set(survivors))

    assert audit["corpus_id"] == "tusz"
    assert audit["removed_count"] == 4


def test_tuep_additionally_keeps_csv(tmp_path, prune_module):
    """TUEP ships `.csv` (seizure markers) — keep them."""
    corpus = tmp_path / "tuep_v3.1.0"
    files = _common_root_docs() + [
        "00_epilepsy/aaaa/s001/aaaa_s001_t000.edf",
        "00_epilepsy/aaaa/s001/aaaa_s001_t000.tse",
        "00_epilepsy/aaaa/s001/aaaa_s001_t000.tse_bi",
        "00_epilepsy/aaaa/s001/aaaa_s001_t000.csv",      # TUH ships
        "00_epilepsy/aaaa/s001/aaaa_s001_t000_labels.npz",  # ours, dies
    ]
    _make_tree(corpus, files)
    prune_module.prune_corpus_tree(corpus, "tuep")
    survivors = sorted(p.relative_to(corpus).as_posix()
                       for p in corpus.rglob("*") if p.is_file())
    assert "00_epilepsy/aaaa/s001/aaaa_s001_t000.csv" in survivors
    assert "00_epilepsy/aaaa/s001/aaaa_s001_t000_labels.npz" not in survivors


def test_tuev_keeps_lbl_and_lbl_bi(tmp_path, prune_module):
    """TUEV double-extension: `.lbl_bi` allowlist match."""
    corpus = tmp_path / "tuev_v2.0.1"
    files = _common_root_docs() + [
        "edf/eval/aaaa/s001/aaaa_s001_t000.edf",
        "edf/eval/aaaa/s001/aaaa_s001_t000.tse",
        "edf/eval/aaaa/s001/aaaa_s001_t000.tse_bi",
        "edf/eval/aaaa/s001/aaaa_s001_t000.lbl",
        "edf/eval/aaaa/s001/aaaa_s001_t000.lbl_bi",
        "edf/eval/aaaa/s001/aaaa_s001_t000_labels.npz",
    ]
    _make_tree(corpus, files)
    prune_module.prune_corpus_tree(corpus, "tuev")
    survivors = sorted(p.relative_to(corpus).as_posix()
                       for p in corpus.rglob("*") if p.is_file())
    assert "edf/eval/aaaa/s001/aaaa_s001_t000.lbl" in survivors
    assert "edf/eval/aaaa/s001/aaaa_s001_t000.lbl_bi" in survivors
    assert "edf/eval/aaaa/s001/aaaa_s001_t000_labels.npz" not in survivors


def test_tuab_minimal_edf_plus_txt(tmp_path, prune_module):
    """TUAB ships only `.edf` + `.txt` annotations."""
    corpus = tmp_path / "tuab_v3.0.1"
    files = _common_root_docs() + [
        "edf/abnormal/aaaa_s001_t000.edf",
        "edf/abnormal/aaaa_s001_t000.txt",
        "edf/abnormal/aaaa_s001_t000.tse",   # NOT shipped by TUAB → dies
        "edf/abnormal/aaaa_s001_t000_labels.npz",
    ]
    _make_tree(corpus, files)
    prune_module.prune_corpus_tree(corpus, "tuab")
    survivors = sorted(p.relative_to(corpus).as_posix()
                       for p in corpus.rglob("*") if p.is_file())
    assert "edf/abnormal/aaaa_s001_t000.edf" in survivors
    assert "edf/abnormal/aaaa_s001_t000.txt" in survivors
    # Our gen-only files die regardless of corpus.
    assert "edf/abnormal/aaaa_s001_t000_labels.npz" not in survivors


def test_summary_txt_preserved_across_corpora(tmp_path, prune_module):
    """`<stem>_summary.txt` files (per-record summaries) survive every corpus."""
    corpus = tmp_path / "tusz_v2.0.6"
    files = ["AAREADME.txt", "edf/aaaa_s001_t000_summary.txt"]
    _make_tree(corpus, files)
    prune_module.prune_corpus_tree(corpus, "tusz")
    assert (corpus / "edf/aaaa_s001_t000_summary.txt").exists()


def test_empty_dirs_cleaned_up(tmp_path, prune_module):
    """Subdirs left empty by pruning get rmdir'd, deepest-first."""
    corpus = tmp_path / "tusz_v2.0.6"
    files = [
        "AAREADME.txt",
        "stale/sub/dir/orphan_labels.npz",   # all-prune subtree
    ]
    _make_tree(corpus, files)
    prune_module.prune_corpus_tree(corpus, "tusz")
    # Pruned to AAREADME.txt only; the `stale/sub/dir/` chain must be gone.
    assert not (corpus / "stale").exists()


def test_audit_json_shape(tmp_path, prune_module):
    """`prune_corpus_tree` returns a dict with the expected fields + counts."""
    corpus = tmp_path / "tuar_v3.0.1"
    files = _common_root_docs() + [
        "edf/01_tcp_ar/aaaa_s001_t000.edf",
        "edf/01_tcp_ar/aaaa_s001_t000.tse_bi",
        "edf/01_tcp_ar/aaaa_s001_t000_labels.npz",
    ]
    _make_tree(corpus, files)
    audit = prune_module.prune_corpus_tree(corpus, "tuar")
    assert audit["corpus_id"] == "tuar"
    assert audit["corpus_dir"] == str(corpus)
    assert set(audit["pre_prune_counts"]) >= {".edf", ".tse_bi", "_labels.npz"}
    assert audit["pre_prune_counts"]["_labels.npz"] == 1
    assert "_labels.npz" not in audit["post_prune_counts"]
    assert audit["removed_count"] == 1


def test_dry_run_does_not_modify_disk(tmp_path, prune_module, capsys):
    """`--dry-run` reports candidates without touching files."""
    corpus = tmp_path / "tusz_v2.0.6"
    files = ["edf/aaaa.edf", "edf/aaaa_labels.npz"]
    _make_tree(corpus, files)
    rc = prune_module.main([str(corpus), "--corpus", "tusz", "--dry-run"])
    assert rc == 0
    # Both files must still exist on disk.
    assert (corpus / "edf/aaaa.edf").exists()
    assert (corpus / "edf/aaaa_labels.npz").exists()
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "WOULD-DELETE" in out or "would-delete" in out


def test_unknown_corpus_id_raises(tmp_path, prune_module):
    """Unknown corpus id rejects with SystemExit, not silent no-op."""
    corpus = tmp_path / "garbage"
    corpus.mkdir()
    with pytest.raises(SystemExit):
        prune_module.prune_corpus_tree(corpus, "doesnotexist")


def test_cli_writes_audit_json(tmp_path, prune_module):
    """End-to-end CLI path writes the audit JSON to the expected location."""
    corpus = tmp_path / "tusz_v2.0.6"
    files = ["AAREADME.txt", "edf/aaaa.edf", "edf/aaaa_labels.npz"]
    _make_tree(corpus, files)
    audit_out = tmp_path / "tusz.prune.json"
    rc = prune_module.main([
        str(corpus), "--corpus", "tusz", "--audit-out", str(audit_out)
    ])
    assert rc == 0
    audit = json.loads(audit_out.read_text())
    assert audit["corpus_id"] == "tusz"
    assert audit["removed_count"] >= 1
    assert not (corpus / "edf/aaaa_labels.npz").exists()
    assert (corpus / "edf/aaaa.edf").exists()
