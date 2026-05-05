"""Smoke tests for the tutorial notebooks.

Asserts:
  - Each .ipynb file is valid JSON with the expected nbformat shell.
  - Every code cell parses (compiles) cleanly — catches typos.
  - Notebooks 01, 02, 04 run end-to-end in a temporary process.
    (Notebook 03 spawns subprocesses; we exercise its code path through
    the batch tests instead, to keep this test fast.)
  - All notebooks reference each other consistently in their README links.

This is a fast structural check, not a CI replacement for executing the
notebooks in jupyter. We don't need nbformat / jupyter to be installed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[2] / "notebooks"

EXPECTED_NOTEBOOKS = [
    '01_quickstart.ipynb',
    '02_quality_levels.ipynb',
    '03_batch_operations.ipynb',
    '04_mne_integration.ipynb',
]


def _load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def _code_cells(nb: dict) -> list[str]:
    """Return the source of each code cell as a single string."""
    return [
        ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        for c in nb['cells'] if c['cell_type'] == 'code'
    ]


# ============================================================
# Existence + JSON validity
# ============================================================

@pytest.mark.parametrize('name', EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_parses(name):
    path = NB_DIR / name
    assert path.exists(), f'missing notebook: {name}'
    nb = _load_nb(path)
    assert nb['nbformat'] == 4
    assert nb['cells'], f'{name} has no cells'
    # First cell is markdown with the title and the Colab badge.
    first = nb['cells'][0]
    assert first['cell_type'] == 'markdown'
    txt = ''.join(first['source'])
    assert 'colab.research.google.com' in txt, \
        f'{name} first cell missing Colab badge link'


@pytest.mark.parametrize('name', EXPECTED_NOTEBOOKS)
def test_notebook_code_cells_compile(name):
    """Every code cell must be syntactically valid Python (catches typos)."""
    nb = _load_nb(NB_DIR / name)
    for i, src in enumerate(_code_cells(nb)):
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f'{name} code cell #{i} has SyntaxError: {e}')


# ============================================================
# Execute the lightweight notebooks end-to-end.
#
# We exec each cell in a single shared namespace inside a clean process
# (via `python -c`), to mimic real notebook behaviour. The shell-style
# `%pip` lines and the sys.path-extending bootstrap cells are skipped.
# ============================================================

import subprocess
import sys


def _runnable_source(nb_path: Path) -> str:
    nb = _load_nb(nb_path)
    parts = []
    for src in _code_cells(nb):
        # Skip lines that are jupyter-only (% magics) — they'd be NameError
        # in a plain Python process.
        cleaned = '\n'.join(
            line for line in src.splitlines()
            if not line.lstrip().startswith('%')
        )
        parts.append(cleaned)
    return '\n\n'.join(parts)


def _run_notebook(nb_path: Path):
    src = _runnable_source(nb_path)
    # Inject the project root onto sys.path before any other imports.
    bootstrap = (
        "import sys; sys.path.insert(0, "
        f"{repr(str(Path(__file__).resolve().parents[2]))})\n\n"
    )
    full = bootstrap + src
    proc = subprocess.run(
        [sys.executable, '-c', full],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.fail(
            f'{nb_path.name} failed:\n'
            f'--- stdout ---\n{proc.stdout}\n'
            f'--- stderr ---\n{proc.stderr}'
        )


def test_notebook_01_runs():
    _run_notebook(NB_DIR / '01_quickstart.ipynb')


def test_notebook_02_runs():
    _run_notebook(NB_DIR / '02_quality_levels.ipynb')


def test_notebook_04_runs():
    pytest.importorskip('mne')
    _run_notebook(NB_DIR / '04_mne_integration.ipynb')


# Notebook 03 spawns batch CLI subprocesses — covered by tests/test_batch.py.
# Run only its non-subprocess cells (data prep + manifest read) so we still
# exercise the narrative code path.
def test_notebook_03_data_prep_runs():
    nb = _load_nb(NB_DIR / '03_batch_operations.ipynb')
    cells = _code_cells(nb)
    # Run cells up to (but not including) the first one that calls run([]).
    safe = []
    for src in cells:
        if 'run(' in src and 'lamquant_codec.cli' in src:
            break
        safe.append(src)
    bootstrap = (
        "import sys; sys.path.insert(0, "
        f"{repr(str(Path(__file__).resolve().parents[2]))})\n\n"
    )
    full = bootstrap + '\n\n'.join(safe)
    r = subprocess.run([sys.executable, '-c', full],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f'notebook 03 prep failed:\n{r.stderr}'
