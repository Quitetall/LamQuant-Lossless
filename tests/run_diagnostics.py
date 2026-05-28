#!/usr/bin/env python3
"""Central diagnostics suite — discovers and runs all test categories.

Three tiers:
    fast    Pre-commit gate. Must stay under 30 seconds. Tests that
            catch the bugs you'd actually introduce in a normal commit:
            roundtrip integrity, edge cases, corruption detection.

    full    Pre-push gate. Everything in fast plus stress tests,
            cross-implementation checks, and real-file I/O.

    ci      CI/nightly. Full plus benchmarks and slow fuzz tests.

Usage:
    python tests/run_diagnostics.py              # fast (default, pre-commit)
    python tests/run_diagnostics.py full         # all categories
    python tests/run_diagnostics.py ci           # everything including benchmarks
    python tests/run_diagnostics.py codec        # one category
    python tests/run_diagnostics.py --list       # show categories
    python tests/run_diagnostics.py -v           # verbose output
    python tests/run_diagnostics.py coverage_floor   # ratchet check vs .coverage_floor.json

Exit code: 0 if all passed, 1 if any failed.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

# ── Category registry ──
# tier: fast (pre-commit), full (pre-push), ci (nightly)
CATEGORIES = {
    # Core codec — fast, no GPU, no disk I/O
    "codec": {
        "desc": "LMQ5 roundtrip, entropy coders, fused pipeline, conformance",
        "tier": "fast",
        # L5 vectorize parity is 80s — exclude from fast tier
        "exclude_patterns": ["test_l5_*"],
    },
    "container": {
        "desc": "LML container, versioning, corruption detection",
        "tier": "fast",
    },
    "edf_reader": {
        "desc": "EDF/EDF+/BDF reader, TAL, int24",
        "tier": "fast",
    },
    # Integration — needs real files, slower
    "integration": {
        "desc": "E2E codec, batch operations, package surface",
        "tier": "full",
    },
    # Training — needs torch, may need GPU
    "training": {
        "desc": "Config, checkpointing, experiment log, safety",
        "tier": "full",
    },
    # Firmware — needs C host toolchain
    "firmware": {
        "desc": "C parity, weight export, cross-impl",
        "tier": "ci",
    },
}

# Tiers: each includes all lower tiers
TIERS = {
    "fast": ["fast"],
    "full": ["fast", "full"],
    "ci":   ["fast", "full", "ci"],
}


def discover(tier_filter=None):
    """Find test categories with test files.

    When tier_filter is set (e.g. ['fast']), categories with
    exclude_patterns drop those files so the fast tier stays fast.
    """
    import fnmatch
    found = {}
    for name, info in CATEGORIES.items():
        cat_dir = _TESTS_DIR / name
        if cat_dir.is_dir():
            tests = sorted(cat_dir.glob("test_*.py"))
            # Apply exclude patterns when running in fast tier
            if tier_filter and 'fast' in tier_filter and info.get('exclude_patterns'):
                excludes = info['exclude_patterns']
                tests = [t for t in tests
                         if not any(fnmatch.fnmatch(t.name, pat) for pat in excludes)]
            if tests:
                found[name] = {**info, "files": tests}
    return found


def run_category(name, files, verbose=False, stop_early=False):
    """Run one category via pytest. Returns (passed, failed, returncode)."""
    args = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
    if stop_early:
        args.append("-x")
    if verbose:
        args.append("-v")
    args.extend(str(f) for f in files)

    result = subprocess.run(
        args, cwd=str(_REPO_ROOT),
        capture_output=not verbose, text=True, timeout=300,
    )

    passed = failed = 0
    output = result.stdout or ""
    for line in output.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m: failed = int(m.group(1))

    if not verbose and result.returncode != 0 and failed == 0:
        # Import error or crash — show tail
        tail = (result.stderr or "")[-300:] + "\n" + (result.stdout or "")[-300:]
        print(tail.strip())

    return passed, failed, result.returncode


def coverage_floor() -> int:
    """Compare current .coverage data against `.coverage_floor.json` floor.

    Reads coverage produced by an earlier `pytest --cov` invocation. If
    no .coverage file is present, runs the python-fast lane equivalent
    inline so the subcommand is usable from a pre-push hook too.

    Returns exit code: 0 if current >= floor, 1 if regression.
    """
    import json

    floor_path = _REPO_ROOT / ".coverage_floor.json"
    if not floor_path.exists():
        print(f"[coverage_floor] missing {floor_path} — set initial floor")
        return 1
    floor_data = json.loads(floor_path.read_text())
    floor_pct = float(floor_data["floor"]) * 100.0
    scope = floor_data.get("scope", ["ai_models", "firmware"])

    cov_data_path = _REPO_ROOT / ".coverage"
    if not cov_data_path.exists():
        # No coverage data — run pytest --cov to produce it.
        print("[coverage_floor] .coverage missing — running pytest --cov")
        cov_args = [
            sys.executable, "-m", "pytest", "tests/",
            "-m", "not slow and not perf and not data and not checkpoint",
            "-q",
        ]
        for src in scope:
            cov_args.extend(["--cov", src])
        rc = subprocess.run(cov_args, cwd=str(_REPO_ROOT)).returncode
        if rc != 0:
            print(f"[coverage_floor] pytest failed (rc={rc}) — abort")
            return rc

    # Compute coverage percentage via the coverage Python API so we
    # bypass pyproject's [tool.coverage.report] fail_under (the real
    # gate is the floor file, not that config knob).
    try:
        import coverage
    except ImportError:
        print("[coverage_floor] `coverage` package missing — `pip install coverage`")
        return 1
    cov = coverage.Coverage()
    try:
        cov.load()
    except coverage.misc.CoverageException as e:
        print(f"[coverage_floor] could not load .coverage: {e}")
        return 1
    with open(os.devnull, "w") as devnull:
        current_pct = cov.report(file=devnull)

    if current_pct < floor_pct:
        print(
            f"[coverage_floor] REGRESSION: {current_pct:.2f}% < floor "
            f"{floor_pct:.2f}% (scope={scope}). Either add tests or "
            f"investigate why coverage dropped — do NOT lower the floor "
            f"to make this pass."
        )
        return 1

    delta = current_pct - floor_pct
    print(
        f"[coverage_floor] OK: {current_pct:.2f}% "
        f"(floor: {floor_pct:.2f}%, gain: +{delta:.2f}%) scope={scope}"
    )
    if delta >= 0.5:
        print(
            f"[coverage_floor] suggestion: bump .coverage_floor.json "
            f"floor to {current_pct / 100.0:.4f} to lock in the gain."
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description="LamQuant diagnostics")
    parser.add_argument("targets", nargs="*", default=["fast"],
                        help="Tier (fast/full/ci) or category names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Coverage ratchet shortcut — runs before normal target dispatch.
    if args.targets and args.targets[0] == "coverage_floor":
        return coverage_floor()

    # Determine tier filter for file exclusions
    tier_filter = TIERS.get(args.targets[0]) if args.targets and args.targets[0] in TIERS else None
    available = discover(tier_filter)

    if args.list:
        print("Categories (tier):")
        for name in sorted(available):
            info = available[name]
            n = len(info["files"])
            print(f"  {name:15s} [{info['tier']:4s}]  {n:2d} files  {info['desc']}")
        print()
        print("Tiers: fast (pre-commit), full (pre-push), ci (nightly)")
        return 0

    # Resolve targets → category list
    selected = {}
    for target in args.targets:
        if target in TIERS:
            allowed = TIERS[target]
            for name, info in available.items():
                if info["tier"] in allowed:
                    selected[name] = info
        elif target in available:
            selected[target] = available[target]
        else:
            print(f"Unknown target: {target}")
            print(f"Available: {', '.join(sorted(available))} | fast, full, ci")
            return 1

    if not selected:
        print("No test categories found.")
        return 1

    n_files = sum(len(info["files"]) for info in selected.values())
    tier_label = args.targets[0] if args.targets[0] in TIERS else "custom"

    print(f"{'='*60}")
    print(f"  LamQuant Diagnostics — {tier_label}")
    print(f"  {len(selected)} categories, {n_files} test files")
    print(f"{'='*60}")
    print()

    t0 = time.time()
    total_p = total_f = 0
    results = {}

    for name in sorted(selected):
        info = selected[name]
        files = info["files"]
        print(f"  [{name}] {info['desc']}")
        p, f, rc = run_category(name, files, verbose=args.verbose,
                                stop_early=(tier_label == "fast"))
        total_p += p
        total_f += f
        ok = f == 0 and rc == 0
        results[name] = "PASS" if ok else "FAIL"
        print(f"    {p} passed, {f} failed  {'PASS' if ok else 'FAIL'}")
        print()
        if tier_label == "fast" and not ok:
            print("  Stopping on first failure (fast tier).")
            break

    elapsed = time.time() - t0

    print(f"{'='*60}")
    for name, status in sorted(results.items()):
        print(f"  {name:15s} {status}")
    print(f"{'='*60}")
    print(f"  {total_p} passed, {total_f} failed ({elapsed:.1f}s)")
    print(f"{'='*60}")

    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
