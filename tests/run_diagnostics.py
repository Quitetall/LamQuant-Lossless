#!/usr/bin/env python3
"""Lossless diagnostics suite — discovers and runs lossless test categories.

This is the LOSSLESS-ONLY port of the monorepo diagnostic runner. The
neural / EAGLE checks (SNN/TNN training, Vocos decoder, paper verify,
benchmark sweeps) were DROPPED in the LamQuant-Lossless carve — they now
live in the LamQuant-Neural sibling repo. See the bottom of this file for
the explicit drop list.

What this harness covers (all deterministic, SDLC-governed, PCCP-out-of-scope):

    Rust gates (mirrors .github/workflows/ci.yml rust-host + firmware-no_std):
        cargo build  --workspace
        cargo test   --workspace --lib          (or `cargo nextest run` if installed)
        cargo test   --test byte_equal_backends  (load-bearing wire-format gate)
        cargo build  -p lamquant-firmware --no-default-features
                     --target riscv32imac-unknown-none-elf   (no_std MCU path)

    Python pytest categories (codec, container, edf_reader, integration,
    firmware C-parity, codec_python_smoke). Neural-coupled smoke tests
    self-skip via `pytest.importorskip` when the ai_models subtree is absent.

Tiers (each includes all lower tiers):
    fast    Pre-commit gate. Codec/container/edf roundtrip + edge cases +
            corruption detection + the Rust workspace build/lib tests +
            the byte-equal wire-format gate.
    full    Pre-push gate. fast + integration E2E + python codec smoke.
    ci      CI/nightly. full + firmware no_std build + C-host parity.

Usage:
    python tests/run_diagnostics.py              # fast (default, pre-commit)
    python tests/run_diagnostics.py full         # + integration + smoke
    python tests/run_diagnostics.py ci           # + firmware no_std + C parity
    python tests/run_diagnostics.py codec        # one python category
    python tests/run_diagnostics.py rust         # only the Rust gates
    python tests/run_diagnostics.py --list       # show categories (no run)
    python tests/run_diagnostics.py --dry-run    # print the plan, run nothing
    python tests/run_diagnostics.py -v           # verbose output

Exit code: 0 if all passed, 1 if any failed.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

# Firmware no_std build target (RP2350 / Hazard3 RISC-V).
_FIRMWARE_TARGET = "riscv32imac-unknown-none-elf"

# ── Rust gate registry ──
# Mirrors .github/workflows/ci.yml. These are the load-bearing lossless
# correctness checks: the cross-backend byte-equality gate guarantees
# ComputeBackend::Firmware and ComputeBackend::Desktop emit identical
# .lml bytes, and the no_std build protects the MCU path.
#
# `cargo nextest run` is preferred when installed (faster, better output);
# we fall back to `cargo test` otherwise. The byte_equal_backends gate
# always goes through `cargo test --test` (it is an integration test bin,
# runnable under either runner, but plain test keeps it dependency-free).
def _rust_gates():
    have_nextest = shutil.which("cargo-nextest") is not None
    if have_nextest:
        workspace_tests = {
            "cmd": ["cargo", "nextest", "run", "--workspace", "--lib"],
            "desc": "workspace lib tests (cargo nextest)",
        }
    else:
        workspace_tests = {
            "cmd": ["cargo", "test", "--workspace", "--lib"],
            "desc": "workspace lib tests (cargo test)",
        }
    return {
        "rust_build": {
            "cmd": ["cargo", "build", "--workspace"],
            "desc": "host workspace builds (default features)",
            "tier": "fast",
        },
        "rust_lib": {
            **workspace_tests,
            "tier": "fast",
        },
        "byte_equal": {
            "cmd": ["cargo", "test", "--test", "byte_equal_backends"],
            "desc": "cross-backend byte-equal wire-format gate (load-bearing)",
            "tier": "fast",
        },
        "firmware_no_std": {
            "cmd": [
                "cargo", "build", "-p", "lamquant-firmware",
                "--no-default-features", "--target", _FIRMWARE_TARGET,
            ],
            "desc": f"firmware no_std build ({_FIRMWARE_TARGET})",
            "tier": "ci",
            # Skip gracefully if the embedded target is not installed.
            "requires_target": _FIRMWARE_TARGET,
        },
    }


# ── Python pytest category registry ──
# tier: fast (pre-commit), full (pre-push), ci (nightly)
#
# DROPPED vs the monorepo registry:
#   - "training"  : neural — needs torch / GPU, lives in LamQuant-Neural.
#   - "decoder"   : neural — Vocos decoder + train_combined, LamQuant-Neural.
# Neural-coupled tests that physically remain under codec_python_smoke/
# self-skip via pytest.importorskip("subband_preprocess" / torch), so the
# category stays green without the ai_models subtree present.
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
    # Python codec smoke — surface-level imports + coverage; neural-coupled
    # files in here self-skip via importorskip.
    "codec_python_smoke": {
        "desc": "Python codec import/coverage smoke (neural-coupled self-skip)",
        "tier": "full",
    },
    # Firmware — C parity host + cross-impl. Needs a C host toolchain.
    "firmware": {
        "desc": "C parity, weight export, cross-impl",
        "tier": "ci",
    },
    "c_host": {
        "desc": "C host reference harness parity",
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
    """Find python test categories with test files.

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


def _target_installed(target):
    """Return True if the given rustup target is installed."""
    try:
        out = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return target in (out.stdout or "")


def run_rust_gate(name, gate, verbose=False, dry_run=False):
    """Run one Rust gate. Returns (status, returncode).

    status is one of "PASS", "FAIL", "SKIP".
    """
    cmd = gate["cmd"]
    if dry_run:
        print(f"    [dry-run] {' '.join(cmd)}")
        return "SKIP", 0

    req = gate.get("requires_target")
    if req and not _target_installed(req):
        print(f"    SKIP (rustup target {req} not installed: "
              f"`rustup target add {req}`)")
        return "SKIP", 0

    result = subprocess.run(
        cmd, cwd=str(_REPO_ROOT),
        capture_output=not verbose, text=True, timeout=1800,
    )
    ok = result.returncode == 0
    if not ok and not verbose:
        tail = ((result.stderr or "")[-800:] + "\n" + (result.stdout or "")[-400:])
        print(tail.strip())
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ("PASS" if ok else "FAIL"), result.returncode


def run_category(name, files, verbose=False, stop_early=False, dry_run=False):
    """Run one python category via pytest. Returns (passed, failed, returncode)."""
    args = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
    if stop_early:
        args.append("-x")
    if verbose:
        args.append("-v")
    args.extend(str(f) for f in files)

    if dry_run:
        print(f"    [dry-run] {' '.join(args[:5])} ... ({len(files)} files)")
        return 0, 0, 0

    result = subprocess.run(
        args, cwd=str(_REPO_ROOT),
        capture_output=not verbose, text=True, timeout=600,
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


def main():
    parser = argparse.ArgumentParser(description="LamQuant-Lossless diagnostics")
    parser.add_argument("targets", nargs="*", default=["fast"],
                        help="Tier (fast/full/ci), 'rust', or python category names")
    parser.add_argument("--list", action="store_true",
                        help="Show categories and Rust gates, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the execution plan without running anything")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.targets:
        args.targets = ["fast"]

    rust_gates = _rust_gates()

    # Determine tier filter for python file exclusions
    primary = args.targets[0]
    tier_filter = TIERS.get(primary) if primary in TIERS else None
    available = discover(tier_filter)

    if args.list:
        print("Rust gates (tier):")
        for name, g in rust_gates.items():
            print(f"  {name:15s} [{g['tier']:4s}]  {g['desc']}")
        print()
        print("Python categories (tier):")
        for name in sorted(available):
            info = available[name]
            n = len(info["files"])
            print(f"  {name:18s} [{info['tier']:4s}]  {n:2d} files  {info['desc']}")
        print()
        print("Tiers: fast (pre-commit), full (pre-push), ci (nightly)")
        print("Special target: 'rust' runs only the Rust gates.")
        return 0

    # Resolve which Rust gates and python categories run.
    selected_rust = {}
    selected_py = {}

    for target in args.targets:
        if target == "rust":
            selected_rust.update(rust_gates)
        elif target in TIERS:
            allowed = TIERS[target]
            for name, g in rust_gates.items():
                if g["tier"] in allowed:
                    selected_rust[name] = g
            for name, info in available.items():
                if info["tier"] in allowed:
                    selected_py[name] = info
        elif target in available:
            selected_py[target] = available[target]
        elif target in rust_gates:
            selected_rust[target] = rust_gates[target]
        else:
            print(f"Unknown target: {target}")
            print(f"Available python: {', '.join(sorted(available))}")
            print(f"Available rust:   {', '.join(rust_gates)}")
            print("Tiers: fast, full, ci | special: rust")
            return 1

    if not selected_rust and not selected_py:
        print("No test categories or Rust gates found.")
        return 1

    tier_label = primary if primary in TIERS else "custom"
    n_py_files = sum(len(info["files"]) for info in selected_py.values())

    print(f"{'='*64}")
    print(f"  LamQuant-Lossless Diagnostics — {tier_label}"
          f"{' (dry-run)' if args.dry_run else ''}")
    print(f"  {len(selected_rust)} rust gates, "
          f"{len(selected_py)} python categories ({n_py_files} files)")
    print(f"{'='*64}")
    print()

    t0 = time.time()
    total_p = total_f = 0
    results = {}
    any_fail = False

    # ── Rust gates first (cheap to fail fast on a broken build) ──
    for name in selected_rust:
        gate = selected_rust[name]
        print(f"  [rust:{name}] {gate['desc']}")
        status, rc = run_rust_gate(name, gate, verbose=args.verbose,
                                   dry_run=args.dry_run)
        results[f"rust:{name}"] = status
        print()
        if status == "FAIL":
            any_fail = True
            if tier_label == "fast":
                print("  Stopping on first failure (fast tier).")
                break

    # ── Python pytest categories ──
    if not (tier_label == "fast" and any_fail):
        for name in sorted(selected_py):
            info = selected_py[name]
            files = info["files"]
            print(f"  [{name}] {info['desc']}")
            p, f, rc = run_category(
                name, files, verbose=args.verbose,
                stop_early=(tier_label == "fast"),
                dry_run=args.dry_run,
            )
            total_p += p
            total_f += f
            ok = f == 0 and rc == 0
            results[name] = "SKIP" if args.dry_run else ("PASS" if ok else "FAIL")
            if not args.dry_run:
                print(f"    {p} passed, {f} failed  {'PASS' if ok else 'FAIL'}")
            print()
            if not ok and not args.dry_run:
                any_fail = True
                if tier_label == "fast":
                    print("  Stopping on first failure (fast tier).")
                    break

    elapsed = time.time() - t0

    print(f"{'='*64}")
    for name, status in sorted(results.items()):
        print(f"  {name:24s} {status}")
    print(f"{'='*64}")
    print(f"  {total_p} passed, {total_f} failed ({elapsed:.1f}s)")
    print(f"{'='*64}")

    if args.dry_run:
        return 0
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())


# ─────────────────────────────────────────────────────────────────────────
# DROPPED in the LamQuant-Lossless carve (these live in LamQuant-Neural now):
#   - python category "training"  : SNN/TNN config, checkpointing, QAT,
#                                    experiment log, safety. Needs torch/GPU.
#   - python category "decoder"   : Vocos decoder + train_combined +
#                                    perceptual losses. Neural, needs torch.
#   - coverage_floor subcommand   : ratcheted .coverage_floor.json over the
#                                    ai_models/ python package, which does not
#                                    exist in the lossless carve.
#   - paper verify / benchmark sweeps : EAGLE/paper reproduction; out of scope
#                                    for the deterministic lossless SDLC gate.
#                                    (Codec perf is its own scripts/run_benchmarks.sh.)
# ─────────────────────────────────────────────────────────────────────────
