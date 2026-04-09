#!/usr/bin/env python3
"""
Verify L3 integrity: recompute L3 from `data` in a sample of q31 NPZ files
and compare bit-exactly against the stored `l3` array.

Categorizes each file as:
  - exact    : np.array_equal(stored, fresh)
  - close    : relative difference < 1e-5 (float32 reduction-order noise)
  - differ   : relative difference >= 1e-5 (real mismatch — orphan recovery
               assigned the file to the wrong target, or data was corrupted)
  - shape    : shape mismatch between stored and fresh
  - no_l3    : file had no `l3` key (not yet processed)
  - error    : load or compute failure

A clean run has zero `differ` and zero `shape` — anything else is a red flag.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

import numpy as np


# Make subband_preprocess importable in each worker
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ai_models", "student"))


def recompute_l3(eeg_q31):
    """Mirror of precompute_file_l3's inner loop — Q31 → float → windowed L3."""
    from subband_preprocess import preprocess_subband_single

    eeg_float = (eeg_q31.astype(np.float32) / 2147483647.0) * 1000.0
    T = eeg_float.shape[1]
    num_windows = (T + 2499) // 2500

    out = []
    for w in range(num_windows):
        start = w * 2500
        end = min(start + 2500, T)
        win = eeg_float[:, start:end].astype(np.float32)
        if win.shape[1] < 2500:
            win = np.pad(win, ((0, 0), (0, 2500 - win.shape[1])))
        l3, _, _ = preprocess_subband_single(win, order=8, autocorr_len=256)
        out.append(l3)
    return np.stack(out, axis=0).astype(np.float32)


def check_file(path):
    """Returns (category, detail_dict) for one file."""
    try:
        with np.load(path) as d:
            if "l3" not in d.files:
                return ("no_l3", {"path": os.path.basename(path)})
            eeg_q31 = np.asarray(d["data"])
            stored = np.asarray(d["l3"])
    except Exception as e:
        return ("error", {"path": os.path.basename(path), "msg": str(e)})

    try:
        fresh = recompute_l3(eeg_q31)
    except Exception as e:
        return ("error", {"path": os.path.basename(path), "msg": f"recompute: {e}"})

    if stored.shape != fresh.shape:
        return ("shape", {
            "path": os.path.basename(path),
            "stored": stored.shape,
            "fresh": fresh.shape,
        })

    if not np.all(np.isfinite(stored)):
        return ("differ", {
            "path": os.path.basename(path),
            "reason": "stored contains NaN/Inf",
            "max_abs": float("nan"),
            "rel": float("nan"),
        })

    if np.array_equal(stored, fresh):
        return ("exact", {"path": os.path.basename(path), "shape": stored.shape})

    max_abs = float(np.max(np.abs(stored - fresh)))
    rms = float(np.sqrt(np.mean(fresh ** 2)))
    rel = max_abs / (rms + 1e-12)
    if rel < 1e-5:
        return ("close", {
            "path": os.path.basename(path),
            "max_abs": max_abs,
            "rel": rel,
        })
    return ("differ", {
        "path": os.path.basename(path),
        "max_abs": max_abs,
        "rel": rel,
        "reason": "rel >= 1e-5",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="ai_models/dataset_sim/q31_events")
    ap.add_argument("--sample", type=int, default=200,
                    help="Number of files to verify")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pattern", default="*_q31.npz")
    args = ap.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    print(f"candidate files in {args.dir}: {len(all_files)}", flush=True)

    # Restrict to files that have l3 so we're actually verifying recovered/
    # processed data rather than picking unprocessed files that would all
    # land in `no_l3`.
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    picked = []
    for p in all_files:
        try:
            with np.load(p) as d:
                if "l3" in d.files:
                    picked.append(p)
                    if len(picked) >= args.sample:
                        break
        except Exception:
            continue
    print(f"sampled {len(picked)} files with 'l3' key", flush=True)
    if not picked:
        print("no files with l3 key — nothing to verify", flush=True)
        return

    t0 = time.time()
    counts = Counter()
    details_differ = []
    details_shape = []
    details_error = []
    close_rels = []

    with mp.Pool(args.workers) as pool:
        n = 0
        for category, info in pool.imap_unordered(check_file, picked, chunksize=1):
            n += 1
            counts[category] += 1
            if category == "differ":
                details_differ.append(info)
                if len(details_differ) <= 20:
                    print(f"  [{n}] DIFFER  {info}", flush=True)
            elif category == "shape":
                details_shape.append(info)
                print(f"  [{n}] SHAPE   {info}", flush=True)
            elif category == "error":
                details_error.append(info)
                print(f"  [{n}] ERROR   {info}", flush=True)
            elif category == "close":
                close_rels.append(info["rel"])
            if n % 20 == 0:
                elapsed = time.time() - t0
                rate = n / elapsed
                eta = (len(picked) - n) / rate if rate > 0 else 0
                print(f"  [{n}/{len(picked)}] {rate:.1f} files/s eta={eta:.0f}s "
                      f"exact={counts['exact']} close={counts['close']} "
                      f"differ={counts['differ']}", flush=True)

    dt = time.time() - t0
    print(flush=True)
    print(f"elapsed: {dt:.1f}s  ({n/dt:.1f} files/s)", flush=True)
    print(flush=True)
    print(f"SUMMARY ({len(picked)} files)", flush=True)
    print(f"  exact:     {counts['exact']:5d}  bit-identical to fresh recomputation", flush=True)
    print(f"  close:     {counts['close']:5d}  within float32 noise (rel < 1e-5)", flush=True)
    print(f"  differ:    {counts['differ']:5d}  rel >= 1e-5 (INVESTIGATE)", flush=True)
    print(f"  shape:     {counts['shape']:5d}  shape mismatch (INVESTIGATE)", flush=True)
    print(f"  no_l3:     {counts['no_l3']:5d}", flush=True)
    print(f"  error:     {counts['error']:5d}", flush=True)

    if close_rels:
        print(flush=True)
        print("close-case relative diff distribution:", flush=True)
        a = np.asarray(close_rels)
        print(f"  min={a.min():.3g}  p50={np.median(a):.3g}  "
              f"p95={np.percentile(a,95):.3g}  max={a.max():.3g}", flush=True)

    passed = (counts["differ"] == 0 and counts["shape"] == 0 and counts["error"] == 0)
    print(flush=True)
    if passed:
        print("VERDICT: L3 data is clean.", flush=True)
        sys.exit(0)
    else:
        print("VERDICT: FAILURES — investigate above.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
