#!/usr/bin/env python3
"""
Recover orphan tempfile NPZs left behind by a buggy precompute_l3_fast.py.

BACKGROUND
----------
`precompute_l3_fast.precompute_file_l3` used this pattern:

    with tempfile.NamedTemporaryFile(dir=..., delete=False) as tmp:
        tmp_path = tmp.name              # e.g. /.../q31_events/tmpABC123
    np.savez_compressed(tmp_path, **payload)   # -> /.../tmpABC123.npz  (!)
    shutil.move(tmp_path, npz_path)       # moves the EMPTY original tempfile

Because `np.savez_compressed` auto-appends `.npz` when the filename doesn't
end in `.npz`, the actual L3-enriched output was written to `<tmp>.npz` while
`tmp_path` still pointed to `<tmp>` (an empty file). The subsequent
`shutil.move` then clobbered the real q31 NPZ with that empty tempfile,
producing thousands of 0-byte files and leaving every legitimately-processed
output stranded as a `tmp*.npz` orphan.

Each orphan still contains the full payload: {data, gain, channels,
seizure_mask, source, dataset, sample_rate, l3}. From `source` + `dataset`
we can reconstruct the intended filename:

    out_name = f"{dataset}_{basename(source, strip .edf)}_q31.npz"

This tool walks the q31_events directory, reads each `tmp*.npz` orphan,
derives its target name, and moves it into place. Existing valid targets
with an `l3` key are left alone; the orphan is deleted. Existing 0-byte
or l3-less targets are replaced.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def derive_target_name(orphan_path: Path) -> str:
    """Compute the canonical q31 filename this orphan should have become."""
    with np.load(orphan_path, allow_pickle=True) as z:
        source = str(z["source"]).strip()
        dataset = str(z["dataset"]).strip()
    # source is the original EDF basename, e.g. "aaaaacon_00000001.edf"
    stem = os.path.splitext(os.path.basename(source))[0]
    return f"{dataset}_{stem}_q31.npz"


def orphan_has_l3(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as z:
            return "l3" in z.files
    except Exception:
        return False


def target_has_l3(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return orphan_has_l3(path)


def recover(q31_dir: Path, dry_run: bool = False) -> dict:
    orphans = sorted(q31_dir.glob("tmp*.npz"))
    stats = {
        "orphans_found": len(orphans),
        "recovered": 0,
        "skipped_target_already_has_l3": 0,
        "skipped_orphan_invalid": 0,
        "skipped_orphan_no_l3": 0,
        "collisions_broken": [],
    }

    for orphan in orphans:
        # Validate orphan
        if orphan.stat().st_size == 0:
            stats["skipped_orphan_invalid"] += 1
            if not dry_run:
                orphan.unlink()
            continue
        try:
            target_name = derive_target_name(orphan)
        except Exception as e:
            stats["skipped_orphan_invalid"] += 1
            stats["collisions_broken"].append(f"{orphan.name}: {e}")
            continue

        if not orphan_has_l3(orphan):
            stats["skipped_orphan_no_l3"] += 1
            if not dry_run:
                orphan.unlink()
            continue

        target = q31_dir / target_name
        if target_has_l3(target):
            stats["skipped_target_already_has_l3"] += 1
            if not dry_run:
                orphan.unlink()
            continue

        if dry_run:
            print(f"[dry-run] {orphan.name} -> {target.name}")
        else:
            # Remove any stale 0-byte file sitting at the target
            if target.exists() and target.stat().st_size == 0:
                target.unlink()
            shutil.move(str(orphan), str(target))
        stats["recovered"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="ai_models/dataset_sim/q31_events",
                    help="Directory containing the q31 NPZ files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without moving anything")
    args = ap.parse_args()

    q31_dir = Path(args.dir).resolve()
    if not q31_dir.is_dir():
        print(f"[!] not a directory: {q31_dir}", file=sys.stderr)
        sys.exit(1)

    stats = recover(q31_dir, dry_run=args.dry_run)

    print(f"orphans found:                     {stats['orphans_found']}")
    print(f"recovered (moved into place):      {stats['recovered']}")
    print(f"skipped (target already has l3):   {stats['skipped_target_already_has_l3']}")
    print(f"skipped (orphan had no l3 key):    {stats['skipped_orphan_no_l3']}")
    print(f"skipped (orphan unreadable):       {stats['skipped_orphan_invalid']}")
    if stats["collisions_broken"]:
        print("errors:")
        for line in stats["collisions_broken"][:20]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
