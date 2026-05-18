#!/usr/bin/env python3
"""
Phase 8 / Item B — dump EEGLAB `.set` metadata into the LamQuant
sidecar JSON that the Rust reader consumes.

Usage:
    python3 tools/dump_eeglab_meta.py recording.set [-o recording.lml-meta.json]

Default output: `<set>.lml-meta.json` next to the source.

The LamQuant `EeglabReader` v1 reads metadata from this sidecar
(n_channels, n_samples, sample_rate, channels) rather than parsing
the full MATLAB 5 binary format inline. Generate once per recording
when you start working with the file; the sidecar lives alongside
the `.set` permanently.

Requires `scipy` (for `scipy.io.loadmat`).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import scipy.io as sio
except ImportError:
    print(
        "scipy not installed — `pip install scipy`. This generator is "
        "human-supervised only; the Rust EeglabReader only needs the "
        "JSON sidecar, not scipy.",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("set_file", type=Path)
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()
    if not args.set_file.exists():
        print(f"no such file: {args.set_file}", file=sys.stderr)
        return 1
    raw = sio.loadmat(str(args.set_file), squeeze_me=True, struct_as_record=False)
    if "EEG" not in raw:
        print(f"{args.set_file}: no `EEG` struct (is this an EEGLAB .set?)",
              file=sys.stderr)
        return 2
    eeg = raw["EEG"]
    n_channels = int(eeg.nbchan)
    n_samples = int(eeg.pnts)
    sample_rate = float(eeg.srate)
    # chanlocs is a struct array; the labels field contains strings.
    channels = []
    if hasattr(eeg, "chanlocs"):
        for chan in eeg.chanlocs:
            label = getattr(chan, "labels", None)
            if label is None:
                label = ""
            channels.append(str(label))
    if len(channels) != n_channels:
        channels = [f"ch{i}" for i in range(n_channels)]

    sidecar = {
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "channels": channels,
        "phys_dim": "uV",
    }
    out_path = args.output or args.set_file.with_suffix(".lml-meta.json")
    out_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out_path}")
    print(
        f"  n_channels={n_channels}  n_samples={n_samples}  "
        f"sample_rate={sample_rate} Hz"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
