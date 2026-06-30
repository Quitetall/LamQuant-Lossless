#!/usr/bin/env bash
# Cross-tool validation of the LML H5Z filter via the system HDF5 CLI
# (ADR 0051 Track 3, mechanism A). Proves the deployable use case:
#   h5repack -f UD=32200 in.nwb out.nwb   →  lossless, smaller, native HDF5.
#
# Requires: system h5repack + h5dump (libhdf5 >= the version the .so was built
# against) and python3 + h5py + numpy to author the fixture and compare.
# Skips (exit 0) if those tools are absent so CI without them stays green.
set -euo pipefail

SO_DIR="${1:-$(cd "$(dirname "$0")/../../target/release" && pwd)}"
SO="$SO_DIR/liblamquant_lml_h5filter.so"
FID=32200

command -v h5repack >/dev/null 2>&1 || { echo "SKIP: no system h5repack"; exit 0; }
python3 -c 'import h5py, numpy' 2>/dev/null  || { echo "SKIP: no python3+h5py+numpy"; exit 0; }
[ -f "$SO" ] || { echo "ERROR: build first: cargo build -p lamquant-lml-h5filter --release"; exit 2; }

export HDF5_PLUGIN_PATH="$SO_DIR"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

python3 - <<'PY'
import numpy as np, h5py
T, C = 2000, 8
data = ((np.sin(np.arange(T)*0.3)[:,None]*1500).astype('<i2') + (np.arange(C)*11).astype('<i2'))
with h5py.File("in.h5","w") as f:
    es = f.create_group("acquisition").create_group("ElectricalSeries")
    es.create_dataset("data", data=np.ascontiguousarray(data), chunks=(500, C))
PY

h5repack -f UD=$FID,0,0 in.h5 out.h5            # encode through LML
h5repack -f NONE        out.h5 roundtrip.h5     # decode (strip filter) via system libhdf5

h5dump -pH out.h5 | grep -q "FILTER_ID $FID" || { echo "FAIL: filter $FID not applied"; exit 1; }

python3 - <<'PY'
import numpy as np, h5py, os, sys
a = h5py.File("in.h5","r")["acquisition/ElectricalSeries/data"][...]
b = h5py.File("roundtrip.h5","r")["acquisition/ElectricalSeries/data"][...]
si, so = os.path.getsize("in.h5"), os.path.getsize("out.h5")
ok = np.array_equal(a, b)
print(f"lossless={ok}  in={si}B  filtered={so}B  ratio={si/so:.3f}x")
sys.exit(0 if (ok and so < si) else 1)
PY
echo "PASS: h5repack LML round-trip lossless + smaller"
