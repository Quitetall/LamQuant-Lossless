# Export

> Cross-format export to scientific containers. CSV, NPY, MATLAB,
> Parquet, HDF5, BIDS, raw int32 stdout. For exporting back to the
> source-format byte-equal (EDF / BrainVision / CNT / EEGLAB / DICOM),
> use `lml extract` — see [Decompression](./02-decompression.md).

Export is one-way: LML → scientific container. It's lossy in the
sense that the resulting file doesn't decode back to a `.lml` —
it's the right format for downstream analysis (MNE, EEGLAB, scipy,
pandas, BIDS pipelines), not for re-encoding.

For round-trippable bit-exact recovery of the original file (EDF,
EEGLAB `.set + .fdt`, BrainVision tri-file, DICOM `.dcm`, etc.),
extract from the LMA archive directly.

## At a glance

| Feature | Format / flag | Status | First shipped | Notes |
|---|---|---|---|---|
| CSV | `lml export -f csv` | shipped | v1.0 (5.1) | Channel-major; default format |
| NumPy `.npy` | `lml export -f npy` | shipped | v1.0 (5.2) | int64 array |
| Raw int32 LE | `lml decode -o -` | shipped | v1.0 (5.3) | Stdout pipeable (R.3) |
| Parquet | `lml export -f parquet` | shipped (feature-gated) | v1.0 (5.4) | `--features parquet`; Snappy int64 column per channel |
| MATLAB `.mat` v5 | `lml export -f mat` | shipped | v1.0 (5.5) | Hand-rolled writer (no matio dep); scipy.io.loadmat verified |
| HDF5 | `lml export -f hdf5` | shipped (feature-gated) | v1.0 (5.6) | `--features hdf5`; `/signal` 2-D int64 via hdf5-metno 0.12 |
| BIDS-EEG layout | `lml export -f bids` | shipped | v1.0 (5.7) | `dataset_description.json` + `sub-01/eeg/...` |
| Streaming-conversion sweep | (automatic) | shipped | v1.0 (5.8) | raw streams via `decode_one_to_raw`; csv/npy/mat/bids small-payload load |
| DICOM Waveform output | (planned) | not shipped | — | Round-trip via `lml extract` works today; explicit write reverse path deferred |

## Commands

### `lml export`

Export an LML's decoded signal to a scientific container format.

Synopsis:
```
lml export [-f <FORMAT>] <INPUT> [-o <OUTPUT>]
```

Format choices: `csv` (default), `npy`, `raw`, `mat`, `bids`,
`parquet` (feature-gated), `hdf5` (feature-gated).

Examples:
```
# CSV (default)
lml export recording.lml -o samples.csv

# NumPy .npy
lml export recording.lml -o samples.npy -f npy

# MATLAB .mat v5 — scipy.io.loadmat reads this
lml export recording.lml -o samples.mat -f mat

# BIDS-EEG skeleton directory
lml export recording.lml -o bids-out/ -f bids

# Parquet (requires --features parquet at build time)
lml export recording.lml -o samples.parquet -f parquet

# HDF5 (requires --features hdf5 at build time)
lml export recording.lml -o samples.h5 -f hdf5
```

For raw int32 LE to stdout, use `lml decode -o -` (see
[Decompression](./02-decompression.md)). `lml export -f raw` is
equivalent.

## Format-by-format notes

### CSV

Channel-major rows, header line lists channel names. Easy for ad-hoc
inspection with `awk` / `cut` / spreadsheet imports. Slow for big
recordings (text encoding overhead).

### NumPy `.npy`

Standard `.npy` v3 file, dtype `int64`, shape `(n_channels, n_samples)`.
Loads directly via `numpy.load(...)`. Pair with `lml info` to fetch
metadata (sample rate, channel names) from the LML JSON metadata
blob.

### Raw int32 LE

Channel-major int32 little-endian, no header. Pipes cleanly into
NumPy (`np.fromfile(..., dtype=np.int32).reshape(n_chan, -1)`),
matplotlib, GNU Octave, custom C / Rust pipelines. The only export
format with no metadata sidecar — caller is responsible for tracking
channel count and sample rate.

### MATLAB `.mat` v5

Hand-rolled MATLAB v5 writer in pure Rust (no `matio` C dep). Output
is a struct with fields:

- `signal` — int64 array, shape `(n_channels, n_samples)`
- `sample_rate` — double scalar
- `channel_names` — cell array of strings

scipy.io.loadmat reads this without modification. MATLAB R2017+ also
reads it.

### BIDS-EEG directory

`-f bids` emits a BIDS-Brain-Imaging-Data-Structure skeleton:

```
bids-out/
├── dataset_description.json
├── participants.tsv
└── sub-01/
    └── eeg/
        ├── sub-01_task-rest_eeg.csv (or .edf)
        ├── sub-01_task-rest_eeg.json
        └── sub-01_task-rest_channels.tsv
```

The skeleton is minimal — operator may need to fill in dataset-level
metadata (`DatasetDOI`, `Authors`, etc.) before downstream BIDS
validators accept it.

### Parquet (`--features parquet`)

One int64 column per channel, Snappy compressed, written via
`parquet::ArrowWriter`. Loads via pyarrow, polars, DuckDB. Footer
preserves per-channel name as the column name. Pair with `lml info`
or BIDS sidecar for sample-rate metadata.

Built only when `--features parquet` is on. Default builds omit the
`parquet` Cargo dependency.

### HDF5 (`--features hdf5`)

`/signal` dataset = 2-D int64, shape `(n_channels, n_samples)`. Via
`hdf5-metno 0.12` (forked from `hdf5` to skip the Windows-toolchain
issues in the upstream crate). h5py reads it round-trip.

Built only when `--features hdf5` is on. The dependency requires
either system libhdf5 or `--features hdf5/static` for a vendored
build.

## Streaming-conversion sweep

The `raw` arm decodes window-by-window via
`decode_one_to_raw`, never materialising the whole signal in memory.
`csv` / `npy` / `mat` / `bids` load small payloads in one shot — fine
for clinical recordings (~10 GB max), would need a streaming refactor
for hour-long high-rate ECoG.

For pipeline composition with arbitrarily large streams, prefer:

```sh
# Raw int32 stream → downstream tool
lml decode big.lml -o - | downstream_tool

# vs the materialised version
lml export big.lml -o big.csv -f csv   # blocks until done
```

## Cross-format round-trip vs export

`lml extract recording.lma` gets you the original file (`.edf`,
`.set + .fdt`, `.vhdr + .vmrk + .eeg`, `.cnt`, `.dcm`, `.raw + .json`)
back byte-identical. This is the **right path** for "I want my source
file back".

`lml export -f X` decodes the signal and writes it in format X.
**It does not write back to the original source format.** Loop-back
encode → export → re-encode won't reproduce the original file.

| If you want... | Use |
|---|---|
| The original `.edf` byte-identical | `lml extract` (any LMA archive) |
| The original `.set + .fdt` byte-identical | `lml extract` |
| The decoded signal as CSV / NPY / MAT / BIDS / etc. | `lml export -f X` |
| Raw int32 piped into another tool | `lml decode -o -` |

## Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `-f`, `--format <FMT>` | enum | `csv` | `csv` / `npy` / `raw` / `mat` / `bids` / `parquet` / `hdf5` |
| `-o`, `--output <PATH>` | path | (derived) | Output file or directory (for `bids`) |

## Error cases

| Trigger | Error |
|---|---|
| `-f parquet` on a binary built without `--features parquet` | "parquet format not compiled in (build with --features parquet)" |
| `-f hdf5` on a binary built without `--features hdf5` | "hdf5 format not compiled in (build with --features hdf5)" |
| `-f bids` with `-o` pointing at a file (not a directory) | refuse — BIDS requires a directory layout |
| Unknown format | "unknown format: <name>" |

## Related

- **Other buckets**:
  - [Decompression](./02-decompression.md) — `lml decode` + `lml extract` (the byte-equal source recovery path)
  - [Compression](./01-compression.md) — supported source formats are also export targets via `lml extract`
  - [Build / Release](./10-build-release.md) — `--features parquet` / `--features hdf5` build matrix
- **Source files**:
  - `lamquant-core/src/bin/lml.rs:392` — `Export` subcommand
  - `lamquant-core/src/lml.rs` — decode helpers reused by every export arm
- **Tests**:
  - `tests/integration/test_export_csv.py`
  - `tests/integration/test_export_npy.py`
  - `tests/integration/test_export_mat.py`
  - `tests/integration/test_export_bids.py`
  - `tests/integration/test_export_parquet.py` (feature-gated)
  - `tests/integration/test_export_hdf5.py` (feature-gated)
- **Commits**:
  - `b4ed11a` — MAT v5 + BIDS (5.5 + 5.7)
  - Phase 5.4 — Parquet (feature-gated)
  - Phase 5.6 — HDF5 (feature-gated)
  - Phase 5.8 — streaming-conversion sweep
- **Cross-cutting docs**:
  - [`../FEATURES.md`](../FEATURES.md) §4 (export targets)
  - [`../FAQ.md`](../FAQ.md) — feature-gated build common questions
