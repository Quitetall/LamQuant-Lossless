# lamquant-lml-h5filter

LamQuant **LML as a registered HDF5 (H5Z) compression filter** — so stock
HDF5/NWB tooling stores integer datasets through the royalty-free LML lossless
codec, and the file stays a fully native HDF5/NWB.

This is ADR 0051 Track 3, mechanism A: *own the NWB flank H.BWC structurally
cannot reach.* NWB (HDF5 + schema) is the AI/BCI/iEEG research container; a
permissive codec can live **inside** it. Unlike a structural transcoder, the
filter preserves 100% of HDF5 structure (groups, attributes, compound electrode
tables, object references) — only the integer dataset *chunks* are recoded.

Filter ID: **32200** (placeholder, pending registration with The HDF Group).

## Build

```sh
cargo build -p lamquant-lml-h5filter --release
# → target/release/liblamquant_lml_h5filter.so
```

The `.so` carries **no `DT_NEEDED` on libhdf5** — HDF5 symbols resolve from the
process that loads it, so it works under whatever libhdf5 the host provides.

## Use it (one command — `lml nwb`)

Build `lml` with `--features nwb` and let it drive everything (locates this
plugin, runs the repack, verifies losslessness via the LML reader):

```sh
lml nwb pack    recording.nwb -o recording.lml.nwb   # shrink (self-verifies)
lml nwb unpack  recording.lml.nwb -o plain.nwb        # restore plain NWB
# --plugin-so <path> or $LAMQUANT_H5FILTER if the .so isn't auto-found.
```

`pack` prints the ratio and confirms every integer dataset round-trips
identically before exiting. Example: a 3000×16 int32 ElectricalSeries packs
4.9× and verifies lossless.

## Use it (system HDF5 tools — the underlying path)

```sh
export HDF5_PLUGIN_PATH=/path/to/target/release

# Losslessly shrink an existing NWB/HDF5 in place (structure fully preserved):
h5repack -f UD=32200,0,0  recording.nwb  recording.lml.nwb

# Confirm the filter + ratio, and that decode is byte-exact:
h5dump -pH recording.lml.nwb | grep FILTER_ID          # → FILTER_ID 32200
h5repack -f NONE recording.lml.nwb decoded.nwb          # decode anywhere
```

Measured on a 2000×8 int16 ElectricalSeries: **2.296:1** dataset compression,
byte-identical round-trip (`scripts/validate_h5repack.sh`).

## Scope / safety

- Applies **only to little-endian integer datasets** (1/2/4/8 byte). Float,
  compound, and big-endian data are left untouched (`can_apply` refuses them) —
  never routed through (or corrupted by) LML.
- Each chunk is self-describing (20-byte header), so decode never depends on the
  HDF5 `cd_values` surviving; if LML doesn't shrink a chunk it is stored raw
  (never expands past the input).
- Host-only. Never part of the no_std firmware floor.

## Caveat: pip `h5py`

The manylinux **pip `h5py` wheel bundles its own libhdf5 with a mangled soname
and local symbol visibility** (auditwheel), so a dlopened filter plugin cannot
resolve HDF5 symbols against it — HDF5 reports the filter "not registered".
This affects pip-`h5py` reads/writes through the plugin, **not** the filter
itself. Use one of:

- **System or conda `h5py`** (links a normally-visible libhdf5) — works.
- The **system HDF5 CLI** (`h5repack`/`h5dump`) — works (validated above).
- `LD_PRELOAD=/path/libhdf5.so` before launching pip `h5py`.

The filter logic is proven independently of this: `tests/roundtrip.rs` drives it
through real libhdf5 in-process, and `tests/`/`src` include an `H5Z_class2_t`
layout cross-check against the `-sys` bindings.
