# lamquant-common

Shared primitives for the LamQuant codec family — used by the lossless codec
(`lamquant-lml`) and the neural codec (`lamquant-lmq`).

`no_std` + `alloc` by default; the `host` feature adds std-only helpers.

## Contents

- **`crc32`** — CRC-32 (ISO 3309), the integrity check used across the
  LamQuant LML/LMA formats. `no_std`.
- **`paths`** *(host)* — path utilities for the CLI tooling.
- **`ingest`** *(host)* — non-EDF signal ingest (ASCII integer lines →
  synthesized EDF wrappers) so every codec consumes one canonical format.

The EDF reader, LMA archive, and codec DSP live in the codec crates
(`lamquant-lml` etc.), not here — this crate is the minimal shared base.

Library name: `lamquant_common`.

## License

AGPL-3.0-or-later; commercial license available — see the repository.
