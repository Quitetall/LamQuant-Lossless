# lmq-common

Shared primitives for the LamQuant codec family — used by the lossless codec
(`lamquant-lml`) and the neural codec (`lmq-codec`).

`no_std` + `alloc` by default; the `host` feature adds std-only helpers.

## Contents

- **crc32** — CRC-32 (ISO 3309), the integrity check used across LML/LMA.
- **EDF/biosignal reader primitives** + **LMA archive** helpers (host feature).
- **DSP traits** common to the lossless and neural pipelines.

Library name: `lamquant_common`.

## License

AGPL-3.0-or-later; commercial license available — see the repository.
