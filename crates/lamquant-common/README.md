# lamquant-common

Shared primitives for the LamQuant codec family. `no_std` + `alloc` by default; `host` feature adds std-only helpers.

- **`crc32`** — CRC-32 ISO 3309; integrity check used in LML/LMA wire formats
- **`paths`** *(host)* — path utilities for the CLI
- **`ingest`** *(host)* — ASCII integer → EDF wrapper so every codec consumes one format

Library name: `lamquant_common`.

## License

AGPL-3.0-or-later. Commercial license available — see the repository.
