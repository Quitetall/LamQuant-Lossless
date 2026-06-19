# lamquant-common

Shared primitives for the LamQuant codec family. `no_std` + `alloc` by default; the `host` feature adds `std`-dependent helpers.

## Contents

| Module | Feature | Description |
|--------|---------|-------------|
| `crc32` | *(always)* | CRC-32 ISO 3309 — per-entry integrity field in LML and LMA wire formats |
| `paths` | `host` | Canonical path construction for LML sidecars, LMA archives, and the CLI output tree |
| `ingest` | `host` | EDF main-header + signal-header parser; normalises per-signal `samples_per_record` to a flat `i64` channel matrix. Shared entry point for `lml encode`, the Python wheel, and the firmware bootloader stub |

The `no_std` constraint on the core module is load-bearing: `crc32` and the LML header structs must compile for `riscv32imac-unknown-none-elf` without a system allocator.

Library name: `lamquant_common`.

## License

AGPL-3.0-or-later. Commercial license available — see the repository.
