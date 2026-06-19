#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_sort_by
)]
//! LamQuant Core — complete lossless EEG codec in Rust.
//!
//! Build modes:
//!   default ("host")  — full library: codec + container + EDF + LMA + CLI/TUI deps
//!   "std"             — codec only, host platform (file I/O, std types)
//!   no features       — `no_std` + `alloc`. Codec only. For RP2350 firmware.
//!   "python"          — PyO3 bindings (implies "host")
//!   "ffi"             — C FFI bindings (implies "host")
//!   "wasm"            — WebAssembly bindings (implies "std")
//!
//! Core (no_std + alloc) modules:
//!   crc32   — CRC-32 ISO 3309
//!   lifting — Le Gall 5/3 integer DWT
//!   lpc     — LPC analysis/synthesis + bias cancellation
//!   golomb  — Golomb-Rice entropy coding
//!   rans    — rANS entropy coding
//!   error   — error types (gated std::io impls)
//!   lml     — LML packet compress/decompress
//!
//! Host-only modules (require `host` feature):
//!   container — file I/O, LML v1 container
//!   edf       — EDF file reader
//!   lma       — LMA archive format
//!   stream    — parallel streaming I/O
//!   tui       — interactive terminal UI

extern crate alloc;

// ─── Core (always available, no_std + alloc) ──────────────────────

pub mod backend;
pub mod bit_pack;
pub mod codec_errors;
// Re-exported from lamquant-common during the 8-repo decomposition (Phase 2).
// Keeps the public `lamquant_core::crc32` path stable for firmware + lsl + lmafs.
pub use lamquant_common::crc32;
// ADR 0023 Track B5+: arithmetic coding is opt-in via the
// `experimental_arithmetic` Cargo feature. Default-builds skip
// the module entirely, so the constriction dep adds no surface
// to the lossless codec hot path or the firmware binary.
#[cfg(feature = "experimental_arithmetic")]
pub mod arithmetic;
// ADR 0051 track 2 P3.5: empirical-categorical range coder (the real lossy
// entropy gap-closer). Same opt-in feature as `arithmetic` (shares the
// constriction dep); firmware fails closed on its payload tag.
#[cfg(feature = "experimental_arithmetic")]
pub mod arith_cat;
pub mod deployment;
pub mod error;
pub mod golomb;
pub mod lifting;
pub mod lml;
pub mod lmqc;
pub mod lpc;
pub mod quant;
pub mod rans;
pub mod zrle;

// ─── archive: file I/O + LMA/EDF/container r/w ───────────────────
// Enabled by: archive, cli, tui, security, host (all superset features).

#[cfg(feature = "async")]
pub mod async_io;
#[cfg(feature = "archive")]
pub mod codec_stages;
#[cfg(feature = "archive")]
pub mod container;
#[cfg(feature = "archive")]
pub mod edf;
// Re-exported from lamquant-common during the 8-repo decomposition (Phase 2).
// Keeps `lamquant_core::ingest::*` callable from existing code (lma.rs,
// container, downstream tests).
#[cfg(feature = "archive")]
pub use lamquant_common::ingest;
#[cfg(feature = "archive")]
pub mod io;
#[cfg(feature = "archive")]
pub mod lma;
#[cfg(feature = "archive")]
pub mod offset_table;
// Re-exported from lamquant-common during the 8-repo decomposition (Phase 2).
#[cfg(feature = "archive")]
pub use lamquant_common::paths;
#[cfg(feature = "archive")]
pub mod pipeline;
#[cfg(feature = "archive")]
pub mod range;
#[cfg(feature = "archive")]
pub mod source;
#[cfg(feature = "archive")]
pub mod stream;

// ─── security: encryption / signing primitives ────────────────────
#[cfg(feature = "security")]
pub mod security;

// ─── tui: interactive TUI panels ──────────────────────────────────
#[cfg(feature = "tui")]
pub mod tui;
#[cfg(feature = "tui")]
pub mod tui_experimental;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "wasm")]
pub mod wasm;

