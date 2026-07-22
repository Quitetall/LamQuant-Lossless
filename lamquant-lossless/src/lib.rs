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

// ─── Core codec (ADR 0052 Tier 1) ─────────────────────────────────
// The no_std integer LML codec now lives in `lamquant-lml-mcu`. It is
// re-exported here so the historical `lamquant_core::{lml, lpc, golomb, ...}`
// paths stay byte-for-byte stable for firmware, lamquant-lsl, the Python
// extension, and every test — none of those call sites change.
pub use lamquant_lml_mcu::{
    bit_pack, codec, codec_errors, crc32, deployment, error, golomb, lifting, lml, lmqc, lpc,
    quant, rans, zrle,
};
// ADR 0023 Track B5+ / ADR 0051 P3.5: arithmetic + empirical-categorical range
// coders are opt-in via `experimental_arithmetic` (re-exported from core).
#[cfg(feature = "experimental_arithmetic")]
pub use lamquant_lml_mcu::{arith_cat, arithmetic};

// ─── Tier crates (ADR 0058) ───────────────────────────────────────
// This umbrella IS the Desktop assembly: it links all three tiers. The MCU tier
// (`lamquant-lml-mcu`) is re-exported module-by-module above (the stable
// `lamquant_core::{lml,lpc,...}` surface). The Desktop tier (host fast path) is
// re-exported as `lamquant_core::desktop` under `archive` (it is std/rayon, so
// the no_std facade build omits it).
#[cfg(feature = "archive")]
pub use lamquant_lml_desktop as desktop;

// ADR 0058 carve-full: the `ComputeBackend` selector + the rayon parallel
// encode/decode now live in the Desktop tier. Re-exported at the stable
// `lamquant_core::backend` path (used by `container`, the `lml` CLI's
// `--backend` flag, …) and the parallel entry points the container hot path
// calls. Firmware (no `archive`) never selects a backend — it runs scalar.
#[cfg(feature = "archive")]
pub use lamquant_lml_desktop::backend;
#[cfg(feature = "archive")]
pub use lamquant_lml_desktop::{
    compress_with_mode_parallel, compress_with_mode_parallel_views, decompress_parallel,
};

// The Optimum (LMO) tier. Re-exported as `lamquant_core::optimum`; it ships the
// LMO decoder always and the encoder under `archive` (which needs the MCU tier's
// RD search), matching where the host codec capabilities already live.
pub use lamquant_lml_optimum as optimum;
/// Independent Optimum-v2 learned-lossless peer (`LMO1` v3 / `BGF1`). Host-only;
/// the firmware/no-default-features graph remains unchanged.
#[cfg(feature = "archive")]
pub use lamquant_lml_optimum_v2 as optimum_v2;

/// Universal magic-dispatch decode (the Desktop full dispatch): routes an LML
/// stream to the integer floor and an LMO stream to the Optimum decoder. Unlike
/// [`codec::decode`] (the Firmware/core view, which returns
/// `OptimumNotInstalled` for LMO), this resolves both formats because the
/// facade always links the LMO decoder.
pub fn decode(bytes: &[u8]) -> Result<codec::Signal, codec::CodecError> {
    #[cfg(feature = "archive")]
    if bytes.get(..4) == Some(&b"LMO1"[..]) && bytes.get(4) == Some(&3) {
        return optimum_v2::OptimumV2Codec
            .decode_window(bytes)
            .map(|window| window.samples)
            .map_err(|error| codec::CodecError::Backend(error.to_string()));
    }
    optimum::decode_any(bytes)
}

// ─── archive: file I/O + LMA/EDF/container r/w ───────────────────
// Enabled by: archive, cli, tui, security, host (all superset features).

#[cfg(feature = "async")]
pub mod async_io;
// ADR 0139/0143: current LML archives are semantic ABIR BCS2 codec bundles.
// Retired containers are reachable only through the supervised legacy Adapter.
#[cfg(feature = "archive")]
pub mod bcs2_container;
// ADR 0069 S7b: the LMQ training normalization pipeline (channel-select →
// resample → 0.5 Hz zero-phase HP → Q31), hoisted from Python. Host-only Lossy
// DSP — see module docs (non-causal filtfilt ⇒ no MCU variant).
#[cfg(feature = "archive")]
pub mod codec_stages;
#[cfg(feature = "archive")]
pub mod normalize;
/// ADR 0075 — the LQTP1 training tensor pack (BFP window store). Host-only.
pub mod tensor_pack;
#[cfg(feature = "archive")]
pub use crate::bcs2_container as container;
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
// NWB/HDF5 integer-signal reader → LML ingest (ADR 0051 Track 3). Host-only:
// the `nwb` feature gates in libhdf5 via hdf5-metno; never in the no_std build.
#[cfg(feature = "nwb")]
pub mod nwb;
// Textual biosignal-IR form (ADR 0069) — the golden/debug serialization of the
// SignalBundle IR. Needs the bundle (archive feature).
#[cfg(feature = "archive")]
pub mod ir;
// Re-exported from lamquant-common during the 8-repo decomposition (Phase 2).
#[cfg(feature = "archive")]
pub use lamquant_common::paths;
#[cfg(feature = "archive")]
pub mod pipeline;
// ADR 0069 Pillar 3 / S3c: the Reversible/Lossy `Pass` framework built on
// top of `pipeline::Stage`. Same gate as `pipeline` — a `Pass` IS a
// `Stage`, so it needs everything `archive` already turns on.
#[cfg(feature = "archive")]
pub mod pass;
// ADR 0069 Pillar 3 / S5 Increment 2 (task #20): the textual pass-pipeline
// DSL built on top of `pass` — parses a pipeline-text config into a
// `PipelineSpec` and resolves it against a `PassRegistry`, reusing
// `run_in_lml`/`DynPass` UNCHANGED (including the runtime Lossy refusal).
#[cfg(feature = "archive")]
pub mod pipeline_dsl;
// `source::descriptor` (ADR 0069 Pillar 3 / S5 Increment 3, task #20) is
// declared inside `source/mod.rs`, not here — it's a submodule of
// `source` (file lives at `src/source/descriptor.rs`), gated by this
// same `archive` feature via the parent `pub mod source;` below. See
// `source::descriptor`'s module docs for the `FormatDescriptor` schema
// and the G5 gotchas (F32 refusal, first-class endian, reciprocal
// sample-rate transform).
#[cfg(feature = "archive")]
pub mod source;

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
