#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_sort_by
)]
//! LamQuant LML — **MCU tier** (ADR 0058): the no_std integer lossless floor +
//! the shared `codec` seam. Firmware links ONLY this crate; the Desktop and
//! Optimum tiers depend on it (they reuse the LML codec + the `Codec` seam). The
//! umbrella crate `lamquant-lml` (lib name `lamquant_core`) re-exports every
//! module here, so existing `use lamquant_core::{lml, lpc, ...}` call sites are
//! unchanged.
//!
//! Build modes:
//!   no features  — `no_std` + `alloc`. Integer codec only. RP2350 firmware.
//!   "std"        — adds the f64 host-side encode helpers (RD search, Anytime
//!                  LPC, blockfloat bench, std::io sinks).
//!   "archive"    — adds the rayon parallel encode/decode (Desktop backend).
//!
//! Modules:
//!   crc32   — CRC-32 ISO 3309 (re-exported from lamquant-common)
//!   lifting — Le Gall 5/3 integer DWT
//!   lpc     — LPC analysis/synthesis + bias cancellation
//!   golomb  — Golomb-Rice entropy coding
//!   rans    — rANS entropy coding
//!   error   — error types (gated std::io impls)
//!   lml     — LML packet compress/decompress (the wire format + reference decoder)

extern crate alloc;

pub mod bit_pack;
// The shared codec seam (ADR 0052): Format/Mode/Codec trait + universal
// magic-dispatch decode. The LML half lives here; LMO is in `lamquant-optimum`.
pub mod codec;
pub mod codec_errors;
// Re-exported from lamquant-common (8-repo decomposition, Phase 2). Keeps the
// public `crc32` path stable for firmware + lsl + lmafs (via the facade).
pub use lamquant_common::crc32;
// ADR 0023 Track B5+: arithmetic coding is opt-in via the
// `experimental_arithmetic` Cargo feature. Default builds skip the module
// entirely, so the constriction dep adds no surface to the firmware binary.
#[cfg(feature = "experimental_arithmetic")]
pub mod arithmetic;
// ADR 0051 track 2 P3.5: empirical-categorical range coder (the real lossy
// entropy gap-closer). Same opt-in feature as `arithmetic`.
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

/// Host-only parallel execution profile. Packet orchestration remains in this
/// codec owner; firmware builds exclude the module and its Rayon dependency.
#[cfg(feature = "parallel")]
pub mod parallel;
