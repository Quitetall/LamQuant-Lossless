//! Compute backend selector — `ComputeBackend` enum (ADR 0058: lives in the
//! Desktop tier, since it *chooses between* the MCU scalar path and the Desktop
//! parallel path. Firmware never selects a backend — it just runs scalar — so
//! this is a host concern.)
//!
//! Same wire format, different machine. Every backend produces byte-identical
//! `.lml` output for the same input; the only difference is HOW it computes.
//! The invariant is enforced by `tests/byte_equal_backends.rs` in this crate.
//!
//! Variants:
//!   * `Firmware` — the reference scalar path (`lamquant_lml_mcu::lml::
//!     compress_with_mode` / `decompress`). The MCU build uses it directly
//!     (without this selector); here it is the byte-equality baseline.
//!   * `Desktop` — rayon per-channel parallelism (+ future SIMD), byte-identical
//!     to `Firmware` by construction (see [`crate::parallel`]).

use core::sync::atomic::{AtomicU8, Ordering};

use lamquant_lml_mcu::error::LmlResult;
use lamquant_lml_mcu::lml;
use lamquant_lml_mcu::lpc::LpcMode;

use alloc::vec::Vec;

/// Which compute backend to dispatch through. `default()` is `Desktop` (the perf
/// path) — this crate is the host fast tier. Output is byte-identical across
/// variants; `tests/byte_equal_backends.rs` locks the invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// Reference scalar Rust path (the MCU tier's codec). Byte-equality baseline.
    Firmware,
    /// Rayon per-channel parallelism (+ future SIMD). Same wire-format output as
    /// `Firmware` by contract. The default on this host fast tier.
    #[default]
    Desktop,
}

// ─── Process-wide backend selector ────────────────────────────────
// Set once at startup by CLI / TUI / library callers; read on every encode.
// Encoded as u8: 0 = unset (use Default), 1 = Firmware, 2 = Desktop. `Relaxed`
// is sufficient — backend choice is sticky configuration, not racing data.

const BACKEND_UNSET: u8 = 0;
const BACKEND_FIRMWARE: u8 = 1;
const BACKEND_DESKTOP: u8 = 2;

static GLOBAL_BACKEND: AtomicU8 = AtomicU8::new(BACKEND_UNSET);

/// Set the process-wide compute backend. Call once at startup (CLI argv parse,
/// TUI settings panel, library init).
pub fn set_global_backend(backend: ComputeBackend) {
    let v = match backend {
        ComputeBackend::Firmware => BACKEND_FIRMWARE,
        ComputeBackend::Desktop => BACKEND_DESKTOP,
    };
    GLOBAL_BACKEND.store(v, Ordering::Relaxed);
}

/// Read the process-wide compute backend. If unset, returns `default()`.
pub fn global_backend() -> ComputeBackend {
    match GLOBAL_BACKEND.load(Ordering::Relaxed) {
        BACKEND_FIRMWARE => ComputeBackend::Firmware,
        BACKEND_DESKTOP => ComputeBackend::Desktop,
        _ => ComputeBackend::default(),
    }
}

impl ComputeBackend {
    /// Parse from CLI string. Returns `Err` for unknown names.
    pub fn parse(s: &str) -> Result<Self, &'static str> {
        match s {
            "firmware" => Ok(ComputeBackend::Firmware),
            "desktop" => Ok(ComputeBackend::Desktop),
            _ => Err("backend must be `firmware` or `desktop`"),
        }
    }

    /// Human-readable name, matches CLI parse value.
    pub fn name(&self) -> &'static str {
        match self {
            ComputeBackend::Firmware => "firmware",
            ComputeBackend::Desktop => "desktop",
        }
    }
}

/// Compress through the selected backend. Byte-identical output for every
/// variant — the invariant the conformance gate locks.
pub fn compress_with_backend(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: LpcMode,
    backend: ComputeBackend,
) -> LmlResult<Vec<u8>> {
    match backend {
        ComputeBackend::Firmware => lml::compress_with_mode(signal, noise_bits, mode),
        ComputeBackend::Desktop => {
            crate::parallel::compress_with_mode_parallel(signal, noise_bits, mode)
        }
    }
}

/// Decompress through the selected backend. Byte-identical signal output across
/// variants.
pub fn decompress_with_backend(data: &[u8], backend: ComputeBackend) -> LmlResult<Vec<Vec<i64>>> {
    match backend {
        ComputeBackend::Firmware => lml::decompress(data),
        ComputeBackend::Desktop => crate::parallel::decompress_parallel(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_desktop() {
        assert_eq!(ComputeBackend::default(), ComputeBackend::Desktop);
    }

    #[test]
    fn global_backend_roundtrip() {
        set_global_backend(ComputeBackend::Firmware);
        assert_eq!(global_backend(), ComputeBackend::Firmware);
        set_global_backend(ComputeBackend::Desktop);
        assert_eq!(global_backend(), ComputeBackend::Desktop);
        GLOBAL_BACKEND.store(BACKEND_UNSET, Ordering::Relaxed);
        assert_eq!(global_backend(), ComputeBackend::default());
    }

    #[test]
    fn parse_roundtrip() {
        for b in [ComputeBackend::Firmware, ComputeBackend::Desktop] {
            assert_eq!(ComputeBackend::parse(b.name()), Ok(b));
        }
        assert!(ComputeBackend::parse("avx2").is_err());
        assert!(ComputeBackend::parse("").is_err());
    }
}
