//! Compute backend selector — `ComputeBackend` enum.
//!
//! Same wire format, different machine. Every backend produces
//! byte-identical `.lml` output for the same input; the only
//! difference is HOW it computes. This invariant is enforced by
//! `lamquant-core/tests/byte_equal_backends.rs`.
//!
//! Variants:
//!   * `Firmware` — the reference scalar Rust implementation. Used
//!     by the MCU build (Cortex-M, no SIMD, no_std + alloc) AND by
//!     the host conformance baseline. Always available.
//!   * `Desktop` — host-only path. Rayon per-channel parallelism +
//!     (future) AVX2/NEON intrinsics. Same output as `Firmware` by
//!     contract; faster on x86_64 / aarch64 / any multi-core host
//!     (workstation, laptop, server, basestation).
//!
//! The split lets us land the byte-equality test gate FIRST (TDD —
//! tests pass because both variants produce identical bytes by
//! construction), then add SIMD / parallelism inside `Desktop`
//! later. The gate catches any drift instantly.
//!
//! Firmware builds (no `host` feature) only see `Firmware`. The CLI
//! refuses `Desktop` on architectures where the host feature isn't
//! compiled in.

use crate::error::LmlResult;
use crate::lml;
use crate::lpc::LpcMode;

// `Vec` is only in `core::alloc::vec::Vec` (no_std + alloc) when std
// isn't pulled in. lib.rs declares `extern crate alloc;`, but each
// module that uses Vec under no_std has to import it explicitly.
use alloc::vec::Vec;

/// Which compute backend to dispatch through.
///
/// `default()` returns `Desktop` on host builds (CPU has rayon +
/// AVX2 — the perf path), `Firmware` on MCU builds (no_std + alloc,
/// scalar only). Output is byte-identical across variants;
/// `tests/byte_equal_backends.rs` locks the invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    /// Reference scalar Rust implementation. Always available — the
    /// MCU firmware build uses this path; the host conformance suite
    /// uses it as the byte-equality baseline.
    Firmware,
    /// Host-only path: rayon over channels + AVX2 autocorr + parallel
    /// decompress. Same wire-format output as `Firmware` by contract.
    /// Targets x86_64 / aarch64 desktops, laptops, servers,
    /// basestations.
    #[cfg(feature = "host")]
    Desktop,
}

impl Default for ComputeBackend {
    /// Host builds default to `Desktop` (the perf path). Firmware
    /// builds (no `host` feature) default to `Firmware`. Callers can
    /// override globally via `set_global_backend()`.
    fn default() -> Self {
        #[cfg(feature = "host")]
        {
            ComputeBackend::Desktop
        }
        #[cfg(not(feature = "host"))]
        {
            ComputeBackend::Firmware
        }
    }
}

// ─── Process-wide backend selector ────────────────────────────────
//
// Set once at startup by CLI / TUI / library callers. Read by
// `container::encode_into` (and any other hot-path entry that wants
// to honour user choice) on every invocation. Process-wide state is
// the cleanest minimal-touch design here — every callsite would
// otherwise need a new `backend` parameter threaded through.
//
// Encoded as u8: 0 = unset (use Default::default()), 1 = Firmware,
// 2 = Desktop. `Relaxed` ordering is sufficient — backend choice
// changes are sticky configuration, not racing data writes.

use core::sync::atomic::{AtomicU8, Ordering};

const BACKEND_UNSET: u8 = 0;
const BACKEND_FIRMWARE: u8 = 1;
#[cfg(feature = "host")]
const BACKEND_DESKTOP: u8 = 2;

static GLOBAL_BACKEND: AtomicU8 = AtomicU8::new(BACKEND_UNSET);

/// Set the process-wide compute backend. Call once at startup
/// (from CLI argv parse, TUI settings panel, library init). Reads
/// after this point see the new value.
pub fn set_global_backend(backend: ComputeBackend) {
    let v = match backend {
        ComputeBackend::Firmware => BACKEND_FIRMWARE,
        #[cfg(feature = "host")]
        ComputeBackend::Desktop => BACKEND_DESKTOP,
    };
    GLOBAL_BACKEND.store(v, Ordering::Relaxed);
}

/// Read the process-wide compute backend. If unset, returns
/// `ComputeBackend::default()` (= Desktop on host, Firmware on
/// firmware build).
pub fn global_backend() -> ComputeBackend {
    match GLOBAL_BACKEND.load(Ordering::Relaxed) {
        BACKEND_FIRMWARE => ComputeBackend::Firmware,
        #[cfg(feature = "host")]
        BACKEND_DESKTOP => ComputeBackend::Desktop,
        _ => ComputeBackend::default(),
    }
}

impl ComputeBackend {
    /// Parse from CLI string. Returns `Err` for unknown names and
    /// for backends not available on this build. The error message
    /// reflects what THIS build actually supports — a `no_std` /
    /// firmware build only lists `firmware`, so the operator isn't
    /// pointed at a variant that doesn't exist.
    pub fn parse(s: &str) -> Result<Self, &'static str> {
        match s {
            "firmware" => Ok(ComputeBackend::Firmware),
            #[cfg(feature = "host")]
            "desktop" => Ok(ComputeBackend::Desktop),
            #[cfg(feature = "host")]
            _ => Err("backend must be `firmware` or `desktop`"),
            #[cfg(not(feature = "host"))]
            _ => Err("backend must be `firmware` (only variant on this build)"),
        }
    }

    /// Human-readable name, matches CLI parse value.
    pub fn name(&self) -> &'static str {
        match self {
            ComputeBackend::Firmware => "firmware",
            #[cfg(feature = "host")]
            ComputeBackend::Desktop => "desktop",
        }
    }
}

/// Compress through the selected backend. Byte-identical output for
/// every variant — that's the invariant the conformance gate locks.
pub fn compress_with_backend(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: LpcMode,
    backend: ComputeBackend,
) -> LmlResult<Vec<u8>> {
    match backend {
        ComputeBackend::Firmware => lml::compress_with_mode(signal, noise_bits, mode),
        #[cfg(feature = "host")]
        ComputeBackend::Desktop => {
            // Host parallel path: rayon over channels. Output is
            // byte-identical to the firmware path; the `byte_equal_
            // backends.rs` conformance gate locks this. SIMD on the
            // LPC autocorr inner loop lands in a follow-up commit
            // and slots in inside `encode_one_channel` -- still
            // byte-equal by construction.
            lml::compress_with_mode_parallel(signal, noise_bits, mode)
        }
    }
}

/// Decompress through the selected backend. Byte-identical signal
/// output across variants.
pub fn decompress_with_backend(
    data: &[u8],
    backend: ComputeBackend,
) -> LmlResult<Vec<Vec<i64>>> {
    match backend {
        ComputeBackend::Firmware => lml::decompress(data),
        #[cfg(feature = "host")]
        ComputeBackend::Desktop => {
            // Host parallel decompress: sequential parse phase
            // (cursor through golomb-encoded payload) + parallel
            // synth + lifting-inverse phase per channel. Same
            // signal output as the firmware serial path, locked by
            // `byte_equal_backends.rs`.
            lml::decompress_parallel(data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_build() {
        // Host builds default to Desktop (perf path), firmware to
        // Firmware (the only variant present). Matches the manual
        // `impl Default`.
        #[cfg(feature = "host")]
        assert_eq!(ComputeBackend::default(), ComputeBackend::Desktop);
        #[cfg(not(feature = "host"))]
        assert_eq!(ComputeBackend::default(), ComputeBackend::Firmware);
    }

    #[test]
    fn global_backend_roundtrip() {
        // Process-wide selector: set + read returns what was set,
        // regardless of how `Default::default()` resolves on this
        // build.
        set_global_backend(ComputeBackend::Firmware);
        assert_eq!(global_backend(), ComputeBackend::Firmware);
        #[cfg(feature = "host")]
        {
            set_global_backend(ComputeBackend::Desktop);
            assert_eq!(global_backend(), ComputeBackend::Desktop);
        }
        // Reset to default so other tests aren't perturbed by
        // this one's side effect.
        GLOBAL_BACKEND.store(BACKEND_UNSET, Ordering::Relaxed);
        assert_eq!(global_backend(), ComputeBackend::default());
    }

    #[test]
    fn parse_firmware_ok() {
        assert_eq!(
            ComputeBackend::parse("firmware"),
            Ok(ComputeBackend::Firmware)
        );
    }

    #[cfg(feature = "host")]
    #[test]
    fn parse_desktop_ok_on_host() {
        assert_eq!(
            ComputeBackend::parse("desktop"),
            Ok(ComputeBackend::Desktop)
        );
    }

    #[test]
    fn parse_unknown_errs() {
        assert!(ComputeBackend::parse("avx2").is_err());
        assert!(ComputeBackend::parse("scalar").is_err()); // old name rejected
        assert!(ComputeBackend::parse("vectorized").is_err()); // old name rejected
        assert!(ComputeBackend::parse("").is_err());
    }

    #[test]
    fn name_parse_roundtrip() {
        // Genuine roundtrip: name() output must parse back to the
        // same variant. Catches drift if a variant gets renamed but
        // parse() isn't updated in lockstep.
        assert_eq!(
            ComputeBackend::parse(ComputeBackend::Firmware.name()),
            Ok(ComputeBackend::Firmware)
        );
        #[cfg(feature = "host")]
        {
            assert_eq!(
                ComputeBackend::parse(ComputeBackend::Desktop.name()),
                Ok(ComputeBackend::Desktop)
            );
        }
    }
}
