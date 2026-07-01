#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant LML — **Desktop tier** (ADR 0058): the host fast path.
//!
//! Desktop is *identical to the MCU tier, just fast* (ADR 0052): same LML wire
//! format, **byte-identical output**, with rayon per-channel parallelism (and
//! future SIMD) on top. After the ADR 0058 carve-full, this crate *physically
//! owns* the parallel orchestration ([`parallel`]) and the [`backend`] selector
//! that chooses between the MCU scalar path and the Desktop parallel path —
//! both built over the MCU tier's exposed codec primitives, so the output is
//! byte-identical to the serial path by construction.
//!
//! The MCU tier (`lamquant-lml-mcu`) is re-exported wholesale, so a Desktop
//! consumer reaches the whole codec (`lml`, `codec`, `lpc`, …) plus the fast
//! path through this one crate.

extern crate alloc;

/// The MCU tier, re-exported under a tier-named alias.
pub use lamquant_lml_mcu as mcu;

// The full MCU codec surface (`lml`, the `codec` seam, `lpc`, `lifting`,
// `golomb`, …) re-exported so Desktop consumers reach everything here.
pub use lamquant_lml_mcu::*;

// ─── The Desktop fast path (rayon). Requires `fast` (the default). ─────────
#[cfg(feature = "fast")]
pub mod backend;
#[cfg(feature = "fast")]
pub mod parallel;

/// The runtime compute-backend selector (Firmware-scalar vs Desktop-parallel).
#[cfg(feature = "fast")]
pub use backend::ComputeBackend;

/// The Desktop fast-path entry points (rayon-parallel encode/decode), byte-
/// identical to the scalar MCU path.
#[cfg(feature = "fast")]
pub use parallel::{
    compress_with_mode_parallel, compress_with_mode_parallel_views, decompress_parallel,
};

#[cfg(all(test, feature = "fast"))]
mod tests {
    /// The authoritative cross-backend golden gate is
    /// `tests/byte_equal_backends.rs` (Firmware-scalar vs Desktop-parallel,
    /// byte-for-byte). This smoke just confirms the Desktop parallel path
    /// round-trips through the re-exported entry points.
    #[test]
    fn desktop_parallel_round_trips() {
        let sig: alloc::vec::Vec<alloc::vec::Vec<i64>> =
            (0..4).map(|c| (0..256).map(|i| ((i * 3 + c) % 50) as i64 - 25).collect()).collect();
        let bytes =
            super::compress_with_mode_parallel(&sig, 0, super::lpc::LpcMode::default())
                .expect("parallel encode");
        let back = super::decompress_parallel(&bytes).expect("parallel decode");
        assert_eq!(back, sig);
    }
}
