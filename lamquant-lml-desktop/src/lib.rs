#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant LML — **Desktop tier** (ADR 0058): the host fast path.
//!
//! Desktop is *identical to the MCU tier, just fast* (ADR 0052): same LML wire
//! format, **byte-identical output**, with rayon per-channel parallelism (and
//! future SIMD) on top. It is an *assembly* of the MCU codec with the host perf
//! path enabled — not a reimplementation — so the only public surface here is
//! the MCU codec re-exported plus the parallel encode/decode entry points.
//!
//! **Carve status:** the rayon code physically lives in `lamquant-lml-mcu`
//! behind its `archive` feature today; this crate enables it (`fast`, on by
//! default) and presents it as the Desktop tier. Relocating that code into this
//! crate — which would move the `byte_equal_backends` Firmware-vs-Desktop gate
//! across the crate boundary — is a tracked follow-up (ADR 0058 Progress Log).

extern crate alloc;

/// The MCU tier, re-exported under a tier-named alias.
pub use lamquant_lml_mcu as mcu;

// The full MCU codec surface (the `lml` codec, the `codec` seam, `lpc`,
// `lifting`, `golomb`, …) so a Desktop consumer reaches everything through this
// one crate.
pub use lamquant_lml_mcu::*;

/// The Desktop fast-path entry points (rayon-parallel encode/decode). Present
/// only with the `fast` feature (the default), which turns on the MCU tier's
/// `archive` rayon path. Output is byte-identical to the scalar MCU path.
#[cfg(feature = "fast")]
pub use lamquant_lml_mcu::lml::{compress_with_mode_parallel, decompress_parallel};

/// The runtime compute-backend selector (shared with the MCU tier). On a Desktop
/// build `ComputeBackend::default()` is `Desktop` (rayon + SIMD).
pub use lamquant_lml_mcu::backend::ComputeBackend;

#[cfg(test)]
mod tests {
    /// Byte-identity is the load-bearing property: the Desktop parallel path
    /// must produce the exact same LML bytes as the scalar MCU path. The
    /// authoritative cross-backend golden gate lives in
    /// `lamquant-lml-mcu/tests/byte_equal_backends.rs` (both backends compile in
    /// the MCU crate today); this smoke just confirms the Desktop tier re-exports
    /// a working parallel round-trip.
    #[cfg(feature = "fast")]
    #[test]
    fn desktop_parallel_round_trips() {
        let sig: alloc::vec::Vec<alloc::vec::Vec<i64>> =
            (0..4).map(|c| (0..256).map(|i| ((i * 3 + c) % 50) as i64 - 25).collect()).collect();
        let bytes = super::compress_with_mode_parallel(
            &sig,
            0,
            super::lpc::LpcMode::default(),
        )
        .expect("parallel encode");
        let back = super::decompress_parallel(&bytes).expect("parallel decode");
        assert_eq!(back, sig);
    }
}
