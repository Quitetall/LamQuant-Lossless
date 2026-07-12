#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant LML — **Desktop tier** (ADR 0058): the host fast path.
//!
//! Desktop is *identical to the MCU tier, just fast* (ADR 0052): same LML wire
//! format, **byte-identical output**, with rayon per-channel parallelism (and
//! future SIMD) on top. The codec owner retains packet orchestration and private
//! channel plans; this crate owns only host backend selection and compatibility
//! entry points for the host execution profile.

extern crate alloc;

// ─── The Desktop fast path (rayon). Requires `fast` (the default). ─────────
#[cfg(feature = "fast")]
pub mod backend;
#[cfg(feature = "std")]
pub mod io;
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
        let sig: alloc::vec::Vec<alloc::vec::Vec<i64>> = (0..4)
            .map(|c| (0..256).map(|i| ((i * 3 + c) % 50) as i64 - 25).collect())
            .collect();
        let bytes =
            super::compress_with_mode_parallel(&sig, 0, lamquant_lml_mcu::lpc::LpcMode::default())
                .expect("parallel encode");
        let back = super::decompress_parallel(&bytes).expect("parallel decode");
        assert_eq!(back, sig);
    }
}
