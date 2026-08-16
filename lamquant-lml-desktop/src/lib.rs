#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant LML — **Desktop tier** (ADR 0058): host fast path.
//!
//! Desktop selects byte-identical host execution profiles from the codec owner
//! and owns host backend selection plus `Read`/`Write` adapters. Packet
//! preparation, channel plans, and wire assembly do not cross this crate seam.

extern crate alloc;

#[cfg(feature = "fast")]
pub mod backend;
#[cfg(feature = "std")]
pub mod io;

#[cfg(feature = "fast")]
pub use backend::ComputeBackend;

#[cfg(all(test, feature = "fast"))]
mod tests {
    #[test]
    fn desktop_parallel_round_trips() {
        let signal: alloc::vec::Vec<alloc::vec::Vec<i64>> = (0..4)
            .map(|channel| {
                (0..256)
                    .map(|index| ((index * 3 + channel) % 50) as i64 - 25)
                    .collect()
            })
            .collect();
        let bytes = super::backend::compress_with_backend(
            &signal,
            0,
            lamquant_lml_mcu::lpc::LpcMode::default(),
            super::ComputeBackend::Desktop,
        )
        .expect("parallel encode");
        let recovered =
            super::backend::decompress_with_backend(&bytes, super::ComputeBackend::Desktop)
                .expect("parallel decode");
        assert_eq!(recovered, signal);
    }
}
