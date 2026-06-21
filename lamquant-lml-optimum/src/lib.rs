#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant **Optimum** (LMO) — the deterministic, maximum-compression-ratio
//! lossless / near-lossless codec (ADR 0052 Tier 3; the H.BWC ratio attack of
//! ADR 0054).
//!
//! **Phase 2 (current):** the LMO wire format + the [`LmoCodec`] implementation
//! of the shared [`Codec`] seam. WP0 is a *faithful re-encode of the LML
//! pipeline* (bit-exact parity baseline) — see [`lmo`]. The crate also provides
//! the *full* magic-dispatch decode ([`decode_any`]) that routes LMO streams
//! here, in contrast to `lamquant_lml_mcu::codec::decode` which returns a
//! typed "LMO not installed" for an LMO stream on a build that lacks this crate.
//!
//! **Phases 3–4 (next):** the ratio attack the bench says wins — Lagrangian
//! per-subband PCRD allocation, RD-optimized quantization, a deeper transform,
//! and a periodicity-aware long-lag predictor for ECG.
//!
//! Invariants (ADR 0052):
//!   * **Deterministic only** — no neural/learned models (those stay in LMQ,
//!     ADR 0049); this keeps LMO out of PCCP scope.
//!   * **`encode` is host-only** — its float/heavy DSP must never enter the
//!     riscv32 firmware dependency graph.
//!   * **`decode` is no_std-capable** — the optional Firmware LMO-decode module.
//!   * LMO carries a **true-lossless WP0** (bit-exact), not lossy-only.

extern crate alloc;

pub mod arith_int;
pub mod crosschan;
pub mod lmo;
pub mod lmo_lossless;
pub mod rls;
pub mod tcq;
pub mod lmo_pcrd97;
pub mod wavelet97;

// Re-export the shared codec seam (defined in -core) so consumers can write
// `use lamquant_lml_optimum::{Codec, Mode, Format, ...}` without reaching across to
// the core crate.
pub use lamquant_lml_mcu::codec::{
    peek_format, Codec, CodecError, Format, Mode, Signal, LML_MAGIC, LMO_MAGIC,
};

// The headline LMO surface.
pub use lmo::{decode, decode_any, LmoCodec, LMO_HEADER_LEN, LMO_VERSION};

#[cfg(feature = "encode")]
pub use lmo::encode;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_lmo1() {
        assert_eq!(LMO_MAGIC.as_slice(), b"LMO1");
    }

    #[test]
    fn lmo_round_trips_via_dispatch() {
        // decode-only builds can't encode; this exercises the full path when
        // the encoder is present.
        #[cfg(feature = "encode")]
        {
            let sig: Vec<Vec<i64>> =
                (0..3).map(|c| (0..200).map(|i| ((i * 5 + c) % 64) as i64 - 32).collect()).collect();
            let stream = encode(&sig, Mode::Lossless).expect("lmo encode");
            assert_eq!(peek_format(&stream), Some(Format::Lmo));
            let back = decode_any(&stream).expect("lmo dispatch decode");
            assert_eq!(back, sig);
        }
    }
}
