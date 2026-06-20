#![cfg_attr(not(feature = "std"), no_std)]
//! LamQuant **Optimum** (LMO) — the deterministic, maximum-compression-ratio
//! lossless / near-lossless codec (ADR 0052 Tier 3; the H.BWC ratio attack of
//! ADR 0054).
//!
//! This is a **skeleton** at warplan Phase 1: it establishes the crate, the
//! `decode`/`encode` feature split, and the dependency direction
//! (`lamquant-optimum` → `lamquant-lossless-core`). The real work lands later:
//!
//!   * **Phase 2** — the LMO wire format (distinct magic), the `Codec` trait it
//!     shares with LML, and a parity WP0 (a faithful re-encode of the LML
//!     pipeline into LMO, bit-exact).
//!   * **Phases 3-4** — the ratio attack the bench says wins: Lagrangian
//!     per-subband PCRD allocation, RD-optimized quantization, a deeper
//!     transform, and a periodicity-aware long-lag predictor for ECG.
//!
//! Invariants (ADR 0052):
//!   * **Deterministic only** — no neural/learned models (those stay in LMQ,
//!     ADR 0049); this keeps LMO out of PCCP scope.
//!   * **`encode` is host-only** — its float/heavy DSP must never enter the
//!     riscv32 firmware dependency graph.
//!   * **`decode` is no_std-capable** — the optional Firmware LMO-decode module.
//!   * LMO carries a **true-lossless WP0** (bit-exact, smaller than LML), not
//!     lossy-only.

#![cfg_attr(not(feature = "encode"), allow(dead_code))]

extern crate alloc;

/// The LMO format magic. A universal `decode(bytes)` dispatches on the leading
/// 4 bytes: LML streams route to `lamquant_lossless_core::lml`, LMO streams
/// route here. A build without this crate returns a typed "LMO not installed"
/// error rather than mis-parsing (wired in Phase 2).
pub const LMO_MAGIC: [u8; 4] = *b"LMO1";

// ── Phase 2 will introduce here ───────────────────────────────────────────
// * `pub trait Codec` — encode(&Signal, Mode) -> Stream / decode(&[u8]) -> Signal,
//   the I/O-contract seam shared with LML (likely defined in -core and impl'd here).
// * `mod lmo` — the LMO container + magic-dispatch decoder (feature = "decode").
// * `mod encode` — the deterministic max-CR analysis (feature = "encode").
//
// Kept intentionally empty in Phase 1 so the crate split is a pure, no-regret
// structural change with zero behavioural surface.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_lmo1() {
        assert_eq!(&LMO_MAGIC, b"LMO1");
    }
}
