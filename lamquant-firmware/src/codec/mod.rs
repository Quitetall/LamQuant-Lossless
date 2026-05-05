//! Codec layer: rANS context, Golomb-Rice, hybrid entropy, FSQ adaptive.
//!
//! Phase 4 — currently empty pending port from
//! `firmware/codec/{rans_context,fsq_adaptive,detail_threshold,lpc_delta}.c`.
//! Underlying `lamquant_core::rans` and `lamquant_core::golomb` already
//! work; this layer adds context selection and adaptive level dispatch.
