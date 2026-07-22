#![cfg_attr(not(feature = "std"), no_std)]

//! # LamQuant LMQ — the Rust neural (lossy) codec shell (ADR 0074 Track N)
//!
//! LMQ is the **lossy/neural** spoke: reconstruction ≠ original, and the wire
//! carries entropy-coded FSQ tokens, not integer samples. It is therefore a
//! *parallel* path to the lossless `Codec` — it does NOT implement that trait.
//!
//! Architecture (a thin Rust **shell** over a swappable backend):
//! - **Wire**: the registered `bcs.lmq.progressive.v1` BCS2 codec profile,
//!   binding canonical ABIR semantics, mandatory model provenance, explicit
//!   non-exact fidelity, and this crate's deterministic token packet.
//! - **Entropy** ([`body`]): FSQ tokens → the byte-exact `lamquant_lml_mcu::rans`
//!   coder, self-described (the frequency model travels in-band).
//! - **Backend** (later phases): a `NeuralBackend` trait — a `PyBackend`
//!   (subprocess to the Python `SubbandCodec`) now, a fully-Rust `RustBackend`
//!   with embedded ternary weights later. The wire never changes between backends.
//!
//! `no_std`-first (the shell + entropy body are `alloc`-only); the Python backend
//! is host/`std`-gated.

extern crate alloc;

pub mod backend;
pub mod body;
#[cfg(feature = "python")]
pub mod py_backend;
pub mod rust_backend;
pub mod shell;
