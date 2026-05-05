//! Neural inference kernels — ternary MAC, FSQ, focal modulation, SNN.
//!
//! Pipeline: L3 approx [21][313] → TNN encoder (focal blocks) → latent
//! [32][T_latent] → WHT → FSQ → entropy coding.
//!
//! Phase 3 status:
//!   - ternary_mac: kernels ported (branchless + bit-serial CPOP paths)
//!   - fsq:         scalar quantization + flat-index encoding ported
//!   - focal:       TODO — needs weight export pipeline (Python → Rust const arrays)
//!   - snn:         TODO — same as focal

pub mod fsq;
pub mod ternary_mac;
