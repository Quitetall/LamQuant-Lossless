//! Codec layer — entropy coders, FSQ adaptive, detail thresholding, mailbox.
//!
//! Phase 4 status:
//!   - mailbox:          inter-core sync (Core 0 ↔ Core 1) ✅
//!   - quality:          quality mode + activity level enums ✅
//!   - detail_threshold: SNN-driven hard thresholding of DWT details ✅
//!   - fsq_adaptive:     variable-L FSQ driven by SNN activity_map ✅
//!   - rans_context:     TODO (316 LoC — context-adaptive rANS state machine)
//!   - lpc_delta:        TODO (lossless path)
//!   - hybrid_entropy:   TODO (orchestrator, wires rANS + Golomb together)

pub mod detail_threshold;
pub mod fsq_adaptive;
pub mod mailbox;
pub mod quality;
