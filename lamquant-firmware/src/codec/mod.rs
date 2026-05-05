//! Codec layer — entropy coders, FSQ adaptive, detail thresholding, mailbox.
//!
//! Phase 4 status:
//!   - mailbox:          inter-core sync (Core 0 ↔ Core 1) ✅
//!   - quality:          quality mode + activity level enums ✅
//!   - detail_threshold: SNN-driven hard thresholding of DWT details ✅
//!   - fsq_adaptive:     variable-L FSQ driven by SNN activity_map ✅
//!   - lpc_delta:        Q31 / Q15 / Q8 delta encoding (lossless) ✅
//!   - rans_context:     context-adaptive rANS encoder ✅
//!   - hybrid_entropy:   orchestrator (Mode 1 rANS / Mode 2 LPC+Rice) ✅

pub mod detail_threshold;
pub mod fsq_adaptive;
pub mod hybrid_entropy;
pub mod lpc_delta;
pub mod mailbox;
pub mod quality;
pub mod rans_context;
