//! Operations — thin wrapper over `lamquant-ops`.
//!
//! Historically this module owned the OpEvent/OpSpec/runner code. As of the
//! UI parity refactor (Phase 1) the canonical implementation lives in the
//! `lamquant-ops` workspace crate so the Tauri GUI and Rust TUI share the
//! exact same event shapes, op-spec lookup, and subprocess supervisor.
//!
//! This file re-exports the bits the rest of `lamquant-core` already
//! consumed, plus a small `channel()` helper that bundles an `MpscSink`
//! with its receiver — that's the in-process pattern the Rust TUI's
//! output panel relies on.

// Backwards-compat submodule. Older code does
// `super::operations::runner::OpHandle`; declaring it here keeps that path
// valid while runner.rs itself only re-exports from `lamquant-ops`.
pub mod runner;

pub use lamquant_ops::{
    launcher, op_spec, spawn_command, spawn_lml, MpscSink, OpEvent, OpEventSink, OpHandle,
    OpProgressSnapshot, OpSpec, OpState,
};

/// Receiver paired with an `MpscSink`. The output panel drains this each
/// `tick()` and updates its line buffer.
pub type OpReceiver = std::sync::mpsc::Receiver<OpEvent>;

/// Convenience alias for the sender side.
pub type OpSender = MpscSink;

/// Create a fresh sink/receiver pair for in-process op runs. Bounded
/// at `DEFAULT_CHANNEL_BOUND` so a fast runner cannot grow the queue
/// unboundedly when the TUI drains slowly. Bible rule 33: every queue
/// has an explicit answer for what happens when it fills — here the
/// runner BLOCKS until the TUI drains, never drops. CR distribution,
/// recent files, and every Progress tick are guaranteed to reach the
/// dashboard.
pub fn channel() -> (MpscSink, OpReceiver) {
    lamquant_ops::sink::bounded_channel(lamquant_ops::sink::DEFAULT_CHANNEL_BOUND)
}
