//! Event sinks — different lifetime models for different consumers.
//!
//! - `MpscSink` is for in-process consumers (Rust TUI). Forwards each
//!   `OpEvent` into a `std::sync::mpsc::Sender` and lets the consumer's
//!   render loop drain on each tick.
//!
//! - Tauri GUI provides its own sink (in `gui/src-tauri/src/op.rs`) that
//!   updates a shared `OpProgressSnapshot` on every event AND emits Tauri
//!   events only on state transitions. Frontend polls `op_snapshot` for
//!   incremental progress at 200ms — Tauri's event delivery is less
//!   reliable than its sync command path under load.
//!
//! The `OpEventSink` trait abstracts both lifetimes without leaking into
//! the runner's signature.

use std::sync::mpsc;

use crate::OpEvent;

/// Coarse-grained state for poll-based readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OpState {
    /// Spawned but no Started event seen yet.
    Pending,
    /// Started, no terminal event yet.
    Running,
    /// User pressed cancel; waiting for the child to exit.
    Cancelling,
    /// Done variant received.
    Done,
    /// Error variant received (real failure OR cancellation).
    Failed,
}

/// Snapshot of op progress for poll-based readers.
///
/// Tauri GUI's frontend polls this every 200ms via the `op_snapshot` Tauri
/// command instead of subscribing to per-event Tauri events. Tauri's event
/// delivery under load (window minimize, tab switch, HMR) drops events;
/// the snapshot is the source of truth for incremental progress.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpProgressSnapshot {
    pub op_id: String,
    pub state: OpState,
    pub current: u64,
    pub total: u64,
    pub message: String,
    /// Most recent FileDone path, if any.
    pub last_file: Option<String>,
    /// Most recent terminal message (Done.message or Error.message).
    pub terminal_message: Option<String>,
    /// Wall-clock ms of the latest update — frontends use this to detect
    /// staleness if the runner hangs.
    pub updated_ms: i64,
}

impl OpProgressSnapshot {
    pub fn new(op_id: impl Into<String>) -> Self {
        Self {
            op_id: op_id.into(),
            state: OpState::Pending,
            current: 0,
            total: 0,
            message: String::new(),
            last_file: None,
            terminal_message: None,
            updated_ms: OpEvent::now_ms(),
        }
    }

    /// Apply an event to the snapshot. Idempotent except for `updated_ms`.
    pub fn apply(&mut self, event: &OpEvent) {
        self.updated_ms = OpEvent::now_ms();
        match event {
            OpEvent::Started { total, .. } => {
                self.state = OpState::Running;
                if let Some(t) = total {
                    self.total = *t;
                }
            }
            OpEvent::Progress { current, total, message, .. } => {
                self.current = *current;
                self.total = *total;
                self.message = message.clone();
            }
            OpEvent::FileDone { path, .. } => {
                self.last_file = Some(path.clone());
            }
            OpEvent::Done { message, .. } => {
                self.state = OpState::Done;
                self.terminal_message = Some(message.clone());
            }
            OpEvent::Error { message, .. } => {
                self.state = OpState::Failed;
                self.terminal_message = Some(message.clone());
            }
            OpEvent::Log { .. } => { /* logs don't change state */ }
        }
    }
}

/// Trait that runner consumers implement to receive op events.
///
/// Sink is shared via `Arc<S>` across the supervisor + stdout/stderr reader
/// threads, so it must be `Send + Sync + 'static`. Implementors typically
/// store any internal channels behind locks (`MpscSink` wraps a clonable
/// `Sender` which is itself `Sync`; the Tauri `TauriSink` puts its snapshot
/// behind a `Mutex`).
pub trait OpEventSink: Send + Sync + 'static {
    /// Forward a single event. MUST be cheap — runners may call this from
    /// the supervisor thread or from an stdout reader thread, and dropping
    /// events is preferable to blocking the runner.
    fn emit(&self, event: OpEvent);
}

/// In-process sink that forwards events to an `mpsc::Sender`. Used by the
/// Rust TUI's output panel: the panel's `tick()` drains the receiver each
/// frame and updates its line buffer.
///
/// `mpsc::Sender` is not `Sync` on stable Rust, so we wrap it in a `Mutex`
/// to satisfy the `OpEventSink` trait bound. Lock contention is negligible
/// — events arrive far slower than a mutex can release.
pub struct MpscSink {
    tx: std::sync::Mutex<mpsc::Sender<OpEvent>>,
}

impl MpscSink {
    pub fn new(tx: mpsc::Sender<OpEvent>) -> Self {
        Self { tx: std::sync::Mutex::new(tx) }
    }
}

impl OpEventSink for MpscSink {
    fn emit(&self, event: OpEvent) {
        // Best-effort send. Receiver may be dropped (UI navigated away);
        // we can't recover so just discard — runner's terminal Done/Error
        // event will be the last thing it tries to deliver anyway.
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(event);
        }
    }
}

/// Receiving half of the op-event channel. Type alias so callers don't
/// have to spell out `mpsc::Receiver<OpEvent>` every time. Used by the
/// [`crate::transport::Transport`] trait return type.
pub type OpReceiver = mpsc::Receiver<OpEvent>;

/// Convenience: create a paired (sender-sink, receiver) for in-process use.
pub fn channel() -> (MpscSink, OpReceiver) {
    let (tx, rx) = mpsc::channel();
    (MpscSink::new(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_progresses_with_events() {
        let mut snap = OpProgressSnapshot::new("encode");
        assert_eq!(snap.state, OpState::Pending);

        snap.apply(&OpEvent::Started { ts_ms: 0, op_id: "encode".into(), total: Some(10) });
        assert_eq!(snap.state, OpState::Running);
        assert_eq!(snap.total, 10);

        snap.apply(&OpEvent::Progress { ts_ms: 0, current: 3, total: 10, message: "f3".into() });
        assert_eq!(snap.current, 3);
        assert_eq!(snap.message, "f3");

        snap.apply(&OpEvent::FileDone {
            ts_ms: 0, path: "f3.lml".into(), success: true, ms: 1,
            cr: Some(2.0), bytes_in: None, bytes_out: None,
            samples: None, duration_s: None, n_channels: None,
            sample_rate: None, sha256: None, n_windows: None,
        });
        assert_eq!(snap.last_file.as_deref(), Some("f3.lml"));

        snap.apply(&OpEvent::Done { ts_ms: 0, message: "ok".into() });
        assert_eq!(snap.state, OpState::Done);
    }

    #[test]
    fn mpsc_sink_round_trip() {
        let (sink, rx) = channel();
        sink.emit(OpEvent::Log { ts_ms: 0, message: "hi".into() });
        match rx.try_recv().unwrap() {
            OpEvent::Log { message, .. } => assert_eq!(message, "hi"),
            _ => panic!("wrong variant"),
        }
    }
}
