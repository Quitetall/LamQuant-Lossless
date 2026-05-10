//! Shared op-runner contract for all three LamQuant front-ends.
//!
//! Every long-running operation flows through this crate. Consumers vary
//! by lifetime model:
//!
//!   - **Rust TUI** runs the runner in-process. It uses `MpscSink` so the
//!     main loop can drain `OpEvent`s from an `mpsc::Receiver` each tick.
//!   - **Tauri GUI** runs the runner in a Tauri command. It uses a
//!     `TauriSink` (defined in `gui/src-tauri/src/op.rs`) that updates a
//!     shared `OpProgressSnapshot` on every event AND emits Tauri events
//!     on state transitions only. Frontend reads progress via 200ms
//!     `op_snapshot` polling — Tauri events are unreliable under load.
//!   - **Python TUI** consumes the JSON-line variant emitted by `lml
//!     --emit-json-events` AND produces matching JSON lines for ops it
//!     spawns natively (training, ML pipelines).
//!
//! The wire format is `specs/op-events.schema.json`. Drift between the
//! Rust enum, Python emitter, and TS types is caught by the parity test
//! at `tests/schema_parity.rs`.

pub mod launcher;
pub mod op_spec;
pub mod runner;
pub mod sink;
pub mod transport;

pub use launcher::launcher;
pub use op_spec::{op_spec, OpSpec};
pub use runner::{spawn_command, spawn_lml, OpHandle};
pub use sink::{MpscSink, OpEventSink, OpProgressSnapshot, OpState};
pub use transport::{
    Peer, PeerHealth, PeerInfo, RemoteHandle, RemotePath, SshConfig,
    Transport, TransportError, TransportKind,
};
pub use transport::ssh::SshTransport;

use serde::{Deserialize, Serialize};

/// Events emitted by background operations. Wire format is the canonical
/// JSON Schema at `specs/op-events.schema.json` — keep this enum in sync.
///
/// Tagged with `type` for JSON-line interoperability with Python and TS:
///
/// ```json
/// {"type":"Started","ts_ms":1730000000000,"op_id":"encode","total":42}
/// {"type":"Progress","ts_ms":1730000001000,"current":5,"total":42,"message":"file 5/42"}
/// {"type":"FileDone","ts_ms":1730000002000,"path":"a.lml","success":true,"cr":2.5,"ms":120}
/// {"type":"Done","ts_ms":1730000003000,"message":"42 files, 120 MiB"}
/// {"type":"Error","ts_ms":1730000003000,"message":"out of disk space"}
/// {"type":"Log","ts_ms":1730000003000,"message":"opening /data/a.edf"}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpEvent {
    /// Op has started. `total` is the work-unit count if known up front
    /// (file count for a directory walk, byte count for a single file).
    /// Required for all ops; it's the only event guaranteed to carry op_id.
    Started {
        ts_ms: i64,
        op_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total: Option<u64>,
    },

    /// Sticky progress update. The runner SHOULD emit one Progress per
    /// 100ms wall clock or per file processed, whichever is longer, so
    /// the UI poll loop has fresh-enough state to render.
    Progress {
        ts_ms: i64,
        current: u64,
        total: u64,
        message: String,
    },

    /// One file in a multi-file op completed (success or failure).
    FileDone {
        ts_ms: i64,
        path: String,
        success: bool,
        ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cr: Option<f64>,
        /// Original (uncompressed) byte size. Drives the dashboard's
        /// throughput + saved-bytes display. Optional for backward
        /// compatibility with pre-dashboard emitters.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes_in: Option<u64>,
        /// Compressed output byte size. Same dashboard role as
        /// bytes_in. Either both fields are present or neither.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes_out: Option<u64>,
    },

    /// Op completed successfully. Final event for a happy path.
    Done { ts_ms: i64, message: String },

    /// Op terminated abnormally — error message OR cancellation.
    /// Cancellations contain "cancelled" in the message; readers
    /// distinguish via substring match (see output panel render logic).
    Error { ts_ms: i64, message: String },

    /// Free-form line of output. Emitted for stdout/stderr passthrough
    /// when there's no structured event to match.
    Log { ts_ms: i64, message: String },
}

impl OpEvent {
    /// Wall-clock milliseconds since UNIX epoch. Used to stamp every event.
    pub fn now_ms() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Serialize as a single JSON line (no trailing newline). Caller
    /// appends `\n` when writing to stdout.
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            // Falling back to a plain Error variant so the consumer still
            // parses something. The ts is best-effort.
            format!(
                r#"{{"type":"Error","ts_ms":{},"message":"failed to serialize event: {}"}}"#,
                Self::now_ms(),
                e
            )
        })
    }

    /// Parse a JSON line (no trailing newline) emitted by another runner.
    pub fn from_json_line(line: &str) -> Result<Self, String> {
        serde_json::from_str(line).map_err(|e| format!("OpEvent parse: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_started_with_total() {
        let ev = OpEvent::Started {
            ts_ms: 1_730_000_000_000,
            op_id: "encode".into(),
            total: Some(42),
        };
        let line = ev.to_json_line();
        let back = OpEvent::from_json_line(&line).unwrap();
        match back {
            OpEvent::Started { op_id, total, .. } => {
                assert_eq!(op_id, "encode");
                assert_eq!(total, Some(42));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_started_without_total() {
        let ev = OpEvent::Started {
            ts_ms: 1_730_000_000_000,
            op_id: "info".into(),
            total: None,
        };
        let line = ev.to_json_line();
        // total: None must not appear in the wire format (skip_serializing_if).
        assert!(!line.contains("total"), "got {}", line);
        let back = OpEvent::from_json_line(&line).unwrap();
        match back {
            OpEvent::Started { total, .. } => assert_eq!(total, None),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn round_trip_filedone_no_cr() {
        let ev = OpEvent::FileDone {
            ts_ms: 1,
            path: "a.lml".into(),
            success: false,
            ms: 0,
            cr: None,
            bytes_in: None,
            bytes_out: None,
        };
        let line = ev.to_json_line();
        assert!(!line.contains("\"cr\""));
        let back = OpEvent::from_json_line(&line).unwrap();
        match back {
            OpEvent::FileDone { success, cr, .. } => {
                assert!(!success);
                assert_eq!(cr, None);
            }
            _ => panic!("wrong variant"),
        }
    }
}
