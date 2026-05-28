//! `StateSnapshot` — serializable view over `AppState`.
//!
//! Phase C.1a of the unified-state deployment plan. The TUI's
//! `AppState` holds OS resources (`OpHandle`, `RemoteHandle`) and
//! monotonic clocks (`std::time::Instant`) that don't `Serialize` and
//! shouldn't be exposed to the GUI bridge anyway. Instead of forcing
//! AppState through awkward serde gymnastics, we project it into a
//! flat snapshot type that:
//!
//!   - excludes runtime handles entirely (booleans expose the
//!     "is something running?" signal the GUI actually needs)
//!   - converts `Instant` deltas to human-readable strings using
//!     existing TrackedProcess helpers (so the wire format is stable
//!     across processes; no monotonic-clock leaking)
//!   - serializes via standard serde so the Tauri bridge can JSON it
//!     for free
//!
//! This is a one-way projection (`From<&AppState>`). The reverse
//! direction (apply a snapshot back to AppState) intentionally does
//! not exist — mutations always flow through `dispatch(Action)`.
//! Loading a snapshot back would defeat the whole point of the
//! redux-style chokepoint.

use serde::{Deserialize, Serialize};

use lamquant_ops::{Peer, TransportKind};

use super::config::LamQuantConfig;
use super::history::History;
use super::state::{AppState, LaunchState, TrackedProcess};

/// Peer with credentials stripped — what the GUI bridge actually needs
/// to render the peers panel. Drops `key_path` and `known_hosts` (which
/// expose the user's filesystem layout) and the `host_fingerprint`
/// (informational but not needed for display). Re-emits a string label
/// for the transport kind so the GUI doesn't need to know about
/// `TransportKind`'s tagged-enum wire format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerSnapshot {
    pub id: String,
    pub display: String,
    pub host: String,
    pub transport_kind: String,
    pub user: Option<String>,
    pub port: Option<u16>,
}

impl From<&Peer> for PeerSnapshot {
    fn from(p: &Peer) -> Self {
        let (kind, user, port) = match &p.transport {
            TransportKind::Ssh(cfg) => ("ssh".to_string(), Some(cfg.user.clone()), Some(cfg.port)),
        };
        Self {
            id: p.id.clone(),
            display: p.display.clone(),
            host: p.host.clone(),
            transport_kind: kind,
            user,
            port,
        }
    }
}

/// One TrackedProcess flattened to wire-friendly strings. Instant fields
/// become elapsed-string projections (`"12s"`, `"3m45s"`) computed at
/// snapshot time so the GUI doesn't need monotonic clock access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackedProcessSnapshot {
    pub label: String,
    pub kind: String,
    pub nav_screen: Option<String>,
    pub elapsed_str: String,
    pub done: bool,
    pub failed: bool,
    pub cancelled: bool,
    /// Empty string when not done; "5m ago" / "12s ago" otherwise.
    pub done_elapsed_str: String,
    /// Peer id when this process was dispatched remotely; None for local.
    pub peer: Option<String>,
}

impl From<&TrackedProcess> for TrackedProcessSnapshot {
    fn from(p: &TrackedProcess) -> Self {
        Self {
            label: p.label.clone(),
            kind: p.kind.to_string(),
            nav_screen: p.nav_screen.clone(),
            elapsed_str: p.elapsed_str(),
            done: p.done,
            failed: p.failed,
            cancelled: p.cancelled,
            done_elapsed_str: if p.done {
                p.done_elapsed_str()
            } else {
                String::new()
            },
            peer: p.peer.clone(),
        }
    }
}

/// LaunchState flattened — Instant fields collapse to a since-launch
/// elapsed millisecond count (relative to the snapshot, so wall-clock
/// jumps don't matter).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LaunchStateSnapshot {
    Idle,
    Launching { tool: String, elapsed_ms: u64 },
    Failed { tool: String },
}

impl From<&LaunchState> for LaunchStateSnapshot {
    fn from(s: &LaunchState) -> Self {
        match s {
            LaunchState::Idle => Self::Idle,
            LaunchState::Launching { tool, started } => Self::Launching {
                tool: tool.clone(),
                elapsed_ms: started.elapsed().as_millis() as u64,
            },
            LaunchState::Failed(tool) => Self::Failed { tool: tool.clone() },
        }
    }
}

/// Read-only projection of AppState for the GUI bridge. Captured each
/// time the bridge polls; ~5-20µs per snapshot for the field set below.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub cwd: String,
    pub status_message: Option<String>,
    pub version: String,
    pub cfg: LamQuantConfig,
    pub history: History,
    pub should_quit: bool,

    /// Op currently being configured. Just the op_id; full PendingOp
    /// is internal and a string is what the GUI needs to render.
    pub pending_op_id: Option<String>,

    /// True when ANY op is in flight (local subprocess or remote
    /// dispatch). Mirrors `App::op_in_flight()` from app.rs.
    pub op_running: bool,
    /// True when the running op was dispatched to a remote peer (i.e.,
    /// `state.remote_handle` is Some). Lets the GUI render an `@<peer>`
    /// indicator without needing to peek into TrackedProcess.peer.
    pub remote_op: bool,

    pub processes: Vec<TrackedProcessSnapshot>,
    pub launch_state: LaunchStateSnapshot,
    pub active_viz_tool: Option<String>,

    pub sidebar_focused: bool,
    pub sidebar_selected: usize,
    pub context_tick: u64,

    pub peers: Vec<PeerSnapshot>,
    pub selected_peer: Option<String>,
    pub peer_override: Option<String>,

    /// Most recent op log lines (capped at OP_LOG_CAP). Updated by
    /// `AppState::apply_op_event` from the dispatched OpEvent stream.
    /// Phase C.3c.
    pub op_log: Vec<String>,
    /// (current, total) progress for the in-flight op, or None if
    /// no op is running or progress hasn't started.
    pub op_progress: Option<(u64, u64)>,
    /// Terminal state of the most recent op: None = in flight,
    /// Some(true) = succeeded, Some(false) = failed/cancelled.
    pub op_terminal_ok: Option<bool>,
}

impl From<&AppState> for StateSnapshot {
    fn from(s: &AppState) -> Self {
        Self {
            cwd: s.cwd.display().to_string(),
            status_message: s.status_message.clone(),
            version: s.version.clone(),
            cfg: s.cfg.clone(),
            history: s.history.clone(),
            should_quit: s.should_quit,
            pending_op_id: s.pending_op.as_ref().map(|p| p.op_id.clone()),
            op_running: s.op_handle.is_some() || s.remote_handle.is_some(),
            remote_op: s.remote_handle.is_some(),
            processes: s.processes.iter().map(Into::into).collect(),
            launch_state: (&s.launch_state).into(),
            active_viz_tool: s.active_viz_tool.clone(),
            sidebar_focused: s.sidebar_focused,
            sidebar_selected: s.sidebar_selected,
            context_tick: s.context_tick,
            peers: s.peers.iter().map(Into::into).collect(),
            op_log: s.op_log.clone(),
            op_progress: s.op_progress,
            op_terminal_ok: s.op_terminal_ok,
            selected_peer: s.selected_peer.clone(),
            peer_override: s.peer_override.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_appstate_round_trips_through_json() {
        let s = AppState::new();
        let snap = StateSnapshot::from(&s);
        // Serialize + deserialize succeeds — the wire format works.
        let j = serde_json::to_string(&snap).expect("snapshot serialize");
        let _: StateSnapshot = serde_json::from_str(&j).expect("snapshot deserialize");
    }

    #[test]
    fn idle_launch_state_serializes() {
        let s = AppState::new();
        let snap = StateSnapshot::from(&s);
        match snap.launch_state {
            LaunchStateSnapshot::Idle => {}
            other => panic!("expected Idle launch state, got {:?}", other),
        }
    }

    #[test]
    fn op_running_false_for_fresh_state() {
        let s = AppState::new();
        let snap = StateSnapshot::from(&s);
        assert!(!snap.op_running, "no handle → op_running=false");
        assert!(!snap.remote_op, "no remote_handle → remote_op=false");
    }

    #[test]
    fn version_propagates() {
        let s = AppState::new();
        let snap = StateSnapshot::from(&s);
        assert_eq!(snap.version, env!("CARGO_PKG_VERSION"));
    }
}
