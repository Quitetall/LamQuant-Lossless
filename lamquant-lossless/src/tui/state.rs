//! Application state — single source of truth for everything app-global.
//!
//! Loaded at startup; mutated only via `App::dispatch(Action)`. Panels
//! receive `&AppState` in commits 3+ and read fields directly instead of
//! caching their own copies.

use std::path::PathBuf;

use lamquant_ops::{Peer, RemoteHandle};

use super::config::LamQuantConfig;
use super::history::History;
use super::operations::runner::OpHandle;
use super::operations::OpSpec;

/// One tracked process entry — shown in the STATUS sidebar.
///
/// Migrated from `app.rs` in refactor commit 2. The `mark_done`/`expired`
/// helpers are unchanged; the type just lives here so `AppState` can own the
/// `Vec<TrackedProcess>` field without app.rs needing to re-export it.
#[derive(Debug, Clone)]
pub struct TrackedProcess {
    pub label: String,
    pub kind: &'static str,
    pub nav_screen: Option<String>,
    pub started: std::time::Instant,
    pub done: bool,
    pub failed: bool,
    pub cancelled: bool,
    /// Set when done. Successful entries auto-prune after 20s;
    /// failed/cancelled stay until the user presses [d]ismiss.
    pub done_at: Option<std::time::Instant>,
    /// Peer id this process is running on. None = local. Surfaces in the
    /// STATUS sidebar so users can tell at a glance which device a row
    /// belongs to. Set by `execute_pending` from the resolved peer.
    pub peer: Option<String>,
}

impl TrackedProcess {
    pub fn new(label: impl Into<String>, kind: &'static str, nav: Option<String>) -> Self {
        Self {
            label: label.into(),
            kind,
            nav_screen: nav,
            started: std::time::Instant::now(),
            done: false,
            failed: false,
            cancelled: false,
            done_at: None,
            peer: None,
        }
    }

    /// Builder-style attribution for remote dispatch. Sticks the peer id
    /// onto an existing TrackedProcess so the sidebar can show `@<peer>`.
    pub fn with_peer(mut self, peer_id: impl Into<String>) -> Self {
        self.peer = Some(peer_id.into());
        self
    }

    pub fn mark_done(&mut self, failed: bool, cancelled: bool) {
        if !self.done {
            self.done = true;
            self.failed = failed;
            self.cancelled = cancelled;
            self.done_at = Some(std::time::Instant::now());
        }
    }

    /// Only successful completions auto-expire; failed/cancelled require [d]ismiss.
    pub fn expired(&self) -> bool {
        if self.failed || self.cancelled {
            return false;
        }
        self.done_at
            .map(|t| t.elapsed() >= std::time::Duration::from_secs(20))
            .unwrap_or(false)
    }

    pub fn elapsed_str(&self) -> String {
        let s = self.started.elapsed().as_secs();
        if s < 60 {
            format!("{}s", s)
        } else if s < 3600 {
            format!("{}m{}s", s / 60, s % 60)
        } else {
            format!("{}h{}m", s / 3600, (s % 3600) / 60)
        }
    }

    /// Time since the process ended (for failed/cancelled timestamps).
    pub fn done_elapsed_str(&self) -> String {
        let Some(t) = self.done_at else {
            return String::new();
        };
        let s = t.elapsed().as_secs();
        if s < 60 {
            format!("{}s ago", s)
        } else if s < 3600 {
            format!("{}m ago", s / 60)
        } else {
            format!("{}h ago", s / 3600)
        }
    }
}

/// Tracks the lifecycle of an external launcher (viz tool, training session).
#[derive(Debug, Clone)]
pub enum LaunchState {
    Idle,
    Launching {
        tool: String,
        started: std::time::Instant,
    },
    Failed(String),
}

/// In-flight operation request — accumulated by start_op as the user picks
/// input/output paths, then consumed by execute_pending.
#[derive(Debug, Clone)]
pub struct PendingOp {
    pub op_id: String,
    pub spec: OpSpec,
    pub input: Option<String>,
    pub output: Option<String>,
}

/// Runtime application state — every app-global mutable field lives here.
/// Mutations flow through `App::dispatch(Action)`.
pub struct AppState {
    // ── Static / config ────────────────────────────────────────────────
    /// Current working directory for file operations.
    pub cwd: PathBuf,
    /// Status bar message (cleared after display).
    pub status_message: Option<String>,
    /// Application version string.
    pub version: String,
    /// Live snapshot of the user config — re-read after wizard/settings save.
    pub cfg: LamQuantConfig,
    /// Persisted user history (recent inputs/outputs, last op, interrupted flag).
    pub history: History,

    // ── Lifecycle / quit ───────────────────────────────────────────────
    /// Main loop checks this after every event tick.
    pub should_quit: bool,

    // ── Op pipeline ────────────────────────────────────────────────────
    /// Op currently being configured (input/output picker flow).
    pub pending_op: Option<PendingOp>,
    /// Handle to the running LOCAL subprocess. Some only on SCREEN_RUNNING
    /// when the op was dispatched locally. Mutually exclusive with
    /// `remote_handle`.
    pub op_handle: Option<OpHandle>,
    /// Handle to the running REMOTE op (returned by Transport::dispatch).
    /// Some only on SCREEN_RUNNING when the op was dispatched to a peer.
    /// Mutually exclusive with `op_handle`. Cancel path checks both.
    pub remote_handle: Option<RemoteHandle>,
    /// All tracked processes (ops + launchers + viz). Drives STATUS sidebar.
    pub processes: Vec<TrackedProcess>,

    // ── External launcher state ────────────────────────────────────────
    pub launch_state: LaunchState,
    /// Name of the viz tool that was last launched (drives EEG sidebar gating).
    pub active_viz_tool: Option<String>,
    /// User-selected input file for the next viz launcher (T5 / ADR 0020).
    /// `start_launcher` substitutes any `$INPUT` token in the launcher's
    /// argv with this path. `None` means the user hasn't picked a file
    /// yet; pipe-shaped viz tools reject the launch with a clear error
    /// instead of opening blank.
    pub viz_selected_input: Option<std::path::PathBuf>,
    /// Firmware export source: weights checkpoint path (T6 / ADR 0019).
    /// `start_launcher` substitutes any `$WEIGHTS` token in the
    /// launcher's argv with this path.
    pub fw_weights_input: Option<std::path::PathBuf>,
    /// Firmware export destination: output bundle path (T6 / ADR 0019).
    /// `start_launcher` substitutes any `$OUTPUT` token in the
    /// launcher's argv with this path.
    pub fw_output_path: Option<std::path::PathBuf>,

    // ── UI / sidebar ───────────────────────────────────────────────────
    /// Whether arrow keys route to the sidebar process list instead of workflows.
    pub sidebar_focused: bool,
    pub sidebar_selected: usize,
    /// Incremented every tick (~50ms); drives context panel animations.
    pub context_tick: u64,

    // ── Multi-device (P-multi-device commit 2/6) ───────────────────────
    /// Configured remote peers loaded from peers.json. Empty if the file
    /// is missing — peer feature is opt-in.
    pub peers: Vec<Peer>,
    /// Sticky peer for outgoing ops. None = local execution. Set/cleared
    /// from the Peers panel.
    pub selected_peer: Option<String>,
    /// One-shot override for the next op. Consumed by `start_op` and
    /// reset to None. Set via the per-op-override key in panels that
    /// dispatch ops.
    pub peer_override: Option<String>,

    // ── Op-event mirror (Phase C.3c) ────────────────────────────────────
    /// Recent op log lines, capped at OP_LOG_CAP entries (FIFO trim).
    /// Updated by `apply_op_event` from drained OpEvents — mirrors
    /// what the OutputPanel renders, but accessible to the GUI bridge
    /// without needing OutputPanel-equivalent on the Tauri side.
    pub op_log: Vec<String>,
    /// Last seen progress (current, total). None until first
    /// `OpEvent::Progress`. Cleared on `OpEvent::Done`/`Error`.
    pub op_progress: Option<(u64, u64)>,
    /// Terminal state of the most recent op: None = no op or in flight,
    /// Some(true) = Done success, Some(false) = Error/cancelled.
    pub op_terminal_ok: Option<bool>,
    /// Timestamp the running op flipped to Done. Used by the TUI's
    /// tick_panels to auto-route back to the previous screen after a
    /// success-grace period (so users don't have to press [b] manually
    /// when a workflow finishes cleanly). None when no op active or
    /// when the op terminal hasn't been observed yet. Reset on each
    /// new Started event. Not serialized — Instant is monotonic and
    /// lives only in-process; the snapshot doesn't expose it.
    pub op_done_at: Option<std::time::Instant>,
}

/// Maximum op_log lines retained in AppState. Older lines drop on push.
/// Keeps StateSnapshot serialization bounded — large logs go to the
/// OutputPanel (TUI) or could be streamed via a separate channel.
pub const OP_LOG_CAP: usize = 200;

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            status_message: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
            cfg: LamQuantConfig::load(),
            history: History::load(),

            should_quit: false,

            pending_op: None,
            op_handle: None,
            remote_handle: None,
            processes: Vec::new(),

            launch_state: LaunchState::Idle,
            active_viz_tool: None,
            viz_selected_input: None,
            fw_weights_input: None,
            fw_output_path: None,

            sidebar_focused: false,
            sidebar_selected: 0,
            context_tick: 0,

            peers: super::peers_config::load().peers,
            selected_peer: None,
            peer_override: None,

            op_log: Vec::new(),
            op_progress: None,
            op_terminal_ok: None,
            op_done_at: None,
        }
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
    }

    pub fn take_status(&mut self) -> Option<String> {
        self.status_message.take()
    }

    /// Recent input paths — single source of truth is `history.recent_inputs`.
    pub fn recent_inputs(&self) -> &[String] {
        &self.history.recent_inputs
    }

    /// Recent output paths — single source of truth is `history.recent_outputs`.
    pub fn recent_outputs(&self) -> &[String] {
        &self.history.recent_outputs
    }

    pub fn add_recent_input(&mut self, path: &str) {
        self.history.add_input(path);
    }

    pub fn add_recent_output(&mut self, path: &str) {
        self.history.add_output(path);
    }

    /// Apply Action::SetCfgWorkers — mutate `cfg.compute.workers`
    /// in memory only. Negatives clamp to 0 (auto-detect). Returns
    /// true if the value changed; callers persist via `save_cfg_now`
    /// when they want it on disk. Splitting mutate from save keeps
    /// unit tests of the reducer fast and free of disk side effects.
    pub fn apply_cfg_workers(&mut self, workers: i64) -> bool {
        let workers = workers.max(0);
        let changed = self.cfg.compute.workers != workers;
        self.cfg.compute.workers = workers;
        self.set_status(format!("cfg.compute.workers = {}", workers));
        changed
    }

    /// Apply Action::SetCfgBackend. Accepts auto/rust/python (case-
    /// insensitive, trimmed); other values leave cfg unchanged and
    /// emit a status warning. Returns true if mutation occurred.
    pub fn apply_cfg_backend(&mut self, mode: String) -> bool {
        let normalized = mode.trim().to_ascii_lowercase();
        if !matches!(normalized.as_str(), "auto" | "rust" | "python") {
            self.set_status(format!(
                "Invalid backend '{}' (expected auto/rust/python) — unchanged.",
                mode,
            ));
            return false;
        }
        let changed = self.cfg.backend.mode != normalized;
        self.cfg.backend.mode = normalized.clone();
        self.set_status(format!("cfg.backend.mode = {}", normalized));
        changed
    }

    /// Apply Action::SetCfgVerification. Accepts paranoid/standard/fast.
    pub fn apply_cfg_verification(&mut self, level: String) -> bool {
        let normalized = level.trim().to_ascii_lowercase();
        if !matches!(normalized.as_str(), "paranoid" | "standard" | "fast",) {
            self.set_status(format!(
                "Invalid verification '{}' (expected paranoid/standard/fast) — unchanged.",
                level,
            ));
            return false;
        }
        let changed = self.cfg.codec.verification != normalized;
        self.cfg.codec.verification = normalized.clone();
        self.set_status(format!("cfg.codec.verification = {}", normalized));
        changed
    }

    /// Persist the in-memory cfg to lamquant.toml. Updates status
    /// with the result. Called by dispatch sites after a successful
    /// apply_cfg_* mutation.
    pub fn save_cfg_now(&mut self) {
        match self.cfg.save() {
            Ok(()) => self.set_status("cfg saved"),
            Err(e) => self.set_status(format!("cfg save failed: {}", e)),
        }
    }

    /// Apply a drained OpEvent to AppState — phase C.3c. Mirrors what
    /// the TUI's OutputPanel renders, but stores it in AppState so the
    /// GUI bridge can surface it through StateSnapshot without needing
    /// an OutputPanel-equivalent on the Tauri side.
    ///
    /// During cohabit (until OutputPanel is fully retired), the TUI
    /// dispatch arm calls BOTH `output_panel.consume(ev)` AND
    /// `state.apply_op_event(ev.clone())` so both consumers see the
    /// same stream.
    pub fn apply_op_event(&mut self, ev: &lamquant_ops::OpEvent) {
        use lamquant_ops::OpEvent;
        match ev {
            OpEvent::Started { op_id, total, .. } => {
                self.op_log.clear();
                self.op_progress = total.map(|t| (0, t));
                self.op_terminal_ok = None;
                self.op_done_at = None; // clear stale auto-back timer
                self.op_log.push(format!("started: {}", op_id));
                self.trim_op_log();
            }
            OpEvent::Progress {
                current,
                total,
                message,
                ..
            } => {
                self.op_progress = Some((*current, *total));
                if !message.is_empty() {
                    self.op_log.push(message.clone());
                    self.trim_op_log();
                }
            }
            OpEvent::FileDone {
                path, success, ms, ..
            } => {
                let mark = if *success { "✓" } else { "✗" };
                self.op_log.push(format!("{} {} ({}ms)", mark, path, ms));
                self.trim_op_log();
            }
            OpEvent::Done { message, .. } => {
                self.op_terminal_ok = Some(true);
                self.op_progress = None;
                self.op_log.push(format!("done: {}", message));
                self.trim_op_log();
            }
            OpEvent::Error { message, .. } => {
                self.op_terminal_ok = Some(false);
                self.op_progress = None;
                self.op_log.push(format!("error: {}", message));
                self.trim_op_log();
            }
            OpEvent::Log { message, .. } => {
                self.op_log.push(message.clone());
                self.trim_op_log();
            }
        }
    }

    fn trim_op_log(&mut self) {
        if self.op_log.len() > OP_LOG_CAP {
            let drop = self.op_log.len() - OP_LOG_CAP;
            self.op_log.drain(..drop);
        }
    }

    /// Apply Action::SelectPeer to AppState. Shared between the TUI's
    /// dispatch chokepoint (app.rs) and the GUI bridge's
    /// apply_action (gui/src-tauri/src/state_bridge.rs) so both
    /// reducers stay strictly identical. Validation rules:
    ///
    ///   - Some("") → treated as None (clear sticky). Empty ids
    ///     would otherwise display as "" in UI; safer to clear.
    ///   - Some(unknown_id) → status warning, no state mutation.
    ///   - None → clear sticky, fall back to local.
    pub fn apply_select_peer(&mut self, id: Option<String>) {
        match id {
            Some(pid) if !pid.is_empty() => {
                let known = self.peers.iter().any(|p| p.id == pid);
                if known {
                    self.set_status(format!("Sticky peer set: {}", pid));
                    self.selected_peer = Some(pid);
                } else {
                    self.set_status(format!("Unknown peer id `{}` — selection unchanged.", pid,));
                }
            }
            _ => {
                self.selected_peer = None;
                self.set_status("Cleared peer selection — running local.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_peer_none_clears() {
        let mut s = AppState::new();
        s.selected_peer = Some("anything".into());
        s.apply_select_peer(None);
        assert_eq!(s.selected_peer, None);
    }

    #[test]
    fn select_peer_empty_string_clears() {
        let mut s = AppState::new();
        s.selected_peer = Some("anything".into());
        s.apply_select_peer(Some(String::new()));
        assert_eq!(s.selected_peer, None, "empty id should clear, not set ''");
    }

    #[test]
    fn select_peer_unknown_id_preserves_existing() {
        let mut s = AppState::new();
        s.peers = Vec::new();
        s.selected_peer = Some("preexisting".into());
        s.apply_select_peer(Some("nonexistent".into()));
        assert_eq!(
            s.selected_peer.as_deref(),
            Some("preexisting"),
            "unknown peer id must not displace an existing selected_peer",
        );
    }

    #[test]
    fn cfg_workers_negative_clamps_to_zero() {
        let mut s = AppState::new();
        s.apply_cfg_workers(-3);
        assert_eq!(
            s.cfg.compute.workers, 0,
            "negative workers must clamp to auto-detect"
        );
    }

    #[test]
    fn cfg_workers_normal_value_applied() {
        let mut s = AppState::new();
        s.apply_cfg_workers(8);
        assert_eq!(s.cfg.compute.workers, 8);
    }

    #[test]
    fn cfg_backend_invalid_no_mutation() {
        let mut s = AppState::new();
        let prev = s.cfg.backend.mode.clone();
        s.apply_cfg_backend("nonsense".into());
        assert_eq!(
            s.cfg.backend.mode, prev,
            "invalid backend must not mutate cfg"
        );
    }

    #[test]
    fn cfg_backend_valid_normalized() {
        let mut s = AppState::new();
        s.apply_cfg_backend("  RUST ".into());
        assert_eq!(s.cfg.backend.mode, "rust", "trim + lowercase normalization");
    }

    #[test]
    fn cfg_verification_invalid_no_mutation() {
        let mut s = AppState::new();
        let prev = s.cfg.codec.verification.clone();
        s.apply_cfg_verification("xtreme".into());
        assert_eq!(s.cfg.codec.verification, prev);
    }

    #[test]
    fn cfg_verification_valid_normalized() {
        let mut s = AppState::new();
        s.apply_cfg_verification("Paranoid".into());
        assert_eq!(s.cfg.codec.verification, "paranoid");
    }

    use lamquant_ops::OpEvent;

    fn ev_started(op_id: &str, total: Option<u64>) -> OpEvent {
        OpEvent::Started {
            ts_ms: 0,
            op_id: op_id.into(),
            total,
        }
    }
    fn ev_progress(c: u64, t: u64, msg: &str) -> OpEvent {
        OpEvent::Progress {
            ts_ms: 0,
            current: c,
            total: t,
            message: msg.into(),
        }
    }
    fn ev_done(msg: &str) -> OpEvent {
        OpEvent::Done {
            ts_ms: 0,
            message: msg.into(),
        }
    }
    fn ev_error(msg: &str) -> OpEvent {
        OpEvent::Error {
            ts_ms: 0,
            message: msg.into(),
        }
    }

    #[test]
    fn op_event_started_resets_state() {
        let mut s = AppState::new();
        s.op_log.push("stale line".into());
        s.op_terminal_ok = Some(true);
        s.apply_op_event(&ev_started("encode", Some(42)));
        assert_eq!(s.op_progress, Some((0, 42)));
        assert_eq!(s.op_terminal_ok, None);
        assert!(s.op_log.iter().any(|l| l.contains("started: encode")));
        assert!(
            !s.op_log.iter().any(|l| l == "stale line"),
            "Started must clear stale log"
        );
    }

    #[test]
    fn op_event_progress_updates_pair() {
        let mut s = AppState::new();
        s.apply_op_event(&ev_started("encode", Some(10)));
        s.apply_op_event(&ev_progress(3, 10, "file 3/10"));
        assert_eq!(s.op_progress, Some((3, 10)));
        assert!(s.op_log.iter().any(|l| l == "file 3/10"));
    }

    #[test]
    fn op_event_done_marks_terminal_ok() {
        let mut s = AppState::new();
        s.apply_op_event(&ev_started("encode", None));
        s.apply_op_event(&ev_done("42 files, 120 MiB"));
        assert_eq!(s.op_terminal_ok, Some(true));
        assert_eq!(
            s.op_progress, None,
            "Done clears the progress so render shows no in-flight bar"
        );
    }

    #[test]
    fn op_event_error_marks_terminal_failed() {
        let mut s = AppState::new();
        s.apply_op_event(&ev_started("encode", None));
        s.apply_op_event(&ev_error("disk full"));
        assert_eq!(s.op_terminal_ok, Some(false));
        assert_eq!(s.op_progress, None);
    }

    #[test]
    fn op_log_caps_at_op_log_cap() {
        let mut s = AppState::new();
        s.apply_op_event(&ev_started("e", Some(1000)));
        for i in 0..(OP_LOG_CAP * 2) {
            s.apply_op_event(&OpEvent::Log {
                ts_ms: 0,
                message: format!("line {}", i),
            });
        }
        assert!(
            s.op_log.len() <= OP_LOG_CAP,
            "op_log unbounded — would leak memory across long runs"
        );
    }
}
