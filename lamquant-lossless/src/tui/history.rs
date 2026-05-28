//! History — re-export from the shared `lamquant-history` workspace crate.
//!
//! As of Phase 2 #34 the canonical implementation lives in
//! `crates/lamquant-history` so the Tauri GUI and Python TUI write the same
//! file with the same locking + atomic-rename guarantees. This module is a
//! thin compatibility layer for the Rust TUI's existing call sites
//! (`app.rs`, `panels/resume.rs`, `panels/file_browser.rs`).
//!
//! The wrapper exposes the legacy field names (`recent_inputs`,
//! `recent_outputs`) the TUI panels were built against, while delegating to
//! the spec-compliant on-disk format.

pub use lamquant_history::{history_path, HistoryOp};

use serde::{Deserialize, Serialize};

/// Public TUI history wrapper. Adapter over `lamquant_history::History`
/// that preserves the field-access pattern used by older call sites.
///
/// Serialize/Deserialize implementations cover the mirrored public
/// fields only — `inner` is intentionally `#[serde(skip)]` because
/// it's a derived disk-sync handle and snapshots reconstruct it from
/// the mirrors anyway. Used by `tui::snapshot::StateSnapshot`.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct History {
    /// Disk-sync handle. Reconstructed from the mirror fields on
    /// deserialize via `Default`. `#[serde(default)]` is explicit so
    /// the dependency on `lamquant_history::History: Default` is
    /// documented and fails loudly if that impl ever disappears.
    #[serde(skip, default)]
    inner: lamquant_history::History,
    /// Mirror of `inner.recent_paths.inputs` — exposed as a flat field so
    /// `app.rs` and `panels/file_browser.rs` continue to compile without a
    /// rename. Sync via `add_input`/`add_output`/`save`.
    pub recent_inputs: Vec<String>,
    /// Mirror of `inner.recent_paths.outputs`.
    pub recent_outputs: Vec<String>,
    /// Mirror of `inner.last_op` etc. for the resume panel.
    pub last_op: Option<String>,
    pub last_input: Option<String>,
    pub last_output: Option<String>,
    pub interrupted: bool,
}

impl History {
    pub fn load() -> Self {
        Self::from_inner(lamquant_history::History::load())
    }

    pub fn save(&self) {
        // Build a fresh inner from the mirrored fields so concurrent writers
        // can pick up our additions via the merge step in
        // `lamquant_history::History::save_to`.
        let mut inner = self.inner.clone();
        inner.recent_paths.inputs = self.recent_inputs.clone();
        inner.recent_paths.outputs = self.recent_outputs.clone();
        inner.last_op = self.last_op.clone();
        inner.last_input = self.last_input.clone();
        inner.last_output = self.last_output.clone();
        inner.interrupted = self.interrupted;
        let _ = inner.save();
    }

    pub fn add_input(&mut self, p: &str) {
        self.recent_inputs.retain(|x| x != p);
        self.recent_inputs.insert(0, p.to_string());
        self.recent_inputs.truncate(20);
    }
    pub fn add_output(&mut self, p: &str) {
        self.recent_outputs.retain(|x| x != p);
        self.recent_outputs.insert(0, p.to_string());
        self.recent_outputs.truncate(20);
    }

    pub fn mark_running(&mut self, op: &str, input: &str, output: Option<&str>) {
        self.last_op = Some(op.to_string());
        self.last_input = Some(input.to_string());
        self.last_output = output.map(|s| s.to_string());
        self.interrupted = true;
        self.save();
    }

    pub fn mark_complete(&mut self) {
        if let (Some(op), Some(input)) = (self.last_op.clone(), self.last_input.clone()) {
            self.inner.record_op(&op, &input, "ok");
        }
        self.interrupted = false;
        self.save();
    }

    /// Read-only view of the canonical recent_operations log. Useful for
    /// rendering "what did I just do" panels without reaching into `inner`.
    pub fn recent_operations(&self) -> &[HistoryOp] {
        &self.inner.recent_operations
    }

    fn from_inner(inner: lamquant_history::History) -> Self {
        let recent_inputs = inner.recent_paths.inputs.clone();
        let recent_outputs = inner.recent_paths.outputs.clone();
        let last_op = inner.last_op.clone();
        let last_input = inner.last_input.clone();
        let last_output = inner.last_output.clone();
        let interrupted = inner.interrupted;
        Self {
            inner,
            recent_inputs,
            recent_outputs,
            last_op,
            last_input,
            last_output,
            interrupted,
        }
    }
}
