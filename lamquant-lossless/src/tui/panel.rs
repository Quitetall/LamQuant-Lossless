//! Panel trait — the core abstraction for all TUI widgets.
//!
//! Every visible component implements Panel. The framework handles:
//!   - Rendering (calls `render` with the assigned Rect)
//!   - Key dispatch (calls `handle_event` when panel is focused)
//!   - Focus management (calls `can_focus` to determine tab order)
//!
//! Panels are stateful — they own their data and update on events.
//!
//! ## Action vs PanelAction (commit 7+ of reactive-store refactor)
//!
//! Two separate enums exist by design:
//! - [`PanelAction`] — what panels are allowed to emit. 9 variants, no
//!   App-internal commands. Returned by [`Panel::handle_event`].
//! - [`Action`] — the umbrella App's `dispatch` accepts. Includes every
//!   PanelAction variant (via `From<PanelAction>`) plus three App-internal
//!   variants the panels cannot construct: `Tick`, `SetStatus`, `BackOne`.
//!
//! This separation makes it impossible for a panel to accidentally emit
//! `Tick` (which would otherwise run reducer animation logic at user-key
//! cadence). Was previously a single Action enum with a doc comment
//! warning panels not to use the App-internal variants — type system
//! now enforces it.

use crossterm::event::KeyEvent;
use ratatui::prelude::*;
use serde::{Deserialize, Serialize};

use super::operations::OpEvent;
use super::state::AppState;

/// Subset of [`Action`] that panels are allowed to emit. Returned by
/// [`Panel::handle_event`]. App converts to `Action` via `From` impl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelAction {
    /// Event consumed, no further action needed.
    Consumed,
    /// Panel wants to navigate to a different screen.
    Navigate(String),
    /// Panel wants to go back (pop router stack).
    Back,
    /// Panel wants to quit the application (routes through exit_confirm).
    Quit,
    /// Panel wants to return to the main menu (q key — "home").
    Home,
    /// Panel wants to run an operation (by ID) with given args.
    RunOperation { op_id: String, args: Vec<String> },
    /// Panel didn't handle the event — pass to next handler.
    Ignored,
    /// Panel wants to show a message in the status bar.
    StatusMessage(String),
    /// Panel submits a value (file path, text input, etc.) to the app.
    Submit(String),
    /// Set or clear the sticky peer. Some(id) makes that peer the
    /// default target for outgoing ops; None clears (run local).
    /// Replaces the "__select_peer:<id>" StatusMessage sentinel
    /// shipped in commit 5 of the multi-device chain.
    SelectPeer(Option<String>),
    /// Settings panel: write `cfg.compute.workers`. 0 = auto-detect.
    SetCfgWorkers(i64),
    /// Settings panel: write `cfg.backend.mode`.
    SetCfgBackend(String),
    /// Settings panel: write `cfg.codec.verification`.
    SetCfgVerification(String),
}

/// Umbrella action processed by `App::dispatch`. Includes every
/// [`PanelAction`] (via `From`) plus three App-internal variants that
/// panels cannot construct.
///
/// Tagged Serialize for the GUI dispatch wire format
/// (see `gui/src-tauri/src/state_bridge.rs::dispatch`). Variants
/// requiring runtime-only payloads (`Tick`, `OpEvent`) are still
/// serializable but the GUI side typically only emits the panel-safe
/// subset; App-internal variants are reserved for the TUI's own use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    // ── Panel-equivalent variants ──────────────────────────────────────
    Consumed,
    Navigate(String),
    Back,
    Quit,
    Home,
    RunOperation {
        op_id: String,
        args: Vec<String>,
    },
    Ignored,
    StatusMessage(String),
    Submit(String),
    SelectPeer(Option<String>),
    /// Settings panel: write `cfg.compute.workers`. 0 = auto-detect.
    /// Mutation is applied immediately + persisted to lamquant.toml.
    SetCfgWorkers(i64),
    /// Settings panel: write `cfg.backend.mode` (auto/rust/python).
    SetCfgBackend(String),
    /// Settings panel: write `cfg.codec.verification` (paranoid/standard/fast).
    SetCfgVerification(String),

    // ── App-internal (panels cannot emit) ──────────────────────────────
    /// Per-frame tick; reducer advances animation counters and drains timeouts.
    Tick,
    /// Direct status set without panel attribution. Used by App's launcher
    /// branches that previously called `state.set_status(...)` ad hoc.
    SetStatus(String),
    /// Pop one screen without the conditional `pending_op = None` cleanup
    /// that `Action::Back` carries — used internally by `execute_pending`.
    BackOne,
    /// One drained subprocess event from a running op (P3 of refactor).
    /// App.tick_panels drains the OpReceiver and emits these so all
    /// state mutations (log lines, progress, terminal flags) flow through
    /// the dispatch chokepoint.
    OpEvent(OpEvent),
}

impl From<PanelAction> for Action {
    fn from(pa: PanelAction) -> Self {
        match pa {
            PanelAction::Consumed => Action::Consumed,
            PanelAction::Navigate(s) => Action::Navigate(s),
            PanelAction::Back => Action::Back,
            PanelAction::Quit => Action::Quit,
            PanelAction::Home => Action::Home,
            PanelAction::RunOperation { op_id, args } => Action::RunOperation { op_id, args },
            PanelAction::Ignored => Action::Ignored,
            PanelAction::StatusMessage(s) => Action::StatusMessage(s),
            PanelAction::Submit(s) => Action::Submit(s),
            PanelAction::SelectPeer(id) => Action::SelectPeer(id),
            PanelAction::SetCfgWorkers(v) => Action::SetCfgWorkers(v),
            PanelAction::SetCfgBackend(v) => Action::SetCfgBackend(v),
            PanelAction::SetCfgVerification(v) => Action::SetCfgVerification(v),
        }
    }
}

/// The core panel trait — implement this for any TUI widget.
pub trait Panel {
    /// Unique identifier for this panel type.
    fn id(&self) -> &str;

    /// Render the panel into the given area.
    fn render(&self, state: &AppState, f: &mut Frame, area: Rect);

    /// Handle a key event. Returns a [`PanelAction`] (panel-safe subset
    /// of [`Action`]). App converts to Action and dispatches.
    fn handle_event(&mut self, event: KeyEvent, state: &AppState) -> PanelAction;

    /// Whether this panel can receive focus (for tab navigation).
    fn can_focus(&self) -> bool {
        true
    }

    /// Called when the panel gains focus.
    fn on_focus(&mut self) {}

    /// Called when the panel loses focus.
    fn on_blur(&mut self) {}

    /// Title shown in the panel border (if bordered).
    fn title(&self) -> &str {
        self.id()
    }

    /// Optional tick — called every frame for animations/updates.
    fn tick(&mut self) {}
}
