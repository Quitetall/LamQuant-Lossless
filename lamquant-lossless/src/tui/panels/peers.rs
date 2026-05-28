//! Peers panel — lists configured remote peers, shows health, lets the
//! user pick a sticky target for outgoing ops.
//!
//! Reads `state.peers` (loaded once at App::new from peers.json).
//! Writes `state.selected_peer` via `PanelAction::StatusMessage` →
//! handled by App tick that mirrors panel intent (no direct mutation
//! since render is &self and event handler is &mut self but state is
//! &AppState immutable). For now we expose a hidden setter via
//! StatusMessage convention until a typed `Action::SelectPeer` lands
//! in commit 6.
//!
//! Peers route reaches `SCREEN_PEERS` (added to router.rs in this
//! commit). Hub tile lives in main_hub.rs additions (key `[N]` chosen
//! to avoid clashes with existing `[1]`–`[5]`/`[s]`/`[i]`/`[t]`).

use crossterm::event::{KeyCode, KeyEvent};
use lamquant_ops::{PeerHealth, SshTransport, Transport, TransportError, TransportKind};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

/// In-memory health snapshot keyed by peer.id. Refresh on `[r]` press
/// (sync, blocks the TUI for one SSH round-trip per peer — fine for
/// LAN; future P-async could move probes to a worker thread).
#[derive(Default)]
pub struct PeersPanel {
    selected: usize,
    /// Cached health: (peer_id, last result). Rebuilt on refresh or
    /// first render. Health probes happen only on explicit refresh —
    /// not every render, since each probe spawns SSH.
    health: Vec<(String, PeerHealth)>,
    /// Last status line shown at footer. Cleared on next interaction.
    status: Option<String>,
}

impl PeersPanel {
    pub fn new() -> Self {
        Self::default()
    }

    fn refresh_health(&mut self, peers: &[lamquant_ops::Peer]) {
        let transport = SshTransport::new();
        self.health.clear();
        for peer in peers {
            let h = match &peer.transport {
                TransportKind::Ssh(_) => match transport.health(peer) {
                    Ok(h) => h,
                    Err(TransportError::VersionMismatch { local, peer: pv }) => {
                        PeerHealth::IncompatibleVersion { local, peer: pv }
                    }
                    Err(_) => PeerHealth::Unreachable,
                },
            };
            self.health.push((peer.id.clone(), h));
        }
    }

    fn health_for<'a>(&'a self, peer_id: &str) -> Option<&'a PeerHealth> {
        self.health
            .iter()
            .find(|(id, _)| id == peer_id)
            .map(|(_, h)| h)
    }
}

impl Panel for PeersPanel {
    fn id(&self) -> &str {
        "peers"
    }
    fn title(&self) -> &str {
        "Peers"
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(Span::styled(" PEERS ", theme::highlight()))
            .borders(Borders::ALL)
            .border_style(theme::dim());
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // header
                Constraint::Min(0),    // peer list
                Constraint::Length(2), // status / hint
            ])
            .split(inner);

        // Header: how many peers + sticky indicator.
        let sticky = state.selected_peer.as_deref().unwrap_or("(local)");
        let header = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{} peers configured", state.peers.len()),
                    theme::dim(),
                ),
                Span::raw("    "),
                Span::styled("→ ", theme::dim()),
                Span::styled(sticky, theme::highlight()),
            ]),
            Line::from(""),
        ];
        f.render_widget(Paragraph::new(header), chunks[0]);

        // Peer list.
        if state.peers.is_empty() {
            let empty = vec![
                Line::from(""),
                Line::from(Span::styled("  No peers configured.", theme::dim())),
                Line::from(""),
                Line::from(Span::styled(
                    "  Add ~/.config/lamquant/peers.json with [[peers]] entries.",
                    theme::dim(),
                )),
            ];
            f.render_widget(Paragraph::new(empty), chunks[1]);
        } else {
            let lines: Vec<Line> = state
                .peers
                .iter()
                .enumerate()
                .map(|(i, peer)| {
                    let cursor = if i == self.selected { "▶ " } else { "  " };
                    let cursor_style = if i == self.selected {
                        theme::highlight()
                    } else {
                        theme::dim()
                    };
                    let sticky_mark = if state.selected_peer.as_deref() == Some(peer.id.as_str()) {
                        Span::styled(" ★", theme::highlight())
                    } else {
                        Span::raw("  ")
                    };
                    let (health_glyph, health_style) = match self.health_for(&peer.id) {
                        Some(PeerHealth::Idle) => ("●", theme::success()),
                        Some(PeerHealth::Busy(_)) => ("◐", theme::warning()),
                        Some(PeerHealth::Unreachable) => ("✗", theme::error()),
                        Some(PeerHealth::IncompatibleVersion { .. }) => ("⚠", theme::error()),
                        None => ("○", theme::dim()), // not probed yet
                    };
                    let detail = match &peer.transport {
                        TransportKind::Ssh(cfg) => {
                            format!("ssh {}@{}:{}", cfg.user, peer.host, cfg.port)
                        }
                    };
                    Line::from(vec![
                        Span::styled(cursor, cursor_style),
                        Span::styled(health_glyph, health_style),
                        Span::raw(" "),
                        Span::styled(format!("{:<14}", peer.display), theme::normal()),
                        sticky_mark,
                        Span::raw("  "),
                        Span::styled(detail, theme::dim()),
                    ])
                })
                .collect();
            f.render_widget(Paragraph::new(lines), chunks[1]);
        }

        // Status / key hints.
        let hint = if let Some(s) = &self.status {
            Line::from(Span::styled(format!("  {}", s), theme::warning()))
        } else {
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[↵]", theme::key_hint()),
                Span::raw(" sticky-set   "),
                Span::styled("[c]", theme::key_hint()),
                Span::raw(" clear (local)   "),
                Span::styled("[r]", theme::key_hint()),
                Span::raw(" refresh   "),
                Span::styled("[b]", theme::key_hint()),
                Span::raw(" back   "),
                Span::styled("[q]", theme::key_hint()),
                Span::raw(" main"),
            ])
        };
        f.render_widget(Paragraph::new(hint), chunks[2]);
    }

    fn handle_event(&mut self, event: KeyEvent, state: &AppState) -> PanelAction {
        // Clear any prior status on next interaction.
        if self.status.is_some() {
            self.status = None;
        }
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !state.peers.is_empty() && self.selected + 1 < state.peers.len() {
                    self.selected += 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                // Sticky-set via the typed Action::SelectPeer (C.2b).
                // Empty id is impossible here: peers_config::load
                // filters those out before they reach state.peers.
                match state.peers.get(self.selected) {
                    Some(peer) => PanelAction::SelectPeer(Some(peer.id.clone())),
                    None => PanelAction::Consumed,
                }
            }
            KeyCode::Char('c') => {
                // Clear sticky → fall back to local.
                PanelAction::SelectPeer(None)
            }
            KeyCode::Char('r') => {
                self.refresh_health(&state.peers);
                self.status = Some(format!("Probed {} peer(s).", state.peers.len()));
                PanelAction::Consumed
            }
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            KeyCode::Char('h') | KeyCode::Char('?') => {
                PanelAction::Navigate(crate::tui::router::SCREEN_HELP.to_string())
            }
            _ => PanelAction::Ignored,
        }
    }
}
