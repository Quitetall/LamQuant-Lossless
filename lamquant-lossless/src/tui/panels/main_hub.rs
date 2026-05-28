//! Main hub — workflow + system tile list (items only).
//!
//! Dividers, hint bar, and the live sidebar are rendered by app.rs so they
//! have access to runtime state (op handle, history, EEG placeholder, etc.).
//! This panel owns selection state and key dispatch only.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Debug, Clone)]
pub struct HubTile {
    pub key: String,
    pub label: String,
    pub desc: String,
    pub target: String,
    /// Short right-aligned status tag shown in dim on the row.
    pub tag: String,
}

impl HubTile {
    pub fn new(key: &str, label: &str, desc: &str, target: &str) -> Self {
        Self::with_tag(key, label, desc, target, "")
    }

    pub fn with_tag(key: &str, label: &str, desc: &str, target: &str, tag: &str) -> Self {
        Self {
            key: key.to_string(),
            label: label.to_string(),
            desc: desc.to_string(),
            target: target.to_string(),
            tag: tag.to_string(),
        }
    }
}

pub struct MainHubPanel {
    id: String,
    pub workflows: Vec<HubTile>,
    pub system: Vec<HubTile>,
    selected: usize,
}

impl MainHubPanel {
    pub fn new(id: &str, workflows: Vec<HubTile>, system: Vec<HubTile>) -> Self {
        Self {
            id: id.to_string(),
            workflows,
            system,
            selected: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.workflows.len() + self.system.len()
    }

    pub fn target_at(&self, idx: usize) -> Option<&str> {
        if idx < self.workflows.len() {
            Some(&self.workflows[idx].target)
        } else {
            self.system
                .get(idx - self.workflows.len())
                .map(|t| t.target.as_str())
        }
    }

    pub fn target_for_key(&self, key: &str) -> Option<&str> {
        self.workflows
            .iter()
            .chain(self.system.iter())
            .find(|t| t.key == key)
            .map(|t| t.target.as_str())
    }

    /// Exact number of rows this panel needs (used by app.rs to size the body chunk).
    pub fn body_height(&self) -> u16 {
        // 1 blank + header + 1 blank + N workflow rows
        // + 1 blank + header + 1 blank + N system rows
        // + 1 trailing blank
        (3 + self.workflows.len() + 3 + self.system.len() + 1) as u16
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    fn item_line(tile: &HubTile, selected: bool, row_w: u16) -> Line<'static> {
        // ASCII '>' marker in magenta — distinct from cyan headers, yellow warnings.
        let marker = if selected {
            Span::styled(" > ", Style::default().fg(Color::Magenta))
        } else {
            Span::raw("   ")
        };
        let label_style = if selected {
            theme::highlight()
        } else {
            theme::normal()
        };

        // Right-align tag: pad label+desc so tag sits at the row edge.
        // Simple approach: fixed label width, then desc, then right-padded tag.
        let tag_w = tile.tag.chars().count();
        let left = format!("[{}]  {:<20}  {:<28}", tile.key, tile.label, tile.desc,);
        let left_w = left.chars().count() + 3; // 3 for marker
        let pad = (row_w as usize).saturating_sub(left_w + tag_w + 2);
        let padding = " ".repeat(pad);

        let mut spans = vec![
            marker,
            Span::styled(format!("[{}]", tile.key), theme::key_hint()),
            Span::raw("  "),
            Span::styled(format!("{:<20}", tile.label), label_style),
            Span::raw("  "),
            Span::styled(format!("{:<28}", tile.desc), theme::dim()),
            Span::raw(padding),
        ];
        if !tile.tag.is_empty() {
            spans.push(Span::styled(tile.tag.clone(), theme::dim()));
            spans.push(Span::raw("  "));
        }
        Line::from(spans)
    }
}

impl Panel for MainHubPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "LamQuant"
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        let w = area.width;
        let wf = self.workflows.len();
        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled("WORKFLOWS", theme::highlight()),
        ]));
        lines.push(Line::from(""));
        for (i, tile) in self.workflows.iter().enumerate() {
            lines.push(Self::item_line(tile, i == self.selected, w));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled("SYSTEM", theme::highlight()),
        ]));
        lines.push(Line::from(""));
        for (i, tile) in self.system.iter().enumerate() {
            lines.push(Self::item_line(tile, wf + i == self.selected, w));
        }
        lines.push(Line::from(""));

        // GUI discovery hint: shown only while ui_preference is unset
        // ("ask"). The wizard clears it on save (sets to "tui") so the
        // hint goes away once the user has expressed any UI preference.
        // Keeps onboarding visible without nagging returning users.
        if state.cfg.ui_preference.trim().eq_ignore_ascii_case("ask") {
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled("tip:", theme::dim()),
                Span::raw(" "),
                Span::styled("lamquant --gui", theme::key_hint()),
                Span::styled(
                    "  opens the desktop GUI (or set LAMQUANT_UI=gui)",
                    theme::dim(),
                ),
            ]));
            lines.push(Line::from(""));
        }

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.total() {
                    self.selected += 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                if let Some(t) = self.target_at(self.selected) {
                    PanelAction::Navigate(t.to_string())
                } else {
                    PanelAction::Consumed
                }
            }
            KeyCode::Char(c) => {
                let key = c.to_string();
                if let Some(t) = self.target_for_key(&key) {
                    return PanelAction::Navigate(t.to_string());
                }
                match c {
                    'x' => PanelAction::Quit,
                    'q' => PanelAction::Home,
                    'b' => PanelAction::Back,
                    'h' | '?' => PanelAction::Navigate(router::SCREEN_HELP.to_string()),
                    _ => PanelAction::Ignored,
                }
            }
            // Esc on the root hub routes to exit-confirm (not Back —
            // Back at root is a no-op, leaving Esc unresponsive).
            // Backspace stays as Back so users can still go up from
            // any sub-screen with the same key.
            KeyCode::Esc => PanelAction::Quit,
            KeyCode::Backspace => PanelAction::Back,
            _ => PanelAction::Ignored,
        }
    }
}
