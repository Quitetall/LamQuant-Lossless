//! Exit confirmation dialog.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

pub struct ExitConfirmPanel;

impl ExitConfirmPanel {
    pub fn new() -> Self {
        Self
    }
}
impl Default for ExitConfirmPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel for ExitConfirmPanel {
    fn id(&self) -> &str {
        "exit_confirm"
    }
    fn title(&self) -> &str {
        "Confirm Exit"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::warning())
            .title(Span::styled(" Confirm Exit ", theme::warning()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let lines = vec![
            Line::from(""),
            Line::from(Span::styled("    Quit LamQuant?", theme::heading())),
            Line::from(""),
            Line::from(vec![
                Span::styled("      [y / Enter] ", theme::key_hint()),
                Span::styled("Yes, quit", theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("      [n / Esc]   ", theme::key_hint()),
                Span::styled("No, stay", theme::normal()),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => PanelAction::Quit,
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Backspace => {
                PanelAction::Back
            }
            _ => PanelAction::Consumed,
        }
    }
}
