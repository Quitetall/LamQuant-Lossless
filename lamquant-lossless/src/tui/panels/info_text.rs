//! Info text panel — static text screen for Train/Eagle/Setup placeholders.
//! Shows guidance + CLI command to run that workflow.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

pub struct InfoTextPanel {
    id: String,
    title: String,
    body: Vec<Line<'static>>,
}

impl InfoTextPanel {
    pub fn new(id: &str, title: &str, body: Vec<Line<'static>>) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            body,
        }
    }
}

impl Panel for InfoTextPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        &self.title
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(format!(" {} ", self.title), theme::heading()));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(self.body.clone()).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') | KeyCode::Enter => {
                PanelAction::Back
            }
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }
}
