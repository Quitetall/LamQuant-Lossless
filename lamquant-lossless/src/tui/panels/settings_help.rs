//! Settings help — full-screen multi-line help for a single setting.
//!
//! Populated by app.rs when the user presses `?` from SettingsPanel. The app
//! reads pending_help from SettingsPanel and calls `set_target()` here.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Default)]
pub struct SettingsHelpPanel {
    label: String,
    dotpath: String,
    help: String,
    desc: String,
    section: String,
    current_value: String,
}

impl SettingsHelpPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_target(
        &mut self,
        section: &str,
        label: &str,
        dotpath: &str,
        desc: &str,
        help: &str,
        current_value: &str,
    ) {
        self.section = section.into();
        self.label = label.into();
        self.dotpath = dotpath.into();
        self.desc = desc.into();
        self.help = help.into();
        self.current_value = current_value.into();
    }
}

impl Panel for SettingsHelpPanel {
    fn id(&self) -> &str {
        "settings_help"
    }
    fn title(&self) -> &str {
        "Setting Help"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(
                format!(" Help — {} ", self.dotpath),
                theme::heading(),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines: Vec<Line<'static>> = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Section:  ", theme::dim()),
                Span::styled(self.section.clone(), theme::title()),
            ]),
            Line::from(vec![
                Span::styled("  Label:    ", theme::dim()),
                Span::styled(self.label.clone(), theme::heading()),
            ]),
            Line::from(vec![
                Span::styled("  Dotpath:  ", theme::dim()),
                Span::styled(self.dotpath.clone(), theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("  Current:  ", theme::dim()),
                Span::styled(self.current_value.clone(), theme::highlight()),
            ]),
            Line::from(vec![
                Span::styled("  Summary:  ", theme::dim()),
                Span::styled(self.desc.clone(), theme::normal()),
            ]),
            Line::from(""),
            Line::from(Span::styled("  Description", theme::title())),
            Line::from(Span::styled("  ───────────", theme::dim())),
        ];
        for raw in self.help.lines() {
            lines.push(Line::from(Span::styled(
                format!("  {}", raw),
                theme::normal(),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  [Esc/b/Enter] Back",
            theme::dim(),
        )));

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
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
