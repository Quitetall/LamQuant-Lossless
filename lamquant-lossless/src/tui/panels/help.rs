//! Help panel — keyboard shortcuts + quick reference.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

pub struct HelpPanel;

impl HelpPanel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HelpPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel for HelpPanel {
    fn id(&self) -> &str {
        "help"
    }
    fn title(&self) -> &str {
        "Help"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(" LamQuant — Help ", theme::heading()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let section =
            |title: &str| Line::from(Span::styled(format!("  {}", title), theme::highlight()));
        let kv = |key: &str, desc: &str| {
            Line::from(vec![
                Span::styled(format!("    {:<14}", key), theme::key_hint()),
                Span::styled(desc.to_string(), theme::normal()),
            ])
        };

        let lines = vec![
            Line::from(""),
            section("Navigation"),
            kv("↑ ↓ / j k", "Move selection"),
            kv("Enter / →", "Open / select"),
            kv("Esc / b / ←", "Go back"),
            kv("q / x", "Quit"),
            kv("Ctrl+C", "Force quit (any screen)"),
            Line::from(""),
            section("Operations"),
            kv("encode", "EDF → .lml lossless compression"),
            kv("decode", ".lml → raw int32 LE binary"),
            kv("verify", "CRC-32 integrity check"),
            kv("info", "Header + per-channel metadata"),
            kv("stats", "Per-channel signal statistics"),
            kv("archive", "Pack directory → .lma"),
            kv("extract", ".lma → directory"),
            Line::from(""),
            section("File browser"),
            kv("↑↓ jk", "Navigate entries"),
            kv("Enter / l", "Enter dir / select file"),
            kv("Backspace / ←", "Parent dir"),
            kv(".", "Toggle hidden files"),
            Line::from(""),
            section("Settings"),
            kv("↑↓ jk", "Move selection"),
            kv("← →", "Toggle bool / cycle enum"),
            kv("Enter", "Inline-edit int / float / string"),
            kv("/", "Search by key, dotpath, or description"),
            kv("?", "Full help for current setting"),
            kv("s  /  r", "Save  /  reset to defaults"),
            kv("Esc / b", "Back (warns if dirty)"),
            Line::from(""),
            section("CLI mode"),
            Line::from(Span::styled("    Run with args to skip TUI:", theme::dim())),
            Line::from(Span::styled(
                "      lml encode file.edf -o out.lml --verify",
                theme::normal(),
            )),
            Line::from(Span::styled("      lml verify out.lml", theme::normal())),
            Line::from(Span::styled("      lml --help", theme::normal())),
            Line::from(""),
            Line::from(Span::styled("  [Esc/b/Enter] Back", theme::dim())),
        ];

        f.render_widget(Paragraph::new(lines), inner);
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
