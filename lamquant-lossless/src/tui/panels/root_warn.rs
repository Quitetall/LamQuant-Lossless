//! Root user warning — shown at startup when running as uid 0 and warn_root=true.
//! User can: continue (e), quit (q), or toggle allow_root (a) to silence forever.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Default)]
pub struct RootWarnPanel {
    pub silence_pressed: bool,
}

impl RootWarnPanel {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn take_silence(&mut self) -> bool {
        let r = self.silence_pressed;
        self.silence_pressed = false;
        r
    }
}

impl Panel for RootWarnPanel {
    fn id(&self) -> &str {
        "root_warn"
    }
    fn title(&self) -> &str {
        "Running as root"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::warning())
            .title(Span::styled(" Running as root ", theme::warning()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  You're running LamQuant as root (uid 0).",
                theme::heading(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  This is rarely necessary. Output files written here will be owned",
                theme::normal(),
            )),
            Line::from(Span::styled(
                "  by root, and a stray rm -rf could nuke far more than you intend.",
                theme::normal(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Recommended: re-run as your normal user. If you must be root,",
                theme::dim(),
            )),
            Line::from(Span::styled(
                "  press [e] to continue (still safe — files just owned by root).",
                theme::dim(),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("    [e] ", theme::key_hint()),
                Span::styled("Continue this session  ", theme::normal()),
            ]),
            Line::from(vec![
                Span::styled("    [a] ", theme::key_hint()),
                Span::styled(
                    "Allow root permanently (sets output.allow_root=true)  ",
                    theme::normal(),
                ),
            ]),
            Line::from(vec![
                Span::styled("    [q] ", theme::key_hint()),
                Span::styled("Quit and re-run as a normal user", theme::normal()),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            // Continue this session — primary affirmative.
            KeyCode::Char('e')
            | KeyCode::Char('E')
            | KeyCode::Char('y')
            | KeyCode::Char('Y')
            | KeyCode::Enter => PanelAction::Back,
            // Permanently allow root.
            KeyCode::Char('a') | KeyCode::Char('A') => {
                self.silence_pressed = true;
                PanelAction::Back
            }
            // Esc / Backspace = cancel the action = continue session (matches exit_confirm pattern).
            KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            // Explicit quit.
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Char('n') | KeyCode::Char('N') => {
                PanelAction::Quit
            }
            // Tier 4 audit: unmatched keys return Ignored so
            // global shortcuts (`?` help, etc.) still fire while
            // the warning modal is on screen. Pre-fix `Consumed`
            // blocked them.
            _ => PanelAction::Ignored,
        }
    }
}
