//! Resume offer — shown at startup when `History.interrupted` is true,
//! meaning the previous session ended in the middle of an op without
//! reaching the success path. User can [r]esume, [d]iscard, or quit.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Default)]
pub struct ResumePanel {
    pub op: String,
    pub input: String,
    pub output: String,
    pub action: ResumeAction,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeAction {
    #[default]
    Pending,
    Resume,
    Discard,
}

impl ResumePanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_target(&mut self, op: &str, input: &str, output: Option<&str>) {
        // Tier 4 audit: clamp pathological-length strings from a
        // corrupted history.json so a multi-MB op/input/output
        // doesn't lock the panel render loop.
        const MAX_FIELD: usize = 4096;
        let clamp = |s: &str| -> String {
            if s.len() > MAX_FIELD {
                let mut t = s[..MAX_FIELD].to_string();
                t.push_str("… (truncated)");
                t
            } else {
                s.to_string()
            }
        };
        // Tier 4 audit: warn loudly if a previously-set action
        // hasn't been taken yet -- a race between navigation and
        // re-target was silently dropping the user's confirmed
        // choice.
        if !matches!(self.action, ResumeAction::Pending) {
            eprintln!(
                "  WARNING: ResumePanel.set_target overwriting a pending action ({:?}); \
                 previous user choice is being discarded.",
                self.action
            );
        }
        self.op = clamp(op);
        self.input = clamp(input);
        self.output = output.map(clamp).unwrap_or_default();
        self.action = ResumeAction::Pending;
    }

    /// Returns the user's chosen action, then resets to Pending.
    pub fn take_action(&mut self) -> ResumeAction {
        let a = self.action;
        self.action = ResumeAction::Pending;
        a
    }
}

impl Panel for ResumePanel {
    fn id(&self) -> &str {
        "resume"
    }
    fn title(&self) -> &str {
        "Resume Interrupted Run"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::warning())
            .title(Span::styled(" Interrupted run detected ", theme::warning()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let mut lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Your last LamQuant session ended without completing.",
                theme::heading(),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("    Op:     ", theme::dim()),
                Span::styled(self.op.clone(), theme::highlight()),
            ]),
            Line::from(vec![
                Span::styled("    Input:  ", theme::dim()),
                Span::styled(self.input.clone(), theme::normal()),
            ]),
        ];
        if !self.output.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("    Output: ", theme::dim()),
                Span::styled(self.output.clone(), theme::normal()),
            ]));
        }
        lines.extend([
            Line::from(""),
            Line::from(Span::styled("  Choose:", theme::heading())),
            Line::from(""),
            Line::from(vec![
                Span::styled("    [r] ", theme::key_hint()),
                Span::styled(
                    "Resume — re-run the op with --skip-existing so finished files are kept",
                    theme::normal(),
                ),
            ]),
            Line::from(vec![
                Span::styled("    [d] ", theme::key_hint()),
                Span::styled(
                    "Discard — clear the interrupted state and start fresh",
                    theme::normal(),
                ),
            ]),
            Line::from(vec![
                Span::styled("    [Esc/b] ", theme::key_hint()),
                Span::styled("Decide later (re-prompt next launch)", theme::normal()),
            ]),
            Line::from(""),
            Line::from(Span::styled("  [q] quit", theme::dim())),
        ]);
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            // Tier 4 audit: Enter was previously aliased to [r]
            // Resume. A user dismissing the dialog with reflex
            // Enter could auto-resume an interrupted op and
            // clobber partial outputs without consent. Now: Enter
            // is a no-op (consumed but does nothing); user must
            // explicitly press 'r' to resume or 'd' to discard.
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.action = ResumeAction::Resume;
                PanelAction::Back
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                self.action = ResumeAction::Discard;
                PanelAction::Back
            }
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => PanelAction::Back,
            KeyCode::Char('q') | KeyCode::Char('Q') => PanelAction::Quit,
            // Tier 4 audit: unmatched keys return Ignored not
            // Consumed so global shortcuts (e.g. `?` help) still
            // fire while this modal is on screen.
            _ => PanelAction::Ignored,
        }
    }
}
