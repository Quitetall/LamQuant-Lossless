//! Dialog panel — generic text-only modal/dialog widget.
//!
//! Collapses what used to be a fleet of near-identical panels
//! (info_text, help, exit_confirm, ...) into one configurable
//! struct. Each former panel is now just a `DialogPanel::new(...)`
//! call with the right id/title/body and a small key-binding map.
//!
//! Behavior surface preserved 1:1 with the originals:
//!   * Border style (theme::dim vs theme::warning)
//!   * Title rendered into the border
//!   * Body lines rendered via `Paragraph::new(...).wrap(...)`
//!   * Key bindings — a `Vec<(KeyCode, PanelAction)>` lookup table
//!   * Unmatched keys fall back to a configurable default action
//!     (Ignored by default, so global shortcuts like `?` still fire)
//!
//! Convenience constructors are provided for the previously-dedicated
//! shapes (`info`, `help`, `exit_confirm`) so the call sites in app.rs
//! stay almost identical to before the merge.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

/// Visual style for the dialog border. Maps to a `theme::*` accessor.
#[derive(Debug, Clone, Copy)]
pub enum DialogStyle {
    /// Dim border + heading title — used for help / info dialogs.
    Info,
    /// Yellow warning border + warning title — used for exit_confirm.
    Warning,
}

impl DialogStyle {
    fn border(self) -> Style {
        match self {
            DialogStyle::Info => theme::dim(),
            DialogStyle::Warning => theme::warning(),
        }
    }
    fn title(self) -> Style {
        match self {
            DialogStyle::Info => theme::heading(),
            DialogStyle::Warning => theme::warning(),
        }
    }
}

/// Generic dialog panel — render text, dispatch a small set of keys.
pub struct DialogPanel {
    id: String,
    title: String,
    body: Vec<Line<'static>>,
    style: DialogStyle,
    bindings: Vec<(KeyCode, PanelAction)>,
    /// Action emitted for any key not in `bindings`. Defaults to
    /// `Ignored` so global shortcuts (`?` help, etc.) still fire
    /// while the dialog is on screen.
    fallback: PanelAction,
}

impl DialogPanel {
    /// Fully-explicit constructor. Most callers should prefer the
    /// preset helpers below.
    pub fn new(
        id: &str,
        title: &str,
        body: Vec<Line<'static>>,
        style: DialogStyle,
        bindings: Vec<(KeyCode, PanelAction)>,
        fallback: PanelAction,
    ) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            body,
            style,
            bindings,
            fallback,
        }
    }

    /// Default "info text" dialog — Esc/Backspace/b/Enter → Back,
    /// q → Home. Replaces the old `InfoTextPanel::new` shape.
    pub fn info(id: &str, title: &str, body: Vec<Line<'static>>) -> Self {
        let bindings = vec![
            (KeyCode::Esc, PanelAction::Back),
            (KeyCode::Backspace, PanelAction::Back),
            (KeyCode::Char('b'), PanelAction::Back),
            (KeyCode::Enter, PanelAction::Back),
            (KeyCode::Char('q'), PanelAction::Home),
        ];
        Self::new(id, title, body, DialogStyle::Info, bindings, PanelAction::Ignored)
    }

    /// Help dialog preset — identical bindings to `info` but with
    /// "Help" as the static id/title slot.
    pub fn help() -> Self {
        let body = Self::help_body();
        Self::info("help", "Help", body)
    }

    /// Exit confirmation preset — y/Y/Enter → Quit, n/N/Esc/Backspace → Back.
    /// Fallback is `Consumed` (matches legacy ExitConfirmPanel behavior:
    /// any other key is swallowed so the modal stays put).
    pub fn exit_confirm() -> Self {
        let body = vec![
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
        let bindings = vec![
            (KeyCode::Char('y'), PanelAction::Quit),
            (KeyCode::Char('Y'), PanelAction::Quit),
            (KeyCode::Enter, PanelAction::Quit),
            (KeyCode::Char('n'), PanelAction::Back),
            (KeyCode::Char('N'), PanelAction::Back),
            (KeyCode::Esc, PanelAction::Back),
            (KeyCode::Backspace, PanelAction::Back),
        ];
        Self::new(
            "exit_confirm",
            "Confirm Exit",
            body,
            DialogStyle::Warning,
            bindings,
            PanelAction::Consumed,
        )
    }

    fn help_body() -> Vec<Line<'static>> {
        let section =
            |title: &str| Line::from(Span::styled(format!("  {}", title), theme::highlight()));
        let kv = |key: &str, desc: &str| {
            Line::from(vec![
                Span::styled(format!("    {:<14}", key), theme::key_hint()),
                Span::styled(desc.to_string(), theme::normal()),
            ])
        };
        vec![
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
            section("Remote & launcher"),
            kv("N", "Peers — pick a remote SSH target for ops"),
            kv("lamquant", "Launch this TUI (default front door)"),
            kv("lamquant --gui", "Launch the desktop GUI instead"),
            kv("lamquant <cmd>", "Pass through to `lml <cmd>`"),
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
        ]
    }
}

impl Panel for DialogPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        &self.title
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.style.border())
            .title(Span::styled(format!(" {} ", self.title), self.style.title()));
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(
            Paragraph::new(self.body.clone()).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        for (code, action) in &self.bindings {
            if *code == event.code {
                return action.clone();
            }
        }
        self.fallback.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn info_back_on_esc() {
        let mut d = DialogPanel::info("x", "X", vec![]);
        let st = AppState::default();
        assert!(matches!(d.handle_event(key(KeyCode::Esc), &st), PanelAction::Back));
    }

    #[test]
    fn info_home_on_q() {
        let mut d = DialogPanel::info("x", "X", vec![]);
        let st = AppState::default();
        assert!(matches!(
            d.handle_event(key(KeyCode::Char('q')), &st),
            PanelAction::Home
        ));
    }

    #[test]
    fn info_unmatched_key_is_ignored() {
        let mut d = DialogPanel::info("x", "X", vec![]);
        let st = AppState::default();
        assert!(matches!(
            d.handle_event(key(KeyCode::Char('z')), &st),
            PanelAction::Ignored
        ));
    }

    #[test]
    fn exit_confirm_y_quits() {
        let mut d = DialogPanel::exit_confirm();
        let st = AppState::default();
        assert!(matches!(
            d.handle_event(key(KeyCode::Char('y')), &st),
            PanelAction::Quit
        ));
        assert!(matches!(
            d.handle_event(key(KeyCode::Enter), &st),
            PanelAction::Quit
        ));
    }

    #[test]
    fn exit_confirm_n_backs() {
        let mut d = DialogPanel::exit_confirm();
        let st = AppState::default();
        assert!(matches!(
            d.handle_event(key(KeyCode::Char('n')), &st),
            PanelAction::Back
        ));
        assert!(matches!(
            d.handle_event(key(KeyCode::Esc), &st),
            PanelAction::Back
        ));
    }

    #[test]
    fn exit_confirm_unmatched_consumed() {
        // Legacy ExitConfirmPanel returned Consumed (not Ignored) for
        // unmatched keys so the modal stays put. Preserve that.
        let mut d = DialogPanel::exit_confirm();
        let st = AppState::default();
        assert!(matches!(
            d.handle_event(key(KeyCode::Char('z')), &st),
            PanelAction::Consumed
        ));
    }
}
