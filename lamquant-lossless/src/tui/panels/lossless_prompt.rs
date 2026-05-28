//! Lossless compress sub-prompt — `[a] LMA archive` vs `[s] LML +
//! siblings (copy)`.
//!
//! Pushed by `mode_panel` when the user presses `[1] Compress` in
//! Lossless mode AND `codec.lossless_default_mode = "prompt"`.
//! When the setting is `"lma"` or `"lml_siblings"` mode_panel
//! dispatches straight to the corresponding op-id and this panel
//! is bypassed.
//!
//! Selection navigates to the appropriate op screen
//! (`OP_ENCODE_LMA` or `OP_ENCODE_LML_SIBLINGS`); `[Esc]` / `[b]`
//! pops back to the Lossless main panel.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Default)]
pub struct LosslessPromptPanel;

impl LosslessPromptPanel {
    pub fn new() -> Self {
        Self
    }
}

impl Panel for LosslessPromptPanel {
    fn id(&self) -> &str {
        router::SCREEN_LOSSLESS_PROMPT
    }
    fn title(&self) -> &str {
        "Choose Lossless mode"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(
                " Choose Lossless mode ",
                theme::heading(),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let opt = |key: &'static str, label: &'static str, desc: &'static str| {
            Line::from(vec![
                Span::raw("   "),
                Span::styled(format!("[{}]", key), theme::key_hint()),
                Span::raw("  "),
                Span::styled(format!("{:<28}", label), theme::normal()),
                Span::styled(desc, theme::dim()),
            ])
        };
        let dim_line = |s: &'static str| Line::from(Span::styled(format!("  {}", s), theme::dim()));

        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    "How do you want to compress this lossless run?",
                    theme::heading(),
                ),
            ]),
            Line::from(""),
            opt(
                "a",
                "LMA archive",
                "everything → one .lma (siblings inside, smallest)",
            ),
            opt(
                "s",
                "LML + siblings (copy)",
                "per-EEG .lml + non-EEG files copied alongside",
            ),
            Line::from(""),
            dim_line("Tip: set Lossless default mode in Settings to skip this prompt."),
            Line::from(""),
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[Esc]", theme::key_hint()),
                Span::styled(" cancel   ", theme::dim()),
                Span::styled("[b]", theme::key_hint()),
                Span::styled(" back   ", theme::dim()),
                Span::styled("[q]", theme::key_hint()),
                Span::styled(" Main menu   ", theme::dim()),
                Span::styled("[x]", theme::key_hint()),
                Span::styled(" Exit", theme::dim()),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('a') | KeyCode::Char('A') => {
                PanelAction::Navigate(router::OP_ENCODE_LMA.to_string())
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                PanelAction::Navigate(router::OP_ENCODE_LML_SIBLINGS.to_string())
            }
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            _ => PanelAction::Ignored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn key_a_routes_to_lma() {
        let mut p = LosslessPromptPanel::new();
        let st = AppState::default();
        match p.handle_event(key('a'), &st) {
            PanelAction::Navigate(t) => assert_eq!(t, router::OP_ENCODE_LMA),
            other => panic!("expected Navigate(OP_ENCODE_LMA), got {:?}", other),
        }
        // Uppercase also accepted -- shift-A shouldn't dead-end.
        match p.handle_event(key('A'), &st) {
            PanelAction::Navigate(t) => assert_eq!(t, router::OP_ENCODE_LMA),
            other => panic!("expected Navigate, got {:?}", other),
        }
    }

    #[test]
    fn key_s_routes_to_lml_siblings() {
        let mut p = LosslessPromptPanel::new();
        let st = AppState::default();
        match p.handle_event(key('s'), &st) {
            PanelAction::Navigate(t) => assert_eq!(t, router::OP_ENCODE_LML_SIBLINGS),
            other => panic!("expected Navigate(OP_ENCODE_LML_SIBLINGS), got {:?}", other),
        }
    }

    #[test]
    fn key_esc_pops_back() {
        let mut p = LosslessPromptPanel::new();
        let st = AppState::default();
        let key_esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert!(matches!(p.handle_event(key_esc, &st), PanelAction::Back));
    }
}
