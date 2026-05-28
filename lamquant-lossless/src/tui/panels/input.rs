//! Input panel — single-line text input backed by ratatui-textarea.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::*;
use ratatui_textarea::{CursorMove, Input, Key, TextArea};

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

fn to_textarea_input(event: KeyEvent) -> Input {
    let ctrl = event.modifiers.contains(KeyModifiers::CONTROL);
    let alt = event.modifiers.contains(KeyModifiers::ALT);
    let shift = event.modifiers.contains(KeyModifiers::SHIFT);
    let key = match event.code {
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Enter => Key::Enter,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Tab => Key::Tab,
        KeyCode::BackTab => Key::Tab,
        KeyCode::Delete => Key::Delete,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Esc => Key::Esc,
        KeyCode::F(n) => Key::F(n),
        _ => Key::Null,
    };
    Input {
        key,
        ctrl,
        alt,
        shift,
    }
}

pub struct InputPanel {
    textarea: TextArea<'static>,
    prompt: String,
    hint: String,
}

impl InputPanel {
    pub fn new(prompt: &str) -> Self {
        let mut ta = TextArea::default();
        ta.set_cursor_line_style(Style::default());
        ta.set_block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::highlight()),
        );
        ta.set_max_histories(0); // disable undo for single-line path input
        Self {
            textarea: ta,
            prompt: prompt.to_string(),
            hint: String::new(),
        }
    }

    pub fn value(&self) -> &str {
        self.textarea
            .lines()
            .first()
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    pub fn clear(&mut self) {
        let hint = self.hint.clone();
        let prompt = self.prompt.clone();
        *self = Self::new(&prompt);
        self.hint = hint;
    }

    pub fn set_prompt(&mut self, prompt: &str) {
        self.prompt = prompt.to_string();
    }

    pub fn set_value(&mut self, value: &str) {
        self.textarea = TextArea::new(vec![value.to_string()]);
        self.textarea.set_cursor_line_style(Style::default());
        self.textarea.set_block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme::highlight()),
        );
        self.textarea.set_max_histories(0);
        self.textarea.move_cursor(CursorMove::End);
    }

    pub fn set_hint(&mut self, hint: &str) {
        self.hint = hint.to_string();
    }
}

impl Panel for InputPanel {
    fn id(&self) -> &str {
        "input"
    }
    fn title(&self) -> &str {
        &self.prompt
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(1), // prompt
                Constraint::Length(3), // textarea
                Constraint::Length(1), // hints
                Constraint::Min(0),
            ])
            .split(area);

        f.render_widget(
            Paragraph::new(Span::styled(&self.prompt, theme::heading())),
            chunks[0],
        );

        f.render_widget(&self.textarea, chunks[1]);

        let mut hint_spans = vec![
            Span::styled(" Enter", theme::key_hint()),
            Span::styled(" submit  ", theme::dim()),
            Span::styled("Esc", theme::key_hint()),
            Span::styled(" cancel  ", theme::dim()),
            Span::styled("Ctrl+U", theme::key_hint()),
            Span::styled(" clear", theme::dim()),
        ];
        if !self.hint.is_empty() {
            hint_spans.push(Span::styled(format!("   • {}", self.hint), theme::dim()));
        }
        f.render_widget(Paragraph::new(Line::from(hint_spans)), chunks[2]);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Enter => PanelAction::Submit(self.value().to_string()),
            KeyCode::Esc => PanelAction::Back,
            // Block newlines; Ctrl+U clears.
            KeyCode::Char('u')
                if event
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::CONTROL) =>
            {
                self.clear();
                PanelAction::Consumed
            }
            _ => {
                self.textarea.input(to_textarea_input(event));
                PanelAction::Consumed
            }
        }
    }
}
