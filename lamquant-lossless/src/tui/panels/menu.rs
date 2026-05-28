//! Menu panel — list-based selection with arrow keys and number shortcuts.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

/// A menu item: shortcut key, label, description, target screen ID.
#[derive(Debug, Clone)]
pub struct MenuItem {
    pub key: String,
    pub label: String,
    pub desc: String,
    pub target: String,
}

impl MenuItem {
    pub fn new(key: &str, label: &str, desc: &str, target: &str) -> Self {
        Self {
            key: key.to_string(),
            label: label.to_string(),
            desc: desc.to_string(),
            target: target.to_string(),
        }
    }
}

/// Generic menu panel — reusable for any screen with a list of options.
pub struct MenuPanel {
    id: String,
    title: String,
    subtitle: String,
    items: Vec<MenuItem>,
    selected: usize,
}

impl MenuPanel {
    pub fn new(id: &str, title: &str, subtitle: &str, items: Vec<MenuItem>) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            subtitle: subtitle.to_string(),
            items,
            selected: 0,
        }
    }

    /// Get currently selected item's target.
    pub fn selected_target(&self) -> &str {
        self.items
            .get(self.selected)
            .map(|i| i.target.as_str())
            .unwrap_or("")
    }
}

impl Panel for MenuPanel {
    fn id(&self) -> &str {
        &self.id
    }

    fn title(&self) -> &str {
        &self.title
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(2), // title + subtitle
                Constraint::Min(0),    // items
            ])
            .split(area);

        // Title block
        let title_text = vec![
            Line::from(Span::styled(&self.title, theme::heading())),
            Line::from(Span::styled(&self.subtitle, theme::dim())),
        ];
        f.render_widget(Paragraph::new(title_text), chunks[0]);

        // Menu items
        let list_items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let style = if i == self.selected {
                    theme::selected()
                } else {
                    theme::normal()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("  [{}] ", item.key), theme::key_hint()),
                    Span::styled(format!("{:<24}", item.label), style),
                    Span::styled(format!(" {}", item.desc), theme::dim()),
                ]))
            })
            .collect();

        let list = List::new(list_items).block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(theme::dim()),
        );
        f.render_widget(list, chunks[1]);
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
                if self.selected < self.items.len().saturating_sub(1) {
                    self.selected += 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                let target = self.selected_target().to_string();
                if target.is_empty() {
                    PanelAction::Consumed
                } else {
                    PanelAction::Navigate(target)
                }
            }
            KeyCode::Char(c) => {
                // Number/letter shortcut
                let key = c.to_string();
                if let Some(item) = self.items.iter().find(|i| i.key == key) {
                    PanelAction::Navigate(item.target.clone())
                } else if c == 'x' {
                    PanelAction::Quit
                } else if c == 'q' {
                    PanelAction::Home
                } else if c == 'b' {
                    PanelAction::Back
                } else if c == 'h' {
                    PanelAction::Navigate(router::SCREEN_HELP.to_string())
                } else {
                    PanelAction::Ignored
                }
            }
            KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            _ => PanelAction::Ignored,
        }
    }
}
