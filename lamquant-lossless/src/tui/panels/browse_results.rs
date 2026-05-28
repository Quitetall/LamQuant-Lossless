//! Codec browse — recursively scan a directory for `.lml` / `.lmq` files
//! and present them as a sortable list. Selecting one returns its path
//! via `PanelAction::Submit` (re-using the input-flow contract).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::{Path, PathBuf};
use tui_scrollview::ScrollViewState;

use crate::tui::clipboard;
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Debug, Clone)]
struct Hit {
    path: PathBuf,
    rel: String, // path relative to cwd for display
    size: u64,
    is_lmq: bool,
}

pub struct BrowseResultsPanel {
    cwd: PathBuf,
    hits: Vec<Hit>,
    selected: usize,
    scroll: usize,
    sort_by_size: bool,
    status: Option<String>,
    /// tui-scrollview state — tracks virtual scroll position for future migration.
    scroll_state: std::cell::RefCell<ScrollViewState>,
}

impl Default for BrowseResultsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowseResultsPanel {
    pub fn new() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut p = Self {
            cwd,
            hits: Vec::new(),
            selected: 0,
            scroll: 0,
            sort_by_size: false,
            status: None,
            scroll_state: std::cell::RefCell::new(ScrollViewState::default()),
        };
        p.rescan();
        p
    }

    pub fn set_root(&mut self, root: &Path) {
        self.cwd = root.to_path_buf();
        self.rescan();
    }

    fn rescan(&mut self) {
        self.hits.clear();
        scan_recursive(&self.cwd, &self.cwd, &mut self.hits, 0);
        self.sort();
        self.selected = 0;
        self.scroll = 0;
        self.status = Some(format!("{} files found", self.hits.len()));
    }

    fn sort(&mut self) {
        if self.sort_by_size {
            self.hits.sort_by(|a, b| b.size.cmp(&a.size));
        } else {
            self.hits
                .sort_by(|a, b| a.rel.to_lowercase().cmp(&b.rel.to_lowercase()));
        }
    }

    fn adjust_scroll(&mut self, visible: usize) {
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + visible {
            self.scroll = self.selected.saturating_sub(visible.saturating_sub(1));
        }
        // Keep tui-scrollview state in sync for future ScrollView rendering.
        let _ = self.scroll_state.borrow();
    }
}

fn scan_recursive(root: &Path, cwd: &Path, out: &mut Vec<Hit>, depth: u32) {
    // Hard-cap recursion depth to avoid runaway scans.
    if depth > 16 {
        return;
    }
    let Ok(read) = fs::read_dir(cwd) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Skip dot-dirs to keep walks fast.
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with('.'))
                .unwrap_or(false)
            {
                continue;
            }
            scan_recursive(root, &path, out, depth + 1);
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        let is_lml = ext.eq_ignore_ascii_case("lml");
        let is_lmq = ext.eq_ignore_ascii_case("lmq");
        if !(is_lml || is_lmq) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let rel = path
            .strip_prefix(root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.display().to_string());
        out.push(Hit {
            path,
            rel,
            size,
            is_lmq,
        });
    }
}

impl Panel for BrowseResultsPanel {
    fn id(&self) -> &str {
        "browse_results"
    }
    fn title(&self) -> &str {
        "Browse"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let title = match &self.status {
            Some(s) => format!(" Browse — {} — {} ", self.cwd.display(), s),
            None => format!(" Browse — {} ", self.cwd.display()),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(title, theme::heading()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(inner);

        let visible = chunks[0].height as usize;
        let end = (self.scroll + visible).min(self.hits.len());

        if self.hits.is_empty() {
            let empty = Paragraph::new(Line::from(Span::styled(
                "  No .lml or .lmq files under this directory.",
                theme::dim(),
            )));
            f.render_widget(empty, chunks[0]);
        } else {
            let items: Vec<ListItem> = self.hits[self.scroll..end]
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    let idx = self.scroll + i;
                    let style = if idx == self.selected {
                        theme::selected()
                    } else {
                        Style::default()
                    };
                    let badge = if h.is_lmq { "LMQ" } else { "LML" };
                    let badge_style = if h.is_lmq {
                        theme::title()
                    } else {
                        theme::highlight()
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("  {} ", badge), badge_style),
                        Span::styled(format!(" {:<60}", h.rel), style),
                        Span::styled(format!("{:>10}", human_size(h.size)), theme::dim()),
                    ]))
                })
                .collect();
            f.render_widget(List::new(items), chunks[0]);
        }

        let sort_label = if self.sort_by_size { "size" } else { "name" };
        let nav = if theme::ascii_only() {
            "[Up/Dn]"
        } else {
            "[↑↓]"
        };
        let hint = format!(
            " {} nav  [Enter] open  [r] rescan  [o] sort ({})  [c] copy  [b] back ",
            nav, sort_label
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[1],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        let n = self.hits.len();
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if n > 0 {
                    self.selected = self.selected.saturating_sub(1);
                    self.adjust_scroll(20);
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if n > 0 && self.selected + 1 < n {
                    self.selected += 1;
                    self.adjust_scroll(20);
                }
                PanelAction::Consumed
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Some(h) = self.hits.get(self.selected) {
                    return PanelAction::Submit(h.path.display().to_string());
                }
                PanelAction::Consumed
            }
            KeyCode::Char('r') => {
                self.rescan();
                PanelAction::Consumed
            }
            KeyCode::Char('o') => {
                self.sort_by_size = !self.sort_by_size;
                self.sort();
                PanelAction::Consumed
            }
            KeyCode::Char('c') => {
                if let Some(h) = self.hits.get(self.selected) {
                    let path = h.path.display().to_string();
                    self.status = Some(match clipboard::copy_to_clipboard(&path) {
                        Ok(b) => format!("Copied via {}", b),
                        Err(e) => format!("Copy failed: {}", e),
                    });
                }
                PanelAction::Consumed
            }
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }
}

fn human_size(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{} B", bytes);
    }
    if bytes < 1024 * 1024 {
        return format!("{:.1} KB", bytes as f64 / 1024.0);
    }
    if bytes < 1024 * 1024 * 1024 {
        return format!("{:.1} MB", bytes as f64 / 1048576.0);
    }
    format!("{:.1} GB", bytes as f64 / 1073741824.0)
}
