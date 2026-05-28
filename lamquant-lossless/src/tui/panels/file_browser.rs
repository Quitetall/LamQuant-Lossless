//! File browser panel — pure ratatui implementation. No external deps.
//!
//! Two submit keys, by user request:
//!   - Enter (or `;` alias) — submit the highlighted entry (file OR dir)
//!   - S                    — submit the directory being browsed (cwd)
//!
//! Navigation:
//!   - ↑↓ / jk      move highlight
//!   - → / l        descend into a dir (no-op on a file)
//!   - ← / h / Bksp parent directory
//!   - .            toggle hidden files
//!   - c            copy selected path to clipboard
//!   - b / Esc      back
//!
//! Save-as mode: constructed via `new_save(title, default_filename)`,
//! renders an inline filename row. `n` enters rename mode; chars edit
//! the filename buffer; Enter or Esc exits edit mode. Enter on a dir
//! submits `<dir>/<filename>`; Enter on a file submits the file path
//! (overwrite). S submits `<cwd>/<filename>`.
//!
//! Recent paths from `History.recent_inputs` are surfaced as virtual
//! `★ <path>` rows at the top of the list when in the user's CWD.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::{Path, PathBuf};
use tui_tree_widget::{TreeItem, TreeState};

use crate::tui::clipboard;
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Debug, Clone)]
struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: u64,
    /// True if this is a "★" virtual entry sourced from recent paths.
    is_recent: bool,
}

pub struct FileBrowserPanel {
    cwd: PathBuf,
    entries: Vec<Entry>,
    selected: usize,
    scroll: usize,
    title: String,
    show_hidden: bool,
    /// Recent paths to surface at the top of the list (sourced from History).
    recent_paths: Vec<String>,
    /// Most recent action result for display in the title bar.
    status: Option<String>,
    /// tui-tree-widget state — ready for full tree rendering in a future pass.
    tree_state: std::cell::RefCell<TreeState<String>>,
    /// When Some, panel is in save-as mode: an inline filename buffer
    /// is composed with the picked directory on submit. Returns
    /// `<chosen_dir>/<filename>` instead of just the directory path.
    save_filename: Option<String>,
    /// True when the user has pressed `n` and is editing
    /// `save_filename`. Printable keys append to the buffer; Backspace
    /// pops; Enter / Esc returns to nav mode.
    editing_filename: bool,
    /// ADR 0022 Group B: save-mode clobber guard. When Submit would
    /// land on an existing path in save mode, we stash the path here
    /// + show a warning row + return Consumed. A second matching
    /// Enter/S press with the same path clears + actually submits.
    /// Esc / any nav key clears the pending state.
    pending_clobber: Option<String>,
}

impl FileBrowserPanel {
    pub fn new(title: &str) -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let mut panel = Self {
            cwd: cwd.clone(),
            entries: Vec::new(),
            selected: 0,
            scroll: 0,
            title: title.to_string(),
            show_hidden: false,
            recent_paths: Vec::new(),
            status: None,
            tree_state: std::cell::RefCell::new(TreeState::default()),
            save_filename: None,
            editing_filename: false,
            pending_clobber: None,
        };
        panel.refresh();
        panel
    }

    /// Save-as constructor. Renders an inline filename row; Enter on a
    /// directory submits `<dir>/<default_filename>`; Enter on a file
    /// submits the file path (overwrite).
    pub fn new_save(title: &str, default_filename: &str) -> Self {
        let mut panel = Self::new(title);
        panel.save_filename = Some(default_filename.to_string());
        panel
    }

    /// Re-seed the panel for a new save flow. Cheaper than constructing
    /// a fresh FileBrowserPanel — preserves the cwd if the caller has
    /// no preference, which lets the user pick siblings of the input.
    pub fn reset_save(&mut self, title: &str, default_filename: &str, cwd: Option<&Path>) {
        self.title = title.to_string();
        self.save_filename = Some(default_filename.to_string());
        self.editing_filename = false;
        self.status = None;
        self.pending_clobber = None;
        if let Some(p) = cwd {
            self.cwd = p.to_path_buf();
        }
        self.refresh();
    }

    /// Re-seed the panel for an input pick — clears any save state so
    /// the same panel instance can be reused for input + output flows
    /// without leaking the filename buffer.
    pub fn reset_input(&mut self, title: &str) {
        self.title = title.to_string();
        self.save_filename = None;
        self.editing_filename = false;
        self.status = None;
        self.pending_clobber = None;
        self.refresh();
    }

    /// Set the recent-paths list (called by app.rs from History.recent_inputs).
    pub fn set_recent(&mut self, recent: Vec<String>) {
        self.recent_paths = recent;
        self.refresh();
    }

    pub fn selected_path(&self) -> Option<&Path> {
        self.entries.get(self.selected).map(|e| e.path.as_path())
    }

    /// ADR 0022 Group B: clobber guard for save-mode Submit. Only
    /// fires when we're in save mode AND the target path already
    /// exists. First press stashes the path + sets a warning status
    /// row + returns Consumed so the caller doesn't clobber. Second
    /// press with the same path passes through to actual Submit.
    /// Returns None when no guard action is needed (caller proceeds
    /// to Submit normally).
    fn clobber_guard(&mut self, path: &str) -> Option<PanelAction> {
        if self.save_filename.is_none() {
            return None;
        }
        if !std::path::Path::new(path).exists() {
            // Stale warning from a prior press would mislead the
            // user; clear both pending + status here.
            self.clear_clobber();
            return None;
        }
        if self.pending_clobber.as_deref() == Some(path) {
            self.pending_clobber = None;
            return None;
        }
        self.pending_clobber = Some(path.to_string());
        self.status = Some(format!(
            "{} exists -- press again to overwrite, Esc to cancel",
            path
        ));
        Some(PanelAction::Consumed)
    }

    /// Clear any pending clobber state. Called by nav/cancel keys so
    /// the user explicitly re-confirms when they navigate away and
    /// come back.
    fn clear_clobber(&mut self) {
        if self.pending_clobber.is_some() {
            self.pending_clobber = None;
            self.status = None;
        }
    }

    fn refresh(&mut self) {
        self.entries.clear();

        // 1. Recent paths as ★ virtual entries.
        //
        // Tier 4 audit: pre-fix used p.exists() only -- no path
        // traversal check. A history.json entry like `../../etc/
        // shadow` or a symlinked path was rendered as a ★ entry
        // and submitted verbatim to the next op. Now: canonicalize
        // each entry; skip if canonicalize fails (broken symlinks /
        // permission denied) or if the canonical path contains
        // `..` components. Render the label with a 200-char cap
        // so a pathological-length history entry can't blow up
        // the renderer.
        const RECENT_DISPLAY_CAP: usize = 200;
        for r in &self.recent_paths {
            let p = PathBuf::from(r);
            // Skip entries whose canonical form fails or contains
            // parent-dir traversal post-resolution.
            let canon = match fs::canonicalize(&p) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if canon
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                continue;
            }
            let is_dir = canon.is_dir();
            let size = fs::metadata(&canon).map(|m| m.len()).unwrap_or(0);
            let display = if r.len() > RECENT_DISPLAY_CAP {
                let cut = r
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|i| *i <= RECENT_DISPLAY_CAP)
                    .last()
                    .unwrap_or(0);
                format!("★ {}…", &r[..cut])
            } else {
                format!("★ {}", r)
            };
            self.entries.push(Entry {
                name: display,
                path: canon,
                is_dir,
                size,
                is_recent: true,
            });
        }

        // 2. CWD listing — dirs first, then files, both alphabetical.
        // Uses `DirEntry::file_type()` (cached on most filesystems) +
        // skips `metadata()` for directories since the size column is
        // hidden for them anyway. Eliminates 2× stat syscalls per
        // entry, which was the bulk of the per-keypress lag in large
        // directories. Surface read_dir failures via self.status so
        // the title bar makes the cause visible — previously an
        // EACCES / ENOENT silently rendered as "empty directory".
        match fs::read_dir(&self.cwd) {
            Err(e) => {
                self.status = Some(format!("read_dir failed: {}", e));
                self.entries.clear();
                self.selected = 0;
                self.scroll = 0;
                return;
            }
            Ok(read_dir) => {
                let mut dirs: Vec<Entry> = Vec::new();
                let mut files: Vec<Entry> = Vec::new();

                for entry in read_dir.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !self.show_hidden && name.starts_with('.') {
                        continue;
                    }
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    let path = entry.path();
                    let size = if is_dir {
                        0
                    } else {
                        entry.metadata().map(|m| m.len()).unwrap_or(0)
                    };

                    let e = Entry {
                        name,
                        path,
                        is_dir,
                        size,
                        is_recent: false,
                    };
                    if is_dir {
                        dirs.push(e);
                    } else {
                        files.push(e);
                    }
                }

                dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
                files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

                self.entries.extend(dirs);
                self.entries.extend(files);
            }
        }
        self.selected = 0;
        self.scroll = 0;
        // Rebuild tree items for future tui-tree-widget rendering.
        self.sync_tree_state();
    }

    /// Rebuilds the TreeState from current entries (dirs become expandable nodes).
    fn sync_tree_state(&mut self) {
        let items: Vec<TreeItem<String>> = self
            .entries
            .iter()
            .map(|e| TreeItem::new_leaf(e.path.display().to_string(), e.name.clone()))
            .collect();
        let mut state = self.tree_state.borrow_mut();
        // Select the first item by default.
        if !items.is_empty() {
            state.select_first();
        }
    }

    fn enter_dir(&mut self) {
        if let Some(entry) = self.entries.get(self.selected) {
            if entry.is_dir {
                self.cwd = entry.path.clone();
                self.pending_clobber = None;
                self.refresh();
            }
        }
    }

    fn go_up(&mut self) {
        if let Some(parent) = self.cwd.parent() {
            self.cwd = parent.to_path_buf();
            self.pending_clobber = None;
            self.refresh();
        }
    }

    fn adjust_scroll(&mut self, visible: usize) {
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + visible {
            self.scroll = self.selected.saturating_sub(visible - 1);
        }
    }

    fn copy_current_to_clipboard(&mut self) {
        if let Some(entry) = self.entries.get(self.selected) {
            let path_str = entry.path.display().to_string();
            self.status = Some(match clipboard::copy_to_clipboard(&path_str) {
                Ok(backend) => format!("Copied via {}: {}", backend, path_str),
                Err(e) => format!("Copy failed: {}", e),
            });
        }
    }
}

impl Panel for FileBrowserPanel {
    fn id(&self) -> &str {
        "file_browser"
    }
    fn title(&self) -> &str {
        &self.title
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let title = match &self.status {
            Some(s) => format!(" {} — {} — {} ", self.title, self.cwd.display(), s),
            None => format!(" {} — {} ", self.title, self.cwd.display()),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(title, theme::heading()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let save_h: u16 = if self.save_filename.is_some() { 1 } else { 0 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(save_h),
                Constraint::Length(1),
            ])
            .split(inner);

        let visible = chunks[0].height as usize;
        let end = (self.scroll + visible).min(self.entries.len());

        // Empty-dir affordance: when there are no entries, render a
        // single dim line so the user isn't staring at a blank pane
        // wondering why Enter does nothing. Save mode keeps `;` as the
        // submit-cwd-with-filename escape hatch.
        if self.entries.is_empty() {
            let msg = "  (empty directory — press [S] to select this dir, [h/←] to go up)";
            f.render_widget(
                Paragraph::new(Line::from(Span::styled(msg, theme::dim()))),
                chunks[0],
            );
        }

        let dir_icon = if theme::ascii_only() { "[d] " } else { "📁 " };
        let file_icon = if theme::ascii_only() { "    " } else { "   " };
        let star_icon = if theme::ascii_only() { "* " } else { "★ " };
        let caret = if theme::ascii_only() { "> " } else { "▶ " };
        let no_caret = "  ";

        let items: Vec<ListItem> = self.entries[self.scroll..end]
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let idx = self.scroll + i;
                let is_selected = idx == self.selected;
                // Selected styling wins over the recent-paths highlight so
                // the cursor is unambiguous even when the first entry is a
                // ★-marked recent path (which previously rendered with the
                // recent highlight color, masking the selection).
                let line_style = if is_selected {
                    theme::selected()
                } else if entry.is_recent {
                    theme::highlight()
                } else {
                    Style::default()
                };
                let icon = if entry.is_recent {
                    star_icon
                } else if entry.is_dir {
                    dir_icon
                } else {
                    file_icon
                };
                let size_str = if entry.is_dir {
                    String::new()
                } else {
                    format!("{:>8}", human_size(entry.size))
                };
                // Visible caret on the selected line — works regardless of
                // theme colour, so the cursor is obvious from the first
                // frame after refresh().
                let cursor_prefix = if is_selected { caret } else { no_caret };
                let name_style = line_style;
                ListItem::new(Line::from(vec![
                    Span::styled(cursor_prefix, theme::key_hint()),
                    Span::raw(icon),
                    Span::styled(format!("{:<40}", entry.name), name_style),
                    Span::styled(size_str, theme::dim()),
                ]))
            })
            .collect();

        f.render_widget(List::new(items), chunks[0]);

        // Save-mode filename row (rendered into chunks[1] when present).
        if let Some(name) = &self.save_filename {
            let cursor = if self.editing_filename { "█" } else { "" };
            let mode_tag = if self.editing_filename {
                "[EDIT]"
            } else {
                "[NAV] "
            };
            let line = Line::from(vec![
                Span::raw(" "),
                Span::styled(mode_tag, theme::key_hint()),
                Span::raw(" Filename: "),
                Span::styled(name.clone(), theme::highlight()),
                Span::styled(cursor, theme::highlight()),
            ]);
            f.render_widget(Paragraph::new(line), chunks[1]);
        }

        let hint = if self.save_filename.is_some() {
            if self.editing_filename {
                " [chars] type   [Backspace] erase   [Enter/Esc] done editing "
            } else {
                " [↑↓/jk] nav   [Enter/;] save in highlighted dir   [S] save in cwd   [l/→] open dir   [h/←] up   [n] rename   [.] hidden   [b] back "
            }
        } else {
            " [↑↓/jk] nav   [Enter/;] select highlighted   [S] select cwd   [l/→] open dir   [h/←] up   [.] hidden   [c] copy   [b] back "
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[2],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        // Save-mode filename editing: while `editing_filename`, ALL
        // printable keys append to the buffer and Backspace pops. Enter
        // and Esc both leave edit mode without changing focus. This
        // routes BEFORE the nav match so j/k/h/l etc. don't fire.
        if self.editing_filename {
            match event.code {
                KeyCode::Char(c) => {
                    if let Some(buf) = self.save_filename.as_mut() {
                        // Cap save-filename at SAVE_NAME_CAP chars so a
                        // wedged keyboard or bracketed paste can't drive
                        // the buffer to OOM. POSIX NAME_MAX is 255; we
                        // give a little headroom.
                        const SAVE_NAME_CAP: usize = 512;
                        if !c.is_control() && buf.chars().count() < SAVE_NAME_CAP {
                            buf.push(c);
                        }
                    }
                    return PanelAction::Consumed;
                }
                KeyCode::Backspace => {
                    if let Some(buf) = self.save_filename.as_mut() {
                        buf.pop();
                    }
                    return PanelAction::Consumed;
                }
                KeyCode::Enter | KeyCode::Esc => {
                    self.editing_filename = false;
                    return PanelAction::Consumed;
                }
                _ => return PanelAction::Consumed,
            }
        }

        let visible = 20usize; // approximate
        let in_save_mode = self.save_filename.is_some();
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.clear_clobber();
                self.selected = self.selected.saturating_sub(1);
                self.adjust_scroll(visible);
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.clear_clobber();
                if self.selected < self.entries.len().saturating_sub(1) {
                    self.selected += 1;
                }
                self.adjust_scroll(visible);
                PanelAction::Consumed
            }
            // Enter (and `;` alias) = SUBMIT highlighted entry. The
            // panel exposes TWO submit keys total: this one and `S`
            // (which submits cwd). Never descends — use `l` / `→` to
            // open a dir for browsing.
            //
            // Save mode: Enter/; on a dir submits `<dir>/<filename>`;
            // on a file submits the file path (overwrite).
            KeyCode::Enter | KeyCode::Char(';') => {
                if let Some(entry) = self.entries.get(self.selected) {
                    let path = if entry.is_dir {
                        if let Some(name) = &self.save_filename {
                            entry.path.join(name).display().to_string()
                        } else {
                            entry.path.display().to_string()
                        }
                    } else {
                        entry.path.display().to_string()
                    };
                    if let Some(action) = self.clobber_guard(&path) {
                        return action;
                    }
                    return PanelAction::Submit(path);
                }
                PanelAction::Consumed
            }
            // l / → = descend into a dir. On a file: no-op (the only
            // way to submit is Enter or S — keeps the keymap legible).
            KeyCode::Right | KeyCode::Char('l') => {
                if let Some(entry) = self.entries.get(self.selected) {
                    if entry.is_dir && !entry.is_recent {
                        self.enter_dir();
                    }
                    // file: silent. Enter is the only file-submit key.
                }
                PanelAction::Consumed
            }
            // h / Left / Backspace = up one directory (vim-style).
            KeyCode::Left | KeyCode::Backspace | KeyCode::Char('h') => {
                self.go_up();
                PanelAction::Consumed
            }
            KeyCode::Char('.') => {
                self.show_hidden = !self.show_hidden;
                self.refresh();
                PanelAction::Consumed
            }
            KeyCode::Char('c') => {
                self.copy_current_to_clipboard();
                PanelAction::Consumed
            }
            // [n] = rename the save buffer (save mode only).
            KeyCode::Char('n') if in_save_mode => {
                self.editing_filename = true;
                PanelAction::Consumed
            }
            // [S] = select the directory currently being browsed
            // (self.cwd). Save mode: submit cwd/filename instead.
            KeyCode::Char('S') => {
                let path = if let Some(name) = &self.save_filename {
                    self.cwd.join(name).display().to_string()
                } else {
                    self.cwd.display().to_string()
                };
                if let Some(action) = self.clobber_guard(&path) {
                    return action;
                }
                PanelAction::Submit(path)
            }
            KeyCode::Esc | KeyCode::Char('b') => {
                if self.pending_clobber.is_some() {
                    // Esc cancels pending overwrite; stay in panel.
                    self.clear_clobber();
                    PanelAction::Consumed
                } else {
                    PanelAction::Back
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR 0022 Group B: save-mode clobber two-tap. First Submit on
    /// an existing path stashes pending state + Consumed; second
    /// matching call returns None (passes through to actual Submit).
    /// A non-existent path returns None immediately. A different
    /// path on the second call re-stashes and Consumes again.
    #[test]
    fn clobber_guard_two_tap() {
        let tmp = std::env::temp_dir().join(format!(
            "lq_clobber_{}_{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&tmp, b"x").expect("write tmp");
        let mut panel = FileBrowserPanel::new("test");
        panel.save_filename = Some("out.lml".to_string());

        let path = tmp.display().to_string();
        // First press: pending stashed, Consumed returned.
        match panel.clobber_guard(&path) {
            Some(PanelAction::Consumed) => {}
            _ => panic!("first press should Consume"),
        }
        assert_eq!(panel.pending_clobber.as_deref(), Some(path.as_str()));
        // Second press same path: pass through.
        assert!(panel.clobber_guard(&path).is_none());
        assert!(panel.pending_clobber.is_none());

        // Non-existent path: pass through immediately.
        assert!(panel.clobber_guard("/no/such/path/xyz").is_none());

        // Different existing path re-stashes.
        let tmp2 = std::env::temp_dir().join(format!(
            "lq_clobber2_{}_{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&tmp2, b"y").expect("write tmp2");
        let p2 = tmp2.display().to_string();
        match panel.clobber_guard(&p2) {
            Some(PanelAction::Consumed) => {}
            _ => panic!("different existing path should Consume"),
        }

        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&tmp2);
    }

    /// Not in save mode: clobber_guard never fires.
    #[test]
    fn clobber_guard_skipped_in_input_mode() {
        let tmp = std::env::temp_dir().join(format!(
            "lq_clobber_inp_{}_{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&tmp, b"x").expect("write tmp");
        let mut panel = FileBrowserPanel::new("test");
        // save_filename is None -> input mode.
        let path = tmp.display().to_string();
        assert!(panel.clobber_guard(&path).is_none());
        assert!(panel.pending_clobber.is_none());
        let _ = std::fs::remove_file(&tmp);
    }
}
