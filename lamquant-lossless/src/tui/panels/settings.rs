//! Settings panel — full configuration editor backed by `LamQuantConfig`.
//!
//! Features:
//!   - Real load/save via `tui::config::LamQuantConfig` (TOML at lamquant.toml)
//!   - 60+ settings across 9 sections (output, codec, compute, integrity,
//!     resume, logging, input, output_files, backend, codec.lossless)
//!   - Arrow / vim navigation (↑↓ jk), ←→ to toggle bool/cycle enum
//!   - Enter to inline-edit int/float/string fields
//!   - `/` to search by key/dotpath/description (live filter)
//!   - `?` opens full help screen for current setting
//!   - `s` saves to lamquant.toml, `r` resets to defaults
//!   - `b` / Esc to go back; warns if dirty
//!
//! Each setting is described by a `SettingDescriptor` with fn-pointer
//! getter/setter — no macro magic, just verbose but explicit.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::config::LamQuantConfig;
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Debug, Clone)]
pub enum SettingValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    StrArr(Vec<String>),
}

impl SettingValue {
    pub fn display(&self) -> String {
        match self {
            Self::Bool(b) => {
                if *b {
                    "ON".into()
                } else {
                    "OFF".into()
                }
            }
            Self::Int(n) => n.to_string(),
            Self::Float(f) => format!("{}", f),
            Self::Str(s) => s.clone(),
            Self::StrArr(v) => v.join(", "),
        }
    }
}

#[derive(Clone, Copy)]
pub enum SettingKind {
    Bool,
    Int,
    Float,
    Enum(&'static [&'static str]),
    Str,
    StrArr,
}

pub struct SettingDescriptor {
    pub section: &'static str,
    pub label: &'static str,
    pub dotpath: &'static str,
    pub desc: &'static str,
    pub help: &'static str,
    pub kind: SettingKind,
    pub get: fn(&LamQuantConfig) -> SettingValue,
    pub set: fn(&mut LamQuantConfig, SettingValue),
}

/// Canonical bucket display order — most-touched first. Rows whose
/// section matches one of these strings sort into the corresponding
/// slot; an unknown section sorts to the end (`usize::MAX`) so a
/// typo in the descriptor table doesn't silently disappear off the
/// top of the panel.
fn section_rank(section: &str) -> usize {
    // Catch typos in dev builds: a row whose section string isn't in
    // the canonical bucket list would silently render under the last
    // bucket's header in release. The `every_descriptor_has_known_
    // section` test catches this at test time; this assert catches
    // it at panel-construction time during local development.
    debug_assert!(
        matches!(
            section,
            "CODEC"
                | "LOSSLESS"
                | "LOSSY/NEURAL"
                | "PERFORMANCE"
                | "SAFETY"
                | "INPUT"
                | "OUTPUT"
                | "DISPLAY"
                | "AUDIT"
        ),
        "settings.rs: descriptor section `{}` not in canonical bucket \
         list — add it to section_rank() or fix the typo",
        section,
    );
    match section {
        "CODEC" => 0,
        "LOSSLESS" => 1,
        "LOSSY/NEURAL" => 2,
        "PERFORMANCE" => 3,
        "SAFETY" => 4,
        "INPUT" => 5,
        "OUTPUT" => 6,
        "DISPLAY" => 7,
        "AUDIT" => 8,
        _ => usize::MAX,
    }
}

/// Returned by `cycle_*` helpers to indicate panel needs redraw.
const _: () = ();

pub struct SettingsPanel {
    cfg: LamQuantConfig,
    descriptors: Vec<SettingDescriptor>,
    /// Visible indices into `descriptors` after applying search filter.
    visible: Vec<usize>,
    cursor: usize,
    scroll: usize,
    dirty: bool,
    message: Option<String>,
    /// `Some(buffer)` when editing the current value's text representation.
    edit_buffer: Option<String>,
    /// `/` typed: collect search query in `search_query`. Esc/Enter exits search input.
    search_active: bool,
    search_query: String,
    /// Set by handler when user wants to navigate to per-setting help screen.
    pending_help: Option<usize>,
    /// Snapshot of `cfg` taken at the moment of a successful `save()`. Consumed
    /// via `take_saved_cfg` so App can sync `state.cfg` once with the EXACT
    /// committed copy — never an uncommitted post-save edit. Closes the
    /// 50ms tick-latency race that would otherwise let edits-after-save
    /// propagate before the next save fires.
    last_saved_cfg: Option<LamQuantConfig>,
}

impl Default for SettingsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl SettingsPanel {
    pub fn new() -> Self {
        let cfg = LamQuantConfig::load();
        Self::from_cfg(cfg)
    }

    pub fn from_cfg(cfg: LamQuantConfig) -> Self {
        let descriptors = build_descriptors();
        // Stable-sort initial visible[] by canonical bucket order so
        // the panel renders one header per bucket from the first
        // frame -- no need to wait for refilter() to be called via
        // search input.
        let mut visible: Vec<usize> = (0..descriptors.len()).collect();
        visible.sort_by_key(|i| section_rank(descriptors[*i].section));
        Self {
            cfg,
            descriptors,
            visible,
            cursor: 0,
            scroll: 0,
            dirty: false,
            message: None,
            edit_buffer: None,
            search_active: false,
            search_query: String::new(),
            pending_help: None,
            last_saved_cfg: None,
        }
    }

    /// True if the panel currently holds unsaved edits.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the snapshot of `cfg` captured at the most recent successful
    /// save(), or None if no save has occurred (or it was already consumed).
    /// The snapshot is taken at save-time so post-save edits cannot leak
    /// into the consumer.
    pub fn take_saved_cfg(&mut self) -> Option<LamQuantConfig> {
        self.last_saved_cfg.take()
    }

    /// Returns the descriptor index for help-screen rendering, then clears.
    pub fn take_pending_help(&mut self) -> Option<usize> {
        self.pending_help.take()
    }

    pub fn descriptor(&self, i: usize) -> Option<&SettingDescriptor> {
        self.descriptors.get(i)
    }

    /// Read-only access to the underlying config (used by settings_help to render current value).
    pub fn cfg_ref(&self) -> &LamQuantConfig {
        &self.cfg
    }

    fn current_idx(&self) -> Option<usize> {
        self.visible.get(self.cursor).copied()
    }

    #[allow(dead_code)]
    fn current_value(&self) -> Option<SettingValue> {
        self.current_idx()
            .map(|i| (self.descriptors[i].get)(&self.cfg))
    }

    fn cycle_current(&mut self, forward: bool) {
        let Some(idx) = self.current_idx() else {
            return;
        };
        let d = &self.descriptors[idx];
        let val = (d.get)(&self.cfg);
        let new_val = match (d.kind, val) {
            (SettingKind::Bool, SettingValue::Bool(b)) => Some(SettingValue::Bool(!b)),
            (SettingKind::Enum(choices), SettingValue::Str(s)) => {
                let i = choices.iter().position(|c| *c == s).unwrap_or(0);
                let next = if forward {
                    (i + 1) % choices.len()
                } else if i == 0 {
                    choices.len() - 1
                } else {
                    i - 1
                };
                Some(SettingValue::Str(choices[next].to_string()))
            }
            _ => None,
        };
        if let Some(v) = new_val {
            (d.set)(&mut self.cfg, v);
            self.dirty = true;
        }
    }

    fn enter_edit(&mut self) -> bool {
        let Some(idx) = self.current_idx() else {
            return false;
        };
        let d = &self.descriptors[idx];
        match d.kind {
            SettingKind::Bool => {
                self.cycle_current(true);
                false
            }
            SettingKind::Enum(_) => {
                self.cycle_current(true);
                false
            }
            SettingKind::Int | SettingKind::Float | SettingKind::Str | SettingKind::StrArr => {
                let val = (d.get)(&self.cfg);
                self.edit_buffer = Some(val.display());
                true
            }
        }
    }

    fn commit_edit(&mut self) -> Result<(), String> {
        let Some(buf) = self.edit_buffer.take() else {
            return Ok(());
        };
        let Some(idx) = self.current_idx() else {
            return Ok(());
        };
        let d = &self.descriptors[idx];
        let new_val = match d.kind {
            SettingKind::Int => buf
                .trim()
                .parse::<i64>()
                .map(SettingValue::Int)
                .map_err(|e| format!("not an integer: {}", e))?,
            SettingKind::Float => buf
                .trim()
                .parse::<f64>()
                .map(SettingValue::Float)
                .map_err(|e| format!("not a float: {}", e))?,
            SettingKind::Str => SettingValue::Str(buf),
            SettingKind::StrArr => {
                let parts: Vec<String> = buf
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                SettingValue::StrArr(parts)
            }
            _ => return Ok(()),
        };
        (d.set)(&mut self.cfg, new_val);
        self.dirty = true;
        Ok(())
    }

    fn cancel_edit(&mut self) {
        self.edit_buffer = None;
    }

    fn save(&mut self) {
        match self.cfg.save() {
            Ok(()) => {
                self.dirty = false;
                // Snapshot at commit time — closes the post-save edit race.
                self.last_saved_cfg = Some(self.cfg.clone());
                // Propagate config-time choices (today: compute
                // backend) into process-wide runtime globals so the
                // codec hot path picks up the user's selection
                // without requiring a restart.
                self.cfg.apply_to_runtime();
                self.message = Some(format!(
                    "Saved to {}",
                    crate::tui::config::config_path().display()
                ));
            }
            Err(e) => self.message = Some(format!("Save failed: {}", e)),
        }
    }

    fn reset_defaults(&mut self) {
        self.cfg = LamQuantConfig::default();
        self.dirty = true;
        self.message = Some("Defaults restored — press s to save".into());
    }

    fn refilter(&mut self) {
        let q = self.search_query.to_lowercase();
        self.visible = (0..self.descriptors.len())
            .filter(|i| {
                if q.is_empty() {
                    return true;
                }
                let d = &self.descriptors[*i];
                d.label.to_lowercase().contains(&q)
                    || d.dotpath.to_lowercase().contains(&q)
                    || d.desc.to_lowercase().contains(&q)
                    || d.section.to_lowercase().contains(&q)
            })
            .collect();
        // Stable-sort by canonical user-flow section order so the
        // renderer's section-break logic groups every row of a given
        // bucket under one header — even if the source descriptor
        // table interleaves them. Lets us keep the descriptor block
        // physically grouped by data layer (codec/output/etc.) while
        // presenting buckets like PERFORMANCE = COMPUTE + BACKEND as
        // one contiguous panel section.
        self.visible.sort_by_key(|i| {
            let sec = self.descriptors[*i].section;
            section_rank(sec)
        });
        self.cursor = self.cursor.min(self.visible.len().saturating_sub(1));
        self.scroll = 0;
    }

    fn adjust_scroll(&mut self, visible_h: usize) {
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if visible_h > 0 && self.cursor >= self.scroll + visible_h {
            self.scroll = self.cursor.saturating_sub(visible_h - 1);
        }
    }
}

impl Panel for SettingsPanel {
    fn id(&self) -> &str {
        "settings"
    }
    fn title(&self) -> &str {
        "Settings"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        // Outer block matches the canonical full-screen panel style:
        // dim border + heading-styled title with single-space padding.
        let outer = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(" Settings ", theme::heading()));
        let inner = outer.inner(area);
        f.render_widget(outer, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(inner);

        // ── Header ────────────────────────────────────────────────────────
        let dirty_mark = if self.dirty { " [modified]" } else { "" };
        let mut header_lines = vec![Line::from(vec![
            Span::styled(dirty_mark, theme::warning()),
            Span::styled(
                format!(
                    "  {} of {} shown",
                    self.visible.len(),
                    self.descriptors.len()
                ),
                theme::dim(),
            ),
        ])];
        if self.search_active {
            header_lines.push(Line::from(vec![
                Span::styled(" /", theme::key_hint()),
                Span::styled(self.search_query.clone(), theme::heading()),
                Span::styled("_", theme::warning()),
            ]));
        } else {
            header_lines.push(Line::from(""));
        }
        f.render_widget(Paragraph::new(header_lines), chunks[0]);

        // ── Items ─────────────────────────────────────────────────────────
        let avail = chunks[1].height as usize;
        let start = self.scroll;
        let end = (start + avail).min(self.visible.len());
        let mut last_section = "";

        let mut rows: Vec<ListItem> = Vec::new();
        for vi in start..end {
            let idx = self.visible[vi];
            let d = &self.descriptors[idx];
            if d.section != last_section {
                rows.push(ListItem::new(Line::from(Span::styled(
                    format!("    {}", d.section),
                    theme::title(),
                ))));
                last_section = d.section;
            }

            let val = (d.get)(&self.cfg);
            let val_str = if vi == self.cursor && self.edit_buffer.is_some() {
                format!("[{}]_", self.edit_buffer.as_deref().unwrap_or(""))
            } else {
                val.display()
            };
            let val_style = match &val {
                SettingValue::Bool(true) => theme::success(),
                SettingValue::Bool(false) => theme::dim(),
                _ => theme::heading(),
            };
            let row_style = if vi == self.cursor {
                theme::selected()
            } else {
                Style::default()
            };
            let marker = if vi == self.cursor { "▸" } else { " " };
            let hint = if vi == self.cursor && self.edit_buffer.is_none() {
                match d.kind {
                    SettingKind::Bool | SettingKind::Enum(_) => " ←→",
                    _ => " ↵ edit",
                }
            } else {
                ""
            };

            rows.push(ListItem::new(Line::from(vec![
                Span::styled(format!(" {} ", marker), theme::key_hint()),
                Span::styled(format!("{:<24}", d.label), row_style),
                Span::styled(format!("{:<28}", trunc(&val_str, 28)), val_style),
                Span::styled(format!(" {}", d.desc), theme::dim()),
                Span::styled(hint, theme::key_hint()),
            ])));
        }

        f.render_widget(List::new(rows), chunks[1]);

        // ── Footer ────────────────────────────────────────────────────────
        let msg_style = if self
            .message
            .as_deref()
            .map(|m| m.starts_with("Save failed"))
            .unwrap_or(false)
        {
            theme::error()
        } else {
            theme::success()
        };
        let msg = self.message.as_deref().unwrap_or("");
        // Bracketed key hints — matches help/output/exit_confirm/settings_help/root_warn.
        let bracket_hints: Vec<(&str, &str)> = if self.edit_buffer.is_some() {
            vec![("Enter", "commit"), ("Esc", "cancel"), ("Backspace", "del")]
        } else if self.search_active {
            vec![
                ("Enter", "exit search"),
                ("Esc", "exit"),
                ("Backspace", "del"),
            ]
        } else {
            vec![
                ("↑↓", "nav"),
                ("←→", "toggle"),
                ("Enter", "edit"),
                ("/", "search"),
                ("?", "help"),
                ("s", "save"),
                ("r", "reset"),
                ("b", "back"),
            ]
        };
        let mut footer_keys: Vec<Span> = Vec::new();
        for (k, v) in bracket_hints {
            footer_keys.push(Span::styled(format!(" [{}] ", k), theme::key_hint()));
            footer_keys.push(Span::styled(v.to_string(), theme::dim()));
        }

        let footer = Paragraph::new(vec![
            Line::from(""),
            Line::from(footer_keys),
            Line::from(Span::styled(format!("  {}", msg), msg_style)),
        ]);
        f.render_widget(footer, chunks[2]);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        // Search input mode
        if self.search_active {
            match event.code {
                KeyCode::Esc | KeyCode::Enter => {
                    self.search_active = false;
                }
                KeyCode::Backspace => {
                    self.search_query.pop();
                    self.refilter();
                }
                KeyCode::Char(c) => {
                    self.search_query.push(c);
                    self.refilter();
                }
                _ => {}
            }
            return PanelAction::Consumed;
        }

        // Edit mode (text input for current value)
        if self.edit_buffer.is_some() {
            match event.code {
                KeyCode::Esc => self.cancel_edit(),
                KeyCode::Enter => {
                    if let Err(e) = self.commit_edit() {
                        self.message = Some(format!("Save failed: {}", e));
                    } else {
                        self.message = Some("Saved value".into());
                    }
                }
                KeyCode::Backspace => {
                    if let Some(b) = self.edit_buffer.as_mut() {
                        b.pop();
                    }
                }
                KeyCode::Char(c) => {
                    if let Some(b) = self.edit_buffer.as_mut() {
                        b.push(c);
                    }
                }
                _ => {}
            }
            return PanelAction::Consumed;
        }

        let n = self.visible.len();
        if n == 0 {
            return match event.code {
                KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') | KeyCode::Char('q') => {
                    PanelAction::Back
                }
                KeyCode::Char('/') => {
                    self.search_active = true;
                    PanelAction::Consumed
                }
                _ => PanelAction::Ignored,
            };
        }

        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.cursor = if self.cursor == 0 {
                    n - 1
                } else {
                    self.cursor - 1
                };
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.cursor = (self.cursor + 1) % n;
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::Home => {
                self.cursor = 0;
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::End => {
                self.cursor = n - 1;
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::PageUp => {
                self.cursor = self.cursor.saturating_sub(10);
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::PageDown => {
                self.cursor = (self.cursor + 10).min(n - 1);
                self.adjust_scroll(20);
                PanelAction::Consumed
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.cycle_current(true);
                PanelAction::Consumed
            }
            KeyCode::Left => {
                self.cycle_current(false);
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                self.enter_edit();
                PanelAction::Consumed
            }
            KeyCode::Char('/') => {
                self.search_active = true;
                self.search_query.clear();
                self.refilter();
                PanelAction::Consumed
            }
            KeyCode::Char('s') => {
                self.save();
                PanelAction::Consumed
            }
            KeyCode::Char('r') => {
                self.reset_defaults();
                PanelAction::Consumed
            }
            KeyCode::Char('?') => {
                self.pending_help = self.current_idx();
                PanelAction::Navigate(router::SCREEN_SETTINGS_HELP.to_string())
            }
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') | KeyCode::Char('q') => {
                if self.dirty {
                    self.message =
                        Some("Unsaved changes — press s to save, b again to discard".into());
                    self.dirty = false; // give the user one warning shot
                    PanelAction::Consumed
                } else {
                    PanelAction::Back
                }
            }
            _ => PanelAction::Ignored,
        }
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

// ── Descriptor table ───────────────────────────────────────────────────────
// One row per editable field in LamQuantConfig. Verbose but explicit.
//
// Row order here = intra-bucket display order. `from_cfg` / `refilter`
// stable-sort by `section_rank()` to group rows of a given bucket
// together at render time; rows within a bucket display in the order
// they appear below. Add new rows next to existing rows of the same
// section so the user-visible order stays predictable.

fn build_descriptors() -> Vec<SettingDescriptor> {
    use SettingKind as K;
    use SettingValue::*;

    macro_rules! d {
        ($section:expr, $label:expr, $dot:expr, $desc:expr, $help:expr, $kind:expr, $get:expr, $set:expr) => {
            SettingDescriptor {
                section: $section,
                label: $label,
                dotpath: $dot,
                desc: $desc,
                help: $help,
                kind: $kind,
                get: $get,
                set: $set,
            }
        };
    }

    fn s(v: &str) -> SettingValue {
        Str(v.to_string())
    }

    vec![
        // ── PERFORMANCE (binary mode + paths) ──────────────────────────────
        d!("PERFORMANCE", "Mode", "backend.mode",
           "auto/rust/python/custom",
           "Which backend performs encode/decode.\n\n  auto   — Use Rust if found, else Python\n  rust   — Force compiled Rust binary (~200 MB/s)\n  python — Pure-Python via numba (~15 MB/s)\n  custom — User-provided binary",
           K::Enum(&["auto","rust","python","custom"]),
           |c| s(&c.backend.mode),
           |c, v| if let Str(x) = v { c.backend.mode = x; }),
        d!("PERFORMANCE", "Rust binary", "backend.rust_binary",
           "name or path",
           "Name or path of the Rust lml binary.\nSearched: explicit → $PATH → target/release/ → target/debug/",
           K::Str,
           |c| s(&c.backend.rust_binary),
           |c, v| if let Str(x) = v { c.backend.rust_binary = x; }),
        d!("PERFORMANCE", "Custom binary", "backend.custom_binary",
           "path (when mode=custom)",
           "Path to user-provided backend binary. Only used when mode=custom.",
           K::Str,
           |c| s(&c.backend.custom_binary),
           |c, v| if let Str(x) = v { c.backend.custom_binary = x; }),

        // ── CODEC ──────────────────────────────────────────────────────────
        d!("CODEC", "Default mode", "codec.default_mode",
           "lossless or neural",
           "Which codec mode by default.\n  lossless — Bit-perfect DWT+LPC+Golomb-Rice\n  neural   — Ternary encoder + SNAC FSQ + rANS",
           K::Enum(&["lossless","neural"]),
           |c| s(&c.codec.default_mode),
           |c, v| if let Str(x) = v { c.codec.default_mode = x; }),
        d!("CODEC", "Lossless default", "codec.lossless_default_mode",
           "prompt / lma / lml_siblings",
           "What the Lossless [1] Compress hotkey does.\n  prompt        — Ask LMA vs LML+siblings every time (default)\n  lma           — Always pack to a single .lma archive\n  lml_siblings  — Per-EEG .lml + copy non-EEG sidecars alongside",
           K::Enum(&["prompt","lma","lml_siblings"]),
           |c| s(&c.codec.lossless_default_mode),
           |c, v| if let Str(x) = v { c.codec.lossless_default_mode = x; }),
        d!("CODEC", "Input bits", "codec.input_bits",
           "ADC bits",
           "Bit depth of the source ADC. Most EEG: 16 or 24.",
           K::Int,
           |c| Int(c.codec.input_bits),
           |c, v| if let Int(x) = v { c.codec.input_bits = x; }),
        d!("CODEC", "Window samples", "codec.window_samples",
           "samples per window",
           "Samples per compression window. 2500 = 10s @ 250 Hz.",
           K::Int,
           |c| Int(c.codec.window_samples),
           |c, v| if let Int(x) = v { c.codec.window_samples = x; }),
        d!("LOSSY/NEURAL", "Noise bits", "codec.noise_bits",
           "0=lossless, 1-15=strip",
           "LSBs to strip. 0 = lossless. 6 ≈ ADS1299 noise floor.",
           K::Int,
           |c| Int(c.codec.noise_bits),
           |c, v| if let Int(x) = v { c.codec.noise_bits = x; }),
        d!("CODEC", "Verification", "codec.verification",
           "paranoid/standard/fast",
           "Integrity check level after encode.\n  paranoid — Full decode + sample-exact compare\n  standard — CRC-32 + SHA-256 (default)\n  fast     — CRC-32 only",
           K::Enum(&["paranoid","standard","fast"]),
           |c| s(&c.codec.verification),
           |c, v| if let Str(x) = v { c.codec.verification = x; }),

        // ── LOSSLESS (entropy coder, LPC, lifting) ─────────────────────────
        d!("LOSSLESS", "Entropy coder", "codec.lossless.entropy_coder",
           "golomb_rice/rans",
           "Entropy coding algorithm.",
           K::Enum(&["golomb_rice","rans"]),
           |c| s(&c.codec.lossless.entropy_coder),
           |c, v| if let Str(x) = v { c.codec.lossless.entropy_coder = x; }),
        d!("LOSSLESS", "LPC order", "codec.lossless.lpc_order",
           "predictor order",
           "Linear-predictor order. 2-4 typical.",
           K::Int,
           |c| Int(c.codec.lossless.lpc_order),
           |c, v| if let Int(x) = v { c.codec.lossless.lpc_order = x; }),
        d!("LOSSLESS", "Use lifting", "codec.lossless.use_lifting",
           "Le Gall 5/3 DWT",
           "Run integer Le Gall 5/3 lifting wavelet before LPC.",
           K::Bool,
           |c| Bool(c.codec.lossless.use_lifting),
           |c, v| if let Bool(x) = v { c.codec.lossless.use_lifting = x; }),

        // ── PERFORMANCE (workers, memory, backend) ─────────────────────────
        d!("PERFORMANCE", "Workers", "compute.workers",
           "0 = auto",
           "Parallel encode workers. 0 = (cores-2) capped by RAM.",
           K::Int,
           |c| Int(c.compute.workers),
           |c, v| if let Int(x) = v { c.compute.workers = x; }),
        d!("PERFORMANCE", "Memory limit (GiB)", "compute.memory_limit_gib",
           "0 = unlimited",
           "Soft per-process RSS limit. 0 = unlimited.",
           K::Float,
           |c| Float(c.compute.memory_limit_gib),
           |c, v| if let Float(x) = v { c.compute.memory_limit_gib = x; }),
        d!("PERFORMANCE", "Numba cache dir", "compute.numba_cache_dir",
           "auto/path",
           "Where numba JIT-compiled artifacts are cached. 'auto' = ~/.numba_cache.",
           K::Str,
           |c| s(&c.compute.numba_cache_dir),
           |c, v| if let Str(x) = v { c.compute.numba_cache_dir = x; }),
        d!("PERFORMANCE", "Codec backend", "compute.backend",
           "firmware/desktop",
           "Lossless-codec compute backend.\n\n  desktop  — Rayon per-channel + AVX2 autocorr + parallel\n             decompress. Host perf path. Default on this build.\n  firmware — Scalar serial path the Cortex-M build also uses.\n             Slower but byte-identical output. Pick this for\n             debugging or firmware-bench parity.\n\nOutput .lml bytes are identical across the two backends\n(locked by tests/byte_equal_backends.rs). Only wall-clock\ndiffers.",
           K::Enum(&["firmware","desktop"]),
           |c| s(&c.compute.backend),
           |c, v| if let Str(x) = v { c.compute.backend = x; }),

        // ── SAFETY (integrity) ─────────────────────────────────────────────
        d!("SAFETY", "Window checksum", "integrity.window_checksum",
           "crc32",
           "Per-window checksum algorithm. crc32 = LML container default.",
           K::Enum(&["crc32"]),
           |c| s(&c.integrity.window_checksum),
           |c, v| if let Str(x) = v { c.integrity.window_checksum = x; }),
        d!("SAFETY", "File checksum", "integrity.file_checksum",
           "sha256/blake3",
           "Whole-file integrity hash for the manifest.",
           K::Enum(&["sha256","blake3"]),
           |c| s(&c.integrity.file_checksum),
           |c, v| if let Str(x) = v { c.integrity.file_checksum = x; }),
        d!("SAFETY", "Verify after write", "integrity.verify_after_write",
           "+50% time",
           "Decode and verify each compressed file post-write. FDA recommended.",
           K::Bool,
           |c| Bool(c.integrity.verify_after_write),
           |c, v| if let Bool(x) = v { c.integrity.verify_after_write = x; }),
        d!("SAFETY", "Verify outliers", "integrity.verify_outliers",
           "abs > 90% range",
           "Re-decode files whose ratio looks anomalous (saturated/sparse).",
           K::Bool,
           |c| Bool(c.integrity.verify_outliers),
           |c, v| if let Bool(x) = v { c.integrity.verify_outliers = x; }),
        d!("SAFETY", "Reject corrupted input", "integrity.reject_corrupted_input",
           "EDF parser",
           "Refuse to compress files with EDF parse errors instead of recovering.",
           K::Bool,
           |c| Bool(c.integrity.reject_corrupted_input),
           |c, v| if let Bool(x) = v { c.integrity.reject_corrupted_input = x; }),
        d!("SAFETY", "Refuse double-strip", "integrity.refuse_double_strip",
           "noise_bits twice",
           "Block stripping a file that already has metadata showing prior LSB strip.",
           K::Bool,
           |c| Bool(c.integrity.refuse_double_strip),
           |c, v| if let Bool(x) = v { c.integrity.refuse_double_strip = x; }),
        d!("SAFETY", "Fail fast", "integrity.fail_fast",
           "abort on first error",
           "Stop the whole batch on first error. OFF = log and continue.",
           K::Bool,
           |c| Bool(c.integrity.fail_fast),
           |c, v| if let Bool(x) = v { c.integrity.fail_fast = x; }),

        // ── SAFETY (resume) ────────────────────────────────────────────────
        d!("SAFETY", "Enabled", "resume.enabled",
           "crash recovery",
           "Track per-file progress in .lamquant_state.json so interrupted runs can resume.",
           K::Bool,
           |c| Bool(c.resume.enabled),
           |c, v| if let Bool(x) = v { c.resume.enabled = x; }),
        d!("SAFETY", "State file", "resume.state_file",
           "filename",
           "Filename for resume state, written next to each output dir.",
           K::Str,
           |c| s(&c.resume.state_file),
           |c, v| if let Str(x) = v { c.resume.state_file = x; }),
        d!("SAFETY", "Checkpoint strategy", "resume.checkpoint_strategy",
           "per_file/per_window",
           "Granularity of resume points.",
           K::Enum(&["per_file","per_window"]),
           |c| s(&c.resume.checkpoint_strategy),
           |c, v| if let Str(x) = v { c.resume.checkpoint_strategy = x; }),
        d!("SAFETY", "On existing state", "resume.on_existing_state",
           "auto/resume/restart/ask",
           "Behaviour when a state file already exists.",
           K::Enum(&["auto","resume","restart","ask"]),
           |c| s(&c.resume.on_existing_state),
           |c, v| if let Str(x) = v { c.resume.on_existing_state = x; }),
        d!("SAFETY", "Skip existing", "resume.skip_existing_output",
           "don't re-compress",
           "Skip files whose output already exists.",
           K::Bool,
           |c| Bool(c.resume.skip_existing_output),
           |c, v| if let Bool(x) = v { c.resume.skip_existing_output = x; }),
        d!("SAFETY", "Verify skipped", "resume.verify_skipped",
           "even if skipped",
           "Run integrity check on skipped outputs anyway.",
           K::Bool,
           |c| Bool(c.resume.verify_skipped),
           |c, v| if let Bool(x) = v { c.resume.verify_skipped = x; }),
        d!("SAFETY", "Quarantine dir", "resume.quarantine_dir",
           "subdir name",
           "Directory under output for files that failed to encode.",
           K::Str,
           |c| s(&c.resume.quarantine_dir),
           |c, v| if let Str(x) = v { c.resume.quarantine_dir = x; }),
        d!("SAFETY", "Max retries", "resume.max_retries",
           "transient errors",
           "Max retries per file on transient I/O errors before quarantining.",
           K::Int,
           |c| Int(c.resume.max_retries),
           |c, v| if let Int(x) = v { c.resume.max_retries = x; }),
        d!("SAFETY", "Retry backoff (s)", "resume.retry_backoff_s",
           "wait between tries",
           "Seconds to wait between retry attempts.",
           K::Float,
           |c| Float(c.resume.retry_backoff_s),
           |c, v| if let Float(x) = v { c.resume.retry_backoff_s = x; }),

        // ── AUDIT (logging + manifest) ─────────────────────────────────────
        d!("AUDIT", "Audit log", "logging.audit_log",
           "filename",
           "Append-only audit log per output directory.",
           K::Str,
           |c| s(&c.logging.audit_log),
           |c, v| if let Str(x) = v { c.logging.audit_log = x; }),
        d!("AUDIT", "Append audit", "logging.append_audit",
           "append vs truncate",
           "Append to audit log instead of truncating per run.",
           K::Bool,
           |c| Bool(c.logging.append_audit),
           |c, v| if let Bool(x) = v { c.logging.append_audit = x; }),
        d!("AUDIT", "Stderr level", "logging.stderr_level",
           "min severity",
           "Minimum severity for messages printed to stderr.",
           K::Enum(&["DEBUG","INFO","WARNING","ERROR"]),
           |c| s(&c.logging.stderr_level),
           |c, v| if let Str(x) = v { c.logging.stderr_level = x; }),
        d!("AUDIT", "File log", "logging.file_log",
           "path or empty",
           "Optional secondary log file path (empty = disabled).",
           K::Str,
           |c| s(&c.logging.file_log),
           |c, v| if let Str(x) = v { c.logging.file_log = x; }),
        d!("AUDIT", "Include tracebacks", "logging.include_tracebacks",
           "verbose errors",
           "Include Python tracebacks in audit/log on errors.",
           K::Bool,
           |c| Bool(c.logging.include_tracebacks),
           |c, v| if let Bool(x) = v { c.logging.include_tracebacks = x; }),
        d!("AUDIT", "Manifest", "logging.manifest",
           "JSON filename",
           "Manifest JSON with SHA-256 of every output file.",
           K::Str,
           |c| s(&c.logging.manifest),
           |c, v| if let Str(x) = v { c.logging.manifest = x; }),
        d!("AUDIT", "Manifest include files", "logging.manifest_include_files",
           "per-file entries",
           "Include per-file SHA-256 entries in manifest. OFF = directory totals only.",
           K::Bool,
           |c| Bool(c.logging.manifest_include_files),
           |c, v| if let Bool(x) = v { c.logging.manifest_include_files = x; }),

        // ── INPUT ──────────────────────────────────────────────────────────
        d!("INPUT", "Extensions", "input.extensions",
           "comma-separated",
           "File extensions to consider as EEG input. Comma-separated.",
           K::StrArr,
           |c| StrArr(c.input.extensions.clone()),
           |c, v| if let StrArr(x) = v { c.input.extensions = x; }),
        d!("INPUT", "Recursive", "input.recursive",
           "scan subdirs",
           "Walk subdirectories when input is a directory.",
           K::Bool,
           |c| Bool(c.input.recursive),
           |c, v| if let Bool(x) = v { c.input.recursive = x; }),
        d!("INPUT", "Follow symlinks", "input.follow_symlinks",
           "during walk",
           "Follow symbolic links when walking directories.",
           K::Bool,
           |c| Bool(c.input.follow_symlinks),
           |c, v| if let Bool(x) = v { c.input.follow_symlinks = x; }),
        d!("INPUT", "Min file size", "input.min_file_size",
           "bytes",
           "Skip files smaller than this many bytes (e.g., empty .edf).",
           K::Int,
           |c| Int(c.input.min_file_size),
           |c, v| if let Int(x) = v { c.input.min_file_size = x; }),
        d!("INPUT", "Max file size", "input.max_file_size",
           "0 = unlimited",
           "Skip files larger than this many bytes. 0 = unlimited.",
           K::Int,
           |c| Int(c.input.max_file_size),
           |c, v| if let Int(x) = v { c.input.max_file_size = x; }),
        d!("INPUT", "Exclude patterns", "input.exclude_patterns",
           "comma-separated globs",
           "Globs of paths to skip. e.g. **/test/**, **/.git/**",
           K::StrArr,
           |c| StrArr(c.input.exclude_patterns.clone()),
           |c, v| if let StrArr(x) = v { c.input.exclude_patterns = x; }),

        // ── OUTPUT (destination + structure) ───────────────────────────────
        // NOTE: `extension` is NOT a configurable field. The codec
        // mode selected at encode time (`CodecMode::ext()` at
        // panels/mode_panel.rs) determines `.lml` (lossless) vs `.lmq`
        // (neural/lossy). Allowing user override would let a `.lml`
        // file carry lossy bytes -- downstream tools that trust the
        // `.lml` suffix could silently mishandle data.
        d!("OUTPUT", "Preserve structure", "output_files.preserve_structure",
           "mirror input tree",
           "Mirror input directory layout in the output tree.",
           K::Bool,
           |c| Bool(c.output_files.preserve_structure),
           |c, v| if let Bool(x) = v { c.output_files.preserve_structure = x; }),
        d!("OUTPUT", "Atomic writes", "output_files.atomic_writes",
           "tmp+rename",
           "Write to .tmp and rename on completion to avoid partial files.",
           K::Bool,
           |c| Bool(c.output_files.atomic_writes),
           |c, v| if let Bool(x) = v { c.output_files.atomic_writes = x; }),
        d!("OUTPUT", "Fsync on write", "output_files.fsync_on_write",
           "force flush",
           "fsync each output file before declaring success.",
           K::Bool,
           |c| Bool(c.output_files.fsync_on_write),
           |c, v| if let Bool(x) = v { c.output_files.fsync_on_write = x; }),

        // ── DISPLAY (UI cosmetics) ─────────────────────────────────────────
        d!("DISPLAY", "Refresh Hz", "output.refresh_hz",
           "dashboard fps",
           "Frame rate for the running-op dashboard. 10.0 = smooth, low CPU.",
           K::Float,
           |c| Float(c.output.refresh_hz),
           |c, v| if let Float(x) = v { c.output.refresh_hz = x; }),
        d!("DISPLAY", "Color", "output.color",
           "auto/always/never",
           "ANSI color usage. auto = detect TTY, NO_COLOR.",
           K::Enum(&["auto","always","never"]),
           |c| s(&c.output.color),
           |c, v| if let Str(x) = v { c.output.color = x; }),
        d!("DISPLAY", "Charset", "output.charset",
           "auto/unicode/ascii",
           "Unicode box-drawing or pure ASCII. auto = detect terminal.",
           K::Enum(&["auto","unicode","ascii"]),
           |c| s(&c.output.charset),
           |c, v| if let Str(x) = v { c.output.charset = x; }),
        d!("DISPLAY", "Show spinner", "output.show_spinner",
           "live indicator",
           "Animated spinner during ops. Off in potato mode.",
           K::Bool,
           |c| Bool(c.output.show_spinner),
           |c, v| if let Bool(x) = v { c.output.show_spinner = x; }),
        d!("DISPLAY", "Show banner", "output.show_banner",
           "ASCII logo",
           "Show the LamQuant ASCII logo on main screen.",
           K::Bool,
           |c| Bool(c.output.show_banner),
           |c, v| if let Bool(x) = v { c.output.show_banner = x; }),
        d!("DISPLAY", "Splash duration (s)", "output.splash_duration",
           "0 = off",
           "Splash screen duration in seconds. 0 = skip.",
           K::Float,
           |c| Float(c.output.splash_duration),
           |c, v| if let Float(x) = v { c.output.splash_duration = x; }),
        d!("DISPLAY", "Autocomplete", "output.autocomplete",
           "tab completion",
           "Tab-completion in path prompts.",
           K::Bool,
           |c| Bool(c.output.autocomplete),
           |c, v| if let Bool(x) = v { c.output.autocomplete = x; }),
        d!("DISPLAY", "Minimal UI", "output.minimal_ui",
           "low-res render",
           "Drop banner + splash + spinner + color. Slower refresh tick. \
            Use on slow SSH or low-power terminals.",
           K::Bool,
           |c| Bool(c.output.minimal_ui),
           |c, v| if let Bool(x) = v { c.output.minimal_ui = x; }),
        d!("DISPLAY", "Allow root", "output.allow_root",
           "sudo allowed",
           "Permit running as root without warning.",
           K::Bool,
           |c| Bool(c.output.allow_root),
           |c, v| if let Bool(x) = v { c.output.allow_root = x; }),
        d!("DISPLAY", "Warn root", "output.warn_root",
           "show warning",
           "Show a warning when running as root.",
           K::Bool,
           |c| Bool(c.output.warn_root),
           |c, v| if let Bool(x) = v { c.output.warn_root = x; }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_descriptor_has_known_section() {
        // section_rank returns usize::MAX for unknown sections so a
        // typo in the descriptor table doesn't silently rank to the
        // top. This test fails loudly if anyone introduces a row
        // with a section string not in the canonical 9-bucket list.
        let descriptors = build_descriptors();
        for d in &descriptors {
            assert!(
                section_rank(d.section) != usize::MAX,
                "descriptor `{}` uses unknown section `{}` — add it to \
                 section_rank() or fix the typo",
                d.dotpath,
                d.section,
            );
        }
    }

    #[test]
    fn from_cfg_groups_by_section_order() {
        // No filter → visible is descriptors sorted by section_rank.
        // Two-descriptors-of-same-section guarantee they're adjacent
        // in the panel even if the source table interleaves them.
        // Also verify stable sort: within each bucket, descriptor
        // indices must be monotonically increasing -- meaning the
        // source-table insertion order is preserved per-bucket.
        let p = SettingsPanel::from_cfg(LamQuantConfig::default());
        assert!(
            !p.visible.is_empty(),
            "default panel should expose all descriptors with empty filter"
        );
        let mut prev_rank = 0usize;
        let mut prev_idx_in_bucket: i64 = -1;
        for &desc_i in &p.visible {
            let sec = p.descriptors[desc_i].section;
            let r = section_rank(sec);
            assert!(
                r >= prev_rank,
                "section order regressed at descriptor `{}`",
                p.descriptors[desc_i].dotpath,
            );
            if r > prev_rank {
                // New bucket: reset stability tracker.
                prev_idx_in_bucket = -1;
            }
            assert!(
                (desc_i as i64) > prev_idx_in_bucket,
                "intra-bucket order regressed at descriptor `{}` \
                 (section={}, idx={}, prev={}) — stable sort broken",
                p.descriptors[desc_i].dotpath,
                sec,
                desc_i,
                prev_idx_in_bucket,
            );
            prev_idx_in_bucket = desc_i as i64;
            prev_rank = r;
        }
    }

    #[test]
    fn refilter_keeps_section_order() {
        // refilter() (not from_cfg) must also produce a bucket-
        // sorted visible[]. Exercise it explicitly with a non-empty
        // search query so we exercise the same code path users hit
        // while typing in the search box.
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        p.search_query = "verify".to_string();
        p.refilter();
        // Guard against vacuous pass: if a future descriptor rename
        // drops every match for "verify", the for-loop body never
        // runs and the test silently green-lights any section-order
        // regression. The query was chosen to match multiple SAFETY
        // rows (verify_after_write, verify_outliers, verify_skipped)
        // so >=1 is a safe lower bound.
        assert!(
            !p.visible.is_empty(),
            "refilter with 'verify' should match >=1 descriptor — \
             update the query if no `verify` rows remain in the table"
        );
        let ranks: Vec<usize> = p
            .visible
            .iter()
            .map(|i| section_rank(p.descriptors[*i].section))
            .collect();
        let mut prev = 0usize;
        for r in &ranks {
            assert!(*r >= prev, "post-refilter section order regressed: {:?}", ranks);
            prev = *r;
        }
    }

    #[test]
    fn descriptors_get_set_round_trip() {
        let mut cfg = LamQuantConfig::default();
        let descriptors = build_descriptors();
        for d in &descriptors {
            let v = (d.get)(&cfg);
            (d.set)(&mut cfg, v.clone());
            // Same value → same display
            let v2 = (d.get)(&cfg);
            assert_eq!(
                v.display(),
                v2.display(),
                "round-trip failed for {}",
                d.dotpath
            );
        }
    }

    #[test]
    fn cycle_bool_toggles() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        // Find first bool descriptor
        let bool_idx = p
            .descriptors
            .iter()
            .position(|d| matches!(d.kind, SettingKind::Bool))
            .unwrap();
        p.cursor = p.visible.iter().position(|i| *i == bool_idx).unwrap();
        let before = (p.descriptors[bool_idx].get)(&p.cfg);
        p.cycle_current(true);
        let after = (p.descriptors[bool_idx].get)(&p.cfg);
        match (before, after) {
            (SettingValue::Bool(a), SettingValue::Bool(b)) => assert_ne!(a, b),
            _ => panic!("expected bool"),
        }
        assert!(p.dirty);
    }

    #[test]
    fn cycle_enum_advances() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        // Find backend.mode (enum auto/rust/python/custom) and point
        // the cursor at its position in `visible` — descriptors[]
        // insertion order doesn't match visible[] order after the
        // canonical-bucket sort.
        let backend_idx = p
            .descriptors
            .iter()
            .position(|d| d.dotpath == "backend.mode")
            .expect("backend.mode descriptor present");
        p.cursor = p
            .visible
            .iter()
            .position(|i| *i == backend_idx)
            .expect("backend.mode visible");
        let before = (p.descriptors[backend_idx].get)(&p.cfg);
        p.cycle_current(true);
        let after = (p.descriptors[backend_idx].get)(&p.cfg);
        assert_ne!(before.display(), after.display());
        // Cycle back lands on previous
        p.cycle_current(false);
        let back = (p.descriptors[backend_idx].get)(&p.cfg);
        assert_eq!(back.display(), before.display());
    }

    #[test]
    fn search_filters_by_label() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        let total = p.descriptors.len();
        p.search_query = "workers".into();
        p.refilter();
        assert!(p.visible.len() < total, "search should reduce visible set");
        assert!(p
            .visible
            .iter()
            .any(|i| p.descriptors[*i].dotpath == "compute.workers"));
    }

    #[test]
    fn search_empty_shows_all() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        let total = p.descriptors.len();
        p.search_query = "this-matches-nothing-zzz".into();
        p.refilter();
        assert_eq!(p.visible.len(), 0);
        p.search_query.clear();
        p.refilter();
        assert_eq!(p.visible.len(), total);
    }

    #[test]
    fn edit_int_commits_on_enter() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        // Find compute.workers
        let idx = p
            .descriptors
            .iter()
            .position(|d| d.dotpath == "compute.workers")
            .unwrap();
        p.cursor = p.visible.iter().position(|i| *i == idx).unwrap();
        assert!(p.enter_edit());
        p.edit_buffer = Some("12".into());
        p.commit_edit().expect("commit");
        match (p.descriptors[idx].get)(&p.cfg) {
            SettingValue::Int(n) => assert_eq!(n, 12),
            _ => panic!("expected int"),
        }
        assert!(p.dirty);
    }

    #[test]
    fn edit_int_rejects_garbage() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        let idx = p
            .descriptors
            .iter()
            .position(|d| d.dotpath == "compute.workers")
            .unwrap();
        p.cursor = p.visible.iter().position(|i| *i == idx).unwrap();
        p.enter_edit();
        p.edit_buffer = Some("not-an-int".into());
        assert!(p.commit_edit().is_err());
    }

    #[test]
    fn edit_str_array_splits_on_comma() {
        let mut p = SettingsPanel::from_cfg(LamQuantConfig::default());
        let idx = p
            .descriptors
            .iter()
            .position(|d| d.dotpath == "input.extensions")
            .unwrap();
        p.cursor = p.visible.iter().position(|i| *i == idx).unwrap();
        p.enter_edit();
        p.edit_buffer = Some("edf, bdf , mef".into());
        p.commit_edit().unwrap();
        match (p.descriptors[idx].get)(&p.cfg) {
            SettingValue::StrArr(v) => assert_eq!(v, vec!["edf", "bdf", "mef"]),
            _ => panic!(),
        }
    }
}
