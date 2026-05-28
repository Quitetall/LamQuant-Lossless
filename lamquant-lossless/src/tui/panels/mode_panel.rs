//! Mode panel — boxed Operations + Status view for LML Lossless and
//! LMQ Neural modes. Replaces the plain `MenuPanel` registrations for
//! `SCREEN_LOSSLESS` / `SCREEN_NEURAL`.
//!
//! Direct port of the old Python `_codec_hub` mode-specific branch
//! (lamquant.py L239-265): banner, mode subtitle, bordered Operations
//! group (5 ops), bordered Status group (Mode / Backend / Workers /
//! Verify / Noise bits or Quality), `[m]` toggle to swap between LML
//! and LMQ.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

/// Which codec mode the panel represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMode {
    Lossless,
    Neural,
}

impl CodecMode {
    fn screen_id(self) -> &'static str {
        match self {
            CodecMode::Lossless => router::SCREEN_LOSSLESS,
            CodecMode::Neural => router::SCREEN_NEURAL,
        }
    }
    fn other_screen(self) -> &'static str {
        match self {
            CodecMode::Lossless => router::SCREEN_NEURAL,
            CodecMode::Neural => router::SCREEN_LOSSLESS,
        }
    }
    fn other_label(self) -> &'static str {
        match self {
            CodecMode::Lossless => "LMQ",
            CodecMode::Neural => "LML",
        }
    }
    fn display(self) -> &'static str {
        match self {
            CodecMode::Lossless => "Lossless (LML)",
            CodecMode::Neural => "Neural (LMQ)",
        }
    }
    fn ext(self) -> &'static str {
        match self {
            CodecMode::Lossless => "lml",
            CodecMode::Neural => "lmq",
        }
    }
    fn src(self) -> &'static str {
        match self {
            CodecMode::Lossless => "EDF/BDF",
            CodecMode::Neural => "EDF/NPY",
        }
    }
    fn subtitle(self) -> &'static str {
        match self {
            CodecMode::Lossless => "Bit-perfect EEG compression",
            CodecMode::Neural => "Adaptive SNAC · Ternary encoder",
        }
    }
    fn banner_title(self) -> &'static str {
        match self {
            CodecMode::Lossless => "LamQuant Lossless Codec",
            CodecMode::Neural => "LamQuant Neural Codec",
        }
    }
    fn op_encode_default(self) -> &'static str {
        // Default Lossless route when the user's
        // `codec.lossless_default_mode` setting is `LmaArchive` (or
        // `Prompt` is bypassed). The bare-LML op (`OP_ENCODE`) skips
        // the LML -> zstd -> store cascade and drops every non-EDF
        // sibling -- the footgun that cost one full TUEG migration
        // cycle, so bare-LML stays CLI-only behind
        // `--bare-lml --i-understand-data-loss`. The TUI picker on
        // `[1] Compress` chooses between OP_ENCODE_LMA and the
        // sibling-preserving OP_ENCODE_LML_SIBLINGS per the user's
        // setting (see `mode_panel::handle_event`).
        match self {
            CodecMode::Lossless => router::OP_ENCODE_LMA,
            CodecMode::Neural => router::OP_ENCODE_NEURAL,
        }
    }
    fn op_decode(self) -> &'static str {
        match self {
            CodecMode::Lossless => router::OP_DECODE,
            CodecMode::Neural => router::OP_DECODE_NEURAL,
        }
    }
}

pub struct ModePanel {
    id: String,
    mode: CodecMode,
}

impl ModePanel {
    pub fn new(mode: CodecMode) -> Self {
        Self {
            id: mode.screen_id().to_string(),
            mode,
        }
    }

    /// Resolve the `[1] Compress` target for the panel's mode.
    ///
    /// Neural mode has only one encode flavour today, so it always
    /// routes to `OP_ENCODE_NEURAL`. Lossless mode honours the
    /// user's `codec.lossless_default_mode` setting (validated to
    /// `prompt | lma | lml_siblings` in `apply_codec`):
    ///   - `lma`          -> OP_ENCODE_LMA (per-EDF .lma archives)
    ///   - `lml_siblings` -> OP_ENCODE_LML_SIBLINGS (.lml + copies)
    ///   - `prompt`       -> SCREEN_LOSSLESS_PROMPT (ask user)
    /// Unknown values fall back to the prompt screen rather than
    /// silently committing to one path; apply_codec rejects unknown
    /// values at TOML load, so the fallback is defensive only.
    fn dispatch_compress(&self, state: &AppState) -> PanelAction {
        if !matches!(self.mode, CodecMode::Lossless) {
            return PanelAction::Navigate(self.mode.op_encode_default().to_string());
        }
        match state.cfg.codec.lossless_default_mode.as_str() {
            "lma" => PanelAction::Navigate(router::OP_ENCODE_LMA.to_string()),
            "lml_siblings" => PanelAction::Navigate(router::OP_ENCODE_LML_SIBLINGS.to_string()),
            _ => PanelAction::Navigate(router::SCREEN_LOSSLESS_PROMPT.to_string()),
        }
    }
}

fn bordered(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(format!(" {} ", title), theme::highlight()))
}

fn opt<'a>(key: &'static str, label: &'static str, desc: String) -> Line<'a> {
    Line::from(vec![
        Span::raw("   "),
        Span::styled(format!("[{}]", key), theme::key_hint()),
        Span::raw("  "),
        Span::styled(format!("{:<14}", label), theme::normal()),
        Span::styled(desc, theme::dim()),
    ])
}

impl Panel for ModePanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        match self.mode {
            CodecMode::Lossless => "LML Lossless",
            CodecMode::Neural => "LMQ Neural",
        }
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // title + subtitle (no big banner)
                Constraint::Length(7), // Operations (5 entries + borders)
                Constraint::Length(7), // Status (5 entries + borders)
                Constraint::Length(1), // footer
                Constraint::Min(0),
            ])
            .split(area);

        // Title — single text banner, no ascii logo. The header acts
        // as the panel's "banner" and reads `LamQuant Lossless Codec`
        // (or Neural).
        let title_lines = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(self.mode.banner_title(), theme::heading()),
                Span::raw("    "),
                Span::styled(self.mode.subtitle(), theme::dim()),
            ]),
            Line::from(""),
        ];
        f.render_widget(Paragraph::new(title_lines), chunks[0]);

        // Operations
        let ops_block = bordered("Operations");
        let ops_inner = ops_block.inner(chunks[1]);
        f.render_widget(ops_block, chunks[1]);
        let ext = self.mode.ext();
        let src = self.mode.src();
        f.render_widget(
            Paragraph::new(vec![
                opt("1", "Compress", format!("{} → .{}", src, ext)),
                opt("2", "Decompress", format!(".{} → {}", ext, src)),
                opt("3", "Verify", "CRC + SHA-256 integrity check".into()),
                opt("4", "Info", "File metadata (no decode)".into()),
                opt("5", "Stats", "Per-channel signal statistics".into()),
            ]),
            ops_inner,
        );

        // Status
        let status_block = bordered("Status");
        let status_inner = status_block.inner(chunks[2]);
        f.render_widget(status_block, chunks[2]);
        let backend = format!(
            "{} (lml {})",
            state.cfg.backend.mode,
            env!("CARGO_PKG_VERSION")
        );
        let workers = state.cfg.compute.workers.to_string();
        let verify = if state.cfg.integrity.verify_after_write {
            "ON".to_string()
        } else {
            "OFF".to_string()
        };
        let last_line = match self.mode {
            CodecMode::Lossless => {
                let nb = state.cfg.codec.noise_bits;
                let v = if nb == 0 {
                    "0  lossless".to_string()
                } else {
                    format!("{}  strip {} LSBs", nb, nb)
                };
                ("Noise bits", v, nb == 0)
            }
            CodecMode::Neural => ("Quality", "clinical  (configurable)".to_string(), true),
        };
        let kv = |k: &str, v: String, hi: bool| {
            Line::from(vec![
                Span::raw("   "),
                Span::styled(format!("{:<13}", k), theme::dim()),
                Span::styled(
                    v,
                    if hi {
                        theme::highlight()
                    } else {
                        theme::normal()
                    },
                ),
            ])
        };
        f.render_widget(
            Paragraph::new(vec![
                kv("Mode", self.mode.display().to_string(), true),
                kv("Backend", backend, true),
                kv("Workers", workers, false),
                kv("Verify", verify, state.cfg.integrity.verify_after_write),
                kv(last_line.0, last_line.1, last_line.2),
            ]),
            status_inner,
        );

        // Footer keys — no divider above (boxes already have borders).
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw("  "),
                Span::styled("[h]", theme::key_hint()),
                Span::styled(" How-to   ", theme::dim()),
                Span::styled("[m]", theme::key_hint()),
                Span::styled(
                    format!(" Switch {}   ", self.mode.other_label()),
                    theme::dim(),
                ),
                Span::styled("[b]", theme::key_hint()),
                Span::styled(" Back   ", theme::dim()),
                Span::styled("[q]", theme::key_hint()),
                Span::styled(" Main menu   ", theme::dim()),
                Span::styled("[x]", theme::key_hint()),
                Span::styled(" Exit", theme::dim()),
            ])),
            chunks[3],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('1') => self.dispatch_compress(state),
            KeyCode::Char('2') => PanelAction::Navigate(self.mode.op_decode().to_string()),
            KeyCode::Char('3') => PanelAction::Navigate(router::OP_VERIFY.to_string()),
            KeyCode::Char('4') => PanelAction::Navigate(router::OP_INFO.to_string()),
            KeyCode::Char('5') => PanelAction::Navigate(router::OP_STATS.to_string()),
            KeyCode::Char('m') => PanelAction::Navigate(self.mode.other_screen().to_string()),
            KeyCode::Char('h') | KeyCode::Char('?') => {
                PanelAction::Navigate(router::SCREEN_TUTORIAL.to_string())
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
    use crate::tui::state::AppState;

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), crossterm::event::KeyModifiers::NONE)
    }

    /// `[6]` was the redundant "Pack LMA" alias for `[1]`. After
    /// the dedup commit it must be an Ignored no-op (not a hidden
    /// alias that still routes through). This test fails if the
    /// keybind is reintroduced.
    #[test]
    fn key_6_is_ignored_after_dedup() {
        let mut panel = ModePanel::new(CodecMode::Lossless);
        let state = AppState::default();
        assert!(matches!(
            panel.handle_event(key('6'), &state),
            PanelAction::Ignored
        ));
        let mut panel_n = ModePanel::new(CodecMode::Neural);
        assert!(matches!(
            panel_n.handle_event(key('6'), &state),
            PanelAction::Ignored
        ));
    }

    /// Neural mode `[1]` always routes to OP_ENCODE_NEURAL. Lossless
    /// `[1]` reads `codec.lossless_default_mode` and dispatches per
    /// the user's choice -- prompt vs lma vs lml_siblings. Tests
    /// below cover the Lossless branches in detail; this one locks
    /// the Neural invariant + the Lossless-default (Prompt) path.
    #[test]
    fn key_1_neural_always_routes_to_neural_encode() {
        let mut panel = ModePanel::new(CodecMode::Neural);
        let state = AppState::default();
        match panel.handle_event(key('1'), &state) {
            PanelAction::Navigate(target) => assert_eq!(target, router::OP_ENCODE_NEURAL),
            other => panic!("expected Navigate(OP_ENCODE_NEURAL), got {:?}", other),
        }
    }

    /// Default `lossless_default_mode = "prompt"` -- [1] shows the
    /// sub-prompt overlay so the user can pick LMA vs LML+siblings.
    /// This is the most common path for new users (they haven't
    /// touched Settings yet) and locks the consent affordance.
    #[test]
    fn key_1_lossless_prompt_routes_to_prompt_screen() {
        let mut panel = ModePanel::new(CodecMode::Lossless);
        let mut state = AppState::default();
        // Default is already "prompt"; set explicitly so the test
        // doesn't silently regress when the Default impl moves.
        state.cfg.codec.lossless_default_mode = "prompt".into();
        match panel.handle_event(key('1'), &state) {
            PanelAction::Navigate(target) => {
                assert_eq!(target, router::SCREEN_LOSSLESS_PROMPT);
            }
            other => panic!("expected Navigate(SCREEN_LOSSLESS_PROMPT), got {:?}", other),
        }
    }

    /// `lossless_default_mode = "lma"` -- [1] bypasses the prompt
    /// and goes straight to OP_ENCODE_LMA (today's per-EDF archive
    /// flow). Saves a keystroke for users who always pick LMA.
    #[test]
    fn key_1_lossless_lma_routes_to_lma() {
        let mut panel = ModePanel::new(CodecMode::Lossless);
        let mut state = AppState::default();
        state.cfg.codec.lossless_default_mode = "lma".into();
        match panel.handle_event(key('1'), &state) {
            PanelAction::Navigate(target) => assert_eq!(target, router::OP_ENCODE_LMA),
            other => panic!("expected Navigate(OP_ENCODE_LMA), got {:?}", other),
        }
    }

    /// `lossless_default_mode = "lml_siblings"` -- [1] bypasses the
    /// prompt and goes straight to OP_ENCODE_LML_SIBLINGS (the new
    /// whole-tree mode added in D/5).
    #[test]
    fn key_1_lossless_lml_siblings_routes_to_lml_siblings() {
        let mut panel = ModePanel::new(CodecMode::Lossless);
        let mut state = AppState::default();
        state.cfg.codec.lossless_default_mode = "lml_siblings".into();
        match panel.handle_event(key('1'), &state) {
            PanelAction::Navigate(target) => {
                assert_eq!(target, router::OP_ENCODE_LML_SIBLINGS);
            }
            other => panic!("expected Navigate(OP_ENCODE_LML_SIBLINGS), got {:?}", other),
        }
    }

    /// Defensive: an unexpected setting value (apply_codec rejects
    /// these at load, but a stale process or test could plant one)
    /// must NOT silently encode -- fall back to the prompt overlay
    /// so the user explicitly chooses a mode.
    #[test]
    fn key_1_lossless_unknown_value_falls_back_to_prompt() {
        let mut panel = ModePanel::new(CodecMode::Lossless);
        let mut state = AppState::default();
        state.cfg.codec.lossless_default_mode = "garbage".into();
        match panel.handle_event(key('1'), &state) {
            PanelAction::Navigate(target) => {
                assert_eq!(target, router::SCREEN_LOSSLESS_PROMPT);
            }
            other => panic!("expected Navigate(SCREEN_LOSSLESS_PROMPT), got {:?}", other),
        }
    }
}
