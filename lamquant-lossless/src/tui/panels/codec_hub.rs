//! Codec Hub — entry point for LML lossless and LMQ neural codecs.
//!
//! Four boxed sections: Codec Modes (1, 2), File Tools (3–7),
//! Export & Recovery (9, e, n, w, R), and Status (live config + last
//! paths from history — read directly from `&AppState`, no caching).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::panels::splash;
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

/// Drop-priority tiers for the banner: splash::LOGO block letters +
/// "── CODEC HUB ──" tag (one banner unit, hand-crafted style); shrink
/// to a single-line dim tag; drop entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BannerKind {
    Full,
    Small,
    None,
}

pub struct CodecHubPanel {
    id: String,
}

impl CodecHubPanel {
    pub fn new() -> Self {
        Self {
            id: "codec_hub".to_string(),
        }
    }

    fn render_footer(&self, f: &mut Frame, area: Rect) {
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw("  "),
                Span::styled("[h]", theme::key_hint()),
                Span::styled(" How-to   ", theme::dim()),
                Span::styled("[b]", theme::key_hint()),
                Span::styled(" Back   ", theme::dim()),
                Span::styled("[q]", theme::key_hint()),
                Span::styled(" Main menu   ", theme::dim()),
                Span::styled("[x]", theme::key_hint()),
                Span::styled(" Exit", theme::dim()),
            ])),
            area,
        );
    }
}

impl Default for CodecHubPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn bordered(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(format!(" {} ", title), theme::highlight()))
}

fn opt<'a>(key: &'static str, label: &'static str, desc: &'static str) -> Line<'a> {
    Line::from(vec![
        Span::raw("   "),
        Span::styled(format!("[{}]", key), theme::key_hint()),
        Span::raw("  "),
        Span::styled(format!("{:<18}", label), theme::normal()),
        Span::styled(desc.to_string(), theme::dim()),
    ])
}

impl Panel for CodecHubPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Codec Hub"
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        // ── Adaptive sizing ──────────────────────────────────────
        // Drop priority (first to disappear → last to disappear):
        //   1. Big banner (full logo)   — replaced with 1-line tag
        //   2. Status box
        //   3. Footer keys (instructions)
        //   4. (modes + tools + export are essentials, never drop)
        // Below the modes+tools+export essentials we render a
        // "terminal too small" message instead.
        const ESSENTIALS_H: u16 = 5 + 7 + 7; // modes + tools + export
        const STATUS_H: u16 = 8;
        const FOOTER_H: u16 = 1;
        // Full banner: splash::LOGO block-letter ASCII (6 rows) with a
        // "── CODEC HUB ──" tag welded to its bottom edge — one banner
        // unit, same hand-crafted style as the boot splash.
        const BANNER_FULL_H: u16 = (splash::LOGO.len() as u16) + 1;
        const BANNER_SMALL_H: u16 = 1; // single "── LAMQUANT CODEC HUB ──" line
        const TOO_SMALL_MIN: u16 = ESSENTIALS_H; // below this, fallback message

        let h = area.height;

        if h < TOO_SMALL_MIN {
            let msg = format!(
                "Terminal too small — need ≥{} rows, have {}",
                TOO_SMALL_MIN, h,
            );
            let lines = vec![
                Line::from(""),
                Line::from(Span::styled(msg, theme::error())),
                Line::from(""),
                Line::from(Span::styled(
                    "Resize the terminal window or zoom out to continue.",
                    theme::dim(),
                )),
            ];
            f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), area);
            return;
        }

        // Greedy fit: rebuild flags + constraints based on what fits
        // around the always-on essentials. Order of additions matches
        // the REVERSE drop order so the most important extras win the
        // remaining rows first.
        let mut used = ESSENTIALS_H;
        let footer_on = if used + FOOTER_H <= h {
            used += FOOTER_H;
            true
        } else {
            false
        };
        let status_on = if used + STATUS_H <= h {
            used += STATUS_H;
            true
        } else {
            false
        };
        let banner = if used + BANNER_FULL_H <= h {
            used += BANNER_FULL_H;
            BannerKind::Full
        } else if used + BANNER_SMALL_H <= h {
            used += BANNER_SMALL_H;
            BannerKind::Small
        } else {
            BannerKind::None
        };
        let _ = used;

        let mut constraints: Vec<Constraint> = Vec::with_capacity(6);
        match banner {
            BannerKind::Full => constraints.push(Constraint::Length(BANNER_FULL_H)),
            BannerKind::Small => constraints.push(Constraint::Length(BANNER_SMALL_H)),
            BannerKind::None => {}
        }
        constraints.push(Constraint::Length(5)); // Codec Modes
        constraints.push(Constraint::Length(7)); // File Tools
        constraints.push(Constraint::Length(7)); // Export & Recovery
        if status_on {
            constraints.push(Constraint::Length(STATUS_H));
        }
        if footer_on {
            constraints.push(Constraint::Length(FOOTER_H));
        }
        constraints.push(Constraint::Min(0));

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        // Index walker — chunks indexed by what's actually present.
        let mut i: usize = 0;

        // Banner — splash::LOGO block letters + welded tag, both in
        // the hand-crafted ANSI Shadow style. One banner unit, no
        // double-rendering.
        match banner {
            BannerKind::Full => {
                let mut lines: Vec<Line> = splash::LOGO
                    .iter()
                    .map(|l| Line::from(Span::styled(*l, theme::title())))
                    .collect();
                lines.push(Line::from(Span::styled(
                    "─────────────  CODEC HUB  ─────────────",
                    theme::heading(),
                )));
                f.render_widget(
                    Paragraph::new(lines).alignment(Alignment::Center),
                    chunks[i],
                );
                i += 1;
            }
            BannerKind::Small => {
                f.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        "─────────────  LAMQUANT CODEC HUB  ─────────────",
                        theme::heading(),
                    )))
                    .alignment(Alignment::Center),
                    chunks[i],
                );
                i += 1;
            }
            BannerKind::None => {}
        }

        // Codec Modes
        let modes_block = bordered("Codec Modes");
        let modes_inner = modes_block.inner(chunks[i]);
        f.render_widget(modes_block, chunks[i]);
        f.render_widget(
            Paragraph::new(vec![
                opt("1", "LML Lossless", "Open mode panel (! quick-decode)"),
                opt("2", "LMQ Neural", "Open mode panel (@ quick-decode)"),
            ]),
            modes_inner,
        );
        i += 1;

        // File Tools
        let tools_block = bordered("File Tools");
        let tools_inner = tools_block.inner(chunks[i]);
        f.render_widget(tools_block, chunks[i]);
        f.render_widget(
            Paragraph::new(vec![
                opt("3", "Inspect", "File metadata (no decode)"),
                opt("4", "Verify", "CRC + SHA-256 integrity check"),
                opt("5", "Verify manifest", "Check manifest.lml.json"),
                opt("6", "Stats", "Per-channel signal statistics"),
                opt("7", "Browse", "Find LML/LMQ files ([c] copies path)"),
            ]),
            tools_inner,
        );
        i += 1;

        // Export & Recovery
        let exp_block = bordered("Export & Recovery");
        let exp_inner = exp_block.inner(chunks[i]);
        f.render_widget(exp_block, chunks[i]);
        f.render_widget(
            Paragraph::new(vec![
                opt("9", "Benchmark", "Speed test (encode/decode throughput)"),
                opt("e", "Export CSV", ".lml → CSV per-channel rows"),
                opt("n", "Export NPY", ".lml → NumPy .npy"),
                opt("w", "Export raw", ".lml → int32 LE raw"),
                opt("R", "Recover", "Salvage damaged .lml file"),
            ]),
            exp_inner,
        );
        i += 1;

        // Status (only when the tier permits)
        if !status_on {
            // Skip without rendering, but still render footer below if on.
            if footer_on {
                self.render_footer(f, chunks[i]);
                // i += 1; // not needed — last addressed chunk
            }
            return;
        }
        let status_block = bordered("Status");
        let status_inner = status_block.inner(chunks[i]);
        f.render_widget(status_block, chunks[i]);
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
        let nb = state.cfg.codec.noise_bits;
        let nb_str = if nb == 0 {
            "0  lossless".to_string()
        } else {
            format!("{}  strip {} LSBs", nb, nb)
        };
        let last_in = state
            .history
            .recent_inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "(none)".into());
        let last_out = state
            .history
            .recent_outputs
            .first()
            .cloned()
            .unwrap_or_else(|| "(none)".into());
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
                kv("Backend", backend, true),
                kv("Workers", workers, false),
                kv("Verify", verify, state.cfg.integrity.verify_after_write),
                kv("Noise bits", nb_str, nb == 0),
                kv("Last input", last_in, false),
                kv("Last output", last_out, false),
            ]),
            status_inner,
        );
        i += 1;

        // Footer keys (instructions) — only when the tier permits.
        if footer_on {
            self.render_footer(f, chunks[i]);
        }
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            // Codec modes — open the boxed mode panel (Operations + Status).
            // Shifted variants `!` / `@` jump straight to the encode/decode
            // op for power users who don't need the full mode view.
            KeyCode::Char('1') => PanelAction::Navigate(router::SCREEN_LOSSLESS.to_string()),
            KeyCode::Char('!') => PanelAction::Navigate(router::OP_DECODE.to_string()),
            KeyCode::Char('2') => PanelAction::Navigate(router::SCREEN_NEURAL.to_string()),
            KeyCode::Char('@') => PanelAction::Navigate(router::OP_DECODE_NEURAL.to_string()),

            // File tools
            KeyCode::Char('3') => PanelAction::Navigate(router::OP_INFO.to_string()),
            KeyCode::Char('4') => PanelAction::Navigate(router::OP_VERIFY.to_string()),
            KeyCode::Char('5') => PanelAction::Navigate(router::OP_VERIFY_MANIFEST.to_string()),
            KeyCode::Char('6') => PanelAction::Navigate(router::OP_STATS.to_string()),
            KeyCode::Char('7') => PanelAction::Navigate(router::SCREEN_BROWSE.to_string()),

            // Export & Recovery
            KeyCode::Char('9') => PanelAction::Navigate(router::OP_BENCH.to_string()),
            KeyCode::Char('e') => PanelAction::Navigate(router::OP_EXPORT_CSV.to_string()),
            KeyCode::Char('n') => PanelAction::Navigate(router::OP_EXPORT_NPY.to_string()),
            KeyCode::Char('w') => PanelAction::Navigate(router::OP_EXPORT_RAW.to_string()),
            KeyCode::Char('R') => PanelAction::Navigate(router::OP_RECOVER.to_string()),

            // Footer
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
