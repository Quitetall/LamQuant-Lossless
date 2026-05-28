//! Codec tutorial — quick how-to for compress/decompress/verify/inspect.
//! Single scrollable screen ported from Python `_codec_tutorial`.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Default)]
pub struct TutorialPanel {
    scroll: u16,
}

impl TutorialPanel {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Panel for TutorialPanel {
    fn id(&self) -> &str {
        "tutorial"
    }
    fn title(&self) -> &str {
        "How to use LamQuant Codec"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(
                " How to use LamQuant Codec ",
                theme::heading(),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let sect = |name: &str| Line::from(Span::styled(format!("  {}", name), theme::title()));
        let dim = |s: &str| Line::from(Span::styled(format!("  {}", s), theme::dim()));
        let cmd = |s: &str| Line::from(Span::styled(format!("    {}", s), theme::normal()));
        let blank = || Line::from("");

        let lines = vec![
            blank(),
            sect("COMPRESS (EDF → LML)"),
            blank(),
            dim("Interactive:"),
            cmd("lml  → [1] LML Lossless → [1] Compress → pick file → run"),
            blank(),
            dim("CLI — one file:"),
            cmd("lml encode recording.edf -o recording.lml"),
            blank(),
            dim("CLI — directory:"),
            cmd("lml encode /data/edf/ -o /data/lml/ -j 8 -r"),
            blank(),
            dim("Resume interrupted run:"),
            cmd("lml encode /data/edf/ -o /data/lml/ -r --skip-existing"),
            blank(),
            sect("DECOMPRESS (LML → raw int32 LE)"),
            blank(),
            dim("Interactive:"),
            cmd("lml → [1] LML Lossless → [2] Decompress → pick file → run"),
            blank(),
            dim("CLI:"),
            cmd("lml decode /data/lml/ -o /data/restored/ -r"),
            blank(),
            sect("VERIFY"),
            blank(),
            dim("Per-window CRC-32 + per-file SHA-256:"),
            cmd("lml verify /data/lml/ -r"),
            blank(),
            dim("Verify a manifest:"),
            cmd("lml verify-manifest /data/lml/manifest.lml.json"),
            blank(),
            sect("INSPECT"),
            blank(),
            dim("Header metadata without decoding:"),
            cmd("lml info recording.lml"),
            blank(),
            dim("Per-channel statistics:"),
            cmd("lml stats recording.lml"),
            blank(),
            sect("ARCHIVE"),
            blank(),
            dim("Pack a directory of LML files into a single .lma:"),
            cmd("lml archive /data/lml/ -o study.lma"),
            blank(),
            dim("Extract:"),
            cmd("lml extract study.lma -o /data/lml/ --verify"),
            blank(),
            sect("KEY CONCEPTS"),
            blank(),
            Line::from(Span::styled(
                "  • LML = lossless. Bit-perfect round-trip — every sample preserved.",
                theme::normal(),
            )),
            Line::from(Span::styled(
                "  • LMQ = neural lossy. Higher CR, controlled quality loss.",
                theme::normal(),
            )),
            Line::from(Span::styled(
                "  • Rust backend ~200 MB/s; Python fallback ~15 MB/s. Auto-detect.",
                theme::normal(),
            )),
            Line::from(Span::styled(
                "  • CRC-32 per window + SHA-256 per file on every output.",
                theme::normal(),
            )),
            Line::from(Span::styled(
                "  • --skip-existing makes any batch command resumable.",
                theme::normal(),
            )),
            blank(),
            Line::from(Span::styled(
                "  Full reference: lml --help, lml <subcommand> --help",
                theme::dim(),
            )),
            blank(),
            Line::from(Span::styled(
                if theme::ascii_only() {
                    "  [Up/Dn PgUp/PgDn] scroll  [Esc/b/Enter] back"
                } else {
                    "  [↑↓ PgUp/PgDn] scroll  [Esc/b/Enter] back"
                },
                theme::dim(),
            )),
        ];

        f.render_widget(
            Paragraph::new(lines)
                .scroll((self.scroll, 0))
                .wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll = self.scroll.saturating_sub(1);
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll = self.scroll.saturating_add(1);
                PanelAction::Consumed
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(8);
                PanelAction::Consumed
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(8);
                PanelAction::Consumed
            }
            KeyCode::Home => {
                self.scroll = 0;
                PanelAction::Consumed
            }
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') | KeyCode::Enter => {
                PanelAction::Back
            }
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }
}
