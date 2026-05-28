//! OpenHuman Portal — Install · Update · Validate hub.
//!
//! Replaces the linear MenuPanel that previously lived at SCREEN_SETUP.
//! Three sections: Status (auto-detected), Install (1–5), Maintain (v/u/c).
//! `[1]` Everything routes to the linear `WizardPanel` for first-time setup.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::path::Path;
use std::process::Command;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

const PORTAL_LOGO: &[&str] = &[
    "  ╔═╗╔═╗╦═╗╔╦╗╔═╗╦  ",
    "  ╠═╝║ ║╠╦╝ ║ ╠═╣║  ",
    "  ╩  ╚═╝╩╚═ ╩ ╩ ╩╩═╝",
];

#[derive(Clone)]
struct StatusEntry {
    label: &'static str,
    detail: String,
    ok: bool,
}

pub struct PortalPanel {
    id: String,
    statuses: Vec<StatusEntry>,
}

impl PortalPanel {
    pub fn new() -> Self {
        Self {
            id: "portal".to_string(),
            statuses: detect_statuses(),
        }
    }

    pub fn refresh(&mut self) {
        self.statuses = detect_statuses();
    }
}

impl Default for PortalPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn cmd_ok(prog: &str, args: &[&str]) -> bool {
    Command::new(prog)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn detect_statuses() -> Vec<StatusEntry> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let venv_path = cwd.join(".venv");
    let lml_local = cwd.join("target/release/lml");
    let gui_local = cwd.join("gui/src-tauri/target/release/lamquant-gui");

    let python_ok = cmd_ok("python3", &["--version"]) || cmd_ok("python", &["--version"]);
    let cargo_ok = cmd_ok("cargo", &["--version"]);
    let node_ok = cmd_ok("node", &["--version"]);
    let venv_ok = venv_path.is_dir();
    let codec_ok = cmd_ok("python3", &["-c", "import lamquant_codec"]);
    let lml_ok = cmd_ok("which", &["lml"]) || lml_local.exists();
    let gui_ok = cmd_ok("which", &["lamquant-gui"]) || gui_local.exists();

    vec![
        StatusEntry {
            label: "Python",
            detail: if python_ok {
                "found".into()
            } else {
                "not found".into()
            },
            ok: python_ok,
        },
        StatusEntry {
            label: "Rust/Cargo",
            detail: if cargo_ok {
                "found".into()
            } else {
                "not found".into()
            },
            ok: cargo_ok,
        },
        StatusEntry {
            label: "Node.js",
            detail: if node_ok {
                "found".into()
            } else {
                "not found".into()
            },
            ok: node_ok,
        },
        StatusEntry {
            label: "Virtual env",
            detail: if venv_ok {
                ".venv/".into()
            } else {
                "not present".into()
            },
            ok: venv_ok,
        },
        StatusEntry {
            label: "lamquant_codec",
            detail: if codec_ok {
                "importable".into()
            } else {
                "not importable".into()
            },
            ok: codec_ok,
        },
        StatusEntry {
            label: "lml binary",
            detail: if lml_ok {
                "found".into()
            } else {
                "not built".into()
            },
            ok: lml_ok,
        },
        StatusEntry {
            label: "Vision GUI",
            detail: if gui_ok {
                "built".into()
            } else {
                "not built".into()
            },
            ok: gui_ok,
        },
    ]
}

fn bordered_block(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(format!(" {} ", title), theme::highlight()))
}

impl Panel for PortalPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Install & Setup"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let logo_h: u16 = (PORTAL_LOGO.len() + 2) as u16;
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(logo_h), // banner + subtitle
                Constraint::Length(9),      // Status box (7 entries + borders)
                Constraint::Length(1),      // gap
                Constraint::Length(7),      // Install box (5 entries + borders)
                Constraint::Length(1),      // gap
                Constraint::Length(6),      // Maintain box (3 entries + footer + borders)
                Constraint::Min(0),
            ])
            .split(area);

        // ── Banner ───────────────────────────────────────────────────────
        let mut banner: Vec<Line> = PORTAL_LOGO
            .iter()
            .map(|l| Line::from(Span::styled(*l, theme::title())))
            .collect();
        banner.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("OpenHuman Portal", theme::highlight()),
            Span::styled("  ·  Install  ·  Update  ·  Validate", theme::dim()),
        ]));
        f.render_widget(Paragraph::new(banner), chunks[0]);

        // ── Status box ───────────────────────────────────────────────────
        let status_block = bordered_block("Status");
        let status_inner = status_block.inner(chunks[1]);
        f.render_widget(status_block, chunks[1]);
        let status_lines: Vec<Line> = self
            .statuses
            .iter()
            .map(|s| {
                let (mark, mark_style) = if s.ok {
                    ("✓", theme::success())
                } else {
                    ("✗", theme::error())
                };
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(mark, mark_style),
                    Span::raw("  "),
                    Span::styled(format!("{:<16}", s.label), theme::normal()),
                    Span::styled(s.detail.clone(), theme::dim()),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(status_lines), status_inner);

        // ── Install box ──────────────────────────────────────────────────
        let install_block = bordered_block("Install");
        let install_inner = install_block.inner(chunks[3]);
        f.render_widget(install_block, chunks[3]);
        let install_rows = [
            ("1", "Everything", "full install (recommended)"),
            ("2", "Python pipeline only", "venv + pip install -e .[all]"),
            ("3", "Rust codec only", "cargo build --release"),
            ("4", "Vision GUI only", "npm + tauri build"),
            ("5", "Firmware tools", "Pico SDK guidance"),
        ];
        let install_lines: Vec<Line> = install_rows
            .iter()
            .map(|(k, l, d)| {
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(format!("[{}]", k), theme::key_hint()),
                    Span::raw("  "),
                    Span::styled(format!("{:<22}", l), theme::normal()),
                    Span::styled(d.to_string(), theme::dim()),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(install_lines), install_inner);

        // ── Maintain box ─────────────────────────────────────────────────
        let maintain_block = bordered_block("Maintain");
        let maintain_inner = maintain_block.inner(chunks[5]);
        f.render_widget(maintain_block, chunks[5]);
        let maintain_rows = [
            ("v", "Validate install", "imports, binaries, smoke test"),
            ("u", "Update (pip + cargo)", "rebuild from source"),
            (
                "c",
                "Clean build artifacts",
                "target/, .venv/, node_modules/",
            ),
            ("n", "What's new", "patchnotes from CHANGELOG.md"),
        ];
        let mut maintain_lines: Vec<Line> = maintain_rows
            .iter()
            .map(|(k, l, d)| {
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(format!("[{}]", k), theme::key_hint()),
                    Span::raw("  "),
                    Span::styled(format!("{:<22}", l), theme::normal()),
                    Span::styled(d.to_string(), theme::dim()),
                ])
            })
            .collect();
        maintain_lines.push(Line::from(vec![
            Span::raw(" "),
            Span::styled("[b]", theme::key_hint()),
            Span::styled(" Back   ", theme::dim()),
            Span::styled("[q]", theme::key_hint()),
            Span::styled(" Main menu   ", theme::dim()),
            Span::styled("[x]", theme::key_hint()),
            Span::styled(" Exit", theme::dim()),
        ]));
        f.render_widget(Paragraph::new(maintain_lines), maintain_inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('1') => PanelAction::Navigate(router::SCREEN_WIZARD.to_string()),
            KeyCode::Char('2') => PanelAction::Navigate(router::LAUNCH_SETUP_PIP.to_string()),
            KeyCode::Char('3') => PanelAction::Navigate(router::LAUNCH_SETUP_CARGO.to_string()),
            KeyCode::Char('4') => PanelAction::StatusMessage(
                "GUI build:  cd gui && npm install && npm run tauri build".into()),
            KeyCode::Char('5') => PanelAction::StatusMessage(
                "Pico SDK:  https://github.com/raspberrypi/pico-sdk  ·  see lamquant-firmware/README".into()),
            KeyCode::Char('v') => PanelAction::Navigate(router::SCREEN_SYSCHECK.to_string()),
            KeyCode::Char('u') => PanelAction::StatusMessage(
                "Update:  pip install -U -e .[all]  &&  cargo build --release".into()),
            KeyCode::Char('c') => PanelAction::StatusMessage(
                "Clean:  cargo clean  &&  rm -rf .venv node_modules".into()),
            KeyCode::Char('n') => {
                // Show first non-empty heading from CHANGELOG.md as a quick teaser.
                let head = std::fs::read_to_string("CHANGELOG.md")
                    .ok()
                    .and_then(|s| s.lines().take(40)
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                        .first().cloned())
                    .unwrap_or_else(|| "CHANGELOG.md not found in cwd".to_string());
                PanelAction::StatusMessage(format!("What's new: {}  ·  see CHANGELOG.md", head))
            },
            KeyCode::Char('r') => { self.refresh(); PanelAction::StatusMessage("Re-detected install state.".into()) },
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            KeyCode::Char('h') | KeyCode::Char('?') => PanelAction::Navigate(router::SCREEN_HELP.to_string()),
            _ => PanelAction::Ignored,
        }
    }
}

#[allow(dead_code)]
fn _unused(_: &Path) {} // placeholder to keep std::path::Path import alive when refactoring
