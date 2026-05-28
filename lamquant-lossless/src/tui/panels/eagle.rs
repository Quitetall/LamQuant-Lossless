//! Eagle Validation Suite — LQS compliance, benchmarks, clinical tasks.
//!
//! Replaces the linear MenuPanel that previously lived at SCREEN_EAGLE.
//! Sections: Compliance (1–3), Benchmarking (4–6), Clinical Validation (7–8),
//! Exploration (9), Registry (p, r). Footer: x export · b back · q exit.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

const EAGLE_LOGO: &[&str] = &[
    "  ███████╗ █████╗  ██████╗ ██╗     ███████╗",
    "  ██╔════╝██╔══██╗██╔════╝ ██║     ██╔════╝",
    "  █████╗  ███████║██║  ███╗██║     █████╗  ",
    "  ██╔══╝  ██╔══██║██║   ██║██║     ██╔══╝  ",
    "  ███████╗██║  ██║╚██████╔╝███████╗███████╗",
    "  ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝",
];

pub struct EaglePanel {
    id: String,
}

impl EaglePanel {
    pub fn new() -> Self {
        Self {
            id: "eagle".to_string(),
        }
    }
}

impl Default for EaglePanel {
    fn default() -> Self {
        Self::new()
    }
}

fn section_header<'a>(title: &'static str, sub: &'static str) -> Line<'a> {
    Line::from(vec![
        Span::raw("    "),
        Span::styled(title, theme::highlight()),
        Span::raw("                          "),
        Span::styled(sub, theme::dim()),
    ])
}

fn option_row<'a>(key: &'static str, label: &'static str, desc: &'static str) -> Line<'a> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("[{}]", key), theme::key_hint()),
        Span::raw("  "),
        Span::styled(format!("{:<24}", label), theme::normal()),
        Span::styled(desc.to_string(), theme::dim()),
    ])
}

impl Panel for EaglePanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Eagle Validation Suite"
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        // ── Banner ───────────────────────────────────────────────────────
        for row in EAGLE_LOGO {
            lines.push(Line::from(Span::styled(*row, theme::title())));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("       "),
            Span::styled(
                "OpenHuman Eagle  ·  Validation Suite for EEG Processing",
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));

        // ── Version row ──────────────────────────────────────────────────
        let version = format!("v{}", env!("CARGO_PKG_VERSION"));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Eagle", theme::key_hint()),
            Span::raw(" "),
            Span::styled("v1.0.0", theme::normal()),
            Span::raw("   "),
            Span::styled("LQS", theme::key_hint()),
            Span::raw(" "),
            Span::styled("v1.0", theme::normal()),
            Span::raw("   "),
            Span::styled("Codec", theme::key_hint()),
            Span::raw(" "),
            Span::styled(format!("LamQuant {}", version), theme::normal()),
        ]));
        lines.push(Line::from(""));

        // ── LamQuant info box (hand-drawn so it sits inline) ─────────────
        let header_label = format!("LamQuant {}", version);
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled(
                format!(
                    "┌─ {} ─────────────────────────────────────────┐",
                    header_label
                ),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled("│ ", theme::dim()),
            Span::styled(format!("{:<8}", "Mode"), theme::dim()),
            Span::styled(
                format!("{:<48}", state.cfg.codec.default_mode.clone()),
                theme::normal(),
            ),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled("│ ", theme::dim()),
            Span::styled(format!("{:<8}", "Target"), theme::dim()),
            Span::styled(format!("{:<48}", "LQS-L/C/M/A compliance"), theme::normal()),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled("│ ", theme::dim()),
            Span::styled(format!("{:<8}", "Status"), theme::dim()),
            Span::styled(format!("{:<48}", "ready to test"), theme::success()),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled(
                "└──────────────────────────────────────────────────────────┘",
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));

        // ── Divider ──────────────────────────────────────────────────────
        let div_w = (area.width as usize).saturating_sub(4);
        lines.push(Line::from(Span::styled(
            format!("  {}", theme::dash(div_w)),
            theme::dim(),
        )));
        lines.push(Line::from(""));

        // ── COMPLIANCE ───────────────────────────────────────────────────
        lines.push(section_header("COMPLIANCE", "What is being verified?"));
        lines.push(Line::from(""));
        lines.push(option_row(
            "1",
            "LQS compliance test",
            "all four levels against holdout",
        ));
        lines.push(option_row(
            "2",
            "Quick quality check",
            "30-second sanity check",
        ));
        lines.push(option_row(
            "3",
            "Targeted level",
            "test one specific LQS level",
        ));
        lines.push(Line::from(""));

        // ── BENCHMARKING ─────────────────────────────────────────────────
        lines.push(section_header("BENCHMARKING", "How does it perform?"));
        lines.push(Line::from(""));
        lines.push(option_row(
            "4",
            "Performance suite",
            "latency p50/p95/p99, throughput",
        ));
        lines.push(option_row(
            "5",
            "Rate-distortion sweep",
            "quality vs CR curve",
        ));
        lines.push(option_row(
            "6",
            "Head-to-head",
            "against gzip, zstd, baselines",
        ));
        lines.push(Line::from(""));

        // ── CLINICAL VALIDATION ──────────────────────────────────────────
        lines.push(section_header(
            "CLINICAL VALIDATION",
            "Is it safe for patients?",
        ));
        lines.push(Line::from(""));
        lines.push(option_row(
            "7",
            "Downstream tasks",
            "seizure, sleep, pathology",
        ));
        lines.push(option_row(
            "8",
            "Hallucination tests",
            "detect generative fabrication",
        ));
        lines.push(Line::from(""));

        // ── EXPLORATION ──────────────────────────────────────────────────
        lines.push(section_header("EXPLORATION", ""));
        lines.push(Line::from(""));
        lines.push(option_row(
            "9",
            "Metrics explorer",
            "drill into last run's metrics",
        ));
        lines.push(Line::from(""));

        // ── REGISTRY ─────────────────────────────────────────────────────
        lines.push(section_header("REGISTRY", ""));
        lines.push(Line::from(""));
        lines.push(option_row(
            "p",
            "Publish badge",
            "signed compliance certificate",
        ));
        lines.push(option_row("r", "Leaderboard", "current state of the field"));
        lines.push(Line::from(""));

        // ── Footer keys ──────────────────────────────────────────────────
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("[x]", theme::key_hint()),
            Span::styled(" Export report   ", theme::dim()),
            Span::styled("[b]", theme::key_hint()),
            Span::styled(" Back   ", theme::dim()),
            Span::styled("[q]", theme::key_hint()),
            Span::styled(" Main menu", theme::dim()),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {}", theme::dash(div_w)),
            theme::dim(),
        )));

        f.render_widget(Paragraph::new(lines), area);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            // Compliance
            KeyCode::Char('1') => PanelAction::Navigate(router::LAUNCH_EAGLE_FULL.to_string()),
            KeyCode::Char('2') => PanelAction::Navigate(router::LAUNCH_EAGLE_QUICK.to_string()),
            KeyCode::Char('3') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            // Benchmarking
            KeyCode::Char('4') => PanelAction::Navigate(router::LAUNCH_EAGLE_PERF.to_string()),
            KeyCode::Char('5') => PanelAction::Navigate(router::LAUNCH_EAGLE_RD.to_string()),
            KeyCode::Char('6') => PanelAction::Navigate(router::LAUNCH_EAGLE_H2H.to_string()),
            // Clinical
            KeyCode::Char('7') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('8') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            // Exploration
            KeyCode::Char('9') => PanelAction::StatusMessage(
                "Metrics explorer — see eagle/runs/<latest>/metrics.json for now.".into(),
            ),
            // Registry
            KeyCode::Char('p') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('r') => PanelAction::StatusMessage(
                "Leaderboard — http://eagle.openhuman.tech/leaderboard (placeholder).".into(),
            ),
            // Footer
            KeyCode::Char('x') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('h') | KeyCode::Char('?') => {
                PanelAction::Navigate(router::SCREEN_HELP.to_string())
            }
            _ => PanelAction::Ignored,
        }
    }
}
