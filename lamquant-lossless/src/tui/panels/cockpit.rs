//! Training Cockpit — pipeline status, resources, all training operations.
//!
//! Replaces the linear MenuPanel at SCREEN_TRAIN with a full hub:
//! Pipeline Status (live process + tmux scan), Resources line (GPU/CPU/RAM),
//! and five sections of operations (Data, Pipeline, Planning, Diagnostics,
//! System).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::process::Command;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Clone, Default)]
struct Pipeline {
    /// Running training process descriptions (one per python -m … train …).
    procs: Vec<String>,
    /// Active tmux sessions whose name contains "train" or starts with "lamquant".
    tmux: Vec<String>,
    /// Resource summary lines (GPU + CPU+RAM+Disk).
    gpu: String,
    cpu: String,
    /// Last 3 entries from runs/ or training history.
    recent: Vec<String>,
}

pub struct CockpitPanel {
    id: String,
    pipeline: Pipeline,
    /// Timestamp of last `[r]` press awaiting confirmation. None means
    /// no pending reset; Some(t) means the user pressed [r] once at t
    /// and a second press within RESET_WINDOW_SECS will fire the
    /// destructive launcher. Avoids one-keystroke wipe of ~/.cache.
    pending_reset: Option<std::time::Instant>,
}

const RESET_WINDOW_SECS: u64 = 3;

impl CockpitPanel {
    pub fn new() -> Self {
        Self {
            id: "cockpit".to_string(),
            pipeline: probe_pipeline(),
            pending_reset: None,
        }
    }
    pub fn refresh(&mut self) {
        self.pipeline = probe_pipeline();
    }
}

impl Default for CockpitPanel {
    fn default() -> Self {
        Self::new()
    }
}

fn cmd_out(prog: &str, args: &[&str]) -> Option<String> {
    Command::new(prog)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn probe_pipeline() -> Pipeline {
    // Training python processes
    let procs: Vec<String> = cmd_out("pgrep", &["-af", "lamquant.py train"])
        .map(|s| {
            s.lines()
                .map(|l| l.split_whitespace().skip(1).collect::<Vec<_>>().join(" "))
                .collect()
        })
        .unwrap_or_default();

    // tmux sessions
    let tmux: Vec<String> = cmd_out("tmux", &["ls"])
        .map(|s| {
            s.lines()
                .filter(|l| l.contains("train") || l.starts_with("lamquant"))
                .map(|l| l.to_string())
                .collect()
        })
        .unwrap_or_default();

    // GPU summary via nvidia-smi
    let gpu = cmd_out(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
    )
    .and_then(|s| {
        s.lines().next().map(|l| {
            let parts: Vec<&str> = l.split(',').map(|p| p.trim()).collect();
            if parts.len() >= 5 {
                let name = parts[0];
                let name_short = if name.len() > 22 { &name[..22] } else { name };
                Some(format!(
                    "GPU {} · {}/{} MiB · {}°C · {} W",
                    name_short, parts[1], parts[2], parts[3], parts[4]
                ))
            } else {
                None
            }
        })
    })
    .flatten()
    .unwrap_or_else(|| "GPU not detected (nvidia-smi missing)".to_string());

    // CPU + RAM + Disk
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let mem_lines = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let read_kb = |key: &str| -> u64 {
        mem_lines
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|n| n.parse().ok())
            .unwrap_or(0)
    };
    let total_kb = read_kb("MemTotal:");
    let avail_kb = read_kb("MemAvailable:");
    let used_kb = total_kb.saturating_sub(avail_kb);
    let to_gib = |kb: u64| -> f64 { kb as f64 / 1024.0 / 1024.0 };
    let disk_free = cmd_out("df", &["-h", "--output=avail", "."])
        .and_then(|s| s.lines().nth(1).map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "?".into());
    let cpu = format!(
        "CPU {} threads · {:.1}/{:.1} GiB RAM · Disk {} free",
        threads,
        to_gib(used_kb),
        to_gib(total_kb),
        disk_free,
    );

    // Recent runs — first 3 directories under ./runs (newest first)
    let recent: Vec<String> = std::fs::read_dir("runs")
        .ok()
        .map(|rd| {
            let mut entries: Vec<_> = rd
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .collect();
            entries
                .sort_by_key(|e| std::cmp::Reverse(e.metadata().and_then(|m| m.modified()).ok()));
            entries
                .into_iter()
                .take(3)
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();

    Pipeline {
        procs,
        tmux,
        gpu,
        cpu,
        recent,
    }
}

fn section_header<'a>(title: &'static str) -> Line<'a> {
    Line::from(vec![
        Span::raw("    "),
        Span::styled(title, theme::highlight()),
    ])
}

fn opt<'a>(key: &'static str, label: &'static str, desc: &'static str) -> Line<'a> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("[{}]", key), theme::key_hint()),
        Span::raw("  "),
        Span::styled(format!("{:<24}", label), theme::normal()),
        Span::styled(desc.to_string(), theme::dim()),
    ])
}

impl Panel for CockpitPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Training Cockpit"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        let div_w = (area.width as usize).saturating_sub(4);
        let dash = theme::dash(div_w);

        // ── Header ───────────────────────────────────────────────────────
        lines.push(Line::from(Span::styled(
            format!("  {}", dash),
            theme::dim(),
        )));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Training Cockpit", theme::highlight()),
            Span::raw(" ".repeat(div_w.saturating_sub(35))),
            Span::styled(
                format!("lamquant {}", env!("CARGO_PKG_VERSION")),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(Span::styled(
            format!("  {}", dash),
            theme::dim(),
        )));
        lines.push(Line::from(""));

        // ── BLUT cockpit handoff banner ─────────────────────────────────
        // BLUT's `blut tui` is now THE canonical training cockpit and a
        // superset of this panel + the retired Python cockpit. The hub's
        // "Train a model" tile execs `blut tui` directly (see
        // app.rs::handle_navigate, SCREEN_TRAIN → exec_handoff). This
        // in-process panel is now only a FALLBACK shown when the `blut`
        // binary is missing from PATH; it stays as an ad-hoc probe (live
        // pgrep, tmux scan) + per-recipe launcher so training is never
        // fully blocked.
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("→ ", theme::highlight()),
            Span::styled(
                "FALLBACK cockpit — `blut tui` is the canonical training cockpit ",
                theme::key_hint(),
            ),
            Span::styled(
                "(install BLUT; this opens automatically when blut is missing)",
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));

        // ── Pipeline Status box (manual borders so it scales with width) ─
        let inner_w = div_w.saturating_sub(2);
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!(
                    "┌─ Pipeline status {}┐",
                    theme::dash(inner_w.saturating_sub(18))
                ),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", theme::dim()),
            Span::raw(" ".repeat(inner_w)),
            Span::styled("│", theme::dim()),
        ]));
        let proc_line = if self.pipeline.procs.is_empty() {
            (
                "   No training processes detected".to_string(),
                theme::dim(),
            )
        } else {
            (
                format!("   {} training process(es)", self.pipeline.procs.len()),
                theme::success(),
            )
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", theme::dim()),
            Span::styled(
                format!("{:<width$}", proc_line.0, width = inner_w),
                proc_line.1,
            ),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", theme::dim()),
            Span::raw(" ".repeat(inner_w)),
            Span::styled("│", theme::dim()),
        ]));
        let tmux_line = if self.pipeline.tmux.is_empty() {
            ("   No training tmux sessions".to_string(), theme::dim())
        } else {
            (
                format!("   {} tmux session(s)", self.pipeline.tmux.len()),
                theme::success(),
            )
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", theme::dim()),
            Span::styled(
                format!("{:<width$}", tmux_line.0, width = inner_w),
                tmux_line.1,
            ),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", theme::dim()),
            Span::raw(" ".repeat(inner_w)),
            Span::styled("│", theme::dim()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("└{}┘", theme::dash(inner_w)), theme::dim()),
        ]));
        lines.push(Line::from(""));

        // ── Resources ────────────────────────────────────────────────────
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Resources         ", theme::key_hint()),
            Span::styled(self.pipeline.gpu.clone(), theme::normal()),
        ]));
        lines.push(Line::from(vec![
            Span::raw("                    "),
            Span::styled(self.pipeline.cpu.clone(), theme::dim()),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {}", dash),
            theme::dim(),
        )));
        lines.push(Line::from(""));

        // ── Sections ─────────────────────────────────────────────────────
        lines.push(section_header("DATA PREPARATION"));
        lines.push(Line::from(""));
        lines.push(opt(
            "d",
            "Data pipeline",
            "LML → L3 windows → memmap (full setup)",
        ));
        lines.push(opt(
            "p",
            "Preprocess LML → L3",
            "EDF/LML → windowed .npz (2500 samples)",
        ));
        lines.push(opt(
            "f",
            "Build fullband memmap",
            "L3 windows → fullband_train.dat",
        ));
        lines.push(opt(
            "n",
            "Build manifest",
            "scan datasets, assign splits + categories",
        ));
        lines.push(Line::from(""));

        lines.push(section_header("PIPELINE OPERATIONS"));
        lines.push(Line::from(""));
        lines.push(opt(
            "1",
            "Start training run",
            "configure and launch in tmux",
        ));
        lines.push(opt("2", "Train single stage", "pick one stage to run"));
        lines.push(opt("3", "Attach to running job", "tmux session viewer"));
        lines.push(opt(
            "4",
            "Queue & schedule",
            "view, reorder, skip pipeline stages",
        ));
        lines.push(Line::from(""));

        lines.push(section_header("PLANNING"));
        lines.push(Line::from(""));
        lines.push(opt(
            "5",
            "Preset management",
            "fast / research / production",
        ));
        lines.push(opt(
            "6",
            "Hyperparameter editor",
            "per-stage config (TrainingConfig)",
        ));
        lines.push(opt("7", "Experiment runner", "systematic A/B sweeps"));
        lines.push(Line::from(""));

        lines.push(section_header("DIAGNOSTICS"));
        lines.push(Line::from(""));
        lines.push(opt("8", "Run history", "past runs with metrics"));
        lines.push(opt("9", "Leaderboard", "all models ranked by R"));
        lines.push(opt("a", "Compare runs", "side-by-side metric table"));
        lines.push(opt("c", "Checkpoints", "list, inspect, prune, export"));
        lines.push(opt("m", "Live metrics", "tail log or matplotlib plot"));
        lines.push(Line::from(""));

        lines.push(section_header("SYSTEM"));
        lines.push(Line::from(""));
        lines.push(opt(
            "r",
            "Reset training state",
            "clean caches, stale sessions",
        ));
        lines.push(Line::from(""));

        // ── Footer keys ──────────────────────────────────────────────────
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("[b]", theme::key_hint()),
            Span::styled(" Back    ", theme::dim()),
            Span::styled("[q]", theme::key_hint()),
            Span::styled(" Main menu    ", theme::dim()),
            Span::styled("[x]", theme::key_hint()),
            Span::styled(" Exit", theme::dim()),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {}", dash),
            theme::dim(),
        )));
        lines.push(Line::from(""));

        // ── Recent runs ──────────────────────────────────────────────────
        if !self.pipeline.recent.is_empty() {
            let rec = self.pipeline.recent.join("  ·  ");
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("Recent:  ", theme::key_hint()),
                Span::styled(rec, theme::dim()),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("Recent:  ", theme::key_hint()),
                Span::styled("(none — start a run with [1])", theme::dim()),
            ]));
        }

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        // Two-press [r] window only lives across consecutive [r] presses.
        // Any other key clears it so a stray sequence like r→j→r within
        // 3 seconds doesn't accidentally fire the destructive launcher.
        if !matches!(event.code, KeyCode::Char('r')) {
            self.pending_reset = None;
        }
        match event.code {
            // Data preparation — [d] dispatches BLUT
            // lamquant_data_prep recipe (LML → LMA pack). [p]/[f]
            // are deprecated (LMA-direct training, ADR 0017
            // eliminates precompute caches). [n] manifest build
            // is folded into lamquant_data_prep + each train
            // recipe internally, so it has no standalone button.
            KeyCode::Char('d') => PanelAction::Navigate(router::LAUNCH_COCKPIT_DATA_PREP.into()),
            KeyCode::Char('p') => PanelAction::StatusMessage(
                "Precompute removed in v7.7 — training reads LMA directly (ADR 0017). Use [d]."
                    .into(),
            ),
            KeyCode::Char('f') => PanelAction::StatusMessage(
                "Memmap removed in v7.7 — training reads LMA directly (ADR 0017). Use [d].".into(),
            ),
            KeyCode::Char('n') => PanelAction::StatusMessage(
                "Manifest is built inline by every BLUT train recipe — no standalone button."
                    .into(),
            ),

            // Pipeline operations — all route through BLUT
            // (status.jsonl tail + StageEvent translation, see
            // lamquant_ops::runner::spawn_blut).
            KeyCode::Char('1') => {
                PanelAction::Navigate(router::LAUNCH_COCKPIT_TRAIN_ENCODER.into())
            }
            KeyCode::Char('2') => PanelAction::Navigate(router::LAUNCH_COCKPIT_TRAIN_SNN.into()),
            KeyCode::Char('3') => PanelAction::Navigate(router::LAUNCH_COCKPIT_TRAIN_ORACLE.into()),
            KeyCode::Char('4') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),

            // Planning
            KeyCode::Char('5') => PanelAction::StatusMessage(
                "Preset management: fast/research/production — TrainingConfig presets.".into(),
            ),
            KeyCode::Char('6') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('7') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),

            // Diagnostics
            KeyCode::Char('8') => PanelAction::StatusMessage(
                "Run history — see runs/ directory or experiments/ log.".into(),
            ),
            KeyCode::Char('9') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('a') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('c') => PanelAction::Navigate(router::LAUNCH_COCKPIT_CHECKPOINTS.into()),
            KeyCode::Char('m') => PanelAction::Navigate(router::LAUNCH_COCKPIT_METRICS.into()),
            // T3.2 — BLUT jobs view (Python cockpit parity for the
            // queue/history screens; BLUT now owns the table).
            KeyCode::Char('j') => PanelAction::Navigate(router::LAUNCH_COCKPIT_JOBS.into()),
            // T3.2 — Export weights (Python cockpit `_screen_export`
            // parity; runs scripts/export_weights.py — same as the
            // `fw_export` firmware-tier launcher).
            KeyCode::Char('e') => PanelAction::Navigate(router::LAUNCH_COCKPIT_EXPORT.into()),

            // System — destructive: rm -rf ~/.cache + tmux kill. Two-press
            // confirm within RESET_WINDOW_SECS so a stray [r] press doesn't
            // wipe state. First press records the timestamp + status hint;
            // second press within the window dispatches the launcher.
            KeyCode::Char('r') => {
                let now = std::time::Instant::now();
                let armed = self
                    .pending_reset
                    .map(|t| now.duration_since(t).as_secs() < RESET_WINDOW_SECS)
                    .unwrap_or(false);
                if armed {
                    self.pending_reset = None;
                    PanelAction::Navigate(router::LAUNCH_COCKPIT_RESET.into())
                } else {
                    self.pending_reset = Some(now);
                    PanelAction::StatusMessage(format!(
                        "Reset cache + tmux: press [r] again within {}s to confirm.",
                        RESET_WINDOW_SECS,
                    ))
                }
            }
            // [e] now wired to cockpit_export (above, T3.2).

            // Refresh detection (lowercase R is reset; capital R = refresh)
            KeyCode::Char('R') => {
                self.refresh();
                PanelAction::StatusMessage("Re-probed pipeline state.".into())
            }

            // Nav
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            KeyCode::Char('h') | KeyCode::Char('?') => {
                PanelAction::Navigate(router::SCREEN_HELP.to_string())
            }
            _ => PanelAction::Ignored,
        }
    }
}
