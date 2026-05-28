//! Syscheck — quick system probe (Python, Rust binary, libc, optional GPU).
//!
//! Each row is a `Check` with a name + status (Ok / Warn / Err) + detail.
//! Refresh runs all checks synchronously (each is a tiny subprocess or
//! filesystem probe).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::process::Command;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Warn,
    Err,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

pub struct SyscheckPanel {
    checks: Vec<Check>,
}

impl Default for SyscheckPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl SyscheckPanel {
    pub fn new() -> Self {
        let mut p = Self { checks: Vec::new() };
        p.refresh();
        p
    }

    pub fn refresh(&mut self) {
        self.checks.clear();
        self.checks.push(probe_python());
        self.checks.push(probe_pip());
        self.checks.push(probe_rust_binary());
        self.checks.push(probe_cargo());
        self.checks.push(probe_lamquant_pkg());
        self.checks.push(probe_gpu());
        self.checks.push(probe_edf_test_files());
        self.checks.push(probe_target_dir_size());
    }
}

impl Panel for SyscheckPanel {
    fn id(&self) -> &str {
        "syscheck"
    }
    fn title(&self) -> &str {
        "System Check"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(" System Check ", theme::heading()));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(inner);

        let mut rows: Vec<Line> = vec![Line::from("")];
        let n_ok = self
            .checks
            .iter()
            .filter(|c| c.status == Status::Ok)
            .count();
        let n_warn = self
            .checks
            .iter()
            .filter(|c| c.status == Status::Warn)
            .count();
        let n_err = self
            .checks
            .iter()
            .filter(|c| c.status == Status::Err)
            .count();
        rows.push(Line::from(vec![
            Span::styled(format!("  {} ok", n_ok), theme::success()),
            Span::styled(format!("    {} warn", n_warn), theme::warning()),
            Span::styled(format!("    {} fail", n_err), theme::error()),
        ]));
        rows.push(Line::from(""));

        for c in &self.checks {
            let (sym, sym_style) = match c.status {
                Status::Ok => (
                    if theme::ascii_only() { "[OK]" } else { "✓" },
                    theme::success(),
                ),
                Status::Warn => (
                    if theme::ascii_only() { "[!!]" } else { "▲" },
                    theme::warning(),
                ),
                Status::Err => (
                    if theme::ascii_only() { "[FAIL]" } else { "✗" },
                    theme::error(),
                ),
            };
            rows.push(Line::from(vec![
                Span::styled(format!("  {}  ", sym), sym_style),
                Span::styled(format!("{:<28}", c.name), theme::heading()),
                Span::styled(c.detail.clone(), theme::dim()),
            ]));
        }
        f.render_widget(Paragraph::new(rows).wrap(Wrap { trim: false }), chunks[0]);

        let hint = " [r] refresh  [Esc/b] back  [q] quit ";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[1],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::Enter => {
                self.refresh();
                PanelAction::Consumed
            }
            KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }
}

// ── Probes ─────────────────────────────────────────────────────────────────

fn probe_python() -> Check {
    match Command::new("python").arg("--version").output() {
        Ok(o) if o.status.success() => Check {
            name: "Python".into(),
            status: Status::Ok,
            detail: String::from_utf8_lossy(&o.stdout)
                .trim()
                .to_string()
                .replace("Python ", "v"),
        },
        Ok(_) => Check {
            name: "Python".into(),
            status: Status::Warn,
            detail: "exits non-zero".into(),
        },
        Err(_) => Check {
            name: "Python".into(),
            status: Status::Err,
            detail: "not found in $PATH".into(),
        },
    }
}

fn probe_pip() -> Check {
    match Command::new("pip").arg("--version").output() {
        Ok(o) if o.status.success() => {
            let line = String::from_utf8_lossy(&o.stdout);
            let v = line.split_whitespace().nth(1).unwrap_or("?").to_string();
            Check {
                name: "pip".into(),
                status: Status::Ok,
                detail: format!("v{}", v),
            }
        }
        _ => Check {
            name: "pip".into(),
            status: Status::Warn,
            detail: "not found".into(),
        },
    }
}

fn probe_rust_binary() -> Check {
    match Command::new("lml").arg("--version").output() {
        Ok(o) if o.status.success() => Check {
            name: "lml binary (lml)".into(),
            status: Status::Ok,
            detail: String::from_utf8_lossy(&o.stdout).trim().to_string(),
        },
        _ => {
            let local = std::path::Path::new("target/release/lml");
            if local.exists() {
                Check {
                    name: "lml binary (target/release)".into(),
                    status: Status::Ok,
                    detail: "found in target/release".into(),
                }
            } else {
                Check {
                    name: "lml binary".into(),
                    status: Status::Err,
                    detail: "not in $PATH or target/release".into(),
                }
            }
        }
    }
}

fn probe_cargo() -> Check {
    match Command::new("cargo").arg("--version").output() {
        Ok(o) if o.status.success() => Check {
            name: "cargo".into(),
            status: Status::Ok,
            detail: String::from_utf8_lossy(&o.stdout).trim().to_string(),
        },
        _ => Check {
            name: "cargo".into(),
            status: Status::Warn,
            detail: "not found (Rust dev only)".into(),
        },
    }
}

fn probe_lamquant_pkg() -> Check {
    match Command::new("python")
        .args([
            "-c",
            "import lamquant_codec; print(lamquant_codec.__version__)",
        ])
        .output()
    {
        Ok(o) if o.status.success() => Check {
            name: "lamquant_codec (Python pkg)".into(),
            status: Status::Ok,
            detail: format!("v{}", String::from_utf8_lossy(&o.stdout).trim()),
        },
        Ok(o) => Check {
            name: "lamquant_codec".into(),
            status: Status::Warn,
            detail: String::from_utf8_lossy(&o.stderr).trim().to_string(),
        },
        Err(_) => Check {
            name: "lamquant_codec".into(),
            status: Status::Err,
            detail: "Python missing — cannot test".into(),
        },
    }
}

/// GPU probe — tries NVIDIA → AMD ROCm → Apple MPS in turn so users on
/// non-NVIDIA hardware don't see a misleading "GPU not found" warning when
/// they have a perfectly functional accelerator. Returns the first hit; if
/// none respond, reports a single Warn explaining what was tried.
fn probe_gpu() -> Check {
    if let Some(c) = probe_gpu_nvidia() {
        return c;
    }
    if let Some(c) = probe_gpu_rocm() {
        return c;
    }
    if let Some(c) = probe_gpu_apple() {
        return c;
    }
    Check {
        name: "GPU".into(),
        status: Status::Warn,
        detail: "no NVIDIA, AMD ROCm, or Apple MPS device detected".into(),
    }
}

fn probe_gpu_nvidia() -> Option<Check> {
    let out = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        // nvidia-smi exists but reports no devices — treat as "no NVIDIA"
        // and let subsequent probes try other vendors.
        return None;
    }
    Some(Check {
        name: "GPU (NVIDIA)".into(),
        status: Status::Ok,
        detail: s.lines().next().unwrap_or("").to_string(),
    })
}

fn probe_gpu_rocm() -> Option<Check> {
    // rocm-smi exit code is 0 even when no AMD GPU is present, so we
    // additionally check the output line count.
    let out = Command::new("rocm-smi")
        .arg("--showproductname")
        .arg("--csv")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    // Skip header + blank lines; treat first non-empty data line as the device.
    let device = body
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with("device"))
        .map(|l| l.trim().to_string())
        .unwrap_or_default();
    if device.is_empty() {
        return None;
    }
    Some(Check {
        name: "GPU (AMD ROCm)".into(),
        status: Status::Ok,
        detail: device,
    })
}

fn probe_gpu_apple() -> Option<Check> {
    // Only meaningful on macOS; Linux/Windows builds short-circuit.
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .arg("-json")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    // Cheap text search — avoids pulling in serde_json just for this probe.
    // Apple Silicon devices (M1/M2/M3) all advertise "Metal Support" in the
    // displays output; a hit means MPS is available to the codec.
    if !body.contains("Metal") {
        return None;
    }
    // Try to extract a "chipset_model" line for the detail. Format is
    // "      \"chipset_model\" : \"Apple M2 Max\"," — match the value.
    let model = body
        .lines()
        .find(|l| l.contains("chipset_model"))
        .and_then(|l| l.split('"').nth(3))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Apple Silicon GPU (Metal)".to_string());
    Some(Check {
        name: "GPU (Apple Metal/MPS)".into(),
        status: Status::Ok,
        detail: model,
    })
}

fn probe_edf_test_files() -> Check {
    let candidates = [
        "reference_software/pyedflib-master/pyedflib/data/test_generator.edf",
        "reference_software/nedc_pyprint_edf/v1.0.0/example.edf",
    ];
    let found: Vec<&str> = candidates
        .iter()
        .copied()
        .filter(|p| std::path::Path::new(p).exists())
        .collect();
    if found.is_empty() {
        Check {
            name: "Sample EDFs".into(),
            status: Status::Warn,
            detail: "none found in reference_software/".into(),
        }
    } else {
        Check {
            name: "Sample EDFs".into(),
            status: Status::Ok,
            detail: format!("{} test file(s) available", found.len()),
        }
    }
}

fn probe_target_dir_size() -> Check {
    use std::fs;
    let target = std::path::Path::new("target");
    if !target.exists() {
        return Check {
            name: "target/ build cache".into(),
            status: Status::Warn,
            detail: "not built yet".into(),
        };
    }
    let mut size = 0u64;
    if let Ok(rd) = fs::read_dir(target) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                size += md.len();
            }
        }
    }
    let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    Check {
        name: "target/ build cache".into(),
        status: if gb > 5.0 { Status::Warn } else { Status::Ok },
        detail: format!("~{:.1} GB", gb),
    }
}
