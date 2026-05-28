//! Pre-flight panel — shown between the file picker and the running
//! op for compress operations. Displays scan results (file count,
//! total bytes, disk free, backend, workers) and gates execution on
//! `[Enter] Start` / `[b] Cancel`.
//!
//! Direct port of the old Python compress pre-flight banner
//! (lamquant.py L420-440). Skips the scan for non-compress ops to
//! match Python's behavior.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::config::LamQuantConfig;
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

/// Bound on the recursive scan. Keeps the UI responsive on huge dirs.
/// When hit, the panel surfaces a "(capped)" suffix so the count isn't
/// silently truncated.
const SCAN_CAP: usize = 200_000;

/// User intent flagged by the preflight panel before it exits. The
/// host reads this via `take_edit()` and routes the user back to the
/// appropriate picker (then re-enters preflight once the new path
/// lands).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditTarget {
    Input,
    Output,
}

pub struct PreflightPanel {
    op_id: String,
    input: String,
    output: Option<String>,
    file_count: u64,
    total_bytes: u64,
    /// True when `scan_inputs` short-circuited at `SCAN_CAP` and the
    /// real file count may be larger than `file_count`.
    scan_capped: bool,
    disk_free: Option<u64>,
    backend: String,
    workers: i64,
    confirmed: bool,
    edit: Option<EditTarget>,
}

impl PreflightPanel {
    pub fn new() -> Self {
        Self {
            op_id: String::new(),
            input: String::new(),
            output: None,
            file_count: 0,
            total_bytes: 0,
            scan_capped: false,
            disk_free: None,
            backend: String::new(),
            workers: 0,
            confirmed: false,
            edit: None,
        }
    }

    /// Populate the panel for a fresh run. Walks the input path for
    /// EDF/BDF files, sums their sizes, and queries the output dir
    /// for free space.
    pub fn prepare(
        &mut self,
        op_id: &str,
        input: &str,
        output: Option<&str>,
        cfg: &LamQuantConfig,
    ) {
        self.op_id = op_id.to_string();
        self.input = input.to_string();
        self.output = output.map(|s| s.to_string());
        self.confirmed = false;
        self.edit = None;
        self.backend = cfg.backend.mode.clone();
        self.workers = cfg.compute.workers;

        let (count, bytes, capped) = scan_inputs(Path::new(input));
        self.file_count = count;
        self.total_bytes = bytes;
        self.scan_capped = capped;

        let disk_target = output
            .map(Path::new)
            .and_then(|p| {
                p.parent()
                    .map(|p| p.to_path_buf())
                    .or_else(|| Some(p.to_path_buf()))
            })
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()));
        self.disk_free = disk_free_bytes(&disk_target);
    }

    /// Read + clear the confirm flag. Returns true exactly once if the
    /// user pressed Enter on this panel.
    pub fn take_confirmed(&mut self) -> bool {
        let v = self.confirmed;
        self.confirmed = false;
        v
    }

    /// Read + clear the "change input/output" flag. Returns Some
    /// exactly once if the user pressed `i` or `o` to edit a path
    /// from the preflight panel.
    pub fn take_edit(&mut self) -> Option<EditTarget> {
        self.edit.take()
    }
}

impl Default for PreflightPanel {
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

fn fmt_bytes(b: u64) -> String {
    let gib = b as f64 / (1u64 << 30) as f64;
    if gib >= 1.0 {
        format!("{:.2} GiB", gib)
    } else {
        let mib = b as f64 / (1u64 << 20) as f64;
        format!("{:.1} MiB", mib)
    }
}

impl Panel for PreflightPanel {
    fn id(&self) -> &str {
        "preflight"
    }
    fn title(&self) -> &str {
        "Pre-flight"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2), // header
                Constraint::Length(1), // divider
                Constraint::Length(7), // Input box
                Constraint::Length(1),
                Constraint::Length(7), // Scan + Resources box
                Constraint::Length(1),
                Constraint::Length(2), // confirm prompt
                Constraint::Min(0),
            ])
            .split(area);

        // Header
        let header = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled("Pre-flight", theme::title()),
                Span::raw("  "),
                Span::styled(format!("· {}", self.op_id), theme::dim()),
            ]),
            Line::from(Span::styled(
                "  Review before starting. [Enter] to launch · [b] to cancel.",
                theme::dim(),
            )),
        ];
        f.render_widget(Paragraph::new(header), chunks[0]);

        let div_w = (area.width as usize).saturating_sub(4);
        let dash = theme::dash(div_w);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("  {}", dash),
                theme::dim(),
            ))),
            chunks[1],
        );

        // Input + Output paths
        let in_block = bordered("Input / Output");
        let in_inner = in_block.inner(chunks[2]);
        f.render_widget(in_block, chunks[2]);
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
        let in_lines = vec![
            kv("Input", self.input.clone(), true),
            kv(
                "Output",
                self.output
                    .clone()
                    .unwrap_or_else(|| "(auto-derive)".into()),
                false,
            ),
            kv("Op", self.op_id.clone(), false),
            Line::from(""),
            Line::from(Span::styled(
                "   Tip: empty output asks the backend to place files next to the source.",
                theme::dim(),
            )),
        ];
        f.render_widget(Paragraph::new(in_lines), in_inner);

        // Scan + Resources
        let res_block = bordered("Scan & Resources");
        let res_inner = res_block.inner(chunks[4]);
        f.render_widget(res_block, chunks[4]);
        let scan_value = if self.file_count == 0 {
            "(0 files — input may not contain EDF/BDF)".to_string()
        } else {
            let suffix = if self.scan_capped {
                " (capped — more files exist)"
            } else {
                ""
            };
            format!(
                "{} file{}, {}{}",
                self.file_count,
                if self.file_count == 1 { "" } else { "s" },
                fmt_bytes(self.total_bytes),
                suffix
            )
        };
        let disk_value = match self.disk_free {
            Some(b) => fmt_bytes(b),
            None => "?".to_string(),
        };
        let res_lines = vec![
            kv("Scan", scan_value, self.file_count > 0),
            kv("Disk free", disk_value, true),
            kv("Backend", self.backend.clone(), true),
            kv("Workers", self.workers.to_string(), false),
            Line::from(""),
            Line::from(Span::styled(
                "   Backend + workers can be changed in Settings.",
                theme::dim(),
            )),
        ];
        f.render_widget(Paragraph::new(res_lines), res_inner);

        // Confirm
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("  {}", dash),
                theme::dim(),
            ))),
            chunks[5],
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw("  "),
                Span::styled("[Enter]", theme::key_hint()),
                Span::styled(" Start   ", theme::normal()),
                Span::styled("[i]", theme::key_hint()),
                Span::styled(" Change input   ", theme::dim()),
                Span::styled("[o]", theme::key_hint()),
                Span::styled(" Change output   ", theme::dim()),
                Span::styled("[b]", theme::key_hint()),
                Span::styled(" Cancel   ", theme::dim()),
                Span::styled("[q]", theme::key_hint()),
                Span::styled(" Main menu   ", theme::dim()),
                Span::styled("[x]", theme::key_hint()),
                Span::styled(" Exit", theme::dim()),
            ])),
            chunks[6],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('i') | KeyCode::Char('I') => {
                self.edit = Some(EditTarget::Input);
                PanelAction::Back
            }
            KeyCode::Char('o') | KeyCode::Char('O') => {
                self.edit = Some(EditTarget::Output);
                PanelAction::Back
            }
            KeyCode::Enter => {
                self.confirmed = true;
                PanelAction::Back
            }
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            _ => PanelAction::Ignored,
        }
    }
}

/// Walk `path` recursively for EDF/BDF files (matches the Python
/// `find_edf_files` rglob behavior). Returns
/// `(file_count, total_bytes, capped)` where `capped` is true if the
/// walk hit the SCAN_CAP guard before exhausting the tree — surfaces
/// in the UI as a "(capped)" note so users know the count may be low.
fn scan_inputs(path: &Path) -> (u64, u64, bool) {
    if path.is_file() {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        return (1, size, false);
    }
    if !path.is_dir() {
        return (0, 0, false);
    }
    let mut count = 0u64;
    let mut bytes = 0u64;
    let mut visited = 0usize;
    let mut capped = false;
    for entry in walkdir::WalkDir::new(path)
        .max_depth(8)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        visited += 1;
        if visited > SCAN_CAP {
            capped = true;
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let ext_ok = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                let e = e.to_ascii_lowercase();
                e == "edf" || e == "bdf"
            })
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        count += 1;
        if let Ok(m) = entry.metadata() {
            bytes += m.len();
        }
    }
    (count, bytes, capped)
}

/// Best-effort disk-free query via `df`. Returns None if the call
/// fails or output can't be parsed. Linux-only — matches the project
/// platform. The value is presentational, so failure degrades
/// gracefully to a "?" string in the UI.
fn disk_free_bytes(path: &Path) -> Option<u64> {
    let out = std::process::Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = std::str::from_utf8(&out.stdout).ok()?;
    stdout
        .lines()
        .nth(1)
        .and_then(|l| l.trim().parse::<u64>().ok())
}
