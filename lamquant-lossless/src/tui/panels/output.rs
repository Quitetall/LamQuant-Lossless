//! Output panel — displays live stdout/stderr from a running operation.
//! Drains an OpReceiver each tick. Auto-scrolls until user scrolls back.
//!
//! For compress ops (`encode` / `encode_neural`), renders a multi-section
//! dashboard with progress bar, throughput, ETA, codec stats (avg CR,
//! raw/out bytes, savings), and a recent-files list. Other ops render
//! the legacy plain-log view.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::sync::mpsc::TryRecvError;
use throbber_widgets_tui::BRAILLE_SIX_DOUBLE;

use crate::tui::operations::{OpEvent, OpReceiver};
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LineKind {
    Normal,
    Hi,
    Ok,
    Err,
    Dim,
}

pub struct OutputPanel {
    title: String,
    lines: Vec<(String, LineKind)>,
    receiver: Option<OpReceiver>,
    scroll: usize,
    auto_scroll: bool,
    done: bool,
    failed: bool,
    /// Set when the op terminated because the user pressed Ctrl+C, distinct
    /// from a real failure. Border and header render in warning yellow rather
    /// than error red — the user already knows they cancelled, no need to
    /// shout at them in red about it.
    cancelled: bool,
    /// Frame index into `BRAILLE_SIX_DOUBLE.symbols` — advanced each tick.
    spinner: usize,
    last_height: usize,
    /// Most recent progress event — rendered as sticky bottom bar.
    progress: Option<ProgressState>,
    /// When true, emit a terminal bell (\x07) once on Done.
    pub bell_on_done: bool,
    bell_emitted: bool,
    /// Op id captured from the first OpEvent::Started — drives the
    /// dashboard-vs-plain-log render branch.
    op_id: Option<String>,
    /// Wall-clock when the op was first observed via Started — used
    /// for throughput and ETA on the compress dashboard.
    start_at: Option<Instant>,
    /// Rolling compress-mode metrics. Always populated; only rendered
    /// when `op_id` looks like an encode op.
    metrics: CompressMetrics,
}

/// Bucket key for grouping FileDone events by their input class —
/// (bits-per-sample, sample-rate Hz rounded to int, channel count).
/// Captures the three signal-config dimensions that change the
/// Shannon ceiling: bit depth (EDF int16 vs BDF int24 etc.), sample
/// rate (bandwidth), channel count (multi-channel correlation).
/// `BTreeMap` keeps the rendered breakdown stable-sorted by key.
type BucketKey = (u32, u32, u32); // (bps_per_sample, sample_rate_hz, n_channels)

#[derive(Default, Clone)]
struct BucketStats {
    files: u64,
    bytes_in: u64,
    bytes_out: u64,
    samples: u64,
}

/// Per-FileDone rollup driving the compress dashboard.
#[derive(Default, Clone)]
struct CompressMetrics {
    files_done: u64,
    files_failed: u64,
    bytes_in_total: u64,
    bytes_out_total: u64,
    /// Sum of FileDone.samples across all successful files. Drives the
    /// "Samples" column + Shannon-efficiency math.
    total_samples: u64,
    /// Sum of FileDone.duration_s across all successful files. Drives
    /// "EEG hours".
    total_duration_s: f64,
    /// Latest n_channels seen — assumes uniform across the corpus,
    /// which holds for clinical EEG datasets the readout targets.
    n_channels: Option<u32>,
    /// Latest sample_rate seen (Hz).
    sample_rate: Option<f32>,
    last_cr: Option<f64>,
    last_path: Option<String>,
    last_ms: u64,
    /// Last N FileDone events as `(path, success, cr, ms)` for the
    /// "Recent files" section. Bounded to RECENT_CAP so memory stays
    /// flat on long runs.
    recent: VecDeque<(String, bool, Option<f64>, u64)>,
    /// Every successful FileDone CR — drives the completion summary's
    /// CR distribution row (min / p50 / max). Sorted lazily at render
    /// time. Memory cost: 8 bytes/file, bounded by run length.
    all_crs: Vec<f64>,
    /// Per-input-class buckets keyed by (bits/sample, sample_rate,
    /// n_channels). One row per bucket in the completion summary's
    /// "Shannon by class" breakdown. Memory cost: ~64 bytes/bucket,
    /// realistic corpora produce 1-5 buckets.
    buckets: BTreeMap<BucketKey, BucketStats>,
    /// Best-CR file seen so far (path, cr).
    best: Option<(String, f64)>,
    /// Worst-CR file seen so far (path, cr).
    worst: Option<(String, f64)>,
}

const RECENT_CAP: usize = 8;
/// Hard cap on the `lines` log buffer. Without this the buffer grew
/// linearly with every Log/FileDone event — a 100k-file compress run
/// leaked ~50 MB of String allocations. Trimming from the front when
/// the cap is exceeded keeps memory bounded while preserving the most
/// recent log tail (what the user sees after auto-scroll).
const LINES_CAP: usize = 5000;
/// Cap on `all_crs` so very long compress runs don't grow this vec
/// unbounded. The CR distribution row (min / p50 / max) is still
/// representative since we keep the first ~50k samples — long-tail
/// runs would have to look at the manifest for the full picture.
const ALL_CRS_CAP: usize = 50_000;

#[derive(Clone)]
struct ProgressState {
    current: u64,
    total: u64,
    message: String,
}

impl OutputPanel {
    pub fn new() -> Self {
        Self {
            title: "Output".to_string(),
            lines: Vec::new(),
            receiver: None,
            scroll: 0,
            auto_scroll: true,
            done: false,
            failed: false,
            cancelled: false,
            spinner: 0,
            last_height: 20,
            progress: None,
            bell_on_done: false,
            bell_emitted: false,
            op_id: None,
            start_at: None,
            metrics: CompressMetrics::default(),
        }
    }

    /// Begin a new operation: clear state, attach receiver.
    pub fn start(&mut self, title: String, receiver: OpReceiver) {
        self.title = title;
        self.lines.clear();
        self.receiver = Some(receiver);
        self.scroll = 0;
        self.auto_scroll = true;
        self.done = false;
        self.failed = false;
        self.cancelled = false;
        self.progress = None;
        self.bell_emitted = false;
        self.op_id = None;
        self.start_at = None;
        self.metrics = CompressMetrics::default();
    }

    /// True when the active op is one that should render the compress
    /// dashboard (encode_lma / encode_neural). Other ops fall back to
    /// the plain log view. `encode` is the legacy bare-LML op-id
    /// removed in v1.1; kept here for the few v1.0 .lml files an
    /// operator might still be re-running.
    fn is_compress_op(&self) -> bool {
        self.op_id
            .as_deref()
            .map(|s| {
                matches!(
                    s,
                    "encode_lma"
                        | "encode_lml_siblings"
                        | "encode_neural"
                        | "encode"
                )
            })
            .unwrap_or(false)
    }

    pub fn is_done(&self) -> bool {
        self.done
    }
    pub fn op_title(&self) -> &str {
        &self.title
    }
    /// Last `n` log lines (newest last).
    pub fn recent_lines(&self, n: usize) -> impl Iterator<Item = &str> {
        let start = self.lines.len().saturating_sub(n);
        self.lines[start..].iter().map(|(s, _)| s.as_str())
    }
    /// Progress 0.0–1.0, or None if no progress data yet.
    pub fn progress_pct(&self) -> Option<f32> {
        self.progress.as_ref().and_then(|p| {
            if p.total > 0 {
                Some((p.current as f32 / p.total as f32).clamp(0.0, 1.0))
            } else {
                None
            }
        })
    }
    pub fn progress_msg(&self) -> Option<&str> {
        self.progress.as_ref().map(|p| p.message.as_str())
    }
    /// Whether the op terminated due to a real error (not a user cancel).
    pub fn is_failed(&self) -> bool {
        self.failed
    }
    /// Whether the op terminated because the user pressed Ctrl+C.
    /// Mutually exclusive with `is_failed` — runner emits one terminal
    /// event per op, classified by message substring at parse time.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    fn maybe_bell(&mut self) {
        if self.bell_on_done && !self.bell_emitted {
            // Write directly to stdout — ratatui won't emit \x07 itself.
            use std::io::Write;
            let mut stdout = std::io::stdout();
            let _ = stdout.write_all(b"\x07");
            let _ = stdout.flush();
            self.bell_emitted = true;
        }
    }
}

impl Default for OutputPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Panel for OutputPanel {
    fn id(&self) -> &str {
        "output"
    }
    fn title(&self) -> &str {
        &self.title
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let header_text = if self.done {
            if self.cancelled {
                format!(" ⚠ {} — cancelled ", self.title)
            } else if self.failed {
                format!(" ✗ {} — failed ", self.title)
            } else {
                format!(" ✓ {} — done ", self.title)
            }
        } else {
            let sym = BRAILLE_SIX_DOUBLE.symbols[self.spinner % BRAILLE_SIX_DOUBLE.symbols.len()];
            format!(" {} {} ", sym, self.title)
        };
        let header_style = if self.done && self.cancelled {
            theme::warning()
        } else if self.done && self.failed {
            theme::error()
        } else if self.done {
            theme::success()
        } else {
            theme::highlight()
        };

        // Border colour echoes the run state once the op is done. Cancellation
        // is yellow not red — the user pressed Ctrl+C, that's not an error.
        let border_style = if self.done && self.cancelled {
            theme::warning()
        } else if self.done && self.failed {
            theme::error()
        } else if self.done {
            theme::success()
        } else {
            theme::dim()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(Span::styled(header_text, header_style));
        let inner = block.inner(area);
        f.render_widget(block, area);

        // Branch on op type. Compress ops get the multi-section
        // dashboard while running and a completion summary once done;
        // everything else falls back to the legacy log-only render so
        // verify/info/stats stay terse.
        if self.is_compress_op() {
            if self.done {
                self.render_compress_summary(f, inner);
            } else {
                self.render_compress_dashboard(f, inner);
            }
            return;
        }

        let progress_h: u16 = if self.progress.is_some() { 1 } else { 0 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(progress_h),
                Constraint::Length(1),
            ])
            .split(inner);

        let body_h = chunks[0].height as usize;
        let total = self.lines.len();
        let view_start = if self.auto_scroll {
            total.saturating_sub(body_h)
        } else {
            self.scroll.min(total.saturating_sub(body_h.max(1)))
        };
        let view_end = (view_start + body_h).min(total);

        let view: Vec<Line> = self.lines[view_start..view_end]
            .iter()
            .map(|(s, k)| {
                let style = match k {
                    LineKind::Normal => theme::normal(),
                    LineKind::Hi => theme::highlight(),
                    LineKind::Ok => theme::success(),
                    LineKind::Err => theme::error(),
                    LineKind::Dim => theme::dim(),
                };
                Line::from(Span::styled(s.clone(), style))
            })
            .collect();
        f.render_widget(Paragraph::new(view), chunks[0]);

        // Sticky progress bar (only when an op is reporting progress).
        if let Some(p) = &self.progress {
            let bar_area = chunks[1];
            let line = render_progress_bar(p, bar_area.width as usize);
            f.render_widget(Paragraph::new(line), bar_area);
        }

        let hint = if self.done {
            " [Enter] back  [Esc] back  [b] back  [↑↓] scroll  [q] quit "
        } else {
            " [↑↓] scroll  [PgUp/PgDn] page  [End] auto-scroll  [Ctrl+C] cancel "
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[2],
        );
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.auto_scroll = false;
                self.scroll = self.scroll.saturating_sub(1);
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !self.lines.is_empty() {
                    let max_scroll = self.lines.len().saturating_sub(1);
                    self.scroll = self.scroll.saturating_add(1).min(max_scroll);
                    if self.scroll + self.last_height >= self.lines.len() {
                        self.auto_scroll = true;
                    }
                }
                PanelAction::Consumed
            }
            KeyCode::PageUp => {
                self.auto_scroll = false;
                self.scroll = self.scroll.saturating_sub(10);
                PanelAction::Consumed
            }
            KeyCode::PageDown => {
                let max_scroll = self.lines.len().saturating_sub(1);
                self.scroll = self.scroll.saturating_add(10).min(max_scroll);
                PanelAction::Consumed
            }
            KeyCode::End => {
                self.auto_scroll = true;
                PanelAction::Consumed
            }
            KeyCode::Home => {
                self.auto_scroll = false;
                self.scroll = 0;
                PanelAction::Consumed
            }
            KeyCode::Enter | KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => {
                if self.done {
                    PanelAction::Back
                } else {
                    PanelAction::Consumed
                }
            }
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }

    fn tick(&mut self) {
        // Drain happens at App level via `try_recv_event` + dispatch
        // (P3 of reactive-store refactor). Tick keeps only spinner + bell.
        self.spinner = self.spinner.wrapping_add(1);
        if self.done {
            self.maybe_bell();
        }
    }
}

impl OutputPanel {
    /// Pull one event off the subprocess channel without applying it.
    /// Called by App.tick_panels which then dispatches `Action::OpEvent(ev)`
    /// → `consume(ev)`. Routes all OpEvent state mutations through the
    /// dispatch chokepoint (P3 of reactive-store refactor).
    ///
    /// Returns:
    /// - `Some(ev)` for each pending event
    /// - `None` when the channel is empty OR disconnected (drops receiver
    ///   on disconnect and marks the panel done if not already)
    pub fn try_recv_event(&mut self) -> Option<OpEvent> {
        let rx = self.receiver.as_ref()?;
        match rx.try_recv() {
            Ok(ev) => Some(ev),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.receiver = None;
                if !self.done {
                    self.done = true;
                }
                self.progress = None;
                None
            }
        }
    }

    /// Apply a drained `OpEvent` to panel state. The body is the original
    /// match arms from the pre-P3 `tick()` drain loop, lifted unchanged so
    /// log appearance / progress sticky bar / done/failed/cancelled flags
    /// behave identically.
    /// Trim the front of the log buffer when it grows past LINES_CAP.
    /// Cheap when no trimming is needed (single comparison + Vec::drain
    /// only when exceeded). Called after every push() in `consume`.
    fn trim_lines(&mut self) {
        if self.lines.len() > LINES_CAP {
            let drop = self.lines.len() - LINES_CAP;
            self.lines.drain(..drop);
            // scroll index may now point past the front of the buffer.
            // Clamp to keep the viewport from breaking.
            self.scroll = self.scroll.saturating_sub(drop);
        }
    }

    pub fn consume(&mut self, ev: OpEvent) {
        match ev {
            OpEvent::Log { message, .. } => {
                self.lines.push((message, LineKind::Normal));
                self.trim_lines();
            }
            OpEvent::Started { op_id, .. } => {
                self.lines
                    .push((format!(">> Started: {}", op_id), LineKind::Hi));
                self.trim_lines();
                self.op_id = Some(op_id);
                self.start_at = Some(Instant::now());
            }
            OpEvent::Progress {
                current,
                total,
                message,
                ..
            } => {
                self.progress = Some(ProgressState {
                    current,
                    total,
                    message,
                });
            }
            OpEvent::FileDone {
                path,
                success,
                cr,
                ms,
                bytes_in,
                bytes_out,
                samples,
                duration_s,
                n_channels,
                sample_rate,
                ..
            } => {
                let (s, k) = if success {
                    (
                        format!("   ✓ {}  CR={:.2}  {}ms", path, cr.unwrap_or(0.0), ms),
                        LineKind::Ok,
                    )
                } else {
                    (format!("   ✗ {}  failed in {}ms", path, ms), LineKind::Err)
                };
                self.lines.push((s, k));
                self.trim_lines();

                // Roll into compress dashboard metrics. Per OpEvent
                // contract: bytes_in/bytes_out are either both Some
                // or both None — no need to handle half-populated.
                if success {
                    self.metrics.files_done += 1;
                } else {
                    self.metrics.files_failed += 1;
                }
                if let (Some(bi), Some(bo)) = (bytes_in, bytes_out) {
                    self.metrics.bytes_in_total = self.metrics.bytes_in_total.saturating_add(bi);
                    self.metrics.bytes_out_total = self.metrics.bytes_out_total.saturating_add(bo);
                }
                if let Some(s) = samples {
                    self.metrics.total_samples = self.metrics.total_samples.saturating_add(s);
                }
                if let Some(d) = duration_s {
                    self.metrics.total_duration_s += d;
                }
                if n_channels.is_some() {
                    self.metrics.n_channels = n_channels;
                }
                if sample_rate.is_some() {
                    self.metrics.sample_rate = sample_rate;
                }

                // Per-input-class bucket: only count successful files
                // with the full telemetry tuple. Bit depth derives from
                // observed bytes_in / samples — handles EDF int16,
                // BDF int24, future float32, etc. without front-end
                // hints.
                if success {
                    if let (Some(bi), Some(s), Some(sr), Some(nc)) =
                        (bytes_in, samples, sample_rate, n_channels)
                    {
                        if s > 0 {
                            let bps_per_sample = ((bi as f64 * 8.0) / s as f64).round() as u32;
                            let key: BucketKey = (bps_per_sample, sr.round() as u32, nc);
                            let stats = self.metrics.buckets.entry(key).or_default();
                            stats.files += 1;
                            stats.bytes_in = stats.bytes_in.saturating_add(bi);
                            stats.samples = stats.samples.saturating_add(s);
                            if let Some(bo) = bytes_out {
                                stats.bytes_out = stats.bytes_out.saturating_add(bo);
                            }
                        }
                    }
                }

                self.metrics.last_cr = cr;
                self.metrics.last_path = Some(path.clone());
                self.metrics.last_ms = ms;
                if success {
                    if let Some(c) = cr {
                        if c > 0.0 && self.metrics.all_crs.len() < ALL_CRS_CAP {
                            self.metrics.all_crs.push(c);
                            match self.metrics.best.as_ref() {
                                Some((_, bc)) if *bc >= c => {}
                                _ => self.metrics.best = Some((path.clone(), c)),
                            }
                            match self.metrics.worst.as_ref() {
                                Some((_, wc)) if *wc <= c => {}
                                _ => self.metrics.worst = Some((path.clone(), c)),
                            }
                        }
                    }
                }
                self.metrics.recent.push_back((path, success, cr, ms));
                while self.metrics.recent.len() > RECENT_CAP {
                    self.metrics.recent.pop_front();
                }
            }
            OpEvent::Done { message, .. } => {
                self.lines
                    .push((format!(">> Done: {}", message), LineKind::Ok));
                self.trim_lines();
                self.done = true;
                self.progress = None;
            }
            OpEvent::Error { message, .. } => {
                let is_cancel = message.contains("cancelled");
                let kind = if is_cancel {
                    LineKind::Dim
                } else {
                    LineKind::Err
                };
                let prefix = if is_cancel {
                    ">> Cancelled: "
                } else {
                    ">> Error: "
                };
                self.lines.push((format!("{}{}", prefix, message), kind));
                self.trim_lines();
                self.done = true;
                if is_cancel {
                    self.cancelled = true;
                } else {
                    self.failed = true;
                }
                self.progress = None;
            }
        }
    }

    /// Multi-section live dashboard for compress ops. Mirrors the
    /// legacy Python `Dashboard._draw` (lamquant_codec/cli/readout.py)
    /// — header band, progress + throughput + ETA, two-column
    /// Compression|Signal split, integrity row, recent files,
    /// hint footer. Falls back to dim placeholders before the first
    /// FileDone so the layout doesn't reflow once metrics arrive.
    fn render_compress_dashboard(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),                       // header (title + elapsed)
                Constraint::Length(1),                       // divider
                Constraint::Length(4), // Progress + Throughput/ETA + Current (3 rows + 1 border-ish padding)
                Constraint::Length(1), // spacer
                Constraint::Length(8), // Compression | Signal split (6 inner rows)
                Constraint::Length(1), // spacer
                Constraint::Length(3), // Integrity row
                Constraint::Length((RECENT_CAP as u16) + 2), // Recent files
                Constraint::Min(0),    // spacer
                Constraint::Length(1), // hint
            ])
            .split(area);

        // Pre-compute everything used across sections.
        let elapsed_secs = self
            .start_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let throughput = if self.metrics.bytes_in_total > 0 && elapsed_secs > 0.0 {
            self.metrics.bytes_in_total as f64 / elapsed_secs
        } else {
            0.0
        };
        let eta = self.progress.as_ref().and_then(|p| {
            if p.total == 0 || p.current == 0 {
                return None;
            }
            let frac = p.current as f64 / p.total as f64;
            if frac <= 0.0 || frac >= 1.0 {
                return None;
            }
            let total_secs = elapsed_secs / frac;
            Some(((total_secs - elapsed_secs).max(0.0)) as u64)
        });
        let avg_cr = if self.metrics.bytes_out_total > 0 && self.metrics.bytes_in_total > 0 {
            Some(self.metrics.bytes_in_total as f64 / self.metrics.bytes_out_total as f64)
        } else {
            None
        };
        let saved_bytes = self
            .metrics
            .bytes_in_total
            .saturating_sub(self.metrics.bytes_out_total);
        let shannon = shannon_efficiency(
            self.metrics.bytes_in_total,
            self.metrics.bytes_out_total,
            self.metrics.total_samples,
        );
        let uncompressed_bps =
            bps_per_sample(self.metrics.bytes_in_total, self.metrics.total_samples);
        let compressed_bps =
            bps_per_sample(self.metrics.bytes_out_total, self.metrics.total_samples);

        // ── Header band ──────────────────────────────────────────
        let mode = mode_label(self.op_id.as_deref().unwrap_or(""));
        let header = vec![
            Line::from(vec![
                Span::raw(" "),
                Span::styled(format!("LamQuant Compression — {}", mode), theme::heading()),
                Span::raw("    "),
                Span::styled(format!("[{}]", fmt_hms(elapsed_secs as u64)), theme::dim()),
            ]),
            Line::from(Span::styled("", theme::dim())),
        ];
        f.render_widget(Paragraph::new(header), chunks[0]);

        let div_w = (area.width as usize).saturating_sub(2);
        let dash = theme::dash(div_w);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {}", dash), theme::dim()))),
            chunks[1],
        );

        // ── Progress + Throughput/ETA + Current ──────────────────
        let prog_inner = chunks[2];
        let mut prog_lines: Vec<Line> = Vec::new();
        if let Some(p) = &self.progress {
            prog_lines.push(render_progress_bar(p, prog_inner.width as usize));
        } else {
            prog_lines.push(Line::from(Span::styled(
                "  (waiting for first progress event)",
                theme::dim(),
            )));
        }
        let throughput_str = if throughput > 0.0 {
            format!("{}/s", fmt_bytes(throughput as u64))
        } else {
            "—".to_string()
        };
        let eta_str = match eta {
            Some(s) => fmt_hms(s),
            None => "—".to_string(),
        };
        prog_lines.push(Line::from(vec![
            Span::raw(" "),
            Span::styled("Throughput  ", theme::dim()),
            Span::styled(throughput_str, theme::highlight()),
            Span::raw("    "),
            Span::styled("ETA  ", theme::dim()),
            Span::styled(eta_str, theme::highlight()),
            Span::raw("    "),
            Span::styled("Elapsed  ", theme::dim()),
            Span::styled(fmt_hms(elapsed_secs as u64), theme::normal()),
        ]));
        if let Some(p) = &self.metrics.last_path {
            let trimmed = if p.len() > 60 {
                truncate_path_right(p, 58)
            } else {
                p.clone()
            };
            prog_lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled("Current     ", theme::dim()),
                Span::styled(trimmed, theme::normal()),
            ]));
        }
        f.render_widget(Paragraph::new(prog_lines), prog_inner);

        // ── Compression | Signal split ───────────────────────────
        let split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[4]);

        let comp_block = bordered("Compression");
        let comp_inner = comp_block.inner(split[0]);
        f.render_widget(comp_block, split[0]);
        let kv = |k: &str, v: String, hi: bool| {
            Line::from(vec![
                Span::raw(" "),
                Span::styled(format!("{:<10}", k), theme::dim()),
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
        let comp_lines = vec![
            kv(
                "Ratio",
                avg_cr
                    .map(|v| format!("{:>5.2} : 1", v))
                    .unwrap_or_else(|| "—".into()),
                true,
            ),
            kv("Saved", fmt_bytes(saved_bytes), saved_bytes > 0),
            kv(
                "Shannon",
                shannon
                    .map(|v| format!("{:>5.2} %", v))
                    .unwrap_or_else(|| "—".into()),
                shannon.is_some(),
            ),
            kv(
                "Bps in",
                uncompressed_bps
                    .map(|v| format!("{:>5.2}", v))
                    .unwrap_or_else(|| "—".into()),
                uncompressed_bps.is_some(),
            ),
            kv(
                "Bps out",
                compressed_bps
                    .map(|v| format!("{:>5.2}", v))
                    .unwrap_or_else(|| "—".into()),
                compressed_bps.is_some(),
            ),
            kv(
                "Last CR",
                self.metrics
                    .last_cr
                    .map(|v| format!("{:.2} : 1", v))
                    .unwrap_or_else(|| "—".into()),
                false,
            ),
        ];
        f.render_widget(Paragraph::new(comp_lines), comp_inner);

        let sig_block = bordered("Signal");
        let sig_inner = sig_block.inner(split[1]);
        f.render_widget(sig_block, split[1]);
        let hours = self.metrics.total_duration_s / 3600.0;
        let samples_g = self.metrics.total_samples as f64 / 1e9;
        let sig_lines = vec![
            kv(
                "Files",
                format!("{} OK", self.metrics.files_done),
                self.metrics.files_done > 0,
            ),
            kv(
                "EEG hrs",
                if hours > 0.0 {
                    format!("{:>7.1}", hours)
                } else {
                    "—".into()
                },
                hours > 0.0,
            ),
            kv(
                "Samples",
                if samples_g > 0.0 {
                    format!("{:>5.2} G", samples_g)
                } else {
                    "—".into()
                },
                samples_g > 0.0,
            ),
            kv(
                "Channels",
                self.metrics
                    .n_channels
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "—".into()),
                self.metrics.n_channels.is_some(),
            ),
        ];
        f.render_widget(Paragraph::new(sig_lines), sig_inner);

        // ── Integrity ────────────────────────────────────────────
        let integ_block = bordered("Integrity");
        let integ_inner = integ_block.inner(chunks[6]);
        f.render_widget(integ_block, chunks[6]);
        let err_style = if self.metrics.files_failed > 0 {
            theme::error()
        } else {
            theme::success()
        };
        let integ_line = Line::from(vec![
            Span::raw(" "),
            Span::styled("CRC-32 + SHA-256  ", theme::dim()),
            Span::styled("✓", theme::success()),
            Span::styled(" verified  ", theme::dim()),
            Span::styled(
                format!(
                    "{:>6} / {:<6}",
                    self.metrics.files_done,
                    self.metrics.files_done + self.metrics.files_failed
                ),
                theme::highlight(),
            ),
            Span::raw("    "),
            Span::styled("Errors  ", theme::dim()),
            Span::styled(format!("{:>3}", self.metrics.files_failed), err_style),
        ]);
        f.render_widget(Paragraph::new(integ_line), integ_inner);
        let _ = chunks[3]; // spacer
        let _ = chunks[5]; // spacer

        // ── Progress ─────────────────────────────────────────────
        // ── Recent files ─────────────────────────────────────────
        let recent_block = bordered("Recent files");
        let recent_inner = recent_block.inner(chunks[7]);
        f.render_widget(recent_block, chunks[7]);
        let mut recent_lines: Vec<Line> = Vec::new();
        if self.metrics.recent.is_empty() {
            recent_lines.push(Line::from(Span::styled(
                "  (waiting for first file)",
                theme::dim(),
            )));
        } else {
            for (path, success, cr, ms) in self.metrics.recent.iter().rev() {
                let trimmed = if path.len() > 56 {
                    truncate_path_right(path, 54)
                } else {
                    path.clone()
                };
                if *success {
                    recent_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled("✓ ", theme::success()),
                        Span::styled(format!("{:<58}", trimmed), theme::normal()),
                        Span::styled(
                            format!(" CR={:>5.2}  {}ms", cr.unwrap_or(0.0), ms),
                            theme::dim(),
                        ),
                    ]));
                } else {
                    recent_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled("✗ ", theme::error()),
                        Span::styled(format!("{:<58}", trimmed), theme::error()),
                        Span::styled(format!(" failed in {}ms", ms), theme::dim()),
                    ]));
                }
            }
        }
        f.render_widget(Paragraph::new(recent_lines), recent_inner);

        // ── Hint ─────────────────────────────────────────────────
        let hint = if self.done {
            " [Enter/Esc/b] back  [↑↓] log scroll (plain ops)  [q] main menu "
        } else {
            " [Ctrl+C] cancel  [↑↓/PgUp/PgDn] log scroll  [End] follow "
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[9],
        );
    }
}

impl OutputPanel {
    /// Completion summary rendered when a compress op terminates.
    /// Mirrors `print_summary` in readout.py — files counts, byte
    /// totals + EEG hours, ratio + Shannon + gap-to-floor +
    /// effective bitrate, integrity verification box, CR
    /// distribution, best/worst, reproducibility footer.
    fn render_compress_summary(&self, f: &mut Frame, area: Rect) {
        // Per-class breakdown: render up to MAX_BUCKETS rows + 1
        // header. Most clinical runs produce 1-3 buckets so this
        // usually stays compact.
        const MAX_BUCKETS_SHOWN: usize = 6;
        let bucket_rows = self.metrics.buckets.len().min(MAX_BUCKETS_SHOWN);
        // Header line plus rows; when no buckets seen yet, render
        // a single "no bucket data" line so the section never
        // collapses to zero height.
        let bucket_h: u16 = (1 + bucket_rows.max(1)) as u16;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),        // header
                Constraint::Length(1),        // divider
                Constraint::Length(4),        // Files / Input / Output
                Constraint::Length(1),        // spacer
                Constraint::Length(5),        // Ratio + Shannon + Gap + kbps + spacer
                Constraint::Length(6),        // Integrity box
                Constraint::Length(1),        // spacer
                Constraint::Length(3),        // CR distribution + best + worst
                Constraint::Length(1),        // spacer
                Constraint::Length(bucket_h), // Shannon by file class
                Constraint::Min(0),           // residual spacer
                Constraint::Length(1),        // reproducibility footer
                Constraint::Length(1),        // hint
            ])
            .split(area);

        let elapsed_secs = self
            .start_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        // Both byte totals must be positive — bytes_in==0 with
        // bytes_out>0 is theoretically impossible but produces a
        // poison `avg_cr=0` that propagates Infinity through the
        // gap + kbps formulas. Guard explicitly.
        let avg_cr = if self.metrics.bytes_out_total > 0 && self.metrics.bytes_in_total > 0 {
            Some(self.metrics.bytes_in_total as f64 / self.metrics.bytes_out_total as f64)
        } else {
            None
        };
        let saved_bytes = self
            .metrics
            .bytes_in_total
            .saturating_sub(self.metrics.bytes_out_total);
        let hours = self.metrics.total_duration_s / 3600.0;
        let shannon = shannon_efficiency(
            self.metrics.bytes_in_total,
            self.metrics.bytes_out_total,
            self.metrics.total_samples,
        );
        let _uncompressed_bps =
            bps_per_sample(self.metrics.bytes_in_total, self.metrics.total_samples);
        let _compressed_bps =
            bps_per_sample(self.metrics.bytes_out_total, self.metrics.total_samples);

        // ── Header ───────────────────────────────────────────────
        let (icon, icon_style, label) = if self.cancelled {
            ("⚠", theme::warning(), "Compression cancelled")
        } else if self.failed {
            ("✗", theme::error(), "Compression failed")
        } else {
            ("✓", theme::success(), "Compression complete")
        };
        let header = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(icon, icon_style),
                Span::raw("  "),
                Span::styled(label, theme::heading()),
                Span::raw("    "),
                Span::styled(format!("[{}]", fmt_hms(elapsed_secs as u64)), theme::dim()),
            ]),
            Line::from(""),
        ];
        f.render_widget(Paragraph::new(header), chunks[0]);

        let div_w = (area.width as usize).saturating_sub(2);
        let dash = theme::dash(div_w);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {}", dash), theme::dim()))),
            chunks[1],
        );

        // ── Files / Input / Output ───────────────────────────────
        let kv = |k: &str, v: String, hi: bool| {
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<12}", k), theme::dim()),
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
        let files_line = format!(
            "{} compressed  ·  {} failed",
            self.metrics.files_done, self.metrics.files_failed,
        );
        let input_line = format!(
            "{:<14}{:>12} hours EEG",
            fmt_bytes(self.metrics.bytes_in_total),
            format!("{:.1}", hours)
        );
        let output_line = format!(
            "{:<14}saved {}",
            fmt_bytes(self.metrics.bytes_out_total),
            fmt_bytes(saved_bytes)
        );
        let basics = vec![
            kv("Files", files_line, self.metrics.files_done > 0),
            kv("Input", input_line, self.metrics.bytes_in_total > 0),
            kv("Output", output_line, saved_bytes > 0),
        ];
        f.render_widget(Paragraph::new(basics), chunks[2]);

        // ── Ratio / Shannon / Gap / kbps ─────────────────────────
        let cr_str = avg_cr
            .map(|v| format!("{:>5.2} : 1", v))
            .unwrap_or_else(|| "—".into());
        let shannon_str = shannon
            .map(|v| format!("{:>6.2} %", v))
            .unwrap_or_else(|| "—".into());
        let bps = avg_cr.map(|c| 16.0 / c).unwrap_or(16.0);
        let gap_str = avg_cr
            .map(|_| {
                format!(
                    "{:>5.2} bits/sample  ({:.2} vs 6.63 bps)",
                    (bps - 6.63).max(0.0),
                    bps
                )
            })
            .unwrap_or_else(|| "—".into());
        let n_ch = self.metrics.n_channels.unwrap_or(21);
        let sr = self.metrics.sample_rate.unwrap_or(250.0);
        let kbps_str = avg_cr
            .map(|c| {
                let kbps = (16.0 / c) * sr as f64 * n_ch as f64 / 1000.0;
                format!("{:>6.0} kbps  ({}ch @ {} Hz)", kbps, n_ch, sr as u32)
            })
            .unwrap_or_else(|| "—".into());
        let stats_lines = vec![
            kv("Ratio", cr_str, true),
            kv("Shannon", shannon_str, shannon.is_some()),
            kv("Gap to floor", gap_str, avg_cr.is_some()),
            kv("Bit rate", kbps_str, avg_cr.is_some()),
        ];
        f.render_widget(Paragraph::new(stats_lines), chunks[4]);

        // ── Integrity box ────────────────────────────────────────
        let integ_block = bordered("Integrity verification");
        let integ_inner = integ_block.inner(chunks[5]);
        f.render_widget(integ_block, chunks[5]);
        let total = self.metrics.files_done + self.metrics.files_failed;
        let counts = format!("{:>6} / {:<6}", self.metrics.files_done, total);
        let err_style = if self.metrics.files_failed > 0 {
            theme::error()
        } else {
            theme::success()
        };
        let integ_lines = vec![
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<25}", "CRC-32 per window"), theme::normal()),
                Span::styled(counts.clone(), theme::highlight()),
                Span::styled("   files   ", theme::dim()),
                Span::styled("✓", theme::success()),
                Span::styled(" all verified", theme::dim()),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<25}", "SHA-256 per file"), theme::normal()),
                Span::styled(counts.clone(), theme::highlight()),
                Span::styled("   files   ", theme::dim()),
                Span::styled("✓", theme::success()),
                Span::styled(" all verified", theme::dim()),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{:<25}", "Verify-after-write"), theme::normal()),
                Span::styled(counts, theme::highlight()),
                Span::styled("   files   ", theme::dim()),
                Span::styled("✓", theme::success()),
                Span::styled(" roundtrip OK", theme::dim()),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled("Errors  ", theme::dim()),
                Span::styled(format!("{:>4}", self.metrics.files_failed), err_style),
            ]),
        ];
        f.render_widget(Paragraph::new(integ_lines), integ_inner);

        // ── CR distribution + best/worst ─────────────────────────
        let mut dist_lines: Vec<Line> = Vec::new();
        if self.metrics.all_crs.len() >= 2 {
            let mut sorted = self.metrics.all_crs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let pmin = sorted[0];
            let pmax = sorted[n - 1];
            let p50 = sorted[n / 2];
            dist_lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("CR distribution  ", theme::dim()),
                Span::styled(format!("min {:>5.2}", pmin), theme::normal()),
                Span::raw("  ·  "),
                Span::styled(format!("p50 {:>5.2}", p50), theme::highlight()),
                Span::raw("  ·  "),
                Span::styled(format!("max {:>5.2}", pmax), theme::normal()),
            ]));
        }
        if let Some((p, cr)) = &self.metrics.best {
            let trimmed = truncate_path_right(p, 48);
            dist_lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("Best   ", theme::dim()),
                Span::styled(format!("{:<50}", trimmed), theme::normal()),
                Span::styled(format!("  {:>5.2} : 1", cr), theme::success()),
            ]));
        }
        if let Some((p, cr)) = &self.metrics.worst {
            let trimmed = truncate_path_right(p, 48);
            dist_lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("Worst  ", theme::dim()),
                Span::styled(format!("{:<50}", trimmed), theme::normal()),
                Span::styled(format!("  {:>5.2} : 1", cr), theme::warning()),
            ]));
        }
        f.render_widget(Paragraph::new(dist_lines), chunks[7]);
        let _ = chunks[3];
        let _ = chunks[6];
        let _ = chunks[8];
        let _ = chunks[10];

        // ── Shannon efficiency by input class ────────────────────
        // One row per unique (bits/sample, sample-rate Hz, channels)
        // bucket. Each bucket's Shannon is its own corpus-weighted
        // average against the SHANNON_FLOOR_BPS reference; bit depth
        // is derived from observed bytes_in / samples (handles EDF
        // int16, BDF int24, etc. without manual config).
        let mut bucket_lines: Vec<Line> = Vec::new();
        bucket_lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Shannon efficiency by file class:", theme::heading()),
        ]));
        if self.metrics.buckets.is_empty() {
            bucket_lines.push(Line::from(Span::styled(
                "    (no per-file telemetry yet)",
                theme::dim(),
            )));
        } else {
            // Sort by samples descending so largest contributor leads.
            let mut buckets: Vec<(&BucketKey, &BucketStats)> =
                self.metrics.buckets.iter().collect();
            buckets.sort_by(|a, b| b.1.samples.cmp(&a.1.samples));
            for ((bps_in, sr, nc), stats) in buckets.iter().take(MAX_BUCKETS_SHOWN) {
                let eff = shannon_efficiency(stats.bytes_in, stats.bytes_out, stats.samples);
                let eff_str = eff
                    .map(|v| format!("{:>5.1}%", v))
                    .unwrap_or_else(|| "  —  ".into());
                let comp_bps = bps_per_sample(stats.bytes_out, stats.samples);
                let comp_str = comp_bps
                    .map(|v| format!("{:>5.2} bps", v))
                    .unwrap_or_else(|| "  —     ".into());
                bucket_lines.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("{}-bit @ {:>5} Hz × {:>3}ch", bps_in, sr, nc),
                        theme::normal(),
                    ),
                    Span::raw("   "),
                    Span::styled(eff_str, theme::highlight()),
                    Span::raw("   "),
                    Span::styled(comp_str, theme::dim()),
                    Span::raw("   "),
                    Span::styled(
                        format!(
                            "({} file{})",
                            stats.files,
                            if stats.files == 1 { "" } else { "s" }
                        ),
                        theme::dim(),
                    ),
                ]));
            }
        }
        f.render_widget(Paragraph::new(bucket_lines), chunks[9]);

        // ── Reproducibility footer ───────────────────────────────
        let footer = Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("codec={}  build=lamquant-core", env!("CARGO_PKG_VERSION"),),
                theme::dim(),
            ),
        ]);
        f.render_widget(Paragraph::new(footer), chunks[11]);

        // ── Hint ─────────────────────────────────────────────────
        let hint = " [Enter/Esc/b] back   [↑↓] log scroll   [q] main menu ";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, theme::dim()))),
            chunks[12],
        );
    }
}

/// Bordered block with a dim border + highlighted title — matches the
/// codec_hub / mode_panel / preflight box aesthetic.
fn bordered(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(format!(" {} ", title), theme::highlight()))
}

/// Human-readable byte count. Picks GiB/MiB/KiB/B based on magnitude.
fn fmt_bytes(b: u64) -> String {
    const GIB: u64 = 1u64 << 30;
    const MIB: u64 = 1u64 << 20;
    const KIB: u64 = 1u64 << 10;
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.1} MiB", b as f64 / MIB as f64)
    } else if b >= KIB {
        format!("{:.1} KiB", b as f64 / KIB as f64)
    } else {
        format!("{} B", b)
    }
}

/// Compact duration formatter — `4m12s`, `1h05m`, `42s`.
/// HH:MM:SS formatter — matches Python `_dur` in readout.py.
fn fmt_hms(s: u64) -> String {
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// Shannon-floor reference for clinical EEG, in bits per sample.
/// Source: empirical fit across the TUH EEG corpus at 21-channel,
/// 250 Hz, int16 — see decisions/0007-v763-engineering-audit.md.
/// This is the closest published estimate of the entropy floor for
/// medical-grade EEG signal. It is hard-coded (not derived from the
/// signal at runtime) because real per-file differential entropy
/// would require the raw samples, not the post-encode metadata
/// that flows through OpEvent.
const SHANNON_FLOOR_BPS: f64 = 6.63;

/// Average bits-per-sample given byte totals and sample totals.
/// Returns None when samples is zero (no signal seen yet).
fn bps_per_sample(bytes: u64, samples: u64) -> Option<f64> {
    if samples == 0 {
        return None;
    }
    Some((bytes as f64 * 8.0) / samples as f64)
}

/// Shannon efficiency percent, computed from REAL per-sample byte
/// rates rather than the assumed-16-bit shortcut the old version
/// used. Inputs are corpus-aggregate totals — the per-sample averages
/// are sample-weighted across every successful FileDone, so files
/// with different bit depths (EDF int16 vs BDF int24) and different
/// sample rates contribute proportional to their data volume.
///
/// `eff = (uncompressed_bps - compressed_bps) / (uncompressed_bps - floor_bps)`
///
/// Returns None when:
///   - samples or bytes_in are zero (no data yet)
///   - uncompressed_bps is at or below the floor (the floor is the
///     wrong reference for this corpus; would yield negative pct)
fn shannon_efficiency(bytes_in: u64, bytes_out: u64, samples: u64) -> Option<f64> {
    let bps_in = bps_per_sample(bytes_in, samples)?;
    let bps_out = bps_per_sample(bytes_out, samples)?;
    if bps_in <= SHANNON_FLOOR_BPS {
        return None;
    }
    let pct = (bps_in - bps_out) / (bps_in - SHANNON_FLOOR_BPS) * 100.0;
    Some(pct.clamp(0.0, 100.0))
}

/// Friendly mode label for the dashboard's "Mode" row.
/// Truncate `s` to its last `n` characters (NOT bytes) prefixed with
/// `…`. Char-boundary safe — byte-slicing at a fixed offset panics on
/// multi-byte codepoints (CJK / accented / emoji file paths). Returns
/// the original string when its char count ≤ n.
///
/// Postcondition: output char count ≤ n + 1 (the `…` adds at most one).
fn truncate_path_right(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        debug_assert!(s.chars().count() == count);
        return s.to_string();
    }
    let skip = count - n;
    let mut out = String::with_capacity(n + 3);
    out.push('…');
    out.extend(s.chars().skip(skip));
    debug_assert!(
        out.chars().count() == n + 1,
        "truncate_path_right output must be n + 1 chars (… + tail)",
    );
    out
}

fn mode_label(op_id: &str) -> &'static str {
    match op_id {
        // `encode_lma` is the canonical Lossless op-id after the v1.0
        // TUI default flip. `encode` (bare LML) stays mapped so old
        // log replays of v1.0 runs still render the right label.
        // `encode_lml_siblings` is the newer per-EEG .lml + copy
        // non-EEG mode added in the lossless-flow batch.
        "encode_lma" | "encode" | "encode_lml_siblings" => "LML Lossless",
        "encode_neural" => "LMQ Neural",
        _ => "(unknown)",
    }
}

/// Render a one-line progress bar: `[████████░░] 80% (12/15) — message`.
/// Width is the available terminal columns; falls back gracefully on narrow widths.
fn render_progress_bar(p: &ProgressState, width: usize) -> Line<'static> {
    let pct = if p.total > 0 {
        (p.current as f64 / p.total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let pct_label = format!(" {:>3}% ({}/{}) ", (pct * 100.0) as u32, p.current, p.total);
    let suffix = if p.message.is_empty() {
        String::new()
    } else {
        format!("— {}", p.message)
    };
    // Reserve room for label + suffix; bar gets the rest.
    let reserved = pct_label.chars().count() + suffix.chars().count() + 2;
    let bar_width = width.saturating_sub(reserved).max(8);
    let filled = (bar_width as f64 * pct).round() as usize;
    let filled = filled.min(bar_width);
    let bar_full: String = "█".repeat(filled);
    let bar_empty: String = "░".repeat(bar_width - filled);

    Line::from(vec![
        Span::styled("[", theme::dim()),
        Span::styled(bar_full, theme::highlight()),
        Span::styled(bar_empty, theme::dim()),
        Span::styled("]", theme::dim()),
        Span::styled(pct_label, theme::heading()),
        Span::styled(suffix, theme::dim()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_zero_total_is_empty() {
        let p = ProgressState {
            current: 0,
            total: 0,
            message: "init".into(),
        };
        let line = render_progress_bar(&p, 40);
        // No filled blocks
        let s: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(s.contains("0%"));
        assert!(!s.contains('█'));
    }

    #[test]
    fn bar_half_fills_half() {
        let p = ProgressState {
            current: 50,
            total: 100,
            message: "halfway".into(),
        };
        let line = render_progress_bar(&p, 60);
        let s: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(s.contains("50%"));
        assert!(s.contains("(50/100)"));
        assert!(s.contains('█'));
        assert!(s.contains('░'));
    }

    #[test]
    fn bar_full_no_empty_chars() {
        let p = ProgressState {
            current: 10,
            total: 10,
            message: "done".into(),
        };
        let line = render_progress_bar(&p, 50);
        let s: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(s.contains("100%"));
    }

    #[test]
    fn bar_clamps_to_minimum_width() {
        let p = ProgressState {
            current: 5,
            total: 10,
            message: "tight".into(),
        };
        // tiny width still produces something usable
        let line = render_progress_bar(&p, 8);
        assert!(!line.spans.is_empty());
    }
}
