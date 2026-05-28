//! First-time setup wizard — 4 linear steps the user steps through on first launch.
//!
//! Step 1: Display preferences (color + show_banner + splash_duration)
//! Step 2: Backend (auto/rust/python)
//! Step 3: Workers (numeric input, 0 = auto)
//! Step 4: Verification (paranoid/standard/fast)
//!
//! Esc/b: return to previous step, or cancel from step 1.
//! Enter: advance to next step, or commit + save on step 4.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::config::LamQuantConfig;
use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

const STEPS: &[&str] = &["Display", "Backend", "Workers", "Verification"];

pub struct WizardPanel {
    step: usize,
    cfg: LamQuantConfig,

    // Step 1 picker state
    display_choice: usize, // 0=full ui, 1=potato

    // Step 2 picker state
    backend_choice: usize, // 0=auto, 1=rust, 2=python

    // Step 3 numeric input state
    workers_buffer: String,

    // Step 4 picker state
    verify_choice: usize, // 0=paranoid, 1=standard, 2=fast

    /// Set when the user has completed the wizard; app polls this
    /// after navigation to detect commit and route to main.
    committed: bool,
    save_message: Option<String>,
}

impl Default for WizardPanel {
    fn default() -> Self {
        Self::new(LamQuantConfig::default())
    }
}

impl WizardPanel {
    pub fn new(cfg: LamQuantConfig) -> Self {
        // Tier 4 audit: warn via save_message when an unknown
        // backend.mode / codec.verification value falls through to
        // the default arm. Pre-fix silently coerced the toml value
        // to "auto" / "standard" without telling the user; the next
        // wizard save then overwrote the original value invisibly.
        let mut save_message: Option<String> = None;
        let known_backend = matches!(
            cfg.backend.mode.as_str(),
            "auto" | "rust" | "python" | "custom" | ""
        );
        if !known_backend {
            save_message = Some(format!(
                "Note: unknown backend.mode {:?} -- defaulting to `auto`. \
                 Confirm in step 2 before save.",
                cfg.backend.mode
            ));
        }
        let known_verify = matches!(
            cfg.codec.verification.as_str(),
            "paranoid" | "standard" | "fast" | ""
        );
        if !known_verify {
            let extra = format!(
                " unknown codec.verification {:?} -- defaulting to `standard`",
                cfg.codec.verification
            );
            save_message = Some(match save_message {
                Some(m) => format!("{};{}", m, extra),
                None => format!("Note:{}", extra),
            });
        }
        Self {
            step: 0,
            workers_buffer: cfg.compute.workers.to_string(),
            display_choice: if cfg.output.minimal_ui { 1 } else { 0 },
            backend_choice: match cfg.backend.mode.as_str() {
                "rust" => 1,
                "python" => 2,
                _ => 0,
            },
            verify_choice: match cfg.codec.verification.as_str() {
                "paranoid" => 0,
                "fast" => 2,
                _ => 1,
            },
            cfg,
            committed: false,
            save_message,
        }
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Test-only accessor for the worker count input buffer. Used by the
    /// `wizard_worker_buffer_caps_at_eight_chars` test to verify the
    /// Phase 0 #18 buffer cap stays in place.
    #[doc(hidden)]
    pub fn workers_buffer_for_test(&self) -> &str {
        &self.workers_buffer
    }

    /// Test-only setter to jump to a specific step without going through
    /// the picker UI. Lets tests target the workers step directly.
    #[doc(hidden)]
    pub fn set_step_for_test(&mut self, step: usize) {
        self.step = step;
    }
    pub fn take_committed(&mut self) -> bool {
        let r = self.committed;
        self.committed = false;
        r
    }

    fn apply_choices(&mut self) {
        // Display
        match self.display_choice {
            0 => {
                self.cfg.output.minimal_ui = false;
                self.cfg.output.show_banner = true;
                self.cfg.output.show_spinner = true;
                self.cfg.output.color = "auto".into();
                if self.cfg.output.splash_duration == 0.0 {
                    self.cfg.output.splash_duration = 0.5;
                }
            }
            _ => {
                self.cfg.output.minimal_ui = true;
                self.cfg.output.show_banner = false;
                self.cfg.output.show_spinner = false;
                self.cfg.output.color = "never".into();
                self.cfg.output.splash_duration = 0.0;
            }
        }
        self.cfg.backend.mode = ["auto", "rust", "python"][self.backend_choice].into();
        self.cfg.compute.workers = self
            .workers_buffer
            .trim()
            .parse::<i64>()
            .unwrap_or(0)
            .max(0);
        self.cfg.codec.verification = ["paranoid", "standard", "fast"][self.verify_choice].into();
        // The user just completed the TUI wizard, so they implicitly chose
        // TUI as their default UI. Clears the "ask" sentinel that the
        // main_hub hint reads — wizard completion is the first-run prompt.
        // Users wanting GUI later can edit the toml or use LAMQUANT_UI=gui.
        // Canonical lowercase + trim so a future read can short-circuit
        // the trim/lowercase dance even though smart_detect_mode does it
        // defensively at the read site.
        if self.cfg.ui_preference.trim().eq_ignore_ascii_case("ask") {
            self.cfg.ui_preference = "tui".into();
        } else {
            self.cfg.ui_preference = self.cfg.ui_preference.trim().to_ascii_lowercase();
        }
    }

    fn finalize(&mut self) {
        self.apply_choices();
        match self.cfg.save() {
            Ok(()) => {
                self.committed = true;
                self.save_message = Some("Saved lamquant.toml".into());
            }
            Err(e) => {
                self.save_message = Some(format!("Save failed: {} — values applied in-memory", e));
                self.committed = true; // continue to main even if save failed
            }
        }
    }
}

impl Panel for WizardPanel {
    fn id(&self) -> &str {
        "wizard"
    }
    fn title(&self) -> &str {
        "First-time Setup"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);

        // Header / progress
        let header = vec![
            Line::from(""),
            Line::from(Span::styled("  Welcome to LamQuant", theme::title())),
            Line::from(Span::styled(
                "  Quick 4-step setup. You can change everything later in Settings.",
                theme::dim(),
            )),
            progress_dots(self.step, STEPS),
        ];
        f.render_widget(Paragraph::new(header), chunks[0]);

        // Step body
        let body = match self.step {
            0 => self.render_step_display(),
            1 => self.render_step_backend(),
            2 => self.render_step_workers(),
            _ => self.render_step_verify(),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim())
            .title(Span::styled(
                format!(
                    " Step {} of {} — {} ",
                    self.step + 1,
                    STEPS.len(),
                    STEPS[self.step]
                ),
                theme::heading(),
            ));
        f.render_widget(
            Paragraph::new(body).wrap(Wrap { trim: false }).block(block),
            chunks[1],
        );

        // Footer
        let last = self.step == STEPS.len() - 1;
        let footer_keys = vec![
            Span::styled(" ↑↓", theme::key_hint()),
            Span::styled(" choose  ", theme::dim()),
            Span::styled("Enter", theme::key_hint()),
            Span::styled(
                if last { " save & start  " } else { " next  " },
                theme::dim(),
            ),
            Span::styled("Esc/b", theme::key_hint()),
            Span::styled(
                if self.step == 0 {
                    " skip wizard  "
                } else {
                    " prev  "
                },
                theme::dim(),
            ),
        ];
        let mut footer_lines = vec![Line::from(footer_keys)];
        if let Some(m) = &self.save_message {
            footer_lines.push(Line::from(Span::styled(
                format!("  {}", m),
                theme::success(),
            )));
        }
        f.render_widget(Paragraph::new(footer_lines), chunks[2]);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        if self.step == 2 {
            // Numeric text input on workers step.
            // Esc always returns to previous step regardless of buffer state.
            match event.code {
                KeyCode::Esc => {
                    self.step -= 1;
                    return PanelAction::Consumed;
                }
                KeyCode::Backspace => {
                    self.workers_buffer.pop();
                    return PanelAction::Consumed;
                }
                KeyCode::Char('b') if self.workers_buffer.is_empty() => {
                    self.step -= 1;
                    return PanelAction::Consumed;
                }
                KeyCode::Char('q') => return PanelAction::Quit,
                KeyCode::Char(c) if c.is_ascii_digit() => {
                    // Worker count realistically tops out around 192 on the
                    // largest dual-socket Threadripper / Xeon Phi nodes some
                    // users compress on, so cap at 8 digits and let the
                    // config validator clamp absurd values. The previous
                    // 4-digit cap silently truncated 99999 → "9999".
                    if self.workers_buffer.len() < 8 {
                        self.workers_buffer.push(c);
                    }
                    return PanelAction::Consumed;
                }
                KeyCode::Enter => {
                    self.step += 1;
                    return PanelAction::Consumed;
                }
                _ => return PanelAction::Ignored,
            }
        }

        let n = match self.step {
            0 => 2, // display: 2 options
            1 => 3, // backend: 3 options
            3 => 3, // verify: 3 options
            _ => 0,
        };
        let cursor = match self.step {
            0 => &mut self.display_choice,
            1 => &mut self.backend_choice,
            3 => &mut self.verify_choice,
            _ => return PanelAction::Ignored,
        };

        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                *cursor = if *cursor == 0 { n - 1 } else { *cursor - 1 };
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                *cursor = (*cursor + 1) % n;
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                if self.step == STEPS.len() - 1 {
                    self.finalize();
                    // Back triggers handle_back which checks take_committed()
                    // and routes to main screen.
                    PanelAction::Back
                } else {
                    self.step += 1;
                    PanelAction::Consumed
                }
            }
            KeyCode::Esc | KeyCode::Char('b') | KeyCode::Backspace => {
                if self.step == 0 {
                    // Cancel wizard — caller should route back to main.
                    PanelAction::Back
                } else {
                    self.step -= 1;
                    PanelAction::Consumed
                }
            }
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }
}

impl WizardPanel {
    fn render_step_display(&self) -> Vec<Line<'static>> {
        wizard_picker(
            "Choose how the UI looks",
            &[
                ("Full UI", "Color + ASCII logo + spinner + splash. Default."),
                (
                    "Potato mode",
                    "No color/banner/spinner. For slow SSH or dumb terminals.",
                ),
            ],
            self.display_choice,
        )
    }
    fn render_step_backend(&self) -> Vec<Line<'static>> {
        wizard_picker(
            "Choose the compression backend",
            &[
                ("auto", "Use Rust if found, else Python. Recommended."),
                ("rust", "Force Rust binary. Fastest (~200 MB/s)."),
                ("python", "Pure-Python via numba. Slower (~15 MB/s)."),
            ],
            self.backend_choice,
        )
    }
    fn render_step_workers(&self) -> Vec<Line<'static>> {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  How many parallel workers?",
                theme::heading(),
            )),
            Line::from(Span::styled(
                "  0 = auto (cores - 2, capped by available RAM).",
                theme::dim(),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Workers: ", theme::normal()),
                Span::styled(format!("[{}_]", self.workers_buffer), theme::highlight()),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Type a number (0-9999), Backspace to edit, Enter to confirm.",
                theme::dim(),
            )),
        ]
    }
    fn render_step_verify(&self) -> Vec<Line<'static>> {
        wizard_picker(
            "Choose verification level",
            &[
                (
                    "paranoid",
                    "Decode + sample-exact compare after every encode. FDA grade.",
                ),
                ("standard", "CRC-32 per window + SHA-256 per file. Default."),
                ("fast", "CRC-32 only. Quickest; not for archival."),
            ],
            self.verify_choice,
        )
    }
}

fn wizard_picker(prompt: &str, options: &[(&str, &str)], cursor: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = vec![
        Line::from(""),
        Line::from(Span::styled(format!("  {}", prompt), theme::heading())),
        Line::from(""),
    ];
    for (i, (label, desc)) in options.iter().enumerate() {
        let marker = if i == cursor { "▸" } else { " " };
        let label_style = if i == cursor {
            theme::selected()
        } else {
            theme::normal()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", marker), theme::key_hint()),
            Span::styled(format!("{:<14}", label.to_string()), label_style),
            Span::styled(format!(" — {}", desc), theme::dim()),
        ]));
    }
    lines
}

fn progress_dots(step: usize, steps: &[&str]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = vec![Span::raw("  ")];
    for (i, name) in steps.iter().enumerate() {
        let style = if i == step {
            theme::highlight()
        } else if i < step {
            theme::success()
        } else {
            theme::dim()
        };
        let symbol = if i < step {
            "●"
        } else if i == step {
            "◉"
        } else {
            "○"
        };
        spans.push(Span::styled(symbol.to_string(), style));
        spans.push(Span::styled(format!(" {}  ", name), theme::dim()));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::AppState;
    use crossterm::event::{KeyEvent, KeyModifiers};

    fn k(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn s() -> AppState {
        AppState::new()
    }

    #[test]
    fn full_walk_through_commits() {
        let mut w = WizardPanel::new(LamQuantConfig::default());
        // Step 0 → 1
        w.handle_event(k(KeyCode::Enter), &s());
        assert_eq!(w.step, 1);
        // Step 1 → 2
        w.handle_event(k(KeyCode::Enter), &s());
        assert_eq!(w.step, 2);
        // Step 2 → 3 (workers buffer pre-filled to "0")
        w.handle_event(k(KeyCode::Enter), &s());
        assert_eq!(w.step, 3);
        // Step 3 finalize — but save will likely fail in test env (no writable cwd).
        w.handle_event(k(KeyCode::Enter), &s());
        assert!(w.committed);
    }

    #[test]
    fn esc_at_step_0_returns_back() {
        let mut w = WizardPanel::new(LamQuantConfig::default());
        let r = w.handle_event(k(KeyCode::Esc), &s());
        assert!(matches!(r, PanelAction::Back));
    }

    #[test]
    fn esc_after_step_1_goes_back() {
        let mut w = WizardPanel::new(LamQuantConfig::default());
        w.handle_event(k(KeyCode::Enter), &s());
        assert_eq!(w.step, 1);
        w.handle_event(k(KeyCode::Esc), &s());
        assert_eq!(w.step, 0);
    }

    #[test]
    fn workers_input_accepts_digits_only() {
        let mut w = WizardPanel::new(LamQuantConfig::default());
        w.step = 2;
        w.workers_buffer.clear();
        w.handle_event(k(KeyCode::Char('1')), &s());
        w.handle_event(k(KeyCode::Char('2')), &s());
        w.handle_event(k(KeyCode::Char('a')), &s()); // ignored
        assert_eq!(w.workers_buffer, "12");
    }

    #[test]
    fn cycles_choice_with_arrows() {
        let mut w = WizardPanel::new(LamQuantConfig::default());
        w.handle_event(k(KeyCode::Down), &s());
        assert_eq!(w.display_choice, 1);
        w.handle_event(k(KeyCode::Down), &s());
        assert_eq!(w.display_choice, 0); // wraps
    }
}
