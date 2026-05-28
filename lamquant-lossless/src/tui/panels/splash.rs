//! Splash panel — boot screen shown for ~700ms on launch with the
//! LAMQUANT block-letter logo, version, and tagline. Any key skips
//! the wait. Auto-emits Back when the timer elapses, which pops the
//! splash off the router stack so whatever screen the App pushed
//! beneath it (Main / Resume / Wizard / RootWarn) becomes active.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::state::AppState;
use crate::tui::theme;

/// Tick budget before auto-advance. App ticks every ~50ms so 14 ticks
/// gives ~700ms — long enough to register the brand, short enough that
/// power users don't get annoyed.
const AUTO_ADVANCE_TICKS: u32 = 14;

pub(crate) const LOGO: &[&str] = &[
    " ██╗      █████╗ ███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗",
    " ██║     ██╔══██╗████╗ ████║██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝",
    " ██║     ███████║██╔████╔██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ",
    " ██║     ██╔══██║██║╚██╔╝██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ",
    " ███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ",
    " ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝",
];

pub struct SplashPanel {
    ticks: u32,
    /// Set on the first tick after a key press OR when the auto-advance
    /// budget elapses. `App` checks via `take_done()` and pops the
    /// router on the next event-loop iteration.
    done: bool,
}

impl SplashPanel {
    pub fn new() -> Self {
        Self {
            ticks: 0,
            done: false,
        }
    }

    /// Read + clear the done flag. Returns true exactly once when the
    /// splash has finished (auto-advance or skip).
    pub fn take_done(&mut self) -> bool {
        let v = self.done;
        self.done = false;
        v
    }
}

impl Default for SplashPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateless splash render used by `tui::mod::run_app` to paint the
/// boot logo BEFORE `App::new()` constructs config / history / peers
/// state. Closes the visible gap between alt-screen entry and the
/// first dashboard frame — without this, the user sees a blank
/// terminal for the duration of App::new()'s I/O.
///
/// No AppState dependency: only the logo + version string + a
/// "loading…" hint. The full SplashPanel takes over once App is
/// constructed and renders the same logo with backend/workers
/// info populated.
pub fn render_boot(f: &mut Frame, area: Rect) {
    let content_h: u16 = (LOGO.len() as u16) + 4;
    let pad_top = area.height.saturating_sub(content_h) / 2;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(pad_top),
            Constraint::Length(LOGO.len() as u16),
            Constraint::Length(1),
            Constraint::Length(1), // tagline
            Constraint::Length(1),
            Constraint::Length(1), // version
            Constraint::Min(0),
            Constraint::Length(1), // hint
        ])
        .split(area);

    let logo_lines: Vec<Line> = LOGO
        .iter()
        .map(|l| Line::from(Span::styled(*l, theme::title())))
        .collect();
    f.render_widget(
        Paragraph::new(logo_lines).alignment(Alignment::Center),
        chunks[1],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Neural EEG Codec  ·  OpenHuman Technologies",
            theme::dim(),
        )))
        .alignment(Alignment::Center),
        chunks[3],
    );
    let version = format!("Version  {}", env!("CARGO_PKG_VERSION"));
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(version, theme::normal())))
            .alignment(Alignment::Center),
        chunks[5],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled("loading…", theme::dim())))
            .alignment(Alignment::Center),
        chunks[7],
    );
}

impl Panel for SplashPanel {
    fn id(&self) -> &str {
        "splash"
    }
    fn title(&self) -> &str {
        "LamQuant"
    }

    fn render(&self, state: &AppState, f: &mut Frame, area: Rect) {
        // Vertical centering: figure out top padding so the logo +
        // tagline land in the middle of the available area.
        let content_h: u16 = (LOGO.len() as u16) + 6; // logo + spacer + 3 info rows + tagline
        let pad_top = area.height.saturating_sub(content_h) / 2;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(pad_top),
                Constraint::Length(LOGO.len() as u16),
                Constraint::Length(1), // spacer
                Constraint::Length(1), // tagline
                Constraint::Length(1), // spacer
                Constraint::Length(1), // version row
                Constraint::Length(1), // backend row
                Constraint::Length(1), // platform row
                Constraint::Min(0),    // bottom spacer
                Constraint::Length(1), // hint
            ])
            .split(area);

        let logo_lines: Vec<Line> = LOGO
            .iter()
            .map(|l| Line::from(Span::styled(*l, theme::title())))
            .collect();
        f.render_widget(
            Paragraph::new(logo_lines).alignment(Alignment::Center),
            chunks[1],
        );

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "Neural EEG Codec  ·  OpenHuman Technologies",
                theme::dim(),
            )))
            .alignment(Alignment::Center),
            chunks[3],
        );

        let version = format!(
            "Version  {}  ·  codec lamquant-core  ·  AGPL-3.0",
            env!("CARGO_PKG_VERSION"),
        );
        let backend = format!(
            "Backend  {}  ·  workers {}",
            state.cfg.backend.mode, state.cfg.compute.workers,
        );
        let platform = format!(
            "Platform {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH,
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(version, theme::normal())))
                .alignment(Alignment::Center),
            chunks[5],
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(backend, theme::normal())))
                .alignment(Alignment::Center),
            chunks[6],
        );
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(platform, theme::dim())))
                .alignment(Alignment::Center),
            chunks[7],
        );

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  press any key to continue · auto-advances in a moment",
                theme::dim(),
            )))
            .alignment(Alignment::Center),
            chunks[9],
        );
    }

    fn handle_event(&mut self, _event: KeyEvent, _state: &AppState) -> PanelAction {
        // Any key skips the splash. Q routes home as elsewhere; X quits.
        match _event.code {
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            _ => {
                self.done = true;
                PanelAction::Back
            }
        }
    }

    fn tick(&mut self) {
        self.ticks = self.ticks.saturating_add(1);
        if self.ticks >= AUTO_ADVANCE_TICKS {
            self.done = true;
        }
    }
}
