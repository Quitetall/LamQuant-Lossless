//! Visualization hub — lists EEG/BCI viewer tools with install status.
//!
//! Each tool shows ✓ installed or ✗ not found.
//! Enter on an installed tool spawns it. Enter on a missing tool
//! shows install instructions in the detail pane below the list.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use tui_widget_list::{ListBuilder, ListState, ListView};

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

// ── Tool descriptor ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct VizTool {
    key: &'static str,
    name: &'static str,
    desc: &'static str,
    /// Shell command / binary to probe for existence.
    probe: &'static str,
    /// How to install / where to download if not found.
    install: &'static str,
    /// Command to launch once confirmed present.
    #[allow(dead_code)]
    launch: &'static str,
    /// Optional `launcher.rs` id that runs an auto-install (e.g.
    /// `viz_install_mne` → `pip install mne`). None for tools that
    /// can't be auto-installed (commercial software, MATLAB-only,
    /// or anything requiring a GUI installer). When Some, [i] in
    /// the panel triggers the install via the standard launcher path.
    /// All install launchers use --force / --force-reinstall so
    /// firing the same launcher on an existing install acts as
    /// repair (rebuilds + replaces the binary).
    install_id: Option<&'static str>,
    /// Optional `launcher.rs` id that uninstalls the tool. Paired
    /// with install_id when set. None for tools where uninstall
    /// makes no sense (manual-install software).
    uninstall_id: Option<&'static str>,
    installed: bool,
}

impl VizTool {
    fn new(
        key: &'static str,
        name: &'static str,
        desc: &'static str,
        probe: &'static str,
        install: &'static str,
        launch: &'static str,
    ) -> Self {
        let installed = command_exists(probe);
        Self {
            key,
            name,
            desc,
            probe,
            install,
            launch,
            install_id: None,
            uninstall_id: None,
            installed,
        }
    }

    fn with_install_id(mut self, id: &'static str) -> Self {
        self.install_id = Some(id);
        self
    }

    fn with_uninstall_id(mut self, id: &'static str) -> Self {
        self.uninstall_id = Some(id);
        self
    }
}

fn command_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// ── Panel ─────────────────────────────────────────────────────────────────────

pub struct VisualizationPanel {
    id: String,
    tools: Vec<VizTool>,
    selected: usize,
    /// tui-widget-list state — synced from `selected` each render.
    list_state: std::cell::RefCell<ListState>,
    /// Last probe wall time. Throttles tick-time re-probes to ~1 Hz —
    /// `which` is fast but firing it 20x per second across 11 tools
    /// is wasteful. Auto-refresh keeps the ✓/✗ status fresh after
    /// install / uninstall completes without forcing the user to
    /// press [r].
    last_probe: std::time::Instant,
}

impl Default for VisualizationPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualizationPanel {
    pub fn new() -> Self {
        let tools = vec![
            // ── EEG / BCI viewers ────────────────────────────────────────
            VizTool::new(
                "1", "Vision GUI",
                "LamQuant live device dashboard (impedance, signals, export)",
                "lamquant-gui",
                "cargo install --path gui/src-tauri --bin lamquant-gui\n  (run from repo root; needs Tauri webview deps on Linux:\n   libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf)",
                "lamquant-gui",
            ).with_install_id("viz_install_lamquant_gui").with_uninstall_id("viz_uninstall_lamquant_gui"),
            VizTool::new(
                "2", "EEGLab",
                "MATLAB/Octave EEG analysis suite — ICA, time-frequency, STUDY",
                "eeglab",
                "Download from:  https://sccn.ucsd.edu/eeglab/downloadtoolbox.php\n  MATLAB required. After install, add EEGLab to MATLAB path and run `eeglab`.",
                "eeglab",
            ),
            VizTool::new(
                "3", "MNE-Python",
                "Open-source MEG/EEG analysis in Python",
                "mne",
                "Install with:  pip install mne\n  Then launch:  python3 -c \"import mne; mne.gui.browse_raw()\"",
                "mne",
            ).with_install_id("viz_install_mne").with_uninstall_id("viz_uninstall_mne"),
            VizTool::new(
                "4", "OpenBCI GUI",
                "Real-time BCI signal viewer — EEG, EMG, EKG",
                "OpenBCIGUI",
                "Download from:  https://openbci.com/downloads\n  Requires Java 8+.",
                "OpenBCIGUI",
            ),
            VizTool::new(
                "5", "scope-tui",
                "Terminal oscilloscope / spectroscope for EEG signal inspection",
                "scope-tui",
                "Install with:  cargo install scope-tui\n  Launch:  scope-tui",
                "scope-tui",
            ).with_install_id("viz_install_scope_tui").with_uninstall_id("viz_uninstall_scope_tui"),
            VizTool::new(
                "6", "BrainVision Analyzer",
                "Clinical-grade EEG analysis (Brain Products)",
                "BVAnalyzer",
                "Download from:  https://www.brainproducts.com/downloads/\n  Commercial license required.",
                "BVAnalyzer",
            ),
            VizTool::new(
                "7", "BESA",
                "Source analysis, dipole fitting, ERP analysis",
                "besa",
                "Download from:  https://www.besa.de/downloads/\n  Commercial license required.",
                "besa",
            ),
            // ── System / corpus tools ─────────────────────────────────────
            VizTool::new(
                "b", "bottom",
                "Cross-platform system monitor — CPU, RAM, GPU during training",
                "btm",
                "Install with:  cargo install bottom\n  or:  brew install bottom",
                "btm",
            ).with_install_id("viz_install_bottom").with_uninstall_id("viz_uninstall_bottom"),
            VizTool::new(
                "t", "television",
                "Fuzzy finder TUI — fast file picking for codec hub",
                "tv",
                "Install with:  cargo install television",
                "tv",
            ).with_install_id("viz_install_television").with_uninstall_id("viz_uninstall_television"),
            VizTool::new(
                "c", "csvlens",
                "Terminal CSV viewer — inspect exported EEG metrics",
                "csvlens",
                "Install with:  cargo install csvlens",
                "csvlens",
            ).with_install_id("viz_install_csvlens").with_uninstall_id("viz_uninstall_csvlens"),
            VizTool::new(
                "g", "gitui",
                "Terminal git client — corpus version control",
                "gitui",
                "Install with:  cargo install gitui",
                "gitui",
            ).with_install_id("viz_install_gitui").with_uninstall_id("viz_uninstall_gitui"),
        ];
        Self {
            id: "visualization".to_string(),
            tools,
            selected: 0,
            list_state: std::cell::RefCell::new(ListState::default()),
            last_probe: std::time::Instant::now(),
        }
    }

    /// Re-probe every tool's installed flag. Cheap — `which` is one
    /// syscall per tool, ~11 tools = sub-millisecond total.
    fn refresh_probes(&mut self) {
        for tool in &mut self.tools {
            tool.installed = command_exists(tool.probe);
        }
        self.last_probe = std::time::Instant::now();
    }

    fn detail_lines(&self) -> Vec<Line<'static>> {
        let tool = &self.tools[self.selected];
        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled(tool.name, theme::highlight()),
            Span::raw("  "),
            Span::styled(tool.desc, theme::dim()),
        ]));
        lines.push(Line::from(""));
        if tool.installed {
            lines.push(Line::from(vec![
                Span::styled("✓  Installed", theme::success()),
                Span::raw("   "),
                Span::styled("[Enter]", theme::key_hint()),
                Span::raw(" launch"),
            ]));
            // Repair + uninstall hints when paired launchers exist.
            if tool.install_id.is_some() || tool.uninstall_id.is_some() {
                let mut spans: Vec<Span> = vec![Span::raw("   ")];
                if tool.install_id.is_some() {
                    spans.push(Span::styled("[R]", theme::key_hint()));
                    spans.push(Span::raw(" repair   "));
                }
                if tool.uninstall_id.is_some() {
                    spans.push(Span::styled("[u]", theme::key_hint()));
                    spans.push(Span::raw(" uninstall"));
                }
                lines.push(Line::from(spans));
            }
        } else {
            lines.push(Line::from(vec![
                Span::styled("✗  Not found", theme::error()),
                Span::raw(format!("  (probe: {})", tool.probe)),
            ]));
            lines.push(Line::from(""));
            if tool.install_id.is_some() {
                lines.push(Line::from(vec![
                    Span::raw("  Press "),
                    Span::styled("[Enter]", theme::key_hint()),
                    Span::raw(" or "),
                    Span::styled("[i]", theme::key_hint()),
                    Span::raw(" to auto-install:"),
                ]));
            } else {
                lines.push(Line::from(Span::styled(
                    "  Manual install required:",
                    theme::dim(),
                )));
            }
            for line in tool.install.split('\n') {
                lines.push(Line::from(Span::styled(
                    format!("  {}", line.trim()),
                    theme::normal(),
                )));
            }
        }
        lines
    }
}

impl Panel for VisualizationPanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Visualization"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),   // tool list
                Constraint::Length(7), // detail pane
            ])
            .split(area);

        // ── Tool list via tui-widget-list ─────────────────────────────────
        let tools_snapshot: Vec<VizTool> = self.tools.clone();
        let selected_idx = self.selected;

        let builder = ListBuilder::new(move |context| {
            let tool = &tools_snapshot[context.index];
            let i = context.index;
            let is_sel = i == selected_idx;

            let status = if tool.installed {
                Span::styled(" ✓ ", theme::success())
            } else {
                Span::styled(" ✗ ", theme::error())
            };
            let label_style = if is_sel {
                theme::selected()
            } else {
                theme::normal()
            };
            let marker = if is_sel {
                Span::styled("▶ ", theme::highlight())
            } else {
                Span::raw("  ")
            };
            let line = Line::from(vec![
                marker,
                Span::styled(format!("[{}]", tool.key), theme::key_hint()),
                status,
                Span::styled(format!("{:<26}", tool.name), label_style),
                Span::styled(format!("  {}", tool.desc), theme::dim()),
            ]);
            (line, 1u16)
        });

        let list = ListView::new(builder, self.tools.len()).block(
            Block::default()
                .title(Span::styled(" EEG / BCI / System Tools ", theme::heading()))
                .borders(Borders::ALL)
                .border_style(theme::dim()),
        );

        let mut state = self.list_state.borrow_mut();
        state.select(Some(self.selected));
        f.render_stateful_widget(list, chunks[0], &mut *state);

        // ── Detail pane ──────────────────────────────────────────────────
        let detail = Paragraph::new(self.detail_lines())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme::dim()),
            )
            .wrap(Wrap { trim: false });
        f.render_widget(detail, chunks[1]);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.tools.len() {
                    self.selected += 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                let tool = &self.tools[self.selected];
                if tool.installed {
                    PanelAction::Navigate(format!("launch:viz_{}", tool.probe))
                } else if let Some(install_id) = tool.install_id {
                    // Auto-install path: Enter on a missing tool that
                    // has a registered installer fires the install
                    // launcher (cargo install / pip install / etc.).
                    PanelAction::Navigate(format!("launch:{}", install_id))
                } else {
                    // No auto-install registered (commercial software,
                    // GUI installer required, etc.) — leave the
                    // detail pane's manual instructions visible.
                    PanelAction::StatusMessage(format!(
                        "{} requires manual install — see instructions below.",
                        tool.name,
                    ))
                }
            }
            KeyCode::Char('i') => {
                let tool = &self.tools[self.selected];
                if tool.installed {
                    PanelAction::StatusMessage(format!(
                        "{} already installed (use [R] to repair / [u] to uninstall).",
                        tool.name,
                    ))
                } else if let Some(install_id) = tool.install_id {
                    PanelAction::Navigate(format!("launch:{}", install_id))
                } else {
                    PanelAction::StatusMessage(format!(
                        "No auto-install for {} — see manual instructions.",
                        tool.name,
                    ))
                }
            }
            // [R] = repair: re-fire the install launcher even if the
            // tool is currently installed. Install entries use
            // --force / --force-reinstall so this rebuilds + replaces.
            KeyCode::Char('R') => {
                let tool = &self.tools[self.selected];
                match tool.install_id {
                    Some(id) => PanelAction::Navigate(format!("launch:{}", id)),
                    None => PanelAction::StatusMessage(format!(
                        "No auto-repair for {} — manual reinstall required.",
                        tool.name,
                    )),
                }
            }
            // [u] = uninstall via the paired uninstall_id launcher.
            KeyCode::Char('u') => {
                let tool = &self.tools[self.selected];
                if !tool.installed {
                    PanelAction::StatusMessage(format!(
                        "{} not installed — nothing to uninstall.",
                        tool.name,
                    ))
                } else if let Some(id) = tool.uninstall_id {
                    PanelAction::Navigate(format!("launch:{}", id))
                } else {
                    PanelAction::StatusMessage(format!("No auto-uninstall for {}.", tool.name,))
                }
            }
            KeyCode::Char(c) => {
                let key = c.to_string();
                if let Some(tool) = self.tools.iter().find(|t| t.key == key) {
                    if tool.installed {
                        return PanelAction::Navigate(format!("launch:viz_{}", tool.probe));
                    } else {
                        let idx = self.tools.iter().position(|t| t.key == key).unwrap_or(0);
                        self.selected = idx;
                        return PanelAction::Consumed;
                    }
                }
                match c {
                    'r' => {
                        self.refresh_probes();
                        PanelAction::StatusMessage("Re-scanned installed tools.".into())
                    }
                    'x' => PanelAction::Quit,
                    'q' => PanelAction::Home,
                    'B' => PanelAction::Back,
                    'h' | '?' => PanelAction::Navigate(router::SCREEN_HELP.to_string()),
                    _ => PanelAction::Ignored,
                }
            }
            KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            _ => PanelAction::Ignored,
        }
    }

    /// Auto-refresh installed status every ~1s. Catches the case
    /// where the user fired install/uninstall, came back to the
    /// panel, and would otherwise see a stale ✓/✗ until pressing
    /// [r] manually.
    fn tick(&mut self) {
        if self.last_probe.elapsed() >= std::time::Duration::from_secs(1) {
            self.refresh_probes();
        }
    }
}
