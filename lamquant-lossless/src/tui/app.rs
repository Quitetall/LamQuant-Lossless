//! App — main event loop. Owns terminal, router, panels, op state, history.
//!
//! Architecture:
//!   - Static panels (menus, settings, help) live in a HashMap by screen ID.
//!   - Stateful "flow" panels (file_browser, input, output) are direct fields
//!     because they get reset/reconfigured per operation.
//!   - PendingOp tracks the user's progress through an op flow:
//!     PickInput -> [PickOutput] -> Running -> Done.

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::Duration;

use super::operations::{channel, op_spec, runner};
use super::panel::{Action, Panel, PanelAction};
use super::panels::browse_results::BrowseResultsPanel;
use super::panels::exit_confirm::ExitConfirmPanel;
use super::panels::file_browser::FileBrowserPanel;
use super::panels::help::HelpPanel;
use super::panels::info_text::InfoTextPanel;
use super::panels::input::InputPanel;
use super::panels::main_hub::{HubTile, MainHubPanel};
use super::panels::menu::{MenuItem, MenuPanel};
use super::panels::output::OutputPanel;
use super::panels::preflight::PreflightPanel;
use super::panels::lossless_prompt::LosslessPromptPanel;
use super::panels::resume::{ResumeAction, ResumePanel};
use super::panels::root_warn::RootWarnPanel;
use super::panels::settings::SettingsPanel;
use super::panels::settings_help::SettingsHelpPanel;
use super::panels::splash::SplashPanel;
use super::panels::syscheck::SyscheckPanel;
use super::panels::tutorial::TutorialPanel;
use super::panels::visualization::VisualizationPanel;
use super::panels::wizard::WizardPanel;
use super::router::{self, Router};
use super::state::{AppState, LaunchState, PendingOp, TrackedProcess};
use super::theme;
use lamquant_ops::{Peer, SshTransport, Transport};

/// Grace period after an op finishes successfully before auto-routing
/// back to the previous screen. Long enough that the user sees the
/// final "✓ done" state on the output panel; short enough that a
/// trivial op doesn't feel sticky. Ops that fail/cancel STAY on
/// SCREEN_RUNNING so the user can read the error.
const AUTO_BACK_GRACE: Duration = Duration::from_millis(1500);

const SCREEN_FILE_PICKER: &str = "file_picker";
/// Screen ID for the codec browse panel (recursively listed lml/lmq files).
/// Same string as `router::SCREEN_BROWSE` so menu items can stay simple.
const SCREEN_BROWSE_RESULTS: &str = "browse";
const SCREEN_TUTORIAL: &str = "tutorial";
// Local alias matching the canonical router::SCREEN_LOSSLESS_PROMPT.
// app.rs declares local consts for every screen id (legacy pattern
// predating router::); keep this one in sync if the router const
// ever moves -- the test `local_screen_consts_match_router` at the
// bottom of this file locks the two strings together.
const SCREEN_LOSSLESS_PROMPT: &str = "lossless_prompt";
const SCREEN_SYSCHECK: &str = "syscheck";
const SCREEN_RESUME: &str = "resume";
const SCREEN_SPLASH: &str = "splash";
const SCREEN_INPUT_PATH: &str = "input_path";
const SCREEN_OUTPUT_PICKER: &str = "output_picker";
const SCREEN_PREFLIGHT: &str = "preflight";
const SCREEN_RUNNING: &str = "running";
const SCREEN_EXIT_CONFIRM: &str = "exit_confirm";
const SCREEN_SETTINGS_HELP: &str = "settings_help";
const SCREEN_WIZARD: &str = "wizard";
const SCREEN_ROOT_WARN: &str = "root_warn";

/// The block-letter LAMQUANT splash. Public so `panels::main_hub` and any
/// future tooling can reuse the exact same glyph block — single source of
/// truth across the GUI's `<Logo/>`, the Python TUI's `_LOGO`, and this.
pub(crate) const LOGO: &[&str] = &[
    " ██╗      █████╗ ███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗",
    " ██║     ██╔══██╗████╗ ████║██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝",
    " ██║     ███████║██╔████╔██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ",
    " ██║     ██╔══██║██║╚██╔╝██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ",
    " ███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ",
    " ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝",
];

// LaunchState, TrackedProcess, PendingOp now live in `super::state`. They
// were moved there in commit 2 of the reactive-store refactor so AppState can
// own the fields (`launch_state`, `processes`, `pending_op`) directly.

pub struct App {
    router: Router,
    /// Single source of truth for app-global mutable state. Mutated only via
    /// `App::dispatch(Action)`. Panels read from this in commits 3+.
    state: AppState,
    panels: HashMap<String, Box<dyn Panel>>,
    file_browser: FileBrowserPanel,
    output_picker: FileBrowserPanel,
    input_panel: InputPanel,
    output_panel: OutputPanel,
    preflight_panel: PreflightPanel,
    exit_confirm: ExitConfirmPanel,
    settings_panel: SettingsPanel,
    settings_help_panel: SettingsHelpPanel,
    wizard_panel: WizardPanel,
    root_warn_panel: RootWarnPanel,
    browse_panel: BrowseResultsPanel,
    tutorial_panel: TutorialPanel,
    lossless_prompt_panel: LosslessPromptPanel,
    syscheck_panel: SyscheckPanel,
    resume_panel: ResumePanel,
    splash_panel: SplashPanel,
    viz_panel: VisualizationPanel,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        // AppState owns cfg + history (loaded once, mutated through dispatch).
        let state = AppState::new();

        // Apply user color/charset preferences from config (or env, or auto-detect).
        super::theme::detect(&state.cfg.output.color, &state.cfg.output.charset);

        let initial_cfg = state.cfg.clone();
        let mut app = Self {
            router: Router::new(),
            state,
            panels: HashMap::new(),
            file_browser: FileBrowserPanel::new("Select File"),
            output_picker: FileBrowserPanel::new_save("Save As", "output.lml"),
            input_panel: InputPanel::new("Enter value"),
            output_panel: OutputPanel::new(),
            preflight_panel: PreflightPanel::new(),
            exit_confirm: ExitConfirmPanel::new(),
            settings_panel: SettingsPanel::new(),
            settings_help_panel: SettingsHelpPanel::new(),
            wizard_panel: WizardPanel::new(initial_cfg),
            root_warn_panel: RootWarnPanel::new(),
            browse_panel: BrowseResultsPanel::new(),
            tutorial_panel: TutorialPanel::new(),
            lossless_prompt_panel: LosslessPromptPanel::new(),
            syscheck_panel: SyscheckPanel::new(),
            resume_panel: ResumePanel::new(),
            splash_panel: SplashPanel::new(),
            viz_panel: VisualizationPanel::new(),
        };
        app.register_panels();
        if app.state.history.interrupted {
            if let (Some(op), Some(input)) = (
                &app.state.history.last_op.clone(),
                &app.state.history.last_input.clone(),
            ) {
                app.resume_panel
                    .set_target(op, input, app.state.history.last_output.as_deref());
                app.router.navigate(SCREEN_RESUME);
            } else {
                // Stale interrupted flag with no detail — silently clear.
                app.state.history.mark_complete();
            }
        }

        // First-time wizard: trigger when no config file exists yet.
        let cfg_path = super::config::config_path();
        if !cfg_path.exists() {
            app.router.navigate(SCREEN_WIZARD);
        } else if running_as_root()
            && app.state.cfg.output.warn_root
            && !app.state.cfg.output.allow_root
        {
            app.router.navigate(SCREEN_ROOT_WARN);
        }
        // Splash screen on top of whichever screen the boot logic
        // landed on. Auto-pops via take_done in tick_panels after
        // ~700ms, or instantly on any keypress.
        app.router.navigate(SCREEN_SPLASH);
        app
    }

    fn register_panels(&mut self) {
        // Main hub — uses the dedicated splash panel with the LAMQUANT
        // logo, sectioned WORKFLOWS / SYSTEM blocks, and the bottom
        // version + Ready footer. Tile keys, IDs, and order MUST match
        // `specs/ui-parity.md::Workflow inventory` byte-for-byte —
        // `crates/lamquant-ops/tests/workflow_parity.rs` enforces parity
        // with the GUI's TS workflows.ts and the Python TUI menu.
        self.panels.insert(
            router::SCREEN_MAIN.to_string(),
            Box::new(MainHubPanel::new(
                router::SCREEN_MAIN,
                vec![
                    HubTile::new(
                        "1",
                        "Visualization",
                        "Live EEG viewers · launch external tools",
                        router::SCREEN_VIZ,
                    ),
                    HubTile::new(
                        "2",
                        "Codec Hub",
                        "Compress · decompress · browse · verify",
                        router::SCREEN_CODEC_HUB,
                    ),
                    HubTile::new(
                        "3",
                        "Validation Suite",
                        "Eagle Validation Suite · LQS compliance · benchmarks",
                        router::SCREEN_EAGLE,
                    ),
                    HubTile::new(
                        "4",
                        "Firmware Hub",
                        "Flash · export · build for MCUs",
                        router::SCREEN_FIRMWARE,
                    ),
                    HubTile::new(
                        "5",
                        "Train a model",
                        "ML training cockpit · experiments",
                        router::SCREEN_TRAIN,
                    ),
                ],
                vec![
                    HubTile::new(
                        "N",
                        "Peers",
                        "Remote LamQuant devices · SSH targets",
                        router::SCREEN_PEERS,
                    ),
                    HubTile::new(
                        "s",
                        "Settings",
                        "Workers · paths · device profiles",
                        router::SCREEN_SETTINGS,
                    ),
                    HubTile::new(
                        "i",
                        "Install & setup",
                        "Wizard · dependencies · syscheck · GPU probe",
                        router::SCREEN_SETUP,
                    ),
                    HubTile::new(
                        "t",
                        "Diagnostics",
                        "Internal Testing Suite · Crashlog Viewer · Health Check",
                        router::SCREEN_TEST,
                    ),
                ],
            )),
        );

        // Boxed mode-specific panels (Operations + Status). Direct port
        // of the old Python `_codec_hub` mode branch — replaces the plain
        // MenuPanel that lived here. `[m]` toggles between LML and LMQ.
        use super::panels::mode_panel::{CodecMode, ModePanel};
        self.panels.insert(
            router::SCREEN_LOSSLESS.to_string(),
            Box::new(ModePanel::new(CodecMode::Lossless)),
        );
        self.panels.insert(
            router::SCREEN_NEURAL.to_string(),
            Box::new(ModePanel::new(CodecMode::Neural)),
        );

        self.panels.insert(
            router::SCREEN_CODEC_HUB.to_string(),
            Box::new(super::panels::codec_hub::CodecHubPanel::new()),
        );

        self.panels.insert(
            router::SCREEN_ARCHIVE.to_string(),
            Box::new(MenuPanel::new(
                router::SCREEN_ARCHIVE,
                "LMA Archives",
                "Pack and unpack LMA archives",
                vec![
                    MenuItem::new("1", "Pack", "Directory → .lma", "op:archive"),
                    MenuItem::new("2", "Extract", ".lma → Directory", "op:extract"),
                    MenuItem::new("3", "List", "Show archive contents", "op:list_archive"),
                    MenuItem::new(
                        "4",
                        "Verify",
                        "Check archive integrity",
                        "op:verify_archive",
                    ),
                ],
            )),
        );

        self.panels.insert(
            router::SCREEN_FIRMWARE.to_string(),
            Box::new(super::panels::firmware::FirmwarePanel::new()),
        );

        // Settings — full editor.
        // Settings + settings_help live as direct fields (App owns them) so
        // the help panel can be populated from settings panel state.

        // Help, Train/Eagle/Setup as info_text.
        self.panels
            .insert(router::SCREEN_HELP.to_string(), Box::new(HelpPanel::new()));
        self.panels.insert(
            router::SCREEN_PEERS.to_string(),
            Box::new(super::panels::peers::PeersPanel::new()),
        );

        // "Coming in v1.1" placeholder for deferred features. Cockpit and
        // Eagle stubs (hyperparam editor, experiment runner, leaderboard,
        // downstream tasks, etc.) navigate here so users see a single
        // clear "not yet built" screen instead of fading status messages.
        // Listed in alphabetical groups so the same screen serves both
        // panels and stays scannable.
        let coming_body: Vec<Line<'static>> = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Features tracked for the v1.1 milestone",
                theme::dim(),
            )),
            Line::from(""),
            Line::from(Span::styled("  Training cockpit:", theme::heading())),
            Line::from(Span::raw("    [4]  Queue & schedule pipeline runner")),
            Line::from(Span::raw("    [6]  Hyperparameter editor (TrainingConfig)")),
            Line::from(Span::raw("    [7]  Experiment runner — A/B sweeps")),
            Line::from(Span::raw("    [9]  Leaderboard — runs ranked by R")),
            Line::from(Span::raw("    [a]  Compare runs — side-by-side metrics")),
            Line::from(""),
            Line::from(Span::styled("  Eagle validation:", theme::heading())),
            Line::from(Span::raw(
                "    [3]  Targeted LQS level (use [1] for full sweep)",
            )),
            Line::from(Span::raw(
                "    [7]  Downstream tasks (seizure / sleep / pathology)",
            )),
            Line::from(Span::raw("    [8]  Hallucination detection suite")),
            Line::from(Span::raw("    [p]  Publish badge — signed compliance cert")),
            Line::from(Span::raw("    [x]  Export report (HTML)")),
            Line::from(""),
            Line::from(Span::styled("  Firmware:", theme::heading())),
            Line::from(Span::raw(
                "    [c]  cmake configure — per-target toolchain wiring",
            )),
            Line::from(Span::raw("    [m]  make build — depends on configure")),
            Line::from(Span::raw(
                "    [f]  picotool/esptool flash — toolchain-specific",
            )),
            Line::from(Span::raw(
                "    [z]  size report — riscv32-elf-size / arm-none-eabi-size",
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  These keys currently route here. Track progress at",
                theme::dim(),
            )),
            Line::from(Span::styled(
                "  https://github.com/openhuman-ai/LamQuant/milestones",
                theme::dim(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press [b] / [Esc] / [Backspace] / [Enter] to return.",
                theme::dim(),
            )),
        ];
        self.panels.insert(
            router::SCREEN_COMING_V11.to_string(),
            Box::new(InfoTextPanel::new(
                router::SCREEN_COMING_V11,
                "Coming in v1.1",
                coming_body,
            )),
        );

        self.panels.insert(
            router::SCREEN_TRAIN.to_string(),
            Box::new(super::panels::cockpit::CockpitPanel::new()),
        );

        // Stale linear menu kept under a hidden screen id for backward compat
        // (no UI references it; left here so existing tests keep building).
        self.panels.insert(
            "train_legacy".to_string(),
            Box::new(MenuPanel::new(
                "train_legacy",
                "Train a Model",
                "legacy",
                vec![
                    MenuItem::new(
                        "1",
                        "Encoder",
                        "V1/V2 lossy encoder",
                        "launch:train_encoder",
                    ),
                    MenuItem::new("2", "SNN", "Mamba SNN seizure detector", "launch:train_snn"),
                    MenuItem::new(
                        "3",
                        "TNN",
                        "Ternary NN firmware-deployable",
                        "launch:train_tnn",
                    ),
                    MenuItem::new(
                        "4",
                        "Resume latest",
                        "Continue interrupted run from checkpoint",
                        "launch:train_resume",
                    ),
                ],
            )),
        );

        self.panels.insert(
            router::SCREEN_EAGLE.to_string(),
            Box::new(super::panels::eagle::EaglePanel::new()),
        );

        self.panels.insert(
            router::SCREEN_SETUP.to_string(),
            Box::new(super::panels::portal::PortalPanel::new()),
        );

        // Tests menu (cmd_test in Python)
        self.panels.insert(
            router::SCREEN_TEST.to_string(),
            Box::new(MenuPanel::new(
                router::SCREEN_TEST,
                "Test Suites",
                "pytest invocations",
                vec![
                    MenuItem::new(
                        "1",
                        "Conformance",
                        "Codec roundtrip + boundary stress",
                        "launch:test_conformance",
                    ),
                    MenuItem::new("2", "Codec", "Codec subsystem only", "launch:test_codec"),
                    MenuItem::new("3", "Full", "All tests", "launch:test_full"),
                    MenuItem::new(
                        "4",
                        "Paranoid",
                        "Full + extended stress (slow)",
                        "launch:test_paranoid",
                    ),
                ],
            )),
        );

        // Codec browse (`browse` screen) is rendered via direct-field
        // BrowseResultsPanel — no panels HashMap entry needed.
    }

    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
        while !self.state.should_quit {
            terminal.draw(|f| self.render(f))?;
            self.handle_events()?;
            self.tick_panels();
        }
        Ok(())
    }

    // ── Test-only helpers ───────────────────────────────────────────────
    // Exposed so integration tests can drive the app with synthetic keys
    // and snapshot the rendered ratatui buffer. Not used in normal runs.

    /// Dispatch a single key event as if the user pressed it.
    /// Mirrors `handle_events` but skips the crossterm event::poll wait.
    pub fn dispatch_key_for_test(&mut self, key: crossterm::event::KeyEvent) {
        // Replicate the global Ctrl+C branch from handle_events so tests can
        // verify the cancel-vs-quit logic.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            // Global Ctrl+C policy:
            //   1) If a running op exists and hasn't been told to cancel yet,
            //      cancel it and stay where we are. The user is mid-task and
            //      wants to abort the work, not abandon the session.
            //   2) If the op has already been cancel-requested OR no op is
            //      running, route through the normal exit-confirm panel
            //      instead of force-quitting. This stops users from
            //      accidentally trashing their session with a fast double
            //      Ctrl+C — the confirm panel still accepts a single Enter
            //      to quit, so the keystroke cost is unchanged for the
            //      "yes, I really mean it" path.
            if self.router.current() == SCREEN_RUNNING && self.cancel_active_op() {
                self.state
                    .set_status("Cancelling op... (Ctrl+C again to confirm exit)");
                return;
            }
            if self.router.current() != SCREEN_EXIT_CONFIRM {
                self.router.navigate(SCREEN_EXIT_CONFIRM);
            }
            return;
        }
        let screen = self.router.current().to_string();
        let action = self.dispatch_event(&screen, key);
        self.dispatch(&screen, action.into());
    }

    /// Render once into the supplied terminal. Generic over backend so tests
    /// can use `ratatui::backend::TestBackend`.
    pub fn render_for_test<B: ratatui::backend::Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> io::Result<()> {
        terminal
            .draw(|f| self.render(f))
            .map_err(|e| io::Error::other(e.to_string()))?;
        Ok(())
    }

    /// Run a tick (panel timers, op event drain) without consuming any input.
    pub fn tick_for_test(&mut self) {
        self.tick_panels();
    }

    /// Test-only: dispatch an Action directly to the reducer, bypassing
    /// the key-event translation layer. Reducer unit tests use this to
    /// assert state transitions for typed Actions without spawning
    /// subprocesses or simulating keystrokes.
    #[doc(hidden)]
    pub fn dispatch_for_test(&mut self, action: Action) {
        let screen = self.router.current().to_string();
        self.dispatch(&screen, action);
    }

    /// Test-only: read access to the AppState for reducer assertions.
    #[doc(hidden)]
    pub fn state_for_test(&self) -> &AppState {
        &self.state
    }

    /// Whether the app is set to quit. Tests check this after sending q/y.
    pub fn should_quit(&self) -> bool {
        self.state.should_quit
    }

    /// Current router screen ID. Tests assert navigation correctness.
    pub fn current_screen(&self) -> String {
        self.router.current().to_string()
    }

    fn tick_panels(&mut self) {
        for panel in self.panels.values_mut() {
            panel.tick();
        }
        self.file_browser.tick();
        self.output_picker.tick();
        self.input_panel.tick();
        self.output_panel.tick();
        self.settings_panel.tick();
        self.settings_help_panel.tick();
        self.wizard_panel.tick();
        self.root_warn_panel.tick();
        self.browse_panel.tick();
        self.tutorial_panel.tick();
        self.syscheck_panel.tick();
        self.resume_panel.tick();
        // Splash auto-advance: tick the panel, then pop the splash off
        // the router stack once it flags done. Pops back to whatever
        // boot screen was pushed beneath (Main / Resume / Wizard /
        // RootWarn). Idempotent — take_done clears the flag.
        if self.router.current() == SCREEN_SPLASH {
            self.splash_panel.tick();
            if self.splash_panel.take_done() {
                self.router.back();
            }
        }
        self.viz_panel.tick();
        self.state.context_tick = self.state.context_tick.wrapping_add(1);

        // P3: drain subprocess events through the dispatch chokepoint.
        // Without this, OpEvent state mutations bypassed `dispatch` —
        // panel state was correct but invisible to any future replay /
        // snapshot / centralized logging path.
        let screen = self.router.current().to_string();
        while let Some(ev) = self.output_panel.try_recv_event() {
            self.dispatch(&screen, Action::OpEvent(ev));
        }

        // Settings save → AppState sync. The snapshot is taken at save-time
        // inside SettingsPanel so post-save edits cannot leak through.
        // Tick fires every ~50ms on every screen → catches all exit paths
        // (b, q, x, navigate-away). Without this, `state.cfg` stayed pinned
        // to the load-at-startup values and every spoke panel rendered stale
        // config until process restart, defeating the reactive-store
        // refactor's whole purpose.
        if let Some(saved_cfg) = self.settings_panel.take_saved_cfg() {
            self.state.cfg = saved_cfg;
            super::theme::detect(&self.state.cfg.output.color, &self.state.cfg.output.charset);
            self.output_panel.bell_on_done = self.state.cfg.output.bell_on_done;
            self.state.set_status("Settings saved — config in sync.");
        }

        // Sync process done/failed/cancelled state from output panel immediately.
        if self.output_panel.is_done() {
            let failed = self.output_panel.is_failed();
            let cancelled = self.output_panel.is_cancelled();
            if let Some(p) = self.state.processes.iter_mut().rev().find(|p| !p.done) {
                p.mark_done(failed, cancelled);
            }

            // Task #49 fix: auto-route back to the previous screen after a
            // successful op finishes, so users don't have to press [b]
            // manually when a workflow completes cleanly. Failed and
            // cancelled ops STAY on SCREEN_RUNNING so the user can read
            // the error / cancellation summary.
            if !failed && !cancelled && self.router.current() == SCREEN_RUNNING {
                if self.state.op_done_at.is_none() {
                    self.state.op_done_at = Some(std::time::Instant::now());
                }
                if let Some(t) = self.state.op_done_at {
                    if t.elapsed() >= AUTO_BACK_GRACE {
                        self.handle_back(SCREEN_RUNNING);
                        self.state.op_done_at = None;
                    }
                }
            } else {
                // Clear the timer on failure/cancel so a previous success
                // doesn't trigger after a manual retry.
                self.state.op_done_at = None;
            }
        } else {
            // Op no longer reports done (rare: re-entered RUNNING for a
            // new op without explicit Started). Clear the timer.
            self.state.op_done_at = None;
        }

        // Prune entries that have been done for > 20s to keep STATUS clean.
        self.state.processes.retain(|p| !p.expired());

        // Launch-state timeout: 5 seconds → Failed.
        if let LaunchState::Launching { tool, started } = &self.state.launch_state {
            if started.elapsed() >= std::time::Duration::from_secs(5) {
                self.state.launch_state = LaunchState::Failed(tool.clone());
            }
        }
    }

    fn handle_events(&mut self) -> io::Result<()> {
        // Block up to 50ms for the first event so the loop doesn't spin
        // when idle. Once one event is ready, drain pending events
        // before returning so terminal-buffered key repeats don't queue
        // up one-per-frame (~500ms lag on fast j/k).
        //
        // Per-tick cap: a multi-megabyte bracketed paste turns into
        // tens of thousands of events. Without a cap the drain blocks
        // for seconds while the renderer is frozen. Cap at DRAIN_MAX so
        // pasting stays responsive — leftover events pick up next tick.
        const DRAIN_MAX: usize = 64;
        if !event::poll(Duration::from_millis(50))? {
            return Ok(());
        }
        for _ in 0..DRAIN_MAX {
            let Event::Key(key) = event::read()? else {
                if !event::poll(Duration::from_millis(0))? {
                    break;
                }
                continue;
            };

            // crossterm 0.27+ may surface Release / Repeat events on
            // terminals that opt into the Kitty Keyboard Protocol. Both
            // would double-fire every keypress — most visibly, a Ctrl+C
            // would (a) cancel the op (Press), then (b) navigate to
            // SCREEN_EXIT_CONFIRM (Release, because was_killed=true so
            // cancel_active_op returns false). Filter to Press only.
            if key.kind != KeyEventKind::Press {
                if !event::poll(Duration::from_millis(0))? {
                    break;
                }
                continue;
            }

            // Global Ctrl+C policy. See `dispatch_key_for_test` for the rationale —
            // the two paths must stay in lockstep so tests reflect runtime behaviour.
            // 1) On SCREEN_RUNNING with a live op: cancel the op, stay put.
            // 2) Otherwise (or after the op has already been cancel-requested):
            //    route through SCREEN_EXIT_CONFIRM rather than force-quitting,
            //    so a stray double Ctrl+C cannot trash the user's session.
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                if self.router.current() == SCREEN_RUNNING && self.cancel_active_op() {
                    self.state
                        .set_status("Cancelling op... (Ctrl+C again to confirm exit)");
                    return Ok(());
                }
                if self.router.current() != SCREEN_EXIT_CONFIRM {
                    self.router.navigate(SCREEN_EXIT_CONFIRM);
                }
                return Ok(());
            }

            let screen = self.router.current().to_string();
            let action = self.dispatch_event(&screen, key);
            self.dispatch(&screen, action.into());

            // Bail if no more events queued — render the result before
            // potentially blocking again.
            if !event::poll(Duration::from_millis(0))? {
                break;
            }
        }
        Ok(())
    }

    fn dispatch_event(&mut self, screen: &str, mut key: crossterm::event::KeyEvent) -> PanelAction {
        // Treat `;` as an alias for Enter everywhere — easier one-hand navigation.
        if key.code == KeyCode::Char(';') {
            key.code = KeyCode::Enter;
        }

        // Tab on main hub toggles sidebar focus regardless of which side is active.
        if screen == router::SCREEN_MAIN && key.code == KeyCode::Tab {
            self.state.sidebar_focused = !self.state.sidebar_focused;
            if self.state.sidebar_focused
                && self.state.sidebar_selected >= self.state.processes.len()
            {
                self.state.sidebar_selected = self.state.processes.len().saturating_sub(1);
            }
            return PanelAction::Consumed;
        }
        match screen {
            SCREEN_FILE_PICKER => self.file_browser.handle_event(key, &self.state),
            SCREEN_OUTPUT_PICKER => self.output_picker.handle_event(key, &self.state),
            SCREEN_INPUT_PATH => self.input_panel.handle_event(key, &self.state),
            SCREEN_RUNNING => self.output_panel.handle_event(key, &self.state),
            SCREEN_PREFLIGHT => self.preflight_panel.handle_event(key, &self.state),
            SCREEN_EXIT_CONFIRM => self.exit_confirm.handle_event(key, &self.state),
            x if x == router::SCREEN_SETTINGS => self.settings_panel.handle_event(key, &self.state),
            SCREEN_SETTINGS_HELP => self.settings_help_panel.handle_event(key, &self.state),
            SCREEN_WIZARD => self.wizard_panel.handle_event(key, &self.state),
            SCREEN_ROOT_WARN => self.root_warn_panel.handle_event(key, &self.state),
            SCREEN_BROWSE_RESULTS => self.browse_panel.handle_event(key, &self.state),
            SCREEN_TUTORIAL => self.tutorial_panel.handle_event(key, &self.state),
            SCREEN_LOSSLESS_PROMPT => {
                self.lossless_prompt_panel.handle_event(key, &self.state)
            }
            SCREEN_SYSCHECK => self.syscheck_panel.handle_event(key, &self.state),
            SCREEN_RESUME => self.resume_panel.handle_event(key, &self.state),
            SCREEN_SPLASH => self.splash_panel.handle_event(key, &self.state),
            x if x == router::SCREEN_VIZ => self.viz_panel.handle_event(key, &self.state),
            // Main hub: Tab toggles sidebar focus; when focused, route to sidebar handler.
            x if x == router::SCREEN_MAIN && self.state.sidebar_focused => {
                self.handle_sidebar_key(key)
            }
            _ => {
                let state = &self.state;
                if let Some(panel) = self.panels.get_mut(screen) {
                    panel.handle_event(key, state)
                } else {
                    // Generic fallback for unregistered screens.
                    match key.code {
                        KeyCode::Char('x') => PanelAction::Quit,
                        KeyCode::Char('q') => PanelAction::Home,
                        KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => PanelAction::Back,
                        _ => PanelAction::Ignored,
                    }
                }
            }
        }
    }

    /// Single mutation chokepoint. Every panel-emitted Action and every
    /// App-internal Action (Tick, SetStatus, BackOne) flows through here.
    /// In commit 1 of the reactive-store refactor this is just the renamed
    /// `process_action` plus three new arms; commit 2 will lift the body
    /// into a pure `reduce(&mut AppState, Action) -> Vec<Cmd>`.
    fn dispatch(&mut self, screen: &str, action: Action) {
        match action {
            Action::Navigate(target) => self.handle_navigate(&target),
            Action::Back => self.handle_back(screen),
            Action::Home => {
                // q = go to main menu; from main menu, escalate to exit_confirm.
                if screen == router::SCREEN_MAIN {
                    self.router.navigate(SCREEN_EXIT_CONFIRM);
                } else {
                    self.router.navigate(router::SCREEN_MAIN);
                }
            }
            Action::Quit => {
                // x = exit application, always routes through exit_confirm.
                if screen == SCREEN_EXIT_CONFIRM {
                    self.state.should_quit = true;
                } else {
                    self.router.navigate(SCREEN_EXIT_CONFIRM);
                }
            }
            Action::StatusMessage(msg) | Action::SetStatus(msg) => {
                self.state.set_status(msg);
            }
            Action::SelectPeer(id) => {
                // Typed peer-set/clear (C.2b — replaces the
                // "__select_peer:<id>" StatusMessage sentinel).
                // Shared reducer in AppState keeps the TUI and the
                // GUI bridge strictly identical.
                self.state.apply_select_peer(id);
            }
            Action::SetCfgWorkers(v) => {
                if self.state.apply_cfg_workers(v) {
                    self.state.save_cfg_now();
                }
            }
            Action::SetCfgBackend(v) => {
                if self.state.apply_cfg_backend(v) {
                    self.state.save_cfg_now();
                }
            }
            Action::SetCfgVerification(v) => {
                if self.state.apply_cfg_verification(v) {
                    self.state.save_cfg_now();
                }
            }
            Action::RunOperation { op_id, args } => {
                self.state
                    .set_status(format!("Running: {} {:?}", op_id, args));
            }
            Action::Submit(value) => self.handle_submit(screen, value),
            Action::BackOne => {
                self.router.back();
            }
            Action::Tick => {
                // Reducer-driven tick is reserved for commit 2+. Today the
                // animation counter is bumped directly in tick_panels(); this
                // arm exists so Tick is a valid Action through the chokepoint
                // and tests can dispatch it without panicking.
            }
            Action::OpEvent(ev) => {
                // P3 + C.3c: subprocess events drained by tick_panels are
                // applied here so all OpEvent state mutations flow through
                // the dispatch chokepoint. Dual-write during cohabit:
                //   1. OutputPanel.consume — TUI rendering (existing path)
                //   2. AppState.apply_op_event — mirrors the stream into
                //      AppState so the GUI bridge can surface progress /
                //      log lines via StateSnapshot without an
                //      OutputPanel-equivalent on the Tauri side.
                self.state.apply_op_event(&ev);
                self.output_panel.consume(ev);
            }
            Action::Consumed | Action::Ignored => {}
        }
    }

    fn handle_navigate(&mut self, target: &str) {
        if let Some(stripped) = target.strip_prefix("op:") {
            self.start_op(stripped);
            return;
        }
        if let Some(stripped) = target.strip_prefix("fw:") {
            self.start_firmware(stripped);
            return;
        }
        if let Some(stripped) = target.strip_prefix("launch:") {
            self.start_launcher(stripped);
            return;
        }
        if target == SCREEN_SETTINGS_HELP {
            // Pull the descriptor index from the settings panel (set when user pressed `?`)
            if let Some(idx) = self.settings_panel.take_pending_help() {
                if let Some(d) = self.settings_panel.descriptor(idx) {
                    let cur = (d.get)(self.settings_panel.cfg_ref()).display();
                    self.settings_help_panel
                        .set_target(d.section, d.label, d.dotpath, d.desc, d.help, &cur);
                }
            }
            self.router.navigate(target);
            return;
        }
        self.router.navigate(target);
    }

    fn handle_back(&mut self, screen: &str) {
        if matches!(
            screen,
            SCREEN_FILE_PICKER | SCREEN_INPUT_PATH | SCREEN_OUTPUT_PICKER
        ) {
            self.state.pending_op = None;
        }
        // Preflight outcomes: Enter (confirmed), i/o (edit request),
        // b/Esc (cancel). Capture both flags BEFORE the router.back()
        // below so the post-back stage can route appropriately:
        //   - confirmed → execute_pending after router.back()
        //   - edit Input → push file_picker after router.back(); keep
        //     pending_op alive so the next file_picker submit hits
        //     `pending_needs_preflight` and re-enters this panel
        //   - edit Output → push output_picker after router.back();
        //     same pending_op preservation
        //   - neither → cancel, clear pending_op
        let mut preflight_confirmed = false;
        let mut preflight_edit: Option<super::panels::preflight::EditTarget> = None;
        if screen == SCREEN_PREFLIGHT {
            preflight_confirmed = self.preflight_panel.take_confirmed();
            preflight_edit = self.preflight_panel.take_edit();
            if !preflight_confirmed && preflight_edit.is_none() {
                self.state.pending_op = None;
            }
        }
        if screen == SCREEN_RUNNING {
            self.state.op_handle = None;
            self.state.remote_handle = None;
            // Clear the auto-back timer too — if the user revisits
            // SCREEN_RUNNING later (e.g. via the sidebar's "click a
            // tracked process" navigation), output_panel.is_done() is
            // still true from the prior op, and a stale op_done_at
            // would instantly fire another auto-back, popping them
            // straight off the screen they just opened. Reset on every
            // SCREEN_RUNNING exit so the next visit starts fresh.
            self.state.op_done_at = None;
            self.state.history.mark_complete();
            self.state.launch_state = LaunchState::Idle;
            // Mark the most recent running process as done (no-op if tick already synced).
            let failed = self.output_panel.is_failed();
            let cancelled = self.output_panel.is_cancelled();
            if let Some(p) = self.state.processes.iter_mut().rev().find(|p| !p.done) {
                p.mark_done(failed, cancelled);
            }
        }
        if screen == SCREEN_WIZARD {
            if self.wizard_panel.take_committed() {
                self.state.cfg = super::config::LamQuantConfig::load();
                super::theme::detect(&self.state.cfg.output.color, &self.state.cfg.output.charset);
                self.settings_panel = SettingsPanel::new();
                self.state.set_status("Setup saved. Welcome to LamQuant!");
            } else {
                self.state
                    .set_status("Setup skipped — using defaults. Settings menu can edit.");
            }
        }
        if screen == SCREEN_ROOT_WARN {
            if self.root_warn_panel.take_silence() {
                self.state.cfg.output.allow_root = true;
                let _ = self.state.cfg.save();
                self.state
                    .set_status("Root warning silenced (allow_root=true).");
            } else {
                self.state
                    .set_status("Continuing as root for this session.");
            }
        }
        // SCREEN_SETTINGS: sync handled by `tick_panels` via
        // `settings_panel.take_saved_cfg()` — fires once per successful
        // save and works on every exit path (b, q, x, navigate-away).
        // Resume flow: stage pending_op here, but DEFER execute_pending
        // until AFTER the router.back() below. Otherwise execute_pending
        // pushes SCREEN_RUNNING, then router.back() immediately pops it
        // off — net effect: pressing `r` looked like a no-op. Same
        // pattern as preflight_confirmed handling.
        let mut resume_confirmed = false;
        if screen == SCREEN_RESUME {
            match self.resume_panel.take_action() {
                ResumeAction::Resume => {
                    let op = self.resume_panel.op.clone();
                    let input = self.resume_panel.input.clone();
                    let output = self.resume_panel.output.clone();
                    if let Some(spec) = op_spec(&op) {
                        self.state.pending_op = Some(PendingOp {
                            op_id: op.clone(),
                            spec,
                            input: Some(input.clone()),
                            output: if output.is_empty() {
                                None
                            } else {
                                Some(output.clone())
                            },
                        });
                        self.state
                            .set_status(format!("Resuming {} on {}", op, input));
                        resume_confirmed = true;
                    } else {
                        self.state
                            .set_status(format!("Cannot resume — op {} no longer exists", op));
                        self.state.history.mark_complete();
                    }
                }
                ResumeAction::Discard => {
                    // Discard nukes the .lamquant-staging subdir of
                    // the last-output dir (if one exists) AND clears
                    // the interrupted flag. The staging subdir is
                    // where the canceled encode wrote its partial
                    // outputs (per the lml --batch-staging behavior),
                    // so this is the only place safe to delete from.
                    let staging_root = self.state.history.last_output.clone();
                    let removed = discard_staging(staging_root.as_deref());
                    self.state.history.mark_complete();
                    self.state.set_status(match removed {
                        DiscardResult::Removed(p) => {
                            format!("Discarded — removed staging at {}", p.display())
                        }
                        DiscardResult::NotFound => {
                            "Discarded interrupted run state (no staging dir).".to_string()
                        }
                        DiscardResult::Skipped(reason) => {
                            format!("Discarded run flag only — staging kept: {}", reason)
                        }
                        DiscardResult::Failed(e) => {
                            format!("Discard flag cleared, staging delete failed: {}", e)
                        }
                    });
                }
                ResumeAction::Pending => {
                    // User backed out — keep the interrupted flag for next session.
                }
            }
        }
        self.router.back();
        if preflight_confirmed || resume_confirmed {
            // Fire AFTER pop so SCREEN_RUNNING lands on top of the picker
            // ancestor — `[b]` from running goes back to mode panel, not
            // back to preflight or resume.
            self.execute_pending();
        } else if let Some(target) = preflight_edit {
            // Preflight asked to change a path. Pending op is preserved
            // so the next picker submit re-enters preflight via
            // `pending_needs_preflight`. Re-seed the relevant picker
            // panel and navigate.
            use super::panels::preflight::EditTarget;
            match target {
                EditTarget::Input => {
                    let op_id = self
                        .state
                        .pending_op
                        .as_ref()
                        .map(|p| p.op_id.clone())
                        .unwrap_or_default();
                    self.file_browser =
                        FileBrowserPanel::new(&format!("Select input for: {}", op_id));
                    self.file_browser
                        .set_recent(self.state.history.recent_inputs.clone());
                    self.router.navigate(SCREEN_FILE_PICKER);
                }
                EditTarget::Output => {
                    let (op_id, input) = {
                        let p = self.state.pending_op.as_ref();
                        (
                            p.map(|p| p.op_id.clone()).unwrap_or_default(),
                            p.and_then(|p| p.input.clone()).unwrap_or_default(),
                        )
                    };
                    let default_output = derive_output_path(&op_id, &input);
                    let default_path = std::path::Path::new(&default_output);
                    let default_dir = default_path
                        .parent()
                        .filter(|p| p.is_dir())
                        .map(|p| p.to_path_buf())
                        .or_else(|| std::env::current_dir().ok())
                        .unwrap_or_else(|| std::path::PathBuf::from("."));
                    let default_name = default_path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| format!("output.{}", op_id));
                    self.output_picker.reset_save(
                        &format!("Save output for: {}", op_id),
                        &default_name,
                        Some(&default_dir),
                    );
                    self.router.navigate(SCREEN_OUTPUT_PICKER);
                }
            }
        }
    }

    fn start_op(&mut self, op_id: &str) {
        let Some(spec) = op_spec(op_id) else {
            self.state.set_status(format!("Unknown op: {}", op_id));
            return;
        };
        let pending = PendingOp {
            op_id: op_id.to_string(),
            spec: spec.clone(),
            input: None,
            output: None,
        };
        self.state.pending_op = Some(pending);

        if spec.input {
            self.file_browser = FileBrowserPanel::new(&format!("Select input for: {}", op_id));
            self.file_browser
                .set_recent(self.state.history.recent_inputs.clone());
            self.router.navigate(SCREEN_FILE_PICKER);
        } else {
            self.execute_pending();
        }
    }

    /// Handle key presses when the sidebar process list is focused.
    fn handle_sidebar_key(&mut self, key: crossterm::event::KeyEvent) -> PanelAction {
        let total = self.state.processes.len();
        match key.code {
            KeyCode::Esc | KeyCode::Tab | KeyCode::Char('b') => {
                self.state.sidebar_focused = false;
                PanelAction::Consumed
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if total > 0 && self.state.sidebar_selected > 0 {
                    self.state.sidebar_selected -= 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if total > 0 && self.state.sidebar_selected + 1 < total {
                    self.state.sidebar_selected += 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Enter => {
                if let Some(proc) = self.state.processes.get(self.state.sidebar_selected) {
                    if let Some(screen) = proc.nav_screen.clone() {
                        self.state.sidebar_focused = false;
                        return PanelAction::Navigate(screen);
                    }
                }
                PanelAction::Consumed
            }
            KeyCode::Char('K') => {
                // Kill the selected process if it's the active op. Goes
                // through the unified cancel path so remote ops dispatch
                // SSH cancel, not OpHandle.kill (which would no-op for
                // a remote dispatch since op_handle is None).
                let label = self
                    .state
                    .processes
                    .get(self.state.sidebar_selected)
                    .filter(|p| !p.done)
                    .map(|p| p.label.clone());
                if let Some(label) = label {
                    if self.cancel_active_op() {
                        self.state.set_status(format!("Sent kill to: {}", label));
                    }
                }
                PanelAction::Consumed
            }
            KeyCode::Char('d') => {
                // Dismiss selected done entry (required for failed/cancelled).
                if self
                    .state
                    .processes
                    .get(self.state.sidebar_selected)
                    .map(|p| p.done)
                    .unwrap_or(false)
                {
                    self.state.processes.remove(self.state.sidebar_selected);
                    if self.state.sidebar_selected >= self.state.processes.len() {
                        self.state.sidebar_selected = self.state.processes.len().saturating_sub(1);
                    }
                }
                PanelAction::Consumed
            }
            KeyCode::Char('x') => PanelAction::Quit,
            KeyCode::Char('q') => PanelAction::Home,
            _ => PanelAction::Ignored,
        }
    }

    /// Kill any in-flight op before starting a new one. Without
    /// this, rapid double-press on `[1]`/`[d]` orphans the first
    /// child (no kill, no wait) and the old subprocess's MpscSink
    /// closes mid-write (broken pipe) when the panel's receiver
    /// is dropped. Caller is responsible for invoking this before
    /// reassigning `op_handle` / restarting `output_panel`.
    fn preempt_inflight_op(&mut self) {
        if let Some(mut h) = self.state.op_handle.take() {
            // Best-effort graceful cancel — spawn_blut routes
            // through `blut cancel <id>`; spawn_command sends
            // SIGKILL via Child::kill. Both are idempotent.
            h.kill();
        }
    }

    /// Spawn a preset external launcher (training, eagle, pytest, install scripts).
    ///
    /// Dispatch order:
    ///   1. `blut_launcher` → `runner::spawn_blut` (status.jsonl
    ///      tail + StageEvent translation). Cockpit training
    ///      tier (`cockpit_train_encoder` etc.) lives here.
    ///   2. `launcher` → `runner::spawn_command`. Eagle, pytest,
    ///      viz tools, install scripts.
    ///
    /// `id` is the bare op id (router strips the `launch:`
    /// prefix in `handle_navigate` before calling here).
    fn start_launcher(&mut self, id: &str) {
        // BLUT-tier dispatch first — these route through
        // spawn_blut so the dashboard receives translated
        // StageEvents, not raw stdout Log lines.
        if let Some((recipe, args_json, label)) = lamquant_ops::blut_launcher(id) {
            self.preempt_inflight_op();
            self.state.launch_state = LaunchState::Launching {
                tool: label.to_string(),
                started: std::time::Instant::now(),
            };
            let title = format!("BLUT recipe: {}", label);
            let (tx, rx) = channel();
            self.output_panel.start(title, rx);
            self.output_panel.bell_on_done = self.state.cfg.output.bell_on_done;
            self.state.op_handle = Some(runner::spawn_blut(
                recipe.to_string(),
                args_json.to_string(),
                tx,
            ));
            self.state.processes.push(TrackedProcess::new(
                label,
                "train",
                Some(SCREEN_RUNNING.to_string()),
            ));
            self.router.navigate(SCREEN_RUNNING);
            return;
        }

        let Some((program, args, label)) = super::operations::launcher(id) else {
            self.state.set_status(format!("Unknown launcher: {}", id));
            return;
        };

        // Track viz tool launches for the EEG sidebar.
        if id.starts_with("viz_") {
            self.state.active_viz_tool = Some(label.to_string());
        }

        self.preempt_inflight_op();
        self.state.launch_state = LaunchState::Launching {
            tool: label.to_string(),
            started: std::time::Instant::now(),
        };

        let title = format!("launcher: {}", label);
        let (tx, rx) = channel();
        self.output_panel.start(title, rx);
        self.output_panel.bell_on_done = self.state.cfg.output.bell_on_done;
        // Argv placeholder substitution. Three tokens supported:
        //   $INPUT   — viz file (T5 / ADR 0020)
        //   $WEIGHTS — firmware-export source ckpt (T6 / ADR 0019)
        //   $OUTPUT  — firmware-export destination bundle (T6 / ADR 0019)
        // Each placeholder draws from a dedicated AppState field;
        // a missing selection aborts the launch with a clear status
        // hint rather than spawning the subprocess with a literal
        // "$INPUT" / "$WEIGHTS" / "$OUTPUT" string.
        let owned_args: Vec<String> = {
            let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            if raw.iter().any(|a| a == "$INPUT") {
                let Some(input) = &self.state.viz_selected_input else {
                    self.state.set_status(
                        "Pick a file via [Tab] file picker before launching this viz tool."
                            .to_string(),
                    );
                    self.state.launch_state = LaunchState::Idle;
                    return;
                };
                let input_str = input.display().to_string();
                raw.into_iter()
                    .map(|a| if a == "$INPUT" { input_str.clone() } else { a })
                    .collect()
            } else if raw.iter().any(|a| a == "$WEIGHTS" || a == "$OUTPUT") {
                let Some(weights) = &self.state.fw_weights_input else {
                    self.state.set_status(
                        "Pick a weights checkpoint via [w] before launching firmware export."
                            .to_string(),
                    );
                    self.state.launch_state = LaunchState::Idle;
                    return;
                };
                let Some(output) = &self.state.fw_output_path else {
                    self.state.set_status(
                        "Pick an output path via [o] before launching firmware export."
                            .to_string(),
                    );
                    self.state.launch_state = LaunchState::Idle;
                    return;
                };
                let w = weights.display().to_string();
                let o = output.display().to_string();
                raw.into_iter()
                    .map(|a| match a.as_str() {
                        "$WEIGHTS" => w.clone(),
                        "$OUTPUT" => o.clone(),
                        _ => a,
                    })
                    .collect()
            } else {
                raw
            }
        };
        self.state.op_handle = Some(runner::spawn_command(program.to_string(), owned_args, tx));

        // Register process for sidebar tracking.
        let kind: &'static str = if id.starts_with("viz_") {
            "viz"
        } else if id.starts_with("train_") {
            "train"
        } else {
            "launch"
        };
        self.state.processes.push(TrackedProcess::new(
            label,
            kind,
            Some(SCREEN_RUNNING.to_string()),
        ));

        self.router.navigate(SCREEN_RUNNING);
    }

    fn start_firmware(&mut self, target: &str) {
        // T6 / ADR 0019: route firmware actions through the launcher
        // table so per-target cargo+probe-rs invocations stay in one
        // place. The pre-T6 hardcoded `scripts/firmware/build_*.sh`
        // references pointed at files that didn't exist; the launcher
        // now uses `cargo build --features target-<id>` directly +
        // `probe-rs run` for flash.
        //
        // Target → launcher-id mapping:
        //   list                 → fw_list_devices (probe-rs list)
        //   rp2350 / nrf54l15 /
        //   esp32p4 / stm32n6    → fw_build_<id>
        //   esp32s3              → fw_legacy_esp32s3 (sequester)
        //   flash_<id>           → fw_flash_<id>
        //   export               → fw_export ($WEIGHTS/$OUTPUT)
        //
        // Any verb not in the table falls through to the legacy
        // path-based dispatch so any in-flight firmware-panel code
        // that hasn't been updated still works.
        let launcher_id: Option<&str> = match target {
            "list" => Some("fw_list_devices"),
            "rp2350" => Some("fw_build_rp2350"),
            "nrf54l15" => Some("fw_build_nrf54l15"),
            "esp32p4" => Some("fw_build_esp32p4"),
            "stm32n6" => Some("fw_build_stm32n6"),
            "esp32s3" => Some("fw_legacy_esp32s3"),
            "flash_rp2350" => Some("fw_flash_rp2350"),
            "flash_nrf54l15" => Some("fw_flash_nrf54l15"),
            "flash_esp32p4" => Some("fw_flash_esp32p4"),
            "flash_stm32n6" => Some("fw_flash_stm32n6"),
            "export" => Some("fw_export"),
            _ => None,
        };
        if let Some(id) = launcher_id {
            self.start_launcher(id);
            return;
        }
        // Fallback for any unmapped verb (defensive — should not be
        // reachable from the current panel surface).
        self.state
            .set_status(format!("Unknown firmware target: {}", target));
    }

    /// Returns true if the pending op should pause for the pre-flight
    /// banner. Mirrors the old Python compress flow — only encode-style
    /// ops get the scan + Enter-to-launch gate; verify/info/stats run
    /// straight through.
    fn pending_needs_preflight(&self) -> bool {
        // All four encode flavours get the scan + Enter-to-launch
        // preflight gate. Skipping the gate for `encode_lma` / `
        // encode_lml_siblings` was a regression introduced when those
        // op-ids landed -- users would drop straight into the running
        // dashboard with no chance to confirm paths or see scan
        // numbers, which is the UX gap the audit caught.
        self.state
            .pending_op
            .as_ref()
            .map(|p| {
                matches!(
                    p.op_id.as_str(),
                    "encode" | "encode_neural" | "encode_lma" | "encode_lml_siblings",
                )
            })
            .unwrap_or(false)
    }

    /// Stage the preflight panel from the current pending op and route
    /// to SCREEN_PREFLIGHT. Caller must already have set input/output
    /// on the pending op.
    fn enter_preflight(&mut self) {
        let (op_id, input, output) = {
            let Some(p) = self.state.pending_op.as_ref() else {
                return;
            };
            (
                p.op_id.clone(),
                p.input.clone().unwrap_or_default(),
                p.output.clone(),
            )
        };
        self.preflight_panel
            .prepare(&op_id, &input, output.as_deref(), &self.state.cfg);
        self.router.navigate(SCREEN_PREFLIGHT);
    }

    fn handle_submit(&mut self, screen: &str, value: String) {
        // Submit from codec browse → start `info` op on selected file.
        if screen == SCREEN_BROWSE_RESULTS {
            self.router.back();
            self.state.pending_op = Some(PendingOp {
                op_id: "info".into(),
                spec: op_spec("info").expect("info op exists"),
                input: Some(value.clone()),
                output: None,
            });
            self.state.history.add_input(&value);
            self.execute_pending();
            return;
        }
        if value.is_empty() && screen == SCREEN_INPUT_PATH {
            // Allow empty output → CLI auto-derives. Continue with None output.
            if let Some(op) = self.state.pending_op.as_mut() {
                op.output = None;
            }
            if self.pending_needs_preflight() {
                self.enter_preflight();
            } else {
                self.execute_pending();
            }
            return;
        }

        match screen {
            SCREEN_FILE_PICKER => {
                // Snapshot the bits we need from pending_op before mutating
                // the rest of state — avoids overlapping mutable borrows.
                let (op_id, wants_output) = {
                    let Some(op) = self.state.pending_op.as_mut() else {
                        return;
                    };
                    op.input = Some(value.clone());
                    (op.op_id.clone(), op.spec.output)
                };
                self.state.history.add_input(&value);
                self.state.add_recent_input(&value);

                if wants_output {
                    let default_output = derive_output_path(&op_id, &value);
                    let default_path = std::path::Path::new(&default_output);
                    let default_dir = default_path
                        .parent()
                        .filter(|p| p.is_dir())
                        .map(|p| p.to_path_buf())
                        .or_else(|| std::env::current_dir().ok())
                        .unwrap_or_else(|| std::path::PathBuf::from("."));
                    let default_name = default_path
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| format!("output.{}", op_id));
                    self.output_picker.reset_save(
                        &format!("Save output for: {}", op_id),
                        &default_name,
                        Some(&default_dir),
                    );
                    self.router.navigate(SCREEN_OUTPUT_PICKER);
                } else if self.pending_needs_preflight() {
                    self.enter_preflight();
                } else {
                    self.execute_pending();
                }
            }
            SCREEN_INPUT_PATH | SCREEN_OUTPUT_PICKER => {
                {
                    let Some(op) = self.state.pending_op.as_mut() else {
                        return;
                    };
                    op.output = Some(value.clone());
                }
                self.state.history.add_output(&value);
                self.state.add_recent_output(&value);
                if self.pending_needs_preflight() {
                    self.enter_preflight();
                } else {
                    self.execute_pending();
                }
            }
            _ => {}
        }
    }

    /// Resolve the peer the next op should run on. Override (one-shot)
    /// wins over sticky; override is consumed by `take()`. If override
    /// names a peer that's no longer in peers.json (stale), surface a
    /// status warning and fall through to sticky.
    fn pick_peer_for_op(&mut self) -> Option<Peer> {
        if let Some(id) = self.state.peer_override.take() {
            if let Some(p) = self.state.peers.iter().find(|p| p.id == id).cloned() {
                return Some(p);
            }
            self.state.set_status(format!(
                "Override peer '{}' not configured — falling back",
                id,
            ));
        }
        let id = self.state.selected_peer.clone()?;
        self.state.peers.iter().find(|p| p.id == id).cloned()
    }

    /// Send a cancel signal to whatever op is in flight (local subprocess
    /// or remote SSH). Returns true if a NEW signal was emitted; returns
    /// false if no op or the op was already cancel-requested.
    ///
    /// Local: kill the OpHandle (idempotent via was_killed flag).
    /// Remote: resolve peer first, then call Transport::cancel. If the
    /// peer can't be resolved (orphan handle — e.g. all tracked processes
    /// already done, or peer.id rotated out of peers.json), fall back to
    /// killing the local SSH PID directly via the handle token. The
    /// remote op exits via SIGPIPE when stdout closes, so this is a
    /// correct best-effort even without an SSH round-trip.
    ///
    /// Order matters: peer is resolved BEFORE take() so an orphan-handle
    /// case still triggers the local-PID fallback rather than silently
    /// dropping the handle.
    fn cancel_active_op(&mut self) -> bool {
        if let Some(h) = self.state.op_handle.as_mut() {
            if !h.was_killed() {
                h.kill();
                return true;
            }
            return false;
        }
        if self.state.remote_handle.is_none() {
            return false;
        }

        // Resolve the peer FIRST, while the handle is still in state.
        let peer = self
            .state
            .processes
            .iter()
            .rev()
            .find(|p| !p.done)
            .and_then(|p| p.peer.clone())
            .and_then(|pid| self.state.peers.iter().find(|p| p.id == pid).cloned());

        let Some(handle) = self.state.remote_handle.take() else {
            return false;
        };

        if let Some(peer) = peer {
            let transport = SshTransport::new();
            let _ = transport.cancel(&peer, &handle);
        } else {
            // Orphan-handle fallback: kill the local SSH child directly.
            // The remote process exits via SIGPIPE when stdout closes.
            if let Ok(pid) = handle.token.parse::<u32>() {
                let _ = std::process::Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .status();
            }
        }
        true
    }

    /// Returns true if either a local subprocess OR a remote dispatch is
    /// still considered in-flight. Used by render code to gate "running"
    /// status indicators across all spoke panels.
    fn op_in_flight(&self) -> bool {
        let local = self
            .state
            .op_handle
            .as_ref()
            .map(|h| !h.was_killed())
            .unwrap_or(false);
        let remote = self.state.remote_handle.is_some() && !self.output_panel.is_done();
        local || remote
    }

    fn execute_pending(&mut self) {
        let Some(pending) = self.state.pending_op.take() else {
            return;
        };
        let title = format!(
            "{} — {}",
            pending.op_id,
            pending.input.as_deref().unwrap_or("")
        );
        let label = format!(
            "{} · {}",
            pending.op_id,
            pending
                .input
                .as_deref()
                .map(|p| trim_path(p, 18))
                .unwrap_or_default(),
        );

        self.state.history.mark_running(
            &pending.op_id,
            pending.input.as_deref().unwrap_or(""),
            pending.output.as_deref(),
        );
        self.state.history.save();

        let peer = self.pick_peer_for_op();
        let peer_id = peer.as_ref().map(|p| p.id.clone());

        let dispatched = match peer.as_ref() {
            Some(p) => self.dispatch_op_remote(p, &pending, &title),
            None => {
                self.dispatch_op_local(&pending, &title);
                true
            }
        };
        if !dispatched {
            // Remote dispatch failed; status set by callee. Don't push a
            // tracked process or navigate — leave the user where they were.
            return;
        }

        self.output_panel.bell_on_done = self.state.cfg.output.bell_on_done;

        let mut tp = TrackedProcess::new(label, "op", Some(SCREEN_RUNNING.to_string()));
        if let Some(pid) = peer_id {
            tp = tp.with_peer(pid);
        }
        self.state.processes.push(tp);

        // Pop browse + input screens off the stack, then push running.
        while matches!(
            self.router.current(),
            SCREEN_FILE_PICKER | SCREEN_INPUT_PATH | SCREEN_OUTPUT_PICKER
        ) {
            if !self.router.back() {
                break;
            }
        }
        self.router.navigate(SCREEN_RUNNING);
    }

    fn dispatch_op_local(&mut self, pending: &PendingOp, title: &str) {
        let args = pending
            .spec
            .build_args(pending.input.as_deref(), pending.output.as_deref());
        let (tx, rx) = channel();
        self.output_panel.start(title.to_string(), rx);
        self.state.op_handle = Some(runner::spawn_lml(args, tx));
        self.state.remote_handle = None;
    }

    /// Returns true on successful dispatch; false if staging or dispatch
    /// failed (status message already set on AppState).
    fn dispatch_op_remote(&mut self, peer: &Peer, pending: &PendingOp, title: &str) -> bool {
        let transport = SshTransport::new();

        // Stage input file if the op consumes one. Hash-detect first, rsync fallback.
        let staged_input: Option<String> = match pending.input.as_deref() {
            Some(local_path) => {
                match transport.stage_input(peer, std::path::Path::new(local_path)) {
                    Ok(remote) => Some(remote.0),
                    Err(e) => {
                        self.state
                            .set_status(format!("stage_input → {}: {}", peer.id, e,));
                        return false;
                    }
                }
            }
            None => None,
        };

        let remote_args = pending
            .spec
            .build_args(staged_input.as_deref(), pending.output.as_deref());

        match transport.dispatch(peer, &pending.op_id, &remote_args) {
            Ok((rx, handle)) => {
                self.output_panel.start(title.to_string(), rx);
                self.state.remote_handle = Some(handle);
                self.state.op_handle = None;
                true
            }
            Err(e) => {
                self.state
                    .set_status(format!("dispatch → {}: {}", peer.id, e,));
                false
            }
        }
    }

    fn render(&mut self, f: &mut Frame) {
        let area = f.area();
        let on_main = self.router.current() == router::SCREEN_MAIN;
        let minimal = self.state.cfg.output.minimal_ui || !self.state.cfg.output.show_banner;
        let logo_h: u16 = if on_main && !minimal {
            (LOGO.len() + 3) as u16
        } else {
            0
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(logo_h),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);

        if on_main && !minimal {
            // Banner: [LOGO (Min)] | [desc box (fixed)] | [version box (fixed)]
            // OVERVIEW + BUILD widths match CONTEXT + STATUS sidebar split below
            // (sidebar_w = 68 → 34 each at 50/50 split).
            let banner_cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Min(74),
                    Constraint::Length(34), // OVERVIEW (matches CONTEXT below)
                    Constraint::Length(34), // BUILD    (matches STATUS below)
                ])
                .split(chunks[0]);

            let mut logo_lines: Vec<Line> = LOGO
                .iter()
                .map(|l| Line::from(Span::styled(*l, theme::title())))
                .collect();
            logo_lines.push(Line::from(""));
            logo_lines.push(Line::from(vec![
                Span::raw("       "),
                Span::styled(
                    "Neural EEG Codec  ·  Gen 7.7  ·  AGPL-3.0  ·  OpenHuman.tech",
                    theme::dim(),
                ),
            ]));
            f.render_widget(Paragraph::new(logo_lines), banner_cols[0]);
            self.render_banner_desc(f, banner_cols[1]);
            self.render_banner_info(f, banner_cols[2]);
        }

        let screen = self.router.current().to_string();
        let state = &self.state;
        match screen.as_str() {
            SCREEN_FILE_PICKER => self.file_browser.render(state, f, chunks[1]),
            SCREEN_OUTPUT_PICKER => self.output_picker.render(state, f, chunks[1]),
            SCREEN_INPUT_PATH => self.input_panel.render(state, f, chunks[1]),
            SCREEN_RUNNING => self.output_panel.render(state, f, chunks[1]),
            SCREEN_PREFLIGHT => self.preflight_panel.render(state, f, chunks[1]),
            SCREEN_EXIT_CONFIRM => self.exit_confirm.render(state, f, chunks[1]),
            x if x == router::SCREEN_SETTINGS => self.settings_panel.render(state, f, chunks[1]),
            SCREEN_SETTINGS_HELP => self.settings_help_panel.render(state, f, chunks[1]),
            SCREEN_WIZARD => self.wizard_panel.render(state, f, chunks[1]),
            SCREEN_ROOT_WARN => self.root_warn_panel.render(state, f, chunks[1]),
            SCREEN_BROWSE_RESULTS => self.browse_panel.render(state, f, chunks[1]),
            SCREEN_TUTORIAL => self.tutorial_panel.render(state, f, chunks[1]),
            SCREEN_LOSSLESS_PROMPT => {
                self.lossless_prompt_panel.render(state, f, chunks[1])
            }
            SCREEN_SYSCHECK => self.syscheck_panel.render(state, f, chunks[1]),
            SCREEN_RESUME => self.resume_panel.render(state, f, chunks[1]),
            SCREEN_SPLASH => self.splash_panel.render(state, f, chunks[1]),
            x if x == router::SCREEN_VIZ => self.viz_panel.render(state, f, chunks[1]),
            x if x == router::SCREEN_MAIN => self.render_main_hub_screen(f, chunks[1]),
            other => {
                if let Some(panel) = self.panels.get(other) {
                    panel.render(state, f, chunks[1]);
                } else {
                    let msg = format!("Screen '{}' — not loaded. [b] Back", other);
                    f.render_widget(
                        Paragraph::new(msg).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_style(theme::dim()),
                        ),
                        chunks[1],
                    );
                }
            }
        }

        // Status bar — sectioned with cyan labels, colored ready dot.
        // Snapshot launch_state so we can mutably take_status() inside arms.
        let launch_snapshot = self.state.launch_state.clone();
        let (dot_style, dot_char, ready_text) = match launch_snapshot {
            LaunchState::Idle => {
                let msg = self.state.take_status().unwrap_or_else(|| {
                    if self.router.depth() > 1 {
                        "back".to_string()
                    } else {
                        "ready".to_string()
                    }
                });
                (theme::success(), "●", msg)
            }
            LaunchState::Launching { tool, .. } => {
                let _ = self.state.take_status();
                (theme::warning(), "●", format!("launching {}…", tool))
            }
            LaunchState::Failed(tool) => {
                let _ = self.state.take_status();
                (
                    theme::error(),
                    "●",
                    format!("failed to launch {tool} — Ctrl-C and rebuild"),
                )
            }
        };

        let git = env!("LAMQUANT_GIT_COMMIT");
        // Build left sections: each label is cyan, value is dim, `·` separators.
        let sections: Vec<Span> = vec![
            Span::raw(" "),
            Span::styled("lamquant", theme::key_hint()),
            Span::raw("  "),
            Span::styled(format!("v{}", self.state.version), theme::dim()),
            Span::styled("  ·  ", theme::dim()),
            Span::styled("cli", theme::key_hint()),
            Span::raw("  "),
            Span::styled("v1.0.0", theme::dim()),
            Span::styled("  ·  ", theme::dim()),
            Span::styled("build", theme::key_hint()),
            Span::raw("  "),
            Span::styled(git.to_string(), theme::dim()),
        ];
        let left_len: usize = sections.iter().map(|s| s.content.chars().count()).sum();
        let right_part = format!("{}  {} ", dot_char, ready_text);
        let pad_w = (area.width as usize).saturating_sub(left_len + right_part.chars().count());
        let mut spans = sections;
        spans.push(Span::raw(" ".repeat(pad_w)));
        spans.push(Span::styled(format!("{}  ", dot_char), dot_style));
        spans.push(Span::styled(ready_text, dot_style));
        spans.push(Span::raw(" "));
        f.render_widget(Paragraph::new(Line::from(spans)), chunks[2]);
    }

    /// Banner middle: what LamQuant is — static description box.
    fn render_banner_desc(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim());
        let inner = block.inner(area);
        f.render_widget(block, area);
        let lines = vec![
            Line::from(Span::styled(" OVERVIEW", theme::highlight())),
            Line::from(Span::styled(" Powerful Open Source EEG", theme::normal())),
            Line::from(Span::styled(" Processing Suite", theme::normal())),
            Line::from(Span::styled(" EEG codec + tools.", theme::dim())),
            Line::from(Span::styled(" Encode · validate · train", theme::dim())),
            Line::from(Span::styled(" firmware · visualize.", theme::dim())),
            Line::from(Span::styled(" AGPL-3.0", theme::dim())),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// Banner right: build/cli/profile KV table in a bordered box.
    fn render_banner_info(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim());
        let inner = block.inner(area);
        f.render_widget(block, area);

        let git = env!("LAMQUANT_GIT_COMMIT");
        let key = |k: &str| Span::styled(format!(" {:<11}", k), theme::dim());
        let val = |v: String| Span::styled(v, theme::normal());
        let val_cy = |v: String| Span::styled(v, theme::highlight());

        let is_running = self.op_in_flight();
        let active_label: String = if is_running {
            self.output_panel.op_title().to_string()
        } else {
            self.state
                .processes
                .iter()
                .rev()
                .find(|p| !p.done)
                .map(|p| p.label.clone())
                .unwrap_or_else(|| "idle".to_string())
        };
        let active_style = if active_label == "idle" {
            theme::dim()
        } else {
            theme::highlight()
        };

        let lines = vec![
            Line::from(Span::styled(" BUILD", theme::highlight())),
            Line::from(vec![
                key("version"),
                val_cy(format!("v{}", self.state.version)),
            ]),
            Line::from(vec![key("commit"), val_cy(git.to_string())]),
            Line::from(vec![key("cli"), val_cy("v1.0.0".into())]),
            Line::from(vec![
                key("profile"),
                val(self.state.cfg.instance_name.clone()),
            ]),
            Line::from(vec![
                key("workers"),
                val(self.state.cfg.compute.workers.to_string()),
            ]),
            Line::from(vec![
                key("active"),
                Span::styled(active_label, active_style),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    /// Full main hub screen: dividers + [items | sidebar] + hint.
    fn render_main_hub_screen(&mut self, f: &mut Frame, area: Rect) {
        let sidebar_w: u16 = 68;

        // Compact body: 1 blank + "WORKFLOWS" header + 1 blank + 5 items
        //             + 1 blank + "SYSTEM" header + 1 blank + 3 items + 1 blank = 15
        let body_h: u16 = 15;

        let vc = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),      // top divider
                Constraint::Length(body_h), // items + sidebar
                Constraint::Length(1),      // bottom divider
                Constraint::Length(1),      // hint
            ])
            .split(area);

        // Full-width dividers
        let rule: String = "─".repeat(area.width as usize);
        let rule_line = Line::from(Span::styled(rule, theme::dim()));
        f.render_widget(Paragraph::new(rule_line.clone()), vc[0]);
        f.render_widget(Paragraph::new(rule_line), vc[2]);

        // [items | sidebar]
        let hc = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(30), Constraint::Length(sidebar_w)])
            .split(vc[1]);

        if let Some(panel) = self.panels.get(router::SCREEN_MAIN) {
            panel.render(&self.state, f, hc[0]);
        }
        self.render_sidebar(f, hc[1]);

        // Hint bar — shows workflow hints normally, sidebar hints when focused.
        let hint = if self.state.sidebar_focused {
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[↑↓]", theme::key_hint()),
                Span::raw(" scroll   "),
                Span::styled("[↵]", theme::key_hint()),
                Span::raw(" jump to   "),
                Span::styled("[d]", theme::key_hint()),
                Span::raw(" dismiss   "),
                Span::styled("[K]", theme::key_hint()),
                Span::raw(" kill   "),
                Span::styled("[Tab]", theme::key_hint()),
                Span::raw(" back to workflows"),
            ])
        } else {
            Line::from(vec![
                Span::raw("  "),
                Span::styled("[j/k ↑↓]", theme::key_hint()),
                Span::raw(" navigate   "),
                Span::styled("[↵ Enter ;]", theme::key_hint()),
                Span::raw(" select   "),
                Span::styled("[Tab]", theme::key_hint()),
                Span::raw(" processes   "),
                Span::styled("[h]", theme::key_hint()),
                Span::raw(" help   "),
                Span::styled("[b]", theme::key_hint()),
                Span::raw(" back   "),
                Span::styled("[q]", theme::key_hint()),
                Span::raw(" main menu   "),
                Span::styled("[x]", theme::key_hint()),
                Span::raw(" quit program"),
            ])
        };
        f.render_widget(Paragraph::new(hint), vc[3]);
    }

    /// Right sidebar: CONTEXT (left, full height) | STATUS (right, full height).
    fn render_sidebar(&self, f: &mut Frame, area: Rect) {
        let hc = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        self.render_context_panel(f, hc[0]);
        self.render_sidebar_status(f, hc[1]);
    }

    fn render_sidebar_status(&self, f: &mut Frame, area: Rect) {
        let is_running = self.op_in_flight();
        let any_running =
            !self.state.processes.is_empty() && self.state.processes.iter().any(|p| !p.done);
        let idle = self.state.processes.is_empty();

        let block =
            Block::default()
                .borders(Borders::ALL)
                .border_style(if self.state.sidebar_focused {
                    theme::highlight()
                } else {
                    theme::dim()
                });

        let inner = block.inner(area);
        f.render_widget(block, area);

        let inner_w = inner.width as usize;
        let mut lines: Vec<Line> = Vec::new();

        // Title inside the box.
        let (badge_str, badge_style) = if any_running {
            (" ● LIVE", theme::success())
        } else {
            (" ○ IDLE", theme::dim())
        };
        lines.push(Line::from(vec![
            Span::styled(" STATUS", theme::highlight()),
            Span::styled(badge_str, badge_style),
        ]));
        lines.push(Line::from(""));

        if idle {
            // Mirror context panel idle animation — pulsing dots + label.
            let phase = (self.state.context_tick / 20 % 4) as usize;
            let dots = ["·  ·  ·", " · · · ", "·  ·  ·", "   ·   "];
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  {}", dots[phase]),
                theme::dim(),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  waiting for workflows",
                theme::dim(),
            )));
        } else {
            // ── Config summary (only when something has run / is running) ──
            let workers = self.state.cfg.compute.workers;
            lines.push(Line::from(vec![
                Span::styled(format!("{:<10}", "codec"), theme::dim()),
                Span::styled(
                    self.state.cfg.codec.default_mode.clone(),
                    theme::highlight(),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(format!("{:<10}", "workers"), theme::dim()),
                Span::styled(format!("{workers} / {workers}"), theme::dim()),
            ]));
            if self.state.history.interrupted {
                lines.push(Line::from(Span::styled(
                    "⚠ interrupted run",
                    theme::warning(),
                )));
            }

            // ── Process list ─────────────────────────────────────────────
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("processes", theme::dim())));
            for (i, proc) in self.state.processes.iter().enumerate() {
                let selected = self.state.sidebar_focused && i == self.state.sidebar_selected;
                let (dot, dot_style) = if proc.failed {
                    ("✗", theme::error())
                } else if proc.cancelled {
                    ("⚠", theme::warning())
                } else if proc.done {
                    ("✓", theme::dim())
                } else if is_running {
                    ("⟳", theme::highlight())
                } else {
                    ("○", theme::dim())
                };

                let kind_tag = format!(" {}", proc.kind);
                let label_max = inner_w.saturating_sub(kind_tag.len() + 4);
                let label = if proc.label.chars().count() > label_max {
                    format!(
                        "{}…",
                        truncate_chars(&proc.label, label_max.saturating_sub(1))
                    )
                } else {
                    proc.label.clone()
                };

                let prefix = if selected { "▶ " } else { "  " };
                let mut spans = vec![
                    Span::raw(prefix),
                    Span::styled(dot, dot_style),
                    Span::raw(" "),
                    Span::styled(
                        label,
                        if selected {
                            theme::selected()
                        } else {
                            theme::normal()
                        },
                    ),
                    Span::styled(kind_tag, theme::dim()),
                ];
                if let Some(pid) = &proc.peer {
                    // @peer suffix surfaces remote dispatch in the sidebar so
                    // users can tell which device is doing the work without
                    // having to open the Peers panel.
                    spans.push(Span::styled(format!(" @{}", pid), theme::highlight()));
                }
                lines.push(Line::from(spans));

                // Second row: elapsed (running) or timestamp (failed/cancelled).
                let detail = if proc.failed || proc.cancelled {
                    let ts = proc.done_elapsed_str();
                    let tag = if proc.cancelled {
                        "cancelled"
                    } else {
                        "failed"
                    };
                    format!("    {} {}  [d] dismiss", tag, ts)
                } else {
                    format!("    {}", proc.elapsed_str())
                };
                let detail_style = if proc.failed {
                    theme::error()
                } else if proc.cancelled {
                    theme::warning()
                } else {
                    theme::dim()
                };
                lines.push(Line::from(Span::styled(detail, detail_style)));
            }
        }

        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    /// Always-visible context panel — adapts to what LamQuant is currently doing.
    fn render_context_panel(&self, f: &mut Frame, area: Rect) {
        let t = self.state.context_tick;
        let is_running = self.op_in_flight();
        let op = self.state.history.last_op.as_deref().unwrap_or("");

        // Determine context mode.
        enum Ctx<'a> {
            Encode,
            Decode,
            Train,
            Viz(&'a str),
            Eagle,
            Firmware,
            Idle,
        }
        let ctx = if is_running {
            match op {
                "encode" | "encode_neural" | "encode_lma" | "encode_lml_siblings" => {
                    Ctx::Encode
                }
                "decode" | "decode_neural" => Ctx::Decode,
                s if s.starts_with("train_") => Ctx::Train,
                s if s.starts_with("eagle_") => Ctx::Eagle,
                s if s.starts_with("fw_") || s == "flash" => Ctx::Firmware,
                _ => Ctx::Idle,
            }
        } else if let Some(tool) = &self.state.active_viz_tool {
            Ctx::Viz(tool.as_str())
        } else {
            Ctx::Idle
        };

        let (title_label, title_extra) = match &ctx {
            Ctx::Encode => (" ENCODE ", " compressing… "),
            Ctx::Decode => (" DECODE ", " expanding…   "),
            Ctx::Train => (" TRAIN ", " model fit…   "),
            Ctx::Viz(t) => (" EEG ", *t),
            Ctx::Eagle => (" EAGLE ", " validating…  "),
            Ctx::Firmware => (" FIRMWARE ", " building…    "),
            Ctx::Idle => (" CONTEXT", " ○ IDLE"),
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::dim());

        let inner = block.inner(area);
        f.render_widget(block, area);

        // Encode/Decode: split horizontally — animation left, PieChart right.
        let (main_area, pie_area) =
            if matches!(&ctx, Ctx::Encode | Ctx::Decode) && inner.width >= 40 {
                let hc = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
                    .split(inner);
                (hc[0], Some(hc[1]))
            } else {
                (inner, None)
            };

        let w = main_area.width as usize;
        let h = main_area.height as usize;
        let mut lines: Vec<Line> = Vec::new();

        // Title inside the box.
        lines.push(Line::from(vec![
            Span::styled(title_label, theme::highlight()),
            Span::styled(title_extra, theme::dim()),
        ]));

        // ── Animation body ─────────────────────────────────────────────
        let anim_rows = h.saturating_sub(4); // reserve title row + 2 for progress + elapsed

        match ctx {
            Ctx::Encode => {
                // Blocks on left being fed into small compressed form on right.
                // Animated feed cursor cycles every 6 ticks.
                let cursor = (t / 6 % 4) as usize;
                let src_chars = ['█', '▓', '▒', '░'];
                let src_ch = src_chars[cursor];
                let src_w = (w * 2 / 3).saturating_sub(4);
                let dst_w = w / 4;
                for row in 0..anim_rows.min(4) {
                    let src: String = std::iter::repeat(src_ch).take(src_w).collect();
                    let dst: String = "█".repeat(dst_w);
                    let arrow = if row == anim_rows.min(4) / 2 {
                        " ⟶ "
                    } else {
                        "    "
                    };
                    lines.push(Line::from(vec![
                        Span::styled(src, theme::dim()),
                        Span::styled(arrow, theme::highlight()),
                        Span::styled(dst, theme::highlight()),
                    ]));
                }
                lines.push(Line::from(""));
            }
            Ctx::Decode => {
                // Small compressed block on left expanding to full on right.
                let cursor = (t / 6 % 4) as usize;
                let dst_chars = ['░', '▒', '▓', '█'];
                let dst_ch = dst_chars[cursor];
                let src_w = w / 4;
                let dst_w = (w * 2 / 3).saturating_sub(4);
                for row in 0..anim_rows.min(4) {
                    let src: String = "█".repeat(src_w);
                    let dst: String = std::iter::repeat(dst_ch).take(dst_w).collect();
                    let arrow = if row == anim_rows.min(4) / 2 {
                        " ⟶ "
                    } else {
                        "    "
                    };
                    lines.push(Line::from(vec![
                        Span::styled(src, theme::highlight()),
                        Span::styled(arrow, theme::highlight()),
                        Span::styled(dst, theme::dim()),
                    ]));
                }
                lines.push(Line::from(""));
            }
            Ctx::Train => {
                // Scrolling loss indicator + epoch counter.
                let epoch_str = self.output_panel.progress_msg().unwrap_or("training…");
                let wave_phase = if w > 0 {
                    (t / 4 % w as u64) as usize
                } else {
                    0
                };
                let wave: String = (0..w.min(24))
                    .map(|i| {
                        let v = ((i + wave_phase) % 8) as u8;
                        [' ', ' ', '·', '·', '-', '-', '▄', '▄'][v as usize]
                    })
                    .collect();
                lines.push(Line::from(Span::styled(wave, theme::dim())));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!("  {}", truncate_chars(epoch_str, w.saturating_sub(3))),
                    theme::normal(),
                )));
                lines.push(Line::from(""));
            }
            Ctx::Viz(tool) => {
                // EEG channel bars (existing logic).
                let bar_w = w.saturating_sub(10);
                let channels: &[(&str, ratatui::style::Color, f32)] = &[
                    ("ch01 Fp1", theme::CYAN, 0.93),
                    ("ch08 O3 ", theme::GREEN, 0.79),
                    ("ch15 Pz ", Color::Blue, 0.86),
                    ("ch21 O2 ", Color::Magenta, 0.58),
                ];
                let _ = tool;
                for (label, color, fill) in channels {
                    let filled = (bar_w as f32 * fill) as usize;
                    let bar: String = "█".repeat(filled);
                    lines.push(Line::from(vec![
                        Span::styled(format!("{label} "), theme::dim()),
                        Span::styled(bar, Style::default().fg(*color)),
                    ]));
                }
                lines.push(Line::from(Span::styled(
                    format!("  0s{}now", " ".repeat(w.saturating_sub(8))),
                    theme::dim(),
                )));
            }
            Ctx::Eagle => {
                let spinner = ['◐', '◓', '◑', '◒'][(t / 5 % 4) as usize];
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{} running validation", spinner), theme::dim()),
                ]));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  LQS suite in progress",
                    theme::dim(),
                )));
            }
            Ctx::Firmware => {
                let spinner =
                    ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][(t / 3 % 10) as usize];
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(format!("{} building firmware", spinner), theme::dim()),
                ]));
            }
            Ctx::Idle => {
                // Pulsing dot grid + waiting label. Layout matches STATUS idle:
                // title → blank → blank → dots → blank → "waiting for workflows".
                let phase = (t / 20 % 4) as usize;
                let dots = ["·  ·  ·", " · · · ", "·  ·  ·", "   ·   "];
                lines.push(Line::from(""));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!("  {}", dots[phase]),
                    theme::dim(),
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  waiting for workflows",
                    theme::dim(),
                )));
            }
        }

        // ── Progress bar (always at bottom 2 rows) ─────────────────────
        let pct = self.output_panel.progress_pct();
        let elapsed = if is_running {
            self.state
                .processes
                .iter()
                .rev()
                .find(|p| !p.done)
                .map(|p| p.elapsed_str())
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Fill remaining space with blank lines before progress bar.
        let body_rows = lines.len();
        let target_body = anim_rows;
        for _ in body_rows..target_body {
            lines.push(Line::from(""));
        }

        // Progress bar row.
        let bar_w = w.saturating_sub(8);
        let (bar_line, pct_str) = if let Some(p) = pct {
            let filled = (bar_w as f32 * p) as usize;
            let empty = bar_w.saturating_sub(filled);
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty),);
            let ps = format!("{:3.0}%", p * 100.0);
            (bar, ps)
        } else if is_running && bar_w > 0 {
            // Indeterminate spinner bar.
            let pos = (t / 3 % bar_w as u64) as usize;
            let bar: String = (0..bar_w)
                .map(|i| {
                    if i == pos || i == pos.saturating_sub(1) {
                        '▓'
                    } else if i == pos.saturating_add(1) && i < bar_w {
                        '▒'
                    } else {
                        '░'
                    }
                })
                .collect();
            (bar, "  …".to_string())
        } else {
            (String::new(), String::new())
        };

        if !bar_line.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(bar_line, theme::key_hint()),
                Span::raw(" "),
                Span::styled(pct_str, theme::dim()),
            ]));
            lines.push(Line::from(Span::styled(
                format!("  elapsed  {}", elapsed),
                theme::dim(),
            )));
        }

        f.render_widget(Paragraph::new(lines), main_area);

        // PieChart in right panel for encode/decode contexts.
        if let Some(pie_rect) = pie_area {
            use tui_piechart::{PieChart, PieSlice};
            let slices = vec![
                PieSlice::new("signal", 75.0, theme::CYAN),
                PieSlice::new("metadata", 13.0, theme::GREEN),
                PieSlice::new("overhead", 12.0, ratatui::style::Color::DarkGray),
            ];
            f.render_widget(
                PieChart::new(slices)
                    .show_legend(true)
                    .show_percentages(true),
                pie_rect,
            );
        }
    }
}

fn trim_path(path: &str, max: usize) -> String {
    if path.len() <= max {
        return path.to_string();
    }
    // Show filename; if still too long, truncate with ellipsis.
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    if name.len() <= max {
        return format!("…{}", name);
    }
    format!(
        "…{}",
        &name[name.len().saturating_sub(max.saturating_sub(1))..]
    )
}

fn derive_output_path(op_id: &str, input: &str) -> String {
    use std::path::Path;
    let p = Path::new(input);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let parent = p.parent().and_then(|p| p.to_str()).unwrap_or(".");
    let ext = match op_id {
        // encode_lma writes a single `.lma` archive; encode_lml_siblings
        // writes a directory tree. Pick `.lma` and `_out` respectively
        // so the output picker offers a sane default rather than
        // falling through to the generic `.out` extension.
        "encode" | "encode_neural" => "lml",
        "encode_lma" => "lma",
        "encode_lml_siblings" => "out", // directory; "_out" suffix handled below
        "decode" | "decode_neural" => "raw",
        "archive" => "lma",
        "extract" => "extracted",
        _ => "out",
    };
    if op_id == "encode_lml_siblings" {
        // Directory output -- no extension. Mirror the input stem with
        // an `_out` suffix to avoid colliding with the source dir.
        return if parent.is_empty() {
            format!("{}_out", stem)
        } else {
            format!("{}/{}_out", parent, stem)
        };
    }
    if op_id == "extract" {
        format!("{}/{}_{}", parent, stem, ext)
    } else if parent.is_empty() {
        format!("{}.{}", stem, ext)
    } else {
        format!("{}/{}.{}", parent, stem, ext)
    }
}

/// Take the first `n` characters of `s` (NOT bytes). Safe on any
/// UTF-8 string — byte-slicing at a fixed index panics on multi-byte
/// codepoints (CJK / accented / emoji), so use this for any display
/// truncation of user-controlled paths, op messages, and process labels.
///
/// Postcondition: output char count == min(n, input char count).
fn truncate_chars(s: &str, n: usize) -> String {
    let out: String = s.chars().take(n).collect();
    debug_assert!(out.chars().count() <= n, "truncate_chars exceeded n");
    debug_assert_eq!(
        out.chars().count(),
        n.min(s.chars().count()),
        "truncate_chars must equal min(n, input)",
    );
    out
}

/// Outcome of `discard_staging`. Reported back to the user via the
/// status bar so a missing dir, refused safety check, or filesystem
/// failure are all distinguishable.
enum DiscardResult {
    Removed(PathBuf),
    NotFound,
    Skipped(String),
    Failed(String),
}

/// Name of the staging subdir produced by `lml encode`. Hard-coded
/// here so a typo can't make the TUI delete the wrong directory; the
/// codec side declares the same literal.
const STAGING_DIRNAME: &str = ".lamquant-staging";

/// Safely delete the .lamquant-staging subdir under `last_output`,
/// if one exists. Five gates protect against deleting anything else:
///
///   1. `last_output` must canonicalize (resolves symlinks + verifies
///      existence). A nonexistent parent → skip.
///   2. The candidate path is `<last_output>/.lamquant-staging`.
///      Built by name join — not from any untrusted input.
///   3. `staging.symlink_metadata` must report a directory and NOT a
///      symlink. Refuse to follow symlinks into arbitrary locations.
///   4. `staging.canonicalize().parent()` must equal
///      `last_output.canonicalize()`. Catches `..` traversal and
///      bind-mount tricks that put the staging-named entry elsewhere.
///   5. Walk the staging tree: every entry must be a regular file
///      whose extension is `lml` / `json` / `tmp` / `log`, OR a
///      subdir matching the same constraint recursively (max depth 4
///      so we never iterate forever on a hostile structure). Any
///      foreign content aborts the delete with `Skipped`.
///
/// After all five pass, `fs::remove_dir_all` runs. Returns the
/// canonical path that was removed for the status-bar message.
fn discard_staging(last_output: Option<&str>) -> DiscardResult {
    let Some(out) = last_output else {
        return DiscardResult::NotFound;
    };
    if out.is_empty() {
        return DiscardResult::NotFound;
    }

    let out_path = std::path::PathBuf::from(out);
    let candidate = out_path.join(STAGING_DIRNAME);
    if !candidate.exists() {
        return DiscardResult::NotFound;
    }

    // Gate 1 + 4: both paths must canonicalize, parent of staging
    // must match canonical(last_output).
    let canon_out = match out_path.canonicalize() {
        Ok(p) => p,
        Err(e) => return DiscardResult::Skipped(format!("last_output unresolved: {}", e)),
    };
    let canon_staging = match candidate.canonicalize() {
        Ok(p) => p,
        Err(e) => return DiscardResult::Skipped(format!("staging unresolved: {}", e)),
    };
    let canon_staging_parent = match canon_staging.parent() {
        Some(p) => p,
        None => return DiscardResult::Skipped("staging has no parent".into()),
    };
    if canon_staging_parent != canon_out {
        return DiscardResult::Skipped(format!(
            "staging parent {} != last_output {}",
            canon_staging_parent.display(),
            canon_out.display(),
        ));
    }
    // Gate 2: basename must equal STAGING_DIRNAME.
    let basename_ok = canon_staging
        .file_name()
        .map(|n| n == STAGING_DIRNAME)
        .unwrap_or(false);
    if !basename_ok {
        return DiscardResult::Skipped(format!("staging basename != {}", STAGING_DIRNAME,));
    }
    // Gate 3: refuse if staging is a symlink (the symlink_metadata
    // check distinguishes the link itself from its target).
    match std::fs::symlink_metadata(&candidate) {
        Ok(m) if m.file_type().is_symlink() => {
            return DiscardResult::Skipped("staging is a symlink".into());
        }
        Ok(m) if !m.is_dir() => {
            return DiscardResult::Skipped("staging not a directory".into());
        }
        Err(e) => {
            return DiscardResult::Skipped(format!("staging stat failed: {}", e));
        }
        _ => {}
    }
    // Gate 5: contents allow-list (recursive, depth-bounded).
    if let Err(reason) = staging_contents_allowed(&canon_staging, 0) {
        return DiscardResult::Skipped(reason);
    }

    match std::fs::remove_dir_all(&canon_staging) {
        Ok(()) => DiscardResult::Removed(canon_staging),
        Err(e) => DiscardResult::Failed(format!("{}", e)),
    }
}

/// Recursive content allow-list for the staging dir. Depth is capped
/// to prevent any pathological filesystem from looping the walk.
/// Files outside the whitelist refuse the discard — caller treats
/// that as "skip the delete, keep the dir for manual inspection".
fn staging_contents_allowed(dir: &std::path::Path, depth: u32) -> Result<(), String> {
    const MAX_DEPTH: u32 = 4;
    const ALLOWED_EXT: &[&str] = &["lml", "json", "tmp", "log"];
    if depth > MAX_DEPTH {
        return Err(format!("staging tree depth > {}", MAX_DEPTH));
    }
    let read = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {}", dir.display(), e))?;
    for entry in read {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let p = entry.path();
        let meta = entry
            .file_type()
            .map_err(|e| format!("file_type {}: {}", p.display(), e))?;
        if meta.is_symlink() {
            return Err(format!("staging contains symlink: {}", p.display()));
        }
        if meta.is_dir() {
            staging_contents_allowed(&p, depth + 1)?;
        } else {
            let ext_ok = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| ALLOWED_EXT.iter().any(|a| a.eq_ignore_ascii_case(e)))
                .unwrap_or(false);
            // Also allow `audit.log` style files with no extension is
            // a fallback for any future file we might add.
            let name_audit = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n == "audit.log" || n == ".lamquant_state.json")
                .unwrap_or(false);
            if !ext_ok && !name_audit {
                return Err(format!("staging contains foreign entry: {}", p.display()));
            }
        }
    }
    Ok(())
}

/// Linux-only uid 0 detection via /proc. On other platforms, returns false.
fn running_as_root() -> bool {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                if let Some(uid) = rest.split_whitespace().next() {
                    return uid == "0";
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod app_tests {
    /// app.rs declares local `const SCREEN_*` mirrors of every
    /// `router::SCREEN_*` constant (legacy pattern -- app.rs predates
    /// the router module). Drift between the two means a navigate
    /// target falls through every match arm and silently renders the
    /// home screen. This test locks the new
    /// `SCREEN_LOSSLESS_PROMPT` pair together at compile time.
    /// Extend the assertion when adding new screens.
    #[test]
    fn local_screen_consts_match_router() {
        assert_eq!(
            super::SCREEN_LOSSLESS_PROMPT,
            super::router::SCREEN_LOSSLESS_PROMPT,
            "local SCREEN_LOSSLESS_PROMPT drifted from router::",
        );
    }
}

