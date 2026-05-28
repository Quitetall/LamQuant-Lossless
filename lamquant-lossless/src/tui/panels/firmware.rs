//! Firmware Hub — toolchain status, targets, and build actions.
//!
//! Replaces the linear MenuPanel at SCREEN_FIRMWARE with a three-section
//! layout: Toolchain Status (auto-detected), Targets (1–3), Actions
//! (e/c/m/f/z + b/q/x).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::process::Command;

use crate::tui::panel::{Panel, PanelAction};
use crate::tui::router;
use crate::tui::state::AppState;
use crate::tui::theme;

const FIRMWARE_LOGO: &[&str] = &[
    "  ███████╗██╗██████╗ ███╗   ███╗██╗    ██╗ █████╗ ██████╗ ███████╗",
    "  ██╔════╝██║██╔══██╗████╗ ████║██║    ██║██╔══██╗██╔══██╗██╔════╝",
    "  █████╗  ██║██████╔╝██╔████╔██║██║ █╗ ██║███████║██████╔╝█████╗  ",
    "  ██╔══╝  ██║██╔══██╗██║╚██╔╝██║██║███╗██║██╔══██║██╔══██╗██╔══╝  ",
    "  ██║     ██║██║  ██║██║ ╚═╝ ██║╚███╔███╔╝██║  ██║██║  ██║███████╗",
    "  ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝",
];

#[derive(Clone)]
struct Tool {
    label: &'static str,
    detail: String,
    ok: bool,
}

pub struct FirmwarePanel {
    id: String,
    tools: Vec<Tool>,
    /// Currently-selected target (0=RP2350, 1=ESP32-S3, 2=ESP32-P4).
    target: usize,
}

impl FirmwarePanel {
    pub fn new() -> Self {
        Self {
            id: "firmware".to_string(),
            tools: detect_tools(),
            target: 0,
        }
    }
    pub fn refresh(&mut self) {
        self.tools = detect_tools();
    }
}

impl Default for FirmwarePanel {
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

fn detect_tools() -> Vec<Tool> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let cmake_ok = cmd_ok("cmake", &["--version"]);
    let pico_path = std::env::var("PICO_SDK_PATH").ok();
    let pico_ok = pico_path
        .as_deref()
        .map(std::path::Path::new)
        .map(|p| p.is_dir())
        .unwrap_or(false);
    let riscv_ok = cmd_ok("riscv32-elf-gcc", &["--version"])
        || cmd_ok("riscv-none-elf-gcc", &["--version"])
        || cmd_ok("riscv32-unknown-elf-gcc", &["--version"]);
    let idf_path = std::env::var("IDF_PATH").ok().or_else(|| {
        let p = cwd.join("sdk/esp-idf");
        if p.is_dir() {
            Some(p.display().to_string())
        } else {
            None
        }
    });
    let xtensa_ok = cmd_ok("xtensa-esp32s3-elf-gcc", &["--version"]);
    let elf_path =
        cwd.join("lamquant-firmware/target/thumbv8m.main-none-eabihf/release/lamquant-firmware");
    let elf_size = std::fs::metadata(&elf_path).ok().map(|m| m.len());

    vec![
        Tool {
            label: "CMake",
            detail: if cmake_ok {
                "found".into()
            } else {
                "not found".into()
            },
            ok: cmake_ok,
        },
        Tool {
            label: "Pico SDK",
            detail: pico_path.unwrap_or_else(|| "not found".into()),
            ok: pico_ok,
        },
        Tool {
            label: "RISC-V GCC",
            detail: if riscv_ok {
                "found".into()
            } else {
                "not found".into()
            },
            ok: riscv_ok,
        },
        Tool {
            label: "ESP-IDF",
            detail: idf_path.clone().unwrap_or_else(|| "not found".into()),
            ok: idf_path.is_some(),
        },
        Tool {
            label: "Xtensa GCC",
            detail: if xtensa_ok {
                "found".into()
            } else {
                "not found (for ESP32-S3)".into()
            },
            ok: xtensa_ok,
        },
        Tool {
            label: "lamquant.elf",
            detail: elf_size
                .map(|n| format!("{} bytes", group_thousands(n)))
                .unwrap_or_else(|| "not built".into()),
            ok: elf_size.is_some(),
        },
    ]
}

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

fn bordered(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(format!(" {} ", title), theme::highlight()))
}

impl Panel for FirmwarePanel {
    fn id(&self) -> &str {
        &self.id
    }
    fn title(&self) -> &str {
        "Firmware Hub"
    }

    fn render(&self, _state: &AppState, f: &mut Frame, area: Rect) {
        let logo_h: u16 = (FIRMWARE_LOGO.len() + 2) as u16; // banner + subtitle + blank
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(logo_h),
                Constraint::Length(8), // toolchain
                Constraint::Length(1),
                Constraint::Length(8), // targets (3 × 2-line entries + borders)
                Constraint::Length(1),
                Constraint::Length(9), // actions (5 × 1 + footer + borders)
                Constraint::Min(0),
            ])
            .split(area);

        // Banner
        let mut banner: Vec<Line> = FIRMWARE_LOGO
            .iter()
            .map(|l| Line::from(Span::styled(*l, theme::title())))
            .collect();
        banner.push(Line::from(""));
        banner.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("Firmware Hub", theme::highlight()),
            Span::styled("  ·  Export  ·  Build  ·  Flash", theme::dim()),
        ]));
        f.render_widget(Paragraph::new(banner), chunks[0]);

        // Toolchain Status
        let tc_block = bordered("Toolchain Status");
        let tc_inner = tc_block.inner(chunks[1]);
        f.render_widget(tc_block, chunks[1]);
        let tc_lines: Vec<Line> = self
            .tools
            .iter()
            .map(|t| {
                let (mark, mark_style) = if t.ok {
                    ("✓", theme::success())
                } else {
                    ("✗", theme::error())
                };
                Line::from(vec![
                    Span::raw("   "),
                    Span::styled(mark, mark_style),
                    Span::raw("  "),
                    Span::styled(format!("{:<14}", t.label), theme::normal()),
                    Span::styled(t.detail.clone(), theme::dim()),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(tc_lines), tc_inner);

        // Targets
        let tg_block = bordered("Targets");
        let tg_inner = tg_block.inner(chunks[3]);
        f.render_widget(tg_block, chunks[3]);
        let target_rows: [(&str, &str, &str, &str); 3] = [
            (
                "1",
                "RP2350 Hazard3",
                "RISC-V, 150 MHz, 520 KB SRAM",
                "Pico 2 — production target, ternary encoder",
            ),
            (
                "2",
                "ESP32-S3",
                "Xtensa, 240 MHz, 512 KB SRAM",
                "BLE 5.0, Wi-Fi, onboard ML accelerator",
            ),
            (
                "3",
                "ESP32-P4",
                "RISC-V, 400 MHz, 768 KB SRAM",
                "Next-gen, MIPI CSI, hardware AI",
            ),
        ];
        let mut tg_lines: Vec<Line> = Vec::new();
        for (i, (k, name, specs, blurb)) in target_rows.iter().enumerate() {
            let sel = i == self.target;
            let marker = if sel {
                Span::styled(" > ", Style::default().fg(Color::Magenta))
            } else {
                Span::raw("   ")
            };
            tg_lines.push(Line::from(vec![
                marker,
                Span::styled(format!("[{}]", k), theme::key_hint()),
                Span::raw("  "),
                Span::styled(
                    format!("{:<18}", name),
                    if sel {
                        theme::highlight()
                    } else {
                        theme::normal()
                    },
                ),
                Span::styled(specs.to_string(), theme::dim()),
            ]));
            tg_lines.push(Line::from(vec![
                Span::raw("        "),
                Span::styled(blurb.to_string(), theme::dim()),
            ]));
        }
        f.render_widget(Paragraph::new(tg_lines), tg_inner);

        // Actions
        let ac_block = bordered("Actions");
        let ac_inner = ac_block.inner(chunks[5]);
        f.render_widget(ac_block, chunks[5]);
        let actions = [
            ("e", "Export weights", "Model → C headers for firmware"),
            ("c", "Configure", "cmake -DPICO_PLATFORM=rp2350-riscv"),
            ("m", "Build", "make -j$(nproc)"),
            ("f", "Flash", "picotool load / esptool.py flash"),
            ("z", "Size report", "SRAM/flash usage breakdown"),
        ];
        let mut ac_lines: Vec<Line> = actions
            .iter()
            .map(|(k, l, d)| {
                Line::from(vec![
                    Span::raw("   "),
                    Span::styled(format!("[{}]", k), theme::key_hint()),
                    Span::raw("  "),
                    Span::styled(format!("{:<18}", l), theme::normal()),
                    Span::styled(d.to_string(), theme::dim()),
                ])
            })
            .collect();
        ac_lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled("[b]", theme::key_hint()),
            Span::styled(" Back   ", theme::dim()),
            Span::styled("[q]", theme::key_hint()),
            Span::styled(" Main menu   ", theme::dim()),
            Span::styled("[x]", theme::key_hint()),
            Span::styled(" Exit", theme::dim()),
        ]));
        f.render_widget(Paragraph::new(ac_lines), ac_inner);
    }

    fn handle_event(&mut self, event: KeyEvent, _state: &AppState) -> PanelAction {
        match event.code {
            KeyCode::Char('1') => {
                self.target = 0;
                PanelAction::StatusMessage("Target → RP2350 Hazard3".into())
            }
            KeyCode::Char('2') => {
                self.target = 1;
                PanelAction::StatusMessage("Target → ESP32-S3".into())
            }
            KeyCode::Char('3') => {
                self.target = 2;
                PanelAction::StatusMessage("Target → ESP32-P4".into())
            }
            KeyCode::Char('e') => PanelAction::Navigate(router::LAUNCH_FW_EXPORT.to_string()),
            KeyCode::Char('c') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('m') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('f') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('z') => PanelAction::Navigate(router::SCREEN_COMING_V11.into()),
            KeyCode::Char('r') => {
                self.refresh();
                PanelAction::StatusMessage("Re-detected toolchain.".into())
            }
            KeyCode::Char('b') | KeyCode::Esc | KeyCode::Backspace => PanelAction::Back,
            KeyCode::Char('q') => PanelAction::Home,
            KeyCode::Char('x') => PanelAction::Quit,
            KeyCode::Char('h') | KeyCode::Char('?') => {
                PanelAction::Navigate(router::SCREEN_HELP.to_string())
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.target > 0 {
                    self.target -= 1;
                }
                PanelAction::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.target < 2 {
                    self.target += 1;
                }
                PanelAction::Consumed
            }
            _ => PanelAction::Ignored,
        }
    }
}
