//! ASCII art pools for the two decorative slots in the TUI.
//!
//! BANNER_POOL — shown to the right of the LAMQUANT splash text.
//! HUB_POOL    — shown between the two dividers on the main hub, right of
//!               the workflow / system item lists.
//!
//! Config keys (lamquant.toml [output]):
//!   art_banner = "random" | "off" | "<name>"
//!   art_hub    = "random" | "off" | "<name>"

pub struct ArtPiece {
    pub name: &'static str,
    pub rows: &'static [&'static str],
}

pub const OFF: &[&str] = &[];

// ── Banner pool ───────────────────────────────────────────────────────────────

pub const BANNER_POOL: &[ArtPiece] = &[ArtPiece {
    name: "placeholder",
    rows: BANNER_PLACEHOLDER,
}];

const BANNER_PLACEHOLDER: &[&str] = &[
    r"                           ",
    r"  ┌───────────────────┐   ",
    r"  │                   │   ",
    r"  │  [ banner art ]   │   ",
    r"  │                   │   ",
    r"  │ set art_banner in │   ",
    r"  │  lamquant.toml    │   ",
    r"  │                   │   ",
    r"  └───────────────────┘   ",
];

// ── Hub pool ──────────────────────────────────────────────────────────────────

pub const HUB_POOL: &[ArtPiece] = &[
    ArtPiece {
        name: "chip",
        rows: CHIP,
    },
    ArtPiece {
        name: "lmq1_die",
        rows: LMQ1_DIE,
    },
    ArtPiece {
        name: "eeg_cap",
        rows: EEG_CAP,
    },
    ArtPiece {
        name: "waveform",
        rows: WAVEFORM,
    },
    ArtPiece {
        name: "nodes",
        rows: NODES,
    },
];

const CHIP: &[&str] = &[
    r"       │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │",
    r"   ┌───┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┐",
    r" ──┤ ╔══════════════════════════════════════════════╗ ├──",
    r" ──┤ ║  ┌─────────┐   ┌──────────────┐   ┌────────┐ ║ ├──",
    r" ──┤ ║  │ HAZARD3 │═══│  ░NEURAL░░   │═══│  SRAM  │ ║ ├──",
    r" ──┤ ║  │  RV32   │   │  ▓ TNN-L5 ▓  │   │ 520 KB │ ║ ├──",
    r" ──┤ ║  │ 150 MHz │═══│  ░░ rANS ░░  │═══│ ▓▓▓▓▓▓ │ ║ ├──",
    r" ──┤ ║  └────┬────┘   └───────┬──────┘   └───┬────┘ ║ ├──",
    r" ──┤ ║       ╚═══════════╤════╧═══════════════╝     ║ ├──",
    r" ──┤ ║         ┌─────┐ ┌─┴─┐ ┌─────┐ ┌────────┐     ║ ├──",
    r" ──┤ ║         │ DMA │ │ADC│ │ PIO │ │ FLASH  │     ║ ├──",
    r" ──┤ ║         └─────┘ └───┘ └─────┘ │ 16 MB  │     ║ ├──",
    r" ──┤ ║       LMQ-1 · QFN-56          │  XIP   │     ║ ├──",
    r" ──┤ ╚═══════════════════════════════└────────┘═════╝ ├──",
    r"   └───┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┘",
    r"       │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │",
];

const LMQ1_DIE: &[&str] = &[
    r"       │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │",
    r"   ┌───┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┐",
    r"   │ ╔══════════════════════════════════════════════════════════╗ │",
    r" ──┤ ║  ┌────────────┐  ┌────────────────┐  ┌────────────────┐  ║ ├──",
    r" ──┤ ║  │  HAZARD3   │  │  NEURAL CODEC  │  │   FSQ ENCODER  │  ║ ├──",
    r" ──┤ ║  │  RV32IMAC  │──│  ░░░TNN-L5░░░  │──│   L=5 · 2.26x  │  ║ ├──",
    r" ──┤ ║  │  150 MHz   │  │  ▓▓▓▓▓▓▓▓▓▓▓▓  │  │   ████ rANS    │  ║ ├──",
    r" ──┤ ║  └─────┬──────┘  └────────┬───────┘  └────────┬───────┘  ║ ├──",
    r" ──┤ ║        │                  │                   │          ║ ├──",
    r" ──┤ ║   ╔════╧══════════════════╧═══════════════════╧═════╗    ║ ├──",
    r" ──┤ ║   ║         AHB-LITE  ·  64-bit  ·  300 MB/s        ║    ║ ├──",
    r" ──┤ ║   ╚═══╤══════════╤══════════╤═════════╤═════════════╝    ║ ├──",
    r" ──┤ ║       │          │          │         │                  ║ ├──",
    r" ──┤ ║  ┌────┴─────┐ ┌──┴───┐  ┌───┴────┐ ┌──┴─────┐  ┌───────┐  ║ ├──",
    r" ──┤ ║  │  SRAM    │ │ DMA  │  │  PIO   │ │ USB-HS │  │ FLASH │  ║ ├──",
    r" ──┤ ║  │  520 KB  │ │ 16ch │  │  8sm   │ │  480M  │  │ 16 MB │  ║ ├──",
    r" ──┤ ║  │ ▓▓▓▓▓▓▓▓ │ └──────┘  └────────┘ └────────┘  │  XIP  │  ║ ├──",
    r" ──┤ ║  └──────────┘                                  └───────┘  ║ ├──",
    r" ──┤ ║                                                           ║ ├──",
    r" ──┤ ║   ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐    ║ ├──",
    r" ──┤ ║   │ ADC ×4 │  │ I²S    │  │ SPI ×2 │  │  CRYPTO  ◇   │    ║ ├──",
    r" ──┤ ║   │ 12 bit │  │ 24 bit │  │ 50 MHz │  │  AES-128/SHA │    ║ ├──",
    r" ──┤ ║   └────────┘  └────────┘  └────────┘  └──────────────┘    ║ ├──",
    r" ──┤ ║                                                           ║ ├──",
    r" ──┤ ║       LMQ-1   ·   die 4.2 × 4.2 mm   ·   QFN-56           ║ ├──",
    r" ──┤ ╚══════════════════════════════════════════════════════════╝ ├──",
    r"   └───┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬───┘",
    r"       │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │",
    r"       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20",
    r"",
    r"       ░ TNN weights (FLASH-resident, XIP)   ▓ active SRAM region",
];

const EEG_CAP: &[&str] = &[
    r"                      ",
    r"     ○   ●   ○   ●    ",
    r"  ○    ○   ○   ○    ○ ",
    r" ○  ●    ○   ○    ●  ○",
    r"  ○   ○   ●   ○   ○   ",
    r" ○  ○   ○   ○   ○  ○  ",
    r"  ○   ○   ●   ○   ○   ",
    r" ○  ●    ○   ○    ●  ○",
    r"  ○    ○   ○   ○    ○ ",
    r"     ○   ●   ○   ●    ",
    r"                      ",
    r"   10-20 electrode    ",
];

const WAVEFORM: &[&str] = &[
    r"  ┌────────────────┐  ",
    r"  │  /\/\    /\/\  │  ",
    r"  │ /    \  /    \ │  ",
    r"  │/      \/      \│  ",
    r"  │ ch01           │  ",
    r"  │    /\/\/\/\/\  │  ",
    r"  │   /          \ │  ",
    r"  │  /   ch02     \│  ",
    r"  │                │  ",
    r"  │  · · · ch32 ·  │  ",
    r"  └────────────────┘  ",
    r"    256 Hz · lml      ",
];

const NODES: &[&str] = &[
    r"                      ",
    r"   ●───────●───────●  ",
    r"   │ ╲   ╱ │ ╲   ╱ │  ",
    r"   │   ╲╱  │   ╲╱  │  ",
    r"   │   ╱╲  │   ╱╲  │  ",
    r"   │ ╱   ╲ │ ╱   ╲ │  ",
    r"   ●───────●───────●  ",
    r"   │ ╲   ╱ │ ╲   ╱ │  ",
    r"   │   ╲╱  │   ╲╱  │  ",
    r"   ●───────●───────●  ",
    r"                      ",
    r"   LMQ encoder graph  ",
];

// ── Selection ─────────────────────────────────────────────────────────────────

/// Return the art rows for `name` from `pool`.
/// "random" → random pool entry each call.
/// "off" or empty → empty slice (no art rendered).
/// Any other string → find by name, fall back to random.
pub fn pick(name: &str, pool: &[ArtPiece]) -> &'static [&'static str] {
    if pool.is_empty() {
        return OFF;
    }
    match name {
        "off" | "" => OFF,
        "random" => pool[random_idx(pool.len())].rows,
        n => pool
            .iter()
            .find(|a| a.name == n)
            .map(|a| a.rows)
            .unwrap_or_else(|| pool[random_idx(pool.len())].rows),
    }
}

fn random_idx(n: usize) -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as usize)
        .unwrap_or(0)
        % n
}
