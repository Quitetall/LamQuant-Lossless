//! Main interactive menu — matches Python lamquant.py TUI.

use super::codec;
use super::style::{self, clear, clear_full, colors, dot, hrule, repeat};
use std::io::{self, BufRead, Write};

const LOGO: &str = r#"
 ██╗      █████╗ ███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ██║     ██╔══██╗████╗ ████║██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
 ██║     ███████║██╔████╔██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
 ██║     ██╔══██║██║╚██╔╝██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
 ███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
 ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
"#;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const W: usize = 72;

fn print_banner(first: bool) {
    let c = colors();
    if first {
        println!("{}{}{}", c.cyn, LOGO, c.rst);
        println!(
            "       {}Neural EEG Codec {} Gen 7.7 {} OpenHuman Technologies{}",
            c.dim,
            dot(),
            dot(),
            c.rst
        );
    } else {
        println!(
            "\n  {}OpenHuman LamQuant{}  {}Gen 7.7 {} v{} {} lml{}",
            c.bld,
            c.rst,
            c.dim,
            dot(),
            VERSION,
            dot(),
            c.rst
        );
    }
    println!("\n  {}{}{}", c.dim, repeat(hrule(), W), c.rst);
}

fn print_main_menu() {
    let c = colors();

    println!("\n    {}{}WORKFLOWS{}\n", c.bld, c.cyn, c.rst);
    let workflows = [
        ("1", "LML Lossless", "Compress / Decompress / Verify"),
        ("2", "Codec Hub", "Browse, inspect, verify files"),
        ("3", "Batch Encode", "Encode entire directories"),
        ("4", "Archive (LMA)", "Pack / unpack LMA archives"),
    ];
    for (k, name, desc) in &workflows {
        println!("  [{}]  {:<26}  {}{}{}", k, name, c.dim, desc, c.rst);
    }

    println!("\n    {}{}TOOLS{}\n", c.bld, c.cyn, c.rst);
    let tools = [
        ("v", "Verify", "CRC-32 integrity check"),
        ("i", "Info", "File metadata"),
        ("s", "Stats", "Per-channel signal statistics"),
        ("b", "Benchmark", "Encode/decode speed test"),
    ];
    for (k, name, desc) in &tools {
        println!("  [{}]  {:<26}  {}{}{}", k, name, c.dim, desc, c.rst);
    }

    println!(
        "\n  [{}h{}] Help    [{}q{}] Quit",
        c.dim, c.rst, c.dim, c.rst
    );

    let info = format!("v{} {} lml", VERSION, dot());
    let ready = format!("{}Ready{}", c.grn, c.rst);
    println!("\n  {}{}{}", c.dim, repeat(hrule(), W), c.rst);
    let pad = W.saturating_sub(info.len() + 5 + 4);
    println!("  {}{}{}{}{}", c.cyn, info, c.rst, " ".repeat(pad), ready);
}

fn read_key() -> String {
    print!("  > ");
    io::stdout().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input).unwrap_or(0);
    input.trim().to_lowercase()
}

fn pause() {
    let c = colors();
    print!("\n  {}Press Enter to return...{}", c.dim, c.rst);
    io::stdout().flush().unwrap_or(());
    let mut buf = String::new();
    let _ = io::stdin().lock().read_line(&mut buf);
}

fn print_help() {
    let c = colors();
    clear();
    println!("\n  {}LamQuant LML — Lossless EEG Codec{}\n", c.bld, c.rst);
    println!("  {}Commands:{}", c.bld, c.rst);
    println!("    lml encode <file.edf> -o <out.lml> --verify");
    println!("    lml decode <file.lml> -o <dir/>");
    println!("    lml verify <file.lml>");
    println!("    lml info <file.lml>");
    println!("    lml stats <file.lml>");
    println!("    lml bench <file.lml>");
    println!("    lml archive <dir/> -o <archive.lma>");
    println!("    lml extract <archive.lma> -o <dir/>");
    println!();
    println!("  {}Interactive mode:{}", c.bld, c.rst);
    println!("    lml              (no arguments → this menu)");
    println!();
    println!("  {}Format:{}", c.bld, c.rst);
    println!("    LML1 v1.0 — bit-exact EDF roundtrip");
    println!("    Le Gall 5/3 lifting DWT + LPC order-8 + Golomb-Rice");
    println!("    Typical CR: 2.0-3.5:1");
    println!();
    println!("  {}Website:{} openhuman.tech", c.bld, c.rst);
    pause();
}

/// Entry point for interactive TUI (no CLI args).
pub fn run_interactive() -> i32 {
    clear_full();
    let mut first = true;

    loop {
        if first {
            clear_full();
        } else {
            clear();
        }
        print_banner(first);
        first = false;
        print_main_menu();

        let choice = read_key();
        match choice.as_str() {
            "1" => codec::run_lossless_menu(),
            "2" => codec::run_codec_hub(),
            "3" => codec::run_batch_encode(),
            "4" => codec::run_archive_menu(),
            "v" => codec::run_verify_prompt(),
            "i" => codec::run_info_prompt(),
            "s" => codec::run_stats_prompt(),
            "b" => codec::run_bench_prompt(),
            "h" => print_help(),
            "q" | "x" => {
                let c = colors();
                println!(
                    "\n  {}{}{} {}Session ended{}",
                    c.grn,
                    style::check(),
                    c.rst,
                    c.bld,
                    c.rst
                );
                println!(
                    "  {}v{} {} lml {} openhuman.tech{}\n",
                    c.dim,
                    VERSION,
                    dot(),
                    dot(),
                    c.rst
                );
                return 0;
            }
            _ => {}
        }
    }
}
