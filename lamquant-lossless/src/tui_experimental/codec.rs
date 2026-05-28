//! Codec operation screens — interactive wrappers for encode/decode/verify/info.

use super::style::{clear, colors, hrule, repeat};
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::Command;

const W: usize = 72;

fn read_path(prompt: &str) -> Option<String> {
    let _c = colors();
    print!("  {}: ", prompt);
    io::stdout().flush().unwrap_or(());
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input).unwrap_or(0);
    let trimmed = input.trim().to_string();
    if trimmed.is_empty() || trimmed == "b" || trimmed == "q" {
        None
    } else {
        Some(trimmed)
    }
}

fn pause() {
    let c = colors();
    print!("\n  {}Press Enter to return...{}", c.dim, c.rst);
    io::stdout().flush().unwrap_or(());
    let mut buf = String::new();
    let _ = io::stdin().lock().read_line(&mut buf);
}

fn run_lml_command(args: &[&str]) {
    let exe = std::env::current_exe().unwrap_or_default();
    let output = Command::new(&exe).args(args).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !stdout.is_empty() {
                print!("{}", stdout);
            }
            if !stderr.is_empty() {
                eprint!("{}", stderr);
            }
        }
        Err(e) => {
            let c = colors();
            eprintln!(
                "  {}{}{}  Failed to run: {}",
                c.red,
                super::style::cross(),
                c.rst,
                e
            );
        }
    }
}

// ── LML Lossless Menu ──────────────────────────────────────────────

pub fn run_lossless_menu() {
    let c = colors();
    loop {
        clear();
        println!("\n  {}LML Lossless{}", c.bld, c.rst);
        println!("  {}Bit-exact EDF compression{}\n", c.dim, c.rst);
        println!("  {}{}{}", c.dim, repeat(hrule(), W), c.rst);

        println!(
            "\n  [{}1{}]  {}Compress{}           EDF → .lml",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}2{}]  {}Decompress{}         .lml → raw signal",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}3{}]  {}Verify{}             CRC-32 integrity",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}4{}]  {}Info{}               File metadata",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}5{}]  {}Stats{}              Per-channel statistics",
            c.bld, c.rst, c.bld, c.rst
        );
        println!("\n  [{}b{}] Back", c.dim, c.rst);

        print!("\n  > ");
        io::stdout().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).unwrap_or(0);

        match input.trim() {
            "1" => run_compress_prompt(),
            "2" => run_decompress_prompt(),
            "3" => run_verify_prompt(),
            "4" => run_info_prompt(),
            "5" => run_stats_prompt(),
            "b" | "q" => return,
            _ => {}
        }
    }
}

// ── Individual prompts ──────────────────────────────────────────────

pub fn run_compress_prompt() {
    let c = colors();
    clear();
    println!("\n  {}Compress{} — EDF → LML\n", c.bld, c.rst);

    let input = match read_path("Input (EDF file or directory)") {
        Some(p) => p,
        None => return,
    };

    let output = match read_path("Output (.lml file or directory)") {
        Some(p) => p,
        None => return,
    };

    println!();
    if Path::new(&input).is_dir() {
        run_lml_command(&["encode", &input, "-o", &output, "-r", "--verify"]);
    } else {
        run_lml_command(&["encode", &input, "-o", &output, "--verify"]);
    }
    pause();
}

pub fn run_decompress_prompt() {
    let c = colors();
    clear();
    println!("\n  {}Decompress{} — LML → raw signal\n", c.bld, c.rst);

    let input = match read_path("Input (.lml file or directory)") {
        Some(p) => p,
        None => return,
    };

    let output = match read_path("Output directory") {
        Some(p) => p,
        None => return,
    };

    println!();
    if Path::new(&input).is_dir() {
        run_lml_command(&["decode", &input, "-o", &output, "-r"]);
    } else {
        run_lml_command(&["decode", &input, "-o", &output]);
    }
    pause();
}

pub fn run_verify_prompt() {
    let c = colors();
    clear();
    println!("\n  {}Verify{} — CRC-32 integrity check\n", c.bld, c.rst);

    let input = match read_path("File or directory to verify") {
        Some(p) => p,
        None => return,
    };

    println!();
    if Path::new(&input).is_dir() {
        run_lml_command(&["verify", &input, "-r"]);
    } else {
        run_lml_command(&["verify", &input]);
    }
    pause();
}

pub fn run_info_prompt() {
    let c = colors();
    clear();
    println!("\n  {}Info{} — File metadata\n", c.bld, c.rst);

    let input = match read_path("File path") {
        Some(p) => p,
        None => return,
    };

    println!();
    run_lml_command(&["info", &input]);
    pause();
}

pub fn run_stats_prompt() {
    let c = colors();
    clear();
    println!(
        "\n  {}Stats{} — Per-channel signal statistics\n",
        c.bld, c.rst
    );

    let input = match read_path("File path") {
        Some(p) => p,
        None => return,
    };

    println!();
    run_lml_command(&["stats", &input]);
    pause();
}

pub fn run_bench_prompt() {
    let c = colors();
    clear();
    println!(
        "\n  {}Benchmark{} — Encode/decode speed test\n",
        c.bld, c.rst
    );

    let input = match read_path("File to benchmark") {
        Some(p) => p,
        None => return,
    };

    println!();
    run_lml_command(&["bench", &input]);
    pause();
}

// ── Codec Hub ───────────────────────────────────────────────────────

pub fn run_codec_hub() {
    let c = colors();
    loop {
        clear();
        println!("\n  {}╔═╗╔═╗╦═╗╔═╗╔═╗{}", c.cyn, c.rst);
        println!("  {}║  ║ ║║ ║╠═ ║  {}", c.cyn, c.rst);
        println!("  {}╚═╝╚═╝╩═╝╚═╝╚═╝{}", c.cyn, c.rst);
        println!("  {}Codec Hub{}\n", c.dim, c.rst);
        println!("  {}{}{}", c.dim, repeat(hrule(), W), c.rst);

        println!(
            "\n  [{}1{}]  {}Compress{}     EDF → .lml",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}2{}]  {}Decompress{}   .lml → raw",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}3{}]  {}Verify{}       CRC-32 check",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}4{}]  {}Info{}         Metadata",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}5{}]  {}Stats{}        Per-channel",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}6{}]  {}Benchmark{}    Speed test",
            c.bld, c.rst, c.bld, c.rst
        );
        println!("\n  [{}b{}] Back", c.dim, c.rst);

        print!("\n  > ");
        io::stdout().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).unwrap_or(0);

        match input.trim() {
            "1" => run_compress_prompt(),
            "2" => run_decompress_prompt(),
            "3" => run_verify_prompt(),
            "4" => run_info_prompt(),
            "5" => run_stats_prompt(),
            "6" => run_bench_prompt(),
            "b" | "q" => return,
            _ => {}
        }
    }
}

// ── Batch Encode ────────────────────────────────────────────────────

pub fn run_batch_encode() {
    let c = colors();
    clear();
    println!(
        "\n  {}Batch Encode{} — Encode entire directory\n",
        c.bld, c.rst
    );

    let input = match read_path("Input directory") {
        Some(p) => p,
        None => return,
    };

    let output = match read_path("Output directory") {
        Some(p) => p,
        None => return,
    };

    println!();
    run_lml_command(&["encode", &input, "-o", &output, "-r", "--verify"]);
    pause();
}

// ── Archive Menu ────────────────────────────────────────────────────

pub fn run_archive_menu() {
    let c = colors();
    loop {
        clear();
        println!("\n  {}LMA Archives{}\n", c.bld, c.rst);
        println!("  {}{}{}", c.dim, repeat(hrule(), W), c.rst);

        println!(
            "\n  [{}1{}]  {}Pack{}       Directory → .lma",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}2{}]  {}Extract{}    .lma → Directory",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}3{}]  {}List{}       Show archive contents",
            c.bld, c.rst, c.bld, c.rst
        );
        println!(
            "  [{}4{}]  {}Verify{}     Check archive integrity",
            c.bld, c.rst, c.bld, c.rst
        );
        println!("\n  [{}b{}] Back", c.dim, c.rst);

        print!("\n  > ");
        io::stdout().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().lock().read_line(&mut input).unwrap_or(0);

        match input.trim() {
            "1" => {
                clear();
                println!("\n  {}Pack Archive{}\n", c.bld, c.rst);
                if let Some(dir) = read_path("Directory to archive") {
                    if let Some(out) = read_path("Output .lma path") {
                        println!();
                        run_lml_command(&["archive", &dir, "-o", &out]);
                        pause();
                    }
                }
            }
            "2" => {
                clear();
                println!("\n  {}Extract Archive{}\n", c.bld, c.rst);
                if let Some(archive) = read_path("Archive path (.lma)") {
                    if let Some(out) = read_path("Output directory") {
                        println!();
                        run_lml_command(&["extract", &archive, "-o", &out]);
                        pause();
                    }
                }
            }
            "3" => {
                clear();
                println!("\n  {}List Archive{}\n", c.bld, c.rst);
                if let Some(archive) = read_path("Archive path (.lma)") {
                    println!();
                    run_lml_command(&["list-archive", &archive]);
                    pause();
                }
            }
            "4" => {
                clear();
                println!("\n  {}Verify Archive{}\n", c.bld, c.rst);
                if let Some(archive) = read_path("Archive path (.lma)") {
                    println!();
                    run_lml_command(&["verify-archive", &archive]);
                    pause();
                }
            }
            "b" | "q" => return,
            _ => {}
        }
    }
}
