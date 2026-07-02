#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::unnecessary_sort_by,
    clippy::too_many_arguments,
    clippy::modulo_one
)]
//! lml — LamQuant Lossless EEG codec CLI.

use clap::{Parser, Subcommand};
use lamquant_core::deployment::LosslessMode;
use lamquant_core::{container, edf, lma, lml, tui};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "lml",
    version,
    about = "LML — Lossless EEG Codec\n\nRun with no arguments for interactive mode.",
    after_help = "\
COMMON WORKFLOWS:
  lml encode foo.edf -o out/foo.lma       # encode (default = per-recording LMA)
  lml extract foo.lma -o restored/        # unpack everything byte-for-byte
  lml ls foo.lma --tree                   # browse archive contents
  lml cat foo.lma recording.tse           # extract single entry to stdout
  lml verify foo.lma                      # archive integrity (auto-dispatches)
  lml info foo.lma                        # tree listing (auto-dispatches)
  lml roundtrip foo.edf                   # paranoid bit-exact verification
  lml encode -r dir/ -o out/              # batch encode (per-recording LMAs)

DOCUMENTATION:
  Full CLI reference: docs/CLI_REFERENCE.md
  Feature inventory:  docs/FEATURES.md
  Wire format spec:   docs/lml-format-v1.md
  Problem statement:  docs/PROBLEM.md
  FAQ:                docs/FAQ.md

PRINCIPLE:
  No byte ever lost by default. Data loss is opt-in via `--no-bundle` /
  `--bare-lml` paired with `--i-understand-data-loss`. See `lml encode
  --help` for the loud warning paragraph.

REPORT BUGS:
  https://github.com/Quitetall/LamQuant/issues"
)]
struct Cli {
    /// Emit one JSON line per OpEvent to stdout instead of pretty progress.
    /// Used by Tauri GUI and Python TUI to consume the same wire format
    /// (specs/op-events.schema.json). When set, all status output goes to
    /// stderr; stdout is reserved for OpEvent JSON lines.
    #[arg(long, global = true)]
    emit_json_events: bool,

    /// Suppress all log output (errors still print). Mutually exclusive
    /// with `-v`. Equivalent to `RUST_LOG=off`.
    #[arg(short = 'q', long, global = true, conflicts_with = "verbose")]
    quiet: bool,

    /// Increase log verbosity. `-v` enables INFO, `-vv` DEBUG, `-vvv` TRACE.
    /// Without flags, the default filter is `lamquant_core=warn,lml=info`.
    /// `RUST_LOG` env var overrides both.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Control ANSI colour in log output. `auto` (default) enables
    /// colour when stderr is a TTY, respects NO_COLOR + CLICOLOR_FORCE.
    /// `always` forces colour; `never` disables. Standard env vars
    /// (NO_COLOR set to any value → disabled; CLICOLOR_FORCE=1 →
    /// forced) still override `auto`.
    #[arg(long, global = true, value_enum, default_value = "auto")]
    color: ColorChoice,

    /// Compute backend (host only): `desktop` (default) uses rayon
    /// per-channel + AVX2 inner loops -- the perf path. `firmware`
    /// uses the scalar serial code path the MCU build also uses --
    /// useful for debugging, firmware-bench parity, or
    /// reproducible-build verification. Output bytes are identical
    /// across backends (locked by
    /// `tests/byte_equal_backends.rs`); only wall-clock differs.
    #[arg(long, global = true, value_enum)]
    backend: Option<BackendChoice>,

    #[command(subcommand)]
    command: Option<Commands>,
}

/// `--backend` CLI value. Mirrors `ComputeBackend` variants exposed
/// on host. The CLI uses its own enum because `clap::ValueEnum`
/// requires the type to be in the binary crate (not a library).
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BackendChoice {
    /// Reference scalar path. Slower; matches MCU build behaviour
    /// byte-for-byte. Use for debugging or conformance checks.
    Firmware,
    /// Host rayon + AVX2 perf path (default). Byte-identical output
    /// to `firmware`.
    Desktop,
}

impl BackendChoice {
    /// Map to the library's `ComputeBackend`. Called from main()
    /// to set the process-wide backend selector before any encode.
    fn to_lib(self) -> lamquant_core::backend::ComputeBackend {
        match self {
            BackendChoice::Firmware => lamquant_core::backend::ComputeBackend::Firmware,
            BackendChoice::Desktop => lamquant_core::backend::ComputeBackend::Desktop,
        }
    }
}

/// `--color` parameter values. `clap::ValueEnum`-derived for stable
/// CLI parsing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ColorChoice {
    /// Detect TTY + env vars (NO_COLOR, CLICOLOR_FORCE).
    Auto,
    /// Force ANSI escape sequences even if stderr isn't a TTY.
    Always,
    /// Never emit ANSI escapes.
    Never,
}

impl ColorChoice {
    /// Resolve to a concrete bool (true = colour on) after consulting
    /// stderr-isatty + env vars. Bible R32 — single deterministic
    /// resolver, no scattered "is colour on?" checks.
    fn resolve(self) -> bool {
        use std::io::IsTerminal;
        // NO_COLOR (any value) is the strongest "no colour" signal —
        // no-color.org standard. Honoured even when --color=always.
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        match self {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => {
                // CLICOLOR_FORCE=1 forces colour on non-TTYs (rarely
                // useful, but standard).
                if std::env::var("CLICOLOR_FORCE").as_deref() == Ok("1") {
                    return true;
                }
                std::io::stderr().is_terminal()
            }
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Encode EDF/BDF/BrainVision/CNT/DICOM/EEGLAB/raw to per-recording `.lma`
    ///
    /// Default mode produces one `.lma` archive per recording containing the
    /// compressed `.lml` signal plus the original source bytes plus every
    /// sibling annotation file (TUH `.tse`, `.csv_bi`, `.lbl_bi`,
    /// `_summary.txt`, etc.) via the LML → zstd → store cascade. No byte
    /// can be silently dropped: every input file produces exactly one LMA
    /// entry.
    ///
    /// Bare `.lml` output is available via `--no-bundle` / `--bare-lml`
    /// but prints a 20-line stderr warning every invocation. Suppress
    /// only with the explicit `--i-understand-data-loss` co-flag.
    #[command(after_help = "\
EXAMPLES:
  # Default: per-recording .lma with sibling preservation
  lml encode recording.edf -o out/recording.lma

  # Recursive batch encode of an entire corpus
  lml encode -r /data/tueg/ -o /backup/tueg/

  # Bare .lml output (with loud data-loss warning)
  lml encode recording.edf -o out/ --no-bundle --i-understand-data-loss

  # Cross-validate: encode then decode-compare against source
  lml encode recording.edf -o out/recording.lma --verify --cross-validate

OUTPUT MODES:
  default (no flag)   per-recording `.lma` -- recommended (zero-loss invariant)
  --no-bundle         bare `.lml` + mirror-copied siblings (data-loss warning)
  --bare-lml          alias for --no-bundle (more discoverable name)
  --lma               legacy: pack the entire corpus into one big `.lma`

SOURCE FORMATS:
  EDF / EDF+C / EDF+D, BDF / BDF+, BrainVision (.vhdr / .vmrk / .eeg),
  NeuroScan CNT, custom raw + sidecar JSON, DICOM Waveform (--features dicom),
  EEGLAB (.set + .fdt + sidecar JSON)")]
    Encode {
        /// Input EDF file or directory
        input: PathBuf,
        /// Output LML file or directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Verify roundtrip after encoding
        #[arg(long)]
        verify: bool,
        /// Cross-validate: decode output and compare SHA-256 against original
        #[arg(long)]
        cross_validate: bool,
        /// Strip N least-significant bits (0 = lossless)
        #[arg(long, default_value = "0")]
        noise_bits: u8,
        /// Track-2 bounded-MAE near-lossless: guarantee max|orig-recon| <= δ
        /// (ADR 0051). Single-file → bare `.lml` via closed-loop DPCM.
        /// Mutually exclusive with `--noise-bits`.
        #[arg(
            long = "max-error",
            value_name = "DELTA",
            conflicts_with = "noise_bits"
        )]
        max_error: Option<u64>,
        /// Track-2 target-BPS rate-controlled lossy: minimize distortion s.t.
        /// bits-per-sample <= X (ADR 0051, the H.BWC WP tier). Single-file →
        /// `.lml`. Mutually exclusive with --noise-bits / --max-error.
        #[arg(
            long = "target-bps",
            value_name = "BPS",
            conflicts_with = "noise_bits",
            conflicts_with = "max_error"
        )]
        target_bps: Option<f64>,
        /// Explicit lossless deployment mode. Omitted = MCU-safe integer mode.
        /// `basestation` requires `--features experimental_basestation`.
        #[arg(long = "lossless-mode", value_name = "MODE", value_parser = ["mcu", "basestation"])]
        lossless_mode: Option<String>,
        /// Samples per compression window
        #[arg(long, default_value = "2500")]
        window_size: usize,
        /// Number of parallel threads (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        threads: usize,
        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,
        /// Skip files whose output already exists (for resuming interrupted runs)
        #[arg(long)]
        skip_existing: bool,
        /// v1.2 I — glob pattern of files to INCLUDE (multiple `--include`
        /// flags accepted). Applied during the input walk; only files
        /// whose relative path matches at least one pattern are encoded.
        /// Default = include everything. Patterns follow tar-style /
        /// gitignore-style globbing (`*.edf`, `**/sub-*/eeg/*.edf`,
        /// `recording_[0-9]*.edf`).
        #[arg(long = "include", value_name = "GLOB")]
        include_globs: Vec<String>,
        /// v1.2 I — glob pattern of files to EXCLUDE (multiple
        /// `--exclude` flags accepted). Applied AFTER `--include`;
        /// matches override. Each excluded file emits a one-line
        /// stderr notice; loss is never silent. Pair with
        /// `--i-understand-data-loss` to suppress the loud per-file
        /// warning paragraph.
        #[arg(long = "exclude", value_name = "GLOB")]
        exclude_globs: Vec<String>,
        /// Scan inputs and print estimated sizes/time without writing anything
        #[arg(long)]
        dry_run: bool,
        /// Legacy "corpus-wide single archive" mode: pack every encode
        /// output into one `.lma` archive at the `-o` path. Mutually
        /// exclusive with `--no-bundle`. Without either flag, the
        /// default is **per-EDF `.lma`** — each input EDF emits one
        /// archive bundling its `.lml` plus every sibling sidecar.
        #[arg(long, conflicts_with = "no_bundle")]
        lma: bool,
        /// Opt out of per-EDF `.lma` bundling and emit a bare `.lml`
        /// per EDF *plus* a mirror-copy of every sibling sidecar in
        /// the output directory. The sidecars are not compressed but
        /// they are preserved — the encoder never silently drops them
        /// regardless of mode. Use this when downstream tooling reads
        /// raw `.lml` and the storage cost of un-bundled sidecars is
        /// acceptable. **Prints a loud data-loss warning unless paired
        /// with `--i-understand-data-loss`.** `--bare-lml` is the
        /// recommended alias.
        #[arg(
            long = "no-bundle",
            alias = "bare-lml",
            alias = "mirror",
            conflicts_with = "lma"
        )]
        no_bundle: bool,
        /// Whole-directory LML+siblings mode. Walks the input
        /// directory tree, encodes every EDF/BDF to per-file `.lml`,
        /// and COPIES every non-EEG file verbatim into the mirrored
        /// output tree (no archive container, no zstd on metadata).
        /// Preserves the source structure 1:1. Mutually exclusive
        /// with `--lma` and `--no-bundle`. Routes to
        /// `lamquant_core::lma::pack_lml_with_siblings`.
        ///
        /// Differs from `--no-bundle` in scope: `--no-bundle` mirrors
        /// only stem-bound sidecars next to each EDF; `--lml-siblings`
        /// preserves the entire source tree (every sibling, every
        /// sub-directory, every loose file). Use this when downstream
        /// tooling expects to see the original tree layout.
        #[arg(
            long = "lml-siblings",
            conflicts_with = "lma",
            conflicts_with = "no_bundle"
        )]
        lml_siblings: bool,
        /// Silence the `--no-bundle` / `--bare-lml` data-loss warning.
        /// The flag is required to suppress the loud reminder that
        /// bare-LML output drops the per-recording `.lma` envelope,
        /// breaking sibling-preservation invariants if the encoded
        /// output is moved away from the source directory. Has no
        /// effect when default per-EDF `.lma` mode is active.
        #[arg(long = "i-understand-data-loss")]
        i_understand_data_loss: bool,
        /// LPC mode controlling the per-subband speed/CR trade-off:
        ///   * adaptive — AIC/MDL byte-cost order search (best CR)
        ///   * fixed    — legacy [3,3,6,8] schedule (fastest)
        ///   * anytime  — fixed first, upgrade to adaptive if budget
        ///     allows (default; equivalent to adaptive on batch CLI
        ///     which has no deadline)
        #[arg(long = "lpc-mode", default_value = "auto", value_parser = ["auto", "adaptive", "fixed", "anytime"])]
        lpc_mode: String,
        /// Abort the batch on first failure. Default = continue
        /// (Phase 1.7 `--continue-on-error` is implicit; failures are
        /// logged and the batch summary reports the count).
        #[arg(long)]
        fail_fast: bool,
        /// Explicit alias for the implicit default. No-op; documents
        /// the per-file error-tolerant behaviour for scripts.
        #[arg(long = "continue-on-error", conflicts_with = "fail_fast")]
        continue_on_error: bool,
    },
    /// Decode LML file(s) to raw signal (int32 LE binary)
    Decode {
        /// Input LML file or directory
        input: PathBuf,
        /// Output file or directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,
        /// Skip files whose output already exists
        #[arg(long)]
        skip_existing: bool,
        /// Number of parallel threads (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        threads: usize,
        /// Reconstruct a full byte-identical EDF/BDF file (header +
        /// data records + trailing) instead of the default raw-int32
        /// sample dump. Required for paranoid bit-exact roundtrip
        /// verification (`lml roundtrip`).
        #[arg(long)]
        to_edf: bool,
        /// Phase 3.2 — partial-channel decode. Comma-separated zero-
        /// based indices into the LML's channel list (e.g. `0,2,4`).
        /// Output is channel-major int32 LE in sort+dedup'd index order.
        /// Refused alongside --to-edf (EDF reconstruction needs every
        /// channel slot). Requires the file carry an LMLFOOT1 seek
        /// table — legacy files error explicitly.
        #[arg(long, value_name = "CSV")]
        channels: Option<String>,
        /// Phase 3.3 — partial-time decode. Inclusive-start, exclusive-
        /// end sample-index range `START:END` (e.g. `0:5000`). Output
        /// covers only the requested sample range. Past-EOF end clamps;
        /// past-EOF start errors. Refused alongside --to-edf.
        #[arg(long, value_name = "START:END")]
        time_range: Option<String>,
        /// Abort the batch on first failure. Default = continue.
        #[arg(long)]
        fail_fast: bool,
        /// Explicit alias for the default behaviour. No-op; documents
        /// the per-file error-tolerant batch path for scripts.
        #[arg(long = "continue-on-error", conflicts_with = "fail_fast")]
        continue_on_error: bool,
    },
    /// Show LML file information
    Info {
        /// Input LML file
        input: PathBuf,
    },
    /// Verify LML file integrity (CRC-32 all windows). Auto-dispatches
    /// to the archive verifier on `LMA1` magic so `lml verify foo.lma`
    /// just works.
    Verify {
        /// Input LML file(s) or directory
        input: PathBuf,
        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,
        /// When the input is an `.lma`, print the auditable per-step
        /// verification chain instead of the compact summary. No-op
        /// on `.lml` inputs (the LML verify path is single-step).
        #[arg(long)]
        explain: bool,
    },
    /// Paranoid full-roundtrip bit-exact verification on EDF/BDF files.
    /// For each input file: encodes to a tempfile.lml, then compares
    /// FOUR SHA-256 slices between original and recovered:
    ///   1. signal samples (channel data)
    ///   2. raw_header (full EDF header bytes)
    ///   3. non_eeg channel data (annotations + non-EEG signals)
    ///   4. trailing_data (partial-record bytes at file tail)
    ///
    /// All four must match exactly. Failure modes are reported per-file
    /// in a structured JSON record. Designed for clinical-grade
    /// verification where any drift is unacceptable.
    Roundtrip {
        /// Input EDF/BDF file or directory
        input: PathBuf,
        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,
        /// Write JSON report to this path (default: stdout)
        #[arg(short = 'o', long)]
        report: Option<PathBuf>,
        /// Bail on first mismatch instead of scanning the whole input
        #[arg(long)]
        fail_fast: bool,
        /// Worker thread count (0 = rayon default = CPU count)
        #[arg(short = 'j', long, default_value = "0")]
        parallel: usize,
    },
    /// Verify a manifest.lml.json: check file existence, size, and SHA-256
    VerifyManifest {
        /// Path to manifest.lml.json
        manifest: PathBuf,
    },
    /// Show per-channel signal statistics for LML file(s)
    Stats {
        /// Input LML file or directory
        input: PathBuf,
        /// Recurse into subdirectories
        #[arg(short, long)]
        recursive: bool,
    },
    /// Export LML to CSV, NPY, or raw format
    Export {
        /// Input LML file
        input: PathBuf,
        /// Output file path (or output directory for `bids`)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format: csv, npy, raw, mat, bids
        #[arg(short, long, default_value = "csv")]
        format: String,
        /// Export metadata deployment mode. Omitted = preserve archive metadata or MCU.
        #[arg(long = "lossless-mode", value_name = "MODE", value_parser = ["mcu", "basestation"])]
        lossless_mode: Option<String>,
    },
    /// Losslessly shrink (or restore) a native NWB/HDF5 file in place using the
    /// LML H5Z filter — only integer datasets are recoded; structure, metadata,
    /// compound electrode tables, and object references are preserved by HDF5.
    /// (Built with `--features nwb`; drives the system `h5repack`.)
    Nwb {
        #[command(subcommand)]
        cmd: NwbCmd,
    },
    /// Compare two LML files sample-by-sample
    Diff {
        /// First LML file
        a: PathBuf,
        /// Second LML file
        b: PathBuf,
    },
    /// Recover valid windows from a damaged LML file
    Recover {
        /// Damaged input LML file
        input: PathBuf,
        /// Recovered output LML file
        output: PathBuf,
    },
    /// Benchmark encode/decode speed
    Bench {
        /// Input EDF file
        input: PathBuf,
    },
    /// Archive a directory into a single .lma file
    Archive {
        /// Input directory to archive
        input: PathBuf,
        /// Output .lma file path
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Zstd compression level for sidecars (1-22, default 9)
        #[arg(long, default_value = "9")]
        zstd_level: i32,
    },
    /// Extract an .lma archive to a directory (byte-exact roundtrip)
    ///
    /// Recovers every byte of every entry: signal `.lml`, original
    /// source-format bytes (`.edf` / `.vhdr+.vmrk+.eeg` / `.dcm` /
    /// `.set+.fdt` / `.cnt` / `.raw+sidecar`), and every sibling
    /// annotation file. SHA-256 verified per entry by default.
    #[command(after_help = "\
EXAMPLES:
  # Standard extract into a directory
  lml extract recording.lma -o restored/

  # Skip per-entry SHA-256 check (faster, only for trusted archives)
  lml extract recording.lma -o restored/ --no-verify

NOTES:
  * Recovers exact-name files (e.g. `recording.set` with original MAT
    struct intact for EEGLAB sources).
  * Path-traversal protected: rejects `..` / absolute / Windows device
    name entries.
  * mtime + Unix mode preserved on extracted entries.")]
    Extract {
        /// Input .lma file
        input: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Verify SHA-256 of each extracted file against the manifest.
        /// Default ON for clinical-grade dataloss safety: archive
        /// corruption (bit-flips, partial writes, hardware errors)
        /// would otherwise extract silently as garbage. Pass
        /// `--no-verify` to skip the SHA pass on trusted archives if
        /// extraction speed is critical (rare).
        #[arg(long, default_value_t = true, overrides_with = "no_verify")]
        verify: bool,
        /// Opt out of the SHA-256 extraction check (default is on).
        #[arg(long)]
        no_verify: bool,
    },
    /// List contents of an .lma archive
    ListArchive {
        /// Input .lma file
        input: PathBuf,
    },
    /// Split an `.lma` archive into fixed-size volumes (v1.2 V)
    ///
    /// Produces `<archive>.001`, `<archive>.002`, … each ≤ `--size`
    /// bytes. Pure byte-stream split; no LMA wire-format change.
    /// Pair with `volume-assemble` to reverse.
    ///
    /// Use for cloud / email / removable-media uploads with size limits.
    #[command(after_help = "\
EXAMPLES:
  # Split into ~100 MB volumes (deletes the original by default)
  lml volume-split recording.lma --size 100M

  # Same but keep the original alongside the volumes
  lml volume-split recording.lma --size 100M --keep")]
    VolumeSplit {
        /// Input `.lma` file
        input: PathBuf,
        /// Volume size with K/M/G suffix (e.g. `100M`, `4G`).
        #[arg(long, short = 's')]
        size: String,
        /// Keep the original `.lma` after splitting. Default = delete.
        #[arg(long)]
        keep: bool,
        /// Overwrite existing volume files.
        #[arg(long)]
        force: bool,
    },
    /// Reassemble volume files into a single `.lma` (v1.2 V)
    ///
    /// Auto-detects the volume set by globbing `<base>.NNN` for
    /// sequential 3-digit suffixes. The volume set must be complete
    /// (no gaps in the numbering); on missing volume → typed error.
    #[command(after_help = "\
EXAMPLES:
  # Reassemble all foo.lma.* into foo.lma
  lml volume-assemble foo.lma.001 -o foo.lma

  # Force overwrite of an existing foo.lma
  lml volume-assemble foo.lma.001 -o foo.lma --force")]
    VolumeAssemble {
        /// Any single volume file (must end in `.NNN` for some N).
        /// The other volumes are auto-discovered alongside.
        input: PathBuf,
        /// Output `.lma` path. Defaults to the input stripped of the
        /// `.NNN` suffix.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Overwrite existing output.
        #[arg(long)]
        force: bool,
    },
    /// Verify .lma archive integrity without extracting
    ///
    /// Walks every check the verifier performs:
    /// archive-wide SHA-256 over content+footer, manifest decompress
    /// + parse, per-entry SHA-256 against the manifest. Default
    /// output is a compact "OK n/n entries" summary; `--explain`
    /// prints the auditable per-step readout.
    #[command(after_help = "\
EXAMPLES:
  # Compact summary (default)
  lml verify-archive recording.lma

  # Auditable per-step readout
  lml verify-archive recording.lma --explain")]
    VerifyArchive {
        /// Input .lma file
        input: PathBuf,
        /// Print the per-step verification chain instead of a summary.
        /// Shows archive SHA, manifest decompress byte counts, per-
        /// entry SHA against the manifest, decompression ratios, and
        /// the cumulative elapsed time. Auditable readout for the
        /// "no black box" contract.
        #[arg(long)]
        explain: bool,
    },
    /// Browse `.lma` archive contents (flat or `--tree`)
    ///
    /// Foundation for OS file-manager integration (Ark on KDE,
    /// Nautilus on GNOME, Finder on macOS, Explorer on Windows).
    /// Plugins shell out to `lml ls` to enumerate archive entries
    /// without unpacking.
    #[command(after_help = "\
EXAMPLES:
  # One entry per line (tar tf style; pipes to grep/awk/fzf cleanly)
  lml ls recording.lma

  # Tree-style with sizes, compression method, sha256 prefix
  lml ls recording.lma --tree

OUTPUT SHAPE (--tree):
  recording.lma (3.3 KB, 8 entries, 1.97x CR)
  ├── recording.lml             1.8 KB  lml      sha256:7f4a...
  ├── recording.edf.zst         312 B   zstd     sha256:a2cd...
  ├── recording.tse              58 B   store    sha256:bc8e...
  └── recording_summary.txt     144 B   store    sha256:de77...")]
    Ls {
        /// Input `.lma` file
        input: PathBuf,
        /// Tree-style output with sizes + method + sha256 prefix
        #[arg(long, conflicts_with = "long")]
        tree: bool,
        /// Long-form machine-readable listing: one entry per line,
        /// TAB-separated `<original_size>\t<compressed_size>\t<method>\t<sha256>\t<path>`.
        /// Stable for OS file-manager / archive-tool integration
        /// (Ark CliPlugin, Nautilus extension, etc.).
        #[arg(long, conflicts_with = "tree")]
        long: bool,
    },
    /// Extract one entry from an `.lma` archive to stdout
    ///
    /// Composes with `less` / `grep` / `fzf` / shell preview panes.
    /// Path-traversal protected: the entry must appear verbatim in
    /// the manifest.
    #[command(after_help = "\
EXAMPLES:
  # Show entry contents
  lml cat recording.lma recording.tse

  # Pipe to less for paging
  lml cat recording.lma recording_summary.txt | less

  # Pipe to grep
  lml cat recording.lma recording.csv_bi | grep seiz")]
    Cat {
        /// Input `.lma` file
        input: PathBuf,
        /// Entry path within the archive (e.g. `recording.tse`,
        /// `__orphans__/notes.txt`)
        entry: String,
    },
    /// Phase 3.5 — split a single LML into N equal-sample-count
    /// chunks. Each chunk is re-encoded with the source's sample rate,
    /// window size, and (a copy of) the source metadata; the metadata
    /// gains a `split_chunk_idx` + `split_n_chunks` field so a future
    /// `lml concat` can detect provenance. Last chunk absorbs any
    /// remainder so concat is lossless. Requires the source carry an
    /// LMLFOOT1 seek table.
    Split {
        /// Input LML file
        input: PathBuf,
        /// Number of output chunks (must be >= 2)
        #[arg(long, default_value = "2")]
        chunks: u32,
        /// Output directory (created if missing). Chunk files are
        /// named `<stem>.part-NN-of-MM.lml`.
        #[arg(short, long)]
        output: PathBuf,
        /// Overwrite existing chunk files inside the output directory.
        #[arg(long)]
        force: bool,
    },
    /// Phase 3.6 — concatenate sibling LMLs (same n_channels +
    /// sample_rate + window_size) into one LML in deterministic order.
    /// If every input carries the `split_chunk_idx` + `split_n_chunks`
    /// metadata fields from `lml split`, inputs are sorted by chunk_idx
    /// and validated for completeness (idx 0..N-1 with no gaps/dups).
    /// Otherwise inputs are concatenated in lexicographic filename
    /// order. Bible R32 — byte-identical regardless of argv order.
    Concat {
        /// Two or more input LML files
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,
        /// Output LML path
        #[arg(short, long)]
        output: PathBuf,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
    },
    /// Phase 1.4 — common `--force` flag rationale: every command
    /// below that writes a single output path defaults to refusing to
    /// overwrite an existing file. Pass `--force` to opt in. The legacy
    /// `encode` / `decode` / `export` paths keep their current silent-
    /// overwrite default to preserve existing CI scripts; they have
    /// `--skip-existing` instead.
    ///
    /// Phase 8.1 — serve Prometheus text-exposition counters on
    /// `--bind ADDR/metrics`. Useful for dashboards observing a
    /// long-running `lml watch` daemon. Requires `--features async`.
    #[cfg(feature = "async")]
    Metrics {
        /// Bind address (host:port). Default 127.0.0.1:9100.
        #[arg(long, default_value = "127.0.0.1:9100")]
        bind: String,
    },
    /// Phase 6.4 — watch a directory for new EDF/BDF files and
    /// auto-encode them to LML in real time. Bounded mpsc with
    /// drop-oldest WARN on backpressure. Stops on SIGINT. Requires
    /// the binary built with `--features async`.
    #[cfg(feature = "async")]
    Watch {
        /// Input directory to watch (recursive)
        input: PathBuf,
        /// Output directory for emitted .lml files
        #[arg(short, long)]
        output: PathBuf,
        /// Queue capacity for the watcher → encoder pipeline.
        #[arg(long, default_value = "256")]
        queue_cap: usize,
        /// Strip N least-significant bits during encode
        #[arg(long, default_value = "0")]
        noise_bits: u8,
        /// Samples per compression window
        #[arg(long, default_value = "2500")]
        window_size: usize,
        /// Source sample rate (Hz)
        #[arg(long, default_value = "250.0")]
        sample_rate: f64,
    },
    /// Phase 6.2 — HTTP fetch a remote file (http/https only). Refuses
    /// non-http(s) schemes. Requires `--features async`.
    #[cfg(feature = "async")]
    Fetch {
        /// URL to fetch
        url: String,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Max bytes to accept (default 8 GiB)
        #[arg(long, default_value = "8589934592")]
        max_bytes: u64,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
    },
    /// Phase 6.6 — POST a webhook callback with op_id idempotency-key.
    /// Exponential backoff retries. Requires `--features async`.
    #[cfg(feature = "async")]
    Notify {
        /// Webhook URL
        url: String,
        /// Operation verb (e.g. "encode", "decode")
        #[arg(long)]
        op: String,
        /// Idempotency key (defaults to a timestamp-derived value)
        #[arg(long)]
        op_id: Option<String>,
        /// Source path field for the payload
        #[arg(long, default_value = "")]
        source_path: String,
        /// Output path field for the payload
        #[arg(long, default_value = "")]
        output_path: String,
        /// SHA-256 hex (or "" if not applicable)
        #[arg(long, default_value = "")]
        content_sha256: String,
        /// Bytes count to report
        #[arg(long, default_value = "0")]
        bytes: u64,
        /// Max retries on transient failure
        #[arg(long, default_value = "3")]
        max_retries: u32,
    },
    /// Phase 7.1 — AES-256-GCM encrypt a file. Reads the 32-byte key
    /// (as 64-char hex) from the `LAMQUANT_KEY` env-var. Output blob
    /// is self-describing (magic `LMLCRYPT` + version + nonce + GCM
    /// ciphertext with auth tag).
    Encrypt {
        /// Input file
        input: PathBuf,
        /// Output ciphertext file
        #[arg(short, long)]
        output: PathBuf,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
        /// v1.2 P — derive the AES-256 key from a password via
        /// Argon2id (OWASP defaults: m=64 MiB, t=3, p=1). When set,
        /// reads the password from `LAMQUANT_PASSWORD` env (if non-
        /// empty) or prompts interactively without echo. The 16-byte
        /// salt + Argon2 params are stored in a sidecar header at
        /// `<output>.lmcrypt.header` so decrypt can re-derive the
        /// same key. Mutually exclusive with `LAMQUANT_KEY` env path.
        #[arg(long)]
        password: bool,
    },
    /// Phase 7.1 — AES-256-GCM decrypt + authenticate. Errors on bad
    /// magic / version / auth-tag mismatch / wrong key.
    Decrypt {
        /// Input ciphertext file (LMLCRYPT)
        input: PathBuf,
        /// Output plaintext file
        #[arg(short, long)]
        output: PathBuf,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
        /// v1.2 P — derive the AES-256 key from a password via
        /// Argon2id. Reads the salt + params from the sidecar at
        /// `<input>.lmcrypt.header`; reads the password from
        /// `LAMQUANT_PASSWORD` env or prompts interactively.
        #[arg(long)]
        password: bool,
    },
    /// Phase 7.2 — HMAC-SHA-256 sign a file. Detached 32-byte tag
    /// written to `<input>.hmac` (or --output).
    Sign {
        /// Input file
        input: PathBuf,
        /// Tag output path. Defaults to `<input>.hmac`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Overwrite an existing tag file.
        #[arg(long)]
        force: bool,
    },
    /// Phase 7.2 — verify the HMAC-SHA-256 tag of a file. Reads the
    /// 32-byte tag from `<input>.hmac` (or --tag).
    VerifySignature {
        /// File whose contents are verified
        input: PathBuf,
        /// Tag path. Defaults to `<input>.hmac`.
        #[arg(long)]
        tag: Option<PathBuf>,
    },
    /// Phase 7.6 — append to or verify the tamper-evident audit log.
    AuditLog {
        #[command(subcommand)]
        action: AuditAction,
    },
    /// Phase 3.7 — pull a single entry out of an LMA archive without
    /// unpacking the rest. Entry is matched by exact path first, then
    /// by suffix (so passing just the basename works). LML entries are
    /// reconstructed to the original EDF/BDF (matching `lml extract`);
    /// zstd entries are decompressed. SHA-256 of the reconstructed
    /// payload is verified against the manifest.
    ExtractEntry {
        /// Input .lma file
        archive: PathBuf,
        /// Entry path within the archive (exact match or suffix match)
        entry: String,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
    },
    /// Phase 3.10 — recompress an LML with new compression parameters.
    /// Decodes the signal from the source and re-encodes it against
    /// the same metadata using the new --noise-bits / --window-size /
    /// --lpc-mode values. Combined with --noise-bits > 0 this is the
    /// recommended path for converting an existing lossless LML into a
    /// lossy one without re-running the EDF reader. Output via
    /// --output PATH or atomic --in-place.
    Recompress {
        /// Input LML file
        input: PathBuf,
        /// Output LML path (required unless --in-place is set)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Atomic-swap the input file in place via same-dir tempfile.
        #[arg(long)]
        in_place: bool,
        /// Strip N least-significant bits during recompression
        /// (0 = lossless).
        #[arg(long, default_value = "0")]
        noise_bits: u8,
        /// Samples per compression window (default 2500 = 10s @ 250Hz
        /// ref; the encoder scales for actual sample rate).
        #[arg(long, default_value = "2500")]
        window_size: usize,
        /// LPC mode for recompressed output.
        #[arg(long, default_value = "fixed", value_parser = ["fixed", "adaptive", "anytime"])]
        lpc_mode: String,
        /// Overwrite an existing --output target.
        #[arg(long)]
        force: bool,
    },
    /// Phase 3.11 — update an LML's metadata JSON via key/value edits
    /// and/or a sidecar JSON file. Signal payload is never touched;
    /// only the metadata blob in the container header is rewritten.
    /// Output via --output PATH or atomic --in-place.
    ///
    /// Edit order: sidecar overlay (top-level merge) → --set values →
    /// --remove keys. --set values are parsed as JSON literals when
    /// syntactically valid (numbers, true/false/null, "...", [...],
    /// {...}); otherwise stored as strings verbatim.
    SetMetadata {
        /// Input LML file
        input: PathBuf,
        /// Output LML path (required unless --in-place is set)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Atomic-swap the input file in place via same-dir tempfile.
        #[arg(long)]
        in_place: bool,
        /// Sidecar JSON file merged top-level into the metadata.
        #[arg(long)]
        sidecar: Option<PathBuf>,
        /// Set one or more keys via KEY=VALUE. Repeatable. Values are
        /// parsed as JSON literals where syntactically valid; otherwise
        /// stored as strings.
        #[arg(long = "set", value_name = "KEY=VALUE")]
        sets: Vec<String>,
        /// Remove one or more top-level keys. Repeatable.
        #[arg(long = "remove", value_name = "KEY")]
        removes: Vec<String>,
        /// Overwrite an existing --output target.
        #[arg(long)]
        force: bool,
    },
    /// Phase 3.9 — strip patient PII from an LML file. Masks the EDF
    /// header's patient_id (bytes 8..88) and recording_id (bytes
    /// 88..168) with spaces. By default also masks start_date and
    /// start_time; pass --keep-dates to retain them. Re-encodes the
    /// container against the original signal so byte-exact decompression
    /// still works.
    ///
    /// The output is written to a separate file; the input is never
    /// modified in place (use `lml strip-pii … --in-place` for an
    /// atomic-swap update of the source file).
    StripPii {
        /// Input LML file
        input: PathBuf,
        /// Output LML path (required unless --in-place is set)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Atomic-swap the input file in place via same-dir tempfile +
        /// rename + fsync. Refuses if --output is also set.
        #[arg(long)]
        in_place: bool,
        /// Skip the start_date / start_time masking. patient_id /
        /// recording_id are always masked.
        #[arg(long)]
        keep_dates: bool,
        /// Overwrite an existing --output target.
        #[arg(long)]
        force: bool,
    },
    /// Phase 3.8 — append a file to an existing LMA archive without
    /// rewriting pre-existing payload bytes. Uses a same-directory
    /// tempfile (WAL) + atomic rename + fsync so the old archive is
    /// either fully replaced or fully retained — never half-updated.
    /// Old archive is retained at `<archive>.lma.bak` unless --no-bak.
    /// Idempotent on identical SHA-256 (no-op); errors on duplicate
    /// path with different content unless --force.
    Append {
        /// Input .lma archive
        archive: PathBuf,
        /// File to append (EDF/BDF goes through LML compression;
        /// other types go through zstd; falls back to store)
        file: PathBuf,
        /// Entry path inside the archive. Defaults to the file's basename.
        #[arg(long, value_name = "PATH")]
        r#as: Option<String>,
        /// Zstd level (1-22, default 9) for non-LML entries.
        #[arg(long, default_value = "9")]
        zstd_level: i32,
        /// Overwrite an entry whose path matches but whose SHA-256
        /// differs (otherwise the call errors).
        #[arg(long)]
        force: bool,
        /// Discard the `<archive>.lma.bak` backup after successful
        /// rename. Default is to keep it.
        #[arg(long)]
        no_bak: bool,
    },
    /// PCCP — model version pins, integrity verify, change history
    Pccp {
        #[command(subcommand)]
        action: PccpAction,
    },
    /// Emit a roff-format man page for `lml` to stdout. Pipe to
    /// `~/.local/share/man/man1/lml.1` (or your system man path) and
    /// run `mandb` to register.
    Manpage,
    /// Built-in self-test: encode a synthetic signal, decode it, and
    /// verify the round-trip is byte-exact + the CRC32 footer matches.
    /// Useful for confirming a fresh binary install works end-to-end
    /// before running it against real data.
    ///
    /// Exits 0 on success, 1 on any mismatch.
    SelfTest,
    /// Generate shell completion script for the given shell.
    ///
    /// Pipe the output to your shell's completion directory:
    ///
    ///   bash:        lml completions bash       > ~/.local/share/bash-completion/completions/lml
    ///   zsh:         lml completions zsh        > "${fpath[1]}/_lml"
    ///   fish:        lml completions fish       > ~/.config/fish/completions/lml.fish
    ///   PowerShell:  lml completions powershell | Out-String | Invoke-Expression
    ///   elvish:      lml completions elvish     > ~/.config/elvish/lib/lml.elv
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[derive(Subcommand)]
enum NwbCmd {
    /// Pack: rewrite a native NWB/HDF5 with its integer datasets LML-compressed.
    /// Output stays a valid NWB/HDF5, readable anywhere the filter is present.
    Pack {
        /// Source .nwb / .h5 / .hdf5
        input: PathBuf,
        /// Destination (native NWB/HDF5, smaller)
        #[arg(short, long)]
        output: PathBuf,
        /// Path to liblamquant_lml_h5filter.so (else $LAMQUANT_H5FILTER, else
        /// searched beside the lml binary and target/release).
        #[arg(long = "plugin-so", value_name = "PATH")]
        plugin_so: Option<PathBuf>,
        /// Skip the lossless round-trip verification (on by default).
        #[arg(long = "no-verify")]
        no_verify: bool,
    },
    /// Unpack: rewrite an LML-filtered NWB/HDF5 back to a plain (unfiltered) one.
    Unpack {
        /// Source LML-filtered .nwb / .h5
        input: PathBuf,
        /// Destination (plain native NWB/HDF5)
        #[arg(short, long)]
        output: PathBuf,
        /// Path to the filter .so (needed to DECODE); see `pack --plugin-so`.
        #[arg(long = "plugin-so", value_name = "PATH")]
        plugin_so: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum AuditAction {
    /// Append a new entry to the audit log.
    Append {
        /// Audit-log file path (JSONL, append-only)
        #[arg(short, long)]
        log: PathBuf,
        /// Operation verb (e.g. "encode", "decode", "decrypt")
        #[arg(long)]
        op: String,
        /// Free-form message body
        #[arg(long)]
        msg: String,
    },
    /// Verify the SHA-chain integrity of an audit-log file.
    Verify {
        /// Audit-log file path (JSONL)
        #[arg(short, long)]
        log: PathBuf,
    },
}

#[derive(Subcommand)]
enum PccpAction {
    /// Print device + model version card from pccp/registry.yaml
    Version,
    /// Verify a checkpoint SHA-256 against registry pin
    Verify {
        /// Model class (encoder | decoder | snn | tnn | oracle)
        model: String,
        /// Path to checkpoint file
        path: PathBuf,
    },
    /// Print recent CHANGELOG entries (default last 5)
    History {
        /// Number of entries to show
        #[arg(short = 'n', long, default_value = "5")]
        count: usize,
    },
}

/// Process-global emit-json-events flag. Set once in main from the CLI
/// arg; queried by emit_* helpers so call sites in encode loops can call
/// them unconditionally and incur zero cost when emission is off.
static EMIT_JSON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Phase 1.7 — process-global `--fail-fast` flag. Default false
/// (continue-on-error). Set by cmd_encode / cmd_decode; consulted by
/// the rayon batch loops to short-circuit further per-file work after
/// the first error. Atomic so par-iter workers don't race the read.
static FAIL_FAST_FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Parse the `--lpc-mode` flag string into a typed [`lamquant_core::lpc::LpcMode`].
/// Clap restricts the string to `adaptive|fixed|anytime` via `value_parser`
/// so the wildcard branch is unreachable; we still return a default rather
/// than panic so a malformed call path can't crash the encoder.
fn parse_lpc_mode(s: &str) -> lamquant_core::lpc::LpcMode {
    use lamquant_core::lpc::LpcMode;
    const ADAPTIVE_MAX_ORDER: usize = 16;
    match s {
        "fixed" => LpcMode::Fixed,
        "adaptive" => LpcMode::Adaptive {
            max_order: ADAPTIVE_MAX_ORDER,
        },
        "anytime" => LpcMode::Anytime {
            max_order: ADAPTIVE_MAX_ORDER,
            deadline: None,
        },
        _ => LpcMode::default(),
    }
}

fn parse_lossless_mode(mode: Option<&str>) -> Result<LosslessMode, String> {
    match mode.unwrap_or("mcu") {
        "mcu" => Ok(LosslessMode::Mcu),
        "basestation" => {
            if cfg!(feature = "experimental_basestation") {
                Ok(LosslessMode::Basestation)
            } else {
                Err("basestation lossless mode requires building lml with `--features experimental_basestation`".to_owned())
            }
        }
        other => Err(format!("unknown lossless mode `{other}`")),
    }
}

fn resolve_lpc_mode(
    lossless_mode: LosslessMode,
    lpc_mode: &str,
) -> Result<lamquant_core::lpc::LpcMode, String> {
    match (lossless_mode, lpc_mode) {
        (LosslessMode::Mcu, "auto" | "fixed") => Ok(LosslessMode::Mcu.default_lpc_mode()),
        (LosslessMode::Mcu, other) => Err(format!(
            "MCU lossless mode only permits --lpc-mode fixed/auto; got `{other}`"
        )),
        (LosslessMode::Basestation, "auto") => Ok(LosslessMode::Basestation.default_lpc_mode()),
        (LosslessMode::Basestation, other) => Ok(parse_lpc_mode(other)),
    }
}

fn reject_explicit_lossless_with_non_lossless(
    lossless_mode: Option<&str>,
    max_error: Option<u64>,
    target_bps: Option<f64>,
) -> Result<(), String> {
    if lossless_mode.is_some() && target_bps.is_some() {
        return Err("--lossless-mode cannot be combined with --target-bps".to_owned());
    }
    if lossless_mode.is_some() && max_error.unwrap_or(0) > 0 {
        return Err("--lossless-mode cannot be combined with --max-error > 0".to_owned());
    }
    Ok(())
}

/// Find every sibling sidecar file next to a given EDF — anything in
/// the same directory whose basename starts with the EDF's stem
/// followed by either `.` (extension separator) or `_` (`_summary.
/// txt`-style suffix), excluding the EDF itself.
///
/// **Stem-collision disambiguation:** if the parent directory contains
/// another EDF/BDF whose stem extends ours (e.g. `recording.edf` and
/// `recording_extra.edf` in the same dir), files matching the longer
/// stem belong to that other EDF — they must not be misattributed
/// here. The function reads every sibling EDF/BDF once up front, then
/// rejects any candidate sidecar whose name starts with a longer EDF
/// stem. This is the failure mode lamu V4 Pro flagged on the initial
/// commit — without the second pass, `recording_extra.tse` would be
/// duplicated into both archives.
///
/// Rationale: the TUH EEG families (TUSZ, TUEV, TUSL, TUAR, TUEP,
/// TUAB) all store labels in sibling files with the same basename as
/// the EDF. The default encode path must preserve every one of them —
/// dropping any one is a silent loss of clinical / training-set
/// supervision. See `tests/integration/test_sidecar_preservation.py`
/// for the contract this enables.
///
/// Returns paths sorted lexicographically for determinism. Directory
/// reads that fail (permission, missing) log a warning to stderr and
/// yield an empty list — the lossless contract relies on the operator
/// seeing the warning, not on the function silently appearing to
/// succeed.
fn find_sidecars(edf_path: &Path) -> Vec<PathBuf> {
    let parent = match edf_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let Some(edf_stem) = edf_path.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let edf_name = edf_path.file_name();

    // Pass 1: collect every other EDF/BDF stem in the parent so we
    // can disambiguate longer-stem collisions in pass 2.
    let mut other_edf_stems: Vec<String> = Vec::new();
    let entries_for_stems = match std::fs::read_dir(&parent) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "Warning: sidecar scan could not read directory {}: {} \
                 — proceeding without bundling sidecars for this EDF",
                parent.display(),
                e
            );
            return Vec::new();
        }
    };
    for entry in entries_for_stems.flatten() {
        let p = entry.path();
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if !matches!(ext.as_deref(), Some("edf") | Some("bdf")) {
            continue;
        }
        if Some(entry.file_name().as_os_str()) == edf_name {
            continue;
        }
        if let Some(s) = p.file_stem().and_then(|s| s.to_str()) {
            if s != edf_stem {
                other_edf_stems.push(s.to_string());
            }
        }
    }
    // Longest-stem first so the longest-match check below short-
    // circuits faster on collision-prone trees.
    other_edf_stems.sort_by_key(|s| std::cmp::Reverse(s.len()));

    // Pass 2: collect sidecars for THIS EDF, rejecting any whose
    // name is actually a sidecar of a longer-stemmed sibling EDF.
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&parent) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "Warning: sidecar scan could not re-read directory {}: {}",
                parent.display(),
                e
            );
            return out;
        }
    };
    for entry in entries.flatten() {
        let name_os = entry.file_name();
        if Some(name_os.as_os_str()) == edf_name {
            continue; // skip the EDF itself
        }
        let Some(name) = name_os.to_str() else {
            continue;
        };
        if !name.starts_with(edf_stem) {
            continue;
        }
        let rest = &name[edf_stem.len()..];
        // Sibling = stem + ('.' or '_') + anything. Strict starts_with
        // alone would also match e.g. `recording.edf2` → not what we
        // want.
        if !(rest.starts_with('.') || rest.starts_with('_')) {
            continue;
        }
        // Disambiguate: if another EDF in this dir has a longer stem
        // that also prefixes this sidecar's name, the sidecar belongs
        // to that other EDF and we must not claim it.
        let belongs_to_longer = other_edf_stems
            .iter()
            .any(|other_stem| other_stem.len() > edf_stem.len() && name.starts_with(other_stem));
        if belongs_to_longer {
            continue;
        }
        let path = entry.path();
        if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Emit a `FileDone` OpEvent JSON line to stdout. Carries the full
/// telemetry the TUI dashboard renders (CR, byte sizes, samples,
/// duration, channels, sample rate, sha256, n_windows). All extended
/// fields are optional — pass None when the value isn't known (e.g.,
/// in the failure branch where encode_one bailed before reading the
/// EDF metadata).
#[allow(clippy::too_many_arguments)]
fn emit_file_done(
    path: &str,
    success: bool,
    ms: u64,
    cr: Option<f64>,
    bytes_in: Option<u64>,
    bytes_out: Option<u64>,
    samples: Option<u64>,
    duration_s: Option<f64>,
    n_channels: Option<u32>,
    sample_rate: Option<f32>,
    sha256: Option<String>,
    n_windows: Option<u32>,
) {
    let ev = lamquant_ops::OpEvent::FileDone {
        ts_ms: lamquant_ops::OpEvent::now_ms(),
        path: path.into(),
        success,
        ms,
        cr,
        bytes_in,
        bytes_out,
        samples,
        duration_s,
        n_channels,
        sample_rate,
        sha256,
        n_windows,
    };
    println!("{}", ev.to_json_line());
}

/// Emit a `Started` OpEvent JSON line to stdout. Helper for the
/// `--emit-json-events` mode. Op-id maps onto the canonical token from
/// `specs/ui-parity.md::Op IDs`.
fn emit_started(op_id: &str, total: Option<u64>) {
    let ev = lamquant_ops::OpEvent::Started {
        ts_ms: lamquant_ops::OpEvent::now_ms(),
        op_id: op_id.into(),
        total,
    };
    println!("{}", ev.to_json_line());
}

/// Emit a `Log` JSON line to stdout. Allows cmd_* functions (or future
/// instrumentation) to push human-readable progress text through the
/// same channel a remote TUI consumer parses. Stable contract — see
/// `crates/lamquant-ops/src/transport/ssh.rs` for the parser.
#[allow(dead_code)]
pub(crate) fn emit_log(message: impl Into<String>) {
    let ev = lamquant_ops::OpEvent::Log {
        ts_ms: lamquant_ops::OpEvent::now_ms(),
        message: message.into(),
    };
    println!("{}", ev.to_json_line());
}

/// Emit a `Progress` JSON line. Used by cmd_* paths that have a known
/// total (encode of N files, batch verify). Optional — parsers must
/// tolerate ops that emit no Progress events at all.
#[allow(dead_code)]
pub(crate) fn emit_progress(current: u64, total: u64, message: impl Into<String>) {
    let ev = lamquant_ops::OpEvent::Progress {
        ts_ms: lamquant_ops::OpEvent::now_ms(),
        current,
        total,
        message: message.into(),
    };
    println!("{}", ev.to_json_line());
}

/// Emit a terminal Done or Error JSON line based on the exit code returned
/// by the underlying cmd_* function. Called once at the end of `main` when
/// the user opted into `--emit-json-events`.
fn emit_terminal(op_id: &str, code: i32, summary: &str) {
    let ev = if code == 0 {
        lamquant_ops::OpEvent::Done {
            ts_ms: lamquant_ops::OpEvent::now_ms(),
            message: summary.into(),
        }
    } else {
        lamquant_ops::OpEvent::Error {
            ts_ms: lamquant_ops::OpEvent::now_ms(),
            message: format!("{} exited with code {}", op_id, code),
        }
    };
    println!("{}", ev.to_json_line());
}

/// Map a `Commands` variant onto its canonical op id from the parity spec.
fn op_id_of(cmd: &Commands) -> &'static str {
    match cmd {
        Commands::Encode { .. } => "encode",
        Commands::Decode { .. } => "decode",
        Commands::Info { .. } => "info",
        Commands::Verify { .. } => "verify",
        Commands::Roundtrip { .. } => "roundtrip",
        Commands::VerifyManifest { .. } => "verify_manifest",
        Commands::Stats { .. } => "stats",
        Commands::Export { format, .. } => match format.as_str() {
            "csv" => "export_csv",
            "npy" => "export_npy",
            "raw" => "export_raw",
            _ => "export_csv",
        },
        Commands::Nwb { cmd } => match cmd {
            NwbCmd::Pack { .. } => "nwb_pack",
            NwbCmd::Unpack { .. } => "nwb_unpack",
        },
        Commands::Diff { .. } => "diff",
        Commands::Recover { .. } => "recover",
        Commands::Bench { .. } => "bench",
        Commands::Archive { .. } => "archive",
        Commands::Extract { .. } => "extract",
        Commands::ListArchive { .. } => "list_archive",
        Commands::VolumeSplit { .. } => "volume_split",
        Commands::VolumeAssemble { .. } => "volume_assemble",
        Commands::VerifyArchive { .. } => "verify_archive",
        Commands::Ls { .. } => "ls",
        Commands::Cat { .. } => "cat",
        Commands::Split { .. } => "split",
        Commands::Concat { .. } => "concat",
        #[cfg(feature = "async")]
        Commands::Metrics { .. } => "metrics",
        #[cfg(feature = "async")]
        Commands::Watch { .. } => "watch",
        #[cfg(feature = "async")]
        Commands::Fetch { .. } => "fetch",
        #[cfg(feature = "async")]
        Commands::Notify { .. } => "notify",
        Commands::Encrypt { .. } => "encrypt",
        Commands::Decrypt { .. } => "decrypt",
        Commands::Sign { .. } => "sign",
        Commands::VerifySignature { .. } => "verify_signature",
        Commands::AuditLog { .. } => "audit_log",
        Commands::ExtractEntry { .. } => "extract_entry",
        Commands::Append { .. } => "append",
        Commands::StripPii { .. } => "strip_pii",
        Commands::SetMetadata { .. } => "set_metadata",
        Commands::Recompress { .. } => "recompress",
        Commands::Pccp { .. } => "pccp",
        Commands::SelfTest => "self_test",
        Commands::Manpage => "manpage",
        Commands::Completions { .. } => "completions",
    }
}

/// Initialise the structured-logging substrate.
///
/// Filter precedence (highest first):
/// 1. `RUST_LOG` env var (standard tracing-subscriber escape hatch)
/// 2. `--quiet` / `-v` CLI flags (Phase 1.1.e):
///    `--quiet`  → `off`        (nothing routes to stderr).
///    default    → `lamquant_core=warn,lml=info`.
///    `-v`       → `lamquant_core=info,lml=info`.
///    `-vv`      → `lamquant_core=debug,lml=debug`.
///    `-vvv`     → `lamquant_core=trace,lml=trace`.
/// 3. Fallback (no flags, no env) = default level above.
///
/// The subscriber writes to stderr to keep stdout reserved for the
/// JSON event stream (`--emit-json-events`) and signal data (Phase
/// 1.5 stdin/stdout piping).
fn init_tracing(quiet: bool, verbose: u8, color: ColorChoice) {
    use tracing_subscriber::EnvFilter;
    let filter = if let Ok(env) = std::env::var("RUST_LOG") {
        // RUST_LOG wins. Standard ecosystem hook.
        EnvFilter::new(env)
    } else if quiet {
        EnvFilter::new("off")
    } else {
        let level = match verbose {
            0 => "lamquant_core=warn,lml=info",
            1 => "lamquant_core=info,lml=info",
            2 => "lamquant_core=debug,lml=debug",
            _ => "lamquant_core=trace,lml=trace",
        };
        EnvFilter::new(level)
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false) // keep lines short; module path is rarely useful at CLI
        .with_ansi(color.resolve())
        .try_init();
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.quiet, cli.verbose, cli.color);

    // Apply the CLI-provided compute backend (if any) before any
    // encode work runs. The selector is process-wide; container::
    // encode_into reads it per window. Omitted flag = library
    // default (Desktop on host).
    if let Some(choice) = cli.backend {
        lamquant_core::backend::set_global_backend(choice.to_lib());
    }
    let emit_json = cli.emit_json_events;
    EMIT_JSON.store(emit_json, std::sync::atomic::Ordering::Relaxed);
    if emit_json {
        if let Some(cmd) = cli.command.as_ref() {
            emit_started(op_id_of(cmd), None);
        }
    }
    let op_id_for_terminal = cli.command.as_ref().map(op_id_of).unwrap_or("interactive");
    let result = match cli.command {
        None => {
            // No subcommand → interactive TUI
            std::process::exit(tui::run_interactive());
        }
        Some(cmd) => match cmd {
            Commands::Encode {
                input,
                output,
                verify,
                cross_validate,
                noise_bits,
                max_error,
                target_bps,
                lossless_mode,
                window_size,
                threads,
                recursive,
                skip_existing,
                dry_run,
                lma,
                no_bundle,
                lml_siblings,
                i_understand_data_loss,
                include_globs,
                exclude_globs,
                lpc_mode,
                fail_fast,
                continue_on_error: _,
            } => {
                if let Err(e) = reject_explicit_lossless_with_non_lossless(
                    lossless_mode.as_deref(),
                    max_error,
                    target_bps,
                ) {
                    Err(e.into())
                } else {
                    match parse_lossless_mode(lossless_mode.as_deref()) {
                        Err(e) => Err(e.into()),
                        Ok(selected_lossless_mode) => {
                            match resolve_lpc_mode(selected_lossless_mode, &lpc_mode) {
                                Err(e) => Err(e.into()),
                                Ok(selected_lpc_mode) => {
                                    // ADR 0051 track 2: bounded-MAE near-lossless. Self-contained
                                    // single-file path → bare `.lml` via closed-loop DPCM, bypasses
                                    // the batch/bundle machinery (the lossy `.lml` has no `.lma`
                                    // sibling-envelope semantics). For the H.BWC working-point
                                    // bench + clinical near-lossless use.
                                    if let Some(tb) = target_bps {
                                        cmd_encode_target_bps(
                                            &input,
                                            output.as_deref(),
                                            tb,
                                            window_size,
                                            parse_lpc_mode(&lpc_mode),
                                            verify,
                                        )
                                    } else if let Some(delta) = max_error {
                                        cmd_encode_bounded_mae(
                                            &input,
                                            output.as_deref(),
                                            delta,
                                            window_size,
                                            parse_lpc_mode(&lpc_mode),
                                            verify,
                                        )
                                    // `--lml-siblings` is the TUI's "LML + copy siblings"
                                    // mode -- walk the source tree, encode every EDF to
                                    // .lml, copy every non-EEG file verbatim into the
                                    // mirrored output. Diverges from --no-bundle in
                                    // scope: --no-bundle handles stem-bound sidecars per
                                    // EDF; --lml-siblings handles the entire tree (every
                                    // file, every sub-directory). Implementation lives
                                    // in lma::pack_lml_with_siblings.
                                    } else if lml_siblings {
                                        // Pre-flight: --lml-siblings demands an explicit
                                        // -o because it writes into a directory, not a
                                        // single file. Same shape as the --lma branch
                                        // already enforces.
                                        match output.as_deref() {
                        None => Err::<(), Box<dyn std::error::Error + Send + Sync>>(
                            "--lml-siblings requires an explicit -o <output_dir>".into(),
                        ),
                        Some(out_dir) => match lma::pack_lml_with_siblings(
                            std::path::Path::new(&input),
                            std::path::Path::new(out_dir),
                            true,
                            None,
                        ) {
                            Err(e) => Err(e),
                            Ok(summary) => {
                                eprintln!(
                                    "  LML+siblings: {} encoded, {} copied, {} errors",
                                    summary.counts_lml,
                                    summary.counts_copied,
                                    summary.errors.len(),
                                );
                                Ok(())
                            }
                        },
                    }
                                    } else {
                                        // Data-loss footgun guard. `--no-bundle` (alias
                                        // `--bare-lml`) emits raw `.lml` without the per-
                                        // recording `.lma` envelope. Sidecars are mirror-copied
                                        // next to the `.lml` (the encoder never silently drops
                                        // them) BUT the operator who moves only the `.lml` to a
                                        // new location, or deletes the source dir after
                                        // verifying signal parity, will lose every sibling
                                        // file. This is the failure mode that cost one full
                                        // TUEG migration cycle. We refuse to be silent about
                                        // it: every invocation of `--no-bundle` prints a loud
                                        // warning unless paired with `--i-understand-data-loss`.
                                        // Tier 3 audit (O2): --no-bundle is now a HARD GATE.
                                        // The pre-fix code printed a loud stderr warning and
                                        // then proceeded with encoding either way -- but
                                        // `2>/dev/null` silences the warning, the exit code
                                        // stayed 0, and downstream automation never knew the
                                        // data-loss path was taken. CLAUDE.md / Bible R5 (no
                                        // silent fallback): require `--i-understand-data-loss`
                                        // as a hard pre-condition. Exit code 2 if missing.
                                        if no_bundle && !i_understand_data_loss {
                                            eprintln!();
                                            eprintln!("  ============================================================");
                                            eprintln!("  ERROR: --no-bundle / --bare-lml refuses to run without ");
                                            eprintln!("         --i-understand-data-loss");
                                            eprintln!("  ============================================================");
                                            eprintln!("  Bare `.lml` output drops the per-recording `.lma` envelope.");
                                            eprintln!("  Sidecar files (.tse, .csv_bi, .lbl_bi, _summary.txt, etc.)");
                                            eprintln!("  are mirror-copied next to the `.lml` so they are NOT");
                                            eprintln!("  silently lost in this run -- but if you later move the");
                                            eprintln!("  `.lml` to a different directory, or delete the source");
                                            eprintln!("  directory after verifying signal parity, every sidecar");
                                            eprintln!(
                                                "  will go with it. There is no second chance."
                                            );
                                            eprintln!();
                                            eprintln!("  Default behaviour (no `--no-bundle` flag) produces a per-");
                                            eprintln!("  recording `.lma` archive that bundles `.lml` + every");
                                            eprintln!("  sibling. The archive travels as one file; sidecars cannot");
                                            eprintln!("  be left behind by accident.");
                                            eprintln!();
                                            eprintln!("  Re-run with `--i-understand-data-loss` to acknowledge the");
                                            eprintln!("  failure mode and proceed. The flag is intentionally verbose.");
                                            eprintln!("  ============================================================");
                                            eprintln!();
                                            std::process::exit(2);
                                        }
                                        cmd_encode(
                                            &input,
                                            output.as_deref(),
                                            verify,
                                            cross_validate,
                                            noise_bits,
                                            window_size,
                                            threads,
                                            recursive,
                                            skip_existing,
                                            dry_run,
                                            lma,
                                            no_bundle,
                                            selected_lpc_mode,
                                            fail_fast,
                                            &include_globs,
                                            &exclude_globs,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Commands::Decode {
                input,
                output,
                recursive,
                skip_existing,
                threads,
                to_edf,
                channels,
                time_range,
                fail_fast,
                continue_on_error: _,
            } => cmd_decode(
                &input,
                output.as_deref(),
                recursive,
                skip_existing,
                threads,
                to_edf,
                channels.as_deref(),
                time_range.as_deref(),
                fail_fast,
            ),
            Commands::Info { input } => cmd_info(&input),
            Commands::Verify {
                input,
                recursive,
                explain,
            } => cmd_verify(&input, recursive, explain),
            Commands::Roundtrip {
                input,
                recursive,
                report,
                fail_fast,
                parallel,
            } => cmd_roundtrip(&input, recursive, report.as_deref(), fail_fast, parallel),
            Commands::VerifyManifest { manifest } => cmd_verify_manifest(&manifest),
            Commands::Stats { input, recursive } => cmd_stats(&input, recursive),
            Commands::Export {
                input,
                output,
                format,
                lossless_mode,
            } => cmd_export(&input, output.as_deref(), &format, lossless_mode.as_deref()),
            Commands::Nwb { cmd } => cmd_nwb(cmd),
            Commands::Diff { a, b } => cmd_diff(&a, &b),
            Commands::Recover { input, output } => cmd_recover(&input, &output),
            Commands::Bench { input } => cmd_bench(&input),
            Commands::Archive {
                input,
                output,
                zstd_level,
            } => cmd_archive(&input, output.as_deref(), zstd_level),
            Commands::Extract {
                input,
                output,
                verify,
                no_verify,
            } => cmd_extract(&input, output.as_deref(), verify && !no_verify),
            Commands::ListArchive { input } => {
                // v1.1: `list-archive` is the legacy tabular listing.
                // `ls --tree` covers the same surface plus a flat
                // `ls` mode. Print a one-line deprecation note so
                // operators migrate scripts; behaviour stays
                // unchanged for one release.
                eprintln!(
                    "note: `list-archive` is deprecated; use `lml ls {} --tree` for the same output (removal planned for v2.0).",
                    input.display(),
                );
                cmd_list_archive(&input)
            }
            Commands::VolumeSplit {
                input,
                size,
                keep,
                force,
            } => cmd_volume_split(&input, &size, keep, force),
            Commands::VolumeAssemble {
                input,
                output,
                force,
            } => cmd_volume_assemble(&input, output.as_deref(), force),
            Commands::VerifyArchive { input, explain } => {
                cmd_verify_archive_explain(&input, explain)
            }
            Commands::Ls { input, tree, long } => cmd_ls(&input, tree, long),
            Commands::Cat { input, entry } => cmd_cat(&input, &entry),
            Commands::Split {
                input,
                chunks,
                output,
                force,
            } => cmd_split(&input, chunks, &output, force),
            Commands::Concat {
                inputs,
                output,
                force,
            } => cmd_concat(&inputs, &output, force),
            #[cfg(feature = "async")]
            Commands::Metrics { bind } => cmd_metrics(&bind),
            #[cfg(feature = "async")]
            Commands::Watch {
                input,
                output,
                queue_cap,
                noise_bits,
                window_size,
                sample_rate,
            } => cmd_watch(
                input,
                output,
                queue_cap,
                noise_bits,
                window_size,
                sample_rate,
            ),
            #[cfg(feature = "async")]
            Commands::Fetch {
                url,
                output,
                max_bytes,
                force,
            } => cmd_fetch(&url, &output, max_bytes, force),
            #[cfg(feature = "async")]
            Commands::Notify {
                url,
                op,
                op_id,
                source_path,
                output_path,
                content_sha256,
                bytes,
                max_retries,
            } => cmd_notify(
                &url,
                &op,
                op_id.as_deref(),
                &source_path,
                &output_path,
                &content_sha256,
                bytes,
                max_retries,
            ),
            Commands::Encrypt {
                input,
                output,
                force,
                password,
            } => cmd_encrypt(&input, &output, force, password),
            Commands::Decrypt {
                input,
                output,
                force,
                password,
            } => cmd_decrypt(&input, &output, force, password),
            Commands::Sign {
                input,
                output,
                force,
            } => cmd_sign(&input, output.as_deref(), force),
            Commands::VerifySignature { input, tag } => {
                cmd_verify_signature(&input, tag.as_deref())
            }
            Commands::AuditLog { action } => cmd_audit_log(action),
            Commands::ExtractEntry {
                archive,
                entry,
                output,
                force,
            } => cmd_extract_entry(&archive, &entry, &output, force),
            Commands::Append {
                archive,
                file,
                r#as,
                zstd_level,
                force,
                no_bak,
            } => cmd_append(&archive, &file, r#as.as_deref(), zstd_level, force, !no_bak),
            Commands::StripPii {
                input,
                output,
                in_place,
                keep_dates,
                force,
            } => cmd_strip_pii(&input, output.as_deref(), in_place, keep_dates, force),
            Commands::SetMetadata {
                input,
                output,
                in_place,
                sidecar,
                sets,
                removes,
                force,
            } => cmd_set_metadata(
                &input,
                output.as_deref(),
                in_place,
                sidecar.as_deref(),
                &sets,
                &removes,
                force,
            ),
            Commands::Recompress {
                input,
                output,
                in_place,
                noise_bits,
                window_size,
                lpc_mode,
                force,
            } => cmd_recompress(
                &input,
                output.as_deref(),
                in_place,
                noise_bits,
                window_size,
                parse_lpc_mode(&lpc_mode),
                force,
            ),
            Commands::Pccp { action } => cmd_pccp(action),
            Commands::SelfTest => cmd_self_test(),
            Commands::Manpage => cmd_manpage(),
            Commands::Completions { shell } => {
                cmd_completions(shell);
                Ok(())
            }
        }, // end Some(cmd)
    }; // end match cli.command

    let exit_code = match &result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {}", e);
            1
        }
    };
    if emit_json {
        let summary = match &result {
            Ok(()) => format!("{} completed", op_id_for_terminal),
            Err(e) => format!("{}: {}", op_id_for_terminal, e),
        };
        emit_terminal(op_id_for_terminal, exit_code, &summary);
    }
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

type R = Result<(), Box<dyn std::error::Error + Send + Sync>>;

/// Recognised source-format extensions for `lml encode`. Phase 4
/// registers BrainVision (.vhdr) alongside the EDF/BDF family;
/// adding CNT / custom raw is a one-line append per format here +
/// dispatch in `encode_one_to_lml`.
fn is_source_extension(ext: &std::ffi::OsStr) -> bool {
    let lower = ext.to_string_lossy().to_ascii_lowercase();
    // Phase 4.4 also matches .raw, but only when a sibling sidecar
    // (<stem>.json / <stem>.meta.json / <path>.json) is present —
    // otherwise the file is just unannotated bytes and we shouldn't
    // claim it. The sibling check happens at dispatch time below;
    // we list .raw here so directory scans pick it up.
    // `dcm` only valid when the binary is built with --features dicom;
    // the encode_one dispatch below returns a typed error if dispatched
    // without the feature.
    matches!(
        lower.as_str(),
        "edf" | "bdf" | "vhdr" | "raw" | "cnt" | "dcm" | "set"
    )
}

fn find_edfs(path: &Path, recursive: bool) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files = Vec::new();
    if recursive {
        for entry in walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let p = entry.path();
            if p.is_file() {
                if let Some(ext) = p.extension() {
                    if is_source_extension(ext) {
                        files.push(p.to_path_buf());
                    }
                }
            }
        }
    } else if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    if let Some(ext) = p.extension() {
                        if is_source_extension(ext) {
                            files.push(p);
                        }
                    }
                }
            }
        }
    }
    files.sort();
    files
}

fn make_output_path(input: &Path, input_root: &Path, output_dir: &Path) -> PathBuf {
    // Preserve directory structure: strip the input root, keep the relative path.
    // /data/tueg/edf/000/patient/session/file.edf with root /data/tueg/edf/
    // → output_dir/000/patient/session/file.lml
    let relative = input.strip_prefix(input_root).unwrap_or(input);
    let mut out = output_dir.join(relative);
    out.set_extension("lml");
    out
}

/// UTC ISO-8601 timestamp without external crate.
fn now_utc() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Convert epoch seconds to UTC date/time (no leap second handling needed for log timestamps)
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;

    // Days since 1970-01-01 → (year, month, day) using the civil_from_days algorithm
    let z = days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let yr = if mon <= 2 { y + 1 } else { y };

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", yr, mon, d, h, m, s)
}

struct FileResult {
    source: String,
    output: String,
    raw_bytes: usize,
    compressed_bytes: usize,
    cr: f64,
    sha256: String,
    verified: bool,
    error: Option<String>,
}

/// Telemetry returned by encode_one. Sourced from ContainerStats plus the
/// SHA-256 hash and verify flag computed in encode_one. Drives both the
/// FileResult that lands in manifest.lml.json AND the OpEvent::FileDone
/// that the TUI dashboard consumes.
struct EncodeMetrics {
    raw_size: usize,
    compressed_size: usize,
    cr: f64,
    sha256: String,
    verified: bool,
    samples: u64,
    duration_s: f64,
    n_channels: u32,
    sample_rate: f32,
    n_windows: u32,
}

/// Phase 4.1 — encode a BrainVision (.vhdr/.eeg/.vmrk) recording.
///
/// Goes through `BrainVisionReader::read_bundle` → `SignalBundle` →
/// `container::write_into`. The full `.vhdr` and any `.vmrk` are
/// preserved in metadata as base64+zstd-encoded blobs (`vhdr_b64` and
/// `vmrk_b64`) so a future `lml decode --to-brainvision` can reverse
/// the encode losslessly. Today there's no `--to-brainvision` flag;
/// the round-trip target is round-tripping the integer sample matrix
/// (decode → raw int32 LE matches the original .eeg byte-for-byte for
/// INT_16 sources, modulo the sample-major i64 widen).
fn encode_one_brainvision(
    vhdr_path: &Path,
    lml_path: &Path,
    verify: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::container;
    use lamquant_core::source::{BrainVisionReader, SignalSourceReader};
    let t0 = Instant::now();
    let mut reader = BrainVisionReader::new(vhdr_path);
    let bundle = reader.read_bundle()?;
    let n_channels = bundle.signal.len() as u32;
    let n_samples = bundle.signal.first().map(|c| c.len()).unwrap_or(0);
    let sample_rate = bundle.sample_rate;

    // SHA-256 of the i64 LE sample matrix — same convention as EDF
    // encode for cross-format checksum compatibility.
    let mut hasher = Sha256::new();
    for ch in &bundle.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    // v1.1: source `.vhdr` + `.vmrk` are now preserved as separate
    // LMA entries (in default mode) or sibling files (in --no-bundle
    // mode), not as b64-zstd blobs inside metadata JSON. The metadata
    // shrinks substantially; the legacy reader fallback in v1 .lml
    // files still works because the fields just resolve to empty
    // strings on absence.

    // JSON encode channel labels + phys_min / phys_max.
    let mut ch_json = String::from("[");
    for (i, name) in bundle.channels.iter().enumerate() {
        if i > 0 {
            ch_json.push(',');
        }
        let safe = name
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        ch_json.push('"');
        ch_json.push_str(&safe);
        ch_json.push('"');
    }
    ch_json.push(']');
    let pmin: Vec<String> = bundle.phys_min.iter().map(|v| format!("{v}")).collect();
    let pmax: Vec<String> = bundle.phys_max.iter().map(|v| format!("{v}")).collect();

    let metadata_json = format!(
        "{{\"source_file\":\"{}\",\"format\":\"BRAINVISION\",\"n_channels\":{},\
         \"sample_rate\":{},\"channels\":{},\"phys_min\":[{}],\"phys_max\":[{}],\
         \"phys_dim\":\"{}\",\"signal_sha256\":\"{}\",\
         \"encoder\":\"lml/{}\",\"noise_bits\":{}}}",
        bundle
            .metadata
            .source_file
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
        n_channels,
        sample_rate,
        ch_json,
        pmin.join(","),
        pmax.join(","),
        bundle.metadata.phys_dim.replace('"', "\\\""),
        sha256_hex,
        env!("CARGO_PKG_VERSION"),
        noise_bits,
    );

    if let Some(parent) = lml_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(lml_path)?);
    let stats = container::write_into(
        &mut sink,
        &bundle.signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(
            kind,
            "encode brainvision: BufWriter flush failed before sync_all",
        )
    })?;
    f.sync_all()?;

    let original_size = std::fs::metadata(vhdr_path).map(|m| m.len()).unwrap_or(0)
        + bundle.signal.len() as u64 * n_samples as u64 * 2; // .eeg bytes (int16 ≈ baseline)
    let compressed_size = stats.compressed_size as u64;
    let cr = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if verify {
        // Round-trip: decompress and compare i64 samples.
        let (recovered, _meta) = container::read_file(lml_path)?;
        if recovered.len() != bundle.signal.len() {
            return Err(format!(
                "verify (brainvision): channel count {} != {}",
                recovered.len(),
                bundle.signal.len()
            )
            .into());
        }
        for (i, (a, b)) in recovered.iter().zip(bundle.signal.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "verify (brainvision): channel {i} mismatch ({} vs {} samples)",
                    a.len(),
                    b.len()
                )
                .into());
            }
        }
    }

    // Byte-exact preservation of the BrainVision tri-file (.vhdr +
    // .vmrk + .eeg). The b64-in-metadata embed above is kept for
    // backward-compatibility with v1 `.lml` readers but is now
    // redundant with the LMA-entry preservation below. New writes
    // can drop the b64 in a follow-up cleanup once we're confident
    // no consumer reads it.
    //
    // Same-dir guard mirrors the EEGLAB encoder: when `-o` points at
    // the source dir (in-place encode), do not rewrite source bytes.
    let lml_parent = lml_path.parent().unwrap_or_else(|| Path::new("."));
    let vhdr_parent = vhdr_path.parent().unwrap_or_else(|| Path::new("."));
    let same_dir = match (lml_parent.canonicalize(), vhdr_parent.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir {
        // Copy `.vhdr` -- exact filename of the source.
        if let Some(vhdr_blob) = bundle.sidecar_first("vhdr_raw") {
            let vhdr_name = vhdr_path
                .file_name()
                .ok_or("brainvision encode: vhdr_path has no file_name")?;
            let dest = lml_parent.join(vhdr_name);
            std::fs::write(&dest, &vhdr_blob.bytes).map_err(|e| {
                format!(
                    "brainvision encode: failed to write `.vhdr` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        // Copy `.eeg` -- name comes from the `DataFile=` field of the
        // source .vhdr so the byte-equal .vhdr post-extract still
        // points at a real file.
        if let (Some(eeg_blob), Some(name_blob)) = (
            bundle.sidecar_first("eeg_raw"),
            bundle.sidecar_first("eeg_filename"),
        ) {
            let name = std::str::from_utf8(&name_blob.bytes)
                .map_err(|e| format!("brainvision encode: eeg_filename not UTF-8: {e}"))?;
            let dest = lml_parent.join(name);
            std::fs::write(&dest, &eeg_blob.bytes).map_err(|e| {
                format!(
                    "brainvision encode: failed to write `.eeg` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        // Copy `.vmrk` if it exists -- name comes from `MarkerFile=`.
        if let (Some(vmrk_blob), Some(name_blob)) = (
            bundle.sidecar_first("vmrk_raw"),
            bundle.sidecar_first("vmrk_filename"),
        ) {
            let name = std::str::from_utf8(&name_blob.bytes)
                .map_err(|e| format!("brainvision encode: vmrk_filename not UTF-8: {e}"))?;
            let dest = lml_parent.join(name);
            std::fs::write(&dest, &vmrk_blob.bytes).map_err(|e| {
                format!(
                    "brainvision encode: failed to write `.vmrk` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
    }

    let _ = t0; // elapsed not in EncodeMetrics today; caller times via outer Instant
    Ok(EncodeMetrics {
        raw_size: original_size as usize,
        compressed_size: compressed_size as usize,
        cr,
        sha256: sha256_hex,
        verified: verify,
        samples: (n_channels as u64).saturating_mul(n_samples as u64),
        duration_s: bundle.duration_s,
        n_channels,
        sample_rate: sample_rate as f32,
        n_windows: stats.n_windows as u32,
    })
}

/// Phase 4.4 — encode a custom raw binary (`.raw` + sidecar JSON).
fn encode_one_raw(
    raw_path: &Path,
    lml_path: &Path,
    verify: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::container;
    use lamquant_core::source::{RawReader, SignalSourceReader};
    let t0 = Instant::now();
    let mut reader = RawReader::new(raw_path);
    let bundle = reader.read_bundle()?;
    let n_channels = bundle.signal.len() as u32;
    let n_samples = bundle.signal.first().map(|c| c.len()).unwrap_or(0);
    let sample_rate = bundle.sample_rate;

    // SHA-256 of the i64 LE sample matrix.
    let mut hasher = Sha256::new();
    for ch in &bundle.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    // v1.1: sidecar JSON now preserved as a real sibling file on
    // disk (named via `raw_sidecar_filename` blob in v1.0); no need
    // to embed it as a b64-zstd blob inside metadata JSON.

    let mut ch_json = String::from("[");
    for (i, name) in bundle.channels.iter().enumerate() {
        if i > 0 {
            ch_json.push(',');
        }
        let safe = name
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        ch_json.push('"');
        ch_json.push_str(&safe);
        ch_json.push('"');
    }
    ch_json.push(']');
    let pmin: Vec<String> = bundle.phys_min.iter().map(|v| format!("{v}")).collect();
    let pmax: Vec<String> = bundle.phys_max.iter().map(|v| format!("{v}")).collect();

    let metadata_json = format!(
        "{{\"source_file\":\"{}\",\"format\":\"RAW\",\"n_channels\":{},\
         \"sample_rate\":{},\"channels\":{},\"phys_min\":[{}],\"phys_max\":[{}],\
         \"phys_dim\":\"{}\",\"signal_sha256\":\"{}\",\
         \"encoder\":\"lml/{}\",\"noise_bits\":{}}}",
        bundle
            .metadata
            .source_file
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
        n_channels,
        sample_rate,
        ch_json,
        pmin.join(","),
        pmax.join(","),
        bundle.metadata.phys_dim.replace('"', "\\\""),
        sha256_hex,
        env!("CARGO_PKG_VERSION"),
        noise_bits,
    );

    if let Some(parent) = lml_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(lml_path)?);
    let stats = container::write_into(
        &mut sink,
        &bundle.signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "encode raw: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    let original_size = std::fs::metadata(raw_path).map(|m| m.len()).unwrap_or(0);
    let compressed_size = stats.compressed_size as u64;
    let cr = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if verify {
        let (recovered, _meta) = container::read_file(lml_path)?;
        if recovered.len() != bundle.signal.len() {
            return Err(format!(
                "verify (raw): channel count {} != {}",
                recovered.len(),
                bundle.signal.len()
            )
            .into());
        }
        for (i, (a, b)) in recovered.iter().zip(bundle.signal.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "verify (raw): channel {i} mismatch ({} vs {} samples)",
                    a.len(),
                    b.len()
                )
                .into());
            }
        }
    }

    // Byte-exact preservation of the `.raw` payload + its sidecar
    // JSON + any stem-matched siblings. Same pattern as the other
    // non-EDF encoders. The codec losslessly reconstructs the signal
    // from i64 samples + sidecar JSON, so the payload entry is
    // strictly redundant for roundtrip -- but the user gave us those
    // exact bytes and the invariant says they come back exactly.
    let lml_parent = lml_path.parent().unwrap_or_else(|| Path::new("."));
    let raw_parent = raw_path.parent().unwrap_or_else(|| Path::new("."));
    let same_dir = match (lml_parent.canonicalize(), raw_parent.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir {
        if let (Some(payload_blob), Some(name_blob)) = (
            bundle.sidecar_first("raw_payload_raw"),
            bundle.sidecar_first("raw_payload_filename"),
        ) {
            let name = std::str::from_utf8(&name_blob.bytes)
                .map_err(|e| format!("raw encode: raw_payload_filename not UTF-8: {e}"))?;
            let dest = lml_parent.join(name);
            std::fs::write(&dest, &payload_blob.bytes).map_err(|e| {
                format!(
                    "raw encode: failed to write `.raw` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        if let (Some(sidecar_blob), Some(name_blob)) = (
            bundle.sidecar_first("raw_sidecar_json"),
            bundle.sidecar_first("raw_sidecar_filename"),
        ) {
            let name = std::str::from_utf8(&name_blob.bytes)
                .map_err(|e| format!("raw encode: raw_sidecar_filename not UTF-8: {e}"))?;
            let dest = lml_parent.join(name);
            std::fs::write(&dest, &sidecar_blob.bytes).map_err(|e| {
                format!(
                    "raw encode: failed to write sidecar JSON preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        for sibling in find_sidecars(raw_path) {
            if let Some(name) = sibling.file_name() {
                let dest = lml_parent.join(name);
                std::fs::copy(&sibling, &dest).map_err(|e| {
                    format!(
                        "raw encode: sibling copy failed: {} -> {}: {}",
                        sibling.display(),
                        dest.display(),
                        e
                    )
                })?;
            }
        }
    }

    let _ = t0;
    Ok(EncodeMetrics {
        raw_size: original_size as usize,
        compressed_size: compressed_size as usize,
        cr,
        sha256: sha256_hex,
        verified: verify,
        samples: (n_channels as u64).saturating_mul(n_samples as u64),
        duration_s: bundle.duration_s,
        n_channels,
        sample_rate: sample_rate as f32,
        n_windows: stats.n_windows as u32,
    })
}

/// Phase 4.4 — encode a NeuroScan CNT (`.cnt`) recording.
fn encode_one_cnt(
    cnt_path: &Path,
    lml_path: &Path,
    verify: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::container;
    use lamquant_core::source::{CntReader, SignalSourceReader};
    let t0 = Instant::now();
    let mut reader = CntReader::new(cnt_path);
    let bundle = reader.read_bundle()?;
    let n_channels = bundle.signal.len() as u32;
    let n_samples = bundle.signal.first().map(|c| c.len()).unwrap_or(0);
    let sample_rate = bundle.sample_rate;

    let mut hasher = Sha256::new();
    for ch in &bundle.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    // v1.1: source `.cnt` is now preserved as a separate LMA entry
    // (default mode) or sibling file (--no-bundle), not as a b64-zstd
    // blob inside metadata JSON. Drops cnt-sized payloads from the
    // .lml header; legacy v1 reader fallback handles old archives.

    let mut ch_json = String::from("[");
    for (i, name) in bundle.channels.iter().enumerate() {
        if i > 0 {
            ch_json.push(',');
        }
        let safe = name
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        ch_json.push('"');
        ch_json.push_str(&safe);
        ch_json.push('"');
    }
    ch_json.push(']');

    let metadata_json = format!(
        "{{\"source_file\":\"{}\",\"format\":\"CNT\",\"n_channels\":{},\
         \"sample_rate\":{},\"channels\":{},\
         \"signal_sha256\":\"{}\",\"encoder\":\"lml/{}\",\"noise_bits\":{}}}",
        bundle
            .metadata
            .source_file
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
        n_channels,
        sample_rate,
        ch_json,
        sha256_hex,
        env!("CARGO_PKG_VERSION"),
        noise_bits,
    );

    if let Some(parent) = lml_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(lml_path)?);
    let stats = container::write_into(
        &mut sink,
        &bundle.signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "encode cnt: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    let original_size = std::fs::metadata(cnt_path).map(|m| m.len()).unwrap_or(0);
    let compressed_size = stats.compressed_size as u64;
    let cr = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if verify {
        let (recovered, _meta) = container::read_file(lml_path)?;
        for (i, (a, b)) in recovered.iter().zip(bundle.signal.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "verify (cnt): channel {i} mismatch ({} vs {} samples)",
                    a.len(),
                    b.len()
                )
                .into());
            }
        }
    }

    // Byte-exact preservation of the source `.cnt` plus any stem-
    // matched sibling annotation files (e.g. `recording_events.csv`,
    // `recording.lbl`). Same pattern as the EDF / BV / EEGLAB
    // encoders. b64-in-metadata stays for v1 reader back-compat;
    // future cleanup drops it from new writes.
    let lml_parent = lml_path.parent().unwrap_or_else(|| Path::new("."));
    let cnt_parent = cnt_path.parent().unwrap_or_else(|| Path::new("."));
    let same_dir = match (lml_parent.canonicalize(), cnt_parent.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir {
        if let Some(cnt_blob) = bundle.sidecar_first("cnt_raw") {
            let cnt_name = cnt_path
                .file_name()
                .ok_or("cnt encode: cnt_path has no file_name")?;
            let dest = lml_parent.join(cnt_name);
            std::fs::write(&dest, &cnt_blob.bytes).map_err(|e| {
                format!(
                    "cnt encode: failed to write `.cnt` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        for sidecar in find_sidecars(cnt_path) {
            if let Some(name) = sidecar.file_name() {
                let dest = lml_parent.join(name);
                std::fs::copy(&sidecar, &dest).map_err(|e| {
                    format!(
                        "cnt encode: sibling copy failed: {} -> {}: {}",
                        sidecar.display(),
                        dest.display(),
                        e
                    )
                })?;
            }
        }
    }

    let _ = t0;
    Ok(EncodeMetrics {
        raw_size: original_size as usize,
        compressed_size: compressed_size as usize,
        cr,
        sha256: sha256_hex,
        verified: verify,
        samples: (n_channels as u64).saturating_mul(n_samples as u64),
        duration_s: bundle.duration_s,
        n_channels,
        sample_rate: sample_rate as f32,
        n_windows: stats.n_windows as u32,
    })
}

/// Phase 8 / Item A — encode a DICOM Waveform recording (`.dcm`).
#[cfg(feature = "dicom")]
fn encode_one_dicom(
    dcm_path: &Path,
    lml_path: &Path,
    verify: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::container;
    use lamquant_core::source::{DicomWaveformReader, SignalSourceReader};
    let t0 = Instant::now();
    let mut reader = DicomWaveformReader::new(dcm_path);
    let bundle = reader.read_bundle()?;
    let n_channels = bundle.signal.len() as u32;
    let n_samples = bundle.signal.first().map(|c| c.len()).unwrap_or(0);
    let sample_rate = bundle.sample_rate;

    let mut hasher = Sha256::new();
    for ch in &bundle.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    // v1.1: source `.dcm` is now preserved as a separate LMA entry
    // (default mode) or sibling file (--no-bundle), not as a b64-zstd
    // blob inside metadata JSON. Multi-MB DICOM files no longer
    // double their footprint in the .lml header.

    let mut ch_json = String::from("[");
    for (i, name) in bundle.channels.iter().enumerate() {
        if i > 0 {
            ch_json.push(',');
        }
        let safe = name
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        ch_json.push('"');
        ch_json.push_str(&safe);
        ch_json.push('"');
    }
    ch_json.push(']');

    let metadata_json = format!(
        "{{\"source_file\":\"{}\",\"format\":\"DICOM_WAVEFORM\",\"n_channels\":{},\
         \"sample_rate\":{},\"channels\":{},\
         \"signal_sha256\":\"{}\",\"encoder\":\"lml/{}\",\"noise_bits\":{}}}",
        bundle
            .metadata
            .source_file
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
        n_channels,
        sample_rate,
        ch_json,
        sha256_hex,
        env!("CARGO_PKG_VERSION"),
        noise_bits,
    );

    if let Some(parent) = lml_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(lml_path)?);
    let stats = container::write_into(
        &mut sink,
        &bundle.signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "encode dicom: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    let original_size = std::fs::metadata(dcm_path).map(|m| m.len()).unwrap_or(0);
    let compressed_size = stats.compressed_size as u64;
    let cr = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if verify {
        let (recovered, _meta) = container::read_file(lml_path)?;
        for (i, (a, b)) in recovered.iter().zip(bundle.signal.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "verify (dicom): channel {i} mismatch ({} vs {} samples)",
                    a.len(),
                    b.len()
                )
                .into());
            }
        }
    }

    // Byte-exact preservation of the source `.dcm` plus stem-matched
    // siblings (rare for DICOM but possible -- e.g. structured-report
    // companions, custom analysis sidecars). Same pattern as the
    // EDF / BV / CNT / EEGLAB encoders.
    let lml_parent = lml_path.parent().unwrap_or_else(|| Path::new("."));
    let dcm_parent = dcm_path.parent().unwrap_or_else(|| Path::new("."));
    let same_dir = match (lml_parent.canonicalize(), dcm_parent.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir {
        if let Some(dicom_blob) = bundle.sidecar_first("dicom_raw") {
            let dcm_name = dcm_path
                .file_name()
                .ok_or("dicom encode: dcm_path has no file_name")?;
            let dest = lml_parent.join(dcm_name);
            std::fs::write(&dest, &dicom_blob.bytes).map_err(|e| {
                format!(
                    "dicom encode: failed to write `.dcm` preservation copy to {}: {}",
                    dest.display(),
                    e
                )
            })?;
        }
        for sidecar in find_sidecars(dcm_path) {
            if let Some(name) = sidecar.file_name() {
                let dest = lml_parent.join(name);
                std::fs::copy(&sidecar, &dest).map_err(|e| {
                    format!(
                        "dicom encode: sibling copy failed: {} -> {}: {}",
                        sidecar.display(),
                        dest.display(),
                        e
                    )
                })?;
            }
        }
    }

    let _ = t0;
    Ok(EncodeMetrics {
        raw_size: original_size as usize,
        compressed_size: compressed_size as usize,
        cr,
        sha256: sha256_hex,
        verified: verify,
        samples: (n_channels as u64).saturating_mul(n_samples as u64),
        duration_s: bundle.duration_s,
        n_channels,
        sample_rate: sample_rate as f32,
        n_windows: stats.n_windows as u32,
    })
}

/// Phase 8 / Item B — encode an EEGLAB `.set + .fdt + .lml-meta.json`
/// triple. Default is lossless: f32 bit-cast → i64; the decoder
/// recovers exact f32 via `f32::from_bits((sample as u32))`.
/// Opt-in `lossy_int16` scales-and-clamps to i16 with sensitivity 1.0.
#[allow(clippy::too_many_arguments)]
fn encode_one_eeglab(
    set_path: &Path,
    lml_path: &Path,
    verify: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
    lossy_int16: bool,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::container;
    use lamquant_core::source::{EeglabReader, SignalSourceReader};
    let t0 = Instant::now();
    let mut reader = EeglabReader::new(set_path).with_lossy_int16(lossy_int16);
    let bundle = reader.read_bundle()?;
    let n_channels = bundle.signal.len() as u32;
    let n_samples = bundle.signal.first().map(|c| c.len()).unwrap_or(0);
    let sample_rate = bundle.sample_rate;

    let mut hasher = Sha256::new();
    for ch in &bundle.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    let mut ch_json = String::from("[");
    for (i, name) in bundle.channels.iter().enumerate() {
        if i > 0 {
            ch_json.push(',');
        }
        let safe = name
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        ch_json.push('"');
        ch_json.push_str(&safe);
        ch_json.push('"');
    }
    ch_json.push(']');

    let eeglab_dtype = if lossy_int16 {
        "lossy_i16_scaled"
    } else {
        "lossless_f32_bitcast"
    };
    let metadata_json = format!(
        "{{\"source_file\":\"{}\",\"format\":\"{}\",\"n_channels\":{},\
         \"sample_rate\":{},\"channels\":{},\"eeglab_dtype\":\"{}\",\
         \"signal_sha256\":\"{}\",\"encoder\":\"lml/{}\",\"noise_bits\":{}}}",
        bundle
            .metadata
            .source_file
            .replace('\\', "\\\\")
            .replace('"', "\\\""),
        bundle.metadata.format,
        n_channels,
        sample_rate,
        ch_json,
        eeglab_dtype,
        sha256_hex,
        env!("CARGO_PKG_VERSION"),
        noise_bits,
    );

    if let Some(parent) = lml_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut sink = std::io::BufWriter::new(std::fs::File::create(lml_path)?);
    let stats = container::write_into(
        &mut sink,
        &bundle.signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(
            kind,
            "encode eeglab: BufWriter flush failed before sync_all",
        )
    })?;
    f.sync_all()?;

    let original_size = std::fs::metadata(set_path).map(|m| m.len()).unwrap_or(0);
    let compressed_size = stats.compressed_size as u64;
    let cr = if compressed_size > 0 {
        original_size as f64 / compressed_size as f64
    } else {
        0.0
    };

    if verify {
        let (recovered, _meta) = container::read_file(lml_path)?;
        for (i, (a, b)) in recovered.iter().zip(bundle.signal.iter()).enumerate() {
            if a != b {
                return Err(format!(
                    "verify (eeglab): channel {i} mismatch ({} vs {} samples)",
                    a.len(),
                    b.len()
                )
                .into());
            }
        }
    }

    // Byte-exact preservation of the original `.set` + `.fdt`. The v1
    // reader only persisted a few JSON fields (nbchan, pnts, srate,
    // channel labels) and silently dropped every other `EEG.*` field
    // -- events, urevent, chanlocs xyz, icaweights, icasphere, reject,
    // history, etc. The bundle now carries the raw bytes; emit them
    // alongside the `.lml` so the outer `pack_archive` pulls them
    // into the per-recording `.lma` (default) or so the `--no-bundle`
    // operator gets them as siblings of the bare `.lml`. Either way,
    // `lml decode --to-eeglab` recovers both files byte-for-byte and
    // the "no data ever lost" invariant holds for EEGLAB.
    //
    // Edge case: if the encoder is invoked in single-file direct-
    // write mode where `lml_path` is the bare `.lml` next to the
    // source `.set`, we'd be re-writing the source bytes back onto
    // themselves. Skip in that case by comparing canonical parents;
    // a fs::copy(self, self) is a no-op but the second write of the
    // identical bytes is wasted I/O.
    let lml_parent = lml_path.parent().unwrap_or_else(|| Path::new("."));
    let set_parent = set_path.parent().unwrap_or_else(|| Path::new("."));
    let same_dir = match (lml_parent.canonicalize(), set_parent.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir {
        // Copy `.set`.
        if let Some(set_blob) = bundle.sidecar_first("set_raw") {
            let set_name = set_path
                .file_name()
                .ok_or("eeglab encode: set_path has no file_name")?;
            let set_dest = lml_parent.join(set_name);
            std::fs::write(&set_dest, &set_blob.bytes).map_err(|e| {
                format!(
                    "eeglab encode: failed to write `.set` preservation copy to {}: {}",
                    set_dest.display(),
                    e
                )
            })?;
        }
        // Copy `.fdt`. Stem matches `.set` stem; rebuild the path so
        // we don't fight the SignalSourceReader internal locator.
        if let Some(fdt_blob) = bundle.sidecar_first("fdt_raw") {
            let stem = set_path
                .file_stem()
                .ok_or("eeglab encode: set_path has no stem")?;
            let fdt_dest = lml_parent.join(format!("{}.fdt", stem.to_string_lossy()));
            std::fs::write(&fdt_dest, &fdt_blob.bytes).map_err(|e| {
                format!(
                    "eeglab encode: failed to write `.fdt` preservation copy to {}: {}",
                    fdt_dest.display(),
                    e
                )
            })?;
        }
    }

    let _ = t0;
    Ok(EncodeMetrics {
        raw_size: original_size as usize,
        compressed_size: compressed_size as usize,
        cr,
        sha256: sha256_hex,
        verified: verify,
        samples: (n_channels as u64).saturating_mul(n_samples as u64),
        duration_s: bundle.duration_s,
        n_channels,
        sample_rate: sample_rate as f32,
        n_windows: stats.n_windows as u32,
    })
}

fn encode_one(
    edf_path: &Path,
    lml_path: &Path,
    verify: bool,
    cross_validate: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
) -> Result<EncodeMetrics, Box<dyn std::error::Error + Send + Sync>> {
    // Phase 8.2 — span over the whole encode call. Fields render
    // into every tracing event emitted from helpers below.
    let _span = tracing::info_span!(
        "encode_one",
        path = %edf_path.display(),
        noise_bits = noise_bits,
        window_size = window_size,
        lpc_mode = ?lpc_mode,
    )
    .entered();
    // Phase 4.1 + 4.4 — dispatch on extension. BrainVision (.vhdr)
    // and Raw (.raw + sidecar) go through the bundle path; EDF/BDF
    // stays on the legacy raw-header + non-EEG-pairs path so
    // byte-exact `--to-edf` reconstruction is preserved.
    if let Some(ext) = edf_path.extension() {
        if ext.eq_ignore_ascii_case("vhdr") {
            return encode_one_brainvision(
                edf_path,
                lml_path,
                verify,
                noise_bits,
                window_size,
                lpc_mode,
            );
        }
        if ext.eq_ignore_ascii_case("raw") {
            return encode_one_raw(
                edf_path,
                lml_path,
                verify,
                noise_bits,
                window_size,
                lpc_mode,
            );
        }
        if ext.eq_ignore_ascii_case("cnt") {
            return encode_one_cnt(
                edf_path,
                lml_path,
                verify,
                noise_bits,
                window_size,
                lpc_mode,
            );
        }
        if ext.eq_ignore_ascii_case("dcm") {
            #[cfg(feature = "dicom")]
            return encode_one_dicom(
                edf_path,
                lml_path,
                verify,
                noise_bits,
                window_size,
                lpc_mode,
            );
            #[cfg(not(feature = "dicom"))]
            return Err(".dcm input requires building lml with `--features dicom`. \
                 The default `host` build doesn't pull in dicom-rs."
                .into());
        }
        if ext.eq_ignore_ascii_case("set") {
            return encode_one_eeglab(
                edf_path,
                lml_path,
                verify,
                noise_bits,
                window_size,
                lpc_mode,
                /* lossy_int16 = */ false,
            );
        }
    }
    let mut edf_data = edf::read_edf(edf_path)?;

    // SHA-256 of original signal bytes (channel-major, i64 LE — matches Python)
    let mut hasher = Sha256::new();
    for ch in &edf_data.signal {
        for &sample in ch {
            hasher.update(sample.to_le_bytes());
        }
    }
    let sha256_hex = format!("{:x}", hasher.finalize());

    // Build rich metadata with full EDF header preservation
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    // Hash the raw EDF header for FDA provenance — SHA-256 of pre-zstd bytes.
    let edf_header_sha = {
        let mut h = Sha256::new();
        h.update(edf_data.raw_header.as_slice());
        format!("{:x}", h.finalize())
    };

    // Zstd-compress the raw EDF header.
    // Audit-2026-05-11 Fix-#34: propagate zstd errors instead of
    // silently emitting an empty payload. An empty `hdr_compressed`
    // would decode to a zero-byte EDF header at unpack time, producing
    // a clinical EDF with no patient ID / channel labels / phys units —
    // a silent loss of provenance.
    let hdr_compressed = zstd::encode_all(edf_data.raw_header.as_slice(), 9)
        .map_err(|e| format!("zstd compress EDF header: {e}"))?;
    let hdr_b64 = b64.encode(&hdr_compressed);

    let encoder_version = format!("lml/{}", env!("CARGO_PKG_VERSION"));

    // Zstd-compress non-EEG channel data
    let mut non_eeg_json = String::from("{");
    for (i, (ch_idx, ch_bytes)) in edf_data.non_eeg_data.iter().enumerate() {
        // Audit-2026-05-11 Fix-#34: propagate zstd errors per channel.
        let compressed = zstd::encode_all(ch_bytes.as_slice(), 9)
            .map_err(|e| format!("zstd compress non-EEG channel {ch_idx}: {e}"))?;
        let encoded = b64.encode(&compressed);
        if i > 0 {
            non_eeg_json.push(',');
        }
        non_eeg_json.push_str(&format!("\"{}\":\"{}\"", ch_idx, encoded));
    }
    non_eeg_json.push('}');

    // Preserve any partial-record bytes at the file tail. Pre-existing
    // encoder dropped these — bit-exact roundtrip on EDFs whose data
    // section doesn't end on a record boundary was impossible. Zstd
    // compresses tiny payloads well; SHA captures the raw bytes for
    // verification without decompression.
    let trailing_sha = {
        let mut h = Sha256::new();
        h.update(edf_data.trailing_data.as_slice());
        format!("{:x}", h.finalize())
    };
    // Audit-2026-05-11 Fix-#34: propagate zstd errors. Empty
    // trailing_data legitimately produces empty payload; non-empty
    // data MUST compress successfully or we lose the partial record.
    let trailing_compressed = if edf_data.trailing_data.is_empty() {
        Vec::new()
    } else {
        zstd::encode_all(edf_data.trailing_data.as_slice(), 9)
            .map_err(|e| format!("zstd compress trailing data: {e}"))?
    };
    let trailing_b64 = b64.encode(&trailing_compressed);

    // Build channel arrays as JSON (escape backslashes BEFORE quotes)
    let escape_json = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    };
    let channels_json: Vec<String> = edf_data
        .channels
        .iter()
        .map(|c| format!("\"{}\"", escape_json(c)))
        .collect();
    let all_labels_json: Vec<String> = edf_data
        .all_labels
        .iter()
        .map(|c| format!("\"{}\"", escape_json(c)))
        .collect();
    let eeg_idx_json: Vec<String> = edf_data.eeg_indices.iter().map(|i| i.to_string()).collect();
    let phys_min_json: Vec<String> = edf_data.phys_min.iter().map(|v| format!("{}", v)).collect();
    let phys_max_json: Vec<String> = edf_data.phys_max.iter().map(|v| format!("{}", v)).collect();
    let dig_min_json: Vec<String> = edf_data.dig_min.iter().map(|v| v.to_string()).collect();
    let dig_max_json: Vec<String> = edf_data.dig_max.iter().map(|v| v.to_string()).collect();
    let ns_json: Vec<String> = edf_data
        .all_ns_per_rec
        .iter()
        .map(|v| v.to_string())
        .collect();

    let meta = format!(
        concat!(
            "{{",
            "\"source_file\":\"{}\",",
            "\"format\":\"{}\",",
            "\"channels\":[{}],",
            "\"n_channels\":{},",
            "\"n_signals_total\":{},",
            "\"sample_rate\":{},",
            "\"n_data_records\":{},",
            "\"record_duration\":{},",
            "\"phys_min\":[{}],",
            "\"phys_max\":[{}],",
            "\"dig_min\":[{}],",
            "\"dig_max\":[{}],",
            "\"phys_dim\":\"{}\",",
            "\"all_labels\":[{}],",
            "\"all_ns_per_rec\":[{}],",
            "\"eeg_channel_indices\":[{}],",
            "\"duration_s\":{},",
            "\"patient_id\":\"{}\",",
            "\"recording_info\":\"{}\",",
            "\"startdate\":\"{}\",",
            "\"edf_header\":\"{}\",",
            "\"edf_header_sha256\":\"{}\",",
            "\"encoder_version\":\"{}\",",
            "\"non_eeg_channels\":{},",
            "\"trailing_data\":\"{}\",",
            "\"trailing_data_sha256\":\"{}\",",
            "\"signal_sha256\":\"{}\"",
            "}}"
        ),
        escape_json(&edf_data.source_file),
        edf_data.format,
        channels_json.join(","),
        edf_data.n_channels,
        edf_data.n_signals_total,
        edf_data.sample_rate,
        edf_data.n_data_records,
        edf_data.record_duration,
        phys_min_json.join(","),
        phys_max_json.join(","),
        dig_min_json.join(","),
        dig_max_json.join(","),
        escape_json(&edf_data.phys_dim),
        all_labels_json.join(","),
        ns_json.join(","),
        eeg_idx_json.join(","),
        edf_data.duration_s,
        escape_json(&edf_data.patient_id),
        escape_json(&edf_data.recording_info),
        escape_json(&edf_data.startdate),
        hdr_b64,
        edf_header_sha,
        escape_json(&encoder_version),
        non_eeg_json,
        trailing_b64,
        trailing_sha,
        sha256_hex,
    );
    let stats = container::write_file_with_mode(
        lml_path,
        &edf_data.signal,
        edf_data.sample_rate,
        window_size,
        noise_bits,
        &meta,
        lpc_mode,
    )?;

    // Take signal out of edf_data, clear the heavy fields to free memory
    // before verify/cross-validate reads the file back.
    let n_channels = edf_data.n_channels;
    let signal = std::mem::take(&mut edf_data.signal);
    // Clear heavy fields explicitly
    edf_data.raw_header = Vec::new();
    edf_data.non_eeg_data = Vec::new();
    drop(edf_data);

    let mut verified = false;
    if verify {
        let (recovered, _) = container::read_file(lml_path)?;
        let mask = if noise_bits > 0 {
            !((1i64 << noise_bits) - 1)
        } else {
            !0i64
        };
        for ch in 0..n_channels {
            let orig = &signal[ch];
            let rec = &recovered[ch];
            let len = orig.len().min(rec.len());
            for i in 0..len {
                if (orig[i] & mask) != (rec[i] & mask) {
                    return Err(format!("Verify FAILED ch {} sample {}", ch, i).into());
                }
            }
        }
        verified = true;
    }

    // Feature 4: Cross-validate — decode the written file, recompute SHA-256
    if cross_validate {
        let (recovered, _) = container::read_file(lml_path)?;
        let mut dec_hasher = Sha256::new();
        let mask = if noise_bits > 0 {
            !((1i64 << noise_bits) - 1)
        } else {
            !0i64
        };
        for ch in &recovered {
            for &sample in ch {
                dec_hasher.update((sample & mask).to_le_bytes());
            }
        }
        let decoded_sha = format!("{:x}", dec_hasher.finalize());

        // Compute masked original SHA for comparison when noise_bits > 0
        let orig_sha = if noise_bits > 0 {
            let mut masked_hasher = Sha256::new();
            for ch in &signal {
                for &sample in ch {
                    masked_hasher.update((sample & mask).to_le_bytes());
                }
            }
            format!("{:x}", masked_hasher.finalize())
        } else {
            sha256_hex.clone()
        };

        if decoded_sha != orig_sha {
            return Err(format!(
                "CROSS-VALIDATE FAILED: original sha256={} decoded sha256={}",
                orig_sha, decoded_sha
            )
            .into());
        }
        verified = true;
    }

    // Sidecar preservation — the core "no silent data loss" contract.
    // Copy every sibling sidecar (TUH `.tse_bi`/`.lbl_bi`/`.csv_bi`/
    // `_summary.txt` and similar) to the same directory the `.lml`
    // landed in. Downstream the caller may pack the directory into a
    // `.lma` (default) or leave the files in place (`--no-bundle`); in
    // both modes the sidecar files must survive the encode step. See
    // `find_sidecars` and `tests/integration/test_sidecar_preservation.py`.
    if let Some(out_dir) = lml_path.parent() {
        for sidecar in find_sidecars(edf_path) {
            if let Some(name) = sidecar.file_name() {
                let dest = out_dir.join(name);
                if let Err(e) = std::fs::copy(&sidecar, &dest) {
                    // Stay loud: a copy failure must surface as an
                    // encode error rather than silently dropping the
                    // sidecar. Wire-format `.lml` is fine but the
                    // labels are missing — clinically that's an
                    // unrecoverable loss for this file.
                    return Err(format!(
                        "Sidecar copy failed: {} → {}: {}",
                        sidecar.display(),
                        dest.display(),
                        e
                    )
                    .into());
                }
            }
        }
    }

    Ok(EncodeMetrics {
        raw_size: stats.raw_size,
        compressed_size: stats.compressed_size,
        cr: stats.cr,
        sha256: sha256_hex,
        verified,
        samples: (stats.n_channels as u64).saturating_mul(stats.total_samples as u64),
        duration_s: stats.duration_s,
        n_channels: stats.n_channels as u32,
        sample_rate: if stats.duration_s.is_finite() {
            {
                if stats.duration_s > 0.0 {
                    (stats.total_samples as f64 / stats.duration_s) as f32
                } else {
                    0.0
                }
            }
        } else {
            0.0
        },
        n_windows: stats.n_windows as u32,
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn cmd_encode(
    input: &Path,
    output: Option<&Path>,
    verify: bool,
    cross_validate: bool,
    noise_bits: u8,
    window_size: usize,
    threads: usize,
    recursive: bool,
    skip_existing: bool,
    dry_run: bool,
    lma: bool,
    no_bundle: bool,
    lpc_mode: lamquant_core::lpc::LpcMode,
    fail_fast: bool,
    include_globs: &[String],
    exclude_globs: &[String],
) -> R {
    // v1.2 I — compile `--include` / `--exclude` glob sets once
    // upfront. Empty include list = include everything (standard
    // tar/gitignore semantics).
    let include_set = if include_globs.is_empty() {
        None
    } else {
        let mut b = globset::GlobSetBuilder::new();
        for pat in include_globs {
            b.add(
                globset::Glob::new(pat)
                    .map_err(|e| format!("encode: --include `{pat}` is not a valid glob ({e})"))?,
            );
        }
        Some(
            b.build()
                .map_err(|e| format!("encode: --include glob set build failed: {e}"))?,
        )
    };
    let exclude_set = if exclude_globs.is_empty() {
        None
    } else {
        let mut b = globset::GlobSetBuilder::new();
        for pat in exclude_globs {
            b.add(
                globset::Glob::new(pat)
                    .map_err(|e| format!("encode: --exclude `{pat}` is not a valid glob ({e})"))?,
            );
        }
        Some(
            b.build()
                .map_err(|e| format!("encode: --exclude glob set build failed: {e}"))?,
        )
    };
    // Phase 1.7 — `fail_fast` checked by the rayon batch loop's
    // per-file error handler below (search for FAIL_FAST_FLAG). The
    // default-continue path is preserved when the flag is unset, so
    // existing CI scripts keep their current behaviour.
    FAIL_FAST_FLAG.store(fail_fast, std::sync::atomic::Ordering::Relaxed);
    if fail_fast {
        tracing::info!("encode: --fail-fast enabled; batch aborts on first failure");
    }
    // R.3 stdin: argv `-` buffers EDF bytes from stdin into a tempfile,
    // then proceeds as a single-file encode. Tempfile is auto-deleted
    // on function exit (drop). isatty guard prevents accidental
    // `lml encode -` at a terminal (would hang reading stdin forever).
    let _stdin_buffer: Option<tempfile::NamedTempFile>;
    let input_owned: PathBuf = if input == Path::new("-") {
        use std::io::IsTerminal as _;
        if std::io::stdin().is_terminal() {
            return Err(
                "`lml encode -` requires EDF bytes on stdin; refusing to read \
                 from a terminal. Pipe a file in (`cat foo.edf | lml encode -`) \
                 or pass a path."
                    .into(),
            );
        }
        let mut tf = tempfile::Builder::new()
            .prefix("lml-stdin-")
            .suffix(".edf")
            .tempfile()
            .map_err(|e| format!("tempfile for stdin buffer: {e}"))?;
        std::io::copy(&mut std::io::stdin().lock(), tf.as_file_mut())
            .map_err(|e| format!("read stdin: {e}"))?;
        tf.as_file_mut()
            .sync_all()
            .map_err(|e| format!("sync stdin tempfile: {e}"))?;
        let p = tf.path().to_path_buf();
        _stdin_buffer = Some(tf);
        p
    } else {
        _stdin_buffer = None;
        input.to_path_buf()
    };
    let input: &Path = &input_owned;

    let edfs_all = find_edfs(input, recursive);
    if edfs_all.is_empty() {
        return Err("No EDF files found".into());
    }

    // v1.2 I — apply --include / --exclude glob filters. Excluded
    // files are named explicitly on stderr (loss is never silent);
    // include-only filter mode keeps everything matching at least
    // one include pattern. Filter applies to the relative path
    // under `input` so users can match against subdirectory
    // structure (e.g. `--include 'sub-*/eeg/*.edf'`).
    let edfs: Vec<PathBuf> = if include_set.is_none() && exclude_set.is_none() {
        edfs_all
    } else {
        let input_root = if input.is_dir() {
            input.to_path_buf()
        } else {
            input.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        let mut included: Vec<PathBuf> = Vec::with_capacity(edfs_all.len());
        let mut excluded: Vec<PathBuf> = Vec::new();
        for p in edfs_all.into_iter() {
            let rel = p.strip_prefix(&input_root).unwrap_or(&p);
            let included_by_filter = match &include_set {
                Some(set) => set.is_match(rel) || set.is_match(&p),
                None => true,
            };
            let excluded_by_filter = match &exclude_set {
                Some(set) => set.is_match(rel) || set.is_match(&p),
                None => false,
            };
            if included_by_filter && !excluded_by_filter {
                included.push(p);
            } else {
                excluded.push(p);
            }
        }
        if !excluded.is_empty() {
            // Loud stderr per excluded file. The --i-understand-
            // data-loss flag separately suppresses the bigger
            // --no-bundle warning paragraph; per-file exclusion
            // notices always print so the operator sees exactly
            // what was filtered out.
            eprintln!(
                "note: filter excluded {} file(s) from the encode set:",
                excluded.len()
            );
            for p in &excluded {
                eprintln!("  excluded: {}", p.display());
            }
        }
        if included.is_empty() {
            return Err(format!(
                "No files matched the --include / --exclude filters \
                 ({} EDFs in input were all excluded). \
                 Check your glob patterns.",
                excluded.len()
            )
            .into());
        }
        included
    };

    // Feature 5: Dry-run — scan inputs, estimate sizes, print summary
    if dry_run {
        let total_files = edfs.len();
        let total_raw: u64 = edfs
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let est_compressed = (total_raw as f64 / 2.26) as u64;
        let est_seconds = total_files as f64 / 12.0;
        println!("Dry-run summary:");
        println!("  Files:              {}", total_files);
        println!("  Total raw size:     {}", human_bytes(total_raw));
        println!(
            "  Est. compressed:    {} (at 2.26:1 avg CR)",
            human_bytes(est_compressed)
        );
        println!("  Est. time:          {}", human_duration(est_seconds));
        println!("  No files written.");
        return Ok(());
    }

    // Set thread pool.
    // Audit-2026-05-11 Fix-#30: warn-once on stderr if the global pool
    // is already initialised. Previously `.ok()` swallowed the error
    // so a daemon / test that re-entered this code path silently used
    // the prior thread count, making `--threads N` look like it
    // applied when it didn't.
    if threads > 0 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            eprintln!(
                "Note: rayon global thread pool already initialised; --threads {} ignored ({})",
                threads, e
            );
        }
    }

    // Output routing:
    //   - --lma flag: encode into a tempfile::TempDir, then archive
    //     the dir into the user's `-o` .lma path on success. Existing
    //     behaviour, preserved.
    //   - Batch mode (dir input OR -o pointing at a directory):
    //     encode into `<final_output>/.lamquant-staging/`. On full
    //     success, atomically move every entry into the final dir
    //     and remove the staging subdir. On any failure / SIGKILL,
    //     staging is left intact so the Resume panel can either
    //     resume (--skip-existing on the staging contents) or
    //     Discard (the TUI fs::remove_dir_all'd path is the entire
    //     staging subdir — predictable + bounded).
    //   - Single-file mode (one EDF + -o pointing at a non-dir):
    //     unchanged. Writes directly to the user's output path.
    let (output_dir, lma_target, staging_dir, batch_final_dir): (
        PathBuf,
        Option<PathBuf>,
        Option<tempfile::TempDir>,
        Option<PathBuf>,
    ) = if lma {
        let lma_path = output
            .map(|p| p.to_path_buf())
            .ok_or("--lma requires an explicit -o <path.lma> output")?;
        // Co-locate staging on the output volume (NOT /tmp). /tmp is
        // often a tmpfs RAM-disk and overflows on corpora bigger
        // than RAM headroom (CHB-MIT 43 GB, TUEG hundreds of GB).
        // Failure mode caught: encode_lma against full CHB-MIT
        // exhausted /tmp partway through pack and reported the dash-
        // board as "100% then failed" with the actual ENOSPC error
        // landing in stderr but never reaching the TUI.
        let staging_parent = lma_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        std::fs::create_dir_all(&staging_parent)?;
        let staging = tempfile::Builder::new()
            .prefix(".lamquant-staging-")
            .tempdir_in(&staging_parent)?;
        let dir = staging.path().to_path_buf();
        (dir, Some(lma_path), Some(staging), None)
    } else {
        let final_dir = output
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf());
        let single_file_mode = edfs.len() == 1 && output.is_some_and(|p| !p.is_dir());
        if single_file_mode {
            // Single-file: direct write, no staging subdir.
            (final_dir, None, None, None)
        } else {
            // Batch: route through .lamquant-staging.
            std::fs::create_dir_all(&final_dir)?;
            let staging = final_dir.join(".lamquant-staging");
            std::fs::create_dir_all(&staging)?;
            (staging, None, None, Some(final_dir))
        }
    };

    if edfs.len() == 1 && !lma && output.is_some_and(|p| !p.is_dir()) {
        // Single file → specific output path. In default bundle mode
        // we stage the .lml + sidecars into a tempdir then pack into
        // the user's `-o <stem>.lma` path; in `--no-bundle` mode we
        // write directly to the user's `-o <stem>.lml` path and copy
        // sidecars to the same parent dir.
        let user_out = output.unwrap_or(&output_dir);
        let t0 = Instant::now();
        let m = if no_bundle {
            encode_one(
                &edfs[0],
                user_out,
                verify,
                cross_validate,
                noise_bits,
                window_size,
                lpc_mode,
            )?
        } else {
            // Default: per-EDF `.lma` bundling. Stage to a tempdir so
            // sidecars from this EDF land in their own private space
            // and the lma pack is byte-deterministic on its contents.
            let staging = tempfile::tempdir()?;
            let stem = edfs[0]
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("input has no file stem")?;
            let staging_lml = staging.path().join(format!("{}.lml", stem));
            let m = encode_one(
                &edfs[0],
                &staging_lml,
                verify,
                cross_validate,
                noise_bits,
                window_size,
                lpc_mode,
            )?;
            // The user's `-o` is the final `.lma` path. Pack the tempdir
            // contents (this EDF's .lml + its sibling sidecars) into it.
            if let Some(parent) = user_out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            lma::pack_archive(staging.path(), user_out, 9, true, None)?;
            // staging TempDir Drop here auto-cleans the scratch dir.
            m
        };
        let elapsed = t0.elapsed();
        let flags = if cross_validate {
            " [cross-validated]"
        } else if verify {
            " [verified]"
        } else {
            ""
        };
        println!(
            "Encoded: {:.2}:1 CR ({} → {} bytes) in {:.0}ms{}  sha256:{}",
            m.cr,
            m.raw_size,
            m.compressed_size,
            elapsed.as_millis(),
            flags,
            m.sha256
        );
        if EMIT_JSON.load(std::sync::atomic::Ordering::Relaxed) {
            emit_file_done(
                &edfs[0].display().to_string(),
                true,
                elapsed.as_millis() as u64,
                Some(m.cr),
                Some(m.raw_size as u64),
                Some(m.compressed_size as u64),
                Some(m.samples),
                Some(m.duration_s),
                Some(m.n_channels),
                Some(m.sample_rate),
                Some(m.sha256.clone()),
                Some(m.n_windows),
            );
        }
        return Ok(());
    }

    // Batch mode
    std::fs::create_dir_all(&output_dir)?;
    let total = edfs.len();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let t0 = Instant::now();

    // Seed the TUI dashboard with total file count. The Started event
    // fires in main() before cmd_encode runs (total unknown at that
    // point), so without this Progress kick the dashboard shows
    // "0 / 0 files" indefinitely.
    if EMIT_JSON.load(Ordering::Relaxed) {
        emit_progress(0, total as u64, format!("0/{}", total));
    }

    println!(
        "Encoding {} files → {} (threads: {})",
        total,
        output_dir.display(),
        if threads == 0 {
            "auto".into()
        } else {
            threads.to_string()
        }
    );

    // Input root: if input is a directory, use it as root for relative paths.
    // If input is a file, use its parent.
    let input_root = if input.is_dir() {
        input.to_path_buf()
    } else {
        input.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    // Pre-create all output directories before the parallel loop.
    // Avoids per-file create_dir_all contention in the threadpool.
    {
        let mut dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        for edf_path in &edfs {
            let out = make_output_path(edf_path, &input_root, &output_dir);
            if let Some(parent) = out.parent() {
                dirs.insert(parent.to_path_buf());
            }
        }
        for dir in &dirs {
            std::fs::create_dir_all(dir)?;
        }
    }

    let skipped = AtomicUsize::new(0);
    let total_raw_bytes = AtomicUsize::new(0);

    // Feature 2: Collect results for JSON manifest
    let results: Mutex<Vec<FileResult>> = Mutex::new(Vec::with_capacity(total));

    // Feature 3: Audit log — line-buffered, crash-safe
    let audit_path = output_dir.join("audit.log");
    let audit: Mutex<BufWriter<std::fs::File>> =
        Mutex::new(BufWriter::new(std::fs::File::create(&audit_path)?));

    // Feature 4: State file counter for periodic checkpoint
    let state_counter = AtomicUsize::new(0);
    let state_failed = &failed; // alias for state writes
    let state_skipped = &skipped;
    let state_output_dir = output_dir.clone();

    // Memory budget: limit concurrent memory usage to prevent OOM on large corpora.
    // Each file needs ~4x its size in memory (EDF read + i64 signal + verify readback).
    const MEM_BUDGET: usize = 32 * 1024 * 1024 * 1024; // 32 GB
    let mem_in_use = AtomicUsize::new(0);
    // Audit-2026-05-11 Fix-#50: count worker panics so the operator
    // sees structured "N panicked" in the summary instead of rayon
    // silently dropping the failure. The body is wrapped in
    // `catch_unwind`; existing `failed` counter still tracks
    // returned-Err paths separately.
    let panicked = AtomicUsize::new(0);

    edfs.par_iter().for_each(|edf_path| {
        // Phase 1.7 — fail-fast short-circuit. Once any worker has
        // reported a failure, subsequent files are skipped without
        // touching the codec hot path. AtomicBool::Relaxed is fine:
        // we only need monotonic eventual-visibility.
        if FAIL_FAST_FLAG.load(Ordering::Relaxed) && failed.load(Ordering::Relaxed) > 0 {
            done.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // Wrap the worker body in catch_unwind so a panic in one
        // file's encode does not gag the rest of the run AND surfaces
        // to the final summary as a structured count.
        let edf_path_for_panic = edf_path.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let out = make_output_path(edf_path, &input_root, &output_dir);
            let now_str = now_utc();

            // Tier 3 audit (O10d for encode-batch): atomic
            // skip_existing via create_new + O_EXCL. Same pattern
            // as cmd_decode batch at line 4878 -- closes the
            // TOCTOU window between `out.exists()` and the
            // subsequent encode-write that would truncate.
            if skip_existing {
                match std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&out)
                {
                    Ok(file) => {
                        drop(file);
                        let _ = std::fs::remove_file(&out);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        done.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(e) => {
                        eprintln!("  FAIL pre-create check {}: {}", out.display(), e);
                        failed.fetch_add(1, Ordering::Relaxed);
                        done.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }
            }

            // Memory throttle: wait if budget exceeded
            let file_bytes = std::fs::metadata(edf_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            let mem_needed = file_bytes.saturating_mul(4).max(1024 * 1024); // 4x file size, min 1MB
            loop {
                let current = mem_in_use.load(Ordering::Acquire);
                if current + mem_needed <= MEM_BUDGET {
                    if mem_in_use
                        .compare_exchange_weak(
                            current,
                            current + mem_needed,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        break;
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }

            let source_rel = edf_path
                .strip_prefix(&input_root)
                .unwrap_or(edf_path)
                .display()
                .to_string();
            // `output_rel` is computed further down — after the `out`
            // shadow that swaps the staging-side `.lml` path for the real
            // destination (`.lma` in bundle mode, `.lml` otherwise).

            let file_t0 = Instant::now();
            // Per-EDF output routing:
            //   * default (per-EDF bundle): stage `.lml` + sidecars into a
            //     fresh tempdir, then pack into `<final_dir>/<stem>.lma`.
            //   * `--no-bundle`: write `.lml` + sidecars directly into the
            //     final dir; encode_one's sidecar copy takes care of the
            //     latter.
            //   * `--lma` (corpus-wide single archive): write into the
            //     shared staging dir as before; the outer pack step at
            //     end-of-batch sweeps the whole staging tree.
            //
            // For default mode the per-file telemetry reports the `.lma`
            // path; for no_bundle/lma it reports the `.lml` path. Both
            // are visible to the operator on stdout.
            let (encode_result, reported_out) = if !lma && !no_bundle {
                let staging = match tempfile::tempdir() {
                    Ok(t) => t,
                    Err(e) => {
                        let err_msg = format!("tempdir: {}", e);
                        eprintln!(
                            "  [{}/{}] FAIL {} ({})",
                            done.load(Ordering::Relaxed) + 1,
                            total,
                            source_rel,
                            err_msg
                        );
                        failed.fetch_add(1, Ordering::Relaxed);
                        done.fetch_add(1, Ordering::Relaxed);
                        mem_in_use.fetch_sub(mem_needed, Ordering::Release);
                        return;
                    }
                };
                let lml_name = out.file_name().map(|n| n.to_owned()).unwrap_or_default();
                let stage_lml = staging.path().join(&lml_name);
                let lma_out = out.with_extension("lma");
                let r = encode_one(
                    edf_path,
                    &stage_lml,
                    verify,
                    cross_validate,
                    noise_bits,
                    window_size,
                    lpc_mode,
                )
                .and_then(|m| {
                    if let Some(p) = lma_out.parent() {
                        std::fs::create_dir_all(p)?;
                    }
                    lma::pack_archive(staging.path(), &lma_out, 9, true, None)
                        .map(|_| m)
                        .map_err(|e| {
                            Box::<dyn std::error::Error + Send + Sync>::from(format!(
                                "lma pack: {}",
                                e
                            ))
                        })
                });
                // staging TempDir dropped at end of this scope → cleanup.
                (r, lma_out)
            } else {
                let r = encode_one(
                    edf_path,
                    &out,
                    verify,
                    cross_validate,
                    noise_bits,
                    window_size,
                    lpc_mode,
                );
                (r, out.clone())
            };
            // Shadow `out` so the rest of the OK/Err branches report the
            // final destination (`.lma` or `.lml`) the operator will see.
            let out = reported_out;
            let output_rel = out
                .strip_prefix(&output_dir)
                .unwrap_or(&out)
                .display()
                .to_string();
            match encode_result {
                Ok(m) => {
                    let file_ms = file_t0.elapsed().as_millis() as u64;
                    total_raw_bytes.fetch_add(m.raw_size, Ordering::Relaxed);
                    // Feature 3: Audit log — OK line
                    {
                        let out_fname = out
                            .file_name()
                            .map(|f| f.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let line = format!(
                            "{} OK {} \u{2192} {} {}\u{2192}{} {:.2}:1 sha256:{}\n",
                            now_str,
                            source_rel,
                            out_fname,
                            m.raw_size,
                            m.compressed_size,
                            m.cr,
                            m.sha256
                        );
                        // Audit-2026-05-11 Fix-#32: recover from poisoned
                        // mutex via `into_inner` so a previous-panic
                        // poisoning does not silently drop subsequent audit
                        // writes. Audit log is append-only and ordering is
                        // not load-bearing — the panic that caused the
                        // poison is already lost; we want the current write
                        // to still land.
                        let mut w = match audit.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        let _ = w.write_all(line.as_bytes());
                        let _ = w.flush();
                    }

                    // Per-file FileDone OpEvent (consumed by the TUI dashboard).
                    if EMIT_JSON.load(Ordering::Relaxed) {
                        emit_file_done(
                            &source_rel,
                            true,
                            file_ms,
                            Some(m.cr),
                            Some(m.raw_size as u64),
                            Some(m.compressed_size as u64),
                            Some(m.samples),
                            Some(m.duration_s),
                            Some(m.n_channels),
                            Some(m.sample_rate),
                            Some(m.sha256.clone()),
                            Some(m.n_windows),
                        );
                    }

                    // Feature 2: Collect result
                    if let Ok(mut res) = results.lock() {
                        res.push(FileResult {
                            source: source_rel,
                            output: output_rel,
                            raw_bytes: m.raw_size,
                            compressed_bytes: m.compressed_size,
                            cr: m.cr,
                            sha256: m.sha256,
                            verified: m.verified,
                            error: None,
                        });
                    }
                }
                Err(e) => {
                    let file_ms = file_t0.elapsed().as_millis() as u64;
                    let err_msg = format!("{}", e);
                    eprintln!(
                        "  FAIL {}: {} — preserving original",
                        edf_path.display(),
                        err_msg
                    );

                    // NEVER LOSE DATA: copy the original file as-is if compression fails
                    if let Some(parent) = out.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let raw_out = out.with_extension("edf");
                    if let Err(copy_err) = std::fs::copy(edf_path, &raw_out) {
                        eprintln!(
                            "  CRITICAL: cannot preserve {}: {}",
                            edf_path.display(),
                            copy_err
                        );
                    }

                    failed.fetch_add(1, Ordering::Relaxed);

                    // Feature 3: Audit log — FAIL line (with preservation note)
                    {
                        let line = format!(
                            "{} FAIL {}: {} (original preserved as .edf)\n",
                            now_str, source_rel, err_msg
                        );
                        // Audit-2026-05-11 Fix-#32: recover from poisoned
                        // mutex via `into_inner` so a previous-panic
                        // poisoning does not silently drop subsequent audit
                        // writes. Audit log is append-only and ordering is
                        // not load-bearing — the panic that caused the
                        // poison is already lost; we want the current write
                        // to still land.
                        let mut w = match audit.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        let _ = w.write_all(line.as_bytes());
                        let _ = w.flush();
                    }

                    // Per-file FileDone OpEvent (failed). bytes_*/cr left
                    // None — the file never compressed. duration_s/etc.
                    // also unavailable since encode_one bailed early.
                    if EMIT_JSON.load(Ordering::Relaxed) {
                        emit_file_done(
                            &source_rel,
                            false,
                            file_ms,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                        );
                    }

                    // Feature 2: Collect failed result
                    if let Ok(mut res) = results.lock() {
                        res.push(FileResult {
                            source: source_rel,
                            output: output_rel,
                            raw_bytes: 0,
                            compressed_bytes: 0,
                            cr: 0.0,
                            sha256: String::new(),
                            verified: false,
                            error: Some(err_msg),
                        });
                    }
                }
            }

            // Release memory budget after processing
            mem_in_use.fetch_sub(mem_needed, Ordering::Release);

            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            // Emit a Progress event after every file when JSON streaming
            // is on so the TUI dashboard's progress bar + ETA math have
            // current values. Cheap (one JSON line per file).
            if EMIT_JSON.load(Ordering::Relaxed) {
                emit_progress(n as u64, total as u64, format!("{}/{}", n, total));
            }
            if n % 100 == 0 || n == total {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = n as f64 / elapsed;
                let eta = (total - n) as f64 / rate;
                let pct = 100.0 * n as f64 / total as f64;
                let bytes_done = total_raw_bytes.load(Ordering::Relaxed) as f64;
                let mbps = bytes_done / (1024.0 * 1024.0) / elapsed;
                eprint!(
                    "\r  [{}/{}] {:.1}%  {:.1} files/s  {:.1} MB/s  ETA {}  elapsed {}   ",
                    n,
                    total,
                    pct,
                    rate,
                    mbps,
                    human_duration(eta),
                    human_duration(elapsed)
                );

                // Feature 4: State file every 100 files
                let sc = state_counter.fetch_add(1, Ordering::Relaxed);
                if sc % 1 == 0 {
                    // Write state (n already accounts for this file)
                    let f_now = state_failed.load(Ordering::Relaxed);
                    let s_now = state_skipped.load(Ordering::Relaxed);
                    let state_json = format!(
                        "{{\"completed\":{},\"failed\":{},\"skipped\":{},\"last_update\":\"{}\"}}",
                        n - f_now - s_now,
                        f_now,
                        s_now,
                        now_utc()
                    );
                    let state_path = state_output_dir.join(".lamquant_state.json");
                    // Audit-2026-05-11 Fix-#33: warn on state-write IO errors
                    // (full disk, permission revoked mid-encode). Best-effort
                    // operation, but operators need diagnostic when the state
                    // file silently goes stale.
                    if let Err(e) = std::fs::write(&state_path, state_json.as_bytes()) {
                        eprintln!(
                            "WARNING: state write to {} failed: {}",
                            state_path.display(),
                            e
                        );
                    }
                }
            }
        }));
        if let Err(panic_info) = result {
            // Audit-2026-05-11 Fix-#50: structured panic record.
            panicked.fetch_add(1, Ordering::Relaxed);
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            eprintln!(
                "PANIC in worker processing {}: {}",
                edf_path_for_panic.display(),
                panic_msg
            );
        }
    });
    eprintln!();

    // Feature 2: Write manifest.lml.json
    let f = failed.load(Ordering::Relaxed);
    let s = skipped.load(Ordering::Relaxed);
    let elapsed = t0.elapsed();
    let p = panicked.load(Ordering::Relaxed);
    if p > 0 {
        eprintln!(
            "WARNING: {} worker(s) panicked during encode; output may be incomplete.",
            p
        );
    }

    {
        let res = results.lock().unwrap();
        let total_raw: usize = res.iter().map(|r| r.raw_bytes).sum();
        let total_comp: usize = res.iter().map(|r| r.compressed_bytes).sum();
        let overall_cr = if total_comp > 0 {
            total_raw as f64 / total_comp as f64
        } else {
            0.0
        };

        let mut files_json = String::from("[");
        for (i, r) in res.iter().enumerate() {
            if i > 0 {
                files_json.push(',');
            }
            if let Some(ref err) = r.error {
                let escaped = err.replace('\\', "\\\\").replace('"', "\\\"");
                files_json.push_str(&format!(
                    concat!(
                        "{{\"source\":\"{}\",\"output\":\"{}\",",
                        "\"raw_bytes\":{},\"compressed_bytes\":{},\"cr\":{:.4},",
                        "\"sha256\":\"{}\",\"verified\":{},\"error\":\"{}\"}}"
                    ),
                    r.source,
                    r.output,
                    r.raw_bytes,
                    r.compressed_bytes,
                    r.cr,
                    r.sha256,
                    r.verified,
                    escaped
                ));
            } else {
                files_json.push_str(&format!(
                    concat!(
                        "{{\"source\":\"{}\",\"output\":\"{}\",",
                        "\"raw_bytes\":{},\"compressed_bytes\":{},\"cr\":{:.4},",
                        "\"sha256\":\"{}\",\"verified\":{}}}"
                    ),
                    r.source, r.output, r.raw_bytes, r.compressed_bytes, r.cr, r.sha256, r.verified
                ));
            }
        }
        files_json.push(']');

        let manifest = format!(
            concat!(
                "{{\"version\":\"1.0\",\"encoder\":\"lml 0.2.0 (Rust)\",",
                "\"created\":\"{}\",",
                "\"n_files\":{},\"n_failed\":{},\"n_skipped\":{},",
                "\"total_raw_bytes\":{},\"total_compressed_bytes\":{},",
                "\"overall_cr\":{:.4},",
                "\"files\":{}}}"
            ),
            now_utc(),
            total - s,
            f,
            s,
            total_raw,
            total_comp,
            overall_cr,
            files_json
        );

        let manifest_path = output_dir.join("manifest.lml.json");
        if let Err(e) = std::fs::write(&manifest_path, manifest.as_bytes()) {
            eprintln!("Warning: failed to write manifest: {}", e);
        }
    }

    // Feature 4: Final state file
    {
        let state_json = format!(
            "{{\"completed\":{},\"failed\":{},\"skipped\":{},\"last_update\":\"{}\"}}",
            total - f - s,
            f,
            s,
            now_utc()
        );
        let state_path = output_dir.join(".lamquant_state.json");
        // Audit-2026-05-11 Fix-#33: warn on state-write IO error.
        if let Err(e) = std::fs::write(&state_path, state_json.as_bytes()) {
            eprintln!(
                "WARNING: final state write to {} failed: {}",
                state_path.display(),
                e
            );
        }
    }

    println!(
        "Done: {} succeeded, {} failed, {} skipped ({:.1}s)",
        total - f - s,
        f,
        s,
        elapsed.as_secs_f64()
    );

    // --lma flag: archive the staging dir into the user's .lma path.
    // Only runs on a fully-successful batch (no failed files) so the
    // archive never contains a partial encode. On any failure the
    // staging dir is preserved via TempDir::into_path so the user can
    // salvage what completed.
    if let (Some(target), Some(staging)) = (lma_target.as_ref(), staging_dir) {
        if f > 0 {
            // `keep()` consumes the TempDir without running its
            // Drop-time cleanup, so the staging dir survives for
            // post-mortem recovery of any files that did encode.
            let kept = staging.keep();
            eprintln!(
                "  --lma: skipped archive ({} file(s) failed). Staging kept at {}",
                f,
                kept.display()
            );
            std::process::exit(1);
        }
        let staging_path = staging.path().to_path_buf();
        println!("Packing {} → {}", staging_path.display(), target.display());
        let summary = lma::pack_archive(&staging_path, target, 9, true, None)?;
        println!(
            "  {} files ({} LML, {} zstd, {} stored)",
            summary.n_files, summary.counts_lml, summary.counts_zstd, summary.counts_store
        );
        // staging TempDir is dropped here → temp dir auto-cleaned.
    }

    // Batch-mode finalize: move every entry from .lamquant-staging
    // into the user's final output dir, then remove the (now-empty)
    // staging subdir. Only fires on full success (no failed files);
    // on failure the staging dir stays so Resume can re-attempt.
    if let Some(final_dir) = batch_final_dir.as_ref() {
        if f == 0 {
            finalize_staging(&output_dir, final_dir)?;
        } else {
            eprintln!(
                "  Note: {} file(s) failed. Partial outputs preserved at {}",
                f,
                output_dir.display()
            );
        }
    }

    if f > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Move every entry from `staging` into `final_dir`, then remove the
/// (now-empty) staging directory. Uses `fs::rename` (atomic within the
/// same filesystem — which holds because staging is always a subdir of
/// final_dir). On rename failure the function bails immediately
/// without touching the remaining entries, so the staging dir survives
/// for inspection.
fn finalize_staging(staging: &Path, final_dir: &Path) -> R {
    debug_assert!(staging.is_dir(), "finalize_staging: staging must be a dir");
    debug_assert!(
        final_dir.is_dir(),
        "finalize_staging: final_dir must be a dir"
    );
    for entry in std::fs::read_dir(staging)? {
        let entry = entry?;
        let src = entry.path();
        let dst = final_dir.join(entry.file_name());
        // If the destination already exists (e.g. interrupted previous
        // run left behind a file), remove it first — staging contents
        // are always the newest verified result.
        if dst.exists() {
            if dst.is_dir() {
                std::fs::remove_dir_all(&dst)?;
            } else {
                std::fs::remove_file(&dst)?;
            }
        }
        std::fs::rename(&src, &dst)?;
    }
    // Staging should be empty now. remove_dir refuses non-empty dirs,
    // so a stray entry produces a clear error instead of silent loss.
    std::fs::remove_dir(staging)?;
    Ok(())
}

fn find_lmls(path: &Path, recursive: bool) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files = Vec::new();
    if recursive {
        for entry in walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let p = entry.path();
            if p.is_file() {
                if let Some(ext) = p.extension() {
                    if ext.eq_ignore_ascii_case("lml") {
                        files.push(p.to_path_buf());
                    }
                }
            }
        }
    } else if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    if let Some(ext) = p.extension() {
                        if ext.eq_ignore_ascii_case("lml") {
                            files.push(p);
                        }
                    }
                }
            }
        }
    }
    files.sort();
    files
}

fn make_decode_output_path(input: &Path, input_root: &Path, output_dir: &Path) -> PathBuf {
    let relative = input.strip_prefix(input_root).unwrap_or(input);
    let mut out = output_dir.join(relative);
    out.set_extension("raw");
    out
}

/// Channel-major raw output requires every window's full channel data
/// before any byte of channel 0 can be flushed past sample W. To stream
/// in channel-major shape without holding all windows in RAM we'd need
/// to either (a) seek N times per channel or (b) buffer per-channel.
/// (a) costs N×n_ch reads; (b) costs n_ch × total_samples × 8 bytes.
///
/// Compromise: window-by-window decode (Phase 3.4), but stage each
/// channel's bytes into per-channel `BufWriter`s pointed at the same
/// destination file via positional writes — channel C writes at
/// `C * total_samples * 4 + sample_cursor * 4`. This keeps RAM at
/// `n_ch * 8KiB` (BufWriter default) instead of the full
/// `n_ch * total_samples * 8 bytes` of the legacy load-everything path,
/// at the cost of `n_ch` open file handles to the same file. For an
/// 82-channel iEEG fixture that's ~656 KiB of buffers vs. ~70 MiB.
fn decode_one_to_raw(
    lml_path: &Path,
    out_path: &Path,
) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    // ADR 0069/0071 L9: dispatch on magic so this streaming path handles
    // both the live `BCS1` wire (write_abir's default output) and old
    // `LML1` files — see `lamquant_core::bcs1_stream`.
    use lamquant_core::bcs1_stream::AnyLmlReader;
    use std::io::{BufWriter, Seek, SeekFrom, Write as _};

    // Phase 8.2 — observability span.
    let _span = tracing::info_span!(
        "decode_one_to_raw",
        path = %lml_path.display(),
        out = %out_path.display(),
    )
    .entered();

    let mut reader = AnyLmlReader::open(lml_path)?;
    let n_ch = reader.header().n_channels;
    let total_samples = reader.header().total_samples;

    // Tier 3 audit (O9): bound n_ch against fd budget. Pre-fix the
    // per-channel `OpenOptions::open` loop opened `n_ch` independent
    // file handles; an adversarial .lml claiming `n_channels =
    // 100000` would exhaust `RLIMIT_NOFILE` mid-loop and leave a
    // partial sparse output file. 4096 channels is a generous
    // clinical ceiling (highest-density real EEG montages are
    // ~256 ch).
    const MAX_DECODE_CHANNELS: usize = 4096;
    if n_ch > MAX_DECODE_CHANNELS {
        return Err(format!(
            "decode_one_to_raw: {} channels exceeds MAX_DECODE_CHANNELS={} \
             (likely corrupted .lml header)",
            n_ch, MAX_DECODE_CHANNELS
        )
        .into());
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Pre-size the file so the per-channel writers can seek into
    // their slabs. Open *separately* per channel — `File::try_clone()`
    // on Linux uses `dup`, which shares the file position across all
    // clones, so multiple BufWriters would corrupt each other's slabs.
    // `OpenOptions::open` returns an independent open file description
    // with its own position. Bible R23 — pin the assumption (each
    // writer must own its position) explicitly.
    {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(out_path)?;
        let total_bytes = (n_ch as u64) * (total_samples as u64) * 4;
        if total_bytes > 0 {
            file.set_len(total_bytes)?;
        }
        file.sync_all()?; // make set_len visible to subsequent opens
    }
    let mut channel_writers: Vec<BufWriter<std::fs::File>> = Vec::with_capacity(n_ch);
    for ch in 0..n_ch {
        let mut f = std::fs::OpenOptions::new().write(true).open(out_path)?;
        let base = (ch as u64) * (total_samples as u64) * 4;
        f.seek(SeekFrom::Start(base))?;
        channel_writers.push(BufWriter::new(f));
    }
    // Window-by-window decode. RAM cost ≈ one window worth of i64
    // samples + per-channel 8 KiB write buffer. Legacy path held the
    // entire `Vec<Vec<i64>>` signal in memory.
    //
    // Tier 3 audit (O9): track i32 truncation. Pre-fix `v as i32`
    // silently wrapped i64 samples outside the i32 range. EDF source
    // is i16/i24 so this is safe for clinical workflows; defensive
    // for accumulator inputs.
    let mut i32_clamps_low: u64 = 0;
    let mut i32_clamps_high: u64 = 0;
    while let Some(window) = reader.next_window() {
        let w = window?;
        for (ch_idx, samples) in w.iter().enumerate() {
            let writer = &mut channel_writers[ch_idx];
            for &v in samples {
                if v < i32::MIN as i64 {
                    i32_clamps_low += 1;
                } else if v > i32::MAX as i64 {
                    i32_clamps_high += 1;
                }
                let clamped: i32 = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                writer.write_all(&clamped.to_le_bytes())?;
            }
        }
    }
    if i32_clamps_low > 0 || i32_clamps_high > 0 {
        eprintln!(
            "  WARNING: decode_one_to_raw: clamped {} samples to i32 range \
             ({} below MIN, {} above MAX). Source had i64 values outside i32; \
             consider --to-edf for native dynamic range.",
            i32_clamps_low + i32_clamps_high,
            i32_clamps_low,
            i32_clamps_high
        );
    }
    for mut w in channel_writers {
        w.flush()?;
    }
    Ok((n_ch, total_samples))
}

/// Reconstruct a byte-identical EDF/BDF file from an LML container.
///
/// Inputs (from container metadata + signal payload):
///   - `raw_header` (zstd+b64, decoded → header bytes verbatim)
///   - `non_eeg_channels` (per-slot raw record bytes)
///   - `trailing_data` (partial-record bytes at file tail)
///   - `eeg_channel_indices`, `all_ns_per_rec`, `n_data_records`
///   - `format` (determines int16 vs int24 sample width)
///   - signal payload (i64 samples for the EEG slots only)
///
/// For each data record we walk the original channel slot order and:
///   - EEG slot: re-emit samples from `signal` as int16 LE (EDF/EDF+)
///     or int24 LE (BDF), preserving sign.
///   - Non-EEG slot: copy the stored record-byte slice verbatim.
///
/// Returns (n_records, n_signals). For paranoid bit-exact use,
/// SHA-256 the resulting bytes and compare against the original
/// EDF's SHA-256 — they must be identical.
fn decode_one_to_edf(
    lml_path: &Path,
    out_path: &Path,
) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let (signal, metadata) = container::read_file(lml_path)?;

    // Pull every reconstruction input from the metadata. Using the
    // helpers from cmd_roundtrip below — same JSON-pulling primitives.
    let raw_header =
        meta_b64_zstd_field(&metadata, "edf_header").ok_or("metadata missing edf_header")?;
    let trailing = meta_b64_zstd_field(&metadata, "trailing_data").unwrap_or_default();
    let non_eeg_pairs = meta_non_eeg(&metadata).ok_or("metadata non_eeg_channels parse failed")?;

    let format = meta_str_field(&metadata, "format").unwrap_or("EDF");
    let is_bdf = format == "BDF";
    let bps = if is_bdf { 3usize } else { 2 };

    let n_data_records =
        meta_str_field_num(&metadata, "n_data_records").ok_or("metadata missing n_data_records")?;

    let eeg_indices = meta_int_array(&metadata, "eeg_channel_indices")
        .ok_or("metadata missing eeg_channel_indices")?;
    let all_ns_per_rec =
        meta_int_array(&metadata, "all_ns_per_rec").ok_or("metadata missing all_ns_per_rec")?;
    let n_signals = all_ns_per_rec.len();

    if signal.len() != eeg_indices.len() {
        return Err(format!(
            "container signal channel count {} != eeg_channel_indices len {}",
            signal.len(),
            eeg_indices.len(),
        )
        .into());
    }

    // Slot → source: EEG (index into signal[]) or non-EEG (index into
    // non_eeg_pairs[]). Built once, used for every record.
    let mut non_eeg_slot_to_idx: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (i, (slot, _)) in non_eeg_pairs.iter().enumerate() {
        non_eeg_slot_to_idx.insert(*slot, i);
    }
    let mut eeg_slot_to_idx: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (i, &slot) in eeg_indices.iter().enumerate() {
        eeg_slot_to_idx.insert(slot as usize, i);
    }

    // Pre-budget the output capacity.
    let bytes_per_record: usize = all_ns_per_rec.iter().map(|n| *n as usize).sum::<usize>() * bps;
    let mut out =
        Vec::with_capacity(raw_header.len() + n_data_records * bytes_per_record + trailing.len());
    out.extend_from_slice(&raw_header);

    for r in 0..n_data_records {
        for slot in 0..n_signals {
            let ns = all_ns_per_rec[slot] as usize;
            if let Some(&k) = eeg_slot_to_idx.get(&slot) {
                // EEG channel: emit ns samples as int16 LE / int24 LE.
                let start = r * ns;
                let end = start + ns;
                if end > signal[k].len() {
                    return Err(format!(
                        "EEG channel {} only has {} samples; record {} needs [{}..{}]",
                        k,
                        signal[k].len(),
                        r,
                        start,
                        end,
                    )
                    .into());
                }
                if is_bdf {
                    for i in start..end {
                        let v = signal[k][i] as i32;
                        out.push((v & 0xff) as u8);
                        out.push(((v >> 8) & 0xff) as u8);
                        out.push(((v >> 16) & 0xff) as u8);
                    }
                } else {
                    for i in start..end {
                        let v = signal[k][i] as i16;
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            } else if let Some(&i) = non_eeg_slot_to_idx.get(&slot) {
                // Non-EEG: copy the stored record byte-slice verbatim.
                let chunk = ns * bps;
                let off = r * chunk;
                let endb = off + chunk;
                let bytes = &non_eeg_pairs[i].1;
                if endb > bytes.len() {
                    return Err(format!(
                        "non-EEG slot {} has {} bytes; record {} needs [{}..{}]",
                        slot,
                        bytes.len(),
                        r,
                        off,
                        endb,
                    )
                    .into());
                }
                out.extend_from_slice(&bytes[off..endb]);
            } else {
                return Err(format!("slot {} not classified as EEG or non-EEG", slot).into());
            }
        }
    }

    out.extend_from_slice(&trailing);

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, &out)?;
    Ok((n_data_records, n_signals))
}

/// Pull a numeric scalar from the metadata JSON: `"key":<num>`. Used by
/// decode_one_to_edf for n_data_records and similar.
fn meta_str_field_num(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse::<usize>().ok()
}

/// Pull an integer array `"key":[1,2,3]` → Vec<usize>.
fn meta_int_array(json: &str, key: &str) -> Option<Vec<u32>> {
    let needle = format!("\"{}\":[", key);
    let start = json.find(&needle)? + needle.len();
    let end = json[start..].find(']').map(|p| start + p)?;
    let body = &json[start..end];
    if body.trim().is_empty() {
        return Some(Vec::new());
    }
    body.split(',')
        .map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

/// Parse `--channels 0,2,4` into a sorted+deduplicated index list.
/// Returns the validated vec; the caller wraps in `RangeQuery::new`,
/// which re-sorts/de-dups defensively.
fn parse_channels_csv(s: &str) -> Result<Vec<usize>, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("--channels: empty list".into());
    }
    let mut out = Vec::new();
    for tok in trimmed.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            return Err(format!("--channels: empty token in '{s}'"));
        }
        let idx: usize = t
            .parse()
            .map_err(|e| format!("--channels: '{t}' is not a non-negative integer ({e})"))?;
        out.push(idx);
    }
    Ok(out)
}

/// Parse `--time-range START:END` (sample-index range, end exclusive).
/// Returns `(start, end_exclusive)`.
fn parse_time_range(s: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(format!(
            "--time-range: expected 'START:END' (sample indices), got '{s}'"
        ));
    }
    let start: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|e| format!("--time-range: start '{}' not a u32 ({e})", parts[0]))?;
    let end: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|e| format!("--time-range: end '{}' not a u32 ({e})", parts[1]))?;
    if end <= start {
        return Err(format!(
            "--time-range: end {end} must be > start {start} (range is half-open [start,end))"
        ));
    }
    Ok((start, end))
}

/// Phase 3.2 + 3.3 — single-file partial decode through `RangeReader`.
///
/// Either or both of `channels` (already sort+dedup'd) and `time_range`
/// may be `Some`. If `time_range` is None, the full sample range is
/// used. If `channels` is None, every channel is decoded.
///
/// Writes channel-major int32 LE to `out_path`. Returns
/// `(n_channels_emitted, n_samples_emitted)`.
fn decode_one_partial_to_raw(
    lml_path: &Path,
    out_path: &Path,
    channels: Option<&[usize]>,
    time_range: Option<(u32, u32)>,
) -> Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    use lamquant_core::range::{RangeQuery, RangeReader};

    // ADR 0069/0071 L9: `RangeReader::open` dispatches on magic (BCS1 vs
    // legacy LML1) instead of requiring the caller to construct an
    // `LmlReader` up front — that construction used to hard-fail with
    // `InvalidMagic` on a BCS1 file before `RangeReader::new` was ever
    // reached.
    let mut rr = RangeReader::open(lml_path)?;
    let total_samples_u32: u32 = rr.header().total_samples.try_into().map_err(|_| {
        format!(
            "decode --time-range: total_samples {} > u32::MAX",
            rr.header().total_samples
        )
    })?;
    let (start, end_exclusive) = time_range.unwrap_or((0, total_samples_u32));
    let q = RangeQuery::new(start, end_exclusive, channels.map(|c| c.to_vec()))?;
    let slice = rr.read(&q)?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let n_ch = slice.signal.len();
    let n_samples = slice.n_samples();
    let mut raw = Vec::with_capacity(n_ch * n_samples * 4);
    for ch in &slice.signal {
        for &v in ch {
            raw.extend_from_slice(&(v as i32).to_le_bytes());
        }
    }
    std::fs::write(out_path, &raw)?;
    Ok((n_ch, n_samples))
}

#[allow(clippy::too_many_arguments)]
fn cmd_decode(
    input: &Path,
    output: Option<&Path>,
    recursive: bool,
    skip_existing: bool,
    threads: usize,
    to_edf: bool,
    channels_arg: Option<&str>,
    time_range_arg: Option<&str>,
    fail_fast: bool,
) -> R {
    FAIL_FAST_FLAG.store(fail_fast, std::sync::atomic::Ordering::Relaxed);
    if fail_fast {
        tracing::info!("decode: --fail-fast enabled; batch aborts on first failure");
    }
    let lmls = find_lmls(input, recursive);
    if lmls.is_empty() {
        return Err("No LML files found".into());
    }

    // Parse partial-decode flags up front so we error on bad syntax
    // before doing any I/O.
    let channels: Option<Vec<usize>> = match channels_arg {
        Some(s) => Some(parse_channels_csv(s)?),
        None => None,
    };
    let time_range: Option<(u32, u32)> = match time_range_arg {
        Some(s) => Some(parse_time_range(s)?),
        None => None,
    };
    let partial = channels.is_some() || time_range.is_some();
    if partial && to_edf {
        return Err(
            "--channels / --time-range are incompatible with --to-edf — EDF \
             reconstruction needs every channel and sample slot. Drop --to-edf \
             or drop the partial flags."
                .into(),
        );
    }
    if partial && (lmls.len() > 1 || input.is_dir()) {
        return Err(format!(
            "--channels / --time-range require a single input LML (got {}). \
             Run partial-decode per-file in a shell loop if you need a batch.",
            lmls.len()
        )
        .into());
    }

    // R.3 stdout: `-o -` writes decoded bytes to stdout. Only the
    // single-file raw-int32 path is supported here; --to-edf to stdout
    // is refused because EDF reconstruction needs metadata round-trip
    // helpers that the current writer assumes are file-backed.
    let stdout_mode = output.map(|p| p == Path::new("-")).unwrap_or(false);
    if stdout_mode {
        if to_edf {
            return Err("`lml decode -o -` doesn't yet support --to-edf. \
                 Write to a file or drop --to-edf to get raw int32 LE on stdout."
                .into());
        }
        if lmls.len() != 1 {
            return Err(format!(
                "`lml decode -o -` requires exactly one input LML (got {}). \
                 Batch decode needs a directory output, not stdout.",
                lmls.len()
            )
            .into());
        }
        use std::io::IsTerminal as _;
        if std::io::stdout().is_terminal() {
            return Err(
                "Refusing to write binary int32 LE signal bytes to a terminal. \
                 Redirect stdout (`lml decode foo.lml -o - > samples.bin`) \
                 or pass a file path."
                    .into(),
            );
        }
        // Partial-decode stdout — funnel through RangeReader so
        // --channels / --time-range produce the trimmed payload.
        if partial {
            use lamquant_core::range::{RangeQuery, RangeReader};
            // ADR 0069/0071 L9: dispatch on magic (see decode_one_partial_to_raw).
            let mut rr = RangeReader::open(&lmls[0])?;
            let total_samples_u32: u32 = rr.header().total_samples.try_into().map_err(|_| {
                format!(
                    "decode --time-range: total_samples {} > u32::MAX",
                    rr.header().total_samples
                )
            })?;
            let (start, end_exclusive) = time_range.unwrap_or((0, total_samples_u32));
            let q = RangeQuery::new(start, end_exclusive, channels.clone())?;
            let slice = rr.read(&q)?;
            let mut out = std::io::stdout().lock();
            for ch in &slice.signal {
                for &v in ch {
                    out.write_all(&(v as i32).to_le_bytes())?;
                }
            }
            out.flush()?;
            eprintln!(
                "Decoded: {}ch x {} samples [{}, {}) -> stdout (int32 LE)",
                slice.n_channels(),
                slice.n_samples(),
                slice.start_sample,
                slice.end_sample_exclusive
            );
            return Ok(());
        }
        // Tier 3 audit (O10e): stream the stdout-mode dump window-
        // by-window instead of loading the entire signal into RAM.
        // Pre-fix `container::read_file` materialised the whole
        // Vec<Vec<i64>> -- multi-GiB inputs OOM'd even when the
        // user just wanted to pipe a few hundred MiB to a tool.
        // Channel-major: emit all of ch0, then all of ch1, ... so
        // the byte layout matches decode_one_to_raw. We get there
        // by accumulating per-channel into BufWriter slabs (sized
        // by total_samples since stdout is unseekable); RAM cost
        // ~ n_ch * 4 bytes * window_size during the loop, plus
        // the final per-channel buffer. For 256-ch × 24h × 256 Hz
        // EEG (44 GiB raw) this is still a multi-GiB allocation
        // -- not OK. For ultra-large inputs the caller should
        // re-encode at a lower window count or use `-o <path>` +
        // a streaming consumer (cat / pv) instead.
        // ADR 0069/0071 L9: dispatch on magic (see decode_one_to_raw).
        use lamquant_core::bcs1_stream::AnyLmlReader;
        use std::io::Write as _;
        let mut reader = AnyLmlReader::open(&lmls[0])?;
        let n_ch = reader.header().n_channels;
        let total_samples = reader.header().total_samples;
        let mut channels: Vec<Vec<i32>> = (0..n_ch)
            .map(|_| Vec::with_capacity(total_samples))
            .collect();
        while let Some(window) = reader.next_window() {
            let w = window?;
            for (ch_idx, samples) in w.iter().enumerate() {
                let dst = &mut channels[ch_idx];
                for &v in samples {
                    let clamped = v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                    dst.push(clamped);
                }
            }
        }
        let mut out = std::io::stdout().lock();
        for ch in &channels {
            for &v in ch {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        out.flush()?;
        eprintln!(
            "Decoded: {}ch x {} samples -> stdout (int32 LE)",
            n_ch, total_samples
        );
        return Ok(());
    }

    // Single file mode
    if lmls.len() == 1 && !input.is_dir() {
        let default_ext = if to_edf { "edf" } else { "raw" };
        let out_path = output
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| lmls[0].with_extension(default_ext));
        // Tier 3 audit (O10 follow-up): refuse output==input.
        // Single-file decode uses `set_len(total_bytes)` (line ~4255)
        // which TRUNCATES the output BEFORE the read finishes. If
        // -o equals the input path, the .lml is destroyed before
        // LmlReader::open even seeks to read it. Pre-fix:
        // `lml decode foo.lml -o foo.lml` silently obliterated foo.
        if out_path == lmls[0] {
            return Err(format!(
                "cmd_decode: output path equals input ({}); decode would truncate the source",
                out_path.display()
            )
            .into());
        }
        if let (Ok(in_canon), Ok(out_canon)) = (lmls[0].canonicalize(), out_path.canonicalize()) {
            if in_canon == out_canon {
                return Err(format!(
                    "cmd_decode: canonical output {} equals input; refusing",
                    out_canon.display()
                )
                .into());
            }
        }
        let t0 = Instant::now();
        if to_edf {
            let (n_records, n_signals) = decode_one_to_edf(&lmls[0], &out_path)?;
            let elapsed = t0.elapsed();
            println!(
                "Decoded EDF: {} records × {} signals -> {} (byte-identical {:.1}ms)",
                n_records,
                n_signals,
                out_path.display(),
                elapsed.as_secs_f64() * 1000.0
            );
        } else if partial {
            let (n_ch, t) =
                decode_one_partial_to_raw(&lmls[0], &out_path, channels.as_deref(), time_range)?;
            let elapsed = t0.elapsed();
            let (start, end_excl) = time_range.unwrap_or((0, t as u32));
            println!(
                "Decoded: {}ch x {} samples [{}, {}) -> {} (int32 LE, {:.1}ms)",
                n_ch,
                t,
                start,
                end_excl,
                out_path.display(),
                elapsed.as_secs_f64() * 1000.0
            );
        } else {
            let (n_ch, t) = decode_one_to_raw(&lmls[0], &out_path)?;
            let elapsed = t0.elapsed();
            println!(
                "Decoded: {}ch x {} samples -> {} (int32 LE, {:.1}ms)",
                n_ch,
                t,
                out_path.display(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
        return Ok(());
    }

    // Batch mode
    if threads > 0 {
        // Tier 3 audit (O10): warn when build_global() fails so the
        // operator knows their --threads N was silently ignored.
        // Pre-fix `.ok()` discarded the AlreadyInitialized error
        // common to a second cmd_* call in the same process.
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            eprintln!(
                "  WARNING: cmd_decode: --threads {} ignored ({}); \
                 using existing rayon pool",
                threads, e
            );
        }
    }

    let output_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf());
    std::fs::create_dir_all(&output_dir)?;

    let input_root = if input.is_dir() {
        input.to_path_buf()
    } else {
        input.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    // Pre-create output directories
    {
        let mut dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        for lml_path in &lmls {
            let out = make_decode_output_path(lml_path, &input_root, &output_dir);
            if let Some(parent) = out.parent() {
                dirs.insert(parent.to_path_buf());
            }
        }
        for dir in &dirs {
            std::fs::create_dir_all(dir)?;
        }
    }

    let total = lmls.len();
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let skipped_count = AtomicUsize::new(0);
    let t0 = Instant::now();

    // Audit log -- Tier 3 audit (O10 follow-up): timestamped
    // filename so we never clobber an operator's pre-existing
    // `audit.log` in their working directory. Pre-fix:
    // unconditional `output_dir.join("audit.log")` overwrote any
    // file by that name, including the operator's own decode-
    // history file from a prior run.
    let now_stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let audit_path = output_dir.join(format!("decode_audit_{}.log", now_stamp));
    let audit: Mutex<BufWriter<std::fs::File>> = Mutex::new(BufWriter::new(
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&audit_path)
            .or_else(|_| std::fs::File::create(&audit_path))?,
    ));

    println!(
        "Decoding {} files -> {} (threads: {})",
        total,
        output_dir.display(),
        if threads == 0 {
            "auto".into()
        } else {
            threads.to_string()
        }
    );

    lmls.par_iter().for_each(|lml_path| {
        // Phase 1.7 — fail-fast short-circuit (Decode batch).
        // Tier 3 audit (O10): when fail-fast fires, emit an audit
        // line so the counter mis-attributing skipped-due-to-
        // fail-fast as "succeeded" is fixed. Pre-fix `return`
        // without writing to audit.log meant the audit trail
        // silently under-counted skipped files.
        if FAIL_FAST_FLAG.load(Ordering::Relaxed) && failed.load(Ordering::Relaxed) > 0 {
            let now_str = now_utc();
            let source_rel = lml_path
                .strip_prefix(&input_root)
                .unwrap_or(lml_path)
                .display()
                .to_string();
            let line = format!("{} FAIL_FAST_SKIP {}\n", now_str, source_rel);
            if let Ok(mut w) = audit.lock() {
                let _ = w.write_all(line.as_bytes());
            }
            skipped_count.fetch_add(1, Ordering::Relaxed);
            done.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut out = make_decode_output_path(lml_path, &input_root, &output_dir);
        if to_edf {
            // make_decode_output_path forces .raw; flip to .edf when
            // reconstructing a real EDF file. Container metadata
            // doesn't currently distinguish .edf vs .bdf; .edf is the
            // safe default since the decoder writes byte-exact whatever
            // format the encoder ingested.
            out.set_extension("edf");
        }

        // Tier 3 audit (O10d): atomic skip_existing check via
        // create_new + O_EXCL. Pre-fix `out.exists()` + write was
        // classic TOCTOU; a symlink swap between check and the
        // decoder's `OpenOptions::truncate(true)` would destroy
        // the target the symlink pointed at. Now: try create_new;
        // on AlreadyExists honour skip; on success delete the
        // zero-byte sentinel so the decoder can re-open without
        // the truncate-already-empty edge case.
        if skip_existing {
            match std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&out)
            {
                Ok(file) => {
                    // We won the create race -- delete the sentinel so
                    // the decoder can re-open clean. The window
                    // between this drop and the decoder's open is
                    // microseconds and only an attacker with write
                    // access could exploit it.
                    drop(file);
                    let _ = std::fs::remove_file(&out);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    skipped_count.fetch_add(1, Ordering::Relaxed);
                    done.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(e) => {
                    eprintln!("  FAIL pre-create check {}: {}", out.display(), e);
                    failed.fetch_add(1, Ordering::Relaxed);
                    done.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        }

        let now_str = now_utc();
        let source_rel = lml_path
            .strip_prefix(&input_root)
            .unwrap_or(lml_path)
            .display()
            .to_string();

        let result = if to_edf {
            decode_one_to_edf(lml_path, &out)
        } else {
            decode_one_to_raw(lml_path, &out)
        };
        match result {
            Ok((n_ch, t)) => {
                let line = format!(
                    "{} OK {} -> {} {}ch x {}\n",
                    now_str,
                    source_rel,
                    out.file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    n_ch,
                    t
                );
                // Audit-2026-05-11 Fix-#32: recover from poisoned mutex.
                let mut w = match audit.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                // Tier 3 audit (O10): propagate write_all + flush
                // errors instead of `let _ = ...`. Pre-fix a full
                // disk produced silent "OK" lines that never
                // landed; `Done: N succeeded` was a lie. On write
                // failure, bump `failed` counter and eprintln.
                if let Err(e) = w.write_all(line.as_bytes()) {
                    eprintln!(
                        "  WARNING: audit.log write_all failed for {}: {}",
                        lml_path.display(),
                        e
                    );
                    failed.fetch_add(1, Ordering::Relaxed);
                } else if let Err(e) = w.flush() {
                    eprintln!(
                        "  WARNING: audit.log flush failed for {}: {}",
                        lml_path.display(),
                        e
                    );
                    failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(e) => {
                eprintln!("  FAIL {}: {}", lml_path.display(), e);
                failed.fetch_add(1, Ordering::Relaxed);
                let line = format!("{} FAIL {}: {}\n", now_str, source_rel, e);
                // Audit-2026-05-11 Fix-#32: recover from poisoned mutex.
                let mut w = match audit.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let _ = w.write_all(line.as_bytes());
                let _ = w.flush();
            }
        }

        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 100 == 0 || n == total {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = n as f64 / elapsed;
            let eta = (total - n) as f64 / rate;
            let pct = 100.0 * n as f64 / total as f64;
            eprint!(
                "\r  [{}/{}] {:.1}%  {:.1} files/s  ETA {}  elapsed {}   ",
                n,
                total,
                pct,
                rate,
                human_duration(eta),
                human_duration(elapsed)
            );
        }
    });
    eprintln!();

    let f = failed.load(Ordering::Relaxed);
    let s = skipped_count.load(Ordering::Relaxed);
    let elapsed = t0.elapsed();
    println!(
        "Done: {} succeeded, {} failed, {} skipped ({:.1}s)",
        total - f - s,
        f,
        s,
        elapsed.as_secs_f64()
    );
    if f > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Render one JSON value as a compact one-line summary for `lml info`.
/// Arrays of strings render as comma-joined (truncated at 8 items); all
/// other types fall back to compact JSON. Long strings truncate.
fn summarise_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => {
            // Tier 3 audit (O5): `&s[..200]` panics if byte 200
            // lands inside a multibyte UTF-8 sequence. Use
            // char_indices to find the next char boundary at or
            // after byte 200.
            if s.len() > 200 {
                let cut = s
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|i| *i <= 200)
                    .last()
                    .unwrap_or(0);
                format!("{}…", &s[..cut])
            } else {
                s.clone()
            }
        }
        serde_json::Value::Array(items) => {
            let strs: Vec<&str> = items.iter().filter_map(|x| x.as_str()).collect();
            if strs.len() == items.len() && !strs.is_empty() {
                let head = strs.iter().take(8).copied().collect::<Vec<_>>().join(", ");
                if strs.len() > 8 {
                    format!("{head}, … (+{} more)", strs.len() - 8)
                } else {
                    head
                }
            } else {
                serde_json::to_string(v).unwrap_or_else(|_| String::from("?"))
            }
        }
        _ => serde_json::to_string(v).unwrap_or_else(|_| String::from("?")),
    }
}

fn cmd_info(input: &Path) -> R {
    use std::io::Read as _;
    let mut f = std::fs::File::open(input)?;
    // Tier 3 audit (O5): keep file_size in u64. Pre-fix `as usize`
    // silently truncated > 4 GiB files on 32-bit MCU/host targets;
    // bounds checks derived from the truncated value were wrong.
    // Use usize only where slice arithmetic forces it, after a
    // bounds check.
    let file_size: u64 = f.metadata()?.len();
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr)?;

    // v1.1 magic-byte auto-dispatch: `lml info foo.lma` now routes
    // to the archive inspector instead of erroring "Not LML". The
    // CLI ergonomics match `tar` / `7z` / `unzip` where one tool
    // handles both container forms.
    if &hdr[0..4] == b"LMA1" {
        eprintln!(
            "note: {} is an LMA archive, dispatching to archive inspector (`lml ls --tree`).",
            input.display(),
        );
        return cmd_ls(input, /*tree=*/ true, /*long=*/ false);
    }

    // ADR 0069/0071 L9: `lml info` on a `BCS1` file (write_abir's default
    // output today) — dispatch to the BCS1-aware reader BEFORE the legacy
    // `hdr[0..3] != b"LML"` guard below would reject it. Reads the
    // remaining bytes to complete the 40-byte typed header (32 already
    // buffered above).
    if &hdr[0..4] == abir::BCS1_MAGIC {
        let mut rest = [0u8; abir::BCS1_HEADER_LEN - 32];
        f.read_exact(&mut rest)?;
        let mut full = [0u8; abir::BCS1_HEADER_LEN];
        full[..32].copy_from_slice(&hdr);
        full[32..].copy_from_slice(&rest);
        return cmd_info_bcs1(input, &mut f, file_size, &full);
    }

    if &hdr[0..3] != b"LML" {
        return Err(format!(
            "Not LML or LMA (magic: {:?}). Expected leading bytes `LML1` or `BCS1` or `LMA1`.",
            &hdr[0..4]
        )
        .into());
    }

    let ver_major = hdr[4];
    let ver_minor = hdr[5];
    let n_ch = u16::from_le_bytes([hdr[6], hdr[7]]);
    let n_win = u16::from_le_bytes([hdr[8], hdr[9]]);
    let total = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]);
    let ws = u16::from_le_bytes([hdr[14], hdr[15]]);
    let sr_mhz = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
    let bit_depth = hdr[20];
    let flags = hdr[21];
    let meta_len = u32::from_le_bytes([hdr[22], hdr[23], hdr[24], hdr[25]]);
    let raw = n_ch as f64 * total as f64 * 2.0;
    let sr_hz = sr_mhz as f64 / 1000.0;
    let duration_s = if sr_hz > 0.0 {
        total as f64 / sr_hz
    } else {
        0.0
    };
    let has_footer = (flags & 0b0000_0001) != 0;

    println!("File:       {}", input.display());
    println!(
        "Format:     LML{} v{}.{}",
        hdr[3] as char, ver_major, ver_minor
    );
    println!("Channels:   {}", n_ch);
    println!("Windows:    {}", n_win);
    println!(
        "Samples:    {} ({:.1}s @ {:.0} Hz)",
        total, duration_s, sr_hz
    );
    println!("Duration:   {}", human_duration(duration_s));
    println!("Window:     {} samples", ws);
    println!("Bit depth:  {}", bit_depth);
    println!(
        "Flags:      0x{:02X}{}",
        flags,
        if has_footer {
            " (HAS_FOOTER)"
        } else {
            " (no footer; legacy file)"
        }
    );
    println!("Size:       {}", human_bytes(file_size));
    if raw > 0.0 {
        println!(
            "CR:         {:.2}:1  ({} raw → {})",
            raw / file_size as f64,
            human_bytes(raw as u64),
            human_bytes(file_size)
        );
    }
    // Tier 3 audit (O5): u64-domain bounds check. Pre-fix
    // `32 + meta_len as usize <= file_size` did the addition in
    // 32-bit usize on MCU targets and could wrap. Now: check in
    // u64 first; only cast to usize if the read fits.
    if meta_len > 0 && 32u64.saturating_add(meta_len as u64) <= file_size {
        let read_len = (meta_len as usize).min(4096);
        let mut meta_buf = vec![0u8; read_len];
        f.read_exact(&mut meta_buf)?;
        let meta = String::from_utf8_lossy(&meta_buf);
        print_metadata_summary(&meta);
    }

    // Probe footer at EOF when present. Surface the seek-table size
    // (the only field that meaningfully informs the user — full table
    // parse is for `lml stats` etc.). Footer absence is already
    // signalled by the Flags line above; flag set but EOF magic
    // mismatched = corrupted random-access trailer, surface as warn.
    if has_footer {
        print_footer_probe(&mut f, file_size, n_win as u32)?;
    }
    Ok(())
}

/// Human-readable metadata JSON dump — shared tail of `lml info` for both
/// the legacy 32-byte LML1 header and the 40-byte BCS1 header (ADR
/// 0069/0071 L9: the metadata blob itself is byte-identical between the
/// two wire formats, only the header framing it differs). Prints known
/// priority fields first, then any leftover keys; falls back to a raw dump
/// for non-object or unparseable JSON.
fn print_metadata_summary(meta: &str) {
    // Try to surface known fields as a structured readout. Fall back to
    // raw text for unknown / unparseable metadata.
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(meta);
    if let Ok(v) = parsed {
        if let Some(o) = v.as_object() {
            // Phase 1.11: human-readable per-key dump. Prioritise
            // well-known fields so the most useful info bubbles to the
            // top of the listing.
            let priority_keys = [
                "tool_version",
                "encoder_version",
                "source",
                "source_file",
                "format",
                "patient_id",
                "recording_info",
                "startdate",
                "phys_dim",
                "channels",
                "channel_labels",
            ];
            for k in priority_keys {
                if let Some(val) = o.get(k) {
                    println!("{:12}{}", format!("{k}:"), summarise_json(val));
                }
            }
            // Then any leftover keys for completeness.
            for (k, val) in o {
                if !priority_keys.contains(&k.as_str()) {
                    println!("{:12}{}", format!("{k}:"), summarise_json(val));
                }
            }
        } else {
            println!("Metadata:   {meta}");
        }
    } else {
        println!("Metadata:   {meta}");
    }
}

/// Probe the `LMLFOOT1` seek-table footer at EOF and print a one-line
/// summary — shared tail of `lml info` for both header formats. The
/// footer's own position/shape (fixed 32 bytes at EOF, preceded by the
/// offset table) does NOT depend on which header the file carries, so this
/// is a straight share, not a clone-with-changes. `n_win_header` is the
/// header's own `n_windows` field, used only to bound-check the footer's
/// claimed count against corruption (Tier 3 audit O5 — a poisoned footer
/// used to print "Seek table: 4294967295 entries" unchecked).
fn print_footer_probe(
    f: &mut std::fs::File,
    file_size: u64,
    n_win_header: u32,
) -> std::io::Result<()> {
    use std::io::{Read as _, Seek as _};
    if file_size < 32 {
        return Ok(());
    }
    f.seek(std::io::SeekFrom::End(-32))?;
    let mut footer = [0u8; 32];
    f.read_exact(&mut footer)?;
    if &footer[0..8] == b"LMLFOOT1" {
        let n_seek_windows = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]);
        if n_seek_windows > n_win_header {
            println!(
                "Seek table: {} entries (CORRUPT: exceeds header n_windows = {})",
                n_seek_windows, n_win_header
            );
        } else {
            println!("Seek table: {} entries (LMLFOOT1 magic OK)", n_seek_windows);
        }
    } else {
        println!(
            "Seek table: FLAG_HAS_FOOTER set but LMLFOOT1 magic missing at EOF \
             — corrupted random-access trailer"
        );
    }
    Ok(())
}

/// `lml info` for a `BCS1` file (ADR 0069/0071 L9) — the BCS1 counterpart
/// of the legacy tail of [`cmd_info`] above, sourced from
/// [`abir::Bcs1Header`]'s parsed fields instead of hand-rolled
/// 32-byte offsets. `hdr_bytes` is the already-buffered 40-byte header
/// (`cmd_info` reads it before dispatching here); `f` is positioned right
/// after it, ready for the metadata read below.
fn cmd_info_bcs1(
    input: &Path,
    f: &mut std::fs::File,
    file_size: u64,
    hdr_bytes: &[u8; abir::BCS1_HEADER_LEN],
) -> R {
    use std::io::Read as _;
    let header = abir::Bcs1Header::parse(hdr_bytes)
        .map_err(|e| format!("invalid BCS1 header: {e}"))?;

    let raw = header.n_channels as f64 * header.total_samples as f64 * 2.0;
    let sr_hz = header.sample_rate_mhz as f64 / 1000.0;
    let duration_s = if sr_hz > 0.0 {
        header.total_samples as f64 / sr_hz
    } else {
        0.0
    };
    let has_footer = (header.flags & 0b0000_0001) != 0;

    println!("File:       {}", input.display());
    println!(
        "Format:     BCS1 v{}.{}",
        header.version_major, header.version_minor
    );
    println!("Channels:   {}", header.n_channels);
    println!("Windows:    {}", header.n_windows);
    println!(
        "Samples:    {} ({:.1}s @ {:.0} Hz)",
        header.total_samples, duration_s, sr_hz
    );
    println!("Duration:   {}", human_duration(duration_s));
    println!("Window:     {} samples", header.window_size);
    println!("Bit depth:  {}", header.bit_depth);
    println!(
        "Modality:   tag={} source={}",
        header.modality_tag, header.modality_source
    );
    println!(
        "Codec:      descriptor={} mode={} tier={} decode_capability={}",
        header.codec_descriptor, header.mode, header.tier, header.decode_capability
    );
    println!(
        "Flags:      0x{:02X}{}",
        header.flags,
        if has_footer {
            " (HAS_FOOTER)"
        } else {
            " (no footer)"
        }
    );
    println!("Size:       {}", human_bytes(file_size));
    if raw > 0.0 {
        println!(
            "CR:         {:.2}:1  ({} raw → {})",
            raw / file_size as f64,
            human_bytes(raw as u64),
            human_bytes(file_size)
        );
    }

    let meta_len = header.metadata_length;
    if meta_len > 0
        && (abir::BCS1_HEADER_LEN as u64).saturating_add(meta_len as u64) <= file_size
    {
        let read_len = (meta_len as usize).min(4096);
        let mut meta_buf = vec![0u8; read_len];
        f.read_exact(&mut meta_buf)?;
        let meta = String::from_utf8_lossy(&meta_buf);
        print_metadata_summary(&meta);
    }

    if has_footer {
        print_footer_probe(f, file_size, header.n_windows as u32)?;
    }
    Ok(())
}

fn cmd_verify(input: &Path, recursive: bool, explain: bool) -> R {
    // Hard-fail on a missing input — clinical-grade contract: silent
    // "0/0 verified" on a typo'd path is unacceptable.
    if !input.exists() {
        return Err(format!("input path does not exist: {}", input.display()).into());
    }

    // Tier 3 audit (O14 follow-up): if user passed `--explain` but
    // the dispatch never reaches the archive verifier (i.e. input
    // is a pure-LML file or LML directory), emit a stderr WARNING
    // so the operator knows the flag is being ignored. The
    // per-window CRC report for LML files is a follow-up feature;
    // until then, the visible diagnostic prevents silent
    // confusion.
    let warn_explain_unused = |scope: &str| {
        if explain {
            eprintln!(
                "  WARNING: --explain currently only applies to LMA archive verification, \
                 not LML files ({}); ignoring for this run.",
                scope
            );
        }
    };

    // v1.1 magic-byte auto-dispatch: if the user passed a single
    // `.lma` archive (or any single file whose leading bytes are
    // `LMA1`), route to the archive verifier instead of trying to
    // walk it as an LML directory. v1.2 X adds `--explain`
    // forwarding so `lml verify foo.lma --explain` renders the
    // auditable per-step readout.
    //
    // Tier 3 audit (O14): surface File::open errors instead of
    // silently falling through to the LML walker. Pre-fix the
    // `.is_ok()` short-circuit meant a permission-denied on an
    // LMA produced a confusing "not LML" error downstream.
    if input.is_file() {
        use std::io::Read as _;
        match std::fs::File::open(input) {
            Ok(mut f) => {
                let mut magic = [0u8; 4];
                if f.read_exact(&mut magic).is_ok() && &magic == b"LMA1" {
                    eprintln!(
                        "note: {} is an LMA archive, dispatching to archive verifier (`lml verify-archive`).",
                        input.display(),
                    );
                    return cmd_verify_archive_explain(input, explain);
                }
            }
            Err(e) => {
                return Err(format!(
                    "cmd_verify: cannot open {} for magic-byte check: {}",
                    input.display(),
                    e
                )
                .into());
            }
        }
    }

    let files = if input.is_file() {
        vec![input.to_path_buf()]
    } else {
        let mut f = Vec::new();
        let walker = if recursive {
            walkdir::WalkDir::new(input)
        } else {
            walkdir::WalkDir::new(input).max_depth(1)
        };
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            // v1.1: include `.lma` in directory walks so `lml verify
            // recursive_dir/` picks up archives alongside `.lml` files.
            //
            // Tier 3 audit (O14): compare against OsStr instead of
            // str so paths with non-UTF-8 components aren't silently
            // dropped from the verify set (pre-fix `.and_then(|e|
            // e.to_str())` returned None and the entry was skipped
            // with no warning; total count under-reported).
            let ext = entry.path().extension();
            let is_lml = ext.is_some_and(|e| e.eq_ignore_ascii_case(std::ffi::OsStr::new("lml")));
            let is_lma = ext.is_some_and(|e| e.eq_ignore_ascii_case(std::ffi::OsStr::new("lma")));
            if is_lml || is_lma {
                f.push(entry.path().to_path_buf());
            }
        }
        f.sort();
        f
    };

    let total = files.len();
    if total == 0 {
        return Err(format!(
            "no .lml files found at {} — verify would silently report \
             0/0 success otherwise",
            input.display()
        )
        .into());
    }
    let mut passed = 0;
    let mut failed = 0;

    // Tier 3 audit (O14): if explain was requested but every file
    // in the batch is LML (not LMA), surface the warning once
    // instead of per-file. Mixed batches don't trigger this; the
    // per-LML-entry path silently skips explain.
    let any_lma = files.iter().any(|f| {
        f.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(std::ffi::OsStr::new("lma")))
    });
    if !any_lma {
        warn_explain_unused("verify batch");
    }
    for f in &files {
        let t0 = Instant::now();
        // v1.1: per-file magic-byte dispatch so a mixed `.lml` +
        // `.lma` corpus walks transparently.
        let is_lma = f
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "lma");
        if is_lma {
            match cmd_verify_archive_explain(f, explain) {
                Ok(()) => {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if total <= 10 {
                        println!("  OK  lma ({:.1}ms) {}", ms, f.display());
                    }
                    passed += 1;
                }
                Err(e) => {
                    println!("  FAIL {} — {}", f.display(), e);
                    failed += 1;
                }
            }
            continue;
        }
        match container::read_file(f) {
            Ok((sig, _)) => {
                let n_ch = sig.len();
                let t = if n_ch > 0 { sig[0].len() } else { 0 };
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                if total <= 10 {
                    println!("  OK  {}ch × {} ({:.1}ms) {}", n_ch, t, ms, f.display());
                }
                passed += 1;
            }
            Err(e) => {
                println!("  FAIL {} — {}", f.display(), e);
                failed += 1;
            }
        }
    }

    println!("{}/{} verified, {} failed", passed, total, failed);
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

// ── Paranoid roundtrip verification — clinical-grade bit-exact check ──
//
// Encodes each EDF/BDF in the input set to a tempfile.lml, then verifies
// FOUR SHA-256 slices match between original and what came out of the
// container. Designed for "must be bit-perfect on real patient data"
// deployments where any drift is unacceptable.
//
// What's checked:
//   1. signal samples — SHA over int24-cast bytes of every channel sample
//   2. raw_header    — SHA over the literal EDF header bytes
//                      (256 main + n_signals*256 channel headers)
//   3. non_eeg_data  — SHA over the concatenated raw bytes of every
//                      non-EEG channel slot (annotations + status etc.)
//   4. trailing_data — SHA over partial-record bytes at the file tail.
//                      Pre-fix encoder dropped these; new encoder
//                      preserves them.
//
// All four MUST match. JSON report records per-file results + aggregate
// counts. Process exits with code 1 if any file fails so CI can gate
// on it.

#[derive(serde::Serialize)]
struct RoundtripResult {
    file: String,
    size_bytes: u64,
    n_channels: usize,
    n_samples: usize,
    elapsed_ms: f64,
    /// SHA-256 of the original EDF/BDF file, full bytes. Strict
    /// definition of bit-exact roundtrip: this and `roundtrip_sha256`
    /// MUST be identical for status=PASS.
    original_sha256: String,
    /// SHA-256 of the file produced by encode → decode-to-EDF. If
    /// equal to original_sha256, the codec is forensically lossless
    /// for this input.
    roundtrip_sha256: String,
    /// True iff original_sha256 == roundtrip_sha256. The single
    /// definition of "did the roundtrip preserve everything".
    file_match: bool,
    /// Diagnostic — populated only on file_match=false. The byte
    /// offset of the first divergence, or None if the lengths
    /// differ at byte 0. Helps narrow which region of the EDF the
    /// codec corrupted (header / data / trailing).
    first_diff_offset: Option<u64>,
    status: &'static str, // "PASS" | "FAIL" | "ERROR"
    error: Option<String>,
}

#[derive(serde::Serialize)]
struct RoundtripReport {
    total: usize,
    passed: usize,
    failed: usize,
    errored: usize,
    elapsed_ms_total: f64,
    encoder_version: String,
    results: Vec<RoundtripResult>,
}

#[allow(dead_code)]
fn sha256_signal(signal: &[Vec<i64>]) -> String {
    // Hash channels in order, samples within each channel as
    // little-endian 8-byte i64. Same byte representation on every
    // platform so the hash is stable across machines.
    let mut h = Sha256::new();
    for ch in signal {
        for &v in ch {
            h.update(v.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn sha256_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    format!("{:x}", h.finalize())
}

#[allow(dead_code)]
fn sha256_non_eeg_pairs(pairs: &[(usize, Vec<u8>)]) -> String {
    // Sort by channel index for stable ordering, then hash channel
    // index (4 bytes LE) + payload length (8 bytes LE) + payload bytes.
    // Length-prefixing protects against (idx=0, ch="abc") vs
    // (idx=0, ch="ab"+"c") collisions.
    let mut sorted: Vec<(usize, &[u8])> = pairs.iter().map(|(i, b)| (*i, b.as_slice())).collect();
    sorted.sort_by_key(|(i, _)| *i);
    let mut h = Sha256::new();
    for (i, b) in sorted {
        h.update((i as u32).to_le_bytes());
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    format!("{:x}", h.finalize())
}

/// Pull `"key":"value"` substring from the metadata JSON. Reuses the
/// pattern from extract_json_str but local to keep the call site
/// self-contained. Returns None if the key is absent or malformed.
fn meta_str_field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let end = json[start..].find('"').map(|p| start + p)?;
    Some(&json[start..end])
}

/// Pull the non_eeg_channels object {idx:b64, ...} and decode each
/// payload (b64 → zstd → bytes). Returns Vec<(idx, bytes)> sorted by
/// idx so sha256_non_eeg_pairs is stable.
fn meta_non_eeg(json: &str) -> Option<Vec<(usize, Vec<u8>)>> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;
    let needle = "\"non_eeg_channels\":{";
    let start = json.find(needle)? + needle.len();
    // Find matching close brace; non_eeg payloads can't contain braces
    // (they're b64 strings) so a simple scan to '}' works.
    let end = json[start..].find('}').map(|p| start + p)?;
    let body = &json[start..end];
    if body.trim().is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for entry in body.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        // Format: "12":"b64..."
        let mut parts = entry.splitn(2, ':');
        let idx_str = parts.next()?.trim().trim_matches('"');
        let val_str = parts.next()?.trim().trim_matches('"');
        let idx: usize = idx_str.parse().ok()?;
        let compressed = b64.decode(val_str.as_bytes()).ok()?;
        let raw = if compressed.is_empty() {
            Vec::new()
        } else {
            zstd::decode_all(compressed.as_slice()).ok()?
        };
        out.push((idx, raw));
    }
    out.sort_by_key(|(i, _)| *i);
    Some(out)
}

/// Decode a base64+zstd-compressed bytes field from the metadata JSON.
/// Returns empty Vec if the field's value is empty.
fn meta_b64_zstd_field(json: &str, key: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;
    let v = meta_str_field(json, key)?;
    if v.is_empty() {
        return Some(Vec::new());
    }
    let compressed = b64.decode(v.as_bytes()).ok()?;
    if compressed.is_empty() {
        return Some(Vec::new());
    }
    zstd::decode_all(compressed.as_slice()).ok()
}

fn first_diff_offset(a: &[u8], b: &[u8]) -> Option<u64> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i as u64);
        }
    }
    if a.len() != b.len() {
        Some(n as u64)
    } else {
        None
    }
}

fn roundtrip_one(edf_path: &Path) -> RoundtripResult {
    let file = edf_path.display().to_string();
    let t0 = Instant::now();

    let size_bytes = std::fs::metadata(edf_path).map(|m| m.len()).unwrap_or(0);

    let make_err = |status: &'static str, e: String| RoundtripResult {
        file: file.clone(),
        size_bytes,
        n_channels: 0,
        n_samples: 0,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        original_sha256: String::new(),
        roundtrip_sha256: String::new(),
        file_match: false,
        first_diff_offset: None,
        status,
        error: Some(e),
    };

    // Read original bytes + compute strict full-file SHA.
    let orig_bytes = match std::fs::read(edf_path) {
        Ok(b) => b,
        Err(e) => return make_err("ERROR", format!("read source: {}", e)),
    };
    let original_sha256 = sha256_bytes(&orig_bytes);

    // Quick stats for reporting (channel count, sample count).
    // Tier 3 audit (O12): log read_edf errors to stderr instead of
    // silent (0, 0) -- pre-fix code produced a JSON report with
    // n_channels=0/n_samples=0 and no indication that EDF parse
    // failed. Operators saw "0ch x 0samples" and assumed empty
    // input. Now: stderr line with the parse error.
    let (n_channels, n_samples) = match lamquant_core::edf::read_edf(edf_path) {
        Ok(e) => (
            e.signal.len(),
            e.signal.first().map(|c| c.len()).unwrap_or(0),
        ),
        Err(e) => {
            eprintln!(
                "  WARNING: roundtrip_one: read_edf failed on {}: {} -- \
                 stats fall back to (0, 0) but the roundtrip continues via raw bytes",
                file, e
            );
            (0, 0)
        }
    };

    // Encode → temp.lml.
    let tmp_lml = match tempfile::Builder::new()
        .prefix("lml-roundtrip-")
        .suffix(".lml")
        .tempfile()
    {
        Ok(t) => t,
        Err(e) => return make_err("ERROR", format!("tempfile.lml: {}", e)),
    };
    let lml_path = tmp_lml.path().to_path_buf();

    if let Err(e) = encode_one(
        edf_path,
        &lml_path,
        false, // verify
        false, // cross_validate (we do our own paranoid full-file SHA below)
        0,     // noise_bits — 0 = lossless
        2500,  // window_size — match the CLI default
        lamquant_core::lpc::LpcMode::default(),
    ) {
        return make_err("ERROR", format!("encode: {}", e));
    }

    // Decode-to-EDF → temp.edf (full byte-identical reconstruction).
    let tmp_edf = match tempfile::Builder::new()
        .prefix("lml-roundtrip-")
        .suffix(".edf")
        .tempfile()
    {
        Ok(t) => t,
        Err(e) => return make_err("ERROR", format!("tempfile.edf: {}", e)),
    };
    let edf_path_round = tmp_edf.path().to_path_buf();
    if let Err(e) = decode_one_to_edf(&lml_path, &edf_path_round) {
        return make_err("ERROR", format!("decode_to_edf: {}", e));
    }

    let round_bytes = match std::fs::read(&edf_path_round) {
        Ok(b) => b,
        Err(e) => return make_err("ERROR", format!("read roundtrip: {}", e)),
    };
    let roundtrip_sha256 = sha256_bytes(&round_bytes);

    let file_match = original_sha256 == roundtrip_sha256;
    let first_diff = if file_match {
        None
    } else {
        first_diff_offset(&orig_bytes, &round_bytes)
    };

    RoundtripResult {
        file,
        size_bytes,
        n_channels,
        n_samples,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        original_sha256,
        roundtrip_sha256,
        file_match,
        first_diff_offset: first_diff,
        status: if file_match { "PASS" } else { "FAIL" },
        error: None,
    }
}

fn cmd_roundtrip(
    input: &Path,
    recursive: bool,
    report: Option<&Path>,
    fail_fast: bool,
    parallel: usize,
) -> R {
    if !input.exists() {
        return Err(format!("input path does not exist: {}", input.display()).into());
    }

    // Collect EDF/BDF files.
    let files: Vec<PathBuf> = if input.is_file() {
        vec![input.to_path_buf()]
    } else {
        let mut f = Vec::new();
        let walker = if recursive {
            walkdir::WalkDir::new(input)
        } else {
            walkdir::WalkDir::new(input).max_depth(1)
        };
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                let lower = ext.to_ascii_lowercase();
                if matches!(lower.as_str(), "edf" | "bdf") {
                    f.push(entry.path().to_path_buf());
                }
            }
        }
        f.sort();
        f
    };

    if files.is_empty() {
        return Err(format!(
            "no .edf/.bdf files at {} — roundtrip would silently report 0/0",
            input.display()
        )
        .into());
    }

    if parallel > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build_global()
            .ok();
    }

    let total = files.len();
    eprintln!(
        "[roundtrip] paranoid bit-exact verification: {} file(s)",
        total
    );

    let t_start = Instant::now();

    // fail_fast short-circuits via AtomicBool. rayon par_iter still runs
    // queued work but we drop results once tripped.
    let stop = std::sync::atomic::AtomicBool::new(false);

    let mut results: Vec<RoundtripResult> = files
        .par_iter()
        .map(|p| {
            if fail_fast && stop.load(std::sync::atomic::Ordering::Relaxed) {
                // Skip — produce a synthetic skipped result.
                return RoundtripResult {
                    file: p.display().to_string(),
                    size_bytes: 0,
                    n_channels: 0,
                    n_samples: 0,
                    elapsed_ms: 0.0,
                    original_sha256: String::new(),
                    roundtrip_sha256: String::new(),
                    file_match: false,
                    first_diff_offset: None,
                    status: "SKIPPED",
                    error: Some("fail_fast: skipped after upstream failure".into()),
                };
            }
            let r = roundtrip_one(p);
            if fail_fast && r.status != "PASS" {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            r
        })
        .collect();

    // Sort results by file path for stable JSON output.
    results.sort_by(|a, b| a.file.cmp(&b.file));

    let elapsed_ms_total = t_start.elapsed().as_secs_f64() * 1000.0;
    let mut passed = 0;
    let mut failed = 0;
    let mut errored = 0;
    for r in &results {
        match r.status {
            "PASS" => passed += 1,
            "FAIL" => failed += 1,
            "ERROR" | "SKIPPED" => errored += 1,
            _ => {}
        }
    }

    // Per-file stderr summary so users see progress without parsing JSON.
    // PASS prints the matching SHA prefix for spot-checks; FAIL prints
    // first-diff-offset so users can locate the corruption boundary
    // (header / data record / trailing section).
    for r in &results {
        let mark = match r.status {
            "PASS" => "  OK",
            "FAIL" => "FAIL",
            _ => " ERR",
        };
        let detail = if r.status == "PASS" {
            format!("sha256:{}", &r.original_sha256[..16])
        } else if r.status == "FAIL" {
            match r.first_diff_offset {
                Some(o) => format!("first_diff@{} bytes", o),
                None => "lengths differ".to_string(),
            }
        } else {
            r.error.as_deref().unwrap_or("").to_string()
        };
        eprintln!("{}  {}  ({:.0}ms)  {}", mark, r.file, r.elapsed_ms, detail,);
    }
    eprintln!(
        "[roundtrip] {}/{} PASS, {} FAIL, {} ERROR — {:.1}s total",
        passed,
        total,
        failed,
        errored,
        elapsed_ms_total / 1000.0,
    );

    let report_obj = RoundtripReport {
        total,
        passed,
        failed,
        errored,
        elapsed_ms_total,
        encoder_version: format!("lml/{}", env!("CARGO_PKG_VERSION")),
        results,
    };
    // Tier 3 audit (O12): hard-Err on serialize failure instead of
    // writing a non-conforming {"error": ...} body. Pre-fix wrote
    // a file claiming to be the report but containing only an
    // error message; callers parsing the JSON silently failed.
    let json = serde_json::to_string_pretty(&report_obj)
        .map_err(|e| format!("cmd_roundtrip: JSON serialize failed: {}", e))?;
    // Tier 3 audit (O12): atomic write via tmp+rename. Pre-fix
    // `std::fs::write` was non-atomic; a kill mid-write left a
    // partially-truncated JSON file.
    match report {
        Some(p) => {
            let tmp = p.with_extension(format!(
                "{}.tmp.{}.{}",
                p.extension().and_then(|s| s.to_str()).unwrap_or("json"),
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::write(&tmp, &json)?;
            std::fs::rename(&tmp, p).map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!(
                    "cmd_roundtrip: atomic rename of {} failed: {}",
                    p.display(),
                    e
                )
            })?;
        }
        None => println!("{}", json),
    }

    if failed > 0 || errored > 0 {
        std::process::exit(1);
    }
    Ok(())
}

// ── Feature 2: verify-manifest ──

fn cmd_verify_manifest(manifest_path: &Path) -> R {
    let data = std::fs::read_to_string(manifest_path)?;
    let manifest_dir = manifest_path.parent().unwrap_or(Path::new("."));

    // Minimal JSON parsing — find "files" array entries
    // Each entry has: "output", "compressed_bytes", "sha256"
    let files_start = data
        .find("\"files\":[")
        .ok_or("manifest missing \"files\" array")?;
    let files_str = &data[files_start + 8..];

    // Split on "},{" to get individual file entries
    let mut entries: Vec<&str> = Vec::new();
    let inner_start = files_str.find('[').unwrap_or(0) + 1;
    let inner_end = files_str.rfind(']').unwrap_or(files_str.len());
    let inner = &files_str[inner_start..inner_end];
    if !inner.trim().is_empty() {
        // Split carefully on "},{" — rejoin with "},{" to re-parse
        let mut depth = 0i32;
        let mut start = 0;
        let bytes = inner.as_bytes();
        for i in 0..bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        entries.push(&inner[start..=i]);
                        // Skip comma
                        if i + 1 < bytes.len() && bytes[i + 1] == b',' {
                            start = i + 2;
                        } else {
                            start = i + 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let total = entries.len();
    let mut passed = 0usize;
    let mut failed_count = 0usize;
    let mut missing = 0usize;

    for entry in &entries {
        let out_path = extract_json_str(entry, "output").unwrap_or_default();
        let expected_size = extract_json_num(entry, "compressed_bytes").unwrap_or(0);
        let expected_sha = extract_json_str(entry, "sha256").unwrap_or_default();

        let full_path = manifest_dir.join(&out_path);

        if !full_path.exists() {
            println!("  MISSING {}", out_path);
            missing += 1;
            continue;
        }

        // Check file size
        let actual_size = std::fs::metadata(&full_path)?.len() as usize;
        if expected_size > 0 && actual_size != expected_size {
            println!(
                "  SIZE MISMATCH {}: expected {} got {}",
                out_path, expected_size, actual_size
            );
            failed_count += 1;
            continue;
        }

        // Decode and compute SHA-256 of signal
        if !expected_sha.is_empty() {
            match container::read_file(&full_path) {
                Ok((signal, _)) => {
                    let mut hasher = Sha256::new();
                    for ch in &signal {
                        for &sample in ch {
                            hasher.update(sample.to_le_bytes());
                        }
                    }
                    let actual_sha = format!("{:x}", hasher.finalize());
                    if actual_sha != expected_sha {
                        println!(
                            "  SHA256 MISMATCH {}: expected {}.. got {}...",
                            out_path,
                            &expected_sha[..16.min(expected_sha.len())],
                            &actual_sha[..16.min(actual_sha.len())]
                        );
                        failed_count += 1;
                        continue;
                    }
                }
                Err(e) => {
                    println!("  DECODE FAIL {}: {}", out_path, e);
                    failed_count += 1;
                    continue;
                }
            }
        }

        passed += 1;
    }

    println!(
        "\nManifest verification: {}/{} passed, {} failed, {} missing",
        passed, total, failed_count, missing
    );
    if failed_count > 0 || missing > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Extract a JSON string value by key (simple, no nested objects).
fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let end = json[start..].find('"')? + start;
    Some(json[start..end].to_string())
}

/// Extract a JSON number value by key.
fn extract_json_num(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\":", key);
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ── Feature 3: stats ──

fn cmd_stats(input: &Path, recursive: bool) -> R {
    let lmls = find_lmls(input, recursive);
    if lmls.is_empty() {
        return Err("No LML files found".into());
    }

    let is_dir = lmls.len() > 1 || input.is_dir();

    if is_dir {
        // CSV summary mode
        println!("file,channels,samples,duration_s,sample_rate,file_bytes");
        // Tier 3 audit (O13): track failures so we can exit non-zero
        // when every file fails. Pre-fix CSV mode returned Ok(())
        // even on 100% failure; CI couldn't gate on stats success.
        let mut failed = 0usize;
        for lml_path in &lmls {
            match read_lml_header_info(lml_path) {
                Ok(info) => {
                    println!(
                        "{},{},{},{:.1},{:.0},{}",
                        lml_path.display(),
                        info.n_ch,
                        info.total_samples,
                        info.duration_s,
                        info.sample_rate,
                        info.file_size
                    );
                }
                Err(e) => {
                    eprintln!("  FAIL {}: {}", lml_path.display(), e);
                    failed += 1;
                }
            }
        }
        if failed > 0 {
            return Err(format!(
                "cmd_stats: CSV summary mode -- {}/{} files failed header read",
                failed,
                lmls.len()
            )
            .into());
        }
    } else {
        // Single file: detailed per-channel stats table
        let lml_path = &lmls[0];
        let (signal, metadata) = container::read_file(lml_path)?;
        let n_ch = signal.len();
        let t = if n_ch > 0 { signal[0].len() } else { 0 };

        // Tier 3 audit (O13): warn when sample_rate fallback fires.
        // Pre-fix silently substituted 250.0 -- `duration_s`
        // numbers were fiction for any file lacking the metadata
        // field, with no warning. Now: track + diagnose.
        let sr_opt = extract_json_num(&metadata, "sample_rate").map(|v| v as f64);
        let sr = match sr_opt {
            Some(v) if v.is_finite() && v > 0.0 => v,
            _ => {
                eprintln!(
                    "  WARNING: cmd_stats: {} lacks finite positive sample_rate in metadata; \
                     duration shown is computed at 250 Hz placeholder",
                    lml_path.display()
                );
                250.0
            }
        };
        let duration = t as f64 / sr;
        let file_size = std::fs::metadata(lml_path)?.len();

        // Tier 3 audit (O13): float-domain CR math. Pre-fix
        // `(n_ch * t * 2) as f64` did the multiply in usize first;
        // for high-density EEG (256ch × 8M samples) it overflows
        // usize on 32-bit MCU targets and prints garbage CR.
        let raw_bytes_f = (n_ch as f64) * (t as f64) * 2.0;
        println!("File:        {}", lml_path.display());
        println!("Channels:    {}", n_ch);
        println!("Samples:     {} ({:.1}s @ {:.0} Hz)", t, duration, sr);
        println!(
            "File size:   {} bytes ({:.2}:1 CR)",
            file_size,
            raw_bytes_f / file_size as f64
        );
        println!();
        println!(
            "{:<6} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Ch", "Min", "Max", "Mean", "Std", "Samples", "NoiseFloor"
        );
        println!("{}", "-".repeat(84));

        for (i, ch) in signal.iter().enumerate() {
            if ch.is_empty() {
                continue;
            }
            let n = ch.len() as f64;
            let min_v = ch.iter().copied().min().unwrap_or(0);
            let max_v = ch.iter().copied().max().unwrap_or(0);
            let sum: f64 = ch.iter().map(|&v| v as f64).sum();
            let mean = sum / n;
            let var: f64 = ch
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let std = var.sqrt();

            // Noise floor estimate: Median Absolute Deviation (MAD).
            // Tier 3 audit (O13): cap MAD computation memory. Pre-fix
            // cloned every channel into Vec<i64> + Vec<f64>; for a 24h
            // × 256ch × 1kHz EEG that's ~5 GiB per channel pair. Cap
            // by sampling: if channel is over 1M samples, sample
            // evenly to a 1M-element sketch. MAD is robust to
            // sub-sampling at this rate.
            const MAD_SKETCH_CAP: usize = 1 << 20; // 1M samples
            let sample_step = ch.len().div_ceil(MAD_SKETCH_CAP).max(1);
            let mut sorted: Vec<i64> = ch.iter().step_by(sample_step).copied().collect();
            sorted.sort();
            let median = sorted[sorted.len() / 2] as f64;
            let mut abs_devs: Vec<f64> =
                sorted.iter().map(|&v| (v as f64 - median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = abs_devs[abs_devs.len() / 2];

            println!(
                "{:<6} {:>12} {:>12} {:>12.1} {:>12.1} {:>12} {:>12.1}",
                i,
                min_v,
                max_v,
                mean,
                std,
                ch.len(),
                mad
            );
        }
    }
    Ok(())
}

struct LmlHeaderInfo {
    n_ch: usize,
    total_samples: usize,
    sample_rate: f64,
    duration_s: f64,
    file_size: u64,
}

fn read_lml_header_info(
    path: &Path,
) -> Result<LmlHeaderInfo, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path)?;
    let file_size = f.metadata()?.len();
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;

    // ADR 0069/0071 L9: `lml stats` CSV/directory mode (this function's
    // only caller) previously hard-parsed a 32-byte LML1 header and
    // rejected everything else, including `BCS1` files (write_abir's
    // default output today). Dispatch on magic first.
    if magic == *abir::BCS1_MAGIC {
        let mut rest = [0u8; abir::BCS1_HEADER_LEN - 4];
        f.read_exact(&mut rest)?;
        let mut hdr = [0u8; abir::BCS1_HEADER_LEN];
        hdr[0..4].copy_from_slice(&magic);
        hdr[4..].copy_from_slice(&rest);
        let header = abir::Bcs1Header::parse(&hdr)
            .map_err(|e| format!("invalid BCS1 header: {e}"))?;
        let sr = header.sample_rate_mhz as f64 / 1000.0;
        let total_samples = header.total_samples as usize;
        let duration_s = if sr > 0.0 {
            total_samples as f64 / sr
        } else {
            0.0
        };
        return Ok(LmlHeaderInfo {
            n_ch: header.n_channels as usize,
            total_samples,
            sample_rate: sr,
            duration_s,
            file_size,
        });
    }

    if &magic[0..3] != b"LML" {
        return Err("Not a valid LML file".into());
    }
    let mut rest = [0u8; 28];
    f.read_exact(&mut rest)?;
    let mut hdr = [0u8; 32];
    hdr[0..4].copy_from_slice(&magic);
    hdr[4..].copy_from_slice(&rest);
    let n_ch = u16::from_le_bytes([hdr[6], hdr[7]]) as usize;
    let total_samples = u32::from_le_bytes([hdr[10], hdr[11], hdr[12], hdr[13]]) as usize;
    let sr_mhz = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
    let sr = sr_mhz as f64 / 1000.0;
    let duration_s = if sr > 0.0 {
        total_samples as f64 / sr
    } else {
        0.0
    };
    Ok(LmlHeaderInfo {
        n_ch,
        total_samples,
        sample_rate: sr,
        duration_s,
        file_size,
    })
}

// ── Feature 7: export ──

fn cmd_export(input: &Path, output: Option<&Path>, format: &str, lossless_mode: Option<&str>) -> R {
    let export_mode = parse_lossless_mode(lossless_mode)
        .map_err(Box::<dyn std::error::Error + Send + Sync>::from)?;
    #[cfg(not(any(feature = "hdf5", feature = "nwb")))]
    let _ = export_mode;
    // Tier 3 audit (O4): refuse output==input across all formats.
    // Every export arm uses `input.with_extension(target_ext)` as
    // the default output path; if input is already named with the
    // target extension (e.g. `lml export -f csv foo.csv` where
    // foo.csv is a misnamed .lml), the default output would
    // silently overwrite the input. Even with an explicit -o,
    // path-equality slips through. Canonicalize both and refuse.
    if let Some(o) = output {
        if o == input {
            return Err(format!(
                "cmd_export: output path equals input ({}); export would overwrite the source",
                input.display()
            )
            .into());
        }
        match (input.canonicalize(), o.canonicalize()) {
            (Ok(ic), Ok(oc)) if ic == oc => {
                return Err(format!(
                    "cmd_export: canonical output path {} equals input; refusing",
                    oc.display()
                )
                .into());
            }
            _ => {}
        }
    }
    let (signal, metadata) = container::read_file(input)?;
    let n_ch = signal.len();
    let t = if n_ch > 0 { signal[0].len() } else { 0 };

    // Phase 1.6: expand {stem} / {name} / {ext} / {parent} in -o.
    // Literal paths (no placeholders) pass through unchanged so
    // existing callers see no behavior change.
    //
    // Tier 3 audit (O16): refuse expanded paths containing `..`
    // path components. A crafted input filename with embedded
    // `../` (or a symlinked input whose stem contains `..`) used
    // to expand through `expand_template` and produce an output
    // path that escaped the operator's intended -o directory.
    // Now: reject any expanded path containing a ParentDir
    // component; clinical operators don't need traversal in
    // template expansion.
    let templated_output: Option<PathBuf> = output.and_then(|p| {
        let s = p.to_str()?;
        if !lamquant_core::paths::has_placeholder(s) {
            return None;
        }
        let expanded = lamquant_core::paths::expand_template(s, input)?;
        if expanded
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            eprintln!(
                "  WARNING: cmd_export: refusing -o template {:?} expansion {:?} \
                 because it contains parent-directory traversal (..); \
                 falling back to literal -o or default extension.",
                s,
                expanded.display()
            );
            return None;
        }
        Some(expanded)
    });
    let output: Option<&Path> = templated_output.as_deref().or(output);

    match format {
        "raw" => {
            // Phase 5.8 — streaming path. `decode_one_to_raw`
            // window-by-window iterates via `LmlReader::next_window` so
            // RAM cost stays at O(window_size × n_ch × 8 B) instead of
            // O(n_ch × total_samples × 8 B). Output bytes are
            // byte-identical to the load-everything path verified in
            // Phase 3.4.
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("raw"));
            let (out_ch, out_t) = decode_one_to_raw(input, &out_path)?;
            println!(
                "Exported: {} (raw int32 LE streaming, {}ch x {})",
                out_path.display(),
                out_ch,
                out_t
            );
            // `signal` (in-RAM) + `t`/`n_ch` are still referenced by
            // other arms below, but the raw arm now writes via the
            // streaming path. Suppress unused-warning when only `raw`
            // is exercised.
            let _ = (&signal, n_ch, t);
        }
        "csv" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("csv"));
            let mut f = BufWriter::new(std::fs::File::create(&out_path)?);

            // Header line: try to get channel names from metadata.
            // Tier 3 audit (O15+O17): use serde_json to parse the
            // channels array instead of substring scanning. Pre-fix
            // hand-rolled parser broke on any channel name
            // containing `]`, silently falling back to ch0..chN with
            // no warning -- operators couldn't tell whether they
            // got real names or generated placeholders. Now: parse
            // properly + emit a warning when the fallback engages.
            let mut ch_names: Vec<String> = Vec::new();
            match serde_json::from_str::<serde_json::Value>(&metadata) {
                Ok(v) => {
                    if let Some(arr) = v.get("channels").and_then(|c| c.as_array()) {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                ch_names.push(s.to_string());
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  WARNING: cmd_export csv: metadata is not valid JSON ({}); \
                         channel names will be ch0..chN",
                        e
                    );
                }
            }
            if ch_names.len() != n_ch {
                if !ch_names.is_empty() {
                    eprintln!(
                        "  WARNING: cmd_export csv: parsed {} channel names but signal has {}; \
                         falling back to ch0..chN",
                        ch_names.len(),
                        n_ch
                    );
                }
                ch_names = (0..n_ch).map(|i| format!("ch{}", i)).collect();
            }
            // Tier 3 audit (O15): escape tab/newline/CR in channel
            // names so an EDF header field containing those bytes
            // can't inject extra columns or rows into the CSV.
            let escape = |s: &str| -> String {
                s.replace('\\', "\\\\")
                    .replace('\t', "\\t")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
            };
            let header: Vec<String> = ch_names.iter().map(|s| escape(s)).collect();
            writeln!(f, "{}", header.join("\t"))?;

            // One row per sample, tab-separated
            for s in 0..t {
                for (i, ch) in signal.iter().enumerate() {
                    if i > 0 {
                        write!(f, "\t")?;
                    }
                    write!(f, "{}", ch[s])?;
                }
                writeln!(f)?;
            }
            f.flush()?;
            println!(
                "Exported: {} (CSV, {}ch x {} samples)",
                out_path.display(),
                n_ch,
                t
            );
        }
        "npy" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("npy"));
            write_npy(&out_path, &signal, n_ch, t)?;
            println!(
                "Exported: {} (NPY int64, shape [{}, {}])",
                out_path.display(),
                n_ch,
                t
            );
        }
        "mat" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("mat"));
            write_mat_v5(&out_path, &signal, n_ch, t, "EEG")?;
            println!(
                "Exported: {} (MATLAB .mat v5 int64, shape [{}, {}])",
                out_path.display(),
                n_ch,
                t
            );
        }
        "bids" => {
            let out_dir = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("bids"));
            write_bids_skeleton(&out_dir, &signal, &metadata, n_ch, t)?;
            println!(
                "Exported: {} (BIDS-EEG layout, {}ch × {} samples)",
                out_dir.display(),
                n_ch,
                t
            );
        }
        #[cfg(feature = "parquet")]
        "parquet" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("parquet"));
            write_parquet(&out_path, &signal, n_ch, t)?;
            println!(
                "Exported: {} (Parquet int64, {} columns × {} rows)",
                out_path.display(),
                n_ch,
                t
            );
        }
        #[cfg(feature = "hdf5")]
        "hdf5" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("h5"));
            write_hdf5(&out_path, &signal, n_ch, t, export_mode)?;
            println!(
                "Exported: {} (HDF5 /signal dataset int64, mode {}, shape [{}, {}])",
                out_path.display(),
                export_mode.as_str(),
                n_ch,
                t
            );
        }
        #[cfg(feature = "nwb")]
        "nwb" => {
            let out_path = output
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| input.with_extension("nwb"));
            write_nwb(&out_path, &signal, n_ch, t, export_mode)?;
            println!(
                "Exported: {} (NWB-like HDF5 ElectricalSeries, mode {}, shape [{}, {}])",
                out_path.display(),
                export_mode.as_str(),
                n_ch,
                t
            );
        }
        _ => {
            return Err(format!(
                "Unknown format: {} (supported: csv, npy, raw, mat, bids; \
                 build with --features parquet, hdf5, or nwb for those)",
                format
            )
            .into());
        }
    }
    Ok(())
}

/// Write minimal NumPy .npy v1.0 file (int64, C-contiguous, shape [n_ch, t]).
fn write_npy(path: &Path, signal: &[Vec<i64>], n_ch: usize, t: usize) -> R {
    let header_str = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({}, {}), }}",
        n_ch, t
    );
    // NPY v1.0: magic(6) + version(2) + HEADER_LEN(2) + header + padding to 64-byte align
    let prefix_len = 10; // magic + version + header_len
    let total_header = prefix_len + header_str.len() + 1; // +1 for newline
    let padded = total_header.div_ceil(64) * 64;
    let padding = padded - prefix_len - header_str.len() - 1;
    // Tier 3 audit (O6): NPY v1.0 HEADER_LEN is u16, capped at
    // 65535 bytes. Very wide signals would produce a shape string
    // exceeding the cap; `as u16` silently truncated the cast and
    // numpy read garbage. v2.0 uses u32 HEADER_LEN; for now we
    // hard-Err with a clear hint to use --format hdf5/mat which
    // don't have this limit.
    let header_body = header_str.len() + padding + 1;
    if header_body > u16::MAX as usize {
        return Err(format!(
            "write_npy: header body {} bytes exceeds NPY v1.0 u16 cap ({}); \
             use --format hdf5 or --format mat for wide-shape exports",
            header_body,
            u16::MAX
        )
        .into());
    }
    let header_len = header_body as u16;

    let mut f = BufWriter::new(std::fs::File::create(path)?);
    // Magic: \x93NUMPY
    f.write_all(&[0x93])?;
    f.write_all(b"NUMPY")?;
    // Version 1.0
    f.write_all(&[1, 0])?;
    // HEADER_LEN (u16 LE)
    f.write_all(&header_len.to_le_bytes())?;
    // Header string
    f.write_all(header_str.as_bytes())?;
    // Padding spaces + newline
    for _ in 0..padding {
        f.write_all(b" ")?;
    }
    f.write_all(b"\n")?;
    // Data: row-major (C-order), int64 LE
    for ch in signal {
        for &v in ch {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

/// Phase 5.5 — write a minimal MATLAB Level-5 `.mat` file containing
/// one `[n_ch, n_samples]` int64 matrix under `name`. Hand-rolled so
/// no `matio`/HDF5 dependency drags in. Spec: MATLAB 7.x .mat format,
/// MAT-File Format documentation §1.
///
/// File layout:
///   - 128-byte text header (124 bytes ASCII description + 2 bytes
///     version 0x0100 + 2 bytes endian marker `MI`)
///   - one `miMATRIX` element (top-level), made of subelements:
///       Array Flags (miUINT32 × 2)         — class = mxINT64_CLASS (12)
///       Dimensions (miINT32 × 2)            — [n_ch, n_samples]
///       Array Name (miINT8 string)
///       Real Part   (miINT64 × n_ch*n_samples)
///   - All elements 8-byte padded.
///
/// Column-major (Fortran order) per MATLAB convention. Signal is
/// row-major in our `Vec<Vec<i64>>`; we transpose on write so MATLAB
/// sees `[n_ch, n_samples]` and `EEG(ch, t)` indexing works directly.
fn write_mat_v5(path: &Path, signal: &[Vec<i64>], n_ch: usize, t: usize, name: &str) -> R {
    use std::io::Write as _;

    // ──── MAT element type IDs (MATLAB MAT-File Format docs) ────
    const MI_INT8: u32 = 1;
    const MI_INT32: u32 = 5;
    const MI_UINT32: u32 = 6;
    const MI_INT64: u32 = 12;
    const MI_MATRIX: u32 = 14;
    // mxINT64_CLASS = 12 (matches MATLAB R2008b+).
    const MX_INT64_CLASS: u32 = 12;

    fn align8(n: usize) -> usize {
        (8 - (n & 7)) & 7
    }

    // Tier 3 audit (O7): validate dimensions fit in MAT-v5's
    // i32 / u32 fields BEFORE writing anything. Pre-fix casts
    // wrapped silently for n_ch > i32::MAX, t > i32::MAX, or
    // n_elems × 8 > u32::MAX, producing structurally-corrupt
    // .mat files with negative dimensions or truncated element
    // sizes.
    if n_ch > i32::MAX as usize || t > i32::MAX as usize {
        return Err(format!(
            "write_mat_v5: dims {}×{} exceed MAT-v5 i32 cap ({}); \
             use --format hdf5 (MAT-v7.3) for larger matrices",
            n_ch,
            t,
            i32::MAX
        )
        .into());
    }
    let n_elems = n_ch
        .checked_mul(t)
        .ok_or_else(|| format!("write_mat_v5: n_elems = {} × {} overflows usize", n_ch, t))?;
    let data_bytes = n_elems
        .checked_mul(8)
        .ok_or_else(|| format!("write_mat_v5: data_bytes = {} × 8 overflows usize", n_elems))?;
    if data_bytes > u32::MAX as usize {
        return Err(format!(
            "write_mat_v5: data_bytes {} exceeds MAT-v5 u32 element-size cap ({}); \
             use --format hdf5 for matrices over 4 GiB",
            data_bytes,
            u32::MAX
        )
        .into());
    }

    // Build the miMATRIX subelement bytes in memory.
    let mut body: Vec<u8> = Vec::new();

    // 1. Array Flags subelement: type miUINT32, size 8 bytes,
    //    flags-class word + nzmax (0 for full).
    body.extend_from_slice(&MI_UINT32.to_le_bytes());
    body.extend_from_slice(&8u32.to_le_bytes());
    let flags_class: u32 = MX_INT64_CLASS; // no complex / global / logical flags
    body.extend_from_slice(&flags_class.to_le_bytes());
    body.extend_from_slice(&0u32.to_le_bytes()); // nzmax

    // 2. Dimensions subelement: miINT32 × 2 = 8 bytes. Safe casts
    // -- bounds checked above.
    body.extend_from_slice(&MI_INT32.to_le_bytes());
    body.extend_from_slice(&8u32.to_le_bytes());
    body.extend_from_slice(&(n_ch as i32).to_le_bytes());
    body.extend_from_slice(&(t as i32).to_le_bytes());

    // 3. Array Name subelement: miINT8 string + 8-byte align padding.
    let name_bytes = name.as_bytes();
    body.extend_from_slice(&MI_INT8.to_le_bytes());
    body.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    body.extend_from_slice(name_bytes);
    for _ in 0..align8(name_bytes.len()) {
        body.push(0);
    }

    // 4. Real Part subelement: miINT64 × (n_ch * t), column-major.
    // `n_elems` + `data_bytes` already computed + bounds-checked
    // at function entry; no duplicate unchecked math here.
    body.extend_from_slice(&MI_INT64.to_le_bytes());
    body.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    // Column-major: for each sample t, then each channel — but our
    // `signal` is row-major (signal[ch][t]). Output [ch, t] in
    // column-major = for s in 0..t { for ch in 0..n_ch { ... } }.
    // (MATLAB reads `EEG(ch, s)` → `signal[ch][s]`.)
    for s in 0..t {
        for ch in 0..n_ch {
            body.extend_from_slice(&signal[ch][s].to_le_bytes());
        }
    }
    for _ in 0..align8(data_bytes) {
        body.push(0);
    }

    // 128-byte header.
    let mut hdr = Vec::with_capacity(128);
    let descr = format!(
        "MATLAB 5.0 MAT-file, Platform: lml/{}, n_channels={}, n_samples={}",
        env!("CARGO_PKG_VERSION"),
        n_ch,
        t
    );
    let descr_bytes = descr.as_bytes();
    let descr_take = descr_bytes.len().min(124);
    hdr.extend_from_slice(&descr_bytes[..descr_take]);
    for _ in descr_take..124 {
        hdr.push(b' ');
    }
    hdr.extend_from_slice(&0x0100u16.to_le_bytes()); // version
    hdr.extend_from_slice(b"IM"); // endian marker (little-endian = 'I' 'M' reading as "MI")
    debug_assert_eq!(hdr.len(), 128);

    let mut f = BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&hdr)?;
    // Top-level miMATRIX tag + body.
    f.write_all(&MI_MATRIX.to_le_bytes())?;
    f.write_all(&(body.len() as u32).to_le_bytes())?;
    f.write_all(&body)?;
    f.flush()?;
    Ok(())
}

/// Phase 5.7 — write a minimal BIDS-EEG directory skeleton.
///
/// Layout produced (single-subject, single-session, single-run):
///
///   <out_dir>/
///     dataset_description.json
///     README
///     sub-01/
///       eeg/
///         sub-01_task-rest_eeg.eeg       (raw int16 LE, channel-major)
///         sub-01_task-rest_eeg.json      (BIDS sidecar)
///         sub-01_task-rest_channels.tsv  (BIDS channels)
///
/// Real BIDS layouts carry subject IDs, session/run numbers, and
/// participant tsv. This export gives downstream BIDS-aware tooling
/// (MNE-BIDS, fmriprep ingest) a starting tree that they can rename
/// to the actual subject naming. Bible R30 — refuse to silently
/// invent IDs from metadata; user renames the `sub-01` dir after.
fn write_bids_skeleton(
    out_dir: &Path,
    signal: &[Vec<i64>],
    metadata: &str,
    n_ch: usize,
    t: usize,
) -> R {
    use std::io::Write as _;
    let sub_dir = out_dir.join("sub-01").join("eeg");
    std::fs::create_dir_all(&sub_dir)?;

    // ── dataset_description.json ──
    let dd = format!(
        "{{\"Name\":\"lamquant export\",\"BIDSVersion\":\"1.8.0\",\
         \"DatasetType\":\"raw\",\"GeneratedBy\":[{{\"Name\":\"lml\",\
         \"Version\":\"{}\"}}]}}",
        env!("CARGO_PKG_VERSION")
    );
    std::fs::write(out_dir.join("dataset_description.json"), dd)?;
    std::fs::write(
        out_dir.join("README"),
        "Generated by `lml export -f bids`. Rename sub-01/ to the actual subject ID before publishing.\n",
    )?;

    // ── sub-01_task-rest_eeg.eeg: int16 LE, channel-major ──
    // Clamp i64 → i16 with saturation; emit a warning summary
    // (Tier 3 audit O3) when clamps occur. Most EDF EEG fits in
    // i16 by design, but BDF i24 sources routinely exceed it --
    // pre-fix code silently saturated every out-of-range sample
    // with no diagnostic. Clinical seizure thresholds could
    // misfire on saturated amplitudes.
    let eeg_path = sub_dir.join("sub-01_task-rest_eeg.eeg");
    let mut f = BufWriter::new(std::fs::File::create(&eeg_path)?);
    let mut clamps_low: u64 = 0;
    let mut clamps_high: u64 = 0;
    for ch in signal {
        for &v in ch {
            if v < i16::MIN as i64 {
                clamps_low += 1;
            } else if v > i16::MAX as i64 {
                clamps_high += 1;
            }
            let clamped = v.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            f.write_all(&clamped.to_le_bytes())?;
        }
    }
    f.flush()?;
    if clamps_low > 0 || clamps_high > 0 {
        eprintln!(
            "  WARNING: BIDS i16 export clamped {} samples (low: {} below i16::MIN, high: {} above i16::MAX). \
             Source likely BDF/i24; consider --format hdf5 or --format mat for lossless export.",
            clamps_low + clamps_high,
            clamps_low,
            clamps_high
        );
    }

    // ── sidecar JSON ──
    // Try to parse sample_rate / channels from container metadata; fall
    // back to safe defaults.
    let sample_rate = {
        let needle = "\"sample_rate\":";
        if let Some(p) = metadata.find(needle) {
            let tail = &metadata[p + needle.len()..];
            let end = tail
                .find(|c: char| {
                    !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E')
                })
                .unwrap_or(tail.len());
            tail[..end].trim().parse::<f64>().unwrap_or(250.0)
        } else {
            250.0
        }
    };
    let mut ch_names: Vec<String> = Vec::new();
    if let Some(start) = metadata.find("\"channels\":[") {
        let rest = &metadata[start + 12..];
        if let Some(end) = rest.find(']') {
            for part in rest[..end].split(',') {
                let name = part.trim().trim_matches('"').to_string();
                if !name.is_empty() {
                    ch_names.push(name);
                }
            }
        }
    }
    if ch_names.len() != n_ch {
        ch_names = (0..n_ch).map(|i| format!("EEG{:03}", i + 1)).collect();
    }
    // Tier 3 audit (O18): bound sample_rate to a clinical floor
    // before division so RecordingDuration can never become Inf
    // (sample_rate.max(1e-9) protected against zero but a finite-
    // tiny value still produced Inf when t was modest, and
    // serde_json/parsers reject "Infinity" as invalid JSON).
    // Use 1.0 Hz as the floor -- any clinical EEG is at least 1 Hz;
    // anything lower is corrupt metadata + we surface as a warning.
    let safe_sample_rate = if sample_rate.is_finite() && sample_rate >= 1.0 {
        sample_rate
    } else {
        eprintln!(
            "  WARNING: BIDS export: sample_rate {} is below the 1 Hz clinical floor; \
             RecordingDuration computed against placeholder 1.0 Hz to keep the JSON valid",
            sample_rate
        );
        1.0
    };
    let recording_duration = t as f64 / safe_sample_rate;
    let sidecar = format!(
        "{{\"TaskName\":\"rest\",\"SamplingFrequency\":{},\"EEGReference\":\"unknown\",\
         \"PowerLineFrequency\":50,\"SoftwareFilters\":\"n/a\",\"EEGChannelCount\":{},\
         \"RecordingDuration\":{}}}",
        safe_sample_rate, n_ch, recording_duration,
    );
    std::fs::write(sub_dir.join("sub-01_task-rest_eeg.json"), sidecar)?;

    // ── channels.tsv ──
    let mut tsv = String::from("name\ttype\tunits\n");
    for ch in &ch_names {
        tsv.push_str(ch);
        tsv.push_str("\tEEG\tuV\n");
    }
    std::fs::write(sub_dir.join("sub-01_task-rest_channels.tsv"), tsv)?;

    Ok(())
}

/// Phase 5.4 — Parquet export. Writes one int64 column per channel,
/// names = `ch0..chN`. Uses arrow-rs + parquet via `ArrowWriter`.
#[cfg(feature = "parquet")]
fn write_parquet(path: &Path, signal: &[Vec<i64>], n_ch: usize, t: usize) -> R {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    // Build an Arrow schema with one Int64 column per channel.
    let fields: Vec<arrow_schema::Field> = (0..n_ch)
        .map(|i| arrow_schema::Field::new(format!("ch{i}"), arrow_schema::DataType::Int64, false))
        .collect();
    let schema = Arc::new(arrow_schema::Schema::new(fields));
    let columns: Vec<Arc<dyn arrow_array::Array>> = signal
        .iter()
        .map(|c| -> Arc<dyn arrow_array::Array> {
            Arc::new(arrow_array::Int64Array::from(c.clone()))
        })
        .collect();
    let batch = arrow_array::RecordBatch::try_new(schema.clone(), columns)?;
    let _ = t; // n rows derived from batch
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Phase 5.6 — HDF5 export. Writes a single `/signal` 2-D int64
/// dataset shaped `[n_ch, t]`, mirroring the NPY export layout.
#[cfg(feature = "hdf5")]
fn write_hdf5(
    path: &Path,
    signal: &[Vec<i64>],
    n_ch: usize,
    t: usize,
    lossless_mode: LosslessMode,
) -> R {
    use hdf5_metno::File;
    let mut flat: Vec<i64> = Vec::with_capacity(n_ch * t);
    for ch in signal {
        flat.extend_from_slice(ch);
    }
    let f = File::create(path)?;
    let arr = ndarray::Array2::from_shape_vec((n_ch, t), flat)?;
    f.new_dataset_builder().with_data(&arr).create("signal")?;
    f.new_dataset_builder()
        .with_data(lossless_mode.as_str().as_bytes())
        .create("lossless_mode")?;
    Ok(())
}

#[cfg(feature = "nwb")]
fn write_nwb(
    path: &Path,
    signal: &[Vec<i64>],
    n_ch: usize,
    t: usize,
    lossless_mode: LosslessMode,
) -> R {
    use hdf5_metno::File;
    let mut flat: Vec<i64> = Vec::with_capacity(n_ch * t);
    for ch in signal {
        flat.extend_from_slice(ch);
    }
    let f = File::create(path)?;
    let acquisition = f.create_group("acquisition")?;
    let series = acquisition.create_group("ElectricalSeries")?;
    let arr = ndarray::Array2::from_shape_vec((n_ch, t), flat)?;
    series
        .new_dataset_builder()
        .with_data(&arr)
        .create("data")?;
    series
        .new_dataset_builder()
        .with_data(lossless_mode.as_str().as_bytes())
        .create("lossless_mode")?;
    series
        .new_dataset_builder()
        .with_data(b"lml")
        .create("source_codec")?;
    Ok(())
}

// ── NWB/HDF5 pack/unpack (ADR 0051 Track 3) ──
//
// Drives the system `h5repack` with the LML H5Z filter. h5repack is used
// deliberately: an in-process repack cannot safely shrink a real NWB — HDF5
// does not reclaim freed space in place, and rebuilding a file piecemeal breaks
// object references (e.g. an ElectricalSeries' electrodes DynamicTableRegion).
// h5repack does the compact rewrite + reference fix-up correctly; `lml nwb`
// just makes it a one-liner (locates the filter, sets the plugin path, and
// verifies losslessness via our own reader). Filter id matches the plugin.
#[cfg(feature = "nwb")]
const FILTER_SO: &str = "liblamquant_lml_h5filter.so";
#[cfg(feature = "nwb")]
const FILTER_UD: &str = "32200";
/// Value-returning result (the bin's `R` alias is fixed to `()`).
#[cfg(feature = "nwb")]
type Rt<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[cfg(feature = "nwb")]
fn cmd_nwb(cmd: NwbCmd) -> R {
    match cmd {
        NwbCmd::Pack {
            input,
            output,
            plugin_so,
            no_verify,
        } => cmd_nwb_pack(&input, &output, plugin_so.as_deref(), !no_verify),
        NwbCmd::Unpack {
            input,
            output,
            plugin_so,
        } => cmd_nwb_unpack(&input, &output, plugin_so.as_deref()),
    }
}

#[cfg(not(feature = "nwb"))]
fn cmd_nwb(_cmd: NwbCmd) -> R {
    Err("`lml nwb` requires building lml with `--features nwb` (pulls in libhdf5). \
         The default `host` build doesn't link HDF5."
        .into())
}

/// Resolve the directory containing `liblamquant_lml_h5filter.so` for
/// `HDF5_PLUGIN_PATH`, trying (in order): an explicit path, `$LAMQUANT_H5FILTER`,
/// next to the running `lml` binary, then common dev build dirs.
#[cfg(feature = "nwb")]
fn filter_plugin_dir(explicit: Option<&Path>) -> Rt<PathBuf> {
    fn so_dir(p: &Path) -> Option<PathBuf> {
        let cand = if p.is_dir() { p.join(FILTER_SO) } else { p.to_path_buf() };
        if cand.is_file() {
            cand.parent().map(|d| d.to_path_buf())
        } else {
            None
        }
    }
    let mut tried = Vec::new();
    let mut try_one = |p: &Path| -> Option<PathBuf> {
        match so_dir(p) {
            Some(d) => Some(d),
            None => {
                tried.push(p.display().to_string());
                None
            }
        }
    };
    if let Some(p) = explicit {
        if let Some(d) = try_one(p) {
            return Ok(d);
        }
    }
    if let Ok(env) = std::env::var("LAMQUANT_H5FILTER") {
        if let Some(d) = try_one(Path::new(&env)) {
            return Ok(d);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(d) = try_one(dir) {
                return Ok(d);
            }
        }
    }
    for p in ["target/release", "target/debug", "codec-lossless/target/release"] {
        if let Some(d) = try_one(Path::new(p)) {
            return Ok(d);
        }
    }
    Err(format!(
        "could not locate {FILTER_SO}. Build it with \
         `cargo build -p lamquant-lml-h5filter --release`, then pass \
         `--plugin-so <path>` or set $LAMQUANT_H5FILTER. Tried: {}",
        tried.join(", ")
    )
    .into())
}

#[cfg(feature = "nwb")]
fn run_h5repack(args: &[&str], plugin_dir: &Path) -> R {
    let status = std::process::Command::new("h5repack")
        .args(args)
        .env("HDF5_PLUGIN_PATH", plugin_dir)
        .status()
        .map_err(|e| {
            format!("failed to launch h5repack ({e}). Install the HDF5 CLI tools (libhdf5).")
        })?;
    if !status.success() {
        return Err(format!("h5repack exited with {status}").into());
    }
    Ok(())
}

#[cfg(feature = "nwb")]
fn path_str(p: &Path) -> Rt<&str> {
    p.to_str()
        .ok_or_else(|| format!("non-UTF-8 path: {}", p.display()).into())
}

#[cfg(feature = "nwb")]
fn cmd_nwb_pack(input: &Path, output: &Path, plugin_so: Option<&Path>, verify: bool) -> R {
    let dir = filter_plugin_dir(plugin_so)?;
    let ud = format!("UD={FILTER_UD},0,0");
    run_h5repack(&["-f", &ud, path_str(input)?, path_str(output)?], &dir)?;

    let si = std::fs::metadata(input)?.len();
    let so = std::fs::metadata(output)?.len();
    let ratio = if so > 0 { si as f64 / so as f64 } else { 0.0 };
    println!(
        "Packed {} -> {}  ({} -> {} bytes, {ratio:.3}x)",
        input.display(),
        output.display(),
        si,
        so
    );

    if verify {
        // Decode path needs the filter discoverable in-process.
        std::env::set_var("HDF5_PLUGIN_PATH", &dir);
        verify_int_datasets_match(input, output)?;
        println!("Verified: lossless (every integer dataset round-trips identically).");
    }
    Ok(())
}

#[cfg(feature = "nwb")]
fn cmd_nwb_unpack(input: &Path, output: &Path, plugin_so: Option<&Path>) -> R {
    let dir = filter_plugin_dir(plugin_so)?;
    // `-f NONE` strips all filters; HDF5_PLUGIN_PATH lets h5repack DECODE the
    // LML-filtered source.
    run_h5repack(&["-f", "NONE", path_str(input)?, path_str(output)?], &dir)?;
    println!(
        "Unpacked {} -> {} (plain native NWB/HDF5)",
        input.display(),
        output.display()
    );
    Ok(())
}

/// Read both files' integer datasets through our reader and assert they match
/// (same paths, same values). The strong lossless gate for pack.
#[cfg(feature = "nwb")]
fn verify_int_datasets_match(a: &Path, b: &Path) -> R {
    use std::collections::BTreeMap;
    let to_map = |p: &Path| -> Rt<BTreeMap<String, Vec<Vec<i64>>>> {
        Ok(lamquant_core::nwb::read_int_signals(p)?
            .into_iter()
            .map(|s| (s.h5_path, s.signal))
            .collect())
    };
    let (ma, mb) = (to_map(a)?, to_map(b)?);
    if ma.keys().ne(mb.keys()) {
        return Err(format!(
            "integer-dataset set differs after pack: {:?} vs {:?}",
            ma.keys().collect::<Vec<_>>(),
            mb.keys().collect::<Vec<_>>()
        )
        .into());
    }
    for (k, va) in &ma {
        if mb.get(k) != Some(va) {
            return Err(format!("dataset {k} differs after pack — NOT lossless").into());
        }
    }
    Ok(())
}

// ── Feature 8: diff ──

fn cmd_diff(a_path: &Path, b_path: &Path) -> R {
    let (sig_a, _meta_a) = container::read_file(a_path)?;
    let (sig_b, _meta_b) = container::read_file(b_path)?;

    let n_ch_a = sig_a.len();
    let n_ch_b = sig_b.len();
    let t_a = if n_ch_a > 0 { sig_a[0].len() } else { 0 };
    let t_b = if n_ch_b > 0 { sig_b[0].len() } else { 0 };

    println!("A: {} ({}ch x {} samples)", a_path.display(), n_ch_a, t_a);
    println!("B: {} ({}ch x {} samples)", b_path.display(), n_ch_b, t_b);

    // Tier 3 audit (O11 follow-up): return Err instead of
    // std::process::exit so the function's owned Vec<Vec<i64>>
    // signals + tracing span guards drop cleanly. main()'s top-
    // level error handler turns the Err into a non-zero exit
    // for the operator. Same visible behavior, no destructor
    // bypass.
    if n_ch_a != n_ch_b {
        return Err(format!(
            "DIFFERENT: channel count mismatch ({} vs {})",
            n_ch_a, n_ch_b
        )
        .into());
    }
    if t_a != t_b {
        return Err(format!("DIFFERENT: sample count mismatch ({} vs {})", t_a, t_b).into());
    }

    let n_ch = n_ch_a;
    let t = t_a;
    let mut total_diffs: u64 = 0;
    let mut first_diff: Option<(usize, usize, i64, i64)> = None;

    // Compare window-by-window (using 2500-sample windows for reporting)
    let window_size = 2500usize;
    let n_windows = t.div_ceil(window_size).max(1);
    let mut windows_match = 0u64;
    let mut windows_differ = 0u64;

    for w in 0..n_windows {
        let start = w * window_size;
        let end = (start + window_size).min(t);
        let mut window_ok = true;
        for ch in 0..n_ch {
            // Tier 3 audit (O11): bounds-check per channel. Pre-fix
            // assumed sig_a[0].len() == every channel.len() == t,
            // but ragged channels (legal after a partial
            // cmd_recover, or any corrupt container with mismatched
            // lengths) caused out-of-bounds panic. Now: cap `end`
            // against the actual channel length and report short-
            // channels as a diff diagnostic.
            let ch_end = end.min(sig_a[ch].len()).min(sig_b[ch].len());
            if ch_end < end {
                total_diffs = total_diffs.saturating_add((end - ch_end) as u64);
                window_ok = false;
                if first_diff.is_none() {
                    first_diff = Some((ch, ch_end, 0, 0));
                }
            }
            for s in start..ch_end {
                if sig_a[ch][s] != sig_b[ch][s] {
                    total_diffs += 1;
                    window_ok = false;
                    if first_diff.is_none() {
                        first_diff = Some((ch, s, sig_a[ch][s], sig_b[ch][s]));
                    }
                }
            }
        }
        if window_ok {
            windows_match += 1;
        } else {
            windows_differ += 1;
        }
    }

    println!(
        "\nWindows: {} match, {} differ (of {})",
        windows_match, windows_differ, n_windows
    );
    if total_diffs == 0 {
        println!("Result: IDENTICAL");
        Ok(())
    } else {
        let mut msg = format!("Result: {} samples differ", total_diffs);
        if let Some((ch, s, va, vb)) = first_diff {
            msg.push_str(&format!(
                " (first diff: ch={} sample={} A={} B={})",
                ch, s, va, vb
            ));
        }
        Err(msg.into())
    }
}

// ── Helper: human-readable byte size ──

fn human_bytes(n: u64) -> String {
    if n < 1024 {
        return format!("{} B", n);
    }
    let mut val = n as f64;
    for unit in &["KB", "MB", "GB", "TB"] {
        val /= 1024.0;
        if val < 1024.0 {
            return format!("{:.1} {}", val, unit);
        }
    }
    format!("{:.1} PB", val / 1024.0)
}

/// Human-readable duration: "1h 23m" or "45m 12s" or "3.2s"
fn human_duration(secs: f64) -> String {
    if secs < 0.0 || secs.is_nan() {
        return "0s".to_string();
    }
    if secs < 60.0 {
        return format!("{:.0}s", secs);
    }
    let total_s = secs as u64;
    let h = total_s / 3600;
    let m = (total_s % 3600) / 60;
    let s = total_s % 60;
    if h > 0 {
        format!("{}h {:02}m", h, m)
    } else {
        format!("{}m {:02}s", m, s)
    }
}

fn cmd_recover(input: &Path, output: &Path) -> R {
    // Audit-2026-05-11 Fix-C1 + Tier 3 audit (O1): route through
    // container::parse_header so the header layout matches what
    // `container::write_file` emits, AND fix the cluster of bugs
    // the audit found:
    //   * pre-fix hardcoded `sample_rate = 250.0` in the output;
    //     non-250 Hz sources were silently re-tagged as 250 Hz.
    //   * pre-fix used `start = w * hdr.window_size` for the
    //     write offset, but real per-window length scales with
    //     sample_rate. Time-shifted garbage for any non-250 Hz.
    //   * pre-fix `end - start` underflowed if `start >=
    //     hdr.total_samples` (corrupted n_windows in header).
    //   * pre-fix accepted `output == input` and silently
    //     truncated the input file.
    //   * pre-fix wrote an all-zeros output and exited 0 when
    //     good_windows == 0.
    //   * pre-fix handed attacker-controlled u32 payload_len
    //     straight to lml::decompress; OOM via a chain of huge
    //     claimed lengths.
    if input == output {
        return Err(format!(
            "cmd_recover: output path equals input ({}); recovery would truncate the source",
            input.display()
        )
        .into());
    }
    let input_canon = input.canonicalize().ok();
    let output_canon = output.canonicalize().ok();
    if let (Some(i), Some(o)) = (&input_canon, &output_canon) {
        if i == o {
            return Err(format!(
                "cmd_recover: canonical output path {} equals input; refusing",
                o.display()
            )
            .into());
        }
    }

    let data = std::fs::read(input)?;
    let hdr =
        container::parse_header(&data).map_err(|e| format!("invalid container header: {}", e))?;

    // Parse sample_rate from the metadata JSON. container::write_file
    // takes sample_rate as a separate argument and the header stores
    // it via the window-length-index, so we recover it via the
    // metadata blob (encoder writes it there). Defaults to 250 Hz
    // ONLY if the metadata genuinely lacks the field, with a stderr
    // warning so the operator knows the rate was a guess.
    let sample_rate: f64 = match serde_json::from_str::<serde_json::Value>(&hdr.metadata)
        .ok()
        .and_then(|v| v.get("sample_rate").and_then(|sr| sr.as_f64()))
    {
        Some(sr) if sr.is_finite() && sr > 0.0 => sr,
        _ => {
            eprintln!(
                "  WARNING: cmd_recover: metadata lacks finite positive sample_rate; \
                 defaulting to 250 Hz (output will be tagged 250 Hz regardless of source)"
            );
            250.0
        }
    };

    // Real per-window stride scales with sample_rate. The encoder
    // packs `actual_window = window_size * sample_rate / 250.0`
    // samples per window (see container::write_file). Recovery
    // MUST use the same stride or the recovered signal is
    // time-shifted garbage.
    let actual_window: usize = ((hdr.window_size as f64) * sample_rate / 250.0) as usize;
    let actual_window = actual_window.max(1); // refuse zero-stride

    // Bound per-window payload to a sane ceiling so an adversarial
    // .lml with a chain of 1 GiB claimed lengths can't OOM the
    // recovery tool. 64 MiB per window is generous (real LML
    // windows are KB-scale).
    const RECOVER_MAX_PAYLOAD: usize = 64 * 1024 * 1024;

    let mut pos = hdr.payload_start;
    let mut recovered_signal = vec![vec![0i64; hdr.total_samples]; hdr.n_ch];
    let mut good_windows = 0usize;
    let mut max_populated: usize = 0;
    let mut bad_windows: Vec<(usize, String)> = Vec::new();

    for w in 0..hdr.n_windows {
        if pos + 4 > data.len() {
            break;
        }
        let payload_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + payload_len > data.len() {
            break;
        }
        if payload_len > RECOVER_MAX_PAYLOAD {
            bad_windows.push((
                w,
                format!(
                    "payload_len {} > recover cap {} (suspected adversarial input)",
                    payload_len, RECOVER_MAX_PAYLOAD
                ),
            ));
            pos += payload_len;
            continue;
        }

        match lml::decompress(&data[pos..pos + payload_len]) {
            Ok(window) => {
                let start = w * actual_window;
                if start >= hdr.total_samples {
                    bad_windows.push((
                        w,
                        format!(
                            "window {} start {} exceeds total_samples {} (header over-claims n_windows)",
                            w, start, hdr.total_samples
                        ),
                    ));
                } else {
                    for ch in 0..hdr.n_ch.min(window.len()) {
                        let end = (start + window[ch].len()).min(hdr.total_samples);
                        let copy_len = end - start; // safe: start < total_samples
                        recovered_signal[ch][start..start + copy_len]
                            .copy_from_slice(&window[ch][..copy_len]);
                        if end > max_populated {
                            max_populated = end;
                        }
                    }
                    good_windows += 1;
                }
            }
            Err(e) => {
                bad_windows.push((w, format!("{}", e)));
            }
        }
        pos += payload_len;
    }

    if good_windows == 0 {
        return Err(format!(
            "cmd_recover: zero windows recovered from {} (refusing to write all-zeros salvage); \
             {} failed windows: {}",
            input.display(),
            bad_windows.len(),
            bad_windows
                .iter()
                .take(3)
                .map(|(w, e)| format!("[{} {}]", w, e))
                .collect::<Vec<_>>()
                .join(" ")
        )
        .into());
    }

    // Trim to the actual maximum populated sample index instead of
    // the wrong `good_windows * window_size` product (the old formula
    // discarded successfully-decoded windows when failures were
    // non-contiguous).
    let trimmed: Vec<Vec<i64>> = recovered_signal
        .iter()
        .map(|ch| ch[..max_populated.min(ch.len())].to_vec())
        .collect();

    container::write_file(
        output,
        &trimmed,
        sample_rate,
        hdr.window_size,
        0,
        &hdr.metadata,
    )?;

    println!("Recovered {}/{} windows", good_windows, hdr.n_windows);
    if !bad_windows.is_empty() {
        println!("Failed windows:");
        for (w, e) in &bad_windows {
            println!("  Window {}: {}", w, e);
        }
    }
    println!("Output: {}", output.display());
    Ok(())
}

fn cmd_archive(input: &Path, output: Option<&Path>, zstd_level: i32) -> R {
    let _span = tracing::info_span!(
        "archive",
        input = %input.display(),
        zstd_level = zstd_level,
    )
    .entered();
    let output_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let name = input
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "archive".to_string());
        input
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}.lma", name))
    });

    let t0 = Instant::now();
    println!("Archiving {} → {}", input.display(), output_path.display());

    let summary = lma::pack_archive(input, &output_path, zstd_level, true, None)?;

    let elapsed = t0.elapsed();
    println!("\nDone in {:.1}s", elapsed.as_secs_f64());
    println!(
        "  {} files ({} LML, {} zstd, {} stored)",
        summary.n_files, summary.counts_lml, summary.counts_zstd, summary.counts_store
    );
    println!(
        "  {} → {} ({:.2}x CR)",
        human_bytes(summary.original_bytes),
        human_bytes(summary.archive_bytes),
        summary.cr
    );
    if !summary.errors.is_empty() {
        println!(
            "  {} files fell back to zstd compression",
            summary.errors.len()
        );
    }
    Ok(())
}

fn cmd_extract(input: &Path, output: Option<&Path>, verify: bool) -> R {
    let _span = tracing::info_span!(
        "extract",
        input = %input.display(),
        verify = verify,
    )
    .entered();
    let output_dir = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "extracted".to_string());
        input.parent().unwrap_or(Path::new(".")).join(stem)
    });

    let t0 = Instant::now();
    println!("Extracting {} → {}", input.display(), output_dir.display());

    let summary = lma::unpack_archive(input, &output_dir, verify, true, None)?;

    let elapsed = t0.elapsed();
    println!("\nDone in {:.1}s", elapsed.as_secs_f64());
    println!(
        "  {} files extracted ({} LML, {} zstd, {} stored)",
        summary.n_files, summary.counts_lml, summary.counts_zstd, summary.counts_store
    );
    if !summary.errors.is_empty() {
        return Err(format!("{} file(s) failed during extraction", summary.errors.len()).into());
    }
    Ok(())
}

fn cmd_list_archive(input: &Path) -> R {
    let entries = lma::list_archive(input)?;

    let mut total_orig: u64 = 0;
    let mut total_comp: u64 = 0;

    println!(
        "{:<60} {:>12} {:>12} {:>8} SHA256",
        "PATH", "ORIGINAL", "COMPRESSED", "METHOD"
    );
    println!("{}", "-".repeat(110));

    for entry in &entries {
        total_orig += entry.original_size;
        total_comp += entry.compressed_size;
        let method_str = match entry.method {
            lma::Method::Lml => "lml",
            lma::Method::Zstd => "zstd",
            lma::Method::Store => "store",
            _ => "unknown",
        };
        let display_path = if entry.path.chars().count() > 60 {
            let skip = entry.path.chars().count() - 60;
            entry.path.chars().skip(skip).collect::<String>()
        } else {
            entry.path.clone()
        };
        let sha_prefix = &entry.sha256[..entry.sha256.len().min(12)];
        println!(
            "{:<60} {:>12} {:>12} {:>8} {}",
            display_path,
            human_bytes(entry.original_size),
            human_bytes(entry.compressed_size),
            method_str,
            sha_prefix
        );
    }

    println!("{}", "-".repeat(110));
    let cr = if total_comp > 0 {
        total_orig as f64 / total_comp as f64
    } else {
        0.0
    };
    println!(
        "{} files, {} original → {} compressed ({:.2}x CR)",
        entries.len(),
        human_bytes(total_orig),
        human_bytes(total_comp),
        cr
    );
    Ok(())
}

/// Phase-Lossless — `lml ls foo.lma [--tree]`. Browse an LMA archive
/// without extracting it. Foundation for OS file-manager backends:
/// Ark / Nautilus / Finder / Explorer plugins all need a way to peek
/// inside the archive without unpacking. Pairs with `lml cat` for
/// single-entry extraction.
fn cmd_ls(input: &Path, tree: bool, long: bool) -> R {
    let entries = lma::list_archive(input)?;

    if long {
        // Machine-readable, versioned wire format for OS plugin
        // shell-outs (Ark CliPlugin, Nautilus extension, etc.).
        //
        //   Line 1:   `#lml-ls schema=1`
        //   Line N>1: <original_size>\t<compressed_size>\t<method>\t<sha256>\t<path>
        //
        // The schema marker line is the FIRST stdout line. Consumers
        // MUST validate it before parsing entries. `#` is a comment
        // glyph that lml's archive writer forbids as the first
        // character of an archive entry path, so it cannot collide
        // with a legitimate entry name.
        //
        // Future schema bumps: if we add a column, bump to schema=2.
        // Old consumers that pinned schema=1 fail closed; new
        // consumers handle both. Path stays last so embedded `/`
        // (nested entries) parses cleanly; any literal `\t`, `\n`,
        // or `\r` inside a path -- which `lml`'s archive writer
        // already rejects -- would corrupt the wire format, so we
        // re-validate per entry below and refuse to emit a malformed
        // line rather than silently truncate downstream.
        println!("#lml-ls schema=1");
        for entry in &entries {
            let method_str = match entry.method {
                lma::Method::Lml => "lml",
                lma::Method::Zstd => "zstd",
                lma::Method::Store => "store",
                _ => "unknown",
            };
            // Byte-level check catches both literal control chars and
            // (theoretically) overlong UTF-8 encodings of NUL (0xC0 0x80).
            // Debug-format the path on rejection so the embedded
            // control byte doesn't break the stderr line format.
            if entry
                .path
                .bytes()
                .any(|b| b == b'\t' || b == b'\n' || b == b'\r' || b == 0)
            {
                return Err(format!(
                    "entry path {:?} contains control byte (tab/newline/null) -- \
                     would corrupt --long wire format",
                    entry.path
                )
                .into());
            }
            println!(
                "{}\t{}\t{}\t{}\t{}",
                entry.original_size, entry.compressed_size, method_str, entry.sha256, entry.path,
            );
        }
        return Ok(());
    }

    if !tree {
        // Flat one-line-per-entry listing. Like `tar tf`.
        for entry in &entries {
            println!("{}", entry.path);
        }
        return Ok(());
    }

    // Tree-style with totals header + per-entry size / method /
    // sha256 prefix. Format chosen to match `tree -L 1` semantics:
    // every entry is a sibling under the root; the manifest is flat.
    let total_orig: u64 = entries.iter().map(|e| e.original_size).sum();
    let total_comp: u64 = entries.iter().map(|e| e.compressed_size).sum();
    let cr = if total_comp > 0 {
        total_orig as f64 / total_comp as f64
    } else {
        0.0
    };

    let archive_name = input
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| input.display().to_string());

    println!(
        "{} ({}, {} entries, {:.2}x CR)",
        archive_name,
        human_bytes(total_comp),
        entries.len(),
        cr
    );

    for (i, entry) in entries.iter().enumerate() {
        let connector = if i == entries.len() - 1 {
            "└── "
        } else {
            "├── "
        };
        let method_str = match entry.method {
            lma::Method::Lml => "lml",
            lma::Method::Zstd => "zstd",
            lma::Method::Store => "store",
            _ => "unknown",
        };
        let sha_prefix = &entry.sha256[..entry.sha256.len().min(12)];
        println!(
            "{}{:<48}  {:>10}  {:<6}  sha256:{}",
            connector,
            entry.path,
            human_bytes(entry.original_size),
            method_str,
            sha_prefix,
        );
    }
    Ok(())
}

/// Phase-Lossless — `lml cat foo.lma <entry-path>`. Extract a single
/// LMA entry to stdout. Composes with `less` / `grep` / `fzf`.
/// Path-traversal-rejected by the same safety logic as
/// `unpack_archive` -- the requested entry must be present in the
/// manifest verbatim.
fn cmd_cat(input: &Path, entry_path: &str) -> R {
    let bytes = lma::read_entry(input, entry_path).map_err(
        |e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("lml cat: failed to read entry {}: {}", entry_path, e).into()
        },
    )?;
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    lock.write_all(&bytes)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("lml cat: stdout write failed: {}", e).into()
        })?;
    Ok(())
}

/// v1.2 V — parse a byte-size string with K/M/G suffix.
///
/// Accepts: `100`, `100K`, `100M`, `100G`, `100KiB`, `100MiB`, `100GiB`.
/// Decimal-only mantissa; case-insensitive suffix. Returns `Err` on
/// empty / overflow / unknown suffix.
fn parse_byte_size(s: &str) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let s = s.trim();
    if s.is_empty() {
        return Err("byte size: empty".into());
    }
    let lower = s.to_ascii_lowercase();
    let (num_part, mul): (&str, u64) = if let Some(stripped) = lower.strip_suffix("gib") {
        (stripped, 1024 * 1024 * 1024)
    } else if let Some(stripped) = lower.strip_suffix("mib") {
        (stripped, 1024 * 1024)
    } else if let Some(stripped) = lower.strip_suffix("kib") {
        (stripped, 1024)
    } else if let Some(stripped) = lower.strip_suffix('g') {
        (stripped, 1024 * 1024 * 1024)
    } else if let Some(stripped) = lower.strip_suffix('m') {
        (stripped, 1024 * 1024)
    } else if let Some(stripped) = lower.strip_suffix('k') {
        (stripped, 1024)
    } else if let Some(stripped) = lower.strip_suffix('b') {
        (stripped, 1)
    } else {
        (lower.as_str(), 1)
    };
    let n: u64 = num_part
        .trim()
        .parse()
        .map_err(|_| format!("byte size: cannot parse `{num_part}` as integer"))?;
    n.checked_mul(mul)
        .ok_or_else(|| format!("byte size: `{s}` overflows u64").into())
}

/// v1.2 V — split an `.lma` archive into fixed-size volume files.
///
/// Pure byte-stream split. Each volume `<archive>.NNN` contains a
/// contiguous slice of the archive bytes. Reassembly is `cat
/// <archive>.* > <archive>` (or `lml volume-assemble <archive>.001`).
fn cmd_volume_split(input: &Path, size_spec: &str, keep: bool, force: bool) -> R {
    let size = parse_byte_size(size_spec)?;
    if size == 0 {
        return Err("volume-split: --size must be > 0".into());
    }
    let input_meta = std::fs::metadata(input)
        .map_err(|e| format!("volume-split: cannot stat {}: {}", input.display(), e))?;
    let total = input_meta.len();
    if total == 0 {
        return Err(format!("volume-split: {} is empty", input.display()).into());
    }
    let n_volumes = (total + size - 1) / size; // ceil
    if n_volumes > 999 {
        return Err(format!(
            "volume-split: would produce {} volumes; max 999 (NNN suffix is 3 digits). \
             Pick a larger --size.",
            n_volumes
        )
        .into());
    }
    // Pre-flight: refuse to clobber unless --force.
    for i in 1..=n_volumes {
        let vol_path = input.with_extension(format!(
            "{}.{:03}",
            input.extension().and_then(|e| e.to_str()).unwrap_or("lma"),
            i
        ));
        if vol_path.exists() && !force {
            return Err(format!(
                "volume-split: {} already exists; pass --force to overwrite",
                vol_path.display()
            )
            .into());
        }
    }
    // Split.
    use std::io::{Read, Write};
    let mut src = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    for i in 1..=n_volumes {
        let vol_path = input.with_extension(format!(
            "{}.{:03}",
            input.extension().and_then(|e| e.to_str()).unwrap_or("lma"),
            i
        ));
        let mut out = std::io::BufWriter::new(std::fs::File::create(&vol_path)?);
        let mut remaining = size.min(total - (i - 1) * size);
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = src.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            remaining -= n as u64;
        }
        let bytes_written = size.min(total - (i - 1) * size);
        println!(
            "  wrote {} ({})",
            vol_path.display(),
            human_bytes(bytes_written)
        );
    }
    drop(src);
    if !keep {
        std::fs::remove_file(input)?;
        println!("  removed original {}", input.display());
    }
    println!("  {} volume(s), {} total", n_volumes, human_bytes(total));
    Ok(())
}

/// v1.2 V — reassemble volume files into a single archive. Auto-
/// discovers the volume set by globbing `<base>.NNN` for sequential
/// 3-digit suffixes. Refuses to proceed on gaps in the sequence.
fn cmd_volume_assemble(input: &Path, output: Option<&Path>, force: bool) -> R {
    // Extract base name + starting NNN from the input path.
    let input_filename = input
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("volume-assemble: input has no filename")?;
    let dot_idx = input_filename
        .rfind('.')
        .ok_or("volume-assemble: input must end in `.NNN`")?;
    let suffix = &input_filename[dot_idx + 1..];
    if !(suffix.len() == 3 && suffix.chars().all(|c| c.is_ascii_digit())) {
        return Err(format!(
            "volume-assemble: input `{}` doesn't end in `.NNN` (got `.{}`)",
            input_filename, suffix
        )
        .into());
    }
    let base = &input_filename[..dot_idx];
    let parent_raw = input.parent().unwrap_or(Path::new("."));
    // `Path::parent` on a bare filename returns Some("") rather than
    // None; treat the empty path as the current directory so
    // `volume-assemble big.lma.001` works without a directory prefix.
    let parent: &Path = if parent_raw.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent_raw
    };

    // Glob the volume set.
    let mut volumes: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(rest) = name.strip_prefix(&format!("{base}.")) {
            if rest.len() == 3 && rest.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(idx) = rest.parse::<u32>() {
                    volumes.push((idx, entry.path()));
                }
            }
        }
    }
    if volumes.is_empty() {
        return Err(format!(
            "volume-assemble: no `{}.NNN` volumes found in {}",
            base,
            parent.display()
        )
        .into());
    }
    volumes.sort_by_key(|(idx, _)| *idx);

    // Verify sequence is complete: indices must be 1..=N with no gaps.
    let n = volumes.len() as u32;
    for (i, (idx, path)) in volumes.iter().enumerate() {
        let expected = (i as u32) + 1;
        if *idx != expected {
            return Err(format!(
                "volume-assemble: sequence gap -- expected `{}.{:03}`, found `{}` at position {}",
                base,
                expected,
                path.display(),
                i
            )
            .into());
        }
    }

    // Default output: strip the `.NNN` suffix.
    let default_out = parent.join(base);
    let out_path = output.map(|p| p.to_path_buf()).unwrap_or(default_out);
    lamquant_core::paths::ensure_can_write(&out_path, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    let mut total: u64 = 0;
    for (idx, vol_path) in &volumes {
        let mut f = std::io::BufReader::new(std::fs::File::open(vol_path)?);
        let n_copied = std::io::copy(&mut f, &mut out)?;
        total += n_copied;
        println!(
            "  read [{:03}/{:03}] {} ({})",
            idx,
            n,
            vol_path.display(),
            human_bytes(n_copied)
        );
    }
    out.flush()?;
    println!(
        "  assembled {} ({}, from {} volumes)",
        out_path.display(),
        human_bytes(total),
        n
    );
    Ok(())
}

/// v1.2 X — dispatches to the explainer when `--explain` is set;
/// otherwise calls the compact existing verifier.
fn cmd_verify_archive_explain(input: &Path, explain: bool) -> R {
    if explain {
        cmd_verify_archive_explainer(input)
    } else {
        cmd_verify_archive(input)
    }
}

/// v1.2 X — auditable per-step readout. Prints the literal chain
/// of checks the verifier performs:
///   1. Archive size + structural minimum
///   2. Archive-wide SHA-256 over content+footer
///   3. Manifest section (zstd decompress + parse)
///   4. Per-entry payload read + method-specific verify + SHA match
///   5. Decompression byte counts + per-entry CR
///   6. Cumulative elapsed time + OK/FAIL summary
///
/// No black box. Operator sees exactly what was checked.
fn cmd_verify_archive_explainer(input: &Path) -> R {
    use sha2::Digest;

    let file_size = std::fs::metadata(input)?.len();
    if file_size < 48 {
        return Err(format!("Archive too small ({} bytes)", file_size).into());
    }

    let t0 = Instant::now();
    println!("Verifying {} (auditable readout)", input.display());
    println!("─────────────────────────────────────────────────────────");
    println!(
        "[1/5] Archive size:        {} ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    // 2. Archive-wide SHA-256 (content + everything before the 32-byte footer)
    print!("[2/5] Archive SHA-256:     ");
    {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(input)?);
        let mut hasher = Sha256::new();
        let content_size = file_size - 32;
        let mut remaining = content_size;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = f.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            remaining -= n as u64;
        }
        let computed = hasher.finalize();
        let mut stored = [0u8; 32];
        f.read_exact(&mut stored)?;
        let computed_bytes = computed.as_slice();
        if computed_bytes != stored {
            println!("FAILED");
            print!("       stored:    ");
            for b in &stored {
                print!("{:02x}", b);
            }
            println!();
            print!("       computed:  ");
            for b in computed_bytes {
                print!("{:02x}", b);
            }
            println!();
            return Err("Archive SHA-256 mismatch — file is corrupted".into());
        }
        print!("OK  sha256:");
        for b in &stored[..8] {
            print!("{:02x}", b);
        }
        println!();
    }

    // 3. Manifest decompress + parse
    let entries = lma::list_archive(input)?;
    println!(
        "[3/5] Manifest:            OK ({} entries enumerated)",
        entries.len()
    );

    // 4. Per-entry payload read + verify
    println!("[4/5] Per-entry verify:");
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut header = [0u8; 16];
    f.read_exact(&mut header)?;
    let manifest_len = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as u64;
    let payload_start = 16 + manifest_len;

    let mut verified = 0usize;
    let mut failed = 0usize;
    let mut total_compressed: u64 = 0;
    let mut total_decompressed: u64 = 0;

    for (i, entry) in entries.iter().enumerate() {
        f.seek(SeekFrom::Start(payload_start + entry.offset))?;
        let mut payload = vec![0u8; entry.compressed_size as usize];
        f.read_exact(&mut payload)?;

        let method_str = match entry.method {
            lma::Method::Lml => "lml",
            lma::Method::Zstd => "zstd",
            lma::Method::Store => "store",
            _ => "unknown",
        };
        let sha_prefix = &entry.sha256[..entry.sha256.len().min(12)];

        let (ok, decompressed_size, detail) = match entry.method {
            lma::Method::Lml => {
                let tmp = tempfile::NamedTempFile::new()?;
                std::fs::write(tmp.path(), &payload)?;
                match container::read_file(tmp.path()) {
                    Ok((signal, _)) => {
                        let bytes = signal.iter().map(|ch| ch.len() * 8).sum::<usize>() as u64;
                        (true, bytes, format!("LML decode OK"))
                    }
                    Err(e) => (false, 0, format!("LML decode error: {}", e)),
                }
            }
            lma::Method::Zstd => match zstd::decode_all(payload.as_slice()) {
                Ok(decompressed) => {
                    let hash = format!("{:x}", Sha256::digest(&decompressed));
                    if hash == entry.sha256 {
                        (true, decompressed.len() as u64, format!("zstd OK"))
                    } else {
                        (
                            false,
                            decompressed.len() as u64,
                            format!(
                                "SHA-256 mismatch (got {}, expected {})",
                                &hash[..12],
                                sha_prefix
                            ),
                        )
                    }
                }
                Err(e) => (false, 0, format!("zstd error: {}", e)),
            },
            lma::Method::Store => {
                let hash = format!("{:x}", Sha256::digest(&payload));
                if hash == entry.sha256 {
                    (true, payload.len() as u64, format!("store OK"))
                } else {
                    (
                        false,
                        payload.len() as u64,
                        format!(
                            "SHA-256 mismatch (got {}, expected {})",
                            &hash[..12],
                            sha_prefix
                        ),
                    )
                }
            }
            _ => (
                false,
                0,
                "unknown compression method (writer newer than reader)".to_string(),
            ),
        };

        let cr = if entry.compressed_size > 0 {
            decompressed_size as f64 / entry.compressed_size as f64
        } else {
            0.0
        };
        let marker = if ok { "✓" } else { "✗" };
        println!(
            "       [{}/{}] {} {:<32}  {:>10}  {:<6}  sha256:{}  CR {:.2}x  ({})",
            i + 1,
            entries.len(),
            marker,
            entry.path,
            human_bytes(entry.compressed_size),
            method_str,
            sha_prefix,
            cr,
            detail,
        );
        if ok {
            verified += 1;
        } else {
            failed += 1;
        }
        total_compressed += entry.compressed_size;
        total_decompressed += decompressed_size;
    }

    let elapsed = t0.elapsed();
    let cr = if total_compressed > 0 {
        total_decompressed as f64 / total_compressed as f64
    } else {
        0.0
    };
    println!("[5/5] Summary:");
    println!(
        "       Compressed total:   {} ({} bytes)",
        human_bytes(total_compressed),
        total_compressed
    );
    println!(
        "       Decompressed total: {} ({} bytes)",
        human_bytes(total_decompressed),
        total_decompressed
    );
    println!("       Archive CR:         {:.2}x", cr);
    println!("       Verified:           {}/{}", verified, entries.len());
    println!("       Failed:             {}/{}", failed, entries.len());
    println!("       Elapsed:            {:.3}s", elapsed.as_secs_f64());
    println!("─────────────────────────────────────────────────────────");
    if failed > 0 {
        println!("Result: FAIL ({} failed entries)", failed);
        Err(format!("{} files failed verification", failed).into())
    } else {
        println!("Result: PASS (archive-wide hash OK + all entries verified)");
        Ok(())
    }
}

fn cmd_verify_archive(input: &Path) -> R {
    let file_size = std::fs::metadata(input)?.len();
    if file_size < 48 {
        return Err(format!("Archive too small ({} bytes)", file_size).into());
    }

    let t0 = Instant::now();
    println!("Verifying {}", input.display());

    // 1. Archive-level SHA-256
    print!("  Archive SHA-256... ");
    {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(input)?);
        let mut hasher = Sha256::new();
        let content_size = file_size - 32;
        let mut remaining = content_size;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        while remaining > 0 {
            let to_read = (remaining as usize).min(buf.len());
            let n = f.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            remaining -= n as u64;
        }
        let computed = hasher.finalize();
        let mut stored = [0u8; 32];
        f.read_exact(&mut stored)?;
        if computed.as_slice() != stored {
            println!("FAILED");
            return Err("Archive SHA-256 mismatch — file is corrupted".into());
        }
        println!("OK");
    }

    // 2. Read manifest
    let entries = lma::list_archive(input)?;
    println!("  Manifest: {} files", entries.len());

    // 3. Verify each entry's payload is readable and consistent.
    use rayon::prelude::*;
    use std::io::{Read, Seek, SeekFrom};

    // Parse the 16-byte header once to locate the payload region.
    let payload_start = {
        let mut header = [0u8; 16];
        std::io::BufReader::new(std::fs::File::open(input)?).read_exact(&mut header)?;
        let manifest_len =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as u64;
        16 + manifest_len
    };

    // Verify entries IN PARALLEL: each rayon worker opens its OWN file handle and
    // decode-checks its payload independently (verify is read-only — no ordering
    // constraint). The per-file LML verify uses a uniquely-named tempfile, so
    // concurrent verifies never collide. `None` = pass; `Some(msg)` = fail.
    // Messages are printed IN MANIFEST ORDER below, so output stays deterministic.
    let verdicts: Vec<Option<String>> = entries
        .par_iter()
        .map(|entry| -> Option<String> {
            let mut f = match std::fs::File::open(input) {
                Ok(f) => f,
                Err(e) => return Some(format!("  FAIL: {} — open error: {}", entry.path, e)),
            };
            if let Err(e) = f.seek(SeekFrom::Start(payload_start + entry.offset)) {
                return Some(format!("  FAIL: {} — seek error: {}", entry.path, e));
            }
            let mut payload = vec![0u8; entry.compressed_size as usize];
            if let Err(e) = f.read_exact(&mut payload) {
                return Some(format!("  FAIL: {} — read error: {}", entry.path, e));
            }

            match entry.method {
                lma::Method::Lml => {
                    // Decodable LML payload (CRC checked inside container::read_file).
                    let tmp = match tempfile::NamedTempFile::new() {
                        Ok(t) => t,
                        Err(e) => return Some(format!("  FAIL: {} — tempfile: {}", entry.path, e)),
                    };
                    if let Err(e) = std::fs::write(tmp.path(), &payload) {
                        return Some(format!("  FAIL: {} — tempfile write: {}", entry.path, e));
                    }
                    match container::read_file(tmp.path()) {
                        Ok(_) => None,
                        Err(e) => Some(format!("  FAIL: {} — LML decode error: {}", entry.path, e)),
                    }
                }
                lma::Method::Zstd => match zstd::decode_all(payload.as_slice()) {
                    Ok(decompressed) => {
                        let hash = format!("{:x}", Sha256::digest(&decompressed));
                        if hash == entry.sha256 {
                            None
                        } else {
                            Some(format!("  FAIL: {} — SHA-256 mismatch", entry.path))
                        }
                    }
                    Err(e) => Some(format!(
                        "  FAIL: {} — zstd decompress error: {}",
                        entry.path, e
                    )),
                },
                lma::Method::Store => {
                    let hash = format!("{:x}", Sha256::digest(&payload));
                    if hash == entry.sha256 {
                        None
                    } else {
                        Some(format!("  FAIL: {} — SHA-256 mismatch", entry.path))
                    }
                }
                // Unknown method (future Method variant). Fail-closed — clinical
                // contract: never treat unknown as Store.
                _ => Some(format!(
                    "  FAIL: {} — unknown compression method (writer is newer than this reader)",
                    entry.path
                )),
            }
        })
        .collect();

    // Aggregate + report in deterministic (manifest) order.
    let mut verified = 0usize;
    let mut failed = 0usize;
    for (i, verdict) in verdicts.iter().enumerate() {
        match verdict {
            None => verified += 1,
            Some(msg) => {
                failed += 1;
                println!("{}", msg);
            }
        }
        if (i + 1) % 500 == 0 {
            println!("    {}/{} verified...", i + 1, entries.len());
        }
    }

    let elapsed = t0.elapsed();
    println!(
        "\n  {} files verified, {} failed, {:.1}s",
        verified,
        failed,
        elapsed.as_secs_f64()
    );

    if failed > 0 {
        Err(format!("{} files failed verification", failed).into())
    } else {
        println!("  INTEGRITY OK — archive is valid.");
        Ok(())
    }
}

/// ADR 0051 track 2: encode a single EDF/BDF to a bare bounded-MAE `.lml`
/// (closed-loop DPCM, guarantees max|orig-recon| <= delta). Self-contained —
/// bypasses the batch/bundle path; the lossy stream has no per-recording
/// `.lma` sibling-envelope semantics. For the H.BWC working-point bench +
/// clinical near-lossless. Prints BPS (and, with --verify, the measured MAE).
fn cmd_encode_bounded_mae(
    input: &Path,
    output: Option<&Path>,
    delta: u64,
    window_size: usize,
    mode: lamquant_core::lpc::LpcMode,
    verify: bool,
) -> R {
    if input.is_dir() {
        return Err("--max-error expects a single EDF/BDF file, not a directory".into());
    }
    let edf = edf::read_edf(input)?;
    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("lml"));
    // Produce a standard .lml container (per-window bounded-MAE payloads) so
    // decode / export / info all work on the output.
    let stats = container::write_file_bounded_mae(
        &out_path,
        &edf.signal,
        edf.sample_rate,
        window_size,
        delta,
        "{}",
        mode,
    )?;

    let nm = (stats.n_channels * stats.total_samples).max(1);
    let bps = stats.compressed_size as f64 * 8.0 / nm as f64;

    if verify {
        // Re-read through the container and confirm the guarantee held.
        let (recon, _meta) = container::read_file(&out_path)?;
        let mut mae = 0i64;
        for (o, r) in edf.signal.iter().zip(recon.iter()) {
            for (a, b) in o.iter().zip(r.iter()) {
                let d = (a - b).abs();
                if d > mae {
                    mae = d;
                }
            }
        }
        if mae > delta as i64 {
            return Err(format!(
                "bounded-MAE VIOLATED: measured MAE {} > delta {}",
                mae, delta
            )
            .into());
        }
        println!(
            "bounded-MAE: {} ({}ch x {}) delta={} -> {} bytes, {:.4} BPS, measured MAE {} (<= {})",
            out_path.display(),
            stats.n_channels,
            stats.total_samples,
            delta,
            stats.compressed_size,
            bps,
            mae,
            delta
        );
    } else {
        println!(
            "bounded-MAE: {} ({}ch x {}) delta={} -> {} bytes, {:.4} BPS",
            out_path.display(),
            stats.n_channels,
            stats.total_samples,
            delta,
            stats.compressed_size,
            bps
        );
    }
    Ok(())
}

/// ADR 0051 track 2 P2: encode a single EDF/BDF to a target-BPS rate-controlled
/// lossy `.lml` container. Minimizes distortion subject to the bits-per-sample
/// ceiling. Prints achieved BPS (and, with --verify, the measured PRD).
fn cmd_encode_target_bps(
    input: &Path,
    output: Option<&Path>,
    target_bps: f64,
    window_size: usize,
    mode: lamquant_core::lpc::LpcMode,
    verify: bool,
) -> R {
    if input.is_dir() {
        return Err("--target-bps expects a single EDF/BDF file, not a directory".into());
    }
    let edf = edf::read_edf(input)?;
    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("lml"));
    let stats = container::write_file_target_bps(
        &out_path,
        &edf.signal,
        edf.sample_rate,
        window_size,
        target_bps,
        "{}",
        mode,
    )?;
    let nm = (stats.n_channels * stats.total_samples).max(1);
    let bps = stats.compressed_size as f64 * 8.0 / nm as f64;

    if verify {
        // Mean-removed PRD (CfP definition) vs the original.
        let (recon, _meta) = container::read_file(&out_path)?;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (o, r) in edf.signal.iter().zip(recon.iter()) {
            let mean = o.iter().sum::<i64>() as f64 / o.len().max(1) as f64;
            for (a, b) in o.iter().zip(r.iter()) {
                let e = (*a - *b) as f64;
                num += e * e;
                den += (*a as f64 - mean) * (*a as f64 - mean);
            }
        }
        let prd = if den == 0.0 {
            0.0
        } else {
            100.0 * (num / den).sqrt()
        };
        println!(
            "target-BPS: {} ({}ch x {}) target={:.3} -> {} bytes, {:.4} BPS, PRD {:.3}%",
            out_path.display(),
            stats.n_channels,
            stats.total_samples,
            target_bps,
            stats.compressed_size,
            bps,
            prd
        );
    } else {
        println!(
            "target-BPS: {} ({}ch x {}) target={:.3} -> {} bytes, {:.4} BPS",
            out_path.display(),
            stats.n_channels,
            stats.total_samples,
            target_bps,
            stats.compressed_size,
            bps
        );
    }
    Ok(())
}

fn cmd_bench(input: &Path) -> R {
    let edf_data = edf::read_edf(input)?;
    let n_ch = edf_data.n_channels;
    let t = edf_data.total_samples;

    // Tier 3 audit (O8): validate sample_rate before `as usize`
    // cast. NaN/zero/Inf would silently produce actual_window = 0,
    // empty window slices, and the subsequent .expect("benchmark
    // window valid") panic with no input-path context. edf::read_edf
    // now rejects non-finite sample rates (Tier 1 L1), so this is
    // defense-in-depth.
    if !(edf_data.sample_rate.is_finite() && edf_data.sample_rate > 0.0) {
        return Err(format!(
            "cmd_bench: {} has non-finite or non-positive sample_rate ({:?}); refusing",
            input.display(),
            edf_data.sample_rate
        )
        .into());
    }
    if n_ch == 0 || t == 0 {
        return Err(format!(
            "cmd_bench: {} has empty signal ({} channels x {} samples)",
            input.display(),
            n_ch,
            t
        )
        .into());
    }

    println!(
        "Benchmarking: {}ch × {} samples ({:.1}s @ {:.0} Hz)",
        n_ch, t, edf_data.duration_s, edf_data.sample_rate
    );

    let window_size = 2500;
    let actual_window = ((window_size as f64 * edf_data.sample_rate / 250.0) as usize).max(1);
    let n_windows = t.div_ceil(actual_window).max(1);

    let end = actual_window.min(t);
    let window: Vec<Vec<i64>> = edf_data
        .signal
        .iter()
        .map(|ch| ch[..end].to_vec())
        .collect();

    // Tier 3 audit (O8): replace .expect() with propagation so the
    // user sees the input path on failure instead of a stack trace.
    let _ = lml::compress(&window, 0);
    let compressed = lml::compress(&window, 0).map_err(|e| {
        format!(
            "cmd_bench: lml::compress failed on warmup window of {}: {}",
            input.display(),
            e
        )
    })?;
    let _ = lml::decompress(&compressed);

    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = lml::compress(&window, 0);
    }
    let compress_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = lml::decompress(&compressed);
    }
    let decompress_us = t0.elapsed().as_micros() as f64 / iters as f64;

    // Tier 3 audit (O8): checked multiplication. n_ch * end * 2
    // overflows usize on a 1024-ch × 2^60-sample adversarial input,
    // producing garbage CR / throughput. checked_mul forces a clean
    // error message instead of silent wraparound.
    let raw_bytes = n_ch
        .checked_mul(end)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            format!(
                "cmd_bench: raw_bytes math overflowed ({} ch × {} samples × 2 bytes)",
                n_ch, end
            )
        })?;
    let cr = raw_bytes as f64 / compressed.len() as f64;

    // Tier 3 audit (O8): guard the throughput division so we never
    // print "inf MB/s" when elapsed rounds to 0 micros.
    let bytes_per_us = |bytes: usize, us: f64| -> String {
        if us > 0.0 {
            format!("{:.0} MB/s", bytes as f64 / us)
        } else {
            "(< 1 us)".to_string()
        }
    };
    println!("\nPer-window ({}ch × {}):", n_ch, end);
    println!(
        "  Compress:   {:.0} us ({})",
        compress_us,
        bytes_per_us(raw_bytes, compress_us)
    );
    println!(
        "  Decompress: {:.0} us ({})",
        decompress_us,
        bytes_per_us(raw_bytes, decompress_us)
    );
    println!(
        "  CR:         {:.2}:1 ({} → {})",
        cr,
        raw_bytes,
        compressed.len()
    );
    println!("\nFull file ({} windows):", n_windows);
    println!(
        "  Compress:   {:.1} ms",
        compress_us * n_windows as f64 / 1000.0
    );
    println!(
        "  Decompress: {:.1} ms",
        decompress_us * n_windows as f64 / 1000.0
    );
    Ok(())
}

// ===========================================================================
// PCCP — version pins, integrity verify, change history
//
// Reads `pccp/registry.yaml` via a tiny hand-rolled parser (no serde_yaml
// dep). Only handles the narrow shape the registry uses: top-level scalars,
// `models.<name>.<key>: <value>` two-deep nesting. Sufficient for the version
// card + integrity pin lookup. If we ever need full YAML parsing, swap in
// serde_yaml under a feature flag.
// ===========================================================================

fn cmd_pccp(action: PccpAction) -> R {
    match action {
        PccpAction::Version => pccp_print_version(),
        PccpAction::Verify { model, path } => pccp_verify(&model, &path),
        PccpAction::History { count } => pccp_print_history(count),
    }
}

/// Emit a shell completion script to stdout for the requested shell.
/// Caller pipes the output to their shell's completion directory.
fn cmd_completions(shell: clap_complete::Shell) {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
}

/// Emit a roff(7) man-page for `lml` to stdout. Caller pipes to
/// `~/.local/share/man/man1/lml.1` (user) or `/usr/local/share/man/man1/lml.1`
/// (system) and runs `mandb` to register.
fn cmd_manpage() -> R {
    use clap::CommandFactory;
    let cmd = Cli::command();
    let man = clap_mangen::Man::new(cmd);
    let mut buf: Vec<u8> = Vec::new();
    man.render(&mut buf)
        .map_err(|e| format!("manpage render failed: {e}"))?;
    use std::io::Write as _;
    std::io::stdout()
        .write_all(&buf)
        .map_err(|e| format!("stdout write failed: {e}"))?;
    Ok(())
}

/// Read `sample_rate` from an LML/BCS1 container header.
///
/// ADR 0069/0071 L9 fix (was the SILENT bug this function used to carry):
/// the magic is now checked FIRST, before any field offset is read.
///   - `b"BCS1"` (the 40-byte typed header `write_abir` emits today) reads
///     `sample_rate_mhz` from its OWN offset, 22..26 (see
///     `abir::bcs1` layout docs) — NOT the legacy 16..20 offset,
///     which in the BCS1 layout holds `total_samples` instead.
///   - `b"LML*"` (the legacy 32-byte header) keeps the original probe +
///     offset: `hdr[4..6]` must equal `1` (the byte-exact 32-byte-header
///     marker; the older 18-/20-byte legacy containers have no
///     sample-rate field and are refused explicitly).
///   - anything else is a clean `Err` — never a silent wrong read.
///
/// Pre-fix, a `BCS1` file coincidentally passed the `probe == 1` check
/// (BCS1's `version_major`/`version_minor` bytes at header offset 4..6 are
/// `01 00`, which reads as `u16::from_le_bytes = 1`, matching the legacy
/// probe purely by byte coincidence) and fell through into the legacy
/// 16..20 read — silently returning BCS1's `total_samples` field
/// reinterpreted as a millihertz sample rate, with no error at all.
fn read_sample_rate_from_header(
    lml_path: &Path,
) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    use abir::{Bcs1Header, BCS1_HEADER_LEN, BCS1_MAGIC};
    use std::io::Read as _;
    let mut f = std::fs::File::open(lml_path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;

    if magic == *BCS1_MAGIC {
        let mut rest = [0u8; BCS1_HEADER_LEN - 4];
        f.read_exact(&mut rest)?;
        let mut hdr = [0u8; BCS1_HEADER_LEN];
        hdr[0..4].copy_from_slice(&magic);
        hdr[4..].copy_from_slice(&rest);
        let header = Bcs1Header::parse(&hdr)
            .map_err(|e| format!("lml split: invalid BCS1 header: {e}"))?;
        return Ok(header.sample_rate_mhz as f64 / 1000.0);
    }

    if &magic[0..3] != b"LML" {
        return Err(format!(
            "lml split: unrecognized container magic {:?} — expected `BCS1` or `LML1`",
            magic
        )
        .into());
    }

    // Legacy 32-byte header: bytes 16-19 = u32 LE sample_rate_mhz.
    let mut rest = [0u8; 28];
    f.read_exact(&mut rest)?;
    let mut hdr = [0u8; 32];
    hdr[0..4].copy_from_slice(&magic);
    hdr[4..].copy_from_slice(&rest);
    // 32-byte header: probe[4..6] = version u16 = 1.
    let probe = u16::from_le_bytes([hdr[4], hdr[5]]);
    if probe != 1 {
        return Err(format!(
            "lml split: header probe {probe} != 1 — legacy 18-byte container has \
             no sample-rate field, re-encode before splitting"
        )
        .into());
    }
    let sr_mhz = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
    Ok(sr_mhz as f64 / 1000.0)
}

/// Patch the source metadata JSON object with `split_chunk_idx` and
/// `split_n_chunks` fields. The source metadata is opaque to lml — we
/// don't parse the structure; we just insert two top-level keys before
/// the closing brace. If the metadata isn't a JSON object literal we
/// wrap it under `{"orig_metadata": <quoted>, ...}`.
fn metadata_with_split_marker(orig: &str, chunk_idx: u32, n_chunks: u32) -> String {
    let trimmed = orig.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        if inner.trim().is_empty() {
            return format!("{{\"split_chunk_idx\":{chunk_idx},\"split_n_chunks\":{n_chunks}}}");
        }
        return format!(
            "{{{inner},\"split_chunk_idx\":{chunk_idx},\"split_n_chunks\":{n_chunks}}}"
        );
    }
    // Non-object metadata — quote into a string field. JSON-escape the
    // double-quotes to keep the wrapper valid.
    let escaped = orig.replace('\\', r"\\").replace('"', r#"\""#);
    format!(
        "{{\"orig_metadata\":\"{escaped}\",\"split_chunk_idx\":{chunk_idx},\"split_n_chunks\":{n_chunks}}}"
    )
}

fn cmd_split(input: &Path, chunks: u32, output_dir: &Path, force: bool) -> R {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;

    if chunks < 2 {
        return Err(format!(
            "lml split: --chunks must be >= 2 (got {chunks}); to split into 1 chunk just copy the file"
        )
        .into());
    }

    let sample_rate = read_sample_rate_from_header(input)?;
    let (signal, metadata) = container::read_file(input)?;
    let n_ch = signal.len();
    let total_samples = if n_ch > 0 { signal[0].len() } else { 0 };
    if total_samples == 0 {
        return Err("lml split: empty signal in input".into());
    }
    if (chunks as usize) > total_samples {
        return Err(format!(
            "lml split: --chunks {chunks} > total_samples {total_samples} (every chunk must hold >= 1 sample)"
        )
        .into());
    }

    // Re-derive window_size from the source by reading via AnyLmlReader
    // (ADR 0069/0071 L9 — dispatches BCS1 vs legacy LML1 on magic), which
    // parses it from the header. Cheaper than re-decoding.
    let window_size = lamquant_core::bcs1_stream::AnyLmlReader::open(input)?
        .header()
        .window_size;
    if window_size == 0 {
        return Err("lml split: source window_size = 0".into());
    }

    std::fs::create_dir_all(output_dir)?;

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("split");

    let chunk_size = total_samples / chunks as usize;
    // Last chunk absorbs the remainder so concat is round-trip lossless.
    let mut start = 0usize;
    let t0 = Instant::now();
    for i in 0..chunks {
        let end = if i + 1 == chunks {
            total_samples
        } else {
            start + chunk_size
        };
        let slice: Vec<Vec<i64>> = signal.iter().map(|ch| ch[start..end].to_vec()).collect();
        let meta = metadata_with_split_marker(&metadata, i, chunks);
        let out_name = format!("{stem}.part-{:02}-of-{:02}.lml", i + 1, chunks);
        let out_path = output_dir.join(out_name);
        lamquant_core::paths::ensure_can_write(&out_path, force)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        let mut sink = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
        container::write_into(
            &mut sink,
            &slice,
            sample_rate,
            window_size,
            0,
            &meta,
            LpcMode::default(),
        )?;
        let f = sink.into_inner().map_err(|e| {
            let kind = e.error().kind();
            std::io::Error::new(kind, "lml split: BufWriter flush failed before sync_all")
        })?;
        f.sync_all()?;
        eprintln!(
            "  chunk {}/{}: {}ch × {} samples [{}, {}) → {}",
            i + 1,
            chunks,
            n_ch,
            end - start,
            start,
            end,
            out_path.display()
        );
        start = end;
    }
    println!(
        "Split {} into {} chunks ({:.1}ms)",
        input.display(),
        chunks,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Pull the `split_chunk_idx` integer (top-level JSON number) from
/// the metadata blob, if present. Returns `None` for plain LMLs that
/// were not produced by `lml split`.
fn metadata_split_idx(meta: &str) -> Option<u32> {
    let key = "\"split_chunk_idx\":";
    let pos = meta.find(key)?;
    let tail = &meta[pos + key.len()..];
    let mut end = 0usize;
    for (i, c) in tail.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else if end == 0 {
            // skip leading whitespace
            if !c.is_whitespace() {
                return None;
            }
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    tail[..end].trim().parse().ok()
}

fn metadata_split_n(meta: &str) -> Option<u32> {
    let key = "\"split_n_chunks\":";
    let pos = meta.find(key)?;
    let tail = &meta[pos + key.len()..];
    let mut end = 0usize;
    for (i, c) in tail.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else if end == 0 {
            if !c.is_whitespace() {
                return None;
            }
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    tail[..end].trim().parse().ok()
}

/// Strip `"split_chunk_idx": N,` and `"split_n_chunks": N,` (and
/// `, "...":...` trailing forms) from the metadata JSON. Best-effort
/// string surgery — keeps the surrounding JSON structure intact in
/// the common case (the keys were appended by `lml split`).
fn metadata_strip_split_markers(meta: &str) -> String {
    let mut out = meta.to_string();
    for key in ["split_chunk_idx", "split_n_chunks"] {
        let needles = [
            format!(",\"{key}\":"),
            format!(", \"{key}\":"),
            format!("\"{key}\":"),
        ];
        for n in needles.iter() {
            if let Some(start) = out.find(n.as_str()) {
                // Find end of the integer value
                let value_start = start + n.len();
                let bytes = out.as_bytes();
                let mut value_end = value_start;
                while value_end < bytes.len() {
                    let b = bytes[value_end];
                    if b.is_ascii_digit() || b == b' ' {
                        value_end += 1;
                    } else {
                        break;
                    }
                }
                // Also gobble a trailing comma if we entered via the
                // bare "..." needle (not the ",..." needle).
                let mut cut_end = value_end;
                if !n.starts_with(',') && bytes.get(cut_end) == Some(&b',') {
                    cut_end += 1;
                }
                out.replace_range(start..cut_end, "");
                break;
            }
        }
    }
    out
}

fn cmd_concat(inputs: &[PathBuf], output: &Path, force: bool) -> R {
    lamquant_core::paths::ensure_can_write(output, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;

    if inputs.len() < 2 {
        return Err("lml concat: need at least 2 inputs".into());
    }

    // Read all inputs up front so we can sort + validate before
    // committing to a write. This trades RAM for one-shot atomicity.
    let mut loaded: Vec<(PathBuf, Vec<Vec<i64>>, String, f64, usize)> = Vec::new();
    for path in inputs {
        let sr = read_sample_rate_from_header(path)?;
        let (sig, meta) = container::read_file(path)?;
        let ws = lamquant_core::bcs1_stream::AnyLmlReader::open(path)?
            .header()
            .window_size;
        loaded.push((path.clone(), sig, meta, sr, ws));
    }

    // Validate compatibility: same n_channels + sample_rate + window_size.
    let (_, ref first_sig, _, sr_ref, ws_ref) = loaded[0];
    let n_ch_ref = first_sig.len();
    for (path, sig, _, sr, ws) in &loaded {
        if sig.len() != n_ch_ref {
            return Err(format!(
                "lml concat: {} has {} channels, expected {n_ch_ref}",
                path.display(),
                sig.len()
            )
            .into());
        }
        if (sr - sr_ref).abs() > 1e-9 {
            return Err(format!(
                "lml concat: {} has sample_rate {sr}, expected {sr_ref}",
                path.display()
            )
            .into());
        }
        if *ws != ws_ref {
            return Err(format!(
                "lml concat: {} has window_size {ws}, expected {ws_ref}",
                path.display()
            )
            .into());
        }
    }

    // Sort: by split_chunk_idx if every input carries the markers AND
    // they agree on n_chunks. Else lexicographic by filename.
    let all_have_markers = loaded
        .iter()
        .all(|(_, _, m, _, _)| metadata_split_idx(m).is_some());
    if all_have_markers {
        // Check n_chunks agreement.
        let ns: std::collections::HashSet<u32> = loaded
            .iter()
            .filter_map(|(_, _, m, _, _)| metadata_split_n(m))
            .collect();
        if ns.len() != 1 {
            return Err(format!(
                "lml concat: split_n_chunks disagrees across inputs ({ns:?}) — refuse to guess order"
            )
            .into());
        }
        let expected_n = *ns.iter().next().unwrap() as usize;
        if expected_n != loaded.len() {
            return Err(format!(
                "lml concat: inputs claim {expected_n} chunks but got {} files",
                loaded.len()
            )
            .into());
        }
        loaded.sort_by_key(|(_, _, m, _, _)| metadata_split_idx(m).unwrap());
        // Check chunk_idx covers 0..N-1 with no gaps.
        for (expected, (_, _, m, _, _)) in loaded.iter().enumerate() {
            let idx = metadata_split_idx(m).unwrap();
            if idx as usize != expected {
                return Err(format!(
                    "lml concat: chunk_idx sequence has gap or dup at position {expected} (got idx {idx})"
                )
                .into());
            }
        }
        eprintln!("lml concat: sorting by split_chunk_idx (every input carries split markers)");
    } else {
        loaded.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!("lml concat: sorting by filename (no split markers detected)");
    }

    // Stitch channels.
    let mut out_signal: Vec<Vec<i64>> = vec![Vec::new(); n_ch_ref];
    for (path, sig, _, _, _) in &loaded {
        eprintln!(
            "  + {} ({}ch × {} samples)",
            path.display(),
            sig.len(),
            sig.first().map(|c| c.len()).unwrap_or(0)
        );
        for (ch_idx, ch) in sig.iter().enumerate() {
            out_signal[ch_idx].extend_from_slice(ch);
        }
    }

    // Output metadata: take chunk 0's metadata, strip split markers.
    let base_meta = &loaded[0].2;
    let out_meta = metadata_strip_split_markers(base_meta);

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let t0 = Instant::now();
    let mut sink = std::io::BufWriter::new(std::fs::File::create(output)?);
    container::write_into(
        &mut sink,
        &out_signal,
        sr_ref,
        ws_ref,
        0,
        &out_meta,
        LpcMode::default(),
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "lml concat: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;
    println!(
        "Concat {} inputs → {} ({}ch × {} samples, {:.1}ms)",
        loaded.len(),
        output.display(),
        n_ch_ref,
        out_signal[0].len(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn cmd_extract_entry(archive: &Path, entry: &str, output: &Path, force: bool) -> R {
    lamquant_core::paths::ensure_can_write(output, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let t0 = Instant::now();
    let n_bytes = lma::extract_entry(archive, entry, output)?;
    println!(
        "Extracted '{}' from {} → {} ({} bytes, {:.1}ms)",
        entry,
        archive.display(),
        output.display(),
        n_bytes,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Mask EDF/EDF+ header PII fields in `hdr` (modifies in place).
/// EDF header layout (first 256 bytes; see specs/lml-format-v1.md):
///   [0..8]    version "0       "
///   [8..88]   patient_id (80 bytes ASCII, ' '-padded)
///   [88..168] recording_id (80 bytes ASCII, ' '-padded)
///   [168..176] start_date "dd.mm.yy"
///   [176..184] start_time "hh.mm.ss"
///   [184..192] header_bytes (ASCII int)
///   ... fixed-width signal headers follow
///
/// Phase 3.9 whitelist-default: ALL bytes in patient_id and
/// recording_id get replaced with spaces. If `keep_dates` is false,
/// start_date is set to "01.01.85" (EDF spec's recommended anonymized
/// value, mirroring `edf-anonymize` and similar tools) and start_time
/// to "00.00.00".
fn mask_edf_pii_fields(hdr: &mut [u8], keep_dates: bool) -> Result<(), String> {
    if hdr.len() < 184 {
        return Err(format!(
            "strip-pii: EDF header too short ({} bytes, need >= 184)",
            hdr.len()
        ));
    }
    // patient_id (bytes 8..88)
    for b in &mut hdr[8..88] {
        *b = b' ';
    }
    // recording_id (bytes 88..168)
    for b in &mut hdr[88..168] {
        *b = b' ';
    }
    if !keep_dates {
        // EDF spec: anonymized "01.01.85" (a known anonymization sentinel).
        hdr[168..176].copy_from_slice(b"01.01.85");
        hdr[176..184].copy_from_slice(b"00.00.00");
    }
    Ok(())
}

/// Re-emit `metadata_json` with the `edf_header` field replaced by the
/// b64+zstd-encoded `new_hdr` bytes. Falls through for non-EDF metadata
/// (no edf_header key) and returns the input unchanged.
fn metadata_replace_edf_header(
    metadata_json: &str,
    new_hdr: &[u8],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use base64::Engine;
    let mut v: serde_json::Value = serde_json::from_str(metadata_json)?;
    let obj = match v.as_object_mut() {
        Some(o) => o,
        None => return Ok(metadata_json.to_string()),
    };
    if !obj.contains_key("edf_header") {
        return Ok(metadata_json.to_string());
    }
    let zstd_bytes = zstd::encode_all(new_hdr, 9)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&zstd_bytes);
    obj.insert("edf_header".into(), serde_json::Value::String(b64));
    Ok(serde_json::to_string(&v)?)
}

fn cmd_strip_pii(
    input: &Path,
    output: Option<&Path>,
    in_place: bool,
    keep_dates: bool,
    force: bool,
) -> R {
    if let Some(p) = output {
        lamquant_core::paths::ensure_can_write(p, force)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    use lamquant_core::bcs1_stream::AnyLmlReader;

    if in_place && output.is_some() {
        return Err("strip-pii: --in-place and --output are mutually exclusive".into());
    }
    if !in_place && output.is_none() {
        return Err("strip-pii: must pass either --output PATH or --in-place".into());
    }

    let t0 = Instant::now();
    let (signal, metadata_json) = container::read_file(input)?;
    let header_bytes = match meta_b64_zstd_field(&metadata_json, "edf_header") {
        Some(bytes) if !bytes.is_empty() => bytes,
        _ => {
            return Err(format!(
                "strip-pii: {} has no embedded EDF header in metadata — \
                 nothing to strip (was this LML produced from an EDF/BDF source?)",
                input.display()
            )
            .into())
        }
    };
    let mut new_hdr = header_bytes.clone();
    mask_edf_pii_fields(&mut new_hdr, keep_dates)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let new_metadata = metadata_replace_edf_header(&metadata_json, &new_hdr)?;

    // Source sample_rate and window_size so the re-encode preserves
    // the wire-format fields (Phase 3.5's helper does exactly this).
    let sample_rate = read_sample_rate_from_header(input)?;
    let window_size = AnyLmlReader::open(input)?.header().window_size;

    // Choose write target: --output path (new file) OR --in-place (same-dir tempfile + atomic rename).
    let target_path: PathBuf = if let Some(p) = output {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        p.to_path_buf()
    } else {
        // in_place: write to <input>.tmp first
        input.with_extension(format!(
            "{}.tmp",
            input.extension().and_then(|s| s.to_str()).unwrap_or("lml")
        ))
    };

    let mut sink = std::io::BufWriter::new(std::fs::File::create(&target_path)?);
    container::write_into(
        &mut sink,
        &signal,
        sample_rate,
        window_size,
        0,
        &new_metadata,
        LpcMode::default(),
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "strip-pii: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    if in_place {
        // Atomic swap: replace input with tempfile.
        std::fs::rename(&target_path, input)?;
        // Fsync parent dir on Unix so rename is durable.
        #[cfg(unix)]
        {
            if let Some(parent) = input.parent() {
                if let Ok(d) = std::fs::File::open(parent) {
                    let _ = d.sync_all();
                }
            }
        }
        println!(
            "Stripped PII from {} (in-place, dates {}, {:.1}ms)",
            input.display(),
            if keep_dates { "kept" } else { "masked" },
            t0.elapsed().as_secs_f64() * 1000.0
        );
    } else {
        println!(
            "Stripped PII: {} → {} (dates {}, {:.1}ms)",
            input.display(),
            target_path.display(),
            if keep_dates { "kept" } else { "masked" },
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

/// Parse `key=value` into (key, JSON value). If the value is a valid
/// JSON literal (number, true/false/null, "...", [...], {...}), it
/// becomes that JSON value. Otherwise, stored as a JSON string verbatim.
fn parse_kv_set(s: &str) -> Result<(String, serde_json::Value), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("--set: expected KEY=VALUE, got '{s}'"))?;
    let key = k.trim();
    if key.is_empty() {
        return Err(format!("--set: empty key in '{s}'"));
    }
    let v_trim = v.trim_start();
    let parsed = match serde_json::from_str::<serde_json::Value>(v_trim) {
        Ok(j) => j,
        Err(_) => serde_json::Value::String(v.to_string()),
    };
    Ok((key.to_string(), parsed))
}

#[allow(clippy::too_many_arguments)]
fn cmd_set_metadata(
    input: &Path,
    output: Option<&Path>,
    in_place: bool,
    sidecar: Option<&Path>,
    sets: &[String],
    removes: &[String],
    force: bool,
) -> R {
    if let Some(p) = output {
        lamquant_core::paths::ensure_can_write(p, force)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;
    use lamquant_core::bcs1_stream::AnyLmlReader;

    if in_place && output.is_some() {
        return Err("set-metadata: --in-place and --output are mutually exclusive".into());
    }
    if !in_place && output.is_none() {
        return Err("set-metadata: must pass either --output PATH or --in-place".into());
    }
    if sidecar.is_none() && sets.is_empty() && removes.is_empty() {
        return Err("set-metadata: no edits requested (pass --sidecar / --set / --remove)".into());
    }

    let t0 = Instant::now();
    let (signal, metadata_json) = container::read_file(input)?;
    let mut v: serde_json::Value = serde_json::from_str(&metadata_json).map_err(|e| {
        format!("set-metadata: existing metadata is not valid JSON ({e}); refuse to clobber")
    })?;
    let obj = v.as_object_mut().ok_or_else(|| {
        "set-metadata: existing metadata is not a JSON object; refuse to overwrite"
    })?;

    // Sidecar overlay first (top-level merge).
    if let Some(side) = sidecar {
        let raw = std::fs::read_to_string(side)
            .map_err(|e| format!("set-metadata: read sidecar {}: {e}", side.display()))?;
        let sv: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            format!(
                "set-metadata: sidecar {} not valid JSON: {e}",
                side.display()
            )
        })?;
        let so = sv.as_object().ok_or_else(|| {
            format!(
                "set-metadata: sidecar {} must be a top-level JSON object",
                side.display()
            )
        })?;
        for (k, val) in so {
            obj.insert(k.clone(), val.clone());
        }
    }
    // --set k=v values.
    for kv in sets {
        let (k, val) = parse_kv_set(kv)?;
        obj.insert(k, val);
    }
    // --remove keys.
    for k in removes {
        obj.remove(k);
    }
    let final_key_count = obj.len();
    let new_metadata = serde_json::to_string(&v)?;

    // Sample rate + window size preserved from the source header.
    let sample_rate = read_sample_rate_from_header(input)?;
    let window_size = AnyLmlReader::open(input)?.header().window_size;

    let target_path: PathBuf = if let Some(p) = output {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        p.to_path_buf()
    } else {
        input.with_extension(format!(
            "{}.tmp",
            input.extension().and_then(|s| s.to_str()).unwrap_or("lml")
        ))
    };

    let mut sink = std::io::BufWriter::new(std::fs::File::create(&target_path)?);
    container::write_into(
        &mut sink,
        &signal,
        sample_rate,
        window_size,
        0,
        &new_metadata,
        LpcMode::default(),
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "set-metadata: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    if in_place {
        std::fs::rename(&target_path, input)?;
        #[cfg(unix)]
        {
            if let Some(parent) = input.parent() {
                if let Ok(d) = std::fs::File::open(parent) {
                    let _ = d.sync_all();
                }
            }
        }
        println!(
            "set-metadata: {} (in-place, {} keys after edit, {:.1}ms)",
            input.display(),
            final_key_count,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    } else {
        println!(
            "set-metadata: {} → {} ({} keys after edit, {:.1}ms)",
            input.display(),
            target_path.display(),
            final_key_count,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_recompress(
    input: &Path,
    output: Option<&Path>,
    in_place: bool,
    noise_bits: u8,
    window_size: usize,
    lpc_mode: lamquant_core::lpc::LpcMode,
    force: bool,
) -> R {
    if let Some(p) = output {
        lamquant_core::paths::ensure_can_write(p, force)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }
    use lamquant_core::container;
    use lamquant_core::bcs1_stream::AnyLmlReader;

    if in_place && output.is_some() {
        return Err("recompress: --in-place and --output are mutually exclusive".into());
    }
    if !in_place && output.is_none() {
        return Err("recompress: must pass either --output PATH or --in-place".into());
    }
    if window_size == 0 {
        return Err("recompress: --window-size must be > 0".into());
    }

    let t0 = Instant::now();
    let (signal, metadata_json) = container::read_file(input)?;
    let sample_rate = read_sample_rate_from_header(input)?;
    let src_window_size = AnyLmlReader::open(input)?.header().window_size;
    let src_size = std::fs::metadata(input)?.len();

    let target_path: PathBuf = if let Some(p) = output {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        p.to_path_buf()
    } else {
        input.with_extension(format!(
            "{}.tmp",
            input.extension().and_then(|s| s.to_str()).unwrap_or("lml")
        ))
    };

    let mut sink = std::io::BufWriter::new(std::fs::File::create(&target_path)?);
    container::write_into(
        &mut sink,
        &signal,
        sample_rate,
        window_size,
        noise_bits,
        &metadata_json,
        lpc_mode,
    )?;
    let f = sink.into_inner().map_err(|e| {
        let kind = e.error().kind();
        std::io::Error::new(kind, "recompress: BufWriter flush failed before sync_all")
    })?;
    f.sync_all()?;

    if in_place {
        std::fs::rename(&target_path, input)?;
        #[cfg(unix)]
        {
            if let Some(parent) = input.parent() {
                if let Ok(d) = std::fs::File::open(parent) {
                    let _ = d.sync_all();
                }
            }
        }
        let new_size = std::fs::metadata(input)?.len();
        println!(
            "Recompressed in-place: {} ({} → {} bytes, noise_bits={}, window {}→{}, lpc={:?}, {:.1}ms)",
            input.display(),
            src_size,
            new_size,
            noise_bits,
            src_window_size,
            window_size,
            lpc_mode,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    } else {
        let new_size = std::fs::metadata(&target_path)?.len();
        println!(
            "Recompressed: {} → {} ({} → {} bytes, noise_bits={}, window {}→{}, lpc={:?}, {:.1}ms)",
            input.display(),
            target_path.display(),
            src_size,
            new_size,
            noise_bits,
            src_window_size,
            window_size,
            lpc_mode,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

fn cmd_append(
    archive: &Path,
    file: &Path,
    as_path: Option<&str>,
    zstd_level: i32,
    force: bool,
    keep_bak: bool,
) -> R {
    let t0 = Instant::now();
    let summary = lma::append_entry(archive, file, as_path, zstd_level, force, keep_bak)?;
    println!(
        "Appended {} → {} (now {} entries, {:.1} MiB total, CR={:.2}x, {:.1}ms)",
        file.display(),
        archive.display(),
        summary.n_files,
        summary.archive_bytes as f64 / (1024.0 * 1024.0),
        summary.cr,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Built-in self-test: synth signal → compress → decompress → verify
/// byte-exact + check container CRC. Returns Err on any mismatch so
/// the process exits with code 1.
///
/// Useful for confirming a fresh binary install (or a CI runner)
/// produces a working codec without needing real EEG fixtures.
fn cmd_self_test() -> R {
    use lamquant_core::container;
    use lamquant_core::lpc::LpcMode;

    let n_ch = 4usize;
    let n_samples = 1024usize;

    // Deterministic synth signal — same PRNG style as the container
    // tests. Different seed per channel keeps inter-channel statistics
    // distinct so a per-channel skip bug would surface.
    let mut state: u64 = 0xDEAD_BEEF_C0FF_EE00;
    let mut sig: Vec<Vec<i64>> = (0..n_ch).map(|_| Vec::with_capacity(n_samples)).collect();
    for ch in &mut sig {
        for _ in 0..n_samples {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ch.push(((state >> 33) as i32) as i64 % 8000);
        }
    }

    println!("self-test: synth {} channels × {} samples", n_ch, n_samples);

    // Encode into Vec sink (no filesystem dependency).
    let mut sink: Vec<u8> = Vec::new();
    let t0 = std::time::Instant::now();
    let stats = container::write_into(
        &mut sink,
        &sig,
        250.0,
        256,
        0,
        "{\"source\":\"self-test\"}",
        LpcMode::default(),
    )
    .map_err(|e| format!("encode failed: {e}"))?;
    let enc_ms = t0.elapsed().as_millis();

    println!(
        "self-test: encoded {} bytes (CR={:.2}×) in {}ms",
        stats.compressed_size, stats.cr, enc_ms
    );

    // Decode from a Cursor.
    let t1 = std::time::Instant::now();
    let mut cursor = std::io::Cursor::new(&sink);
    let (recovered, meta) =
        container::read_from(&mut cursor).map_err(|e| format!("decode failed: {e}"))?;
    let dec_ms = t1.elapsed().as_millis();

    if meta != "{\"source\":\"self-test\"}" {
        return Err(format!("metadata round-trip mismatch: {meta:?}").into());
    }
    if recovered.len() != n_ch {
        return Err(format!(
            "channel count mismatch: encoded {n_ch}, recovered {}",
            recovered.len()
        )
        .into());
    }
    for ch in 0..n_ch {
        if recovered[ch] != sig[ch] {
            return Err(format!(
                "channel {ch} signal mismatch ({} vs {} samples)",
                recovered[ch].len(),
                sig[ch].len()
            )
            .into());
        }
    }

    println!(
        "self-test: decoded + verified {} bytes in {}ms",
        sink.len(),
        dec_ms
    );
    println!("self-test: OK");
    Ok(())
}

fn pccp_registry_path() -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(p) = std::env::var("LAMQUANT_REGISTRY_PATH") {
        return Ok(PathBuf::from(p));
    }
    // Walk up from the binary location and from cwd looking for pccp/registry.yaml.
    let candidates = std::iter::empty()
        .chain(std::env::current_exe().ok())
        .chain(std::env::current_dir().ok());
    for start in candidates {
        let mut cur: &Path = &start;
        loop {
            let candidate = cur.join("pccp").join("registry.yaml");
            if candidate.exists() {
                return Ok(candidate);
            }
            match cur.parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
    Err("pccp/registry.yaml not found (set LAMQUANT_REGISTRY_PATH)".into())
}

/// Strongly-typed view of `pccp/registry.yaml`. We deserialize via serde_yaml
/// (NOT a hand-rolled parser) so the audit trail interpretation matches the
/// gate hook's PyYAML view byte-for-byte. V4 Pro Finding 1 on the
/// bones-A+B+C commit.
#[derive(serde::Deserialize, Default, Debug)]
struct PccpRegistry {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    last_updated: String,
    #[serde(default)]
    last_updated_by: String,
    /// Per-model field bag. Values are kept as YAML scalars (string-typed
    /// here; serde_yaml converts numbers to their repr) which is fine for
    /// the version-card rendering that's the only consumer.
    #[serde(default)]
    models:
        std::collections::BTreeMap<String, std::collections::BTreeMap<String, serde_yaml::Value>>,
}

impl PccpRegistry {
    fn model_field(&self, model: &str, field: &str) -> Option<String> {
        self.models.get(model)?.get(field).map(yaml_value_to_string)
    }
}

/// Canonical fixed-decimal float repr — never scientific notation.
///
/// Uses `{:.12}` to force fixed-point with 12 fractional digits (more
/// than enough for PCCP metrics: R, FPR, KB sizes, latency_ms), then
/// strips trailing zeros and a trailing dot for a clean canonical form.
/// 12 digits is the IEEE-754 round-trip threshold for f64 values in the
/// magnitude ranges we handle (|f| ∈ [1e-9, 1e9]).
fn format_float_canonical(f: f64) -> String {
    if !f.is_finite() {
        return f.to_string(); // NaN / inf — leave Display form.
    }
    let s = format!("{:.12}", f);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn yaml_value_to_string(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::Null => String::new(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        // Canonical numeric form: integers as plain decimal, floats via
        // `format_float_canonical` which forces fixed-point notation and
        // strips trailing zeros. Rust's `f64::to_string()` uses Display
        // which CAN emit scientific notation for very small/large
        // magnitudes — not safe for PCCP audit comparison vs the Python
        // gate's PyYAML rendering (V4 Pro Findings 1 of the bones-A+B+C-
        // fixes commits — the prior `to_string()` and `format!("{:.}")`
        // approaches were both flagged).
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else if let Some(f) = n.as_f64() {
                format_float_canonical(f)
            } else {
                n.to_string()
            }
        }
        serde_yaml::Value::String(s) => s.clone(),
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

fn pccp_load_registry() -> Result<PccpRegistry, Box<dyn std::error::Error + Send + Sync>> {
    let path = pccp_registry_path()?;
    let text = std::fs::read_to_string(&path)?;
    let reg: PccpRegistry =
        serde_yaml::from_str(&text).map_err(|e| format!("registry.yaml parse error: {}", e))?;
    Ok(reg)
}

fn pccp_print_version() -> R {
    let reg = pccp_load_registry()?;
    let device_version = env!("CARGO_PKG_VERSION");

    println!("LamQuant Neural EEG Codec");
    println!("  Device version:   {}", device_version);
    println!("  Registry schema:  {}", reg.schema_version);
    println!(
        "  Last updated:     {}  (by {})",
        reg.last_updated, reg.last_updated_by
    );
    println!();
    println!("  Models:");
    for name in reg.models.keys() {
        let version = reg
            .model_field(name, "production_version")
            .unwrap_or_else(|| "?".to_string());
        let ckpt = reg
            .model_field(name, "production_checkpoint")
            .unwrap_or_else(|| "?".to_string());
        let sha = reg
            .model_field(name, "production_sha256")
            .unwrap_or_else(|| "?".to_string());
        let sha_short = if sha.starts_with("PLACEHOLDER") {
            "(unpinned)".to_string()
        } else {
            let take = 16.min(sha.len());
            sha[..take].to_string()
        };
        println!(
            "    {:<10} v{:<8}  sha256:{}…  {}",
            name, version, sha_short, ckpt
        );
    }
    println!();
    println!("  PCCP authority:  pccp/PCCP.md");
    println!("  Modification log: pccp/CHANGELOG.md");
    Ok(())
}

fn pccp_verify(model: &str, path: &Path) -> R {
    let reg = pccp_load_registry()?;
    let expected = reg
        .model_field(model, "production_sha256")
        .ok_or_else(|| format!("no production_sha256 for model '{}'", model))?;
    if expected.starts_with("PLACEHOLDER") {
        return Err(format!(
            "model '{}' has placeholder SHA — capture via `python ai_models/pccp_gate.py --capture --model {} --candidate {}`",
            model, model, path.display()
        ).into());
    }

    if !path.exists() {
        return Err(format!("checkpoint not found: {}", path.display()).into());
    }
    let mut hasher = Sha256::new();
    let f = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(f);
    std::io::copy(&mut reader, &mut hasher)?;
    let actual = format!("{:x}", hasher.finalize());

    if actual == expected {
        println!(
            "OK  model={}  sha256={}  path={}",
            model,
            actual,
            path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "INTEGRITY FAILURE for model '{}'\n  expected: {}\n  actual:   {}\n  path:     {}",
            model,
            expected,
            actual,
            path.display()
        )
        .into())
    }
}

fn pccp_print_history(count: usize) -> R {
    let path = pccp_registry_path()?
        .parent()
        .ok_or("registry.yaml has no parent dir")?
        .join("CHANGELOG.md");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("CHANGELOG.md: {}", e))?;

    // Only `## YYYY-MM-DD …` lines start a real entry. Doc-style sections
    // (e.g. "## Format") are skipped.
    let lines: Vec<&str> = text.split('\n').collect();
    let is_entry_header = |s: &str| -> bool {
        if !s.starts_with("## ") {
            return false;
        }
        let rest = &s[3..];
        if rest.len() < 10 {
            return false;
        }
        let bytes = rest.as_bytes();
        bytes[0..4].iter().all(|c| c.is_ascii_digit())
            && bytes[4] == b'-'
            && bytes[5..7].iter().all(|c| c.is_ascii_digit())
            && bytes[7] == b'-'
            && bytes[8..10].iter().all(|c| c.is_ascii_digit())
    };

    let mut entries: Vec<String> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        if is_entry_header(line) {
            if let Some(s) = start {
                entries.push(lines[s..i].join("\n"));
            }
            start = Some(i);
        }
    }
    if let Some(s) = start {
        entries.push(lines[s..].join("\n"));
    }

    let shown = count.min(entries.len());
    println!(
        "Showing {} most recent of {} CHANGELOG entries",
        shown,
        entries.len()
    );
    println!();
    for entry in entries.iter().take(shown) {
        println!("{}", entry);
    }
    Ok(())
}

// ─── Phase 7 security subcommand wrappers ────────────────────────────

/// v1.2 P — read a password for encryption.
///
/// Priority: `LAMQUANT_PASSWORD` env (when non-empty, useful for
/// scripted contexts) → interactive prompt via rpassword (reads from
/// /dev/tty so stdin can carry plaintext through a pipe).
///
/// Empty passwords are refused at the KDF call site (see
/// `Key::from_password`).
fn read_password_for_encrypt() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(env_pw) = std::env::var("LAMQUANT_PASSWORD") {
        if !env_pw.is_empty() {
            return Ok(env_pw);
        }
    }
    eprintln!("Encrypting with password-derived key (Argon2id, OWASP defaults).");
    let pw1 = rpassword::prompt_password("Password: ")
        .map_err(|e| format!("encrypt: failed to read password from terminal: {e}"))?;
    let pw2 = rpassword::prompt_password("Confirm: ")
        .map_err(|e| format!("encrypt: failed to read confirmation: {e}"))?;
    if pw1 != pw2 {
        return Err("encrypt: password mismatch (typed twice; refusing to proceed)".into());
    }
    if pw1.is_empty() {
        return Err("encrypt: empty password not allowed".into());
    }
    Ok(pw1)
}

/// v1.2 P — read a password for decryption (no confirmation).
fn read_password_for_decrypt() -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(env_pw) = std::env::var("LAMQUANT_PASSWORD") {
        if !env_pw.is_empty() {
            return Ok(env_pw);
        }
    }
    let pw = rpassword::prompt_password("Password: ")
        .map_err(|e| format!("decrypt: failed to read password from terminal: {e}"))?;
    Ok(pw)
}

fn cmd_encrypt(input: &Path, output: &Path, force: bool, password: bool) -> R {
    lamquant_core::paths::ensure_can_write(output, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    use lamquant_core::security::{encrypt_aes_gcm, Argon2Params, Key, LmcryptHeader};
    let (key, kdf_header) = if password {
        // v1.2 P — derive key via Argon2id.
        let pw = read_password_for_encrypt()?;
        let header = LmcryptHeader::new_random(Argon2Params::default())?;
        let key = Key::from_password(&pw, &header)?;
        (key, Some(header))
    } else {
        (Key::from_env()?, None)
    };
    let plaintext = std::fs::read(input)?;
    let blob = encrypt_aes_gcm(&key, &plaintext)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, &blob)?;
    // Sidecar with salt + Argon2 params for password-derived keys.
    if let Some(header) = kdf_header {
        let sidecar_path = output.with_extension("lmcrypt.header");
        std::fs::write(&sidecar_path, header.to_bytes())?;
        println!(
            "Encrypted {} → {} ({} → {} bytes, AES-256-GCM); KDF sidecar {} (Argon2id m={} t={} p={})",
            input.display(),
            output.display(),
            plaintext.len(),
            blob.len(),
            sidecar_path.display(),
            header.params.m_kib,
            header.params.t_cost,
            header.params.p_cost,
        );
    } else {
        println!(
            "Encrypted {} → {} ({} → {} bytes, AES-256-GCM)",
            input.display(),
            output.display(),
            plaintext.len(),
            blob.len()
        );
    }
    Ok(())
}

fn cmd_decrypt(input: &Path, output: &Path, force: bool, password: bool) -> R {
    lamquant_core::paths::ensure_can_write(output, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    use lamquant_core::security::{decrypt_aes_gcm, Key, LmcryptHeader};
    let key = if password {
        // v1.2 P — re-derive key via Argon2id using the salt + params
        // stored in the sidecar header alongside the ciphertext.
        let sidecar_path = input.with_extension("lmcrypt.header");
        let header_bytes = std::fs::read(&sidecar_path).map_err(|e| {
            format!(
                "decrypt: --password requires sidecar {}; cannot read: {}",
                sidecar_path.display(),
                e
            )
        })?;
        let header = LmcryptHeader::from_bytes(&header_bytes)?;
        let pw = read_password_for_decrypt()?;
        Key::from_password(&pw, &header)?
    } else {
        Key::from_env()?
    };
    let blob = std::fs::read(input)?;
    let plaintext = decrypt_aes_gcm(&key, &blob)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, &plaintext)?;
    println!(
        "Decrypted {} → {} ({} → {} bytes, AES-256-GCM auth OK)",
        input.display(),
        output.display(),
        blob.len(),
        plaintext.len()
    );
    Ok(())
}

fn cmd_sign(input: &Path, output: Option<&Path>, force: bool) -> R {
    if let Some(p) = output {
        lamquant_core::paths::ensure_can_write(p, force)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    }
    use lamquant_core::security::{hmac_sign, Key};
    let key = Key::from_env()?;
    let bytes = std::fs::read(input)?;
    let tag = hmac_sign(&key, &bytes);
    let tag_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        input.with_extension(format!(
            "{}.hmac",
            input.extension().and_then(|s| s.to_str()).unwrap_or("bin")
        ))
    });
    if let Some(parent) = tag_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&tag_path, tag)?;
    println!(
        "Signed {} → {} (HMAC-SHA-256, 32-byte tag)",
        input.display(),
        tag_path.display()
    );
    Ok(())
}

fn cmd_verify_signature(input: &Path, tag_path_arg: Option<&Path>) -> R {
    use lamquant_core::security::{hmac_verify, Key, HMAC_TAG_LEN};
    let key = Key::from_env()?;
    let bytes = std::fs::read(input)?;
    let tag_path = tag_path_arg.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        input.with_extension(format!(
            "{}.hmac",
            input.extension().and_then(|s| s.to_str()).unwrap_or("bin")
        ))
    });
    let tag_bytes = std::fs::read(&tag_path)?;
    if tag_bytes.len() != HMAC_TAG_LEN {
        return Err(format!(
            "verify-signature: tag file {} has {} bytes, expected {HMAC_TAG_LEN}",
            tag_path.display(),
            tag_bytes.len()
        )
        .into());
    }
    let mut tag = [0u8; HMAC_TAG_LEN];
    tag.copy_from_slice(&tag_bytes);
    if hmac_verify(&key, &bytes, &tag) {
        println!(
            "verify-signature: OK — {} matches {} (HMAC-SHA-256)",
            input.display(),
            tag_path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "verify-signature: FAIL — {} does NOT match {} (wrong key, tampered file, or stale tag)",
            input.display(),
            tag_path.display()
        )
        .into())
    }
}

fn cmd_audit_log(action: AuditAction) -> R {
    use lamquant_core::security::AuditLog;
    match action {
        AuditAction::Append { log, op, msg } => {
            let al = AuditLog::new(&log);
            al.append(&op, &msg)?;
            println!("Appended audit entry to {} (op={op})", log.display());
            Ok(())
        }
        AuditAction::Verify { log } => {
            let al = AuditLog::new(&log);
            let n = al.verify()?;
            println!(
                "audit-log {}: chain intact, {n} entries verified",
                log.display()
            );
            Ok(())
        }
    }
}

// ─── Phase 6 async subcommand wrappers (feature-gated) ───────────────

#[cfg(feature = "async")]
fn build_async_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
}

#[cfg(feature = "async")]
fn cmd_metrics(bind: &str) -> R {
    let rt = build_async_runtime()?;
    let bind_owned = bind.to_string();
    rt.block_on(async move {
        lamquant_core::async_io::serve_metrics(&bind_owned, async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
    })?;
    Ok(())
}

#[cfg(feature = "async")]
fn cmd_watch(
    input: PathBuf,
    output: PathBuf,
    queue_cap: usize,
    noise_bits: u8,
    window_size: usize,
    sample_rate: f64,
) -> R {
    let rt = build_async_runtime()?;
    let processed = rt.block_on(async move {
        lamquant_core::async_io::watch_dir(
            input,
            output,
            sample_rate,
            window_size,
            noise_bits,
            queue_cap,
            async {
                let _ = tokio::signal::ctrl_c().await;
            },
        )
        .await
    })?;
    println!("watch: stopped, {processed} files encoded");
    Ok(())
}

#[cfg(feature = "async")]
fn cmd_fetch(url: &str, output: &Path, max_bytes: u64, force: bool) -> R {
    lamquant_core::paths::ensure_can_write(output, force)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let rt = build_async_runtime()?;
    let url_owned = url.to_string();
    let bytes = rt
        .block_on(async move { lamquant_core::async_io::fetch_url(&url_owned, max_bytes).await })?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(output, &bytes)?;
    println!(
        "Fetched {} → {} ({} bytes)",
        url,
        output.display(),
        bytes.len()
    );
    Ok(())
}

#[cfg(feature = "async")]
#[allow(clippy::too_many_arguments)]
fn cmd_notify(
    url: &str,
    op: &str,
    op_id: Option<&str>,
    source_path: &str,
    output_path: &str,
    content_sha256: &str,
    bytes: u64,
    max_retries: u32,
) -> R {
    use lamquant_core::async_io::{post_webhook, WebhookPayload};
    let op_id_owned = op_id.map(|s| s.to_string()).unwrap_or_else(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("lamquant-{ts}")
    });
    let payload = WebhookPayload {
        op_id: op_id_owned.clone(),
        op: op.to_string(),
        source_path: source_path.to_string(),
        output_path: output_path.to_string(),
        content_sha256: content_sha256.to_string(),
        bytes,
    };
    let rt = build_async_runtime()?;
    let url_owned = url.to_string();
    rt.block_on(async move { post_webhook(&url_owned, &payload, max_retries).await })?;
    println!("notify: POSTed to {url} (op_id={op_id_owned}, op={op}, bytes={bytes})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_lossless_mode_is_mcu_when_flag_missing() {
        assert_eq!(parse_lossless_mode(None).unwrap(), LosslessMode::Mcu);
    }

    #[test]
    fn mcu_mode_resolves_to_fixed_lpc_when_auto() {
        assert!(matches!(
            resolve_lpc_mode(LosslessMode::Mcu, "auto").unwrap(),
            lamquant_core::lpc::LpcMode::Fixed
        ));
    }

    #[test]
    fn mcu_mode_rejects_adaptive_lpc() {
        let err = resolve_lpc_mode(LosslessMode::Mcu, "adaptive").unwrap_err();
        assert!(err.contains("MCU lossless mode only permits"));
    }

    #[test]
    fn explicit_lossless_mode_rejects_target_bps() {
        let err =
            reject_explicit_lossless_with_non_lossless(Some("mcu"), None, Some(3.0)).unwrap_err();
        assert!(err.contains("--target-bps"));
    }

    #[test]
    fn explicit_lossless_mode_rejects_max_error() {
        let err =
            reject_explicit_lossless_with_non_lossless(Some("mcu"), Some(1), None).unwrap_err();
        assert!(err.contains("--max-error"));
    }

    #[cfg(not(feature = "experimental_basestation"))]
    #[test]
    fn basestation_requires_experimental_feature() {
        let err = parse_lossless_mode(Some("basestation")).unwrap_err();
        assert!(err.contains("experimental_basestation"));
    }
}
