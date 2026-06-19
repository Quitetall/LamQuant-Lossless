//! LamQuant configuration — TOML reader/writer at lamquant.toml.
//!
//! Mirrors the Python `lamquant_codec/cli/config.py` schema (9 sections).
//! Best-effort parsing — unknown keys are preserved on round-trip so we don't
//! drop user customisations the Rust side doesn't understand yet.
//! Zero external deps; follows the pattern in `tui/history.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// Serialize/Deserialize derives below are for the `tui::snapshot::
// StateSnapshot` GUI bridge ONLY. The canonical config format on
// disk is TOML, parsed/written by the hand-rolled `parse_toml` /
// `serialize_toml` pair lower in this file. DO NOT use serde to
// read or write `lamquant.toml` — the wire formats differ (serde
// would emit JSON-style key ordering, no preserved unknown keys,
// and miss the embedded comment header).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LamQuantConfig {
    pub schema_version: String,
    pub instance_name: String,
    /// Preferred UI for `lamquant` no-arg invocations: "tui", "gui", or
    /// "ask" (no preference yet — first-run prompt sets a value).
    /// Read by lamquant.rs::smart_detect_mode; written by the wizard
    /// after the first-run prompt and by the future settings panel.
    /// Env var LAMQUANT_UI=tui|gui|ask overrides this at launch time.
    pub ui_preference: String,
    pub output: OutputCfg,
    pub codec: CodecCfg,
    pub compute: ComputeCfg,
    pub integrity: IntegrityCfg,
    pub resume: ResumeCfg,
    pub logging: LoggingCfg,
    pub input: InputCfg,
    pub output_files: OutputFilesCfg,
    pub backend: BackendCfg,
    /// Unknown sections/keys preserved verbatim across round-trip.
    pub extra: BTreeMap<String, BTreeMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputCfg {
    pub refresh_hz: f64,
    pub color: String,
    pub charset: String,
    pub dashboard_width: i64,
    pub show_spinner: bool,
    pub spinner_style: String,
    pub verbose_per_file: bool,
    pub truncate_filenames: bool,
    pub show_banner: bool,
    pub show_summary: bool,
    pub json_summary: bool,
    pub splash_duration: f64,
    pub autocomplete: bool,
    /// Low-resource render mode. Drops the banner block and slows the
    /// refresh tick. Useful on slow terminals (ssh over high-latency
    /// link, low-power MCU console, etc.). Renamed from `potato_mode`
    /// in commit B of the TUI settings cleanup batch.
    pub minimal_ui: bool,
    pub allow_root: bool,
    pub warn_root: bool,
    pub bell_on_done: bool,
    /// Art shown right of the LAMQUANT banner. "random" | "off" | "<name>"
    pub art_banner: String,
    /// Art shown between the hub dividers. "random" | "off" | "<name>"
    pub art_hub: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LosslessCfg {
    pub entropy_coder: String,
    pub lpc_order: i64,
    pub use_lifting: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecCfg {
    pub default_mode: String,
    pub input_bits: i64,
    pub window_samples: i64,
    pub noise_bits: i64,
    pub verification: String,
    /// Lossless compression default. One of `"prompt"`, `"lma"`,
    /// `"lml_siblings"`. Drives the `[1] Compress` flow in
    /// `tui::panels::mode_panel`:
    ///   - `"prompt"` (default) -- show the [a] LMA / [s] LML + siblings
    ///     picker overlay before the file explorer.
    ///   - `"lma"` -- skip the picker, encode straight to a single
    ///     `.lma` archive (today's behaviour pre-batch).
    ///   - `"lml_siblings"` -- skip the picker, encode each EEG file
    ///     to `.lml` and copy every non-EEG sibling alongside (no
    ///     archive container, preserves directory tree).
    /// Validated string in `apply_codec`; downstream consumers must
    /// match the literal forms above.
    pub lossless_default_mode: String,
    pub lossless: LosslessCfg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeCfg {
    pub workers: i64,
    pub memory_limit_gib: f64,
    pub numba_cache_dir: String,
    /// Lossless-codec compute backend. `"desktop"` selects the host
    /// rayon-parallel + AVX2 perf path (default on host); `"firmware"`
    /// selects the scalar serial path the MCU build also uses (for
    /// debugging or firmware-bench parity). Output bytes are
    /// identical across the two -- only wall-clock differs.
    pub backend: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegrityCfg {
    pub window_checksum: String,
    pub file_checksum: String,
    pub verify_after_write: bool,
    pub verify_outliers: bool,
    pub reject_corrupted_input: bool,
    pub refuse_double_strip: bool,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResumeCfg {
    pub enabled: bool,
    pub state_file: String,
    pub checkpoint_strategy: String,
    pub on_existing_state: String,
    pub skip_existing_output: bool,
    pub verify_skipped: bool,
    pub quarantine_dir: String,
    pub max_retries: i64,
    pub retry_backoff_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggingCfg {
    pub audit_log: String,
    pub append_audit: bool,
    pub stderr_level: String,
    pub file_log: String,
    pub include_tracebacks: bool,
    pub manifest: String,
    pub manifest_include_files: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputCfg {
    pub extensions: Vec<String>,
    pub recursive: bool,
    pub follow_symlinks: bool,
    pub min_file_size: i64,
    pub max_file_size: i64,
    pub exclude_patterns: Vec<String>,
}

/// Output-file destination knobs.
///
/// IMPORTANT: extension is NOT a configurable field. The codec mode
/// selected at encode time determines it: lossless → `.lml`, neural /
/// lossy → `.lmq`. Allowing the user to override would let a `.lml`
/// filename carry lossy bytes — downstream tools that assume bit-
/// exact reconstruction from `.lml` could silently mishandle seizure
/// data.
///
/// The mapping lives in `tui::panels::mode_panel::CodecMode::ext()`
/// for the TUI/GUI flows and is hardcoded to `.lml` in the CLI
/// encoder at `bin/lml.rs` (the CLI is lossless-only today).
///
/// TODO(lmq-cli): when the CLI gains the neural-encode mode switch,
/// update this comment + the CLI to read suffix from
/// `CodecMode::ext()` so TUI / GUI / CLI share one source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputFilesCfg {
    pub preserve_structure: bool,
    pub atomic_writes: bool,
    pub fsync_on_write: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendCfg {
    pub mode: String,
    pub rust_binary: String,
    pub custom_binary: String,
}

// ── Defaults match Python config.py ────────────────────────────────────────

impl Default for OutputCfg {
    fn default() -> Self {
        Self {
            refresh_hz: 10.0,
            color: "always".into(),
            charset: "auto".into(),
            dashboard_width: 0,
            show_spinner: true,
            spinner_style: "braille".into(),
            verbose_per_file: false,
            truncate_filenames: true,
            show_banner: true,
            show_summary: true,
            json_summary: false,
            splash_duration: 0.5,
            autocomplete: true,
            minimal_ui: false,
            allow_root: false,
            warn_root: true,
            bell_on_done: false,
            art_banner: "random".into(),
            art_hub: "chip".into(),
        }
    }
}

impl Default for LosslessCfg {
    fn default() -> Self {
        Self {
            entropy_coder: "golomb_rice".into(),
            lpc_order: 2,
            use_lifting: true,
        }
    }
}

impl Default for CodecCfg {
    fn default() -> Self {
        Self {
            default_mode: "lossless".into(),
            input_bits: 16,
            window_samples: 2500,
            noise_bits: 0,
            verification: "standard".into(),
            // Default = Prompt. New users hit the LMA / LML+siblings
            // picker on every compress until they pick a default in
            // Settings -- safer than silently choosing one for them.
            lossless_default_mode: "prompt".into(),
            lossless: LosslessCfg::default(),
        }
    }
}

impl Default for ComputeCfg {
    fn default() -> Self {
        Self {
            workers: 0,
            memory_limit_gib: 0.0,
            numba_cache_dir: "auto".into(),
            // Backend default matches what THIS build can actually
            // run: `"desktop"` on host (rayon + AVX2 path is
            // compiled in), `"firmware"` on no_std builds (the only
            // variant the firmware binary knows about). Without
            // this conditional, a fresh-config firmware launch
            // would land in `apply_to_runtime`'s "unrecognised"
            // arm and emit a spurious WARN. (V4 Pro / V6 R review
            // of 08ae620.)
            backend: if cfg!(feature = "host") {
                "desktop".into()
            } else {
                "firmware".into()
            },
        }
    }
}

impl Default for IntegrityCfg {
    fn default() -> Self {
        Self {
            window_checksum: "crc32".into(),
            file_checksum: "sha256".into(),
            verify_after_write: true,
            verify_outliers: true,
            reject_corrupted_input: true,
            refuse_double_strip: true,
            fail_fast: false,
        }
    }
}

impl Default for ResumeCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            state_file: ".lamquant_state.json".into(),
            checkpoint_strategy: "per_file".into(),
            on_existing_state: "auto".into(),
            skip_existing_output: true,
            verify_skipped: false,
            quarantine_dir: "quarantine".into(),
            max_retries: 2,
            retry_backoff_s: 5.0,
        }
    }
}

impl Default for LoggingCfg {
    fn default() -> Self {
        Self {
            audit_log: "audit.log".into(),
            append_audit: true,
            stderr_level: "WARNING".into(),
            file_log: String::new(),
            include_tracebacks: false,
            manifest: "manifest.lml.json".into(),
            manifest_include_files: true,
        }
    }
}

impl Default for InputCfg {
    fn default() -> Self {
        Self {
            extensions: vec!["edf".into(), "bdf".into()],
            recursive: true,
            follow_symlinks: false,
            min_file_size: 1024,
            max_file_size: 0,
            exclude_patterns: vec!["**/test/**".into(), "**/.git/**".into()],
        }
    }
}

impl Default for OutputFilesCfg {
    fn default() -> Self {
        Self {
            preserve_structure: true,
            atomic_writes: true,
            fsync_on_write: true,
        }
    }
}

impl Default for BackendCfg {
    fn default() -> Self {
        Self {
            mode: "auto".into(),
            rust_binary: "lml".into(),
            custom_binary: String::new(),
        }
    }
}

impl Default for LamQuantConfig {
    fn default() -> Self {
        Self {
            schema_version: "1.0".into(),
            instance_name: "default".into(),
            ui_preference: "ask".into(),
            output: OutputCfg::default(),
            codec: CodecCfg::default(),
            compute: ComputeCfg::default(),
            integrity: IntegrityCfg::default(),
            resume: ResumeCfg::default(),
            logging: LoggingCfg::default(),
            input: InputCfg::default(),
            output_files: OutputFilesCfg::default(),
            backend: BackendCfg::default(),
            extra: BTreeMap::new(),
        }
    }
}

// ── Disk I/O ───────────────────────────────────────────────────────────────

/// Resolution order matches Python `_find_config_file`:
/// CLI override > ./lamquant.toml > $XDG_CONFIG_HOME/lamquant/lamquant.toml > /etc.
pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var("LAMQUANT_CONFIG") {
        return PathBuf::from(p);
    }
    let cwd_local = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("lamquant.toml");
    if cwd_local.exists() {
        return cwd_local;
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        let p = PathBuf::from(xdg).join("lamquant").join("lamquant.toml");
        if p.exists() {
            return p;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".config/lamquant/lamquant.toml");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("lamquant.toml")
}

impl LamQuantConfig {
    /// Load from default path. Returns defaults + any errors logged to stderr.
    pub fn load() -> Self {
        let path = config_path();
        match fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text).unwrap_or_else(|e| {
                eprintln!("config parse error in {}: {}", path.display(), e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Save to default path. Atomic via temp file + rename.
    ///
    /// If the rename fails (e.g. cross-filesystem boundary, Windows
    /// `ERROR_ACCESS_DENIED` on antivirus locks, or the destination
    /// being suddenly read-only), we explicitly remove the orphaned
    /// `.toml.tmp` so repeated save attempts don't litter the user's
    /// CWD with `lamquant.toml.tmp` files. The rename error is then
    /// surfaced to the caller untouched.
    pub fn save(&self) -> std::io::Result<()> {
        let path = config_path();
        let tmp = path.with_extension("toml.tmp");
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        // Write + fsync the temp file before swapping in. fsync ensures the
        // serialized bytes are durable on disk before rename — survives a
        // crash between rename and the kernel flushing the page cache.
        {
            let mut f = fs::File::create(&tmp)?;
            std::io::Write::write_all(&mut f, self.serialize().as_bytes())?;
            // Best-effort fsync. On platforms where fsync isn't available
            // (e.g. some networked filesystems) this returns Err which we
            // log and ignore — the rename below still gives us atomicity.
            let _ = f.sync_all();
        }
        match fs::rename(&tmp, &path) {
            Ok(()) => Ok(()),
            Err(rename_err) => {
                // Best-effort cleanup of the orphan; log the secondary error
                // but always return the original rename failure so the user
                // sees the root cause, not the cleanup symptom.
                if let Err(unlink_err) = fs::remove_file(&tmp) {
                    eprintln!(
                        "config save: rename failed ({}) and tmp cleanup failed ({}). \
                         Manual cleanup may be required: {}",
                        rename_err,
                        unlink_err,
                        tmp.display()
                    );
                }
                Err(rename_err)
            }
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        parse_toml(text)
    }

    pub fn serialize(&self) -> String {
        serialize_toml(self)
    }

    /// Push config-time choices into process-wide runtime globals.
    ///
    /// Today this is only the compute backend selector (Desktop vs
    /// Firmware). Call once after loading lamquant.toml and after
    /// every successful Settings panel save -- without this, the
    /// TOML field exists but the codec hot path keeps using
    /// `ComputeBackend::default()`.
    ///
    /// Unknown / invalid `compute.backend` values are logged at
    /// WARN and left untouched (Default::default() applies). Avoids
    /// turning a typo in the config into a hard startup failure.
    pub fn apply_to_runtime(&self) {
        use crate::backend::{set_global_backend, ComputeBackend};
        match self.compute.backend.as_str() {
            "firmware" => set_global_backend(ComputeBackend::Firmware),
            #[cfg(feature = "tui")]
            "desktop" => set_global_backend(ComputeBackend::Desktop),
            other => {
                tracing::warn!(
                    "compute.backend = {:?} unrecognised; falling back to default",
                    other
                );
            }
        }
    }
}

// ── TOML parser (subset) ───────────────────────────────────────────────────
// Supports: comments (#...), [section], [section.subsection],
// scalars (bool, int, float, string), string arrays.
// Unknown keys preserved into self.extra.

fn parse_toml(text: &str) -> Result<LamQuantConfig, String> {
    let mut cfg = LamQuantConfig::default();
    let mut current_section = String::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            current_section = rest.trim().to_string();
            continue;
        }
        let Some((key, val)) = split_kv(line) else {
            return Err(format!("line {}: not a key = value: {}", lineno + 1, raw));
        };
        if let Err(e) = apply_kv(&mut cfg, &current_section, &key, &val) {
            // Unknown keys preserved verbatim, but malformed values fail.
            if e.starts_with("unknown:") {
                cfg.extra
                    .entry(current_section.clone())
                    .or_default()
                    .insert(key, val);
            } else {
                return Err(format!("line {}: {}", lineno + 1, e));
            }
        }
    }
    Ok(cfg)
}

fn split_kv(line: &str) -> Option<(String, String)> {
    let eq = line.find('=')?;
    let k = line[..eq].trim().to_string();
    // Strip trailing comment after value (only if not inside a string).
    let raw_val = &line[eq + 1..];
    let val = strip_trailing_comment(raw_val).trim().to_string();
    Some((k, val))
}

fn strip_trailing_comment(s: &str) -> &str {
    let mut in_str = false;
    let mut prev = '\0';
    for (i, c) in s.char_indices() {
        if c == '"' && prev != '\\' {
            in_str = !in_str;
        } else if c == '#' && !in_str {
            return &s[..i];
        }
        prev = c;
    }
    s
}

fn apply_kv(cfg: &mut LamQuantConfig, section: &str, key: &str, val: &str) -> Result<(), String> {
    match section {
        "" => match key {
            "schema_version" => cfg.schema_version = parse_string(val)?,
            "instance_name" => cfg.instance_name = parse_string(val)?,
            "ui_preference" => cfg.ui_preference = parse_string(val)?,
            _ => return Err(format!("unknown:root.{}", key)),
        },
        "output" => apply_output(&mut cfg.output, key, val)?,
        "codec" => apply_codec(&mut cfg.codec, key, val)?,
        "codec.lossless" => apply_lossless(&mut cfg.codec.lossless, key, val)?,
        "compute" => apply_compute(&mut cfg.compute, key, val)?,
        "integrity" => apply_integrity(&mut cfg.integrity, key, val)?,
        "resume" => apply_resume(&mut cfg.resume, key, val)?,
        "logging" => apply_logging(&mut cfg.logging, key, val)?,
        "input" => apply_input(&mut cfg.input, key, val)?,
        "output_files" => apply_output_files(&mut cfg.output_files, key, val)?,
        "backend" => apply_backend(&mut cfg.backend, key, val)?,
        _ => return Err(format!("unknown:{}.{}", section, key)),
    }
    Ok(())
}

fn apply_output(c: &mut OutputCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "refresh_hz" => c.refresh_hz = parse_float(v)?,
        "color" => c.color = parse_string(v)?,
        "charset" => c.charset = parse_string(v)?,
        "dashboard_width" => c.dashboard_width = parse_int(v)?,
        "show_spinner" => c.show_spinner = parse_bool(v)?,
        "spinner_style" => c.spinner_style = parse_string(v)?,
        "verbose_per_file" => c.verbose_per_file = parse_bool(v)?,
        "truncate_filenames" => c.truncate_filenames = parse_bool(v)?,
        "show_banner" => c.show_banner = parse_bool(v)?,
        "show_summary" => c.show_summary = parse_bool(v)?,
        "json_summary" => c.json_summary = parse_bool(v)?,
        "splash_duration" => c.splash_duration = parse_float(v)?,
        "autocomplete" => c.autocomplete = parse_bool(v)?,
        "minimal_ui" => c.minimal_ui = parse_bool(v)?,
        // Pre-release alias: `potato_mode` was renamed to `minimal_ui`
        // in commit B of the TUI settings cleanup batch. Accept the
        // old key for one minor release so existing dev configs keep
        // working; drop at next major.
        "potato_mode" => c.minimal_ui = parse_bool(v)?,
        "allow_root" => c.allow_root = parse_bool(v)?,
        "warn_root" => c.warn_root = parse_bool(v)?,
        // `instant_nav` field was removed (commit A of TUI settings
        // cleanup). Accept + silently drop any stale value carried
        // forward from an older lamquant.toml so a working config
        // doesn't trip "unknown key" loudly on first read after
        // upgrade. Drop this arm entirely at next major.
        "instant_nav" => { /* deprecated, ignored */ }
        "bell_on_done" => c.bell_on_done = parse_bool(v)?,
        "art_banner" => c.art_banner = parse_string(v)?,
        "art_hub" => c.art_hub = parse_string(v)?,
        _ => return Err(format!("unknown:output.{}", k)),
    }
    Ok(())
}

fn apply_codec(c: &mut CodecCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "default_mode" => c.default_mode = parse_string(v)?,
        "input_bits" => c.input_bits = parse_int(v)?,
        "window_samples" => c.window_samples = parse_int(v)?,
        "noise_bits" => c.noise_bits = parse_int(v)?,
        "verification" => c.verification = parse_string(v)?,
        "lossless_default_mode" => {
            let s = parse_string(v)?;
            // Reject anything outside the 3 valid forms so a typo'd
            // TOML doesn't silently flip the default behaviour --
            // prevents silent fallback to default if mode_panel
            // dispatch (commit E/5) sees an unknown value.
            match s.as_str() {
                "prompt" | "lma" | "lml_siblings" => {
                    c.lossless_default_mode = s;
                }
                other => {
                    return Err(format!(
                        "invalid:codec.lossless_default_mode={} \
                         (expected prompt|lma|lml_siblings)",
                        other,
                    ));
                }
            }
        }
        _ => return Err(format!("unknown:codec.{}", k)),
    }
    Ok(())
}

fn apply_lossless(c: &mut LosslessCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "entropy_coder" => c.entropy_coder = parse_string(v)?,
        "lpc_order" => c.lpc_order = parse_int(v)?,
        "use_lifting" => c.use_lifting = parse_bool(v)?,
        _ => return Err(format!("unknown:codec.lossless.{}", k)),
    }
    Ok(())
}

fn apply_compute(c: &mut ComputeCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "workers" => c.workers = parse_int(v)?,
        "memory_limit_gib" => c.memory_limit_gib = parse_float(v)?,
        "numba_cache_dir" => c.numba_cache_dir = parse_string(v)?,
        "backend" => {
            let s = parse_string(v)?;
            if s != "firmware" && s != "desktop" {
                return Err(format!(
                    "compute.backend must be \"firmware\" or \"desktop\" (got {:?})",
                    s
                ));
            }
            c.backend = s;
        }
        _ => return Err(format!("unknown:compute.{}", k)),
    }
    Ok(())
}

fn apply_integrity(c: &mut IntegrityCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "window_checksum" => c.window_checksum = parse_string(v)?,
        "file_checksum" => c.file_checksum = parse_string(v)?,
        "verify_after_write" => c.verify_after_write = parse_bool(v)?,
        "verify_outliers" => c.verify_outliers = parse_bool(v)?,
        "reject_corrupted_input" => c.reject_corrupted_input = parse_bool(v)?,
        "refuse_double_strip" => c.refuse_double_strip = parse_bool(v)?,
        "fail_fast" => c.fail_fast = parse_bool(v)?,
        _ => return Err(format!("unknown:integrity.{}", k)),
    }
    Ok(())
}

fn apply_resume(c: &mut ResumeCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "enabled" => c.enabled = parse_bool(v)?,
        "state_file" => c.state_file = parse_string(v)?,
        "checkpoint_strategy" => c.checkpoint_strategy = parse_string(v)?,
        "on_existing_state" => c.on_existing_state = parse_string(v)?,
        "skip_existing_output" => c.skip_existing_output = parse_bool(v)?,
        "verify_skipped" => c.verify_skipped = parse_bool(v)?,
        "quarantine_dir" => c.quarantine_dir = parse_string(v)?,
        "max_retries" => c.max_retries = parse_int(v)?,
        "retry_backoff_s" => c.retry_backoff_s = parse_float(v)?,
        _ => return Err(format!("unknown:resume.{}", k)),
    }
    Ok(())
}

fn apply_logging(c: &mut LoggingCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "audit_log" => c.audit_log = parse_string(v)?,
        "append_audit" => c.append_audit = parse_bool(v)?,
        "stderr_level" => c.stderr_level = parse_string(v)?,
        "file_log" => c.file_log = parse_string(v)?,
        "include_tracebacks" => c.include_tracebacks = parse_bool(v)?,
        "manifest" => c.manifest = parse_string(v)?,
        "manifest_include_files" => c.manifest_include_files = parse_bool(v)?,
        _ => return Err(format!("unknown:logging.{}", k)),
    }
    Ok(())
}

fn apply_input(c: &mut InputCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "extensions" => c.extensions = parse_string_array(v)?,
        "recursive" => c.recursive = parse_bool(v)?,
        "follow_symlinks" => c.follow_symlinks = parse_bool(v)?,
        "min_file_size" => c.min_file_size = parse_int(v)?,
        "max_file_size" => c.max_file_size = parse_int(v)?,
        "exclude_patterns" => c.exclude_patterns = parse_string_array(v)?,
        _ => return Err(format!("unknown:input.{}", k)),
    }
    Ok(())
}

fn apply_output_files(c: &mut OutputFilesCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        // `extension` was removed (commit C of TUI settings cleanup
        // batch). Codec mode now determines the suffix at encode
        // time (`CodecMode::ext()`). Accept + silently drop the
        // stale value so existing lamquant.toml files don't trip
        // "unknown key". Drop this arm at next major.
        "extension" => { /* deprecated, ignored — derived from CodecMode */ }
        "preserve_structure" => c.preserve_structure = parse_bool(v)?,
        "atomic_writes" => c.atomic_writes = parse_bool(v)?,
        "fsync_on_write" => c.fsync_on_write = parse_bool(v)?,
        _ => return Err(format!("unknown:output_files.{}", k)),
    }
    Ok(())
}

fn apply_backend(c: &mut BackendCfg, k: &str, v: &str) -> Result<(), String> {
    match k {
        "mode" => c.mode = parse_string(v)?,
        "rust_binary" => c.rust_binary = parse_string(v)?,
        "custom_binary" => c.custom_binary = parse_string(v)?,
        _ => return Err(format!("unknown:backend.{}", k)),
    }
    Ok(())
}

// ── value parsers ──────────────────────────────────────────────────────────

fn parse_string(v: &str) -> Result<String, String> {
    let v = v.trim();
    if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
        Ok(unescape(&v[1..v.len() - 1]))
    } else {
        Err(format!("expected quoted string, got: {}", v))
    }
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v.trim() {
        "true" => Ok(true),
        "false" => Ok(false),
        x => Err(format!("expected bool, got: {}", x)),
    }
}

fn parse_int(v: &str) -> Result<i64, String> {
    v.trim()
        .parse::<i64>()
        .map_err(|e| format!("expected int: {}", e))
}

fn parse_float(v: &str) -> Result<f64, String> {
    v.trim()
        .parse::<f64>()
        .map_err(|e| format!("expected float: {}", e))
}

fn parse_string_array(v: &str) -> Result<Vec<String>, String> {
    let v = v.trim();
    if !v.starts_with('[') || !v.ends_with(']') {
        return Err(format!("expected array, got: {}", v));
    }
    let inner = &v[1..v.len() - 1];
    let mut out = Vec::new();
    let chars = inner.chars().peekable();
    let mut buf = String::new();
    let mut in_str = false;
    let mut prev = '\0';
    for c in chars {
        if c == '"' && prev != '\\' {
            in_str = !in_str;
            if !in_str {
                out.push(unescape(&buf));
                buf.clear();
            }
        } else if in_str {
            buf.push(c);
        }
        prev = c;
    }
    Ok(out)
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(o) => {
                    out.push('\\');
                    out.push(o);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            x => out.push(x),
        }
    }
    out
}

// ── Serializer ─────────────────────────────────────────────────────────────

fn serialize_toml(cfg: &LamQuantConfig) -> String {
    let mut s = String::new();
    s.push_str("# LamQuant configuration — schema v1.0\n");
    s.push_str("# Generated by lml. Edit values; comments outside known keys are not preserved.\n");
    s.push('\n');
    s.push_str(&format!(
        "schema_version = \"{}\"\n",
        escape(&cfg.schema_version)
    ));
    s.push_str(&format!(
        "instance_name = \"{}\"\n",
        escape(&cfg.instance_name)
    ));
    s.push_str(&format!(
        "ui_preference = \"{}\"\n",
        escape(&cfg.ui_preference)
    ));

    let o = &cfg.output;
    s.push_str("\n[output]\n");
    s.push_str(&kv_float("refresh_hz", o.refresh_hz));
    s.push_str(&kv_str("color", &o.color));
    s.push_str(&kv_str("charset", &o.charset));
    s.push_str(&kv_int("dashboard_width", o.dashboard_width));
    s.push_str(&kv_bool("show_spinner", o.show_spinner));
    s.push_str(&kv_str("spinner_style", &o.spinner_style));
    s.push_str(&kv_bool("verbose_per_file", o.verbose_per_file));
    s.push_str(&kv_bool("truncate_filenames", o.truncate_filenames));
    s.push_str(&kv_bool("show_banner", o.show_banner));
    s.push_str(&kv_bool("show_summary", o.show_summary));
    s.push_str(&kv_bool("json_summary", o.json_summary));
    s.push_str(&kv_float("splash_duration", o.splash_duration));
    s.push_str(&kv_bool("autocomplete", o.autocomplete));
    s.push_str(&kv_bool("minimal_ui", o.minimal_ui));
    s.push_str(&kv_bool("allow_root", o.allow_root));
    s.push_str(&kv_bool("warn_root", o.warn_root));
    s.push_str(&kv_bool("bell_on_done", o.bell_on_done));
    s.push_str(&kv_str("art_banner", &o.art_banner));
    s.push_str(&kv_str("art_hub", &o.art_hub));

    let c = &cfg.codec;
    s.push_str("\n[codec]\n");
    s.push_str(&kv_str("default_mode", &c.default_mode));
    s.push_str(&kv_int("input_bits", c.input_bits));
    s.push_str(&kv_int("window_samples", c.window_samples));
    s.push_str(&kv_int("noise_bits", c.noise_bits));
    s.push_str(&kv_str("verification", &c.verification));
    s.push_str(&kv_str("lossless_default_mode", &c.lossless_default_mode));

    let l = &c.lossless;
    s.push_str("\n[codec.lossless]\n");
    s.push_str(&kv_str("entropy_coder", &l.entropy_coder));
    s.push_str(&kv_int("lpc_order", l.lpc_order));
    s.push_str(&kv_bool("use_lifting", l.use_lifting));

    let cm = &cfg.compute;
    s.push_str("\n[compute]\n");
    s.push_str(&kv_int("workers", cm.workers));
    s.push_str(&kv_float("memory_limit_gib", cm.memory_limit_gib));
    s.push_str(&kv_str("numba_cache_dir", &cm.numba_cache_dir));
    s.push_str(&kv_str("backend", &cm.backend));

    let i = &cfg.integrity;
    s.push_str("\n[integrity]\n");
    s.push_str(&kv_str("window_checksum", &i.window_checksum));
    s.push_str(&kv_str("file_checksum", &i.file_checksum));
    s.push_str(&kv_bool("verify_after_write", i.verify_after_write));
    s.push_str(&kv_bool("verify_outliers", i.verify_outliers));
    s.push_str(&kv_bool("reject_corrupted_input", i.reject_corrupted_input));
    s.push_str(&kv_bool("refuse_double_strip", i.refuse_double_strip));
    s.push_str(&kv_bool("fail_fast", i.fail_fast));

    let r = &cfg.resume;
    s.push_str("\n[resume]\n");
    s.push_str(&kv_bool("enabled", r.enabled));
    s.push_str(&kv_str("state_file", &r.state_file));
    s.push_str(&kv_str("checkpoint_strategy", &r.checkpoint_strategy));
    s.push_str(&kv_str("on_existing_state", &r.on_existing_state));
    s.push_str(&kv_bool("skip_existing_output", r.skip_existing_output));
    s.push_str(&kv_bool("verify_skipped", r.verify_skipped));
    s.push_str(&kv_str("quarantine_dir", &r.quarantine_dir));
    s.push_str(&kv_int("max_retries", r.max_retries));
    s.push_str(&kv_float("retry_backoff_s", r.retry_backoff_s));

    let lg = &cfg.logging;
    s.push_str("\n[logging]\n");
    s.push_str(&kv_str("audit_log", &lg.audit_log));
    s.push_str(&kv_bool("append_audit", lg.append_audit));
    s.push_str(&kv_str("stderr_level", &lg.stderr_level));
    s.push_str(&kv_str("file_log", &lg.file_log));
    s.push_str(&kv_bool("include_tracebacks", lg.include_tracebacks));
    s.push_str(&kv_str("manifest", &lg.manifest));
    s.push_str(&kv_bool(
        "manifest_include_files",
        lg.manifest_include_files,
    ));

    let ip = &cfg.input;
    s.push_str("\n[input]\n");
    s.push_str(&kv_str_array("extensions", &ip.extensions));
    s.push_str(&kv_bool("recursive", ip.recursive));
    s.push_str(&kv_bool("follow_symlinks", ip.follow_symlinks));
    s.push_str(&kv_int("min_file_size", ip.min_file_size));
    s.push_str(&kv_int("max_file_size", ip.max_file_size));
    s.push_str(&kv_str_array("exclude_patterns", &ip.exclude_patterns));

    let of = &cfg.output_files;
    s.push_str("\n[output_files]\n");
    s.push_str(&kv_bool("preserve_structure", of.preserve_structure));
    s.push_str(&kv_bool("atomic_writes", of.atomic_writes));
    s.push_str(&kv_bool("fsync_on_write", of.fsync_on_write));

    let b = &cfg.backend;
    s.push_str("\n[backend]\n");
    s.push_str(&kv_str("mode", &b.mode));
    s.push_str(&kv_str("rust_binary", &b.rust_binary));
    s.push_str(&kv_str("custom_binary", &b.custom_binary));

    // Preserved unknown sections last, so we never silently drop user customisations.
    for (section, kv) in &cfg.extra {
        if section.is_empty() {
            for (k, v) in kv {
                s.push_str(&format!("{} = {}\n", k, v));
            }
        } else {
            s.push_str(&format!("\n[{}]\n", section));
            for (k, v) in kv {
                s.push_str(&format!("{} = {}\n", k, v));
            }
        }
    }
    s
}

fn kv_str(k: &str, v: &str) -> String {
    format!("{} = \"{}\"\n", k, escape(v))
}
fn kv_bool(k: &str, v: bool) -> String {
    format!("{} = {}\n", k, v)
}
fn kv_int(k: &str, v: i64) -> String {
    format!("{} = {}\n", k, v)
}
fn kv_float(k: &str, v: f64) -> String {
    // Always include decimal point so re-parsing as f64 round-trips.
    if v.fract() == 0.0 && v.is_finite() {
        format!("{} = {:.1}\n", k, v)
    } else {
        format!("{} = {}\n", k, v)
    }
}
fn kv_str_array(k: &str, v: &[String]) -> String {
    let parts: Vec<String> = v.iter().map(|s| format!("\"{}\"", escape(s))).collect();
    format!("{} = [{}]\n", k, parts.join(", "))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip() {
        let cfg = LamQuantConfig::default();
        let text = cfg.serialize();
        let parsed = LamQuantConfig::parse(&text).expect("parse");
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn modified_round_trip() {
        let mut cfg = LamQuantConfig::default();
        cfg.compute.workers = 7;
        cfg.compute.memory_limit_gib = 12.5;
        cfg.codec.noise_bits = 4;
        cfg.codec.window_samples = 5000;
        cfg.codec.lossless.lpc_order = 4;
        // Cover the new lossless_default_mode field with a non-
        // default value so the round-trip exercises both
        // apply_codec parse + serialize for it.
        cfg.codec.lossless_default_mode = "lml_siblings".into();
        cfg.input.extensions = vec!["edf".into(), "bdf".into(), "mef".into()];
        cfg.backend.rust_binary = "/usr/local/bin/lml".into();
        let text = cfg.serialize();
        let parsed = LamQuantConfig::parse(&text).expect("parse");
        assert_eq!(cfg, parsed);
    }

    /// apply_codec rejects values outside the canonical 3 forms for
    /// `lossless_default_mode`. A typo'd TOML must error at load
    /// time so the panel dispatch logic never sees an unknown mode.
    #[test]
    fn lossless_default_mode_rejects_invalid_value() {
        let mut cfg = LamQuantConfig::default().codec;
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"lma\"").is_ok());
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"prompt\"").is_ok());
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"lml_siblings\"").is_ok());
        // Any other value -- including the old "lossless" alias --
        // must surface as a load-time error.
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"lossless\"").is_err());
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"\"").is_err());
        assert!(apply_codec(&mut cfg, "lossless_default_mode", "\"LMA\"").is_err());
    }

    #[test]
    fn escapes_special_chars_in_strings() {
        let mut cfg = LamQuantConfig::default();
        cfg.backend.rust_binary = r#"/path with "quotes" and \backslashes"#.into();
        cfg.logging.audit_log = "tab\there".into();
        let text = cfg.serialize();
        let parsed = LamQuantConfig::parse(&text).expect("parse escaped");
        assert_eq!(cfg.backend.rust_binary, parsed.backend.rust_binary);
        assert_eq!(cfg.logging.audit_log, parsed.logging.audit_log);
    }

    #[test]
    fn preserves_unknown_keys_in_known_sections() {
        let mut cfg = LamQuantConfig::default();
        cfg.extra
            .entry("output".into())
            .or_default()
            .insert("future_key".into(), "\"future_value\"".into());
        let text = cfg.serialize();
        let parsed = LamQuantConfig::parse(&text).expect("parse");
        assert_eq!(
            parsed.extra.get("output").and_then(|m| m.get("future_key")),
            Some(&"\"future_value\"".to_string())
        );
    }

    #[test]
    fn preserves_unknown_sections() {
        let cfg_text = r#"
schema_version = "1.0"
instance_name = "default"

[experimental]
secret_flag = true
custom_path = "/opt/foo"
"#;
        let parsed = LamQuantConfig::parse(cfg_text).expect("parse");
        assert!(parsed.extra.contains_key("experimental"));
        let text = parsed.serialize();
        let reparsed = LamQuantConfig::parse(&text).expect("reparse");
        assert_eq!(parsed.extra, reparsed.extra);
    }

    #[test]
    fn ignores_comments_and_blanks() {
        let txt = r#"
# top comment
schema_version = "1.0"  # trailing
instance_name = "x"

[output]
# section comment
color = "auto"
"#;
        let parsed = LamQuantConfig::parse(txt).expect("parse");
        assert_eq!(parsed.schema_version, "1.0");
        assert_eq!(parsed.output.color, "auto");
    }

    #[test]
    fn rejects_malformed_value() {
        let txt = r#"
schema_version = 1.0
"#;
        assert!(LamQuantConfig::parse(txt).is_err());
    }

    #[test]
    fn float_defaults_have_decimal_point() {
        let cfg = LamQuantConfig::default();
        let text = cfg.serialize();
        assert!(text.contains("refresh_hz = 10.0"));
        assert!(text.contains("splash_duration = 0.5"));
        assert!(text.contains("retry_backoff_s = 5.0"));
    }
}
