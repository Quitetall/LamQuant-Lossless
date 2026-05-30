//! Router — screen navigation with history stack.
//!
//! Screens are identified by string IDs. The router maintains a stack
//! so "Back" pops to the previous screen. Navigation is decoupled from
//! rendering — panels don't need to know about each other.

/// A screen in the navigation stack.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenId(pub String);

impl ScreenId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Well-known screen IDs
pub const SCREEN_MAIN: &str = "main";
pub const SCREEN_LOSSLESS: &str = "lossless";
/// Sub-prompt for Lossless `[1] Compress` when
/// `codec.lossless_default_mode = "prompt"`: asks the user to pick
/// between `[a] LMA archive` and `[s] LML + siblings (copy)`.
/// Skipped when the setting is `lma` or `lml_siblings` (mode_panel
/// dispatch routes straight to the op).
pub const SCREEN_LOSSLESS_PROMPT: &str = "lossless_prompt";
pub const SCREEN_NEURAL: &str = "neural";
pub const SCREEN_CODEC_HUB: &str = "codec_hub";
pub const SCREEN_BATCH: &str = "batch";
pub const SCREEN_ARCHIVE: &str = "archive";
pub const SCREEN_VERIFY: &str = "verify";
pub const SCREEN_INFO: &str = "info";
pub const SCREEN_STATS: &str = "stats";
pub const SCREEN_BENCH: &str = "bench";
pub const SCREEN_SETTINGS: &str = "settings";
pub const SCREEN_SETUP: &str = "setup";
pub const SCREEN_HELP: &str = "help";
pub const SCREEN_BROWSE: &str = "browse";
pub const SCREEN_TEST: &str = "test";
pub const SCREEN_PEERS: &str = "peers";

// ── Op routes (handle_navigate dispatches `op:` prefix to start_op) ──
// Use these constants from panels to prevent typos. Compile-error if you
// reference a name that doesn't exist; runtime "Screen X — not loaded"
// fallback no longer triggered for typo'd op IDs.
pub const OP_ENCODE: &str = "op:encode";
pub const OP_ENCODE_LMA: &str = "op:encode_lma";
/// Lossless encode that emits per-EEG `.lml` files and COPIES every
/// non-EEG sibling verbatim into the output tree (no archive, no
/// zstd-on-metadata). Chosen by the user via the `[1] Compress`
/// picker overlay or the `codec.lossless_default_mode` setting.
pub const OP_ENCODE_LML_SIBLINGS: &str = "op:encode_lml_siblings";
pub const OP_DECODE: &str = "op:decode";
pub const OP_ENCODE_NEURAL: &str = "op:encode_neural";
pub const OP_DECODE_NEURAL: &str = "op:decode_neural";
pub const OP_VERIFY: &str = "op:verify";
pub const OP_VERIFY_MANIFEST: &str = "op:verify_manifest";
pub const OP_INFO: &str = "op:info";
pub const OP_STATS: &str = "op:stats";
pub const OP_BENCH: &str = "op:bench";
pub const OP_EXPORT_CSV: &str = "op:export_csv";
pub const OP_EXPORT_NPY: &str = "op:export_npy";
pub const OP_EXPORT_RAW: &str = "op:export_raw";
pub const OP_RECOVER: &str = "op:recover";

// ── Launcher routes (handle_navigate dispatches `launch:` prefix) ──
pub const LAUNCH_SETUP_PIP: &str = "launch:setup_pip";
pub const LAUNCH_SETUP_CARGO: &str = "launch:setup_cargo";
// LAUNCH_EAGLE_* consts removed in ADR 0026 W2 phase A.6 — Eagle tile
// dropped from lml codec-only hub. Umbrella copy with full launcher
// set lives at /lamquant/src/tui/router.rs (meta-repo).

// ── Sub-screens ──
pub const SCREEN_PREFLIGHT: &str = "preflight";
pub const SCREEN_TUTORIAL: &str = "tutorial";
pub const SCREEN_SETTINGS_HELP: &str = "settings_help";
pub const SCREEN_WIZARD: &str = "wizard";
pub const SCREEN_SYSCHECK: &str = "syscheck";

pub struct Router {
    stack: Vec<ScreenId>,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    pub fn new() -> Self {
        Self {
            stack: vec![ScreenId::new(SCREEN_MAIN)],
        }
    }

    /// Current screen ID.
    pub fn current(&self) -> &str {
        self.stack.last().map(|s| s.as_str()).unwrap_or(SCREEN_MAIN)
    }

    /// Navigate to a new screen (push onto stack).
    pub fn navigate(&mut self, screen: &str) {
        self.stack.push(ScreenId::new(screen));
    }

    /// Go back (pop stack). Returns false if already at root.
    pub fn back(&mut self) -> bool {
        if self.stack.len() > 1 {
            self.stack.pop();
            true
        } else {
            false
        }
    }

    /// Reset to main screen (clear stack).
    pub fn home(&mut self) {
        self.stack.clear();
        self.stack.push(ScreenId::new(SCREEN_MAIN));
    }

    /// Stack depth (for UI indicators).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}
