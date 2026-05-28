//! Peer registry — loads `~/.config/lamquant/peers.json` (XDG path).
//!
//! Why a separate file from `lamquant.toml`? The main config uses a
//! hand-rolled TOML parser (see `config.rs`); the peer list maps
//! naturally to JSON via `serde_json` (already a `host` feature dep) so
//! we get parsing for free without expanding the toml parser.
//!
//! Schema (peers.json):
//!
//! ```json
//! {
//!   "peers": [
//!     {
//!       "id": "gpu-box",
//!       "display": "Lab GPU box",
//!       "host": "10.0.0.10",
//!       "transport": {
//!         "kind": "ssh",
//!         "user": "lamquant",
//!         "key_path": "/home/me/.ssh/lamquant_gpu_box",
//!         "host_fingerprint": "SHA256:abc123…",
//!         "port": 22
//!       }
//!     }
//!   ]
//! }
//! ```
//!
//! Missing file → empty registry (peers feature off).
//! Malformed file → empty registry + status message.

use lamquant_ops::Peer;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level peers.json schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeersConfig {
    #[serde(default)]
    pub peers: Vec<Peer>,
}

/// Path to peers.json — derived from the resolved lamquant.toml dir so the
/// two configs stay co-located regardless of XDG / cwd-override order.
pub fn peers_config_path() -> PathBuf {
    let cfg_path = super::config::config_path();
    cfg_path
        .parent()
        .map(|p| p.join("peers.json"))
        .unwrap_or_else(|| PathBuf::from("peers.json"))
}

/// Load peers.json. Returns empty registry if the file is missing or
/// malformed (with a logged warning to stderr) — peers are an optional
/// feature; absence must not crash the TUI.
pub fn load() -> PeersConfig {
    let path = peers_config_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return PeersConfig::default(),
    };
    match serde_json::from_str::<PeersConfig>(&text) {
        Ok(mut cfg) => {
            // Empty-id peers collide with the "__select_peer:" clear-sentinel
            // (sentinel with empty body == clear sticky). Drop them defensively;
            // the peers.json schema requires a non-empty id.
            let before = cfg.peers.len();
            cfg.peers.retain(|p| !p.id.is_empty());
            let dropped = before - cfg.peers.len();
            if dropped > 0 {
                eprintln!(
                    "[lamquant] peers.json: dropped {} peer(s) with empty id",
                    dropped
                );
            }
            cfg
        }
        Err(e) => {
            eprintln!("[lamquant] peers.json parse error: {} — peers disabled", e);
            PeersConfig::default()
        }
    }
}
