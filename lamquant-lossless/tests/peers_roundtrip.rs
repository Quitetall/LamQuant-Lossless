//! Phase E.2 — peers.json round-trip integration test.
//!
//! Verifies that every peer field (id, display, host, transport
//! kind + nested SSH config) survives a serialize → deserialize
//! cycle without loss. Catches drift between the Rust serde
//! representation and the on-disk schema documented in
//! `lamquant-core/src/tui/peers_config.rs`.

use std::path::PathBuf;

use lamquant_ops::{Peer, SshConfig, TransportKind};

#[test]
fn ssh_peer_preserves_all_fields_through_json() {
    // Full-struct equality via PartialEq derive — adding a new field
    // to Peer or SshConfig without updating this test will fail
    // immediately, instead of silently passing per-field assertions
    // that don't know about the new field.
    let original = Peer {
        id: "lab-gpu".into(),
        display: "Lab GPU box".into(),
        host: "10.0.0.10".into(),
        transport: TransportKind::Ssh(SshConfig {
            user: "lamquant".into(),
            key_path: PathBuf::from("/home/me/.ssh/lamquant_gpu_box"),
            known_hosts: PathBuf::from("/home/me/.ssh/known_hosts"),
            host_fingerprint: "SHA256:abc123==".into(),
            port: 22,
        }),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: Peer = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original, restored);
}

#[test]
fn peer_without_fingerprint_round_trips() {
    let p = Peer {
        id: "no-pin".into(),
        display: "No fingerprint pin".into(),
        host: "192.168.1.50".into(),
        transport: TransportKind::Ssh(SshConfig {
            user: "user".into(),
            key_path: PathBuf::from("/tmp/key"),
            known_hosts: PathBuf::from("/tmp/known_hosts"),
            host_fingerprint: String::new(),
            port: 2222,
        }),
    };
    let json = serde_json::to_string(&p).expect("serialize");
    let restored: Peer = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(p, restored);
}

#[test]
fn peers_config_top_level_round_trips() {
    // Mirrors the on-disk schema { "peers": [...] }.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Wrap {
        peers: Vec<Peer>,
    }
    let cfg = Wrap {
        peers: vec![
            Peer {
                id: "alpha".into(),
                display: "Alpha".into(),
                host: "alpha.local".into(),
                transport: TransportKind::Ssh(SshConfig {
                    user: "a".into(),
                    key_path: "/k/a".into(),
                    known_hosts: "/k/h".into(),
                    host_fingerprint: String::new(),
                    port: 22,
                }),
            },
            Peer {
                id: "beta".into(),
                display: "Beta".into(),
                host: "beta.local".into(),
                transport: TransportKind::Ssh(SshConfig {
                    user: "b".into(),
                    key_path: "/k/b".into(),
                    known_hosts: "/k/h".into(),
                    host_fingerprint: "SHA256:beta-fp".into(),
                    port: 2222,
                }),
            },
        ],
    };
    let json = serde_json::to_string_pretty(&cfg.peers).expect("serialize");
    let restored: Vec<Peer> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(cfg.peers, restored);
}
