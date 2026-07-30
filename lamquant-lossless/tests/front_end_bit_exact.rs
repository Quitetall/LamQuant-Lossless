//! Front-end bit-exact baseline — freeze current behavior so the ABIR migration
//! (ADR 0069: frontends → typed IR → backends) can be proven byte-identical (Step A).
//!
//! Four locks bracket the refactor:
//!   * **codec-wire** (always runs, no fixtures): deterministic signals →
//!     `lml::compress` → sha256. Locks the per-window LML1 payload — the backend
//!     invariant that must NEVER change however the front-end is rewritten.
//!   * **container** (`--features archive`): synthetic signals →
//!     ABIR-native container encode → authenticated
//!     ABIR/BCS2 bundle whose enclosed LML1 packets match the frozen kernel.
//!   * **EDF→container** (`--features archive`): a synth EDF → `edf::read_edf` →
//!     the same `.lml` sink. Locks that the EDF front-end serializes identically.
//!   * **front-end (NWB)** (`--features nwb`): an h5py fixture → ABIR tensors
//!     → sha256 of the parsed IR signal. Skips when python3+h5py absent.
//!
//! Scope note (deliberate): the container/EDF locks pin metadata to `"{}"`, so they
//! freeze the shared IR→container byte path the ABIR refactor touches — NOT the
//! bin/lml.rs CLI metadata (encoder_version / source_file / zstd header), which is
//! version-fragile and out of Step-A scope. Never hash the `.lma` packer (mtime).
//!
//! Regenerate goldens after an INTENTIONAL change (reference Linux x86_64 toolchain
//! + pinned Cargo.lock; record the why in the commit message):
//!   LAMQUANT_REGEN_FRONTEND=1 cargo test --features archive --test front_end_bit_exact -- --nocapture  # LML1 wire
//!   LAMQUANT_REGEN_FRONTEND=1 cargo test --features nwb     --test front_end_bit_exact -- --nocapture  # + NWB
//!   then paste the printed shas into the GOLDEN_* tables below.

#[cfg(feature = "nwb")]
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[cfg(feature = "archive")]
mod common;

/// sha256 over channel-major signal: count, then per-channel (len + i64-LE samples).
#[cfg(feature = "nwb")]
fn sha_signal(signal: &[Vec<i64>]) -> String {
    let mut h = Sha256::new();
    h.update((signal.len() as u64).to_le_bytes());
    for ch in signal {
        h.update((ch.len() as u64).to_le_bytes());
        for &sample in ch {
            h.update(sample.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn sha_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(b).finalize())
}

fn regen() -> bool {
    std::env::var("LAMQUANT_REGEN_FRONTEND").is_ok()
}

#[cfg(feature = "nwb")]
#[derive(Debug, Deserialize)]
struct NwbSlot {
    h5_path: String,
    first_ch: usize,
    n_ch: usize,
}

#[cfg(feature = "nwb")]
fn nwb_slots(read: &lamquant_core::source::SemanticRead) -> Vec<NwbSlot> {
    let capsule = read
        .opened
        .dataset()
        .source_capsules()
        .iter()
        .find(|capsule| {
            capsule.source().namespace() == "source.nwb.capsule.1"
                && capsule.source().value() == "nwb_slots"
        })
        .expect("slot metadata capsule present");
    let bytes = read
        .opened
        .access()
        .payload_bytes(capsule.content_id())
        .expect("slot metadata payload available");
    serde_json::from_slice(bytes).expect("slot metadata decodes")
}

#[cfg(feature = "nwb")]
fn signal_from_read(read: &lamquant_core::source::SemanticRead) -> Vec<Vec<i64>> {
    let mut signal = Vec::with_capacity(read.mapping.channels.len());
    for channel in &read.mapping.channels {
        let block = read
            .opened
            .block_view(channel.atom_id)
            .expect("NWB tensor payload block view");
        assert_eq!(
            block.bytes().len() % 8,
            0,
            "NWB tensor payload is i64 bytes"
        );
        signal.push(
            block
                .bytes()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("exact chunk")))
                .collect(),
        );
    }
    signal
}

/// Deterministic test signals — varied shapes the codec must encode identically.
fn fixtures() -> Vec<(&'static str, Vec<Vec<i64>>)> {
    let ramp: Vec<i64> = (0..2000).map(|t| (t % 257) as i64 - 128).collect();
    let sine: Vec<i64> = (0..2000)
        .map(|t| ((t as f64 * 0.13).sin() * 1000.0) as i64)
        .collect();
    let flat: Vec<i64> = vec![42; 2000];
    vec![
        ("single_ramp", vec![ramp.clone()]),
        (
            "multi_4ch",
            vec![ramp.clone(), sine.clone(), flat.clone(), ramp.clone()],
        ),
        ("flat_const", vec![flat]),
    ]
}

// Codec-wire goldens (lossless, noise_bits=0). Captured on main pre-IR-refactor.
const GOLDEN_WIRE: &[(&str, &str)] = &[
    (
        "single_ramp",
        "f3a890974f399ea71cde88ca7073c4bfa31765bd46d0c5d2e75170d75111d3d4",
    ),
    (
        "multi_4ch",
        "fb45d32714be7746c24be2d2c8cba83da7722f4cc69ea0848c7bf59a83cca64b",
    ),
    (
        "flat_const",
        "32faeba1ce70f364add756a863cad497da3ed48715fa12653832916b33226a3b",
    ),
];

#[test]
fn codec_wire_bytes_locked() {
    for (name, signal) in fixtures() {
        let bytes = lamquant_core::lml::compress(&signal, 0).expect("compress");
        let got = sha_bytes(&bytes);
        if regen() {
            println!("WIRE {name} = {got}");
            continue;
        }
        let want = GOLDEN_WIRE.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(got, want, "codec wire bytes changed for {name}");
    }
}

#[cfg(feature = "nwb")]
mod nwb_frontend {
    use super::*;
    use std::process::Command;

    const GOLDEN_NWB_SIGNAL: &str =
        "7dbf6db220c3e098e3a07b22f3e8a27d8d8fe88ab4e27665677f7c76cb7ca102";

    fn make_fixture(path: &std::path::Path) -> bool {
        let script = format!(
            r#"
import sys
try:
    import h5py, numpy as np
except Exception:
    sys.exit(42)
T, C = 800, 6
data = (np.arange(T).reshape(T,1)*7 + np.arange(C).reshape(1,C)).astype('<i2')
flags = (np.arange(300) % 11).astype('u1')
with h5py.File(r"{}", "w") as f:
    f.create_group("acquisition").create_group("ElectricalSeries").create_dataset("data", data=data, chunks=(200,C))
    f.create_dataset("flags", data=flags)
sys.exit(0)
"#,
            path.display()
        );
        matches!(
            Command::new("python3").arg("-c").arg(&script).status(),
            Ok(s) if s.success()
        )
    }

    #[test]
    fn nwb_reader_signal_locked() {
        let dir = tempfile::tempdir().unwrap();
        let fx = dir.path().join("fx.nwb");
        if !make_fixture(&fx) {
            eprintln!("SKIP nwb_reader_signal_locked: h5py unavailable");
            return;
        }
        let read = lamquant_core::nwb::read_semantic(&fx).expect("read_semantic");
        let read_dataset = read.opened.dataset();
        assert_eq!(read_dataset.recordings().len(), 1);
        assert_eq!(read_dataset.streams().len(), 1);
        assert!(!read.mapping.channels.is_empty());
        let slots = nwb_slots(&read);
        assert!(!slots.is_empty(), "NWB fixture should yield slot metadata");
        let mut total_slot_channels = 0usize;
        for slot in slots {
            assert!(!slot.h5_path.is_empty(), "slot path should be non-empty");
            assert!(slot.n_ch > 0);
            total_slot_channels = total_slot_channels
                .checked_add(slot.n_ch)
                .expect("slot channel count should not overflow");
            assert!(slot.first_ch + slot.n_ch <= read.mapping.channel_count);
        }
        assert_eq!(total_slot_channels, read.mapping.channel_count);
        let signal = signal_from_read(&read);
        let got = sha_signal(&signal);
        if regen() {
            println!("NWB signal = {got}");
            return;
        }
        assert_eq!(got, GOLDEN_NWB_SIGNAL, "NWB reader signal changed");
    }
}

/// Current-container gates (ADR 0139/0143). Locks the canonical ABIR/BCS2
/// envelope, exact reopen, same-process determinism, and the frozen LML1 packet
/// nested inside it. The full bundle intentionally includes an implementation
/// build ID (target, profile, features, compiler), so a cross-build whole-file
/// hash would reject valid provenance changes rather than wire-format drift.
///
/// SCOPE: it does NOT hash the bin/lml.rs CLI metadata (encoder_version /
/// source_file / zstd header) — that version-fragile path is out of Step-A scope and
/// would need its own version-normalized fixture.
///
/// Gated on `archive` (where `container`/`edf`/`ingest` live). The gate is load-bearing:
/// CI pins it explicitly via a `--features archive` step (ci.yml) — relying on the
/// default→host→archive chain alone is a false-green risk (the sibling `nwb` gate is
/// already dormant in the default lane).
#[cfg(feature = "archive")]
mod container_full {
    use super::*;
    use lamquant_core::lpc::LpcMode;

    /// Encode a signal to a full `.lml` in memory. Pins every
    /// byte-determining input: 250 Hz (exactly representable → stable
    /// `sample_rate_mhz`), 256-sample windows, `noise_bits=0`, `"{}"` metadata
    /// (the only metadata in the bytes is the deterministic codec stamp serde
    /// injects), and `LpcMode::default()` (Anytime{deadline:None} — never samples
    /// `Instant::now()`). No fs, no `.lma`. Backend is the shipped default
    /// (Desktop/rayon); the bytes are backend-independent by the
    /// `byte_equal_backends` contract (Firmware == Desktop), and this file never
    /// sets a backend, so there is no process-wide-`AtomicU8` ordering hazard.
    fn container_bytes(signal: &[Vec<i64>]) -> Vec<u8> {
        common::encode_uniform(signal, 250.0, 256, "{}", LpcMode::default())
    }

    /// Refuse to run with any byte-affecting encoder env override set. The first
    /// three are experimental encoder paths (lamquant-lml-mcu lml.rs). `LMQ_LEVINSON`
    /// (lpc.rs — integer block-float Levinson) and `LAMQUANT_PCRD` (lml.rs — lossy
    /// target-bps RD) are INERT under this golden's `Anytime{None}` lossless config
    /// (analyze_adaptive bypasses analyze()), but are listed so the lock stays honest
    /// if its `LpcMode` ever changes to Fixed/Adaptive (which DO route through analyze()).
    fn assert_clean_env() {
        for v in [
            "LAMQUANT_TRY_BIT_PACK",
            "LAMQUANT_TRY_ARITHMETIC",
            "LAMQUANT_TRY_EXTENDED_LPC",
            "LMQ_LEVINSON",
            "LAMQUANT_PCRD",
        ] {
            assert!(
                std::env::var(v).is_err(),
                "{v} is set — would change encoder bytes; unset it before running the golden"
            );
        }
    }

    #[test]
    fn container_bytes_locked() {
        assert_clean_env();
        for (name, signal) in fixtures() {
            let bytes = container_bytes(&signal);
            assert_eq!(&bytes[..4], b"ABIR", "{name} did not use current BCS2 wire");
            assert_eq!(
                bytes,
                container_bytes(&signal),
                "non-deterministic encode for {name}"
            );
            let opened = lamquant_core::container::open(&bytes).expect("open current container");
            assert_eq!(opened.signal(), signal, "semantic reopen changed {name}");
            let mut start = 0_usize;
            for (ordinal, (packet, &samples)) in opened
                .packets()
                .zip(opened.packet_sample_counts())
                .enumerate()
            {
                let end = start + samples;
                let window = signal
                    .iter()
                    .map(|channel| channel[start..end].to_vec())
                    .collect::<Vec<_>>();
                let expected = lamquant_core::lml::compress(&window, 0).unwrap();
                assert_eq!(
                    packet, expected,
                    "nested LML1 packet {ordinal} changed for {name}"
                );
                start = end;
            }
            assert_eq!(start, signal[0].len(), "incomplete packet sequence");
        }
    }

    #[test]
    fn edf_container_locked() {
        assert_clean_env();
        // Deterministic single-channel i16 ramp. SINGLE channel + SINGLE sample-rate
        // so the EDF reader's `sr_weights` HashMap has exactly one entry — no
        // per-process `max_by_key` tie-break (edf.rs:182-199). 250 Hz matches the
        // rate supplied to the ABIR encoder.
        let samples: Vec<i16> = (0..2000).map(|t| ((t % 257) - 128) as i16).collect();
        let edf_bytes = lamquant_core::ingest::synth_single_channel_edf(&samples, 250.0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synth.edf");
        std::fs::write(&path, &edf_bytes).unwrap();
        let edf = lamquant_core::edf::read_edf(&path).expect("read_edf");
        let got = container_bytes(&edf.signal);
        assert_eq!(
            got,
            container_bytes(&edf.signal),
            "non-deterministic .lml encode for EDF signal"
        );
        let ramp = fixtures().remove(0).1;
        assert_eq!(
            got,
            container_bytes(&ramp),
            "EDF and direct signal paths emitted different current containers"
        );
    }
}
