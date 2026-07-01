//! L1 DIFFERENTIAL ORACLE (ADR 0069) — pins the LML-v1 container's byte-identity,
//! round-trip, decode-old-files, and hardening reference BEFORE any code is
//! relocated into `lamquant-lml-legacy`. READ-ONLY: it exercises only the existing
//! decode/encode surface (no new encoder).
//!
//! Once green, every later relocation (L2–L8) is *provably* byte-identical: the
//! clean `write_abir` must reproduce these exact shas + round-trips + rejections,
//! or the oracle fails loudly. The cutover to the clean writer ships only when this
//! harness is green across the corpus.
//!
//! Corpus (all deterministic, in-test): A lossless-golden, B Fixed/Adaptive LPC,
//! C BoundedMae (δ=0/8), D TargetBps (lossy), E decode-old-files KAT, F header
//! parse, G hardening-rejects. (Synthetic 18/20-byte header variants + the
//! `specs/conformance/vectors` port are a tracked fast-follow.)
//!
//! **ADR 0069 L8 (post-cutover) — oracle-integrity note.** `lamquant_core::container`
//! (`lamquant_core::{container, ...}`) now ALIASES the clean `abir_container`
//! facade, whose write half dispatches through `write_abir` — the very thing this
//! oracle is supposed to check independently. So every `container::*` call in this
//! file imports `lamquant_lml_legacy::container` DIRECTLY (below), bypassing that
//! alias entirely, and every `write_abir` comparison is against genuinely
//! independent code: `lamquant_lml_legacy::container::{write_into,write_file_bounded_mae,
//! write_file_target_bps}` (the retiring `encode_into`-based v1 writer, linked via
//! the `oracle` feature's `lamquant-lml-legacy/legacy-encode`) vs
//! `lamquant_core::abir_container::write_abir_to_vec` (sourced from `Abir`). Two
//! separate implementations, asserted byte-equal against each other AND the frozen
//! S1/ORACLE goldens — not `write_abir` vs itself.
//!
//! Run: `cargo test --features oracle --test oracle_diff`
//! Regen the B/C/D determinism shas after an INTENTIONAL change:
//!   `LAMQUANT_REGEN_ORACLE=1 cargo test --features oracle --test oracle_diff -- --nocapture`
//!
//! **ADR 0069/0071 L9 — the ONE deliberate byte change.** `write_abir` now
//! wraps its output in the new `BCS1` 40-byte typed header instead of
//! reproducing the legacy `LML1` 32-byte header — so arms A–D can no longer
//! assert `write_abir(x) == container::write_into(x)` (they are, by design,
//! now DIFFERENT containers: same payload tail, different header). Each arm
//! keeps its legacy-vs-golden assertions UNCHANGED (that pins
//! `lamquant_lml_legacy::container::write_into`, which is untouched by L9)
//! and REPLACES the `write_abir == legacy` equality with two things: (1) the
//! real correctness proof, `decode(write_abir(x)) == x` (round-trip via the
//! new `bcs1_read_bytes`, independent of any golden — this is what makes the
//! restructured arms trustworthy, not weaker than before), and (2) a
//! REGEN-gated BCS1 golden sha (`BCS1_ORACLE_GOLDEN`, all "REGEN" sentinels
//! today — a human freezes them after reviewing the fresh values printed by
//! `LAMQUANT_REGEN_ORACLE=1 -- --nocapture`).
#![cfg(feature = "oracle")]

use lamquant_abir::Abir;
use lamquant_core::abir_container::{bcs1_read_bytes, write_abir_to_vec};
use lamquant_core::error::LmlError;
use lamquant_core::lml;
use lamquant_core::lpc::LpcMode;
// ADR 0069 L8: the INDEPENDENT legacy reference — deliberately NOT
// `lamquant_core::container` (that alias now resolves to `abir_container`,
// i.e. `write_abir` itself post-cutover). Every `container::*` call below is
// the retiring `legacy-encode` v1 writer / frozen v1 reader, linked directly
// so the diff against `write_abir_to_vec` stays a real two-implementation
// check. See module docs above.
use lamquant_lml_legacy::container;
use sha2::{Digest, Sha256};
use std::sync::atomic::Ordering;

// ───────────────────────── helpers ─────────────────────────

fn sha_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(b).finalize())
}

/// sha256 over decoded samples (channel-major, each i64 LE). Matches
/// legacy_crc_decode.rs::samples_sha256 so the two goldens agree.
fn samples_sha256(signal: &[Vec<i64>]) -> String {
    let mut h = Sha256::new();
    for ch in signal {
        for &s in ch {
            h.update(s.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}

fn regen() -> bool {
    std::env::var("LAMQUANT_REGEN_ORACLE").is_ok()
}

/// Refuse byte-affecting encoder env overrides. The lossless `Anytime{None}` golden
/// is inert to these, but Fixed/Adaptive/target_bps actually route through the
/// gated `analyze()` / PCRD paths — so EVERY arm calls this.
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
            "{v} is set — would change encoder bytes; unset it before the oracle"
        );
    }
}

/// The 3 deterministic synthetic signals (identical to front_end_bit_exact.rs::fixtures).
fn fixtures() -> Vec<(&'static str, Vec<Vec<i64>>)> {
    let ramp: Vec<i64> = (0..2000).map(|t| (t % 257) as i64 - 128).collect();
    let sine: Vec<i64> = (0..2000)
        .map(|t| ((t as f64 * 0.13).sin() * 1000.0) as i64)
        .collect();
    let flat: Vec<i64> = vec![42; 2000];
    vec![
        ("single_ramp", vec![ramp.clone()]),
        ("multi_4ch", vec![ramp.clone(), sine.clone(), flat.clone(), ramp.clone()]),
        ("flat_const", vec![flat]),
    ]
}

/// Encode a signal to a full `.lml` in memory under `lpc` and return the bytes.
/// Pins the same byte-determining inputs as the S1 golden: 250 Hz / 256-window /
/// noise_bits=0 / `"{}"` metadata.
fn write_into_vec(signal: &[Vec<i64>], lpc: LpcMode) -> Vec<u8> {
    let mut buf = Vec::new();
    container::write_into(&mut buf, signal, 250.0, 256, 0, "{}", lpc).expect("write_into");
    buf
}

/// Locate the first per-window LML1 packet inside a 32-byte-header container.
/// Lifted from legacy_crc_decode.rs::first_window_packet.
fn first_window_packet(buf: &[u8]) -> (usize, usize) {
    assert!(&buf[0..4] == lml::MAGIC, "container starts with LML1 magic");
    assert!(
        buf.len() >= 32 && matches!(buf[20], 16 | 24 | 32),
        "expected 32-byte container header (data[20] ∈ {{16,24,32}})"
    );
    let n_windows = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    assert!(n_windows >= 1, "container has at least one window");
    let meta_len = u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]) as usize;
    let index_start = 32usize + meta_len;
    let payload_start = index_start + n_windows * 4;
    let rel_off = u32::from_le_bytes([
        buf[index_start],
        buf[index_start + 1],
        buf[index_start + 2],
        buf[index_start + 3],
    ]) as usize;
    let block_pos = payload_start + rel_off;
    let len = u32::from_le_bytes([
        buf[block_pos],
        buf[block_pos + 1],
        buf[block_pos + 2],
        buf[block_pos + 3],
    ]) as usize;
    let packet_start = block_pos + 4;
    assert!(packet_start + len <= buf.len(), "window packet overruns buffer");
    (packet_start, len)
}

// Frozen S1 container goldens (MUST match front_end_bit_exact.rs::GOLDEN_CONTAINER).
const GOLDEN_CONTAINER: &[(&str, &str)] = &[
    ("single_ramp", "bf74545d5e5f5907244f4d738185f3b50fbb9359c607c9554a7b169688328b8b"),
    ("multi_4ch", "4c363b0c7abe9120ded6a16a53e604b79c6ea95a97a37f66070ff8ce749370e6"),
    ("flat_const", "e7b4bfdcefc0ac5ce5d1044e70f4c736b7f12a14d6eb9c71fb276fe8c1409e88"),
];

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/legacy_payload_crc.lml");
const DECODED_SAMPLES_SHA256: &str =
    "2dd003d15e50b8d5e7d927923ea1b9e7a6d5b07e6cbce319153ce6d8926377c0";

// Oracle-local determinism shas for the non-default modes (each LpcMode / codec_mode
// stamps a distinct sha — the S1 golden only pins Anytime{None}). Frozen via
// LAMQUANT_REGEN_ORACLE=1. These are a determinism tripwire, NOT an S1 contract:
// the load-bearing assertion for B/C/D is the round-trip + same-process determinism.
const ORACLE_GOLDEN: &[(&str, &str)] = &[
    ("B.single_ramp.fixed", "c3a2fb8567de5b2e697344f6fb928e425d3654377e7a8cec17879098abea45b0"),
    ("B.single_ramp.adaptive16", "582c7cf4c3958496457b25484c836bea3e9bf602a1f0f4cbee69ca13edc82ff0"),
    ("B.multi_4ch.fixed", "97638cce8766bd77f628f9d87b8404de950baa539d773979b2b7c07e463fae11"),
    ("B.multi_4ch.adaptive16", "832267dfdc296b0a0f06250d0ada52ca64760c1511afe90ef39538ccb3af9a5f"),
    ("C.single_ramp.d0", "702cf7b49be19c2a430e6fd3a9914cc3cde621ac656a1758ac06f6caf3507806"),
    ("C.single_ramp.d8", "7892e8d0614a51548ff8f60515373bb72a976ca59d1a8f3db7056e0b9a7a3356"),
    ("C.multi_4ch.d0", "23fbe82c46c4c1f050e5ccca58bbdf750d43feabe8a1fecf3c756821d6f5849d"),
    ("C.multi_4ch.d8", "446ab3eb9fcf74771c36da60d501287305b24a997c1a42de5e08cc24f819dd2f"),
    ("D.single_ramp.bps4", "54e5e0d242d0835195c91d8c71a50f85ee7835f124f29a87abb196fbb09677b3"),
    ("D.multi_4ch.bps4", "aa483784078d4019ed84baf7fa7db12bcb13c6510807f939c49078589f800dfa"),
];

/// regen-print or assert the frozen oracle-local sha for `key`.
fn check_oracle_sha(key: &str, got: &str) {
    if regen() {
        println!("ORACLE {key} = {got}");
        return;
    }
    let want = ORACLE_GOLDEN.iter().find(|(k, _)| *k == key).map(|(_, s)| *s);
    if let Some(want) = want {
        if want != "REGEN" {
            assert_eq!(got, want, "oracle sha drift: {key}");
        }
    }
}

// ─────────────────── ADR 0069/0071 L9 — BCS1 wire goldens ───────────────────
//
// The `write_abir_to_vec` output sha per fixture/mode, keyed the same way as
// `ORACLE_GOLDEN` (arm A uses the bare fixture name; B/C/D reuse their own
// `key` string). Left at the "REGEN" sentinel deliberately — this commit
// computes + prints the fresh values (via `LAMQUANT_REGEN_ORACLE=1 cargo
// test --features oracle --test oracle_diff -- --nocapture`, look for the
// `BCS1 <key> = <sha>` lines) but does NOT paste them in. Per
// `check_bcs1_oracle_sha`'s REGEN-skip below, a "REGEN" entry disables the
// byte-identity assertion for that key WITHOUT weakening the round-trip
// proof each arm runs unconditionally (`decode(write_abir(x)) == x`) — a
// human reviews the printed shas and freezes them here once the L9 wire is
// accepted.
const BCS1_ORACLE_GOLDEN: &[(&str, &str)] = &[
    ("A.single_ramp", "a00664d6a34bd203a7161b5a7db9b21c9c406bccfd49e6c4e33b2d22b499611d"),
    ("A.multi_4ch", "faec6946dbbc40373908a55f6722b059ec8a3fc0ba2c73dfcca6e879a262ce3e"),
    ("A.flat_const", "7fc7d5066e45aaf86b7838d2f6becdcc24a93f8942850a954ac82d079c82e0d9"),
    ("B.single_ramp.fixed", "dee345997f221a73c05ca2f7d5c81e8152a8312a9d2384978f324f2222f44536"),
    ("B.single_ramp.adaptive16", "cdf0cb6c32fdce9016507cf2b45401400337a31b5adae64c6ed8612c28fdf690"),
    ("B.multi_4ch.fixed", "7df6b5d0c0bcd863b8affbfde95dfd5500c2140293c8926c2b44a242782bb096"),
    ("B.multi_4ch.adaptive16", "2a456d7775f38664cda097a26e40193853a287129f95c30da64299348d183012"),
    ("C.single_ramp.d0", "f85276b55457e0b4274127a9b52993a7fff1eb19ac86b25e6d8096fd40f2c351"),
    ("C.single_ramp.d8", "56b900d52a62383304b39d9e19185cdf7176a95c3f895e22bf10ce35ecd4d51c"),
    ("C.multi_4ch.d0", "dd2cc9371f82a2af2ca5d98ef109f1da7b2764ee78cdaff051fada2e268540ba"),
    ("C.multi_4ch.d8", "170bd68cff8418464ca96d35ffc64543e67ad8afafd8b48b87e2b85c9c37bfa3"),
    ("D.single_ramp.bps4", "a43659355effef246465921763b26a1d2aad600f023e4bfcf16418f1e8803095"),
    ("D.multi_4ch.bps4", "0de8686d655b4d8ed5e3d7e88fd8cb234c258f74037879e8fb926c5f1c4154f2"),
];

/// regen-print or assert the frozen BCS1-container sha for `key`. Mirrors
/// `check_oracle_sha`'s REGEN-sentinel behavior exactly (see module note
/// above `BCS1_ORACLE_GOLDEN`).
fn check_bcs1_oracle_sha(key: &str, got: &str) {
    if regen() {
        println!("BCS1 {key} = {got}");
        return;
    }
    let want = BCS1_ORACLE_GOLDEN.iter().find(|(k, _)| *k == key).map(|(_, s)| *s);
    if let Some(want) = want {
        if want != "REGEN" {
            assert_eq!(got, want, "BCS1 oracle sha drift: {key}");
        }
    }
}

// ───────────────────────── A — lossless byte-identity + MAE=0 ─────────────────────────

#[test]
fn arm_a_lossless_golden() {
    assert_clean_env();
    for (name, sig) in fixtures() {
        let buf = write_into_vec(&sig, LpcMode::default());
        let got = sha_bytes(&buf);
        // byte-identity vs the frozen S1 golden (the load-bearing pin for the
        // LEGACY writer — `container::write_into` is untouched by L9, still
        // emits the old LML1 32-byte-header container).
        let want = GOLDEN_CONTAINER.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(got, want, "arm A byte-identity drift: {name}");
        // same-process determinism (legacy writer)
        assert_eq!(got, sha_bytes(&write_into_vec(&sig, LpcMode::default())), "arm A nondeterministic: {name}");

        // ADR 0069/0071 L9: `write_abir` now wraps its output in the NEW
        // BCS1 header — it INTENTIONALLY no longer reproduces the legacy
        // bytes (see module docs). The correctness proof that replaces the
        // old `abir_bytes == legacy` equality is the round-trip below,
        // independent of any golden; the BCS1 sha itself is still tracked
        // (REGEN-gated) so a silent regression stays visible once a human
        // freezes it.
        let abir = Abir::from_channels_i64(sig.clone(), 250.0);
        let abir_bytes = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec");
        assert_eq!(
            &abir_bytes[0..4],
            lamquant_abir::BCS1_MAGIC,
            "write_abir must emit the BCS1 magic: {name}"
        );
        // same-process determinism (BCS1 writer)
        let abir_bytes_2 = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", LpcMode::default(), None, None)
            .expect("write_abir_to_vec (2nd)");
        assert_eq!(abir_bytes, abir_bytes_2, "arm A write_abir nondeterministic: {name}");
        check_bcs1_oracle_sha(&format!("A.{name}"), &sha_bytes(&abir_bytes));

        // THE REAL CORRECTNESS PROOF (independent of any golden): decoding
        // write_abir's BCS1 output must reproduce `sig` byte-exact, MAE=0.
        let (rec_abir, _meta_abir) =
            bcs1_read_bytes(&abir_bytes).expect("bcs1_read_bytes(write_abir_to_vec(x))");
        assert_eq!(rec_abir.len(), sig.len(), "arm A BCS1 round-trip channel count: {name}");
        for ch in 0..sig.len() {
            assert_eq!(rec_abir[ch], sig[ch], "arm A BCS1 round-trip MAE!=0 ch{ch}: {name}");
        }

        // round-trip MAE=0 (exact) — legacy writer/reader, unchanged.
        let (rec, _) = container::read_bytes(&buf).expect("read_bytes");
        assert_eq!(rec.len(), sig.len(), "arm A channel count: {name}");
        for ch in 0..sig.len() {
            assert_eq!(rec[ch], sig[ch], "arm A MAE!=0 ch{ch}: {name}");
        }
    }
}

// ───────────────────────── B — Fixed/Adaptive LPC, lossless ─────────────────────────

#[test]
fn arm_b_lpc_modes_lossless() {
    assert_clean_env();
    let modes = [("fixed", LpcMode::Fixed), ("adaptive16", LpcMode::Adaptive { max_order: 16 })];
    for (name, sig) in fixtures().into_iter().take(2) {
        for (mname, lpc) in modes {
            let buf = write_into_vec(&sig, lpc);
            let key = format!("B.{name}.{mname}");
            let got = sha_bytes(&buf);
            assert_eq!(got, sha_bytes(&write_into_vec(&sig, lpc)), "arm B nondeterministic: {key}");
            check_oracle_sha(&key, &got);
            // ADR 0069/0071 L9: write_abir now emits BCS1, so it no longer
            // reproduces the legacy bytes (see module docs) — the round-trip
            // below is the correctness proof, for every LpcMode, not just
            // the default.
            let abir = Abir::from_channels_i64(sig.clone(), 250.0);
            let abir_bytes = write_abir_to_vec(&abir, 250.0, 256, 0, "{}", lpc, None, None)
                .expect("write_abir_to_vec");
            assert_eq!(
                &abir_bytes[0..4],
                lamquant_abir::BCS1_MAGIC,
                "write_abir must emit the BCS1 magic: {key}"
            );
            check_bcs1_oracle_sha(&key, &sha_bytes(&abir_bytes));
            let (rec_abir, _) =
                bcs1_read_bytes(&abir_bytes).expect("bcs1_read_bytes(write_abir_to_vec(x))");
            for ch in 0..sig.len() {
                assert_eq!(rec_abir[ch], sig[ch], "arm B BCS1 round-trip MAE!=0 {key} ch{ch}");
            }
            // lossless for ANY LpcMode → exact round-trip (legacy writer/reader)
            let (rec, _) = container::read_bytes(&buf).expect("read_bytes");
            for ch in 0..sig.len() {
                assert_eq!(rec[ch], sig[ch], "arm B MAE!=0 {key} ch{ch}");
            }
        }
    }
}

// ───────────────────────── C — BoundedMae (δ=0 exact, δ=8 ≤δ) ─────────────────────────

#[test]
fn arm_c_bounded_mae() {
    assert_clean_env();
    let dir = tempfile::tempdir().unwrap();
    for (name, sig) in fixtures().into_iter().take(2) {
        for delta in [0u64, 8] {
            let p = dir.path().join(format!("{name}_d{delta}.lml"));
            container::write_file_bounded_mae(&p, &sig, 250.0, 256, delta, "{}", LpcMode::default())
                .expect("write_file_bounded_mae");
            let bytes = std::fs::read(&p).unwrap();
            let key = format!("C.{name}.d{delta}");
            check_oracle_sha(&key, &sha_bytes(&bytes));
            // ADR 0069/0071 L9: write_abir now emits BCS1 in bounded-MAE
            // mode too (write_file_bounded_mae feeds noise_bits=0,
            // Some(delta), target_bps=None) — no longer legacy-byte-
            // identical (see module docs); the round-trip below (respecting
            // the SAME δ-bound semantics as the legacy round-trip check) is
            // the correctness proof.
            let abir = Abir::from_channels_i64(sig.clone(), 250.0);
            let abir_bytes = write_abir_to_vec(
                &abir,
                250.0,
                256,
                0,
                "{}",
                LpcMode::default(),
                Some(delta),
                None,
            )
            .expect("write_abir_to_vec");
            assert_eq!(
                &abir_bytes[0..4],
                lamquant_abir::BCS1_MAGIC,
                "write_abir must emit the BCS1 magic: {key}"
            );
            check_bcs1_oracle_sha(&key, &sha_bytes(&abir_bytes));
            let (rec_abir, _) =
                bcs1_read_bytes(&abir_bytes).expect("bcs1_read_bytes bounded");
            if delta == 0 {
                for ch in 0..sig.len() {
                    assert_eq!(rec_abir[ch], sig[ch], "arm C BCS1 δ=0 MAE!=0 {key} ch{ch}");
                }
            } else {
                let mut maxd_abir = 0i64;
                for ch in 0..sig.len() {
                    for i in 0..sig[ch].len() {
                        maxd_abir = maxd_abir.max((sig[ch][i] - rec_abir[ch][i]).abs());
                    }
                }
                assert!(
                    maxd_abir as u64 <= delta,
                    "arm C BCS1 δ={delta} bound violated: max|diff|={maxd_abir} {key}"
                );
            }

            // legacy writer/reader round-trip, unchanged.
            let (rec, _) = container::read_file(&p).expect("read_file bounded");
            if delta == 0 {
                for ch in 0..sig.len() {
                    assert_eq!(rec[ch], sig[ch], "arm C δ=0 MAE!=0 {key} ch{ch}");
                }
            } else {
                let mut maxd = 0i64;
                for ch in 0..sig.len() {
                    for i in 0..sig[ch].len() {
                        maxd = maxd.max((sig[ch][i] - rec[ch][i]).abs());
                    }
                }
                assert!(maxd as u64 <= delta, "arm C δ={delta} bound violated: max|diff|={maxd} {key}");
            }
        }
    }
}

// ───────────────────────── D — TargetBps (lossy: decode-only) ─────────────────────────

#[test]
fn arm_d_target_bps() {
    assert_clean_env();
    let dir = tempfile::tempdir().unwrap();
    for (name, sig) in fixtures().into_iter().take(2) {
        let p = dir.path().join(format!("{name}_bps.lml"));
        container::write_file_target_bps(&p, &sig, 250.0, 256, 4.0, "{}", LpcMode::default())
            .expect("write_file_target_bps");
        let bytes = std::fs::read(&p).unwrap();
        let key = format!("D.{name}.bps4");
        check_oracle_sha(&key, &sha_bytes(&bytes));
        // ADR 0069/0071 L9: write_abir now emits BCS1 in target-BPS mode too
        // (write_file_target_bps feeds noise_bits=0, delta=None, Some(4.0))
        // — no longer legacy-byte-identical (see module docs); the
        // round-trip below (lossy: shape-only, matching the legacy
        // round-trip check) is the correctness proof.
        let abir = Abir::from_channels_i64(sig.clone(), 250.0);
        let abir_bytes = write_abir_to_vec(
            &abir,
            250.0,
            256,
            0,
            "{}",
            LpcMode::default(),
            None,
            Some(4.0),
        )
        .expect("write_abir_to_vec");
        assert_eq!(
            &abir_bytes[0..4],
            lamquant_abir::BCS1_MAGIC,
            "write_abir must emit the BCS1 magic: {key}"
        );
        check_bcs1_oracle_sha(&key, &sha_bytes(&abir_bytes));
        // lossy → decode must SUCCEED with correct shape; NEVER assert MAE=0.
        let (rec_abir, _) = bcs1_read_bytes(&abir_bytes).expect("bcs1_read_bytes target_bps");
        assert_eq!(rec_abir.len(), sig.len(), "arm D BCS1 channel count: {key}");
        for ch in 0..sig.len() {
            assert_eq!(rec_abir[ch].len(), sig[ch].len(), "arm D BCS1 sample count {key} ch{ch}");
        }

        // legacy writer/reader, unchanged.
        let (rec, _) = container::read_file(&p).expect("read_file target_bps");
        assert_eq!(rec.len(), sig.len(), "arm D channel count: {key}");
        for ch in 0..sig.len() {
            assert_eq!(rec[ch].len(), sig[ch].len(), "arm D sample count {key} ch{ch}");
        }
    }
}

// ───────────────────────── E — decode-old-files KAT (frozen decoder) ─────────────────────────

#[test]
fn arm_e_decode_old_files_positive() {
    lml::SAW_LEGACY_CRC.store(false, Ordering::Relaxed);
    let (signal, _meta) = container::read_file(std::path::Path::new(FIXTURE))
        .expect("legacy container must decode via the payload-only CRC fallback");
    assert!(!signal.is_empty() && !signal[0].is_empty(), "decoded legacy signal non-empty");
    assert!(
        lml::SAW_LEGACY_CRC.load(Ordering::Relaxed),
        "SAW_LEGACY_CRC must latch on the pre-a81cd04 fixture"
    );
    assert_eq!(samples_sha256(&signal), DECODED_SAMPLES_SHA256, "legacy decode drifted");
}

#[test]
fn arm_e_corruption_still_rejected() {
    let buf = std::fs::read(FIXTURE).expect("read fixture");
    let (start, len) = first_window_packet(&buf);
    let mut packet = buf[start..start + len].to_vec();
    lml::SAW_LEGACY_CRC.store(false, Ordering::Relaxed);
    lml::decompress(&packet).expect("carved packet decodes cleanly before corruption");
    let magic_off = packet.windows(4).position(|w| w == lml::MAGIC).expect("magic");
    let flip_at = magic_off + 22 + 64; // deep in lpc_meta/payload, covered by BOTH CRC scopes
    assert!(flip_at < packet.len(), "packet large enough to corrupt payload");
    packet[flip_at] ^= 0x01;
    assert!(
        matches!(lml::decompress(&packet), Err(LmlError::CrcMismatch { .. })),
        "corrupt payload must be rejected with CrcMismatch (both scopes miss)"
    );
}

// ───────────────────────── F — header parse (real 32-byte container) ─────────────────────────

#[test]
fn arm_f_header_parse_32byte() {
    let sig = fixtures().into_iter().find(|(n, _)| *n == "multi_4ch").unwrap().1;
    let buf = write_into_vec(&sig, LpcMode::default());
    let h = container::parse_header(&buf).expect("parse 32-byte header");
    assert_eq!(h.n_ch, 4, "n_ch");
    assert_eq!(h.total_samples, 2000, "total_samples");
    assert_eq!(h.window_size, 256, "window_size");
    assert!(h.payload_start >= 32, "payload_start past the 32-byte header");
}

// ───────────────────────── G — hardening (input → Err, no panic) ─────────────────────────

#[test]
fn arm_g_hardening_rejects() {
    // valid baseline to mutate
    let sig = fixtures().into_iter().next().unwrap().1; // single_ramp
    let good = write_into_vec(&sig, LpcMode::default());

    // truncation: a too-short buffer → Truncated, no panic.
    let short = vec![b'L', b'M', b'L', b'1', 0, 0, 0, 0, 0, 0]; // 10 bytes < 18
    assert!(
        matches!(container::read_bytes(&short), Err(LmlError::Truncated { .. })),
        "too-short header must be Truncated"
    );

    // bad magic: 32 bytes starting "XXXX" → InvalidMagic.
    let mut badmagic = good.clone();
    badmagic[0..4].copy_from_slice(b"XXXX");
    assert!(
        matches!(container::read_bytes(&badmagic), Err(LmlError::InvalidMagic(_))),
        "bad magic must be InvalidMagic"
    );

    // bad version: "LML9" → UnsupportedVersion.
    let mut badver = good.clone();
    badver[3] = b'9';
    assert!(
        matches!(container::read_bytes(&badver), Err(LmlError::UnsupportedVersion(_))),
        "bad version must be UnsupportedVersion"
    );

    // oversized total_samples (u32::MAX @ [10..14]) → InvalidHeader.
    let mut bigtotal = good.clone();
    bigtotal[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(
        matches!(container::read_bytes(&bigtotal), Err(_)),
        "oversized total_samples must be rejected"
    );

    // n_ch = 0 (@ [6..8]) → InvalidHeader.
    let mut zeroch = good.clone();
    zeroch[6..8].copy_from_slice(&0u16.to_le_bytes());
    assert!(
        matches!(container::read_bytes(&zeroch), Err(_)),
        "zero channels must be rejected"
    );

    // single-bit flip in a real window payload → CrcMismatch.
    let mut corrupt = good.clone();
    let (start, len) = first_window_packet(&corrupt);
    let magic_off = corrupt[start..start + len]
        .windows(4)
        .position(|w| w == lml::MAGIC)
        .expect("packet magic");
    corrupt[start + magic_off + 22 + 64] ^= 0x01;
    assert!(
        matches!(container::read_bytes(&corrupt), Err(LmlError::CrcMismatch { .. })),
        "corrupt window payload must be CrcMismatch"
    );

    // bounds-safety sweep: NO truncation may PANIC. The full 0..len sweep proves
    // the `pos+N>len` / checked-arith guards hold at every boundary (the call simply
    // returning — Ok or Err — is the no-panic proof).
    for k in 0..good.len() {
        let _ = container::read_bytes(&good[..k]);
    }
    // Truncations that remove header/metadata/window DATA must Err. (A footer-only
    // truncation may legitimately still decode — read_bytes tolerates a missing
    // LMLFOOT1 — so that is covered as no-panic above, not asserted as Err.)
    for k in [4usize, 18, 32, good.len() / 2] {
        assert!(
            container::read_bytes(&good[..k]).is_err(),
            "data-truncation at {k} must Err"
        );
    }

    // Regression (oracle-found L1): a `probe==1` buffer truncated to 18 bytes used
    // to OOB-panic in the 20-byte-header branch (container.rs parse_header/read_bytes)
    // — now a clean Truncated. Frozen so the to-be-legacy decoder stays panic-free.
    assert!(
        matches!(container::read_bytes(&good[..18]), Err(LmlError::Truncated { .. })),
        "18-byte truncation must be Truncated, not a panic"
    );
}
