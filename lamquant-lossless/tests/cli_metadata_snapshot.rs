//! CLI embedded-metadata snapshot — the **L5(a) safety net** for the ABIR
//! migration (ADR 0069). Freezes the EXACT metadata JSON string the `lml`
//! binary embeds in a `.lml` container, per input format.
//!
//! `front_end_bit_exact.rs`'s container/EDF locks deliberately pin
//! `metadata_json = "{}"` — they lock the shared IR→container byte path, NOT
//! the `bin/lml.rs` CLI's per-format `metadata_json` builders (the big inline
//! `format!` blocks around `encode_one_brainvision` / `encode_one_raw` /
//! `encode_one_cnt` / `encode_one_dicom` / `encode_one_eeglab` /
//! `cmd_encode`'s EDF tail). Those builders are exactly what a Step-2+ ABIR
//! relocation would move — so they need their own guard. This file is it.
//!
//! For each of the 6 auto-detected input formats (dispatch is by extension —
//! see `encode_one` in `src/bin/lml.rs`): synthesize (or load, for DICOM) a
//! tiny deterministic fixture, run the REAL `lml encode` binary against it
//! with `--no-bundle --i-understand-data-loss` (bare `.lml`, no `.lma`
//! wrapping so `container::read_file` can read it straight back), pull the
//! embedded metadata string via `lamquant_core::container::read_file`,
//! normalize the fields that are inherently run-to-run volatile (tempdir
//! path, crate version), and pin `sha256(normalized)`.
//!
//! **Why the container round-trip re-serializes the JSON:** every
//! `container::write_into*` call routes through
//! `lamquant-lml-legacy::container::metadata_with_codec_mode`, which parses
//! the hand-built `format!` JSON via `serde_json::Value`, inserts
//! `codec_mode` / (`lossless_mode` | `max_error` | `target_bps`) / `lpc_mode`,
//! and re-serializes. Since this crate's `serde_json` is NOT built with the
//! `preserve_order` feature, `serde_json::Value::Object` is a `BTreeMap` —
//! keys come back **alphabetically sorted**, not in the original `format!`
//! order. That's fine: we hash whatever `lml encode` + `container::read_file`
//! actually hand back (the CURRENT observable behavior), not the source
//! `format!` template. A refactor that changes field order, field names, or
//! field values will still flip this golden.
//!
//! Volatile fields normalized before hashing (see `NORMALIZED FIELDS` note
//! on each fixture fn below):
//!   - `source_file` — BrainVision/Raw/CNT/DICOM/EEGLAB embed the FULL
//!     encode-time path (`<reader>.rs`: `.display().to_string()`), which
//!     bakes in the tempdir. EDF is the one exception: `edf.rs::read_edf`
//!     stores `path.file_name()` only (basename), so with a fixed literal
//!     input filename it is ALREADY deterministic — deliberately left
//!     un-normalized so a future accidental switch to a full-path EDF
//!     `source_file` still flips this golden instead of being masked.
//!   - `encoder` (BrainVision/Raw/CNT/DICOM/EEGLAB) / `encoder_version`
//!     (EDF) — `lml/<CARGO_PKG_VERSION>`, changes on every version bump.
//!
//! Everything else scanned and found deterministic given a fixed fixture:
//! `signal_sha256` / `edf_header_sha256` / `trailing_data_sha256` (hash of
//! content we control), `edf_header` / `trailing_data` (zstd level 9 is a
//! pure function of its input bytes — no embedded timestamp), `codec_mode`
//! / `lossless_mode` / `lpc_mode` (CLI defaults resolve to
//! `LosslessMode::Mcu.default_lpc_mode()` = `LpcMode::Fixed`, which carries
//! no `Instant` — see `resolve_lpc_mode` / `LosslessMode::default_lpc_mode`
//! in `lamquant-lml-mcu/src/deployment.rs`).
//!
//! Regenerate after an INTENTIONAL change to a metadata builder (record the
//! why in the commit message):
//!   LAMQUANT_REGEN_CLI_META=1 cargo test -p lamquant-lml --features archive,dicom \
//!     --test cli_metadata_snapshot -- --nocapture
//! then paste the printed shas into `FROZEN` below. `assert_clean_env` is
//! EXPECTED to fail during a regen run (see its doc comment) — that failure
//! is the reminder to unset the var afterward, not a real regression.
#![cfg(feature = "archive")]

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

// ───────────────────────── plumbing ─────────────────────────

fn sha_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(b).finalize())
}

fn regen() -> bool {
    std::env::var("LAMQUANT_REGEN_CLI_META").is_ok()
}

/// Locate the `lml` binary cargo just built for THIS invocation's feature
/// set (mirrors `tests/op_e2e.rs::lml_path()` — `env!("CARGO_BIN_EXE_lml")`
/// is not used anywhere else in this crate's test suite, so we follow the
/// precedent that's actually proven to work here rather than introduce a
/// second, untested lookup convention).
fn lml_bin() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .expect("workspace root")
        .join("target");
    for c in &[
        target.join("debug").join("lml"),
        target.join("debug").join("lml.exe"),
        target.join("release").join("lml"),
        target.join("release").join("lml.exe"),
    ] {
        if c.exists() {
            return c.clone();
        }
    }
    panic!("lml binary not built; run `cargo build --bin lml --features archive` first");
}

/// Run the real `lml encode` CLI: bare `.lml` output (`--no-bundle`, ack'd
/// via `--i-understand-data-loss`), no `.lma` wrapping, so the resulting
/// file is exactly what `lamquant_core::container::read_file` expects.
/// Every other flag stays at its CLI default (lossless, `--lpc-mode auto`,
/// `--window-size 2500`) — this pins actual `lml encode <input> -o <out>`
/// behavior, not some hand-tuned invocation.
fn run_encode(input: &Path, output: &Path) {
    let out = Command::new(lml_bin())
        .arg("encode")
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg("--no-bundle")
        .arg("--i-understand-data-loss")
        .output()
        .expect("spawn lml encode");
    assert!(
        out.status.success(),
        "lml encode {} -o {} failed: status={:?}\nstdout={}\nstderr={}",
        input.display(),
        output.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Replace the value of a top-level JSON string field `"field":"..."` with
/// `replacement`, leaving every other byte of `json` untouched. Escape-aware
/// (walks `\`-escapes so a value containing an escaped quote doesn't
/// truncate early) but otherwise a raw byte scan — deliberately NOT a
/// parse-then-reserialize round-trip, so it can't accidentally normalize
/// away a real formatting regression in an unrelated field (key order,
/// number formatting, whitespace, ...).
fn replace_json_string_field(json: &str, field: &str, replacement: &str) -> String {
    let needle = format!("\"{field}\":\"");
    let start = json
        .find(&needle)
        .unwrap_or_else(|| panic!("field `{field}` not found in metadata JSON: {json}"));
    let val_start = start + needle.len();
    let bytes = json.as_bytes();
    let mut i = val_start;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => break,
            _ => i += 1,
        }
    }
    assert!(i < bytes.len(), "unterminated string value for `{field}`");
    format!("{}{}{}", &json[..val_start], replacement, &json[i..])
}

// ───────────────────────── fixtures ─────────────────────────
// Every fixture is fully self-contained and deterministic byte-for-byte
// given a fixed tempdir *layout* (the tempdir *path itself* is never baked
// into the hashed sha except through the fields we explicitly normalize —
// see the module doc comment).

/// EDF. `lamquant_core::ingest::synth_single_channel_edf` (shared with
/// `front_end_bit_exact.rs`) fixes every ASCII header field (patient_id,
/// startdate "01.01.01", starttime "00.00.00", ...) — no normalization
/// needed beyond `encoder_version`. Filename is a fixed literal so EDF's
/// basename-only `source_file` (see module doc) is deterministic too.
fn edf_fixture(dir: &Path) -> PathBuf {
    let samples: Vec<i16> = (0..500).map(|t| ((t % 97) - 48) as i16).collect();
    let bytes = lamquant_core::ingest::synth_single_channel_edf(&samples, 250.0);
    let p = dir.join("synth.edf");
    std::fs::write(&p, &bytes).expect("write synth edf");
    p
}

/// BrainVision (`.vhdr` + `.eeg` + `.vmrk`). Mirrors
/// `src/source/brainvision.rs`'s own `#[cfg(test)] synth_vhdr_int16_multiplexed`
/// and `read_bundle_int16_multiplexed_round_trip` fixture shape (that helper
/// is `#[cfg(test)]`-private to the lib crate, so an external integration
/// test can't reuse it directly — reproduced here byte-for-byte equivalent).
/// NORMALIZED: `source_file` (full `.vhdr` path), `encoder`.
fn brainvision_fixture(dir: &Path) -> PathBuf {
    let n_ch = 2usize;
    let n_samples = 50usize;
    let mut eeg_bytes = Vec::with_capacity(n_ch * n_samples * 2);
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let v = (s as i16) * (ch as i16 + 1) - 25;
            eeg_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("rec.eeg"), &eeg_bytes).expect("write .eeg");

    let mut vhdr = String::new();
    vhdr.push_str("Brain Vision Data Exchange Header File Version 1.0\n");
    vhdr.push_str("[Common Infos]\n");
    vhdr.push_str("DataFile=rec.eeg\n");
    vhdr.push_str("MarkerFile=rec.vmrk\n");
    vhdr.push_str("DataFormat=BINARY\n");
    vhdr.push_str("DataOrientation=MULTIPLEXED\n");
    vhdr.push_str(&format!("NumberOfChannels={n_ch}\n"));
    vhdr.push_str("SamplingInterval=4000\n");
    vhdr.push_str("\n[Binary Infos]\n");
    vhdr.push_str("BinaryFormat=INT_16\n");
    vhdr.push_str("\n[Channel Infos]\n");
    for i in 1..=n_ch {
        vhdr.push_str(&format!("Ch{i}=Ch{i}_name,REF,0.5,uV\n"));
    }
    std::fs::write(dir.join("rec.vhdr"), vhdr.as_bytes()).expect("write .vhdr");
    std::fs::write(dir.join("rec.vmrk"), b"; vmrk stub\n").expect("write .vmrk");
    dir.join("rec.vhdr")
}

/// Raw binary + JSON sidecar. Mirrors `src/source/raw.rs`'s
/// `#[cfg(test)] good_sidecar` shape. NORMALIZED: `source_file`, `encoder`.
fn raw_fixture(dir: &Path) -> PathBuf {
    let n_ch = 2usize;
    let n_samples = 50usize;
    let mut bytes = Vec::with_capacity(n_ch * n_samples * 2);
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let v = (s as i16) * (ch as i16 + 1);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let raw_path = dir.join("data.raw");
    std::fs::write(&raw_path, &bytes).expect("write .raw");
    let sidecar = format!(
        "{{\"n_channels\":{n_ch},\"sample_rate\":250.0,\"dtype\":\"int16\",\
         \"orientation\":\"multiplexed\",\"channels\":[\"ch0\",\"ch1\"],\
         \"phys_min\":[-200.0,-200.0],\"phys_max\":[200.0,200.0],\"phys_dim\":\"uV\"}}"
    );
    std::fs::write(dir.join("data.json"), sidecar).expect("write sidecar json");
    raw_path
}

/// NeuroScan CNT. Mirrors `src/source/cnt.rs`'s `#[cfg(test)] synth_cnt`.
/// NORMALIZED: `source_file`, `encoder`.
fn cnt_fixture(dir: &Path) -> PathBuf {
    const SETUP_HEADER_LEN: usize = 900;
    const ELECTLOC_LEN: usize = 75;
    let n_ch = 2usize;
    let n_samples = 50usize;
    let sample_rate: u16 = 250;

    let mut buf = vec![0u8; SETUP_HEADER_LEN];
    buf[370..372].copy_from_slice(&(n_ch as u16).to_le_bytes());
    buf[376..378].copy_from_slice(&sample_rate.to_le_bytes());
    for ch in 0..n_ch {
        let mut rec = vec![0u8; ELECTLOC_LEN];
        let label = format!("E{ch:02}");
        rec[..label.len()].copy_from_slice(label.as_bytes());
        buf.extend_from_slice(&rec);
    }
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let v = (s as i16) * (ch as i16 + 1);
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    let p = dir.join("rec.cnt");
    std::fs::write(&p, &buf).expect("write .cnt");
    p
}

/// DICOM Waveform. No tiny synthetic byte-crafted fixture (the format needs
/// preamble + DICM magic + real transfer-syntax elements — `dicom_parity.rs`
/// explicitly avoids hand-crafting one for the same reason). Reuses the
/// committed `tests/fixtures/dicom/general_ecg.dcm` — synthesized
/// deterministically by `tools/make_general_ecg_fixture.py`, already the
/// basis of `dicom_parity.rs::parse_general_ecg_fixture_matches_synthetic_golden`.
/// NORMALIZED: `source_file`, `encoder`.
#[cfg(feature = "dicom")]
fn dicom_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dicom")
        .join("general_ecg.dcm")
}

/// EEGLAB `.set` + `.fdt` + `.lml-meta.json`. The v1 reader (`src/source/
/// eeglab.rs`) never parses the MAT struct — only the sidecar JSON + `.fdt`
/// matter, so `.set` can be an arbitrary stub (it's preserved byte-exact as
/// a sidecar, never decoded). NORMALIZED: `source_file`, `encoder`.
fn eeglab_fixture(dir: &Path) -> PathBuf {
    let n_ch = 2usize;
    let n_samples = 50usize;
    let mut fdt = Vec::with_capacity(n_ch * n_samples * 4);
    for ch in 0..n_ch {
        for s in 0..n_samples {
            let f = (s as f32) * (ch as f32 + 1.0) - 10.0;
            fdt.extend_from_slice(&f.to_le_bytes());
        }
    }
    std::fs::write(dir.join("rec.fdt"), &fdt).expect("write .fdt");
    let meta = format!(
        "{{\"n_channels\":{n_ch},\"n_samples\":{n_samples},\"sample_rate\":250.0,\
         \"channels\":[\"ch0\",\"ch1\"],\"phys_dim\":\"uV\"}}"
    );
    std::fs::write(dir.join("rec.lml-meta.json"), meta).expect("write meta sidecar");
    let set_path = dir.join("rec.set");
    std::fs::write(
        &set_path,
        b"MATLAB 5.0 MAT-file, stub for cli_metadata_snapshot",
    )
    .expect("write .set stub");
    set_path
}

// ───────────────────────── frozen goldens ─────────────────────────

/// sha256(normalized metadata JSON) per format. Frozen via
/// `LAMQUANT_REGEN_CLI_META=1 cargo test -p lamquant-lml --features archive,dicom \
///  --test cli_metadata_snapshot -- --nocapture` on a clean env (see
/// `assert_clean_env`), Linux x86_64, `lml` v0.9.3.
const FROZEN: &[(&str, &str)] = &[
    (
        "edf",
        "e3c6d2d433ae3e38a6f529be65a1db37e93c9d697c975e54d91d76f93a53f9cd",
    ),
    (
        "brainvision",
        "2a2fff6b159cd6a293fd722b2467e5eb5de6fd02c241dbbf568b64a2b791f6c2",
    ),
    (
        "raw",
        "f19772945101d06d8d605265cbfd5c4e5000ab0216dd0164a15523f2901d37f0",
    ),
    (
        "cnt",
        "78a8d7493c2c2a8e579fe5ebf0831a54f71a560c410e4b4a9302fbcc29b16680",
    ),
    (
        "dicom",
        "f282987e7ccf279d7e3aeb07cda5e76f9ab5e341e6025032f25314bd3fcbbcd7",
    ),
    (
        "eeglab",
        "6abe26ffe9badaa0380565668ef5720ad677d795b647252d7182cf8319147a64",
    ),
];

fn check(name: &str, got: &str) {
    if regen() {
        println!("CLI_META {name} = {got}");
        return;
    }
    let want = FROZEN
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("no FROZEN entry for `{name}`"));
    assert_ne!(
        want, "REGEN",
        "`{name}` still has a REGEN placeholder — run with LAMQUANT_REGEN_CLI_META=1 \
         and paste the printed sha into FROZEN"
    );
    assert_eq!(got, want, "CLI embedded-metadata JSON drifted for `{name}`");
}

// ───────────────────────── tests ─────────────────────────

/// Guards the regen footgun: if a dev runs `LAMQUANT_REGEN_CLI_META=1 cargo
/// test -- --nocapture` to harvest fresh shas and forgets to unset the var
/// before a normal `cargo test`, every `check()` call above silently
/// degrades into a `println!` and the whole file would report a false
/// PASS. This test has NO regen-mode escape hatch: it fails loudly whenever
/// the var is set, in ANY invocation. During an intentional regen run this
/// is the one EXPECTED red test in an otherwise-printing run — see it fail,
/// harvest the shas from `--nocapture` stdout, unset the var, done.
#[test]
fn assert_clean_env() {
    assert!(
        std::env::var("LAMQUANT_REGEN_CLI_META").is_err(),
        "LAMQUANT_REGEN_CLI_META is set — every snapshot assertion in this file just \
         printed instead of asserting. Unset it before trusting a green run."
    );
}

#[test]
fn edf_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = edf_fixture(dir.path());
    let output = dir.path().join("out").join("synth.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    // EDF's `source_file` is basename-only (see module doc) — deliberately
    // NOT normalized so a future regression to full-path source_file still
    // flips this golden. Assert the fixed literal survived before hashing.
    assert!(
        meta.contains("\"source_file\":\"synth.edf\""),
        "EDF source_file expected to be the basename literal, got: {meta}"
    );
    let normalized = replace_json_string_field(&meta, "encoder_version", "lml/NORMALIZED");
    check("edf", &sha_bytes(normalized.as_bytes()));
}

#[test]
fn brainvision_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = brainvision_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    let normalized = replace_json_string_field(&meta, "source_file", "NORMALIZED_SOURCE_FILE");
    let normalized = replace_json_string_field(&normalized, "encoder", "lml/NORMALIZED");
    check("brainvision", &sha_bytes(normalized.as_bytes()));
}

#[test]
fn raw_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = raw_fixture(dir.path());
    let output = dir.path().join("out").join("data.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    let normalized = replace_json_string_field(&meta, "source_file", "NORMALIZED_SOURCE_FILE");
    let normalized = replace_json_string_field(&normalized, "encoder", "lml/NORMALIZED");
    check("raw", &sha_bytes(normalized.as_bytes()));
}

#[test]
fn cnt_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = cnt_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    let normalized = replace_json_string_field(&meta, "source_file", "NORMALIZED_SOURCE_FILE");
    let normalized = replace_json_string_field(&normalized, "encoder", "lml/NORMALIZED");
    check("cnt", &sha_bytes(normalized.as_bytes()));
}

#[cfg(feature = "dicom")]
#[test]
fn dicom_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = dicom_fixture();
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    let normalized = replace_json_string_field(&meta, "source_file", "NORMALIZED_SOURCE_FILE");
    let normalized = replace_json_string_field(&normalized, "encoder", "lml/NORMALIZED");
    check("dicom", &sha_bytes(normalized.as_bytes()));
}

#[cfg(not(feature = "dicom"))]
#[test]
fn dicom_metadata_locked() {
    eprintln!(
        "SKIP dicom_metadata_locked: built without `--features dicom` \
         (lml refuses .dcm input without it)"
    );
}

#[test]
fn eeglab_metadata_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = eeglab_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let (_signal, meta) = lamquant_core::container::read_file(&output).unwrap();
    let normalized = replace_json_string_field(&meta, "source_file", "NORMALIZED_SOURCE_FILE");
    let normalized = replace_json_string_field(&normalized, "encoder", "lml/NORMALIZED");
    check("eeglab", &sha_bytes(normalized.as_bytes()));
}
