//! CLI semantic-root snapshots for Package 26.
//!
//! Each supported source runs through real `lml encode`, authenticated BCS2
//! reopen, and RFC 8785 ABIR canonical debug serialization. Frozen hashes guard
//! typed source metadata, channel semantics, payload identity, and retained
//! capsule identity without reviving a second JSON metadata carrier.
//!
//! Regenerate only after an intentional semantic change:
//! `LAMQUANT_REGEN_CLI_SEMANTICS=1 cargo test -p lamquant-lml \
//! --features archive,dicom --test cli_semantic_snapshot -- --nocapture \
//! --skip assert_clean_env`.
#![cfg(feature = "archive")]

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

// ───────────────────────── plumbing ─────────────────────────

fn sha_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::new().chain_update(b).finalize())
}

fn regen() -> bool {
    std::env::var("LAMQUANT_REGEN_CLI_SEMANTICS").is_ok()
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
/// file is exactly what authenticated `container::read_bytes` expects.
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

fn read_semantics(path: &Path) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("read encoded LML");
    let opened = lamquant_core::container::open(&bytes).expect("authenticate BCS2 LML");
    semantic_abir::canonical_debug_json(opened.dataset()).expect("canonical ABIR semantics")
}

// ───────────────────────── fixtures ─────────────────────────
// Every fixture is fully self-contained and deterministic byte-for-byte
// given a fixed tempdir *layout* (the tempdir *path itself* is never baked
// into the hashed sha except through the fields we explicitly normalize —
// see the module doc comment).

/// EDF. `lamquant_core::ingest::synth_single_channel_edf` (shared with
/// `front_end_bit_exact.rs`) fixes every ASCII header field (patient_id,
/// startdate "01.01.01", starttime "00.00.00", ...). Filename is fixed, so
/// source identity remains deterministic.
fn edf_fixture(dir: &Path) -> PathBuf {
    let samples: Vec<i16> = (0..500).map(|t| ((t % 97) - 48) as i16).collect();
    let bytes = lamquant_core::ingest::synth_single_channel_edf(&samples, 250.0);
    let p = dir.join("synth.edf");
    std::fs::write(&p, &bytes).expect("write synth edf");
    p
}

/// BrainVision (`.vhdr` + `.eeg` + `.vmrk`). Mirrors
/// `src/source/brainvision.rs`'s own `#[cfg(test)] synth_vhdr_int16_multiplexed`
/// and `lower_int16_multiplexed_round_trip` fixture shape (that helper
/// is `#[cfg(test)]`-private to the lib crate, so an external integration
/// test can't reuse it directly — reproduced here byte-for-byte equivalent).
/// Reader records basename-only source identity.
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
/// `#[cfg(test)] good_sidecar` shape.
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
/// Source facts lower into ABIR recording/channel keys.
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
/// Source facts lower into ABIR recording/channel keys.
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
/// a sidecar, never decoded).
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
        b"MATLAB 5.0 MAT-file, stub for cli_semantic_snapshot",
    )
    .expect("write .set stub");
    set_path
}

// ───────────────────────── frozen goldens ─────────────────────────

/// SHA-256 of canonical ABIR semantics per source format.
const FROZEN: &[(&str, &str)] = &[
    (
        "edf",
        "652ea055d7bc87bc1dbf86c4dfbb6630463ec229f222d1caa5bdf1f628ce4367",
    ),
    (
        "brainvision",
        "fbb699088aee50d38775af9162656fb207be2cb93210253263e3c507dcdcd38d",
    ),
    (
        "raw",
        "e991e97288ef9d31a45d8984978d87fa707662056a66edeb3c7b1aa753876ccc",
    ),
    (
        "cnt",
        "efed98b906d580d7cab32d0b52a0f7752f3bcf93d5313c23742f423a04a83196",
    ),
    (
        "dicom",
        "2c6d5a89af85c6b4fc873d1704dcc63afc5a4971ddbb7af6481bf4f3f4d82ade",
    ),
    (
        "eeglab",
        "5c0976e32a09f3c0b31c40067df77d29eaff5147e67a4be847cd8ef03ceae9a7",
    ),
];

fn check(name: &str, got: &str) {
    if regen() {
        println!("CLI_SEMANTICS {name} = {got}");
        return;
    }
    let want = FROZEN
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or_else(|| panic!("no FROZEN entry for `{name}`"));
    assert_ne!(
        want, "REGEN",
        "`{name}` still has a REGEN placeholder — run with LAMQUANT_REGEN_CLI_SEMANTICS=1 \
         and paste the printed sha into FROZEN"
    );
    assert_eq!(got, want, "CLI ABIR semantics drifted for `{name}`");
}

// ───────────────────────── tests ─────────────────────────

/// Guards the regen footgun: if a dev runs `LAMQUANT_REGEN_CLI_SEMANTICS=1 cargo
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
        std::env::var("LAMQUANT_REGEN_CLI_SEMANTICS").is_err(),
        "LAMQUANT_REGEN_CLI_SEMANTICS is set — every snapshot assertion in this file just \
         printed instead of asserting. Unset it before trusting a green run."
    );
}

#[test]
fn edf_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = edf_fixture(dir.path());
    let output = dir.path().join("out").join("synth.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("edf", &sha_bytes(&semantics));
}

#[test]
fn brainvision_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = brainvision_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("brainvision", &sha_bytes(&semantics));
}

#[test]
fn raw_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = raw_fixture(dir.path());
    let output = dir.path().join("out").join("data.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("raw", &sha_bytes(&semantics));
}

#[test]
fn cnt_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = cnt_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("cnt", &sha_bytes(&semantics));
}

#[cfg(feature = "dicom")]
#[test]
fn dicom_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = dicom_fixture();
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("dicom", &sha_bytes(&semantics));
}

#[cfg(not(feature = "dicom"))]
#[test]
fn dicom_semantics_locked() {
    eprintln!(
        "SKIP dicom_semantics_locked: built without `--features dicom` \
         (lml refuses .dcm input without it)"
    );
}

#[test]
fn eeglab_semantics_locked() {
    let dir = tempfile::tempdir().unwrap();
    let input = eeglab_fixture(dir.path());
    let output = dir.path().join("out").join("rec.lml");
    run_encode(&input, &output);
    let semantics = read_semantics(&output);
    check("eeglab", &sha_bytes(&semantics));
}
