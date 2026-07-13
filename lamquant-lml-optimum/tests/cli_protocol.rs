#![cfg(feature = "benchmark-cli")]

use std::fs;
use std::path::Path;
use std::process::Command;

fn fixture_bytes(sample_rate_mhz: u32, bit_depth: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, bit_depth, 0]);
    bytes.extend_from_slice(&sample_rate_mhz.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    for sample in [1i32, 2, 3, 4, -4, -3, -2, -1] {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn write_metadata(path: &std::path::Path, sample_rate_mhz: u32, bit_depth: u8) {
    fs::write(
        path,
        format!(
            "{{\"bit_depth\":{bit_depth},\"n_channels\":2,\"n_samples\":4,\"sample_rate_mhz\":{sample_rate_mhz}}}\n"
        ),
    )
    .unwrap();
}

fn codec_lossless_git(args: &[&str]) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("optimum package must live under codec-lossless");
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[test]
fn benchmark_cli_emits_raw_lmo_v2_and_round_trips_lqraw() {
    let root = std::env::temp_dir().join(format!("optimum_v1_cli_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let stream = root.join("output.lmo");
    let repeated_stream = root.join("output-repeated.lmo");
    let recon = root.join("recon.lqraw");
    let metadata = root.join("metadata.json");
    let expected = fixture_bytes(250_000, 16);
    fs::write(&raw, &expected).unwrap();
    write_metadata(&metadata, 250_000, 16);

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args(["encode", raw.to_str().unwrap(), stream.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let stream_bytes = fs::read(&stream).unwrap();
    assert_eq!(&stream_bytes[0..5], b"LMO1\x02");
    assert_ne!(&stream_bytes[0..4], b"LMOF");
    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args([
            "encode",
            raw.to_str().unwrap(),
            repeated_stream.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    assert_eq!(fs::read(&repeated_stream).unwrap(), stream_bytes);

    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args(["decode", stream.to_str().unwrap(), recon.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    assert_eq!(fs::read(&recon).unwrap(), expected);

    // Provenance belongs to the build, not to whichever directory the
    // benchmark runner copies the executable into.
    let copied_binary = root.join("copied-optimum-v1-codec");
    fs::copy(binary, &copied_binary).unwrap();
    let descriptor = root.join("descriptor.json");
    assert!(Command::new(&copied_binary)
        .args(["describe", descriptor.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let value: serde_json::Value = serde_json::from_slice(&fs::read(descriptor).unwrap()).unwrap();
    assert_eq!(value["codec"], "LamQuant Optimum v1 deterministic baseline");
    assert_eq!(value["wire"], "raw LMO1-v2");
    assert_eq!(value["package"]["name"], "lamquant-lml-optimum");
    assert_eq!(value["package"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(
        value["executable"]["path"],
        fs::canonicalize(copied_binary).unwrap().to_str().unwrap()
    );
    assert!(value["executable"]["bytes"].as_u64().unwrap() > 0);
    assert_eq!(value["executable"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(value["source_git"]["repository"], "codec-lossless");
    assert_eq!(value["source_git"]["capture"], "compile-time");
    assert_eq!(
        value["source_git"]["head"],
        codec_lossless_git(&["rev-parse", "HEAD"])
    );
    assert_eq!(
        value["source_git"]["dirty"],
        !codec_lossless_git(&["status", "--porcelain=v1", "--untracked-files=all"]).is_empty()
    );
    assert_eq!(value["build"]["id"].as_str().unwrap().len(), 64);
    assert!(!value["build"]["profile"].as_str().unwrap().is_empty());
    assert!(!value["build"]["target"].as_str().unwrap().is_empty());
    let features = value["build"]["features"].as_array().unwrap();
    for required in ["benchmark-cli", "decode", "encode", "std"] {
        assert!(features.iter().any(|feature| feature == required));
    }
    assert!(value["build"]["rustc"]["version"]
        .as_str()
        .unwrap()
        .starts_with("rustc "));
    assert_eq!(
        value["build"]["rustc"]["commit"].as_str().unwrap().len(),
        40
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn decoder_uses_benchmark_metadata_for_rate_and_bit_depth() {
    let root = std::env::temp_dir().join(format!("optimum_v1_meta_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let stream = root.join("output.lmo");
    let recon = root.join("recon.lqraw");
    let encode_metadata = root.join("encode-metadata.json");
    let decode_metadata = root.join("decode-metadata.json");
    fs::write(&raw, fixture_bytes(250_000, 16)).unwrap();
    write_metadata(&encode_metadata, 250_000, 16);
    write_metadata(&decode_metadata, 256_000, 24);

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &encode_metadata)
        .args(["encode", raw.to_str().unwrap(), stream.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &decode_metadata)
        .args(["decode", stream.to_str().unwrap(), recon.to_str().unwrap()])
        .status()
        .unwrap()
        .success());

    let reconstructed = fs::read(recon).unwrap();
    assert_eq!(
        u32::from_le_bytes(reconstructed[8..12].try_into().unwrap()),
        256_000
    );
    assert_eq!(reconstructed[6], 24);
    assert_eq!(&reconstructed[20..], &fixture_bytes(250_000, 16)[20..]);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn encoder_rejects_metadata_that_disagrees_with_lqraw() {
    let root = std::env::temp_dir().join(format!("optimum_v1_reject_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let stream = root.join("must-not-exist.lmo");
    let metadata = root.join("metadata.json");
    fs::write(&raw, fixture_bytes(250_000, 16)).unwrap();
    write_metadata(&metadata, 256_000, 16);

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    assert!(!Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args(["encode", raw.to_str().unwrap(), stream.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    assert!(!stream.exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn encoder_rejects_truncated_and_malformed_lqraw() {
    let root = std::env::temp_dir().join(format!("optimum_v1_lqraw_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let metadata = root.join("metadata.json");
    write_metadata(&metadata, 250_000, 16);

    let truncated = root.join("truncated.lqraw");
    fs::write(&truncated, b"LQR1\x01").unwrap();
    let mut malformed_bytes = fixture_bytes(250_000, 16);
    malformed_bytes[7] = 1;
    let malformed = root.join("malformed.lqraw");
    fs::write(&malformed, malformed_bytes).unwrap();

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    for (index, input) in [truncated, malformed].iter().enumerate() {
        let output = root.join(format!("must-not-exist-{index}.lmo"));
        assert!(!Command::new(binary)
            .env("LQ_CODEC_META_JSON", &metadata)
            .args(["encode", input.to_str().unwrap(), output.to_str().unwrap()])
            .status()
            .unwrap()
            .success());
        assert!(!output.exists());
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn decoder_rejects_non_lossless_mode_and_invalidates_stale_output() {
    let root = std::env::temp_dir().join(format!("optimum_v1_mode_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let stream = root.join("output.lmo");
    let tampered = root.join("tampered.lmo");
    let recon = root.join("must-not-survive.lqraw");
    let metadata = root.join("metadata.json");
    fs::write(&raw, fixture_bytes(250_000, 16)).unwrap();
    write_metadata(&metadata, 250_000, 16);

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    assert!(Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args(["encode", raw.to_str().unwrap(), stream.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let mut bytes = fs::read(stream).unwrap();
    bytes[5] = 2;
    fs::write(&tampered, bytes).unwrap();
    fs::write(&recon, b"stale").unwrap();

    assert!(!Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args([
            "decode",
            tampered.to_str().unwrap(),
            recon.to_str().unwrap()
        ])
        .status()
        .unwrap()
        .success());
    assert!(!recon.exists());

    let oversized = root.join("oversized.lmo");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(2 * (20 + 8_388_608 * 4) as u64 + 1024 * 1024 + 1)
        .unwrap();
    assert!(!Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args([
            "decode",
            oversized.to_str().unwrap(),
            recon.to_str().unwrap()
        ])
        .status()
        .unwrap()
        .success());
    assert!(!recon.exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn invalid_protocol_invalidates_stale_output_without_deleting_inputs() {
    let root = std::env::temp_dir().join(format!("optimum_v1_protocol_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let metadata = root.join("metadata.json");
    let stale = root.join("stale-output.lmo");
    fs::write(&raw, fixture_bytes(250_000, 16)).unwrap();
    write_metadata(&metadata, 250_000, 16);
    fs::write(&stale, b"stale").unwrap();

    let binary = env!("CARGO_BIN_EXE_optimum-v1-codec");
    assert!(!Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args(["invalid", raw.to_str().unwrap(), stale.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    assert!(!stale.exists());

    let describe_output = root.join("describe-output.json");
    let unrelated_extra = root.join("unrelated-extra.txt");
    fs::write(&describe_output, b"stale").unwrap();
    fs::write(&unrelated_extra, b"must survive").unwrap();
    assert!(!Command::new(binary)
        .args([
            "describe",
            describe_output.to_str().unwrap(),
            unrelated_extra.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    assert!(!describe_output.exists());
    assert_eq!(fs::read(&unrelated_extra).unwrap(), b"must survive");

    let extra_argument = root.join("extra.txt");
    fs::write(&stale, b"stale").unwrap();
    assert!(!Command::new(binary)
        .env("LQ_CODEC_META_JSON", &metadata)
        .args([
            "encode",
            raw.to_str().unwrap(),
            stale.to_str().unwrap(),
            extra_argument.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    assert!(!stale.exists());

    for protected in [&raw, &metadata] {
        let before = fs::read(protected).unwrap();
        assert!(!Command::new(binary)
            .env("LQ_CODEC_META_JSON", &metadata)
            .args(["encode", raw.to_str().unwrap(), protected.to_str().unwrap()])
            .status()
            .unwrap()
            .success());
        assert_eq!(fs::read(protected).unwrap(), before);
    }

    let _ = fs::remove_dir_all(root);
}
