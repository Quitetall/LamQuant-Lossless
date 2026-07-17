use std::fs;
use std::process::Command;

use sha2::{Digest, Sha256};

#[test]
fn benchmark_cli_protocol_round_trips_lqraw() {
    let root = std::env::temp_dir().join(format!("optimum_v2_cli_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let raw = root.join("input.lqraw");
    let stream = root.join("output.lmo");
    let recon = root.join("recon.lqraw");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, 16, 0]);
    bytes.extend_from_slice(&250_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    for sample in [1i32, 2, 3, 4, -4, -3, -2, -1] {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(&raw, &bytes).unwrap();

    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    assert!(Command::new(binary)
        .args(["encode", raw.to_str().unwrap(), stream.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let stream_bytes = fs::read(&stream).unwrap();
    assert_eq!(&stream_bytes[0..5], b"LMO1\x03");
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&stream_bytes)),
        [
            0x6a, 0xbd, 0x83, 0x33, 0xf5, 0xdf, 0xb8, 0x90, 0xf1, 0x9e, 0x00, 0x6f, 0x19, 0x17,
            0x9e, 0x36, 0x0b, 0xae, 0x68, 0x6e, 0x82, 0x7d, 0xd9, 0xfe, 0xaf, 0x21, 0xd0, 0x09,
            0xd0, 0xb1, 0x66, 0xef,
        ]
    );
    assert!(Command::new(binary)
        .args(["decode", stream.to_str().unwrap(), recon.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    assert_eq!(fs::read(&recon).unwrap(), bytes);

    let descriptor = root.join("descriptor.json");
    assert!(Command::new(binary)
        .args(["describe", descriptor.to_str().unwrap()])
        .status()
        .unwrap()
        .success());
    let value: serde_json::Value = serde_json::from_slice(&fs::read(descriptor).unwrap()).unwrap();
    assert_eq!(value["wire"], "LMO1-v3/BGF1-v1/DIX1-v2-construction");

    let oversized = root.join("oversized.lqraw");
    let file = fs::File::create(&oversized).unwrap();
    file.set_len((20 + 8_388_608 * 4 + 1) as u64).unwrap();
    assert!(!Command::new(binary)
        .args([
            "encode",
            oversized.to_str().unwrap(),
            root.join("must-not-exist.lmo").to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());

    let _ = fs::remove_dir_all(root);
}
