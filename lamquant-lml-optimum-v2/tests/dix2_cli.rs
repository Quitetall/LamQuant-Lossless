//! Bounded stdio worker contract for source-frozen DIX2 screens.

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn lqraw_fixture(channels: usize, samples: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, 16, 0]);
    bytes.extend_from_slice(&250_000u32.to_le_bytes());
    bytes.extend_from_slice(&(channels as u32).to_le_bytes());
    bytes.extend_from_slice(&(samples as u32).to_le_bytes());
    for channel in 0..channels {
        for sample in 0..samples {
            let common = i32::try_from(sample).unwrap() * 7 - 400;
            let value = common + i32::try_from((sample + channel * 3) % 9).unwrap() - 4;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn stdio_worker(binary: &str, arguments: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(binary)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn dix2_stdio_worker_roundtrips_every_profile_without_scratch_files() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture(2, 257);
    let metadata = r#"{"channel_labels":["EEG Fp1-Ref","EEG Fp2-Ref"]}"#;

    for (profile, wire_profile) in [
        ("product", 0u8),
        ("native", 1),
        ("raw", 2),
        ("delta", 3),
        ("temporal", 4),
        ("tree", 5),
    ] {
        let encoded = stdio_worker(binary, &["dix2-encode-stdio", profile, metadata], &raw);
        assert!(
            encoded.status.success(),
            "DIX2 stdio encode failed: {}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        assert_eq!(&encoded.stdout[..12], b"LMO1\x03\x00\x03DIX2\x01");
        assert_eq!(encoded.stdout[14], wire_profile);

        let decoded = stdio_worker(binary, &["dix2-decode-stdio"], &encoded.stdout);
        assert!(
            decoded.status.success(),
            "DIX2 stdio decode failed: {}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert_eq!(decoded.stdout, raw);
    }
}
