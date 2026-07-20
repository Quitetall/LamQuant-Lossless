//! Bounded stdio worker contract for raw-free MIX1 measurement.

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn lqraw_fixture() -> Vec<u8> {
    let channels = 3usize;
    let samples = 48usize;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, 16, 0]);
    bytes.extend_from_slice(&256_000u32.to_le_bytes());
    bytes.extend_from_slice(&(channels as u32).to_le_bytes());
    bytes.extend_from_slice(&(samples as u32).to_le_bytes());
    for channel in 0..channels {
        for time in 0..samples {
            let value =
                (channel as i32 + 1) * (time as i32 * 3 - 17) + ((time + channel * 5) % 7) as i32;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn peer_magic(packet: &[u8]) -> &[u8] {
    if packet.get(4) == Some(&4) {
        &packet[24..28]
    } else {
        &packet[72..76]
    }
}

fn stdio_worker(binary: &str, arguments: &[&str], input: &[u8]) -> Output {
    let mut child = Command::new(binary)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn MIX1 worker");
    child
        .stdin
        .take()
        .expect("piped standard input")
        .write_all(input)
        .expect("write worker input");
    child.wait_with_output().expect("wait for MIX1 worker")
}

#[test]
fn mix1_stdio_worker_roundtrips_every_score_shift_without_scratch_files() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();

    for score_shift in ["2", "3", "4", "5", "6", "7", "8"] {
        let encoded = stdio_worker(binary, &["mix1-encode-stdio", score_shift], &raw);
        assert!(
            encoded.status.success(),
            "MIX1 stdio encode failed: {}",
            String::from_utf8_lossy(&encoded.stderr)
        );
        assert_eq!(&encoded.stdout[..4], b"OV2P");
        assert_eq!(
            &encoded.stdout[72..78],
            [b'M', b'I', b'X', b'1', 0xA7, score_shift.parse().unwrap()]
        );

        let decoded = stdio_worker(binary, &["mix1-decode-stdio"], &encoded.stdout);
        assert!(
            decoded.status.success(),
            "MIX1 stdio decode failed: {}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert_eq!(decoded.stdout, raw);
    }
}

#[test]
fn mix1_stdio_worker_rejects_out_of_range_score_shift() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let encoded = stdio_worker(binary, &["mix1-encode-stdio", "1"], &lqraw_fixture());

    assert!(!encoded.status.success());
    assert!(String::from_utf8_lossy(&encoded.stderr).contains("score shift"));
}

#[test]
fn mix1_best_stdio_worker_selects_an_actual_complete_packet() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let best = stdio_worker(binary, &["mix1-encode-best-stdio"], &raw);
    assert!(best.status.success());

    let individual = ["2", "3", "4", "5", "6", "7", "8"]
        .map(|shift| stdio_worker(binary, &["mix1-encode-stdio", shift], &raw));
    assert_eq!(
        best.stdout.len(),
        individual
            .iter()
            .map(|output| output.stdout.len())
            .min()
            .unwrap()
    );
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &best.stdout);
    assert_eq!(restored.stdout, raw);
}

#[test]
fn peer_stdio_worker_emits_a_complete_exact_packet() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let incumbent = stdio_worker(binary, &["mix1-encode-best-stdio"], &raw);
    let best = stdio_worker(binary, &["mix1-peer-encode-best-stdio"], &raw);

    assert!(best.status.success());
    assert!(best.stdout.len() <= incumbent.stdout.len());
    assert!(matches!(
        peer_magic(&best.stdout),
        b"MIX1" | b"MMV1" | b"MCH1" | b"MCX1" | b"MQX1" | b"MPX1" | b"APX1" | b"BQX1" | b"ALX1"
    ));
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &best.stdout);
    assert_eq!(restored.stdout, raw);
}

#[test]
fn peer_no_alias_control_worker_emits_a_complete_exact_packet() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let control = stdio_worker(binary, &["mix1-peer-encode-best-no-alias-stdio"], &raw);

    assert!(control.status.success());
    assert_ne!(peer_magic(&control.stdout), b"ALX1");
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &control.stdout);
    assert_eq!(restored.stdout, raw);
}

#[test]
fn permuted_peer_stdio_worker_emits_a_complete_exact_packet() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let encoded = stdio_worker(binary, &["mix1-peer-permuted-encode-stdio", "4", "7"], &raw);

    assert!(
        encoded.status.success(),
        "permuted peer encode failed: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    assert_eq!(&encoded.stdout[72..79], b"MPX1\xA7\x04\x07");
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &encoded.stdout);
    assert_eq!(restored.stdout, raw);
}
