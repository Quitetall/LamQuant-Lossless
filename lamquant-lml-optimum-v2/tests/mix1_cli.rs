//! Bounded stdio worker contract for raw-free MIX1 measurement.

use std::io::Write;
use std::process::{Command, Output, Stdio};

use lamquant_lml_optimum_v2::mix1::Mix1Codec;

fn fixture_signal() -> Vec<Vec<i64>> {
    let channels = 3usize;
    let samples = 48usize;
    (0..channels)
        .map(|channel| {
            (0..samples)
                .map(|time| {
                    i64::from(
                        (channel as i32 + 1) * (time as i32 * 3 - 17)
                            + ((time + channel * 5) % 7) as i32,
                    )
                })
                .collect()
        })
        .collect()
}

fn lqraw_fixture() -> Vec<u8> {
    let signal = fixture_signal();
    let channels = signal.len();
    let samples = signal[0].len();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LQR1");
    bytes.extend_from_slice(&[1, 4, 16, 0]);
    bytes.extend_from_slice(&256_000u32.to_le_bytes());
    bytes.extend_from_slice(&(channels as u32).to_le_bytes());
    bytes.extend_from_slice(&(samples as u32).to_le_bytes());
    for channel in signal {
        for value in channel {
            bytes.extend_from_slice(&(value as i32).to_le_bytes());
        }
    }
    bytes
}

fn parse_candidate_bundle(packed: &[u8]) -> Vec<(String, String, Vec<u8>)> {
    assert_eq!(&packed[..4], b"A5B1");
    let count = usize::from(u16::from_le_bytes(packed[4..6].try_into().unwrap()));
    let mut offset = 6usize;
    let mut candidates = Vec::with_capacity(count);
    for _ in 0..count {
        let id_len = usize::from(u16::from_le_bytes(
            packed[offset..offset + 2].try_into().unwrap(),
        ));
        offset += 2;
        let family_len = usize::from(u16::from_le_bytes(
            packed[offset..offset + 2].try_into().unwrap(),
        ));
        offset += 2;
        let packet_len = usize::try_from(u32::from_le_bytes(
            packed[offset..offset + 4].try_into().unwrap(),
        ))
        .unwrap();
        offset += 4;
        let id = String::from_utf8(packed[offset..offset + id_len].to_vec()).unwrap();
        offset += id_len;
        let family = String::from_utf8(packed[offset..offset + family_len].to_vec()).unwrap();
        offset += family_len;
        let packet = packed[offset..offset + packet_len].to_vec();
        offset += packet_len;
        candidates.push((id, family, packet));
    }
    assert_eq!(offset, packed.len());
    candidates
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
        b"MIX1"
            | b"MMV1"
            | b"MCH1"
            | b"MCX1"
            | b"MQX1"
            | b"MPX1"
            | b"APX1"
            | b"BQX1"
            | b"ALX1"
            | b"BLX1"
    ));
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &best.stdout);
    assert_eq!(restored.stdout, raw);
}

#[test]
fn peer_r2_stdio_worker_is_exact_and_never_larger_than_peer_r1() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let peer_r1 = stdio_worker(binary, &["mix1-peer-encode-best-stdio"], &raw);
    let peer_r2 = stdio_worker(binary, &["mix1-peer-r2-encode-best-stdio"], &raw);

    assert!(peer_r1.status.success());
    assert!(
        peer_r2.status.success(),
        "peer-r2 encode failed: {}",
        String::from_utf8_lossy(&peer_r2.stderr)
    );
    assert!(peer_r2.stdout.len() <= peer_r1.stdout.len());
    let restored = stdio_worker(binary, &["mix1-peer-r2-decode-stdio"], &peer_r2.stdout);
    assert_eq!(restored.stdout, raw);
}

#[test]
fn peer_candidate_bundle_worker_is_ordered_exact_and_matches_production() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let bundled = stdio_worker(binary, &["mix1-peer-candidate-bundle-stdio"], &raw);
    assert!(
        bundled.status.success(),
        "candidate bundle failed: {}",
        String::from_utf8_lossy(&bundled.stderr)
    );
    let candidates = parse_candidate_bundle(&bundled.stdout);

    assert_eq!(candidates.len(), 50);
    assert_eq!(candidates.first().unwrap().0, "score-s2");
    assert_eq!(candidates.first().unwrap().1, "score");
    assert_eq!(candidates.last().unwrap().0, "bitplane");
    assert_eq!(candidates.last().unwrap().1, "bitplane");
    for (id, _family, packet) in &candidates {
        assert_eq!(
            Mix1Codec
                .decode_window(packet)
                .unwrap_or_else(|error| panic!("decode {id}: {error}"))
                .samples,
            fixture_signal()
        );
    }

    let production = stdio_worker(binary, &["mix1-peer-encode-best-stdio"], &raw);
    assert!(production.status.success());
    let strict_minimum = candidates
        .iter()
        .min_by_key(|(_, _, packet)| packet.len())
        .unwrap();
    assert_eq!(strict_minimum.2, production.stdout);
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
fn bitplane_layer_worker_emits_a_complete_exact_packet() {
    let binary = env!("CARGO_BIN_EXE_optimum-v2-codec");
    let raw = lqraw_fixture();
    let encoded = stdio_worker(binary, &["mix1-peer-bitplane-encode-stdio"], &raw);

    assert!(
        encoded.status.success(),
        "BLX1 encode failed: {}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    assert_eq!(peer_magic(&encoded.stdout), b"BLX1");
    let restored = stdio_worker(binary, &["mix1-decode-stdio"], &encoded.stdout);
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
