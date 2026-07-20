//! Cross-language differential gate for construction-private DIX2 packets.

use std::io::Write;
use std::process::{Command, Stdio};

use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix2_carrier::{Dix2CarrierMode, Dix2ConstructionCodec};
use serde_json::Value;

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

fn identities() -> Vec<ChannelIdentity> {
    vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ]
}

#[test]
fn standalone_python_decoder_matches_every_profile_and_boundary() {
    let codec = Dix2ConstructionCodec;
    let identities = identities();
    let signal = common_reference_signal(257);
    let mut cases = vec![
        codec
            .encode_window(&signal, &identities, 500_000, 24)
            .unwrap(),
        codec
            .encode_native_window(&signal, &identities, 500_000, 24)
            .unwrap(),
    ];
    for mode in [
        Dix2CarrierMode::Raw,
        Dix2CarrierMode::Delta,
        Dix2CarrierMode::TemporalRans,
        Dix2CarrierMode::TreeMedRans,
    ] {
        cases.push(
            codec
                .encode_forced(&signal, &identities, 500_000, 24, mode)
                .unwrap(),
        );
    }
    for samples in [1usize, 127, 128, 129, 512] {
        let boundary = common_reference_signal(samples);
        cases.push(
            codec
                .encode_window(&boundary, &identities, 500_000, 24)
                .unwrap(),
        );
    }
    let extreme = vec![
        vec![i32::MIN as i64, i32::MAX as i64, i32::MIN as i64],
        vec![i32::MAX as i64, i32::MIN as i64, i32::MAX as i64],
        vec![0, i32::MIN as i64, i32::MAX as i64],
        vec![-1, i32::MAX as i64, i32::MIN as i64],
    ];
    cases.push(
        codec
            .encode_forced(
                &extreme,
                &identities,
                500_000,
                32,
                Dix2CarrierMode::TreeMedRans,
            )
            .unwrap(),
    );
    let mixed = mixed_escape_signal();
    let mixed_packet = codec
        .encode_window(&mixed, &identities, 500_000, 32)
        .unwrap();
    let mixed_modes = codec.decode_window(&mixed_packet).unwrap().tile_modes;
    assert!(matches!(
        mixed_modes[0],
        Dix2CarrierMode::Raw | Dix2CarrierMode::Delta
    ));
    assert!(mixed_modes[1..].contains(&Dix2CarrierMode::TreeMedRans));
    cases.push(mixed_packet);

    // Mixed referential/bipolar labels carry frozen DIX1 topology priors even
    // while incidence prediction is disabled. This catches an oracle that
    // incorrectly starts the temporal state machine with empty supports.
    let mixed_montage_identities = vec![
        identity(0, "F3-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-C3"),
        identity(3, "ECG"),
    ];
    let mixed_montage = common_reference_signal(257);
    for mode in [Dix2CarrierMode::TemporalRans, Dix2CarrierMode::TreeMedRans] {
        cases.push(
            codec
                .encode_forced(&mixed_montage, &mixed_montage_identities, 500_000, 24, mode)
                .unwrap(),
        );
    }

    for packet in cases {
        let production = codec.decode_window(&packet).expect("Rust decode");
        let reference = reference_decode(&packet).expect("Python decode");
        assert_eq!(
            reference["samples"],
            serde_json::to_value(production.samples).unwrap()
        );
        assert_eq!(
            reference["tile_modes"],
            serde_json::to_value(
                production
                    .tile_modes
                    .iter()
                    .map(|mode| *mode as u8)
                    .collect::<Vec<_>>()
            )
            .unwrap()
        );
        assert_eq!(reference["event_count"], production.event_count);
    }
}

#[test]
fn standalone_decoder_matches_rust_on_deterministic_packet_mutations() {
    let codec = Dix2ConstructionCodec;
    let packet = codec
        .encode_window(&common_reference_signal(257), &identities(), 500_000, 24)
        .unwrap();
    reference_decode(&packet).expect("valid packet");
    assert!(reference_decode(&packet[..packet.len() - 1]).is_err());

    let mut state = 0xd1c2_f022_5eed_cafeu64;
    let mut compared = 0usize;
    for case in 0..96 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let index = state as usize % packet.len();
        if (83..87).contains(&index) {
            continue;
        }
        let mut mutated = packet.clone();
        mutated[index] ^= 1 << ((state >> 32) & 7);
        refresh_packet_crc(&mut mutated);
        let production = codec.decode_window(&mutated);
        let reference = reference_decode(&mutated);
        assert_eq!(
            production.is_ok(),
            reference.is_ok(),
            "acceptance mismatch case {case} byte {index}: Rust={production:?} Python={reference:?}"
        );
        if let (Ok(production), Ok(reference)) = (production, reference) {
            assert_eq!(
                reference["samples"],
                serde_json::to_value(production.samples).unwrap()
            );
            assert_eq!(reference["event_count"], production.event_count);
        }
        compared += 1;
    }
    assert!(compared >= 90);
}

fn reference_decode(packet: &[u8]) -> Result<Value, String> {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let script = format!(
        "{}/tests/reference/dix2_v1_decoder.py",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut child = Command::new(&python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {python}: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "reference decoder stdin is unavailable".to_owned())?
        .write_all(packet)
        .map_err(|error| format!("cannot write packet: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("cannot wait for decoder: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid reference JSON: {error}"))
}

fn common_reference_signal(samples: usize) -> Vec<Vec<i64>> {
    let mut signal = (0..4)
        .map(|_| Vec::with_capacity(samples))
        .collect::<Vec<_>>();
    for time in 0..samples {
        let common = ((time as i64 * 97) % 4096) - 2048;
        signal[0].push(common + (time as i64 % 5));
        signal[1].push(common - (time as i64 % 7));
        signal[2].push(common + (time as i64 % 3));
        signal[3].push(common - (time as i64 % 11));
    }
    signal
}

fn mixed_escape_signal() -> Vec<Vec<i64>> {
    let mut signal = common_reference_signal(512);
    let mut state = 0x7a3c_19d2_u64;
    for sample in 0..128 {
        for channel in &mut signal {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            channel[sample] = i64::from(state as u32 as i32);
        }
    }
    signal
}

fn refresh_packet_crc(packet: &mut [u8]) {
    packet[83..87].fill(0);
    let crc = crc32c(packet);
    packet[83..87].copy_from_slice(&crc.to_le_bytes());
}

fn crc32c(data: &[u8]) -> u32 {
    let mut state = !0u32;
    for &byte in data {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0x82f6_3b78 & 0u32.wrapping_sub(state & 1));
        }
    }
    state ^ !0u32
}
