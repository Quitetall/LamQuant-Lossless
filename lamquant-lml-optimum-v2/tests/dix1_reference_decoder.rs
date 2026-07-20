//! Cross-language differential gate for the construction-private DIX1 v2 carrier.

use std::io::Write;
use std::process::{Command, Stdio};

use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix1_carrier::{Dix1CarrierMode, Dix1ConstructionCodec};
use serde_json::Value;

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

fn identities() -> Vec<ChannelIdentity> {
    vec![
        identity(0, "F3-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-C3"),
        identity(3, "ECG"),
    ]
}

#[test]
fn standalone_python_decoder_matches_all_block_profiles_and_boundaries() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let boundary_signal = incidence_signal(257);
    let mut cases = vec![
        (
            "raw",
            codec
                .encode_forced(
                    &boundary_signal,
                    &identities,
                    500_000,
                    24,
                    Dix1CarrierMode::Raw,
                )
                .unwrap(),
            boundary_signal.clone(),
        ),
        (
            "native",
            codec
                .encode_native_window(&boundary_signal, &identities, 500_000, 24)
                .unwrap(),
            boundary_signal.clone(),
        ),
        (
            "delta",
            codec
                .encode_forced(
                    &boundary_signal,
                    &identities,
                    500_000,
                    24,
                    Dix1CarrierMode::Delta,
                )
                .unwrap(),
            boundary_signal.clone(),
        ),
        (
            "incidence",
            codec
                .encode_forced(
                    &boundary_signal,
                    &identities,
                    500_000,
                    24,
                    Dix1CarrierMode::IncidenceRans,
                )
                .unwrap(),
            boundary_signal.clone(),
        ),
        (
            "no-incidence",
            codec
                .encode_forced(
                    &boundary_signal,
                    &identities,
                    500_000,
                    24,
                    Dix1CarrierMode::NoIncidenceRans,
                )
                .unwrap(),
            boundary_signal,
        ),
    ];
    for samples in [1usize, 127, 128, 129, 512] {
        let signal = incidence_signal(samples);
        cases.push((
            "product-boundary",
            codec
                .encode_window(&signal, &identities, 500_000, 24)
                .unwrap(),
            signal,
        ));
    }
    let extreme_signal = vec![
        vec![
            i64::from(i32::MIN),
            i64::from(i32::MAX),
            i64::from(i32::MIN),
        ],
        vec![
            i64::from(i32::MAX),
            i64::from(i32::MIN),
            i64::from(i32::MAX),
        ],
        vec![0, i64::from(i32::MIN), i64::from(i32::MAX)],
        vec![-1, i64::from(i32::MAX), i64::from(i32::MIN)],
    ];
    cases.push((
        "extreme-incidence",
        codec
            .encode_forced(
                &extreme_signal,
                &identities,
                500_000,
                32,
                Dix1CarrierMode::IncidenceRans,
            )
            .unwrap(),
        extreme_signal,
    ));
    let mixed_signal = raw_escape_then_zero_signal();
    let mixed_packet = codec
        .encode_window(&mixed_signal, &identities, 500_000, 32)
        .unwrap();
    let mixed_modes = packet_modes(&mixed_packet);
    assert_eq!(mixed_modes[0], Dix1CarrierMode::Raw as u8);
    assert!(mixed_modes[1..].contains(&(Dix1CarrierMode::IncidenceRans as u8)));
    cases.push(("mixed-product", mixed_packet, mixed_signal));

    for (label, packet, expected_signal) in cases {
        let reference = reference_decode(&packet)
            .unwrap_or_else(|error| panic!("{label} reference decode failed: {error}"));
        assert_eq!(
            reference["samples"],
            serde_json::to_value(expected_signal).unwrap()
        );
        assert_eq!(reference["sample_rate_mhz"], 500_000);
        assert_eq!(reference["bit_depth"], u64::from(packet[13]));
        assert_eq!(reference["stable_ids"], serde_json::json!([0, 1, 2, 3]));

        let expected_modes = packet_modes(&packet);
        assert_eq!(
            reference["tile_modes"],
            serde_json::to_value(expected_modes).unwrap()
        );
        let production = codec.decode_window(&packet).unwrap();
        assert_eq!(reference["event_count"], production.event_count);
    }
}

#[test]
fn standalone_python_decoder_rejects_corrupt_or_trailed_packets() {
    let codec = Dix1ConstructionCodec;
    let packet = codec
        .encode_forced(
            &incidence_signal(129),
            &identities(),
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();

    reference_decode(&packet).expect("valid reference packet");

    assert!(reference_decode(&packet[..packet.len() - 1]).is_err());

    let mut corrupted = packet.clone();
    let payload = packet_payload_range(&corrupted);
    corrupted[payload.start + payload.len() / 2] ^= 0x20;
    refresh_packet_crc(&mut corrupted);
    assert!(reference_decode(&corrupted).is_err());

    let compressible = incidence_signal(129);
    let mut noncanonical_product = codec
        .encode_forced(
            &compressible,
            &identities(),
            500_000,
            24,
            Dix1CarrierMode::Raw,
        )
        .unwrap();
    noncanonical_product[14] = 0;
    refresh_packet_crc(&mut noncanonical_product);
    assert!(codec.decode_window(&noncanonical_product).is_err());
    assert!(reference_decode(&noncanonical_product).is_err());

    let mut noncanonical_native = noncanonical_product;
    noncanonical_native[14] = 1;
    refresh_packet_crc(&mut noncanonical_native);
    assert!(codec.decode_window(&noncanonical_native).is_err());
    assert!(reference_decode(&noncanonical_native).is_err());

    let mut trailed = packet;
    trailed.push(0);
    refresh_packet_crc(&mut trailed);
    assert!(reference_decode(&trailed).is_err());
}

#[test]
fn standalone_python_decoder_matches_rust_on_deterministic_packet_mutations() {
    let codec = Dix1ConstructionCodec;
    let original = codec
        .encode_window(&incidence_signal(257), &identities(), 500_000, 24)
        .unwrap();
    let mut state = 0xd1c1_f022_5eed_cafeu64;
    let mut compared = 0usize;
    for case in 0..96 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let index = state as usize % original.len();
        if (83..87).contains(&index) {
            continue;
        }
        let mut packet = original.clone();
        packet[index] ^= 1 << ((state >> 32) & 7);
        refresh_packet_crc(&mut packet);

        let production = codec.decode_window(&packet);
        let reference = reference_decode(&packet);
        assert_eq!(
            production.is_ok(),
            reference.is_ok(),
            "acceptance mismatch for mutation {case} at byte {index}: production={production:?} reference={reference:?}"
        );
        if let (Ok(production), Ok(reference)) = (production, reference) {
            assert_eq!(
                reference["samples"],
                serde_json::to_value(production.samples).unwrap()
            );
            assert_eq!(
                reference["stable_ids"],
                serde_json::to_value(
                    production
                        .identities
                        .iter()
                        .map(|identity| identity.stable_id)
                        .collect::<Vec<_>>()
                )
                .unwrap()
            );
            assert_eq!(
                reference["labels"],
                serde_json::to_value(
                    production
                        .identities
                        .iter()
                        .map(|identity| identity.label.as_str())
                        .collect::<Vec<_>>()
                )
                .unwrap()
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
        compared += 1;
    }
    assert!(
        compared >= 90,
        "insufficient deterministic mutation coverage"
    );
}

fn reference_decode(packet: &[u8]) -> Result<Value, String> {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let script = format!(
        "{}/tests/reference/dix1_v2_decoder.py",
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
        .map_err(|error| format!("cannot write reference packet: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("cannot wait for reference decoder: {error}"))?;
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

fn packet_modes(packet: &[u8]) -> Vec<u8> {
    let identity_len = u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize;
    let topology_len = u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize;
    let directory_len = u32::from_le_bytes(packet[71..75].try_into().unwrap()) as usize;
    let start = 87 + identity_len + topology_len;
    packet[start..start + directory_len]
        .chunks_exact(5)
        .map(|entry| entry[0])
        .collect()
}

fn incidence_signal(count: usize) -> Vec<Vec<i64>> {
    let mut state = 0x8b8b_8b8b_1234_5678u64;
    let mut signal = (0..4)
        .map(|_| Vec::with_capacity(count))
        .collect::<Vec<_>>();
    for time in 0..count {
        let common = bounded_noise(&mut state, 180_000);
        let drift = if time < count / 2 {
            time as i64 * 73
        } else {
            -(time as i64) * 91
        };
        let f3 = common + bounded_noise(&mut state, 70_000) + drift;
        let c3 = common + bounded_noise(&mut state, 70_000) - drift / 2;
        let ecg = bounded_noise(&mut state, 300_000);
        for (channel, sample) in signal.iter_mut().zip([f3, c3, f3 - c3, ecg]) {
            channel.push(sample);
        }
    }
    signal
}

fn raw_escape_then_zero_signal() -> Vec<Vec<i64>> {
    let mut state = 0xd1c1_5eed_f00d_cafeu64;
    let mut signal = (0..4).map(|_| Vec::with_capacity(512)).collect::<Vec<_>>();
    for _ in 0..128 {
        for channel in &mut signal {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            channel.push(i64::from((state >> 32) as u32 as i32));
        }
    }
    for channel in &mut signal {
        channel.resize(512, 0);
    }
    signal
}

fn bounded_noise(state: &mut u64, bound: i64) -> i64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let span = (2 * bound + 1) as u64;
    (*state % span) as i64 - bound
}

fn packet_payload_range(packet: &[u8]) -> std::ops::Range<usize> {
    let identity_len = u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize;
    let topology_len = u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize;
    let directory_len = u32::from_le_bytes(packet[71..75].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(packet[75..79].try_into().unwrap()) as usize;
    let start = 87 + identity_len + topology_len + directory_len;
    start..start + payload_len
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
