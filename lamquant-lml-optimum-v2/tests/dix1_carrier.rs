//! Complete-byte construction gates for the experimental DIX1 carrier.

use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix1_carrier::{Dix1CarrierMode, Dix1ConstructionCodec};
use sha2::{Digest, Sha256};

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
fn every_forced_mode_is_self_decoding_repeatable_and_fully_accounted() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let signal = incidence_signal(512);
    let values = signal.len() * signal[0].len();
    let mut sizes = Vec::new();
    for mode in [
        Dix1CarrierMode::Raw,
        Dix1CarrierMode::Delta,
        Dix1CarrierMode::IncidenceRans,
        Dix1CarrierMode::NoIncidenceRans,
    ] {
        let first = codec
            .encode_forced(&signal, &identities, 500_000, 24, mode)
            .expect("forced construction packet");
        let second = codec
            .encode_forced(&signal, &identities, 500_000, 24, mode)
            .expect("repeat construction packet");
        assert_eq!(first, second);
        let decoded = codec.decode_window(&first).expect("self decode");
        assert_eq!(decoded.samples, signal);
        assert_eq!(decoded.identities, identities);
        assert_eq!(decoded.sample_rate_mhz, 500_000);
        assert_eq!(decoded.bit_depth, 24);
        assert_eq!(decoded.mode, mode);
        if matches!(
            mode,
            Dix1CarrierMode::IncidenceRans | Dix1CarrierMode::NoIncidenceRans
        ) {
            assert!(decoded.event_count as usize >= values);
        } else {
            assert_eq!(decoded.event_count, 0);
        }
        let (identity_len, topology_len, directory_len, payload_len) = packet_sections(&first);
        assert_eq!(identity_len, 32);
        assert_eq!(topology_len, 8);
        assert_eq!(directory_len, 24);
        assert_eq!(
            first.len(),
            87 + identity_len + topology_len + directory_len + payload_len
        );
        assert_eq!(first.len() - payload_len, 151);
        let (expected_len, expected_sha256) = match mode {
            Dix1CarrierMode::Raw => (
                8_343,
                "add9052bd22f1463cd3496ed8b6ca3d27cc1cda1868839a09e408d5158200de4",
            ),
            Dix1CarrierMode::Delta => (
                6_201,
                "a9bde46ccdd6006bf6a82f9fc881555e4e6086d1c7412ab9be432524a5e4c500",
            ),
            Dix1CarrierMode::IncidenceRans => (
                4_874,
                "deebd7d8b041e8158bc8de8bf279e0a6fbcac63789930a257690c1333745b09c",
            ),
            Dix1CarrierMode::NoIncidenceRans => (
                5_083,
                "998da6ee6a9a7f597be568fe7a7edd8a64e7e427e58a47645b4a3e00687eb8d3",
            ),
        };
        assert_eq!(first.len(), expected_len);
        assert_eq!(format!("{:x}", Sha256::digest(&first)), expected_sha256);
        println!(
            "DIX1 construction packet golden: mode={mode:?} length={} sha256={:x}",
            first.len(),
            Sha256::digest(&first)
        );
        sizes.push((mode, first.len()));
    }

    let incidence = sizes
        .iter()
        .find(|(mode, _)| *mode == Dix1CarrierMode::IncidenceRans)
        .unwrap()
        .1;
    let no_incidence = sizes
        .iter()
        .find(|(mode, _)| *mode == Dix1CarrierMode::NoIncidenceRans)
        .unwrap()
        .1;
    println!(
        "DIX1 complete-byte synthetic carrier: incidence={} no_incidence={} saving={:.6}% modes={sizes:?}",
        incidence,
        no_incidence,
        100.0 * (no_incidence - incidence) as f64 / no_incidence as f64
    );
    assert!(
        incidence < no_incidence,
        "same-topology incidence carrier must beat its disabled control: {incidence} vs {no_incidence}"
    );
}

#[test]
fn product_and_native_escape_select_the_smallest_complete_packet() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let signal = incidence_signal(512);
    let raw = codec
        .encode_forced(&signal, &identities, 500_000, 24, Dix1CarrierMode::Raw)
        .expect("raw");
    let delta = codec
        .encode_forced(&signal, &identities, 500_000, 24, Dix1CarrierMode::Delta)
        .expect("delta");
    let incidence = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .expect("incidence");
    let product = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("product");
    let native = codec
        .encode_native_window(&signal, &identities, 500_000, 24)
        .expect("native");
    assert_eq!(
        product.len(),
        raw.len().min(delta.len()).min(incidence.len())
    );
    assert_eq!(native.len(), raw.len().min(delta.len()));
    let decoded_product = codec.decode_window(&product).unwrap();
    let decoded_native = codec.decode_window(&native).unwrap();
    assert_eq!(decoded_product.samples, signal);
    assert_eq!(decoded_native.samples, signal);
    assert_eq!(decoded_product.mode, Dix1CarrierMode::IncidenceRans);
    assert_eq!(decoded_native.mode, Dix1CarrierMode::Delta);

    let random = independent_signal(512);
    let random_raw = codec
        .encode_forced(&random, &identities, 500_000, 24, Dix1CarrierMode::Raw)
        .unwrap();
    let random_delta = codec
        .encode_forced(&random, &identities, 500_000, 24, Dix1CarrierMode::Delta)
        .unwrap();
    let random_native = codec
        .encode_native_window(&random, &identities, 500_000, 24)
        .unwrap();
    let random_product = codec
        .encode_window(&random, &identities, 500_000, 24)
        .unwrap();
    assert_eq!(
        random_native.len(),
        random_raw.len().min(random_delta.len())
    );
    assert!(random_product.len() <= random_native.len());
}

#[test]
fn incidence_disabled_is_an_exact_payload_control_without_derivation_supports() {
    let codec = Dix1ConstructionCodec;
    let identities = vec![
        identity(0, "ECG"),
        identity(1, "RESP"),
        identity(2, "EMG"),
        identity(3, "EOG"),
    ];
    let signal = independent_signal(256);
    let incidence = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    let no_incidence = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::NoIncidenceRans,
        )
        .unwrap();

    assert_eq!(incidence.len(), no_incidence.len());
    assert_eq!(packet_payload(&incidence), packet_payload(&no_incidence));
    assert_eq!(
        packet_event_count(&incidence),
        packet_event_count(&no_incidence)
    );
    assert_eq!(codec.decode_window(&incidence).unwrap().samples, signal);
    assert_eq!(codec.decode_window(&no_incidence).unwrap().samples, signal);
}

#[test]
fn packets_are_invariant_when_stable_identities_move_with_channels() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let signal = incidence_signal(256);
    let permutation = [3usize, 2, 0, 1];
    let permuted_identities: Vec<_> = permutation
        .iter()
        .map(|&index| identities[index].clone())
        .collect();
    let permuted_signal: Vec<_> = permutation
        .iter()
        .map(|&index| signal[index].clone())
        .collect();
    for mode in [
        Dix1CarrierMode::Raw,
        Dix1CarrierMode::Delta,
        Dix1CarrierMode::IncidenceRans,
        Dix1CarrierMode::NoIncidenceRans,
    ] {
        let canonical = codec
            .encode_forced(&signal, &identities, 500_000, 24, mode)
            .unwrap();
        let permuted = codec
            .encode_forced(&permuted_signal, &permuted_identities, 500_000, 24, mode)
            .unwrap();
        assert_eq!(canonical, permuted);
    }
}

#[test]
fn packet_corruption_topology_tampering_and_truncation_fail_closed() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let signal = incidence_signal(64);
    let packet = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();

    for &cut in &[0usize, 3, 7, 86, 87, packet.len() - 2, packet.len() - 1] {
        assert!(codec.decode_window(&packet[..cut]).is_err(), "cut {cut}");
    }
    let mut corrupted = packet.clone();
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 0x40;
    assert!(codec.decode_window(&corrupted).is_err());

    let identity_len = u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize;
    let topology_len = u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize;
    let topology_start = 87 + identity_len;
    let mut topology_tamper = packet.clone();
    topology_tamper[topology_start] ^= 1;
    refresh_packet_crc(&mut topology_tamper);
    let error = codec
        .decode_window(&topology_tamper)
        .expect_err("semantic topology tamper");
    assert!(error.to_string().contains("topology"));

    let directory_start = topology_start + topology_len;
    let mut count_tamper = packet.clone();
    count_tamper[directory_start + 20..directory_start + 24].copy_from_slice(&1u32.to_le_bytes());
    refresh_packet_crc(&mut count_tamper);
    assert!(codec.decode_window(&count_tamper).is_err());

    let mut payload_tamper = packet.clone();
    let payload = packet_payload_range(&payload_tamper);
    payload_tamper[payload.start + payload.len() / 2] ^= 0x20;
    refresh_packet_crc(&mut payload_tamper);
    let error = codec
        .decode_window(&payload_tamper)
        .expect_err("rANS payload tamper with valid packet CRC");
    assert!(!error.to_string().contains("packet CRC32C mismatch"));

    let mut trailed = packet;
    trailed.push(0);
    refresh_packet_crc(&mut trailed);
    assert!(codec.decode_window(&trailed).is_err());
}

#[test]
fn carrier_rejects_noncanonical_identities_labels_shapes_and_samples() {
    let codec = Dix1ConstructionCodec;
    let signal = incidence_signal(8);
    let mut bad_ids = identities();
    bad_ids[3].stable_id = 9;
    assert!(codec.encode_window(&signal, &bad_ids, 500_000, 24).is_err());

    let mut bad_label = identities();
    bad_label[0].label = "F3-µV".into();
    assert!(codec
        .encode_window(&signal, &bad_label, 500_000, 24)
        .is_err());
    assert!(codec.encode_window(&signal, &identities(), 0, 24).is_err());
    assert!(codec
        .encode_window(&signal, &identities(), 500_000, 0)
        .is_err());

    let mut bad_shape = signal.clone();
    bad_shape[0].pop();
    assert!(codec
        .encode_window(&bad_shape, &identities(), 500_000, 24)
        .is_err());
    let mut bad_sample = signal;
    bad_sample[0][0] = 1 << 23;
    assert!(codec
        .encode_window(&bad_sample, &identities(), 500_000, 24)
        .is_err());
}

#[test]
fn every_mode_roundtrips_signed_32_bit_boundaries_and_extreme_deltas() {
    let codec = Dix1ConstructionCodec;
    let signals = [
        vec![
            vec![i64::from(i32::MIN)],
            vec![i64::from(i32::MAX)],
            vec![0],
            vec![-1],
        ],
        vec![
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
        ],
    ];
    for signal in signals {
        for mode in [
            Dix1CarrierMode::Raw,
            Dix1CarrierMode::Delta,
            Dix1CarrierMode::IncidenceRans,
            Dix1CarrierMode::NoIncidenceRans,
        ] {
            let packet = codec
                .encode_forced(&signal, &identities(), 4_000_000, 32, mode)
                .unwrap();
            let decoded = codec.decode_window(&packet).unwrap();
            assert_eq!(decoded.samples, signal);
            assert_eq!(decoded.bit_depth, 32);
            assert_eq!(decoded.sample_rate_mhz, 4_000_000);
        }
    }
}

#[test]
fn exact_ascii_labels_survive_normalization_and_input_permutation() {
    let codec = Dix1ConstructionCodec;
    let identities = vec![
        identity(0, "EEG F3-REF."),
        identity(1, "  C3-REF  "),
        identity(2, "F3-C3"),
        identity(3, "ECG lead"),
    ];
    let signal = incidence_signal(32);
    let packet = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    let decoded = codec.decode_window(&packet).unwrap();
    assert_eq!(decoded.identities, identities);

    let permutation = [2usize, 0, 3, 1];
    let permuted_identities: Vec<_> = permutation
        .iter()
        .map(|&index| identities[index].clone())
        .collect();
    let permuted_signal: Vec<_> = permutation
        .iter()
        .map(|&index| signal[index].clone())
        .collect();
    assert_eq!(
        codec
            .encode_forced(
                &permuted_signal,
                &permuted_identities,
                500_000,
                24,
                Dix1CarrierMode::IncidenceRans,
            )
            .unwrap(),
        packet
    );
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
        let row = [f3, c3, f3 - c3, ecg];
        for (channel, sample) in signal.iter_mut().zip(row) {
            channel.push(sample);
        }
    }
    signal
}

fn independent_signal(count: usize) -> Vec<Vec<i64>> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut signal = (0..4)
        .map(|_| Vec::with_capacity(count))
        .collect::<Vec<_>>();
    for _ in 0..count {
        for channel in &mut signal {
            channel.push(bounded_noise(&mut state, 8_000_000));
        }
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

fn refresh_packet_crc(packet: &mut [u8]) {
    const OFFSET: usize = 83;
    packet[OFFSET..OFFSET + 4].fill(0);
    let crc = crc32c(packet);
    packet[OFFSET..OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

fn packet_sections(packet: &[u8]) -> (usize, usize, usize, usize) {
    (
        u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize,
        u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize,
        u32::from_le_bytes(packet[71..75].try_into().unwrap()) as usize,
        u32::from_le_bytes(packet[75..79].try_into().unwrap()) as usize,
    )
}

fn packet_payload_range(packet: &[u8]) -> std::ops::Range<usize> {
    let (identity_len, topology_len, directory_len, payload_len) = packet_sections(packet);
    let start = 87 + identity_len + topology_len + directory_len;
    start..start + payload_len
}

fn packet_payload(packet: &[u8]) -> &[u8] {
    &packet[packet_payload_range(packet)]
}

fn packet_event_count(packet: &[u8]) -> u32 {
    let (identity_len, topology_len, _, _) = packet_sections(packet);
    let directory_start = 87 + identity_len + topology_len;
    u32::from_le_bytes(
        packet[directory_start + 20..directory_start + 24]
            .try_into()
            .unwrap(),
    )
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut state = !0u32;
    for &byte in bytes {
        state ^= u32::from(byte);
        for _ in 0..8 {
            state = (state >> 1) ^ (0x82F6_3B78 & (0u32.wrapping_sub(state & 1)));
        }
    }
    state ^ !0u32
}
