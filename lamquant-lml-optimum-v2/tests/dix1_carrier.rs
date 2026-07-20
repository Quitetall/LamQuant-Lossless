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
fn construction_v2_uses_canonical_128_row_compact_blocks() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    for samples in [1usize, 127, 128, 129, 512] {
        let signal = incidence_signal(samples);
        let packet = codec
            .encode_forced(&signal, &identities, 500_000, 24, Dix1CarrierMode::Raw)
            .unwrap();
        let block_count = samples.div_ceil(128);
        assert_eq!(packet[11], 2, "body version for {samples} rows");
        assert_eq!(packet[14], 2, "forced-raw profile for {samples} rows");
        assert_eq!(
            usize::from(u16::from_le_bytes(packet[17..19].try_into().unwrap())),
            block_count
        );
        let (_, _, directory_len, payload_len) = packet_sections(&packet);
        assert_eq!(directory_len, block_count * 5);
        assert_eq!(payload_len, identities.len() * samples * 4);

        let directory_start = packet_directory_start(&packet);
        for block in 0..block_count {
            let entry = directory_start + block * 5;
            let rows = (samples - block * 128).min(128);
            assert_eq!(packet[entry], 0);
            assert_eq!(
                u32::from_le_bytes(packet[entry + 1..entry + 5].try_into().unwrap()) as usize,
                identities.len() * rows * 4
            );
        }
        assert_eq!(codec.decode_window(&packet).unwrap().samples, signal);
    }
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
        assert_eq!(decoded.tile_modes, vec![mode; 4]);
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
        assert_eq!(directory_len, 20);
        assert_eq!(
            first.len(),
            87 + identity_len + topology_len + directory_len + payload_len
        );
        assert_eq!(first.len() - payload_len, 147);
        let (expected_len, expected_sha256) = match mode {
            Dix1CarrierMode::Raw => (
                8_339,
                "61f20ae238b52b15d967ac9a4a6860bb8ae22e6ccb2cb35aa4054c0168d4f004",
            ),
            Dix1CarrierMode::Delta => (
                6_198,
                "e6ffb5e911c475b55955be95bc30d9f71c4313393f85a048e4120728c433047e",
            ),
            Dix1CarrierMode::IncidenceRans => (
                5_043,
                "25b611906a0d4d54160f4dabc2f94bb94546b70d4486904a7b79ada7c99d1a70",
            ),
            Dix1CarrierMode::NoIncidenceRans => (
                5_260,
                "b56f75f2f78176a9cd42d811ac01d8a7f6427b8a9228b8df00930ff8b66feb31",
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
    assert!(product.len() <= raw.len().min(delta.len()).min(incidence.len()));
    assert!(native.len() <= raw.len().min(delta.len()));
    let decoded_product = codec.decode_window(&product).unwrap();
    let decoded_native = codec.decode_window(&native).unwrap();
    assert_eq!(decoded_product.samples, signal);
    assert_eq!(decoded_native.samples, signal);
    let expected_product = expected_product_modes(&raw, &delta, &incidence);
    let expected_native = expected_native_modes(&raw, &delta);
    assert_eq!(expected_product, vec![Dix1CarrierMode::IncidenceRans; 4]);
    assert_eq!(expected_native, vec![Dix1CarrierMode::Delta; 4]);
    assert_eq!(decoded_product.tile_modes, expected_product);
    assert_eq!(decoded_native.tile_modes, expected_native);
    assert_eq!(product.len(), 5_043);
    assert_eq!(
        format!("{:x}", Sha256::digest(&product)),
        "748144213a5a95c429a5ceddfacf7e4a40a180d97b3961b726de93e99ce31e8d"
    );
    assert_eq!(native.len(), 6_198);
    assert_eq!(
        format!("{:x}", Sha256::digest(&native)),
        "017b7fc3d385257e6798464dd22d9e83f6beed8f5f4758d1474c61652e4ae150"
    );
    assert_eq!(
        packet_sections(&product).3,
        packet_block_entries(&product)
            .iter()
            .map(|(_, length)| length)
            .sum()
    );

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
    assert!(random_native.len() <= random_raw.len().min(random_delta.len()));
    assert!(random_product.len() <= random_native.len());
    let random_decoded_native = codec.decode_window(&random_native).unwrap();
    let random_decoded_product = codec.decode_window(&random_product).unwrap();
    assert_eq!(
        random_decoded_native.tile_modes,
        expected_native_modes(&random_raw, &random_delta)
    );
    let random_incidence = codec
        .encode_forced(
            &random,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    assert_eq!(
        random_decoded_product.tile_modes,
        expected_product_modes(&random_raw, &random_delta, &random_incidence)
    );
}

#[test]
fn raw_escape_advances_predictor_before_a_later_incidence_block() {
    let codec = Dix1ConstructionCodec;
    let identities = identities();
    let signal = raw_escape_then_zero_signal();
    let raw = codec
        .encode_forced(&signal, &identities, 500_000, 32, Dix1CarrierMode::Raw)
        .unwrap();
    let delta = codec
        .encode_forced(&signal, &identities, 500_000, 32, Dix1CarrierMode::Delta)
        .unwrap();
    let incidence = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            32,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    let product = codec
        .encode_window(&signal, &identities, 500_000, 32)
        .unwrap();
    let expected = expected_product_modes(&raw, &delta, &incidence);
    assert_eq!(expected[0], Dix1CarrierMode::Raw);
    assert!(expected[1..].contains(&Dix1CarrierMode::IncidenceRans));
    let decoded = codec.decode_window(&product).unwrap();
    assert_eq!(decoded.tile_modes, expected);
    assert_eq!(decoded.samples, signal);
}

#[test]
fn product_ties_choose_raw_before_delta() {
    let codec = Dix1ConstructionCodec;
    let identities = vec![identity(0, "ECG")];
    let signal = vec![vec![1 << 20]];
    let raw = codec
        .encode_forced(&signal, &identities, 500_000, 24, Dix1CarrierMode::Raw)
        .unwrap();
    let delta = codec
        .encode_forced(&signal, &identities, 500_000, 24, Dix1CarrierMode::Delta)
        .unwrap();
    let incidence = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    let lengths = [
        packet_sections(&raw).3,
        packet_sections(&delta).3,
        packet_sections(&incidence).3,
    ];
    assert_eq!(lengths, [4, 4, 9]);

    let product = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .unwrap();
    let decoded = codec.decode_window(&product).unwrap();
    assert_eq!(decoded.tile_modes, [Dix1CarrierMode::Raw]);
    assert_eq!(decoded.samples, signal);
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
    let decoded_incidence = codec.decode_window(&incidence).unwrap();
    let decoded_no_incidence = codec.decode_window(&no_incidence).unwrap();
    assert_eq!(
        decoded_incidence.event_count,
        decoded_no_incidence.event_count
    );
    assert_eq!(decoded_incidence.samples, signal);
    assert_eq!(decoded_no_incidence.samples, signal);
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
    assert_eq!(
        codec
            .encode_window(&signal, &identities, 500_000, 24)
            .unwrap(),
        codec
            .encode_window(&permuted_signal, &permuted_identities, 500_000, 24)
            .unwrap()
    );
    assert_eq!(
        codec
            .encode_native_window(&signal, &identities, 500_000, 24)
            .unwrap(),
        codec
            .encode_native_window(&permuted_signal, &permuted_identities, 500_000, 24,)
            .unwrap()
    );
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
    let mut mode_tamper = packet.clone();
    mode_tamper[directory_start] = 0;
    refresh_packet_crc(&mut mode_tamper);
    let error = codec
        .decode_window(&mode_tamper)
        .expect_err("forced-incidence profile rejects raw block");
    assert!(error.to_string().contains("profile"));

    let mut length_tamper = packet.clone();
    let length = u32::from_le_bytes(
        length_tamper[directory_start + 1..directory_start + 5]
            .try_into()
            .unwrap(),
    );
    length_tamper[directory_start + 1..directory_start + 5]
        .copy_from_slice(&(length + 1).to_le_bytes());
    refresh_packet_crc(&mut length_tamper);
    let error = codec
        .decode_window(&length_tamper)
        .expect_err("compact directory length tamper");
    assert!(error.to_string().contains("cover the payload"));

    let mut redistributed = codec
        .encode_forced(
            &incidence_signal(256),
            &identities,
            500_000,
            24,
            Dix1CarrierMode::IncidenceRans,
        )
        .unwrap();
    let directory_start = packet_directory_start(&redistributed);
    let first_len = u32::from_le_bytes(
        redistributed[directory_start + 1..directory_start + 5]
            .try_into()
            .unwrap(),
    );
    let second_len = u32::from_le_bytes(
        redistributed[directory_start + 6..directory_start + 10]
            .try_into()
            .unwrap(),
    );
    redistributed[directory_start + 1..directory_start + 5]
        .copy_from_slice(&(first_len + 1).to_le_bytes());
    redistributed[directory_start + 6..directory_start + 10]
        .copy_from_slice(&(second_len - 1).to_le_bytes());
    refresh_packet_crc(&mut redistributed);
    assert!(codec.decode_window(&redistributed).is_err());

    let mut profile_tamper = packet.clone();
    profile_tamper[14] = 6;
    refresh_packet_crc(&mut profile_tamper);
    assert!(codec.decode_window(&profile_tamper).is_err());

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
fn refreshed_crc_single_bit_mutations_never_panic_or_decode_noncanonically() {
    let codec = Dix1ConstructionCodec;
    let packet = codec
        .encode_window(&incidence_signal(512), &identities(), 500_000, 24)
        .unwrap();
    for index in (0..packet.len()).step_by(17) {
        let mut mutated = packet.clone();
        mutated[index] ^= 1u8 << (index % 8);
        refresh_packet_crc(&mut mutated);
        if codec.decode_window(&mutated).is_ok() {
            assert_eq!(mutated, packet, "accepted mutation at byte {index}");
        }
    }
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

fn packet_directory_start(packet: &[u8]) -> usize {
    let (identity_len, topology_len, _, _) = packet_sections(packet);
    87 + identity_len + topology_len
}

fn packet_payload(packet: &[u8]) -> &[u8] {
    &packet[packet_payload_range(packet)]
}

fn packet_block_entries(packet: &[u8]) -> Vec<(u8, usize)> {
    let (_, _, directory_len, _) = packet_sections(packet);
    assert_eq!(directory_len % 5, 0);
    let directory_start = packet_directory_start(packet);
    (0..directory_len / 5)
        .map(|block| {
            let entry = directory_start + block * 5;
            (
                packet[entry],
                u32::from_le_bytes(packet[entry + 1..entry + 5].try_into().unwrap()) as usize,
            )
        })
        .collect()
}

fn expected_product_modes(raw: &[u8], delta: &[u8], incidence: &[u8]) -> Vec<Dix1CarrierMode> {
    packet_block_entries(raw)
        .into_iter()
        .zip(packet_block_entries(delta))
        .zip(packet_block_entries(incidence))
        .map(
            |(((raw_mode, raw_len), (delta_mode, delta_len)), (incidence_mode, incidence_len))| {
                [
                    (raw_len, raw_mode, Dix1CarrierMode::Raw),
                    (delta_len, delta_mode, Dix1CarrierMode::Delta),
                    (
                        incidence_len,
                        incidence_mode,
                        Dix1CarrierMode::IncidenceRans,
                    ),
                ]
                .into_iter()
                .min_by_key(|(length, mode, _)| (*length, *mode))
                .unwrap()
                .2
            },
        )
        .collect()
}

fn expected_native_modes(raw: &[u8], delta: &[u8]) -> Vec<Dix1CarrierMode> {
    packet_block_entries(raw)
        .into_iter()
        .zip(packet_block_entries(delta))
        .map(|((raw_mode, raw_len), (delta_mode, delta_len))| {
            [
                (raw_len, raw_mode, Dix1CarrierMode::Raw),
                (delta_len, delta_mode, Dix1CarrierMode::Delta),
            ]
            .into_iter()
            .min_by_key(|(length, mode, _)| (*length, *mode))
            .unwrap()
            .2
        })
        .collect()
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
