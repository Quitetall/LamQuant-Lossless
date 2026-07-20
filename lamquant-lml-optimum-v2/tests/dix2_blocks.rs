//! Actual-byte construction seam for DIX2 TreeMED blocks.

use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix1_carrier::{Dix1CarrierMode, Dix1ConstructionCodec};
use lamquant_lml_optimum_v2::dix2_blocks::{Dix2BlockCodec, Dix2CarrierMode};

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

#[test]
fn tree_med_rans_beats_a_byte_identical_dix1_temporal_control() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
    let signal = common_reference_signal(512);
    let dix1 = Dix1ConstructionCodec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::NoIncidenceRans,
        )
        .expect("frozen DIX1 control");
    let codec = Dix2BlockCodec;
    let temporal = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix2CarrierMode::TemporalRans,
        )
        .expect("DIX2 temporal control");
    let tree = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix2CarrierMode::TreeMedRans,
        )
        .expect("DIX2 TreeMED candidate");

    assert_eq!(temporal.payload, dix1_payload(&dix1));
    assert!(
        tree.payload.len() < temporal.payload.len(),
        "TreeMED actual rANS must beat its exact DIX1 temporal control: {} vs {}",
        tree.payload.len(),
        temporal.payload.len()
    );
    println!(
        "DIX2 synthetic actual-rANS: temporal={} TreeMED={} saving={:.6}%",
        temporal.payload.len(),
        tree.payload.len(),
        100.0 * (temporal.payload.len() - tree.payload.len()) as f64
            / temporal.payload.len() as f64
    );

    let product = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("DIX2 product");
    assert_eq!(product.modes, vec![Dix2CarrierMode::TreeMedRans; 4]);
    assert_eq!(
        codec
            .decode_window(&product, &identities, 500_000, 24, 512)
            .expect("DIX2 decode")
            .samples,
        signal
    );
    assert_eq!(
        product,
        codec
            .encode_window(&signal, &identities, 500_000, 24)
            .expect("repeat DIX2 product")
    );
}

#[test]
fn product_bytes_follow_identity_not_presentation_order() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
    let signal = common_reference_signal(257);
    let permutation = [2usize, 0, 3, 1];
    let permuted_identities = permutation
        .iter()
        .map(|&channel| identities[channel].clone())
        .collect::<Vec<_>>();
    let permuted_signal = permutation
        .iter()
        .map(|&channel| signal[channel].clone())
        .collect::<Vec<_>>();
    let codec = Dix2BlockCodec;
    let canonical = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("canonical product");
    let permuted = codec
        .encode_window(&permuted_signal, &permuted_identities, 500_000, 24)
        .expect("permuted product");

    assert_eq!(canonical, permuted);
    assert_eq!(
        codec
            .decode_window(&permuted, &permuted_identities, 500_000, 24, 257)
            .expect("permuted decode")
            .samples,
        signal
    );
}

#[test]
fn native_escape_advances_temporal_state_before_tree_med_blocks() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
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
    let codec = Dix2BlockCodec;
    let product = codec
        .encode_window(&signal, &identities, 500_000, 32)
        .expect("mixed product");

    assert!(matches!(
        product.modes[0],
        Dix2CarrierMode::Raw | Dix2CarrierMode::Delta
    ));
    assert!(product.modes[1..].contains(&Dix2CarrierMode::TreeMedRans));
    assert_eq!(
        codec
            .decode_window(&product, &identities, 500_000, 32, 512)
            .expect("mixed decode")
            .samples,
        signal
    );
}

#[test]
fn decoder_rejects_noncanonical_directory_metadata() {
    let identities = vec![identity(0, "C3-REF"), identity(1, "F3-REF")];
    let signal = common_reference_signal(129)[..2].to_vec();
    let codec = Dix2BlockCodec;
    let encoded = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("product");

    let mut invalid_mode = encoded.clone();
    invalid_mode.directory[0] = 0xff;
    assert!(codec
        .decode_window(&invalid_mode, &identities, 500_000, 24, 129)
        .is_err());

    let mut invalid_summary = encoded;
    invalid_summary.modes[0] = Dix2CarrierMode::Raw;
    assert!(codec
        .decode_window(&invalid_summary, &identities, 500_000, 24, 129)
        .is_err());
}

#[test]
fn tree_med_guard_bit_roundtrips_declared_32_bit_extremes() {
    let identities = vec![identity(0, "C3-REF"), identity(1, "F3-REF")];
    let signal = vec![
        vec![i32::MIN as i64, i32::MAX as i64, i32::MIN as i64],
        vec![i32::MAX as i64, i32::MIN as i64, i32::MAX as i64],
    ];
    let codec = Dix2BlockCodec;
    let tree = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            32,
            Dix2CarrierMode::TreeMedRans,
        )
        .expect("guard-bit TreeMED");
    assert_eq!(
        codec
            .decode_window(&tree, &identities, 500_000, 32, 3)
            .expect("guard-bit decode")
            .samples,
        signal
    );
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

fn dix1_payload(packet: &[u8]) -> Vec<u8> {
    let identity_len = u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize;
    let topology_len = u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize;
    let directory_len = u32::from_le_bytes(packet[71..75].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(packet[75..79].try_into().unwrap()) as usize;
    let start = 87 + identity_len + topology_len + directory_len;
    assert_eq!(packet.len(), start + payload_len);
    packet[start..].to_vec()
}
