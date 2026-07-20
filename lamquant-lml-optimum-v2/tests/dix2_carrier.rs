//! Complete-byte packet seam for the construction-private DIX2 carrier.

use lamquant_lml_optimum_v2::derivation_incidence::ChannelIdentity;
use lamquant_lml_optimum_v2::dix1_carrier::{Dix1CarrierMode, Dix1ConstructionCodec};
use lamquant_lml_optimum_v2::dix2_carrier::{Dix2CarrierMode, Dix2ConstructionCodec};
use sha2::{Digest, Sha256};

fn identity(stable_id: u16, label: &str) -> ChannelIdentity {
    ChannelIdentity::new(stable_id, label)
}

#[test]
fn dix2_packet_accounts_for_every_byte_and_preserves_the_dix1_control() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
    let signal = common_reference_signal(512);
    let codec = Dix2ConstructionCodec;
    let temporal = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix2CarrierMode::TemporalRans,
        )
        .expect("DIX2 temporal packet");
    let tree = codec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix2CarrierMode::TreeMedRans,
        )
        .expect("DIX2 TreeMED packet");
    let dix1 = Dix1ConstructionCodec
        .encode_forced(
            &signal,
            &identities,
            500_000,
            24,
            Dix1CarrierMode::NoIncidenceRans,
        )
        .expect("frozen DIX1 temporal packet");

    assert_eq!(&temporal[..7], b"LMO1\x03\x00\x03");
    assert_eq!(&temporal[7..12], b"DIX2\x01");
    assert_eq!(packet_payload(&temporal), packet_payload(&dix1));
    assert!(tree.len() < temporal.len());
    assert_eq!(
        codec.decode_window(&tree).expect("TreeMED decode").samples,
        signal
    );
    assert_eq!(
        codec
            .decode_window(&temporal)
            .expect("temporal decode")
            .tile_modes,
        vec![Dix2CarrierMode::TemporalRans; 4]
    );
    assert_eq!(section_total(&tree), tree.len());
    println!(
        "DIX2 packet synthetic: temporal={} TreeMED={} saving={:.6}% tree_sha256={:x}",
        temporal.len(),
        tree.len(),
        100.0 * (temporal.len() - tree.len()) as f64 / temporal.len() as f64,
        Sha256::digest(&tree)
    );
    assert_eq!(
        tree,
        codec
            .encode_forced(
                &signal,
                &identities,
                500_000,
                24,
                Dix2CarrierMode::TreeMedRans,
            )
            .expect("repeat TreeMED packet")
    );
}

#[test]
fn all_profiles_roundtrip_and_product_selects_exact_complete_bytes() {
    let identities = vec![
        identity(0, "FP1-REF"),
        identity(1, "C3-REF"),
        identity(2, "F3-REF"),
        identity(3, "F4-REF"),
    ];
    let signal = common_reference_signal(512);
    let codec = Dix2ConstructionCodec;
    let modes = [
        Dix2CarrierMode::Raw,
        Dix2CarrierMode::Delta,
        Dix2CarrierMode::TemporalRans,
        Dix2CarrierMode::TreeMedRans,
    ];
    let forced = modes
        .iter()
        .map(|&mode| {
            let packet = codec
                .encode_forced(&signal, &identities, 500_000, 24, mode)
                .expect("forced packet");
            let decoded = codec.decode_window(&packet).expect("forced decode");
            assert_eq!(decoded.samples, signal);
            assert_eq!(decoded.identities, identities);
            assert_eq!(decoded.tile_modes, vec![mode; 4]);
            println!(
                "DIX2 forced golden: mode={mode:?} len={} sha256={:x}",
                packet.len(),
                Sha256::digest(&packet)
            );
            let (expected_len, expected_sha256) = match mode {
                Dix2CarrierMode::Raw => (
                    8_352,
                    "649e5073c392a9268f9f88a2124ede3875c9bf6dc310c37402c0b46407f749a5",
                ),
                Dix2CarrierMode::Delta => (
                    4_256,
                    "199a6f6cf3c43bf73aeeba0f58ad16bd218906286ac4390871524e09055139d9",
                ),
                Dix2CarrierMode::TemporalRans => (
                    2_605,
                    "333289bfebbf2422ca6bd6df2115fa8eb96add315874334060e942352fe64ac3",
                ),
                Dix2CarrierMode::TreeMedRans => (
                    1_371,
                    "2668126c3ff3f94c23bcc91fd5d68cdcd1ad3495661676f00618796aa7a15a1a",
                ),
            };
            assert_eq!(packet.len(), expected_len);
            assert_eq!(format!("{:x}", Sha256::digest(&packet)), expected_sha256);
            (mode, packet)
        })
        .collect::<Vec<_>>();
    let product = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("product");
    let native = codec
        .encode_native_window(&signal, &identities, 500_000, 24)
        .expect("native");
    println!(
        "DIX2 product golden: len={} sha256={:x}",
        product.len(),
        Sha256::digest(&product)
    );
    println!(
        "DIX2 native golden: len={} sha256={:x}",
        native.len(),
        Sha256::digest(&native)
    );
    assert_eq!(product.len(), 1_371);
    assert_eq!(
        format!("{:x}", Sha256::digest(&product)),
        "960aae8e28178846f7f2b905d15c48bbc9e5f027c09e24710537fbce031707e3"
    );
    assert_eq!(native.len(), 4_256);
    assert_eq!(
        format!("{:x}", Sha256::digest(&native)),
        "bbb649ee39ac694d88882de5a02d0ed3c177c2bfb3f19906dd342713ccee6243"
    );
    let smallest = forced.iter().map(|(_, packet)| packet.len()).min().unwrap();
    let native_smallest = forced[..2]
        .iter()
        .map(|(_, packet)| packet.len())
        .min()
        .unwrap();

    assert_eq!(product.len(), smallest);
    assert_eq!(native.len(), native_smallest);
    assert_eq!(
        codec.decode_window(&product).unwrap().tile_modes,
        vec![Dix2CarrierMode::TreeMedRans; 4]
    );
    assert!(codec
        .decode_window(&native)
        .unwrap()
        .tile_modes
        .iter()
        .all(|mode| matches!(mode, Dix2CarrierMode::Raw | Dix2CarrierMode::Delta)));
}

#[test]
fn packet_is_invariant_when_identities_and_channels_move_together() {
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
    let codec = Dix2ConstructionCodec;

    assert_eq!(
        codec
            .encode_window(&signal, &identities, 500_000, 24)
            .unwrap(),
        codec
            .encode_window(&permuted_signal, &permuted_identities, 500_000, 24)
            .unwrap()
    );
}

#[test]
fn packet_corruption_and_noncanonical_profiles_fail_closed() {
    let identities = vec![identity(0, "C3-REF"), identity(1, "F3-REF")];
    let signal = common_reference_signal(129)[..2].to_vec();
    let codec = Dix2ConstructionCodec;
    let packet = codec
        .encode_window(&signal, &identities, 500_000, 24)
        .expect("packet");

    for offset in [
        0usize,
        7,
        12,
        14,
        63,
        67,
        71,
        75,
        79,
        83,
        87,
        packet.len() - 1,
    ] {
        let mut corrupt = packet.clone();
        corrupt[offset] ^= 1;
        assert!(
            codec.decode_window(&corrupt).is_err(),
            "mutation at {offset} must fail"
        );
    }
    for end in [0usize, 6, 11, 86, packet.len() - 1] {
        assert!(codec.decode_window(&packet[..end]).is_err());
    }
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

fn section_total(packet: &[u8]) -> usize {
    87 + u32::from_le_bytes(packet[63..67].try_into().unwrap()) as usize
        + u32::from_le_bytes(packet[67..71].try_into().unwrap()) as usize
        + u32::from_le_bytes(packet[71..75].try_into().unwrap()) as usize
        + u32::from_le_bytes(packet[75..79].try_into().unwrap()) as usize
}

fn packet_payload(packet: &[u8]) -> &[u8] {
    let payload_len = u32::from_le_bytes(packet[75..79].try_into().unwrap()) as usize;
    &packet[packet.len() - payload_len..]
}
