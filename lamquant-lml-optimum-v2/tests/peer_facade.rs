use lamquant_lml_optimum_v2::mix1::Mix1Codec;
use lamquant_lml_optimum_v2::{
    PeerCodec, PeerEncodeContext, PeerPacketKind, PEER_KERNEL_ID, PEER_MAX_CHANNELS,
    PEER_MAX_SAMPLES, PEER_MAX_VALUES,
};

fn fixture() -> Vec<Vec<i64>> {
    vec![
        (0..128)
            .map(|time| i64::from((time * 7 + time / 11) % 257) - 128)
            .collect(),
        (0..128)
            .map(|time| i64::from((time * 5 + time / 13) % 193) - 96)
            .collect(),
    ]
}

#[test]
fn production_facade_is_byte_equal_to_frozen_peer_portfolio() {
    let signal = fixture();
    let context = PeerEncodeContext {
        sample_rate_mhz: 256_000,
        bit_depth: 16,
    };

    let encoded = PeerCodec
        .encode_window_with_report(&signal, context)
        .expect("encode through production facade");
    let direct = Mix1Codec
        .encode_best_peer_window(&signal, context.sample_rate_mhz, context.bit_depth)
        .expect("encode through frozen peer portfolio");

    assert_eq!(encoded.packet, direct);
    assert_eq!(encoded.report.packet_bytes, encoded.packet.len());
    assert_eq!(encoded.report.source_value_count, 256);
    assert_eq!(encoded.report.channel_count, 2);
    assert_eq!(encoded.report.sample_count, 128);
    assert!(matches!(
        encoded.report.selected_kind,
        PeerPacketKind::Mix1
            | PeerPacketKind::Multivariate
            | PeerPacketKind::HierarchicalMultivariate
            | PeerPacketKind::ChannelContext
            | PeerPacketKind::CommonMode
            | PeerPacketKind::PermutedCommonMode
            | PeerPacketKind::AdaptivePermuted
            | PeerPacketKind::CompactCommon
            | PeerPacketKind::Alias
            | PeerPacketKind::Bitplane
    ));

    let decoded = PeerCodec
        .decode_window(&encoded.packet)
        .expect("decode production packet");
    assert_eq!(decoded.samples, signal);
    assert_eq!(decoded.context, context);
}

#[test]
fn production_contract_exposes_exact_identity_and_resource_limits() {
    assert_eq!(
        PEER_KERNEL_ID,
        "org.quitetall.lamquant.lml-optimum-v2.peer-v4"
    );
    assert_eq!(PEER_MAX_CHANNELS, 256);
    assert_eq!(PEER_MAX_SAMPLES, 32_768);
    assert_eq!(PEER_MAX_VALUES, 131_072);

    let too_many_channels = vec![vec![0_i64]; PEER_MAX_CHANNELS + 1];
    assert!(PeerCodec
        .encode_window(
            &too_many_channels,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .is_err());
}

#[test]
fn report_rejects_noncanonical_or_unknown_packets() {
    assert!(PeerCodec.inspect_packet(b"").is_err());

    let signal = fixture();
    let packet = PeerCodec
        .encode_window(
            &signal,
            PeerEncodeContext {
                sample_rate_mhz: 256_000,
                bit_depth: 16,
            },
        )
        .expect("encode packet");
    let mut corrupted = packet;
    corrupted[0] ^= 1;
    assert!(PeerCodec.inspect_packet(&corrupted).is_err());
}
