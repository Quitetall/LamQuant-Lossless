use lamquant_lml_optimum_v2::mix1::{Mix1Codec, Mix1Decoded};

fn signal() -> Vec<Vec<i64>> {
    vec![
        (0..20)
            .map(|time| 3 * time + time % 5 - 2)
            .map(i64::from)
            .collect(),
        (0..20)
            .map(|time| 6 * time + time % 7 - 3)
            .map(i64::from)
            .collect(),
    ]
}

fn python_golden() -> Vec<u8> {
    hex(
        "4f5632500200100200e803000200000014000000be000000280000001a000000\
         16cb91e800000000000000000000000000000000000000000000000000000000\
         000000002d46d0f84d495831a704ffffffffffffffdac5bcd53cdac59e597c1fd1\
         5c14ad7ae51164d3436538b5ffc800f2da94087f183064df02c0a01cb3f0d2a1\
         faf1b47d9102cf66af",
    )
}

#[test]
fn rust_mix1_is_byte_exact_with_python_and_decodes_fresh() {
    let packet = Mix1Codec
        .encode_window(&signal(), 256_000, 16, 4)
        .expect("encode bounded MIX1 signal");

    assert_eq!(packet, python_golden());
    assert_eq!(
        Mix1Codec
            .decode_window(&packet)
            .expect("decode Rust packet"),
        Mix1Decoded {
            samples: signal(),
            sample_rate_mhz: 256_000,
            bit_depth: 16,
            score_shift: 4,
        }
    );
}

#[test]
fn rust_mix1_decodes_the_independent_python_packet() {
    let decoded = Mix1Codec
        .decode_window(&python_golden())
        .expect("decode independent Python packet");

    assert_eq!(decoded.samples, signal());
    assert_eq!(decoded.sample_rate_mhz, 256_000);
    assert_eq!(decoded.bit_depth, 16);
    assert_eq!(decoded.score_shift, 4);
}

#[test]
fn rust_mix1_is_deterministic_and_rejects_tampering() {
    let first = Mix1Codec
        .encode_window(&signal(), 256_000, 16, 2)
        .expect("first encode");
    let second = Mix1Codec
        .encode_window(&signal(), 256_000, 16, 2)
        .expect("second encode");
    assert_eq!(first, second);
    assert_eq!(
        Mix1Codec
            .decode_window(&first)
            .expect("exact roundtrip")
            .samples,
        signal()
    );

    let mut corrupted = first;
    *corrupted.last_mut().expect("nonempty packet") ^= 1;
    assert!(Mix1Codec.decode_window(&corrupted).is_err());
    assert!(Mix1Codec.encode_window(&signal(), 256_000, 16, 1).is_err());
}

#[test]
fn score_family_reuses_analysis_but_matches_independent_packets() {
    let family = Mix1Codec
        .encode_score_family(&signal(), 256_000, 16, &[2, 4, 8])
        .expect("encode MIX1 score family");

    for (score_shift, packet) in family {
        assert_eq!(
            packet,
            Mix1Codec
                .encode_window(&signal(), 256_000, 16, score_shift)
                .expect("encode independent score shift")
        );
    }
    let best = Mix1Codec
        .encode_best_score_window(&signal(), 256_000, 16)
        .expect("encode best score shift");
    let all = Mix1Codec
        .encode_score_family(&signal(), 256_000, 16, &[2, 3, 4, 5, 6, 7, 8])
        .expect("encode full score family");
    assert_eq!(
        best.len(),
        all.iter().map(|(_, packet)| packet.len()).min().unwrap()
    );
    assert_eq!(Mix1Codec.decode_window(&best).unwrap().samples, signal());
}

#[test]
fn peer_multivariate_carriers_are_exact_deterministic_and_never_larger_than_mix1() {
    let incumbent = Mix1Codec
        .encode_best_score_window(&signal(), 256_000, 16)
        .expect("encode MIX1 incumbent");
    let multivariate = Mix1Codec
        .encode_multivariate_window(&signal(), 256_000, 16, 4)
        .expect("encode multivariate carrier");
    assert_eq!(&multivariate[72..76], b"MMV1");
    assert_eq!(
        Mix1Codec.decode_window(&multivariate).unwrap().samples,
        signal()
    );
    assert_eq!(
        multivariate,
        Mix1Codec
            .encode_multivariate_window(&signal(), 256_000, 16, 4)
            .expect("repeat multivariate carrier")
    );

    let hierarchical = Mix1Codec
        .encode_hierarchical_multivariate_window(&signal(), 256_000, 16, 4)
        .expect("encode hierarchical multivariate carrier");
    assert_eq!(&hierarchical[72..76], b"MCH1");
    assert_eq!(
        Mix1Codec.decode_window(&hierarchical).unwrap().samples,
        signal()
    );
    assert_eq!(
        hierarchical,
        Mix1Codec
            .encode_hierarchical_multivariate_window(&signal(), 256_000, 16, 4)
            .expect("repeat hierarchical carrier")
    );

    for packet in [&multivariate, &hierarchical] {
        for end in 0..packet.len() {
            assert!(
                Mix1Codec.decode_window(&packet[..end]).is_err(),
                "accepted truncated peer prefix {end}"
            );
        }
        let mut corrupted = packet.clone();
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(Mix1Codec.decode_window(&corrupted).is_err());
        let mut trailed = packet.clone();
        trailed.push(0);
        assert!(Mix1Codec.decode_window(&trailed).is_err());
    }

    let best = Mix1Codec
        .encode_best_peer_window(&signal(), 256_000, 16)
        .expect("encode best peer carrier");
    assert!(best.len() <= incumbent.len());
    assert!(matches!(
        &best[72..76],
        b"MIX1" | b"MMV1" | b"MCH1" | b"MCX1" | b"MQX1" | b"MPX1"
    ));
    assert_eq!(Mix1Codec.decode_window(&best).unwrap().samples, signal());
    assert_eq!(
        best,
        Mix1Codec
            .encode_best_peer_window(&signal(), 256_000, 16)
            .expect("repeat best peer carrier")
    );
}

#[test]
fn peer_channel_context_carrier_roundtrips_and_rejects_noncanonical_masks() {
    let packet = Mix1Codec
        .encode_channel_context_window(&signal(), 256_000, 16, 4, 7)
        .expect("encode all-event channel-context carrier");
    assert_eq!(&packet[72..76], b"MCX1");
    assert_eq!(packet[78], 7);
    assert_eq!(Mix1Codec.decode_window(&packet).unwrap().samples, signal());
    assert_eq!(
        packet,
        Mix1Codec
            .encode_channel_context_window(&signal(), 256_000, 16, 4, 7)
            .expect("repeat all-event channel-context carrier")
    );

    for mask in [0, 1, 8] {
        assert!(Mix1Codec
            .encode_channel_context_window(&signal(), 256_000, 16, 4, mask)
            .is_err());
    }
}

#[test]
fn peer_common_mode_carrier_roundtrips_deterministically() {
    let packet = Mix1Codec
        .encode_common_mode_window(&signal(), 256_000, 16, 4, 7)
        .expect("encode causal common-mode carrier");
    assert_eq!(&packet[72..76], b"MQX1");
    assert_eq!(packet[78], 7);
    assert_eq!(Mix1Codec.decode_window(&packet).unwrap().samples, signal());
    assert_eq!(
        packet,
        Mix1Codec
            .encode_common_mode_window(&signal(), 256_000, 16, 4, 7)
            .expect("repeat causal common-mode carrier")
    );
}

#[test]
fn peer_permuted_common_mode_carrier_roundtrips_deterministically() {
    let packet = Mix1Codec
        .encode_permuted_common_mode_window(&signal(), 256_000, 16, 4, 7)
        .expect("encode permuted causal common-mode carrier");
    assert_eq!(&packet[72..76], b"MPX1");
    assert_eq!(packet[78], 7);
    assert_eq!(Mix1Codec.decode_window(&packet).unwrap().samples, signal());
    assert_eq!(
        packet,
        Mix1Codec
            .encode_permuted_common_mode_window(&signal(), 256_000, 16, 4, 7)
            .expect("repeat permuted causal common-mode carrier")
    );
    for end in 0..packet.len() {
        assert!(
            Mix1Codec.decode_window(&packet[..end]).is_err(),
            "accepted truncated MPX1 prefix {end}"
        );
    }
    let mut corrupted = packet;
    corrupted[79] = corrupted[80];
    assert!(Mix1Codec.decode_window(&corrupted).is_err());
}

#[test]
fn peer_permuted_common_mode_roundtrips_correlated_channel_families() {
    for channels in 1..=6 {
        let signal = (0..channels)
            .map(|channel| {
                (0..37)
                    .map(|time| {
                        let shared = 5 * time as i64 + ((time * 7) % 11) as i64 - 5;
                        shared + channel as i64 * 3 + ((time + channel * 2) % 5) as i64
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for mask in 2..=7 {
            let packet = Mix1Codec
                .encode_permuted_common_mode_window(&signal, 512_000, 16, 5, mask)
                .expect("encode correlated MPX1 family");
            assert_eq!(
                Mix1Codec.decode_window(&packet).unwrap().samples,
                signal,
                "MPX1 roundtrip differs for {channels} channels and mask {mask}"
            );
        }
    }
}

#[test]
fn rust_mix1_rejects_every_truncated_python_prefix() {
    let packet = python_golden();
    for end in 0..packet.len() {
        assert!(
            Mix1Codec.decode_window(&packet[..end]).is_err(),
            "accepted truncated prefix {end}"
        );
    }
}

fn hex(text: &str) -> Vec<u8> {
    let compact: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    assert_eq!(compact.len() % 2, 0);
    compact
        .chunks_exact(2)
        .map(|pair| {
            let high = digit(pair[0]);
            let low = digit(pair[1]);
            (high << 4) | low
        })
        .collect()
}

fn digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("invalid fixture hex"),
    }
}
