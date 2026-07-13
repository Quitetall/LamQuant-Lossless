use lamquant_lml_optimum_v2::{
    EncodeContext, OptimumV2Codec, BGF_MAGIC, LMO_MAGIC, LMO_V3_HEADER_LEN, LMO_VERSION,
};

#[test]
fn public_codec_round_trips_lmo1_v3_losslessly() {
    let signal: Vec<Vec<i64>> = vec![
        vec![0, 1, 2, 4, 8, 16, -4, -8],
        vec![100, 99, 98, 97, 96, 95, 94, 93],
    ];
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: vec!["C3".into(), "C4".into()],
    };
    let codec = OptimumV2Codec;

    let stream = codec.encode_window(&signal, &context).expect("encode");
    assert_eq!(&stream[0..4], LMO_MAGIC);
    assert_eq!(stream[4], LMO_VERSION);
    assert_eq!(&stream[LMO_V3_HEADER_LEN..LMO_V3_HEADER_LEN + 4], BGF_MAGIC);

    let decoded = codec.decode_window(&stream).expect("decode");
    assert_eq!(decoded.samples, signal);
    assert_eq!(decoded.context, context);
}

#[test]
fn native_delta_mode_reduces_smooth_signal_bytes_without_old_codec_fallback() {
    let signal = vec![(0..4096).map(|sample| sample as i64).collect::<Vec<_>>()];
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: vec!["Cz".into()],
    };
    let codec = OptimumV2Codec;

    let stream = codec.encode_window(&signal, &context).expect("encode");
    let raw_i32_payload = signal[0].len() * 4;
    assert!(
        stream.len() < raw_i32_payload,
        "native v2 delta mode should beat unframed raw i32 on a ramp"
    );
    assert_eq!(
        codec.decode_window(&stream).expect("decode").samples,
        signal
    );
}

#[test]
fn fixed_header_tampering_fails_integrity() {
    let signal = vec![vec![1, 2, 3, 4]];
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: vec!["Cz".into()],
    };
    let codec = OptimumV2Codec;
    let mut stream = codec.encode_window(&signal, &context).expect("encode");

    // LMO header is 7 bytes; BGF sample_rate_mhz starts at BGF offset 16.
    stream[LMO_V3_HEADER_LEN + 16] ^= 1;
    let error = codec
        .decode_window(&stream)
        .expect_err("tampered header must fail");
    assert!(
        error.to_string().contains("integrity"),
        "unexpected error: {error}"
    );
}

#[test]
fn encoding_is_byte_deterministic() {
    let signal = vec![vec![3, -2, 9, 9], vec![4, 4, 4, -1]];
    let context = EncodeContext {
        sample_rate_mhz: 256_000,
        bit_depth: 16,
        channel_labels: vec!["F3".into(), "F4".into()],
    };
    let codec = OptimumV2Codec;
    assert_eq!(
        codec.encode_window(&signal, &context).unwrap(),
        codec.encode_window(&signal, &context).unwrap()
    );
}

#[test]
fn dimension_product_is_bounded_before_allocation() {
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: (0..257).map(|index| format!("ch{index}")).collect(),
    };
    let signal = vec![vec![0]; 257];
    let error = OptimumV2Codec
        .encode_window(&signal, &context)
        .expect_err("257 channels must fail");
    assert!(error.to_string().contains("channel count"));
}

#[test]
fn decode_rejects_truncation_at_every_wire_boundary() {
    let signal = vec![vec![1, 2, 3, 4]];
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: vec!["Cz".into()],
    };
    let codec = OptimumV2Codec;
    let stream = codec.encode_window(&signal, &context).unwrap();
    for end in 0..stream.len() {
        assert!(
            codec.decode_window(&stream[..end]).is_err(),
            "accepted prefix {end}"
        );
    }
}
