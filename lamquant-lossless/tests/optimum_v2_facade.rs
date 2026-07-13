#![cfg(feature = "archive")]

use lamquant_core::optimum_v2::{EncodeContext, OptimumV2Codec};

#[test]
fn host_facade_exposes_independent_optimum_v2_codec() {
    let samples = vec![vec![0, 1, 2, 3], vec![10, 9, 8, 7]];
    let context = EncodeContext {
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        channel_labels: vec!["C3".into(), "C4".into()],
    };
    let codec = OptimumV2Codec;
    let packet = codec.encode_window(&samples, &context).unwrap();
    let decoded = codec.decode_window(&packet).unwrap();
    assert_eq!(decoded.samples, samples);
    assert_eq!(decoded.context, context);
    assert_eq!(lamquant_core::decode(&packet).unwrap(), samples);
}
