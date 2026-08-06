use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use lamquant_lml_optimum_v2::{PeerCodec, PeerEncodeContext};

const CHANNELS: usize = 21;
const SAMPLES: usize = 2_560;

fn eeg_window() -> Vec<Vec<i64>> {
    (0..CHANNELS)
        .map(|channel| {
            (0..SAMPLES)
                .map(|sample| {
                    let drift = ((sample * 3 + channel * 7) % 512) as i64 - 256;
                    let local = ((sample * sample + channel * 13) % 97) as i64 - 48;
                    drift * 40 + local
                })
                .collect()
        })
        .collect()
}

fn peer_cost(criterion: &mut Criterion) {
    let signal = eeg_window();
    let context = PeerEncodeContext {
        sample_rate_mhz: 256_000,
        bit_depth: 16,
    };
    let packet = PeerCodec
        .encode_window(&signal, context)
        .expect("benchmark fixture must encode");
    let decoded = PeerCodec
        .decode_window(&packet)
        .expect("benchmark fixture must decode");
    assert_eq!(decoded.samples, signal);
    assert_eq!(decoded.context, context);

    let source_bytes = (CHANNELS * SAMPLES * core::mem::size_of::<i32>()) as u64;
    let mut group = criterion.benchmark_group("optimum_v2_peer");
    group.throughput(Throughput::Bytes(source_bytes));
    group.bench_function("encode", |bencher| {
        bencher.iter(|| {
            black_box(
                PeerCodec
                    .encode_window(black_box(&signal), context)
                    .expect("encode benchmark window"),
            )
        })
    });
    group.bench_function("decode", |bencher| {
        bencher.iter(|| {
            black_box(
                PeerCodec
                    .decode_window(black_box(&packet))
                    .expect("decode benchmark packet"),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, peer_cost);
criterion_main!(benches);
