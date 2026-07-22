//! End-to-end codec throughput benchmark.
//!
//! Runs LML encode + decode at three working shapes — one window
//! single-channel (DSP-only hot path), one window multi-channel
//! (typical clinical), and a full container round-trip (encoder +
//! window framing + decoder). Synthetic deterministic signal so the
//! numbers are comparable across machines.
//!
//! Run:
//!
//!     cargo bench --features host --bench codec
//!
//! Criterion HTML reports land under `target/criterion/`.
//!
//! Regression policy: any commit that drops a throughput number by
//! more than the criterion noise threshold (10% default) shows as
//! "regressed" in the HTML diff. Treat that as a failing CI signal
//! the same way `cargo test` is treated.
//!
//! These numbers feed the perf-hardening workstream (SIMD on LPC +
//! ternary MAC, per-channel rayon, etc.). Land them under
//! `target/criterion/` before any optimisation work so we have a
//! stable baseline to diff against.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;

use lamquant_core::backend::{compress_with_backend, ComputeBackend};
use lamquant_core::container;
use lamquant_core::golomb;
use lamquant_core::lifting;
use lamquant_core::lml;
use lamquant_core::lpc::{self, LpcMode};

// ─── Deterministic synthetic signal ────────────────────────────────

/// xorshift64 — small fast deterministic PRNG. Same seed → same
/// signal across machines and architectures.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a `[n_ch][T]` signal that imitates EEG dynamic range:
/// ±2000 int16-ish counts, low-pass-correlated within each channel.
fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut signal = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        let mut state = seed.wrapping_add((c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut ch = Vec::with_capacity(t);
        let mut prev: i64 = 0;
        for _ in 0..t {
            // Random walk with bounded amplitude. Adds inter-sample
            // correlation so LPC has something to predict (otherwise
            // we'd benchmark a pathological worst case for entropy).
            let step = (xorshift64(&mut state) as i32 >> 24) as i64;
            prev = (prev + step).clamp(-2000, 2000);
            ch.push(prev);
        }
        signal.push(ch);
    }
    signal
}

// ─── Benchmarks ────────────────────────────────────────────────────

// Throughput byte count uses 2 bytes/sample (int16) -- the
// LamQuant wire format stores samples as int16 (or i24 for BDF).
// The bench's in-memory representation is `Vec<Vec<i64>>` to match
// the public API signature, but those 8-byte slots are populated
// with ±2000 values that fit in i16. Reporting MiB/s as
// "int16-equivalent" makes the numbers directly comparable to
// gzip/zstd/flac throughput on raw EDF (which IS int16 on disk).
const SAMPLE_BYTES: usize = 2;

// `noise_bits = 0` keeps the lossless contract: every input sample
// is exactly reconstructible. Non-zero values strip LSBs and produce
// lossy output; those paths are out of scope for the perf baseline
// since CR + speed both shift under truncation.
const LOSSLESS_NOISE_BITS: u8 = 0;

/// Single-window codec hot path: 4 ch × 2500 samples = 10 s @ 250 Hz.
fn bench_compress_single_window(c: &mut Criterion) {
    let signal = synth_signal(4, 2500, 0xDEAD_BEEF);
    let bytes = 4 * 2500 * SAMPLE_BYTES;
    let mut group = c.benchmark_group("compress_single_window");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("anytime", |b| {
        b.iter(|| {
            let _ = lml::compress(black_box(&signal), LOSSLESS_NOISE_BITS).expect("compress");
        });
    });
    group.bench_function("fixed", |b| {
        b.iter(|| {
            let _ =
                lml::compress_with_mode(black_box(&signal), LOSSLESS_NOISE_BITS, LpcMode::Fixed)
                    .expect("compress");
        });
    });
    group.finish();
}

/// Single-window decode hot path on the same shape.
fn bench_decompress_single_window(c: &mut Criterion) {
    let signal = synth_signal(4, 2500, 0xDEAD_BEEF);
    let bytes = 4 * 2500 * SAMPLE_BYTES;
    let encoded = lml::compress(&signal, LOSSLESS_NOISE_BITS).expect("compress");
    let mut group = c.benchmark_group("decompress_single_window");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("anytime", |b| {
        b.iter(|| {
            let _ = lml::decompress(black_box(&encoded)).expect("decompress");
        });
    });
    group.finish();
}

/// Multi-channel single window: 32 ch × 2500 samples = scalp-EEG
/// shape. Stresses the per-channel inner loop (where SIMD + rayon
/// will pay off).
fn bench_compress_multi_channel(c: &mut Criterion) {
    let signal = synth_signal(32, 2500, 0xCAFE_BABE);
    let bytes = 32 * 2500 * SAMPLE_BYTES;
    let mut group = c.benchmark_group("compress_multi_channel_32ch");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("firmware", |b| {
        b.iter(|| {
            let _ = compress_with_backend(
                black_box(&signal),
                LOSSLESS_NOISE_BITS,
                LpcMode::default(),
                ComputeBackend::Firmware,
            )
            .expect("firmware compress");
        });
    });
    // Desktop backend = rayon per-channel (today) + future AVX2/NEON
    // SIMD on the LPC autocorr inner loop. Same wire-format output
    // (proven byte-equal by the conformance gate); only wall-clock
    // differs. 32-channel input is the shape where the parallelism
    // actually pays off; the smaller 4ch single-window bench would
    // have its gain swamped by rayon scheduling overhead.
    group.bench_function("desktop_parallel", |b| {
        b.iter(|| {
            let _ = compress_with_backend(
                black_box(&signal),
                LOSSLESS_NOISE_BITS,
                LpcMode::default(),
                ComputeBackend::Desktop,
            )
            .expect("desktop compress");
        });
    });
    group.finish();
}

/// Multi-channel decode comparison: Firmware (serial) vs Desktop
/// (sequential parse + parallel synth/lift). Same 32-channel shape
/// as the encode bench so the speedup numbers are directly
/// comparable across compress vs decompress.
fn bench_decompress_multi_channel(c: &mut Criterion) {
    use lamquant_core::backend::decompress_with_backend;
    let signal = synth_signal(32, 2500, 0xCAFE_BABE);
    let bytes = 32 * 2500 * SAMPLE_BYTES;
    // Pre-encode once outside the timed loop.
    let encoded =
        lamquant_core::lml::compress(&signal, LOSSLESS_NOISE_BITS).expect("setup compress");

    let mut group = c.benchmark_group("decompress_multi_channel_32ch");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("firmware", |b| {
        b.iter(|| {
            let _ = decompress_with_backend(black_box(&encoded), ComputeBackend::Firmware)
                .expect("firmware decompress");
        });
    });
    group.bench_function("desktop_parallel", |b| {
        b.iter(|| {
            let _ = decompress_with_backend(black_box(&encoded), ComputeBackend::Desktop)
                .expect("desktop decompress");
        });
    });
    group.finish();
}

/// Full container round-trip: 16 channels × 25 000 samples
/// (≈100 s @ 250 Hz), 10 windows. Exercises encoder framing + window
/// split + per-window LML compress + container metadata path.
fn bench_container_roundtrip(c: &mut Criterion) {
    let n_ch = 16usize;
    let total = 25_000usize;
    let window = 2500usize;
    let sample_rate = 250.0;
    let signal = synth_signal(n_ch, total, 0xF00D_BEEF);
    let bytes = n_ch * total * SAMPLE_BYTES;
    let metadata = r#"{"format":"synthetic","bench":true}"#;

    let mut group = c.benchmark_group("container_roundtrip_16ch_25k");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut sink = Vec::with_capacity(bytes / 2);
            container::write_into(
                &mut sink,
                black_box(&signal),
                sample_rate,
                window,
                LOSSLESS_NOISE_BITS,
                metadata,
                LpcMode::default(),
            )
            .expect("write_into");
            sink
        });
    });

    // Encode once for the decode side; the cost is amortised by
    // criterion's iter count so per-iter timing is decode only.
    let mut encoded = Vec::with_capacity(bytes / 2);
    container::write_into(
        &mut encoded,
        &signal,
        sample_rate,
        window,
        LOSSLESS_NOISE_BITS,
        metadata,
        LpcMode::default(),
    )
    .expect("write_into for decode setup");
    group.bench_function("decode", |b| {
        b.iter(|| {
            let _ = container::read_bytes(black_box(&encoded)).expect("read_bytes");
        });
    });

    group.finish();
}

fn bench_container_roundtrip_32ch_100k(c: &mut Criterion) {
    let n_ch = 32usize;
    let total = 100_000usize;
    let window = 2500usize;
    let sample_rate = 250.0;
    let signal = synth_signal(n_ch, total, 0xABAB_CDCD_EFEF_0101);
    let bytes = n_ch * total * SAMPLE_BYTES;
    let metadata = r#"{"format":"synthetic","bench":true}"#;

    let mut group = c.benchmark_group("container_roundtrip_32ch_100k");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    lamquant_core::backend::set_global_backend(ComputeBackend::Firmware);
    group.bench_function("firmware", |b| {
        b.iter(|| {
            let mut sink = Vec::with_capacity(bytes / 2);
            container::write_into(
                black_box(&mut sink),
                &signal,
                sample_rate,
                window,
                LOSSLESS_NOISE_BITS,
                metadata,
                LpcMode::default(),
            )
            .expect("write_into (firmware)");
            sink
        });
    });
    lamquant_core::backend::set_global_backend(ComputeBackend::Desktop);
    group.bench_function("desktop_parallel", |b| {
        b.iter(|| {
            let mut sink = Vec::with_capacity(bytes / 2);
            container::write_into(
                black_box(&mut sink),
                &signal,
                sample_rate,
                window,
                LOSSLESS_NOISE_BITS,
                metadata,
                LpcMode::default(),
            )
            .expect("write_into (desktop)");
            sink
        });
    });

    group.finish();
}

// ─── Per-stage isolation benches ──────────────────────────────────
//
// These pin individual codec stages so we can localise where the
// per-window encode cost lives without needing root + `perf record`.
// Whichever stage shows lowest MiB/s is the bottleneck and the
// primary SIMD/parallelism target.
//
// Inputs match what a single 2500-sample window flowing through
// `lml::compress_with_mode` would feed each stage at the typical
// subband sizes (3-level lifting: 2500 → 1250+1250 → 625+625 →
// 313+313).

fn bench_stage_lifting_forward(c: &mut Criterion) {
    // One channel, 2500 samples. The lml encoder runs this once per
    // channel per window.
    let signal: Vec<i64> = synth_signal(1, 2500, 0xAA_BB_CC_DD)
        .into_iter()
        .next()
        .unwrap();
    let bytes = signal.len() * SAMPLE_BYTES;
    let mut group = c.benchmark_group("stage_lifting_forward_3level");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("2500_samples", |b| {
        b.iter(|| {
            let _ = lifting::forward_3level(black_box(&signal));
        });
    });
    group.finish();
}

fn bench_stage_lpc_analyze(c: &mut Criterion) {
    // Largest subband after 3-level lifting = `l1_detail`, 1250 samples.
    // In lml::compress_with_mode's subband vec the ordering is
    // (a3, d3, d2, d1) -> sb_idx 0..=3, so l1_detail is `sb_idx = 3`.
    // FIXED_ORDER_SCHEDULE = [3, 3, 6, 8], so this also exercises the
    // worst-case Fixed-mode order (8) -- the real per-window cost.
    const SB_IDX_L1_DETAIL: usize = 3;
    let signal: Vec<i64> = synth_signal(1, 1250, 0xBB_CC_DD_EE)
        .into_iter()
        .next()
        .unwrap();
    let bytes = signal.len() * SAMPLE_BYTES;
    let mut group = c.benchmark_group("stage_lpc_analyze_with_mode");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("anytime_l1_detail", |b| {
        b.iter(|| {
            let _ = lpc::analyze_with_mode(
                black_box(&signal),
                SB_IDX_L1_DETAIL,
                LpcMode::Anytime {
                    max_order: 16,
                    deadline: None,
                },
                8, // ctx_len
                None,
            );
        });
    });
    group.bench_function("fixed_l1_detail", |b| {
        b.iter(|| {
            let _ = lpc::analyze_with_mode(
                black_box(&signal),
                SB_IDX_L1_DETAIL,
                LpcMode::Fixed,
                8,
                None,
            );
        });
    });
    group.finish();
}

fn bench_stage_golomb_encode(c: &mut Criterion) {
    // Synthetic LPC residual — small int magnitudes since LPC removes
    // most signal energy. Match an entropy-like distribution by
    // generating bounded values.
    let mut state: u64 = 0xCAFE_FACE;
    let residual: Vec<i64> = (0..1250)
        .map(|_| {
            // Small zero-centred values, like a real LPC residual.
            let r = xorshift64(&mut state) as i32 >> 27; // ~±15
            r as i64
        })
        .collect();
    let bytes = residual.len() * SAMPLE_BYTES;
    let mut group = c.benchmark_group("stage_golomb_encode_dense");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("1250_residuals", |b| {
        b.iter(|| {
            let _ = golomb::encode_dense(black_box(&residual));
        });
    });
    group.finish();
}

/// ADR 0023 Track B3 — sweep encode + decode throughput across
/// single-window signal lengths spanning regimes 1, 2, and 3 (the
/// `auto_window_size` cliff at MAX_AUTO_WINDOW = 16384 sits at
/// length 16384 here since the test uses 250 Hz reference). This
/// is a pure-codec sweep (single window per call); the ingest
/// dispatcher's window-size pick is exercised in
/// `tests/window_size_transition.rs`. Useful for spotting
/// length-dependent perf cliffs in the codec hot path independent
/// of the dispatcher.
fn bench_window_size_sweep(c: &mut Criterion) {
    // Lengths chosen to cover regimes 1 / 2 / 3 (250 Hz reference)
    // plus ±1 sample around the regime-2 / regime-3 boundary at
    // 16384 samples.
    let lengths: &[usize] = &[2_500, 4_096, 8_000, 12_000, 16_384, 16_385, 24_000, 32_768];

    let mut group = c.benchmark_group("window_size_sweep_compress");
    group.measurement_time(Duration::from_secs(3));
    for &t in lengths {
        let signal = synth_signal(1, t, 0xC0DE_FACE_C0DE_FACE);
        let bytes = t * SAMPLE_BYTES;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(format!("n_{}", t), |b| {
            b.iter(|| {
                let _ = lml::compress(black_box(&signal), LOSSLESS_NOISE_BITS).expect("compress");
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("window_size_sweep_decompress");
    group.measurement_time(Duration::from_secs(3));
    for &t in lengths {
        let signal = synth_signal(1, t, 0xC0DE_FACE_C0DE_FACE);
        let bytes = t * SAMPLE_BYTES;
        let encoded = lml::compress(&signal, LOSSLESS_NOISE_BITS).expect("compress");
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(format!("n_{}", t), |b| {
            b.iter(|| {
                let _ = lml::decompress(black_box(&encoded)).expect("decompress");
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress_single_window,
    bench_decompress_single_window,
    bench_compress_multi_channel,
    bench_decompress_multi_channel,
    bench_container_roundtrip,
    bench_container_roundtrip_32ch_100k,
    bench_stage_lifting_forward,
    bench_stage_lpc_analyze,
    bench_stage_golomb_encode,
    bench_window_size_sweep,
);
criterion_main!(benches);
