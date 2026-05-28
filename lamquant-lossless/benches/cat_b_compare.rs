//! Cat B empirical comparison bench (2026-05-22).
//!
//! User direction: *"Test whether constriction is more efficient than
//! the golomb rice implementation empirically."*
//!
//! Same synthetic LPC residual block fed to both encoders. Reports
//! throughput MiB/s + compressed bytes for each path:
//!
//! - **golomb-rice** (hand-rolled, in-tree default, `golomb::encode_dense`)
//! - **constriction rANS** (opt-in `experimental_arithmetic` feature,
//!   `arithmetic::encode_dense`)
//!
//! Run:
//!   cargo bench --features "host experimental_arithmetic" \
//!     -p lamquant-core --bench cat_b_compare
//!
//! Verdict is purely empirical: lower bytes-per-residual = better CR;
//! higher MiB/s = better throughput; the codec choice rides on the
//! product of the two (rate × throughput) for actual encode wall time.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lamquant_core::{edf, golomb, lma, lpc};

#[cfg(feature = "experimental_arithmetic")]
use lamquant_core::arithmetic;

const RESIDUAL_LEN: usize = 1250;
const SAMPLE_BYTES: usize = 8; // i64

const CHBMIT_LMA_PATH: &str = "/mnt/4tb/data/Archive/lma/physionet/chbmit.lma";
/// Fallback list of CHB-MIT inner paths to try. Some EDFs in this
/// corpus have non-standard header fields the reader rejects;
/// iterate until one parses. (Verified 2026-05-22: chb01_01 is
/// the canonical reference and parses cleanly.)
const CHBMIT_INNER_CANDIDATES: &[&str] = &[
    "chb01/chb01_01.edf",
    "chb01/chb01_02.edf",
    "chb01/chb01_03.edf",
    "chb06/chb06_01.edf",
];
/// Direct on-disk fallback when the LMA isn't installed or the inner
/// EDF parse fails for every candidate.
const CHBMIT_DIRECT_DIR: &str = "/mnt/4tb/data/Archive/edf/physionet/chbmit";

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Synthetic LPC residual — small zero-centred i64 values (~±15)
/// mirrors `bench_stage_golomb_encode` in `benches/codec.rs` so the
/// two harnesses are directly comparable.
fn make_residual(seed: u64) -> Vec<i64> {
    let mut state = seed;
    (0..RESIDUAL_LEN)
        .map(|_| {
            let r = xorshift64(&mut state) as i32 >> 27;
            r as i64
        })
        .collect()
}

fn bench_entropy_compare(c: &mut Criterion) {
    let residual = make_residual(0xCAFE_FACE);
    let bytes_in = residual.len() * SAMPLE_BYTES; // input bytes / iter

    // --- Print one-shot compressed size for the verdict header ---
    let gr_size = golomb::encode_dense(&residual)
        .expect("golomb encode for size measurement")
        .len();
    eprintln!(
        "\n[cat_b_compare] residual_len={}, input_bytes={}, golomb_rice_bytes_out={} ({:.2} bits/sym)",
        RESIDUAL_LEN,
        bytes_in,
        gr_size,
        (gr_size as f64 * 8.0) / RESIDUAL_LEN as f64,
    );
    #[cfg(feature = "experimental_arithmetic")]
    {
        let rans_size = arithmetic::encode_dense(&residual).len();
        eprintln!(
            "[cat_b_compare] constriction_rANS_bytes_out={} ({:.2} bits/sym), CR_delta_vs_golomb={:+.1}%",
            rans_size,
            (rans_size as f64 * 8.0) / RESIDUAL_LEN as f64,
            ((rans_size as f64 - gr_size as f64) / gr_size as f64) * 100.0,
        );
    }
    #[cfg(not(feature = "experimental_arithmetic"))]
    eprintln!(
        "[cat_b_compare] constriction NOT built — rerun with --features experimental_arithmetic for A/B",
    );

    // --- Throughput benches ---
    let mut group = c.benchmark_group("cat_b_entropy_encode_1250_residuals");
    group.throughput(Throughput::Bytes(bytes_in as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("golomb_rice", |b| {
        b.iter(|| {
            let _ = golomb::encode_dense(black_box(&residual));
        });
    });

    #[cfg(feature = "experimental_arithmetic")]
    group.bench_function("constriction_rans", |b| {
        b.iter(|| {
            let _ = arithmetic::encode_dense(black_box(&residual));
        });
    });

    group.finish();
}

/// Try LMA-then-direct EDF paths until one decodes; on success, run
/// LPC analysis on channel 0's first `RESIDUAL_LEN` samples and
/// return the residual. Bench gracefully skips the real-data section
/// if the corpus isn't installed locally.
fn load_chbmit_residual() -> Option<Vec<i64>> {
    let edf_file = try_load_chbmit_edf()?;
    if edf_file.signal.is_empty() || edf_file.signal[0].len() < RESIDUAL_LEN {
        eprintln!(
            "[cat_b_compare] CHB-MIT EDF too short: {} ch, ch0 has {} samples",
            edf_file.signal.len(),
            edf_file.signal.first().map(|c| c.len()).unwrap_or(0),
        );
        return None;
    }
    // First channel, first 1250 samples (5 s @ 250 Hz) → LPC analyze
    // produces a 1250-residual block matching the synthetic harness
    // shape for a direct A/B.
    let channel: Vec<i64> = edf_file.signal[0][..RESIDUAL_LEN].to_vec();
    let (_coeffs, residual) = lpc::analyze(&channel, 8, 256);
    Some(residual)
}

/// Probe both LMA and direct-tree fallback. Returns the first EDF
/// that parses cleanly + writes the chosen source path to stderr.
fn try_load_chbmit_edf() -> Option<edf::EdfFile> {
    // 1) LMA path: try each candidate inner entry.
    let lma_path = Path::new(CHBMIT_LMA_PATH);
    if lma_path.exists() {
        for inner in CHBMIT_INNER_CANDIDATES {
            let bytes = match lma::read_entry(lma_path, inner) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut tmp = match tempfile::NamedTempFile::new() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if tmp.write_all(&bytes).is_err() {
                continue;
            }
            match edf::read_edf(tmp.path()) {
                Ok(f) => {
                    eprintln!(
                        "[cat_b_compare] loaded CHB-MIT from LMA: {} :: {}",
                        CHBMIT_LMA_PATH, inner,
                    );
                    return Some(f);
                }
                Err(_) => continue,
            }
        }
    }
    // 2) Direct on-disk fallback.
    for inner in CHBMIT_INNER_CANDIDATES {
        let direct = Path::new(CHBMIT_DIRECT_DIR).join(inner);
        if !direct.exists() {
            continue;
        }
        match edf::read_edf(&direct) {
            Ok(f) => {
                eprintln!(
                    "[cat_b_compare] loaded CHB-MIT from disk: {}",
                    direct.display(),
                );
                return Some(f);
            }
            Err(e) => {
                eprintln!(
                    "[cat_b_compare] direct edf parse failed for {}: {}",
                    direct.display(), e,
                );
            }
        }
    }
    eprintln!("[cat_b_compare] no CHB-MIT EDF parsed — skipping real-data A/B");
    None
}

fn bench_entropy_compare_chbmit(c: &mut Criterion) {
    let Some(residual) = load_chbmit_residual() else {
        return;
    };
    let bytes_in = residual.len() * SAMPLE_BYTES;

    let gr_size = golomb::encode_dense(&residual)
        .expect("golomb encode for size measurement")
        .len();
    eprintln!(
        "\n[cat_b_compare/CHBMIT real] residual_len={}, input_bytes={}, golomb_rice_bytes_out={} ({:.2} bits/sym)",
        residual.len(),
        bytes_in,
        gr_size,
        (gr_size as f64 * 8.0) / residual.len() as f64,
    );
    #[cfg(feature = "experimental_arithmetic")]
    {
        let rans_size = arithmetic::encode_dense(&residual).len();
        eprintln!(
            "[cat_b_compare/CHBMIT real] constriction_rANS_bytes_out={} ({:.2} bits/sym), CR_delta_vs_golomb={:+.1}%",
            rans_size,
            (rans_size as f64 * 8.0) / residual.len() as f64,
            ((rans_size as f64 - gr_size as f64) / gr_size as f64) * 100.0,
        );
    }

    let mut group = c.benchmark_group("cat_b_entropy_encode_chbmit_real");
    group.throughput(Throughput::Bytes(bytes_in as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("golomb_rice", |b| {
        b.iter(|| {
            let _ = golomb::encode_dense(black_box(&residual));
        });
    });

    #[cfg(feature = "experimental_arithmetic")]
    group.bench_function("constriction_rans", |b| {
        b.iter(|| {
            let _ = arithmetic::encode_dense(black_box(&residual));
        });
    });

    group.finish();
}

// ─── Cat B comparator #2: idsp DirectForm1 biquad vs hand-rolled ──
//
// `idsp::iir::Biquad<f32>` + `DirectForm1<f32>` is the public f32
// path through dsp_process::SplitProcess. Compare against a 10-line
// inline DF1 f32 hand-roll on the same input + same coefficients.
// f32 (not i32) chosen because idsp's i32 Biquad uses a Q<i32, i64, F>
// internal type with non-trivial construction; this bench is about
// API/lib overhead, not the fixed-point representation. Firmware
// uses its own integer biquad — not affected by this result.

/// Minimal Direct-Form-1 biquad — f32 in/out, f32 coefficients.
struct HandRolledDf1F32 {
    b0: f32, b1: f32, b2: f32, a1: f32, a2: f32,
    x1: f32, x2: f32, y1: f32, y2: f32,
}

impl HandRolledDf1F32 {
    fn new(b: [f32; 3], a: [f32; 2]) -> Self {
        Self { b0: b[0], b1: b[1], b2: b[2], a1: a[0], a2: a[1],
               x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
              - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn bench_biquad_compare(c: &mut Criterion) {
    use dsp_process::SplitProcess;
    use idsp::iir::{Biquad, DirectForm1};

    let mut state: u64 = 0xDEAD_BEEF_CAFE;
    let samples: Vec<f32> = (0..1024)
        .map(|_| ((xorshift64(&mut state) as i32 as f32) / (i32::MAX as f32)) * 1024.0)
        .collect();

    // High-pass biquad at 0.5 Hz / 250 Hz nyquist (matches the
    // firmware HP filter operating point in spirit).
    const BA: [f32; 5] = [
        0.99986_f32, -1.99973, 0.99986,  // b0, b1, b2
        -1.99959, 0.99986,                // a1, a2
    ];

    let bytes = samples.len() * core::mem::size_of::<f32>();
    let mut group = c.benchmark_group("cat_b_biquad_df1_f32_1024_samples");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("handrolled_df1_f32", |b| {
        b.iter(|| {
            let mut filter = HandRolledDf1F32::new(
                [BA[0], BA[1], BA[2]],
                [BA[3], BA[4]],
            );
            let mut sum: f32 = 0.0;
            for &x in samples.iter() {
                sum += filter.process(black_box(x));
            }
            black_box(sum);
        });
    });

    group.bench_function("idsp_directform1_f32", |b| {
        b.iter(|| {
            let biquad: Biquad<f32> = Biquad::from(BA);
            let mut state_idsp: DirectForm1<f32> = DirectForm1::default();
            let mut sum: f32 = 0.0;
            for &x in samples.iter() {
                let y = biquad.process(&mut state_idsp, black_box(x));
                sum += y;
            }
            black_box(sum);
        });
    });

    group.finish();
}

// Q-format firmware-realistic A/B — same comparison but with the
// Q30 i32 path firmware actually uses on RP2350. Per user direction
// (2026-05-22): if f32 tied, still measure Q-format to confirm the
// real-firmware shape isn't a different story.

/// Minimal i32 Q30 DF1 biquad — matches the firmware
/// `BiquadState::process` shape (i64 accumulator, round-half-up >>30,
/// saturating i32 cast).
struct HandRolledDf1Q30 {
    b0: i32, b1: i32, b2: i32, a1: i32, a2: i32,
    x1: i32, x2: i32, y1: i32, y2: i32,
}

impl HandRolledDf1Q30 {
    fn new(ba: [i32; 5]) -> Self {
        Self {
            b0: ba[0], b1: ba[1], b2: ba[2], a1: ba[3], a2: ba[4],
            x1: 0, x2: 0, y1: 0, y2: 0,
        }
    }

    #[inline]
    fn process(&mut self, x: i32) -> i32 {
        let acc: i64 =
            (self.b0 as i64) * (x as i64)
          + (self.b1 as i64) * (self.x1 as i64)
          + (self.b2 as i64) * (self.x2 as i64)
          - (self.a1 as i64) * (self.y1 as i64)
          - (self.a2 as i64) * (self.y2 as i64);
        let rounded = (acc + (1i64 << 29)) >> 30;
        let y = rounded.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn bench_biquad_compare_q30(c: &mut Criterion) {
    use dsp_fixedpoint::Q;
    use dsp_process::SplitProcess;
    use idsp::iir::{Biquad, DirectForm1Wide};

    let mut state: u64 = 0xFADE_BEEF;
    let samples: Vec<i32> =
        (0..1024).map(|_| (xorshift64(&mut state) as i32) >> 11).collect();

    // Same HP at 0.5 Hz / 250 Hz Nyquist, expressed in Q30 i32:
    //   round(coeff * 2^30) for coeff ∈ [-2.0, 2.0)
    const BA_Q30: [i32; 5] = [
        1073312824,   // b0 ≈ +0.9998
        -2146625648,  // b1 ≈ -1.9996
        1073312824,   // b2 ≈ +0.9998
        -2146458584,  // a1 ≈ -1.9994
        1072458672,   // a2 ≈ +0.9989
    ];

    let bytes = samples.len() * core::mem::size_of::<i32>();
    let mut group = c.benchmark_group("cat_b_biquad_df1_q30_i32_1024_samples");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("handrolled_df1_q30_i32", |b| {
        b.iter(|| {
            let mut filter = HandRolledDf1Q30::new(BA_Q30);
            let mut sum: i64 = 0;
            for &x in samples.iter() {
                sum = sum.wrapping_add(filter.process(black_box(x)) as i64);
            }
            black_box(sum);
        });
    });

    group.bench_function("idsp_directform1_q30_i32", |b| {
        b.iter(|| {
            // idsp's `Biquad<Q<i32, i64, 30>>` impls `SplitProcess<i32,
            // i32, DirectForm1Wide>`. `Q::new` wraps the raw Q30 bits.
            let ba_q: [Q<i32, i64, 30>; 5] = [
                Q::new(BA_Q30[0]),
                Q::new(BA_Q30[1]),
                Q::new(BA_Q30[2]),
                Q::new(BA_Q30[3]),
                Q::new(BA_Q30[4]),
            ];
            let biquad: Biquad<Q<i32, i64, 30>> = Biquad { ba: ba_q };
            let mut state_idsp: DirectForm1Wide = DirectForm1Wide::default();
            let mut sum: i64 = 0;
            for &x in samples.iter() {
                let y = biquad.process(&mut state_idsp, black_box(x));
                sum = sum.wrapping_add(y as i64);
            }
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_entropy_compare,
    bench_entropy_compare_chbmit,
    bench_biquad_compare,
    bench_biquad_compare_q30,
);
criterion_main!(benches);
