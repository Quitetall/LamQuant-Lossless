//! Autocorrelation kernel microbench (ADR 0054 perf probe, 2026-06-21).
//!
//! Replicates the three candidate autocorr kernels standalone (no crate
//! internals) and times them on the real operating point — `seg_len=256`,
//! order 16 (the default `Anytime{max_order:16}`) — to ground the "is there
//! bit-exact SIMD headroom?" question:
//!
//!   * `scalar`        — the firmware/no_std reference.
//!   * `avx2_current`  — the in-tree kernel: 4 lags per vector, `_mm256_set_pd`
//!     (4 scalar i64 loads + `as f64` convert per i), mul+add.
//!   * `avx2_hoisted`  — the only bit-exact win available: convert i64→f64 ONCE
//!     up front (each sample converted once, not ~order times)
//!     and use a real `_mm256_loadu_pd` instead of `set_pd`.
//!     Same per-lag accumulation order ⇒ bit-identical.
//!
//! Bit-exactness is asserted (all three must produce identical f64 bits).
//! Reassociating the per-lag sum (the textbook 8-wide dot product) or using FMA
//! would be faster but changes rounding ⇒ breaks byte-equality, so neither is a
//! candidate.
//!
//! ```text
//! cargo run -p lamquant-lml-mcu --release --example autocorr_bench
//! ```

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
type AutocorrKernel = unsafe fn(&[i64], usize, usize) -> Vec<f64>;

fn autocorr_scalar(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; order + 1];
    for lag in 0..=order {
        let end = seg_len.saturating_sub(lag);
        let mut s = 0.0f64;
        for i in 0..end {
            s += subband[i] as f64 * subband[i + lag] as f64;
        }
        r[lag] = s;
    }
    r
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorr_avx2_current(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    use core::arch::x86_64::*;
    let mut r = vec![0.0f64; order + 1];
    let lags = order + 1;
    let mut lag_base = 0;
    while lag_base + 4 <= lags {
        let common_end = seg_len.saturating_sub(lag_base + 3);
        let mut accs = _mm256_setzero_pd();
        for i in 0..common_end {
            let s_i = _mm256_set1_pd(subband[i] as f64);
            let v = _mm256_set_pd(
                subband[i + lag_base + 3] as f64,
                subband[i + lag_base + 2] as f64,
                subband[i + lag_base + 1] as f64,
                subband[i + lag_base] as f64,
            );
            accs = _mm256_add_pd(accs, _mm256_mul_pd(s_i, v));
        }
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), accs);
        r[lag_base..lag_base + 4].copy_from_slice(&buf);
        for k in 0..3 {
            let lag = lag_base + k;
            let end = seg_len.saturating_sub(lag);
            for i in common_end..end {
                r[lag] += subband[i] as f64 * subband[i + lag] as f64;
            }
        }
        lag_base += 4;
    }
    while lag_base < lags {
        let lag = lag_base;
        let end = seg_len.saturating_sub(lag);
        let mut s = 0.0f64;
        for i in 0..end {
            s += subband[i] as f64 * subband[i + lag] as f64;
        }
        r[lag] = s;
        lag_base += 1;
    }
    r
}

/// Hoisted-conversion variant: convert i64→f64 once, vector-load the lag window.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorr_avx2_hoisted(subband: &[i64], order: usize, seg_len: usize) -> Vec<f64> {
    use core::arch::x86_64::*;
    // Convert the slice we touch (up to seg_len+order) to f64 ONCE.
    let span = (seg_len + order).min(subband.len());
    let f: Vec<f64> = subband[..span].iter().map(|&x| x as f64).collect();

    let mut r = vec![0.0f64; order + 1];
    let lags = order + 1;
    let mut lag_base = 0;
    while lag_base + 4 <= lags {
        let common_end = seg_len.saturating_sub(lag_base + 3);
        let mut accs = _mm256_setzero_pd();
        for i in 0..common_end {
            let s_i = _mm256_set1_pd(f[i]);
            // Lanes 0..3 = f[i+lag_base .. +3] — same lane assignment as set_pd.
            let v = _mm256_loadu_pd(f.as_ptr().add(i + lag_base));
            accs = _mm256_add_pd(accs, _mm256_mul_pd(s_i, v));
        }
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), accs);
        r[lag_base..lag_base + 4].copy_from_slice(&buf);
        for k in 0..3 {
            let lag = lag_base + k;
            let end = seg_len.saturating_sub(lag);
            for i in common_end..end {
                r[lag] += f[i] * f[i + lag];
            }
        }
        lag_base += 4;
    }
    while lag_base < lags {
        let lag = lag_base;
        let end = seg_len.saturating_sub(lag);
        let mut s = 0.0f64;
        for i in 0..end {
            s += f[i] * f[i + lag];
        }
        r[lag] = s;
        lag_base += 1;
    }
    r
}

fn main() {
    let seg_len = 256usize;
    let order = 16usize;
    let n = seg_len + order + 8;
    // Realistic-ish EEG-magnitude samples (deterministic).
    let mut st = 0x9e3779b97f4a7c15u64;
    let sub: Vec<i64> = (0..n)
        .map(|_| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st as i64 % 5000) - 2500
        })
        .collect();

    let r_sc = autocorr_scalar(&sub, order, seg_len);
    #[cfg(target_arch = "x86_64")]
    let (r_cur, r_hoist) = unsafe {
        (
            autocorr_avx2_current(&sub, order, seg_len),
            autocorr_avx2_hoisted(&sub, order, seg_len),
        )
    };

    // Bit-exactness gate.
    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(r_sc, r_cur, "avx2_current must be bit-identical to scalar");
        assert_eq!(
            r_sc, r_hoist,
            "avx2_hoisted must be bit-identical to scalar"
        );
        println!("# bit-exactness: scalar == avx2_current == avx2_hoisted  ✓");
    }

    let iters = 2_000_000u64;
    let bytes_per = (seg_len * 8) as f64;

    let t = Instant::now();
    let mut acc = 0.0;
    for _ in 0..iters {
        acc += autocorr_scalar(std::hint::black_box(&sub), order, seg_len)[0];
    }
    let sc_ns = t.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(acc);

    println!("# seg_len={seg_len} order={order}  ({iters} iters)");
    println!(
        "# {:<16} {:>10} {:>12} {:>9}",
        "kernel", "ns/call", "GiB/s", "speedup"
    );
    println!(
        "  {:<16} {:>10.1} {:>12.2} {:>9}",
        "scalar",
        sc_ns,
        bytes_per / sc_ns,
        "1.00x"
    );

    #[cfg(target_arch = "x86_64")]
    {
        let kernels: [(&str, AutocorrKernel); 2] = [
            ("avx2_current", autocorr_avx2_current),
            ("avx2_hoisted", autocorr_avx2_hoisted),
        ];
        for (name, f) in kernels {
            let t = Instant::now();
            let mut acc = 0.0;
            for _ in 0..iters {
                acc += unsafe { f(std::hint::black_box(&sub), order, seg_len) }[0];
            }
            let ns = t.elapsed().as_nanos() as f64 / iters as f64;
            std::hint::black_box(acc);
            println!(
                "  {:<16} {:>10.1} {:>12.2} {:>8.2}x",
                name,
                ns,
                bytes_per / ns,
                sc_ns / ns
            );
        }
    }
}
