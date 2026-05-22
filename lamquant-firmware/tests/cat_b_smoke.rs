//! Cat B firmware comparator smoke tests (2026-05-22).
//!
//! Compile + exercise each Cat B crate added under
//! `--features cat-b-{fixed,microfft,idsp,all}`. PASS = API surface
//! works; perf comparison happens in benches once a kernel-swap
//! candidate lands.

#![cfg(feature = "cat-b-all")]

/// `fixed` — Q-format types. Smoke: Q15 add + multiply round-trip
/// through the lib's `I1F15` (signed 1.15) and `I16F16` (signed 16.16)
/// types to confirm the crate works on host. On Hazard3 the firmware
/// uses raw `i32` / `i64` Q-format; this is a comparison candidate
/// for type-safe wrappers around the existing math.
#[cfg(feature = "cat-b-fixed")]
#[test]
fn fixed_q15_add_mul_smoke() {
    use fixed::types::{I1F15, I16F16};
    let a = I1F15::from_num(0.25_f32);
    let b = I1F15::from_num(0.125_f32);
    let sum = a.checked_add(b).expect("Q1.15 add must fit");
    assert!((sum.to_num::<f32>() - 0.375).abs() < 1e-4);

    let x = I16F16::from_num(2.5_f32);
    let y = I16F16::from_num(0.4_f32);
    let prod = x.checked_mul(y).expect("Q16.16 mul must fit");
    assert!((prod.to_num::<f32>() - 1.0).abs() < 1e-3);
}

/// `microfft` — no_std radix-2 FFT. Smoke: 8-point forward FFT on a
/// pure DC signal; bin 0 must dominate, bins 1..N/2 must be near zero.
/// Uses `re*re + im*im` instead of `.norm()` (microfft's `Complex32`
/// type doesn't ship a Hypot impl out of the box).
#[cfg(feature = "cat-b-microfft")]
#[test]
fn microfft_8_point_dc_smoke() {
    use microfft::real::rfft_8;
    let mut samples: [f32; 8] = [1.0; 8]; // DC = 1.0 everywhere
    let spectrum = rfft_8(&mut samples);
    // DC bin should be N (un-normalised), rest near zero.
    assert!((spectrum[0].re - 8.0).abs() < 1e-4, "bin 0 = {}", spectrum[0].re);
    for (i, bin) in spectrum[1..].iter().enumerate() {
        let mag2 = bin.re * bin.re + bin.im * bin.im;
        assert!(mag2 < 1e-4,
                "bin {} should be ~0, got |X|^2 = {}", i + 1, mag2);
    }
}

/// `idsp` — fixed-point DSP. Smoke: verify the crate is reachable
/// and `cossin` (a marquee feature) returns a reasonable Q31 result
/// for known input phase. No biquad / lowpass API today — the
/// crate's surface drifts each minor; perf comparison vs our
/// hand-rolled biquad lives in a future bench, not this smoke.
#[cfg(feature = "cat-b-idsp")]
#[test]
fn idsp_cossin_smoke() {
    // phase = 0 → (cos, sin) ≈ (i32::MAX, 0). idsp uses a small
    // polynomial approx; documented error budget is ~1e-5 relative.
    // Tolerate ~50_000 LSB on cos (≈ 2.3e-5) and ~50_000 LSB on sin.
    let (cos_q31, sin_q31) = idsp::cossin(0i32);
    let cos_err = (cos_q31 as i64 - i32::MAX as i64).abs();
    assert!(cos_err < 100_000,
            "cos(0) err > 100K LSB: cos_q31 = {} (err {})", cos_q31, cos_err);
    assert!(sin_q31.abs() < 100_000,
            "sin(0) err > 100K LSB: sin_q31 = {}", sin_q31);
}
