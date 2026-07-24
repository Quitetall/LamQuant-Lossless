//! ADR 0069 S7b — the LMQ training normalization pipeline, hoisted from Python
//! (`lamquant_codec/training/lma_dataset.py::decode_lma_signal`) into Rust so
//! LML + LMQ draw from ONE definition. These are **Lossy** transforms (the LML
//! lossless backend must never run them; the pass framework enforces that) and
//! are **host-only** (`#[cfg(feature = "archive")]`) — `sosfiltfilt` is a
//! non-causal forward-backward filter (a T2/basestation pass), so there is no
//! MCU/no_std variant here (a causal streaming HP for T0/T1 is a deferred
//! follow-up, flagged in the pass metadata).
//!
//! Calibration (digital→µV) is upstream and already bit-exact in Rust
//! (`container_read_phys_f32`); this module is everything AFTER it: channel
//! select → resample→250 Hz → 0.5 Hz zero-phase highpass → Q31.
//!
//! Parity: bit-exactness vs scipy/LAPACK is impossible for the FP stages, so
//! `tests/normalize_parity.rs` asserts exact equality on the final int32 Q31
//! output where deterministic and a documented ±LSB tolerance on the
//! FP-divergent stages. See that gate before switching `lma_dataset.py` over.

/// Target sample rate the LMQ path normalizes to (Hz).
pub const TARGET_SR: f64 = 250.0;
/// Q31 headroom (`convert_lml` default; `lma_dataset.py::Q31_HEADROOM`).
pub const Q31_HEADROOM: f64 = 0.72;
/// Highpass cutoff (Hz) — `lma_dataset.py::HIGHPASS_HZ`.
pub const HIGHPASS_HZ: f64 = 0.5;

/// `scipy.signal.butter(2, 0.5, btype='high', fs=250.0, output='sos')` — a
/// single second-order section `[b0, b1, b2, a0, a1, a2]`, dumped verbatim from
/// scipy 1.18.0. Fixed operating point (250 Hz / 0.5 Hz HP = [`HIGHPASS_HZ`] /
/// [`TARGET_SR`]); regenerate (and re-dump the golden) if either changes.
/// `a0 == 1.0`. These are NOT derived from the symbolic consts at runtime (no
/// Rust `butter` impl); the `sosfiltfilt_matches_scipy_oracle_on_ramp20` test
/// is the guard — wrong coeffs make its output diverge from the scipy oracle.
const HP_B: [f64; 3] = [0.991153595101663, -1.982307190203326, 0.991153595101663];
const HP_A: [f64; 3] = [1.0, -1.9822289297925284, 0.9823854506141251];
/// `scipy.signal.sosfilt_zi(sos)` for the section above — the steady-state
/// initial condition filtfilt scales by the first (odd-extended) sample.
const HP_ZI: [f64; 2] = [-0.991153595101663, 0.991153595101663];
/// scipy `sosfiltfilt` default edge/pad length for a single section:
/// `3 * (2 * n_sections + 1)` = 9.
const HP_PADLEN: usize = 9;

/// Odd extension of `x` by `n` samples each end — `scipy.signal._arraytools.odd_ext`:
/// left = `2*x[0] - x[n..1]`, right = `2*x[-1] - x[-2..-(n+1)]`. Requires `x.len() > n`.
/// Private; the sole caller (`sosfiltfilt_hp`) enforces the precondition — the
/// `debug_assert` makes the contract explicit if that ever changes.
fn odd_ext(x: &[f64], n: usize) -> Vec<f64> {
    debug_assert!(
        x.len() > n,
        "odd_ext requires x.len() ({}) > n ({n})",
        x.len()
    );
    let len = x.len();
    let mut out = Vec::with_capacity(len + 2 * n);
    // left: 2*x[0] - x[n], 2*x[0] - x[n-1], …, 2*x[0] - x[1]
    for j in (1..=n).rev() {
        out.push(2.0 * x[0] - x[j]);
    }
    out.extend_from_slice(x);
    // right: 2*x[-1] - x[len-2], 2*x[-1] - x[len-3], …, 2*x[-1] - x[len-1-n]
    for k in 0..n {
        out.push(2.0 * x[len - 1] - x[len - 2 - k]);
    }
    out
}

/// One second-order section, Direct-Form-II-Transposed (scipy `sosfilt`'s
/// recurrence), with initial state `zi`. Returns the filtered signal.
fn sosfilt_one(b: [f64; 3], a: [f64; 3], x: &[f64], zi: [f64; 2]) -> Vec<f64> {
    let (b0, b1, b2) = (b[0], b[1], b[2]);
    let (a1, a2) = (a[1], a[2]); // a0 == 1.0
    let mut z0 = zi[0];
    let mut z1 = zi[1];
    let mut y = Vec::with_capacity(x.len());
    for &xn in x {
        let yn = b0 * xn + z0;
        z0 = b1 * xn - a1 * yn + z1;
        z1 = b2 * xn - a2 * yn;
        y.push(yn);
    }
    y
}

/// Zero-phase 0.5 Hz Butterworth highpass — the Rust port of
/// `sosfiltfilt(butter(2, 0.5, fs=250, 'high'), x)`. Forward-backward with odd
/// padding + `sosfilt_zi`-scaled initial conditions, matching scipy's algorithm
/// (validated to <1e-9 against a scipy oracle in the unit tests). Input/output
/// f64; a channel shorter than `HP_PADLEN` is returned unfiltered (scipy would
/// raise — the caller guarantees window length ≫ 9).
pub fn sosfiltfilt_hp(x: &[f64]) -> Vec<f64> {
    let n = HP_PADLEN;
    if x.len() <= n {
        return x.to_vec();
    }
    let ext = odd_ext(x, n);
    let x0 = ext[0];
    let fwd = sosfilt_one(HP_B, HP_A, &ext, [HP_ZI[0] * x0, HP_ZI[1] * x0]);

    // reverse, filter again, reverse back
    let mut rev: Vec<f64> = fwd.into_iter().rev().collect();
    let y0 = rev[0];
    let bwd = sosfilt_one(HP_B, HP_A, &rev, [HP_ZI[0] * y0, HP_ZI[1] * y0]);
    rev = bwd.into_iter().rev().collect();

    // trim the odd-extension padding
    rev[n..rev.len() - n].to_vec()
}

/// Q31 quantization — `lma_dataset.py:429-430`:
/// `gain = 0.72 / max|data|; (data * gain * 2147483647.0).astype(int32)`.
/// The two multiplies are left-to-right f64 (matching numpy's evaluation
/// order), and `as i32` truncates toward zero — the values are in `[-0.72,
/// 0.72] * (2^31 - 1)`, safely inside the i32 range, so no saturation. Returns
/// `None` for an all-flat signal (`max|data| < 1e-12`), matching the Python.
pub fn q31_normalize(data: &[Vec<f64>]) -> Option<Vec<Vec<i32>>> {
    let max_abs = data
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    if max_abs < 1e-12 {
        return None;
    }
    let gain = Q31_HEADROOM / max_abs;
    Some(
        data.iter()
            .map(|row| {
                row.iter()
                    .map(|&v| (v * gain * 2_147_483_647.0) as i32)
                    .collect()
            })
            .collect(),
    )
}

/// EEG normalization for a signal ALREADY at 250 Hz with channels already in
/// the 21-target order (channel-select + resample are identity here). Per-channel
/// zero-phase HP → Q31. Returns `None` on an all-flat signal.
pub fn normalize_eeg_250hz(data: &[Vec<f64>]) -> Option<Vec<Vec<i32>>> {
    let filtered: Vec<Vec<f64>> = data.iter().map(|ch| sosfiltfilt_hp(ch)).collect();
    q31_normalize(&filtered)
}

// ───────────────────────────── resample → 250 Hz ─────────────────────────────
//
// Port of `lma_dataset.py:408-421`. Branch on the gcd-reduced (up, down):
// `resample_poly` (polyphase, the common EEG rates) when both ≤ 256, else
// `scipy.signal.resample` (FFT) — the FFT branch is not yet ported (deferred;
// see `NormalizeError::FftResampleUnsupported`). Unlike the bit-exact HP+Q31,
// `resample_poly` uses transcendentals (sinc, Kaiser I0), so its match to scipy
// is tolerance-bounded, not bit-exact.

/// A normalization failure the caller must handle (rather than silently degrade).
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizeError {
    /// The gcd-reduced up/down exceeds 256 → scipy takes the FFT
    /// (`scipy.signal.resample`) branch, which is not yet ported to Rust. The
    /// caller (pyfunction / cutover) falls back to the Python path for this rate.
    FftResampleUnsupported { orig_sr: f64 },
}

impl core::fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NormalizeError::FftResampleUnsupported { orig_sr } => write!(
                f,
                "resample {orig_sr} Hz → 250 Hz needs the scipy FFT-resample branch (up/down > 256), \
                 not yet ported to Rust"
            ),
        }
    }
}
impl std::error::Error for NormalizeError {}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Modified Bessel function of the first kind, order 0 — the Kaiser-window
/// weight function. Power series `Σ (x²/4)^k / (k!)²`; f64-accurate.
fn i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    loop {
        term *= y / (k * k);
        sum += term;
        if term <= 1e-18 * sum {
            break;
        }
        k += 1.0;
    }
    sum
}

/// `sin(πx)/(πx)`, `sinc(0)=1` — the numpy `np.sinc` convention.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = core::f64::consts::PI * x;
        px.sin() / px
    }
}

/// `scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta))` for a single
/// lowpass band `[0, cutoff]` (cutoff in Nyquist units, `pass_zero`, `scale=True`):
/// a Kaiser-windowed sinc normalized to unity DC gain.
fn firwin_lowpass_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let i0_beta = i0(beta);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha; // symmetric about 0
            let ideal = cutoff * sinc(cutoff * m);
            // Kaiser window w[i] = I0(beta*sqrt(1-(m/alpha)^2)) / I0(beta)
            let r = (1.0 - (m / alpha).powi(2)).max(0.0).sqrt();
            ideal * (i0(beta * r) / i0_beta)
        })
        .collect();
    let s: f64 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= s;
    }
    h
}

/// `scipy.signal._upfirdn._output_len` — length of an `upfirdn(h, x, up, down)`.
fn output_len(len_h: usize, n_in: usize, up: usize, down: usize) -> usize {
    (((n_in - 1) * up + len_h) - 1) / down + 1
}

/// `scipy.signal.upfirdn(h, x, up, down)` — upsample x by `up` (zero-stuff),
/// FIR-filter with `h`, downsample by `down`. Direct polyphase evaluation:
/// `y[m] = Σ_k h[k] · x_up[m·down − k]` where `x_up[j] = x[j/up]` iff `j ≥ 0 ∧
/// j mod up == 0 ∧ j/up < len(x)`, else 0. The inner loop steps `k` by `up`
/// (only those k give a nonzero upsampled tap).
fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let n_in = x.len();
    let len_h = h.len();
    let out_len = output_len(len_h, n_in, up, down);
    let mut y = vec![0.0f64; out_len];
    for (m, ym) in y.iter_mut().enumerate() {
        let base = m * down;
        let mut acc = 0.0f64;
        // k ≡ base (mod up) ⇒ (base-k) is a multiple of up ⇒ a real sample.
        let mut k = base % up;
        while k < len_h && k <= base {
            let xi = (base - k) / up;
            if xi < n_in {
                acc += h[k] * x[xi];
            }
            k += up;
        }
        *ym = acc;
    }
    y
}

/// `scipy.signal.resample_poly(x, up, down, window=('kaiser', 5.0))` — polyphase
/// rational resample with the center-trim padding scipy applies to remove the
/// filter group delay. Matches scipy to a tolerance (transcendental filter design).
pub fn resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = {
        let raw = n_in * up;
        raw / down + usize::from(raw % down != 0)
    };
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let mut h = firwin_lowpass_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0);
    for v in h.iter_mut() {
        *v *= up as f64;
    }
    // scipy's center-trim padding (resample_poly source).
    let n_pre_pad = down - (half_len % down);
    let mut n_post_pad = 0usize;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut hp = vec![0.0f64; n_pre_pad];
    hp.extend_from_slice(&h);
    hp.extend(core::iter::repeat(0.0).take(n_post_pad));

    let y = upfirdn(&hp, x, up, down);
    y[n_pre_remove..n_pre_remove + n_out].to_vec()
}

/// Port of `scipy.signal.resample(x, num)` for REAL input — the FFT branch of the
/// 250 Hz resample, taken when the gcd-reduced up/down exceeds 256 (rare, prime-ish
/// rates > 256 Hz). Fourier-domain: `rfft` → resize the spectrum (Nyquist-aware) →
/// `irfft`, scaled by `num/Nx`. Within-noise-floor parity with scipy (FFT
/// transcendentals), exactly like the polyphase branch.
fn resample_fft(x: &[f64], num: usize) -> Vec<f64> {
    let nx = x.len();
    if nx == 0 || num == 0 {
        return vec![0.0; num];
    }
    if num == nx {
        return x.to_vec();
    }
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(nx);
    let mut input = x.to_vec();
    let mut spec = fwd.make_output_vec(); // len nx/2+1
    fwd.process(&mut input, &mut spec).expect("rfft");

    let out_bins = num / 2 + 1;
    let mut y = vec![realfft::num_complex::Complex::new(0.0_f64, 0.0); out_bins];
    let n = num.min(nx);
    let copy = (n / 2 + 1).min(spec.len()).min(out_bins);
    y[..copy].copy_from_slice(&spec[..copy]);
    // scipy scales the `n = min(num, nx)` Nyquist bin (index n/2 in the OUTPUT
    // spectrum) when n is even: downsampling ×2 (folds in the discarded high-freq
    // energy), upsampling ×0.5 (splits energy into the new bin pair).
    if n % 2 == 0 {
        let idx = n / 2;
        if idx < out_bins {
            if num < nx {
                y[idx] *= 2.0;
            } else if num > nx {
                y[idx] *= 0.5;
            }
        }
    }
    // numpy irfft treats the DC and (even-length) output-Nyquist bins as purely
    // real — mirror that so realfft's c2r matches numpy exactly.
    y[0].im = 0.0;
    if num % 2 == 0 {
        y[out_bins - 1].im = 0.0;
    }
    let inv = planner.plan_fft_inverse(num);
    let mut out = inv.make_output_vec(); // len num
    inv.process(&mut y, &mut out).expect("irfft");
    // numpy irfft is 1/num-normalized; realfft's inverse is unnormalized (×num);
    // scipy multiplies by num/Nx → the net scale is 1/Nx.
    let inv_nx = 1.0 / nx as f64;
    for v in out.iter_mut() {
        *v *= inv_nx;
    }
    out
}

/// Resample one channel to 250 Hz — `lma_dataset.py:408-421`. Identity when
/// `|orig_sr − 250| ≤ 0.5`; else gcd-reduce (up=250, down=int(orig_sr)): the
/// polyphase branch ([`resample_poly`]) when both ≤ 256, else the FFT branch
/// ([`resample_fft`], `num = int(Nx·250/orig)`). Never errors now — every rate is
/// handled. (`Result` kept for API stability + `NormalizeError::FftResampleUnsupported`
/// retained but unreachable.)
pub fn resample_to_250(x: &[f64], orig_sr: f64) -> Result<Vec<f64>, NormalizeError> {
    if (orig_sr - TARGET_SR).abs() <= 0.5 {
        return Ok(x.to_vec());
    }
    let up = TARGET_SR as usize; // 250
    let down = orig_sr as usize; // int(orig_sr), truncates (matches Python int())
    let g = gcd(up, down);
    let (u, d) = (up / g, down / g);
    if u > 256 || d > 256 {
        // scipy FFT branch: num = int(Nx · 250 / orig_sr) (int() truncates).
        let num = (x.len() as f64 * TARGET_SR / orig_sr) as usize;
        return Ok(resample_fft(x, num));
    }
    Ok(resample_poly(x, u, d))
}

/// Full EEG normalization for channels already in 21-target order: resample→250
/// → 0.5 Hz zero-phase HP → Q31. `Ok(None)` on an all-flat signal;
/// `Err(FftResampleUnsupported)` when the rate needs the unported FFT branch.
pub fn normalize_eeg(
    data: &[Vec<f64>],
    orig_sr: f64,
) -> Result<Option<Vec<Vec<i32>>>, NormalizeError> {
    let resampled: Vec<Vec<f64>> = data
        .iter()
        .map(|ch| resample_to_250(ch, orig_sr))
        .collect::<Result<_, _>>()?;
    let filtered: Vec<Vec<f64>> = resampled.iter().map(|ch| sosfiltfilt_hp(ch)).collect();
    Ok(q31_normalize(&filtered))
}

/// The Q31 → float32 round-trip that `decode_lma_signal` actually RETURNS
/// (`lma_dataset.py:438`): `(q31.astype(f32) / 2147483647.0) * 1000.0`, computed
/// entirely in f32 (numpy NEP-50: the python scalars are weak, so a float32
/// array stays float32). This is NOT a no-op despite the int32 detour — it
/// reproduces the Q31 fixed-point quantization the DEPLOYED codec applies, so the
/// model trains on the same representation it sees on-device (MiMo #1: the detour
/// is deployment-quantization simulation, not waste). `q as f32` matches numpy's
/// `astype(float32)` round-to-nearest-even, so this is bit-exact to the Python.
pub fn q31_to_signal_f32(q31: &[Vec<i32>]) -> Vec<Vec<f32>> {
    q31.iter()
        .map(|row| {
            row.iter()
                .map(|&q| (q as f32 / 2_147_483_647.0_f32) * 1000.0_f32)
                .collect()
        })
        .collect()
}

/// The full `decode_lma_signal` tail as the model consumes it: resample→250 → HP
/// → Q31 → the float32 round-trip. `Ok(None)` on an all-flat signal. This is the
/// exact array `decode_lma_signal` returns (the int32 is an internal boundary).
pub fn normalize_eeg_signal_f32(
    data: &[Vec<f64>],
    orig_sr: f64,
) -> Result<Option<Vec<Vec<f32>>>, NormalizeError> {
    Ok(normalize_eeg(data, orig_sr)?.map(|q31| q31_to_signal_f32(&q31)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate `sosfiltfilt_hp` against a scipy 1.18.0 oracle:
    /// `sosfiltfilt(butter(2,0.5,fs=250,'high'), arange(20.0))`.
    #[test]
    fn sosfiltfilt_matches_scipy_oracle_on_ramp20() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y = sosfiltfilt_hp(&x);
        assert_eq!(y.len(), 20);
        // scipy 1.18.0 reference (dumped): first 5 and last 3 samples.
        let head = [
            -14.1847655339,
            -13.6226265744,
            -13.0645789879,
            -12.5106270293,
            -11.9607746278,
        ];
        let tail = [-5.1870319854, -4.6948357005, -4.2067679559];
        for (i, &e) in head.iter().enumerate() {
            assert!(
                (y[i] - e).abs() < 1e-9,
                "head[{i}]: got {}, want {e} (Δ={:.3e})",
                y[i],
                (y[i] - e).abs()
            );
        }
        for (k, &e) in tail.iter().enumerate() {
            let i = 20 - 3 + k;
            assert!(
                (y[i] - e).abs() < 1e-9,
                "tail[{k}]: got {}, want {e} (Δ={:.3e})",
                y[i],
                (y[i] - e).abs()
            );
        }
    }

    /// `q31_to_signal_f32` is bit-exact to numpy's
    /// `(q31.astype(f32) / 2147483647.0) * 1000.0` (NEP-50 f32). Reference
    /// values dumped from numpy 2.5.0.
    #[test]
    fn q31_to_signal_f32_matches_numpy() {
        let q31 = vec![vec![123456789i32, -2000000000, 0, 715827882]];
        let out = q31_to_signal_f32(&q31);
        let expect: [f32; 4] = [57.489_048, -931.322_6, 0.0, 333.333_34];
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(
                out[0][i].to_bits(),
                e.to_bits(),
                "sample {i}: {} != {e}",
                out[0][i]
            );
        }
    }

    /// A constant signal → zero-phase HP removes the DC → ~0 → q31 returns None
    /// (all-flat guard), matching `lma_dataset.py`'s `max_abs < 1e-12` branch.
    #[test]
    fn flat_signal_q31_returns_none() {
        let flat = vec![vec![5.0f64; 64]; 3];
        let filtered: Vec<Vec<f64>> = flat.iter().map(|c| sosfiltfilt_hp(c)).collect();
        assert!(q31_normalize(&filtered).is_none());
    }

    /// Validate `resample_poly` against scipy 1.18.0 reference outputs on a
    /// deterministic 64-sample signal (`x[t] = (t*3) % 101 - 50`), for the two
    /// common branches 200→250 (up=5,down=4) and 500→250 (up=1,down=2).
    #[test]
    fn resample_poly_matches_scipy_oracle() {
        let x: Vec<f64> = (0..64).map(|t| ((t * 3) % 101) as f64 - 50.0).collect();

        let y_200 = resample_poly(&x, 5, 4);
        assert_eq!(y_200.len(), 80, "200→250 output length");
        let ref_200 = [-50.0325258, -50.8568344, -42.0090913, -44.9844302];
        for (i, &e) in ref_200.iter().enumerate() {
            assert!(
                (y_200[i] - e).abs() < 1e-6,
                "200→250[{i}]: got {}, want {e} (Δ={:.2e})",
                y_200[i],
                (y_200[i] - e).abs()
            );
        }

        let y_500 = resample_poly(&x, 1, 2);
        assert_eq!(y_500.len(), 32, "500→250 output length");
        let ref_500 = [-37.0421411, -47.4262811, -36.2815836, -33.0847358];
        for (i, &e) in ref_500.iter().enumerate() {
            assert!(
                (y_500[i] - e).abs() < 1e-6,
                "500→250[{i}]: got {}, want {e} (Δ={:.2e})",
                y_500[i],
                (y_500[i] - e).abs()
            );
        }
    }

    /// Tight-tolerance check on the exact parity input (160-sample synth, ch0)
    /// to surface any Rust-specific resample bug the looser oracle missed.
    #[test]
    fn resample_poly_tight_on_parity_input() {
        let x: Vec<f64> = (0..160)
            .map(|t| (((t * 5) % 4001) as i64 - 2000) as f64)
            .collect();
        let y = resample_poly(&x, 5, 4);
        assert_eq!(y.len(), 200);
        let refv = [-2001.301033, -2121.074965, -1868.163585, -2071.997204];
        let mut maxd = 0.0f64;
        for (i, &e) in refv.iter().enumerate() {
            maxd = maxd.max((y[i] - e).abs());
        }
        assert!(
            maxd < 1e-4,
            "resample tight: max|Δ|={maxd:.6e}; y[:4]={:?}",
            &y[..4]
        );
    }

    /// `resample_to_250` is identity within ±0.5 Hz and supports the FFT branch.
    #[test]
    fn resample_to_250_branch_logic() {
        let x: Vec<f64> = (0..32).map(|t| t as f64).collect();
        assert_eq!(resample_to_250(&x, 250.0).unwrap(), x); // exact
        assert_eq!(resample_to_250(&x, 250.4).unwrap(), x); // within tolerance
        assert_eq!(resample_to_250(&x, 500.0).unwrap().len(), 16); // poly branch
                                                                   // 257 Hz: gcd(250,257)=1 → down=257 > 256 → FFT branch.
        assert_eq!(resample_to_250(&x, 257.0).unwrap().len(), 31);
    }

    /// Q31 truncates toward zero (not round), left-to-right f64 multiply.
    #[test]
    fn q31_truncates_toward_zero() {
        // max_abs = 1.0 → gain = 0.72; 0.5 * 0.72 * 2147483647 = 773094113.0-ish
        let data = vec![vec![0.5f64, -0.5, 1.0, -1.0]];
        let q = q31_normalize(&data).unwrap();
        let expect = |v: f64| -> i32 { (v * (0.72 / 1.0) * 2_147_483_647.0) as i32 };
        assert_eq!(q[0][0], expect(0.5));
        assert_eq!(q[0][1], expect(-0.5));
        assert_eq!(q[0][2], expect(1.0));
        assert_eq!(q[0][3], expect(-1.0));
    }
}
