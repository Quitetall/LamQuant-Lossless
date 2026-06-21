//! Integer NLMS adaptive-predictor probe (ADR 0054): HHI's edge is its per-sample
//! adaptive NLMS-16 predictor; ours is block-static LPC. A *normalized* LMS
//! (energy-normalized fixed-point update — the lossless-audio / MPEG-4 ALS
//! approach) should track non-stationary ECG/EEG far better than block LPC, and
//! it is integer-deterministic ⇒ losslessly reversible (the decoder re-runs the
//! same predict+update on the reconstructed history).
//!
//! Measures golomb(NLMS residual) vs the current 5/3+LPC path, with a round-trip
//! verification of the integer NLMS itself.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example nlms_probe -- /tmp/ecg_100.bin
//! ```

use std::fs;

use lamquant_lml_mcu::lpc::LpcMode;
use lamquant_lml_mcu::{golomb, lifting, lml, lpc};

fn read_window(path: &str) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).expect("read");
    let n_ch = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut sig = Vec::with_capacity(n_ch);
    for _ in 0..n_ch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        sig.push(ch);
    }
    sig
}

const WS: u32 = 14; // weight Q-shift
const LEAK: u32 = 13; // weight leakage shift (decay ≈ 1/2^LEAK per step)

/// Integer NLMS. `forward=true`: input is x[n], returns residual e[n] and advances
/// state. `forward=false`: input is e[n], returns reconstructed x[n]. Identical
/// state machine ⇒ decode reproduces encode (lossless).
struct Nlms {
    w: Vec<i64>,
    hist: Vec<i64>,
    order: usize,
    mu_log2: i32,
    min_e_log2: u32,
}

impl Nlms {
    fn new(order: usize, mu_log2: i32, min_e_log2: u32) -> Self {
        Self { w: vec![0i64; order], hist: vec![0i64; order], order, mu_log2, min_e_log2 }
    }
    fn predict(&self) -> i64 {
        let mut acc = 0i64;
        for k in 0..self.order {
            acc += self.w[k] * self.hist[k];
        }
        acc >> WS
    }
    fn update(&mut self, e: i64, x: i64) {
        let mut energy = 1i64;
        for k in 0..self.order {
            energy += self.hist[k] * self.hist[k];
        }
        let mut eshift = (64 - energy.leading_zeros()) as i32; // ≈ log2(energy)+1
        if eshift < self.min_e_log2 as i32 {
            eshift = self.min_e_log2 as i32;
        }
        let sh = eshift - WS as i32 + self.mu_log2;
        for k in 0..self.order {
            let prod = e * self.hist[k];
            let dw = if sh >= 0 { prod >> sh } else { prod << (-sh) };
            // Leakage: decay weights toward 0 (keeps w bounded; on white input it
            // converges to "predict nothing" instead of random-walking).
            self.w[k] += dw - (self.w[k] >> LEAK);
        }
        for k in (1..self.order).rev() {
            self.hist[k] = self.hist[k - 1];
        }
        self.hist[0] = x;
    }
    fn encode(&mut self, x: i64) -> i64 {
        let e = x - self.predict();
        self.update(e, x);
        e
    }
    fn decode(&mut self, e: i64) -> i64 {
        let x = e + self.predict();
        self.update(e, x);
        x
    }
}

fn nlms_residual(x: &[i64], order: usize, mu: i32, mn: u32) -> (Vec<i64>, bool) {
    let mut enc = Nlms::new(order, mu, mn);
    let res: Vec<i64> = x.iter().map(|&v| enc.encode(v)).collect();
    // round-trip verify
    let mut dec = Nlms::new(order, mu, mn);
    let rt = res.iter().map(|&e| dec.decode(e)).collect::<Vec<_>>() == x;
    (res, rt)
}

/// Two-stage cascade: short fast filter then long slow filter (Monkey's-Audio
/// style; the long filter adaptively captures beat-period / long-lag structure).
fn cascade_residual(
    x: &[i64],
    s1: (usize, i32, u32),
    s2: (usize, i32, u32),
) -> (Vec<i64>, bool) {
    let mut e1 = Nlms::new(s1.0, s1.1, s1.2);
    let r1: Vec<i64> = x.iter().map(|&v| e1.encode(v)).collect();
    let mut e2 = Nlms::new(s2.0, s2.1, s2.2);
    let r2: Vec<i64> = r1.iter().map(|&v| e2.encode(v)).collect();
    // round-trip: r2 → r1 → x
    let mut d2 = Nlms::new(s2.0, s2.1, s2.2);
    let r1b: Vec<i64> = r2.iter().map(|&e| d2.decode(e)).collect();
    let mut d1 = Nlms::new(s1.0, s1.1, s1.2);
    let xb: Vec<i64> = r1b.iter().map(|&e| d1.decode(e)).collect();
    (r2, xb == x)
}

fn current_bytes(x: &[i64]) -> usize {
    let (a3, d3, d2, d1) = lifting::forward_3level(x);
    let mut b = 0;
    for (sb, sub) in [a3, d3, d2, d1].iter().enumerate() {
        let scoped = lml::scope_lpc_mode(LpcMode::default(), lml::lpc_max_order(sub.len()));
        let (c, r, _) = lpc::analyze_with_mode(sub, sb, scoped, lml::BIAS_CTX, None);
        b += 1 + 4 * c.len() + golomb::encode_dense(&r).expect("g").len();
    }
    b
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/ecg_100.bin".to_string());
    let w: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30000);
    let sig = read_window(&path);
    let t = sig[0].len().min(w);
    let nm = (sig.len() * t) as f64;

    let cur: usize = sig.iter().map(|f| current_bytes(&f[..t])).sum();
    println!("# NLMS probe: {} ({}ch, {}). current 5/3+LPC = {} B ({:.3} bps)", path, sig.len(), t, cur, cur as f64 * 8.0 / nm);
    println!("  {:<24} {:>9} {:>8} {:>9} {:>5}", "config", "bytes", "Δvs cur", "bps", "rt");
    for &(order, mu, mn) in &[(16usize, 1i32, 14u32), (16, 2, 14), (32, 1, 14), (32, 2, 16), (32, 0, 16), (16, 1, 18)] {
        let mut tot = 0usize;
        let mut allrt = true;
        for f in &sig {
            let (res, rt) = nlms_residual(&f[..t], order, mu, mn);
            allrt &= rt;
            tot += golomb::encode_dense(&res).expect("g").len();
        }
        let d = -100.0 * (cur as f64 - tot as f64) / cur as f64;
        println!("  NLMS o={:<2} mu=2^-{} mn={:<2}    {:>9} {:>+7.1}% {:>9.3} {:>5}", order, mu, mn, tot, d, tot as f64 * 8.0 / nm, if allrt { "ok" } else { "FAIL" });
    }
    println!("# -- cascade (short fast → long slow) --");
    let cascades: [(&str, (usize, i32, u32), (usize, i32, u32)); 6] = [
        ("16/1 → 256/8", (16, 1, 14), (256, 8, 18)),
        ("16/1 → 256/10", (16, 1, 14), (256, 10, 18)),
        ("16/1 → 256/12", (16, 1, 14), (256, 12, 18)),
        ("16/2 → 512/12", (16, 2, 14), (512, 12, 20)),
        ("16/2 → 512/14", (16, 2, 14), (512, 14, 20)),
        ("32/2 → 256/10", (32, 2, 14), (256, 10, 18)),
    ];
    for (name, s1, s2) in cascades {
        let mut tot = 0usize;
        let mut allrt = true;
        for f in &sig {
            let (res, rt) = cascade_residual(&f[..t], s1, s2);
            allrt &= rt;
            tot += golomb::encode_dense(&res).expect("g").len();
        }
        let d = -100.0 * (cur as f64 - tot as f64) / cur as f64;
        println!("  cascade {:<14}       {:>9} {:>+7.1}% {:>9.3} {:>5}", name, tot, d, tot as f64 * 8.0 / nm, if allrt { "ok" } else { "FAIL" });
    }
    println!("# negative Δ = beats current 5/3+LPC. (golomb on residual, no wavelet, no LPC header.)");
}
