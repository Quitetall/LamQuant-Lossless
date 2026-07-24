//! Guaranteed-bound near-lossless on the MV-RLS residual — the "is there a regime
//! we beat H.BWC" probe (the σ/δ angle).
//!
//! H.BWC has NO guaranteed per-sample error mode: to bound max|error| ≤ δ it must
//! sweep QP finer and over-allocate bits. Our `BoundedMae(δ)` gives a HARD bound.
//! But our SHIPPED near-lossless rides the LML floor (~10 bps lossless), a 2×
//! handicap vs the Optimum keep-best (~5.75). This probe measures the POTENTIAL
//! near-lossless built on the mv_rls residual instead.
//!
//! Closed-loop near-lossless: quantize the residual r to the (2δ+1)-grid,
//! q = round(r/(2δ+1)); reconstruct x̂ = pred + (2δ+1)·q ⇒ |x − x̂| ≤ δ GUARANTEED
//! by construction. Code q losslessly. bits ≈ entropy::encode(q). (We use the
//! open-loop mv_rls residual as a tight estimate of the closed-loop one — same
//! statistics; this is a go/no-go on the bits, not a shipping encoder.)
//!
//! Reports bps vs δ. δ=0 == Optimum lossless (sanity ~5.75). Compare each δ to
//! H.BWC's min-bits config that achieves max|err| ≤ δ (measured separately).
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example near_lossless_mvrls_probe -- <bin>...

use lamquant_lml_optimum::{entropy, mv_rls};
use std::fs;

const W: usize = 32768;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t {
            ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64);
            off += 4;
        }
        s.push(ch);
    }
    s
}

fn entropy_bytes(q: &[i64]) -> usize {
    let mut tot = 0;
    let mut s = 0;
    while s < q.len() {
        let e = (s + W).min(q.len());
        tot += entropy::encode(&q[s..e])
            .map(|g| g.len())
            .unwrap_or(1 << 30)
            + 4;
        s = e;
    }
    tot
}

// --- TRUE closed-loop mv_rls near-lossless (real bits + GUARANTEE verification) ---
const K: usize = 8;
const M: usize = 32;
const RESET: usize = 8192;
struct Rls {
    n: usize,
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
    lambda: f64,
}
impl Rls {
    fn new(n: usize, lambda: f64) -> Self {
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            p[i][i] = 1.0;
        }
        Self {
            n,
            w: vec![0.0; n],
            p,
            lambda,
        }
    }
    fn predict(&self, r: &[f64]) -> f64 {
        (0..self.n).map(|k| self.w[k] * r[k]).sum()
    }
    fn adapt(&mut self, r: &[f64], x: f64, pred: f64) {
        let n = self.n;
        let mut px = vec![0.0; n];
        for i in 0..n {
            px[i] = (0..n).map(|j| self.p[i][j] * r[j]).sum();
        }
        let mut den = self.lambda;
        for j in 0..n {
            den += r[j] * px[j];
        }
        let inv = 1.0 / den;
        let e = x - pred;
        for i in 0..n {
            self.w[i] += px[i] * inv * e;
        }
        let il = 1.0 / self.lambda;
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n {
                self.p[i][j] = (self.p[i][j] - ki * px[j]) * il;
            }
        }
    }
}

/// Closed-loop: predictor adapts on RECONSTRUCTED values; cross-channel refs read
/// reconstructed prior channels. Returns (bytes, max_err). max_err MUST be ≤ δ.
fn closed_loop_nl(sig: &[Vec<i64>], delta: i64) -> (usize, i64) {
    let grid = 2 * delta + 1;
    let (nch, t) = (sig.len(), sig[0].len());
    let mut xhat: Vec<Vec<i64>> = Vec::with_capacity(nch); // reconstructed channels
    let mut max_err = 0i64;
    let mut bytes = 0usize;
    for c in 0..nch {
        let xref = c.min(M);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        let mut rls = Rls::new(order, 0.999);
        let mut own = vec![0.0f64; K]; // reconstructed own history
        let mut q_res = Vec::with_capacity(t);
        let mut rec = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % RESET == 0 {
                rls = Rls::new(order, 0.999);
            } // reset RLS only (mv_rls keeps own)
            let mut reg = vec![0.0f64; order];
            reg[..K].copy_from_slice(&own);
            for (i, &j) in refs.iter().enumerate() {
                reg[K + i] = xhat[j][n] as f64;
            } // reconstructed refs
            let pred = rls.predict(&reg);
            let pr = pred.round() as i64;
            let r = sig[c][n] - pr;
            let q = (r as f64 / grid as f64).round() as i64;
            let xh = pr + grid * q; // reconstruction
            let err = (sig[c][n] - xh).abs();
            if err > max_err {
                max_err = err;
            }
            q_res.push(q);
            rec.push(xh);
            rls.adapt(&reg, xh as f64, pred); // adapt on reconstructed
            for k in (1..K).rev() {
                own[k] = own[k - 1];
            }
            own[0] = xh as f64;
        }
        bytes += entropy_bytes(&q_res);
        xhat.push(rec);
    }
    (bytes, max_err)
}

fn main() {
    println!(
        "# Guaranteed-bound near-lossless on the MV-RLS residual (δ = max per-sample error)\n"
    );
    let deltas = [0i64, 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64];
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let nm = (nch * t) as f64;
        let res = mv_rls::residuals(&sig, 0, 0); // shipped Optimum residual
        let name = path.rsplit('/').next().unwrap_or(&path);
        println!("## {} ({}ch x {})", name, nch, t);
        println!(
            "  {:>4} | {:>5} | {:>10} {:>9} | {:>10} {:>9} {:>7}",
            "δ", "grid", "OL-bytes", "OL-bps", "CL-bytes", "CL-bps", "maxErr"
        );
        for &d in &deltas {
            let grid = (2 * d + 1) as f64;
            let mut ol = 0usize;
            for ch in &res {
                let q: Vec<i64> = ch
                    .iter()
                    .map(|&r| (r as f64 / grid).round() as i64)
                    .collect();
                ol += entropy_bytes(&q);
            }
            let (cl, me) = closed_loop_nl(&sig, d);
            let guard = if me <= d { "ok" } else { "VIOLATED!" };
            println!(
                "  {:>4} | {:>5} | {:>10} {:>9.4} | {:>10} {:>9.4} {:>4}{}",
                d,
                2 * d + 1,
                ol,
                ol as f64 * 8.0 / nm,
                cl,
                cl as f64 * 8.0 / nm,
                me,
                guard
            );
        }
        println!();
    }
    println!("# δ=0 == Optimum lossless. Compare each δ-row bps to H.BWC's smallest bitstream");
    println!("# that achieves max|err| ≤ δ. If we are LOWER at small δ, the guaranteed-bound");
    println!("# near-lossless regime is a real LamQuant win (a capability H.BWC lacks entirely).");
}
