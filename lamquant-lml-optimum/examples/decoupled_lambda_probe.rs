//! Decoupled-λ MV-RLS — does separating the temporal vs spatial adaptation RATE
//! help? (the "RLS does too many things at once / HHI clamps each lever" hypothesis)
//!
//! mv_rls runs ONE RLS per channel over [K=8 temporal taps | m spatial taps] with
//! ONE forgetting factor λ. Spatial structure (geometry / volume conduction /
//! reference) is STABLE over a recording; temporal rhythms drift FASTER. A single
//! λ must compromise. H.BWC instead uses separate stages (slow per-block spatial LS
//! + fast per-sample temporal LMS), clamping each lever independently.
//!
//! This probe keeps mv_rls's JOINT estimator (so it retains the cross-term capture
//! that beat SHOT 1 / RICCT) but gives the temporal and spatial coordinate BLOCKS
//! their own forgetting factors via "inflate-first" RLS: before each measurement
//! update, P ← D·P·D with D = diag(1/√λ_coord). For uniform λ this is PROVABLY
//! identical to scalar RLS (sanity row λ_t=λ_s=0.999 reproduces mv_rls cfg-0).
//!
//! GATE: some (λ_t, λ_s) with λ_s > λ_t (longer spatial memory) beats the single-λ
//! baseline on the referential LOSE-set WITHOUT regressing the task WIN-set ⇒ the
//! prediction axis (which I'd called saturated) partially reopens. Round-trip gated.
//!
//! Run UNDER the cap: `ulimit -v 8388608`.
//! cargo run -p lamquant-lml-optimum --features encode --release --example decoupled_lambda_probe -- <bin>...

use std::fs;
use lamquant_lml_optimum::entropy;

const K: usize = 8;          // temporal taps (mv_rls K)
const M: usize = 32;         // cross-channel cap (cfg-0)
const RESET: usize = 8192;   // periodic reset (cfg-0)

fn read_bin(path: &str) -> Vec<Vec<i64>> {
    let b = fs::read(path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut s = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as i64); off += 4; }
        s.push(ch);
    }
    s
}

#[inline]
fn rnd(x: f64) -> i64 { x.round() as i64 }

/// Block-forgetting RLS: coords 0..K forget at λ_t, coords K..n at λ_s.
/// `d[i] = 1/√λ_coord[i]`; inflate P←D·P·D then measurement-update with λ=1.
struct BRls { n: usize, w: Vec<f64>, p: Vec<Vec<f64>>, d: Vec<f64> }
impl BRls {
    fn new(n: usize, lt: f64, ls: f64) -> Self {
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n { p[i][i] = 1.0; }
        let d: Vec<f64> = (0..n).map(|i| 1.0 / (if i < K { lt } else { ls }).sqrt()).collect();
        Self { n, w: vec![0.0; n], p, d }
    }
    fn predict(&self, reg: &[f64]) -> f64 {
        (0..self.n).map(|k| self.w[k] * reg[k]).sum()
    }
    fn adapt(&mut self, reg: &[f64], x: f64, pred: f64) {
        let n = self.n;
        // inflate: P[i][j] *= d[i]*d[j]
        for i in 0..n { for j in 0..n { self.p[i][j] *= self.d[i] * self.d[j]; } }
        // measurement update with λ=1
        let mut px = vec![0.0f64; n];
        for i in 0..n { px[i] = (0..n).map(|j| self.p[i][j] * reg[j]).sum(); }
        let mut denom = 1.0;
        for j in 0..n { denom += reg[j] * px[j]; }
        let inv = 1.0 / denom;
        let e = x - pred;
        for i in 0..n { self.w[i] += px[i] * inv * e; }
        for i in 0..n {
            let ki = px[i] * inv;
            for j in 0..n { self.p[i][j] -= ki * px[j]; }
        }
    }
}

fn regressor(own: &[f64], prior: &[Vec<i64>], refs: &[usize], n: usize) -> Vec<f64> {
    let mut reg = vec![0.0f64; own.len() + refs.len()];
    reg[..own.len()].copy_from_slice(own);
    for (i, &j) in refs.iter().enumerate() { reg[own.len() + i] = prior[j][n] as f64; }
    reg
}

/// Total entropy::encode bytes of the decoupled-λ residual + round-trip flag.
fn encode_bytes(signal: &[Vec<i64>], lt: f64, ls: f64) -> (usize, bool) {
    let n_ch = signal.len();
    let t = signal[0].len();
    let mut tot = 0usize;
    let mut ok = true;
    for c in 0..n_ch {
        let xref = c.min(M);
        let refs: Vec<usize> = (0..xref).map(|r| c - 1 - r).collect();
        let order = K + xref;
        // encode
        let mut rls = BRls::new(order, lt, ls);
        let mut own = vec![0.0f64; K];
        let mut res = Vec::with_capacity(t);
        for n in 0..t {
            if n != 0 && n % RESET == 0 { rls = BRls::new(order, lt, ls); }
            let reg = regressor(&own, signal, &refs, n);
            let pred = rls.predict(&reg);
            res.push(signal[c][n] - rnd(pred));
            rls.adapt(&reg, signal[c][n] as f64, pred);
            for q in (1..K).rev() { own[q] = own[q - 1]; }
            own[0] = signal[c][n] as f64;
        }
        // round-trip (decoder re-runs identical recursion on reconstructed history)
        let mut dec = BRls::new(order, lt, ls);
        let mut downn = vec![0.0f64; K];
        let mut rec = vec![0i64; t];
        for n in 0..t {
            if n != 0 && n % RESET == 0 { dec = BRls::new(order, lt, ls); }
            let reg = regressor(&downn, signal, &refs, n); // refs are prior channels (already exact)
            let pred = dec.predict(&reg);
            rec[n] = res[n] + rnd(pred);
            dec.adapt(&reg, rec[n] as f64, pred);
            for q in (1..K).rev() { downn[q] = downn[q - 1]; }
            downn[0] = rec[n] as f64;
        }
        if rec != signal[c] { ok = false; }
        // entropy::encode per 32768 window (golomb u16 cap) — full-length overflows
        let mut s = 0;
        while s < res.len() {
            let e = (s + 32768).min(res.len());
            tot += entropy::encode(&res[s..e]).map(|g| g.len()).unwrap_or(1 << 30) + 4;
            s = e;
        }
    }
    (tot, ok)
}

fn main() {
    println!("# Decoupled-λ MV-RLS — temporal λ_t vs spatial λ_s (cfg-0: reset=8192, m=32, K=8)\n");
    // λ_s >= λ_t: longer (stable) spatial memory, faster temporal forgetting. Baseline = (0.999,0.999).
    let lt_set = [0.9995, 0.999, 0.997, 0.995];
    let ls_set = [0.999, 0.9995, 0.9998, 1.0];
    for path in std::env::args().skip(1) {
        let sig = read_bin(&path);
        let (nch, t) = (sig.len(), sig[0].len());
        let name = path.rsplit('/').next().unwrap_or(&path);
        let nm = (nch * t) as f64;
        // baseline = single-λ (the mv_rls cfg-0 equivalent)
        let (base, base_ok) = encode_bytes(&sig, 0.999, 0.999);
        println!("## {} ({}ch x {})  single-λ baseline = {} ({:.4} bps) rt={}",
                 name, nch, t, base, base as f64 * 8.0 / nm, if base_ok {"ok"} else {"FAIL"});
        println!("  {:<8} | {:>10} {:>10} {:>10} {:>10}   (cols = λ_s)", "λ_t \\ λ_s", ls_set[0], ls_set[1], ls_set[2], ls_set[3]);
        for &lt in &lt_set {
            print!("  {:<8} |", lt);
            for &ls in &ls_set {
                let (b, ok) = encode_bytes(&sig, lt, ls);
                let d = 100.0 * (b as f64 - base as f64) / base as f64;
                print!(" {:>+9.2}%{}", d, if ok {" "} else {"!"});
            }
            println!();
        }
        println!();
    }
    println!("# negative = decoupled λ beats the single-λ baseline. Win on the referential LOSE-set");
    println!("# (eegmmidb/siena/tusz/tuar) WITHOUT regressing the WIN-set (ma/chb) ⇒ hypothesis confirmed.");
}
