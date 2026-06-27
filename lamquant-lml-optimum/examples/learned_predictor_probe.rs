//! Stage 0b — learned-lossless-predictor headroom + cross-corpus probe.
//!
//! Q1 (headroom): is there structure LEFT in the RLS residual that a learned
//! predictor / context-model could exploit BEYOND scale_cond's order-0? Measure
//! H0 (order-0 entropy, ~the scale_cond ceiling) vs conditional entropy
//! H(x_t | qbucket(x_{t-1}[, x_{t-2}])). If H_cond ≈ H0, the residual is white →
//! NO predictor headroom → a learned predictor is a dead end. If H_cond ≪ H0,
//! a context model could capture it.
//!
//! Q2 (the killer — generalization): FIT a conditional model on corpus A,
//! EVALUATE its cross-entropy on corpus B. If cross-entropy ≫ B's own conditional
//! entropy (like the frozen-table probe's +5.9%), a FROZEN learned model fails
//! cross-corpus → only an online-adaptive form (like scale_cond) generalizes.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example learned_predictor_probe -- <corpusA_label> <A1.bin> [A2.bin ...] -- <corpusB_label> <B1.bin> ...
//! ```
//! Pass two corpora separated by a literal `--` for the cross-corpus test.

use std::collections::HashMap;
use std::fs;

use lamquant_lml_optimum::rls;

fn read_bin(path: &str) -> Vec<Vec<i64>> {
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

/// signed magnitude bucket of a residual: sign · bit_length(|x|) → small alphabet.
fn qbucket(x: i64) -> i32 {
    let m = x.unsigned_abs();
    let b = if m == 0 { 0 } else { 64 - m.leading_zeros() as i32 };
    if x < 0 { -b } else { b }
}

/// All RLS residuals of a corpus, concatenated per channel (kept as channels so
/// the temporal context doesn't cross channel boundaries).
fn corpus_residuals(paths: &[String]) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    for p in paths {
        for ch in read_bin(p) {
            out.push(rls::residual(&ch));
        }
    }
    out
}

fn entropy_of(counts: &HashMap<i64, u64>) -> f64 {
    let n: u64 = counts.values().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    -counts.values().map(|&c| { let p = c as f64 / nf; p * p.log2() }).sum::<f64>()
}

/// Order-0 entropy (bits/symbol) over all residuals.
fn h0(res: &[Vec<i64>]) -> f64 {
    let mut h: HashMap<i64, u64> = HashMap::new();
    let mut total = 0u64;
    for ch in res {
        for &x in ch {
            *h.entry(x).or_insert(0) += 1;
            total += 1;
        }
    }
    let _ = total;
    entropy_of(&h)
}

/// Conditional model P(x | ctx) as per-context histograms. ctx = qbucket of the
/// previous `order` residuals (causal). Returns (avg conditional entropy bits/sym, model).
type CondModel = HashMap<Vec<i32>, HashMap<i64, u64>>;

fn fit_conditional(res: &[Vec<i64>], order: usize) -> (f64, CondModel) {
    let mut model: CondModel = HashMap::new();
    for ch in res {
        if ch.len() <= order {
            continue;
        }
        for t in order..ch.len() {
            let ctx: Vec<i32> = (1..=order).map(|k| qbucket(ch[t - k])).collect();
            *model.entry(ctx).or_default().entry(ch[t]).or_insert(0) += 1;
        }
    }
    // average conditional entropy weighted by context frequency
    let mut bits = 0.0f64;
    let mut n = 0u64;
    for h in model.values() {
        let cn: u64 = h.values().sum();
        bits += entropy_of(h) * cn as f64;
        n += cn;
    }
    (if n > 0 { bits / n as f64 } else { 0.0 }, model)
}

/// Cross-entropy of corpus B's residuals under corpus A's frozen conditional model
/// (bits/sym). Laplace-smoothed; unseen contexts fall back to a global order-0 model.
fn cross_entropy(res_b: &[Vec<i64>], model_a: &CondModel, order: usize) -> f64 {
    // global fallback from the model
    let mut global: HashMap<i64, u64> = HashMap::new();
    for h in model_a.values() {
        for (&x, &c) in h {
            *global.entry(x).or_insert(0) += c;
        }
    }
    let gtot: u64 = global.values().sum::<u64>().max(1);
    let gdist = global.len() as u64;
    let mut bits = 0.0f64;
    let mut n = 0u64;
    for ch in res_b {
        if ch.len() <= order {
            continue;
        }
        for t in order..ch.len() {
            let ctx: Vec<i32> = (1..=order).map(|k| qbucket(ch[t - k])).collect();
            let p = match model_a.get(&ctx) {
                Some(h) => {
                    let tot: u64 = h.values().sum();
                    let d = h.len() as u64;
                    (*h.get(&ch[t]).unwrap_or(&0) as f64 + 1.0) / (tot as f64 + d as f64 + 1.0)
                }
                None => (*global.get(&ch[t]).unwrap_or(&0) as f64 + 1.0) / (gtot as f64 + gdist as f64 + 1.0),
            };
            bits += -p.log2();
            n += 1;
        }
    }
    if n > 0 { bits / n as f64 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--");
    let (a_args, b_args) = match split {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => (&args[..], &args[0..0]),
    };
    let a_label = a_args.first().cloned().unwrap_or_default();
    let a_paths: Vec<String> = a_args[1.min(a_args.len())..].to_vec();
    let res_a = corpus_residuals(&a_paths);

    let h0a = h0(&res_a);
    let (h1a, model1a) = fit_conditional(&res_a, 1);
    let (h2a, _model2a) = fit_conditional(&res_a, 2);
    println!("# Stage-0b: learned-predictor headroom on the RLS residual (bits/sym)");
    println!("  {:<14} | {:>8} {:>8} {:>8} | {:>10} {:>10}", "corpus", "H0", "H|1", "H|2", "head1%", "head2%");
    let row = |lbl: &str, h0v: f64, h1v: f64, h2v: f64| {
        println!(
            "  {:<14} | {:>8.4} {:>8.4} {:>8.4} | {:>9.2}% {:>9.2}%",
            lbl, h0v, h1v, h2v,
            100.0 * (h0v - h1v) / h0v,
            100.0 * (h0v - h2v) / h0v
        );
    };
    row(&a_label, h0a, h1a, h2a);

    if !b_args.is_empty() {
        let b_label = b_args[0].clone();
        let b_paths: Vec<String> = b_args[1.min(b_args.len())..].to_vec();
        let res_b = corpus_residuals(&b_paths);
        let h0b = h0(&res_b);
        let (h1b, _m1b) = fit_conditional(&res_b, 1);
        let (h2b, _m2b) = fit_conditional(&res_b, 2);
        row(&b_label, h0b, h1b, h2b);

        // cross-corpus: A's order-1 model applied to B
        let xeb = cross_entropy(&res_b, &model1a, 1);
        println!("\n# Stage-0b CROSS-CORPUS (the killer test): fit on {a_label}, eval on {b_label}");
        println!("  {b_label} own H|1 = {h1b:.4} bits/sym");
        println!("  {b_label} under {a_label}'s frozen order-1 model = {xeb:.4} bits/sym  ({:+.2}% vs own, {:+.2}% vs B's H0)",
            100.0 * (xeb - h1b) / h1b, 100.0 * (xeb - h0b) / h0b);
        println!("\n# VERDICT");
        println!("# - headroom: if head2% is large (≫ scale_cond's residual), a context-model/predictor");
        println!("#   has real structure to exploit; if ~0, the residual is white → predictor is a dead end.");
        println!("# - generalization: if the frozen cross-corpus cross-entropy is WORSE than B's own H0,");
        println!("#   a FROZEN learned model fails cross-corpus (like the +5.9% probe) → only ONLINE-adaptive");
        println!("#   (scale_cond-style) generalizes ⇒ marginal. Both must pass to justify a learned predictor.");
    }
}
