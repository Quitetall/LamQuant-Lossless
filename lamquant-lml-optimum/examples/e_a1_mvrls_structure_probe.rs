//! Research E-A1 (Front A): does the SHIPPED MV-RLS residual retain structure a
//! NONLINEAR predictor could exploit? (The order-1 probe wrongly used the per-channel
//! RLS residual; the container ships MV-RLS.) MV-RLS whitens to LINEAR prediction, so
//! we test for residual structure under: order-1 linear (prev bucket), and NONLINEAR
//! features MV-RLS's linear filter can't capture — sign-correlation, magnitude
//! interaction, and same-instant cross-channel MAGNITUDE coupling (volume-conduction
//! nonlinearity). Conditional-entropy reduction vs order-0 = exploitable structure.
//!
//! GATE: if every reduction ≈ 0, the MV-RLS residual is truly white ⇒ Front A is dead
//! (no predictor helps) ⇒ research = Front B (the neural decoder). Any clear nonlinear
//! reduction ⇒ a nonlinear residual predictor could widen the win ⇒ Front A lives.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example e_a1_mvrls_structure_probe -- <label> <bin> [...]
//! ```

use std::collections::HashMap;
use std::fs;

use lamquant_lml_optimum::mv_rls;

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

fn bit_len(m: u64) -> i32 {
    if m == 0 {
        0
    } else {
        64 - m.leading_zeros() as i32
    }
}
fn mag(x: i64) -> i32 {
    bit_len(x.unsigned_abs()).min(15)
}

fn entropy(h: &HashMap<i64, u64>) -> f64 {
    let n: u64 = h.values().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    -h.values()
        .map(|&c| {
            let p = c as f64 / nf;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Average conditional entropy H(x | ctx) over all (ctx, x), weighted by freq.
fn cond_entropy(pairs: &[(u64, i64)]) -> f64 {
    let mut models: HashMap<u64, HashMap<i64, u64>> = HashMap::new();
    for &(c, x) in pairs {
        *models.entry(c).or_default().entry(x).or_insert(0) += 1;
    }
    let mut bits = 0.0;
    let mut n = 0u64;
    for h in models.values() {
        let cn: u64 = h.values().sum();
        bits += entropy(h) * cn as f64;
        n += cn;
    }
    if n > 0 {
        bits / n as f64
    } else {
        0.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "  {:>14} | {:>6} | {:>7} {:>7} {:>7} {:>7} {:>7}",
        "rec", "H0", "ord1", "sign2", "magX", "ccMag", "best%"
    );
    let mut i = 0;
    while i + 1 < args.len() {
        let label = args[i].clone();
        let path = args[i + 1].clone();
        i += 2;
        let sig = read_bin(&path);
        // shipped residual: MV-RLS config 1 (the faster config that wins ma), seg off.
        let res = mv_rls::residuals(&sig, 1, 0);
        let t = res[0].len();

        // order-0
        let mut h0m: HashMap<i64, u64> = HashMap::new();
        for ch in &res {
            for &x in ch {
                *h0m.entry(x).or_insert(0) += 1;
            }
        }
        let h0 = entropy(&h0m);

        // order-1 linear: ctx = signed prev bucket
        let mut p_ord1 = Vec::new();
        // sign-correlation (nonlinear): ctx = sign(prev)*2 + sign(prev2)
        let mut p_sign2 = Vec::new();
        // magnitude interaction (nonlinear): ctx = mag(prev) + mag(prev2)*16
        let mut p_magx = Vec::new();
        for ch in &res {
            for n in 2..ch.len() {
                let p1 = ch[n - 1];
                let p2 = ch[n - 2];
                let signed_prev = (mag(p1) as u64) * 2 + if p1 < 0 { 1 } else { 0 };
                p_ord1.push((signed_prev, ch[n]));
                let s = (if p1 < 0 { 2u64 } else { 0 }) + (if p2 < 0 { 1 } else { 0 });
                p_sign2.push((s, ch[n]));
                let mx = mag(p1) as u64 + mag(p2) as u64 * 16;
                p_magx.push((mx, ch[n]));
            }
        }
        // cross-channel magnitude coupling (nonlinear): condition x_t^c on the
        // magnitude bucket of the prior channel at the SAME instant (what MV-RLS's
        // linear same-instant cross-channel term can't capture if coupling is in variance).
        let mut p_ccmag = Vec::new();
        for c in 1..res.len() {
            for n in 0..t {
                let cc = mag(res[c - 1][n]) as u64;
                p_ccmag.push((cc, res[c][n]));
            }
        }

        let (e1, es, em, ec) = (
            cond_entropy(&p_ord1),
            cond_entropy(&p_sign2),
            cond_entropy(&p_magx),
            cond_entropy(&p_ccmag),
        );
        let red = |h: f64| 100.0 * (h0 - h) / h0;
        let best = red(e1).max(red(es)).max(red(em)).max(red(ec));
        println!(
            "  {:>14} | {:>6.3} | {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
            label,
            h0,
            red(e1),
            red(es),
            red(em),
            red(ec),
            best
        );
    }
    println!("\n# Reductions vs order-0 H0 on the SHIPPED MV-RLS residual. ord1=linear order-1;");
    println!("# sign2/magX/ccMag = NONLINEAR (sign-corr / magnitude-interaction / cross-channel-magnitude).");
    println!("# GATE: all ≈0 ⇒ MV-RLS residual is white ⇒ Front A (nonlinear lossless predictor) is DEAD");
    println!("# ⇒ pivot research to Front B (neural decoder). A clear nonlinear reduction ⇒ Front A lives.");
}
