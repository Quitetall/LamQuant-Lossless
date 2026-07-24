//! Stage-1 pre-build check: does a REALIZABLE online-adaptive ORDER-1 context coder
//! (context = scale × prev-residual-bucket) actually beat scale_cond (scale-only,
//! online) once it pays the adaptation cost of the larger context space? The 0b H|2
//! ceiling (~13% on ma) is idealized; this measures the realizable online delta.
//!
//! Both are online-adaptive (generalize by construction). Bytes are an exact-online
//! adaptive-arithmetic estimate (KT/Laplace, no transmitted model) — the same metric
//! scale_cond's integer range coder reaches to within ~1 byte.
//!
//! ```text
//! cargo run -p lamquant-lml-optimum --features encode --release \
//!   --example order1_context_probe -- <label> <bin> [<label2> <bin2> ...]
//! ```

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

fn bit_len(m: u64) -> i32 {
    if m == 0 {
        0
    } else {
        64 - m.leading_zeros() as i32
    }
}

/// Online-adaptive coder, context = a function of recent state. `ctx_fn` returns the
/// context key BEFORE coding x_t (causal). One Laplace model per context, updated after.
fn adaptive_ctx_bytes<F: FnMut(&CtxState, i64) -> u64>(res: &[i64], mut ctx_fn: F) -> f64 {
    let mut models: HashMap<u64, (HashMap<i64, u64>, u64)> = HashMap::new();
    let mut st = CtxState::default();
    let mut bits = 0.0f64;
    for &x in res {
        let key = ctx_fn(&st, x); // NB: must not depend on x's value (causal); x passed only for type
        let (counts, total) = models.entry(key).or_default();
        let d = counts.len() as u64;
        let c = *counts.get(&x).unwrap_or(&0);
        let p = (c as f64 + 1.0) / (*total as f64 + d as f64 + 1.0);
        bits += -p.log2();
        *counts.entry(x).or_insert(0) += 1;
        *total += 1;
        st.update(x);
    }
    bits / 8.0
}

#[derive(Default)]
struct CtxState {
    ema: f64,  // EMA of |x| (scale)
    prev: i64, // previous residual
    inited: bool,
}
impl CtxState {
    fn scale_bucket(&self) -> u64 {
        let e = if self.inited { self.ema.max(1.0) } else { 1.0 };
        (bit_len(e as u64).min(15)) as u64
    }
    fn prev_bucket(&self) -> u64 {
        let b = bit_len(self.prev.unsigned_abs()).min(15);
        let sign = if self.prev < 0 { 1u64 } else { 0u64 };
        (b as u64) * 2 + sign // signed magnitude bucket, ~32 values
    }
    fn update(&mut self, x: i64) {
        let a = x.unsigned_abs() as f64;
        self.ema = if self.inited {
            0.95 * self.ema + 0.05 * a
        } else {
            a
        };
        self.prev = x;
        self.inited = true;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "  {:>26} | {:>11} {:>11} {:>11} | {:>9} {:>9}",
        "recording", "scaleOnly", "scale×prev", "prevOnly", "s×p vs sc", "best"
    );
    let (mut tsc, mut tsp, mut tpr) = (0f64, 0f64, 0f64);
    let mut i = 0;
    while i + 1 < args.len() {
        let label = args[i].clone();
        let path = args[i + 1].clone();
        i += 2;
        let sig = read_bin(&path);
        let (mut sc, mut sp, mut pr) = (0f64, 0f64, 0f64);
        for ch in &sig {
            let res = rls::residual(ch);
            // scale-only (the scale_cond baseline context)
            sc += adaptive_ctx_bytes(&res, |st, _| st.scale_bucket());
            // scale × prev-bucket (order-1 context)
            sp += adaptive_ctx_bytes(&res, |st, _| st.scale_bucket() * 32 + st.prev_bucket());
            // prev-only (order-1, no scale)
            pr += adaptive_ctx_bytes(&res, |st, _| st.prev_bucket());
        }
        let vs = 100.0 * (sp - sc) / sc;
        let best = sp.min(sc).min(pr);
        let bestlbl = if best == sp {
            "s×p"
        } else if best == sc {
            "scale"
        } else {
            "prev"
        };
        println!(
            "  {:>26} | {:>11.0} {:>11.0} {:>11.0} | {:>8.2}% {:>9}",
            label, sc, sp, pr, vs, bestlbl
        );
        tsc += sc;
        tsp += sp;
        tpr += pr;
    }
    println!(
        "  {:>26} | {:>11.0} {:>11.0} {:>11.0} | {:>8.2}% {:>9}",
        "TOTAL",
        tsc,
        tsp,
        tpr,
        100.0 * (tsp - tsc) / tsc,
        ""
    );
    println!("\n# scale×prev (order-1 online context) vs scaleOnly (the scale_cond baseline). Negative =");
    println!(
        "# the realizable order-1 context coder beats scale_cond. If ~0/positive, the order-1"
    );
    println!("# headroom is eaten by adaptation cost → not worth the production build. keep-best of all 3");
    println!("# is the shippable never-worse design.");
}
