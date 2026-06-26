//! Online **scale-conditioned** adaptive integer range coder for lossless
//! residual entropy coding (ADR 0054 Phase A — the learned-conditional lever,
//! integer/no_std incarnation of `examples/scale_cond_entropy_probe.rs`).
//!
//! The probe proved that conditioning an *online* adaptive order-0 model on the
//! **local scale** of the residual (a causal EMA of `|x|`, log2-bucketed) wins
//! −2 to −3% on `eegmmidb` and −4 to −5% on `ma` over production block-Golomb,
//! reaching the learned-conditional ceiling — and crucially it FIXES the `ma`
//! case where a plain (unconditioned) adaptive order-0 model *lost* +5.8%
//! (its single global model can't track ma's non-stationary scale). Scale is a
//! universal signal property, so the model adapts per-signal with **no
//! transmitted/frozen table** ⇒ it generalizes by construction.
//!
//! This module is the deterministic, integer-only, `no_std` production coder
//! (the bucketed incarnation the parameter sweep selected). Measured on the real
//! held-out residuals (`rls::residual` of the eegmmidb S010/S011 64-ch + ma
//! Subject10/11 21-ch windows) vs production `entropy::encode` (Golomb
//! keep-best): **total −3.25%, ma −3.89%, eeg −1.05%** with the shipped params;
//! the EMA-shift variant `EMA_K_DOWN=2 / EMA_K_UP=0` (the default below) reaches
//! **total −3.58%, ma −4.34%, eeg −0.96%**. The f64 probe's −4.14% was an
//! idealized bit-accounting estimate (KT smoothing over only the distinct-seen
//! symbols, flat-bit escape) that excludes the real range-coder + bucketing
//! overhead, so it overstates the achievable win; the integer coder lands the
//! bulk of it and FIXES the `ma` non-stationary case (its win is largest there).
//!
//!   * **scale context** — a fixed-point integer EMA of `|x|`, `ctx =
//!     bit-length(ema >> EMA_SHIFT)` capped to [`N_CTX`] buckets, computed
//!     CAUSALLY (from the running estimate *before* coding `x`, updated after)
//!     so the decoder reproduces the identical context from already-decoded
//!     symbols. The EMA is **asymmetric** (rises faster than it decays) so it
//!     tracks `ma`'s bursty non-stationary scale without adding noise on the
//!     stationary `eegmmidb` residuals. No floats anywhere.
//!   * **alphabet** — zigzag-map the signed residual `u = (x<<1)^(x>>63)`, then
//!     code its bit-length **bucket** `b = bit_len(u)` (a small adaptive symbol,
//!     `[0, N_BUCKETS)`) per scale context. The `(b-1)` mantissa bits below the
//!     implicit top bit follow: the top [`ABITS`] of them are coded by a second
//!     per-`(ctx, bucket)` adaptive model (capturing the residual fine structure
//!     cheaply), the rest as raw uniform range-coder bits. Bucketing keeps the
//!     adaptive alphabets tiny so the leaky floor never dilutes the model.
//!   * **model** — online adaptive frequency tables (a Fenwick/BIT tree for
//!     O(log A) cumulative + update + symbol-search), incremented after each
//!     symbol with periodic halving rescale to bound the total and keep
//!     adaptation fresh. Range-coded with the same integer Subbotin coder family
//!     as [`crate::arith_int`].
//!
//! Deterministic integer math ⇒ identical encode/decode on host and MCU. Pure
//! `alloc`, no `std`, no `libm`, no `f64`.
//!
//! ## Wire layout
//!
//! ```text
//!   [n: u32]                    number of residual symbols (0 ⇒ stop)
//!   [body]                      raw range-coder byte stream
//! ```
//! Everything else (the per-context models, the EMA, the bucket/mantissa coding)
//! is reconstructed online and identically on both sides; only `n` + the coded
//! body cross the wire.

use alloc::vec;
use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};

// ─── tuned parameters (selected by the held-out parameter sweep) ──────────────

/// Bit-length bucket count for the zigzag value. A zigzag `u64` has bit-length
/// in `0..=64` (bucket 0 = `u==0`, bucket `b>=1` covers `[2^(b-1), 2^b-1]`), so
/// the alphabet is 65 symbols. Tiny ⇒ the leaky floor is negligible.
const N_BUCKETS: usize = 65;

/// Number of mantissa bits (below the implicit top bit of a bucket) coded with a
/// dedicated per-`(ctx, bucket)` adaptive model; the remaining low bits are raw.
/// `ABITS=2` measured best (more makes the small model too sparse to learn).
const ABITS: u32 = 2;
/// Size of the per-`(ctx, bucket)` top-mantissa-bits adaptive model.
const MANT_SYMS: usize = 1 << ABITS;
/// Highest bucket that gets the adaptive top-mantissa-bits treatment. Above this
/// the mantissa is entirely raw (large-magnitude residuals are rare and flat).
const FINE_MAX: usize = 20;

/// Number of scale contexts (log2 buckets of the EMA estimate), capped.
const N_CTX: usize = 16;

/// Fixed-point shift for the EMA accumulator (`ema` stores `|x|` scaled by
/// `1<<EMA_SHIFT`). Keeps fractional precision in the integer EMA recurrence.
const EMA_SHIFT: u32 = 8;
/// EMA decay shift (the SLOW direction): `ema += (target - ema) >> EMA_K_DOWN`.
const EMA_K_DOWN: u32 = 2;
/// EMA rise shift (the FAST direction): `EMA_K_UP < EMA_K_DOWN` so the estimate
/// climbs quickly onto a burst (tracks `ma`'s scale jumps) but decays slowly (no
/// noise on the stationary `eegmmidb` residuals). `0` ⇒ an upward move snaps the
/// estimate straight to `|x|`; measured the strongest total win.
const EMA_K_UP: u32 = 0;

// Range-coder normalization constants (match [`crate::arith_int`]). The adaptive
// model's running total is held `<= MAX_TOTAL` (< 1<<16) so `cum*range` stays
// inside `u32`/`u64`; raw-bit chunks use `tot = 1<<chunk` with `chunk <= 16`.
const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 16;

/// Rescale (halve all counts) when a context's running total reaches this. Kept
/// strictly below `1<<16` so every cumulative frequency is a valid range-coder
/// argument and adaptation stays responsive. `1<<14` (a faster rescale cadence)
/// measured best — it keeps the model tracking the local distribution.
const MAX_TOTAL: u32 = 1 << 14;

/// Per-observation count increment. A larger increment lets a freshly observed
/// symbol dominate the always-`>=1` leaky floor quickly; the periodic halving
/// keeps the model adaptive. `24` measured best on the held-out residuals.
const INC: u32 = 24;

// ─── integer scale context ───────────────────────────────────────────────────

/// Bit-length of a `u64` (0 for 0, else `floor(log2(v)) + 1`).
#[inline]
fn bit_len(v: u64) -> u32 {
    64 - v.leading_zeros()
}

/// Online causal scale-context estimator. `ctx()` is read BEFORE coding `x`;
/// `update()` folds `|x|` in AFTER. Pure integer fixed-point EMA.
struct ScaleCtx {
    ema: u64, // running mean |x|, scaled by 1<<EMA_SHIFT
}

impl ScaleCtx {
    #[inline]
    fn new() -> Self {
        // Seed ema = 1.0 (probe seeded ema=1.0) in fixed point.
        Self { ema: 1u64 << EMA_SHIFT }
    }

    /// Current scale context: log2-bucket of the integer EMA estimate, capped.
    #[inline]
    fn ctx(&self) -> usize {
        let est = self.ema >> EMA_SHIFT; // back to |x| units
        let b = bit_len(est) as usize; // 0 for est==0
        if b >= N_CTX {
            N_CTX - 1
        } else {
            b
        }
    }

    /// Fold `|x|` into the EMA after coding `x` (asymmetric: fast up, slow down).
    #[inline]
    fn update(&mut self, abs_x: u64) {
        let target = (abs_x as i128) << EMA_SHIFT;
        let cur = self.ema as i128;
        let k = if target > cur { EMA_K_UP } else { EMA_K_DOWN };
        self.ema = (cur + ((target - cur) >> k)) as u64;
    }
}

// ─── Fenwick (BIT) adaptive frequency model ──────────────────────────────────

/// An online adaptive frequency model over `[0, size)` backed by a Fenwick tree
/// for O(log size) cumulative-prefix, point-update, and find-by-cumulative. Every
/// symbol starts at count 1 (leaky ⇒ always codable); the running total is held
/// below [`MAX_TOTAL`] by halving on overflow (which re-floors every count to 1).
struct FenwickModel {
    size: usize,
    tree: Vec<u32>,    // 1-indexed BIT over per-symbol counts
    counts: Vec<u32>,  // explicit per-symbol counts (for halving + freq lookup)
    total: u32,
}

impl FenwickModel {
    fn new(size: usize) -> Self {
        let mut m = Self {
            size,
            tree: vec![0u32; size + 1],
            counts: vec![0u32; size],
            total: 0,
        };
        for s in 0..size {
            m.add(s, 1);
        }
        m
    }

    /// Add `delta` to symbol `s` (BIT point update + bookkeeping).
    #[inline]
    fn add(&mut self, s: usize, delta: u32) {
        self.counts[s] += delta;
        self.total += delta;
        let mut i = s + 1;
        while i <= self.size {
            self.tree[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Exclusive prefix sum `cum[s] = sum(counts[0..s])`.
    #[inline]
    fn cum(&self, s: usize) -> u32 {
        let mut i = s; // prefix of length s ⇒ sum tree[1..=s]
        let mut acc = 0u32;
        while i > 0 {
            acc += self.tree[i];
            i -= i & i.wrapping_neg();
        }
        acc
    }

    /// `(cum, freq)` for symbol `s`.
    #[inline]
    fn cum_freq(&self, s: usize) -> (u32, u32) {
        (self.cum(s), self.counts[s])
    }

    /// Smallest `s` with `cum(s+1) > target` — the symbol whose cumulative
    /// interval `[cum(s), cum(s)+freq(s))` contains `target` (`target < total`).
    #[inline]
    fn find(&self, target: u32) -> usize {
        // Standard Fenwick lower-bound: walk bits from high to low.
        let mut pos = 0usize;
        let mut rem = target;
        let mut bit = 1usize << (usize::BITS - 1 - self.size.leading_zeros());
        while bit != 0 {
            let next = pos + bit;
            if next <= self.size && self.tree[next] <= rem {
                pos = next;
                rem -= self.tree[next];
            }
            bit >>= 1;
        }
        pos // pos = number of symbols whose interval ends <= target ⇒ symbol index
    }

    /// Increment symbol `s` after coding it by [`INC`]; halve all counts if the
    /// total would exceed [`MAX_TOTAL`] (keeps every count >= 1 and the total
    /// bounded below `1<<16`).
    #[inline]
    fn update(&mut self, s: usize) {
        if self.total + INC >= MAX_TOTAL {
            self.rescale();
        }
        self.add(s, INC);
    }

    /// Halve every count (floored at 1) and rebuild the BIT.
    fn rescale(&mut self) {
        for c in self.counts.iter_mut() {
            *c = (*c >> 1).max(1);
        }
        // Rebuild the tree from counts.
        for t in self.tree.iter_mut() {
            *t = 0;
        }
        self.total = 0;
        // Build in O(size) using the counts directly.
        for s in 0..self.size {
            let c = self.counts[s];
            self.total += c;
            let mut i = s + 1;
            while i <= self.size {
                self.tree[i] += c;
                i += i & i.wrapping_neg();
            }
        }
    }
}

// ─── Subbotin carryless range coder (mirrors arith_int) ───────────────────────

#[cfg(feature = "encode")]
struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
}

#[cfg(feature = "encode")]
impl RangeEncoder {
    fn new() -> Self {
        Self { low: 0, range: 0xFFFF_FFFF, out: Vec::new() }
    }

    /// Encode a symbol occupying `[cum, cum+freq)` of a `tot`-sized alphabet.
    #[inline]
    fn encode(&mut self, cum: u32, freq: u32, tot: u32) {
        self.range /= tot;
        self.low = self.low.wrapping_add(cum.wrapping_mul(self.range));
        self.range = self.range.wrapping_mul(freq);
        self.normalize();
    }

    #[inline]
    fn normalize(&mut self) {
        while (self.low ^ self.low.wrapping_add(self.range)) < TOP
            || (self.range < BOT && {
                self.range = self.low.wrapping_neg() & (BOT - 1);
                true
            })
        {
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut d = Self { low: 0, range: 0xFFFF_FFFF, code: 0, data, pos: 0 };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte() as u32;
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Cumulative-frequency target in `[0, tot)` for the current symbol.
    #[inline]
    fn decode_freq(&mut self, tot: u32) -> u32 {
        self.range /= tot;
        let v = self.code.wrapping_sub(self.low) / self.range;
        if v >= tot {
            tot - 1
        } else {
            v
        }
    }

    #[inline]
    fn decode_update(&mut self, cum: u32, freq: u32) {
        self.low = self.low.wrapping_add(cum.wrapping_mul(self.range));
        self.range = self.range.wrapping_mul(freq);
        while (self.low ^ self.low.wrapping_add(self.range)) < TOP
            || (self.range < BOT && {
                self.range = self.low.wrapping_neg() & (BOT - 1);
                true
            })
        {
            self.code = (self.code << 8) | self.next_byte() as u32;
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

#[cfg(feature = "encode")]
#[inline]
fn zigzag(x: i64) -> u64 {
    ((x << 1) ^ (x >> 63)) as u64
}

#[inline]
fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Bit-length bucket of a magnitude: 0 for 0, else `floor(log2(m)) + 1`.
#[cfg(feature = "encode")]
#[inline]
fn bucket(m: u64) -> usize {
    bit_len(m) as usize
}

/// Encode `n_bits` raw bits of `value` (LSB-first) as uniform range-coder symbols,
/// at most 16 bits per call (so `tot = 1<<chunk <= 1<<16`). Deterministic.
#[cfg(feature = "encode")]
#[inline]
fn encode_raw_bits(enc: &mut RangeEncoder, value: u64, n_bits: u32) {
    let mut remaining = n_bits;
    let mut v = value;
    while remaining > 0 {
        let chunk = remaining.min(16);
        let tot = 1u32 << chunk;
        let sym = (v & ((tot as u64) - 1)) as u32;
        enc.encode(sym, 1, tot);
        v >>= chunk;
        remaining -= chunk;
    }
}

/// Decode `n_bits` raw bits (LSB-first), mirroring [`encode_raw_bits`].
#[inline]
fn decode_raw_bits(dec: &mut RangeDecoder, n_bits: u32) -> u64 {
    let mut out = 0u64;
    let mut shift = 0u32;
    let mut remaining = n_bits;
    while remaining > 0 {
        let chunk = remaining.min(16);
        let tot = 1u32 << chunk;
        let target = dec.decode_freq(tot);
        // uniform symbols: cum==sym, freq==1
        dec.decode_update(target, 1);
        out |= (target as u64) << shift;
        shift += chunk;
        remaining -= chunk;
    }
    out
}

// ─── encode (host-only) ──────────────────────────────────────────────────────

/// Encode a residual block with the online scale-conditioned adaptive coder.
#[cfg(feature = "encode")]
pub fn encode(res: &[i64]) -> LmlResult<Vec<u8>> {
    let n = res.len();
    let mut out = Vec::with_capacity(4 + n);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    if n == 0 {
        return Ok(out);
    }

    // bucket model per scale context; top-mantissa-bits model per (ctx, bucket).
    let mut bkt: Vec<FenwickModel> = (0..N_CTX).map(|_| FenwickModel::new(N_BUCKETS)).collect();
    let mut mant: Vec<FenwickModel> =
        (0..N_CTX * (FINE_MAX + 1)).map(|_| FenwickModel::new(MANT_SYMS)).collect();
    let mut scale = ScaleCtx::new();
    let mut enc = RangeEncoder::new();

    for &x in res {
        let ctx = scale.ctx();
        let zz = zigzag(x);
        let b = bucket(zz); // 0 for zz==0

        let bm = &mut bkt[ctx];
        let (bcum, bfreq) = bm.cum_freq(b);
        enc.encode(bcum, bfreq, bm.total);
        bm.update(b);

        if b >= 1 {
            // mantissa = the (b-1) bits of zz below the implicit top bit.
            let m = zz - (1u64 << (b - 1));
            let nb = (b - 1) as u32;
            if nb >= 1 && b <= FINE_MAX {
                let take = nb.min(ABITS);
                let top = ((m >> (nb - take)) & ((1u64 << take) - 1)) as usize;
                let mm = &mut mant[ctx * (FINE_MAX + 1) + b];
                let (mcum, mfreq) = mm.cum_freq(top);
                enc.encode(mcum, mfreq, mm.total);
                mm.update(top);
                let restn = nb - take;
                encode_raw_bits(&mut enc, m & ((1u64 << restn) - 1), restn);
            } else {
                encode_raw_bits(&mut enc, m, nb);
            }
        }

        scale.update(x.unsigned_abs());
    }

    let body = enc.finish();
    out.extend_from_slice(&body);
    Ok(out)
}

// ─── decode (no_std) ─────────────────────────────────────────────────────────

/// Decode a slice produced by [`encode`].
pub fn decode(data: &[u8]) -> LmlResult<Vec<i64>> {
    if data.len() < 4 {
        return Err(LmlError::Truncated { expected: 4, actual: data.len(), context: "scale_cond n" });
    }
    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let body = &data[4..];

    let mut bkt: Vec<FenwickModel> = (0..N_CTX).map(|_| FenwickModel::new(N_BUCKETS)).collect();
    let mut mant: Vec<FenwickModel> =
        (0..N_CTX * (FINE_MAX + 1)).map(|_| FenwickModel::new(MANT_SYMS)).collect();
    let mut scale = ScaleCtx::new();
    let mut dec = RangeDecoder::new(body);
    let mut out = Vec::with_capacity(n);

    for _ in 0..n {
        let ctx = scale.ctx();

        let bm = &mut bkt[ctx];
        let btarget = dec.decode_freq(bm.total);
        let b = bm.find(btarget);
        let (bcum, bfreq) = bm.cum_freq(b);
        dec.decode_update(bcum, bfreq);
        bm.update(b);

        let zz = if b == 0 {
            0u64
        } else {
            let nb = (b - 1) as u32;
            let m = if nb >= 1 && b <= FINE_MAX {
                let take = nb.min(ABITS);
                let mm = &mut mant[ctx * (FINE_MAX + 1) + b];
                let mtarget = dec.decode_freq(mm.total);
                let top = mm.find(mtarget);
                let (mcum, mfreq) = mm.cum_freq(top);
                dec.decode_update(mcum, mfreq);
                mm.update(top);
                let restn = nb - take;
                let low = decode_raw_bits(&mut dec, restn);
                ((top as u64) << restn) | low
            } else {
                decode_raw_bits(&mut dec, nb)
            };
            (1u64 << (b - 1)) + m
        };

        let x = unzigzag(zz);
        out.push(x);
        scale.update(x.unsigned_abs());
    }

    Ok(out)
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    fn rt(v: &[i64]) {
        let enc = encode(v).unwrap();
        let dec = decode(&enc).unwrap();
        assert_eq!(dec, v, "scale_cond roundtrip mismatch (len {})", v.len());
    }

    #[test]
    fn roundtrip_edge_cases() {
        rt(&[]); // empty
        rt(&[0]); // single zero
        rt(&[42]); // single positive
        rt(&[-42]); // single negative
        rt(&[0, 0, 0, 0, 0, 0]); // all zero
        rt(&[7, 7, 7, 7, 7]); // all same positive
        rt(&[-9, -9, -9, -9]); // all same negative
        rt(&[5, -5, 5, -5, 5, -5, 5, -5]); // alternating
        rt(&[0, 1, 0, -1, 0, 2, 0, -2]); // small around zero
        // large positive AND negative outliers (escape path, big buckets)
        rt(&[2_000_000, -2_000_000, 0, 1, -1, 2_000_000, -2_000_000]);
        rt(&[i32::MAX as i64, i32::MIN as i64, 0, 5, -5]);
        // CAP boundary values (direct/escape transition)
        rt(&[127, -127, 128, -128, 200, -200, 300, -300]);
        // a value whose zigzag is exactly CAP and CAP-1
        rt(&[128, 127, -128, -127]);
    }

    #[test]
    fn roundtrip_nonstationary_scale() {
        // scale jumps between blocks — exercises the scale context + rescale.
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for n in [255usize, 256, 257, 1000, 5000, 20000] {
            let v: Vec<i64> = (0..n)
                .map(|i| {
                    let scale = if (i / 300) % 2 == 0 { 2i64 } else { 4000 };
                    ((next() % (scale as u64 * 2 + 1)) as i64) - scale
                })
                .collect();
            rt(&v);
        }
    }

    #[test]
    fn roundtrip_fuzz() {
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let n = (next() % 3000) as usize + 1;
            let spread = (next() % 4096) as i64 + 1;
            let bias = (next() % 41) as i64 - 20;
            let v: Vec<i64> = (0..n)
                .map(|_| {
                    let u = (next() % (spread as u64 + 1)) as i64;
                    let sq = (u * u) / (spread + 1);
                    let sign = if next() & 1 == 0 { 1 } else { -1 };
                    sign * sq + bias
                })
                .collect();
            rt(&v);
        }
    }

    #[test]
    fn extreme_outliers_roundtrip() {
        // Full i64 range stress (zigzag of i64::MIN is u64::MAX) — must not panic
        // and must round-trip bit-exact.
        rt(&[i64::MAX, i64::MIN, 0]);
        rt(&[i64::MIN, i64::MIN, i64::MAX, i64::MAX]);
    }
}
