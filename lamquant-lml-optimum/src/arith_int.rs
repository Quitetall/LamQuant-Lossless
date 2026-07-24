//! no_std **integer** empirical-categorical range coder (ADR 0054 Phase 3,
//! lever-3 stage 3b — the PRODUCTION coder).
//!
//! Stage 3a proved that arithmetic coding of the 9/7 LPC residual is the lever
//! that closes the bulk of the HHI gap (~25% PRD @ 2.0 BPS), but it measured the
//! win with the std-only `constriction`-backed [`arith_cat`]. LMO decode must
//! stay `no_std`-capable, so the production coder is this from-scratch
//! **integer-only** range coder: a classic Subbotin carryless 32-bit range
//! coder driving a static empirical-categorical model (histogram the block, ship
//! a leaky 16-bit fixed-point PMF, range-code against it). It captures the
//! deadzone zero-spike + the real tail with no Golomb-style 1-bit/symbol floor —
//! the same family as HHI's CABAC, minus the float dependency.
//!
//! Two models, mirroring `arith_cat`:
//!   * order-0 ([`encode_dense`]/[`decode_dense`]) — one PMF over the block;
//!   * order-1 context ([`encode_dense_ctx`]/[`decode_dense_ctx`]) — one PMF per
//!     previous-coefficient magnitude bucket (4 contexts), capturing the
//!     significance clustering that dominates at low rates.
//!
//! Deterministic integer math ⇒ identical encode/decode on host and MCU. Pure
//! `alloc`, no `std`, no `libm`. The std `arith_cat` stays the host-side oracle
//! this coder's compression is validated against.
//!
//! ## Wire layout (per subband, self-delimiting)
//!
//! ```text
//!   order-0:  [min:i32][k:u32][n:u32]            (k==0 empty, k==1 constant: stop here)
//!             [k×freq:u16][body_len:u32][body]   (k>=2)
//!   order-1:  [min:i32][k:u32][n:u32]            (k<=1 as above)
//!             [N_CTX×k×freq:u16][body_len:u32][body]   (k>=2)
//! ```
//! For `k>=2` every fixed-point freq is in `1..=TOTAL-(k-1) <= 65535`, so `u16`
//! holds it exactly. The body is the raw range-coder byte stream.

use alloc::vec::Vec;

use lamquant_lml_mcu::error::{LmlError, LmlResult};

/// Cumulative-frequency total (the fixed-point PMF sums to exactly this). 16 bit
/// keeps `cum * range` inside `u32`/`u64` for the 32-bit range coder.
const TOTAL: u32 = 1 << 16;
const TOP: u32 = 1 << 24;
const BOT: u32 = 1 << 16;

/// Max alphabet width we will arithmetic-code. Beyond this the freq table dwarfs
/// any coding gain (and `TOTAL` can't keep every entry `>=1` past 65536); the
/// caller falls back to Golomb/zRLE via keep-smallest. Arithmetic only wins in
/// the low-rate / narrow-alphabet regime, so this loses no measured gain.
const MAX_ALPHABET: usize = 8192;
/// Tighter cap for the context model — `N_CTX` freq tables, so the header grows
/// `N_CTX×` faster; a wide context alphabet never wins keep-smallest.
const MAX_ALPHABET_CTX: usize = 2048;

/// Number of contexts, keyed on the previous coefficient's magnitude bucket
/// (identical bucketing to `arith_cat`, so the two coders model the same source).
const N_CTX: usize = 4;

#[inline]
fn ctx_of(prev: i64) -> usize {
    let a = prev.unsigned_abs();
    if a == 0 {
        0
    } else if a == 1 {
        1
    } else if a <= 4 {
        2
    } else {
        3
    }
}

// ─── leaky fixed-point quantization ───────────────────────────────────────────

/// Quantize raw counts to a fixed-point PMF: every entry `>= 1` ("leaky" — even
/// absent interior symbols stay codable) and the sum is exactly [`TOTAL`].
/// Deterministic integer math ⇒ identical model on both sides.
fn leaky_quantize(counts: &[u64]) -> Vec<u32> {
    let k = counts.len();
    let n: u64 = counts.iter().sum::<u64>().max(1);
    let mut freqs: Vec<u32> = counts
        .iter()
        .map(|&c| (((c as u128 * TOTAL as u128) / n as u128) as u64).max(1) as u32)
        .collect();
    let sum: u64 = freqs.iter().map(|&f| f as u64).sum();
    if sum < TOTAL as u64 {
        // Hand the slack to the largest-count symbol (most slack, no underflow).
        let big = (0..k).max_by_key(|&i| counts[i]).unwrap_or(0);
        freqs[big] += (TOTAL as u64 - sum) as u32;
    } else if sum > TOTAL as u64 {
        let mut excess = sum - TOTAL as u64;
        // Trim from the largest freqs first, keeping every entry >= 1.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| freqs[b].cmp(&freqs[a]));
        for &i in &order {
            if excess == 0 {
                break;
            }
            let removable = (freqs[i] as u64).saturating_sub(1);
            let take = removable.min(excess);
            freqs[i] -= take as u32;
            excess -= take;
        }
    }
    debug_assert_eq!(freqs.iter().map(|&f| f as u64).sum::<u64>(), TOTAL as u64);
    freqs
}

/// Exclusive-prefix cumulative table: `cum[0]=0 .. cum[k]=TOTAL`.
fn cumulative(freqs: &[u32]) -> Vec<u32> {
    let mut cum = Vec::with_capacity(freqs.len() + 1);
    let mut acc = 0u32;
    cum.push(0);
    for &f in freqs {
        acc += f;
        cum.push(acc);
    }
    cum
}

/// Largest `s` with `cum[s] <= target` (the symbol whose interval holds target).
fn symbol_for(cum: &[u32], target: u32) -> usize {
    // cum is strictly increasing; binary-search the half-open interval.
    let mut lo = 0usize;
    let mut hi = cum.len() - 1; // == k
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if cum[mid] <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─── Subbotin carryless range coder ──────────────────────────────────────────

struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            out: Vec::new(),
        }
    }

    #[inline]
    fn encode(&mut self, cum: u32, freq: u32) {
        self.range /= TOTAL;
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
        let mut d = Self {
            low: 0,
            range: 0xFFFF_FFFF,
            code: 0,
            data,
            pos: 0,
        };
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

    /// Cumulative-frequency target in `[0, TOTAL)` for the current symbol.
    #[inline]
    fn decode_freq(&mut self) -> u32 {
        self.range /= TOTAL;
        let v = self.code.wrapping_sub(self.low) / self.range;
        if v >= TOTAL {
            TOTAL - 1
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

// ─── order-0 ─────────────────────────────────────────────────────────────────

/// Header for the `k<=1` (empty / constant) blocks: `[min][k][n]`, 12 bytes.
fn write_trivial_header(mn: i64, k: u32, n: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&(mn as i32).to_le_bytes());
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out
}

fn rd_u32(data: &[u8], pos: usize) -> LmlResult<u32> {
    if pos + 4 > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + 4,
            actual: data.len(),
            context: "arith_int header",
        });
    }
    Ok(u32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

fn rd_u16(data: &[u8], pos: usize) -> LmlResult<u16> {
    if pos + 2 > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + 2,
            actual: data.len(),
            context: "arith_int freq",
        });
    }
    Ok(u16::from_le_bytes([data[pos], data[pos + 1]]))
}

/// Encode a residual block with the integer order-0 empirical-categorical coder.
/// Returns `Err` if the alphabet is too wide to be worth coding (caller falls
/// back via keep-smallest).
pub fn encode_dense(values: &[i64]) -> LmlResult<Vec<u8>> {
    let n = values.len();
    if n == 0 {
        return Ok(write_trivial_header(0, 0, 0));
    }
    let mn = *values.iter().min().unwrap();
    let mx = *values.iter().max().unwrap();
    let k_u128 = (mx as i128 - mn as i128 + 1) as u128;
    if k_u128 > MAX_ALPHABET as u128 {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "arith_int alphabet too wide ({k_u128})"
        )));
    }
    let k = k_u128 as usize;
    if k == 1 {
        // Constant block: header carries everything (decoder fills n×mn).
        return Ok(write_trivial_header(mn, 1, n as u32));
    }

    let mut counts = alloc::vec![0u64; k];
    for &v in values {
        counts[(v - mn) as usize] += 1;
    }
    let freqs = leaky_quantize(&counts);
    let cum = cumulative(&freqs);

    let mut enc = RangeEncoder::new();
    for &v in values {
        let s = (v - mn) as usize;
        enc.encode(cum[s], freqs[s]);
    }
    let body = enc.finish();

    let mut out = Vec::with_capacity(12 + 2 * k + 4 + body.len());
    out.extend_from_slice(&write_trivial_header(mn, k as u32, n as u32));
    for &f in &freqs {
        out.extend_from_slice(&(f as u16).to_le_bytes());
    }
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode an order-0 block written by [`encode_dense`] at `offset`. Returns
/// `(values, bytes_consumed_from_offset)`.
pub fn decode_dense(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    let mn = rd_u32(data, offset)? as i32 as i64;
    let k = rd_u32(data, offset + 4)? as usize;
    let n = rd_u32(data, offset + 8)? as usize;
    if k == 0 || n == 0 {
        return Ok((Vec::new(), 12));
    }
    if k == 1 {
        return Ok((alloc::vec![mn; n], 12));
    }
    let mut pos = offset + 12;
    let mut freqs = Vec::with_capacity(k);
    for _ in 0..k {
        freqs.push(rd_u16(data, pos)? as u32);
        pos += 2;
    }
    let body_len = rd_u32(data, pos)? as usize;
    pos += 4;
    if pos + body_len > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + body_len,
            actual: data.len(),
            context: "arith_int body",
        });
    }
    let cum = cumulative(&freqs);
    let mut dec = RangeDecoder::new(&data[pos..pos + body_len]);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let target = dec.decode_freq();
        let s = symbol_for(&cum, target);
        dec.decode_update(cum[s], freqs[s]);
        out.push(mn + s as i64);
    }
    Ok((out, (pos + body_len) - offset))
}

// ─── order-1 context ─────────────────────────────────────────────────────────

/// Encode a residual block with the integer order-1 context-adaptive coder.
pub fn encode_dense_ctx(values: &[i64]) -> LmlResult<Vec<u8>> {
    let n = values.len();
    if n == 0 {
        return Ok(write_trivial_header(0, 0, 0));
    }
    let mn = *values.iter().min().unwrap();
    let mx = *values.iter().max().unwrap();
    let k_u128 = (mx as i128 - mn as i128 + 1) as u128;
    if k_u128 > MAX_ALPHABET_CTX as u128 {
        return Err(LmlError::InvalidHeader(alloc::format!(
            "arith_int_ctx alphabet too wide ({k_u128})"
        )));
    }
    let k = k_u128 as usize;
    if k == 1 {
        return Ok(write_trivial_header(mn, 1, n as u32));
    }

    let mut counts = alloc::vec![alloc::vec![0u64; k]; N_CTX];
    let mut prev = 0i64;
    for &v in values {
        counts[ctx_of(prev)][(v - mn) as usize] += 1;
        prev = v;
    }
    let freqs: Vec<Vec<u32>> = counts.iter().map(|c| leaky_quantize(c)).collect();
    let cums: Vec<Vec<u32>> = freqs.iter().map(|f| cumulative(f)).collect();

    let mut enc = RangeEncoder::new();
    let mut prev = 0i64;
    for &v in values {
        let c = ctx_of(prev);
        let s = (v - mn) as usize;
        enc.encode(cums[c][s], freqs[c][s]);
        prev = v;
    }
    let body = enc.finish();

    let mut out = Vec::with_capacity(12 + N_CTX * 2 * k + 4 + body.len());
    out.extend_from_slice(&write_trivial_header(mn, k as u32, n as u32));
    for f in &freqs {
        for &x in f {
            out.extend_from_slice(&(x as u16).to_le_bytes());
        }
    }
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode an order-1 context block written by [`encode_dense_ctx`] at `offset`.
pub fn decode_dense_ctx(data: &[u8], offset: usize) -> LmlResult<(Vec<i64>, usize)> {
    let mn = rd_u32(data, offset)? as i32 as i64;
    let k = rd_u32(data, offset + 4)? as usize;
    let n = rd_u32(data, offset + 8)? as usize;
    if k == 0 || n == 0 {
        return Ok((Vec::new(), 12));
    }
    if k == 1 {
        return Ok((alloc::vec![mn; n], 12));
    }
    let mut pos = offset + 12;
    let mut freqs: Vec<Vec<u32>> = Vec::with_capacity(N_CTX);
    for _ in 0..N_CTX {
        let mut f = Vec::with_capacity(k);
        for _ in 0..k {
            f.push(rd_u16(data, pos)? as u32);
            pos += 2;
        }
        freqs.push(f);
    }
    let body_len = rd_u32(data, pos)? as usize;
    pos += 4;
    if pos + body_len > data.len() {
        return Err(LmlError::Truncated {
            expected: pos + body_len,
            actual: data.len(),
            context: "arith_int_ctx body",
        });
    }
    let cums: Vec<Vec<u32>> = freqs.iter().map(|f| cumulative(f)).collect();
    let mut dec = RangeDecoder::new(&data[pos..pos + body_len]);
    let mut out = Vec::with_capacity(n);
    let mut prev = 0i64;
    for _ in 0..n {
        let c = ctx_of(prev);
        let target = dec.decode_freq();
        let s = symbol_for(&cums[c], target);
        dec.decode_update(cums[c][s], freqs[c][s]);
        let v = mn + s as i64;
        out.push(v);
        prev = v;
    }
    Ok((out, (pos + body_len) - offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt0(v: &[i64]) {
        let enc = encode_dense(v).unwrap();
        let (dec, consumed) = decode_dense(&enc, 0).unwrap();
        assert_eq!(dec, v, "order-0 roundtrip mismatch");
        assert_eq!(consumed, enc.len(), "order-0 consumed != encoded length");
    }

    fn rt1(v: &[i64]) {
        let enc = encode_dense_ctx(v).unwrap();
        let (dec, consumed) = decode_dense_ctx(&enc, 0).unwrap();
        assert_eq!(dec, v, "order-1 roundtrip mismatch");
        assert_eq!(consumed, enc.len(), "order-1 consumed != encoded length");
    }

    #[test]
    fn roundtrip_shapes_order0() {
        rt0(&[]);
        rt0(&[0]);
        rt0(&[7, 7, 7, 7]); // constant (k==1)
        rt0(&[-5, -5, -5]); // negative constant
        rt0(&[3, -2, 0, 5, -9, 0, 0, 1]);
        let mut sparse = vec![0i64; 4000];
        sparse[5] = 11;
        sparse[3999] = -4;
        rt0(&sparse);
        let mixed: Vec<i64> = (0..3000).map(|i| ((i * 31) % 17) as i64 - 8).collect();
        rt0(&mixed);
    }

    #[test]
    fn roundtrip_shapes_order1() {
        rt1(&[]);
        rt1(&[0]);
        rt1(&[5, 5, 5]);
        rt1(&[0, 0, 9, 0, 0, -4, 0, 0, 0, 1]);
        let clustered: Vec<i64> = (0..4000)
            .map(|i| {
                if (i / 50) % 2 == 0 {
                    0
                } else {
                    ((i % 7) - 3) as i64
                }
            })
            .collect();
        rt1(&clustered);
    }

    /// Deterministic pseudo-random streams across a range of alphabets/skews —
    /// the range coder must round-trip exactly for every one.
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
            let n = (next() % 2000) as usize + 1;
            let spread = (next() % 64) as i64 + 1;
            let bias = (next() % 21) as i64 - 10;
            let v: Vec<i64> = (0..n)
                .map(|_| {
                    // Skew toward zero (deadzone-like): square the uniform.
                    let u = (next() % (spread as u64 + 1)) as i64;
                    let sq = (u * u) / (spread + 1);
                    let sign = if next() & 1 == 0 { 1 } else { -1 };
                    sign * sq + bias
                })
                .collect();
            rt0(&v);
            rt1(&v);
        }
    }

    #[test]
    fn alphabet_cap_bails_not_panics() {
        // Wider than MAX_ALPHABET ⇒ Err (keep-smallest falls back), never panic.
        let wide: Vec<i64> = (0..(MAX_ALPHABET as i64 + 100)).collect();
        assert!(encode_dense(&wide).is_err());
        let wide_ctx: Vec<i64> = (0..(MAX_ALPHABET_CTX as i64 + 100)).collect();
        assert!(encode_dense_ctx(&wide_ctx).is_err());
    }

    #[test]
    fn beats_golomb_on_skewed_zero_spike() {
        // Deadzone-quantized-like: a big zero spike + a sparse ±1 tail. Empirical
        // categorical should beat Golomb (geometric assumption) here.
        let mut v = vec![0i64; 6000];
        for i in 0..300 {
            v[i * 20] = if i % 3 == 0 { 1 } else { -1 };
        }
        let a = encode_dense(&v).unwrap().len();
        let g = lamquant_lml_mcu::golomb::encode_dense(&v).unwrap().len();
        assert!(
            a < g,
            "arith_int {a} should beat golomb {g} on skewed zero-spike"
        );
    }
}
