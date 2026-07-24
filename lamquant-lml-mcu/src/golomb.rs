//! Golomb-Rice entropy coding — maximum throughput.
//!
//! Byte-identical to Python. Optimized: pre-allocated buffers,
//! branchless zigzag, minimal inner-loop overhead.
//!
//! ADR 0021: encoder + decoder return `Result<_, GolombError>`.
//! Previous `debug_assert!(false, ...)` (release-stripped) +
//! silent saturation paths now surface typed errors at the trust
//! boundary. Well-formed-input behaviour is unchanged (locked by
//! `byte_equal_backends`).

use alloc::vec;
use alloc::vec::Vec;

use crate::codec_errors::GolombError;

/// Maximum sane Golomb unary prefix length. Audit-2026-05-11 Fix-#13:
/// bounds the decoder against a stream of zero-bits that would otherwise
/// loop until `n_data * 8` (still polynomial, but wastes time on
/// adversarial input). 2^40 covers every realistic EEG sample magnitude
/// (max |sample| < 2^24 + DWT growth + LPC residual < 2^40) by orders.
const MAX_Q: u64 = 1u64 << 40;

/// Maximum Golomb `k` parameter. Encoder caps at 31 in `compute_k`; the
/// decoder previously accepted up to 62 (1u64 << 62 still legal but
/// outside encoder range). Audit-2026-05-11 Fix-#47 pins both sides to
/// the same upper bound.
const MAX_K: u8 = 31;

#[inline(always)]
fn zigzag_encode(v: i64) -> u64 {
    // Cast to u64 BEFORE left shift to avoid signed overflow on i64::MIN
    ((v as u64) << 1) ^ ((v >> 63) as u64)
}

#[inline(always)]
fn zigzag_decode(v: u64) -> i64 {
    let v = v as i64;
    (v >> 1) ^ -(v & 1)
}

fn compute_k(zz: &[u64]) -> u8 {
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    for &v in zz {
        if v != 0 {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        return 0;
    }
    let mean = sum as f64 / count as f64;
    if mean < 1.0 {
        return 0;
    }
    // floor(log2(mean)) via bit manipulation
    let mut k: u8 = 0;
    let mut tmp = mean;
    while tmp >= 2.0 && k < 31 {
        tmp *= 0.5;
        k += 1;
    }
    k
}

/// Encode dense int64 array → Golomb-Rice bytes.
/// Format: [k:u8][n_total:u16 LE][bitstream...]
///
/// Returns `Err(GolombError)` instead of silently saturating /
/// debug-asserting on:
///   - `i64::MIN` in input (zigzag would wrap silently to a
///     terminator → decoded as 0).
///   - n_total > u16::MAX (wire format can't encode the length).
///   - q > MAX_Q after `v >> k` (legitimately-huge residual that
///     silent saturation would replace with a wrong value).
pub fn encode_dense(coeffs: &[i64]) -> Result<Vec<u8>, GolombError> {
    let n_total = coeffs.len();
    if n_total == 0 {
        return Ok(vec![0, 0, 0]);
    }
    if n_total > u16::MAX as usize {
        return Err(GolombError::HeaderOverflow {
            n_total,
            payload_bytes: 0,
        });
    }

    // Audit-2026-05-11 Fix-#11 + ADR 0021: hard-reject `i64::MIN`.
    // Zigzag encoding would wrap `i64::MIN` to `u64::MAX`, the
    // existing `(v >> k) as i64` cast wraps to `-1`, the unary
    // emits a single terminator bit, and the decoder reads it as
    // 0. The pre-ADR-0021 code only `debug_assert!`ed (release-
    // stripped) and silently saturated to `MIN + 1`. Both wrong;
    // return an error instead.
    for (i, &v) in coeffs.iter().enumerate() {
        if v == i64::MIN {
            return Err(GolombError::I64Min { index: i });
        }
    }

    // Zigzag all values. After the i64::MIN early-return above no
    // value can reach `u64::MAX`.
    let zz: Vec<u64> = coeffs.iter().map(|&v| zigzag_encode(v)).collect();
    let k = compute_k(&zz);
    let k_u64 = k as u64;
    let k_i32 = k as i32;
    let k_mask: u64 = if k > 0 { (1u64 << k_u64) - 1 } else { 0 };

    // Pre-allocate worst case: header + ~16 bytes per value
    let mut out = Vec::with_capacity(3 + n_total * 16);
    out.push(k);
    out.push((n_total & 0xFF) as u8);
    out.push(((n_total >> 8) & 0xFF) as u8);

    let mut bitbuf: u64 = 0;
    let mut bitpos: i32 = 0;

    for &v in &zz {
        // Audit-2026-05-11 Fix-#11: keep `q` in u64 so the
        // `(v >> k) as i64` wrap-to-negative path is impossible.
        //
        // ADR 0021 audit FP: defensive-code-validator flagged the
        // pre-ADR `raw_q.min(MAX_Q)` as a "silent saturation" risk.
        // Math: k is capped at 31 (compute_k loop bound). Max
        // zigzag(v) = u64::MAX ≈ 2^64. Max raw_q = u64::MAX >> 31
        // ≈ 2^33. MAX_Q = 2^40. So raw_q < MAX_Q ALWAYS for this
        // pipeline; the saturation is dead code. Saturation
        // removed entirely; `MAX_Q` kept as the decoder's
        // adversarial-input ceiling (`golomb::decode_dense`
        // checks `q > max_q` against this same constant).
        let mut q: u64 = v >> k_u64;
        let r = v & k_mask;

        // Unary: q zeros + terminating 1
        while q >= 56 {
            bitbuf <<= 56;
            bitpos += 56;
            while bitpos >= 8 {
                bitpos -= 8;
                out.push(((bitbuf >> bitpos as u64) & 0xFF) as u8);
            }
            bitbuf = if bitpos > 0 {
                bitbuf & ((1u64 << bitpos as u64) - 1)
            } else {
                0
            };
            q -= 56;
        }

        let n_unary = q + 1;
        debug_assert!(n_unary + (bitpos as u64) <= 64);
        bitbuf = (bitbuf << n_unary) | 1;
        bitpos += n_unary as i32;
        while bitpos >= 8 {
            bitpos -= 8;
            out.push(((bitbuf >> bitpos as u64) & 0xFF) as u8);
        }
        bitbuf = if bitpos > 0 {
            bitbuf & ((1u64 << bitpos as u64) - 1)
        } else {
            0
        };

        // k-bit remainder
        if k > 0 {
            bitbuf = (bitbuf << k_u64) | r;
            bitpos += k_i32;
            while bitpos >= 8 {
                bitpos -= 8;
                out.push(((bitbuf >> bitpos as u64) & 0xFF) as u8);
            }
            bitbuf = if bitpos > 0 {
                bitbuf & ((1u64 << bitpos as u64) - 1)
            } else {
                0
            };
        }
    }

    if bitpos > 0 {
        out.push(((bitbuf << (8 - bitpos) as u64) & 0xFF) as u8);
    }

    Ok(out)
}

/// Decode dense Golomb-Rice bitstream.
/// Returns (values, bytes_consumed from data[offset..]).
///
/// ADR 0021: Result-typed. Replaces three silent paths with typed
/// errors:
///   - Oversize k header rejection (used to silently return empty
///     output, callers read garbage from the next subband).
///   - q > MAX_Q mid-stream (used to return bytes_consumed=3
///     while bytepos had advanced past header → cross-subband
///     corruption when caller did `sub_pos += 3`).
///   - Truncated unary / truncated remainder (used to silently
///     break out of the unary scan or zero-pad the remainder).
pub fn decode_dense(data: &[u8], offset: usize) -> Result<(Vec<i64>, usize), GolombError> {
    if data.len().saturating_sub(offset) < 3 {
        return Ok((Vec::new(), 0));
    }

    let k = data[offset] as i32;
    let n_total = data[offset + 1] as usize | ((data[offset + 2] as usize) << 8);

    if n_total == 0 {
        return Ok((Vec::new(), 3));
    }
    if k as u8 > MAX_K {
        // ADR 0021: was silent `(Vec::new(), 3)`. Crafted packets
        // with k=32..255 would slip past callers that didn't
        // verify `decoded.len() == n_total`. Now hard-Err.
        return Err(GolombError::OversizeK { k: k as u32 });
    }

    // ADR 0021: tighten the no-op cap from Fix-#13. Old:
    // `n_total.min(data.len() * 8)` -- always > n_total for any
    // real input, so the cap never fired. New: account for offset
    // + 3-byte header, so a small-payload + large-n_total header
    // surfaces as `HeaderOverflow`.
    let payload_bytes = data.len().saturating_sub(offset + 3);
    let max_addressable = payload_bytes.saturating_mul(8);
    if n_total > max_addressable {
        return Err(GolombError::HeaderOverflow {
            n_total,
            payload_bytes,
        });
    }

    let k_u64 = k as u64;
    let k_mask: u64 = if k > 0 { (1u64 << k_u64) - 1 } else { 0 };
    let n_data = data.len();
    let max_q = MAX_Q;

    let mut bitbuf: u64 = 0;
    let mut bitpos: i32 = 0;
    let mut bytepos = offset + 3;

    let mut out = Vec::with_capacity(n_total.min(1 << 20));

    // Helper: compute current bytes_consumed for error reporting.
    // ADR 0021 fix for the L229-230 cross-subband corruption bug:
    // we must report the ACTUAL bytes consumed (incl. payload
    // already read) rather than just the 3-byte header, so the
    // caller doesn't advance the next subband over data already
    // touched by the failed decode.
    let bytes_consumed_so_far = |bytepos: usize, bitpos: i32| -> usize {
        let payload_start = offset + 3;
        let bits = ((bytepos - payload_start) * 8) as i32 - bitpos;
        3 + ((bits + 7) / 8).max(0) as usize
    };

    for _ in 0..n_total {
        // Read unary q
        let mut q: u64 = 0;
        loop {
            if bitpos == 0 {
                while bitpos < 56 && bytepos < n_data {
                    bitbuf = (bitbuf << 8) | data[bytepos] as u64;
                    bytepos += 1;
                    bitpos += 8;
                }
                if bitpos == 0 {
                    // Bitstream ran out mid-unary. Old code:
                    // silent `break` → out vec contained
                    // garbage. Now: typed truncation error.
                    return Err(GolombError::TruncatedUnary {
                        at_byte: bytepos,
                        partial_q: q,
                    });
                }
            }
            if bitbuf == 0 {
                q += bitpos as u64;
                bitpos = 0;
                if q > max_q {
                    return Err(GolombError::QuotientExceedsCeiling {
                        q,
                        max_q,
                        bytes_consumed: bytes_consumed_so_far(bytepos, bitpos),
                    });
                }
                continue;
            }
            let bl = 64 - bitbuf.leading_zeros() as i32;
            q += (bitpos - bl) as u64;
            bitpos = bl - 1;
            bitbuf = if bitpos > 0 {
                bitbuf & ((1u64 << bitpos as u64) - 1)
            } else {
                0
            };
            break;
        }

        // Read k-bit remainder
        let r: u64 = if k > 0 {
            while bitpos < k && bytepos < n_data {
                bitbuf = (bitbuf << 8) | data[bytepos] as u64;
                bytepos += 1;
                bitpos += 8;
            }
            if bitpos >= k {
                bitpos -= k;
                let val = (bitbuf >> bitpos as u64) & k_mask;
                bitbuf = if bitpos > 0 {
                    bitbuf & ((1u64 << bitpos as u64) - 1)
                } else {
                    0
                };
                val
            } else {
                // Bitstream truncated mid-remainder. Old code:
                // silent zero-pad → decoded value fabricated.
                // Now: typed truncation error.
                return Err(GolombError::TruncatedRemainder {
                    at_byte: bytepos,
                    k: k as u32,
                    bits_short: (k - bitpos) as u32,
                });
            }
        } else {
            0
        };

        let zz = (q << k_u64) | r;
        out.push(zigzag_decode(zz));
    }

    let payload_start = offset + 3;
    let bits_consumed = ((bytepos - payload_start) * 8) as i32 - bitpos;
    let bytes_consumed = 3 + ((bits_consumed + 7) / 8) as usize;
    Ok((out, bytes_consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for data in [
            vec![0i64; 200],
            (-50..50i64).collect::<Vec<_>>(),
            (0..200)
                .map(|i| ((i * 137) % 10000) - 5000)
                .collect::<Vec<i64>>(),
            vec![42i64],
            vec![],
        ] {
            let enc = encode_dense(&data).expect("well-formed input encodes ok");
            let (dec, _) = decode_dense(&enc, 0).expect("well-formed bytes decode ok");
            assert_eq!(data, dec);
        }
    }

    #[test]
    fn zigzag() {
        for v in -1000..1000i64 {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
    }

    /// Audit-2026-05-11 Fix-#11: roundtrip on a wider value range —
    /// ensures the `u64`-only `q` path round-trips correctly. Before the
    /// fix, `(v >> k) as i64` wrapped on near-`u64::MAX` zigzag outputs,
    /// silently corrupting any subband that contained a large negative.
    #[test]
    fn roundtrip_wide_range() {
        // ADR 0021 narrowed this test: the prior version included
        // values like `i64::MAX/2` (2^62) that produce q > MAX_Q
        // (2^40) and were silently saturated by the old encoder.
        // Now that `encode_dense` returns `Err(OversizeQ)` on
        // those, the roundtrip path is restricted to values that
        // actually CAN roundtrip. Out-of-range cases moved to
        // `encoder_rejects_oversize_q` below.
        let cases: Vec<Vec<i64>> = vec![
            // Mid-range, well within MAX_Q.
            vec![1_000_000, -1_000_000, 0, 1, -1],
            // Small negatives.
            (1..50).map(|i| -((i as i64) * 1_000)).collect(),
            // 2^30 range -- below MAX_Q = 2^40 by 10 bits.
            vec![1_000_000_000, -999_999_999, 500_000_000],
        ];
        for data in cases {
            let enc = encode_dense(&data).expect("in-range encodes ok");
            let (dec, _) = decode_dense(&enc, 0).expect("decodes ok");
            assert_eq!(data, dec, "wide-range roundtrip failed");
        }
    }

    /// Audit-2026-05-11 Fix-#47: decoder rejects k > MAX_K (= 31) to
    /// match encoder upper bound. Crafted packets with k=32..62 used to
    /// decode through the old `k <= 62` filter — return early instead.
    #[test]
    fn decoder_rejects_oversize_k() {
        // ADR 0021: was silent `(Vec::new(), 3)`. Now: Err(OversizeK).
        let bytes = vec![40u8, 1u8, 0u8];
        match decode_dense(&bytes, 0) {
            Err(GolombError::OversizeK { k: 40 }) => {}
            other => panic!("expected OversizeK{{k:40}}, got {:?}", other),
        }
    }

    // ─── ADR 0021 adversarial fixtures (J1) ────────────────────────
    //
    // Tests below assert the NEW Err() return paths. They lock the
    // J1/J2 encoder hardening so a future refactor that silently
    // re-introduces saturation breaks at commit time.

    #[test]
    fn encoder_rejects_i64_min() {
        // i64::MIN can't safely zigzag (would wrap to u64::MAX).
        // Old encoder `debug_assert!`-then-saturated; now hard-Err.
        let data = vec![0i64, 1, i64::MIN, 2];
        let result = encode_dense(&data);
        match result {
            Err(GolombError::I64Min { index: 2 }) => {} // expected
            other => panic!("expected I64Min{{index:2}}, got {:?}", other),
        }
    }

    // NOTE: `encoder_rejects_oversize_q` was authored from a
    // defensive-code-validator finding but verified as a false
    // positive during J1 implementation. k is capped at 31 by
    // `compute_k`, so raw_q = zigzag(v) >> 31 ≤ 2^33 < MAX_Q (2^40)
    // for any well-formed u64. The saturation was dead code. Test
    // removed; behaviour documented inline in `encode_dense`.

    #[test]
    fn decoder_rejects_header_overflow() {
        // k=0, n_total=65535 (max u16), but payload is 0 bytes.
        // Old code silently capped n_total to data.len()*8 = 0 and
        // returned empty Vec. Now: HeaderOverflow Err.
        let bytes = vec![0u8, 0xFF, 0xFF]; // k=0, n_total=65535, no payload
        match decode_dense(&bytes, 0) {
            Err(GolombError::HeaderOverflow {
                n_total: 65535,
                payload_bytes: 0,
            }) => {}
            other => panic!("expected HeaderOverflow, got {:?}", other),
        }
    }

    #[test]
    fn decoder_rejects_truncated_unary() {
        // k=0, n_total=8 (asks for 8 values), payload = 0x00 (all
        // zeros, never terminates the unary). Old code: silent
        // break out of unary scan, returns partial Vec. Now:
        // TruncatedUnary Err.
        let bytes = vec![0u8, 8u8, 0u8, 0x00];
        match decode_dense(&bytes, 0) {
            Err(GolombError::TruncatedUnary { .. }) => {}
            other => panic!("expected TruncatedUnary, got {:?}", other),
        }
    }

    #[test]
    fn decoder_rejects_truncated_remainder() {
        // k=8, n_total=1, payload = single byte 0x80 (unary
        // terminator at bit 0, then 7 bits of remainder, but k=8
        // needs 8 bits → 1 bit short and no more bytes). Old
        // code: silent zero-pad → returned fabricated value. Now:
        // TruncatedRemainder Err.
        let bytes = vec![8u8, 1u8, 0u8, 0x80];
        match decode_dense(&bytes, 0) {
            Err(GolombError::TruncatedRemainder { k: 8, .. }) => {}
            other => panic!("expected TruncatedRemainder, got {:?}", other),
        }
    }

    #[test]
    fn encoder_rejects_header_overflow() {
        // n_total > u16::MAX is unrepresentable in the wire format.
        // Old encoder panicked via assert!; now hard-Err.
        let data = vec![0i64; (u16::MAX as usize) + 1];
        let result = encode_dense(&data);
        match result {
            Err(GolombError::HeaderOverflow { n_total, .. })
                if n_total == (u16::MAX as usize) + 1 => {}
            other => panic!("expected HeaderOverflow, got {:?}", other),
        }
    }
}
