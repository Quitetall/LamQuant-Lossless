//! Ternary multiply-accumulate kernels for the TNN encoder.
//!
//! Weight encoding: 2 bits per weight, packed 4 weights per byte.
//!   00 = 0    (zero)
//!   01 = +1
//!   10 = -1
//!   11 = 0    (pad)
//!
//! Two execution paths:
//!   Path 1 (Branchless scalar): 4 weights/call via conditional negate.
//!     ~3 cycles/MAC on Hazard3. Used for n_ch < 32.
//!   Path 2 (Bit-serial CPOP):   32 channels/popcount. Used for wide layers.
//!     Requires bitplane-transposed activations.
//!
//! Boot KAT verifies the LUT/branchless paths agree bit-for-bit before any
//! patient data is processed (Phase 14 Total Verification).
//!
//! All operations are pure integer. No float, no libm. Saturating Q31 used
//! only at LSQ alpha scaling.

/// Q31 multiplication: `(a * b) >> 31`, signed.
#[inline(always)]
fn mul_q31(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 31) as i32
}

// ─── Path 1: Branchless scalar MAC ────────────────────────────────

/// Process 4 ternary weights (one packed byte) against 4 activations.
///
/// Branchless. No LUT. No multiply.
/// Decode: `w = (packed >> (2*i)) & 0b11`
///   neg = -(w >> 1)         # 0 if w in {00, 01}, -1 if w in {10, 11}
///   val = (act XOR neg) - neg   # negate act when w == 10
///   nonzero = (w & 1) ^ (w >> 1) # 1 for {01, 10}, 0 for {00, 11}
///   acc += val if nonzero else 0
#[inline(always)]
pub fn mac_byte_fast(packed_w: u8, act: &[i16; 4]) -> i32 {
    let mut acc = 0i32;
    for i in 0..4 {
        let w = ((packed_w >> (i * 2)) & 0b11) as i32;
        let a = act[i] as i32;
        let neg = -(w >> 1);
        let val = (a ^ neg).wrapping_sub(neg);
        let nonzero = (w & 1) ^ (w >> 1);
        acc = acc.wrapping_add(val & -nonzero);
    }
    acc
}

/// LUT-based reference path — kept for KAT verification. Not used in hot path.
#[inline(always)]
fn mac_byte_lut(packed_w: u8, act: &[i16; 4]) -> i32 {
    const LUT: [i32; 4] = [0, 1, -1, 0];
    let mut acc = 0i32;
    for i in 0..4 {
        let w = ((packed_w >> (i * 2)) & 0b11) as usize;
        acc = acc.wrapping_add((act[i] as i32).wrapping_mul(LUT[w]));
    }
    acc
}

/// Single output channel of a 1D ternary convolution.
///
/// `act_in`: input activations, `in_channels * kernel_size` elements.
/// `packed_weights`: ceil(total/4) bytes, 2-bit ternary packed.
/// `lsq_alpha_q31`: digital gain restoration factor (Q31).
///
/// Returns alpha-scaled accumulator output.
pub fn conv1d_channel(
    act_in: &[i16],
    in_channels: usize,
    kernel_size: usize,
    packed_weights: &[u8],
    lsq_alpha_q31: i32,
) -> i32 {
    let total_weights = in_channels * kernel_size;
    let num_bytes = total_weights / 4;
    let mut accumulator = 0i32;

    // 4-at-a-time inner loop.
    for i in 0..num_bytes {
        let chunk: &[i16; 4] = act_in[i * 4..i * 4 + 4].try_into().unwrap();
        accumulator = accumulator.wrapping_add(mac_byte_fast(packed_weights[i], chunk));
    }

    // Tail: remaining weights (< 4).
    let remainder = total_weights % 4;
    if remainder > 0 {
        const LUT: [i32; 4] = [0, 1, -1, 0];
        let last_byte = packed_weights[num_bytes];
        for j in 0..remainder {
            let w = ((last_byte >> (2 * j)) & 0b11) as usize;
            accumulator = accumulator.wrapping_add(
                (act_in[num_bytes * 4 + j] as i32).wrapping_mul(LUT[w]),
            );
        }
    }

    // Apply LSQ alpha scaling (Q31).
    mul_q31(accumulator, lsq_alpha_q31)
}

// ─── Path 2: Bit-serial CPOP (32 channels per popcount) ───────────

/// Bit-serial ternary dot product via popcount.
///
/// Activations supplied as bitplanes: `bitplanes[bit][word]` has bit `bit`
/// of 32 consecutive activations packed into a u32.
///
/// Weights supplied as `sign_words` (1 = negative) and `mask_words`
/// (1 = nonzero).
///
/// Hazard3 has Zbb (bit manipulation) → `__builtin_popcount` is 1 cycle.
pub fn dot_bitserial(
    bitplanes: &[[u32; 16]],
    sign_words: &[u32],
    mask_words: &[u32],
    n_words: usize,
    act_bits: usize,
) -> i32 {
    let mut result = 0i32;
    for b in 0..act_bits {
        let mut bit_acc = 0i32;
        for w in 0..n_words {
            let act_word = bitplanes[b][w];
            let mask = mask_words[w];
            let sign = sign_words[w];

            let pos = act_word & mask & !sign;
            let neg = act_word & mask & sign;

            bit_acc = bit_acc.wrapping_add(pos.count_ones() as i32);
            bit_acc = bit_acc.wrapping_sub(neg.count_ones() as i32);
        }
        result = result.wrapping_add(bit_acc << b);
    }
    result
}

/// Pack int16 activations into bit-planar format for the CPOP path.
///
/// `bitplanes[b][w]` = bit `b` of `act[w*32 .. w*32+31]` (absolute value).
/// 8 planes (one per activation bit), `n_words = ceil(n_ch / 32)`.
pub fn activations_to_bitplanes(
    act: &[i16],
    n_ch: usize,
    bitplanes: &mut [[u32; 16]],
    n_words: usize,
) {
    // Zero output.
    for plane in bitplanes.iter_mut().take(8) {
        for word in plane.iter_mut().take(n_words) {
            *word = 0;
        }
    }

    for i in 0..n_ch {
        let abs_val = (act[i].unsigned_abs()) as u32;
        let word = i / 32;
        let bit = 1u32 << (i & 31);
        for b in 0..8 {
            if abs_val & (1u32 << b) != 0 {
                bitplanes[b][word] |= bit;
            }
        }
    }
}

/// Convert packed 2-bit ternary weights to sign/mask word format.
pub fn weights_to_signmask(
    packed: &[u8],
    n_weights: usize,
    sign_out: &mut [u32],
    mask_out: &mut [u32],
    n_words: usize,
) {
    for w in 0..n_words {
        sign_out[w] = 0;
        mask_out[w] = 0;
    }

    for i in 0..n_weights {
        let byte_pos = i / 4;
        let bit_pos = (i % 4) * 2;
        let w2 = (packed[byte_pos] >> bit_pos) & 0b11;

        let word = i / 32;
        let bit = 1u32 << (i & 31);

        let nonzero = (w2 & 1) ^ (w2 >> 1); // 1 for 01, 10
        let sign = w2 >> 1; // 1 for 10 (-1)

        if nonzero != 0 {
            mask_out[word] |= bit;
        }
        if sign != 0 {
            sign_out[word] |= bit;
        }
    }
}

// ─── Boot Known-Answer Test (parity sign-off) ─────────────────────

/// Verify the branchless ternary MAC matches the LUT reference bit-for-bit.
/// Mandatory before processing any patient data — catches compiler bitfield
/// rotation or struct-packing surprises.
///
/// Returns `Ok(())` on PASS, `Err(code)` on fatal parity error.
pub fn boot_parity_kat() -> Result<(), i32> {
    let test_act: [i16; 4] = [100, 200, 300, 400];

    // Packed weights: [+1, -1, 0, +1]
    //   pos 0 = 0b01 (+1)
    //   pos 1 = 0b10 (-1)
    //   pos 2 = 0b00 (0)
    //   pos 3 = 0b01 (+1)
    // → 0x01 | (0x02 << 2) | (0x00 << 4) | (0x01 << 6) = 0x49
    let test_packed: u8 = 0x49;

    // Expected: (100·1) + (200·-1) + (300·0) + (400·1) = 300
    let r_fast = mac_byte_fast(test_packed, &test_act);
    let r_lut = mac_byte_lut(test_packed, &test_act);

    if r_fast != 300 {
        return Err(-1);
    }
    if r_lut != 300 {
        return Err(-2);
    }
    if r_fast != r_lut {
        return Err(-3);
    }
    Ok(())
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn kat_passes() {
        boot_parity_kat().unwrap();
    }

    #[test]
    fn fast_matches_lut_on_random() {
        for seed in 0..256 {
            let act = [
                (seed * 31) as i16,
                (seed * 53) as i16,
                (seed * 71) as i16,
                (seed * 97) as i16,
            ];
            let packed = (seed % 256) as u8;
            assert_eq!(mac_byte_fast(packed, &act), mac_byte_lut(packed, &act));
        }
    }

    #[test]
    fn signmask_round_trip_zero_weights() {
        let packed = [0u8, 0u8, 0u8, 0u8];
        let mut sign = [0u32; 1];
        let mut mask = [0u32; 1];
        weights_to_signmask(&packed, 16, &mut sign, &mut mask, 1);
        assert_eq!(sign[0], 0);
        assert_eq!(mask[0], 0);
    }
}
