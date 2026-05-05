//! 32-point Walsh-Hadamard Transform (Stage 5C of pipeline).
//!
//! Applied per-timestep to the 32-element latent vector before FSQ.
//! Decorrelates latent dimensions so per-dimension uniform quantization
//! is closer to optimal.
//!
//! Math: orthogonal transform, additions only — no multiplications.
//!   - 32-point WHT: 80 add/sub, 0 mul (5 stages × 16 butterflies)
//!   - Compute: ~80 cycles @ 150 MHz = 0.5 µs (negligible)
//!   - Self-inverse up to scale: H · H = N · I → inverse = forward; >> 5

pub const WHT_N: usize = 32;

/// Forward 32-point WHT in-place. 80 add/sub, log2(32)=5 stages.
#[inline]
pub fn forward(x: &mut [i32; WHT_N]) {
    // Stage 1: stride 1
    let mut i = 0;
    while i < 32 {
        let a = x[i];
        let b = x[i + 1];
        x[i] = a + b;
        x[i + 1] = a - b;
        i += 2;
    }
    // Stage 2: stride 2
    let mut i = 0;
    while i < 32 {
        let a0 = x[i];
        let a1 = x[i + 1];
        let b0 = x[i + 2];
        let b1 = x[i + 3];
        x[i] = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a0 - b0;
        x[i + 3] = a1 - b1;
        i += 4;
    }
    // Stage 3: stride 4
    let mut i = 0;
    while i < 32 {
        for j in 0..4 {
            let a = x[i + j];
            let b = x[i + j + 4];
            x[i + j] = a + b;
            x[i + j + 4] = a - b;
        }
        i += 8;
    }
    // Stage 4: stride 8
    let mut i = 0;
    while i < 32 {
        for j in 0..8 {
            let a = x[i + j];
            let b = x[i + j + 8];
            x[i + j] = a + b;
            x[i + j + 8] = a - b;
        }
        i += 16;
    }
    // Stage 5: stride 16
    for j in 0..16 {
        let a = x[j];
        let b = x[j + 16];
        x[j] = a + b;
        x[j + 16] = a - b;
    }
}

/// Inverse 32-point WHT in-place. WHT is self-inverse up to scale: H·H = N·I.
/// So inverse = forward followed by `>> 5` (divide by N=32).
#[inline]
pub fn inverse(x: &mut [i32; WHT_N]) {
    forward(x);
    for v in x.iter_mut() {
        *v >>= 5;
    }
}

/// Apply forward WHT to all timesteps of a `[32][T]` latent tensor.
///
/// Latent is column-major in firmware: `latent[d][t]`. For each timestep
/// gather across dimensions, transform, scatter back.
pub fn apply_latent<const T: usize>(latent: &mut [[i32; T]; WHT_N]) {
    let mut tmp = [0i32; WHT_N];
    for t in 0..T {
        for d in 0..WHT_N {
            tmp[d] = latent[d][t];
        }
        forward(&mut tmp);
        for d in 0..WHT_N {
            latent[d][t] = tmp[d];
        }
    }
}

/// Apply inverse WHT to all timesteps (decoder side).
pub fn inverse_latent<const T: usize>(latent: &mut [[i32; T]; WHT_N]) {
    let mut tmp = [0i32; WHT_N];
    for t in 0..T {
        for d in 0..WHT_N {
            tmp[d] = latent[d][t];
        }
        inverse(&mut tmp);
        for d in 0..WHT_N {
            latent[d][t] = tmp[d];
        }
    }
}

// ─── Tests (host only) ────────────────────────────────────────────

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_random() {
        let mut x = [0i32; WHT_N];
        for i in 0..WHT_N {
            x[i] = ((i as i32 * 7919).wrapping_mul(31)) & 0xFFFF;
        }
        let original = x;
        forward(&mut x);
        inverse(&mut x);
        // After forward+inverse, we recover the original.
        // (Forward followed by forward = N · original; then >> 5 = original.)
        for i in 0..WHT_N {
            assert_eq!(x[i], original[i], "WHT roundtrip mismatch at i={i}");
        }
    }

    #[test]
    fn forward_dc_concentrates_first_bin() {
        // All-DC input should yield all energy in bin 0.
        let mut x = [100i32; WHT_N];
        forward(&mut x);
        assert_eq!(x[0], 3200); // 100 * 32
        for i in 1..WHT_N {
            assert_eq!(x[i], 0, "non-DC bin {i} should be zero");
        }
    }
}
