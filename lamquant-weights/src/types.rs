//! Typed wrappers over the generated weight tables.
//!
//! Hand-written. Stable across exports. Generated tables in `src/generated/`
//! reference these types by name. If you change a struct shape here you must
//! also update the corresponding Jinja2 template in `firmware/templates/rust/`.

/// 2-bit packed ternary convolution weights with per-output-channel LSQ alpha.
///
/// Packing: 4 weights per byte, 2 bits each.
///   `00 = 0`, `01 = +1`, `10 = -1`, `11 = ERROR (reserved, never generated)`
///
/// LSQ alpha is stored as Q15 (i16) to halve metadata size; firmware promotes
/// to Q31 with a single shift: `alpha_q31 = (alpha_q15 as i32) << 16`.
///
/// `IN`, `OUT`, `K` are const generics so callers get const-folded shape
/// constants and the compiler can validate `packed.len()` against
/// `expected_packed_len()` at compile time via `const_assert!`.
#[derive(Copy, Clone)]
pub struct TernaryConvWeights<const IN: usize, const OUT: usize, const K: usize> {
    pub packed: &'static [u8],
    pub alphas_q15: &'static [i16; OUT],
    pub norm: Option<NormParams<OUT>>,
}

impl<const IN: usize, const OUT: usize, const K: usize> TernaryConvWeights<IN, OUT, K> {
    /// `ceil(IN * OUT * K / 4)` — the expected length of `packed`.
    /// Const-evaluable so callers can assert at compile time.
    pub const fn expected_packed_len() -> usize {
        (IN * OUT * K + 3) / 4
    }

    /// True if `packed.len()` matches the shape. Used by `validate()`.
    pub const fn is_packed_len_valid(&self) -> bool {
        self.packed.len() == Self::expected_packed_len()
    }
}

/// Depthwise ternary 1D conv: one independent kernel per channel.
/// Shape: weight is `[C, 1, K]` → `n_weights = C * K`.
#[derive(Copy, Clone)]
pub struct DepthwiseTernaryConvWeights<const C: usize, const K: usize> {
    pub packed: &'static [u8],
    pub alphas_q15: &'static [i16; C],
}

impl<const C: usize, const K: usize> DepthwiseTernaryConvWeights<C, K> {
    pub const fn expected_packed_len() -> usize {
        (C * K + 3) / 4
    }

    pub const fn is_packed_len_valid(&self) -> bool {
        self.packed.len() == Self::expected_packed_len()
    }
}

/// GroupNorm/LayerNorm parameters: weight (Q7, i8) + bias (Q15, i16).
#[derive(Copy, Clone)]
pub struct NormParams<const C: usize> {
    pub weight_q7: &'static [i8; C],
    pub bias_q15: &'static [i16; C],
}

/// INT8 1D convolution weights (used for `bneck_v` output bottleneck).
/// Higher precision than ternary because the bottleneck output sets the
/// scale of the latent that downstream FSQ quantizes.
#[derive(Copy, Clone)]
pub struct Int8ConvWeights<const IN: usize, const OUT: usize, const K: usize> {
    pub weights: &'static [i8],
    /// Per-output-channel quantization scale, Q15.
    pub scales_q15: &'static [i16; OUT],
}

impl<const IN: usize, const OUT: usize, const K: usize> Int8ConvWeights<IN, OUT, K> {
    pub const fn expected_weights_len() -> usize {
        IN * OUT * K
    }

    pub const fn is_weights_len_valid(&self) -> bool {
        self.weights.len() == Self::expected_weights_len()
    }
}

/// 32-dim Cayley rotation matrix in Q15 (row-major, 1024 elements).
/// Applied to the latent vector before FSQ quantization to decorrelate
/// dimensions: `Q = (I - A)(I + A)^{-1}` where A is skew-symmetric.
#[derive(Copy, Clone)]
pub struct RotationMatrix32 {
    pub q_q15: &'static [i16; 1024],
}

/// FSQ lattice configuration — 4D product quantizer + rANS frequency table.
#[derive(Copy, Clone)]
pub struct FsqLattice {
    /// Per-dimension levels. Total codebook = product(levels).
    pub levels: &'static [i32; 4],
    /// Default Q31 projection scale (overridden at runtime by adaptive gain).
    pub quant_scale_q31: i32,
    /// Per-bin frequency for rANS encode (sums to `rans_total`).
    pub rans_freq: &'static [u32; 16],
    /// Cumulative starts for rANS decode.
    pub rans_start: &'static [u32; 16],
    /// Total frequency budget (2^N for fast renormalization).
    pub rans_total: u32,
    /// Calibrated latent range, Q31-scaled by 1000.
    pub vmin_q31: i32,
    pub vmax_q31: i32,
    /// Pre-computed `(num_levels << 30) / range` for integer division.
    pub inv_range_q31: i32,
}

/// SNN (Mamba state-space) tensor with optional per-tensor float32 scale.
#[derive(Copy, Clone)]
pub struct SnnInt8Tensor {
    pub data: &'static [i8],
    /// Float32 scale; `f32_value = data[i] as f32 * scale`. None when `f32` direct.
    pub scale: f32,
}

/// SNN A_log state-transition tensor (Q15).
#[derive(Copy, Clone)]
pub struct SnnALogQ15<const N: usize> {
    pub data: &'static [i16; N],
}
