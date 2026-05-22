// **GENERATED — DO NOT EDIT.**
//
// Source:    weights/snn/mamba_snn_best.pt
// SHA-256:   34e008106a9b908b4d344f15e253d40a0fd19d206ef0a42a66c81f5f68be3e71
// Generated: 2026-05-22T15:37:53+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --snn-checkpoint <path>
//
// Q15 quantization with per-tensor f32 scale. To dequantize at runtime:
//   f32 = (q15 / 32_767.0) * SCALE

/// norm weight, shape=(40,)
pub const NORM_WEIGHT_LEN: usize = 40;
pub static NORM_WEIGHT: [i8; 40] = [
    37, 87, 11, 127, -50, -34, 13, -45, 32, 80, -28, -56, -30, 49, -2, 41, 
     59, -8, -37, -7, -43, 13, -4, 20, -6, -35, 5, -8, 81, -44, -53, -20, 
     -9, -28, -51, 3, -7, 96, -18, -10,
];

/// Dequantize: real = q8 * (Q15 / 32768) (Q15, pre-baked from f32=3.274733861e-03)
pub const NORM_WEIGHT_SCALE_Q15: i32 = 107;

/// norm bias, shape=(40,)
pub const NORM_BIAS_LEN: usize = 40;
pub static NORM_BIAS: [i8; 40] = [
    -3, 16, -4, 127, 9, 14, 7, 9, -1, -24, 7, 3, 2, -9, 20, -6, 
     9, -3, -3, 7, 29, -2, 1, 1, -5, 11, -2, -7, -26, 2, 3, 4, 
     -13, 8, -19, -16, 6, 14, 13, 58,
];

pub const NORM_BIAS_SCALE_Q15: i32 = 354;
