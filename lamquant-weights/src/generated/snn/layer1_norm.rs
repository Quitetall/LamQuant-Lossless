// **GENERATED — DO NOT EDIT.**
//
// Source:    weights/snn/mamba_snn_best.pt
// SHA-256:   1d92fedec6ffa2c95d5acd4d7bf1166a987d1cc8b98d475d6e438cfd467dbf7e
// Generated: 2026-05-05T18:06:02+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --snn-checkpoint <path>
//
// Q15 quantization with per-tensor f32 scale. To dequantize at runtime:
//   f32 = (q15 / 32_767.0) * SCALE

/// norm weight, shape=(40,)
pub const NORM_WEIGHT_LEN: usize = 40;
pub static NORM_WEIGHT: [i8; 40] = [
    109, 105, 88, 104, 122, 94, 118, 84, 76, 102, 62, 71, 99, 103, 111, 101, 
     76, 101, 114, 101, 109, 92, 111, 118, 92, 116, 89, 127, 77, 76, 72, 99, 
     80, 77, 96, 85, 116, 94, 107, 91,
];

/// Dequantize: f32 = q8 * SCALE
pub const NORM_WEIGHT_SCALE: f32 = 1.139676102e-02;

/// norm bias, shape=(40,)
pub const NORM_BIAS_LEN: usize = 40;
pub static NORM_BIAS: [i8; 40] = [
    55, 6, 8, -55, 16, -26, -15, -30, -9, -27, -46, 32, 70, 12, 43, 43, 
     -46, 15, -8, -127, 5, 37, -7, -105, -25, -5, -60, 35, -17, -44, -64, -13, 
     31, 13, 1, -93, -25, 5, -28, -6,
];

pub const NORM_BIAS_SCALE: f32 = 6.643170916e-03;
