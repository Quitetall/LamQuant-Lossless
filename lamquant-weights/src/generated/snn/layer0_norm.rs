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
    68, 67, 90, 61, 51, 79, 127, 74, 83, 62, 103, 78, 67, 71, 69, 77, 
     71, 67, 70, 89, 58, 63, 70, 74, 53, 81, 81, 106, 127, 62, 79, 63, 
     77, 63, 48, 86, 78, 69, 88, 86,
];

/// Dequantize: f32 = q8 * SCALE
pub const NORM_WEIGHT_SCALE: f32 = 1.528579795e-02;

/// norm bias, shape=(40,)
pub const NORM_BIAS_LEN: usize = 40;
pub static NORM_BIAS: [i8; 40] = [
    -56, -45, 7, 42, -32, -82, -47, -25, 17, 24, -74, 54, 42, -51, 35, -21, 
     24, -70, 37, -27, 118, 28, -52, -127, -66, 43, 16, 7, -45, 21, -16, 74, 
     -57, -68, -49, 34, 26, -9, 20, 26,
];

pub const NORM_BIAS_SCALE: f32 = 2.706974275e-03;
