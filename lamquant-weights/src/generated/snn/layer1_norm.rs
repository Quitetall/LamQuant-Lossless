// **GENERATED — DO NOT EDIT.**
//
// Source:    /mnt/4tb/LamQuant/weights/snn/mamba_snn_best.pt
// SHA-256:   34e008106a9b908b4d344f15e253d40a0fd19d206ef0a42a66c81f5f68be3e71
// Generated: 2026-05-22T03:29:10+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --snn-checkpoint <path>
//
// Q15 quantization with per-tensor f32 scale. To dequantize at runtime:
//   f32 = (q15 / 32_767.0) * SCALE

/// norm weight, shape=(40,)
pub const NORM_WEIGHT_LEN: usize = 40;
pub static NORM_WEIGHT: [i8; 40] = [
    -82, -43, -48, 3, -114, 109, -19, 69, -109, -25, -72, 36, -3, -1, 18, -60, 
     41, 21, -33, 60, -90, 35, -114, -107, 58, 104, -56, -45, -82, -17, 46, 4, 
     -49, -35, 38, 22, -127, -87, 41, -102,
];

/// Dequantize: real = q8 * (Q15 / 32768) (Q15, pre-baked from f32=1.288335154e-03)
pub const NORM_WEIGHT_SCALE_Q15: i32 = 42;

/// norm bias, shape=(40,)
pub const NORM_BIAS_LEN: usize = 40;
pub static NORM_BIAS: [i8; 40] = [
    16, 42, -63, -49, 8, 65, 17, -11, -16, 25, 119, -3, -25, -105, -13, -54, 
     15, 127, -10, 1, 21, 33, -28, -17, -35, -24, -45, -118, -59, 2, -64, 25, 
     77, 97, -59, 55, 1, -8, -14, 53,
];

pub const NORM_BIAS_SCALE_Q15: i32 = 35;
