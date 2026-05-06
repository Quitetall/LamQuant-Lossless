// **GENERATED — DO NOT EDIT.**
//
// Source:    /mnt/4tb/LamQuant/weights/student_subband_gold.ckpt
// SHA-256:   0e8562c377f6081fe5de90f71b476302a53ac9d3e511dc51116ab9c7d6ee2957
// Architecture: subband_v1 (TernaryMobileNetV5_Subband)
// Schema:    1.0
// Exporter:  1.0.0
// Generated: 2026-05-05T18:06:01.935213+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --arch subband_v1
//! dw_gate: 128 → 128, k=3, stride=1, depthwise (groups=128).
//! Ternary 2-bit-packed weights, per-output Q15 alphas.

use crate::types::DepthwiseTernaryConvWeights;

pub const IN_CHANNELS: usize = 128;
pub const OUT_CHANNELS: usize = 128;
pub const KERNEL_SIZE: usize = 3;
pub const STRIDE: usize = 1;
pub const GROUPS: usize = 128;
pub const N_PACKED_BYTES: usize = 96;
pub const N_WEIGHTS: usize = 384;

#[link_section = ".sram4_tnn"]
pub static PACKED_WEIGHTS: [u8; N_PACKED_BYTES] = [0x40, 0x56, 0x41, 0x80, 0x58, 0x68, 0x95, 0x59, 0x55, 0x00, 0x50, 0x00,0x40, 0x85, 0xAA, 0x6A, 0x54, 0x55, 0x96, 0xAA, 0x22, 0x48, 0x05, 0x50,0xA8, 0x82, 0xAA, 0x55, 0x55, 0xA9, 0x2A, 0x40, 0xA1, 0x80, 0x8A, 0xA9,0x15, 0x50, 0x55, 0x50, 0x45, 0x55, 0x68, 0xA5, 0x56, 0x29, 0x00, 0x90,0x11, 0xA5, 0x46, 0x55, 0xA1, 0xA6, 0x29, 0x50, 0x50, 0xAA, 0x55, 0x89,0x99, 0xA8, 0xA4, 0x00, 0x40, 0x65, 0x90, 0x85, 0xA2, 0xAA, 0x4A, 0x00,0x95, 0x5A, 0x55, 0x55, 0x85, 0xA2, 0x59, 0x5A, 0x15, 0x54, 0xA2, 0xAA,0x00, 0x00, 0x00, 0x55, 0x06, 0x44, 0x69, 0x45, 0x54, 0x06, 0x50, 0x01,];

pub static ALPHAS_Q15: [i16; OUT_CHANNELS] = [32767, 32767, 32767, 32767, 32767, 32767, 28688, 2228, 31442, 4539, 32767, 14392,32767, 32767, 32767, 32767, 32767, 13161, 32767, 16902, 32767, 26434, 15135, 9927,29345, 32767, 32767, 32767, 32767, 5998, 32767, 22849, 32767, 25345, 32767, 2170,8772, 4560, 32767, 32767, 32767, 32767, 10347, 32033, 32767, 32767, 11780, 30646,6677, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 3259, 32767, 32767,32767, 32767, 32767, 15432, 7656, 32767, 32767, 13508, 19419, 26542, 25752, 2086,30206, 32767, 32767, 14558, 25773, 4807, 32767, 32767, 6179, 32767, 32767, 3567,32767, 32767, 32767, 8539, 14562, 32767, 12884, 10842, 23656, 32767, 17796, 32767,16202, 13608, 32767, 3571, 32767, 32767, 10252, 32767, 8179, 25913, 32767, 29904,18685, 32705, 1150, 9286, 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,4724, 24869, 32767, 32767, 10400, 32767, 32767, 32767,];


pub const WEIGHTS: DepthwiseTernaryConvWeights<OUT_CHANNELS, KERNEL_SIZE> =
    DepthwiseTernaryConvWeights {
        packed: &PACKED_WEIGHTS,
        alphas_q15: &ALPHAS_Q15,
    };

const _: () = assert!(
    WEIGHTS.is_packed_len_valid(),
    "dw_gate: packed weights length mismatches expected"
);
