// **GENERATED — DO NOT EDIT.**
//
// Source:    /mnt/4tb/LamQuant/weights/student_subband_gold.ckpt
// SHA-256:   0e8562c377f6081fe5de90f71b476302a53ac9d3e511dc51116ab9c7d6ee2957
// Architecture: subband_v1 (TernaryMobileNetV5_Subband)
// Schema:    1.0
// Exporter:  1.0.0
// Generated: 2026-05-05T15:50:59.776447+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --arch subband_v1
//! FSQ lattice + rANS frequency table. Calibrated from 50
//! random-clamped EEG forward passes. Latent range [-1.000, 1.000],
//! entropy 3.924 bps.

use crate::types::FsqLattice;

pub const NUM_LEVELS: usize = 16;
pub const RANS_TOTAL: u32 = 4096;

pub static LEVELS: [i32; 4] = [
8, 6, 5, 5,];

pub const QUANT_SCALE_Q31: i32 = 1073741824;

pub static RANS_FREQ: [u32; NUM_LEVELS] = [225, 124, 169, 218, 269, 315, 350, 377,366, 349, 315, 271, 220, 171, 125, 232,];

pub static RANS_START: [u32; NUM_LEVELS] = [0, 225, 349, 518, 736, 1005, 1320, 1670,2047, 2413, 2762, 3077, 3348, 3568, 3739, 3864,];

pub const VMIN_Q31: i32 = -1000;
pub const VMAX_Q31: i32 = 1000;
pub const INV_RANGE_Q31: i32 = 8589934;

pub const LATTICE: FsqLattice = FsqLattice {
    levels: &LEVELS,
    quant_scale_q31: QUANT_SCALE_Q31,
    rans_freq: &RANS_FREQ,
    rans_start: &RANS_START,
    rans_total: RANS_TOTAL,
    vmin_q31: VMIN_Q31,
    vmax_q31: VMAX_Q31,
    inv_range_q31: INV_RANGE_Q31,
};
