// **GENERATED — DO NOT EDIT.**
//
// Source:    /mnt/4tb/LamQuant/weights/student_subband_gold.ckpt
// SHA-256:   0e8562c377f6081fe5de90f71b476302a53ac9d3e511dc51116ab9c7d6ee2957
// Architecture: subband_v1 (TernaryMobileNetV5_Subband)
// Schema:    1.0
// Exporter:  1.0.0
// Generated: 2026-05-05T16:47:12.074522+00:00
//
// Regenerate via:
//   python firmware/export_firmware.py --target rust --arch subband_v1
//! LFSR seeds for per-channel Toeplitz compressed-sensing rows.
//! Constants — same across all checkpoints.

pub const N_CHANNELS: usize = 21;

pub static SEEDS: [u32; N_CHANNELS] = [0xACE1, 0xBE37, 0xCAFE, 0xDEAD, 0xF00D, 0x1337,0xB00B, 0xFACE, 0xD00D, 0xBEEF, 0xC0DE, 0xBAD1,0xFEED, 0xDAD1, 0xAB1E, 0xACDC, 0xB1A5, 0xCA5E,0xDE1F, 0xEF01, 0xF1A7,];
