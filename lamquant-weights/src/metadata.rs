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
//! Build-time metadata. Mirrors `.exportlock.json` exactly.

pub const SCHEMA_VERSION: &str = "1.0";
pub const MODEL_VERSION: &str = "7.7.0";
pub const ARCHITECTURE: &str = "subband_v1";
pub const ENCODER_CLASS: &str = "TernaryMobileNetV5_Subband";
pub const ENCODER_WIDTH: usize = 128;
pub const N_FOCAL_BLOCKS: usize = 3;
pub const LATENT_DIMS: usize = 32;
pub const LATENT_TIMESTEPS: usize = 79;

pub const CHECKPOINT_NAME: &str = "student_subband_gold.ckpt";
pub const CHECKPOINT_SHA256: [u8; 32] = [    0x0e,    0x85,    0x62,    0xc3,    0x77,    0xf6,    0x08,    0x1f,    0xe5,    0xde,    0x90,    0xf7,    0x1b,    0x47,    0x63,    0x02,    0xa5,    0x3a,    0xc9,    0xd3,    0xe5,    0x11,    0xdc,    0x51,    0x11,    0x6a,    0xb9,    0xc7,    0xd6,    0xee,    0x29,    0x57,];

pub const EXPORTER_VERSION: &str = "1.0.0";
pub const GIT_COMMIT: &str = "17d80c7";
pub const EXPORT_TIMESTAMP_UNIX: u64 = 1777996259;

/// CRC-32 over all weight byte arrays in deterministic enumeration order.
/// Verified at boot; mismatch → safe mode.
pub const FIRMWARE_CRC32: u32 = 0xCA6AC746;

/// True once the codegen tool has populated this crate.
pub const IS_INITIALIZED: bool = true;
