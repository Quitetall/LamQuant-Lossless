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
//! CRC-32 over all weight byte arrays. Verified at boot.
//! Mirrors `metadata::FIRMWARE_CRC32`.

pub const FIRMWARE_EXPECTED_CRC32: u32 = 0xCA6AC746;

/// Order of buffers used to compute the CRC. Firmware must walk this list
/// in the same order during boot integrity check.
pub const CRC_BUFFER_ORDER: &[&str] = &[
"focal::premix::PACKED_WEIGHTS",
"focal::focal2::PACKED_WEIGHTS",
"focal::focal3::PACKED_WEIGHTS",
"focal::dw_gate::PACKED_WEIGHTS",
"focal::bneck_g::PACKED_WEIGHTS",
"focal::bneck_v::WEIGHTS_RAW",
"rotation::ROTATION_Q_Q15",
];
