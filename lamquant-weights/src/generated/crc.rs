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
//! CRC-32 over all weight byte arrays. Verified at boot.
//! Mirrors `metadata::FIRMWARE_CRC32`.

pub const FIRMWARE_EXPECTED_CRC32: u32 = 0x63EE6A12;

/// Order of buffers used to compute the CRC. Firmware must walk this list
/// in the same order during boot integrity check.
pub const CRC_BUFFER_ORDER: &[&str] = &[
"focal::premix::PACKED_WEIGHTS",
"focal::focal1_conv::PACKED_WEIGHTS",
"focal::focal2::PACKED_WEIGHTS",
"focal::focal3::PACKED_WEIGHTS",
"focal::dw_gate::PACKED_WEIGHTS",
"focal::bneck_g::PACKED_WEIGHTS",
"focal::bneck_v::WEIGHTS_RAW",
"rotation::ROTATION_Q_Q15",
"snn::spatial_mix::SPATIAL_MIX_WEIGHT",
"snn::spatial_mix::SPATIAL_MIX_BIAS",
"snn::readout::READOUT_WEIGHT",
"snn::readout::READOUT_BIAS",
"snn::layer0_norm::NORM_WEIGHT",
"snn::layer0_norm::NORM_BIAS",
"snn::layer0_fwd::IN_PROJ_W",
"snn::layer0_fwd::X_PROJ_W",
"snn::layer0_fwd::OUT_PROJ_W",
"snn::layer0_fwd::CONV1D_W",
"snn::layer0_fwd::CONV1D_B",
"snn::layer0_fwd::A_LOG",
"snn::layer0_fwd::D",
"snn::layer0_fwd::DT_BIAS",
"snn::layer0_bwd::IN_PROJ_W",
"snn::layer0_bwd::X_PROJ_W",
"snn::layer0_bwd::OUT_PROJ_W",
"snn::layer0_bwd::CONV1D_W",
"snn::layer0_bwd::CONV1D_B",
"snn::layer0_bwd::A_LOG",
"snn::layer0_bwd::D",
"snn::layer0_bwd::DT_BIAS",
"snn::layer1_norm::NORM_WEIGHT",
"snn::layer1_norm::NORM_BIAS",
"snn::layer1_fwd::IN_PROJ_W",
"snn::layer1_fwd::X_PROJ_W",
"snn::layer1_fwd::OUT_PROJ_W",
"snn::layer1_fwd::CONV1D_W",
"snn::layer1_fwd::CONV1D_B",
"snn::layer1_fwd::A_LOG",
"snn::layer1_fwd::D",
"snn::layer1_fwd::DT_BIAS",
"snn::layer1_bwd::IN_PROJ_W",
"snn::layer1_bwd::X_PROJ_W",
"snn::layer1_bwd::OUT_PROJ_W",
"snn::layer1_bwd::CONV1D_W",
"snn::layer1_bwd::CONV1D_B",
"snn::layer1_bwd::A_LOG",
"snn::layer1_bwd::D",
"snn::layer1_bwd::DT_BIAS",
];
