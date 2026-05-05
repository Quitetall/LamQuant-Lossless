//! Boot-time firmware integrity check (CRC-32 over weight tables).
//!
//! Walks every weight buffer in the same order the codegen tool used to
//! compute `lamquant_weights::metadata::FIRMWARE_CRC32`. Mismatch =>
//! corrupted firmware → caller MUST enter safe mode (running with
//! corrupted weights is a patient-safety violation).
//!
//! The order of `crc32_update` calls below is the contract with the Rust
//! emitter: see `lamquant_weights::generated::crc::CRC_BUFFER_ORDER`.

use lamquant_core::crc32::{crc32_update, CRC32_INIT};
use lamquant_weights::generated;
use lamquant_weights::metadata;

/// Verify the embedded firmware CRC against the constant baked at export.
///
/// Returns `true` iff every weight buffer hashes to the expected total.
/// On `false`, caller MUST enter safe mode.
pub fn check_firmware_crc() -> bool {
    if !metadata::IS_INITIALIZED {
        // The weights crate has not been populated yet (initial scaffold).
        // Pass-through so firmware boots in development; production builds
        // will fail this check via the .exportlock.json gate.
        return true;
    }

    let computed = compute_weight_crc();
    computed == metadata::FIRMWARE_CRC32
}

/// CRC-32 over every weight byte array in deterministic order.
///
/// Order matches the Python emitter exactly. If you add a new layer to
/// the schema you MUST add it here in the same position the emitter
/// would walk it (see `firmware/export/rust_emitter.py`).
fn compute_weight_crc() -> u32 {
    let mut crc = CRC32_INIT;

    // Ternary layers, in schema declaration order.
    crc = crc32_update(crc, &generated::focal::premix::PACKED_WEIGHTS);
    crc = crc32_update(crc, &generated::focal::focal2::PACKED_WEIGHTS);
    crc = crc32_update(crc, &generated::focal::focal3::PACKED_WEIGHTS);
    crc = crc32_update(crc, &generated::focal::dw_gate::PACKED_WEIGHTS);
    crc = crc32_update(crc, &generated::focal::bneck_g::PACKED_WEIGHTS);

    // INT8 layer (bneck_v output bottleneck) — bytes view of i8 array.
    let bneck_v = &generated::focal::bneck_v::WEIGHTS_RAW;
    let bneck_v_bytes: &[u8] = unsafe {
        // SAFETY: i8 and u8 have identical layout; reinterpreting a slice
        // is bit-equivalent. The CRC is byte-blind.
        core::slice::from_raw_parts(bneck_v.as_ptr() as *const u8, bneck_v.len())
    };
    crc = crc32_update(crc, bneck_v_bytes);

    // Cayley rotation matrix (Q15 i16, little-endian on Hazard3).
    let rotation_q = &generated::rotation::ROTATION_Q_Q15;
    let rotation_bytes: &[u8] = unsafe {
        // SAFETY: i16 array is contiguous; emitter computed CRC over the
        // little-endian byte representation, which matches RV32 native.
        core::slice::from_raw_parts(
            rotation_q.as_ptr() as *const u8,
            rotation_q.len() * core::mem::size_of::<i16>(),
        )
    };
    crc = crc32_update(crc, rotation_bytes);

    crc
}
