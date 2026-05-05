//! LamQuant TNN/SNN weights — generated artifact.
//!
//! This crate is **generated** by `firmware/export_firmware.py --target rust`
//! from a model checkpoint. Hand-edited files: `lib.rs` (this), `types.rs`.
//! Everything under `src/generated/` is rewritten on each export run.
//!
//! ## Usage
//!
//! ```ignore
//! use lamquant_weights::generated::focal::focal1_conv;
//! let weights = focal1_conv::WEIGHTS;
//! let acc = ternary_mac::conv1d_channel(
//!     act,
//!     focal1_conv::IN_CHANNELS,    // const-folded
//!     focal1_conv::KERNEL_SIZE,
//!     weights.packed,
//!     (weights.alphas_q15[ch] as i32) << 16,
//! );
//! ```
//!
//! ## Architecture variants
//!
//! Selected at build time via Cargo features:
//!   - `subband_v1` (default): Gen 7.1 width=128
//!   - `subband_v2`: Gen 7.6.1 width=216 (depthwise-separable)
//!   - `legacy_v7_0`: Gen 7.0 width=96 (deprecated)
//!
//! Exactly one feature must be enabled.
//!
//! ## Boot integrity
//!
//! `metadata::FIRMWARE_CRC32` is the CRC-32 over all weight byte arrays in
//! deterministic enumeration order. Firmware verifies this at boot via
//! `lamquant_firmware::integrity::check_firmware_crc()`. Mismatch → safe mode.
//!
//! ## Versioning
//!
//! `metadata::CHECKPOINT_SHA256` pins the source checkpoint. The codegen
//! tool refuses to overwrite `src/generated/` if this value differs from
//! the committed `.exportlock.json` unless `--force` is passed.

#![no_std]
#![allow(clippy::unreadable_literal)]
#![allow(dead_code)]

pub mod metadata;
pub mod types;

#[cfg(feature = "subband_v1")]
pub mod generated;

#[cfg(any(
    all(feature = "subband_v1", feature = "subband_v2"),
    all(feature = "subband_v1", feature = "legacy_v7_0"),
    all(feature = "subband_v2", feature = "legacy_v7_0"),
))]
compile_error!(
    "lamquant-weights: select EXACTLY ONE architecture feature \
    (subband_v1 / subband_v2 / legacy_v7_0)"
);

#[cfg(not(any(feature = "subband_v1", feature = "subband_v2", feature = "legacy_v7_0")))]
compile_error!(
    "lamquant-weights: no architecture feature selected. \
    Add `features = [\"subband_v1\"]` (or another variant) to your Cargo.toml."
);
