//! C FFI — stable ABI for third-party language bindings.
//!
//! Usage: compile with `cargo build --release --features ffi`
//! then link `liblml.so` / `lml.dll` / `liblml.dylib`.
//!
//! All functions return 0 on success, negative on error.

use crate::{error::LmlError, lml};
use std::os::raw::c_char;

/// Error codes returned by all FFI functions.
pub const LML_OK: i32 = 0;
pub const LML_ERROR_INVALID_INPUT: i32 = -1;
pub const LML_ERROR_CORRUPT_DATA: i32 = -2;
pub const LML_ERROR_BUFFER_TOO_SMALL: i32 = -3;
pub const LML_ERROR_UNSUPPORTED_VERSION: i32 = -4;
pub const LML_ERROR_IO: i32 = -5;

/// Version string.
#[no_mangle]
pub extern "C" fn lml_version() -> *const c_char {
    c"0.2.0".as_ptr()
}

/// Compress interleaved i64 signal → LML packet bytes.
///
/// signal: pointer to [n_channels * n_samples] i64 values (channel-major)
/// out_buf: pre-allocated output buffer
/// out_len: on success, receives actual bytes written
///
/// Returns LML_OK or negative error code.
///
/// # Safety
///
/// All pointers must be valid for their declared lengths. `out_buf` must be
/// writable for `out_cap` bytes and `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn lml_compress(
    signal: *const i64,
    n_channels: u32,
    n_samples: u32,
    noise_bits: u8,
    out_buf: *mut u8,
    out_cap: u32,
    out_len: *mut u32,
) -> i32 {
    if signal.is_null() || out_buf.is_null() || out_len.is_null() {
        return LML_ERROR_INVALID_INPUT;
    }

    let n_ch = n_channels as usize;
    let n_samp = n_samples as usize;

    // Build Vec<Vec<i64>> from flat pointer
    let total = n_ch * n_samp;
    let slice = std::slice::from_raw_parts(signal, total);
    let sig: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| slice[ch * n_samp..(ch + 1) * n_samp].to_vec())
        .collect();

    // Fix-C3: lml::compress now returns Result; map InvalidHeader to
    // LML_ERROR_INVALID_INPUT so FFI callers get a code, not a panic.
    let compressed = match lml::compress(&sig, noise_bits) {
        Ok(b) => b,
        Err(_) => return LML_ERROR_INVALID_INPUT,
    };
    if compressed.len() > out_cap as usize {
        return LML_ERROR_BUFFER_TOO_SMALL;
    }

    std::ptr::copy_nonoverlapping(compressed.as_ptr(), out_buf, compressed.len());
    *out_len = compressed.len() as u32;
    LML_OK
}

/// Decompress LML packet → flat i64 signal.
///
/// data: compressed bytes
/// signal_out: pre-allocated [n_channels * n_samples] i64 buffer
/// n_channels_out, n_samples_out: receive actual dimensions
///
/// Returns LML_OK or negative error code.
///
/// # Safety
///
/// `data` must be readable for `data_len` bytes. `signal_out` must be writable
/// for `signal_cap` samples. Non-null dimension outputs must be writable.
#[no_mangle]
pub unsafe extern "C" fn lml_decompress(
    data: *const u8,
    data_len: u32,
    signal_out: *mut i64,
    signal_cap: u32,
    n_channels_out: *mut u32,
    n_samples_out: *mut u32,
) -> i32 {
    if data.is_null() || signal_out.is_null() {
        return LML_ERROR_INVALID_INPUT;
    }

    let bytes = std::slice::from_raw_parts(data, data_len as usize);
    match lml::decompress(bytes) {
        Ok(signal) => {
            let n_ch = signal.len();
            let n_samp = if n_ch > 0 { signal[0].len() } else { 0 };
            let total = n_ch * n_samp;

            if total > signal_cap as usize {
                return LML_ERROR_BUFFER_TOO_SMALL;
            }

            // Write channel-major flat output
            let out = std::slice::from_raw_parts_mut(signal_out, total);
            for ch in 0..n_ch {
                out[ch * n_samp..(ch + 1) * n_samp].copy_from_slice(&signal[ch]);
            }

            if !n_channels_out.is_null() {
                *n_channels_out = n_ch as u32;
            }
            if !n_samples_out.is_null() {
                *n_samples_out = n_samp as u32;
            }
            LML_OK
        }
        Err(e) => match e {
            LmlError::CrcMismatch { .. } => LML_ERROR_CORRUPT_DATA,
            LmlError::UnsupportedVersion(_) => LML_ERROR_UNSUPPORTED_VERSION,
            LmlError::InvalidMagic(_) => LML_ERROR_INVALID_INPUT,
            LmlError::Truncated { .. } => LML_ERROR_CORRUPT_DATA,
            _ => LML_ERROR_INVALID_INPUT,
        },
    }
}

/// Get human-readable error string for an error code.
#[no_mangle]
pub extern "C" fn lml_error_string(code: i32) -> *const c_char {
    match code {
        LML_OK => c"Success".as_ptr(),
        LML_ERROR_INVALID_INPUT => c"Invalid input".as_ptr(),
        LML_ERROR_CORRUPT_DATA => c"Corrupt data (CRC mismatch or truncated)".as_ptr(),
        LML_ERROR_BUFFER_TOO_SMALL => c"Output buffer too small".as_ptr(),
        LML_ERROR_UNSUPPORTED_VERSION => c"Unsupported LML version".as_ptr(),
        LML_ERROR_IO => c"I/O error".as_ptr(),
        _ => c"Unknown error".as_ptr(),
    }
}
