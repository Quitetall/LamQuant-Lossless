//! WebAssembly bindings via wasm-bindgen.
//!
//! Build: wasm-pack build lamquant-core --target web --features wasm
//! Usage: import { compress, decompress, info } from 'lml-codec';

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Compress flat i64 signal → LML packet bytes.
/// Signal is [n_channels * n_samples] channel-major.
///
/// Returns `Err(JsValue)` on invalid header dimensions (Fix-C3: previously
/// panicked the WASM module via `assert!` in `lml::compress`).
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn compress(
    signal: &[i64],
    n_channels: u32,
    n_samples: u32,
    noise_bits: u8,
) -> Result<Vec<u8>, JsValue> {
    let n_ch = n_channels as usize;
    let n_samp = n_samples as usize;
    let sig: Vec<Vec<i64>> = (0..n_ch)
        .map(|ch| signal[ch * n_samp..(ch + 1) * n_samp].to_vec())
        .collect();
    crate::lml::compress(&sig, noise_bits).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Decompress LML packet → flat i64 signal (channel-major).
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decompress(data: &[u8]) -> Result<Vec<i64>, JsValue> {
    let signal = crate::lml::decompress(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let n_ch = signal.len();
    let n_samp = if n_ch > 0 { signal[0].len() } else { 0 };
    let mut flat = Vec::with_capacity(n_ch * n_samp);
    for ch in &signal {
        flat.extend_from_slice(ch);
    }
    Ok(flat)
}
