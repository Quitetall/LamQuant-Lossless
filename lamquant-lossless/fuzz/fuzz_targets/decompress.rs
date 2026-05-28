//! Fuzz the LML decompressor with arbitrary bytes.
//! Must never crash, hang, or consume unbounded memory.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // This must NEVER panic. Errors are fine, crashes are bugs.
    let _ = lamquant_core::lml::decompress(data);
});
