//! Fuzz the current ABIR/BCS2 container authentication and semantic open.
//!
//! Feeds arbitrary bytes into `container::read_bytes`.
//! The reader must never panic on adversarial input — every malformed
//! magic byte / bad length / bogus catalog combination has to surface as
//! a typed `LmlError`. Catches integer overflow, OOB read, alloc bomb
//! patterns that hand-rolled test cases can miss.
//!
//! Run with:
//!     cd lamquant-core
//!     cargo +nightly fuzz run container_header -- -max_total_time=60

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Don't care about the result — only that the parser doesn't
    // panic. `container::read_from` should return Err on every
    // malformed input the fuzzer produces.
    let _ = lamquant_core::container::read_bytes(data);
});
