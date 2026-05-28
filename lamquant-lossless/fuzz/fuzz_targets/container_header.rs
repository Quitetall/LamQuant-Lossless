//! Phase 8 / Item F — fuzz the container header + first window parse.
//!
//! Feeds arbitrary bytes into `container::read_from(&mut Cursor)`.
//! The reader must never panic on adversarial input — every malformed
//! magic byte / bad length / bogus flag combination has to surface as
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
    let mut cursor = std::io::Cursor::new(data);
    let _ = lamquant_core::container::read_from(&mut cursor);
});
