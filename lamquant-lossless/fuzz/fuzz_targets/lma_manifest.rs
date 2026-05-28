//! Phase 8 / Item F — fuzz the LMA manifest parser.
//!
//! `list_archive` parses an LMA archive's header + (zstd-compressed)
//! JSON manifest. Catches:
//!   - manifest-bomb (zstd decompression target far larger than the
//!     declared upper bound)
//!   - malformed JSON in the manifest body
//!   - integer overflow in the offset / length fields
//!
//! The fuzzer writes arbitrary bytes to a tempfile and passes the
//! path to `lma::list_archive`. Bible R23 requires the parser to
//! refuse anything that doesn't match the wire format BEFORE doing
//! any work that depends on user-controlled lengths.
//!
//! Run with:
//!     cd lamquant-core
//!     cargo +nightly fuzz run lma_manifest -- -max_total_time=60

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Write the fuzzer payload to a tempfile (list_archive expects
    // a path). Surface any I/O panic so the fuzzer can flag it; the
    // parser itself must never panic on the bytes themselves.
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    let _ = lamquant_core::lma::list_archive(tmp.path());
});
