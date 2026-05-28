//! Fuzz the `LmlReader` stream API on arbitrary bytes.
//!
//! `lmafs` mounts user-supplied `.lma` archives and reads LML payloads
//! out of them via `LmlReader::from_source(Cursor::new(...))` + random
//! `seek_to_window` / `next_window` calls. This target hammers exactly
//! that path: arbitrary fuzzer bytes as the LML buffer + a fuzzer-
//! supplied window index. Any panic, integer overflow, or unchecked
//! slice index here is reachable from the FUSE attack surface.
//!
//! Cat A3 expansion (2026-05-21).

#![no_main]
use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    // First byte = candidate window index (0..255); rest = LML buffer.
    let idx = data[0] as usize;
    let buf = &data[1..];

    // `Cursor<&[u8]>` implements `Read + Seek` via `T: AsRef<[u8]>`,
    // so we can skip the `.to_vec()` copy and feed the fuzzer slice
    // directly. (lamu review fix on 88b7868.)
    let mut reader = match lamquant_core::stream::LmlReader::from_source(
        Cursor::new(buf),
    ) {
        Ok(r) => r,
        Err(_) => return, // header rejected — fine
    };

    // Try to seek to a fuzzer-chosen window index. Either side of
    // `Ok` / `Err` is acceptable; what's forbidden is `panic!` or
    // unchecked slice indexing inside the stream layer.
    let _ = reader.seek_to_window(idx);

    // Pull up to 4 windows. Each must return either `Some(Ok)`,
    // `Some(Err)`, or `None` without panicking.
    for _ in 0..4 {
        match reader.next_window() {
            Some(Ok(_)) | Some(Err(_)) | None => {}
        }
    }
});
