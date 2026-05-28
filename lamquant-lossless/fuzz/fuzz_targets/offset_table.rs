//! Phase 8 / Item F — fuzz the LMLFOOT1 seek-table parser.
//!
//! Catches integer-overflow / OOM-alloc bugs in `OffsetTable::read_
//! from_buffer`. The footer's `n_windows` × `ENTRY_SIZE` math is the
//! main attack surface — a malicious file can claim huge n_windows
//! to force a giant Vec allocation. Bible R23 — the parser must
//! validate that count BEFORE allocating.
//!
//! Run with:
//!     cd lamquant-core
//!     cargo +nightly fuzz run offset_table -- -max_total_time=60

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = lamquant_core::offset_table::OffsetTable::read_from_buffer(data);
});
