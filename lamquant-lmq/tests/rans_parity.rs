//! ADR 0074 Track N — rANS / LMQ token-body **wire-stability** gate.
//!
//! Pins the exact bytes the Rust neural body produces for a fixed
//! (tokens, schedule, model) input. The lossless codec's "byte-equality never
//! broken" discipline now extends to the neural wire: a refactor of `rans` or the
//! body framing that silently changes these bytes must be a DELIBERATE, reviewed
//! regen (re-run with `--nocapture` to read the new value), never an accident.
//!
//! On cross-implementation parity: the Python reference
//! (`lamquant_codec.ops.rans`) has no independent encoder in the no-numba
//! environment — it **delegates to this same Rust code** (`lamquant_core`, the
//! PyO3 wrapper over `lamquant_lml_mcu::rans`) whenever numba is absent, so in
//! production they are byte-identical by construction. A live Rust-vs-numba
//! cross-check is env-gated (needs numba) and deferred, like the SNN PCCP gates.

use lamquant_lmq::body::{decode_body, encode_body};

const TOKENS: [i64; 20] = [0, 1, 4, 2, 3, 3, 1, 0, 4, 2, 2, 2, 3, 4, 0, 1, 1, 2, 3, 4];
const SCHEDULE: [u8; 20] = [5, 5, 5, 3, 3, 2, 2, 5, 5, 3, 3, 3, 5, 5, 2, 2, 3, 3, 5, 5];
const COUNTS: [i32; 5] = [3, 3, 3, 3, 4]; // per-symbol freq, Σ = 16

/// The frozen LMQ token-body wire for the fixture above: 11-byte prefix (version,
/// n_symbols, alphabet), the 5×u32 counts, the length-prefixed 20-byte schedule,
/// and the 9-byte rANS stream of the tokens. Regen DELIBERATELY only.
const BODY_GOLDEN: &[u8] = &[
    1, // version
    20, 0, 0, 0, // n_symbols = 20
    5, 0, // alphabet = 5
    3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, // counts
    20, 0, 0, 0, // schedule len
    5, 5, 5, 3, 3, 2, 2, 5, 5, 3, 3, 3, 5, 5, 2, 2, 3, 3, 5, 5, // schedule
    9, 0, 0, 0, // rANS len
    230, 98, 120, 241, 230, 17, 62, 6, 0, // rANS stream
];

#[test]
fn lmq_body_wire_is_byte_stable_and_roundtrips() {
    let body = encode_body(&TOKENS, &SCHEDULE, &COUNTS).expect("encode_body");
    assert_eq!(
        body, BODY_GOLDEN,
        "LMQ token body wire drifted from the frozen golden — if intended, regen \
         (--nocapture prints the new bytes); otherwise the neural wire broke."
    );
    let (dt, ds, _alpha) = decode_body(&body).expect("decode_body");
    assert_eq!(dt, TOKENS, "tokens must round-trip");
    assert_eq!(ds, SCHEDULE, "schedule must round-trip");
}
