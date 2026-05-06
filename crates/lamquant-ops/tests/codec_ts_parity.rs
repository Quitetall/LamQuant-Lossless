//! Cross-language parity: every op id the GUI's `gui/src/lib/codec.ts`
//! exposes MUST resolve to a real `lamquant_ops::op_spec`. Drift here
//! means a route renders a Run button that Tauri rejects with "unknown
//! op id" — the worst kind of UI contract failure.
//!
//! Done with a tiny regex on the TS source rather than a full TS parser
//! because `CODEC_OPS` is hand-maintained and the literal block is the
//! source of truth. If the file format ever changes (e.g. moves into a
//! generated file), update this test in lockstep.

use lamquant_ops::op_spec;

const TS_PATH: &str = "../../gui/src/lib/codec.ts";

fn extract_op_ids() -> Vec<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(TS_PATH);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    let mut ids = Vec::new();
    for line in text.lines() {
        // Match lines like:   { id: 'encode',          ... }
        if let Some(id_start) = line.find("id:") {
            let after = &line[id_start + 3..];
            // Find first quote.
            let q = after.find('\'');
            if let Some(q) = q {
                let after_q = &after[q + 1..];
                if let Some(end) = after_q.find('\'') {
                    ids.push(after_q[..end].to_string());
                }
            }
        }
    }
    ids
}

#[test]
fn every_codec_ts_op_id_has_a_rust_spec() {
    let ids = extract_op_ids();
    assert!(
        ids.len() >= 8,
        "expected at least 8 ops in codec.ts, parsed only {} ({:?})",
        ids.len(),
        ids
    );

    let mut missing = Vec::new();
    for id in &ids {
        if op_spec(id).is_none() {
            missing.push(id.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "TS codec.ts contains {} op id(s) without a Rust op_spec entry: {:?}",
        missing.len(),
        missing,
    );
}
