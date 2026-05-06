//! Cross-language parity for the top-level workflow inventory.
//!
//! Three places define the 10 workflows with their canonical keys:
//!
//!   - `specs/ui-parity.md::Workflow inventory` (markdown — human canonical)
//!   - `gui/src/lib/workflows.ts::WORKFLOWS` (TS — GUI hub tiles)
//!   - `lamquant.py:1963-1984` (Python — TUI main menu)
//!
//! Drift between the TS hub and the Python TUI lets users hit a tile via
//! `[2]` in one and end up somewhere different in the other. This test
//! parses both sources and asserts the (key, id-ish) pairs match.

use std::collections::BTreeMap;

const WORKFLOWS_TS: &str = "../../gui/src/lib/workflows.ts";
const LAMQUANT_PY: &str = "../../lamquant.py";

fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // manifest = .../crates/lamquant-ops; workspace root is two up.
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

/// Extract `(key, title)` pairs from `workflows.ts`. Each entry uses
/// `key: '1'` and `title: 'LML Lossless'` literals — single regex covers it.
fn extract_ts_pairs() -> BTreeMap<String, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(WORKFLOWS_TS);
    let text = std::fs::read_to_string(&path).expect("read workflows.ts");
    let mut out = BTreeMap::new();
    for line in text.lines() {
        // Accept lines like:
        //   { key: '1', id: 'lossless', title: 'LML Lossless', ... }
        let key = match find_field(line, "key") {
            Some(k) => k,
            None => continue,
        };
        let title = match find_field(line, "title") {
            Some(t) => t,
            None => continue,
        };
        out.insert(key, title);
    }
    out
}

/// Extract `(key, label)` pairs from `lamquant.py` between WORKFLOWS and
/// SYSTEM blocks. Match `("1", "LML Lossless", "...")` 3-tuples.
fn extract_python_pairs() -> BTreeMap<String, String> {
    let path = workspace_root().join("lamquant.py").canonicalize().unwrap_or_else(|_| {
        std::path::PathBuf::from(LAMQUANT_PY)
    });
    let text = std::fs::read_to_string(&path).expect("read lamquant.py");
    let mut out = BTreeMap::new();
    // Heuristic: lines with three quoted strings and a numeric/single-letter
    // first quote — that's how the menu is written.
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with('(') || !line.ends_with(')') && !line.ends_with("),") {
            continue;
        }
        // Pull out the three quoted segments in order.
        let mut chars = line.chars().peekable();
        let mut quoted: Vec<String> = Vec::new();
        while let Some(c) = chars.next() {
            if c == '"' {
                let mut buf = String::new();
                for c2 in chars.by_ref() {
                    if c2 == '"' { break; }
                    buf.push(c2);
                }
                quoted.push(buf);
                if quoted.len() == 3 { break; }
            }
        }
        if quoted.len() != 3 { continue; }
        let key = &quoted[0];
        // Filter out the system block items keyed by single letters.
        if !is_workflow_or_system_key(key) {
            continue;
        }
        out.insert(key.clone(), quoted[1].clone());
    }
    out
}

fn find_field(line: &str, name: &str) -> Option<String> {
    let needle = format!("{}:", name);
    let i = line.find(&needle)?;
    let after = &line[i + needle.len()..];
    let q = after.find('\'')?;
    let after_q = &after[q + 1..];
    let end = after_q.find('\'')?;
    Some(after_q[..end].to_string())
}

fn is_workflow_or_system_key(k: &str) -> bool {
    matches!(k, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "s" | "i" | "t")
}

#[test]
fn workflow_keys_match_between_ts_and_python() {
    let ts = extract_ts_pairs();
    let py = extract_python_pairs();

    if ts.is_empty() {
        panic!("could not parse any workflows from workflows.ts");
    }
    if py.is_empty() {
        panic!("could not parse any workflows from lamquant.py main menu");
    }

    let ts_keys: std::collections::BTreeSet<_> = ts.keys().cloned().collect();
    let py_keys: std::collections::BTreeSet<_> = py.keys().cloned().collect();
    let missing_in_py: Vec<_> = ts_keys.difference(&py_keys).cloned().collect();
    let missing_in_ts: Vec<_> = py_keys.difference(&ts_keys).cloned().collect();
    assert!(
        missing_in_py.is_empty(),
        "TS has workflow keys absent from Python lamquant.py: {:?}",
        missing_in_py
    );
    assert!(
        missing_in_ts.is_empty(),
        "Python has workflow keys absent from TS workflows.ts: {:?}",
        missing_in_ts
    );

    // Spot-check the LamQuant identity: keys 1 and 2 are the codec subhubs.
    assert!(ts.contains_key("1"));
    assert!(ts.contains_key("2"));
    assert!(ts.get("1").map(|t| t.contains("LML")).unwrap_or(false));
    assert!(ts.get("2").map(|t| t.contains("LMQ")).unwrap_or(false));
}
