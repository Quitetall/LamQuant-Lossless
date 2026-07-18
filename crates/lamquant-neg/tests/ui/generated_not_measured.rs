// The load-bearing invariant of ADR 0114: a Generated node (a synthetic/enhanced
// sample) can NEVER be passed where Measured evidence is required. This file must
// FAIL to compile with a type mismatch (E0308) — trybuild pins the diagnostic.

use lamquant_neg::class::{Generated, Measured};
use lamquant_neg::{Node, NodePayload, Provenance};

// A consumer that only accepts measured evidence.
fn diagnose(_evidence: &Node<Measured>) {}

fn main() {
    let synthetic = Node::<Generated>::new(
        NodePayload::default(),
        Provenance::root("lmq-generative-decoder"),
        None,
    );

    // ERROR: expected `&Node<Measured>`, found `&Node<Generated>`.
    diagnose(&synthetic);
}
