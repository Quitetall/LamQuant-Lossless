//! N1 (ADR 0114) — the type barrier as a first-class, CI-visible `#[test]`.
//!
//! The `compile_fail,E0308` doctest in `lib.rs` already proves `Node<Generated>`
//! cannot reach a `&Node<Measured>` consumer, but doctests run only under
//! `cargo test --doc`. This promotes the barrier to a normal test that runs in
//! the `--lib`/`--test` lanes too, and (via the committed `.stderr` fixture)
//! pins the *exact* compiler diagnostic — so a refactor that accidentally makes
//! the two node classes interchangeable fails here loudly, not silently.
//!
//! To regenerate the fixture after a deliberate change (or a toolchain bump):
//! `TRYBUILD=overwrite cargo test -p lamquant-neg --test compile_fail`.

#[test]
fn generated_cannot_reach_a_measured_consumer() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/generated_not_measured.rs");
}
