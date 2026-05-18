# Full-decode conformance — out-of-scope for v1, see verify.py

The conformance suite in this directory ships a **validation-only**
harness (`verify.py`). It inspects the wire-format envelope — header,
window-level CRC-32, LMLFOOT1 footer, metadata JSON — but does not
invert the codec's three signal passes (entropy decode → LPC inverse
→ Le Gall 5/3 lifting inverse).

The authoritative full decoder is the LamQuant Rust crate. If your
implementation needs sample-level parity with LamQuant, run the
decoded sample matrix through both:

```sh
# LamQuant reference
cargo build --release --bin lml --features host
target/release/lml decode foo.lml -o golden.raw

# Your implementation
your-decoder foo.lml > your.raw

# Compare
cmp golden.raw your.raw
```

A clean-room pure-Python reference decoder is a future ambition. If
you build one (≤ 600 LOC seems achievable per our internal estimate
based on the entropy + LPC + lifting LOC in `lamquant-core/src/`),
open a PR linking it here.

Honest position: the codec design is intentionally simple at the
byte level — integer-only, no FP rounding, three well-documented
passes — so a small reference decoder is feasible. But "small" still
means more than `verify.py` could reasonably ship as a single
self-contained file, so we shipped the validation-only harness now
and left the full decoder for a follow-up.
