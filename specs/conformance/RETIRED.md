# RETIRED 2026-08-16 — `verify.py` here is superseded; do not run it

The live conformance harness is in the meta repository:

    docs/specs/conformance/verify.py     (301 lines, 30 vectors)

This directory holds the 2026-05-18 original (168 lines, 27 vectors). Nothing
runs it: this repository's CI exercises the cargo integration suites
(`conformance`, `lma_conformance`, the ABIR/BCS2 and LMQ wire lanes), and the
meta's `ci.yml::conformance` job invokes the meta's copy.

## Why running this one is actively harmful, not just redundant

It defaults to `--binary target/release/lml`:

```python
default=str(Path(__file__).resolve().parent.parent.parent / "target" / "release" / "lml")
```

The vectors carry frozen legacy magics (LML1). **The current `lml` binary
refuses those by design** — decode-forever is the legacy Adapter's contract, not
main's. So this harness points at a component that is supposed to reject its
inputs, every negative vector "fails", and the obvious way to make it pass is to
weaken main's refusal. That is the trap; `cad84002` (2026-07-31, "point the
conformance suite at the component that owns these vectors") is the fix, and it
landed in the meta's copy only.

The live harness also stopped flattening the error taxonomy. It speaks the
Adapter's JSON protocol and maps its typed `code` field:

| adapter code    | conformance kind |
|-----------------|------------------|
| `crc-mismatch`  | `CrcMismatch`    |
| `truncated`     | `Truncated`      |
| `unknown-magic` | `InvalidMagic`   |

A corrupted payload and an incomplete file stay distinguishable. The copy here
predates that and sniffs substrings out of a human-readable stderr stream.

## Why it is kept

Retire by sequester, never delete. `bonsai.toml` declares
`codec-lossless/specs/conformance` and its `vectors/` and `ascii_vectors/`
children as structural nodes, and the vectors themselves are a publishable
artifact third-party LML readers are invited to run against. Only `verify.py` is
superseded — take the vectors, use the meta's runner.
