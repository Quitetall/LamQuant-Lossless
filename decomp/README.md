# LamQuant Monorepo Decomposition Planning

This directory contains the strategic and tactical planning documents for decomposing the LamQuant monorepo into 8 separate repositories following Unix philosophy principles.

## Documents

### fixture_plan.md (1191 lines)
**Complete fixture infrastructure audit and migration strategy**

- Root `conftest.py` inventory (sys.path, fixtures, hooks, markers)
- Per-subdir conftest analysis (codec, training)
- `tests/fixtures/` file-by-file breakdown (synthetic.py, __init__.py)
- `tests/helpers/` file-by-file breakdown (7 modules: asserts, data_paths, edf_factory, roundtrip, rust_bindings, signals)
- Cross-repo coupling identification and resolution
- Per-repo conftest.py drafts (Lossless, Neural, Eagle)
- lamquant_common_testdata package structure
- benchmark_c_parity decision (leave as-is, Eagle installs Lossless)
- Shared fixture strategy (chose: Python package in Lossless repo)
- 7-phase executable migration steps with verification gates
- Risk assessment and timeline

**Key findings:**
- Moderate coupling via real-EDF loaders, signal generators, and sys.path manipulation
- **Single resolution**: Create `lamquant_common_testdata` Python package in Lossless repo
- All 3 destination repos install it as a dependency
- No fixture duplication; no drift risk

### import_map.md (baseline)
Pre-decomposition import dependency snapshot for reference.

---

## Destination Repos

After split, LamQuant becomes **8 separate repos**:

### Public (can be released as open-source packages)
1. **lamquant-lossless** — Codec, container, firmware, C parity, EDF reader
   - Contains `lamquant_common_testdata` Python package
   - Used by Neural and Eagle as a dependency
   
2. **lamquant-eagle** — Public benchmarks, validation, audits
   - Depends on Lossless (public)
   - Can run tests standalone

### Private (internal use)
3. **lamquant-neural** — Training, SNN, student models, dataset_sim, architectures
   - Depends on Lossless (public)
   - Cannot be released without licensing review

### Utilities
4. **lamquant-common** — Shared build config, CI templates, docs
5. **lamquant-firmware-stubs** — (if firmware is vendored)
6. **lamquant-reference-software** — (external data corpus)
7. **lamquant-weights** — (pre-trained checkpoints)
8. **lamquant-docs** — (consolidated documentation)

---

## Migration Workflow

**Phase 1: Create Common package** (fixture_plan.md, Phase 1)
- Create `tests/common_testdata/` in Lossless repo
- Move fixtures and helpers
- Test Lossless suite passes

**Phase 2: Update Lossless conftest** (fixture_plan.md, Phase 2)
- Import fixtures from `lamquant_common_testdata`
- Remove sys.path for `ai_models/*`
- Keep `firmware/` on sys.path
- Test Lossless suite passes

**Phase 3: Migrate Lossless subdir conftests** (fixture_plan.md, Phase 3)
- Copy `tests/codec/conftest.py` (no changes needed)
- Already self-contained

**Phase 4–5: Set up Neural and Eagle** (fixture_plan.md, Phase 4–5)
- Create minimal conftests in each repo
- Add Lossless as a dependency
- Install and test independently

**Phase 6: Smoke test all repos** (fixture_plan.md, Phase 6)
- Run all three suites together

**Phase 7: Test standalone execution** (fixture_plan.md, Phase 7)
- Delete sibling repos; verify Eagle (or Neural) runs alone
- Confirms no hidden dependencies

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| **Shared package location: Lossless repo** | Lossless is the only repo suitable for public release; contains all fixture/helper code; no circular deps |
| **Package name: `lamquant_common_testdata`** | Descriptive; installed to site-packages as `lamquant_common_testdata` |
| **Conditional imports in roundtrip.py** | Makes module safe to import by Neural/Eagle even if lamquant_codec unavailable |
| **benchmark_c_parity stays in Eagle** | Benchmark is for public use; Lossless is a public dependency; no splits needed |
| **No fixture duplication** | Single source of truth; risk of drift eliminated |
| **ternary_model in Common** | Used by Lossless (firmware) + Neural (training); must be shared |
| **sample_eeg_q31/float stay in codec/conftest** | Codec-specific; no cross-repo deps; no need to move |
| **real-EDF loaders in Common** | Multi-repo use; skip-safe design; no private module imports |
| **Per-repo sys.path in each conftest** | Clear intent; no global coupling; each repo declares its own needs |

---

## Files Involved in Split

### Moves to `lamquant_common_testdata/` (in Lossless repo)
```
tests/fixtures/__init__.py                  → fixtures/__init__.py
tests/fixtures/synthetic.py                 → fixtures/synthetic.py
tests/helpers/__init__.py                   → helpers/__init__.py
tests/helpers/asserts.py                    → helpers/asserts.py (make imports conditional)
tests/helpers/data_paths.py                 → helpers/data_paths.py
tests/helpers/edf_factory.py                → helpers/edf_factory.py
tests/helpers/roundtrip.py                  → helpers/roundtrip.py (make imports conditional)
tests/helpers/rust_bindings.py              → helpers/rust_bindings.py
tests/helpers/signals.py                    → helpers/signals.py
tests/conftest.py (core fixture defs)       → conftest.py (extracted)
```

### Stays in each destination repo
```
Lossless:
  tests/codec/conftest.py (unchanged)
  tests/conftest.py (new, imports from Common)

Neural:
  tests/training/conftest.py (unchanged)
  tests/conftest.py (new, imports from Common)

Eagle:
  tests/conftest.py (minimal, imports from Common)
```

### Deleted (no longer needed)
```
None — all content is preserved in Common or destination conftests
```

---

## Verification Checklist

After each migration phase, run:

```bash
# Phase 1 (Lossless with Common package created)
pytest tests/codec/ tests/firmware/ -x

# Phase 2 (Lossless conftest updated)
pytest tests/codec/ tests/firmware/ tests/c_host/ -x

# Phase 3 (Lossless subdir conftests)
pytest tests/codec/ -x

# Phase 4 (Neural setup)
cd ../lamquant-neural && pytest tests/training/ tests/snn/ -x

# Phase 5 (Eagle setup)
cd ../lamquant-eagle && pytest tests/benchmarks/ tests/validation/ -x

# Phase 7 (Standalone Eagle test — siblings not checked out)
mkdir /tmp/eagle-isolated && cd /tmp/eagle-isolated
git clone <eagle-repo> .
pip install lamquant-lossless
pytest tests/ -x
```

---

## Contact & Questions

Refer to **fixture_plan.md** for:
- Detailed per-fixture migration notes
- Import path adjustments required
- Risk assessment and mitigation strategies
- Executable migration steps with full context

