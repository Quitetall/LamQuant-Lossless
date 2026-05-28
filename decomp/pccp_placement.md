# PCCP Scaffold Placement Audit

**Date:** 2026-05-27  
**Auditor:** Automated PCCP scope analysis  
**Status:** Final recommendation ready for Phase 2

---

## Inventory of pccp/

### Root-level files (governance docs)
| File | Size | Description |
|---|---|---|
| `PCCP.md` | 7.7K | FDA-shaped scaffold root + PCCP lifecycle flowchart |
| `01-modifications.md` | 6.0K | Description of Modifications (DoM) — Class A/B/C/D authorization |
| `02-protocol.md` | 9.1K | Modification Protocol (MP) — how changes are verified |
| `03-impact.md` | 8.1K | Impact Assessment (IA) — risk-benefit analysis |
| `CHANGELOG.md` | 4.9K | Chronological modification log (3 entries to date) |
| `registry.yaml` | 7.4K | Machine-readable model+criteria registry |
| `.registry.lock` | 0 bytes | fcntl-lockfile for atomic registry updates |

### Templates
| File | Size | Description |
|---|---|---|
| `templates/change_request.md` | 2.7K | Pre-change form (sections 1–11: ID, classification, criteria) |
| `templates/verification_report.md` | 3.0K | Post-change form (gate results, subgroup analysis) |
| `templates/algorithm_change_audit.md` | 942 bytes | Optional decision-log capture (Class C/D only) |
| `templates/model_card.md` | 3.2K | Per-checkpoint identity template |

### Per-model cards
| File | Size | Model | Role |
|---|---|---|---|
| `models/encoder.md` | 4.5K | TernaryMobileNetV5_Subband | Lossy EEG encoder (firmware + cloud) |
| `models/decoder.md` | 2.6K | LamQuant Decoder | Cloud-side reconstruction |
| `models/snn.md` | 5.1K | MambaSNN | On-device seizure detection |
| `models/tnn.md` | 4.0K | Ternary Firmware NN | RP2350/ESP32 seizure detection |
| `models/oracle.md` | 2.1K | LamQuant Oracle | Teacher model (training-time only) |

### Verification records
| File | Size | Change ID | Target model(s) | Status |
|---|---|---|---|---|
| `verification_records/2026-05-09-initial-baseline.md` | 3.1K | PCCP-CHG-2026-000-INIT | encoder, decoder, snn, tnn, oracle | Initial baseline (no changes) |
| `verification_records/PCCP-CHG-2026-002.gate.json` | 3.6K | PCCP-CHG-2026-002-FIRMWARE-MAMBA | snn (firmware implementation) | Gate FAIL (deferral) |
| `verification_records/PCCP-CHG-2026-003-N80-BASELINE.gate.json` | 3.7K | PCCP-CHG-2026-003 | snn | Baseline capture |
| `verification_records/PCCP-CHG-DRYRUN.gate.json` | 3.4K | PCCP-CHG-DRYRUN | (test run) | Dry-run test |

---

## Findings per task

### 1. Registry scope: which models?

**All five models are neural-only. No lossless:**

```yaml
models:
  encoder:          # TernaryMobileNetV5_Subband (lossy LMQ encoding)
    production_version: 1.2.0
    authorized_classes: [A.1, A.2, B.3, B.4, C.1, D.1, D.2, D.3]
  
  decoder:          # Cloud reconstruction (lossy)
    production_version: 1.2.0
    authorized_classes: [A.1, B.3, B.4, C.1]
  
  snn:              # MambaSNN (seizure detection)
    production_version: 0.4.1
    authorized_classes: [A.1, A.3, B.1, B.2, B.3, B.4, C.2]
  
  tnn:              # Ternary Firmware NN (seizure detection)
    production_version: 0.3.0
    authorized_classes: [A.1, A.3, B.1, B.2, C.3, D.1, D.2]
  
  oracle:           # Teacher model (training-time only)
    production_version: 0.5.0
    authorized_classes: [A.1, B.4]
```

**Evidence:** PCCP.md §2 Scope explicitly lists: "Lossless components (LML codec, EDF reader, container, archive) are **deterministic, bit-exact, and not AI/ML** — they are governed by the standard SDLC and do not appear in this PCCP."

---

### 2. Modification Classes (01-modifications.md): any lossless scope?

**No. All four Classes (A, B, C, D) are model-only:**

| Class | Focus | Example |
|---|---|---|
| **A** | Re-training on additional data | A.1: periodic EEG corpus expansion; A.2: noise-floor recalibration; A.3: active-learning |
| **B** | Hyperparameter / threshold tuning | B.1: seizure threshold; B.2: post-processing windows; B.3: loss weights; B.4: optimizer choice |
| **C** | Architecture variant within family | C.1: encoder backbone swap; C.2: SNN block count; C.3: TNN compression ratio |
| **D** | Bit-width / quantization changes | D.1: activation precision; D.2: weight precision; D.3: CDF-LUT entry count |

**Section "Out-of-scope"** explicitly lists:
- New model classes
- New dataset modalities
- New target hardware
- Change of intended use
- **"Removal of lossless-mode bit-exact guarantee"** (D.3 is quantization, not lossless-mode removal)

**Conclusion:** The modification framework is entirely neural-model-centric. Lossless codec changes would go through standard SDLC, not PCCP.

---

### 3. `ai_models/pccp_gate.py` runtime gate: neural-only or codec hooks?

**100% neural-model focused. Zero lossless coupling.**

Key findings:

| Aspect | Evidence |
|---|---|
| **Model evaluators** | Only `METRIC_EVALUATORS` dict has five keys: `encoder`, `decoder`, `snn`, `tnn`, `oracle` — no lossless components |
| **Acceptance metrics** | pearson_r (codec), sensitivity/FPR (seizure detection), latency, memory — **none measure lossless properties** |
| **Predicate-delta tolerances** | All delta specs compare learned-model metrics (R, sensitivity, FPR, latency). Zero codec-metric fields. |
| **Registry load** | Gate reads `pccp/registry.yaml` solely for model specs + acceptance criteria. No codec version pinning, no LML wire-format checks. |
| **Subprocess calls** | evaluate_encoder shells out to `ai_models/student/eval_fullband.py` (neural); evaluate_snn shells out to `scripts/snn_pccp_eval.py` (neural). No lossless evaluators exist. |
| **Promotion side effects** | On PASS: updates `registry.yaml` model SHA-256 + last_metrics, appends CHANGELOG. Zero writes to lossless-codec paths or build artifacts. |

**Concrete example from code (lines 508–519):**
```python
_ENCODER_KEYS = (
    "pearson_r", "cr_avg", "cr_worst", "param_count",
    "weight_memory_kb", "activation_memory_kb", "latency_rp2350_ms",
)
_DECODER_KEYS = ("pearson_r_cloud", "latency_gpu_ms", "gpu_memory_gb")
_SNN_KEYS = ("sensitivity", "false_positive_rate_per_hour", "param_count", "latency_gpu_ms")
_TNN_KEYS = (
    "sensitivity", "false_positive_rate_per_hour",
    "latency_rp2350_ms", "latency_esp32s3_ms", "latency_esp32p4_ms",
    "sram_footprint_kb", "flash_footprint_kb",
)
_ORACLE_KEYS = ("training_set_r",)
```

All metrics are neural-model outputs. No LML codec stats.

---

### 4. CHANGELOG.md: any lossless-related entries?

**No. All three entries are model modifications:**

| Entry | Class | Models | Description |
|---|---|---|---|
| PCCP-CHG-2026-000-INIT | initial-baseline | encoder, decoder, snn, tnn, oracle | Baseline establishment (no changes) |
| PCCP-CHG-2026-001-ADAPTIVE-FSQ-WIRED | C.3 | snn → codec wiring | "Wired production neural codec so SNN activity drives FSQ level selection" (code-only, no weights) |
| PCCP-CHG-2026-002-FIRMWARE-MAMBA-INFERENCE | C.2 | snn (firmware) | "Replaced dLIF stub with Mamba pipeline" (firmware implementation, same weights) |

**Note on CHG-2026-001:** While it mentions "neural codec wiring," the change is in how the neural SNN's output drives the codec's adaptive FSQ—this is neural-model orchestration, not lossless codec modification. The registry pins and metrics remain model-centric.

---

### 5. Change-request template: generic or neural-specific?

**Generic structure, but every example is neural-model focused:**

| Section | Template scope | Actual usage |
|---|---|---|
| 1. Identification | Any change ID format | Always PCCP-CHG-YYYY-NNN |
| 2. Classification | Reference to modification class A/B/C/D | All classes defined for models only |
| 3. Description | Plain-language change summary | All examples: model training, architecture, hyperparameter |
| 5. Proposed parameters | Hyperparams, training cmd, architecture variant | Neural training commands; no codec build flags |
| 6. Acceptance criteria check | Reference registry.yaml metrics | Registry only lists model metrics (R, sensitivity, FPR, latency) |
| 8. Subgroup analysis plan | "Sample sizes, age bands, seizure types, sites" | Explicit clinical subgroup enumeration (pediatric, adult, geriatric, per-site) |

**Conclusion:** Template is **structurally reusable** but **operationally neural-only** because the entire classification scheme (A–D) and acceptance framework assume AI/ML model changes.

---

### 6. Per-model verification records: which repo?

**All target neural models:**

| Record | Model(s) | Repo destination |
|---|---|---|
| 2026-05-09-initial-baseline.md | encoder, decoder, snn, tnn, oracle | LamQuant-Neural (PRIVATE) |
| PCCP-CHG-2026-002.gate.json | snn (firmware path) | LamQuant-Neural (PRIVATE) |
| PCCP-CHG-2026-003-N80-BASELINE.gate.json | snn | LamQuant-Neural (PRIVATE) |
| PCCP-CHG-DRYRUN.gate.json | (test run) | LamQuant-Neural (PRIVATE) |

**Evidence:** Every verification record contains:
- `models_affected: [encoder | decoder | snn | tnn | oracle]`
- References to `registry.yaml` entries (neural-model-specific)
- Acceptance result columns (metric-specific to each model)
- Signatures from neural-team reviewers

---

### 7. Test suite (`tests/pccp/`): which bucket?

**Both test files are pure neural-gate unit tests:**

| Test file | Coverage | Scope | Repo |
|---|---|---|---|
| `test_pccp_gate.py` | Input validators, SHA-256, criterion evaluation, predicate-delta parsing, registry I/O | Pure logic of gate (no model eval, no subprocess) | LamQuant-Neural |
| `test_pccp_gate_integration.py` | run_gate, promote, append_changelog, write_verdict_log, main CLI | End-to-end gate orchestration + filesystem atomicity | LamQuant-Neural |

Both import `from ai_models import pccp_gate` and mock evaluators. Zero codec tests.

**Mark:** `pytestmark = pytest.mark.l2` (L2 = "core AI/ML logic", per project convention).

---

### 8. Cross-coupling: lossless code importing PCCP?

**Minimal, read-only, informational only:**

| File | Import | Usage | Impact |
|---|---|---|---|
| `lamquant-core/src/bin/lml.rs` (lines 9091–9108) | `pccp/registry.yaml` search path | Locate registry to print `lml --version` model card | **Read-only:** gate does NOT promote lossless. Gate gate cannot break lossless backward-compat. |
| `lamquant-core/src/bin/lml.rs` (line 1009) | Comment: "Print device + model version card from pccp/registry.yaml" | Extract model version strings for user-facing version output | **Informational:** registry is source-of-truth for model version labeling |
| `reference_implementations/python_codec/lamquant_codec/cli_codec.py` | Mentioned in comment about "registry pin from pccp/registry.yaml" | Presumably checks SNN weight pin for adaptive FSQ feature | **Feature coupling:** codec adaptive-FSQ wiring requires SNN registry pin; but this is neural-model orchestration, not codec modification |

**Verification:** No `grep` match for `import pccp` or `from pccp` in codec/core files. The lml binary parses YAML but doesn't execute gate logic.

**Conclusion:** Read-only informational dependency. Lossless code **queries** the registry (model version, SNN pin for adaptive FSQ) but never **modifies** PCCP artifacts. Gate cannot cause lossless breakage.

---

### 9. Summary: scope determination

| Component | Governed by | Owner repo |
|---|---|---|
| **Registry (model pins + acceptance)** | PCCP gate | LamQuant-Neural |
| **Gate enforcement (pccp_gate.py)** | PCCP gate | LamQuant-Neural |
| **Change templates** | PCCP gate | LamQuant-Neural |
| **Verification records** | PCCP gate | LamQuant-Neural |
| **Per-model cards** | PCCP gate | LamQuant-Neural |
| **CHANGELOG (model changes)** | PCCP gate | LamQuant-Neural |
| **Test suite (gate logic)** | PCCP gate | LamQuant-Neural |
| **lml --version info** | Informational only | LamQuant-Codec (reads registry, no write) |
| **Codec adaptive-FSQ wiring** | Neural orchestration | LamQuant-Neural (SNN pin → codec feature flag) |

---

## Final Recommendation: **P1 (Neural-owned)**

### Justification

1. **Registry scope is 100% neural-model-centric.** Five models (encoder, decoder, snn, tnn, oracle) + five hyperparameter sets, all AI/ML. CLAUDE.md explicitly excludes lossless from PCCP scope.

2. **Gate enforcement is neural-only.** The `pccp_gate.py` script reads model metrics (pearson_r, sensitivity, FPR, latency, memory) that are **produced by neural evaluators only**. It never measures or enforces lossless-codec properties.

3. **Modification classes (A–D) enumerate neural-model changes exclusively.** Class C (architecture variant) and Class D (bit-width) explicitly exclude "removal of lossless-mode bit-exact guarantee"—implying lossless is outside PCCP governance.

4. **Verification records, templates, and model cards are all neural-model forms.** The subgroup-analysis structure (pediatric vs adult vs geriatric, per seizure type, per site) is clinical-evidence oriented, not codec-performance oriented.

5. **Test coverage is gate-logic only.** No codec tests; all unit tests exercise PCCP gate validators and orchestration.

6. **Lossless coupling is read-only and informational.** The lml binary queries the registry for model version strings (for user-facing labeling) and the codec reads the SNN registry pin to enable/disable adaptive-FSQ feature, but:
   - Neither path modifies PCCP artifacts
   - Neither path enforces PCCP acceptance criteria
   - Codec can always fall back to non-adaptive-FSQ (safety mechanism in place per CHANGELOG-CHG-2026-001: "SNN load fails loudly if registry pin unset")

7. **Regulatory posture is FDA-shaped change-control for AI/ML.** The PCCP references FDA guidance on "Marketing Submission Recommendations for AI-Enabled Device Software Functions"—the regulatory entity is the **neural model changes**, not the lossless transform codec.

8. **Ownership alignment:** PCCP author is Brian Lam (regulatory affairs, AI/ML focus). Lossless codec development is under separate SDLC (codec perf benchmarking, byte-equal golden gates, firmware/no_std CI).

### P1 Implementation

**Entire PCCP scaffold (directories + files below) ships in LamQuant-Neural (PRIVATE):**

```
LamQuant-Neural/
├── pccp/                          (entire directory)
│   ├── PCCP.md
│   ├── 01-modifications.md
│   ├── 02-protocol.md
│   ├── 03-impact.md
│   ├── CHANGELOG.md
│   ├── registry.yaml
│   ├── .registry.lock
│   ├── models/
│   │   ├── encoder.md
│   │   ├── decoder.md
│   │   ├── snn.md
│   │   ├── tnn.md
│   │   └── oracle.md
│   ├── templates/
│   │   ├── change_request.md
│   │   ├── verification_report.md
│   │   ├── algorithm_change_audit.md
│   │   └── model_card.md
│   └── verification_records/
│       └── (all signed gate logs)
├── ai_models/
│   ├── pccp_gate.py               (gate enforcement)
│   ├── pccp_gate.py.tests → tests/pccp/
│   └── (all neural model code)
├── tests/
│   └── pccp/
│       ├── test_pccp_gate.py
│       └── test_pccp_gate_integration.py
└── docs/
    └── (compliance docs referencing pccp/)
```

**LamQuant-Codec (PUBLIC) retains:**
- Read-only queries to `registry.yaml` (model version labeling, SNN pin for adaptive FSQ)
- No modification of PCCP state
- No PCCP enforcement logic
- Standard SDLC for lossless codec changes (byte-equal gates, perf benchmarks, CI lanes)

---

## Cross-coupling discovered

### Read-only PCCP dependency in lml binary

**File:** `/mnt/4tb/LamQuant/lamquant-core/src/bin/lml.rs`  
**Lines:** 9091–9108 (registry path search); 1009 (comment)  
**Nature:** Informational coupling (read-only)

```rust
fn pccp_registry_path() -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if let Ok(p) = std::env::var("LAMQUANT_REGISTRY_PATH") {
        return Ok(PathBuf::from(p));
    }
    // Walk up from the binary location and from cwd looking for pccp/registry.yaml.
    for start in candidates {
        let mut cur: &Path = &start;
        loop {
            let candidate = cur.join("pccp").join("registry.yaml");
            if candidate.exists() {
                return Ok(candidate);
            }
            // ...
        }
    }
    Err("pccp/registry.yaml not found (set LAMQUANT_REGISTRY_PATH)".into())
}
```

**Usage:** Populate `lml --version` card with model metadata (encoder v1.2.0, snn v0.4.1, etc.).

**Impact on decomposition:**
- lml binary is in `lamquant-core` → LamQuant-Codec (PUBLIC).
- PCCP registry is in LamQuant-Neural (PRIVATE).
- Solution: At link/package time, LamQuant-Codec build includes a read-only snapshot of `registry.yaml` (model version strings only), not live gate state.
- **Action item:** Create `lamquant_codec/registry_snapshot.yaml` (or inline model version enum) that mirrors the neural registry for version labeling. Update at release time, not at gate time.

### Codec adaptive-FSQ wiring

**File:** `reference_implementations/python_codec/lamquant_codec/cli_codec.py`  
**Nature:** Feature flag (codec behavior depends on SNN registry pin)

Per CHANGELOG-CHG-2026-001: "Wired the production neural codec so the MambaSNN's per-timestep activity classification drives FSQ level selection (L=2 quiet / L=3 active / L=5 seizure)."

**Impact on decomposition:**
- Codec adaptive-FSQ feature is **enabled by presence of valid SNN pin** in registry.
- If SNN pin is missing or invalid, codec fails loudly (no silent fallback).
- Solution: SNN registry pin is canonical source-of-truth (in LamQuant-Neural). At release time, codec reads this pin and bakes it into the release artifact. After release, codec behavior is frozen until next release update.
- **Action item:** Codec release process includes `registry.yaml` SNN-pin extraction → compiled feature flag (not runtime lookup).

---

## Action items for Phase 2

- [ ] **Move `pccp/` directory** from `/mnt/4tb/LamQuant/` to `LamQuant-Neural/` repo (PRIVATE)
- [ ] **Move `ai_models/pccp_gate.py`** to `LamQuant-Neural/ai_models/pccp_gate.py`
- [ ] **Move `tests/pccp/`** to `LamQuant-Neural/tests/pccp/`
- [ ] **Create `lamquant_codec/registry_snapshot.yaml`** containing read-only model version strings (encoder, decoder, snn, tnn, oracle versions only; no acceptance criteria or gate state)
- [ ] **Update `lamquant-core/src/bin/lml.rs`** to:
  - First try env var `LAMQUANT_REGISTRY_SNAPSHOT_PATH`
  - Fall back to `lamquant_codec/registry_snapshot.yaml` (baked into Codec release)
  - Remove fallback search walk-up to `pccp/registry.yaml` (no longer available post-split)
- [ ] **Update codec CI/release pipeline** to:
  - Accept `registry.yaml` SNN-pin as parameter at release time
  - Bake pin into codec artifact (adaptive-FSQ feature flag)
  - No runtime dependency on LamQuant-Neural registry
- [ ] **Update CLAUDE.md** PCCP section to reflect new ownership: "PCCP scaffold (gate, registry, templates, verification records) is housed in LamQuant-Neural (PRIVATE). Codec reads model version strings from a released snapshot only (no live dependency on gate state)."
- [ ] **Update decomposition plan** (per `.claude/plans/optimized-seeking-bentley.md`) to reflect P1 finalization and cross-repo read-only dependency mitigation strategy

---

## Conclusion

**The PCCP scaffold is 100% neural-model governance. P1 (Neural-owned) is correct and required for regulatory coherence.**

The only lossless coupling is read-only (version labeling) and feature-gating (SNN pin → codec adaptive-FSQ flag). Both can be satisfied post-decomposition via snapshot + release-time binding, eliminating live cross-repo dependencies on gate state.

