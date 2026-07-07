# LamQuant Optimum (LMO) — Lossless Codec Specification

**Version:** v1 (specification) · **Date:** 2026-07-06 · **Tier:** ADR 0052 Tier-3 (Optimum)

**Provenance:** synthesized from the `lamquant-lml-optimum` source + the shared MCU kernel, ADRs 0011 / 0051 / 0052 / 0054 / 0062 / 0064 / 0076, and the 2026-06→07 technique-mine + measurement registries. The H.BWC win-record and the lever ranking were **re-verified against ADR 0064** (the authoritative 2026-06-28 full held-out sweep) before writing — they overturned the earlier 2026-06-21 projection docs (and my own prior recollection): the measured dominant lever is `mv_rls`, not arithmetic coding, and the honest record is +3.06% larger / 3-of-42 wins, not "5/8".

**Reading the tags:** every claim carries `[landed]` / `[partial]` / `[planned]` / `[probe]` / `[rejected]`; every number carries **MEASURED** / **DERIVED** / **PROJECTED**. This spec obeys the project's hard no-overclaim rule — nothing aspirational is written as shipped, no unmeasured figure as measured. The inline "do not overclaim / do not re-quote" notes are deliberate guardrails, kept in the shipped spec.

Crate: `codec-lossless/lamquant-lml-optimum` (lib `lamquant_lml_optimum`, **v0.0.1**). Shared kernel: `codec-lossless/lamquant-lml-mcu/src/{lml,lifting,lpc,rans,golomb,codec}.rs` (crate renamed `lamquant-core → lamquant-lml-mcu` per ADR 0058, which amends ADR 0052's crate names).

> **Stale-map warning:** there is **no** `lamquant-lml-optimum/src/lml.rs`. The Optimum sources are `arith_int/crosschan/entropy/lmo/lmo_lossless/lmo_pcrd97/scale_cond/rls/mv_rls/tcq/wavelet97/segmentation/montage.rs`; the DWT/LPC/Golomb/rANS primitives are re-exported from the MCU crate. `entropy_backend_probe.rs` and the ~55 files under `examples/` are **research probes**, not production.

---

## 1. Charter & scope

**What Optimum is** `[landed]` — ADR 0052 Tier-3. The LMO codec is the **deterministic, maximum-compression-ratio, host-only** lossless/near-lossless tier. It is the "encode-once, archival" ratio attack: it may spend arbitrary encode time and memory, it carries **no MCU / latency / SRAM budget on the encoder**, and its mission is to beat ITU-T **H.BWC** on non-stationary clinical EEG.
*Grounding: `docs/decisions/0052-...md:60-82`; `lamquant-lml-optimum/src/lib.rs:1-24`; `Cargo.toml:1-14`.*

**Charter permissions** `[landed]` — "no restrictions on technique": float math (f64), deeper/better transforms, RD-optimal PCRD allocation, context arithmetic / range-coding tools, and H.BWC-class tools **re-implemented in patent-clean Rust**. This distinguishes Optimum from Desktop, which is *byte-identical* to the Firmware LML floor and never compresses harder itself — Desktop merely *carries* the LMO codec.
*Grounding: `0052-...md:57-82,140-143`.*

**Structural constraints** `[landed]`:
- **Deterministic ONLY** — no neural / learned models. Those live in the LMQ tier (ADR 0049), which keeps LMO out of PCCP scope. Enforced by design, not a code gate. *(`0052-...md:152-154`; `lib.rs:17-19`.)*
- **Encode is host-only** (`std` + `f64` + `rayon`) and is structurally excluded from the `riscv32` firmware graph via Cargo features. **Decode is `no_std`-capable** and is an optional Firmware module (`default = ["decode"]`). The encode/decode feature split is what makes the purity *structural*. *(`Cargo.toml:20-54`: `default=["decode"]`, `encode=["std","decode","dep:rayon"]`, `std=["lamquant-lml-mcu/std"]`; `0052-...md:96-111`.)*

**What Optimum is NOT:**
- **Not MCU-decodable in practice** — the 9/7 float path is soft-float-heavy; decode is *`no_std`-capable by construction* (integer or IEEE `+ − × ÷` only) but has **not** been demonstrated on RP2350 silicon. Only the **floor** codec is silicon-proven. *(gotcha set 4.)*
- **Not neural.** ADR 0076's learned entropy model is banned here (§3.6). *(`0076-...:71-75`.)*
- **Not product-wired for encode** today (§12). Only decode is reachable through a product surface. *(status set.)*

---

## 2. Design goals & non-goals

**Goals (falsifiable):**

| # | Goal | Target | Status |
|---|------|--------|--------|
| G1 | Bit-exact lossless (R=1.0, PRD=0), byte-identical round-trip across the corpus | Provable | `[landed]` |
| G2 | **Never-worse than the 5/3 floor** — keep-best container, per-channel keep-smaller | Provable by construction | `[landed]` |
| G3 | Deterministic host↔decoder identity (integer or IEEE `+−×÷`) | Bit-exact | `[landed]` |
| G4 | Size reduction vs the internal 5/3 floor that **scales with inter-channel redundancy** | eegmmidb 64ch **−23.0%**, CHB-MIT 23ch **−10.7%**, ma 21ch **−9.6%**, ECG 2ch ~0 | `[landed]` MEASURED |
| G5 | **Beat H.BWC lossless on the non-stationary/clinically-active EEG niche** | Win *some* held-out task/ictal/bipolar EEG | `[partial]` — honest niche below |
| G6 | Emit **H.BWC-conformant** streams as a measurement tier | Decode oracle 9/9 goldens | `[landed]` (oracle, decode-only) |

**The honest niche for G5.** There are two measurement rounds; the **later, more honest full sweep is authoritative**:
- Early headline (`hbwc-lossless-results-2026-06.md:31-63`): LamQuant **wins 5/8** held-out mental_arithmetic recordings (margins **−1.55% to −6.84%** bytes); 3 lose (+1.24% to +3.81%); chb05 seizure **parity** (+0.05%); static 64ch eegmmidb **behind** (+1.66% to +3.29%).
- **Authoritative full held-out multi-corpus sweep, 2026-06-28** (ADR 0064:37-52): overall **+3.06% total bytes — LamQuant is ~3% LARGER**, with only **3/42 recording wins**. Per-corpus: ma **−3.79%** (win), chbmit **−0.39%** (parity), eegmmidb **+2.30%**, helsinki_neonatal **+1.78%**, tuar **+2.90%**, tuev **+2.92%**, siena **+4.16%**, tusz **+6.03%** (down from +23.4% before the LSB split).

> **Scoping rule (binding):** "beats H.BWC" is TRUE *only* for **lossless on non-stationary / clinically-active EEG** (task / ictal / bipolar). Across the full clinical corpus set LamQuant is ~3% larger and wins 3/42. Unqualified "we beat H.BWC" is an overclaim.

**Non-goals (explicit, evidence-backed):**
- **Lossy is ceded.** On the *same* EEG it wins at lossless, H.BWC is **15–100× better PRD** at every lossy working point (WP1 ×100, WP3 ×50, WP4 ×38, WP5 ×21, WP6 ×15). Lossy is a deep-transform + RD-allocation game — H.BWC's fortress. *(`lossy_head_to_head_2026-06-28.md`.)* `[landed]`
- **ECG is ceded.** H.BWC is +20.9%–37% ahead on lossless ECG; DCT+TCQ+CABAC decorrelates quasi-periodic QRS in a way sample-wise linear prediction cannot (five convergent negatives). Budget is spent on EEG. *(`0054-...:32`; `warplan-v2-...:46-62`.)* `[rejected]` as a target
- **Not FDA-cleared, not "official/licensed/vendor" H.BWC.** LMO is out of PCCP scope (standard SDLC). The reference clone is "independent, BSD-3 Rust H.BWC-draft decoder", never "official/endorsed". *(§5, `0062-...:92-99`.)*

---

## 3. The pipeline (stage-DAG)

Two families exist under one container: the **Optimum-lossless body** (`transform_id=2`) and the **9/7 float PCRD body** (`transform_id=1`, lossy). `transform_id=0` is a faithful re-encode of the LML floor (WP0 baseline). The container **auto-picks the smaller of transform_id 0 and 2 for lossless** — never worse than the floor. *(`lmo.rs:122-171`.)*

**CRITICAL lever-ranking correction.** The user's recollection matches the **2026-06-21 projection docs** (context-adaptive arithmetic = the biggest lever; tANS = the entropy upgrade; learned lifting 3–8%). The **measurement registry (updated 2026-06-28→07-05)** OVERTURNED those ranks. The **measured dominant lever is `mv_rls` — a joint spatio-temporal *predictor*, not an entropy upgrade and not a transform.** Entropy is measured **near-spent** (~0.3% realizable). Both projected and measured verdicts are reported below.
*Grounding: `optimal-eeg-codec-experiments-2026-06.md:41,96-134` vs `lossless-technique-mine-2026-06.md:59-79`.*

### Stage 0 — Spatial decorrelation (as PREDICTION, not transform)

Two distinct "cross-channel" things with **opposite verdicts**:

- **(a) Multi-reference cross-channel INTEGER prediction** `[landed]` — the core of `transform_id=2`. ≤3 references (`K≤3`); prediction `round_q16(Σ gᵣ · ch[refᵣ])`, gains fit by joint least-squares (Gaussian elimination on ridge-regularized normal equations), quantized to **Q16 i32**, shipped in the header. References chosen **byte-greedy (1st) + energy-greedy (rest)**; **per-channel keep-best raw vs predicted**. Pure integer ⇒ `no_std`/MCU-decodable. **This is the biggest multichannel win** (G4). *(`lmo_lossless.rs:1-86,219-327,331-527`.)*
- **(b) Multivariate cross-channel RLS (`mv_rls`)** `[landed]` — **the measured dominant lever** (see Stage 2).

- **KLT / orthogonal spatial transform — ABSENT, and this is deliberate** `[rejected]`. There is no KLT. A full causal cross-channel LS predictor exists in `crosschan.rs` (Cholesky solve, Q16 coeffs) but is a **retained negative result** — explicitly **not given an LMO `transform_id`**: predict-then-wavelet cascade coded **+11% PRD worse** than the 9/7 baseline. The RICCT reversible-integer KLT probe decorrelates (floor −0.5/−4.2/−12%) but is **strictly dominated by `mv_rls`** (+7/+16/+28% worse full-container) — pre-rotating destroys the joint structure `mv_rls` exploits. Literature "joint coding 11.92% / SVD 50% / MVAR 70.64%" all refuted 0–3% under adversarial verification. *(`crosschan.rs:1-36,122-230`; `research-direction-2026-06.md:236-320`; `optimal-...:62-66`.)*

> **Do not** describe KLT/RICCT as a wired stage. Do **not** conflate cross-channel *decorrelation-as-transform* (killed) with cross-channel *prediction* (the win).

### Stage 1 — Temporal transform: integer 5/3 lifting DWT `[landed]`

The LeGall **5/3 integer-lifting wavelet** — **~80% optimal for EEG lossless**; the transform is **not** the dominant lever. Reused from the MCU kernel (`lamquant-lml-mcu`), not reimplemented. Subband layout `[a3,d3,d2,d1]`. Deeper transforms are a **don't-retry**: L4 +0–0.5%, wavelet-packet ma **+6.4% worse** (RAW beats 5/3 — each level adds per-subband LPC overhead + boundary leakage). Learned integer lifting also killed (2-tap LS lands on ≈[0.5,0.5]; +0.34%/+0.98% worse — 5/3 predict is already LS-optimal). *(`lossless-technique-mine-2026-06.md:15-19,156-158`; `optimal-...:73-74`.)*

For the **9/7 body** (`transform_id=1`, lossy) the transform swaps to the **float CDF 9/7 wavelet** (`wavelet97.rs`): canonical Daubechies–Sweldens constants `ALPHA/BETA/GAMMA/DELTA/K`, whole-sample symmetric reflection, forward+inverse algebraic inverses (lossy only via f64 round-off), `no_std`-clean `round_i64` (no libm). Layout matches the 5/3 floor so downstream code is transform-blind. Measured **~6% lower PRD than 5/3 at matched rate** (auto-pick guarantees never-worse). *(`wavelet97.rs:34-47,176-277`; `lmo.rs:26-27`.)*

### Stage 2 — Prediction (the H.BWC-beating engine)

Inside `transform_id=2` the residual coder is **keep-best over predictors**:

- **`CODER_LML=0`** `[landed]` — the MCU 5/3 + adaptive-order LPC + Golomb stream on the residual.
- **`CODER_RLS=1`** `[landed]` — per-channel RLS (MPEG-4-ALS-style): **order 8, λ=0.999, δ=1.0, periodic reset every 16384 samples**. Measured **−27.9%** on hard 21ch EEG ("ma"), −9.3% on ECG vs static 5/3+LPC (probe).
- **`CODER_MV_RLS=2`** `[landed]` — **the measured dominant lever.** Multivariate cross-channel RLS: **K=8 own-history taps + M cross-channel taps**, `M∈{4,8,32}`, a **7-config (λ / reset / M) keep-best grid**, change-point segmentation variants, rayon-parallel per-channel + per-config. **Faster-forgetting λ=0.997 / reset=4096** tracks non-stationary inter-channel coupling drift (vs old λ=0.999). Integer-exact IEEE `+−×÷` ⇒ deterministic host↔MCU, `no_std`-decodable despite being float. Measured: ma 6.530→6.184 bps, win-rate 2/8→5/8. *(`rls.rs:1-149,223-304`; `mv_rls.rs:46-63,139-316`; `optimal-...:41`.)*
- **`CODER_RLS_SEG=3`** `[landed]-decode` — still **decoded** but **dropped from the encode search** (a 40-recording tally found it never wins).

**Adaptive-order LPC schedule** `[landed]` — Burg N/8 rule: `lpc_max_order(subband_len) = subband_len/8` clamped `[0,16]` (`LPC_ORDER_HARD_CAP`); `scope_lpc_mode` clamps the mode's `max_order` to that per-subband ceiling; `analyze_with_mode` picks the actual order per subband (Fixed schedule / Adaptive AIC walk / Anytime deadline). No hardcoded per-subband schedule. *(`lml.rs:357,374-431`; `lpc.rs:226-306`.)*

**Change-point segmentation** `[landed]` (`segmentation.rs`) — causal dual-EWMA energy detector (**fast α=0.05 / slow α=0.002, RATIO_UP=4.0, MIN_DWELL=256, WARMUP=512**); emits adaptive-filter reset points from *losslessly-exact reconstructed samples* so decode reproduces them with a **1-bit on/off flag** — zero side-info. Structural answer to H.BWC's fixed `IntraPeriod`. *(`segmentation.rs:1-100`.)*

### Stage 3 — Context-adaptive bias cancellation (Sriraam) `[landed]`

`bias_cancel`/`bias_restore` over a **`BIAS_CTX=32`** ring buffer, subtracting `floor_div(running_sum, ctx)` per sample — exact O(T) integer inverse; order-0 path is pure `bias_cancel`. Shared kernel, applied inside `lpc::analyze/synthesize`. Measured **+6.0%** (LPC-8+GR 2.85:1 → 3.02:1), zero decoder overhead, bit-perfect. *(`lml.rs:357`; `lpc.rs:792-847,327-360`; `0007-...:601-610`.)*

### Stage 4 — Entropy coding (NOT the dominant lever — measured near-spent)

**The Optimum entropy coder is NOT rANS / CDF-LUT.** rANS (`mcu/src/rans.rs`) is used only by the LML floor and is **never invoked** from the Optimum path (verified by word-boundary grep). LMO uses two from-scratch **integer Subbotin carryless 32-bit range coders**:

- **`arith_int`** `[landed]` — static empirical-categorical: order-0 `encode_dense` + order-1 4-context `encode_dense_ctx`, leaky 16-bit fixed-point PMF, `MAX_ALPHABET` 8192/2048. Integer-only, `no_std`, deterministic host↔MCU.
- **`scale_cond`** `[landed]` — **online** scale-conditioned adaptive: Fenwick/BIT model, causal **log2-bucketed EMA** context, `EMA_K_DOWN=2 / EMA_K_UP=0`. This is the **realized entropy win** — online order-0 Laplace per causal-EMA-of-|x| bucket, zero frozen table. Measured held-out **−2.6% eegmmidb / −4.6% ma**, container **−2.7%**; already **at** the causal-scale information floor (−0.82%/+0.09%/−0.36% vs realizable).
*(`arith_int.rs:1-70,136-427`; `scale_cond.rs:1-57,265-533`; `optimal-...:44,96-102`.)*

**Residual coder keep-best** `[landed]` — `crate::entropy` picks the smallest of single-k Golomb, 256-block-adaptive Golomb, and `scale_cond`. The 9/7 PCRD per-subband coder (`encode_residual`) additionally keeps-best over **Golomb(0) / zRLE(1) / arith_int order-0 (tag4) / arith_int order-1 (tag5)**. `arith_cat` (tags 2/3, std/`constriction`) exists **only** under the `experimental_arithmetic` feature as an A/B oracle — **never emitted in a default build**. *(`entropy.rs:1-95`; `lmo_pcrd97.rs:143-228`; `Cargo.toml:41-47`.)*

**LSB bit-plane split** `[landed]` — `SPLIT_MAGIC=0xFE`, up to **2 low planes**: strips heavily-biased low bit-planes (per-bit entropy **< 0.92**), codes the upper signal through the normal pipeline and each channel's LSB plane via `crate::entropy` separately; keep-best vs no-split (s=0 ⇒ byte-identical). Reproduces H.BWC's own `zero_lsb` win on biased linked-ear montage: took tusz from **+23.67% → +5.62%** vs H.BWC, never-worse, bit-exact, `no_std`. *(`lmo_lossless.rs:56-66,175-215,331-394`; `0064-...:143-152`.)*

> **Overturned projections (don't re-quote as current plan):** context-adaptive arithmetic coding "worth 2–3×" — **rejected**, entropy near-spent. **tANS** projected 10–15% ratio win — measured **−1.2% EEG / +5.8% WORSE on ma** (LPC leaves a near-Laplacian residual for which Golomb-Rice is already the optimal prefix code); survives only as an MCU code-word-variance candidate. Order-1 context coder on the shipped `mv_rls` residual — **spent** (the whiter `mv_rls` residual has no order-1 structure `scale_cond` misses); reverted, byte-identical. A **frozen** learned conditional-entropy table — **killed** cross-corpus (ma→eegmmidb +5.9% worse; only the *online* `scale_cond` form generalizes). *(`optimal-...:67,71-77,124`.)*

### Stage 5 — RD quantization (9/7 lossy path only) `[landed]`

For `transform_id=1` / `Mode::TargetBps`, `lmo_pcrd97.rs` mirrors the MCU 5/3 PCRD exactly (same **46-step geometric quantizer grid**, same greedy Lagrangian ΔD/ΔR bit-allocation, same LPC+entropy chain) — only the transform (9/7) and quant boundary change. Two quantizers, `quant_mode` body byte, keep-best:
- **RDOQ-lite deadzone**: `idx = sign · floor(|c|/q + δ)`; encoder keeps-best over **δ∈{0.5, 0.42, 0.375}** (decode is δ-agnostic, `recon = idx·q`).
- **TCQ** (VVC dependent quantization): two interleaved grids, 4-state machine, Viterbi minimizing `Σ(G·d² + λ·rate)`; decoder replays the state machine — **no side info**.
*(`lmo_pcrd97.rs:47-75,126-141,282-472`; `tcq.rs:1-126`.)*

### Stage 6 — Optional learned entropy model (ADR 0076) — PROJECTED, banned from Optimum

`[planned]` / `[rejected]` for this tier. ADR 0076 is a **causal-Transformer categorical autoregressive** model over the FSQ `[32,79]` index grid (`R = Σ −log2 P(idxᵢ | idx_<i, θ)`, ~2–5M params). **Status: DESIGNED-NOT-BUILT** ("Proposed 2026-06-21", three phases all unimplemented). **PROJECTED** 2–3× CR at fixed R (Phase-1 gate ≥30% CR at same R); improves **CR only, not reconstruction R**. It operates on **lossy FSQ-quantized latents inside the neural LMQ codec** — **learned models are banned from the deterministic Optimum tier** (ADR 0052/0049). It is listed here only to mark the boundary; it is **not** part of the LMO pipeline. *(`0076-...:34-75,146-197`.)*

---

## 4. The H.BWC-beating mechanism

**H.BWC** `[landed]` is the ITU-T SG16 Q6/16 (VCEG) draft Recommendation "Coding of biomedical waveform data" (CfP Rennes, 26 Apr 2024). Rapporteur **Gary Sullivan** (HEVC/VVC), coordinator **Jonathan Pfaff** (Fraunhofer HHI), DICOM WG32 liaison **Jonathan Halford** (MUSC); committed contributors **Fraunhofer HHI + Dolby**. Modalities **EEG/ECG/EMG** (not PPG). Its pipeline is **VVC/VTM retargeted to 1-D biosignals**: integer block **DCT-II → TCQ4/RDOQ → CABAC (+Rice) → LMS-16 intra temporal predictor + cross-channel LMS (≤~31 refs) + block-matching + linear model**. It **does** apply the integer DCT in lossless mode (pre-shift, no post-shift). *(`0051-...:29-45`; `0054-...:216-234`; `0064-...:68-72,177`.)*

**The exploited weakness** `[landed]` — **non-stationarity.** H.BWC is tuned to stationary spectral structure: fixed `IntraPeriod`, **per-QP frozen CABAC context tables**, fixed-order LMS. It under-adapts to per-recording non-stationarity. **RLS ≫ LMS on drifting/peaky input** is the exploit. *(`hbwc-lossless-results-2026-06.md:70-74`; `0064-...:159-161`.)*

**Out-adaptation levers** `[landed]` (three online-adaptive vs H.BWC's frozen/block codec):
1. **`scale_cond`** scale-conditioned adaptive entropy (integer EMA, `no_std` bit-exact; commit e5f53bf).
2. **`mv_rls`** with the keep-best (λ, reset) grid — **highest-impact axis**; faster-forgetting λ=0.997/reset=4096 tracks non-stationary inter-channel coupling drift (commit 2506ca9).
3. **`segmentation`** signal-derived change-point detector resetting adaptive filters at regime boundaries with **zero side-info** — the structural answer to fixed `IntraPeriod` (commit a561d13).
4. **LSB bit-plane split** (commit 110317e) — the "oracle → clean-reproduce" playbook: instrument an HHI win, reproduce with patent-clean integer math.

**Why the gap can't be closed by entropy alone** `[landed]` — total **oracle** conditional-entropy headroom on the *actually-shipped* `mv_rls` residual is only **~1.2–1.9%** (zero model cost), **smaller** than the HHI referential gap (eegmmidb +2.3%, siena +4.16%). Even a perfect entropy coder cannot close it; the remainder is structurally H.BWC's **joint DCT+TCQ+CABAC RD allocation**, which a *keep-best-of-separate* architecture under-serves. The deterministic lever space is declared **at ceiling** on the stationary-referential lose-set (Front A / nonlinear predictor DEAD, E-A1 2026-06-27; RICCT killed). *(`optimal-...:96-134`; `research-front-...:18-36`; `0064-...:198-209`.)*

**Honest measured record** (win-map is signal physics — win non-stationary/clinically-active EEG, lose resting-referential clinical EEG):

| Corpus | vs H.BWC (bytes) | Verdict | Source |
|---|---|---|---|
| mental_arithmetic 21ch (task) | **−3.79%** (full sweep) / 5-8 win at −1.55…−6.84% (early) | **WIN** | ADR 0064:37-52 |
| chbmit 23ch (ictal, bipolar) | **−0.39%** / chb05 +0.05% | parity | ADR 0064 |
| eegmmidb 64ch (resting, referential) | **+2.30%** | behind | ADR 0064 |
| helsinki_neonatal | **+1.78%** | behind | ADR 0064 |
| tuar | **+2.90%** | behind | ADR 0064 |
| tuev | **+2.92%** | behind | ADR 0064 |
| siena | **+4.16%** | behind | ADR 0064 |
| tusz | **+6.03%** (from +23.4% pre-LSB-split) | behind | ADR 0064 |
| **Overall (42 recordings)** | **+3.06% larger, 3/42 wins** | **niche** | ADR 0064 |
| Lossy WP1–WP6 (same EEG) | H.BWC **15–100× better PRD** | LOST | lossy_head_to_head |
| ECG lossless | H.BWC **+20.9%–37%** ahead | CEDED | 0054/warplan-v2 |

> The "least clinically valuable regime" framing for the ceded resting-referential loss (ADR 0064:52,167) is **editorial positioning, not a measured fact** — flag it as such if repeated.

---

## 5. H.BWC conformance sub-tier

`[landed]` (oracle) / `[planned]` (encoder). `reference/_ports/codec-hbwc` is a **component-by-component derivative-work Rust port** of Fraunhofer HHI BWC `EncoderApp`/`DecoderApp` v5.0 — **8 crates** (transform/predict/quant/entropy/crosschan/decoder/encoder/testkit), **BSD-3-Clause, `publish=false`**. Its role is a **MEASUREMENT ORACLE only**; it is **never a `codec-lossless` dependency edge** — nothing patent-encumbered enters the shippable tiers.

- **Decoder:** reached **9/9 XR-B decode-conformance** (all real EEG + ECG + synthetic multi-frame goldens bit-exact) on **2026-06-24** (ref e2f961f). This is the conformance oracle behind the win claims. **Decode-only — no working encoder** (`EncoderApp` port emits only the WPS packet). Not shipped. *(`analysis/conformance-backlog-2026-06.md`.)*
- **Stale-doc flag** `[partial]`: ADR 0062 (2026-06-23) and the top-level `codec-hbwc` README still say the decoder "does not yet decode a single real stream" — that was true on 23 Jun, superseded by the 24 Jun 9/9 completion; the texts were not updated.
- **"Conformance" here** = 9/9 *internal* decode goldens vs the reference reconstruction, **NOT** a passed official ITU-T conformance suite.

**Patent posture** `[landed]` — the reference is BSD-3 with an **explicit patent non-grant** ("NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED"); VVC-derived tools (TCQ, CABAC, block-matching) likely read on essential patents. The port is **derivative** (each file cites `reference File.cpp:line`); copyright-clean ≠ patent-clean; `HLSUtility*` files are CfP-evaluation-licensed (reimplement, never copy). **Claims discipline (binding):** must NOT write "official", "the reference implementation", "endorsed by ITU-T/Fraunhofer", "fully-licensed", or "vendor". Honest claim = **"independent, BSD-3 Rust H.BWC-draft decoder."** A licensed SKU is a multi-year, presently "unbuyable" gated goal. *(`0062-...:43-99`; `0064-...:22-23`.)*

**Parked Shot-4** `[planned]` — a licensed cat-1 wrapper that includes `EncoderApp` output as a keep-best candidate would be *never-worse-than-H.BWC by construction*, but only as a **separately-licensed archival/cloud SKU** for customers already paying VVC/H.BWC royalties — fenced off from the royalty-free MCU/Desktop/Optimum tiers. *(`0064-...:132-138`.)*

---

## 6. Modes

Driven by the `Mode` enum; the container records the winner in `mode_tag` + `transform_id`. *(`lmo.rs:106-171`.)*

| Mode | Behavior | Machinery | Status |
|---|---|---|---|
| **`Mode::Lossless`** (byte-exact) | Auto-picks the **smaller** of the LML 5/3 floor (`transform_id=0`) and the Optimum-lossless body (`transform_id=2`, cross-channel). Both bit-exact ⇒ container never worse than the floor. | `lmo.rs:148-157` | `[landed]` |
| **`Mode::BoundedMae`** (near-lossless, ε=δ) | **Delegates to the LML integer 5/3 floor** (9/7 is float ⇒ not bit-exact for a hard MAE bound). | `lmo.rs:158-161` | `[landed]` |
| **`Mode::TargetBps`** (rate-target RD search) | Runs **both** ratio attacks at the target rate — integer 5/3 PCRD (`lml::compress_target_bps_pcrd`) **and** float 9/7 PCRD (`encode_target_bps_97`) — decodes each, keeps the **lower-PRD** reconstruction; `transform_id` records the winner ⇒ a 9/7 stream is never worse than the 5/3 floor. | `lmo.rs:126-147`; `lml.rs:1071-1190` | `[landed]` |

> **Guaranteed-δ near-lossless on `mv_rls`** `[probe]` — a per-sample error bound H.BWC lacks entirely, and the **only** path shown to beat H.BWC on the *static referential* regime (val δ1 3.72 vs 4.87 = 1.31×; siena δ1 4.50 vs 5.33 = 1.18×, δ32 1.16 vs 2.69 = **2.32×**). **PROBE — needs `mv_rls` `BoundedMae` wired to ship.** Do not present as shipped. *(`optimal-...:47`.)*

---

## 7. Wire format

**Production LMO uses its own self-describing `LMO1` container** (not BCS1/LMA). *(`lmo.rs:11-48`.)*

- **Magic** `b"LMO1"` (4) + **version** (1, `=2`) + **mode_tag** (1) + **transform_id** (1) = **7-byte header**.
- **`transform_id`** is the per-codec descriptor: **0** = inner LML integer 5/3 floor; **1** = LMO-native 9/7 float PCRD body (lossy); **2** = Optimum-lossless body (cross-channel + lml).
- Body-internal tags: residual `CODER_{LML=0,RLS=1,MV_RLS=2,RLS_SEG=3}`; `SPLIT_MAGIC=0xFE`; `quant_mode` (0=scalar, 1=TCQ).

**BCS1 / ABIR neutral wire** `[partial]` — a descriptor is **reserved** at BCS1 header **byte offset 8** (`codec_descriptor`), values numerically identical to `transform_id`: **0=`CODEC_LML_53`, 1=`CODEC_LMO_97`, 2=`CODEC_LMO_LOSSLESS`, 0x10=`CODEC_LMQ_FSQ`** (neural). `CodecDescriptor::{Lml53,Lmo97,LmoLossless,LmqFsq}` mirror them. **But BCS1 does NOT yet decode the Optimum descriptors:** the L9 writer only ever emits `CODEC_LML_53`, and `bcs1_gate_decodable` / the read dispatch accept only `CODEC_LML_53`. Values 1 and 2 **parse cleanly (header round-trips) but are refused fail-closed** as "not wired to a decoder in this build" (deferred follow-up). *(`abir/src/bcs1.rs:6-45,66-132`; `abir_container.rs:442,529-566,1409-1423`.)*

**Universal decode dispatch** `[landed]` — the host facade `lamquant_core` exposes **`decode_any`** (routes `LML1` → integer floor, `LMO1` → `optimum::decode`); the MCU `-core` decode returns typed `CodecError::OptimumNotInstalled` for an `LMO1` stream on a build without the optimum decoder linked; `peek_format` classifies by magic. *(`lamquant-lossless/src/lib.rs:70-82`; `lmo.rs:201-213`; `mcu/src/codec.rs:97-132`.)*

---

## 8. Parameters / config table

| Stage / knob | Symbol / function | Value | Status |
|---|---|---|---|
| Temporal DWT | LeGall 5/3 integer lifting, layout `[a3,d3,d2,d1]` | **3 levels** (a3/d3/d2/d1) | default `[landed]` |
| 9/7 DWT (lossy) | CDF 9/7, `ALPHA/BETA/GAMMA/DELTA/K` | canonical Daubechies–Sweldens | `[landed]` |
| LPC per-subband order | `lpc_max_order(len)=len/8` clamp `[0,16]`; `scope_lpc_mode`; `analyze_with_mode` | cap **`LPC_ORDER_HARD_CAP=16`**; Fixed/Adaptive-AIC/Anytime | `[landed]` |
| Bias cancellation | Sriraam `bias_cancel`, `floor_div(running_sum, ctx)` | **`BIAS_CTX=32`** | `[landed]` |
| Per-channel RLS | `rls.rs` order/λ/δ/reset | **order 8, λ=0.999, δ=1.0, reset 16384** | `[landed]` |
| Multivariate RLS | `mv_rls.rs` K / M / configs | **K=8**, **M∈{4,8,32}**, **7 (λ,reset,M) configs**; non-stationary config **λ=0.997/reset=4096** | `[landed]` |
| Change-point seg. | dual-EWMA `α_fast/α_slow`, `RATIO_UP`, `MIN_DWELL`, `WARMUP` | **0.05 / 0.002, 4.0, 256, 512**; 1-bit flag | `[landed]` |
| Cross-channel pred. | multi-ref joint-LS, Q16 gains, `round_q16` | **K≤3 refs**; byte-greedy(1st)+energy-greedy(rest); per-ch keep-best | `[landed]` |
| LSB split | `SPLIT_MAGIC=0xFE`, per-bit entropy threshold | **≤2 low planes**, entropy **<0.92** | `[landed]` |
| Static range coder | `arith_int` order-0 / order-1 4-ctx, PMF | 16-bit fixed-point, `MAX_ALPHABET` 8192/2048 | `[landed]` |
| Online range coder | `scale_cond` Fenwick, log2-bucket EMA | **`EMA_K_DOWN=2`, `EMA_K_UP=0`** | `[landed]` |
| Golomb keep-best | `entropy.rs` single-k / block-adaptive | **256-sample blocks** | `[landed]` |
| RDOQ-lite deadzone | `idx=sign·floor(|c|/q+δ)` | **δ∈{0.5,0.42,0.375}** | `[landed]` |
| TCQ | 2 grids, 4-state Viterbi `Σ(G·d²+λ·rate)` | `quant_mode=1` | `[landed]` |
| PCRD grid | geometric quantizer steps | **46 steps** | `[landed]` |
| Learned entropy (ADR 0076) | causal-Transformer over FSQ `[32,79]` | ~2–5M params | **`[planned]`, LMQ-only, banned here** |

---

## 9. Determinism & gates

- **Byte-identity round-trip** `[landed]` — the load-bearing correctness proof: every candidate is kept **only if smaller than the current body**, so adding a lever can never regress any recording; correctness proven by bit-exact round-trip across the corpus. *(`lmo_lossless.rs:487-526`; `0064-...:29-32`.)*
- **Determinism** `[landed]` — decode side is integer or IEEE `+−×÷` float only ⇒ deterministic host↔MCU. Always-compiled decode set: `arith_int` decode, `scale_cond` decode, `wavelet97` inverse, `rls`/`mv_rls` decode, `lmo_lossless` decode, `lmo_pcrd97 decode_97`, `segmentation`, `tcq` dequantize. *(`lib.rs:27-40`.)*
- **Conformance oracle** `[landed]` — the independent Rust H.BWC decoder must stay green at **9/9 decode goldens** (`goldens/decode/*.gv`, ref e2f961f) for any head-to-head claim.
- **Floor byte-equal gate (inherited)** — the shared MCU kernel remains under `byte_equal_backends` (Firmware ≡ Desktop identical bytes); Optimum re-exports that kernel, so any change to it must keep that gate green.
- **What is NOT yet gated** `[partial]` — there is **no committed bench harness** for Optimum (only `examples/` + a single `tests/dual_format_battery.rs`); throughput is unmeasured (§10). BCS1 Optimum-descriptor decode is fail-closed by design until wired.

---

## 10. Performance

**Compression (measured):**
- **Base lossless CR** `[landed]` MEASURED: **~2.26:1** on 16-bit clinical EEG = **94% of the Shannon limit** (6.63 b/sample = 2.41:1). Spread **2.31–2.40:1** (20-file GR-only → adaptive); 2.43:1 single-file verified; 2.37:1 paper baseline; **2.3× on full TUEG** (1.7 TB → 735 GB, 69,671 files). *(`0011-...:22-94`; `GENERATIONS.md:1321`.)*
  - **OVERCLAIM — do not use:** `docs/SPEC.md:35` and `api_reference.md:55` say **"3.76:1"** — unsourced, no run produces it. ADR 0011 (2.26:1) is authoritative.
- **Optimum delta-CR vs the 5/3 floor** `[landed]` MEASURED, bit-exact, v2 multi-ref: eegmmidb 64ch **−23.0%**, CHB-MIT 23ch **−10.7%**, mental_arithmetic 21ch **−9.6%**, MIT-BIH ECG 2ch **+0.04%** (7-byte header). These are **size reductions vs the floor, not absolute CRs**. *(`TRUTH_LEDGER.md §2L`.)*
  - **DERIVED** (not directly measured) absolute CRs: eegmmidb ~3.35:1, CHB-MIT ~2.72:1, ma ~1.93:1 — mark DERIVED.
  - **Do not conflate:** these §2L deltas are vs LamQuant's **own** floor, **not** margins vs H.BWC (those are §4).
- **H.BWC head-to-head** — see §4 table. Niche win; +3.06% larger overall; lossy and ECG lost.

**Throughput (measured — FLOOR only):**
- `[landed]` MEASURED (criterion, host): `compress_single_window` **~150–160 MiB/s** (fixed/anytime), firmware backend **~120–145 MiB/s**, `decompress_single` **~215–230 MiB/s**; container encode ~130–450 MiB/s, decode ~185–200 MiB/s. The `multi_channel_32ch`/`desktop_parallel` rows (~330–670 MiB/s) swing ±50% with host load — **flagged untrustworthy**; cite only per-stage/single-window rows. *(`BENCH_RESULTS.md:48-290`.)*
- `[partial]` **There is NO measured MiB/s for the Optimum (`transform_id=2`) path.** BENCH_RESULTS benches only the floor (`-p lamquant-lml`); Optimum is exercised only via `examples/lossless_scoreboard.rs` (size + bit-exact, no throughput). Any MiB/s cited is the **floor**, not Optimum.

**Silicon (measured — FLOOR only):**
- `[landed]` RP2350 (Pico 2 die, 2026-05-27), **FLOOR** lossless codec, 21ch × 2500 samples: **86.3 ms/window, 0.60 Msa/s, 116× realtime headroom**, `.lml` = 239 bytes (crc32 0x8ac7922c). Verilator RTL sim of the taped-out Hazard3: 0.627 Msa/s, 119×. `byte_equal_backends` passes across x86 scalar / x86 vectorised / RISC-V.
- **Do not claim "Optimum runs on silicon."** Optimum decode is integer/`no_std` **by construction**; only the **floor** is silicon-proven.

**Speed vs H.BWC — CORRECTED:**
- `[landed]` The retracted claim "**Optimum encodes ~10× faster than EncoderApp**" (`hbwc-lossless-results-2026-06.md:118`) is **WRONG**. ADR 0064:170-176 corrects it: the Optimum tier is **SLOWER** than EncoderApp (it is an encode-once archival keep-best). Only the deployed **MCU floor** (single 5/3+LPC pass) is **~110–200×** faster. Attribute the speed win to the **MCU floor**, not Optimum.

**Neural R — never attach.** Lossless/Optimum is bit-exact (**R=1.0, PRD=0**). The 0.81/0.85/0.93 R numbers are neural-track mirages (honest neural fullband ~0.42–0.73, continuous-latent, out of this story). *(`TRUTH_LEDGER §4.5`.)*

---

## 11. Build & integration

- **Crate:** `codec-lossless/lamquant-lml-optimum` (lib `lamquant_lml_optimum`, **v0.0.1**). *(`Cargo.toml:9`.)*
- **Feature gating** `[landed]`: `default = ["decode"]`; `encode = ["std","decode","dep:rayon"]`; `std = ["lamquant-lml-mcu/std"]`; `experimental_arithmetic` (the `arith_cat` A/B oracle only). Decode is `no_std`-capable and an optional Firmware module; encode is host-only and structurally excluded from the `riscv32imac-unknown-none-elf` graph. **`montage` is the only `#[cfg(feature="encode")]`-gated module** at the file level. *(`Cargo.toml:20-54`; `lib.rs:27-40`; `montage.rs:10`.)*
- **Reuse of the MCU kernel** `[landed]`: Optimum **re-exports the whole `Codec` seam** from `lamquant-lml-mcu` and calls it — the 5/3 lifting DWT, adaptive-order LPC (+ per-subband order schedule), Sriraam bias cancellation, Golomb-Rice, zRLE, and the 5/3 PCRD are **not reimplemented**. `lmo_pcrd97` calls `lpc::analyze_with_mode`, `golomb`, `zrle`, `lml::BIAS_CTX`, `lml::scope_lpc_mode`, `lml::lpc_max_order`. **rANS (`mcu/src/rans.rs`) is NOT called** from Optimum. *(`lib.rs:45-53`; `lmo_pcrd97.rs:9-16,40-43,246-280`.)*
- **Encode-side host-only symbols** `[landed]`: `joint_ls`/`quantize_gains`/reference-search (`lmo_lossless`), `RangeEncoder` + zigzag/bucket helpers (`arith_int`/`scale_cond` encode), TCQ Viterbi (`tcq::quantize_tcq`/`candidates`/`rate_bits`), `encode_target_bps_97`, `mv_rls` encode (rayon), `crosschan::fit_predictor`, and the entire `montage` module.
- **Product wiring** `[partial]` — **encode is NOT product-reachable.** No CLI subcommand (the `lml` bin has zero LMO refs), no container writer, and nothing in `lamquant-lossless` / `lamquant-lml-desktop` calls `lmo::encode` / `LmoCodec` / `encode_target_bps_97` / `lmo_lossless::encode` (verified by grep). **Only DECODE is product-wired** (via `decode_any`). To exercise the ratio attack today, call the crate API directly (`lamquant_lml_optimum::encode`, `lmo_lossless::encode`, `lmo_pcrd97::encode_target_bps_97`).
- **Tests:** single integration test `tests/dual_format_battery.rs`; ~55 `examples/` are research probes reading `/tmp` window dumps (`[n_ch u32][t u32][i32 samples]`), printing byte-count / entropy-floor comparisons (e.g. `entropy_backend_probe` = "would tANS beat block-Golomb on the RLS residual"; `scale_cond_entropy_probe`). Pub-but-not-shipped research helpers `mv_rls::residuals` (E-A1) and `mv_rls::encode_len_params` (TUH param search) compute measurements, not wire output.

---

## 12. Roadmap — landed vs the target spec

**Ships today (library-landed, roundtrip-tested, in the crate public API):**

| Capability | Module | Status |
|---|---|---|
| `LMO1` container + `transform_id` dispatch | `lmo.rs` | `[landed]` |
| WP0 floor re-encode (`transform_id=0`) — bit-exact baseline to beat | `lmo.rs`/`lib.rs` | `[landed]` |
| Optimum-lossless body (`transform_id=2`): multi-ref K≤3 joint-LS cross-channel prediction | `lmo_lossless.rs` | `[landed]` |
| `mv_rls` (dominant lever) + per-channel `rls` + change-point `segmentation` | `mv_rls/rls/segmentation.rs` | `[landed]` |
| `scale_cond` online entropy + `arith_int` static range coder + Golomb keep-best | `scale_cond/arith_int/entropy.rs` | `[landed]` |
| LSB bit-plane split | `lmo_lossless.rs` | `[landed]` |
| 9/7 float PCRD lossy path + RDOQ-lite + TCQ | `lmo_pcrd97/wavelet97/tcq.rs` | `[landed]` |
| Lossless / BoundedMae / TargetBps modes | `lmo.rs` | `[landed]` |
| Product **decode** via `decode_any` | `lamquant_core` | `[landed]` |
| H.BWC decode conformance oracle (9/9) | `reference/_ports/codec-hbwc` | `[landed]` (oracle, not shipped) |

**This spec proposes to build (not shipped) — the concrete next levers:**

| Item | What | Effort / gate | Status |
|---|---|---|---|
| **Wire Optimum encode to a product surface** | `lml --optimum` subcommand + container writer calling `lmo::encode` | required to ship the ratio attack at all | `[partial]` (unwired) |
| **BCS1/LMA Optimum decode** | Wire `CODEC_LMO_97` / `CODEC_LMO_LOSSLESS` through `bcs1_gate_decodable` + read dispatch (currently fail-closed) | deferred follow-up | `[partial]` |
| **Optimum bench harness** | Committed MiB/s harness for the `transform_id=2` path (none exists) | none today | `[planned]` |
| **Next lever #1 — cross-channel-magnitude entropy context** | Condition `scale_cond` on a reference channel's same-time residual magnitude bucket | MEASURED realizable **−0.23% eegmmidb / −0.62% ma / −0.19% siena**, ~0.3% never-worse; survives model cost; Low-Med effort | `[partial]` |
| **Next lever #2 — guaranteed-δ near-lossless on `mv_rls`** | Wire `mv_rls` `BoundedMae`; the only path shown to beat H.BWC on static referential (siena δ32 **2.32×**) | PROBE → ship | `[probe]` |
| **Next lever #3 — H.BWC keep-best oracle candidate** | Wrap `EncoderApp` output as a keep-best candidate (never-loses; HHI-proprietary/archival SKU) | separate license, fenced off | `[planned]` |
| MCU cost-only (not ratio) | rANS replacement for `arith_int`; sign-sign LMS/NLMS for ECG; cascade RLS→LMS refinement | code-word-variance only | `[planned]` |
| ADR 0076 learned entropy | **LMQ-only, banned from Optimum**; PROJECTED 2–3× CR at fixed R (CR-only) | DESIGNED-NOT-BUILT | `[planned]` |

**Honest bottom line** `[landed]`: the **deterministic Optimum codec is at its practical ceiling** on the stationary-referential lose-set — prediction saturated (`mv_rls`), entropy near-spent (~0.3% realizable via cross-channel context), DCT+CABAC refuted at realizable complexity. It **beats H.BWC only on the non-stationary/clinically-active EEG niche** (task/ictal/bipolar), is ~3% larger across the full 42-recording held-out clinical set, and loses all lossy working points and ECG. The larger remaining win lives in the separate, capacity-bound, unvalidated neural **LMQ** codec (where ADR 0076 applies), not in this deterministic tier. *(`optimal-...:128-134`; `research-front-...:56-77`.)*