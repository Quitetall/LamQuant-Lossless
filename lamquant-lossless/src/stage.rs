//! ADR 0074 · Track M — the codec as a typed morphism category (host-only, `archive`).
//!
//! An IR = **types + typed ops**. The *objects* are the data representations
//! (`Raw → Transformed → Quantized → Residuals → Coded → Packet`); a *pass is a
//! morphism* typed by its `(input, output)` endpoints; a *codec is any
//! endpoint-matching chain* `Raw → Packet`. Every morphism **DISPATCHES** to the
//! kernel — it never reimplements the DSP (ADR 0074's "dispatch, not codegen") —
//! so a composed chain is byte-identical to the shipped kernel by construction.
//!
//! The canonical stage TYPES, in order, are `Transform → Quantize → Predict →
//! Entropy → Assemble`. Lossless AND lossy are both first-class citizens of this
//! ONE pipeline, differing only in the **Quantize** morphism: lossless uses
//! `Identity` (Reversible, a no-op that lowers away → byte-identical to the fused
//! kernel), rate-lossy uses a `deadzone` quantizer (Lossy). Coupled/fused coders
//! (closed-loop DPCM, neural) enter later as *wider* morphisms (e.g. `Transformed
//! → Residuals`) — no special-casing, because endpoint-typing already admits them.
//!
//! M3 builds the objects + the LOSSLESS morphism chain and proves it byte-identical
//! to `compress_with_mode_views`. The reverse morphisms, the `LmlPipeline`/`Pass`
//! wiring, oracle arm H, and the lossy Quantize morphism follow.

use core::marker::PhantomData;

use abir::{Abir, Modality, ModalitySource, Mode};

use crate::error::LmlResult;
use crate::golomb;
use crate::lml::{
    assemble_lml_packet, compress_bounded_mae, compress_target_bps, compress_with_mode_views,
    compute_n_levels, decompress, forward_subbands, lpc_max_order, scope_lpc_mode, BIAS_CTX,
};
use crate::lpc::{self, LpcMode};
use crate::pass::{LmlPipeline, Lossy, Pass, Reversible};
use crate::pipeline::Stage;
use crate::quant;

// ─── Objects (the IR's types) ────────────────────────────────────────────────

/// The **Raw** stage — one window of the typed recording; the DAG's input. The
/// whole `Abir` is treated as one window (`n_samples` × `n_channels`), matching
/// the per-window kernel entry.
///
/// Stage + modality are both in the type, so mixing them is a compile error. A
/// `Coded` is not a `Raw` (distinct stages) — a single `compile_fail` block stops
/// at the first error, so the two properties get one block each:
///
/// ```compile_fail
/// use lamquant_core::stage::{Coded, Raw};
/// fn takes_eeg_raw(_: Raw<abir::Eeg>) {}
/// fn bad(c: Coded<abir::Eeg>) { takes_eeg_raw(c); }
/// ```
///
/// And `Raw<Ecg>` is not `Raw<Eeg>` (distinct modalities — Pillar 1):
///
/// ```compile_fail
/// use lamquant_core::stage::Raw;
/// fn takes_eeg_raw(_: Raw<abir::Eeg>) {}
/// fn bad(r: Raw<abir::Ecg>) { takes_eeg_raw(r); }
/// ```
#[derive(Debug, Clone)]
pub struct Raw<M: Modality> {
    abir: Abir<M>,
}

impl<M: Modality> Raw<M> {
    /// Wrap a typed window.
    pub fn new(abir: Abir<M>) -> Self {
        Self { abir }
    }
    /// Borrow the underlying typed currency.
    pub fn abir(&self) -> &Abir<M> {
        &self.abir
    }
    /// Consume back into the typed `Abir`.
    pub fn into_abir(self) -> Abir<M> {
        self.abir
    }
}

/// The **Transformed** stage — post-`Transform`: per-channel ordered subbands
/// `[ch][subband][sample]` + the chosen `n_levels` (carried so the inverse lift
/// can invert it).
#[derive(Debug, Clone)]
pub struct Transformed<M: Modality> {
    per_channel: Vec<Vec<Vec<i64>>>,
    n_levels: u8,
    _m: PhantomData<M>,
}

impl<M: Modality> Transformed<M> {
    /// The chosen wavelet level count.
    pub fn n_levels(&self) -> u8 {
        self.n_levels
    }
    /// The per-channel subbands.
    pub fn subbands(&self) -> &[Vec<Vec<i64>>] {
        &self.per_channel
    }
}

/// The **Quantized** stage — post-`Quantize`: same shape as `Transformed`. For
/// lossless the `Quantize` morphism is `Identity`, so this is the transformed
/// coefficients unchanged; **M4 diverges** — rate-lossy holds deadzone-quantized
/// indices here (a `Lossy` morphism), which is why this is a distinct type from
/// `Transformed` even though the fields currently match.
#[derive(Debug, Clone)]
pub struct Quantized<M: Modality> {
    per_channel: Vec<Vec<Vec<i64>>>,
    n_levels: u8,
    _m: PhantomData<M>,
}

/// One subband's LPC result — the product output that makes `Predict` reversible:
/// the residual is the main lane, `(order, coeffs)` the carry lane the inverse
/// re-synthesizes from.
#[derive(Debug, Clone)]
pub struct SubbandResidual {
    pub order: usize,
    pub coeffs: Vec<i32>,
    pub residual: Vec<i64>,
}

/// The **Residuals** stage — post-`Predict`: per-channel, per-subband
/// `{order, coeffs, residual}`.
#[derive(Debug, Clone)]
pub struct Residuals<M: Modality> {
    per_channel: Vec<Vec<SubbandResidual>>,
    n_levels: u8,
    _m: PhantomData<M>,
}

impl<M: Modality> Residuals<M> {
    /// The per-channel, per-subband LPC results (inspection).
    pub fn per_channel(&self) -> &[Vec<SubbandResidual>] {
        &self.per_channel
    }
    /// The wavelet level count carried from `Transformed`.
    pub fn n_levels(&self) -> u8 {
        self.n_levels
    }
}

/// One channel's entropy-coded bytes: `meta = [order:u8][coeffs_i32_LE…]` per
/// subband, `payload` = the concatenated Golomb streams.
#[derive(Debug, Clone)]
pub struct ChannelCoded {
    pub meta: Vec<u8>,
    pub payload: Vec<u8>,
}

/// The **Coded** stage — post-`Entropy`: per-channel `{meta, payload}` + the
/// header dimensions `Assemble` needs.
#[derive(Debug, Clone)]
pub struct Coded<M: Modality> {
    per_channel: Vec<ChannelCoded>,
    n_ch: usize,
    t: usize,
    n_levels: u8,
    _m: PhantomData<M>,
}

impl<M: Modality> Coded<M> {
    /// The per-channel entropy-coded `{meta, payload}` (inspection).
    pub fn per_channel(&self) -> &[ChannelCoded] {
        &self.per_channel
    }
}

/// The **Packet** stage — post-`Assemble`: the framed LML1 packet; the DAG's
/// output. Carries the modality type so decode can restore a typed `Raw<M>`.
/// (`M` is a compile-time-only tag — `size_of::<Packet<M>>() == size_of::<Vec<u8>>()`.)
#[derive(Debug, Clone)]
pub struct Packet<M: Modality> {
    bytes: Vec<u8>,
    _m: PhantomData<M>,
}

impl<M: Modality> Packet<M> {
    /// The framed packet bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Consume into the raw packet bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

// ─── The lowering (dispatch to the fused kernel) ─────────────────────────────

/// Lower `Raw → Packet` by **dispatching** to the fused kernel over the zero-copy
/// window views — the DAG's `lower()`: a dispatch to `compress_with_mode_views`,
/// never a reimplementation, so it is byte-identical to the shipped kernel.
///
/// ```
/// use lamquant_core::stage::{lower_encode, lower_decode, Raw};
/// use lamquant_core::lpc::LpcMode;
/// use abir::{Abir, Eeg, ModalitySource};
///
/// let sig = vec![vec![10i64, -20, 30, -40, 50, -60, 70, -80]; 4];
/// let raw: Raw<Eeg> =
///     Raw::new(Abir::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual));
/// let packet = lower_encode(&raw, 0, LpcMode::Fixed).unwrap();
/// let back = lower_decode(&packet, 250.0).unwrap();
/// assert_eq!(back.abir().n_channels(), 4);
/// ```
pub fn lower_encode<M: Modality>(
    raw: &Raw<M>,
    noise_bits: u8,
    mode: LpcMode,
) -> LmlResult<Packet<M>> {
    let n = raw.abir.n_samples();
    let views = raw.abir.window_views(0, n);
    let refs: Vec<&[i64]> = views.iter().map(|c| c.as_ref()).collect();
    let bytes = compress_with_mode_views(&refs, noise_bits, mode)?;
    Ok(Packet { bytes, _m: PhantomData })
}

/// Lower `Packet → Raw` (decode), restoring the modality type `M`. `sample_rate`
/// is caller-supplied (the LML1 packet doesn't carry it — it lives in the BCS1
/// header). `into_modality::<M>` runs before return, so no `Untyped` window
/// escapes; the modality is asserted by the caller's type argument.
pub fn lower_decode<M: Modality>(packet: &Packet<M>, sample_rate: f64) -> LmlResult<Raw<M>> {
    let channels = decompress(packet.bytes())?;
    let abir =
        Abir::from_channels_i64(channels, sample_rate).into_modality::<M>(ModalitySource::Manual);
    Ok(Raw::new(abir))
}

// ─── The lossless morphism chain (each delegates to the kernel) ───────────────

/// **Transform** morphism `Raw → Transformed` (Reversible). Delegates to the
/// kernel's `forward_subbands` per channel; `n_levels` chosen by `compute_n_levels`
/// exactly as `validate_and_levels` does.
pub fn transform<M: Modality>(raw: &Raw<M>) -> Transformed<M> {
    let n = raw.abir.n_samples();
    let n_levels = compute_n_levels(n);
    let views = raw.abir.window_views(0, n);
    let per_channel = views
        .iter()
        .map(|c| forward_subbands(c.as_ref(), n_levels))
        .collect();
    Transformed { per_channel, n_levels, _m: PhantomData }
}

/// **Quantize = Identity** morphism `Transformed → Quantized` (Reversible). The
/// lossless quantizer: a no-op that lowers away. (Rate-lossy replaces this with a
/// deadzone quantizer — a `Lossy` morphism — without touching any other stage.)
pub fn quantize_identity<M: Modality>(t: Transformed<M>) -> Quantized<M> {
    Quantized { per_channel: t.per_channel, n_levels: t.n_levels, _m: PhantomData }
}

/// **Predict** morphism `Quantized → Residuals` (Reversible *because* it carries
/// `coeffs`+`order`). Delegates to `lpc::analyze_with_mode` per subband with the
/// same per-subband ceiling (`lpc_max_order`) + `scope_lpc_mode` the kernel uses.
pub fn predict<M: Modality>(q: Quantized<M>, mode: LpcMode) -> Residuals<M> {
    let per_channel = q
        .per_channel
        .iter()
        .map(|subbands| {
            subbands
                .iter()
                .enumerate()
                .map(|(sb_idx, sub)| {
                    // `sb_idx` is the subband's position in `forward_subbands`'s
                    // output order — it MUST match the kernel's own loop
                    // (`encode_one_channel` lml.rs:1524); the arm-H byte-identity
                    // test is what guards that coupling.
                    let scoped = scope_lpc_mode(mode, lpc_max_order(sub.len()));
                    let (coeffs, residual, order) =
                        lpc::analyze_with_mode(sub, sb_idx, scoped, BIAS_CTX, /* time_remaining = */ None);
                    SubbandResidual { order, coeffs, residual }
                })
                .collect()
        })
        .collect();
    Residuals { per_channel, n_levels: q.n_levels, _m: PhantomData }
}

/// **Entropy** morphism `Residuals → Coded` (Reversible). Serializes
/// `[order:u8][coeffs_i32_LE…]` into `meta` and delegates the residual to
/// `golomb::encode_dense` into `payload` — exactly `encode_one_channel`'s body.
/// `n_ch`/`t` (needed by the header) are DERIVED: `n_ch` is the channel count and
/// `t` is a channel's total residual length (lift + LPC are length-preserving, so
/// the per-channel residuals sum to the original window length) — so `entropy`
/// depends only on `Residuals` and can be a single-input `Stage`.
pub fn entropy<M: Modality>(r: Residuals<M>) -> LmlResult<Coded<M>> {
    let n_ch = r.per_channel.len();
    let t = r
        .per_channel
        .first()
        .map_or(0, |subs| subs.iter().map(|sb| sb.residual.len()).sum());
    debug_assert!(
        r.per_channel
            .iter()
            .all(|subs| subs.iter().map(|sb| sb.residual.len()).sum::<usize>() == t),
        "entropy: channels disagree on the derived window length t={t}"
    );
    let mut per_channel = Vec::with_capacity(r.per_channel.len());
    for subbands in &r.per_channel {
        let mut meta = Vec::new();
        let mut payload = Vec::new();
        for sb in subbands {
            // The wire stores the order in one byte (same cast as the kernel,
            // lml.rs:1557); orders are bounded by `lpc_max_order` in practice.
            debug_assert!(sb.order <= u8::MAX as usize, "LPC order {} exceeds the u8 wire field", sb.order);
            meta.push(sb.order as u8);
            for &c in &sb.coeffs {
                meta.extend_from_slice(&c.to_le_bytes());
            }
            payload.extend_from_slice(&golomb::encode_dense(&sb.residual)?);
        }
        per_channel.push(ChannelCoded { meta, payload });
    }
    Ok(Coded { per_channel, n_ch, t, n_levels: r.n_levels, _m: PhantomData })
}

/// **Assemble** morphism `Coded → Packet` (Reversible) — the fan-in reduce node.
/// Concatenates all channels' meta then all channels' payload (the
/// `finalize_channels` reorder) and delegates the framing (header + CRC) to
/// `assemble_lml_packet`. `noise_bits` is a parameter (0 for pure lossless; the
/// near-lossless noise-shaping variant sets it) so the morphism's signature is
/// stable when M4's lossy chains reuse it.
pub fn assemble<M: Modality>(c: Coded<M>, noise_bits: u8) -> Packet<M> {
    let mut lpc_meta = Vec::new();
    let mut payload = Vec::new();
    for ch in &c.per_channel {
        lpc_meta.extend_from_slice(&ch.meta);
    }
    for ch in &c.per_channel {
        payload.extend_from_slice(&ch.payload);
    }
    // `any_bit_pack_wins = false`: the experimental per-subband entropy
    // *selection* (Golomb vs arith vs bit-pack) is not modeled here yet — the
    // default build is flat Golomb, the byte-locked reference.
    let bytes =
        assemble_lml_packet(c.n_ch, c.t, c.n_levels, noise_bits, false, &lpc_meta, &payload);
    Packet { bytes, _m: PhantomData }
}

/// The composed **lossless codec** as an explicit morphism chain
/// `Raw → Transform → Quantize(Identity) → Predict → Entropy → Assemble → Packet`.
/// Materializes each stage (the authoring path); the fused kernel is the schedule.
/// Proven byte-identical to `lower_encode` / `compress_with_mode_views` — that
/// equality (arm H) is what licenses the scheduler to run the fused form.
pub fn encode_lossless<M: Modality>(raw: &Raw<M>, mode: LpcMode) -> LmlResult<Packet<M>> {
    let transformed = transform(raw);
    let quantized = quantize_identity(transformed);
    let residuals = predict(quantized, mode);
    let coded = entropy(residuals)?;
    Ok(assemble(coded, /* noise_bits = */ 0))
}

// ─── Lossy: the Mode-aware lowering + the deadzone Quantize morphism ──────────

/// Lower `Raw → Packet` for ANY codec [`Mode`] by **dispatching** to the matching
/// kernel — the lossy counterpart of [`lower_encode`], making lossless and lossy
/// equally first-class: `Lossless` → `compress_with_mode_views`, `BoundedMae(δ)`
/// → `compress_bounded_mae`, `TargetBps` → `compress_target_bps`. Byte-identical
/// to the kernel by construction (it IS the kernel); the lossy arms materialize
/// the window (the cold path already does). The rate-controller search
/// (`TargetBps`) is a *schedule* concern that lives inside the kernel — the DAG
/// dispatches the whole optimized encode, it does not model the search.
///
/// This is the *dispatch* lowering; [`encode_lossless`] is the *decomposed*
/// authoring chain. They coexist by design (algorithm vs schedule), and their
/// byte-equality for `Mode::Lossless` is exactly oracle arm H.
pub fn lower_encode_mode<M: Modality>(
    raw: &Raw<M>,
    mode: Mode,
    lpc_mode: LpcMode,
) -> LmlResult<Packet<M>> {
    let n = raw.abir.n_samples();
    let views = raw.abir.window_views(0, n);
    let bytes = match mode {
        Mode::Lossless => {
            let refs: Vec<&[i64]> = views.iter().map(|c| c.as_ref()).collect();
            compress_with_mode_views(&refs, 0, lpc_mode)?
        }
        // The lossy kernels take `&[Vec<i64>]`, so own here (cold lossy path).
        Mode::BoundedMae(delta) => {
            let signal: Vec<Vec<i64>> = views.iter().map(|c| c.as_ref().to_vec()).collect();
            compress_bounded_mae(&signal, delta, lpc_mode)?
        }
        Mode::TargetBps(bps) => {
            let signal: Vec<Vec<i64>> = views.iter().map(|c| c.as_ref().to_vec()).collect();
            compress_target_bps(&signal, bps, lpc_mode)?
        }
    };
    Ok(Packet { bytes, _m: PhantomData })
}

/// **Quantize = deadzone** morphism `Transformed → Quantized` — the LOSSY variant
/// of the Quantize slot (delegates `quant::quantize` per subband). This is the
/// ONLY morphism in the pipeline that discards information; swapping it for
/// [`quantize_identity`] is the entire lossless↔lossy difference. `steps[i]` is
/// the per-subband deadzone step (finer where synthesis gain amplifies error —
/// see `quant::steps_for_scale`); the last step repeats if fewer than the subband
/// count. A `Lossy` morphism: it must never enter the lossless builder (enforced
/// once the morphisms become `Pass` impls, M5).
pub fn quantize_deadzone<M: Modality>(t: Transformed<M>, steps: &[i64]) -> Quantized<M> {
    debug_assert!(!steps.is_empty(), "quantize_deadzone: steps must be non-empty");
    let per_channel = t
        .per_channel
        .iter()
        .map(|subbands| {
            subbands
                .iter()
                .enumerate()
                .map(|(i, sub)| quant::quantize(sub, steps[i.min(steps.len() - 1)]))
                .collect()
        })
        .collect();
    Quantized { per_channel, n_levels: t.n_levels, _m: PhantomData }
}

/// The (lossy) inverse of [`quantize_deadzone`] — `idx * q` per subband. NOT a
/// true inverse: `dequantize(quantize(x))` is within `⌈q/2⌉` of `x` (round-to-
/// nearest), which is the near-lossless bound the property test checks.
pub fn dequantize_deadzone<M: Modality>(q: Quantized<M>, steps: &[i64]) -> Transformed<M> {
    debug_assert!(!steps.is_empty(), "dequantize_deadzone: steps must be non-empty");
    let per_channel = q
        .per_channel
        .iter()
        .map(|subbands| {
            subbands
                .iter()
                .enumerate()
                .map(|(i, idx)| quant::dequantize(idx, steps[i.min(steps.len() - 1)]))
                .collect()
        })
        .collect();
    Transformed { per_channel, n_levels: q.n_levels, _m: PhantomData }
}

// ─── The reversible DSP inverses (Transform⁻¹, Predict⁻¹) ─────────────────────
//
// These are the `unprocess` of the two morphisms that DO the reversible integer
// math; each delegates to the kernel's own inverse primitive so it can never drift
// from the forward transform. (The serialization inverses — Entropy⁻¹ Golomb-decode
// and Assemble⁻¹ packet-parse — are the kernel's tested `decompress`; a typed
// reproduction of those is a follow-up. `lower_decode` already gives the full
// Packet→Raw decode via the kernel.)

/// **Transform⁻¹** `Transformed → per-channel samples` — inverse lifting
/// (delegates `quant::inverse_for_levels`). `inverse_transform(transform(raw))`
/// recovers the samples EXACTLY (5/3 integer lifting is bit-exact reversible).
pub fn inverse_transform<M: Modality>(t: &Transformed<M>) -> Vec<Vec<i64>> {
    t.per_channel
        .iter()
        .map(|subs| quant::inverse_for_levels(t.n_levels, subs))
        .collect()
}

/// **Predict⁻¹** `Residuals → Quantized` — LPC synthesis per subband (delegates
/// `lpc::synthesize` with the carried coeffs+order). `unpredict(predict(q))`
/// recovers the quantized subbands EXACTLY — the coeffs+order carried in
/// `Residuals` are precisely what makes `predict` reversible.
pub fn unpredict<M: Modality>(r: Residuals<M>) -> Quantized<M> {
    let per_channel = r
        .per_channel
        .iter()
        .map(|subbands| {
            subbands
                .iter()
                .map(|sb| lpc::synthesize(&sb.residual, &sb.coeffs, sb.order, BIAS_CTX))
                .collect()
        })
        .collect();
    Quantized { per_channel, n_levels: r.n_levels, _m: PhantomData }
}

// ─── The morphisms as `Pass` impls — the compile-time lossless↔lossy firewall ──
//
// Wrapping each morphism as a `Stage` + `Pass` lets the lossless codec compose
// through `LmlPipeline`, whose `start`/`then` are bounded `Rev = Reversible`. A
// `Lossy` pass (e.g. `DeadzoneQuantizePass`) therefore CANNOT be composed into the
// lossless pipeline — a COMPILE error, not a convention. This is "lossless can
// never hide a lossy step" expressed in the type system.

/// `Transform` (`Raw → Transformed`), Reversible.
#[derive(Debug)]
pub struct LiftPass<M>(PhantomData<M>);
impl<M> Default for LiftPass<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<M: Modality> Stage for LiftPass<M> {
    type Input = Raw<M>;
    type Output = Transformed<M>;
    fn process(&mut self, raw: Raw<M>) -> LmlResult<Transformed<M>> {
        Ok(transform(&raw))
    }
}
impl<M: Modality> Pass for LiftPass<M> {
    type Rev = Reversible;
    const NAME: &'static str = "lift";
}

/// `Quantize = Identity` (`Transformed → Quantized`), Reversible — the lossless quantizer.
#[derive(Debug)]
pub struct IdentityQuantizePass<M>(PhantomData<M>);
impl<M> Default for IdentityQuantizePass<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<M: Modality> Stage for IdentityQuantizePass<M> {
    type Input = Transformed<M>;
    type Output = Quantized<M>;
    fn process(&mut self, t: Transformed<M>) -> LmlResult<Quantized<M>> {
        Ok(quantize_identity(t))
    }
}
impl<M: Modality> Pass for IdentityQuantizePass<M> {
    type Rev = Reversible;
    const NAME: &'static str = "quantize_identity";
}

/// `Quantize = deadzone` (`Transformed → Quantized`), **Lossy** — refused by `LmlPipeline`.
#[derive(Debug)]
pub struct DeadzoneQuantizePass<M> {
    steps: Vec<i64>,
    _m: PhantomData<M>,
}
impl<M> DeadzoneQuantizePass<M> {
    pub fn new(steps: Vec<i64>) -> Self {
        Self { steps, _m: PhantomData }
    }
}
impl<M: Modality> Stage for DeadzoneQuantizePass<M> {
    type Input = Transformed<M>;
    type Output = Quantized<M>;
    fn process(&mut self, t: Transformed<M>) -> LmlResult<Quantized<M>> {
        Ok(quantize_deadzone(t, &self.steps))
    }
}
impl<M: Modality> Pass for DeadzoneQuantizePass<M> {
    type Rev = Lossy;
    const NAME: &'static str = "quantize_deadzone";
}

/// `Predict` (`Quantized → Residuals`), Reversible (carries coeffs+order).
#[derive(Debug)]
pub struct PredictPass<M> {
    mode: LpcMode,
    _m: PhantomData<M>,
}
impl<M> PredictPass<M> {
    pub fn new(mode: LpcMode) -> Self {
        Self { mode, _m: PhantomData }
    }
}
impl<M: Modality> Stage for PredictPass<M> {
    type Input = Quantized<M>;
    type Output = Residuals<M>;
    fn process(&mut self, q: Quantized<M>) -> LmlResult<Residuals<M>> {
        Ok(predict(q, self.mode))
    }
}
impl<M: Modality> Pass for PredictPass<M> {
    type Rev = Reversible;
    const NAME: &'static str = "predict";
}

/// `Entropy` (`Residuals → Coded`), Reversible.
#[derive(Debug)]
pub struct EntropyPass<M>(PhantomData<M>);
impl<M> Default for EntropyPass<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<M: Modality> Stage for EntropyPass<M> {
    type Input = Residuals<M>;
    type Output = Coded<M>;
    fn process(&mut self, r: Residuals<M>) -> LmlResult<Coded<M>> {
        entropy(r)
    }
}
impl<M: Modality> Pass for EntropyPass<M> {
    type Rev = Reversible;
    const NAME: &'static str = "entropy";
}

/// `Assemble` (`Coded → Packet`), Reversible — the fan-in.
#[derive(Debug)]
pub struct AssemblePass<M> {
    noise_bits: u8,
    _m: PhantomData<M>,
}
impl<M> AssemblePass<M> {
    pub fn new(noise_bits: u8) -> Self {
        Self { noise_bits, _m: PhantomData }
    }
}
impl<M: Modality> Stage for AssemblePass<M> {
    type Input = Coded<M>;
    type Output = Packet<M>;
    fn process(&mut self, c: Coded<M>) -> LmlResult<Packet<M>> {
        Ok(assemble(c, self.noise_bits))
    }
}
impl<M: Modality> Pass for AssemblePass<M> {
    type Rev = Reversible;
    const NAME: &'static str = "assemble";
}

/// The lossless codec composed through the **reversible-only** [`LmlPipeline`] —
/// the SAME five morphisms as [`encode_lossless`], but now a `Lossy` pass anywhere
/// in the chain is a compile error. Byte-identical to `encode_lossless`.
///
/// The firewall in action — swapping the Identity quantizer for the deadzone
/// (Lossy) one does not compile:
///
/// ```compile_fail
/// use lamquant_core::stage::{DeadzoneQuantizePass, LiftPass};
/// use lamquant_core::pass::LmlPipeline;
/// let _ = LmlPipeline::start(LiftPass::<abir::Eeg>::default())
///     .then(DeadzoneQuantizePass::<abir::Eeg>::new(vec![9])); // Rev = Lossy → rejected
/// ```
pub fn encode_lossless_pipeline<M: Modality>(raw: Raw<M>, mode: LpcMode) -> LmlResult<Packet<M>> {
    let mut pipe = LmlPipeline::start(LiftPass::<M>::default())
        .then(IdentityQuantizePass::<M>::default())
        .then(PredictPass::<M>::new(mode))
        .then(EntropyPass::<M>::default())
        .then(AssemblePass::<M>::new(0));
    pipe.process(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use abir::Eeg;

    fn synth(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
        (0..n_ch)
            .map(|c| (0..t).map(|i| (((i * 3 + c * 7) % 512) as i64 - 256) * 40).collect())
            .collect()
    }

    fn eeg_raw(sig: Vec<Vec<i64>>) -> Raw<Eeg> {
        Raw::new(Abir::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual))
    }

    #[test]
    fn lossless_morphism_chain_is_byte_identical_to_the_fused_kernel() {
        // The heart of M3 / arm H at the smallest granularity: the explicit typed
        // chain equals the fused kernel byte-for-byte across n_levels 0..3 + modes.
        let shapes = [(1usize, 4usize), (1, 8), (1, 20), (4, 2500), (8, 313), (32, 2500)];
        let modes =
            [LpcMode::Fixed, LpcMode::Adaptive { max_order: 16 }, LpcMode::Anytime { max_order: 16, deadline: None }];
        for &(n_ch, t) in &shapes {
            let sig = synth(n_ch, t);
            let raw = eeg_raw(sig.clone());
            let views: Vec<&[i64]> = sig.iter().map(|c| c.as_slice()).collect();
            for mode in modes {
                let dag = encode_lossless(&raw, mode).expect("morphism chain");
                let fused = compress_with_mode_views(&views, 0, mode).expect("fused");
                assert_eq!(
                    dag.bytes(),
                    fused.as_slice(),
                    "M3 morphism-chain DRIFT vs fused ({n_ch}ch × {t}, {mode:?})"
                );
                assert_eq!(decompress(dag.bytes()).expect("decode"), sig, "roundtrip ({mode:?})");
            }
        }
    }

    #[test]
    fn intermediate_objects_are_inspectable_and_well_formed() {
        let raw = eeg_raw(synth(4, 2500));
        let transformed = transform(&raw);
        assert_eq!(transformed.n_levels(), 3, "t=2500 → 3 levels");
        assert_eq!(transformed.subbands().len(), 4, "4 channels");
        assert_eq!(transformed.subbands()[0].len(), 4, "3 levels → 4 subbands (a3,d3,d2,d1)");
        let residuals = predict(quantize_identity(transformed), LpcMode::Fixed);
        assert_eq!(residuals.per_channel().len(), 4);
        assert_eq!(residuals.per_channel()[0].len(), 4, "one residual set per subband");
        // Every subband carries the coeffs that make Predict reversible.
        let sb0 = &residuals.per_channel()[0][0];
        assert_eq!(sb0.coeffs.len(), sb0.order);
    }

    #[test]
    fn typed_endpoint_lower_decode_round_trips() {
        // Covers lower_encode → lower_decode (the into_modality::<M> restore path),
        // both the fixed and the reproducible-anytime mode.
        let sig = synth(4, 2500);
        let raw = eeg_raw(sig.clone());
        for mode in [LpcMode::Fixed, LpcMode::Anytime { max_order: 16, deadline: None }] {
            let packet = lower_encode(&raw, 0, mode).expect("encode");
            let back = lower_decode(&packet, 250.0).expect("decode");
            assert_eq!(back.abir().n_channels(), 4);
            assert_eq!(back.abir().n_samples(), 2500);
            let n = back.abir().n_samples();
            let got: Vec<Vec<i64>> =
                back.abir().window_views(0, n).iter().map(|c| c.as_ref().to_vec()).collect();
            assert_eq!(got, sig, "lower_decode round-trip lost samples ({mode:?})");
        }
    }

    #[test]
    fn lml_pipeline_composition_equals_chain_and_fused() {
        // The reversible-only LmlPipeline composition (the compile-firewalled form)
        // produces the SAME bytes as the free morphism chain AND the fused kernel.
        let sig = synth(4, 2500);
        for mode in [LpcMode::Fixed, LpcMode::Anytime { max_order: 16, deadline: None }] {
            let via_pipeline = encode_lossless_pipeline(eeg_raw(sig.clone()), mode).unwrap();
            let via_chain = encode_lossless(&eeg_raw(sig.clone()), mode).unwrap();
            let views: Vec<&[i64]> = sig.iter().map(|c| c.as_slice()).collect();
            let fused = compress_with_mode_views(&views, 0, mode).unwrap();
            assert_eq!(via_pipeline.bytes(), via_chain.bytes(), "pipeline != chain ({mode:?})");
            assert_eq!(via_pipeline.bytes(), fused.as_slice(), "pipeline != fused ({mode:?})");
        }
    }

    #[test]
    fn dsp_morphisms_are_reversible() {
        // Closes the "reversibility declared, not proven" hole for the DSP
        // morphisms: unprocess(process(x)) == x, checked directly on the stage data.
        let raw = eeg_raw(synth(4, 2500));
        let n = raw.abir().n_samples();
        let orig: Vec<Vec<i64>> =
            raw.abir().window_views(0, n).iter().map(|c| c.as_ref().to_vec()).collect();

        // Transform⁻¹ ∘ Transform == identity (on samples).
        assert_eq!(inverse_transform(&transform(&raw)), orig, "inverse_transform ∘ transform ≠ id");

        // Predict⁻¹ ∘ Predict == identity (on the quantized subbands), all modes.
        for mode in [LpcMode::Fixed, LpcMode::Adaptive { max_order: 16 }, LpcMode::Anytime { max_order: 16, deadline: None }] {
            let q = quantize_identity(transform(&raw));
            let before = q.per_channel.clone();
            let after = unpredict(predict(q, mode));
            assert_eq!(after.per_channel, before, "unpredict ∘ predict ≠ id ({mode:?})");
        }
    }

    #[test]
    fn packet_phantom_is_zero_overhead() {
        assert_eq!(core::mem::size_of::<Packet<Eeg>>(), core::mem::size_of::<Vec<u8>>());
    }

    #[test]
    fn lossy_lowering_is_byte_identical_to_the_kernels() {
        // Both modes first-class: the Mode-aware lowering dispatches faithfully.
        let sig = synth(4, 2500);
        let raw = eeg_raw(sig.clone());
        let views: Vec<&[i64]> = sig.iter().map(|c| c.as_slice()).collect();
        assert_eq!(
            lower_encode_mode(&raw, Mode::Lossless, LpcMode::Fixed).unwrap().bytes(),
            compress_with_mode_views(&views, 0, LpcMode::Fixed).unwrap().as_slice(),
            "Lossless lowering != kernel"
        );
        assert_eq!(
            lower_encode_mode(&raw, Mode::BoundedMae(8), LpcMode::Fixed).unwrap().bytes(),
            compress_bounded_mae(&sig, 8, LpcMode::Fixed).unwrap().as_slice(),
            "BoundedMae lowering != kernel"
        );
        assert_eq!(
            lower_encode_mode(&raw, Mode::TargetBps(4.0), LpcMode::Fixed).unwrap().bytes(),
            compress_target_bps(&sig, 4.0, LpcMode::Fixed).unwrap().as_slice(),
            "TargetBps lowering != kernel"
        );
    }

    #[test]
    fn bounded_mae_respects_the_error_bound() {
        let sig = synth(4, 2500);
        let raw = eeg_raw(sig.clone());
        let delta = 8u64;
        let packet = lower_encode_mode(&raw, Mode::BoundedMae(delta), LpcMode::Fixed).unwrap();
        let decoded = decompress(packet.bytes()).unwrap();
        let mut max_err = 0i64;
        for (c, ch) in sig.iter().enumerate() {
            for (i, &orig) in ch.iter().enumerate() {
                max_err = max_err.max((orig - decoded[c][i]).abs());
            }
        }
        assert!(max_err as u64 <= delta, "bounded-MAE δ={delta} violated: max|Δ|={max_err}");
    }

    #[test]
    fn deadzone_quantize_round_trips_within_ceil_half_step() {
        // The Lossy Quantize morphism: dequantize(quantize(x)) within ⌈q/2⌉
        // (round-to-nearest). Swapping quantize_identity → quantize_deadzone is the
        // whole lossless↔lossy difference in the ONE pipeline.
        let raw = eeg_raw(synth(4, 2500));
        let transformed = transform(&raw);
        let q = 9i64;
        let bound = (q + 1) / 2; // ⌈q/2⌉
        let back = dequantize_deadzone(quantize_deadzone(transformed.clone(), &[q]), &[q]);
        for (ch_o, ch_b) in transformed.subbands().iter().zip(back.subbands()) {
            for (sb_o, sb_b) in ch_o.iter().zip(ch_b) {
                for (i, &o) in sb_o.iter().enumerate() {
                    assert!(
                        (o - sb_b[i]).abs() <= bound,
                        "deadzone error {} > ⌈q/2⌉={bound}",
                        (o - sb_b[i]).abs()
                    );
                }
            }
        }
    }
}
