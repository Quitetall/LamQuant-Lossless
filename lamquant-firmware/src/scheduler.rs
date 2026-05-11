//! v7.7 pipeline scheduler — single mode, no parallel fallback.
//!
//! Two execution modes (chosen at session start by host serial command):
//!
//!   Mode 1 (Neural):   ADC DMA → biquad → LPC → lifting DWT → SNN classify
//!                      → dispatch Core 1 → TNN encoder on L3 → adaptive FSQ
//!                      → context-adaptive rANS → BLE TX (.lmq packet)
//!
//!   Mode 2 (Lossless): ADC DMA → biquad → LPC → lifting DWT → detail
//!                      threshold → LPC delta + Golomb-Rice → BLE TX (.lml)
//!
//! v7.1 had a "golden/lightning" deadline race; v7.7 dropped it. 264 ms
//! TNN inference fits 38× within a 10 s window so the race was always
//! pointless. Real-time margin = 36×–38× regardless of mode.
//!
//! The scheduler ties together every other module. This file is the
//! single integration point: peripheral handles in, BLE bytes out.

use crate::codec::detail_threshold;
use crate::codec::fsq_adaptive::FsqAdaptive;
use crate::codec::hybrid_entropy::{encode_lossless, encode_neural};
use crate::codec::lpc_delta::LpcDelta;
use crate::codec::quality::QualityMode;
use crate::codec::rans_context::{RansContextEncoder, RansTables, RANS_BUF_SIZE};
use crate::dsp::biquad::{HpFilter, HpFilterBank, NUM_CHANNELS, WINDOW_SAMPLES};
use crate::dsp::lifting::{forward_all_channels, LiftingScratch, Subbands};
use crate::dsp::lpc::{analyze_all_channels, LpcOutput};
use crate::neural::{focal, snn};
use crate::safety::SafetyState;

/// Active codec mode for the current session. Set by host before recording
/// starts — the scheduler does not switch modes mid-session.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum CodecMode {
    Neural = 1,
    Lossless = 2,
}

/// Output mode for transmission: BLE, USB, or both.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum OutputMode {
    BleOnly = 0,
    UsbOnly = 1,
    Dual = 2,
}

/// Output of one window-encode cycle. Caller is responsible for moving the
/// bytes onto the wire (BLE DMA TX or USB CDC write).
pub struct EncodeResult<'a> {
    pub bytes: &'a [u8],
    pub mode: CodecMode,
}

/// Owns every per-window allocation: filter state, LPC scratch, subbands,
/// FSQ adaptive state, rANS encoder + tables, LPC delta encoder, output buf.
///
/// Sized to fit comfortably in SRAM. ~110 KB total. Allocated as a `static`
/// in `main.rs` to ensure deterministic placement.
pub struct PipelineScheduler {
    pub hp: HpFilterBank,
    pub lpc: LpcOutput,
    pub subbands: Subbands,
    pub lifting_scratch: LiftingScratch,
    pub fsq: FsqAdaptive,
    pub rans: RansContextEncoder,
    pub lpc_delta: LpcDelta,
    pub lossless_buf: [u8; RANS_BUF_SIZE],

    pub codec_mode: CodecMode,
    pub output_mode: OutputMode,
    pub quality: QualityMode,
    pub hp_cutoff: HpFilter,
    pub channel_mask: u32,
    pub frame_seq: u32,
}

impl PipelineScheduler {
    pub fn new() -> Self {
        Self {
            hp: HpFilterBank::new(),
            lpc: LpcOutput::zeroed(),
            subbands: Subbands::zeroed(),
            lifting_scratch: LiftingScratch::zeroed(),
            fsq: FsqAdaptive::new(),
            rans: RansContextEncoder::new(RansTables::default_placeholder()),
            lpc_delta: LpcDelta::new(),
            lossless_buf: [0; RANS_BUF_SIZE],

            codec_mode: CodecMode::Neural,
            output_mode: OutputMode::BleOnly,
            quality: QualityMode::Clinical,
            hp_cutoff: HpFilter::DEFAULT,
            channel_mask: (1 << NUM_CHANNELS) - 1,
            frame_seq: 0,
        }
    }

    pub fn set_codec_mode(&mut self, m: CodecMode) {
        self.codec_mode = m;
        // Reset stateful encoders so the next packet is a clean keyframe.
        self.lpc_delta.reset();
    }

    pub fn set_quality(&mut self, q: QualityMode) {
        self.quality = q;
    }

    pub fn set_hp_cutoff(&mut self, c: HpFilter) {
        self.hp_cutoff = c;
        self.hp.reset();
    }

    pub fn set_channel_mask(&mut self, mask: u32) {
        self.channel_mask = mask & ((1 << NUM_CHANNELS) - 1);
    }

    /// Run the full pipeline on one 2500-sample window.
    ///
    /// Inputs:
    ///   `signal`         — Q31 ADC buffer, one window worth, modified in-place
    ///                      by the biquad
    ///   `activity_map`   — SNN per-group output. In Phase 5 stub (zeros)
    ///                      until SNN inference lands. Caller pre-populates.
    ///   `safety`         — patient-safety state for event logging + retry
    ///                      buffer + impedance trend
    ///
    /// Returns the encoded BLE packet bytes (Mode 1: rANS; Mode 2: LPC + Rice).
    pub fn encode_window<'a>(
        &'a mut self,
        signal: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS],
        activity_map: &[[u8; 79]; 8],
        snn_activity_sum: u8,
        safety: &mut SafetyState,
        now_ms: u32,
    ) -> EncodeResult<'a> {
        // Stage 1: HP biquad prefilter (in-place).
        self.hp
            .run(signal, WINDOW_SAMPLES, self.hp_cutoff, self.channel_mask);

        // Stage 2: LPC analysis (order-8 per channel).
        analyze_all_channels(signal, &mut self.lpc);

        // Stage 2b: pre-ictal snapshot of the LPC residual (first 313 samples,
        // matching the L3-approx temporal length). Captured BEFORE lifting
        // because lifting clobbers `self.lpc.residual` in place to avoid a
        // 10 KB scratch buffer.
        safety.push_preictal(&self.lpc.residual_l3_view(), now_ms);

        // Stage 3: 3-level lifting DWT on the LPC residual (in place).
        forward_all_channels(
            &mut self.lpc.residual,
            &mut self.lifting_scratch,
            &mut self.subbands,
        );

        // Stage 4: encode per active codec mode.
        let result = match self.codec_mode {
            CodecMode::Neural => {
                // 4a: convert L3 approximation [21][313] from i32 → i16 with
                //     saturation. The TNN encoder operates in i16 act
                //     (W2A16) — anything beyond i16 range is clamped, not
                //     wrapped, so artifacts saturate to peak rather than
                //     wrap into the wrong sign.
                let mut l3_i16 = [[0i16; 313]; NUM_CHANNELS];
                for ch in 0..NUM_CHANNELS {
                    for t in 0..313 {
                        let v = self.subbands.l3_approx[ch][t];
                        l3_i16[ch][t] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                }

                // 4b: SNN classify the L3 — populates the activity map the
                //     FSQ adaptive encoder consumes for per-block level
                //     selection. The caller-supplied `activity_map` arg
                //     stays as the reference path (lets scheduler tests
                //     inject deterministic activity) but we call the real
                //     SNN unconditionally so its membrane state stays in
                //     sync window-to-window.
                snn::inference(&l3_i16);

                // 4c: TNN focal forward — produces the [32][79] latent that
                //     FSQ + rANS compress. This replaces the prior stub
                //     that copied L3 directly into the latent slots.
                let mut latent_i16 = [[0i16; 79]; 32];
                focal::forward(&l3_i16, &mut latent_i16);

                // FSQ adaptive expects an i32 latent; widen.
                let mut latent_i32 = [[0i32; 79]; 32];
                for d in 0..32 {
                    for t in 0..79 {
                        latent_i32[d][t] = latent_i16[d][t] as i32;
                    }
                }

                self.fsq.encode(&latent_i32, activity_map, self.quality);
                let out = encode_neural(&mut self.rans, &self.fsq, snn_activity_sum);
                EncodeResult {
                    bytes: out.bytes,
                    mode: CodecMode::Neural,
                }
            }
            CodecMode::Lossless => {
                // Threshold detail subbands per quality mode.
                detail_threshold::apply(
                    &mut self.subbands.l3_detail,
                    &mut self.subbands.l2_detail,
                    &mut self.subbands.l1_detail,
                    self.quality,
                );
                let out = encode_lossless(
                    &mut self.lossless_buf,
                    &mut self.lpc_delta,
                    &self.lpc.coeffs,
                    &self.subbands,
                    self.quality,
                );
                EncodeResult {
                    bytes: out.bytes,
                    mode: CodecMode::Lossless,
                }
            }
        };

        // Stage 5: safety hooks — push to BLE retry buffer, update counters.
        // Pre-ictal capture moved to stage 2b (before lifting clobbers residual).
        safety.ble_push_packet(result.bytes, self.frame_seq);
        safety.faults.total_windows_encoded =
            safety.faults.total_windows_encoded.saturating_add(1);

        let _ = now_ms;

        self.frame_seq = self.frame_seq.wrapping_add(1);
        result
    }
}

impl Default for PipelineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

impl LpcOutput {
    /// View of `residual[ch][..313]` as a `[21][313]` array for the
    /// pre-ictal buffer push. The pre-ictal buffer captures the LPC-
    /// decorrelated signal so when a seizure fires we can transmit a
    /// 20-second lookback at clinical quality.
    pub fn residual_l3_view(&self) -> [[i32; 313]; NUM_CHANNELS] {
        let mut out = [[0i32; 313]; NUM_CHANNELS];
        for ch in 0..NUM_CHANNELS {
            for i in 0..313 {
                out[ch][i] = self.residual[ch][i];
            }
        }
        out
    }
}
