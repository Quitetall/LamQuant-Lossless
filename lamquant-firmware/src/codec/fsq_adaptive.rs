//! Adaptive FSQ — variable-L quantization driven by SNN activity_map.
//!
//!   Quiescent  L=2  → 1.00 bits/dim → 32 bits/group
//!   Active     L=3  → 1.58 bits/dim → 50.6 bits/group
//!   High       L=5  → 2.32 bits/dim → 74.2 bits/group
//!   Clinical   L=32 → 5.00 bits/dim → 160 bits/group (overrides activity)
//!
//! In Clinical mode every timestep uses L=32 regardless of SNN activity.
//! The FSQ level bitmap is transmitted in the BLE packet header so the
//! decoder knows which alphabet was used per timestep.

use super::quality::{ActivityLevel, QualityMode};
use lamquant_weights::generated::fsq;

/// Latent dimensions emitted by the encoder. Const, never changes per arch.
pub const LATENT_DIMS: usize = 32;
/// Latent timesteps after stride-4 (313 / 4 ≈ 79 with ceil-padding).
pub const LATENT_TIMESTEPS: usize = 79;
/// Number of SNN output groups (same as encoder.bneck output groups).
pub const N_GROUPS: usize = 8;

/// Number of available FSQ level configurations.
pub const N_CONFIGS: usize = 4;

/// L values per configuration index. Order is contractual (encoded in
/// the BLE packet's 2-bit per-timestep level bitmap).
pub const LEVELS: [u32; N_CONFIGS] = [2, 3, 5, 32];

/// Map ActivityLevel + QualityMode to FSQ config index.
#[inline]
pub fn config_idx_for(level: ActivityLevel, mode: QualityMode) -> usize {
    if mode == QualityMode::Clinical {
        return 3; // L=32 always in clinical
    }
    match level {
        ActivityLevel::Quiescent => 0, // L=2
        ActivityLevel::Active => 1,    // L=3
        ActivityLevel::High => 2,      // L=5
    }
}

/// Per-config quantization parameters. inv_range_q31 = (L * 2^31) / range.
#[derive(Copy, Clone)]
pub struct LevelConfig {
    pub num_levels: u32,
    pub inv_range_q31: i32,
}

/// State machine for adaptive FSQ. Holds precomputed configs (rebuilt at
/// init from the latent vmin/vmax baked into `lamquant-weights`) plus the
/// per-timestep level bitmap and quantized symbol output buffer.
pub struct FsqAdaptive {
    configs: [LevelConfig; N_CONFIGS],
    /// Per-timestep config index (0..3). Transmitted with the packet.
    level_bitmap: [u8; LATENT_TIMESTEPS],
    /// Quantized symbol output, dim-major.
    output: [u32; LATENT_DIMS * LATENT_TIMESTEPS],
    output_len: usize,
}

impl FsqAdaptive {
    /// Build a config from the latent vmin/vmax in the weights crate.
    /// Mirrors the C `fsq_adaptive_init` exactly.
    pub fn new() -> Self {
        let range = (fsq::VMAX_Q31 as i64 - fsq::VMIN_Q31 as i64).max(1);
        let mut configs = [LevelConfig {
            num_levels: 0,
            inv_range_q31: 0,
        }; N_CONFIGS];
        let mut i = 0;
        while i < N_CONFIGS {
            let l = LEVELS[i] as i64;
            configs[i] = LevelConfig {
                num_levels: LEVELS[i],
                inv_range_q31: ((l << 31) / range) as i32,
            };
            i += 1;
        }
        Self {
            configs,
            level_bitmap: [0; LATENT_TIMESTEPS],
            output: [0; LATENT_DIMS * LATENT_TIMESTEPS],
            output_len: 0,
        }
    }

    /// Quantize one Q31 latent value into a discrete bin index.
    #[inline]
    fn quantize(&self, val: i32, cfg_idx: usize) -> u32 {
        let cfg = &self.configs[cfg_idx];
        let shifted = (val.saturating_sub(fsq::VMIN_Q31)).max(0);
        let product = (shifted as i64) * (cfg.inv_range_q31 as i64);
        let bin = (product >> 31) as i32;
        bin.clamp(0, (cfg.num_levels - 1) as i32) as u32
    }

    /// Encode the full latent tensor with adaptive FSQ. Returns symbol count.
    ///
    /// `latent[d][t]` is column-major (Q31). `activity_map[g][t]` is the
    /// SNN classification per group per timestep. We take the max activity
    /// across groups at each timestep — single L for all 32 dims at that t.
    pub fn encode(
        &mut self,
        latent: &[[i32; LATENT_TIMESTEPS]; LATENT_DIMS],
        activity_map: &[[u8; LATENT_TIMESTEPS]; N_GROUPS],
        mode: QualityMode,
    ) -> usize {
        self.output_len = 0;

        for t in 0..LATENT_TIMESTEPS {
            // Max activity across groups at this timestep.
            let mut max_a = ActivityLevel::Quiescent;
            for g in 0..N_GROUPS {
                let a = ActivityLevel::from_u8(activity_map[g][t]);
                if a > max_a {
                    max_a = a;
                }
            }

            let cfg_idx = config_idx_for(max_a, mode);
            self.level_bitmap[t] = cfg_idx as u8;

            // Quantize all 32 dims at this timestep.
            for d in 0..LATENT_DIMS {
                self.output[self.output_len] = self.quantize(latent[d][t], cfg_idx);
                self.output_len += 1;
            }
        }

        self.output_len
    }

    pub fn symbols(&self) -> &[u32] {
        &self.output[..self.output_len]
    }

    pub fn level_bitmap(&self) -> &[u8; LATENT_TIMESTEPS] {
        &self.level_bitmap
    }

    /// Number of FSQ levels used at timestep `t`.
    pub fn num_levels_at(&self, t: usize) -> u32 {
        self.configs[self.level_bitmap[t] as usize].num_levels
    }

    /// 16-bit summary: 2 bits per group, packed. Used in BLE header.
    /// Picks the most frequent activity level per group across the frame.
    pub fn build_level_summary(
        &self,
        activity_map: &[[u8; LATENT_TIMESTEPS]; N_GROUPS],
    ) -> u16 {
        let mut summary: u16 = 0;
        for g in 0..N_GROUPS {
            let mut counts = [0u32; 3];
            for t in 0..LATENT_TIMESTEPS {
                let a = activity_map[g][t] as usize;
                if a < 3 {
                    counts[a] += 1;
                }
            }
            let mut best = 0;
            if counts[1] > counts[best] {
                best = 1;
            }
            if counts[2] > counts[best] {
                best = 2;
            }
            summary |= ((best as u16) & 0x03) << (g * 2);
        }
        summary
    }
}

impl Default for FsqAdaptive {
    fn default() -> Self {
        Self::new()
    }
}
