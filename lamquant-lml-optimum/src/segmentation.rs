//! Change-point segmentation (ADR 0054 Lever C) — a causal, deterministic,
//! integer-fed change-point detector that emits adaptive-filter reset points at
//! signal-derived regime boundaries, IN ADDITION to the fixed periodic reset.
//!
//! HHI's BWC uses a *fixed* IntraPeriod; our RLS/MV-RLS already reset on a fixed
//! period too. A signal-derived reset lets the adaptive predictor restart exactly
//! when the statistics change (seizure onset, movement artefact, electrode pop),
//! re-converging on the new regime instead of dragging stale weights across the
//! boundary.
//!
//! ## Determinism / decodability
//! The detector consumes the **losslessly-exact reconstructed samples** (the same
//! integers the encoder saw), one at a time, in causal order. It uses only f64
//! `+ − × ÷` (IEEE-754 correctly-rounded ⇒ bit-identical across host/MCU, exactly
//! like the RLS recursion). Therefore the decoder, replaying the identical sample
//! stream through [`ChangePoint::observe`], recomputes the *same* boundary set
//! with **zero per-boundary side information** — only a 1-bit "segmentation on/off"
//! flag is signaled.
//!
//! ## Detector
//! A dual-EWMA of the windowed sample energy: a fast mean tracks the local energy,
//! a slow mean the background. When the fast mean diverges from the slow mean by a
//! multiplicative threshold (and a minimum dwell since the last boundary has
//! elapsed, to avoid thrashing), a reset is declared and the slow mean is snapped
//! to the fast mean so the next boundary needs a *fresh* divergence.
//!
//! The detector is per-channel: each channel runs its own [`ChangePoint`], so a
//! regime change on one electrode does not reset the others.

/// Fast-EWMA smoothing (per sample). Larger ⇒ tracks local energy faster.
const ALPHA_FAST: f64 = 0.05;
/// Slow-EWMA smoothing (per sample). The background reference.
const ALPHA_SLOW: f64 = 0.002;
/// Multiplicative divergence threshold: a boundary fires when the fast energy
/// exceeds `RATIO_UP ×` the slow energy (a burst) OR drops below `1/RATIO_UP ×`
/// the slow energy (a lull) — either direction is a regime change the predictor
/// should re-adapt to.
const RATIO_UP: f64 = 4.0;
/// Minimum samples between two detected boundaries (dwell) — prevents a single
/// noisy transient from emitting a reset every sample.
const MIN_DWELL: usize = 256;
/// Warm-up: do not emit any boundary until the slow EWMA has seen at least this
/// many samples (otherwise the cold-start transient trips immediately).
const WARMUP: usize = 512;

/// Causal per-channel change-point detector. Encode and decode each construct one
/// per channel and feed it the identical (reconstructed) integer samples.
pub struct ChangePoint {
    fast: f64,
    slow: f64,
    seen: usize,
    /// samples since the last emitted boundary (or since start)
    since: usize,
}

impl ChangePoint {
    /// Fresh detector (mirrors [`crate::rls::Rls::new`] — same state both sides).
    pub fn new() -> Self {
        Self {
            fast: 0.0,
            slow: 0.0,
            seen: 0,
            since: 0,
        }
    }

    /// Feed the next integer sample. Returns `true` iff a regime boundary is
    /// declared AT this sample (i.e. the predictor should reset BEFORE coding it).
    /// Pure `+ − × ÷` on f64 ⇒ identical encode/decode.
    pub fn observe(&mut self, x: i64) -> bool {
        let e = x as f64 * x as f64;
        if self.seen == 0 {
            self.fast = e;
            self.slow = e;
        } else {
            self.fast += ALPHA_FAST * (e - self.fast);
            self.slow += ALPHA_SLOW * (e - self.slow);
        }
        self.seen += 1;
        self.since += 1;

        if self.seen < WARMUP || self.since < MIN_DWELL {
            return false;
        }
        // Multiplicative divergence either way. Guard the divisions (slow can be 0
        // on a flat-line prefix); a 0 background can't diverge, so no boundary.
        let burst = self.slow > 0.0 && self.fast > self.slow * RATIO_UP;
        let lull = self.fast > 0.0 && self.slow > self.fast * RATIO_UP;
        if burst || lull {
            // Snap the background to the new regime so the next boundary needs a
            // fresh divergence, and restart the dwell counter.
            self.slow = self.fast;
            self.since = 0;
            true
        } else {
            false
        }
    }
}

impl Default for ChangePoint {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat then high-energy regime must emit exactly one boundary (after warm-up
    /// + dwell), and encode/decode replaying the SAME samples agree bit-for-bit.
    #[test]
    fn fires_on_regime_change_and_is_deterministic() {
        let mut seq: Vec<i64> = (0..2000)
            .map(|i| ((i as f64 * 0.1).sin() * 5.0) as i64)
            .collect();
        seq.extend((0..2000).map(|i| ((i as f64 * 0.1).sin() * 5000.0) as i64));

        let mut det_a = ChangePoint::new();
        let mut det_b = ChangePoint::new();
        let mut bounds_a = alloc::vec::Vec::new();
        let mut bounds_b = alloc::vec::Vec::new();
        for (i, &x) in seq.iter().enumerate() {
            if det_a.observe(x) {
                bounds_a.push(i);
            }
            if det_b.observe(x) {
                bounds_b.push(i);
            }
        }
        assert_eq!(bounds_a, bounds_b, "detector must be deterministic");
        assert!(
            !bounds_a.is_empty(),
            "a 1000x energy jump must trip a boundary"
        );
        // every boundary respects the dwell + warm-up
        assert!(bounds_a[0] >= WARMUP);
        for w in bounds_a.windows(2) {
            assert!(w[1] - w[0] >= MIN_DWELL, "dwell honored");
        }
    }

    /// A stationary signal emits NO boundary (seg=on degenerates to the fixed reset).
    #[test]
    fn no_boundary_on_stationary() {
        let mut det = ChangePoint::new();
        let mut n = 0;
        for i in 0..5000i64 {
            if det.observe(((i as f64 * 0.07).sin() * 1000.0) as i64) {
                n += 1;
            }
        }
        assert_eq!(n, 0, "stationary signal must not segment");
    }
}
