//! Real-time sample-rate pacing primitive.
//!
//! ADR 0023 / 0024 — Phase 2. World-class real-time replay needs
//! a pacer that:
//!
//!   * tracks elapsed-vs-target per sample without accumulating
//!     drift (sleeping by `period` per iteration drifts; anchoring
//!     each target to `start + n × period` doesn't);
//!   * supports pause / resume (interactive replay UIs);
//!   * exposes both a blocking `await_next()` and a non-blocking
//!     `should_emit_now()` so callers in async / event-loop contexts
//!     can yield to other work between samples;
//!   * stays microsecond-accurate via `std::time::Instant` on the
//!     sync path (`tokio::time::sleep` has ~1 ms granularity, too
//!     coarse for ≥ 1 kHz sources).
//!
//! `Outlet::push_all` (Phase 1) inlines a simpler version of this
//! logic; Phase 4 (CLI integration) and Phase 3 (Inlet) will use
//! this standalone primitive for richer control.

use std::time::{Duration, Instant};

/// Replay-rate selector. Mirrors `outlet::Rate` for the standalone
/// pacer — kept separate so users of the Pacer don't have to pull
/// in the (liblsl-feature-gated) Outlet module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaceRate {
    /// Match the nominal sample rate exactly.
    RealTime,
    /// Burst — no pacing. `should_emit_now` always returns true.
    Burst,
    /// Real-time × the scalar. `0.0` collapses to `Burst`.
    Multiplier(f64),
}

impl PaceRate {
    /// Nanoseconds per sample at this rate + nominal sample rate.
    /// Returns `None` in burst mode.
    pub fn sample_period_nanos(self, nominal_srate: f64) -> Option<u64> {
        let srate = nominal_srate.max(f64::MIN_POSITIVE);
        match self {
            Self::Burst => None,
            Self::RealTime => Some((1.0e9 / srate) as u64),
            Self::Multiplier(x) => {
                if x <= 0.0 {
                    None
                } else {
                    Some((1.0e9 / (srate * x)) as u64)
                }
            }
        }
    }
}

/// Pacer state.
///
/// Built once at the start of a replay session; `await_next` (or
/// `should_emit_now`) is called once per sample-to-emit. The pacer
/// tracks the elapsed time against `start + n × period`, never
/// against `previous_emit_time + period`, so cumulative drift is
/// bounded by the precision of the clock (microseconds on
/// modern Linux/macOS/Windows).
///
/// **Pause/resume** moves the `start` anchor by the paused
/// duration. Resume picks up where the pause began; the consumer
/// doesn't see a time jump.
pub struct Pacer {
    /// When the replay session began (monotonic).
    start: Instant,
    /// Total time spent in pause states. Subtracted from
    /// `Instant::now() - start` to get the effective elapsed
    /// playback time.
    paused_total: Duration,
    /// `Some(when_paused)` if currently paused; `None` if running.
    paused_at: Option<Instant>,
    /// Period between samples, in nanoseconds. `None` = burst mode.
    period_nanos: Option<u64>,
    /// Count of `await_next` / `should_emit_now` calls; used to
    /// compute the target wake time as
    /// `start + (n + 1) × period`.
    samples_emitted: u64,
}

impl Pacer {
    /// Build a pacer for `nominal_srate` Hz at the given replay rate.
    pub fn new(nominal_srate: f64, rate: PaceRate) -> Self {
        Self {
            start: Instant::now(),
            paused_total: Duration::ZERO,
            paused_at: None,
            period_nanos: rate.sample_period_nanos(nominal_srate),
            samples_emitted: 0,
        }
    }

    /// Block until the next sample should be emitted. Returns
    /// immediately in burst mode.
    pub fn await_next(&mut self) {
        if let Some(target) = self.next_target_instant() {
            let now = Instant::now();
            if target > now {
                std::thread::sleep(target - now);
            }
        }
        self.samples_emitted = self.samples_emitted.saturating_add(1);
    }

    /// Non-blocking probe. Returns `true` if the caller should emit
    /// now, `false` if it should yield + try again later. In burst
    /// mode always returns `true`. Increments the sample counter
    /// only on `true` (so callers can poll without skipping
    /// samples).
    pub fn should_emit_now(&mut self) -> bool {
        let Some(target) = self.next_target_instant() else {
            self.samples_emitted = self.samples_emitted.saturating_add(1);
            return true;
        };
        if Instant::now() >= target {
            self.samples_emitted = self.samples_emitted.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Pause the pacer. Subsequent `await_next` / `should_emit_now`
    /// calls return as-if-time-froze until `resume` is called.
    /// Idempotent — calling pause while already paused is a no-op.
    pub fn pause(&mut self) {
        if self.paused_at.is_none() {
            self.paused_at = Some(Instant::now());
        }
    }

    /// Resume the pacer. The total paused duration is folded into
    /// the start-anchor offset so the next sample comes after the
    /// same effective elapsed time, just shifted forward in
    /// wall-clock by the pause duration. Idempotent on a running
    /// pacer.
    pub fn resume(&mut self) {
        if let Some(at) = self.paused_at.take() {
            self.paused_total = self.paused_total.saturating_add(at.elapsed());
        }
    }

    /// `Some(target)` = next sample's wake-time. `None` = burst
    /// mode (no scheduling).
    ///
    /// Sample N's target is `start + N × period`, so sample 0 emits
    /// at `start` (immediately) and sample 1 at `start + period`.
    /// `samples_emitted` IS the index of the next sample to emit,
    /// so it's used unmodified — `+1` was an off-by-one bug.
    fn next_target_instant(&self) -> Option<Instant> {
        let period = self.period_nanos?;
        let offset_ns = self.samples_emitted.saturating_mul(period);
        // If paused, anchor moves forward by `paused_total +
        // current_pause_duration` so the offset stays effective.
        let pause_offset = self.paused_total
            + self
                .paused_at
                .map(|p| p.elapsed())
                .unwrap_or_default();
        Some(self.start + Duration::from_nanos(offset_ns) + pause_offset)
    }

    /// Number of samples the pacer has gated through so far.
    pub fn samples_emitted(&self) -> u64 {
        self.samples_emitted
    }

    /// Cumulative elapsed playback time excluding paused stretches.
    pub fn effective_elapsed(&self) -> Duration {
        let now = Instant::now();
        let raw = now - self.start;
        let pause_offset = self.paused_total
            + self
                .paused_at
                .map(|p| p.elapsed())
                .unwrap_or_default();
        raw.saturating_sub(pause_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pace_rate_period_realtime() {
        assert_eq!(PaceRate::RealTime.sample_period_nanos(256.0), Some(3_906_250));
    }

    #[test]
    fn pace_rate_period_burst() {
        assert!(PaceRate::Burst.sample_period_nanos(256.0).is_none());
    }

    #[test]
    fn pace_rate_multiplier_2x() {
        let p_1x = PaceRate::RealTime.sample_period_nanos(256.0).unwrap();
        let p_2x = PaceRate::Multiplier(2.0).sample_period_nanos(256.0).unwrap();
        assert!((p_2x as i64 - (p_1x / 2) as i64).abs() < 100);
    }

    #[test]
    fn pace_rate_zero_multiplier_is_burst() {
        assert!(PaceRate::Multiplier(0.0).sample_period_nanos(256.0).is_none());
    }

    #[test]
    fn burst_pacer_never_waits() {
        let mut pacer = Pacer::new(256.0, PaceRate::Burst);
        let start = Instant::now();
        for _ in 0..1000 {
            pacer.await_next();
        }
        // 1000 burst-mode pacer cycles should complete way under
        // 100 ms on any modern machine — no sleep involved.
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(pacer.samples_emitted(), 1000);
    }

    #[test]
    fn pacer_realtime_within_tolerance() {
        // 100 samples at 1000 Hz = 100 ms of pacing.
        let mut pacer = Pacer::new(1000.0, PaceRate::RealTime);
        let start = Instant::now();
        for _ in 0..100 {
            pacer.await_next();
        }
        let elapsed = start.elapsed();
        // OS scheduler jitter on Linux is typically < 10 ms;
        // for a 100 ms target give ±50 ms slack so the test
        // doesn't flake on busy CI.
        assert!(
            elapsed >= Duration::from_millis(80),
            "100 samples at 1 kHz should take ≥ 80 ms, took {:?}",
            elapsed
        );
        assert!(
            elapsed <= Duration::from_millis(200),
            "100 samples at 1 kHz should take ≤ 200 ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn should_emit_now_doesnt_advance_when_false() {
        // 100 Hz = 10 ms period. Right after construction, first
        // emit should be ~immediate; subsequent calls before the
        // period elapses should return false without advancing
        // the counter.
        let mut pacer = Pacer::new(100.0, PaceRate::RealTime);
        let first = pacer.should_emit_now();
        assert!(first); // sample 0 emits immediately
        assert_eq!(pacer.samples_emitted(), 1);
        // No sleep — should return false, counter unchanged.
        let second = pacer.should_emit_now();
        assert!(!second);
        assert_eq!(pacer.samples_emitted(), 1);
    }

    #[test]
    fn pause_resume_extends_wall_time() {
        // 100 Hz = 10 ms period. Pause for 50 ms, resume, drain
        // 5 samples. Effective elapsed playback should match the
        // 5-sample target (~50 ms), but wall-clock will be ~100 ms.
        let mut pacer = Pacer::new(100.0, PaceRate::RealTime);
        // Emit 1 sample, pause, sleep 50 ms wall, resume.
        pacer.await_next();
        pacer.pause();
        std::thread::sleep(Duration::from_millis(50));
        pacer.resume();
        // Now drain 4 more samples — effective elapsed across all 5
        // should be ~50 ms, not ~100 ms.
        for _ in 0..4 {
            pacer.await_next();
        }
        let effective = pacer.effective_elapsed();
        // ~50 ms effective with generous tolerance for OS jitter.
        assert!(
            effective < Duration::from_millis(150),
            "effective_elapsed should exclude the 50 ms pause; got {:?}",
            effective
        );
    }
}
