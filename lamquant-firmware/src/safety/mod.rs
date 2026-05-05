//! Patient-safety subsystems for FDA 510(k) Class II/III readiness.
//!
//! Eight independent SRAM-resident state machines:
//!   1. BLE retry buffer        (1.9 KB) — packet retransmission
//!   2. Impedance trend monitor (3.4 KB) — electrode failure prediction
//!   3. Pre-ictal ring buffer   (51.3 KB) — ~20 s lookback for seizure onset
//!   4. Channel quality scores  (0.2 KB) — per-channel composite metric
//!   5. Event log / audit trail (1.6 KB) — 100-entry circular log
//!   6. Seizure timestamp log   (0.8 KB) — onset + duration for 50 events
//!   7. Battery monitor         (0.1 KB) — voltage / capacity history
//!   8. Watchdog fault counters (0.1 KB) — reliability metrics
//!
//! Total: ~59 KB SRAM. All operations O(1), no malloc, no float, no division.
//! Port of `firmware/core/safety_features.{h,c}`.

pub mod state;

pub use state::{
    EventType, SafetyState, BLE_RETRY_PKT_SIZE, BLE_RETRY_SLOTS,
    BATTERY_HISTORY, EVENT_LOG_SIZE, IMPEDANCE_CHANNELS, IMPEDANCE_HISTORY,
    PREICTAL_CHANNELS, PREICTAL_SAMPLES, PREICTAL_WINDOWS, SEIZURE_LOG_SIZE,
};
