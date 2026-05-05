//! Safety state — eight subsystems consolidated into one BSS-resident struct.
//!
//! Layout matches the C `safety_state_t` byte-for-byte so existing
//! cross-language tools and FDA audit dumps still work.

// ─── Constants ──────────────────────────────────────────────────────

pub const BLE_RETRY_SLOTS: usize = 8;
pub const BLE_RETRY_PKT_SIZE: usize = 240;

pub const IMPEDANCE_CHANNELS: usize = 21;
pub const IMPEDANCE_HISTORY: usize = 60;

pub const PREICTAL_WINDOWS: usize = 2;
pub const PREICTAL_CHANNELS: usize = 21;
pub const PREICTAL_SAMPLES: usize = 313;

pub const EVENT_LOG_SIZE: usize = 100;
pub const SEIZURE_LOG_SIZE: usize = 50;
pub const BATTERY_HISTORY: usize = 60;

const NO_ACTIVE_SEIZURE: u8 = 0xFF;

// ─── Event types ────────────────────────────────────────────────────

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum EventType {
    BootCold = 0x01,
    BootWarm = 0x02,
    WatchdogReset = 0x03,
    SeizureDetect = 0x10,
    SeizureEnd = 0x11,
    BleDisconnect = 0x20,
    BleReconnect = 0x21,
    BleRetry = 0x22,
    ImpedanceAlert = 0x30,
    ChannelDead = 0x31,
    ChannelRecovered = 0x32,
    BatteryLow = 0x40,
    BatteryCritical = 0x41,
    StackOverflow = 0x50,
    CrcFail = 0x51,
    ModeSwitch = 0x60,
    Core1Timeout = 0x61,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct EventEntry {
    pub timestamp_ms: u32,
    pub r#type: u8,
    pub channel: u8,
    pub data_hi: u8,
    pub data_lo: u8,
    pub sequence: u32,
    pub _reserved: u32,
}

#[repr(C)]
pub struct EventLog {
    pub entries: [EventEntry; EVENT_LOG_SIZE],
    pub write_idx: u8,
    pub total_events: u32,
    pub boot_count: u32,
}

impl Default for EventLog {
    fn default() -> Self {
        Self {
            entries: [EventEntry::default(); EVENT_LOG_SIZE],
            write_idx: 0,
            total_events: 0,
            boot_count: 0,
        }
    }
}

// ─── Subsystem structs ──────────────────────────────────────────────

#[repr(C)]
pub struct BleRetryBuf {
    pub packets: [[u8; BLE_RETRY_PKT_SIZE]; BLE_RETRY_SLOTS],
    pub lengths: [u16; BLE_RETRY_SLOTS],
    pub sequence: [u32; BLE_RETRY_SLOTS],
    pub head: u8,
    pub count: u8,
    pub tail: u8,
    pub ack_pending: bool,
}

impl Default for BleRetryBuf {
    fn default() -> Self {
        Self {
            packets: [[0; BLE_RETRY_PKT_SIZE]; BLE_RETRY_SLOTS],
            lengths: [0; BLE_RETRY_SLOTS],
            sequence: [0; BLE_RETRY_SLOTS],
            head: 0,
            count: 0,
            tail: 0,
            ack_pending: false,
        }
    }
}

#[repr(C)]
pub struct ImpedanceMonitor {
    pub history: [[u16; IMPEDANCE_HISTORY]; IMPEDANCE_CHANNELS],
    pub write_idx: u8,
    pub samples_filled: u8,
    pub slope_per_hour: [i16; IMPEDANCE_CHANNELS],
    pub current: [u16; IMPEDANCE_CHANNELS],
    pub alert_flags: u8,
}

impl Default for ImpedanceMonitor {
    fn default() -> Self {
        Self {
            history: [[0; IMPEDANCE_HISTORY]; IMPEDANCE_CHANNELS],
            write_idx: 0,
            samples_filled: 0,
            slope_per_hour: [0; IMPEDANCE_CHANNELS],
            current: [0; IMPEDANCE_CHANNELS],
            alert_flags: 0,
        }
    }
}

#[repr(C)]
pub struct PreIctalBuf {
    pub windows: [[[i32; PREICTAL_SAMPLES]; PREICTAL_CHANNELS]; PREICTAL_WINDOWS],
    pub timestamps: [u32; PREICTAL_WINDOWS],
    pub write_idx: u8,
    pub seizure_triggered: bool,
    pub trigger_time: u32,
}

impl Default for PreIctalBuf {
    fn default() -> Self {
        Self {
            windows: [[[0; PREICTAL_SAMPLES]; PREICTAL_CHANNELS]; PREICTAL_WINDOWS],
            timestamps: [0; PREICTAL_WINDOWS],
            write_idx: 0,
            seizure_triggered: false,
            trigger_time: 0,
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct ChannelQuality {
    pub quality: [u8; IMPEDANCE_CHANNELS],
    pub artifact_count: [u8; IMPEDANCE_CHANNELS],
    pub last_update_ms: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct SeizureEntry {
    pub onset_ms: u32,
    pub duration_s: u16,
    pub severity: u8,
    pub channels: u8,
}

#[repr(C)]
pub struct SeizureLog {
    pub entries: [SeizureEntry; SEIZURE_LOG_SIZE],
    pub write_idx: u8,
    pub active_seizure: u8, // 0xFF = none
    pub total_seizures: u16,
}

impl Default for SeizureLog {
    fn default() -> Self {
        Self {
            entries: [SeizureEntry::default(); SEIZURE_LOG_SIZE],
            write_idx: 0,
            active_seizure: NO_ACTIVE_SEIZURE,
            total_seizures: 0,
        }
    }
}

#[repr(C)]
pub struct BatteryMonitor {
    pub voltage_pct: [u8; BATTERY_HISTORY],
    pub write_idx: u8,
    pub current_pct: u8,
    pub mv: u16,
    pub low_warning: bool,
    pub critical_warning: bool,
}

impl Default for BatteryMonitor {
    fn default() -> Self {
        Self {
            voltage_pct: [0; BATTERY_HISTORY],
            write_idx: 0,
            current_pct: 0,
            mv: 0,
            low_warning: false,
            critical_warning: false,
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub struct FaultCounters {
    pub watchdog_resets: u32,
    pub core1_timeouts: u32,
    pub crc_failures: u32,
    pub stack_overflows: u32,
    pub ble_drops: u32,
    pub uptime_seconds: u32,
    pub total_windows_encoded: u32,
    pub total_seizures_detected: u32,
}

// ─── Master state ───────────────────────────────────────────────────

#[repr(C)]
pub struct SafetyState {
    pub ble_retry: BleRetryBuf,
    pub impedance: ImpedanceMonitor,
    pub preictal: PreIctalBuf,
    pub channel_quality: ChannelQuality,
    pub event_log: EventLog,
    pub seizure_log: SeizureLog,
    pub battery: BatteryMonitor,
    pub faults: FaultCounters,
}

impl Default for SafetyState {
    fn default() -> Self {
        Self {
            ble_retry: Default::default(),
            impedance: Default::default(),
            preictal: Default::default(),
            channel_quality: Default::default(),
            event_log: Default::default(),
            seizure_log: Default::default(),
            battery: Default::default(),
            faults: Default::default(),
        }
    }
}

impl SafetyState {
    /// Initialize at boot. Logs a cold-boot event into the audit trail.
    pub fn init(&mut self, now_ms: u32) {
        self.event_log.boot_count = self.event_log.boot_count.saturating_add(1);
        self.log_event(now_ms, EventType::BootCold, 0xFF, 0);
    }

    // ── Event log ──────────────────────────────────────────────────

    pub fn log_event(&mut self, now_ms: u32, ty: EventType, channel: u8, data: u16) {
        let log = &mut self.event_log;
        let idx = log.write_idx as usize;
        let e = &mut log.entries[idx];
        e.timestamp_ms = now_ms;
        e.r#type = ty as u8;
        e.channel = channel;
        e.data_hi = (data >> 8) as u8;
        e.data_lo = (data & 0xFF) as u8;
        e.sequence = log.total_events;
        e._reserved = 0;
        log.write_idx = (log.write_idx + 1) % EVENT_LOG_SIZE as u8;
        log.total_events = log.total_events.saturating_add(1);
    }

    // ── Impedance trend ────────────────────────────────────────────

    /// Caller iterates channels 0..=20 in order; advance happens on ch==20.
    pub fn update_impedance(&mut self, now_ms: u32, channel: u8, kohm: u16) {
        if channel as usize >= IMPEDANCE_CHANNELS {
            return;
        }
        let imp = &mut self.impedance;
        let widx = imp.write_idx as usize;
        imp.history[channel as usize][widx] = kohm;
        imp.current[channel as usize] = kohm;

        // Slope per hour: (newest - oldest) × 6 (= ×4 + ×2; no division).
        if imp.samples_filled as usize >= IMPEDANCE_HISTORY {
            let oldest_idx = (imp.write_idx as usize + 1) % IMPEDANCE_HISTORY;
            let diff = kohm as i32 - imp.history[channel as usize][oldest_idx] as i32;
            imp.slope_per_hour[channel as usize] = ((diff << 2) + (diff << 1)) as i16;
        }

        // 50 kohm threshold → alert.
        let bit = 1u8 << (channel & 7);
        if kohm > 50 {
            imp.alert_flags |= bit;
            self.log_event(now_ms, EventType::ImpedanceAlert, channel, kohm);
        } else {
            self.impedance.alert_flags &= !bit;
        }

        // Advance index after channel 20 sweep.
        if channel as usize == IMPEDANCE_CHANNELS - 1 {
            let imp = &mut self.impedance;
            imp.write_idx = (imp.write_idx + 1) % IMPEDANCE_HISTORY as u8;
            if (imp.samples_filled as usize) < IMPEDANCE_HISTORY {
                imp.samples_filled += 1;
            }
        }
    }

    // ── Battery ────────────────────────────────────────────────────

    pub fn update_battery(&mut self, now_ms: u32, mv: u16, pct: u8) {
        let bat = &mut self.battery;
        bat.voltage_pct[bat.write_idx as usize] = pct;
        bat.write_idx = (bat.write_idx + 1) % BATTERY_HISTORY as u8;
        bat.current_pct = pct;
        bat.mv = mv;

        let was_low = bat.low_warning;
        let was_critical = bat.critical_warning;
        bat.low_warning = pct < 20;
        bat.critical_warning = pct < 5;

        if bat.low_warning && !was_low {
            self.log_event(now_ms, EventType::BatteryLow, 0xFF, mv);
        }
        if self.battery.critical_warning && !was_critical {
            self.log_event(now_ms, EventType::BatteryCritical, 0xFF, mv);
        }
    }

    // ── Pre-ictal ring buffer ──────────────────────────────────────

    pub fn push_preictal(
        &mut self,
        l3_approx: &[[i32; PREICTAL_SAMPLES]; PREICTAL_CHANNELS],
        timestamp: u32,
    ) {
        let pb = &mut self.preictal;
        let idx = pb.write_idx as usize;
        pb.windows[idx] = *l3_approx;
        pb.timestamps[idx] = timestamp;
        pb.write_idx = (pb.write_idx + 1) % PREICTAL_WINDOWS as u8;
    }

    // ── Seizure diary ──────────────────────────────────────────────

    pub fn on_seizure_start(&mut self, now_ms: u32, severity: u8, channel_mask: u8) {
        let sl = &mut self.seizure_log;
        let idx = sl.write_idx as usize;
        sl.entries[idx] = SeizureEntry {
            onset_ms: now_ms,
            duration_s: 0,
            severity,
            channels: channel_mask,
        };
        sl.active_seizure = sl.write_idx;

        self.preictal.seizure_triggered = true;
        self.preictal.trigger_time = now_ms;

        self.log_event(now_ms, EventType::SeizureDetect, channel_mask, severity as u16);
        self.faults.total_seizures_detected =
            self.faults.total_seizures_detected.saturating_add(1);
    }

    pub fn on_seizure_end(&mut self, now_ms: u32) {
        let sl = &mut self.seizure_log;
        if sl.active_seizure == NO_ACTIVE_SEIZURE {
            return;
        }
        let idx = sl.active_seizure as usize;
        let e = &mut sl.entries[idx];
        let elapsed_ms = now_ms.wrapping_sub(e.onset_ms);
        e.duration_s = (elapsed_ms >> 10) as u16; // ms→s via shift, ~2.4% error

        sl.active_seizure = NO_ACTIVE_SEIZURE;
        sl.write_idx = (sl.write_idx + 1) % SEIZURE_LOG_SIZE as u8;
        sl.total_seizures = sl.total_seizures.saturating_add(1);

        self.preictal.seizure_triggered = false;
        let dur = e.duration_s;
        self.log_event(now_ms, EventType::SeizureEnd, 0xFF, dur);
    }

    // ── BLE retry buffer ───────────────────────────────────────────

    pub fn ble_push_packet(&mut self, data: &[u8], seq: u32) {
        let rb = &mut self.ble_retry;
        let len = data.len().min(BLE_RETRY_PKT_SIZE);
        let slot = rb.head as usize;
        rb.packets[slot][..len].copy_from_slice(&data[..len]);
        rb.lengths[slot] = len as u16;
        rb.sequence[slot] = seq;
        rb.head = (rb.head + 1) % BLE_RETRY_SLOTS as u8;
        if (rb.count as usize) < BLE_RETRY_SLOTS {
            rb.count += 1;
        } else {
            // Overwriting oldest unACKed.
            rb.tail = rb.head;
        }
    }

    pub fn ble_has_retry(&self) -> bool {
        self.ble_retry.count > 0
    }

    /// Returns (packet_bytes, sequence) of the oldest unACKed packet.
    pub fn ble_peek_retry(&self) -> Option<(&[u8], u32)> {
        let rb = &self.ble_retry;
        if rb.count == 0 {
            return None;
        }
        let slot = rb.tail as usize;
        Some((&rb.packets[slot][..rb.lengths[slot] as usize], rb.sequence[slot]))
    }

    pub fn ble_ack_retry(&mut self) {
        let rb = &mut self.ble_retry;
        if rb.count == 0 {
            return;
        }
        rb.tail = (rb.tail + 1) % BLE_RETRY_SLOTS as u8;
        rb.count -= 1;
    }

    // ── Channel quality ────────────────────────────────────────────

    pub fn update_channel_quality(&mut self, now_ms: u32, channel: u8, quality: u8) {
        if channel as usize >= IMPEDANCE_CHANNELS {
            return;
        }
        let prev = self.channel_quality.quality[channel as usize];
        self.channel_quality.quality[channel as usize] = quality;
        self.channel_quality.last_update_ms = (now_ms & 0xFFFF) as u16;

        if quality == 0 && prev > 0 {
            self.log_event(now_ms, EventType::ChannelDead, channel, prev as u16);
        } else if quality > 0 && prev == 0 {
            self.log_event(now_ms, EventType::ChannelRecovered, channel, quality as u16);
        }
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn boot_logs_event() {
        let mut s = SafetyState::default();
        s.init(1000);
        assert_eq!(s.event_log.boot_count, 1);
        assert_eq!(s.event_log.entries[0].r#type, EventType::BootCold as u8);
    }

    #[test]
    fn ble_retry_fifo() {
        let mut s = SafetyState::default();
        s.ble_push_packet(b"hello", 1);
        s.ble_push_packet(b"world", 2);
        let (data, seq) = s.ble_peek_retry().unwrap();
        assert_eq!(data, b"hello");
        assert_eq!(seq, 1);
        s.ble_ack_retry();
        let (data, seq) = s.ble_peek_retry().unwrap();
        assert_eq!(data, b"world");
        assert_eq!(seq, 2);
        s.ble_ack_retry();
        assert!(!s.ble_has_retry());
    }

    #[test]
    fn impedance_above_threshold_alerts() {
        let mut s = SafetyState::default();
        s.update_impedance(0, 3, 60);
        assert_eq!(s.impedance.alert_flags & (1 << 3), 1 << 3);
    }

    #[test]
    fn seizure_lifecycle() {
        let mut s = SafetyState::default();
        s.on_seizure_start(1000, 200, 0b0000_0011);
        assert_eq!(s.seizure_log.active_seizure, 0);
        assert!(s.preictal.seizure_triggered);
        s.on_seizure_end(11_000);
        assert_eq!(s.seizure_log.active_seizure, 0xFF);
        assert!(!s.preictal.seizure_triggered);
        let dur_s = s.seizure_log.entries[0].duration_s;
        // 10 s elapsed; >> 10 of 10000 ms ≈ 9 s.
        assert!((9..=10).contains(&dur_s));
    }
}
