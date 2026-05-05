//! BLE radio transport — SPI1 + DMA TX to nRF52840 BLE controller.
//!
//! Pin assignments (matches firmware/transport/ble_spi_host.c):
//!   GPIO10 = SCK, GPIO11 = MOSI, GPIO12 = MISO, GPIO13 = CS
//!   SPI1 @ 8 MHz, MSB first, CPOL=0 CPHA=0.
//!
//! Adaptive coding: PSRR-driven (Q10 dB) channel coding rate switching.
//!   ≥ 60.0 dB → 1/1 full rate, normal FSQ
//!   < 60.0 dB → 2/3 redundancy, reduced FSQ
//!   < 40.0 dB → 1/2 redundancy, reduced FSQ
//!
//! Phase 4: protocol + state machine ported. Phase 5 wires actual SPI1
//! peripheral handles + DMA channel claim from `rp235x_hal::dma`.

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, Ordering};

/// Channel coding rate. Lower fraction = more redundancy = harder for
/// noisy environment to corrupt the packet.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum ChannelCoding {
    /// 1/1 — no FEC overhead, max throughput.
    Rate1_1 = 0,
    /// 2/3 — moderate redundancy.
    Rate2_3 = 1,
    /// 1/2 — heavy redundancy, half throughput.
    Rate1_2 = 2,
}

/// Snapshot of the runtime BLE state. Atomic so Core 0 (scheduler) can
/// publish from `on_emc_detected` while Core 1 (encoder) reads
/// `is_fsq_reduced` to pick the FSQ table.
pub struct BleState {
    coding: AtomicU8,
    fsq_reduced: AtomicBool,
    /// Last measured PSRR in Q10 dB (1 LSB = 0.1 dB). 600 = 60.0 dB.
    psrr_q10: AtomicI32,
    initialized: AtomicBool,
}

impl BleState {
    pub const fn new() -> Self {
        Self {
            coding: AtomicU8::new(ChannelCoding::Rate1_1 as u8),
            fsq_reduced: AtomicBool::new(false),
            psrr_q10: AtomicI32::new(680), // 68.0 dB nominal
            initialized: AtomicBool::new(false),
        }
    }

    pub fn coding(&self) -> ChannelCoding {
        match self.coding.load(Ordering::Relaxed) {
            0 => ChannelCoding::Rate1_1,
            1 => ChannelCoding::Rate2_3,
            _ => ChannelCoding::Rate1_2,
        }
    }

    pub fn set_coding(&self, c: ChannelCoding) {
        self.coding.store(c as u8, Ordering::Relaxed);
    }

    pub fn is_fsq_reduced(&self) -> bool {
        self.fsq_reduced.load(Ordering::Relaxed)
    }

    pub fn psrr_q10(&self) -> i32 {
        self.psrr_q10.load(Ordering::Relaxed)
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }

    pub fn mark_initialized(&self) {
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn mark_uninitialized(&self) {
        self.initialized.store(false, Ordering::Relaxed);
    }

    /// Update PSRR + recompute coding rate. Called from main loop after
    /// each ADS1299 PSRR measurement.
    pub fn on_emc_detected(&self, psrr_q10: i32) {
        self.psrr_q10.store(psrr_q10, Ordering::Relaxed);
        if psrr_q10 < 400 {
            // < 40 dB — severe interference.
            self.set_coding(ChannelCoding::Rate1_2);
            self.fsq_reduced.store(true, Ordering::Relaxed);
        } else if psrr_q10 < 600 {
            // < 60 dB — mild interference.
            self.set_coding(ChannelCoding::Rate2_3);
            self.fsq_reduced.store(true, Ordering::Relaxed);
        } else {
            // Clean environment.
            self.set_coding(ChannelCoding::Rate1_1);
            self.fsq_reduced.store(false, Ordering::Relaxed);
        }
    }

    /// Safe-mode hook — discard all transmit state cleanly.
    pub fn emergency_flush(&self) {
        self.set_coding(ChannelCoding::Rate1_1);
        self.fsq_reduced.store(false, Ordering::Relaxed);
    }
}

/// Singleton state shared across scheduler + encoder + safety hooks.
pub static BLE_STATE: BleState = BleState::new();

// ─── SPI hardware constants ──────────────────────────────────────

pub const BLE_SPI_BAUD_HZ: u32 = 8_000_000;
pub const BLE_PIN_SCK: u8 = 10;
pub const BLE_PIN_MOSI: u8 = 11;
pub const BLE_PIN_MISO: u8 = 12;
pub const BLE_PIN_CS: u8 = 13;

// ─── BLE alert packet (12-byte SNN seizure event) ───────────────

pub const ALERT_SYNC: u8 = 0xAA;
pub const ALERT_TYPE_SNN: u8 = 0x04;

/// Build a 12-byte SNN alert packet to be transmitted out-of-band.
/// Best-effort fire-and-forget; caller checks `BLE_STATE.is_initialized()`
/// before invoking the SPI write.
pub fn build_snn_alert(level: u8, timestamp_ms: u32) -> [u8; 12] {
    let mut pkt = [0u8; 12];
    pkt[0] = ALERT_SYNC;
    pkt[1] = ALERT_TYPE_SNN;
    pkt[2] = level;
    pkt[3] = (timestamp_ms & 0xFF) as u8;
    pkt[4] = ((timestamp_ms >> 8) & 0xFF) as u8;
    pkt[5] = ((timestamp_ms >> 16) & 0xFF) as u8;
    pkt[6] = ((timestamp_ms >> 24) & 0xFF) as u8;
    // 7..11 already zero
    pkt
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn psrr_drives_coding() {
        let s = BleState::new();
        s.on_emc_detected(700);
        assert_eq!(s.coding(), ChannelCoding::Rate1_1);
        assert!(!s.is_fsq_reduced());

        s.on_emc_detected(500);
        assert_eq!(s.coding(), ChannelCoding::Rate2_3);
        assert!(s.is_fsq_reduced());

        s.on_emc_detected(300);
        assert_eq!(s.coding(), ChannelCoding::Rate1_2);
        assert!(s.is_fsq_reduced());
    }

    #[test]
    fn snn_alert_layout() {
        let pkt = build_snn_alert(2, 0xCAFE_BABE);
        assert_eq!(pkt[0], 0xAA);
        assert_eq!(pkt[1], 0x04);
        assert_eq!(pkt[2], 2);
        assert_eq!(pkt[3], 0xBE);
        assert_eq!(pkt[4], 0xBA);
        assert_eq!(pkt[5], 0xFE);
        assert_eq!(pkt[6], 0xCA);
    }
}
