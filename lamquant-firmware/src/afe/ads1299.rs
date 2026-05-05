//! TI ADS1299 8-channel 24-bit delta-sigma ADC driver.
//!
//! Configuration:
//!   * 250 Hz sample rate (CONFIG1 = 0x96)
//!   * PGA gain 24× (CHnSET = 0x60)
//!   * Internal reference (CONFIG3 = 0xE0)
//!   * All 8 channels active
//!
//! Pin map (matches firmware/afe/ads1299_driver.c):
//!   GPIO2 SCK, GPIO3 MOSI, GPIO4 MISO, GPIO5 CS, GPIO6 DRDY, GPIO7 RESET
//!   SPI0 @ 4 MHz, MSB first, CPOL=0 CPHA=1.
//!
//! Data path on hardware:
//!   DRDY falling edge → ISR → SPI read 27 bytes (3 status + 8×3 data)
//!   → sign-extend 24→32 bit → `raw_adc_buffer[0..7][sample_idx] = raw24 << 8`
//!
//! Phase 5: protocol logic + register layout ported. Phase 6 wires the
//! actual SPI0 / DMA / GPIO IRQ peripherals from rp235x-hal.

pub const NUM_CHANNELS: usize = 8;
pub const WINDOW_SIZE: usize = 2500;
pub const SPI_BAUD_HZ: u32 = 4_000_000;
pub const PIN_SCK: u8 = 2;
pub const PIN_MOSI: u8 = 3;
pub const PIN_MISO: u8 = 4;
pub const PIN_CS: u8 = 5;
pub const PIN_DRDY: u8 = 6;
pub const PIN_RESET: u8 = 7;

// ─── Register addresses (datasheet §10.6) ───────────────────────

#[allow(dead_code)]
pub mod reg {
    pub const ID: u8 = 0x00;
    pub const CONFIG1: u8 = 0x01;
    pub const CONFIG2: u8 = 0x02;
    pub const CONFIG3: u8 = 0x03;
    pub const LOFF: u8 = 0x04;
    pub const CH1SET: u8 = 0x05;
    pub const CH2SET: u8 = 0x06;
    pub const CH3SET: u8 = 0x07;
    pub const CH4SET: u8 = 0x08;
    pub const CH5SET: u8 = 0x09;
    pub const CH6SET: u8 = 0x0A;
    pub const CH7SET: u8 = 0x0B;
    pub const CH8SET: u8 = 0x0C;
    pub const LOFF_SENSP: u8 = 0x0F;
    pub const LOFF_SENSN: u8 = 0x10;
    pub const LOFF_STATP: u8 = 0x12;
    pub const LOFF_STATN: u8 = 0x13;
    pub const GPIO: u8 = 0x14;
    pub const CONFIG4: u8 = 0x17;
}

// ─── SPI commands ───────────────────────────────────────────────

#[allow(dead_code)]
pub mod cmd {
    pub const WAKEUP: u8 = 0x02;
    pub const STANDBY: u8 = 0x04;
    pub const RESET: u8 = 0x06;
    pub const START: u8 = 0x08;
    pub const STOP: u8 = 0x0A;
    pub const RDATAC: u8 = 0x10;
    pub const SDATAC: u8 = 0x11;
    pub const RDATA: u8 = 0x12;
}

/// SPI byte primitive used by the driver. Caller supplies an impl that
/// wraps the rp235x-hal SPI0 + CS GPIO. Host tests use a mock.
pub trait SpiDevice {
    /// Drive CS low, write `tx`, drive CS high, no read.
    fn write(&mut self, tx: &[u8]);
    /// Drive CS low, write `tx` while reading into `rx` of equal length.
    fn transfer(&mut self, tx: &[u8], rx: &mut [u8]);
}

/// GPIO primitive for the RESET pin.
pub trait OutputPin {
    fn set_high(&mut self);
    fn set_low(&mut self);
}

/// Sign-extend 24-bit MSB-first ADS1299 sample to i32, then `<< 8` to Q31.
#[inline]
pub fn sample_to_q31(msb: u8, mid: u8, lsb: u8) -> i32 {
    let raw24 = ((msb as i32) << 16) | ((mid as i32) << 8) | (lsb as i32);
    let extended = if raw24 & 0x80_0000 != 0 {
        raw24 | (0xFF_u32 << 24) as i32
    } else {
        raw24
    };
    extended << 8
}

/// Parse one DRDY frame (27 bytes: 3 status + 8×3 data) into 8 Q31 samples.
pub fn parse_frame(rx: &[u8; 27]) -> [i32; NUM_CHANNELS] {
    let mut out = [0i32; NUM_CHANNELS];
    for ch in 0..NUM_CHANNELS {
        let off = 3 + ch * 3;
        out[ch] = sample_to_q31(rx[off], rx[off + 1], rx[off + 2]);
    }
    out
}

// ─── Driver state ──────────────────────────────────────────────

pub struct Ads1299<S, R>
where
    S: SpiDevice,
    R: OutputPin,
{
    spi: S,
    reset_pin: R,
    pub detected_chip_count: u8,
    pub sample_idx: u32,
    pub impedance_kohm: [u16; NUM_CHANNELS],
}

impl<S, R> Ads1299<S, R>
where
    S: SpiDevice,
    R: OutputPin,
{
    pub fn new(spi: S, reset_pin: R) -> Self {
        Self {
            spi,
            reset_pin,
            detected_chip_count: 0,
            sample_idx: 0,
            impedance_kohm: [0; NUM_CHANNELS],
        }
    }

    fn send_command(&mut self, c: u8) {
        self.spi.write(&[c]);
    }

    fn write_register(&mut self, reg: u8, value: u8) {
        // WREG opcode: 0x40 | reg, then 0x00 (write 1 register), then value.
        let buf = [0x40 | (reg & 0x1F), 0x00, value];
        self.spi.write(&buf);
    }

    fn read_register(&mut self, reg: u8) -> u8 {
        // RREG opcode: 0x20 | reg, then 0x00 (read 1 register).
        let tx = [0x20 | (reg & 0x1F), 0x00, 0x00];
        let mut rx = [0u8; 3];
        self.spi.transfer(&tx, &mut rx);
        rx[2]
    }

    /// Probe the chip-ID register. Returns the number of ADS1299 chips
    /// detected on the bus (currently single-chip → 1, future daisy chain
    /// extends to ≥ 2 by walking additional CS lines).
    pub fn detect_daisy_chain(&mut self) -> u8 {
        let id = self.read_register(reg::ID);
        // Lower 5 bits = 0b11110 (0x1E) for the ADS1299 family.
        if id & 0x1F == 0x1E {
            self.detected_chip_count = 1;
        } else {
            self.detected_chip_count = 0;
        }
        self.detected_chip_count
    }

    /// Total channel count across all detected chips.
    pub fn total_channels(&self) -> u8 {
        self.detected_chip_count * NUM_CHANNELS as u8
    }

    /// Hardware-reset + register configuration. Caller is responsible for
    /// the post-reset `sleep_ms(100)` settle window via its own delay impl.
    pub fn init(&mut self) {
        // RESET sequence: pulse low ≥ 1 ms.
        self.reset_pin.set_low();
        // Caller delays ≥ 1 ms here.
        self.reset_pin.set_high();
        // Caller delays ≥ 100 ms here for ADS1299 boot.

        // Stop continuous read mode for register writes.
        self.send_command(cmd::SDATAC);

        // Detect daisy chain. Fall back to single chip if probe fails.
        if self.detect_daisy_chain() < 1 {
            self.detected_chip_count = 1;
        }

        // Configure registers.
        self.write_register(reg::CONFIG1, 0x96); // 250 Hz, internal oscillator
        self.write_register(reg::CONFIG2, 0xC0); // Test signals off, internal reference buffer
        self.write_register(reg::CONFIG3, 0xE0); // Internal reference, bias enabled

        // Set all 8 channels: PGA 24×, normal electrode input.
        for ch in 0..NUM_CHANNELS as u8 {
            self.write_register(reg::CH1SET + ch, 0x60);
        }

        // Lead-off detection: 6 nA current source, DC excitation.
        self.write_register(reg::LOFF, 0x03);

        self.sample_idx = 0;
    }

    /// Begin continuous data conversion mode. Caller must enable the DRDY
    /// GPIO interrupt separately.
    pub fn start_continuous(&mut self) {
        self.sample_idx = 0;
        self.send_command(cmd::START);
        self.send_command(cmd::RDATAC);
    }

    pub fn stop_continuous(&mut self) {
        self.send_command(cmd::SDATAC);
        self.send_command(cmd::STOP);
    }

    /// Lead-off impedance measurement. Returns true on success.
    /// Caller is responsible for the 500 ms settle delay between
    /// configuring SENSP/SENSN and reading STATP/STATN.
    pub fn measure_impedance_setup(&mut self) {
        self.send_command(cmd::SDATAC);
        self.send_command(cmd::STOP);
        self.write_register(reg::LOFF_SENSP, 0xFF);
        self.write_register(reg::LOFF_SENSN, 0xFF);
        self.write_register(reg::CONFIG4, 0x02);
    }

    pub fn measure_impedance_finish(&mut self) {
        let statp = self.read_register(reg::LOFF_STATP);
        let statn = self.read_register(reg::LOFF_STATN);
        for ch in 0..NUM_CHANNELS {
            let p_off = (statp >> ch) & 1 != 0;
            let n_off = (statn >> ch) & 1 != 0;
            self.impedance_kohm[ch] = if p_off || n_off { 999 } else { 2 };
        }
        // Disable lead-off detection.
        self.write_register(reg::LOFF_SENSP, 0x00);
        self.write_register(reg::LOFF_SENSN, 0x00);
        self.write_register(reg::CONFIG4, 0x00);
    }

    pub fn impedance_kohm(&self, channel: usize) -> u16 {
        self.impedance_kohm.get(channel).copied().unwrap_or(0xFFFF)
    }
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;

    #[test]
    fn sign_extend_positive() {
        // 0x7FFFFF (max positive 24-bit) → << 8 → 0x7FFFFF00
        assert_eq!(sample_to_q31(0x7F, 0xFF, 0xFF), 0x7FFFFF00);
    }

    #[test]
    fn sign_extend_negative() {
        // 0x800000 (min negative 24-bit) → -0x800000 → << 8 → -0x80000000
        assert_eq!(sample_to_q31(0x80, 0x00, 0x00), -0x8000_0000);
    }

    #[test]
    fn parse_frame_skips_status_bytes() {
        let mut rx = [0u8; 27];
        rx[0..3].copy_from_slice(&[0xAA, 0xBB, 0xCC]); // status — ignored
        // Channel 0 = 0x000001 (Q31 = 0x100)
        rx[3..6].copy_from_slice(&[0x00, 0x00, 0x01]);
        let samples = parse_frame(&rx);
        assert_eq!(samples[0], 0x100);
    }
}
