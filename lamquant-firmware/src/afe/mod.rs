//! Analog Front-End drivers.
//!
//! Phase 5 status:
//!   - ads1299: TI ADS1299 8-channel 24-bit AFE driver ✅
//!
//! ADS1299 connects via SPI0 (production hardware) and emits a 250 Hz
//! sample stream. DRDY interrupt drives DMA reads; sign-extended samples
//! land in `raw_adc_buffer` which the scheduler picks up window-by-window.

pub mod ads1299;
