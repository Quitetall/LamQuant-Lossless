//! Transport layer — peripheral I/O for BLE radio + USB CDC.
//!
//! Compiled only on `target_arch = "riscv32"` because each module touches
//! rp235x-hal peripherals (SPI1, USB serial) directly. Hybrid entropy
//! orchestration lives in `codec::hybrid_entropy`.
//!
//! Phase 4 status:
//!   - ble:  SPI1 + DMA TX to nRF52840 + adaptive coding rate ✅
//!   - usb:  raw 24-bit ADC stream over USB CDC ✅
//!
//! Both modules are skeletons: protocol logic is complete, but actual
//! peripheral wiring (claim DMA channel, attach interrupt handler, etc.)
//! happens in Phase 5 when the scheduler integrates the full pipeline.

pub mod ble;
pub mod usb;
