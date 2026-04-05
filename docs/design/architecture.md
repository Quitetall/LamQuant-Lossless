# LamQuant Gen 7 Deployment Architecture

Three-phase pipeline: PyTorch training → RP2350 firmware → base station decode.

## Phase 1: AI Distillation (PyTorch)

1. **Teacher Training**: FP32 Transformer autoencoder trained on clinical EEG (CHB-MIT). Produces validation masks and distillation targets.
2. **Student Distillation**: TernaryMobileNetV5 autoencoder (W2A16) distilled from teacher via LSQ. 4 focal layers (stride 1→2→2→2 = 8x), 96-wide, 32-dim latent.
3. **Firmware Export**: `export_firmware.py` serializes ternary weights (2-bit packed, 4 per byte), Q31 alphas, GroupNorm params, and FSQ/rANS frequency tables into C headers. All arrays SRAM4-pinned, 32-bit aligned.

## Phase 2: Embedded Inference (RP2350 C)

1. **Acquisition**: PIO + DMA stream 21-channel EEG at 250Hz into SRAM0. CPU sleeps (WFI) until 2500-sample window completes. Production path: ADS1299 SPI AFE. Dev path: LC comparator + PIO.
2. **Biquad Prefilter**: 3-stage cascade (HP 0.5Hz → LP 50Hz → notch 60Hz), Q30 fixed-point, all 21 channels in-place.
3. **Path Selection**: Variance-based mode detection. Seizure or budget overrun → lightning path.
   - **Golden**: TNN encoder → FSQ → rANS → BLE. Full quality.
   - **Lightning**: Toeplitz CS → 2D lifting → LPC → Golomb-Rice → BLE. Guaranteed deadline.
4. **Transmit**: DMA-paced SPI1 to nRF52840. Max 240-byte BLE packet per 4ms window.

## Phase 3: Base Station Decode

Route A: Student ternary decoder (always available, lightweight).
Route B: Teacher FP32 decoder (higher quality, requires latent upsampling T/8 → T via linear interpolation).
