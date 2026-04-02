# LamQuant Gen 6 Deployment Architecture

LamQuant operates as a three-phase pipeline tracking data from high-level neural training to bare-metal microcontroller execution.

## Phase 1: AI Distillation Stack (PyTorch)
The training pipeline prepares the neural manifold for hardware deployment:
1. **Teacher Training**: A large-scale Transformer model is trained on clinical EEG data to establish a high-fidelity baseline and generate validation masks.
2. **Student Distillation**: A Ternary MobileNetV5-based autoencoder is distilled from the teacher. It uses Learned Step Size Quantization (LSQ) to maintain clinical fidelity within the RP2350's memory limits.
3. **Firmware Export**: The `export_firmware.py` script serializes trained weights and Q31 alphas into C-headers, applying 32-bit alignment and SRAM4 memory pinning.

## Phase 2: Embedded Inference Stack (RP2350 C)
The firmware executes real-time signal processing and inference:
1. **Data Acquisition**: PIO and DMA engines stream 21-channel EEG data into SRAM buffers, allowing the CPU to remain in low-power states until a 2500-sample frame is ready.
2. **Neural Inference**: The core engine executes ternary convolutions using the Hazard3 RISC-V Bitmanip (Zbb) extensions, completing inference within the 4.0ms real-time deadline.
3. **Telemetry**: Compressed latent variables are packaged and transmitted via BLE or high-speed SPI.

## Phase 3: Base Station Telemetry
A base station receives the compressed stream and applies Bayesian Step-by-Step Learning (BSBL) to reconstruct the signal for clinical analysis and seizure detection.
