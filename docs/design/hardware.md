# LamQuant Gen 7: Hardware Interface

Target: RP2350 with dual Hazard3 RISC-V cores at 150MHz.

## 1. Hazard3 RISC-V

Firmware uses `Zbb` (bit-manipulation) extensions for branchless ternary MAC. No FPU — all arithmetic is Q31/Q30 fixed-point. Compile flags: `-march=rv32imac_zbb -mabi=ilp32 -Os`.

## 2. Memory Layout

All TNN weights pinned to SRAM4 for 0-waitstate access (no XIP flash stalls).

| Region | Size | Contents |
|--------|------|----------|
| SRAM0 | 64KB | ADC buffer (`raw_adc_buffer[21][2500]`) |
| SRAM4 | 64KB | TNN weights (~33KB packed) + workspace |
| SRAM5 | 64KB | Activation double-buffers, latent output |
| Stack | 2.4KB | Enforced via `-Wstack-usage=2400 -Werror` |

Placement: `__attribute__((section(".sram4_tnn"), aligned(4)))`.

## 3. PMP Stack Guard

`stack_guard.c` configures PMP entry 0 in NAPOT mode to poison a 2KB region at `0x20007800`. Lock bit set — immutable after boot. Any stack overflow triggers a hardware exception → `enter_safe_mode()` → BLE flush → WFI halt.

## 4. SPI Bus Allocation

| Bus | Slave | Clock | Use |
|-----|-------|-------|-----|
| SPI0 | ADS1299 | 4MHz | AFE data acquisition |
| SPI1 | nRF52840 | 8MHz | BLE packet transmission |
