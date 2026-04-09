# Hardware Reference

Target: RP2350 with dual Hazard3 RISC-V cores at 150 MHz.

---

## 1. Hazard3 RISC-V Cores

The RP2350 contains two Hazard3 RISC-V cores implementing RV32IMAC + Zbb (bit-manipulation extensions). LamQuant uses:

- **Core 0**: All firmware execution (boot, scheduler, DSP, TNN inference, BLE TX)
- **Core 1**: Reserved for future SNN wake detection (Gen 7). Currently idle.

Compiler flags: `-march=rv32imac_zbb -mabi=ilp32 -Os`

The `Zbb` extension provides hardware `clz`, `ctz`, `cpop`, `min`, `max`, `orc.b`, and `rev8` instructions. These enable branchless ternary MAC operations without pipeline stalls.

There is no FPU. All arithmetic is Q31 or Q30 fixed-point using 64-bit intermediate precision (see [mathematics.md](mathematics.md)).

---

## 2. Memory Layout

| Region | Address | Size | Contents |
|--------|---------|------|----------|
| SRAM0 | 0x20000000 | 64 KB | `raw_adc_buffer[21][2500]` — ADC DMA target, biquad workspace |
| SRAM4 | 0x20040000 | 64 KB | TNN weights (~42.3 KB packed), FSQ tables, rANS freq tables |
| SRAM5 | 0x20050000 | 64 KB | Activation double-buffers `act_buf_a/b[96][2500]`, `latent_output[32][312]`, `lifting_tile_2d[6][32]` |
| Stack | 0x20007800-0x20008000 | 2.4 KB | Call frames. PMP-guarded below 0x20007800 |
| Flash (XIP) | 0x10000000 | 16 Mbit | Firmware binary, read via XIP cache |

### Placement attributes

Weights are pinned to SRAM4 for zero-waitstate access (no XIP flash stalls during inference):

```c
__attribute__((section(".sram4_tnn"), aligned(4)))
```

Activation workspace is pinned to SRAM5 to isolate from SRAM4 weight reads and SRAM0 ADC DMA:

```c
__attribute__((section(".workspace_sram5")))
```

The ADC buffer lives in SRAM0, isolated from Core 1 and SRAM4 traffic:

```c
__attribute__((aligned(4), section(".workspace_sram0")))
```

### Stack budget

The 2.4 KB stack limit is enforced at compile time:

```cmake
-Wstack-usage=2400
-Werror=stack-usage=2400
```

Any function exceeding this causes a build failure. The deepest call path is: `lamquant_scheduler_run()` -> `run_golden_path()` -> `run_tnn_encoder_inference()` -> `run_focal_block()` -> `ternary_conv1d_single()`.

---

## 3. PMP Stack Guard

`stack_guard.c` configures PMP entry 0 at boot:

```c
// NAPOT mode: protect 2 KB region at 0x20007800
uint32_t pmpaddr0 = (0x20007800 >> 2) | 0xFF;
uint32_t pmpcfg = PMP_CFG_L | PMP_CFG_A_NAPOT;  // Lock + NAPOT, no R/W/X
```

- **Address**: 0x20007800 (immediately below stack top at 0x20008000)
- **Size**: 2 KB NAPOT region
- **Permissions**: None (R=0, W=0, X=0) — any access triggers a hardware exception
- **Lock bit**: Set (L=1) — immutable after boot, survives until hard reset

When the stack overflows into this region, the hardware exception handler `stack_overflow_exception()` fires:

```c
void __attribute__((naked)) stack_overflow_exception(void) {
    __asm__ volatile (
        "li   sp, 0x20008000\n"  // Reset SP to stack top
        "j    enter_safe_mode\n"  // Jump to safe mode (never returns)
    );
}
```

The `stack_reset_canaries()` function is an intentional no-op — PMP entries with the Lock bit cannot be modified after boot.

---

## 4. SPI Bus Allocation

| Bus | Master | Slave | Clock | CPOL/CPHA | Use |
|-----|--------|-------|-------|-----------|-----|
| SPI0 | RP2350 | ADS1299 | 4 MHz | 0/1 | AFE data acquisition (production) |
| SPI1 | RP2350 | nRF52840 | 8 MHz | 0/0 | BLE packet transmission |

SPI0 and SPI1 are mutually exclusive with other peripherals on their respective GPIO groups. The ADS1299 uses CPHA=1 (data valid on trailing edge) per the TI datasheet.

---

## 5. Pin Map (RP2350)

| GPIO | Function | Direction | Notes |
|------|----------|-----------|-------|
| 0 | UART TX (debug) | OUT | Debug console output |
| 1 | UART RX (debug) | IN | Debug console input |
| 2 | SPI0 SCK / LC-ADC input | OUT / IN | Mutually exclusive via compile flag |
| 3 | SPI0 MOSI | OUT | ADS1299 command/data |
| 4 | SPI0 MISO | IN | ADS1299 read data |
| 5 | SPI0 CS (ADS1299) | OUT | Active-low chip select |
| 6 | ADS1299 DRDY | IN | Data-ready interrupt (falling edge) |
| 7 | ADS1299 RESET | OUT | Active-low hardware reset |
| 10 | SPI1 SCK (BLE) | OUT | nRF52840 SPI slave clock |
| 11 | SPI1 MOSI (BLE) | OUT | Entropy buffer -> nRF52840 |
| 12 | SPI1 MISO (BLE) | IN | nRF52840 -> RP2350 (unused currently) |
| 13 | SPI1 CS (BLE) | OUT | Active-low chip select |

---

## 6. DMA Configuration

### ADC DMA (LC-ADC path)

```
Source:  PIO0 RX FIFO (pio->rxf[sm])
Dest:    raw_adc_buffer[21][2500]
Size:    32-bit transfers
Count:   21 * 2500 = 52,500 words
Pacing:  PIO RX DREQ (triggered when LC comparator fires)
IRQ:     DMA_IRQ_0 -> on_adc_dma_complete()
```

### BLE TX DMA

```
Source:  bit_buffer[240] (entropy-coded payload)
Dest:    SPI1 TX FIFO (spi_get_hw(spi1)->dr)
Size:    8-bit transfers
Count:   1-240 bytes (variable)
Pacing:  SPI1 TX DREQ
Mode:    Non-blocking (DMA runs while scheduler re-arms ADC)
```

---

## 7. ADS1299 Configuration

The TI ADS1299 is an 8-channel 24-bit delta-sigma ADC designed for biopotential measurement.

| Register | Value | Setting |
|----------|-------|---------|
| CONFIG1 | 0x96 | 250 Hz sample rate, internal oscillator |
| CONFIG2 | 0xC0 | Test signals off, internal reference buffer |
| CONFIG3 | 0xE0 | Internal reference enabled, bias enabled |
| CHnSET | 0x60 | PGA gain = 24x, normal electrode input |
| LOFF | 0x03 | Lead-off detection: 6 nA DC current source |

Data format: 27 bytes per sample — 3 status bytes + 8 channels x 3 bytes (24-bit signed). The driver sign-extends to 32-bit and left-shifts by 8 to convert to Q31 scale.

The DRDY interrupt fires at 250 Hz. Each interrupt reads 27 bytes via SPI0, parses 8 channels, and stores into `raw_adc_buffer[0..7][sample_idx]`. When `sample_idx` reaches 2500, `on_adc_dma_complete()` signals the scheduler.

---

## 8. Watchdog

- **Timeout**: 500 ms
- **Pet location**: Main loop (`watchdog_update()` after each `lamquant_scheduler_run()`)
- **Reset behavior**: Warm boot — SRAM4 weights survive soft reset, so the boot sequence skips KAT and CRC checks
- **Safe mode**: `enter_safe_mode()` enters infinite WFI, deliberately triggering a watchdog reset

---

## 9. Power States

Defined in `power_states.c`:

| State | Entry | Behavior |
|-------|-------|----------|
| Normal | Default | Scheduler loop: sleep -> process window -> transmit |
| Dormant | `enter_dormant_state()` | WFI instruction, wakes on any IRQ (DMA, timer) |
| Safe Mode | `enter_safe_mode()` | BLE flush -> abort inference -> reset DSP -> WFI forever (watchdog reset) |

Safe mode sequence:
1. `ble_spi_emergency_flush()` — Drop pending BLE packets
2. `ble_enter_standby()` — De-assert CS, disable SPI1
3. `scheduler_abort_inference()` — Zero lifting workspace, reset state to SLEEP
4. `dsp_reset_pipeline()` — Set `filters_initialized = false` (forces biquad re-init on resume)
5. Infinite WFI (watchdog resets the chip after 500 ms)
