/* LamQuant v7.7 firmware linker script — ESP32-P4 / HP400 RISC-V.
 *
 * Status: scaffold, pending HAL bringup (ADR 0019 T6.b follow-up).
 *
 * ESP32-P4 datasheet (Espressif):
 *   - HP400 RV32IMAFC dual-core RISC-V, 360-400 MHz
 *   - 768 KB HP SRAM   (0x4FF00000)
 *   - 256 KB LP SRAM   (0x50000000) — retained across deep-sleep
 *   - Flash: external XIP via SPI (board-specific; Raytac default 4 MB
 *     at 0x42000000 cached, 0x44000000 uncached)
 *   - 32 MB PSRAM optional (board-specific, 0x48000000)
 *
 * probe-rs upstream does not yet support ESP32-P4 chip-id flashing;
 * the `fw_flash_esp32p4` launcher label calls this out. Use
 * `cargo build` + esp-idf-monitor as the interim workflow.
 *
 * Once the per-target build.rs dispatch lands, this file is copied to
 * OUT_DIR/memory.x when CARGO_FEATURE_TARGET_ESP32P4 is active.
 */

MEMORY {
    /* External XIP flash (cached). 4 MB Raytac default; widen on
     * larger flash boards via board-overlay rather than editing here. */
    FLASH       : ORIGIN = 0x42000000, LENGTH = 4096K
    /* HP SRAM — primary working memory for compute kernels. */
    RAM         : ORIGIN = 0x4FF00000, LENGTH = 768K
    /* LP SRAM — survives deep-sleep; reserve for retained state. */
    RETAIN_RAM  : ORIGIN = 0x50000000, LENGTH = 256K
}

REGION_ALIAS("REGION_TEXT",   FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA",   RAM);
REGION_ALIAS("REGION_BSS",    RAM);
REGION_ALIAS("REGION_HEAP",   RAM);
REGION_ALIAS("REGION_STACK",  RAM);

SECTIONS {
    .bss.snn_overlay (NOLOAD) : ALIGN(8) {
        __snn_overlay_start = .;
        *(.bss.snn_overlay)
        *(.bss.snn_overlay.*)
        . = ALIGN(8);
        __snn_overlay_end = .;
    } > RAM
} INSERT AFTER .bss;
