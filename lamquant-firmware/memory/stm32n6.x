/* LamQuant v7.7 firmware linker script — STM32N6 / Cortex-M55 + NPU.
 *
 * Status: scaffold, pending HAL bringup (ADR 0019 T6.b follow-up).
 *
 * STM32N6 datasheet (ST):
 *   - Cortex-M55 with Helium MVE (M-Profile Vector Extension), 800 MHz
 *   - Neural-ART NPU accelerator (~0.6 TOPS int8) — separate AXI bus
 *   - 4.2 MB AXI-SRAM (0x34000000) — primary working memory, partitioned
 *     across CPU + NPU
 *   - 4 KB backup RAM (0x40036400) — survives VBAT
 *   - 128 KB flash (0x08000000) — code-only; production firmware lives
 *     in external XIP flash via FMC/OCTOSPI at 0x90000000 (board-specific,
 *     Raytac default 16 MB)
 *   - 2 MB NPU shared memory at 0x34200000 — reserved for tensor
 *     activations + weight cache; carved out of AXI-SRAM
 *
 * NPU enablement is out-of-scope for the firmware-hub bringup (per
 * ADR 0019); the canonical compute pipeline runs CPU-only initially.
 * NPU support lands in a separate sprint once Cube-AI / cmsis-nn
 * codegen integrates with the LamQuant TNN exporter.
 *
 * Once the per-target build.rs dispatch lands, this file is copied to
 * OUT_DIR/memory.x when CARGO_FEATURE_TARGET_STM32N6 is active.
 */

MEMORY {
    /* External XIP flash via OCTOSPI — code + rodata live here. */
    FLASH       : ORIGIN = 0x90000000, LENGTH = 16384K
    /* AXI-SRAM allocated to the CPU side. Reserve 2 MB at the top for
     * NPU activations / weight cache (NPU_SHARED below). */
    RAM         : ORIGIN = 0x34000000, LENGTH = 2048K
    NPU_SHARED  : ORIGIN = 0x34200000, LENGTH = 2048K
    /* On-chip 128 KB flash — used for boot stage / NSC region only. */
    BOOT_FLASH  : ORIGIN = 0x08000000, LENGTH = 128K
    /* VBAT-backed retain region. */
    RETAIN_RAM  : ORIGIN = 0x40036400, LENGTH = 4K
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

/* NPU shared region — Neural-ART activations + weight cache.
 * Items tagged `#[link_section = ".npu_shared"]` land here. NPU
 * codegen integration writes to this region during inference.
 */
SECTIONS {
    .npu_shared (NOLOAD) : ALIGN(16) {
        __npu_shared_start = .;
        *(.npu_shared)
        *(.npu_shared.*)
        . = ALIGN(16);
        __npu_shared_end = .;
    } > NPU_SHARED
} INSERT AFTER .bss.snn_overlay;
