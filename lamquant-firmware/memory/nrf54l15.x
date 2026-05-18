/* LamQuant v7.7 firmware linker script — NRF54L15 / Cortex-M33.
 *
 * Status: scaffold, pending HAL bringup (ADR 0019 T6.b follow-up).
 *
 * NRF54L15 datasheet (Nordic):
 *   - Cortex-M33 with TrustZone + FPU, 128 MHz
 *   - 1.5 MB Flash (RRAM, 0x00000000)
 *   - 256 KB RAM   (0x20000000)
 *   - 22 KB RAM additional for low-power retention region (0x2003C000)
 *   - Raytac MDBT54L module exposes the secure/non-secure split via PSA;
 *     this scaffold pins the non-secure-callable region at the lower
 *     half of FLASH per Raytac reference layout. Refine when nrf54l15-hal
 *     ships upstream.
 *
 * Once the per-target build.rs dispatch lands, this file is copied to
 * OUT_DIR/memory.x when CARGO_FEATURE_TARGET_NRF54L15 is active.
 */

MEMORY {
    /* Non-secure callable region — refined per PSA partitioning. */
    FLASH       : ORIGIN = 0x00000000, LENGTH = 1536K
    RAM         : ORIGIN = 0x20000000, LENGTH = 256K
    /* Low-power retention RAM (write before suspend, read after wake). */
    RETAIN_RAM  : ORIGIN = 0x2003C000, LENGTH = 22K
}

REGION_ALIAS("REGION_TEXT",   FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA",   RAM);
REGION_ALIAS("REGION_BSS",    RAM);
REGION_ALIAS("REGION_HEAP",   RAM);
REGION_ALIAS("REGION_STACK",  RAM);

/* SNN scratch overlay (Track B.4) — shares physical RAM with the
 * lifting + LPC residual buffers that run earlier in the pipeline.
 * See lamquant-firmware/memory/rp2350.x for the design rationale.
 */
SECTIONS {
    .bss.snn_overlay (NOLOAD) : ALIGN(8) {
        __snn_overlay_start = .;
        *(.bss.snn_overlay)
        *(.bss.snn_overlay.*)
        . = ALIGN(8);
        __snn_overlay_end = .;
    } > RAM
} INSERT AFTER .bss;
