/* LamQuant v7.7 firmware linker script — RP2350 / Hazard3 RISC-V.
 *
 * Entry script via rustflags `-Tmemory.x`.
 * Order of resolution:
 *   memory.x (this file) — defines MEMORY + REGION_ALIAS
 *     INCLUDE link.x       — riscv-rt section layout (.text, .data, .bss, ...)
 *       INCLUDE device.x   — rp235x-pac IRQ vector PROVIDEs
 *   then INSERT AFTER for our RP2350-specific sections (.bi_entries, .start_block)
 *
 * NOTE: name `memory.x`, not `device.x` — rp235x-pac already emits a
 * device.x with IRQ PROVIDEs. Naming collision would clobber MEMORY.
 */

MEMORY {
    BOOT_LOADER : ORIGIN = 0x10000000, LENGTH = 0x100
    FLASH       : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    RAM         : ORIGIN = 0x20000000, LENGTH = 512K
}

REGION_ALIAS("REGION_TEXT",   FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA",   RAM);
REGION_ALIAS("REGION_BSS",    RAM);
REGION_ALIAS("REGION_HEAP",   RAM);
REGION_ALIAS("REGION_STACK",  RAM);

/* link.x (riscv-rt) is added separately via rustflags `-Tlink.x` to avoid
 * INCLUDE search-path issues with rust-lld. */

/* RP2350 image_def header — required by boot ROM to identify firmware. */
SECTIONS {
    .start_block : ALIGN(4) {
        __start_block_addr = .;
        KEEP(*(.start_block));
    } > FLASH
} INSERT AFTER .text;

/* picotool / debugger metadata (rp-binary-info entries). */
SECTIONS {
    .bi_entries : ALIGN(4) {
        __bi_entries_start = .;
        KEEP(*(.bi_entries));
        . = ALIGN(4);
        __bi_entries_end = .;
    } > FLASH
} INSERT AFTER .rodata;

/* SNN scratch overlay (Track B.4).
 *
 * SsmScratch + activation ping-pongs in `neural::snn::STATE` consume
 * ~270 KB worst-case under the current T=313 full-buffer impl. These
 * buffers are only live during `snn::inference()`, which the
 * scheduler runs AFTER `lifting_scratch` and `lpc.residual` have been
 * consumed. We collapse the address ranges so the live-at-different-
 * times buffers share physical RAM.
 *
 * Items tagged `#[link_section = ".bss.snn_overlay"]` land here.
 * cargo size --release prints the section before/after to verify the
 * overlap doesn't push past the 510 KB budget.
 *
 * Current impl note: SsmScratch is sized for full T=313 buffers. The
 * production target is T-chunked CHUNK=32 buffers (~25 KB) per the
 * v7.7.1 plan — that refactor is a follow-up commit. Until then, the
 * overlay is necessary-but-not-sufficient.
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
