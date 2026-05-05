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
