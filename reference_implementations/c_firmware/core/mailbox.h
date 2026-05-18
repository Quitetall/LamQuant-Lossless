/*
 * LamQuant Gen 7 — Inter-Core Mailbox
 * ====================================
 * 32-byte shared structure in SRAM8 for Core 0 ↔ Core 1 signaling.
 *
 * Protocol:
 *   Core 0 detects activity → writes DISPATCH to mailbox → SEV
 *   Core 1 wakes on WFE → reads mailbox → runs TNN+FSQ+rANS
 *   Core 1 finishes → writes DONE + result pointer → SEV
 *   Core 0 reads result, transmits BLE packet
 *
 * Invariant: Lightning Path ALWAYS runs on Core 0 regardless of Core 1 state.
 */

#ifndef MAILBOX_H
#define MAILBOX_H

#include <stdint.h>
#include <stdbool.h>
#include "hardware/sync.h"

/* Mailbox commands */
typedef enum {
    MBOX_IDLE     = 0x00,   /* No work pending */
    MBOX_DISPATCH = 0x01,   /* Core 0 → Core 1: run golden path */
    MBOX_DONE     = 0x02,   /* Core 1 → Core 0: golden path complete */
    MBOX_ABORT    = 0x03,   /* Core 0 → Core 1: cancel (timeout) */
    MBOX_ERROR    = 0xFF,   /* Core 1 → Core 0: error occurred */
} mbox_cmd_t;

/* Shared mailbox structure — exactly 32 bytes, aligned to cache line */
typedef struct __attribute__((aligned(32))) {
    volatile uint8_t  cmd;              /* mbox_cmd_t */
    volatile uint8_t  activity_sum;     /* SNN activity summary */
    volatile uint8_t  fsq_level_bitmap; /* Per-group FSQ level (2 bits × 8 = 16 bits, low byte) */
    volatile uint8_t  fsq_level_hi;     /* High byte of FSQ level bitmap */
    volatile uint32_t golden_buf_addr;  /* Pointer to golden entropy buffer */
    volatile uint32_t golden_buf_len;   /* Bytes used in golden buffer */
    volatile uint32_t sequence_num;     /* Frame sequence counter */
    volatile uint32_t encode_cycles;    /* Core 1 encode time (perf counter) */
    volatile uint32_t reserved[3];      /* Pad to 32 bytes */
} mailbox_t;

/* Mailbox lives in SRAM8 */
extern mailbox_t shared_mailbox __attribute__((section(".mailbox_sram8")));

/* ================================================================
 * Core 0 API (scheduler side)
 * ================================================================ */

/* Initialize mailbox to IDLE state */
static inline void mbox_init(void) {
    shared_mailbox.cmd = MBOX_IDLE;
    shared_mailbox.activity_sum = 0;
    shared_mailbox.golden_buf_addr = 0;
    shared_mailbox.golden_buf_len = 0;
    shared_mailbox.sequence_num = 0;
    shared_mailbox.encode_cycles = 0;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
}

/* Dispatch Core 1 to run golden path */
static inline void mbox_dispatch(uint8_t activity_sum, uint32_t seq) {
    shared_mailbox.activity_sum = activity_sum;
    shared_mailbox.sequence_num = seq;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    shared_mailbox.cmd = MBOX_DISPATCH;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    /* Wake Core 1 with SEV */
    __sev();
}

/* Check if Core 1 is done */
static inline bool mbox_is_done(void) {
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    return shared_mailbox.cmd == MBOX_DONE;
}

/* Abort Core 1 (timeout) */
static inline void mbox_abort(void) {
    shared_mailbox.cmd = MBOX_ABORT;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
}

/* ================================================================
 * Core 1 API (encoder side)
 * ================================================================ */

/* Wait for work (called in Core 1 idle loop) */
static inline void mbox_wait_for_dispatch(void) {
    while (shared_mailbox.cmd != MBOX_DISPATCH) {
        __wfe();
    }
}

/* Signal completion to Core 0 */
static inline void mbox_signal_done(uint32_t buf_addr, uint32_t buf_len,
                                     uint32_t cycles) {
    shared_mailbox.golden_buf_addr = buf_addr;
    shared_mailbox.golden_buf_len = buf_len;
    shared_mailbox.encode_cycles = cycles;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    shared_mailbox.cmd = MBOX_DONE;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    __sev();
}

/* Signal error to Core 0 */
static inline void mbox_signal_error(void) {
    shared_mailbox.cmd = MBOX_ERROR;
    __asm__ volatile ("fence iorw, iorw" ::: "memory");
    __sev();
}

/* Check if abort requested */
static inline bool mbox_should_abort(void) {
    return shared_mailbox.cmd == MBOX_ABORT;
}

#endif /* MAILBOX_H */
