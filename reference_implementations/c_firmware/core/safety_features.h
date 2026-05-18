#ifndef SAFETY_FEATURES_H
#define SAFETY_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

/* ── BLE Retry Buffer (1.9 KB) ──────────────────────────
 * Circular buffer of 8 compressed packets for retransmission
 * when BLE link drops. 80 seconds of dropout tolerance. */
#define BLE_RETRY_SLOTS 8
#define BLE_RETRY_PKT_SIZE 240

typedef struct {
    uint8_t packets[BLE_RETRY_SLOTS][BLE_RETRY_PKT_SIZE];
    uint16_t lengths[BLE_RETRY_SLOTS];    /* actual bytes per slot */
    uint32_t sequence[BLE_RETRY_SLOTS];   /* sequence number per slot */
    uint8_t head;                          /* next write slot */
    uint8_t count;                         /* slots with unACKed data */
    uint8_t tail;                          /* next retransmit slot */
    volatile bool ack_pending;
} ble_retry_buf_t;

/* ── Impedance Trend Monitor (3.4 KB) ──────────────────
 * Per-channel impedance history for predicting electrode failure.
 * 60 samples per channel (~10 min at 1 sample/10 sec). */
#define IMPEDANCE_CHANNELS 21
#define IMPEDANCE_HISTORY 60

typedef struct {
    uint16_t history[IMPEDANCE_CHANNELS][IMPEDANCE_HISTORY]; /* kohm */
    uint8_t write_idx;                /* circular index */
    uint8_t samples_filled;           /* 0..60 */
    int16_t slope_per_hour[IMPEDANCE_CHANNELS]; /* impedance trend */
    uint16_t current[IMPEDANCE_CHANNELS];       /* latest reading */
    uint8_t alert_flags;              /* bits: channels above threshold */
} impedance_monitor_t;

/* ── Pre-Ictal Ring Buffer (51.3 KB) ───────────────────
 * Circular buffer of 2 L3 approximation windows (20 sec lookback).
 * When SNN fires seizure, transmit this buffer at clinical quality. */
#define PREICTAL_WINDOWS 2
#define PREICTAL_CHANNELS 21
#define PREICTAL_SAMPLES 313       /* L3 approx length */

typedef struct {
    int32_t windows[PREICTAL_WINDOWS][PREICTAL_CHANNELS][PREICTAL_SAMPLES];
    uint32_t timestamps[PREICTAL_WINDOWS]; /* ms since boot */
    uint8_t write_idx;
    bool seizure_triggered;
    uint32_t trigger_time;
} preictal_buf_t;

/* ── Channel Quality Scores (0.2 KB) ──────────────────
 * Per-channel quality metric: 0=dead, 255=perfect. */
typedef struct {
    uint8_t quality[IMPEDANCE_CHANNELS];     /* 0-255 composite score */
    uint8_t artifact_count[IMPEDANCE_CHANNELS]; /* rolling artifact counter */
    uint16_t last_update_ms;
} channel_quality_t;

/* ── Event Log (1.6 KB) ───────────────────────────────
 * Circular log of 100 device events for FDA audit trail. */
#define EVENT_LOG_SIZE 100

typedef enum {
    EVT_BOOT_COLD = 0x01,
    EVT_BOOT_WARM = 0x02,
    EVT_WATCHDOG_RESET = 0x03,
    EVT_SEIZURE_DETECT = 0x10,
    EVT_SEIZURE_END = 0x11,
    EVT_BLE_DISCONNECT = 0x20,
    EVT_BLE_RECONNECT = 0x21,
    EVT_BLE_RETRY = 0x22,
    EVT_IMPEDANCE_ALERT = 0x30,
    EVT_CHANNEL_DEAD = 0x31,
    EVT_CHANNEL_RECOVERED = 0x32,
    EVT_BATTERY_LOW = 0x40,
    EVT_BATTERY_CRITICAL = 0x41,
    EVT_STACK_OVERFLOW = 0x50,
    EVT_CRC_FAIL = 0x51,
    EVT_MODE_SWITCH = 0x60,
    EVT_CORE1_TIMEOUT = 0x61,
} event_type_t;

typedef struct {
    uint32_t timestamp_ms;
    uint8_t type;          /* event_type_t */
    uint8_t channel;       /* 0xFF if not channel-specific */
    uint8_t data_hi;       /* event-specific */
    uint8_t data_lo;       /* event-specific */
    uint32_t sequence;     /* global sequence counter */
    uint32_t reserved;     /* alignment + future */
} event_entry_t;          /* 16 bytes */

typedef struct {
    event_entry_t entries[EVENT_LOG_SIZE];
    uint8_t write_idx;
    uint32_t total_events;
    uint32_t boot_count;
} event_log_t;

/* ── Seizure Timestamp Log (0.8 KB) ──────────────────
 * Compact seizure diary: onset + duration for 50 events. */
#define SEIZURE_LOG_SIZE 50

typedef struct {
    uint32_t onset_ms;     /* ms since boot */
    uint16_t duration_s;   /* seconds (0 = ongoing) */
    uint8_t severity;      /* SNN confidence: 0-255 */
    uint8_t channels;      /* bitmask of involved channels (low 8) */
} seizure_entry_t;         /* 8 bytes */

typedef struct {
    seizure_entry_t entries[SEIZURE_LOG_SIZE];
    uint8_t write_idx;
    uint8_t active_seizure; /* 0xFF = none, else index */
    uint16_t total_seizures;
} seizure_log_t;

/* ── Battery Monitor (0.1 KB) ────────────────────────
 * Voltage history for power source monitoring. */
#define BATTERY_HISTORY 60   /* 10 min at 1 sample/10 sec */

typedef struct {
    uint8_t voltage_pct[BATTERY_HISTORY]; /* 0-100% */
    uint8_t write_idx;
    uint8_t current_pct;
    uint16_t mv;                          /* current millivolts */
    bool low_warning;                     /* <20% */
    bool critical_warning;                /* <5% */
} battery_monitor_t;

/* ── Watchdog Fault Counter (0.1 KB) ─────────────────
 * Reliability metrics across warm reboots. */
typedef struct {
    uint32_t watchdog_resets;
    uint32_t core1_timeouts;
    uint32_t crc_failures;
    uint32_t stack_overflows;
    uint32_t ble_drops;
    uint32_t uptime_seconds;
    uint32_t total_windows_encoded;
    uint32_t total_seizures_detected;
} fault_counters_t;

/* ── Master Safety State ─────────────────────────────
 * Single struct for all safety features. Allocated in BSS. */
typedef struct {
    ble_retry_buf_t ble_retry;
    impedance_monitor_t impedance;
    preictal_buf_t preictal;
    channel_quality_t channel_quality;
    event_log_t event_log;
    seizure_log_t seizure_log;
    battery_monitor_t battery;
    fault_counters_t faults;
} safety_state_t;

/* Global instance (BSS, zeroed at boot) */
extern safety_state_t safety;

/* ── API ─────────────────────────────────────────── */

void safety_init(void);
void safety_log_event(event_type_t type, uint8_t channel, uint16_t data);
void safety_update_impedance(uint8_t channel, uint16_t kohm);
void safety_update_battery(uint16_t mv, uint8_t pct);
void safety_push_preictal(const int32_t l3_approx[][313], uint32_t timestamp);
void safety_on_seizure_start(uint8_t severity, uint8_t channel_mask);
void safety_on_seizure_end(void);
void safety_ble_push_packet(const uint8_t *data, uint16_t len, uint32_t seq);
bool safety_ble_has_retry(void);
const uint8_t *safety_ble_peek_retry(uint16_t *out_len, uint32_t *out_seq);
void safety_ble_ack_retry(void);
void safety_update_channel_quality(uint8_t channel, uint8_t quality);

#endif /* SAFETY_FEATURES_H */
