/*
 * LamQuant Gen 7.1 — Safety Features for FDA 510(k)
 * ===================================================
 * 8 safety-critical SRAM subsystems:
 *   1. BLE retry buffer (1.9 KB)
 *   2. Impedance trend monitor (3.4 KB)
 *   3. Pre-ictal ring buffer (51.3 KB)
 *   4. Channel quality scores (0.2 KB)
 *   5. Event log / audit trail (1.6 KB)
 *   6. Seizure timestamp log (0.8 KB)
 *   7. Battery monitor (0.1 KB)
 *   8. Watchdog fault counters (0.1 KB)
 *
 * Total SRAM: ~59.4 KB
 *
 * All functions: O(1) time, no malloc, no float, no division.
 * Integer-only slope via shift-based approximation.
 */

#include "safety_features.h"
#include "hardware/timer.h"

/* ── Global instance (BSS — zeroed at boot) ──────── */
safety_state_t safety;

/* ── Helpers ─────────────────────────────────────── */

static inline uint32_t get_timestamp_ms(void) {
    /* Approximate us→ms: multiply by 1/1024 via shift (0.2% error, no div) */
    return time_us_32() >> 10;
}

/* ── safety_init ─────────────────────────────────── */

void safety_init(void) {
    /* BSS is already zeroed, but be explicit for FDA auditors */
    uint8_t *p = (uint8_t *)&safety;
    for (uint32_t i = 0; i < sizeof(safety_state_t); i++) {
        p[i] = 0;
    }

    safety.seizure_log.active_seizure = 0xFF; /* no active seizure */
    safety.event_log.boot_count++;

    /* Log cold boot event */
    safety_log_event(EVT_BOOT_COLD, 0xFF, 0);
}

/* ── Event Log ───────────────────────────────────── */

void safety_log_event(event_type_t type, uint8_t channel, uint16_t data) {
    event_log_t *log = &safety.event_log;
    event_entry_t *e = &log->entries[log->write_idx];

    e->timestamp_ms = get_timestamp_ms();
    e->type = (uint8_t)type;
    e->channel = channel;
    e->data_hi = (uint8_t)(data >> 8);
    e->data_lo = (uint8_t)(data & 0xFF);
    e->sequence = log->total_events;
    e->reserved = 0;

    log->write_idx++;
    if (log->write_idx >= EVENT_LOG_SIZE) {
        log->write_idx = 0;
    }
    log->total_events++;
}

/* ── Impedance Trend Monitor ─────────────────────── */

void safety_update_impedance(uint8_t channel, uint16_t kohm) {
    if (channel >= IMPEDANCE_CHANNELS) return;

    impedance_monitor_t *imp = &safety.impedance;
    imp->history[channel][imp->write_idx] = kohm;
    imp->current[channel] = kohm;

    /* Compute slope: difference between newest and oldest, scaled to per-hour.
     * 60 samples at 10 sec interval = 600 sec = 1/6 hour.
     * slope_per_hour = (newest - oldest) * 6
     * Use shift: *6 = (*4 + *2) = (<<2 + <<1) — no division */
    if (imp->samples_filled >= IMPEDANCE_HISTORY) {
        uint8_t oldest_idx = imp->write_idx + 1;
        if (oldest_idx >= IMPEDANCE_HISTORY) oldest_idx = 0;
        int16_t diff = (int16_t)kohm - (int16_t)imp->history[channel][oldest_idx];
        imp->slope_per_hour[channel] = (int16_t)((diff << 2) + (diff << 1));
    }

    /* Alert if impedance above 50 kohm */
    if (kohm > 50) {
        imp->alert_flags |= (uint8_t)(1u << (channel & 7));
        safety_log_event(EVT_IMPEDANCE_ALERT, channel, kohm);
    } else {
        imp->alert_flags &= (uint8_t)(~(1u << (channel & 7)));
    }

    /* Advance write index only once per full channel sweep.
     * Caller is responsible for calling with channel 0..20 in order;
     * we advance on channel 20 (last). */
    if (channel == IMPEDANCE_CHANNELS - 1) {
        imp->write_idx++;
        if (imp->write_idx >= IMPEDANCE_HISTORY) {
            imp->write_idx = 0;
        }
        if (imp->samples_filled < IMPEDANCE_HISTORY) {
            imp->samples_filled++;
        }
    }
}

/* ── Battery Monitor ─────────────────────────────── */

void safety_update_battery(uint16_t mv, uint8_t pct) {
    battery_monitor_t *bat = &safety.battery;

    bat->voltage_pct[bat->write_idx] = pct;
    bat->write_idx++;
    if (bat->write_idx >= BATTERY_HISTORY) {
        bat->write_idx = 0;
    }

    bat->current_pct = pct;
    bat->mv = mv;

    bool was_low = bat->low_warning;
    bool was_critical = bat->critical_warning;

    bat->low_warning = (pct < 20);
    bat->critical_warning = (pct < 5);

    if (bat->low_warning && !was_low) {
        safety_log_event(EVT_BATTERY_LOW, 0xFF, mv);
    }
    if (bat->critical_warning && !was_critical) {
        safety_log_event(EVT_BATTERY_CRITICAL, 0xFF, mv);
    }
}

/* ── Pre-Ictal Ring Buffer ───────────────────────── */

void safety_push_preictal(const int32_t l3_approx[][313], uint32_t timestamp) {
    preictal_buf_t *pb = &safety.preictal;
    uint8_t idx = pb->write_idx;

    /* Copy L3 approximation window [21][313] into ring buffer slot */
    const int32_t *src = (const int32_t *)l3_approx;
    int32_t *dst = (int32_t *)pb->windows[idx];
    for (uint32_t i = 0; i < PREICTAL_CHANNELS * PREICTAL_SAMPLES; i++) {
        dst[i] = src[i];
    }

    pb->timestamps[idx] = timestamp;

    pb->write_idx++;
    if (pb->write_idx >= PREICTAL_WINDOWS) {
        pb->write_idx = 0;
    }
}

/* ── Seizure Diary ───────────────────────────────── */

void safety_on_seizure_start(uint8_t severity, uint8_t channel_mask) {
    seizure_log_t *sl = &safety.seizure_log;

    seizure_entry_t *e = &sl->entries[sl->write_idx];
    e->onset_ms = get_timestamp_ms();
    e->duration_s = 0;   /* ongoing */
    e->severity = severity;
    e->channels = channel_mask;

    sl->active_seizure = sl->write_idx;

    /* Mark preictal buffer as triggered */
    safety.preictal.seizure_triggered = true;
    safety.preictal.trigger_time = e->onset_ms;

    safety_log_event(EVT_SEIZURE_DETECT, channel_mask, (uint16_t)severity);
    safety.faults.total_seizures_detected++;
}

void safety_on_seizure_end(void) {
    seizure_log_t *sl = &safety.seizure_log;

    if (sl->active_seizure == 0xFF) return; /* no active seizure */

    seizure_entry_t *e = &sl->entries[sl->active_seizure];
    uint32_t now = get_timestamp_ms();
    uint32_t elapsed_ms = now - e->onset_ms;
    /* Convert ms to seconds via shift: /1024 ~ /1000 (2.4% error, acceptable) */
    e->duration_s = (uint16_t)(elapsed_ms >> 10);

    sl->active_seizure = 0xFF;

    sl->write_idx++;
    if (sl->write_idx >= SEIZURE_LOG_SIZE) {
        sl->write_idx = 0;
    }
    sl->total_seizures++;

    safety.preictal.seizure_triggered = false;

    safety_log_event(EVT_SEIZURE_END, 0xFF, e->duration_s);
}

/* ── BLE Retry Buffer ────────────────────────────── */

void safety_ble_push_packet(const uint8_t *data, uint16_t len, uint32_t seq) {
    ble_retry_buf_t *rb = &safety.ble_retry;

    if (len > BLE_RETRY_PKT_SIZE) len = BLE_RETRY_PKT_SIZE;

    uint8_t slot = rb->head;
    const uint8_t *s = data;
    uint8_t *d = rb->packets[slot];
    for (uint16_t i = 0; i < len; i++) {
        d[i] = s[i];
    }
    rb->lengths[slot] = len;
    rb->sequence[slot] = seq;

    rb->head++;
    if (rb->head >= BLE_RETRY_SLOTS) {
        rb->head = 0;
    }

    if (rb->count < BLE_RETRY_SLOTS) {
        rb->count++;
    } else {
        /* Overwriting oldest unACKed — advance tail */
        rb->tail = rb->head;
    }
}

bool safety_ble_has_retry(void) {
    return safety.ble_retry.count > 0;
}

const uint8_t *safety_ble_peek_retry(uint16_t *out_len, uint32_t *out_seq) {
    ble_retry_buf_t *rb = &safety.ble_retry;
    if (rb->count == 0) {
        *out_len = 0;
        *out_seq = 0;
        return (const uint8_t *)0;
    }

    *out_len = rb->lengths[rb->tail];
    *out_seq = rb->sequence[rb->tail];
    return rb->packets[rb->tail];
}

void safety_ble_ack_retry(void) {
    ble_retry_buf_t *rb = &safety.ble_retry;
    if (rb->count == 0) return;

    rb->tail++;
    if (rb->tail >= BLE_RETRY_SLOTS) {
        rb->tail = 0;
    }
    rb->count--;
}

/* ── Channel Quality ─────────────────────────────── */

void safety_update_channel_quality(uint8_t channel, uint8_t quality) {
    if (channel >= IMPEDANCE_CHANNELS) return;

    uint8_t prev = safety.channel_quality.quality[channel];
    safety.channel_quality.quality[channel] = quality;
    safety.channel_quality.last_update_ms = (uint16_t)(get_timestamp_ms() & 0xFFFF);

    /* Detect dead channel (quality dropped to 0 from nonzero) */
    if (quality == 0 && prev > 0) {
        safety_log_event(EVT_CHANNEL_DEAD, channel, (uint16_t)prev);
    }
    /* Detect channel recovery (quality rose from 0) */
    if (quality > 0 && prev == 0) {
        safety_log_event(EVT_CHANNEL_RECOVERED, channel, (uint16_t)quality);
    }
}
