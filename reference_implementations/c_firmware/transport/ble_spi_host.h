#ifndef BLE_SPI_HOST_H
#define BLE_SPI_HOST_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    CODING_RATE_1_1 = 0,
    CODING_RATE_2_3 = 1,
    CODING_RATE_1_2 = 2,
} ChannelCoding;

extern volatile bool ble_initialized;

void adaptive_reduce_fsq_levels(void);
void adaptive_restore_fsq_levels(void);
bool fsq_is_reduced(void);
void ble_set_coding_rate(ChannelCoding rate);
ChannelCoding ble_get_coding_rate(void);
void on_emc_detected(int32_t current_psrr_measure_q10);
void ble_spi_emergency_flush(void);
void ble_enter_standby(void);
void ble_spi_init(void);
void trigger_ble_dma_tx(void);
void ble_spi_send_snn_alert(uint8_t level, uint32_t timestamp_ms);

#endif /* BLE_SPI_HOST_H */
