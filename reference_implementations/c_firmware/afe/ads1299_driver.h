#ifndef ADS1299_DRIVER_H
#define ADS1299_DRIVER_H

#include <stdint.h>
#include <stdbool.h>

/*
 * ADS1299 8-Channel 24-Bit AFE Driver
 * ====================================
 * Production ADC path for LamQuant Gen 7.
 * Communicates via SPI0 at 4 MHz.
 *
 * Pin assignments:
 *   GPIO2 = SCK, GPIO3 = MOSI, GPIO4 = MISO
 *   GPIO5 = CS,  GPIO6 = DRDY, GPIO7 = RESET
 *
 * The ADS1299 has 8 physical channels. These map to the first 8
 * positions of raw_adc_buffer[21][2500]. Channels 8-20 are zero-filled.
 */

#define ADS1299_NUM_CHANNELS 8

// Maximum daisy-chain length: up to 32 ADS1299 chips (32 * 8 = 256 channels).
// Limited by available CS GPIO lines and SPI bandwidth.
#define ADS1299_MAX_CHIPS 32
#define ADS1299_MAX_CHANNELS (ADS1299_MAX_CHIPS * ADS1299_NUM_CHANNELS)

// Detect how many ADS1299 chips are present on the SPI bus.
// Probes each CS line and reads the device ID register (expected 0x3E).
// Returns the count of chips that responded with a valid ID.
// Called automatically by ads1299_init().
int ads1299_detect_daisy_chain(void);

// Get the total number of channels currently active in the system.
// Equals num_detected_chips * 8. Returns 0 if init has not run.
int ads1299_get_total_channels(void);

// Initialize SPI0 and configure ADS1299 registers.
// Must be called during main() boot sequence.
void ads1299_init(void);

// Start continuous data capture (RDATAC mode).
// DMA-paced reads on DRDY interrupt fill raw_adc_buffer.
void ads1299_start_continuous(void);

// Stop continuous data capture.
void ads1299_stop_continuous(void);

// Run lead-off impedance measurement.
// Results are stored in impedance_kohm[] (one per channel).
// Returns true if measurement completed successfully.
bool ads1299_measure_impedance(void);

// Get impedance result for a channel (in kOhm, 0 = good contact).
uint16_t ads1299_get_impedance_kohm(int channel);

#endif // ADS1299_DRIVER_H
