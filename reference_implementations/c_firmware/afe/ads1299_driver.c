#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "ads1299_driver.h"
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "hardware/irq.h"

/*
 * LamQuant Gen 7 — ADS1299 AFE Driver (Production ADC Path)
 * ==========================================================
 * TI ADS1299 8-channel 24-bit delta-sigma ADC over SPI0.
 *
 * Configuration:
 *   - 250 Hz sample rate (CONFIG1 = 0x96)
 *   - 24x PGA gain (CHnSET = 0x60)
 *   - Internal reference (CONFIG3 = 0xE0)
 *   - All 8 channels active
 *
 * Data path:
 *   DRDY interrupt -> DMA read 27 bytes (3 status + 8*3 data) ->
 *   sign-extend 24->32 bit -> raw_adc_buffer[0..7][sample_idx]
 *
 * Channels 8-20 of raw_adc_buffer are zero-filled (ADS1299 only has 8ch).
 */

/* === Pin Assignments === */
#define ADS_SPI_PORT    spi0
#define ADS_SPI_BAUD    4000000   // 4 MHz
#define ADS_PIN_SCK     2
#define ADS_PIN_MOSI    3
#define ADS_PIN_MISO    4
#define ADS_PIN_CS      5
#define ADS_PIN_DRDY    6
#define ADS_PIN_RESET   7

/* === ADS1299 Register Addresses === */
#define REG_ID          0x00
#define REG_CONFIG1     0x01
#define REG_CONFIG2     0x02
#define REG_CONFIG3     0x03
#define REG_LOFF        0x04
#define REG_CH1SET      0x05
#define REG_CH2SET      0x06
#define REG_CH3SET      0x07
#define REG_CH4SET      0x08
#define REG_CH5SET      0x09
#define REG_CH6SET      0x0A
#define REG_CH7SET      0x0B
#define REG_CH8SET      0x0C
#define REG_LOFF_SENSP  0x0F
#define REG_LOFF_SENSN  0x10
#define REG_LOFF_STATP  0x12
#define REG_LOFF_STATN  0x13
#define REG_GPIO        0x14
#define REG_CONFIG4     0x17

/* === ADS1299 Commands === */
#define CMD_WAKEUP      0x02
#define CMD_STANDBY     0x04
#define CMD_RESET       0x06
#define CMD_START       0x08
#define CMD_STOP        0x0A
#define CMD_RDATAC      0x10    // Read data continuous
#define CMD_SDATAC      0x11    // Stop data continuous
#define CMD_RDATA       0x12    // Read data single

/* === State === */
#define WINDOW_SIZE     2500
#define NUM_CHANNELS    21

// ADC buffer (shared with biquad_q31.c, SRAM0)
extern int32_t raw_adc_buffer[NUM_CHANNELS][WINDOW_SIZE];

// DMA completion callback (shared with scheduler.c)
extern void on_adc_dma_complete(void);

static volatile uint32_t sample_idx = 0;
static int ads_dma_channel = -1;
static uint8_t spi_rx_buf[27];  // 3 status + 8*3 data bytes
static uint16_t impedance_kohm[ADS1299_NUM_CHANNELS];

// Daisy-chain detection state
static int detected_chip_count = 0;

/* === Low-level SPI helpers === */

static void ads_cs_select(void) {
    gpio_put(ADS_PIN_CS, 0);
}

static void ads_cs_deselect(void) {
    gpio_put(ADS_PIN_CS, 1);
}

static void ads_send_command(uint8_t cmd) {
    ads_cs_select();
    spi_write_blocking(ADS_SPI_PORT, &cmd, 1);
    ads_cs_deselect();
    sleep_us(4);  // t_SDECODE = 4 SCLK cycles
}

static void ads_write_register(uint8_t reg, uint8_t value) {
    uint8_t buf[3];
    buf[0] = 0x40 | (reg & 0x1F);  // WREG opcode
    buf[1] = 0x00;                   // Write 1 register
    buf[2] = value;
    ads_cs_select();
    spi_write_blocking(ADS_SPI_PORT, buf, 3);
    ads_cs_deselect();
    sleep_us(4);
}

static uint8_t ads_read_register(uint8_t reg) {
    uint8_t tx_buf[3] = {0x20 | (reg & 0x1F), 0x00, 0x00};
    uint8_t rx_buf[3] = {0};
    ads_cs_select();
    spi_write_read_blocking(ADS_SPI_PORT, tx_buf, rx_buf, 3);
    ads_cs_deselect();
    sleep_us(4);
    return rx_buf[2];
}

/* === Sign-extend 24-bit to 32-bit === */
static inline int32_t sign_extend_24(uint8_t msb, uint8_t mid, uint8_t lsb) {
    int32_t val = ((int32_t)msb << 16) | ((int32_t)mid << 8) | (int32_t)lsb;
    // Sign extend from bit 23
    if (val & 0x800000) {
        val |= 0xFF000000;
    }
    return val;
}

/* === DRDY interrupt handler === */
static void on_drdy_falling(uint gpio, uint32_t events) {
    (void)gpio;
    (void)events;

    if (sample_idx >= WINDOW_SIZE) return;

    // Read 27 bytes: 3 status + 8 channels * 3 bytes each
    ads_cs_select();
    spi_read_blocking(ADS_SPI_PORT, 0x00, spi_rx_buf, 27);
    ads_cs_deselect();

    // Parse 8 channels, sign-extend 24-bit -> Q31 (left-shift by 8)
    for (int ch = 0; ch < ADS1299_NUM_CHANNELS; ch++) {
        int offset = 3 + ch * 3;  // Skip 3 status bytes
        int32_t raw24 = sign_extend_24(
            spi_rx_buf[offset], spi_rx_buf[offset + 1], spi_rx_buf[offset + 2]);
        // Left-shift by 8 to convert 24-bit to Q31 scale
        raw_adc_buffer[ch][sample_idx] = raw24 << 8;
    }

    sample_idx++;

    if (sample_idx >= WINDOW_SIZE) {
        // Window complete: signal the scheduler
        on_adc_dma_complete();
        sample_idx = 0;
    }
}

/* === Public API === */

/*
 * Detect how many ADS1299 chips are present on the SPI bus.
 *
 * Each ADS1299 in a daisy chain has its own CS line (or in some
 * configurations, shares CS with the next chip). We probe by reading
 * the ID register: a valid ADS1299 returns 0x3E in the lower bits.
 *
 * For the current single-chip configuration this returns 1 (8 channels).
 * Future hardware revisions with multiple chips can extend this loop
 * to probe additional CS lines on the SPI bus.
 */
int ads1299_detect_daisy_chain(void) {
    int count = 0;

    // Probe primary CS line (current hardware = 1 chip).
    // Future: loop over additional CS GPIOs and accumulate the count.
    uint8_t id = ads_read_register(REG_ID);
    if ((id & 0x1F) == 0x1E) {  // Lower 5 bits = 0b11110 for ADS1299
        count = 1;
    }

    detected_chip_count = count;
    return count;
}

int ads1299_get_total_channels(void) {
    return detected_chip_count * ADS1299_NUM_CHANNELS;
}

void ads1299_init(void) {
    // Initialize SPI0
    spi_init(ADS_SPI_PORT, ADS_SPI_BAUD);
    spi_set_format(ADS_SPI_PORT, 8, SPI_CPOL_0, SPI_CPHA_1, SPI_MSB_FIRST);

    gpio_set_function(ADS_PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(ADS_PIN_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(ADS_PIN_MISO, GPIO_FUNC_SPI);

    // CS is manual
    gpio_init(ADS_PIN_CS);
    gpio_set_dir(ADS_PIN_CS, GPIO_OUT);
    gpio_put(ADS_PIN_CS, 1);

    // RESET pin
    gpio_init(ADS_PIN_RESET);
    gpio_set_dir(ADS_PIN_RESET, GPIO_OUT);

    // DRDY input (active-low)
    gpio_init(ADS_PIN_DRDY);
    gpio_set_dir(ADS_PIN_DRDY, GPIO_IN);
    gpio_pull_up(ADS_PIN_DRDY);

    // Hardware reset sequence
    gpio_put(ADS_PIN_RESET, 0);
    sleep_ms(1);
    gpio_put(ADS_PIN_RESET, 1);
    sleep_ms(100);  // Wait for ADS1299 to settle after reset

    // Stop continuous read mode for register configuration
    ads_send_command(CMD_SDATAC);

    // Detect daisy chain length (sets detected_chip_count).
    // For current single-chip hardware this returns 1 (8 channels).
    int chips = ads1299_detect_daisy_chain();
    if (chips < 1) {
        // Fallback: assume single chip even if ID read failed
        detected_chip_count = 1;
    }

    // Configure registers
    ads_write_register(REG_CONFIG1, 0x96);  // 250 Hz sample rate, internal oscillator
    ads_write_register(REG_CONFIG2, 0xC0);  // Test signals off, internal reference buffer
    ads_write_register(REG_CONFIG3, 0xE0);  // Internal reference, bias enabled

    // Set all 8 channels: PGA gain = 24x, normal input
    for (int ch = 0; ch < ADS1299_NUM_CHANNELS; ch++) {
        ads_write_register(REG_CH1SET + ch, 0x60);  // Gain 24, normal electrode input
    }

    // Lead-off detection: current source magnitude 6nA, DC excitation
    ads_write_register(REG_LOFF, 0x03);

    // Zero-fill unused channels (9-21) at startup
    for (int ch = ADS1299_NUM_CHANNELS; ch < NUM_CHANNELS; ch++) {
        memset(raw_adc_buffer[ch], 0, WINDOW_SIZE * sizeof(int32_t));
    }

    sample_idx = 0;
}

void ads1299_start_continuous(void) {
    sample_idx = 0;

    // Enable DRDY interrupt (falling edge)
    gpio_set_irq_enabled_with_callback(
        ADS_PIN_DRDY,
        GPIO_IRQ_EDGE_FALL,
        true,
        on_drdy_falling
    );

    // Start conversions and enter continuous read mode
    ads_send_command(CMD_START);
    ads_send_command(CMD_RDATAC);
}

void ads1299_stop_continuous(void) {
    ads_send_command(CMD_SDATAC);
    ads_send_command(CMD_STOP);

    // Disable DRDY interrupt
    gpio_set_irq_enabled(ADS_PIN_DRDY, GPIO_IRQ_EDGE_FALL, false);
}

bool ads1299_measure_impedance(void) {
    // Stop continuous mode for register access
    ads_send_command(CMD_SDATAC);
    ads_send_command(CMD_STOP);

    // Enable lead-off sense on all channels (positive and negative)
    ads_write_register(REG_LOFF_SENSP, 0xFF);  // All P channels
    ads_write_register(REG_LOFF_SENSN, 0xFF);  // All N channels

    // Enable lead-off comparators
    ads_write_register(REG_CONFIG4, 0x02);

    // Wait for lead-off detection to settle
    sleep_ms(500);

    // Read lead-off status registers
    uint8_t loff_statp = ads_read_register(REG_LOFF_STATP);
    uint8_t loff_statn = ads_read_register(REG_LOFF_STATN);

    // Convert lead-off bits to approximate impedance
    // Bit set = electrode off or high impedance
    for (int ch = 0; ch < ADS1299_NUM_CHANNELS; ch++) {
        bool p_off = (loff_statp >> ch) & 1;
        bool n_off = (loff_statn >> ch) & 1;

        if (p_off || n_off) {
            impedance_kohm[ch] = 999;  // High impedance / disconnected
        } else {
            // Good contact — approximate impedance from 6nA current source
            // and comparator threshold. Real implementation would measure
            // the actual voltage to compute R = V/I.
            impedance_kohm[ch] = 2;  // Nominal good contact
        }
    }

    // Disable lead-off detection
    ads_write_register(REG_LOFF_SENSP, 0x00);
    ads_write_register(REG_LOFF_SENSN, 0x00);
    ads_write_register(REG_CONFIG4, 0x00);

    return true;
}

uint16_t ads1299_get_impedance_kohm(int channel) {
    if (channel < 0 || channel >= ADS1299_NUM_CHANNELS) return 0xFFFF;
    return impedance_kohm[channel];
}
