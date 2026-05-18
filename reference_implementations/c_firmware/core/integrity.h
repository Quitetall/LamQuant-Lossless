#ifndef INTEGRITY_H
#define INTEGRITY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len);
bool verify_firmware_crc(void);

#endif /* INTEGRITY_H */
