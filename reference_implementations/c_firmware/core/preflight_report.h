#ifndef PREFLIGHT_REPORT_H
#define PREFLIGHT_REPORT_H

#include <stdint.h>
#include <stdbool.h>

void preflight_set_boot_results(bool kat_ok, bool crc_ok,
                                uint32_t expected, uint32_t computed);
void emit_preflight_report(void);

#endif /* PREFLIGHT_REPORT_H */
