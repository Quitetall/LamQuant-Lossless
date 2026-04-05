# LamQuant Gen 7 Validation Harness

Automated stress profiles validate firmware before deployment.

## Stress Profiles

| Profile | Condition | Pass Criteria |
|---------|-----------|---------------|
| `baseline_nominal` | Normal EEG rhythms | PRD < 2.0% |
| `seizure_burst` | High-frequency ictal discharge | PRD < 2.5% |
| `emc_burst` | IEC 61000-4-4 transient injection | < 15% packet loss |
| `thermal_derating` | 85°C junction temp, clock throttle | Latency < 4.0ms |
| `flash_corruption` | Induced byte-flips in weight storage | Safe mode entry, no TX |
| `electrode_pop` | DC offset jump + drift | PRD < 40%, R ≥ 0.85 |

## Graduation Thresholds

A build ships when all of these hold:

- **Fidelity**: Pearson R > 0.85 on held-out patients (chb15–chb20)
- **Latency**: < 4.0ms per window (ADC complete → BLE TX start)
- **Stack**: < 2.4KB usage (`-Werror=stack-usage=2400`)
- **Parity**: 0.000 numerical drift between Python and C encoder output
- **Compression**: ≥ 5.0x ratio at R ≥ 0.85
- **Memory**: Encoder ≤ 43KB (SRAM4 budget)
