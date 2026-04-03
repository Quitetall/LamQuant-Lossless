# LamQuant Continuous Integration Validation Harness

Before deployment, LamQuant validates embedded behaviors against hardware and clinical metrics using automated stress profiles.

## The Stress Profile Architecture
Our automated CI suite mimics extreme environments to evaluate firmware robustness:

1. **`baseline_nominal`**: Normal baseline EEG rhythms. Target: **PRD < 2.0%**.
2. **`seizure_burst`**: High-frequency seizure oscillations and voltage excursions. Target: **PRD < 2.5%**.
3. **`emc_burst`**: Synthetic IEC 61000-4-4 transient bursts simulated against the BLE transmission stack. Target: Signal recovery with **< 15.0%** total packet loss.
4. **`thermal_derating`**: CPU clock throttling simulation across an 85°C thermal range. Target: Inference latency must remain beneath the **4.0ms** real-time deadline.
5. **`flash_corruption`**: Induced byte-flips to test CRC32 integrity anchors. Target: Graceful failure trapping and mandatory entry into `Safe Mode`.

## Graduation Thresholds (Clinical Signoff)
A `lamquant_gen6` build must satisfy the following technical constraints for deployment:
- **Clinical Readiness**: Pearson Correlation **$R > 0.85$** across all held-out clinical datasets.
- **End-to-End Latency**: **< 300ms** total pipeline lag (ADC capture to encrypted BLE payload).
- **Stack Depth**: **< 2.4KB** total usage, verified by stack canary analysis.
- **Parity Proof**: **0.000** numerical drift between Python simulation and C-header integer exports.
