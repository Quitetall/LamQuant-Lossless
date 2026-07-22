# DICOM mapping profile `dicom.ps3.2026c.ecg-i16`

- Current semantic coverage is the validated 12-Lead ECG and General ECG
  Waveform Storage subset using signed 16-bit samples.
- Projects channel order, integer samples, declared sampling frequency, labels,
  and ECG modality; calibration/sensitivity and absolute timing remain in the
  complete DICOM source capsule.
- Patient, study, series, device, annotation, report, private-tag, sensitivity,
  and referenced-media promotion is not yet claimed.
- Unsupported Waveform IODs, sample interpretations, and incompatible multiplex
  groups fail closed.
- The bounded 12-lead fixture is prepared by
  `tools/generate_standard_adapter_fixtures.py` and passes `dciodvfy` with zero
  `Error -` diagnostics; warnings remain outside the bounded semantic claim.
- The broad DICOM PS3 2026c profile remains non-first-class until those mappings
  and independent `dciodvfy` evidence pass.
- This bounded profile also remains forensic/projected until validator receipts
  are consumed by the runtime status decision rather than documented only.
