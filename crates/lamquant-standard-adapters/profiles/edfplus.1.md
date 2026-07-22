# EDF/EDF+/BDF mapping profile `edfplus.1.signal`

- Projects decoded channel order, integer sample values, regular sample rate,
  and labels into ABIR. Physical calibration, absolute recording origin, and
  EDF+ discontinuities are not promoted; EDF+D inputs fail closed.
- Binds the complete original file as an exact source capsule.
- Exact export currently restores that capsule; synthesized export from a
  transformed ABIR dataset is not yet claimed.
- EDF Annotations and off-rate/non-signal channels remain recoverable in the
  source capsule but are not yet promoted into ABIR temporal tables.
- Status remains forensic/projected. First-class status requires calibration
  and timing promotion plus independent EDFbrowser evidence in addition to the
  internal malformed-input and round-trip suite.
