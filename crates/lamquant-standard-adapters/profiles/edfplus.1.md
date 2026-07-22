# EDF/EDF+/BDF mapping profile `edfplus.1.signal`

- Maps every decoded signal channel, channel order, sample value, regular sample
  rate, label, and physical min/max into semantic ABIR.
- Binds the complete original file as an exact source capsule.
- Exact export currently restores that capsule; synthesized export from a
  transformed ABIR dataset is not yet claimed.
- EDF Annotations and off-rate/non-signal channels remain recoverable in the
  source capsule but are not yet promoted into ABIR temporal tables.
- First-class status requires independent EDFbrowser evidence in addition to
  the internal malformed-input and round-trip suite.
