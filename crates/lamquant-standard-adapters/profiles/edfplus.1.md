# EDF/EDF+/BDF mapping profile `edfplus.1`

- Promotes every non-annotation channel, including off-rate channels, as an
  exact integer signal with its own rational sample rate and exact affine
  physical calibration.
- Maps EDF+D record timekeeping annotations to piecewise time axes. Missing or
  malformed timekeeping evidence fails closed instead of being flattened.
- Promotes EDF+ TAL annotations to explicit events and an exact annotation
  payload. Annotation text, onset, and optional duration are retained.
- Preserves the local recording date/time and explicitly marks the absent EDF
  timezone rather than inventing UTC.
- Supports EDF, EDF+C, EDF+D, and signed 24-bit BDF samples.
- Binds the complete original file as an exact source capsule.
- Exact export restores that identity-bound capsule. A changed semantic root
  cannot authorize a stale source capsule.
- The first-class profile is independently checked with pyedflib across EDF+C,
  EDF+D, and BDF fixtures; the source capsule additionally preserves private or
  future header fields outside the standardized semantic surface.
