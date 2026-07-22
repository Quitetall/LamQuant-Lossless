# BIDS mapping profile `bids.1.11.1.single-edf-eeg`

- Requires a pinned BIDS 1.11.1 `dataset_description.json` and exactly one
  EDF/BDF EEG recording in a portable, duplicate-free dataset tree.
- Maps every decoded signal channel, channel order, regular sample rate, label,
  integer sample, and physical range into semantic ABIR.
- Preserves every dataset object as a source capsule. Sidecars remain explicitly
  quarantined until their semantics are promoted by a broader profile.
- Exact export is available only while the current ABIR interchange identity
  matches the identity bound at import; stale capsules cannot authorize a false
  semantic-equivalence claim.
- Multi-recording datasets, non-EDF/BDF recordings, and other BIDS modalities
  fail closed under this bounded profile.
