"""Shared test helpers — one place, no copy-paste.

AUDIT (2026-04-28): Created to eliminate duplicated helpers across 6+ test
files. Each helper was independently maintained in multiple files, leading
to divergent assertions, tolerance thresholds, and error messages.

Modules:
  edf_factory  — synthetic EDF/BDF file generator (was inline in test_lml_adversarial.py)
  roundtrip    — lossless roundtrip assertion with diagnostic failure messages
  signals      — canonical adversarial/synthetic signal generators
"""
