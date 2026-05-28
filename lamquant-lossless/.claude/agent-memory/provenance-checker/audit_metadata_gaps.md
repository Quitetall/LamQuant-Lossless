---
name: Metadata field coverage gaps between Python and Rust encode paths
description: Fields present in Python read_edf_digital metadata dict that are absent from Rust encode_edf_to_lml and CLI encode_one metadata JSON strings
type: project
---

Python read_edf_digital() produces these metadata fields (edf_to_lml.py lines 734-772):
  source_file, source_path, format, channels, n_channels, n_signals_total,
  sample_rate, bits_per_sample, n_data_records, record_duration,
  phys_min, phys_max, dig_min, dig_max, phys_dim,
  all_labels, all_ns_per_rec, all_phys_dims, transducers, prefilterings,
  eeg_channel_indices, annotation_channel_indices,
  duration_s, continuous, annotations, patient_id, recording_info, startdate,
  edf_header, non_eeg_channels, conversion_date,
  [conditional] trailing_data, trailing_data_size, signal_sha256

Rust encode_edf_to_lml() (lma.rs lines 324-374) produces:
  source_file, format, channels, n_channels, n_signals_total,
  sample_rate, n_data_records, record_duration,
  phys_min, phys_max, dig_min, dig_max, phys_dim,
  all_labels, all_ns_per_rec, eeg_channel_indices,
  duration_s, patient_id, recording_info, startdate,
  edf_header, non_eeg_channels,
  signal_sha256 (EMPTY STRING)

MISSING from Rust LMA path vs Python path:
  - source_path (full path, not just filename)
  - bits_per_sample (16 for EDF, 24 for BDF — BDF roundtrip critical)
  - all_phys_dims (per-channel physical dimensions)
  - transducers (per-channel transducer info from signal header field 1)
  - prefilterings (per-channel prefiltering info from signal header field 7)
  - annotation_channel_indices (needed for reconstruct_edf to correctly place non-EEG channels)
  - continuous (EDF+D discontinuity flag)
  - annotations (parsed TAL annotations)
  - conversion_date (audit trail)
  - trailing_data / trailing_data_size (bytes after last complete record)

CONSEQUENCE OF MISSING annotation_channel_indices:
  reconstruct_edf() in edf_to_lml.py line 302-304 falls back to:
    eeg_idx = [i for i in range(n_signals) if i not in meta.get('annotation_channel_indices', [])]
  If annotation_channel_indices is missing, it defaults to [] and all channels are treated as EEG.
  This is actually SAFE for reconstruction because eeg_channel_indices IS present — but the
  fallback code path is confusing and should be verified.

CONSEQUENCE OF MISSING trailing_data:
  Rust edf.rs does NOT capture trailing partial record bytes. The Python reader captures
  trailing_bytes only if the bulk read returns fewer bytes than expected (truncation case).
  Neither path captures extra bytes BEYOND the last complete record. If an EDF file has
  N+epsilon records (i.e., the file is longer than n_data_records * bytes_per_record),
  those extra bytes are silently discarded.

  The Rust reader: data_start + usable_records * bytes_per_rec is the cutoff. Anything
  beyond that in the file is not read. No trailing_data field in EdfFile struct.

**How to apply:** For 510(k), add trailing_data capture to Rust edf.rs. Add the missing
metadata fields to the Rust encode metadata strings. The most critical for bit-exact
roundtrip is trailing_data.
