# NWB mapping profile `nwb.2.10.0`

- Current semantic coverage is one acquisition TimeSeries containing
  little-endian integer data and a finite positive `starting_time.rate`.
- Maps integer samples, regular timing, channel order, and payload identity;
  preserves the complete HDF5/NWB file as an exact source capsule.
- Mixed-length or multiple acquisition series fail closed pending direct
  mixed-rate ABIR lowering.
- Electrode tables, intervals, behavior, stimulus, namespaces, external assets,
  and derived-data promotion are not yet claimed.
- The broad NWB 2.10.0 profile remains non-first-class until those mappings and
  independent NWB Inspector evidence pass.
