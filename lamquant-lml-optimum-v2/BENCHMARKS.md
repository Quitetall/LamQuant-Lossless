# Optimum-v2 peer cost log

## 2026-07-30 — generation-v4 production facade

- CPU: Intel Core i5-13600K, 14 cores / 20 threads
- Build: Cargo `bench` profile, optimized
- Workload: deterministic 21-channel × 2,560-sample, 16-bit EEG-shaped window
- Logical input: 215,040 bytes (canonical 32-bit samples)
- Harness: `benches/peer_codec.rs`

Decode command:

```text
cargo bench -p lamquant-lml-optimum-v2 --bench peer_codec -- decode \
  --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Criterion result:

```text
time:   [1.9980 s 2.0865 s 2.2272 s]
thrpt:  [94.291 KiB/s 100.65 KiB/s 105.10 KiB/s]
```

Encode command:

```text
cargo bench -p lamquant-lml-optimum-v2 --bench peer_codec -- \
  --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Criterion estimated 456.62 seconds for ten encode samples after warm-up. Run
was stopped before collection, so this is not an accepted encode measurement.
It is a blocking performance observation: exhaustive generation-v4 candidate
selection is ratio-first and not production-speed. P1 cost gate remains open
until a complete encode distribution is recorded and byte-preserving shared
analysis or proven candidate pruning reduces cost.
