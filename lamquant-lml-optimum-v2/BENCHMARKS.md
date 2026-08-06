# Optimum-v2 peer cost log

## 2026-08-06 — P1 shared-analysis and bounded parallel portfolio

- CPU: Intel Core i5-13600K, 14 cores / 20 threads
- Build: Cargo `bench` profile, optimized, `parallel` feature
- Rayon bound: at most 49 complete candidate tasks; run used 20 threads
- Workload: deterministic 21-channel × 2,560-sample, 16-bit EEG-shaped window
- Window duration: 10 seconds at 256 Hz
- Logical input: 215,040 bytes (canonical 32-bit samples)
- Harness: `benches/peer_codec.rs`
- Frozen facade packet SHA-256: `edd9427620badeee8b239a90f4800fb772a4af3826499605284eb07b04349377`

Command:

```text
RAYON_NUM_THREADS=20 cargo bench -p lamquant-lml-optimum-v2 \
  --features parallel --bench peer_codec -- \
  --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Accepted Criterion distributions:

```text
encode time:   [15.552 s 15.737 s 15.950 s]
encode thrpt:  [13.166 KiB/s 13.345 KiB/s 13.503 KiB/s]
decode time:   [2.1569 s 2.1760 s 2.1953 s]
decode thrpt:  [95.658 KiB/s 96.506 KiB/s 97.363 KiB/s]
```

Shared analysis alone measured an accepted sequential encode median of
22.642 seconds. Ordered parallel candidate evaluation lowers this to 15.737
seconds, a 30.5% reduction. Against the previous 45.662-second-per-window
Criterion estimate, P1 is 65.5% faster. Encode real-time factor remains 1.574
for this 10-second window; production operation is therefore ratio-first batch
or buffered host/BLUT execution, not live real-time encoding.

Peak resident memory was sampled from `/proc` through `ps` every 100 ms while
the Criterion executable ran its encode test. Peak process RSS was 86,436 KiB
(84.4 MiB). Exit status was zero. This is host evidence, not a portable memory
guarantee; static codec input bounds remain 256 channels, 32,768 samples per
channel, and 131,072 total values.

Generation-v4 bytes did not change. Facade equivalence, frozen SHA-256, and the
independent candidate-path characterization pass with and without `parallel`.

## 2026-07-30 — pre-optimization production facade

Same CPU and workload. Decode median was 2.0865 seconds. Criterion estimated
456.62 seconds for ten encode samples after warm-up; collection was stopped.
That estimate is retained only as the pre-optimization cost observation, not
as an accepted distribution.
