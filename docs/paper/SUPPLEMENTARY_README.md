# Supplementary Material — LamQuant Lossless

Reproducibility package for *LamQuant Lossless: State-of-the-Art Lossless EEG Compression Designed for Microcontroller Deployment* (IEEE TBioCAS submission, May 2026).

Contents:

```
supplementary/
├── README.md                              this file
├── manuscript/
│   ├── lamquant_lossless.tex              source LaTeX
│   ├── lamquant_lossless.pdf              compiled PDF
│   └── references.bib                     BibTeX entries
├── tools/                                 bench + verify scripts (Python 3.10+)
│   ├── bench_gzip_baseline.sh             gzip -9 baseline on EDF tree
│   ├── bench_chbmit.py                    CHB-MIT 3-mode LPC comparator
│   ├── bench_tueg_subsets.py              TUEG per-montage CR (parallel)
│   ├── bench_per_file_cr.py               per-file CR distribution
│   ├── bench_shannon_entropy.py           empirical Shannon H_0(X) + H_0(ΔX)
│   ├── bench_edf_reader_parity.py         pyedflib + MNE cross-validation
│   └── verify_paper_claims.py             cross-checks every numerical
│                                          claim against the JSON evidence
│                                          (60/0 PASS at submission)
└── evidence/                              raw bench outputs (JSON)
    ├── gzip_baseline_000.json
    ├── chbmit_lpc_mode_compare.json
    ├── tueg_subset_breakdown_montage.json
    ├── per_file_cr_{chbmit_full, siena, sleep_edf, cap_sleep,
    │                eegmmidb, openneuro_ds004100_unique,
    │                tueg_edf000_full, tueg_sample,
    │                tuab, tuar, tuep, tuev, tusl, tusz}.json
    ├── shannon_entropy_{tuar, tueg, tuab, tuep, tuev, tusl, tusz,
    │                    chbmit, siena, sleep_edf, cap_sleep,
    │                    eegmmidb, openneuro_ds004100}.json
    ├── shannon_entropy_full_summary.json
    ├── corpus_inventory_v3.json
    ├── corpus_file_counts.json
    ├── edf_reader_parity.json
    ├── tueg_AAREADME_v2.0.1.txt           prior bench reference
    └── tueg_subset_breakdown_montage_v202.json  headline snapshot
```

## Reproducing the headline numbers

All commands below assume Python 3.10+ and a local mirror of the
listed corpora.

### TUEG v2.0.2 headline (2.287:1)

```
# Rsync v2.0.2 from upstream (auth required: TUH NEDC SSH key)
rsync -auvxLP --delete \
  nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg/v2.0.2/ \
  /path/to/local/tueg_v2.0.2/

# Per-montage CR (parallel, ~50 min on 20-core host)
python3 tools/bench_tueg_subsets.py \
  --tree /path/to/local/tueg_v2.0.2 \
  --group-by montage \
  --jobs 19

# Output → outputs/paper/tueg_subset_breakdown_montage.json
```

Expected aggregate: 70,831 files, 1,756,355,590,458 B raw → 768,043,519,030 B compressed, **CR 2.287:1**.

### CHB-MIT (2.723:1)

```
# Download from PhysioNet
wget -r -np https://physionet.org/files/chbmit/1.0.0/ -P /local/chbmit/

# Bench all 3 LPC modes
python3 tools/bench_chbmit.py --tree /local/chbmit/

# Adaptive mode wins: 686 files, 45.76 GB → 16.80 GB, CR 2.7229:1
```

### Per-file CR boxplot (Fig. 3)

```
# Six corpora used in Fig. 3:
python3 tools/bench_per_file_cr.py \
  --tree tueg_sample:/local/tueg_v2.0.2 \
  --tree chbmit:/local/chbmit \
  --tree tuev:/local/tuev_v2.0.1 \
  --tree sleep_edf:/local/sleep_edf \
  --tree eegmmidb:/local/eegmmidb \
  --tree openneuro_ds004100:/local/openneuro/ds004100 \
  --sample 200      # 200-file subsample for TUEG; --sample 0 for others
```

JSON output per corpus carries the IQR + min/max used for the boxplot whiskers.

### Shannon entropy ceilings (Table V + Appendix A)

```
python3 tools/bench_shannon_entropy.py \
  --auto-tuh \
  --tree chbmit:/local/chbmit \
  --tree siena:/local/siena \
  --tree sleep_edf:/local/sleep_edf \
  --tree cap_sleep:/local/cap_sleep \
  --tree eegmmidb:/local/eegmmidb \
  --tree openneuro_ds004100:/local/openneuro/ds004100 \
  --files-per-corpus 200

# Output → shannon_entropy_<corpus>.json + shannon_entropy_summary.json
```

13-corpus weighted aggregate over 56.27 × 10^9 samples: H_0(X) = 10.066, H_0(ΔX) = 7.659.

### Verification script

```
python3 tools/verify_paper_claims.py
# Expected: === 60 PASS / 0 FAIL ===
```

Cross-checks every numerical claim in the paper against the
`evidence/*.json` files. PASS-tally is a precondition for submission.

## RP2350 throughput — measurement provenance

The Table III "LamQuant (RP2350 encoder)" row reports the cycle
count **measured via RTL-level Verilator simulation** of the
official `Wren6991/Hazard3` v1.1 RTL (the same Verilog that taped
out as the RP2350 Hazard3 cores per the RP2350 datasheet).
Measured: **12,548,067 cycles/window** $\Rightarrow$
**83.66 ms/window @ 150 MHz** $\Rightarrow$ **0.627 Msa/s** =
**119× real-time** for a 21 ch × 250 Hz input stream
(CPI 1.071, IPC 0.93 — within 7 % of the 1.0 ceiling for a
single-issue 3-stage in-order pipeline).

The bench harness, runbook, and evidence JSON live in the public
repository — not in this supplementary zip, to avoid dual
maintenance and stay under the TBioCAS supplementary size budget:

- **Bench harness**: `tools/hazard3_bench/src/bench_encode.rs` —
  a bare-metal `riscv32imac-unknown-none-elf` binary that boots,
  builds a 21 ch × 2500 sample deterministic xorshift window,
  runs `PipelineScheduler::encode_window` eight times under a
  64-bit `mcycle` + `minstret` bracket, and writes results to
  the testbench print MMIO at `0xC000_0000`. Repository:
  <https://github.com/Quitetall/LamQuant/tree/main/tools/hazard3_bench>.
- **Runbook**: `tools/bench_rp2350_silicon.md` documents three
  identical-cycle paths:
    1. Verilator on the official Hazard3 RTL (the path used for
       Table III). Cycle-perfect against the silicon-equivalent
       Verilog.
    2. CXXRTL on the same RTL (Yosys backend, slower compile,
       same cycle counts).
    3. Real Pico 2 board over probe-rs / SWD RTT — pending
       confirmation, expected to match Verilator within ±1 cycle.
  Repository: <https://github.com/Quitetall/LamQuant/blob/main/tools/bench_rp2350_silicon.md>.
- **Evidence**: `outputs/paper/rp2350_silicon_bench.json` in the
  repo captures the full bench configuration, all CSR readouts,
  and the provenance chain back to the RTL source.

A reviewer who wants to reproduce the cycle count can clone the
repo, run `cd tools/hazard3_bench && cargo build --release`,
then load the produced ELF into any of the three test environments
above. The Verilator path takes ≈20 s to build the testbench (with
`VK_PCH_I_FAST= VK_PCH_I_SLOW=` to bypass a GCC 16 PCH bug — see
runbook) and ≈26 s of wall clock to run the 129 M-cycle sim.

## Software environment

- Python: 3.10+
- pyedflib: 0.1.36+
- mne-python: 1.6+
- numpy: 1.24+
- rsync: 3.x
- LamQuant Lossless: built via `cargo build --release --bin lml`
  from <https://github.com/Quitetall/LamQuant> at commit `34349e8`
  or later.

## License + Patent

Source code: permissive open-source (license file in repository).
Patent disclosure: Patent Pending US #64/032,641 (commercial
implementation rights only; academic and derivative research are
unaffected).

## Contact

Brian Lam, briankhanglam@usf.edu
ORCID: 0009-0001-5463-2324
OpenHuman Technologies LLC, Florida, USA
