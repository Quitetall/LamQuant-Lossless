# Cover Letter — IEEE TBioCAS Submission

**Date:** May 26, 2026

**To:** Editor-in-Chief
IEEE Transactions on Biomedical Circuits and Systems

**Re:** Submission of manuscript "LamQuant Lossless: State-of-the-Art Lossless EEG Compression Designed for Microcontroller Deployment"

---

Dear Editor,

I am pleased to submit the enclosed manuscript, *LamQuant Lossless: State-of-the-Art Lossless EEG Compression Designed for Microcontroller Deployment*, for consideration as a regular paper in IEEE Transactions on Biomedical Circuits and Systems.

## Why TBioCAS

The work bridges biomedical signal processing, embedded systems implementation, and clinical archival — a triple intersection that TBioCAS is uniquely positioned to evaluate. The codec is built specifically for deployment on a 150 MHz Hazard3 RISC-V microcontroller (RP2350) operating within 280 KB of SRAM, while remaining byte-identical to a host-side implementation. The paper reports the first lossless EEG compression benchmark conducted at the full scale of the Temple University EEG Corpus (1.76 TB, 70,831 files), validated bit-exact across 88,147 encode/decode operations spanning thirteen clinical and research EEG datasets.

## Key contributions

1. **State-of-the-art compression ratio**: 2.72:1 on CHB-MIT (15.9% improvement over Chen *et al.* 2018), 2.287:1 on the full TUEG v2.0.2 corpus. Per-file CR up to 3.573:1 on intracranial EEG (OpenNeuro ds004100).
2. **First full-corpus TUEG benchmark**: 70,831 files (v2.0.2 AAREADME canonical), 1.756 TB raw → 768 GB compressed, byte-exact reconstruction verified end-to-end.
3. **Microcontroller deployability**: integer-only Le Gall 5/3 lifting DWT + per-subband adaptive LPC (orders 3, 3, 6, 8) + Golomb-Rice coder, all running byte-identically on x86, ARM, and RISC-V backends.
4. **Empirical Shannon-entropy ceilings on 13 corpora** (52.6 × 10^9 samples): aggregate H₀(X) = 10.07, H₀(ΔX) = 7.59 bits/sample. LamQuant beats both first-order ceilings (CR_raw 1.59:1, CR_diff 2.09:1) → effective residual entropy ≤ 7.0 bits/sample after the lifting + LPC pipeline.
5. **Methodological contribution**: "Why Benchmark Scale Matters" — per-file CR distributions show that single-file results reach 6.9:1 while corpus aggregates settle at 2.287:1, formalising why headline numbers from small-corpus benchmarks should be discounted.
6. **Commutative-monoid wire format** enabling embarrassingly parallel encoding, append-without-re-encode, and local fault containment — properties the byte-equality conformance gate enforces across host backends.

## Open science

All source code, bench scripts, per-corpus JSON evidence, and reproducibility documentation are released open-source at https://github.com/openhumantech/lamquant (`docs/paper/` and `tools/`). Every numerical claim in the paper can be reproduced by running the listed scripts against a local mirror of the publicly available corpora.

## Novelty

To the best of my knowledge, this is the first paper to:
- Report a lossless EEG compression benchmark at the full scale of TUEG v2.0.2 (1.76 TB);
- Provide measured Shannon-entropy ceilings on a thirteen-corpus mixture covering scalp + intracranial + sleep + motor-imagery EEG;
- Demonstrate that the codec beats first-order Shannon ceilings on every benched corpus while remaining bit-equal across x86 / ARM / RISC-V.

## Suggested reviewers (optional)

Per TBioCAS standard practice, I am happy to suggest reviewers familiar with embedded EEG codec implementation or biomedical lossless compression on request.

## Conflict of interest

The author is the founder of OpenHuman Technologies LLC and has filed a patent application related to the codec (Patent Pending US #64/032,641). The patent covers commercial implementation rights; the source code is released open-source for academic and derivative research.

## Funding

This work was self-funded by OpenHuman Technologies LLC, Florida, USA. No external grants contributed to this research.

## Originality

The manuscript is original work that has not been published elsewhere and is not under consideration by any other journal or conference. It is not under review at any other venue.

I look forward to the reviewers' assessment of the work and to working with the editorial team toward publication.

Sincerely,

Brian Lam
Founder, OpenHuman Technologies LLC
Florida, USA
briankhanglam@usf.edu
