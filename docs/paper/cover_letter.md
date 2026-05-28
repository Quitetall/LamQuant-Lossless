# Cover Letter — IEEE TBioCAS Submission

**Date:** May 26, 2026

**To:** Editor-in-Chief
IEEE Transactions on Biomedical Circuits and Systems

**Re:** Submission of manuscript "LamQuant Lossless: State-of-the-Art Lossless EEG Compression Designed for Microcontroller Deployment"

---

Dear Editor,

**This paper presents the first lossless EEG compression benchmark at the scale of the full Temple University EEG Corpus (1.76 TB, 70,831 files) and demonstrates state-of-the-art microcontroller-deployable compression on commodity RISC-V hardware (RP2350 Hazard3, 119× real-time, RTL-measured).**

I am pleased to submit the enclosed manuscript, *LamQuant Lossless: State-of-the-Art Lossless EEG Compression Designed for Microcontroller Deployment*, for consideration as a regular paper in IEEE Transactions on Biomedical Circuits and Systems.

## Why TBioCAS

The work bridges biomedical signal processing, embedded systems implementation, and clinical archival — a triple intersection that TBioCAS is uniquely positioned to evaluate. The codec is built specifically for deployment on a 150 MHz Hazard3 RISC-V microcontroller (RP2350) operating within 280 KB of SRAM, while remaining byte-identical to a host-side implementation. The paper reports the first lossless EEG compression benchmark conducted at the full scale of the Temple University EEG Corpus (1.76 TB, 70,831 files), validated bit-exact across 88,147 encode/decode operations spanning thirteen clinical and research EEG datasets.

## Key contributions

1. **State-of-the-art lossless EEG compression**: 2.7229:1 on CHB-MIT (a 15.9 % improvement over Chen *et al.* 2018) and **the first full-corpus benchmark on the Temple University EEG Corpus v2.0.2** (70,831 files, 1.756 TB raw → 768 GB compressed, 2.287:1, bit-exact reconstruction verified across every file).
2. **Microcontroller-deployable integer codec verified byte-identical across x86, ARM, and RISC-V**: integer-only Le Gall 5/3 lifting DWT + per-subband adaptive LPC + Golomb-Rice, with the RP2350 (Hazard3 RISC-V, 150 MHz) encode path **measured at 119× real-time via RTL-level Verilator simulation** of the same Verilog that taped out as RP2350 (12,548,067 cycles per 21 ch × 2500 sample window, 0.627 Msa/s, CPI 1.071).
3. **End-to-end reproducibility + methodological contributions**: byte-exact reconstruction across **88,147 encode/decode operations on 13 corpora with zero failures**, accompanied by (a) **empirical Shannon-entropy ceilings** on the 13-corpus mixture (H₀(X) = 10.07, H₀(ΔX) = 7.66 bits/sample) and (b) **per-file CR distributions formalising why small-corpus benchmarks overstate CR** (CHB-MIT 2.7229:1 vs full TUEG 2.287:1 — corpus diversity tightens the achievable aggregate).

## Open science

All source code, bench scripts, per-corpus JSON evidence, and reproducibility documentation are released open-source at https://github.com/Quitetall/LamQuant (`docs/paper/` and `tools/`). Every numerical claim in the paper can be reproduced by running the listed scripts against a local mirror of the publicly available corpora.

## Novelty

To the best of my knowledge, this is the first paper to:
- Report a lossless EEG compression benchmark at the full scale of TUEG v2.0.2 (1.76 TB);
- Provide measured Shannon-entropy ceilings on a thirteen-corpus mixture covering scalp + intracranial + sleep + motor-imagery EEG;
- Demonstrate that the codec beats first-order Shannon ceilings on every benched corpus while remaining bit-equal across x86 / ARM / RISC-V.

## Suggested reviewers

- **Prof. Joseph Picone** (Temple University, Neural Engineering Data Consortium) — maintainer of the TUEG corpus; expertise in clinical EEG storage at scale.
- **Dr. Geoffrey Higgins** (NUI Galway, Tyndall National Institute) — recent IEEE TBME work on lossless biosignal compression for implantable hardware (region-adaptive LPC + Golomb-Rice, our [13]); directly aligned with the MCU-deployable kernel choices in this paper.
- **Prof. N. Sriraam** (REVA University) — long-standing contributor to lossless EEG compression (our [7], [12]); referenced for the bias-cancellation lineage in §II.E.

The author defers selection of an additional comparator AE from the current TBioCAS editorial board to the handling editor.

## Conflict of interest

The author is the founder of OpenHuman Technologies LLC and has filed a (provisional) patent application related to the codec (Patent Pending US #64/032,641). The source code is released open-source under the GNU General Public License version 3 (GPLv3). Section 11 of GPLv3 conveys an **automatic, irrevocable patent license** from every contributor to every downstream user of the licensed implementation, covering any claims that read on the released code. This is materially stronger than a unilateral non-enforcement commitment: every user of the GPLv3-licensed implementation already holds an irrevocable patent license for the act of using, modifying, and redistributing that implementation, by operation of the license terms.

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
