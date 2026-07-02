//! ADR 0074 · Track M · Phase 0 (build_first) — pin the DWT→LPC→entropy split.
//!
//! Before any stage is *extracted* from the fused kernel, this test pins the
//! decomposition as an executable contract: a hand-composed reference built ONLY
//! from the public stage primitives
//!   `lifting::{forward,forward_3level}` → `lpc::analyze_with_mode` →
//!   `golomb::encode_dense` → `assemble_lml_packet`
//! must be **byte-identical** to the fused kernel `compress_with_mode_views`, and
//! the fused bytes must decode-reconstruct the input.
//!
//! This is the equivalence theorem of the migration (`lower(dag) == fused ==
//! decode⁻¹`) at the smallest granularity, and it is **zero production change**:
//! the reference lives entirely here and reproduces the kernel's own split
//! (`encode_one_channel` lml.rs:1501, `forward_subbands` lml.rs:917,
//! `finalize_channels` lml.rs:576, `assemble_lml_packet` lml.rs:605). If a future
//! extraction drifts, this fails FIRST — before the wire goldens do.
//!
//! Default-feature build only (no `experimental_*`): the flat-Golomb payload is
//! the byte-locked reference the `byte_equal_backends` goldens pin.
#![cfg(feature = "archive")]

use lamquant_core::golomb;
use lamquant_core::lifting;
use lamquant_core::lml::{
    assemble_lml_packet, compress_with_mode_views, compute_n_levels, decompress, lpc_max_order,
    scope_lpc_mode, BIAS_CTX,
};
use lamquant_core::lpc::{self, LpcMode};

/// Reproduces the private `forward_subbands` (lml.rs:917) via the public lifting
/// primitives — the DWT stage: `[approx, detail_top, …, detail_1]`.
fn ref_forward_subbands(sig: &[i64], n_levels: u8) -> Vec<Vec<i64>> {
    match n_levels {
        3 => {
            let (a3, d3, d2, d1) = lifting::forward_3level(sig);
            vec![a3, d3, d2, d1]
        }
        2 => {
            let (l1a, l1d) = lifting::forward(sig);
            let (l2a, l2d) = lifting::forward(&l1a);
            vec![l2a, l2d, l1d]
        }
        1 => {
            let (a, d) = lifting::forward(sig);
            vec![a, d]
        }
        _ => vec![sig.to_vec()],
    }
}

/// Reproduces `encode_one_channel`'s default-build per-channel `(meta, payload)`
/// (lml.rs:1524-1578): DWT → per-subband `[order:u8][coeffs_i32_LE…]` into meta +
/// `golomb::encode_dense(residual)` appended to a flat payload. `try_extended_lpc`
/// defaults off → the ceiling is `lpc_max_order` (not the extended variant).
fn ref_channel(sig: &[i64], n_levels: u8, mode: LpcMode) -> (Vec<u8>, Vec<u8>) {
    let subbands = ref_forward_subbands(sig, n_levels);
    let mut meta = Vec::new();
    let mut payload = Vec::new();
    for (sb_idx, sub) in subbands.iter().enumerate() {
        let scoped = scope_lpc_mode(mode, lpc_max_order(sub.len()));
        let (coeffs, residual, order) =
            lpc::analyze_with_mode(sub, sb_idx, scoped, BIAS_CTX, /* time_remaining = */ None);
        meta.push(order as u8);
        for &c in &coeffs {
            meta.extend_from_slice(&c.to_le_bytes());
        }
        payload.extend_from_slice(&golomb::encode_dense(&residual).expect("golomb encode"));
    }
    (meta, payload)
}

/// The full reference packet = `map(ref_channel) >> finalize (concat all metas,
/// then all payloads) >> assemble_lml_packet`. Mirrors `encode_channels_core`
/// (lml.rs:739-755) + `finalize_channels` (lml.rs:576, default branch: wins=false).
fn ref_packet(signal: &[Vec<i64>], mode: LpcMode) -> Vec<u8> {
    assert!(!signal.is_empty(), "ref_packet needs ≥1 channel (the codec rejects n_ch=0)");
    let n_ch = signal.len();
    let t = signal[0].len();
    let n_levels = compute_n_levels(t);
    let mut lpc_meta = Vec::new();
    let mut payload = Vec::new();
    for ch in signal {
        let (m, p) = ref_channel(ch, n_levels, mode);
        lpc_meta.extend_from_slice(&m);
        payload.extend_from_slice(&p);
    }
    assemble_lml_packet(n_ch, t, n_levels, /* noise_bits = */ 0, /* wins = */ false, &lpc_meta, &payload)
}

/// Deterministic multi-channel signal (smooth ramp + bounded wobble so lifting,
/// LPC and Golomb all exercise real paths). Pure index arithmetic — no RNG.
fn synth(n_ch: usize, t: usize) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|c| {
            (0..t)
                .map(|i| {
                    let base = ((i as i64 * 3 + c as i64 * 7) % 512) - 256;
                    let wobble = (((i * i + c) % 97) as i64) - 48;
                    base * 40 + wobble
                })
                .collect()
        })
        .collect()
}

#[test]
fn dag_reference_is_byte_identical_to_fused_and_decodes() {
    // Shapes span every n_levels the encoder picks: t=4→0, t=8→1, t=20→2, t≥32→3.
    let shapes = [(1usize, 4usize), (1, 8), (1, 20), (1, 100), (4, 2500), (8, 313), (32, 2500)];
    let modes: [(&str, LpcMode); 3] = [
        ("fixed", LpcMode::Fixed),
        ("adaptive", LpcMode::Adaptive { max_order: 16 }),
        ("anytime_none", LpcMode::Anytime { max_order: 16, deadline: None }),
    ];
    for &(n_ch, t) in &shapes {
        let signal = synth(n_ch, t);
        let views: Vec<&[i64]> = signal.iter().map(|c| c.as_slice()).collect();
        for (name, mode) in modes {
            let fused = compress_with_mode_views(&views, 0, mode).expect("fused encode");
            let reference = ref_packet(&signal, mode);
            assert_eq!(
                reference, fused,
                "M0 split DRIFT: hand-composed stage reference != fused kernel \
                 (mode={name}, {n_ch}ch × {t}); the DWT→LPC→entropy decomposition no \
                 longer reproduces encode_one_channel byte-for-byte"
            );
            let decoded = decompress(&fused).expect("decode");
            assert_eq!(decoded, signal, "roundtrip failed (mode={name}, {n_ch}ch × {t})");
        }
    }
}
