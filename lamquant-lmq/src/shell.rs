//! ADR 0074 Track N — the LMQ **shell**: `Abir<M>` ⇄ a lossy-signed BCS1-Lmq
//! container, over a [`NeuralBackend`]. The shell owns the whole wire-critical
//! path (header, entropy body, lossy-signing, modality provenance); the backend
//! owns only the network.

use alloc::vec;
use alloc::vec::Vec;

use abir::{
    Abir, Bcs1Header, Modality, ModalityProvenance, ModalitySource, Untyped, BCS1_HEADER_LEN,
    BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LMQ_FSQ,
};

use crate::backend::{BackendError, NeuralBackend, NeuralTokens};
use crate::body::{decode_body, encode_body, BodyError};

/// The `decode_capability` byte a neural file carries. Any non-zero value makes
/// every lossless BCS1 reader refuse it fail-closed (they accept only `0` = the
/// integer floor). Together with `codec_descriptor = CODEC_LMQ_FSQ` this is the
/// permanent lossy signature on the wire — a `.lmq` can never be mis-decoded as
/// integer samples.
pub const LMQ_DECODE_CAPABILITY: u8 = 1;

/// The fixed rANS model total the shell normalizes every frequency table to, so
/// the model size never scales with token count (which would exceed
/// `body::MAX_MODEL_TOTAL`). Matches the Python reference's `total_freq=4096`.
pub const RANS_MODEL_TOTAL: u64 = 4096;

/// Failure encoding/decoding a BCS1-Lmq container.
#[derive(Debug)]
pub enum LmqError {
    /// The backend (inference) failed.
    Backend(BackendError),
    /// The body codec failed.
    Body(BodyError),
    /// The header could not be parsed / the buffer is too short.
    Header,
    /// The stream is not a BCS1-Lmq container (wrong magic or codec_descriptor).
    NotLmq,
    /// A token fell outside `[0, alphabet)` (a buggy backend must not corrupt the
    /// frequency model).
    BadTokens,
    /// The header's `modality_source` byte is not a recognized [`ModalitySource`]
    /// (corrupt / foreign) — refused fail-closed rather than defaulted.
    BadModality,
}

impl From<BodyError> for LmqError {
    fn from(e: BodyError) -> Self {
        LmqError::Body(e)
    }
}

/// Encode `abir` into a lossy-signed BCS1-Lmq container via `backend`.
///
/// The result is PERMANENTLY lossy: `codec_descriptor = CODEC_LMQ_FSQ` +
/// `decode_capability = LMQ_DECODE_CAPABILITY`, so every lossless reader refuses
/// it. The modality provenance from `abir` is preserved in header bytes 6/7.
pub fn encode<M: Modality>(abir: &Abir<M>, backend: &dyn NeuralBackend) -> Result<Vec<u8>, LmqError> {
    // Whole-recording encode. NB: the rANS coder caps a single stream at
    // `MAX_RANS_SYMBOLS` (1<<20) tokens, so a very long recording that produces
    // more tokens than that must be windowed before this call — a documented
    // N-follow-up (the real backend emits latent tokens; the 1:1 StubBackend is
    // the one that hits it soonest). The model total is already token-count-
    // independent (see `histogram`).
    debug_assert!(abir.sample_rate > 0.0, "sample_rate must be positive");
    let n = abir.n_samples();
    let signal: Vec<Vec<i64>> =
        abir.window_views(0, n).iter().map(|c| c.as_ref().to_vec()).collect();
    let tokens = backend.encode(&signal, abir.sample_rate).map_err(LmqError::Backend)?;

    let counts = histogram(&tokens.tokens, tokens.alphabet)?;
    let tokens_i64: Vec<i64> = tokens.tokens.iter().map(|&t| t as i64).collect();
    let body = encode_body(&tokens_i64, &tokens.schedule, &counts)?;

    let header = Bcs1Header {
        version_major: BCS1_VERSION_MAJOR,
        version_minor: BCS1_VERSION_MINOR,
        modality_tag: abir.provenance().tag,
        modality_source: abir.provenance().source.to_u8(),
        codec_descriptor: CODEC_LMQ_FSQ,
        // Neural is signaled by the descriptor; `mode` (a lossless RD concept) is
        // a documented placeholder here.
        mode: 0,
        tier: 0,
        decode_capability: LMQ_DECODE_CAPABILITY,
        n_channels: tokens.n_channels,
        n_windows: 1,
        total_samples: tokens.n_samples,
        // Not meaningful for a token stream; carries the FSQ alphabet for `info`.
        window_size: tokens.alphabet,
        sample_rate_mhz: (abir.sample_rate * 1000.0) as u32,
        bit_depth: 16,
        flags: 0,
        // The backend's opaque decode state rides in the metadata section, between
        // the 40-byte header and the body. Keeps the N0 body format untouched.
        metadata_length: tokens.backend_meta.len() as u32,
    };

    let mut out = Vec::with_capacity(BCS1_HEADER_LEN + tokens.backend_meta.len() + body.len());
    out.extend_from_slice(&header.to_bytes());
    out.extend_from_slice(&tokens.backend_meta);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a BCS1-Lmq container back into an `Abir<Untyped>` (its modality carried
/// in provenance) via `backend`. Refuses anything that is not a BCS1-Lmq stream.
pub fn decode(bytes: &[u8], backend: &dyn NeuralBackend) -> Result<Abir<Untyped>, LmqError> {
    let header = Bcs1Header::parse(bytes).map_err(|_| LmqError::Header)?;
    if header.codec_descriptor != CODEC_LMQ_FSQ {
        return Err(LmqError::NotLmq);
    }
    // Split the metadata section (backend_meta) from the body. Both offsets are
    // bounds-checked — a crafted metadata_length can't slice past the buffer.
    let meta_len = header.metadata_length as usize;
    let after_header = bytes.get(BCS1_HEADER_LEN..).ok_or(LmqError::Header)?;
    let backend_meta = after_header.get(..meta_len).ok_or(LmqError::Header)?.to_vec();
    let body = after_header.get(meta_len..).ok_or(LmqError::Header)?;
    let (tokens_i64, schedule, alphabet) = decode_body(body)?;

    // Defense-in-depth at the trust boundary: rANS decode only emits symbols in
    // `[0, alphabet)` for a valid in-band model (so this is unreachable today, and
    // `alphabet <= 65535` means the `as i32` can't truncate), but validate before
    // handing tokens to a backend — a real one may index a codebook by them.
    // Mirrors the encode-side range check in `histogram`.
    let mut tokens_i32 = Vec::with_capacity(tokens_i64.len());
    for &t in &tokens_i64 {
        if t < 0 || t >= alphabet as i64 {
            return Err(LmqError::BadTokens);
        }
        tokens_i32.push(t as i32);
    }
    let tokens = NeuralTokens {
        tokens: tokens_i32,
        schedule,
        alphabet,
        n_channels: header.n_channels,
        n_samples: header.total_samples,
        backend_meta,
    };
    let signal = backend.decode(&tokens).map_err(LmqError::Backend)?;

    let sample_rate = header.sample_rate_mhz as f64 / 1000.0;
    let mut abir = Abir::<Untyped>::from_channels_i64(signal, sample_rate);
    // Restore the modality provenance recorded in header bytes 6/7 (source + tag),
    // exactly as it was born. A `modality_source` byte outside {0,1,2} is corrupt/
    // foreign — refuse it fail-closed rather than fabricating `Manual` (from_u8's
    // documented contract: "callers must not silently default to a source").
    let source = ModalitySource::from_u8(header.modality_source).ok_or(LmqError::BadModality)?;
    abir.prov = ModalityProvenance { source, tag: header.modality_tag };
    Ok(abir)
}

/// The rANS frequency model from `tokens`, NORMALIZED to a fixed total
/// ([`RANS_MODEL_TOTAL`] = 4096) so the model size never scales with token count
/// (an un-normalized histogram's total = alphabet + n_tokens, which blows past
/// `body::MAX_MODEL_TOTAL` for long recordings). Every symbol in `[0, alphabet)`
/// keeps freq >= 1 (no zero-frequency symbol reaches rANS). The normalized counts
/// travel in-band, so `decode_body` rebuilds the identical model and tokens
/// round-trip exactly. Errors if a token is out of range — a buggy backend must
/// not corrupt the model or panic the histogram.
fn histogram(tokens: &[i32], alphabet: u16) -> Result<Vec<i32>, LmqError> {
    let a = alphabet as usize;
    if a == 0 || a as u64 > RANS_MODEL_TOTAL {
        return Err(LmqError::BadTokens);
    }
    let mut raw = vec![0u64; a];
    for &t in tokens {
        if t < 0 || t as usize >= a {
            return Err(LmqError::BadTokens);
        }
        raw[t as usize] += 1;
    }
    // Every symbol starts at 1 (no zero-freq); distribute the rest of the fixed
    // budget proportionally to the observed counts. Result sums to exactly
    // RANS_MODEL_TOTAL regardless of token count.
    let mut freq = vec![1i32; a];
    let sum: u64 = raw.iter().sum();
    if sum == 0 {
        return Ok(freq); // no tokens → uniform model (m = alphabet)
    }
    let budget = RANS_MODEL_TOTAL - a as u64; // reserved 1 per symbol
    let mut assigned = 0u64;
    for i in 0..a {
        let extra = raw[i] * budget / sum; // floor
        freq[i] += extra as i32;
        assigned += extra;
    }
    // The flooring remainder (< alphabet) → the highest-count symbol, deterministically.
    let remainder = budget - assigned;
    if remainder > 0 {
        let best = (0..a).max_by_key(|&i| raw[i]).unwrap_or(0);
        freq[best] += remainder as i32;
    }
    Ok(freq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StubBackend;
    use abir::{Eeg, ModalitySource};

    fn eeg_abir() -> Abir<Eeg> {
        let sig: Vec<Vec<i64>> =
            (0..4).map(|c| (0..500).map(|i| ((i * 3 + c * 7) % 40) as i64 - 20).collect()).collect();
        Abir::<Untyped>::from_channels_i64(sig, 250.0).into_modality::<Eeg>(ModalitySource::Manual)
    }

    #[test]
    fn shell_roundtrips_through_the_wire_and_is_lossy_signed() {
        let abir = eeg_abir();
        let backend = StubBackend { alphabet: 5 };
        let bytes = encode(&abir, &backend).unwrap();

        // Lossy-signed header: BCS1 magic, LMQ descriptor, a refusing capability,
        // and the Eeg modality preserved in bytes 6/7.
        assert_eq!(&bytes[0..4], b"BCS1");
        assert_eq!(bytes[8], CODEC_LMQ_FSQ, "codec_descriptor = LmqFsq");
        assert_eq!(bytes[11], LMQ_DECODE_CAPABILITY, "decode_capability refuses lossless readers");
        assert_eq!(bytes[6], Eeg::TAG, "modality tag preserved");
        assert_eq!(bytes[7], ModalitySource::Manual.to_u8(), "modality source preserved");

        // The wire round-trip equals the backend's own lossy reconstruction.
        let decoded = decode(&bytes, &backend).unwrap();
        let orig: Vec<Vec<i64>> =
            abir.window_views(0, abir.n_samples()).iter().map(|c| c.as_ref().to_vec()).collect();
        let expect = backend.decode(&backend.encode(&orig, 250.0).unwrap()).unwrap();
        let got: Vec<Vec<i64>> =
            decoded.window_views(0, decoded.n_samples()).iter().map(|c| c.as_ref().to_vec()).collect();
        assert_eq!(got, expect, "wire round-trip must equal the backend reconstruction");
        assert_eq!(decoded.provenance().tag, Eeg::TAG, "modality survives to the decoded Abir");
        assert_eq!(decoded.sample_rate, 250.0);
    }

    #[test]
    fn decode_refuses_a_non_lmq_stream() {
        let backend = StubBackend::default();
        let mut fake = [0u8; BCS1_HEADER_LEN + 4];
        fake[0..4].copy_from_slice(b"BCS1");
        fake[4] = BCS1_VERSION_MAJOR;
        // codec_descriptor (byte 8) stays 0 (Lml53) — not an LMQ stream.
        assert!(matches!(decode(&fake, &backend), Err(LmqError::NotLmq)));
    }

    #[test]
    fn histogram_total_is_bounded_regardless_of_token_count() {
        // 5M tokens: an un-normalized model would be m ≈ 5M (>> body::MAX_MODEL_TOTAL
        // 1<<20); normalized it stays == RANS_MODEL_TOTAL (4096) with every freq >= 1.
        // (This is the model fix from the adversarial review. Note: rANS separately
        // caps n_symbols at 1<<20, so encoding a WHOLE recording of >1M tokens still
        // needs windowing — a documented N-follow-up; here we test the model alone.)
        let tokens: Vec<i32> = (0..5_000_000).map(|i| (i % 5) as i32).collect();
        let counts = histogram(&tokens, 5).unwrap();
        let m: i64 = counts.iter().map(|&c| c as i64).sum();
        assert_eq!(m as u64, RANS_MODEL_TOTAL, "model total must be fixed, not token-count-scaled");
        assert!(counts.iter().all(|&c| c >= 1), "every symbol freq must be >= 1");
    }

    #[test]
    fn backend_metadata_round_trips_through_the_header() {
        // A metadata-bearing backend (mimics the Python codec, which needs its
        // per-channel preprocessing state on decode). The meta must survive the
        // wire verbatim and reach the backend's decode.
        use crate::backend::{BackendError, NeuralTokens};
        struct MetaStub;
        const META: &[u8] = &[0xAB, 0xCD, 0xEF, 0x01];
        impl NeuralBackend for MetaStub {
            fn encode(&self, signal: &[Vec<i64>], _sr: f64) -> Result<NeuralTokens, BackendError> {
                Ok(NeuralTokens {
                    tokens: signal.iter().flat_map(|c| c.iter().map(|&s| s.rem_euclid(3) as i32)).collect(),
                    schedule: vec![3u8; signal[0].len()],
                    alphabet: 3,
                    n_channels: signal.len() as u16,
                    n_samples: signal[0].len() as u32,
                    backend_meta: META.to_vec(),
                })
            }
            fn decode(&self, t: &NeuralTokens) -> Result<Vec<Vec<i64>>, BackendError> {
                assert_eq!(t.backend_meta, META, "backend_meta must survive the wire verbatim");
                let n_s = t.n_samples as usize;
                Ok((0..t.n_channels as usize)
                    .map(|c| t.tokens[c * n_s..(c + 1) * n_s].iter().map(|&x| x as i64).collect())
                    .collect())
            }
        }
        let abir = eeg_abir();
        let bytes = encode(&abir, &MetaStub).unwrap();
        // metadata_length (header bytes 28..32) records the meta length.
        assert_eq!(u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]), META.len() as u32);
        // decode delivers the meta back to the backend (its assert fires if not).
        let _ = decode(&bytes, &MetaStub).unwrap();
    }

    #[test]
    fn decode_rejects_a_corrupt_modality_source_byte() {
        let abir = eeg_abir();
        let backend = StubBackend { alphabet: 5 };
        let mut bytes = encode(&abir, &backend).unwrap();
        bytes[7] = 42; // modality_source outside {0,1,2}
        assert!(matches!(decode(&bytes, &backend), Err(LmqError::BadModality)));
    }
}
