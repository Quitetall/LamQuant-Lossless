//! Encode-only **montage geometry**: an electrode-position search prior for the
//! cross-channel reference selection (Phase A1, `eeg-codec-design-from-port` §2 Stage 1).
//!
//! Geometry NEVER reaches the wire. It only guides WHICH prior channels the encoder
//! tries as cross-channel references; the chosen `(ref_idx, gain)` pairs are serialized
//! exactly as today (`lmo_lossless` id=2 body), so the wire format and the `no_std`
//! decoder are unchanged and geometry-free encoding is byte-identical to today. The
//! win is keep-smaller-safe: the geometry candidate is just one more reference set the
//! per-channel search keeps only if it shrinks the channel.
#![cfg(feature = "encode")]

use alloc::vec::Vec;

/// Per-channel electrode coordinates in meters (head-centered); `None` = unresolved
/// (unknown montage position ⇒ that channel contributes no geometry prior).
pub struct MontageGeometry {
    coords: Vec<Option<[f64; 3]>>,
}

impl MontageGeometry {
    /// Build from per-channel coords (e.g. the `.lmq` container's `coords` field, or a
    /// resolver over channel labels). `coords.len()` should equal the channel count.
    pub fn new(coords: Vec<Option<[f64; 3]>>) -> Self {
        Self { coords }
    }

    /// The `k` nearest PRIOR channels (index `< i`) that have resolved coords, ordered
    /// by squared Euclidean electrode distance to channel `i`. Empty if channel `i` is
    /// unresolved (⇒ the caller falls back to the non-geometry search for it).
    pub fn nearest_prior(&self, i: usize, k: usize) -> Vec<usize> {
        let Some(ci) = self.coords.get(i).copied().flatten() else {
            return Vec::new();
        };
        let mut cand: Vec<(f64, usize)> = (0..i)
            .filter_map(|j| {
                self.coords[j].map(|cj| {
                    let d = (0..3).map(|a| (ci[a] - cj[a]).powi(2)).sum::<f64>();
                    (d, j)
                })
            })
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        cand.into_iter().take(k).map(|(_, j)| j).collect()
    }

    /// Number of channels (resolved or not) this geometry covers.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}
