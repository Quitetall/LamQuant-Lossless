"""
LamQuant Gen 7.6 EEG Neural Codec — 4-mode compression.

Mode 0: Neural only    (274:1, R≈0.90)  TNN → FSQ → rANS, no details
Mode 1: Neural + lossy ( 42:1, R≈0.87)  + PCA-projected details, threshold 8
Mode 2: Neural + clean ( 22:1, R≈0.89)  + full 21ch details, threshold 2
Mode 3: Lossless DSP   (4.8:1, R=1.000) KLT → LPC → lifting → Golomb-Rice

Classes:
  TernaryCodec:   Gen 7.0 (raw → latent → rANS)
  SubbandCodec:   Gen 7.6 Modes 0-2 (neural codec + detail subbands)
  LosslessCodec:  Gen 7.6 Mode 3 (pure DSP, no neural network)
"""
import struct
import sys
import os
import numpy as np
import torch

from lamquant_codec.codec_types import EEGPacket

# Entropy coding primitives — the pure mechanism layer.
from lamquant_codec.ops.golomb import (
    BitWriter, BitReader,
    zigzag_encode as _zigzag_encode,
    zigzag_decode as _zigzag_decode,
    compute_adaptive_k as _compute_adaptive_k,
    encode_dense as _encode_dense_subband,
    decode_dense as _decode_dense_subband,
    encode_detail as _encode_detail_subband,
    decode_detail as _decode_detail_subband,
)
from lamquant_codec.ops.rans import (
    encode as _rans_encode_symbols,
    decode as _rans_decode_symbols,
)

from lamquant_codec._paths import REPO_ROOT as _REPO_ROOT
_ROOT = str(_REPO_ROOT)


def _resolve_checkpoint(checkpoint_path, filename):
    """Resolve a codec checkpoint path with strict, training-tree-free probing.

    Probe order (ADR 0018 — codec reads weights/ only):

      1. Explicit ``checkpoint_path`` argument (caller wins).
      2. ``$LAMQUANT_WEIGHTS_DIR/<filename>`` if env var is set.
      3. ``<repo>/weights/<filename>`` (repo-relative default).

    On miss: ``FileNotFoundError`` enumerating every probed path so the
    user sees exactly where the loader looked. No silent fallback to
    ``ai_models/student/`` (that was the pre-ADR-0018 leak — codec
    runtime is no longer aware of the training tree).
    """
    if checkpoint_path is not None:
        return checkpoint_path
    probed = []
    env_dir = os.environ.get("LAMQUANT_WEIGHTS_DIR")
    if env_dir:
        p = os.path.join(env_dir, filename)
        probed.append(p)
        if os.path.exists(p):
            return p
    p = os.path.join(_ROOT, "weights", filename)
    probed.append(p)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(
        f"codec checkpoint '{filename}' not found. Probed:\n  "
        + "\n  ".join(probed)
        + "\n\nPlace the checkpoint under ./weights/ or set "
          "$LAMQUANT_WEIGHTS_DIR (see ADR 0018)."
    )

from lamquant_codec.models.encoder import (
    TernaryMobileNetV5, TernaryMobileNetV5_Subband, TernaryMobileNetV5_Subband_V2,
)


def _safe_torch_load(path, map_location='cpu'):
    """Load checkpoint preferring weights_only=True for security."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError):
        import warnings
        warnings.warn(
            f"weights_only=True failed for {path}; falling back to "
            f"weights_only=False. Ensure this checkpoint is from a "
            f"trusted source.",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


class TernaryCodec:
    """
    End-to-end EEG neural codec wrapping TernaryMobileNetV5.

    Provides:
      - encode(x) -> latent
      - decode(latent) -> reconstructed
      - compress(latent) -> bytes (FSQ + rANS)
      - decompress(data) -> latent
    """

    def __init__(self, model, fsq_levels=16, rans_total_freq=4096):
        self.model = model
        self.model.eval()
        self.fsq_levels = fsq_levels
        self.rans_total_freq = rans_total_freq

    @classmethod
    def from_checkpoint(cls, checkpoint_path=None, in_ch=21, latent_dim=32, **kwargs):
        """Load codec from a saved checkpoint.

        Resolution: explicit arg → ``$LAMQUANT_WEIGHTS_DIR`` →
        ``./weights/`` (ADR 0018). On miss raises ``FileNotFoundError``
        listing every probed path. No ``ai_models/`` fallback.
        """
        checkpoint_path = _resolve_checkpoint(checkpoint_path, "student_hardened.ckpt")
        model = TernaryMobileNetV5(in_ch=in_ch, latent_dim=latent_dim)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        return cls(model, **kwargs)

    def encode(self, x):
        """Encode EEG tensor [B, 21, T] -> latent [B, 32, T/8]."""
        with torch.no_grad():
            return self.model.encode(x, quantize=True)

    def decode(self, latent):
        """Decode latent [B, 32, T/8] -> reconstructed [B, 21, T]."""
        with torch.no_grad():
            return self.model.decode(latent)

    # compress() and decompress() were removed — the inline Python rANS
    # they contained produced "LMQ v2" packets that never decoded correctly.
    # Use SubbandCodec (LMQ1/LMQ3 via ops/rans.py) for neural compression.


# ------------------------------------------------------------
# Golomb-Rice / BitWriter / BitReader / zigzag / dense+detail
# helpers moved to lamquant_codec/ops/golomb.py (pure mechanism).
# They are imported at the top of this file for backward compat.
# ------------------------------------------------------------

class SubbandCodec:
    """
    Gen 7.1 "Subband" EEG neural codec.

    Full pipeline:
      Encode: EEG [21, 2500] → HP → LPC → lifting → L3 [21, 313]
              → TNN encode → latent [32, 79] → WHT → FSQ → rANS
      Decode: rANS → FSQ → WHT inverse → TNN decode → L3 [21, 313]
              → inverse lifting → LPC synthesis → EEG [21, 2500]

    Provides:
      - encode(x) -> (latent, metadata)
      - decode(latent, metadata) -> reconstructed EEG
      - compress(latent, lpc_coeffs, subbands, quality_mode) -> bytes
      - decompress(data) -> (latent, lpc_coeffs, detail_subbands)
    """

    from lamquant_codec.ops.constants import QUALITY_ALERTING, QUALITY_MONITORING, QUALITY_CLINICAL, FSQ_LEVELS_BY_MODE

    def __init__(self, model, rans_total_freq=4096, snn=None):
        """Construct a SubbandCodec.

        Args:
          model: the TernaryMobileNetV5_Subband (or _V2) encoder/decoder pair.
          rans_total_freq: rANS frequency-table denominator.
          snn: Optional MambaSNN instance for adaptive FSQ level scheduling.
            When attached, neural compress paths can call
            `snn.classify_per_timestep(...)` to produce per-timestep FSQ
            levels (LMQ3 wire format). Default None = legacy uniform LMQ1.
            Use `set_snn(snn)` to attach lazily after construction.
        """
        # Boundary checks BEFORE touching `model.eval()` — otherwise None
        # raises AttributeError ahead of the explicit TypeError (V4 Flash
        # Finding 1 of the bones-A.1 commit f64589d).
        if model is None:
            raise TypeError("SubbandCodec.model must not be None")
        # bool is a subclass of int in Python; reject it explicitly so
        # `True` / `False` cannot smuggle past the type check (V4 Pro
        # Finding 2 / V4 Flash Finding 2 of f64589d).
        if isinstance(rans_total_freq, bool) or not isinstance(rans_total_freq, int) \
                or rans_total_freq <= 0:
            raise ValueError(
                f"rans_total_freq must be positive int, got {rans_total_freq!r}"
            )
        self.model = model
        self.model.eval()
        self.rans_total_freq = rans_total_freq
        # Route through set_snn so the classify_per_timestep guard fires
        # at construction too (V4 Pro Finding 1 of f64589d — hostile-caller
        # hardening must hold for both __init__ and set_snn entry points).
        self._snn = None
        self.set_snn(snn)

    def set_snn(self, snn) -> None:
        """Attach (or detach) the SNN used for adaptive FSQ level scheduling.

        Args:
          snn: MambaSNN-shaped object (must expose `classify_per_timestep`),
            or None to detach.
        Raises:
          TypeError: snn is non-None and lacks `classify_per_timestep`.
        """
        if snn is not None and not hasattr(snn, "classify_per_timestep"):
            raise TypeError(
                "SubbandCodec.set_snn: object must expose classify_per_timestep "
                f"(got {type(snn).__name__})"
            )
        self._snn = snn

    @property
    def snn(self):
        """The attached SNN, or None. Read-only — use `set_snn` to mutate."""
        return self._snn

    @classmethod
    def from_checkpoint(cls, checkpoint_path=None, in_ch=21, latent_dim=32, **kwargs):
        """Load codec from a saved checkpoint.

        Resolution: explicit arg → ``$LAMQUANT_WEIGHTS_DIR`` →
        ``./weights/`` (ADR 0018). On miss raises ``FileNotFoundError``
        listing every probed path. No ``ai_models/`` fallback.
        """
        checkpoint_path = _resolve_checkpoint(checkpoint_path, "student_subband.ckpt")
        raw = _safe_torch_load(checkpoint_path, map_location='cpu')
        # Handle both old (raw state_dict) and new (dict with 'state_dict'
        # key + provenance; produced by CheckpointManager from refactor #72+).
        if isinstance(raw, dict) and 'state_dict' in raw:
            state = raw['state_dict']
        else:
            state = raw
        # Auto-detect V2 (DW-sep focal blocks) from checkpoint keys
        is_v2 = any('focal2.dw.' in k or 'focal4.dw.' in k for k in state.keys())
        if is_v2:
            model = TernaryMobileNetV5_Subband_V2(in_ch=in_ch, latent_dim=latent_dim)
        else:
            model = TernaryMobileNetV5_Subband(in_ch=in_ch, latent_dim=latent_dim)
        model.load_state_dict(state, strict=False)
        return cls(model, **kwargs)

    def preprocess(self, x_np):
        """Run LPC + lifting preprocessing on numpy signal [C, T].
        Returns (l3_approx, lpc_coeffs, subbands_per_ch).
        """
        from lamquant_codec.ops.pipeline import preprocess_subband
        return preprocess_subband(x_np, order=8, autocorr_len=256)

    def encode(self, x):
        """Encode EEG tensor [B, 21, 2500] through full subband pipeline.
        Returns (latent [B, 32, 79], metadata list).
        """
        from lamquant_codec.ops.pipeline import preprocess_subband_torch
        from lamquant_codec.ops.wht import forward_32_torch as wht32_forward_torch

        with torch.no_grad():
            l3, metadata = preprocess_subband_torch(x)
            l3 = l3.to(x.device)
            latent = self.model.encode(l3, quantize=True)
            latent_wht = wht32_forward_torch(latent)
        return latent_wht, metadata

    def decode(self, latent, metadata):
        """Decode latent through inverse WHT → TNN → inverse lifting → LPC synthesis.
        Returns reconstructed EEG [B, 21, 2500].
        """
        from lamquant_codec.ops.wht import inverse_32_torch as wht32_inverse_torch
        from lamquant_codec.ops.pipeline import reconstruct_subband_torch

        with torch.no_grad():
            latent_unwht = wht32_inverse_torch(latent)
            l3_recon = self.model.decode(latent_unwht, target_len=313, quantize=True)
        recon = reconstruct_subband_torch(l3_recon, metadata)
        return recon

    def compress(self, latent, lpc_coeffs=None, subbands_per_ch=None,
                 quality_mode=2):
        """Compress latent + side information to LMQ v2 bytes.

        Delegates to `lamquant_codec.compress._compress_bytes` — the single
        source of truth for the wire format. No dataclass overhead on this
        hot path; the primitive takes the raw latent array directly.
        """
        from lamquant_codec.compress import _compress_bytes
        latent_np = latent.detach().cpu().numpy() if hasattr(latent, 'detach') else np.asarray(latent)
        return _compress_bytes(
            latent_np,
            lpc_coeffs=lpc_coeffs,
            subbands_per_ch=subbands_per_ch,
            quality_mode=quality_mode,
            fsq_levels=self.FSQ_LEVELS_BY_MODE.get(quality_mode, 16),
            rans_total=self.rans_total_freq,
        )

    def decompress(self, data):
        """Decompress LMQ v2 packet back to latent tensor + side-info.

        Delegates to `lamquant_codec.decompress._decompress_bytes`. Returns
        the same 4-tuple as before (latent, quality_mode, lpc_bytes,
        detail_bytes) so existing callers work unchanged. Latent is wrapped
        in a torch.Tensor to preserve the original signature.
        """
        from lamquant_codec.decompress import _decompress_bytes
        latent_np, quality_mode, _L, _fsq, lpc_bytes, detail_bytes = _decompress_bytes(data)
        latent = torch.from_numpy(latent_np.astype(np.float32))
        return latent, quality_mode, lpc_bytes, detail_bytes

    def compress_q2d2(self, latent, quality_mode=2, lpc_coeffs=None):
        """Q2D2: Pairwise 2D grid quantization (pairs adjacent channels).

        Instead of quantizing each of 32 channels independently at L levels,
        pairs adjacent channels and quantizes on an L×L 2D grid. Captures
        inter-channel correlations that independent FSQ misses.

        For L=5: 25 joint codes per pair vs 10 independent codes.
        16 pairs × 79 timesteps = 1,264 2D symbols (vs 2,528 1D symbols).
        """
        L = self.FSQ_LEVELS_BY_MODE.get(quality_mode, 16)
        L2 = L * L  # 2D symbol space

        lat_np = latent.numpy() if hasattr(latent, 'numpy') else latent
        if lat_np.ndim == 3:
            lat_np = lat_np[0]  # [32, 79]
        lat_dim, lat_T = lat_np.shape
        n_pairs = lat_dim // 2

        vmin, vmax = float(lat_np.min()), float(lat_np.max())
        span = vmax - vmin + 1e-8

        # Normalize to [0, 1]
        normalized = (lat_np - vmin) / span

        # Pair adjacent channels and compute 2D symbols
        symbols_2d = []
        for t in range(lat_T):
            for p in range(n_pairs):
                ch_a = int(np.clip(normalized[2 * p, t] * L, 0, L - 1))
                ch_b = int(np.clip(normalized[2 * p + 1, t] * L, 0, L - 1))
                symbols_2d.append(ch_a * L + ch_b)  # 2D → 1D index

        symbols_2d = np.array(symbols_2d, dtype=np.int64)

        # rANS encode on L² symbol space
        rans_bytes, freq = _rans_encode_symbols(symbols_2d, total_freq=self.rans_total_freq)

        # LPC payload
        lpc_payload = bytearray()
        if lpc_coeffs is not None:
            lpc_q15 = np.round(np.clip(lpc_coeffs.astype(np.float64).flatten(),
                                        -1.0, 1.0) * 32767.0).astype(np.int16)
            deltas = np.diff(lpc_q15, prepend=0).astype(np.int16)
            lpc_payload = bytearray(deltas.tobytes())

        freq_bytes = freq.astype(np.uint16).tobytes()

        # Header (Q2D2 variant)
        header = struct.pack('<4sBBHHffIII',
                             b'QD2D',
                             quality_mode, L,
                             lat_dim, lat_T,
                             vmin, vmax,
                             len(rans_bytes),
                             len(lpc_payload), 0)

        return (bytes(header) + freq_bytes + rans_bytes + bytes(lpc_payload))

    def decompress_q2d2(self, data):
        """Decompress Q2D2 packet back to latent tensor."""
        hdr_size = 4 + 1 + 1 + 2 + 2 + 4 + 4 + 4 + 4 + 4  # 30 bytes
        (magic, quality_mode, L, lat_dim, lat_T, vmin, vmax,
         rans_len, lpc_len, detail_len) = struct.unpack(
            '<4sBBHHffIII', data[:hdr_size])
        if magic != b'QD2D':
            raise ValueError(f"Invalid Q2D2 header: expected b'QD2D', got {magic!r}")

        L2 = L * L
        n_pairs = lat_dim // 2
        span = vmax - vmin + 1e-8
        pos = hdr_size

        # Read frequency table (L² entries)
        freq = np.frombuffer(data[pos:pos + L2 * 2], dtype=np.uint16).astype(np.int32).copy()
        pos += L2 * 2

        # rANS decode
        n_symbols = n_pairs * lat_T
        symbols_2d = _rans_decode_symbols(data[pos:pos + rans_len], freq, n_symbols)
        pos += rans_len

        # Reconstruct latent from 2D symbols
        latent = np.zeros((lat_dim, lat_T), dtype=np.float32)
        idx = 0
        for t in range(lat_T):
            for p in range(n_pairs):
                s = int(symbols_2d[idx])
                ch_a = s // L
                ch_b = s % L
                latent[2 * p, t] = vmin + (ch_a + 0.5) / L * span
                latent[2 * p + 1, t] = vmin + (ch_b + 0.5) / L * span
                idx += 1

        latent = torch.from_numpy(latent).unsqueeze(0)
        lpc_bytes = data[pos:pos + lpc_len] if lpc_len > 0 else b''
        return latent, quality_mode, lpc_bytes, b''

    def compress_adaptive(self, latent, level_schedule, lpc_coeffs=None):
        """Compress with per-timestep adaptive FSQ levels.

        Sub-window adaptation: different FSQ levels per temporal segment.
        The SNN classifies each timestep and assigns L=2 (quiet), L=3 (active),
        or L=5 (event). Saves 50-63% on event windows by spending bits only
        where the event is.

        Packet format (LMQ3):
          [0-3]    'LMQ3' sync
          [4]      n_runs (uint8, run-length schedule entries)
          [5-6]    latent_dim (uint16)
          [7-8]    latent_T (uint16)
          [9-12]   vmin (float32)
          [13-16]  vmax (float32)
          [17-20]  rans_payload_length (uint32)
          [21-24]  lpc_payload_length (uint32)
          [25+]    schedule: n_runs × (L_value:uint8, count:uint8)
                   per-level freq tables
                   rANS payload
                   LPC payload (optional)

        Args:
            latent: [1, 32, 79] tensor (post-CDF, uniform [-1,1])
            level_schedule: [79] array of FSQ levels (2, 3, or 5)
            lpc_coeffs: optional LPC coefficients (None for Route B)
        """
        lat_np = latent.numpy() if hasattr(latent, 'numpy') else latent
        if lat_np.ndim == 3:
            lat_np = lat_np[0]  # [32, 79]
        lat_dim, lat_T = lat_np.shape
        level_schedule = np.asarray(level_schedule, dtype=np.int32)

        vmin, vmax = float(lat_np.min()), float(lat_np.max())
        span = vmax - vmin + 1e-8

        # Run-length encode the schedule
        runs = []  # (L_value, count)
        cur_L, cur_count = int(level_schedule[0]), 1
        for t in range(1, lat_T):
            if int(level_schedule[t]) == cur_L:
                cur_count += 1
            else:
                runs.append((cur_L, cur_count))
                cur_L, cur_count = int(level_schedule[t]), 1
        runs.append((cur_L, cur_count))

        # Quantize each timestep with its assigned L and collect symbols
        all_symbols = []
        for t in range(lat_T):
            L = int(level_schedule[t])
            col = lat_np[:, t]  # [32] values for this timestep
            normalized = (col - vmin) / span
            symbols = np.clip((normalized * L).astype(np.int32), 0, L - 1)
            all_symbols.append((L, symbols))

        # Build per-level frequency tables and rANS encode per segment
        rans_total = self.rans_total_freq
        rans_output = bytearray()
        freq_tables = {}  # L → (freq_array, start_array)

        # Collect all symbols per level for frequency estimation
        level_symbols = {}
        for L, syms in all_symbols:
            if L not in level_symbols:
                level_symbols[L] = []
            level_symbols[L].extend(syms.tolist())

        for L in sorted(level_symbols.keys()):
            syms = np.array(level_symbols[L], dtype=np.int64)
            counts = np.bincount(syms, minlength=L)
            freq = np.maximum(1, (counts / max(counts.sum(), 1) * rans_total).astype(np.int32))
            freq[np.argmax(freq)] += rans_total - freq.sum()
            start = np.zeros(L, dtype=np.int32)
            for i in range(1, L):
                start[i] = start[i - 1] + freq[i - 1]
            freq_tables[L] = (freq, start)

        # rANS encode all symbols as one stream, grouped by segment
        # Encode in reverse order (rANS is LIFO)
        RANS_L = 256 * rans_total
        state = RANS_L
        byte_stream = bytearray()

        # Flatten all symbols in temporal order, then reverse for rANS
        flat_symbols = []
        flat_levels = []
        for t in range(lat_T):
            L, syms = all_symbols[t]
            for s in syms:
                flat_symbols.append(int(s))
                flat_levels.append(L)

        for idx in range(len(flat_symbols) - 1, -1, -1):
            sym = flat_symbols[idx]
            L = flat_levels[idx]
            freq, start = freq_tables[L]
            f = int(freq[sym])
            s = int(start[sym])
            threshold = ((RANS_L // rans_total) * f) << 8
            while state >= threshold:
                byte_stream.append(state & 0xFF)
                state >>= 8
            state = (state // f) * rans_total + (state % f) + s

        for _ in range(4):
            byte_stream.append(state & 0xFF)
            state >>= 8

        # LPC payload
        lpc_payload = bytearray()
        if lpc_coeffs is not None:
            lpc_q15 = np.round(np.clip(lpc_coeffs.astype(np.float64).flatten(),
                                        -1.0, 1.0) * 32767.0).astype(np.int16)
            deltas = np.diff(lpc_q15, prepend=0).astype(np.int16)
            lpc_payload = bytearray(deltas.tobytes())

        # Schedule payload
        schedule_payload = bytearray()
        for L_val, count in runs:
            schedule_payload.append(L_val & 0xFF)
            schedule_payload.append(count & 0xFF)

        # Freq table payload (one table per distinct L, ordered)
        freq_payload = bytearray()
        sorted_levels = sorted(freq_tables.keys())
        for L in sorted_levels:
            freq, _ = freq_tables[L]
            freq_payload.append(L & 0xFF)  # which L this table is for
            freq_payload.extend(freq.astype(np.uint16).tobytes())

        # Header
        header = struct.pack('<4sBHHffII',
                             b'LMQ3',
                             len(runs),           # n_runs
                             lat_dim, lat_T,
                             vmin, vmax,
                             len(byte_stream),     # rans length
                             len(lpc_payload))     # lpc length

        return (bytes(header) + bytes(schedule_payload) +
                bytes(freq_payload) + bytes(byte_stream) + bytes(lpc_payload))

    def decompress_adaptive(self, data):
        """Decompress LMQ3 adaptive packet."""
        # Parse header (25 bytes)
        hdr_size = 4 + 1 + 2 + 2 + 4 + 4 + 4 + 4  # 25
        (magic, n_runs, lat_dim, lat_T, vmin, vmax,
         rans_len, lpc_len) = struct.unpack('<4sBHHffII', data[:hdr_size])
        if magic != b'LMQ3':
            raise ValueError(f"Invalid LMQ3 header: expected b'LMQ3', got {magic!r}")
        pos = hdr_size

        span = vmax - vmin + 1e-8

        # Parse schedule
        runs = []
        for _ in range(n_runs):
            L_val, count = data[pos], data[pos + 1]
            runs.append((L_val, count))
            pos += 2

        # Expand schedule to per-timestep levels
        level_schedule = []
        for L_val, count in runs:
            level_schedule.extend([L_val] * count)
        if len(level_schedule) != lat_T:
            raise ValueError(
                f"Schedule length mismatch: expanded {len(level_schedule)} "
                f"timesteps but header declares lat_T={lat_T}")

        # Parse freq tables
        freq_tables = {}
        # Read until we've consumed all distinct levels from the schedule
        distinct_levels = sorted(set(level_schedule))
        for _ in range(len(distinct_levels)):
            L = data[pos]
            pos += 1
            freq = np.frombuffer(data[pos:pos + L * 2], dtype=np.uint16).astype(np.int32).copy()
            pos += L * 2
            start = np.zeros(L, dtype=np.int32)
            for i in range(1, L):
                start[i] = start[i - 1] + freq[i - 1]
            # Build lookup
            total = int(freq.sum())
            cum2sym = np.zeros(total, dtype=np.int32)
            for s in range(L):
                cum2sym[start[s]:start[s] + freq[s]] = s
            freq_tables[L] = (freq, start, cum2sym)

        # rANS decode
        rans_total = self.rans_total_freq
        RANS_L = 256 * rans_total
        rans_payload = list(data[pos:pos + rans_len])
        pos += rans_len

        byte_idx = len(rans_payload) - 1
        state = 0
        for _ in range(4):
            if byte_idx >= 0:
                state = (state << 8) | rans_payload[byte_idx]
                byte_idx -= 1

        # Decode all symbols
        n_total = lat_dim * lat_T
        flat_levels = []
        for t in range(lat_T):
            flat_levels.extend([level_schedule[t]] * lat_dim)

        symbols = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            L = flat_levels[i]
            freq, start, cum2sym = freq_tables[L]
            slot = state % rans_total
            sym = int(cum2sym[slot])
            f = int(freq[sym])
            s_val = int(start[sym])
            state = f * (state // rans_total) + slot - s_val
            while state < RANS_L and byte_idx >= 0:
                state = (state << 8) | rans_payload[byte_idx]
                byte_idx -= 1
            symbols[i] = sym

        # Dequantize per timestep
        latent = np.zeros((lat_dim, lat_T), dtype=np.float32)
        for t in range(lat_T):
            L = level_schedule[t]
            syms = symbols[t * lat_dim:(t + 1) * lat_dim]
            latent[:, t] = vmin + (syms.astype(np.float32) + 0.5) / L * span

        latent = torch.from_numpy(latent).unsqueeze(0)  # [1, 32, 79]

        lpc_bytes = data[pos:pos + lpc_len] if lpc_len > 0 else b''
        return latent, level_schedule, lpc_bytes

    def compress_to_packet(self, latent, l3_signal, quality_mode=2,
                           lpc_coeffs=None, subbands_per_ch=None,
                           level_schedule=None):
        """Compress and return an EEGPacket (neural mode).

        Args:
            latent: [1, 32, 79] encoded latent
            l3_signal: [21, 313] L3 approximation (for decode shape)
            quality_mode: 0/1/2
            lpc_coeffs: optional LPC coefficients
            subbands_per_ch: optional detail subbands
            level_schedule: if provided, uses adaptive FSQ (LMQ3)
        """
        if level_schedule is not None:
            compressed = self.compress_adaptive(latent, level_schedule,
                                                lpc_coeffs=lpc_coeffs)
            lat_dec, sched_dec, _ = self.decompress_adaptive(compressed)
        else:
            compressed = self.compress(latent, lpc_coeffs=lpc_coeffs,
                                       subbands_per_ch=subbands_per_ch,
                                       quality_mode=quality_mode)
            lat_dec, _, _, _ = self.decompress(compressed)

        with torch.no_grad():
            recon_l3 = self.model.decode(lat_dec, target_len=313, quantize=True)
        recon_np = recon_l3[0].numpy()

        return EEGPacket.from_reconstruction(
            signal=recon_np,
            compressed_bytes=len(compressed),
            mode='neural',
            metadata={
                'quality_mode': quality_mode,
                'adaptive': level_schedule is not None,
                'latent_shape': list(latent.shape),
            },
        )



# ============================================================
# Mode 3: Lossless DSP Codec
# ============================================================
# Moved to lamquant_codec/lossless.py — the wire format (LMQ v4) and
# all KLT / lifting helpers now live there. We re-export LosslessCodec
# here so existing `from lamquant_codec.codec import LosslessCodec`
# imports keep working.

from lamquant_codec.lossless import (  # noqa: E402, F401
    LosslessCodec,
    compute_klt,
    compute_lifting_rotations,
    apply_lifting_klt_forward,
    apply_lifting_klt_inverse,
    LIFT_PREC,
    LIFT_HALF,
)
