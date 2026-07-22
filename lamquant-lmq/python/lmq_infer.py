#!/usr/bin/env python3
"""ADR 0074 Track N — the PyBackend inference helper.

Reads ONE JSON request from stdin, writes ONE JSON response to stdout. Driven by
`lamquant-lmq`'s `PyBackend` over a subprocess (the only Rust->Python precedent in
this repo). Two modes:

  * "selftest" — a deterministic, weightless transform (each sample -> residue mod
    L, and back). Proves the subprocess bridge + JSON protocol WITHOUT any model or
    weights. This is what the Rust `py_backend` unit test exercises.

  * "model" — the real Gen-7.6 `SubbandCodec` (codec-neural). Requires the
    `lamquant_neural` + `lamquant_codec` packages importable and a checkpoint
    resolvable via $LAMQUANT_WEIGHTS_DIR. This path is env-gated: the Rust
    integration gate SKIPS it when the environment/weights are absent.

Protocol (all arrays are plain JSON numbers):
  encode  req : {op:"encode", mode, sample_rate, signal:[[i64]...]}
          resp: {tokens:[i32], schedule:[u8], alphabet, n_channels, n_samples,
                 backend_meta:[u8]}
  decode  req : {op:"decode", mode, tokens, schedule, alphabet, n_channels,
                 n_samples, backend_meta:[u8]}
          resp: {signal:[[i64]...]}
"""
import json
import hashlib
import sys

SELFTEST_ALPHABET = 5


def _to_jsonable(obj):
    """Recursively convert numpy arrays/scalars into JSON-safe values, so the
    backend metadata can be carried as JSON (NEVER pickle — backend_meta round-trips
    through the untrusted .lmq wire, and pickle.loads is arbitrary code execution)."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _from_jsonable(obj):
    """Inverse of _to_jsonable."""
    import numpy as np

    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=obj["dtype"])
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(x) for x in obj]
    return obj


def selftest_encode(signal, _sample_rate):
    """Deterministic weightless quantizer: sample -> residue mod L, channel-major."""
    l = SELFTEST_ALPHABET
    n_channels = len(signal)
    n_samples = len(signal[0]) if signal else 0
    tokens = [int(s % l) for ch in signal for s in ch]
    return {
        "tokens": tokens,
        "schedule": [l] * n_samples,
        "alphabet": l,
        "n_channels": n_channels,
        "n_samples": n_samples,
        # No model state to carry; a couple of bytes prove the meta round-trips.
        "backend_meta": [0x53, 0x54],  # 'ST'
    }


def selftest_decode(req):
    """Inverse of selftest_encode: reshape the residues back to [n_ch][n_samples]."""
    n_ch = int(req["n_channels"])
    n_s = int(req["n_samples"])
    tokens = req["tokens"]
    signal = [[int(tokens[c * n_s + i]) for i in range(n_s)] for c in range(n_ch)]
    return {"signal": signal}


def _load_bound_model(req):
    """Resolve once, bind the exact checkpoint bytes, then load that path."""
    from lamquant_neural.codec import SubbandCodec, _resolve_checkpoint

    checkpoint_path = _resolve_checkpoint(None, "student_subband.ckpt")
    with open(checkpoint_path, "rb") as checkpoint:
        actual_sha256 = hashlib.sha256(checkpoint.read()).hexdigest()
    expected_sha256 = req.get("expected_checkpoint_sha256")
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "checkpoint provenance mismatch: "
            f"expected {expected_sha256!r}, loaded {actual_sha256}"
        )
    return SubbandCodec.from_checkpoint(checkpoint_path), actual_sha256


def model_encode(req):
    """Drive the real SubbandCodec (env-gated). Returns integer FSQ tokens + the
    per-channel preprocessing metadata (serialized into backend_meta) the decoder
    needs. Raises if the codec-neural env / weights are unavailable — the Rust gate
    treats a non-zero exit as SKIP when it detects a missing environment."""
    import numpy as np
    import torch
    signal = req["signal"]
    codec, checkpoint_sha256 = _load_bound_model(req)
    x = torch.tensor(np.asarray(signal, dtype=np.float32)).unsqueeze(0)  # [1, C, T]
    latent, metadata = codec.encode(x)  # latent [1, 32, 79] float, metadata list
    l = 32  # CLINICAL FSQ level (FSQ_LEVELS_BY_MODE[2])
    lat = latent.detach().cpu().numpy()[0]  # [32, 79]
    vmin, vmax = float(lat.min()), float(lat.max())
    norm = (lat - vmin) / (vmax - vmin + 1e-8)
    toks = np.clip((norm * l).astype(np.int32), 0, l - 1).reshape(-1)
    # Carry vmin/vmax + the metadata as JSON bytes (never pickle) so decode inverts
    # exactly — safe against a crafted .lmq (backend_meta is untrusted on decode).
    meta_bytes = json.dumps(
        _to_jsonable({"vmin": vmin, "vmax": vmax, "shape": [int(s) for s in lat.shape], "metadata": metadata})
    ).encode("utf-8")
    return {
        "tokens": [int(t) for t in toks],
        "schedule": [l] * lat.shape[1],
        "alphabet": l,
        "n_channels": len(signal),
        "n_samples": len(signal[0]) if signal else 0,
        "backend_meta": list(meta_bytes),
        "checkpoint_sha256": checkpoint_sha256,
    }


def model_decode(req):
    import numpy as np
    import torch
    codec, checkpoint_sha256 = _load_bound_model(req)
    meta = _from_jsonable(json.loads(bytes(req["backend_meta"]).decode("utf-8")))
    l = int(req["alphabet"])
    shape = meta["shape"]
    toks = np.asarray(req["tokens"], dtype=np.float32).reshape(shape)
    norm = (toks + 0.5) / l
    lat = norm * (meta["vmax"] - meta["vmin"]) + meta["vmin"]
    latent = torch.tensor(lat).unsqueeze(0)
    recon = codec.decode(latent, meta["metadata"])  # [1, C, T]
    sig = recon.detach().cpu().numpy()[0]
    return {
        "signal": [[int(round(v)) for v in ch] for ch in sig],
        "checkpoint_sha256": checkpoint_sha256,
    }


def main():
    try:
        req = json.load(sys.stdin)
        op, mode = req["op"], req.get("mode", "model")
        if mode == "selftest":
            resp = selftest_encode(req["signal"], req["sample_rate"]) if op == "encode" else selftest_decode(req)
        elif op == "encode":
            resp = model_encode(req)
        else:
            resp = model_decode(req)
        json.dump(resp, sys.stdout)
    except Exception:
        # Full traceback → stderr, non-zero exit → the Rust side reports it as a
        # BackendError with this stderr attached (debuggable, never a silent hang).
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
