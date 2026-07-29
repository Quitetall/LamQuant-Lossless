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
    resolvable via $LAMQUANT_WEIGHTS_DIR. Developer runs may skip this path when
    dependencies are absent; `LAMQUANT_LMQ_REQUIRE_MODEL_TEST=1` makes missing
    configuration, dependencies, weights, or inference a gate failure.

Protocol (all arrays are plain JSON numbers):
  encode  req : {op:"encode", mode, sample_rate, signal_domain, signal:[[i64]...]}
          resp: {tokens:[i32], schedule:[u8], alphabet, n_channels, n_samples,
                 backend_meta:[u8]}
  decode  req : {op:"decode", mode, signal_domain, tokens, schedule, alphabet,
                 n_channels, n_samples, backend_meta:[u8]}
          resp: {signal:[[i64]...]}
"""
import json
import hashlib
import math
import sys

SELFTEST_ALPHABET = 5
DIGITAL_DOMAIN = "digital-integer"
MODEL_DOMAIN = "physical-microvolt-q16"
MICROVOLT_Q16 = 65_536.0
MODEL_MAX_METADATA_ARRAY_ELEMENTS = 21 * (2_500 + 8)


def _reject_nonfinite_json_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _require_finite_json(value, path="value"):
    """Reject non-finite binary64 values, including exponent overflow accepted
    by Python's JSON parser (for example `1e10000`)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{path}[{index}]")


def _require_signal_domain(actual, expected, backend):
    if actual != expected:
        raise ValueError(f"{backend} requires {expected}, got {actual!r}")


def _new_array_budget():
    return {"remaining_elements": MODEL_MAX_METADATA_ARRAY_ELEMENTS}


def _to_jsonable(obj, array_budget=None):
    """Recursively convert numpy arrays/scalars into JSON-safe values, so the
    backend metadata can be carried as JSON (NEVER pickle — backend_meta round-trips
    through the untrusted .lmq wire, and pickle.loads is arbitrary code execution)."""
    import numpy as np

    if array_budget is None:
        array_budget = _new_array_budget()
    if isinstance(obj, np.ndarray):
        if obj.dtype != np.dtype("float64"):
            raise ValueError(f"unsupported model metadata dtype {obj.dtype!s}")
        if obj.size > array_budget["remaining_elements"]:
            raise ValueError("model metadata arrays exceed element budget")
        array_budget["remaining_elements"] -= int(obj.size)
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v, array_budget) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x, array_budget) for x in obj]
    return obj


def _array_leaf_count(value):
    stack = [value]
    count = 0
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            count += 1
        else:
            raise ValueError("model metadata ndarray contains a non-numeric value")
    return count


def _from_jsonable(obj, array_budget=None):
    """Inverse of _to_jsonable."""
    import numpy as np

    if array_budget is None:
        array_budget = _new_array_budget()
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            if set(obj) != {"__ndarray__", "dtype"}:
                raise ValueError("malformed model metadata ndarray envelope")
            if obj["dtype"] != "float64":
                raise ValueError(f"unsupported model metadata dtype {obj['dtype']!r}")
            count = _array_leaf_count(obj["__ndarray__"])
            if count > array_budget["remaining_elements"]:
                raise ValueError("model metadata arrays exceed element budget")
            array_budget["remaining_elements"] -= count
            return np.asarray(obj["__ndarray__"], dtype=np.float64)
        return {k: _from_jsonable(v, array_budget) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(x, array_budget) for x in obj]
    return obj


def selftest_encode(signal, _sample_rate, signal_domain):
    """Deterministic weightless quantizer: sample -> residue mod L, channel-major."""
    _require_signal_domain(signal_domain, DIGITAL_DOMAIN, "selftest")
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
    _require_signal_domain(req.get("signal_domain"), DIGITAL_DOMAIN, "selftest")
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
    """Drive the real SubbandCodec. Returns integer FSQ tokens + the
    per-channel preprocessing metadata (serialized into backend_meta) the decoder
    needs. Raises if the codec-neural environment, weights, or inference are
    unavailable. Rust decides whether that failure is an optional developer skip
    or a required gate failure."""
    import numpy as np
    import torch
    _require_signal_domain(req.get("signal_domain"), MODEL_DOMAIN, "model")
    signal = req["signal"]
    codec, checkpoint_sha256 = _load_bound_model(req)
    # Rust shell already applied exact ABIR calibration. Integer transport keeps
    # JSON deterministic and avoids binary64 text becoming a second unit seam.
    physical_microvolts_f64 = np.asarray(signal, dtype=np.float64) / MICROVOLT_Q16
    if not np.isfinite(physical_microvolts_f64).all():
        raise ValueError("model input contains non-finite physical samples")
    physical_microvolts = physical_microvolts_f64.astype(np.float32)
    if not np.isfinite(physical_microvolts).all():
        raise ValueError("model input overflows float32 physical samples")
    x = torch.tensor(physical_microvolts).unsqueeze(0)  # [1, C, T]
    latent, metadata = codec.encode(x)  # latent [1, 32, 79] float, metadata list
    l = 32  # CLINICAL FSQ level (FSQ_LEVELS_BY_MODE[2])
    lat = latent.detach().cpu().numpy()[0]  # [32, 79]
    if not np.isfinite(lat).all():
        raise ValueError("model latent contains non-finite values")
    vmin, vmax = float(lat.min()), float(lat.max())
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError("model latent extrema are non-finite")
    norm = (lat - vmin) / (vmax - vmin + 1e-8)
    if not np.isfinite(norm).all():
        raise ValueError("model latent normalization contains non-finite values")
    quantized = norm * l
    if not np.isfinite(quantized).all():
        raise ValueError("model quantization contains non-finite values")
    toks = np.clip(quantized.astype(np.int32), 0, l - 1).reshape(-1)
    # Carry vmin/vmax + the metadata as JSON bytes (never pickle) so decode inverts
    # exactly — safe against a crafted .lmq (backend_meta is untrusted on decode).
    meta_bytes = json.dumps(
        _to_jsonable(
            {
                "vmin": vmin,
                "vmax": vmax,
                "shape": [int(s) for s in lat.shape],
                "metadata": metadata,
            }
        ),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "tokens": toks.tolist(),
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
    _require_signal_domain(req.get("signal_domain"), MODEL_DOMAIN, "model")
    codec, checkpoint_sha256 = _load_bound_model(req)
    metadata_json = json.loads(
        bytes(req["backend_meta"]).decode("utf-8"),
        parse_constant=_reject_nonfinite_json_constant,
    )
    _require_finite_json(metadata_json, "backend_meta")
    meta = _from_jsonable(metadata_json)
    l = int(req["alphabet"])
    shape = meta["shape"]
    toks = np.asarray(req["tokens"], dtype=np.float32).reshape(shape)
    norm = (toks + 0.5) / l
    lat = norm * (meta["vmax"] - meta["vmin"]) + meta["vmin"]
    if not np.isfinite(lat).all():
        raise ValueError("model latent reconstruction contains non-finite values")
    latent = torch.tensor(lat).unsqueeze(0)
    recon = codec.decode(latent, meta["metadata"])  # [1, C, T]
    sig = recon.detach().cpu().numpy()[0]
    scaled = np.rint(sig.astype(np.float64) * MICROVOLT_Q16)
    if not np.isfinite(scaled).all():
        raise ValueError("model reconstruction contains non-finite samples")
    # `float(np.iinfo(np.int64).max)` rounds upward to 2**63, so comparing
    # against `info.max` would admit that exclusive upper bound and let NumPy
    # wrap it to i64::MIN during `astype`. Use exact power-of-two boundaries.
    if (scaled < -(2**63)).any() or (scaled >= 2**63).any():
        raise OverflowError("model reconstruction exceeds Q47.16 microvolt range")
    signal_q16 = scaled.astype(np.int64)
    return {
        "signal": signal_q16.tolist(),
        "checkpoint_sha256": checkpoint_sha256,
    }


def main():
    try:
        req = json.load(sys.stdin, parse_constant=_reject_nonfinite_json_constant)
        _require_finite_json(req, "request")
        op, mode = req["op"], req.get("mode", "model")
        if mode == "selftest":
            resp = (
                selftest_encode(
                    req["signal"], req["sample_rate"], req.get("signal_domain")
                )
                if op == "encode"
                else selftest_decode(req)
            )
        elif op == "encode":
            resp = model_encode(req)
        else:
            resp = model_decode(req)
        json.dump(resp, sys.stdout, allow_nan=False)
    except Exception:
        # Full traceback → stderr, non-zero exit → the Rust side reports it as a
        # BackendError with this stderr attached (debuggable, never a silent hang).
        import traceback

        traceback.print_exc()
        # EX_USAGE-style reserved code: Rust treats this as a deterministic
        # model/helper rejection. Signals and other exit codes remain process
        # failures and retain distinct retry/operations semantics.
        sys.exit(64)


if __name__ == "__main__":
    main()
