#!/usr/bin/env python3
"""ADR 0074 Track N — the PyBackend inference helper.

Reads ONE JSON request from stdin, writes ONE JSON response to stdout. Driven by
`lamquant-lmq`'s `PyBackend` over a subprocess (the only Rust->Python precedent in
this repo). Two modes:

  * "selftest" — a deterministic, weightless transform (each sample -> residue mod
    L, and back). Proves the subprocess bridge + JSON protocol WITHOUT any model or
    weights. This is what the Rust `py_backend` unit test exercises.

  * "model" — the production encoder + fixed scalar FSQ + Vocos decoder bound
    by `codec_artifact_set.json`. Requires the `lamquant_neural` package and an
    artifact set resolvable via $LAMQUANT_WEIGHTS_DIR. BCS2 wire remains
    Rust-owned.
    Developer runs may skip this path when dependencies are absent;
    `LAMQUANT_LMQ_REQUIRE_MODEL_TEST=1` makes missing configuration,
    dependencies, weights, or inference a gate failure.

Protocol (all arrays are plain JSON numbers):
  encode  req : {op:"encode", mode, sample_rate, signal_domain, signal:[[i64]...]}
          resp: {tokens:[i32], schedule:[u8], alphabet, n_channels, n_samples,
                 backend_meta:[u8]}
  decode  req : {op:"decode", mode, signal_domain, tokens, schedule, alphabet,
                 n_channels, n_samples, backend_meta:[u8]}
          resp: {signal:[[i64]...]}
"""
import json
import math
import sys

SELFTEST_ALPHABET = 5
DIGITAL_DOMAIN = "digital-integer"
MODEL_DOMAIN = "physical-microvolt-q16"
MICROVOLT_Q16 = 65_536.0
MODEL_CHANNELS = 21
MODEL_SAMPLES = 2_500
MODEL_LATENT_CHANNELS = 32
MODEL_LATENT_FRAMES = 79
MODEL_FSQ_LEVEL = 32
MODEL_ALPHABET = 33


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
    """Resolve and load one exact encoder-decoder artifact set."""
    from lamquant_neural.artifact_set import ARTIFACT_SET_FILENAME
    from lamquant_neural.codec import ProductionSubbandCodec, _resolve_checkpoint

    expected_sha256 = req.get("expected_artifact_set_sha256")
    if not isinstance(expected_sha256, str):
        raise ValueError("expected artifact-set SHA-256 is required")
    artifact_set_path = _resolve_checkpoint(None, ARTIFACT_SET_FILENAME)
    codec = ProductionSubbandCodec.from_artifact_set(
        artifact_set_path,
        expected_sha256=expected_sha256,
    )
    return codec, codec.artifact_set.manifest_sha256


def model_encode(req):
    """Encode one exact production window into canonical unsigned FSQ symbols."""
    import numpy as np
    import torch
    _require_signal_domain(req.get("signal_domain"), MODEL_DOMAIN, "model")
    signal = req["signal"]
    if req.get("sample_rate") != 250.0:
        raise ValueError("production model requires 250 Hz input")
    codec, artifact_set_sha256 = _load_bound_model(req)
    # Rust shell already applied exact ABIR calibration. Integer transport keeps
    # JSON deterministic and avoids binary64 text becoming a second unit seam.
    physical_microvolts_f64 = np.asarray(signal, dtype=np.float64) / MICROVOLT_Q16
    if not np.isfinite(physical_microvolts_f64).all():
        raise ValueError("model input contains non-finite physical samples")
    physical_microvolts = physical_microvolts_f64.astype(np.float32)
    if not np.isfinite(physical_microvolts).all():
        raise ValueError("model input overflows float32 physical samples")
    x = torch.tensor(physical_microvolts).unsqueeze(0)  # [1, C, T]
    encoded = codec.encode(x)
    symbols = encoded.symbols.detach().cpu()
    if (
        tuple(symbols.shape) != (1, MODEL_LATENT_CHANNELS, MODEL_LATENT_FRAMES)
        or symbols.dtype not in (torch.int32, torch.int64)
        or encoded.level != MODEL_FSQ_LEVEL
        or encoded.alphabet != MODEL_ALPHABET
        or (symbols < 0).any().item()
        or (symbols >= MODEL_ALPHABET).any().item()
    ):
        raise ValueError("production codec returned an invalid FSQ envelope")
    return {
        "tokens": symbols.reshape(-1).tolist(),
        "schedule": [MODEL_FSQ_LEVEL] * MODEL_LATENT_FRAMES,
        "alphabet": MODEL_ALPHABET,
        "n_channels": len(signal),
        "n_samples": len(signal[0]) if signal else 0,
        "backend_meta": [],
        "artifact_set_sha256": artifact_set_sha256,
    }


def model_decode(req):
    import numpy as np
    import torch
    from lamquant_neural.codec import QuantizedLatent

    _require_signal_domain(req.get("signal_domain"), MODEL_DOMAIN, "model")
    tokens = req.get("tokens")
    if (
        req.get("alphabet") != MODEL_ALPHABET
        or req.get("n_channels") != MODEL_CHANNELS
        or req.get("n_samples") != MODEL_SAMPLES
        or req.get("backend_meta") != []
        or req.get("schedule") != [MODEL_FSQ_LEVEL] * MODEL_LATENT_FRAMES
        or not isinstance(tokens, list)
        or len(tokens) != MODEL_LATENT_CHANNELS * MODEL_LATENT_FRAMES
        or any(type(token) is not int for token in tokens)
        or any(token < 0 or token >= MODEL_ALPHABET for token in tokens)
    ):
        raise ValueError("production decode envelope differs from artifact set")
    codec, artifact_set_sha256 = _load_bound_model(req)
    symbols = torch.tensor(tokens, dtype=torch.int32).reshape(
        1, MODEL_LATENT_CHANNELS, MODEL_LATENT_FRAMES
    )
    recon = codec.decode(
        QuantizedLatent(
            symbols=symbols,
            level=MODEL_FSQ_LEVEL,
            alphabet=MODEL_ALPHABET,
        )
    )
    if (
        not isinstance(recon, torch.Tensor)
        or tuple(recon.shape) != (1, MODEL_CHANNELS, MODEL_SAMPLES)
        or not recon.is_floating_point()
        or not torch.isfinite(recon).all().item()
    ):
        raise ValueError("production decoder returned an invalid reconstruction")
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
        "artifact_set_sha256": artifact_set_sha256,
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
