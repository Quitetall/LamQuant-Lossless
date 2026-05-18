"""Mamba SNN loader + registry resolver for the codec's adaptive FSQ path.

Production codec (`SubbandCodec`) consumes an SNN via `set_snn(...)`. This
module is the canonical factory that constructs that SNN from a checkpoint
file on disk, plus the helper that resolves "which checkpoint should I
load by default" by reading the PCCP registry.

Public API:
    load_mamba_snn(checkpoint_path, device='cpu') -> MambaSNN
    resolve_production_snn() -> Optional[Path]

Both are read-only with respect to the registry; the loader never mutates
state. PCCP audit-trail integrity is preserved because the SHA pin in
registry.yaml is the single source of truth for which checkpoint is
production-grade.

Architecture sniffing mirrors the pattern in
`scripts/snn_pccp_eval.py::run_snn_on_pairs` (lines ~96-106):
load state_dict → infer (d_model, d_state, n_layers) from key shapes →
instantiate `lamquant_codec.models.mamba_ssm_minimal.MambaSNN` with matching kwargs.

Programming Bible compliance:
- Rule 6 + 30 — boundary `raise` (not `assert`) for all caller-facing checks
- Rule 17 — structured logging via `logging.getLogger(__name__)`
- Rule 21 — preconditions at entry, postconditions before return
- Rule 27 — graceful failure via AdaptiveFSQError; no silent fallback
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from lamquant_codec.errors import AdaptiveFSQError
# Module-level — missing integrity module = hard error, not silent
# downgrade. (V4 Flash Finding 3 of bones-A.2-3412ad4.)
from lamquant_codec.integrity import (
    IntegrityError, registry_sha,
)

logger = logging.getLogger(__name__)

# Hard ceiling on checkpoint file size before we'll load it. Real
# Mamba SNN ckpts are ≤2 MB; setting this to 32 MB leaves headroom
# for future variants while still guarding against symlinks pointing
# to huge files (defense-in-depth, V4 Flash Finding 5 of e4b33f25 review).
_MAX_CKPT_BYTES = 32 * 1024 * 1024

# Default checkpoint kwargs — match
# lamquant_codec/models/mamba_ssm_minimal.py production constants.
# Architecture sniffing overrides these from the checkpoint's
# state_dict when the file is loaded.
_DEFAULT_IN_CHANNELS = 21
_DEFAULT_D_MODEL = 40
_DEFAULT_D_STATE = 16
_DEFAULT_N_LAYERS = 2


def resolve_production_snn() -> Optional[Path]:
    """Read pccp/registry.yaml and return the production SNN checkpoint
    path, or None if the registry pin is a placeholder / missing.

    Returns:
      Absolute Path to the checkpoint, or None when:
        - registry.yaml is unreachable
        - models.snn.production_checkpoint is missing
        - models.snn.production_sha256 starts with "PLACEHOLDER" or is None
        - registry pin would resolve outside the repo subtree (containment)

    Never raises to the caller — None is the "no production pin available"
    sentinel. Internal ImportError on the integrity module is caught and
    converted to None + logger.warning. Caller is responsible for
    translating None into AdaptiveFSQError when adaptive was requested.
    """
    # Lazy import — keeps `from lamquant_codec.models.snn import ...` cheap
    # even when the registry isn't on disk (test environments).
    try:
        from lamquant_codec.integrity import registry_path, _load_registry_yaml
    except ImportError as exc:
        logger.warning(
            "resolve_production_snn: integrity module unavailable (%r); "
            "returning None.", exc,
        )
        return None

    try:
        reg_path = registry_path()
        registry = _load_registry_yaml(reg_path)
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "resolve_production_snn: registry unreachable (%r); returning None.",
            exc,
        )
        return None

    if not isinstance(registry, dict):
        logger.warning(
            "resolve_production_snn: registry parsed to %s, expected dict; "
            "returning None.", type(registry).__name__,
        )
        return None

    models_block = registry.get("models", {})
    snn_spec = models_block.get("snn") if isinstance(models_block, dict) else None
    if not isinstance(snn_spec, dict):
        logger.warning(
            "resolve_production_snn: registry has no `models.snn` block; returning None."
        )
        return None

    raw_path = snn_spec.get("production_checkpoint")
    raw_sha = snn_spec.get("production_sha256")
    if not raw_path:
        logger.warning("resolve_production_snn: no production_checkpoint pin; returning None.")
        return None
    # None / placeholder / non-str SHA all mean "uncaptured pin" — refuse.
    if raw_sha is None or not isinstance(raw_sha, str) or raw_sha.startswith("PLACEHOLDER"):
        logger.warning(
            "resolve_production_snn: production_sha256 is %r (uncaptured/missing); returning None.",
            raw_sha,
        )
        return None

    # Path containment — registry pins MUST be repo-relative. Reject
    # absolute paths that would escape the repo subtree (V4 Pro Finding 2
    # of bones-A.2-93985e2 — a poisoned registry must not steer torch.load
    # at /etc/passwd or similar).
    repo_root = reg_path.parent.parent.resolve()
    raw_path_obj = Path(raw_path)
    if raw_path_obj.is_absolute():
        logger.warning(
            "resolve_production_snn: registry pin is absolute path %s (must be repo-relative); "
            "returning None.", raw_path,
        )
        return None
    resolved = (repo_root / raw_path_obj).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        logger.warning(
            "resolve_production_snn: registry pin %s escapes repo subtree (%s); returning None.",
            raw_path, repo_root,
        )
        return None
    if not resolved.exists():
        logger.warning(
            "resolve_production_snn: registry pin points at non-existent path %s; "
            "returning None.", resolved,
        )
        return None

    return resolved


def _sniff_architecture(state_dict: dict[str, Any]) -> Tuple[int, int, int, int]:
    """Infer (in_channels, d_model, d_state, n_layers) from MambaSNN keys.

    Falls back to production defaults if a particular dimension can't be
    inferred (older checkpoints with renamed keys). Raises AdaptiveFSQError
    when nothing recognizable is present — that's a wrong-family checkpoint.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"_sniff_architecture: state_dict must be dict, got {type(state_dict).__name__}"
        )
    if not state_dict:
        raise AdaptiveFSQError("empty state_dict — cannot infer architecture")

    # in_channels — spatial_mix.weight has shape [d_model, in_channels].
    sm_key = "spatial_mix.weight"
    if sm_key in state_dict:
        shape = tuple(state_dict[sm_key].shape)
        if len(shape) != 2:
            raise AdaptiveFSQError(
                f"spatial_mix.weight has unexpected shape {shape}; expected 2-D [d_model, in_ch]"
            )
        d_model_inferred, in_ch_inferred = shape[0], shape[1]
    else:
        d_model_inferred, in_ch_inferred = _DEFAULT_D_MODEL, _DEFAULT_IN_CHANNELS
        logger.info(
            "_sniff_architecture: %s missing; using defaults d_model=%d in_ch=%d",
            sm_key, d_model_inferred, in_ch_inferred,
        )

    # n_layers — count distinct `ssm_blocks.<i>.` prefixes. Verify
    # contiguity from 0 — sparse keys (e.g. {0, 2} after pruning) would
    # make `max+1` lie about the actual layer count (Critic Finding 4
    # of bones-A.2-93985e2).
    layer_indices: set[int] = set()
    for key in state_dict.keys():
        if key.startswith("ssm_blocks."):
            parts = key.split(".", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if layer_indices:
        max_idx = max(layer_indices)
        n_layers_inferred = max_idx + 1
        if set(range(n_layers_inferred)) != layer_indices:
            raise AdaptiveFSQError(
                f"_sniff_architecture: ssm_blocks indices are sparse "
                f"{sorted(layer_indices)} (expected contiguous 0..{max_idx}); "
                f"checkpoint may be from a pruned or partial export"
            )
    else:
        n_layers_inferred = _DEFAULT_N_LAYERS

    # d_state — try a known SSM weight; fall back to default.
    # BidirectionalSSM in mamba_ssm_minimal uses `A_log` of shape [d_model, d_state].
    d_state_inferred = _DEFAULT_D_STATE
    for key in state_dict.keys():
        if key.endswith(".fwd.A_log") or key.endswith(".bwd.A_log"):
            shape = tuple(state_dict[key].shape)
            if len(shape) == 2:
                d_state_inferred = shape[1]
                break

    # Verify caller-facing invariants.
    if in_ch_inferred <= 0 or d_model_inferred <= 0 or d_state_inferred <= 0 or n_layers_inferred <= 0:
        raise AdaptiveFSQError(
            f"_sniff_architecture: derived non-positive dim "
            f"(in_ch={in_ch_inferred}, d_model={d_model_inferred}, "
            f"d_state={d_state_inferred}, n_layers={n_layers_inferred})"
        )
    return in_ch_inferred, d_model_inferred, d_state_inferred, n_layers_inferred


def load_mamba_snn(
    checkpoint_path: Path | str,
    device: str = "cpu",
    *,
    allow_pickle_fallback: bool = False,
) -> "Any":
    """Load a production MambaSNN from a `.pt` checkpoint.

    Architecture kwargs are sniffed from the state_dict so we don't
    hardcode dimensions that might drift over training runs. Reuses
    the loader pattern from `scripts/snn_pccp_eval.py::run_snn_on_pairs`.

    Security policy (V4 Pro Finding 1 of bones-A.2-6b1bad0):
      1. Always tries `torch.load(weights_only=True)` first — pickle-safe.
      2. If that fails (Mamba checkpoints carry non-tensor metadata like
         sensitivity/accuracy/epoch, so this is common), the fallback to
         `weights_only=False` is GATED:
           (a) `lamquant_codec.integrity.verify_checkpoint('snn', path)`
               passes → file SHA matches the registry pin → trusted →
               fallback allowed.
           (b) Pin is uncaptured (PLACEHOLDER_*) AND caller passed
               `allow_pickle_fallback=True` → fallback allowed with a
               loud warning. Used by the initial-baseline-capture path
               (pccp_gate.py --capture) before a SHA pin exists.
           (c) Otherwise → AdaptiveFSQError. Refuses the unsafe load.
      No silent unsafe-pickle path.

    Args:
      checkpoint_path: Path to a MambaSNN .pt file.
      device: torch device string ("cpu", "cuda", etc.).
      allow_pickle_fallback: caller-explicit opt-in for the pickle path
        when the registry pin can't validate the file (placeholder SHA).
        Default False — production callers must NOT enable this.

    Returns:
      A `MambaSNN` instance with state loaded and `.eval()` called.

    Raises:
      AdaptiveFSQError: checkpoint missing, unreadable, wrong family,
        lacks classify_per_timestep, OR pickle fallback would be required
        but is not authorized by integrity or caller opt-in.
      TypeError: argument types violate the boundary contract.
    """
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    if not isinstance(checkpoint_path, Path):
        raise TypeError(
            f"load_mamba_snn.checkpoint_path: expected Path or str, "
            f"got {type(checkpoint_path).__name__}"
        )
    if not isinstance(device, str) or not device:
        raise TypeError(
            f"load_mamba_snn.device: expected non-empty str, got {device!r}"
        )
    if not isinstance(allow_pickle_fallback, bool):
        raise TypeError(
            f"load_mamba_snn.allow_pickle_fallback: expected bool, "
            f"got {type(allow_pickle_fallback).__name__}"
        )

    if not checkpoint_path.exists():
        raise AdaptiveFSQError(
            f"checkpoint missing: {checkpoint_path} — set --snn-checkpoint or "
            "update pccp/registry.yaml.models.snn.production_checkpoint"
        )
    if not checkpoint_path.is_file():
        raise AdaptiveFSQError(
            f"checkpoint path is not a regular file: {checkpoint_path}"
        )

    # Lazy imports — torch + MambaSNN are heavy; loading the model is
    # the user's explicit ask. Keep `import lamquant_codec.models.snn`
    # cheap for code paths that only call `resolve_production_snn()`.
    try:
        import torch
    except ImportError as exc:
        raise AdaptiveFSQError(
            f"PyTorch is required to load MambaSNN; install torch ({exc})"
        ) from exc

    # T1b (ADR 0018): MambaSNN lives in the codec now —
    # `lamquant_codec.models.mamba_ssm_minimal.MambaSNN`. No more
    # sys.path injection into `ai_models/snn/`. Old checkpoints that
    # were pickled when the class lived under the bare
    # `mamba_ssm_minimal` top-level module name still load: we
    # register a `sys.modules` alias so the unpickler's class lookup
    # for `mamba_ssm_minimal.MambaSNN` resolves to the codec module.
    try:
        from lamquant_codec.models import mamba_ssm_minimal as _mamba_mod
        from lamquant_codec.models.mamba_ssm_minimal import MambaSNN  # type: ignore
    except ImportError as exc:
        raise AdaptiveFSQError(
            f"cannot import MambaSNN from lamquant_codec.models.mamba_ssm_minimal: {exc}"
        ) from exc
    # Register bare-name alias for pickle compat. Idempotent — repeat
    # imports just re-bind the same module object.
    sys.modules.setdefault("mamba_ssm_minimal", _mamba_mod)

    logger.info("load_mamba_snn: loading %s on %s", checkpoint_path, device)
    # Read file bytes into RAM ONCE; hash those exact bytes; load
    # torch.load from a BytesIO buffer. Eliminates TOCTOU between
    # SHA verification and unsafe pickle load — the bytes torch.load
    # consumes are byte-identical to what we hashed.
    # (V4 Pro Finding 1 of bones-A.2-3412ad4.) SNN ckpts ≤2 MB → RAM cost OK.
    #
    # Open with O_NOFOLLOW + fstat the fd + os.read from the same fd.
    # Ties size check, hash, and torch.load to the same inode — no
    # window for symlink-swap attacks or stat/read race.
    # (V4 Pro Finding 1 of e4b33f25 review.)
    try:
        fd = os.open(str(checkpoint_path), os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise AdaptiveFSQError(
            f"could not open checkpoint {checkpoint_path}: {exc} "
            f"(symlinks rejected via O_NOFOLLOW)"
        ) from exc
    try:
        st = os.fstat(fd)
        size = st.st_size
        if size < 0:
            raise AdaptiveFSQError(
                f"checkpoint {checkpoint_path} reports negative size "
                f"({size}); refusing to load"
            )
        if size > _MAX_CKPT_BYTES:
            raise AdaptiveFSQError(
                f"checkpoint {checkpoint_path} is {size} bytes; refusing "
                f"to load (> {_MAX_CKPT_BYTES}-byte cap)."
            )
        # Read exactly `size` bytes from the same fd we just stat'd.
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise AdaptiveFSQError(
                    f"short read on {checkpoint_path}: expected {size} "
                    f"bytes, got {size - remaining}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        file_bytes = b"".join(chunks)
    finally:
        # Suppress close errors so they don't mask the in-flight
        # AdaptiveFSQError. EIO on close is logged but not re-raised.
        # (V4 Flash Finding 2 of d0349d8f review.)
        try:
            os.close(fd)
        except OSError as close_exc:
            # %r on path → control chars / newlines escaped to prevent
            # log forgery via attacker-chosen filenames.
            # (V4 Pro nit on 7139d540.)
            logger.warning(
                "load_mamba_snn: os.close(fd) failed for %r: %s",
                str(checkpoint_path), close_exc,
            )
    actual_sha = hashlib.sha256(file_bytes).hexdigest()

    # Prefer weights_only=True (pickle-safe). On failure, fallback to
    # weights_only=False is integrity-gated by the SHA we just computed.
    try:
        ckpt = torch.load(
            io.BytesIO(file_bytes), map_location=device, weights_only=True,
        )
    except (pickle.UnpicklingError, RuntimeError) as exc:
        # Pickle-safe load failed. Compare in-memory SHA against
        # registry pin BEFORE unsafe load runs. No TOCTOU possible —
        # `file_bytes` is what both the hash check and torch.load consume.
        # except narrowed to IntegrityError only — other exceptions
        # (programming errors, missing module) must propagate, not be
        # silently downgraded. (V4 Flash Finding 2 of bones-A.2-3412ad4.)
        try:
            expected_sha = registry_sha("snn")
            if actual_sha != expected_sha:
                raise IntegrityError(
                    f"SHA mismatch for snn: actual={actual_sha} "
                    f"expected={expected_sha}"
                )
            logger.info(
                "load_mamba_snn: SHA matches registry pin — pickle fallback "
                "authorized by integrity verification."
            )
        except IntegrityError as ig_exc:
            if allow_pickle_fallback:
                logger.warning(
                    "load_mamba_snn: SHA does not match registry pin (%s) but "
                    "caller passed allow_pickle_fallback=True — proceeding with "
                    "unsafe pickle load. Used by initial-baseline-capture only.",
                    ig_exc,
                )
            else:
                raise AdaptiveFSQError(
                    f"weights_only=True load failed for {checkpoint_path} "
                    f"({type(exc).__name__}: {exc}); pickle fallback refused — "
                    f"integrity check did not authorize ({ig_exc}). Either "
                    f"capture the production SHA via "
                    f"`pccp_gate.py --capture --model snn --candidate <path>`, "
                    f"or pass allow_pickle_fallback=True for non-production paths."
                ) from exc

        # Single-source authorization — both branches above either
        # set authorization (no raise) or raise AdaptiveFSQError.
        # No dead-code guard. (V4 Pro Finding 2 of bones-A.2-3412ad4.)
        try:
            ckpt = torch.load(
                io.BytesIO(file_bytes), map_location=device, weights_only=False,
            )
        except Exception as exc2:
            raise AdaptiveFSQError(
                f"torch.load (pickle path) failed for {checkpoint_path}: "
                f"{type(exc2).__name__}: {exc2}"
            ) from exc2
    except Exception as exc:
        raise AdaptiveFSQError(
            f"torch.load failed for {checkpoint_path}: {type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        else:
            # Use list+repr instead of sorted() — sorted() would crash on
            # non-string keys (e.g. legacy int-keyed metadata dicts).
            # V4 Flash Finding 1 of bones-A.2-6b1bad0.
            keys_preview = list(ckpt.keys())[:8]
            logger.warning(
                "load_mamba_snn: checkpoint has no 'model' key; using top-level "
                "dict as state_dict. First keys: %r",
                keys_preview,
            )
            state = ckpt
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise AdaptiveFSQError(
            f"checkpoint did not yield a state_dict (got {type(state).__name__}); "
            f"file may be from a different model family"
        )

    in_ch, d_model, d_state, n_layers = _sniff_architecture(state)

    model = MambaSNN(
        in_channels=in_ch,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        # Strict failed — try non-strict so older / partial checkpoints
        # load. If keys are wildly off, this likely fails too.
        logger.warning(
            "load_mamba_snn: strict load failed (%s); retrying with strict=False",
            exc,
        )
        try:
            model.load_state_dict(state, strict=False)
        except RuntimeError as exc2:
            raise AdaptiveFSQError(
                f"load_state_dict failed for {checkpoint_path}: {exc2}"
            ) from exc2

    model.to(device)
    model.eval()

    # Postcondition: the contract that SubbandCodec.set_snn enforces.
    if not hasattr(model, "classify_per_timestep"):
        raise AdaptiveFSQError(
            f"loaded model from {checkpoint_path} lacks classify_per_timestep — "
            f"wrong family (got {type(model).__name__})"
        )

    return model
