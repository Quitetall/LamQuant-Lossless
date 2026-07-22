#!/usr/bin/env python3
"""Emit fail-closed independent-validator receipts for bounded Adapter profiles.

Each primary receipt conforms to ABIR ``adapter-validation-v1``.  It binds the
fixture, adapter revision, validator executable, authority artifact, exact
validation argv, raw output hashes, diagnostic counts, and outcome.  The
NWBInspector run is deliberately supporting parser evidence; only the PyNWB
namespace validator can promote the bounded NWB semantic profile.

This command proves only the committed single-fixture BIDS, DICOM, and NWB
profiles.  It does not claim edition-wide or broad first-class conformance.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple


REPO = Path(__file__).resolve().parents[1]
DEFAULT_NWB = (
    REPO
    / "crates"
    / "lamquant-standard-adapters"
    / "tests"
    / "fixtures"
    / "single_integer_timeseries.nwb"
)
DEFAULT_DICOM = (
    REPO
    / "lamquant-lossless"
    / "tests"
    / "fixtures"
    / "dicom"
    / "12lead_ecg.dcm"
)
DEFAULT_BIDS = (
    REPO
    / "crates"
    / "lamquant-standard-adapters"
    / "tests"
    / "fixtures"
    / "bids-single-edf-eeg"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
DEFAULT_PINS = REPO / "tools" / "standard_adapter_validator_pins_v1.json"
DEFAULT_PYTHON_ENVIRONMENT = (
    REPO / "tools" / "standard_adapter_python_environment_v1.json"
)
DEFAULT_ABIR = REPO.parent / "abir"


class Execution(NamedTuple):
    exit_code: int
    stdout: bytes
    stderr: bytes


class ValidationTimeout(OSError):
    """A bounded validator execution exceeded its declared timeout."""


def canonical_json(value: object) -> str:
    """Return the deterministic compact JSON form used for receipt bundles."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise OSError(f"not a regular symlink-free file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_resolved_file(path: Path) -> str:
    return _hash_file(path.resolve())


def _hash_tree(path: Path, domain: bytes) -> str:
    if not path.is_dir() or path.is_symlink():
        raise OSError(f"not a symlink-free directory: {path}")
    entries: list[tuple[str, Path]] = []
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise OSError(f"tree contains a symlink: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise OSError(f"tree contains a non-regular entry: {entry}")
        entries.append((entry.relative_to(path).as_posix(), entry))
    if not entries:
        raise OSError(f"tree contains no regular files: {path}")
    digest = hashlib.sha256(domain)
    for relative, entry in sorted(entries):
        encoded = relative.encode("utf-8")
        payload = entry.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def fixture_sha256(path: Path, kind: str) -> str:
    """Hash fixtures with the exact ABIR adapter-validation algorithm."""

    if kind == "file":
        return _hash_file(path)
    if kind == "tree":
        return _hash_tree(path, b"abir.adapter.fixture-tree.v1\0")
    raise ValueError(f"unsupported fixture kind: {kind}")


def authority_sha256(path: Path) -> str:
    """Hash a validator schema, dictionary, bundle, or package artifact."""

    if path.is_file() and not path.is_symlink():
        return _hash_file(path)
    return _hash_tree(path, b"abir.adapter.authority-tree.v1\0")


def run(argv: list[str], *, cwd: Path, timeout_seconds: float = 300.0) -> Execution:
    """Run one exact argv without shell interpretation or output normalization."""

    environment = os.environ.copy()
    for name in (
        "DYLD_INSERT_LIBRARIES",
        "HDF5_DRIVER",
        "HDF5_EXT_PREFIX",
        "HDF5_PLUGIN_PATH",
        "HDF5_VOL_CONNECTOR",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValidationTimeout(
            f"validator exceeded {timeout_seconds:g} seconds"
        ) from error
    return Execution(completed.returncode, completed.stdout, completed.stderr)


def _combined(stdout: bytes, stderr: bytes) -> list[str]:
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines()]


def classify_bids(exit_code: int, stdout: bytes, stderr: bytes) -> tuple[str, int, int]:
    lines = _combined(stdout, stderr)
    errors = sum(line.startswith("[ERROR]") for line in lines)
    warnings = sum(line.startswith("[WARNING]") for line in lines)
    has_summary = any("Summary:" in line for line in lines)
    if exit_code != 0 or errors:
        return "reject", errors, warnings
    if has_summary:
        return "accept", 0, warnings
    return "indeterminate", 0, warnings


def classify_pynwb(exit_code: int, stdout: bytes, stderr: bytes) -> tuple[str, int, int]:
    lines = _combined(stdout, stderr)
    errors = sum(
        "error" in line.lower() and "no errors found" not in line.lower()
        for line in lines
    )
    warnings = sum("warning" in line.lower() for line in lines)
    no_errors = any("no errors found" in line.lower() for line in lines)
    if exit_code != 0 or errors:
        return "reject", errors, warnings
    if no_errors:
        return "accept", 0, warnings
    return "indeterminate", 0, warnings


def classify_nwbinspector(
    exit_code: int, stdout: bytes, stderr: bytes
) -> tuple[str, int, int]:
    lines = _combined(stdout, stderr)
    no_issues = any("No issues found!" in line for line in lines)
    errors = sum(line.startswith("CRITICAL") for line in lines)
    warnings = sum(
        line.startswith(("BEST_PRACTICE_VIOLATION", "PYNWB_WARNING"))
        for line in lines
    )
    if exit_code != 0 or errors:
        return "reject", errors, warnings
    if no_issues:
        return "accept", 0, warnings
    return "indeterminate", 0, warnings


def classify_dicom(exit_code: int, stdout: bytes, stderr: bytes) -> tuple[str, int, int]:
    lines = _combined(stdout, stderr)
    errors = sum(line.startswith("Error -") or " - Error - " in line for line in lines)
    warnings = sum(line.startswith("Warning -") or " - Warning - " in line for line in lines)
    recognized_iod = any("TwelveLeadECG" in line for line in lines)
    if exit_code != 0 or errors:
        return "reject", errors, warnings
    if recognized_iod:
        return "accept", 0, warnings
    return "indeterminate", 0, warnings


def _relative_fixture(path: Path, fixture_root: Path) -> str:
    root = fixture_root.resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"fixture is outside fixture root: {path}") from error
    if not relative or relative == ".":
        raise ValueError("fixture path must identify a child of fixture root")
    return relative


def make_receipt(
    *,
    profile: str,
    edition: str,
    adapter_revision: str,
    fixture: Path,
    fixture_root: Path,
    fixture_kind: str,
    expected_outcome: str,
    internal_valid: bool,
    validator_name: str,
    validator_version: str,
    validator_executable: Path,
    authority_artifact: Path,
    argv: list[str],
    execution: Execution,
    executed_at_utc: str,
    authority: str,
    observed_outcome: str,
    error_count: int,
    warning_count: int,
    semantic_profile: bool,
    fixture_hash: str | None = None,
    executable_hash: str | None = None,
    authority_hash: str | None = None,
    extra_diagnostics: list[str] | None = None,
) -> dict[str, object]:
    """Construct one derived adapter-validation-v1 receipt."""

    expected_observed = observed_outcome == expected_outcome
    internal_matches = internal_valid == (expected_outcome == "accept")
    passed = internal_matches and expected_observed and authority == "conformance"
    diagnostics: list[str] = list(extra_diagnostics or [])
    if execution.exit_code != 0:
        diagnostics.append(f"validator exit code: {execution.exit_code}")
    if error_count:
        diagnostics.append(f"validator errors: {error_count}")
    if warning_count:
        diagnostics.append(f"validator warnings: {warning_count}")
    if observed_outcome == "indeterminate":
        diagnostics.append("validator outcome was indeterminate")
    if not internal_matches:
        diagnostics.append("internal adapter outcome did not match the fixture expectation")
    if not expected_observed:
        diagnostics.append("independent validator outcome did not match the fixture expectation")
    if authority != "conformance":
        diagnostics.append("supporting parser evidence cannot promote a semantic profile")

    return {
        "schema_version": 1,
        "profile": profile,
        "edition": edition,
        "adapter_revision": adapter_revision,
        "fixture": {
            "kind": fixture_kind,
            "path": _relative_fixture(fixture, fixture_root),
            "sha256": fixture_hash or fixture_sha256(fixture, fixture_kind),
            "expected_outcome": expected_outcome,
        },
        "internal_valid": internal_valid,
        "independent_evidence": {
            "validator_name": validator_name,
            "validator_version": validator_version,
            "validator_executable_sha256": executable_hash
            or _hash_file(validator_executable),
            "schema_or_dictionary_sha256": authority_hash
            or authority_sha256(authority_artifact),
            "argv": argv,
            "executed_at_utc": executed_at_utc,
            "exit_code": execution.exit_code,
            "stdout_sha256": hashlib.sha256(execution.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
            "error_count": error_count,
            "warning_count": warning_count,
            "authority": authority,
            "observed_outcome": observed_outcome,
            "expected_outcome_observed": expected_observed,
        },
        "semantic_profile_promoted": passed and semantic_profile,
        "pass": passed,
        "diagnostics": diagnostics,
    }


def make_unavailable_receipt(
    *,
    profile: str,
    edition: str,
    adapter_revision: str,
    fixture: Path,
    fixture_kind: str,
    internal_valid: bool,
    diagnostic: str,
) -> dict[str, object]:
    """Represent unavailable independent evidence without inventing identity."""

    return {
        "schema_version": 1,
        "profile": profile,
        "edition": edition,
        "adapter_revision": adapter_revision,
        "fixture": {
            "kind": fixture_kind,
            "path": _relative_fixture(fixture, REPO),
            "sha256": fixture_sha256(fixture, fixture_kind),
            "expected_outcome": "accept",
        },
        "internal_valid": internal_valid,
        "independent_evidence": None,
        "semantic_profile_promoted": False,
        "pass": False,
        "diagnostics": [diagnostic],
    }


def _executable(value: str) -> Path:
    located = shutil.which(value)
    if located is None:
        raise OSError(f"validator executable not found: {value}")
    path = Path(located).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OSError(f"validator executable is not executable: {path}")
    return path


def _adapter_revision(value: str | None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError(f"cannot resolve adapter revision: {completed.stderr.strip()}")
    head = completed.stdout.strip()
    if value is None:
        value = head
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("adapter revision must be a 40- or 64-character lowercase hex ID")
    if value != head:
        raise ValueError("adapter revision must equal the checked-out HEAD")
    adapter_paths = [
        "Cargo.toml",
        "Cargo.lock",
        "crates/lamquant-standard-adapters",
        "tools/standard_adapter_validator_pins_v1.json",
        "tools/standard_adapter_python_environment_v1.json",
        "tools/validate_standard_adapter_fixtures.py",
    ]
    for staged in (False, True):
        command = ["git", "diff"]
        if staged:
            command.append("--cached")
        command.extend(["--quiet", "HEAD", "--", *adapter_paths])
        if subprocess.run(command, cwd=REPO, check=False).returncode != 0:
            raise ValueError("adapter implementation differs from the declared HEAD revision")
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "crates/lamquant-standard-adapters",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise ValueError("adapter implementation has untracked or unreadable source")
    return value


def _abir_revision(abir_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=abir_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError(f"cannot resolve ABIR revision: {completed.stderr.strip()}")
    paths = [
        "registries/adapter-profiles-v1.json",
        "schema/adapter-validation-v1.schema.json",
        "tools/verify_adapter_contract.py",
    ]
    for staged in (False, True):
        command = ["git", "diff"]
        if staged:
            command.append("--cached")
        command.extend(["--quiet", "HEAD", "--", *paths])
        if subprocess.run(command, cwd=abir_root, check=False).returncode != 0:
            raise ValueError("normative ABIR receipt contract differs from its HEAD")
    return completed.stdout.strip()


def _execution_time(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("executed-at-utc must be an ISO 8601 timestamp") from error
    if not value.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("executed-at-utc must use the UTC Z suffix")
    return value


def _validated_authority(path: Path) -> Path:
    path = path.absolute()
    authority_sha256(path)
    return path


def _load_pins(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("manifest_version") != 1 or not isinstance(
        payload.get("validators"), dict
    ):
        raise ValueError("validator pin manifest is malformed")
    return payload["validators"]


def _verify_record(record: Path, environment_root: Path) -> set[Path]:
    site_packages = record.parent.parent
    allowed: set[Path] = set()
    with record.open(newline="") as source:
        rows = list(csv.reader(source))
    for row in rows:
        if not row:
            continue
        candidate = (site_packages / row[0]).resolve()
        try:
            candidate.relative_to(environment_root)
        except ValueError as error:
            raise OSError(f"distribution RECORD escapes its environment: {row[0]}") from error
        try:
            allowed.add(candidate.relative_to(site_packages))
        except ValueError:
            pass
        if len(row) < 2 or not row[1].startswith("sha256="):
            continue
        encoded = row[1].removeprefix("sha256=")
        expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
        if _hash_file(candidate) != expected:
            raise OSError(f"distribution file does not match RECORD: {candidate}")
    return allowed


def _reject_unrecorded_site_content(site_packages: Path, allowed: set[Path]) -> None:
    for entry in site_packages.rglob("*"):
        relative = entry.relative_to(site_packages)
        if "__pycache__" in relative.parts or entry.suffix == ".pyc":
            continue
        if entry.is_file() and relative not in allowed:
            raise OSError(f"unrecorded site-packages file: {relative}")
        if entry.is_dir() and not any(
            candidate == relative or relative in candidate.parents for candidate in allowed
        ):
            raise OSError(f"unrecorded site-packages directory: {relative}")


def _verify_python_runtime(executable: Path, pin: dict[str, Any]) -> Path | None:
    runtime = pin.get("python_runtime")
    if runtime is None:
        return None
    first_line = executable.read_bytes().splitlines()[0]
    if not first_line.startswith(b"#!"):
        raise OSError(f"pinned Python launcher has no shebang: {executable}")
    interpreter = Path(first_line[2:].decode("utf-8"))
    if _hash_resolved_file(interpreter) != runtime.get("interpreter_sha256"):
        raise OSError("Python interpreter does not match the reviewed runtime pin")

    environment_root = executable.parent.parent.resolve()
    candidates = sorted((environment_root / "lib").glob("python*/site-packages"))
    if len(candidates) != 1:
        raise OSError("Python validator environment has ambiguous site-packages")
    site_packages = candidates[0]
    startup_files = runtime.get("startup_files")
    if not isinstance(startup_files, dict):
        raise OSError("Python validator runtime pin has no startup-file policy")
    observed_startup = {
        path.name
        for pattern in ("*.pth", "sitecustomize.py", "usercustomize.py")
        for path in site_packages.glob(pattern)
    }
    allowed_startup = {
        name
        for name in startup_files
        if name.endswith(".pth") or name in {"sitecustomize.py", "usercustomize.py"}
    }
    if observed_startup != allowed_startup:
        raise OSError("Python validator environment has unpinned startup files")
    for relative, expected_hash in startup_files.items():
        if _hash_file(site_packages / relative) != expected_hash:
            raise OSError(f"Python startup file is not pinned: {relative}")
    distributions = runtime.get("distributions")
    if not isinstance(distributions, dict) or not distributions:
        raise OSError("Python validator runtime pin has no distributions")
    for distribution, expected_record_hash in distributions.items():
        normalized = distribution.replace("-", "_")
        records = sorted(site_packages.glob(f"{normalized}-*.dist-info/RECORD"))
        if len(records) != 1:
            raise OSError(f"cannot identify pinned distribution: {distribution}")
        record = records[0]
        if _hash_file(record) != expected_record_hash:
            raise OSError(f"distribution RECORD is not pinned: {distribution}")
        _verify_record(record, environment_root)
    environment = json.loads(DEFAULT_PYTHON_ENVIRONMENT.read_text())
    expected_environment = environment.get("distributions")
    if environment.get("environment_version") != 1 or not isinstance(
        expected_environment, dict
    ):
        raise OSError("Python validator environment manifest is malformed")
    observed_records = {
        record.parent.name.removesuffix(".dist-info"): record
        for record in site_packages.glob("*.dist-info/RECORD")
    }
    if set(observed_records) != set(expected_environment):
        raise OSError("Python validator environment has unpinned distributions")
    allowed_content = {Path(relative) for relative in startup_files}
    for distribution, expected_record_hash in expected_environment.items():
        record = observed_records[distribution]
        if _hash_file(record) != expected_record_hash:
            raise OSError(f"environment distribution is not pinned: {distribution}")
        allowed_content.update(_verify_record(record, environment_root))
    _reject_unrecorded_site_content(site_packages, allowed_content)
    return interpreter


def _bind_validator(
    *,
    name: str,
    profile: str,
    version: str,
    evidence_authority: str,
    executable: str,
    authority_artifact: Path,
    pins: dict[str, dict[str, Any]],
) -> tuple[list[str], Path, Path, Path, str, str, str]:
    pin = pins.get(name)
    if pin is None:
        raise ValueError(f"validator is not pinned: {name}")
    validator_executable = _executable(executable)
    authority_path = _validated_authority(authority_artifact)
    launcher_hash = _hash_file(validator_executable)
    interpreter = _verify_python_runtime(validator_executable, pin)
    executed_program = interpreter or validator_executable
    executable_hash = _hash_resolved_file(executed_program)
    authority_hash = authority_sha256(authority_path)
    expected = {
        "version": version,
        "executable_sha256": executable_hash,
        "authority_sha256": authority_hash,
        "authority": (
            "conformance" if evidence_authority == "conformance" else "supporting-only"
        ),
    }
    if interpreter is not None:
        expected["launcher_sha256"] = launcher_hash
    pinned_profiles = pin.get("profiles")
    if pinned_profiles is None:
        pinned_profiles = [pin.get("profile")]
    if (
        not isinstance(pinned_profiles, list)
        or not pinned_profiles
        or any(not isinstance(item, str) or not item for item in pinned_profiles)
        or profile not in pinned_profiles
    ):
        raise ValueError(f"{name} profile does not match the reviewed validator pin")
    for field, observed in expected.items():
        if pin.get(field) != observed:
            raise ValueError(
                f"{name} {field} does not match the reviewed validator pin"
            )
    argv_prefix = (
        [
            str(interpreter),
            "-I",
            "-B",
            "-X",
            "pycache_prefix=/__abir_disabled_pycache__",
            str(validator_executable),
        ]
        if interpreter is not None
        else [str(validator_executable)]
    )
    return (
        argv_prefix,
        executed_program,
        validator_executable,
        authority_path,
        executable_hash,
        launcher_hash,
        authority_hash,
    )


def _internal_validation(
    command: list[str], timeout_seconds: float = 300.0
) -> tuple[bool, dict[str, object]]:
    executable_hash: str | None = None
    try:
        executable_hash = _hash_file(Path(command[0]))
    except OSError:
        pass
    failure_reason: str | None = None
    try:
        execution = run(command, cwd=REPO, timeout_seconds=timeout_seconds)
    except ValidationTimeout as error:
        failure_reason = f"timeout: {error}"
        execution = Execution(124, b"", str(error).encode("utf-8"))
    except OSError as error:
        failure_reason = f"unavailable: {error}"
        encoded = str(error).encode("utf-8", errors="replace")
        execution = Execution(127, b"", encoded)
    return execution.exit_code == 0, {
        "argv": command,
        "executable_sha256": executable_hash,
        "exit_code": execution.exit_code,
        "stdout_sha256": hashlib.sha256(execution.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "failure_reason": failure_reason,
    }


def _receipt_for(
    *,
    profile: str,
    edition: str,
    revision: str,
    fixture: Path,
    fixture_kind: str,
    validator_name: str,
    validator_version: str,
    executable: Path,
    launcher: Path,
    authority_artifact: Path,
    argv_prefix: list[str],
    argv_tail: list[str],
    executed_at_utc: str,
    evidence_authority: str,
    classifier: Callable[[int, bytes, bytes], tuple[str, int, int]],
    internal_valid: bool,
    executable_hash: str,
    launcher_hash: str,
    authority_hash: str,
    runtime_pin: dict[str, Any],
    timeout_seconds: float,
    semantic_profile: bool,
) -> dict[str, object]:
    relative_fixture = _relative_fixture(fixture, REPO)
    fixture_hash = fixture_sha256(fixture, fixture_kind)
    argv = [*argv_prefix, *argv_tail, relative_fixture]
    execution = run(argv, cwd=REPO, timeout_seconds=timeout_seconds)
    observed, errors, warnings = classifier(
        execution.exit_code, execution.stdout, execution.stderr
    )
    mutations: list[str] = []
    try:
        post_hashes = (
            fixture_sha256(fixture, fixture_kind),
            _hash_resolved_file(executable),
            _hash_file(launcher),
            authority_sha256(authority_artifact),
        )
        _verify_python_runtime(launcher, runtime_pin)
    except OSError as error:
        mutations.append(f"bound input disappeared or became invalid: {error}")
    else:
        if post_hashes != (
            fixture_hash,
            executable_hash,
            launcher_hash,
            authority_hash,
        ):
            mutations.append("a bound input changed during validator execution")
    if mutations:
        observed = "indeterminate"
    return make_receipt(
        profile=profile,
        edition=edition,
        adapter_revision=revision,
        fixture=fixture,
        fixture_root=REPO,
        fixture_kind=fixture_kind,
        expected_outcome="accept",
        internal_valid=internal_valid,
        validator_name=validator_name,
        validator_version=validator_version,
        validator_executable=executable,
        authority_artifact=authority_artifact,
        argv=argv,
        execution=execution,
        executed_at_utc=executed_at_utc,
        authority=evidence_authority,
        observed_outcome=observed,
        error_count=errors,
        warning_count=warnings,
        semantic_profile=semantic_profile,
        fixture_hash=fixture_hash,
        executable_hash=executable_hash,
        authority_hash=authority_hash,
        extra_diagnostics=mutations,
    )


def _attempt_receipt(
    *,
    profile: str,
    edition: str,
    revision: str,
    fixture: Path,
    fixture_kind: str,
    validator_name: str,
    validator_version: str,
    executable_name: str,
    authority_artifact: Path,
    argv_tail: list[str],
    executed_at_utc: str,
    evidence_authority: str,
    classifier: Callable[[int, bytes, bytes], tuple[str, int, int]],
    internal_valid: bool,
    pins: dict[str, dict[str, Any]],
    timeout_seconds: float,
) -> dict[str, object]:
    try:
        semantic_profile = _profile_is_semantic(DEFAULT_ABIR, profile)
        (
            argv_prefix,
            executable,
            launcher,
            authority_path,
            executable_hash,
            launcher_hash,
            authority_hash,
        ) = _bind_validator(
            name=validator_name,
            profile=profile,
            version=validator_version,
            evidence_authority=evidence_authority,
            executable=executable_name,
            authority_artifact=authority_artifact,
            pins=pins,
        )
    except (OSError, ValueError) as error:
        return make_unavailable_receipt(
            profile=profile,
            edition=edition,
            adapter_revision=revision,
            fixture=fixture,
            fixture_kind=fixture_kind,
            internal_valid=internal_valid,
            diagnostic=f"independent validator evidence unavailable: {error}",
        )
    try:
        return _receipt_for(
            profile=profile,
            edition=edition,
            revision=revision,
            fixture=fixture,
            fixture_kind=fixture_kind,
            validator_name=validator_name,
            validator_version=validator_version,
            executable=executable,
            launcher=launcher,
            authority_artifact=authority_path,
            argv_prefix=argv_prefix,
            argv_tail=argv_tail,
            executed_at_utc=executed_at_utc,
            evidence_authority=evidence_authority,
            classifier=classifier,
            internal_valid=internal_valid,
            executable_hash=executable_hash,
            launcher_hash=launcher_hash,
            authority_hash=authority_hash,
            runtime_pin=pins[validator_name],
            timeout_seconds=timeout_seconds,
            semantic_profile=semantic_profile,
        )
    except OSError as error:
        return make_unavailable_receipt(
            profile=profile,
            edition=edition,
            adapter_revision=revision,
            fixture=fixture,
            fixture_kind=fixture_kind,
            internal_valid=internal_valid,
            diagnostic=f"independent validator execution unavailable: {error}",
        )


def _abir_contract(abir_root: Path) -> Any:
    script = abir_root / "tools" / "verify_adapter_contract.py"
    spec = importlib.util.spec_from_file_location("abir_adapter_contract", script)
    if spec is None or spec.loader is None:
        raise OSError(f"cannot load ABIR Adapter verifier: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_is_semantic(abir_root: Path, profile_id: str) -> bool:
    registry = json.loads(
        (abir_root / "registries" / "adapter-profiles-v1.json").read_text()
    )
    matches = [
        profile for profile in registry.get("profiles", []) if profile.get("id") == profile_id
    ]
    if len(matches) != 1:
        raise ValueError(f"profile is not uniquely registered: {profile_id}")
    return matches[0].get("status") == "semantic"


def _verify_receipts(receipts: list[dict[str, object]], abir_root: Path) -> None:
    contract = _abir_contract(abir_root)
    for receipt in receipts:
        errors = contract.receipt_errors(receipt, fixture_root=REPO)
        if errors:
            raise ValueError("ABIR receipt verification failed: " + "; ".join(errors))


def _require_normative_validator(
    abir_root: Path, profile: str, validator_name: str
) -> None:
    registry = json.loads(
        (abir_root / "registries" / "adapter-profiles-v1.json").read_text()
    )
    profiles = {entry["id"]: entry for entry in registry["profiles"]}
    registered = profiles.get(profile)
    if registered is None:
        raise ValueError(f"ABIR profile is not registered: {profile}")
    if registered.get("validator") != validator_name:
        raise ValueError(
            f"ABIR profile {profile} requires validator "
            f"{registered.get('validator')}, not {validator_name}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids", type=Path, default=DEFAULT_BIDS)
    parser.add_argument("--nwb", type=Path, default=DEFAULT_NWB)
    parser.add_argument("--dicom", type=Path, default=DEFAULT_DICOM)
    parser.add_argument("--pynwb-validate", default="pynwb-validate")
    parser.add_argument("--nwbinspector", default="nwbinspector")
    parser.add_argument("--dciodvfy", default="dciodvfy")
    parser.add_argument("--bids-validator", default="bids-validator-deno")
    parser.add_argument("--pynwb-version", required=True)
    parser.add_argument("--nwbinspector-version", required=True)
    parser.add_argument("--dciodvfy-version", required=True)
    parser.add_argument("--bids-validator-version", required=True)
    parser.add_argument("--pynwb-authority", type=Path, required=True)
    parser.add_argument("--nwbinspector-authority", type=Path, required=True)
    parser.add_argument("--dicom-authority", type=Path, required=True)
    parser.add_argument("--bids-authority", type=Path, required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--executed-at-utc",
        help="fixed UTC Z timestamp; defaults to the current UTC second",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    revision = _adapter_revision(args.adapter_revision)
    if not 0.0 < args.timeout_seconds <= 3600.0:
        raise ValueError("timeout-seconds must be greater than zero and at most 3600")
    executed_at_utc = _execution_time(args.executed_at_utc)
    pins = _load_pins(DEFAULT_PINS)
    abir_root = DEFAULT_ABIR
    abir_revision = _abir_revision(abir_root)
    bids = args.bids.absolute()
    dicom = args.dicom.absolute()
    nwb = args.nwb.absolute()
    _require_normative_validator(
        abir_root, "bids.1.11.1.single-edf-eeg", "bids-validator"
    )
    _require_normative_validator(
        abir_root, "bids.1.11.1.single-edf-eeg-events", "bids-validator"
    )
    _require_normative_validator(
        abir_root, "nwb.2.10.0.single-integer-timeseries", "pynwb.validate"
    )

    cargo = shutil.which("cargo") or "cargo"
    bids_internal, bids_internal_receipt = _internal_validation(
        [cargo, "test", "-p", "lamquant-standard-adapters", "--test", "bids_adapter"],
        args.timeout_seconds,
    )
    waveform_internal, waveform_internal_receipt = _internal_validation(
        [
            cargo,
            "test",
            "-p",
            "lamquant-standard-adapters",
            "--test",
            "dicom_nwb_adapters",
        ],
        args.timeout_seconds,
    )

    primary = [
        _attempt_receipt(
            profile="bids.1.11.1.single-edf-eeg",
            edition="1.11.1",
            revision=revision,
            fixture=bids,
            fixture_kind="tree",
            validator_name="bids-validator",
            validator_version=args.bids_validator_version,
            executable_name=args.bids_validator,
            authority_artifact=args.bids_authority,
            argv_tail=[],
            executed_at_utc=executed_at_utc,
            evidence_authority="conformance",
            classifier=classify_bids,
            internal_valid=bids_internal,
            pins=pins,
            timeout_seconds=args.timeout_seconds,
        ),
        _attempt_receipt(
            profile="bids.1.11.1.single-edf-eeg-events",
            edition="1.11.1",
            revision=revision,
            fixture=bids,
            fixture_kind="tree",
            validator_name="bids-validator",
            validator_version=args.bids_validator_version,
            executable_name=args.bids_validator,
            authority_artifact=args.bids_authority,
            argv_tail=[],
            executed_at_utc=executed_at_utc,
            evidence_authority="conformance",
            classifier=classify_bids,
            internal_valid=bids_internal,
            pins=pins,
            timeout_seconds=args.timeout_seconds,
        ),
        make_unavailable_receipt(
            profile="dicom.ps3.2026c.ecg-i16",
            edition="PS3 2026c",
            adapter_revision=revision,
            fixture=dicom,
            fixture_kind="file",
            internal_valid=waveform_internal,
            diagnostic=(
                "no pinned PS3 2026c conformance validator is available; "
                "dciodvfy 20240118 is supporting evidence only"
            ),
        ),
        _attempt_receipt(
            profile="nwb.2.10.0.single-integer-timeseries",
            edition="2.10.0",
            revision=revision,
            fixture=nwb,
            fixture_kind="file",
            validator_name="pynwb.validate",
            validator_version=args.pynwb_version,
            executable_name=args.pynwb_validate,
            authority_artifact=args.pynwb_authority,
            argv_tail=["--no-cached-namespace"],
            executed_at_utc=executed_at_utc,
            evidence_authority="conformance",
            classifier=classify_pynwb,
            internal_valid=waveform_internal,
            pins=pins,
            timeout_seconds=args.timeout_seconds,
        ),
    ]
    supporting = [
        _attempt_receipt(
            profile="dicom.ps3.2026c.ecg-i16",
            edition="PS3 2026c",
            revision=revision,
            fixture=dicom,
            fixture_kind="file",
            validator_name="dciodvfy",
            validator_version=args.dciodvfy_version,
            executable_name=args.dciodvfy,
            authority_artifact=args.dicom_authority,
            argv_tail=[],
            executed_at_utc=executed_at_utc,
            evidence_authority="parser-only",
            classifier=classify_dicom,
            internal_valid=waveform_internal,
            pins=pins,
            timeout_seconds=args.timeout_seconds,
        ),
        _attempt_receipt(
            profile="nwb.2.10.0.single-integer-timeseries",
            edition="2.10.0",
            revision=revision,
            fixture=nwb,
            fixture_kind="file",
            validator_name="nwbinspector",
            validator_version=args.nwbinspector_version,
            executable_name=args.nwbinspector,
            authority_artifact=args.nwbinspector_authority,
            argv_tail=["--threshold", "CRITICAL", "--progress-bar", "False", "--detailed"],
            executed_at_utc=executed_at_utc,
            evidence_authority="parser-only",
            classifier=classify_nwbinspector,
            internal_valid=waveform_internal,
            pins=pins,
            timeout_seconds=args.timeout_seconds,
        ),
    ]
    _verify_receipts(primary + supporting, abir_root)
    bundle = {
        "bundle_schema_version": 1,
        "scope": "bounded-standard-adapter-fixtures-only",
        "adapter_revision": revision,
        "abir_contract_revision": abir_revision,
        "timeout_seconds": args.timeout_seconds,
        "receipts": primary,
        "supporting_receipts": supporting,
        "validator_runtime_bindings": {
            name: pins[name]
            for name in (
                "bids-validator",
                "dciodvfy",
                "nwbinspector",
                "pynwb.validate",
            )
        },
        "python_environment_manifest_sha256": _hash_file(
            DEFAULT_PYTHON_ENVIRONMENT
        ),
        "internal_validation": {
            "bids": bids_internal_receipt,
            "dicom_nwb": waveform_internal_receipt,
        },
        "passed": all(receipt["pass"] for receipt in primary),
        "limitations": [
            "No receipt establishes edition-wide BIDS, DICOM, or NWB conformance.",
            "DICOM PS3 2026c conformance evidence is unavailable; "
            "dciodvfy 20240118 is supporting-only.",
            "EDFbrowser conformance evidence is not included in this bundle.",
            "NWBInspector is supporting parser evidence and cannot promote a profile.",
            "Runtime hashes bind installed content; they are not supply-chain provenance.",
        ],
    }
    print(canonical_json(bundle))
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
