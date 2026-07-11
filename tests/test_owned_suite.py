"""Fail-closed tests for Lossless A02 owner manifest."""
from pathlib import Path

import pytest

from tests.run_owned_tests import MANIFEST, SkipAudit, SkipRule, load_manifest


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ownership.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_repository_manifest_loads() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.paths
    assert "tests/test_owned_suite.py" in manifest.paths


def test_retired_ai_models_aggregation_smoke_stays_absent() -> None:
    retired = MANIFEST.parent / "codec_python_smoke/test_ai_models_smoke.py"
    assert not retired.exists()


@pytest.mark.parametrize("version", [True, "1", 2])
def test_schema_version_is_exact_integer(tmp_path: Path, version: object) -> None:
    rendered = str(version).lower() if isinstance(version, bool) else repr(version)
    path = _write(tmp_path, f"schema_version = {rendered}\n")
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(path)


def test_suite_ids_must_be_unique(tmp_path: Path) -> None:
    body = """schema_version = 1
[[suite]]
id = "same"
paths = ["tests/test_owned_suite.py"]
[[suite]]
id = "same"
paths = ["tests/test_owned_suite.py"]
"""
    with pytest.raises(ValueError, match="duplicate suite id"):
        load_manifest(_write(tmp_path, body))


@pytest.mark.parametrize(
    "path",
    [
        "../test_escape.py",
        "/tmp/test_absolute.py",
        "other/test_wrong_root.py",
        "tests/not_a_suite.py",
    ],
)
def test_paths_must_be_owned_test_files(tmp_path: Path, path: str) -> None:
    body = f"""schema_version = 1
[[suite]]
id = "bad"
paths = [{path!r}]
"""
    with pytest.raises(ValueError, match="invalid owned test path"):
        load_manifest(_write(tmp_path, body))


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    body = """schema_version = 1
[[suite]]
id = "bad"
paths = ["tests/test_owned_suite.py", "tests/test_owned_suite.py"]
"""
    with pytest.raises(ValueError, match="duplicate owned test path"):
        load_manifest(_write(tmp_path, body))


def test_manifest_rejects_missing_paths(tmp_path: Path) -> None:
    body = """schema_version = 1
[[suite]]
id = "bad"
paths = ["tests/test_missing_owner_case.py"]
"""
    with pytest.raises(ValueError, match="does not exist"):
        load_manifest(_write(tmp_path, body))


def test_skip_audit_rejects_zero_collection() -> None:
    audit = SkipAudit()
    assert audit.diagnostics((), ("tests/test_owned_suite.py",)) == (
        "owned test path collected no tests: tests/test_owned_suite.py",
    )


def test_skip_audit_rejects_unclassified_skip() -> None:
    audit = SkipAudit()
    audit.collected.add("tests/test_owned_suite.py")
    audit.skips.append(("tests/test_owned_suite.py::test_case", "because"))
    diagnostics = audit.diagnostics((), ("tests/test_owned_suite.py",))
    assert diagnostics == (
        "unclassified skip: tests/test_owned_suite.py::test_case: because",
    )


def test_skip_audit_enforces_count_limit() -> None:
    audit = SkipAudit()
    audit.collected.add("tests/test_owned_suite.py")
    audit.skips.extend(
        [
            ("tests/test_owned_suite.py::test_one", "fixture absent"),
            ("tests/test_owned_suite.py::test_two", "fixture absent"),
        ]
    )
    rule = SkipRule("tests/test_owned_suite.py", "fixture absent", 1)
    diagnostics = audit.diagnostics((rule,), ("tests/test_owned_suite.py",))
    assert diagnostics == (
        "skip count for tests/test_owned_suite.py is 2, above 1",
    )
