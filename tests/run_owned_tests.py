#!/usr/bin/env python3
"""Run implementation tests rehomed from LamQuant meta-repository."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import sys
import tomllib


SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("ownership.toml")


@dataclass(frozen=True)
class SkipRule:
    path: str
    reason_contains: str
    max_count: int


@dataclass(frozen=True)
class OwnedManifest:
    paths: tuple[str, ...]
    allowed_skips: tuple[SkipRule, ...]


def load_manifest(manifest: Path = MANIFEST) -> OwnedManifest:
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError(f"{manifest}: schema_version must be {SCHEMA_VERSION}")
    suites = data.get("suite")
    if not isinstance(suites, list) or not suites:
        raise ValueError(f"{manifest}: at least one [[suite]] is required")

    paths: list[str] = []
    suite_ids: set[str] = set()
    for index, suite in enumerate(suites):
        if not isinstance(suite, dict):
            raise ValueError(f"{manifest}: suite[{index}] must be a table")
        suite_id = suite.get("id")
        if not isinstance(suite_id, str) or not suite_id.strip():
            raise ValueError(f"{manifest}: suite[{index}].id must be a non-empty string")
        if suite_id in suite_ids:
            raise ValueError(f"{manifest}: duplicate suite id {suite_id}")
        suite_ids.add(suite_id)
        suite_paths = suite.get("paths")
        if not isinstance(suite_paths, list) or not suite_paths:
            raise ValueError(f"{manifest}: suite[{index}].paths must be a non-empty array")
        for raw in suite_paths:
            if not isinstance(raw, str):
                raise ValueError(f"{manifest}: owned test paths must be strings")
            path = PurePosixPath(raw)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.name.startswith("test_")
                or path.parts[0] != "tests"
            ):
                raise ValueError(f"{manifest}: invalid owned test path {raw!r}")
            relative = path.as_posix()
            if relative in paths:
                raise ValueError(f"{manifest}: duplicate owned test path {relative}")
            if not (REPO / path).is_file():
                raise ValueError(f"{manifest}: owned test path does not exist: {relative}")
            paths.append(relative)

    skip_rules: list[SkipRule] = []
    for index, raw in enumerate(data.get("allowed_skip", [])):
        if not isinstance(raw, dict):
            raise ValueError(f"{manifest}: allowed_skip[{index}] must be a table")
        path = raw.get("path")
        reason = raw.get("reason_contains")
        max_count = raw.get("max_count")
        if not isinstance(path, str) or not isinstance(reason, str):
            raise ValueError(f"{manifest}: allowed_skip[{index}] path and reason must be strings")
        if path not in paths:
            raise ValueError(f"{manifest}: allowed skip path is not owned: {path}")
        if (
            not reason
            or isinstance(max_count, bool)
            or not isinstance(max_count, int)
            or max_count < 1
        ):
            raise ValueError(f"{manifest}: allowed_skip[{index}] has invalid reason or max_count")
        if any(rule.path == path for rule in skip_rules):
            raise ValueError(f"{manifest}: duplicate allowed skip path {path}")
        skip_rules.append(SkipRule(path, reason, max_count))
    return OwnedManifest(tuple(paths), tuple(skip_rules))


class SkipAudit:
    def __init__(self) -> None:
        self.collected: set[str] = set()
        self.skips: list[tuple[str, str]] = []

    def pytest_collection_finish(self, session) -> None:
        self.collected.update(item.nodeid.split("::", 1)[0] for item in session.items)

    def pytest_collectreport(self, report) -> None:
        if report.skipped:
            self.skips.append((report.nodeid, str(report.longrepr)))

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped:
            self.skips.append((report.nodeid, str(report.longrepr)))

    def diagnostics(
        self,
        rules: tuple[SkipRule, ...],
        paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        counts = {rule.path: 0 for rule in rules}
        diagnostics = [
            f"owned test path collected no tests: {path}"
            for path in paths
            if path not in self.collected
        ]
        for nodeid, reason in self.skips:
            matches = [
                rule
                for rule in rules
                if (nodeid == rule.path or nodeid.startswith(rule.path + "::"))
                and rule.reason_contains in reason
            ]
            if len(matches) != 1:
                diagnostics.append(f"unclassified skip: {nodeid}: {reason}")
                continue
            counts[matches[0].path] += 1
        for rule in rules:
            if counts[rule.path] > rule.max_count:
                diagnostics.append(
                    f"skip count for {rule.path} is {counts[rule.path]}, above {rule.max_count}"
                )
        return tuple(diagnostics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    if args.list:
        print("\n".join(manifest.paths))
        return 0

    command = ["-q", "--rootdir=."]
    if args.collect_only:
        command.append("--collect-only")
    audit = SkipAudit()
    import pytest

    previous = Path.cwd()
    try:
        os.chdir(REPO)
        result = pytest.main([*command, *manifest.paths], plugins=[audit])
    finally:
        os.chdir(previous)
    if result != pytest.ExitCode.OK:
        return int(result)
    diagnostics = audit.diagnostics(manifest.allowed_skips, manifest.paths)
    for diagnostic in diagnostics:
        print(f"FAIL: {diagnostic}", file=sys.stderr)
    return 2 if diagnostics else 0


if __name__ == "__main__":
    sys.exit(main())
