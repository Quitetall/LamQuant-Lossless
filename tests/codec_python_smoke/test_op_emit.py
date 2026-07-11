"""Owner tests for the Python OpEvent producer/consumer contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from lamquant_codec.cli.op_emit import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_FILE_DONE,
    EVENT_LOG,
    EVENT_PROGRESS,
    EVENT_STARTED,
    OpEvent,
    parse_line,
    parse_lines,
)
from lamquant_codec._paths import REPO_ROOT


FIXTURE = REPO_ROOT / "tests" / "fixtures" / "op-events-sample.jsonl"
PYTHON_SOURCE = REPO_ROOT / "reference_implementations" / "python_codec"


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"type": EVENT_STARTED, "ts_ms": 1, "op_id": "encode", "total": 42},
         {"op_id": "encode", "total": 42}),
        ({"type": EVENT_STARTED, "ts_ms": 1, "op_id": "info"},
         {"op_id": "info", "total": None}),
        ({"type": EVENT_PROGRESS, "ts_ms": 2, "current": 5, "total": 42,
          "message": "file 5/42"}, {"current": 5, "total": 42}),
        ({"type": EVENT_FILE_DONE, "ts_ms": 3, "path": "/data/a.lml",
          "success": True, "ms": 120, "cr": 2.5},
         {"path": "/data/a.lml", "success": True, "cr": 2.5}),
        ({"type": EVENT_FILE_DONE, "ts_ms": 3, "path": "/data/b.lml",
          "success": False, "ms": 0}, {"success": False, "cr": None}),
        ({"type": EVENT_DONE, "ts_ms": 4, "message": "done"},
         {"message": "done"}),
        ({"type": EVENT_ERROR, "ts_ms": 4, "message": "out of disk"},
         {"message": "out of disk"}),
        ({"type": EVENT_LOG, "ts_ms": 4, "message": "WARN x"},
         {"message": "WARN x"}),
    ),
)
def test_parse_line_preserves_variant_fields(payload: dict, expected: dict) -> None:
    event = parse_line(json.dumps(payload))

    assert isinstance(event, OpEvent)
    assert event.type == payload["type"]
    assert event.ts_ms == payload["ts_ms"]
    for field, value in expected.items():
        assert getattr(event, field) == value


@pytest.mark.parametrize(
    ("line", "message"),
    (
        (json.dumps({"type": "Mystery", "ts_ms": 1}), "unknown OpEvent type"),
        ("{not json", "OpEvent JSON parse"),
        (json.dumps({"type": EVENT_LOG, "message": "x"}), "ts_ms"),
    ),
)
def test_parse_line_rejects_invalid_events(line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_line(line)


def test_parse_lines_reports_and_skips_malformed_input(capsys) -> None:
    lines = (
        json.dumps({"type": EVENT_LOG, "ts_ms": 1, "message": "one"}),
        "",
        "{not json",
        json.dumps({"type": EVENT_LOG, "ts_ms": 2, "message": "two"}),
    )

    events = list(parse_lines(lines))

    assert [event.message for event in events] == ["one", "two"]
    assert "dropping malformed event line" in capsys.readouterr().err


@pytest.mark.parametrize("explicit_fixture", (False, True))
def test_check_command_validates_owner_fixture(explicit_fixture: bool) -> None:
    # Cover both owner-default dispatch and the explicit path used by composition.
    command = [sys.executable, "-m", "lamquant_codec.cli.op_emit", "--check"]
    if explicit_fixture:
        command.extend(("--fixture", str(FIXTURE)))
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(PYTHON_SOURCE),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "op_emit --check: OK" in result.stdout
