"""Owner-local source contracts retained from the retired root format audit."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
CODEC = REPO / "reference_implementations/python_codec/lamquant_codec"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assigned_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ()
        if isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _literal_assignment(path: Path, name: str) -> object:
    tree = _tree(path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: missing literal assignment {name}")


def test_legacy_container_header_size_stays_pinned() -> None:
    assert _literal_assignment(CODEC / "edf_to_lml.py", "_LML_HEADER_SIZE") == 32


def test_wire_constants_have_one_definition() -> None:
    constants = CODEC / "ops/constants.py"
    assert _literal_assignment(constants, "MAGIC_LMQ") == b"LMQ1"
    assert _literal_assignment(constants, "MAGIC_LML") == b"LML1"
    assert _literal_assignment(constants, "BIAS_CTX_LEN") == 32

    for relative in ("compress.py", "decompress.py"):
        assigned = _assigned_names(_tree(CODEC / relative))
        assert "MAGIC" not in assigned
        assert "_DEFAULT_TOTAL" not in assigned


def test_active_write_paths_do_not_restore_lmq5_magic() -> None:
    for relative in ("compress.py", "decompress.py", "ops/fused_lml.py"):
        tree = _tree(CODEC / relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = (node.target,)
                value = node.value
            else:
                continue
            target_names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if target_names & {"MAGIC", "MAGIC_LMQ"}:
                assert ast.literal_eval(value) != b"LMQ5"


def test_fused_codec_reads_bias_context_constant() -> None:
    tree = _tree(CODEC / "ops/fused_lml.py")
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "lamquant_codec.ops.constants"
        for alias in node.names
    }
    hardcoded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "ctx_len"
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "int64"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
    ]

    assert "BIAS_CTX_LEN" in imported
    assert not hardcoded


def test_codec_does_not_import_retired_training_preprocessor() -> None:
    violations = []
    for path in CODEC.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module == "subband_preprocess":
                violations.append(path.relative_to(REPO).as_posix())
            if isinstance(node, ast.Import) and any(
                alias.name == "subband_preprocess" for alias in node.names
            ):
                violations.append(path.relative_to(REPO).as_posix())
    assert not violations


@pytest.mark.parametrize(
    "module",
    ("constants", "lifting", "lpc", "bias", "wht", "pipeline", "golomb", "rans"),
)
def test_ops_modules_declare_public_exports(module: str) -> None:
    exports = _literal_assignment(CODEC / "ops" / f"{module}.py", "__all__")
    assert isinstance(exports, list) and exports
