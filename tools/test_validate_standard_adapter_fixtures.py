#!/usr/bin/env python3
"""Hermetic tests for independent standard-adapter validation receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("validate_standard_adapter_fixtures.py")
SPEC = importlib.util.spec_from_file_location("standard_receipts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
standard_receipts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(standard_receipts)


class ReceiptTests(unittest.TestCase):
    def test_fixture_tree_hash_uses_abir_domain_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z.txt").write_bytes(b"last")
            (root / "nested").mkdir()
            (root / "nested" / "a.txt").write_bytes(b"first")

            digest = hashlib.sha256(b"abir.adapter.fixture-tree.v1\0")
            for relative, payload in (("nested/a.txt", b"first"), ("z.txt", b"last")):
                encoded = relative.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
            self.assertEqual(
                standard_receipts.fixture_sha256(root, "tree"), digest.hexdigest()
            )

            (root / "link").symlink_to(root / "z.txt")
            with self.assertRaisesRegex(OSError, "symlink"):
                standard_receipts.fixture_sha256(root, "tree")

    def test_receipt_binds_exact_execution_and_is_schema_shaped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"fixture")
            authority = root / "schema.yaml"
            authority.write_bytes(b"schema")
            executable = root / "validator"
            executable.write_text(
                "#!/bin/sh\nprintf 'Summary: valid\\n'\nprintf 'one warning\\n' >&2\n"
            )
            executable.chmod(0o755)
            argv = [str(executable), "fixture.bin"]

            execution = standard_receipts.run(argv, cwd=root)
            receipt = standard_receipts.make_receipt(
                profile="bids.1.11.1.single-edf-eeg",
                edition="1.11.1",
                adapter_revision="a" * 40,
                fixture=fixture,
                fixture_root=root,
                fixture_kind="file",
                expected_outcome="accept",
                internal_valid=True,
                validator_name="bids-validator",
                validator_version="3.0.1",
                validator_executable=executable,
                authority_artifact=authority,
                argv=argv,
                execution=execution,
                executed_at_utc="2026-07-22T12:00:00Z",
                authority="conformance",
                observed_outcome="accept",
                error_count=0,
                warning_count=1,
            )

            self.assertEqual(
                set(receipt),
                {
                    "schema_version",
                    "profile",
                    "edition",
                    "adapter_revision",
                    "fixture",
                    "internal_valid",
                    "independent_evidence",
                    "semantic_profile_promoted",
                    "pass",
                    "diagnostics",
                },
            )
            evidence = receipt["independent_evidence"]
            self.assertEqual(evidence["argv"], argv)
            self.assertEqual(
                evidence["stdout_sha256"], hashlib.sha256(execution.stdout).hexdigest()
            )
            self.assertEqual(
                evidence["stderr_sha256"], hashlib.sha256(execution.stderr).hexdigest()
            )
            self.assertEqual(
                evidence["validator_executable_sha256"],
                hashlib.sha256(executable.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                evidence["schema_or_dictionary_sha256"],
                hashlib.sha256(authority.read_bytes()).hexdigest(),
            )
            self.assertTrue(receipt["pass"])
            self.assertTrue(receipt["semantic_profile_promoted"])

    def test_parser_only_supporting_receipt_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"fixture")
            authority = root / "authority"
            authority.write_bytes(b"authority")
            executable = root / "validator"
            executable.write_bytes(b"validator")
            execution = standard_receipts.Execution(0, b"No issues found!\n", b"")

            receipt = standard_receipts.make_receipt(
                profile="nwb.2.10.0.single-integer-timeseries",
                edition="2.10.0",
                adapter_revision="b" * 40,
                fixture=fixture,
                fixture_root=root,
                fixture_kind="file",
                expected_outcome="accept",
                internal_valid=True,
                validator_name="nwbinspector",
                validator_version="0.7.2",
                validator_executable=executable,
                authority_artifact=authority,
                argv=[str(executable), "fixture.bin"],
                execution=execution,
                executed_at_utc="2026-07-22T12:00:00Z",
                authority="parser-only",
                observed_outcome="accept",
                error_count=0,
                warning_count=0,
            )

            self.assertFalse(receipt["pass"])
            self.assertFalse(receipt["semantic_profile_promoted"])

    def test_error_diagnostics_produce_reject_outcomes(self) -> None:
        self.assertEqual(
            standard_receipts.classify_bids(1, b"", b"[ERROR] bad\n"),
            ("reject", 1, 0),
        )
        self.assertEqual(
            standard_receipts.classify_dicom(0, b"Error - bad\n", b""),
            ("reject", 1, 0),
        )
        self.assertEqual(
            standard_receipts.classify_pynwb(0, b"No errors found.\n", b""),
            ("accept", 0, 0),
        )
        self.assertEqual(
            standard_receipts.classify_nwbinspector(0, b"No issues found!\n", b""),
            ("accept", 0, 0),
        )

    def test_receipt_serialization_is_canonical_json(self) -> None:
        payload = {"z": 1, "a": [True, None]}
        self.assertEqual(
            standard_receipts.canonical_json(payload),
            '{"a":[true,null],"z":1}',
        )

    def test_adapter_revision_cannot_be_caller_forged(self) -> None:
        with self.assertRaisesRegex(ValueError, "checked-out HEAD"):
            standard_receipts._adapter_revision("f" * 40)

    def test_validator_identity_must_match_reviewed_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "validator"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            authority = root / "authority"
            authority.write_bytes(b"authority")
            pins = {
                "validator": {
                    "version": "1",
                    "executable_sha256": "0" * 64,
                    "authority_sha256": hashlib.sha256(b"authority").hexdigest(),
                }
            }
            with self.assertRaisesRegex(ValueError, "reviewed validator pin"):
                standard_receipts._bind_validator(
                    name="validator",
                    profile="profile",
                    version="1",
                    evidence_authority="conformance",
                    executable=str(executable),
                    authority_artifact=authority,
                    pins=pins,
                )

    def test_unavailable_evidence_is_a_schema_valid_failed_receipt(self) -> None:
        receipt = standard_receipts.make_unavailable_receipt(
            profile="nwb.2.10.0.single-integer-timeseries",
            edition="2.10.0",
            adapter_revision="a" * 40,
            fixture=standard_receipts.DEFAULT_NWB,
            fixture_kind="file",
            internal_valid=True,
            diagnostic="validator unavailable",
        )
        self.assertIsNone(receipt["independent_evidence"])
        self.assertFalse(receipt["pass"])
        self.assertFalse(receipt["semantic_profile_promoted"])

        contract = standard_receipts._abir_contract(standard_receipts.DEFAULT_ABIR)
        self.assertEqual(
            contract.receipt_errors(receipt, fixture_root=standard_receipts.REPO), []
        )

    def test_internal_validation_result_is_derived_from_execution(self) -> None:
        valid, success = standard_receipts._internal_validation(
            ["/bin/sh", "-c", "printf ok"]
        )
        invalid, failure = standard_receipts._internal_validation(
            ["/bin/sh", "-c", "printf bad >&2; exit 9"]
        )
        self.assertTrue(valid)
        self.assertEqual(success["exit_code"], 0)
        self.assertFalse(invalid)
        self.assertEqual(failure["exit_code"], 9)

    def test_validator_environment_ignores_pythonpath_shadowing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            injected = root / "injected"
            injected.mkdir()
            (injected / "shadowed.py").write_text("VALUE = 'forged'\n")
            launcher = root / "launcher"
            launcher.mkdir()
            script = launcher / "validator"
            script.write_text(
                "#!/usr/bin/python3\nimport shadowed\nprint(shadowed.VALUE)\n"
            )
            script.chmod(0o755)
            with mock.patch.dict(os.environ, {"PYTHONPATH": str(injected)}):
                execution = standard_receipts.run([str(script)], cwd=root)
            self.assertNotEqual(execution.exit_code, 0)
            self.assertNotIn(b"forged", execution.stdout)

    def test_validator_timeout_is_reason_coded(self) -> None:
        with self.assertRaisesRegex(
            standard_receipts.ValidationTimeout, "exceeded"
        ):
            standard_receipts.run(
                ["/bin/sh", "-c", "sleep 1"],
                cwd=standard_receipts.REPO,
                timeout_seconds=0.01,
            )
        valid, receipt = standard_receipts._internal_validation(
            ["/bin/sh", "-c", "sleep 1"], timeout_seconds=0.01
        )
        self.assertFalse(valid)
        self.assertEqual(receipt["exit_code"], 124)
        self.assertIn("timeout", receipt["failure_reason"])

    def test_external_timeout_becomes_null_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "validator"
            executable.write_text("#!/bin/sh\nsleep 1\n")
            executable.chmod(0o755)
            authority = root / "authority"
            authority.write_bytes(b"authority")
            pins = {
                "pynwb.validate": {
                    "authority": "conformance",
                    "authority_sha256": hashlib.sha256(b"authority").hexdigest(),
                    "executable_sha256": hashlib.sha256(
                        executable.read_bytes()
                    ).hexdigest(),
                    "profile": "nwb.2.10.0.single-integer-timeseries",
                    "version": "test",
                }
            }
            receipt = standard_receipts._attempt_receipt(
                profile="nwb.2.10.0.single-integer-timeseries",
                edition="2.10.0",
                revision="a" * 40,
                fixture=standard_receipts.DEFAULT_NWB,
                fixture_kind="file",
                validator_name="pynwb.validate",
                validator_version="test",
                executable_name=str(executable),
                authority_artifact=authority,
                argv_tail=[],
                executed_at_utc="2026-07-22T12:00:00Z",
                evidence_authority="conformance",
                classifier=standard_receipts.classify_pynwb,
                internal_valid=True,
                pins=pins,
                timeout_seconds=0.01,
            )
            self.assertIsNone(receipt["independent_evidence"])
            self.assertFalse(receipt["pass"])
            self.assertIn("exceeded", receipt["diagnostics"][0])

    def test_unrecorded_package_shadow_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site_packages = Path(temporary)
            package = site_packages / "pynwb"
            package.mkdir()
            recorded = package / "validation_cli.py"
            recorded.write_text("VALID = True\n")
            shadow = package / "validation_cli"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("VALID = False\n")
            with self.assertRaisesRegex(OSError, "unrecorded"):
                standard_receipts._reject_unrecorded_site_content(
                    site_packages, {Path("pynwb/validation_cli.py")}
                )


if __name__ == "__main__":
    unittest.main()
