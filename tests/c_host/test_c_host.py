"""
L2/L5 — C firmware host-compilation and execution tests.

Compiles the firmware C test harness (tests/c_host/test_c_firmware.c) on the
host using gcc, then runs the resulting binary. The C harness contains 186
self-contained unit tests covering:

  - Q31/Q30 fixed-point math primitives (mul, add_sat, sub_sat)
  - Ternary MAC (KAT, edge cases, exhaustive 256-combo, Q31 alpha)
  - CRC32 known vectors
  - LFSR period + batch32 equivalence
  - Integer square root
  - Biquad IIR filters (DC rejection, LP stability, impulse, cascade, state)
  - Lifting DWT (roundtrip even/odd/2500, constant, boundary, neg rounding)
  - WHT32 (roundtrip, delta function)
  - LPC Levinson-Durbin (flat signal, overflow guard)
  - LPC delta codec (keyframe roundtrip, Q8 roundtrip, no-prev guard)
  - FSQ quantization (range, monotonicity)
  - Raw USB packet format
  - Output mode enum

If gcc is not installed or compilation fails, tests skip gracefully.

Paranoia levels:
  L2 — compilation without errors is a mathematical invariant
  L5 — numerical correctness of firmware primitives
"""
import os
import subprocess
import pytest

C_TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # was tests/c_host/ subdir; file moved into tests/c_host/
C_SOURCE = os.path.join(C_TEST_DIR, 'test_c_firmware.c')
C_BINARY = os.path.join(C_TEST_DIR, 'test_c_firmware')


@pytest.mark.l2
@pytest.mark.l5
@pytest.mark.c_host
class TestCFirmwareHost:
    """AUDIT (2026-04-27): Hardened compilation and execution error handling.
    Previously test_run called subprocess.run(..., check=True) for compilation
    which raised CalledProcessError — a valid exception, but the error message
    didn't include the compiler output. Now both tests assert explicitly with
    full stdout/stderr in the failure message.
    """

    def test_compile(self):
        """C test harness must compile without errors on the host."""
        result = subprocess.run(
            ['gcc', '-O2', '-Wall', '-Wextra',
             '-I', C_TEST_DIR,
             '-o', C_BINARY, C_SOURCE, '-lm'],
            capture_output=True, text=True
        )
        assert result.returncode == 0, \
            f"Compilation failed (rc={result.returncode}):\n" \
            f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_run(self):
        """All C unit tests must pass.

        AUDIT (2026-04-27): If binary is missing, we compile first with
        explicit error checking (not check=True which hides compiler errors).
        """
        if not os.path.exists(C_BINARY):
            compile_result = subprocess.run(
                ['gcc', '-O2', '-I', C_TEST_DIR,
                 '-o', C_BINARY, C_SOURCE, '-lm'],
                capture_output=True, text=True
            )
            assert compile_result.returncode == 0, (
                f"C compilation failed in test_run (rc={compile_result.returncode}):\n"
                f"stdout: {compile_result.stdout}\nstderr: {compile_result.stderr}"
            )

        result = subprocess.run(
            [C_BINARY], capture_output=True, text=True, timeout=30
        )
        # Always include output in test record for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr}")
        assert result.returncode == 0, \
            f"C tests failed (rc={result.returncode}):\n" \
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
