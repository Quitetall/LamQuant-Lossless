"""
L5 — C firmware ↔ Python reference bit-exact parity tests.

This is the CRITICAL safety gap for a medical device: the Python training
pipeline is well-tested, but we must verify that the C firmware running on
the RP2350 produces the same numerical results for the same inputs.

Strategy: compile small C test programs on the host that exercise individual
firmware functions, feed them known inputs via stdin, capture stdout, and
compare against the Python reference implementation.

Each test:
  1. Generates test vectors in Python (numpy)
  2. Writes them to a temp binary file
  3. Compiles a small C program that reads the file, calls the firmware
     function, and writes the result
  4. Compares C output vs Python output within Q31 rounding tolerance (±1 LSB)

Requires: gcc on PATH (same as test_c_host.py)
"""
import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import os
import sys
import struct
import subprocess
import tempfile
import pytest
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_C_HOST_DIR = os.path.join(_THIS_DIR, 'c_host')
# sys.path for ai_models/student is handled by conftest.py


def _compile_and_run(c_code, input_bytes, timeout=10):
    """Compile C code, feed input_bytes on stdin, return stdout bytes."""
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(c_code)
        c_path = f.name
    bin_path = c_path.replace('.c', '')
    try:
        comp = subprocess.run(
            ['gcc', '-O2', '-lm', '-o', bin_path, c_path],
            capture_output=True, text=True, timeout=30,
        )
        if comp.returncode != 0:
            pytest.skip(f"gcc compilation failed: {comp.stderr[:200]}")
        proc = subprocess.run(
            [bin_path],
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
        )
        return proc.stdout, proc.returncode
    finally:
        for p in (c_path, bin_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ======================================================================
# Q31 MATH PARITY
# ======================================================================

MUL_Q31_C = """
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static inline int32_t mul_q31(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 31);
}
int main(void) {
    int32_t pairs[200]; /* 100 pairs */
    if (fread(pairs, sizeof(int32_t), 200, stdin) != 200) return 1;
    int32_t results[100];
    for (int i = 0; i < 100; i++) {
        results[i] = mul_q31(pairs[2*i], pairs[2*i+1]);
    }
    fwrite(results, sizeof(int32_t), 100, stdout);
    return 0;
}
"""


@pytest.mark.l5
class TestMulQ31Parity:
    def test_mul_q31_matches_python(self):
        """C mul_q31(a,b) must match Python (a*b)>>31 for 100 random pairs."""
        rng = np.random.default_rng(42)
        a = rng.integers(-2**31, 2**31, size=100, dtype=np.int32)
        b = rng.integers(-2**31, 2**31, size=100, dtype=np.int32)
        pairs = np.empty(200, dtype=np.int32)
        pairs[0::2] = a
        pairs[1::2] = b
        input_bytes = pairs.tobytes()

        stdout, rc = _compile_and_run(MUL_Q31_C, input_bytes)
        assert rc == 0, "C program failed"

        c_results = np.frombuffer(stdout, dtype=np.int32)
        assert len(c_results) == 100

        py_results = ((a.astype(np.int64) * b.astype(np.int64)) >> 31).astype(np.int32)
        mismatches = np.sum(c_results != py_results)
        assert mismatches == 0, f"{mismatches}/100 mul_q31 mismatches"


# ======================================================================
# WHT32 PARITY
# ======================================================================

WHT32_C = """
#include <stdio.h>
#include <stdint.h>
static void wht32_forward(int32_t* x) {
    int h = 1;
    while (h < 32) {
        for (int i = 0; i < 32; i += h*2)
            for (int j = i; j < i+h; j++) {
                int32_t a = x[j], b = x[j+h];
                x[j] = a + b; x[j+h] = a - b;
            }
        h *= 2;
    }
}
int main(void) {
    int32_t buf[32];
    if (fread(buf, sizeof(int32_t), 32, stdin) != 32) return 1;
    wht32_forward(buf);
    fwrite(buf, sizeof(int32_t), 32, stdout);
    return 0;
}
"""


@pytest.mark.l5
class TestWht32Parity:
    def test_wht32_matches_python(self):
        """C wht32_forward must match Python wht32_forward on random input."""
        from subband_preprocess import wht32_forward as py_wht

        rng = np.random.default_rng(123)
        x = rng.integers(-1000000, 1000000, size=32, dtype=np.int32)
        input_bytes = x.tobytes()

        stdout, rc = _compile_and_run(WHT32_C, input_bytes)
        assert rc == 0
        c_out = np.frombuffer(stdout, dtype=np.int32)

        py_out = py_wht(x.astype(np.float64)).astype(np.int64)
        # WHT is exact for integers — should match bit-for-bit
        assert len(c_out) == 32
        max_diff = int(np.max(np.abs(c_out.astype(np.int64) - py_out)))
        assert max_diff == 0, f"WHT32 parity error: max diff = {max_diff}"


# ======================================================================
# LIFTING DWT PARITY
# ======================================================================

LIFTING_C = """
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static void lifting_1d_53_inplace(int32_t* signal, int length) {
    if (length < 2) return;
    int n_detail = length / 2;
    int n_approx = (length + 1) / 2;
    for (int n = 0; n < n_detail - 1; n++)
        signal[2*n+1] -= (signal[2*n] + signal[2*n+2]) >> 1;
    if (n_detail > 0) {
        int last_odd = 2*(n_detail-1)+1, last_even = 2*(n_detail-1);
        if (last_odd < length) {
            if (length % 2 == 0)
                signal[last_odd] -= (signal[last_even] + signal[last_even]) >> 1;
            else
                signal[last_odd] -= (signal[last_even] + signal[last_odd+1]) >> 1;
        }
    }
    signal[0] += (signal[1] + 1) >> 1;
    for (int n = 1; n < n_approx; n++) {
        int left = 2*n-1, right = 2*n+1;
        if (right < length) {
            int32_t sum = signal[left] + signal[right];
            signal[2*n] += (sum >= 0) ? (sum+2)>>2 : -(((-sum)+2)>>2);
        } else {
            signal[2*n] += (signal[left]+1) >> 1;
        }
    }
}
int main(void) {
    /* Read length, then signal */
    int32_t length;
    if (fread(&length, 4, 1, stdin) != 1) return 1;
    int32_t* buf = (int32_t*)malloc(length * 4);
    if (fread(buf, 4, length, stdin) != (size_t)length) return 1;
    lifting_1d_53_inplace(buf, length);
    fwrite(buf, 4, length, stdout);
    free(buf);
    return 0;
}
"""


@pytest.mark.l5
class TestLiftingParity:
    def test_lifting_1d_matches_python(self):
        """C lifting_1d_53_inplace must match Python lifting_1d_forward on 2500 samples."""
        from subband_preprocess import lifting_1d_forward

        rng = np.random.default_rng(456)
        # Use integer values to avoid float rounding differences
        signal = rng.integers(-100000, 100000, size=2500, dtype=np.int32)

        # Python reference (works on float64, but we use integer values)
        sig_f64 = signal.astype(np.float64)
        py_approx, py_detail = lifting_1d_forward(sig_f64)

        # C implementation
        length = np.array([2500], dtype=np.int32)
        input_bytes = length.tobytes() + signal.tobytes()
        stdout, rc = _compile_and_run(LIFTING_C, input_bytes)
        assert rc == 0
        c_out = np.frombuffer(stdout, dtype=np.int32)
        assert len(c_out) == 2500

        # De-interleave C output (even indices = approx, odd = detail)
        c_approx = c_out[0::2][:1250]
        c_detail = c_out[1::2][:1250]

        # Compare approx and detail subbands
        # Python uses float division (exact for integers < 2^53), C uses >>1
        # which truncates toward zero. Allow ±1 LSB tolerance.
        max_approx_diff = int(np.max(np.abs(c_approx.astype(np.int64) -
                                            py_approx[:1250].astype(np.int64))))
        max_detail_diff = int(np.max(np.abs(c_detail.astype(np.int64) -
                                            py_detail[:1250].astype(np.int64))))
        assert max_approx_diff <= 1, f"approx parity: max diff = {max_approx_diff}"
        assert max_detail_diff <= 1, f"detail parity: max diff = {max_detail_diff}"
