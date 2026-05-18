/*
 * LamQuant Gen 7.1 — 32-point Walsh-Hadamard Transform (Phase 5C)
 * ================================================================
 * In-place 32-point WHT applied to latent dimensions before FSQ.
 *
 * Purpose: Decorrelate the 32 latent dimensions so that FSQ's
 * per-dimension independent quantization is closer to optimal.
 * The TNN encoder's latent dimensions are correlated (neighboring
 * dims encode similar temporal features). WHT rotates them into a
 * basis where dimensions are more statistically independent.
 *
 * The WHT is an orthogonal transform using only additions and
 * subtractions — no multiplications:
 *   - 32-point WHT: 80 additions, 0 multiplications
 *   - Compute time: ~80 cycles at 150 MHz ≈ 0.5 µs (negligible)
 *   - Perfectly invertible (WHT is its own inverse up to scaling)
 *
 * The transform is applied per-timestep to the 32-element latent
 * vector. For T_latent=79 timesteps: 79 × 80 = 6,320 additions total.
 *
 * Normalization: We use the unnormalized WHT (no 1/sqrt(N) scaling)
 * because the FSQ quantizer adapts its range to the actual distribution.
 * The inverse WHT at the decoder divides by 32 (right shift by 5).
 *
 * All integer arithmetic. No float. In-place.
 */

#include <stdint.h>
#include "wht32.h"

/*
 * In-place 32-point Walsh-Hadamard Transform.
 *
 * Uses the butterfly decomposition: log2(32) = 5 stages,
 * each stage does N/2 = 16 butterfly operations.
 * Total: 5 × 16 = 80 additions/subtractions.
 *
 * @param x  32-element array of int32_t values (modified in-place)
 */
void wht32_forward(int32_t x[32]) {
    /* Stage 1: stride 1 */
    for (int i = 0; i < 32; i += 2) {
        int32_t a = x[i];
        int32_t b = x[i + 1];
        x[i]     = a + b;
        x[i + 1] = a - b;
    }

    /* Stage 2: stride 2 */
    for (int i = 0; i < 32; i += 4) {
        int32_t a0 = x[i];
        int32_t a1 = x[i + 1];
        int32_t b0 = x[i + 2];
        int32_t b1 = x[i + 3];
        x[i]     = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a0 - b0;
        x[i + 3] = a1 - b1;
    }

    /* Stage 3: stride 4 */
    for (int i = 0; i < 32; i += 8) {
        for (int j = 0; j < 4; j++) {
            int32_t a = x[i + j];
            int32_t b = x[i + j + 4];
            x[i + j]     = a + b;
            x[i + j + 4] = a - b;
        }
    }

    /* Stage 4: stride 8 */
    for (int i = 0; i < 32; i += 16) {
        for (int j = 0; j < 8; j++) {
            int32_t a = x[i + j];
            int32_t b = x[i + j + 8];
            x[i + j]     = a + b;
            x[i + j + 8] = a - b;
        }
    }

    /* Stage 5: stride 16 */
    for (int j = 0; j < 16; j++) {
        int32_t a = x[j];
        int32_t b = x[j + 16];
        x[j]      = a + b;
        x[j + 16] = a - b;
    }
}

/*
 * In-place 32-point inverse Walsh-Hadamard Transform.
 *
 * The WHT is self-inverse: H * H = N * I.
 * So inverse WHT = forward WHT followed by division by N=32 (>>5).
 *
 * @param x  32-element array of int32_t values (modified in-place)
 */
void wht32_inverse(int32_t x[32]) {
    /* Apply forward WHT (self-inverse) */
    wht32_forward(x);

    /* Normalize by N=32 */
    for (int i = 0; i < 32; i++) {
        x[i] >>= 5;
    }
}

/*
 * Apply WHT to all timesteps of the latent tensor.
 *
 * The latent is stored as [32][T_latent] in column-major order.
 * For each timestep t, we gather the 32 values across dimensions,
 * apply WHT, and scatter back.
 *
 * @param latent     [32][T_latent] latent tensor (modified in-place)
 * @param T_latent   Number of timesteps
 */
void wht32_apply_latent(int32_t latent[][79], int T_latent) {
    int32_t temp[32];

    for (int t = 0; t < T_latent; t++) {
        /* Gather: latent[d][t] → temp[d] */
        for (int d = 0; d < 32; d++) {
            temp[d] = latent[d][t];
        }

        wht32_forward(temp);

        /* Scatter: temp[d] → latent[d][t] */
        for (int d = 0; d < 32; d++) {
            latent[d][t] = temp[d];
        }
    }
}

/*
 * Apply inverse WHT to all timesteps (decoder side).
 */
void wht32_inverse_latent(int32_t latent[][79], int T_latent) {
    int32_t temp[32];

    for (int t = 0; t < T_latent; t++) {
        for (int d = 0; d < 32; d++) {
            temp[d] = latent[d][t];
        }

        wht32_inverse(temp);

        for (int d = 0; d < 32; d++) {
            latent[d][t] = temp[d];
        }
    }
}
