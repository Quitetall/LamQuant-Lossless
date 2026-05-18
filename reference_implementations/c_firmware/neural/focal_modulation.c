#include <stdint.h>
#include <stddef.h>
#include "../firmware_export/focal_net_weights.h"
#include "../core/math_utils.h"
#include "ternary_mac.h"
#include "focal_modulation.h"

/* Suppress array-bounds for intentional union overlay alias of raw_adc_buffer.
 * The TNN activation workspace union is larger than the _overlay member, but
 * accesses are bounded by runtime T_cur which shrinks after each strided block.
 * This is validated by the linker (raw_adc_buffer is 210,000 bytes). */
#pragma GCC diagnostic ignored "-Warray-bounds"

/*
 * LamQuant Gen 7.1 "Subband" — TNN Encoder Inference (Golden Path)
 * =================================================================
 * 112-wide strided ternary encoder on L3 approximation [21][313]:
 *
 *   premix: TernaryConv1d(21->21, k=1, s=1)
 *
 *   focal1: TernaryConv1d(21->112, k=7, s=2) + GN(4,112) + ReLU
 *           + TernaryConv1d shortcut(21->112, k=1, s=2)
 *
 *   focal2: TernaryConv1d(112->112, k=5, s=1) + GN(4,112) + ReLU
 *           + identity shortcut
 *
 *   focal3: TernaryConv1d(112->112, k=3, s=2) + GN(4,112) + ReLU
 *           + TernaryConv1d shortcut(112->112, k=1, s=2)
 *
 *   GLU bottleneck:
 *     value: TernaryConv1d(112->32, k=1, s=1) + bias
 *     gate:  TernaryConv1d(112->32, k=1, s=1) + bias
 *     latent = value * sigmoid_lut[gate]
 *
 * Total temporal stride: 2 × 1 × 2 = 4x. Input T=313 -> Latent T=79.
 *
 * All weights: 2-bit ternary {-1, 0, 1} packed 4-per-byte.
 * Alphas: Q31. Activations: int16 (W2A16).
 * Encoder packed: ~42 KB. SRAM4 budget: 43 KB.
 */

/* TERNARY_LUT removed: inner loop uses branchless conditional negate (Path 1+3) */

/* Activation buffers — much smaller than Gen 7.0.
 * Max spatial dim is T=313 (L3 approximation length).
 * After focal1 (s=2): 157.
 * After focal2 (s=1): 157.
 * After focal3 (s=2): 79.
 */
#define MAX_T 313
#define MAX_CH 112

/* TNN activation workspace — aliases raw_adc_buffer via linker.
 * Safe: TNN runs strictly after DMA + biquad + LPC + raw USB TX complete.
 * int16 mode: 3 * 112 * 313 * 2 = 210,672 bytes < raw_adc_buffer (210,000).
 * int8 mode:  3 * 112 * 313 * 1 = 105,168 bytes — fits easily.
 * The slight overshoot in int16 mode (672 bytes) is fine — we only use 21 of
 * 112 channels at the input stage, and after focal1 the temporal dim drops
 * to 157. */
extern int32_t raw_adc_buffer[21][2500];

static union {
    struct {
        act_t a[MAX_CH][MAX_T];
        act_t b[MAX_CH][MAX_T];
        act_t sc[MAX_CH][MAX_T];
    } bufs;
    int32_t _overlay[21][2500];
} *const tnn_ws_ptr = (void*)raw_adc_buffer;

#define act_buf_a   (tnn_ws_ptr->bufs.a)
#define act_buf_b   (tnn_ws_ptr->bufs.b)
#define shortcut_buf (tnn_ws_ptr->bufs.sc)

/* DMA weight prefetch ping-pong buffer (2 KB total, 2 x 1 KB)
 *
 * Each 1 KB page holds weights for one OC of the widest layer
 * (focal2: 112 inputs * 5 kernel / 4 = 140 bytes per OC, fits easily).
 *
 * Prefetch pattern:
 * 1. Core 1 starts computing OC N using weight_prefetch[0]
 * 2. DMA copies OC N+1 weights from flash into weight_prefetch[1]
 * 3. When OC N completes, swap buffers: Core 1 uses [1], DMA fills [0]
 * 4. Hides XIP cache misses completely for sequential OC processing
 *
 * Guarded by ENABLE_DMA_PREFETCH — define when DMA channel is wired up.
 * Without the flag, these consume zero SRAM (declaration only).
 */
#ifdef ENABLE_DMA_PREFETCH
static uint8_t weight_prefetch[2][1024] __attribute__((aligned(4)));
static volatile int prefetch_active_buf;  /* 0 or 1 */
#endif

/* Optimization 4: Quantize int16 post-ReLU activations to int8.
 * ReLU guarantees non-negative, so we only need unsigned clamping.
 * Only compiled when W2A8 mode is enabled. */
#ifdef USE_INT8_ACTIVATIONS
static void quantize_act_to_i8(
    int16_t src[][MAX_T], int8_t dst[][MAX_T],
    int channels, int T, int shift
) {
    for (int c = 0; c < channels; c++) {
        for (int t = 0; t < T; t++) {
            int32_t v = src[c][t] >> shift;
            if (v > 127) v = 127;
            dst[c][t] = (int8_t)v;
        }
    }
}
#endif

/* Latent output: 32 dims x 79 timesteps */
#define MAX_LAT_T 79
static int32_t latent_output[32][MAX_LAT_T] __attribute__((aligned(4)));

/*
 * Ternary 1D convolution: one output channel, one time step.
 *
 * Path 1 (SW pipelining): caller processes 2 OCs per loop iteration.
 * Path 2 (bit-serial CPOP): used when in_ch >= 32 (see below).
 * Path 3 (Zbb sat): caller uses sat_i16() for output clamping.
 *
 * Branchless conditional negate — no LUT, no multiply.
 * Weight encoding: 00=0, 01=+1, 10=-1, 11=0(pad)
 */
/**
 * @brief Optimized ternary 1D convolution — one output channel, one timestep.
 *
 * Three key optimizations over the naive per-element loop:
 *   1. Batch-4 MAC via ternary_mac_byte_fast (processes 4 weights per call,
 *      no per-element byte extraction or modulo arithmetic)
 *   2. Split boundary handling: bulk middle region has no bounds checks,
 *      only the first/last half_k positions check boundaries
 *   3. Contiguous activation gather into aligned scratch buffer for
 *      sequential memory access (eliminates strided [ic][ti] addressing
 *      in the inner loop)
 */
static inline int32_t ternary_conv1d_single(
    const act_t act[][MAX_T],
    int t_center,
    int in_ch,
    int kernel_size,
    int groups,
    int group_id,
    const uint8_t* packed_weights,
    int32_t alpha_q31  /* caller widens q15 -> q31 via << 16 */
) {
    int ch_per_group = in_ch / groups;
    int ch_start = group_id * ch_per_group;
    int half_k = kernel_size / 2;
    int32_t acc = 0;

    /* Gather activations into contiguous scratch for batch-4 MAC.
     * Max gather size: 112 channels * 7 kernel = 784 elements.
     * Static: single-threaded Core 1 inference, no reentrancy. */
    static int16_t scratch[784];
    int total = 0;

    for (int ic = ch_start; ic < ch_start + ch_per_group; ic++) {
        for (int ki = -half_k; ki <= half_k; ki++) {
            int ti = t_center + ki;
            scratch[total++] = (ti >= 0 && ti < MAX_T) ? (int16_t)act[ic][ti] : 0;
        }
    }

    /* Batch-4 MAC: process 4 weights at a time using branchless path */
    int num_bytes = total >> 2;  /* total / 4 */
    for (int i = 0; i < num_bytes; i++) {
        acc += ternary_mac_byte_fast(packed_weights[i], &scratch[i << 2]);
    }

    /* Remainder (0-3 elements) */
    int rem_start = num_bytes << 2;
    if (rem_start < total) {
        uint8_t last = packed_weights[num_bytes];
        for (int j = 0; j < total - rem_start; j++) {
            uint32_t w = ((uint32_t)last >> (j * 2)) & 3;
            int32_t neg = -(int32_t)(w >> 1);
            int32_t val = ((int32_t)scratch[rem_start + j] ^ neg) - neg;
            uint32_t nz = (w & 1) ^ (w >> 1);
            acc += val & (-(int32_t)nz);
        }
    }

    return mul_q31(acc, alpha_q31);
}

/*
 * GroupNorm (groups=4) + ReLU, in-place on act_t activations.
 */
static void groupnorm_relu_inplace(
    act_t buf[][MAX_T],
    int channels,
    int T,
    const int8_t* gamma_q7,
    const int16_t* beta_q15
) {
    int groups = 4;
    int ch_per_group = channels / groups;

    for (int g = 0; g < groups; g++) {
        int ch0 = g * ch_per_group;

        int64_t sum = 0;
        int64_t sum_sq = 0;
        int64_t count = (int64_t)ch_per_group * T;

        for (int c = ch0; c < ch0 + ch_per_group; c++) {
            for (int t = 0; t < T; t++) {
                int32_t v = buf[c][t];
                sum += v;
                sum_sq += v * v;
            }
        }

        int32_t mean = (int32_t)(sum / count);
        int64_t var = (sum_sq / count) - (int64_t)mean * mean;
        if (var < 1) var = 1;

        /* Fast integer sqrt via CLZ seed + 2 Newton iterations.
         * CLZ (Zbb: single-cycle on Hazard3) gives bit-width → initial
         * estimate 1 << (bit_width/2). Then 2 Newton iterations converge
         * to <1% error. Saves 1 division vs 3-iteration Newton from
         * a blind seed. */
        uint32_t v32 = (uint32_t)var;
        int clz = __builtin_clz(v32 | 1);  /* |1 guards zero */
        uint32_t std_approx = (uint32_t)1 << ((32 - clz) >> 1);
        std_approx = (std_approx + v32 / std_approx) >> 1;
        std_approx = (std_approx + v32 / std_approx) >> 1;
        if (std_approx == 0) std_approx = 1;

        for (int c = ch0; c < ch0 + ch_per_group; c++) {
            int32_t gamma = (int32_t)gamma_q7[c] << 24;   /* q7 -> q31 */
            int32_t beta  = (int32_t)beta_q15[c] << 16;   /* q15 -> q31 */
            for (int t = 0; t < T; t++) {
                int32_t normed = ((int32_t)buf[c][t] - mean) * 128
                                 / (int32_t)std_approx;
                int32_t scaled = mul_q31(normed, gamma) + (beta >> 16);
                if (scaled < 0) scaled = 0;
                if (scaled > ACT_MAX) scaled = ACT_MAX;
                buf[c][t] = (act_t)scaled;
            }
        }
    }
}

/*
 * Run a ternary conv + shortcut + groupnorm + relu block.
 * Returns the output temporal length.
 *
 * Optimization 3: Conv and shortcut are fused into a single pass — each OC
 * pair computes conv + shortcut + add in one iteration, eliminating the
 * separate shortcut loop and the final add-residual loop.
 *
 * Optimization 5: XIP cache-aware OC tiling — when the weight footprint
 * exceeds the 16 KB XIP cache, we tile the OC loop so each tile's weights
 * fit in cache, reducing flash access stalls.
 */
#define XIP_CACHE_SIZE 16384
#define OC_TILE_SIZE 64

static int run_focal_block(
    const act_t act_in[][MAX_T],
    act_t act_out[][MAX_T],
    int T_in,
    int in_ch, int out_ch, int kernel_size, int stride, int groups,
    const uint8_t* conv_weights,
    const int16_t* conv_alphas_q15,
    const uint8_t* sc_weights,
    const int16_t* sc_alphas_q15,
    const int8_t* norm_gamma_q7,
    const int16_t* norm_beta_q15
) {
    int T_out = (T_in + stride - 1) / stride;  /* Ceiling division for odd lengths */
    int ch_per_group = in_ch / groups;

    int weights_per_oc = ch_per_group * kernel_size;
    int bytes_per_oc = (weights_per_oc + 3) / 4;

    /* Shortcut weight geometry (k=1 pointwise) */
    int sc_bytes_per_oc = (in_ch + 3) / 4;

    /* Opt 5: If weights exceed XIP cache, tile the OC loop */
    int total_weight_bytes = out_ch * bytes_per_oc;
    int oc_tile = (total_weight_bytes > XIP_CACHE_SIZE) ? OC_TILE_SIZE : out_ch;

    /* Fused conv + shortcut path — Path 1: process 2 output channels per
     * iteration to hide pipeline latency.  Shortcut is computed inline and
     * added to conv result before the single sat_act() store.
     * Path 3: sat_act() for Zbb min/max saturation. */

    int groups_div = out_ch / groups;

    /* Path selection: bit-serial CPOP for wide layers (ch_per_group >= 32),
     * scalar batch-4 for narrow layers. CPOP processes 32 channels per
     * popcount vs 4 per scalar MAC → 8x fewer arithmetic ops on focal2/3.
     * Crossover point: ch_per_group >= 32 where bitplane conversion overhead
     * is amortized over enough channels. */
    int use_bitserial = (ch_per_group >= 32 && groups == 1);

    if (use_bitserial) {
        /* ============================================================
         * Bit-serial CPOP path (Path 2) — for focal2, focal3
         * Precompute sign/mask words per OC (amortized over all T).
         * Then per timestep: gather activations → bitplanes → cpop dot.
         * ============================================================ */
        int n_words = (ch_per_group * kernel_size + 31) / 32;
        /* Static buffers — safe because TNN inference is single-threaded
         * (Core 1 only, no preemption during encode). Avoids 2 KB stack. */
        static uint32_t sign_w[16], mask_w[16];
        static uint32_t bitplanes[8][16];

        for (int oc = 0; oc < out_ch; oc++) {
            const uint8_t* w = &conv_weights[oc * bytes_per_oc];
            int32_t alpha = (int32_t)conv_alphas_q15[oc] << 16;

            /* Precompute sign/mask for this OC's weights (once per OC) */
            weights_to_signmask(w, weights_per_oc, sign_w, mask_w, n_words);

            for (int ot = 0; ot < T_out; ot++) {
                int t_in = ot * stride;

                /* Gather activations for this conv window into flat buffer */
                static int16_t gather_buf[784];
                int half_k = kernel_size / 2;
                int gi = 0;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ki = -half_k; ki <= half_k; ki++) {
                        int ti = t_in + ki;
                        gather_buf[gi++] = (ti >= 0 && ti < MAX_T) ?
                            (int16_t)act_in[ic][ti] : 0;
                    }
                }

                /* Convert to bitplanes and dot product */
                activations_to_bitplanes(gather_buf, gi, bitplanes, n_words);
                int32_t v = ternary_dot_bitserial(
                    (const uint32_t (*)[16])bitplanes,
                    sign_w, mask_w, n_words, 8);
                v = mul_q31(v, alpha);

                /* Shortcut */
                if (sc_weights) {
                    v += ternary_conv1d_single(
                        act_in, t_in, in_ch, 1, 1, 0,
                        &sc_weights[oc * sc_bytes_per_oc],
                        (int32_t)sc_alphas_q15[oc] << 16);
                } else if (in_ch == out_ch) {
                    v += (int32_t)act_in[oc][t_in];
                }

                act_out[oc][ot] = sat_act(v);
            }
        }
    } else {
        /* ============================================================
         * Scalar batch-4 path (Path 1) — for premix, focal1, bneck
         * Paired OC processing to hide pipeline latency.
         * ============================================================ */

    for (int oc_start = 0; oc_start < out_ch; oc_start += oc_tile) {
        int oc_end = (oc_start + oc_tile < out_ch) ? oc_start + oc_tile : out_ch;

        /* Paired iteration: 2 OCs at a time */
        int oc = oc_start;
        for (; oc + 1 < oc_end; oc += 2) {
            int gid0 = oc / groups_div;
            int gid1 = (oc + 1) / groups_div;
            const uint8_t* w0 = &conv_weights[oc * bytes_per_oc];
            const uint8_t* w1 = &conv_weights[(oc + 1) * bytes_per_oc];
            int32_t alpha0 = (int32_t)conv_alphas_q15[oc] << 16;
            int32_t alpha1 = (int32_t)conv_alphas_q15[oc + 1] << 16;

            /* Shortcut weight pointers (only used if sc_weights != NULL) */
            const uint8_t* sw0 = sc_weights ? &sc_weights[oc * sc_bytes_per_oc] : NULL;
            const uint8_t* sw1 = sc_weights ? &sc_weights[(oc + 1) * sc_bytes_per_oc] : NULL;
            int32_t sa0 = sc_weights ? ((int32_t)sc_alphas_q15[oc] << 16) : 0;
            int32_t sa1 = sc_weights ? ((int32_t)sc_alphas_q15[oc + 1] << 16) : 0;

            for (int ot = 0; ot < T_out; ot++) {
                int t_in = ot * stride;
                /* Two independent convolutions — fills pipeline stalls */
                int32_t v0 = ternary_conv1d_single(
                    act_in, t_in, in_ch, kernel_size, groups, gid0, w0, alpha0);
                int32_t v1 = ternary_conv1d_single(
                    act_in, t_in, in_ch, kernel_size, groups, gid1, w1, alpha1);

                /* Fused shortcut (Opt 3): add residual inline */
                if (sc_weights) {
                    v0 += ternary_conv1d_single(
                        act_in, t_in, in_ch, 1, 1, 0, sw0, sa0);
                    v1 += ternary_conv1d_single(
                        act_in, t_in, in_ch, 1, 1, 0, sw1, sa1);
                } else if (in_ch == out_ch) {
                    /* Identity shortcut */
                    v0 += (int32_t)act_in[oc][t_in];
                    v1 += (int32_t)act_in[oc + 1][t_in];
                }
                /* else: no shortcut (channel mismatch, zero residual) */

                act_out[oc][ot] = sat_act(v0);
                act_out[oc + 1][ot] = sat_act(v1);
            }
        }
        /* Handle odd output channel count within this tile */
        if (oc < oc_end) {
            int gid = oc / groups_div;
            const uint8_t* cw = &conv_weights[oc * bytes_per_oc];
            int32_t ca = (int32_t)conv_alphas_q15[oc] << 16;

            const uint8_t* sw = sc_weights ? &sc_weights[oc * sc_bytes_per_oc] : NULL;
            int32_t sa = sc_weights ? ((int32_t)sc_alphas_q15[oc] << 16) : 0;

            for (int ot = 0; ot < T_out; ot++) {
                int t_in = ot * stride;
                int32_t v0 = ternary_conv1d_single(
                    act_in, t_in, in_ch, kernel_size, groups, gid, cw, ca);

                if (sc_weights) {
                    v0 += ternary_conv1d_single(
                        act_in, t_in, in_ch, 1, 1, 0, sw, sa);
                } else if (in_ch == out_ch) {
                    v0 += (int32_t)act_in[oc][t_in];
                }

                act_out[oc][ot] = sat_act(v0);
            }
        }
    }
    } /* end else (scalar path) */

    /* GroupNorm + ReLU (shortcut already added — no separate residual pass) */
    groupnorm_relu_inplace(act_out, out_ch, T_out, norm_gamma_q7, norm_beta_q15);

    return T_out;
}

/*
 * Run full TNN encoder inference.
 *
 * Input:  L3 approximation [21][313] (from lifting DWT, Q31)
 * Output: latent_output[32][79] (Q31, ready for FSQ)
 *
 * Called by Core 1 via scheduler_v7_1.c.
 */
void run_tnn_encoder_inference(const int32_t input[][313], int T) {
    /* Clamp T to MAX_T */
    if (T > MAX_T) T = MAX_T;

    /* Convert Q31 input to activation type (truncate Q31 -> Q15, then clamp) */
    for (int c = 0; c < 21; c++) {
        for (int t = 0; t < T; t++) {
            act_buf_a[c][t] = sat_act(input[c][t] >> 16);
        }
    }

    int T_cur = T;

    /* Spatial pre-mixing: 21->21, k=1, s=1
     * Learned common spatial pattern — decorrelates channels. */
    for (int t = 0; t < T_cur; t++) {
        act_t tmp[21];
        for (int oc = 0; oc < 21; oc++) {
            tmp[oc] = sat_act(ternary_conv1d_single(
                (const act_t(*)[MAX_T])act_buf_a,
                t, 21, 1, 1, 0,
                premix_weights + oc * ((21 + 3) / 4),
                (int32_t)premix_alphas_q15[oc] << 16) >> 16);
        }
        for (int c = 0; c < 21; c++) {
            act_buf_a[c][t] = tmp[c];
        }
    }

    /* focal1: 21->112, k=7, s=2 + shortcut(21->112, k=1, s=2)
     * 313 -> 157 */
    T_cur = run_focal_block(
        (const act_t(*)[MAX_T])act_buf_a,
        act_buf_b, T_cur,
        21, 112, 7, 2, 1,
        focal1_conv_weights, focal1_conv_alphas_q15,
        NULL, NULL,  /* No exported shortcut weights for focal1 */
        focal1_norm_weight_q7, focal1_norm_bias_q15);

    /* Ping-pong: focal2 reads from buf_b, writes to buf_a.
     * Eliminates 3 inter-block memcpy loops (112 * T_cur * sizeof(act_t)
     * each = ~70 KB total). Zero-cost buffer swap. */

    /* focal2: 112->112, k=5, s=1 + identity shortcut (reads b, writes a)
     * 157 -> 157 */
    T_cur = run_focal_block(
        (const act_t(*)[MAX_T])act_buf_b,
        act_buf_a, T_cur,
        112, 112, 5, 1, 1,
        focal2_conv_weights, focal2_conv_alphas_q15,
        NULL, NULL,  /* Identity shortcut -- same channels, stride 1 */
        focal2_norm_weight_q7, focal2_norm_bias_q15);

    /* focal3: 112->112, k=3, s=2 + shortcut (reads a, writes b)
     * 157 -> 79 */
    T_cur = run_focal_block(
        (const act_t(*)[MAX_T])act_buf_a,
        act_buf_b, T_cur,
        112, 112, 3, 2, 1,
        focal3_conv_weights, focal3_conv_alphas_q15,
        focal3_shortcut_weights, focal3_shortcut_alphas_q15,
        focal3_norm_weight_q7, focal3_norm_bias_q15);

    /* GLU Bottleneck: value * sigmoid(gate)
     *
     * value = TernaryConv1d(112→32, k=1) + bias
     * gate  = TernaryConv1d(112→32, k=1) + bias
     * latent = value * sigmoid_lut[gate]
     *
     * Sigmoid LUT: 256 entries indexed by (gate >> 23) & 0xFF
     *
     * Reads from act_buf_b (output of focal3 after ping-pong).
     */
    int bn_weights_per_oc = 112;
    int bn_bytes_per_oc = (bn_weights_per_oc + 3) / 4;

    for (int oc = 0; oc < 32; oc++) {
        /* Value path */
        const uint8_t* vw = &bneck_v_weights[oc * bn_bytes_per_oc];
        int32_t v_alpha = (int32_t)bneck_v_alphas_q15[oc] << 16;
        int32_t v_bias  = (int32_t)bneck_v_bias_q15[oc] << 16;

        /* Gate path */
        const uint8_t* gw = &bneck_g_weights[oc * bn_bytes_per_oc];
        int32_t g_alpha = (int32_t)bneck_g_alphas_q15[oc] << 16;
        int32_t g_bias  = (int32_t)bneck_g_bias_q15[oc] << 16;

        for (int t = 0; t < T_cur; t++) {
            int32_t value = ternary_conv1d_single(
                (const act_t(*)[MAX_T])act_buf_b,
                t, 112, 1, 1, 0, vw, v_alpha);
            value += (v_bias >> 16);

            int32_t gate_raw = ternary_conv1d_single(
                (const act_t(*)[MAX_T])act_buf_b,
                t, 112, 1, 1, 0, gw, g_alpha);
            gate_raw += (g_bias >> 16);

            /* Sigmoid via LUT */
            uint8_t lut_idx = (uint8_t)((gate_raw >> 23) + 128);
            int32_t gate = sigmoid_lut_q31[lut_idx];

            latent_output[oc][t] = mul_q31(value, gate);
        }
    }
}

/* Accessor for FSQ stage */
int get_latent_temporal_length(void) {
    return MAX_LAT_T;
}

const int32_t (*get_latent_output(void))[MAX_LAT_T] {
    return (const int32_t (*)[MAX_LAT_T])latent_output;
}

/* Accessor for DMA prefetch buffer — active when ENABLE_DMA_PREFETCH is set.
 * Returns pointer to the ping-pong buffer and current active index. */
#ifdef ENABLE_DMA_PREFETCH
uint8_t (*get_weight_prefetch_buf(int *active_idx))[1024] {
    if (active_idx) *active_idx = prefetch_active_buf;
    return weight_prefetch;
}
#endif
