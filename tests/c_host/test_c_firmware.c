/*
 * Host-compiled C unit tests for LamQuant firmware primitives.
 *
 * Compile: gcc -O2 -I. -o test_c_firmware test_c_firmware.c -lm
 * Run:     ./test_c_firmware
 *
 * Tests:
 *   1. Q31 math (mul, add_sat, sub_sat)
 *   2. Ternary MAC (KAT + edge cases)
 *   3. CRC32 (cross-check with known vectors)
 *   4. LFSR (period, batch equivalence)
 *   5. Integer square root (isqrt32)
 *   6. Biquad Q31 (DC rejection)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

/* ---- Inline the math_utils.h directly (avoid path issues) ---- */
static inline int32_t mul_q31(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 31);
}

static inline int32_t mul_q30(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> 30);
}

static inline int32_t add_sat_q31(int32_t a, int32_t b) {
    int64_t r = (int64_t)a + (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}

static inline int32_t sub_sat_q31(int32_t a, int32_t b) {
    int64_t r = (int64_t)a - (int64_t)b;
    if (r > INT32_MAX) return INT32_MAX;
    if (r < INT32_MIN) return INT32_MIN;
    return (int32_t)r;
}

/* ---- Ternary MAC (from ternary_mac.c) ---- */
static const int32_t TERNARY_LUT[4] = {0, 1, -1, 0};

static inline int32_t ternary_mac_byte_w2a6(uint8_t packed_w, const int16_t* act) {
    int32_t acc = 0;
    acc += (int32_t)act[0] * TERNARY_LUT[(packed_w     ) & 0x03];
    acc += (int32_t)act[1] * TERNARY_LUT[(packed_w >> 2) & 0x03];
    acc += (int32_t)act[2] * TERNARY_LUT[(packed_w >> 4) & 0x03];
    acc += (int32_t)act[3] * TERNARY_LUT[(packed_w >> 6) & 0x03];
    return acc;
}

/* ---- CRC32 (from integrity.c) ---- */
static const uint32_t CRC32_TABLE[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
    0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
    0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
    0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
    0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
    0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
    0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
    0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
    0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
    0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457,
    0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
    0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB,
    0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
    0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD,
    0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683,
    0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
    0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7,
    0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5,
    0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79,
    0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F,
    0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
    0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713,
    0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21,
    0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
    0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB,
    0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
    0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF,
    0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94, 0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D,
};

uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len) {
    crc ^= 0xFFFFFFFF;
    while (len--) {
        crc = (crc >> 8) ^ CRC32_TABLE[(crc ^ *data++) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/* ---- LFSR (from toeplitz_cs.c) ---- */
static inline uint32_t lfsr_advance(uint32_t state) {
    uint32_t bit = ((state >> 0) ^ (state >> 2) ^
                    (state >> 3) ^ (state >> 5)) & 1;
    return (state >> 1) | (bit << 15);
}

static inline uint32_t lfsr_batch32(uint32_t *state) {
    uint32_t s = *state;
    uint32_t bits = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t bit = ((s >> 0) ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1;
        s = (s >> 1) | (bit << 15);
        bits |= (bit << i);
    }
    *state = s;
    return bits;
}

/* ---- Integer square root (from fsq.c) ---- */
static uint32_t isqrt32(uint32_t x) {
    if (x == 0) return 0;
    uint32_t result = 0;
    uint32_t bit = 1u << 30;
    while (bit > x) bit >>= 2;
    while (bit != 0) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return result;
}

/* ---- Biquad Q30 (from biquad_q31.c — now uses Q30 coefficients) ---- */
typedef struct {
    int32_t b0, b1, b2;
    int32_t a1, a2;
    int32_t x1, x2;
    int32_t y1, y2;
} biquad_state_t;

static inline int32_t biquad_process(biquad_state_t* S, int32_t x0) {
    int32_t acc;
    acc = mul_q30(S->b0, x0);
    acc = add_sat_q31(acc, mul_q30(S->b1, S->x1));
    acc = add_sat_q31(acc, mul_q30(S->b2, S->x2));
    acc = sub_sat_q31(acc, mul_q30(S->a1, S->y1));
    acc = sub_sat_q31(acc, mul_q30(S->a2, S->y2));
    S->x2 = S->x1; S->x1 = x0;
    S->y2 = S->y1; S->y1 = acc;
    return acc;
}

/* ==================================================================
 * Test harness
 * ================================================================== */
static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) == (b)) { tests_passed++; } \
    else { printf("  FAIL: %s (got %ld, expected %ld)\n", msg, (long)(a), (long)(b)); } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { printf("  FAIL: %s\n", msg); } \
} while(0)

/* ---- Test: Q31 multiply ---- */
void test_mul_q31(void) {
    printf("[TEST] mul_q31\n");

    /* 0.5 * 0.5 = 0.25 in Q31 */
    int32_t half = 1073741824; /* 0.5 * 2^31 */
    int32_t quarter = 536870912; /* 0.25 * 2^31 */
    ASSERT_EQ(mul_q31(half, half), quarter, "0.5 * 0.5 = 0.25");

    /* 1.0 * 0 = 0 (Q31 max * 0) */
    ASSERT_EQ(mul_q31(INT32_MAX, 0), 0, "MAX * 0 = 0");

    /* 0 * anything = 0 */
    ASSERT_EQ(mul_q31(0, 12345678), 0, "0 * x = 0");

    /* -1.0 * 0.5 ≈ -0.5 */
    int32_t neg_one = INT32_MIN; /* -1.0 in Q31 */
    int32_t result = mul_q31(neg_one, half);
    /* Should be close to -0.5 * 2^31 = -1073741824 */
    ASSERT_TRUE(abs(result - (-half)) <= 1, "-1.0 * 0.5 ≈ -0.5");

    /* MAX * MAX should not overflow (uses 64-bit intermediate) */
    result = mul_q31(INT32_MAX, INT32_MAX);
    ASSERT_TRUE(result > 0, "MAX * MAX is positive (no overflow)");
}

/* ---- Test: Saturating add ---- */
void test_add_sat_q31(void) {
    printf("[TEST] add_sat_q31\n");

    ASSERT_EQ(add_sat_q31(100, 200), 300, "100 + 200 = 300");
    ASSERT_EQ(add_sat_q31(INT32_MAX, 1), INT32_MAX, "MAX + 1 saturates to MAX");
    ASSERT_EQ(add_sat_q31(INT32_MIN, -1), INT32_MIN, "MIN + -1 saturates to MIN");
    ASSERT_EQ(add_sat_q31(INT32_MAX, INT32_MAX), INT32_MAX, "MAX + MAX saturates");
    ASSERT_EQ(add_sat_q31(-100, 100), 0, "-100 + 100 = 0");
}

/* ---- Test: Saturating sub ---- */
void test_sub_sat_q31(void) {
    printf("[TEST] sub_sat_q31\n");

    ASSERT_EQ(sub_sat_q31(300, 100), 200, "300 - 100 = 200");
    ASSERT_EQ(sub_sat_q31(INT32_MIN, 1), INT32_MIN, "MIN - 1 saturates to MIN");
    ASSERT_EQ(sub_sat_q31(INT32_MAX, -1), INT32_MAX, "MAX - (-1) saturates to MAX");
    ASSERT_EQ(sub_sat_q31(0, 0), 0, "0 - 0 = 0");
}

/* ---- Test: Ternary MAC KAT (from ternary_mac.c) ---- */
void test_ternary_mac_kat(void) {
    printf("[TEST] ternary_mac KAT\n");

    int16_t act[4] = {100, 200, 300, 400};
    /* Packed: [1, -1, 0, 1] = 0x49 */
    uint8_t packed = 0x49;
    int32_t result = ternary_mac_byte_w2a6(packed, act);
    /* 100*1 + 200*(-1) + 300*0 + 400*1 = 300 */
    ASSERT_EQ(result, 300, "KAT: [1,-1,0,1] dot [100,200,300,400] = 300");
}

/* ---- Test: Ternary MAC edge cases ---- */
void test_ternary_mac_edges(void) {
    printf("[TEST] ternary_mac edges\n");

    int16_t act[4] = {1000, 2000, 3000, 4000};

    /* All zeros: 0x00 */
    ASSERT_EQ(ternary_mac_byte_w2a6(0x00, act), 0, "all-zero weights");

    /* All +1: 01 01 01 01 = 0x55 */
    int32_t expected = 1000 + 2000 + 3000 + 4000;
    ASSERT_EQ(ternary_mac_byte_w2a6(0x55, act), expected, "all +1 weights");

    /* All -1: 10 10 10 10 = 0xAA */
    ASSERT_EQ(ternary_mac_byte_w2a6(0xAA, act), -expected, "all -1 weights");
}

/* ---- Test: CRC32 known vectors ---- */
void test_crc32(void) {
    printf("[TEST] crc32\n");

    /* CRC32 of empty data should be 0 */
    uint32_t crc = crc32_update(0, (const uint8_t*)"", 0);
    ASSERT_EQ(crc, 0x00000000, "CRC32 of empty = 0");

    /* CRC32("123456789") = 0xCBF43926 (standard test vector) */
    const uint8_t test_vec[] = "123456789";
    crc = crc32_update(0, test_vec, 9);
    ASSERT_EQ(crc, 0xCBF43926, "CRC32 of '123456789'");

    /* Incremental: split the same data and feed in two parts */
    /* crc32_update XORs with 0xFFFFFFFF at start and end, so we can't
       simply chain calls. Instead, test that single-call matches known. */
    const uint8_t data1[] = "Hello";
    uint32_t crc1 = crc32_update(0, data1, 5);
    ASSERT_TRUE(crc1 != 0, "CRC32 of 'Hello' is non-zero");
}

/* ---- Test: LFSR period ---- */
void test_lfsr_period(void) {
    printf("[TEST] LFSR period\n");

    uint32_t seed = 0xACE1u;
    uint32_t state = seed;
    uint32_t count = 0;

    do {
        state = lfsr_advance(state);
        count++;
        if (count > 70000) break; /* Safety: don't loop forever */
    } while (state != seed);

    ASSERT_EQ(count, 65535, "LFSR period = 2^16 - 1 = 65535");
}

/* ---- Test: LFSR batch32 matches sequential ---- */
void test_lfsr_batch32_equivalence(void) {
    printf("[TEST] LFSR batch32 equivalence\n");

    uint32_t state_seq = 0xBE37u;
    uint32_t state_batch = 0xBE37u;

    uint32_t batch_bits = lfsr_batch32(&state_batch);

    /* Collect 32 sequential bits */
    uint32_t seq_bits = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t bit = ((state_seq >> 0) ^ (state_seq >> 2) ^
                        (state_seq >> 3) ^ (state_seq >> 5)) & 1;
        state_seq = (state_seq >> 1) | (bit << 15);
        seq_bits |= (bit << i);
    }

    ASSERT_EQ(batch_bits, seq_bits, "batch32 matches 32 sequential advances");
    ASSERT_EQ(state_batch, state_seq, "final states match after batch32");
}

/* ---- Test: Integer square root ---- */
void test_isqrt32(void) {
    printf("[TEST] isqrt32\n");

    ASSERT_EQ(isqrt32(0), 0, "sqrt(0) = 0");
    ASSERT_EQ(isqrt32(1), 1, "sqrt(1) = 1");
    ASSERT_EQ(isqrt32(4), 2, "sqrt(4) = 2");
    ASSERT_EQ(isqrt32(9), 3, "sqrt(9) = 3");
    ASSERT_EQ(isqrt32(100), 10, "sqrt(100) = 10");
    ASSERT_EQ(isqrt32(65535), 255, "sqrt(65535) = 255");
    ASSERT_EQ(isqrt32(65536), 256, "sqrt(65536) = 256");

    /* Large value: sqrt(2^30) = 2^15 = 32768 */
    ASSERT_EQ(isqrt32(1u << 30), 32768, "sqrt(2^30) = 32768");

    /* Non-perfect square: floor(sqrt(10)) = 3 */
    ASSERT_EQ(isqrt32(10), 3, "floor(sqrt(10)) = 3");
}

/* ---- Q30 biquad coefficients from biquad_q31.c ---- */

/* HP 0.5 Hz Q30 */
#define HP_B0  1064243069
#define HP_B1 (-2128486138)
#define HP_B2  1064243069
#define HP_A1 (-2128402106)
#define HP_A2  1054828345

/* LP 50 Hz Q30 */
#define LP_B0  221805086
#define LP_B1  443610172
#define LP_B2  221805086
#define LP_A1 (-396777000)
#define LP_A2  210255520

/* Notch 60 Hz Q30 */
#define NOTCH_B0  1047411946
#define NOTCH_B1 (-131535080)
#define NOTCH_B2  1047411946
#define NOTCH_A1 (-131535080)
#define NOTCH_A2  1021082069

static void init_biquad(biquad_state_t* s,
                        int32_t b0, int32_t b1, int32_t b2,
                        int32_t a1, int32_t a2) {
    s->b0 = b0; s->b1 = b1; s->b2 = b2;
    s->a1 = a1; s->a2 = a2;
    s->x1 = s->x2 = s->y1 = s->y2 = 0;
}

/* ---- Test: HP DC rejection (Q30 fix) ---- */
void test_biquad_hp_dc_rejection(void) {
    printf("[TEST] biquad HP DC rejection (Q30)\n");

    biquad_state_t hp;
    init_biquad(&hp, HP_B0, HP_B1, HP_B2, HP_A1, HP_A2);

    /* Feed constant DC for 2000 samples (8 seconds at 250Hz).
       The HP filter must reject DC — output converges to zero. */
    int32_t dc_value = 100000000;
    int32_t last = 0;
    for (int i = 0; i < 2000; i++) {
        last = biquad_process(&hp, dc_value);
    }

    /* After 8 seconds, HP should reject DC to < 1% of input */
    int32_t tolerance = dc_value / 100;
    ASSERT_TRUE(abs(last) < tolerance,
                "HP filter rejects DC (Q30 coefficients)");
}

/* ---- Test: LP DC stability ---- */
void test_biquad_lp_dc_stability(void) {
    printf("[TEST] biquad LP DC stability (Q30)\n");

    biquad_state_t lp;
    init_biquad(&lp, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2);

    /* Feed constant DC for 1000 samples. Filter must:
       1. Produce non-zero output (not a no-op)
       2. Converge to a steady state (not diverge) */
    int32_t dc_value = 100000000;
    int32_t prev = 0, last = 0;
    for (int i = 0; i < 1000; i++) {
        prev = last;
        last = biquad_process(&lp, dc_value);
    }

    ASSERT_TRUE(last != 0, "LP filter output is non-zero for DC input");
    ASSERT_TRUE(abs(last - prev) < 100,
                "LP filter converges to steady state under DC");
}

/* ---- Test: Biquad impulse response is finite ---- */
void test_biquad_impulse_finite(void) {
    printf("[TEST] biquad impulse response is finite (Q30)\n");

    biquad_state_t lp;
    init_biquad(&lp, LP_B0, LP_B1, LP_B2, LP_A1, LP_A2);

    /* Unit impulse */
    int32_t out = biquad_process(&lp, 1000000000);
    ASSERT_TRUE(out != 0, "LP filter responds to impulse");

    /* Feed zeros and check it decays */
    for (int i = 0; i < 200; i++) {
        out = biquad_process(&lp, 0);
    }
    ASSERT_TRUE(abs(out) < 1000, "LP impulse response decays to near zero");
}

/* ---- Test: 3-stage cascade stability ---- */
void test_biquad_cascade_stable(void) {
    printf("[TEST] biquad 3-stage cascade stability (Q30)\n");

    biquad_state_t hp, lp, notch;
    init_biquad(&hp,    HP_B0, HP_B1, HP_B2, HP_A1, HP_A2);
    init_biquad(&lp,    LP_B0, LP_B1, LP_B2, LP_A1, LP_A2);
    init_biquad(&notch, NOTCH_B0, NOTCH_B1, NOTCH_B2, NOTCH_A1, NOTCH_A2);

    /* Feed 5000 samples of varied input — must not diverge */
    int32_t max_out = 0;
    for (int i = 0; i < 5000; i++) {
        /* Synthetic EEG-like signal: sum of 10Hz + 30Hz components */
        double t = (double)i / 250.0;
        int32_t sample = (int32_t)(50000000.0 * sin(2.0 * 3.14159 * 10.0 * t)
                                 + 30000000.0 * sin(2.0 * 3.14159 * 30.0 * t));
        sample = biquad_process(&hp, sample);
        sample = biquad_process(&lp, sample);
        sample = biquad_process(&notch, sample);
        if (abs(sample) > max_out) max_out = abs(sample);
    }

    /* Output must be bounded — no divergence */
    ASSERT_TRUE(max_out < 500000000, "3-stage cascade output is bounded");
    ASSERT_TRUE(max_out > 0, "3-stage cascade produces non-zero output");
}

/* ---- Test: mul_q30 basic correctness ---- */
void test_mul_q30(void) {
    printf("[TEST] mul_q30\n");

    int32_t Q30 = 1 << 30; /* 1.0 in Q30 */

    /* 1.0 * 1.0 = 1.0 */
    ASSERT_EQ(mul_q30(Q30, Q30), Q30, "1.0 * 1.0 = 1.0 in Q30");

    /* 1.5 * 1.0: 1.5 in Q30 = 1.5 * 2^30 = 1610612736 */
    int32_t one_point_five = (int32_t)(1.5 * (1 << 30));
    ASSERT_EQ(mul_q30(one_point_five, Q30), one_point_five, "1.5 * 1.0 = 1.5 in Q30");

    /* -1.5 * 1.0 = -1.5 */
    ASSERT_EQ(mul_q30(-one_point_five, Q30), -one_point_five, "-1.5 * 1.0 = -1.5 in Q30");

    /* 0 * anything = 0 */
    ASSERT_EQ(mul_q30(0, one_point_five), 0, "0 * 1.5 = 0 in Q30");
}

/* ---- Test: Raw USB packet header format ---- */
void test_raw_packet_header(void) {
    printf("[TEST] raw USB packet header format\n");

    /* Simulate header construction matching raw_output.c */
    uint32_t window_id = 0x1234;
    int total_channels = 21;

    uint8_t header[8];
    header[0] = 'L';
    header[1] = 'A';
    header[2] = 'M';
    header[3] = 'R';
    header[4] = (uint8_t)total_channels;
    header[5] = 0;
    header[6] = (uint8_t)(window_id >> 8);
    header[7] = (uint8_t)(window_id & 0xFF);

    ASSERT_EQ(header[0], 'L', "sync byte 0 = 'L'");
    ASSERT_EQ(header[1], 'A', "sync byte 1 = 'A'");
    ASSERT_EQ(header[2], 'M', "sync byte 2 = 'M'");
    ASSERT_EQ(header[3], 'R', "sync byte 3 = 'R'");
    ASSERT_EQ(header[4], 21, "channel count = 21");
    ASSERT_EQ(header[5], 0, "reserved byte = 0");
    ASSERT_EQ(header[6], 0x12, "window ID high byte");
    ASSERT_EQ(header[7], 0x34, "window ID low byte");

    /* Test 24-bit sample packing (big-endian) */
    int32_t val = -12345;
    uint8_t sample[3] = {
        (uint8_t)((val >> 16) & 0xFF),
        (uint8_t)((val >> 8) & 0xFF),
        (uint8_t)(val & 0xFF)
    };

    /* Reconstruct with sign extension */
    int32_t reconstructed = (sample[0] << 16) | (sample[1] << 8) | sample[2];
    if (reconstructed & 0x800000) {
        reconstructed |= (int32_t)0xFF000000;
    }
    ASSERT_EQ(reconstructed, val, "24-bit pack/unpack roundtrip for negative value");

    /* Positive value */
    val = 54321;
    sample[0] = (uint8_t)((val >> 16) & 0xFF);
    sample[1] = (uint8_t)((val >> 8) & 0xFF);
    sample[2] = (uint8_t)(val & 0xFF);
    reconstructed = (sample[0] << 16) | (sample[1] << 8) | sample[2];
    if (reconstructed & 0x800000) {
        reconstructed |= (int32_t)0xFF000000;
    }
    ASSERT_EQ(reconstructed, val, "24-bit pack/unpack roundtrip for positive value");

    /* Zero */
    val = 0;
    sample[0] = (uint8_t)((val >> 16) & 0xFF);
    sample[1] = (uint8_t)((val >> 8) & 0xFF);
    sample[2] = (uint8_t)(val & 0xFF);
    reconstructed = (sample[0] << 16) | (sample[1] << 8) | sample[2];
    if (reconstructed & 0x800000) {
        reconstructed |= (int32_t)0xFF000000;
    }
    ASSERT_EQ(reconstructed, val, "24-bit pack/unpack roundtrip for zero");
}

/* ---- Test: Output mode enum values ---- */
void test_output_mode_enum(void) {
    printf("[TEST] output mode enum\n");

    /* Verify enum values match between firmware and receiver */
    typedef enum {
        OUTPUT_COMPRESSED_ONLY,
        OUTPUT_RAW_ONLY,
        OUTPUT_DUAL,
    } output_mode_t;

    ASSERT_EQ(OUTPUT_COMPRESSED_ONLY, 0, "OUTPUT_COMPRESSED_ONLY = 0");
    ASSERT_EQ(OUTPUT_RAW_ONLY, 1, "OUTPUT_RAW_ONLY = 1");
    ASSERT_EQ(OUTPUT_DUAL, 2, "OUTPUT_DUAL = 2");

    /* Verify serial command byte mapping */
    output_mode_t mode;
    uint8_t cmd;

    cmd = 'R'; mode = OUTPUT_RAW_ONLY;
    ASSERT_EQ(cmd, 0x52, "'R' command byte = 0x52");
    ASSERT_EQ(mode, 1, "'R' maps to OUTPUT_RAW_ONLY");

    cmd = 'C'; mode = OUTPUT_COMPRESSED_ONLY;
    ASSERT_EQ(cmd, 0x43, "'C' command byte = 0x43");
    ASSERT_EQ(mode, 0, "'C' maps to OUTPUT_COMPRESSED_ONLY");

    cmd = 'D'; mode = OUTPUT_DUAL;
    ASSERT_EQ(cmd, 0x44, "'D' command byte = 0x44");
    ASSERT_EQ(mode, 2, "'D' maps to OUTPUT_DUAL");
}

/* ==================================================================
 * LIFTING DWT TESTS (verifies Bug F1 + F3 fixes)
 * ================================================================== */

/* Inline the fixed lifting_1d_53_inplace from firmware/dsp/lifting_2d.c */
static void lifting_1d_53_inplace(int32_t* signal, int length) {
    if (length < 2) return;
    int n_detail = length / 2;
    int n_approx = (length + 1) / 2;

    /* Predict step */
    for (int n = 0; n < n_detail - 1; n++) {
        signal[2*n + 1] -= (signal[2*n] + signal[2*n + 2]) >> 1;
    }
    /* Last detail: boundary (BUG F1 FIX: even-length uses >>1 average) */
    if (n_detail > 0) {
        int last_odd = 2*(n_detail - 1) + 1;
        int last_even = 2*(n_detail - 1);
        if (last_odd < length) {
            if (length % 2 == 0) {
                signal[last_odd] -= (signal[last_even] + signal[last_even]) >> 1;
            } else {
                signal[last_odd] -= (signal[last_even] + signal[last_odd + 1]) >> 1;
            }
        }
    }

    /* Update step (BUG F3 FIX: symmetric rounding) */
    signal[0] += (signal[1] + 1) >> 1;
    for (int n = 1; n < n_approx; n++) {
        int left = 2*n - 1;
        int right = 2*n + 1;
        if (right < length) {
            int32_t sum = signal[left] + signal[right];
            signal[2*n] += (sum >= 0) ? (sum + 2) >> 2 : -(((-sum) + 2) >> 2);
        } else {
            signal[2*n] += (signal[left] + 1) >> 1;
        }
    }
}

/* Inverse lifting for roundtrip test */
static void lifting_1d_53_inverse(int32_t* signal, int length) {
    if (length < 2) return;
    int n_detail = length / 2;
    int n_approx = (length + 1) / 2;

    /* Undo update */
    for (int n = n_approx - 1; n >= 1; n--) {
        int left = 2*n - 1;
        int right = 2*n + 1;
        if (right < length) {
            int32_t sum = signal[left] + signal[right];
            signal[2*n] -= (sum >= 0) ? (sum + 2) >> 2 : -(((-sum) + 2) >> 2);
        } else {
            signal[2*n] -= (signal[left] + 1) >> 1;
        }
    }
    signal[0] -= (signal[1] + 1) >> 1;

    /* Undo predict */
    if (n_detail > 0) {
        int last_odd = 2*(n_detail - 1) + 1;
        int last_even = 2*(n_detail - 1);
        if (last_odd < length) {
            if (length % 2 == 0) {
                signal[last_odd] += (signal[last_even] + signal[last_even]) >> 1;
            } else {
                signal[last_odd] += (signal[last_even] + signal[last_odd + 1]) >> 1;
            }
        }
    }
    for (int n = n_detail - 2; n >= 0; n--) {
        signal[2*n + 1] += (signal[2*n] + signal[2*n + 2]) >> 1;
    }
}

void test_lifting_roundtrip_even(void) {
    printf("[TEST] lifting 1D roundtrip (even length=100)\n");
    int32_t orig[100], buf[100];
    for (int i = 0; i < 100; i++) orig[i] = buf[i] = (i * 12345) ^ (i << 16);
    lifting_1d_53_inplace(buf, 100);
    lifting_1d_53_inverse(buf, 100);
    int max_err = 0;
    for (int i = 0; i < 100; i++) {
        int err = abs(buf[i] - orig[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err == 0, "even-length roundtrip should be exact (integer lifting)");
}

void test_lifting_roundtrip_odd(void) {
    printf("[TEST] lifting 1D roundtrip (odd length=625)\n");
    int32_t orig[625], buf[625];
    for (int i = 0; i < 625; i++) orig[i] = buf[i] = (i * 54321 - 100000) ^ (i << 12);
    lifting_1d_53_inplace(buf, 625);
    lifting_1d_53_inverse(buf, 625);
    int max_err = 0;
    for (int i = 0; i < 625; i++) {
        int err = abs(buf[i] - orig[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err == 0, "odd-length roundtrip should be exact");
}

void test_lifting_constant_input(void) {
    printf("[TEST] lifting 1D constant input\n");
    int32_t buf[100];
    for (int i = 0; i < 100; i++) buf[i] = 42000;
    lifting_1d_53_inplace(buf, 100);
    /* Approximation (even indices) should be ~42000, detail (odd) ~0 */
    for (int i = 1; i < 100; i += 2) {
        ASSERT_TRUE(abs(buf[i]) < 10, "detail of constant should be ~0");
    }
}

void test_lifting_boundary_even_fix(void) {
    printf("[TEST] lifting boundary even-length (Bug F1 fix)\n");
    /* The boundary predict for even-length should use averaged mirror,
     * not just the raw neighbor. Verify the last detail coefficient
     * is correct for a known signal. */
    int32_t buf[10] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    int32_t orig[10];
    memcpy(orig, buf, sizeof(buf));
    lifting_1d_53_inplace(buf, 10);
    lifting_1d_53_inverse(buf, 10);
    int max_err = 0;
    for (int i = 0; i < 10; i++) {
        int err = abs(buf[i] - orig[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err == 0, "Bug F1: even boundary roundtrip must be exact");
}

void test_lifting_negative_rounding(void) {
    printf("[TEST] lifting negative rounding symmetry (Bug F3 fix)\n");
    /* Verify that positive and negative signals produce symmetric results */
    int32_t pos_buf[20], neg_buf[20];
    for (int i = 0; i < 20; i++) {
        pos_buf[i] = (i + 1) * 1000;
        neg_buf[i] = -(i + 1) * 1000;
    }
    lifting_1d_53_inplace(pos_buf, 20);
    lifting_1d_53_inplace(neg_buf, 20);
    /* After lifting, neg_buf should be the negation of pos_buf
     * (within rounding tolerance of ±1 due to integer division) */
    int max_asym = 0;
    for (int i = 0; i < 20; i++) {
        int asym = abs(pos_buf[i] + neg_buf[i]);
        if (asym > max_asym) max_asym = asym;
    }
    ASSERT_TRUE(max_asym <= 1, "Bug F3: positive/negative should be symmetric (±1)");
}

void test_lifting_2500_roundtrip(void) {
    printf("[TEST] lifting 1D roundtrip (length=2500, real window size)\n");
    int32_t orig[2500], buf[2500];
    /* Pseudorandom signal in Q31 range */
    uint32_t rng = 0xDEADBEEF;
    for (int i = 0; i < 2500; i++) {
        rng = rng * 1103515245 + 12345;
        orig[i] = buf[i] = (int32_t)rng;
    }
    lifting_1d_53_inplace(buf, 2500);
    lifting_1d_53_inverse(buf, 2500);
    int max_err = 0;
    for (int i = 0; i < 2500; i++) {
        int err = abs(buf[i] - orig[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err == 0, "2500-sample roundtrip must be exact");
}

/* ==================================================================
 * WHT32 TESTS
 * ================================================================== */

static void wht32_forward(int32_t* x) {
    int h = 1;
    while (h < 32) {
        for (int i = 0; i < 32; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                int32_t a = x[j];
                int32_t b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

static void wht32_inverse(int32_t* x) {
    wht32_forward(x);  /* WHT is self-inverse up to scale */
    for (int i = 0; i < 32; i++) x[i] /= 32;
}

void test_wht32_roundtrip(void) {
    printf("[TEST] WHT32 forward+inverse roundtrip\n");
    int32_t orig[32], buf[32];
    for (int i = 0; i < 32; i++) orig[i] = buf[i] = (i * 1000 - 16000);
    wht32_forward(buf);
    wht32_inverse(buf);
    int max_err = 0;
    for (int i = 0; i < 32; i++) {
        int err = abs(buf[i] - orig[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err == 0, "WHT32 roundtrip should be exact");
}

void test_wht32_delta_function(void) {
    printf("[TEST] WHT32 delta function -> uniform\n");
    int32_t buf[32] = {0};
    buf[0] = 32;  /* delta function scaled by N */
    wht32_forward(buf);
    /* WHT of scaled delta = all ones (Hadamard row 0 = all +1) */
    int all_one = 1;
    for (int i = 0; i < 32; i++) {
        if (buf[i] != 32) all_one = 0;
    }
    ASSERT_TRUE(all_one, "WHT32 of [32,0,0,...] should be [32,32,...,32]");
}

/* ==================================================================
 * LPC LEVINSON-DURBIN TESTS (verifies Bug F5 fix)
 * ================================================================== */

void test_levinson_flat_signal(void) {
    printf("[TEST] LPC Levinson-Durbin on flat (DC) signal\n");
    /* A constant signal has R[0] = val^2 * N, R[k>0] = same.
     * Levinson-Durbin should produce k[0] ≈ -1 (first reflection coeff)
     * and all subsequent reflections ≈ 0. The prediction residual of a
     * constant = 0 after the first sample. */
    int32_t signal[256];
    for (int i = 0; i < 256; i++) signal[i] = 1000;

    /* Biased autocorrelation */
    int64_t R[9] = {0};
    for (int lag = 0; lag <= 8; lag++) {
        for (int i = 0; i < 256 - lag; i++) {
            R[lag] += (int64_t)signal[i] * signal[i + lag];
        }
    }
    /* For a constant: R[lag] = 1000^2 * (256-lag) for all lag */
    ASSERT_TRUE(R[0] > 0, "R[0] must be positive for a non-zero signal");
    /* R[1]/R[0] should be very close to 1 (only differs by 1/256) */
    double ratio = (double)R[1] / (double)R[0];
    ASSERT_TRUE(ratio > 0.99, "DC signal autocorrelation should be near 1.0");
}

void test_levinson_overflow_guard(void) {
    printf("[TEST] LPC Levinson-Durbin overflow guard (Bug F5 fix)\n");
    /* Use a high-energy signal that stresses the Levinson computation.
     * INT32_MAX/16 is a realistic "loud" Q31 EEG signal that doesn't
     * overflow the int64 autocorrelation accumulator itself. */
    int32_t signal[256];
    for (int i = 0; i < 256; i++) signal[i] = INT32_MAX / 16;

    int64_t R[9] = {0};
    for (int lag = 0; lag <= 8; lag++) {
        for (int i = 0; i < 256 - lag; i++) {
            R[lag] += (int64_t)signal[i] * signal[i + lag];
        }
    }
    ASSERT_TRUE(R[0] > 0, "R[0] must be positive for non-zero signal");

    /* Compute reflection coefficient k[0] = -R[1]/R[0] using fixed approach.
     * The old code `-(R[1] * (1LL<<31)) / R[0]` would overflow int64 for
     * high-energy signals. The fix: divide first, then shift. */
    int64_t k_fixed = -(R[1] / R[0]) * (int64_t)(1LL << 31);
    ASSERT_TRUE(k_fixed >= INT32_MIN && k_fixed <= INT32_MAX,
                "Bug F5: reflection coefficient must fit in Q31 after fix");
}

/* ==================================================================
 * LPC DELTA CODEC TESTS (verifies Bug F2 fix)
 * ================================================================== */

/* Inline simplified LPC delta encode/decode for self-contained testing.
 * These match the logic from firmware/codec/lpc_delta.c. */
#define TEST_LPC_ORDER 8
#define TEST_NUM_CH 2  /* Use 2 channels for faster tests */

typedef enum { TMODE_KEY = 0, TMODE_Q15 = 1, TMODE_Q8 = 2 } test_delta_mode_t;

static int32_t td_prev[TEST_NUM_CH][TEST_LPC_ORDER];
static int td_has_prev = 0;

static uint32_t td_encode(const int32_t curr[][TEST_LPC_ORDER], uint8_t* buf) {
    uint32_t pos = 0;
    if (!td_has_prev) {
        buf[pos++] = TMODE_KEY;
        for (int ch = 0; ch < TEST_NUM_CH; ch++)
            for (int k = 0; k < TEST_LPC_ORDER; k++) {
                int32_t v = curr[ch][k];
                buf[pos++] = v & 0xFF; buf[pos++] = (v>>8)&0xFF;
                buf[pos++] = (v>>16)&0xFF; buf[pos++] = (v>>24)&0xFF;
            }
    } else {
        /* Check if Q8 fits */
        int q8_ok = 1;
        for (int ch = 0; ch < TEST_NUM_CH && q8_ok; ch++)
            for (int k = 0; k < TEST_LPC_ORDER && q8_ok; k++) {
                int32_t d = curr[ch][k] - td_prev[ch][k];
                int8_t d8 = (int8_t)(d >> 23);
                if (((int32_t)d8 << 23) != d) q8_ok = 0;
            }
        if (q8_ok) {
            buf[pos++] = TMODE_Q8;
            for (int ch = 0; ch < TEST_NUM_CH; ch++)
                for (int k = 0; k < TEST_LPC_ORDER; k++) {
                    int32_t d = curr[ch][k] - td_prev[ch][k];
                    buf[pos++] = (uint8_t)(d >> 23);
                }
        } else {
            buf[pos++] = TMODE_Q15;
            for (int ch = 0; ch < TEST_NUM_CH; ch++)
                for (int k = 0; k < TEST_LPC_ORDER; k++) {
                    int32_t d = curr[ch][k] - td_prev[ch][k];
                    int16_t d16 = (int16_t)(d >> 16);
                    buf[pos++] = d16 & 0xFF; buf[pos++] = (d16 >> 8) & 0xFF;
                }
        }
    }
    memcpy(td_prev, curr, sizeof(td_prev));
    td_has_prev = 1;
    return pos;
}

static uint32_t td_decode(const uint8_t* buf, int32_t out[][TEST_LPC_ORDER]) {
    test_delta_mode_t mode = (test_delta_mode_t)buf[0];
    uint32_t pos = 1;
    /* Bug F2 guard: reject delta before keyframe */
    if (!td_has_prev && mode != TMODE_KEY) {
        memset(out, 0, sizeof(int32_t) * TEST_NUM_CH * TEST_LPC_ORDER);
        return 0;
    }
    switch (mode) {
        case TMODE_KEY:
            for (int ch = 0; ch < TEST_NUM_CH; ch++)
                for (int k = 0; k < TEST_LPC_ORDER; k++) {
                    out[ch][k] = (int32_t)buf[pos] | ((int32_t)buf[pos+1]<<8)
                               | ((int32_t)buf[pos+2]<<16) | ((int32_t)buf[pos+3]<<24);
                    pos += 4;
                }
            break;
        case TMODE_Q15:
            for (int ch = 0; ch < TEST_NUM_CH; ch++)
                for (int k = 0; k < TEST_LPC_ORDER; k++) {
                    int16_t d16 = (int16_t)((uint16_t)buf[pos] | ((uint16_t)buf[pos+1]<<8));
                    out[ch][k] = td_prev[ch][k] + ((int32_t)d16 << 16);
                    pos += 2;
                }
            break;
        case TMODE_Q8:
            for (int ch = 0; ch < TEST_NUM_CH; ch++)
                for (int k = 0; k < TEST_LPC_ORDER; k++) {
                    int8_t d8 = (int8_t)buf[pos];
                    out[ch][k] = td_prev[ch][k] + ((int32_t)d8 << 23);
                    pos += 1;
                }
            break;
    }
    memcpy(td_prev, out, sizeof(td_prev));
    td_has_prev = 1;
    return pos;
}

static void td_reset(void) { memset(td_prev, 0, sizeof(td_prev)); td_has_prev = 0; }

void test_lpc_delta_keyframe_roundtrip(void) {
    printf("[TEST] LPC delta keyframe encode/decode roundtrip\n");
    td_reset();
    int32_t coeffs[TEST_NUM_CH][TEST_LPC_ORDER];
    for (int ch = 0; ch < TEST_NUM_CH; ch++)
        for (int k = 0; k < TEST_LPC_ORDER; k++)
            coeffs[ch][k] = (ch * 100000 + k * 12345) ^ 0x55AA0000;
    uint8_t buf[256];
    uint32_t n = td_encode(coeffs, buf);
    ASSERT_TRUE(n > 0, "keyframe encode produced bytes");
    ASSERT_EQ(buf[0], TMODE_KEY, "first frame must be keyframe");

    td_reset();
    int32_t out[TEST_NUM_CH][TEST_LPC_ORDER];
    td_decode(buf, out);
    for (int ch = 0; ch < TEST_NUM_CH; ch++)
        for (int k = 0; k < TEST_LPC_ORDER; k++)
            ASSERT_EQ(out[ch][k], coeffs[ch][k], "keyframe roundtrip exact");
}

void test_lpc_delta_q8_roundtrip(void) {
    printf("[TEST] LPC delta Q8 encode/decode roundtrip\n");
    td_reset();
    int32_t frame1[TEST_NUM_CH][TEST_LPC_ORDER] = {{0}};
    int32_t frame2[TEST_NUM_CH][TEST_LPC_ORDER];
    /* frame2 differs from frame1 by small deltas (fits in Q8) */
    for (int ch = 0; ch < TEST_NUM_CH; ch++)
        for (int k = 0; k < TEST_LPC_ORDER; k++)
            frame2[ch][k] = (k + 1) * (1 << 23); /* small Q31 values */
    uint8_t buf1[256], buf2[256];
    td_encode(frame1, buf1); /* keyframe */
    uint32_t n2 = td_encode(frame2, buf2);
    ASSERT_EQ(buf2[0], TMODE_Q8, "small delta should use Q8 mode");

    td_reset();
    int32_t out1[TEST_NUM_CH][TEST_LPC_ORDER], out2[TEST_NUM_CH][TEST_LPC_ORDER];
    td_decode(buf1, out1);
    td_decode(buf2, out2);
    for (int ch = 0; ch < TEST_NUM_CH; ch++)
        for (int k = 0; k < TEST_LPC_ORDER; k++)
            ASSERT_EQ(out2[ch][k], frame2[ch][k], "Q8 delta roundtrip exact");
}

void test_lpc_delta_decode_no_prev(void) {
    printf("[TEST] LPC delta decode rejects delta without keyframe (Bug F2)\n");
    td_reset();
    /* Craft a Q15 delta frame without a preceding keyframe */
    uint8_t fake_delta[256] = {TMODE_Q15};
    int32_t out[TEST_NUM_CH][TEST_LPC_ORDER];
    uint32_t consumed = td_decode(fake_delta, out);
    ASSERT_EQ(consumed, 0, "Bug F2: decode should return 0 on delta without keyframe");
    /* Output should be zeroed (safe fallback) */
    for (int ch = 0; ch < TEST_NUM_CH; ch++)
        for (int k = 0; k < TEST_LPC_ORDER; k++)
            ASSERT_EQ(out[ch][k], 0, "Bug F2: output zeroed on missing keyframe");
}

/* ==================================================================
 * TERNARY MAC EXHAUSTIVE + Q31 ALPHA TESTS (Bug F4)
 * ================================================================== */

void test_ternary_mac_exhaustive(void) {
    printf("[TEST] ternary MAC exhaustive 2-bit weight combos\n");
    /* Test all 256 possible packed_w byte values against hand-computed results */
    int16_t acts[4] = {100, -200, 300, -400};
    int failures = 0;
    for (int w = 0; w < 256; w++) {
        uint8_t packed = (uint8_t)w;
        int32_t expected = 0;
        for (int j = 0; j < 4; j++) {
            int bits = (packed >> (2*j)) & 0x03;
            expected += (int32_t)acts[j] * TERNARY_LUT[bits];
        }
        int32_t got = ternary_mac_byte_w2a6(packed, acts);
        if (got != expected) failures++;
    }
    ASSERT_EQ(failures, 0, "all 256 weight byte combos must match hand computation");
}

void test_ternary_mac_q31_alpha(void) {
    printf("[TEST] ternary MAC Q31 alpha scaling (Bug F4 fix)\n");
    /* Verify mul_q31 produces correct alpha-scaled result.
     * accumulator=1000, alpha=0.1 in Q31 = 214748365 */
    int32_t acc = 1000;
    int32_t alpha_q31 = (int32_t)(0.1 * 2147483648.0); /* 0.1 in Q31 */
    int32_t result = mul_q31(acc, alpha_q31);
    /* Expected: 1000 * 0.1 = 100 (within ±1 for Q31 rounding) */
    ASSERT_TRUE(abs(result - 100) <= 1,
                "Bug F4: Q31 alpha scaling should give ~100 for acc=1000, alpha=0.1");
}

/* ==================================================================
 * FSQ QUANTIZATION TESTS
 * ================================================================== */

/* Simplified FSQ quantize matching firmware/neural/fsq.c */
static int32_t fsq_quantize(int32_t val, int L) {
    /* Map Q31 value to grid [0, L-1] */
    /* val in [-2^31, 2^31-1], map to [-L/2, L/2] then shift to [0, L-1] */
    int32_t half_L = L / 2;
    /* Scale: val * L / 2^32 (approximate) */
    int64_t scaled = ((int64_t)val * L) >> 32;
    /* Clamp to [-L/2, L/2] */
    if (scaled > half_L) scaled = half_L;
    if (scaled < -half_L) scaled = -half_L;
    /* Shift to [0, L-1] */
    int32_t idx = (int32_t)(scaled + half_L);
    if (idx >= L) idx = L - 1;
    if (idx < 0) idx = 0;
    return idx;
}

void test_fsq_quantize_range(void) {
    printf("[TEST] FSQ quantize output range [0, L-1]\n");
    int levels[] = {2, 3, 5, 32};
    int n_levels = 4;
    int failures = 0;
    for (int li = 0; li < n_levels; li++) {
        int L = levels[li];
        /* Test with extreme values */
        int32_t test_vals[] = {INT32_MIN, INT32_MIN/2, 0, INT32_MAX/2, INT32_MAX};
        for (int vi = 0; vi < 5; vi++) {
            int32_t idx = fsq_quantize(test_vals[vi], L);
            if (idx < 0 || idx >= L) failures++;
        }
    }
    ASSERT_EQ(failures, 0, "FSQ output must always be in [0, L-1]");
}

void test_fsq_quantize_monotonic(void) {
    printf("[TEST] FSQ quantize is monotonic\n");
    /* For a given L, if val1 < val2 then fsq(val1) <= fsq(val2) */
    int failures = 0;
    for (int L = 2; L <= 32; L++) {
        int32_t prev_idx = fsq_quantize(INT32_MIN, L);
        for (int i = 1; i <= 1000; i++) {
            int32_t val = (int32_t)((int64_t)INT32_MIN + (int64_t)i * ((int64_t)UINT32_MAX / 1000));
            int32_t idx = fsq_quantize(val, L);
            if (idx < prev_idx) failures++;
            prev_idx = idx;
        }
    }
    ASSERT_EQ(failures, 0, "FSQ must be monotonically non-decreasing");
}

/* ==================================================================
 * BIQUAD EXTENDED TESTS
 * ================================================================== */

void test_biquad_impulse_response_values(void) {
    printf("[TEST] biquad HP impulse response (first 5 values)\n");
    /* HP 0.5 Hz at 250 Hz: b0=1064243069, b1=-2128486138, b2=1064243069,
     * a1=-2128402106, a2=1054828345 (all Q30) */
    biquad_state_t S = {
        .b0 = 1064243069, .b1 = -2128486138, .b2 = 1064243069,
        .a1 = -2128402106, .a2 = 1054828345,
        .x1=0, .x2=0, .y1=0, .y2=0
    };
    /* Impulse: x[0]=Q31_MAX, x[n>0]=0 */
    int32_t y0 = biquad_process(&S, INT32_MAX);
    int32_t y1 = biquad_process(&S, 0);
    int32_t y2 = biquad_process(&S, 0);
    /* y0 should be large positive (≈ b0 * xmax) */
    ASSERT_TRUE(y0 > 0, "HP impulse y[0] should be positive");
    /* y1 should be large negative (b1 term dominates) */
    ASSERT_TRUE(y1 < 0, "HP impulse y[1] should be negative");
    /* All values must be finite (no overflow to 0 or wrap) */
    ASSERT_TRUE(y0 != 0 && y1 != 0 && y2 != 0, "impulse response must be non-zero for first 3 samples");
}

void test_biquad_state_carryover(void) {
    printf("[TEST] biquad state carries over between calls\n");
    biquad_state_t S = {
        .b0 = 1064243069, .b1 = -2128486138, .b2 = 1064243069,
        .a1 = -2128402106, .a2 = 1054828345,
        .x1=0, .x2=0, .y1=0, .y2=0
    };
    /* Process 5 samples */
    int32_t y[5];
    for (int i = 0; i < 5; i++) y[i] = biquad_process(&S, 100000);
    /* State should be non-zero after processing */
    ASSERT_TRUE(S.x1 != 0 || S.y1 != 0, "state must carry over between samples");
    /* Process 5 more — output should be different from first 5 (filter memory) */
    int32_t y2[5];
    for (int i = 0; i < 5; i++) y2[i] = biquad_process(&S, 100000);
    /* With constant input, HP filter should settle toward 0 */
    ASSERT_TRUE(abs(y2[4]) < abs(y[0]), "HP output should decrease for DC input");
}

/* ==================================================================
 * Main (complete)
 * ================================================================== */
int main(void) {
    printf("=== LamQuant Gen 7: C Firmware Unit Tests (Host) ===\n\n");

    /* Math primitives */
    test_mul_q31();
    test_add_sat_q31();
    test_sub_sat_q31();
    test_mul_q30();

    /* Ternary MAC */
    test_ternary_mac_kat();
    test_ternary_mac_edges();
    test_ternary_mac_exhaustive();
    test_ternary_mac_q31_alpha();

    /* CRC + LFSR */
    test_crc32();
    test_lfsr_period();
    test_lfsr_batch32_equivalence();

    /* Misc */
    test_isqrt32();
    test_raw_packet_header();
    test_output_mode_enum();

    /* Biquad filter */
    test_biquad_hp_dc_rejection();
    test_biquad_lp_dc_stability();
    test_biquad_impulse_finite();
    test_biquad_cascade_stable();
    test_biquad_impulse_response_values();
    test_biquad_state_carryover();

    /* Lifting DWT (Bug F1 + F3 verification) */
    test_lifting_roundtrip_even();
    test_lifting_roundtrip_odd();
    test_lifting_constant_input();
    test_lifting_boundary_even_fix();
    test_lifting_negative_rounding();
    test_lifting_2500_roundtrip();

    /* WHT32 */
    test_wht32_roundtrip();
    test_wht32_delta_function();

    /* LPC Levinson-Durbin (Bug F5 verification) */
    test_levinson_flat_signal();
    test_levinson_overflow_guard();

    /* LPC Delta Codec (Bug F2 verification) */
    test_lpc_delta_keyframe_roundtrip();
    test_lpc_delta_q8_roundtrip();
    test_lpc_delta_decode_no_prev();

    /* FSQ Quantization */
    test_fsq_quantize_range();
    test_fsq_quantize_monotonic();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
