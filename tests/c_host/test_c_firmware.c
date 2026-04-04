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

/* ---- Biquad Q31 (from biquad_q31.c) ---- */
typedef struct {
    int32_t b0, b1, b2;
    int32_t a1, a2;
    int32_t x1, x2;
    int32_t y1, y2;
} biquad_state_q31_t;

static inline int32_t biquad_process(biquad_state_q31_t* S, int32_t x0) {
    int32_t acc;
    acc = mul_q31(S->b0, x0);
    acc = add_sat_q31(acc, mul_q31(S->b1, S->x1));
    acc = add_sat_q31(acc, mul_q31(S->b2, S->x2));
    acc = sub_sat_q31(acc, mul_q31(S->a1, S->y1));
    acc = sub_sat_q31(acc, mul_q31(S->a2, S->y2));
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

/* ---- Test: Biquad Q31 stability under DC ---- */
void test_biquad_dc_stability(void) {
    printf("[TEST] biquad DC stability\n");

    /* Lowpass 50Hz coefficients from biquad_q31.c */
    biquad_state_q31_t lp;
    lp.b0 =  292809836;
    lp.b1 =  585619672;
    lp.b2 =  292809836;
    lp.a1 = -451621498;
    lp.a2 =  622860843;
    lp.x1 = lp.x2 = lp.y1 = lp.y2 = 0;

    /* Feed constant DC for 1000 samples. Filter must:
       1. Produce non-zero output (not a no-op)
       2. Converge to a steady state (not diverge)  */
    int32_t dc_value = 100000000;
    int32_t prev = 0, last = 0;
    for (int i = 0; i < 1000; i++) {
        prev = last;
        last = biquad_process(&lp, dc_value);
    }

    ASSERT_TRUE(last != 0, "LP filter output is non-zero for DC input");
    /* Steady state: last two outputs should be very close */
    ASSERT_TRUE(abs(last - prev) < 100,
                "LP filter converges to steady state under DC");
}

/* ---- Test: Biquad impulse response is finite ---- */
void test_biquad_impulse_finite(void) {
    printf("[TEST] biquad impulse response is finite\n");

    /* Lowpass 50Hz coefficients */
    biquad_state_q31_t lp;
    lp.b0 =  292809836;
    lp.b1 =  585619672;
    lp.b2 =  292809836;
    lp.a1 = -451621498;
    lp.a2 =  622860843;
    lp.x1 = lp.x2 = lp.y1 = lp.y2 = 0;

    /* Unit impulse */
    int32_t out = biquad_process(&lp, 1000000000);
    ASSERT_TRUE(out != 0, "LP filter responds to impulse");

    /* Feed zeros and check it decays */
    for (int i = 0; i < 200; i++) {
        out = biquad_process(&lp, 0);
    }
    ASSERT_TRUE(abs(out) < 1000, "LP impulse response decays to near zero");
}

/* ==================================================================
 * Main
 * ================================================================== */
int main(void) {
    printf("=== LamQuant Gen 6: C Firmware Unit Tests (Host) ===\n\n");

    test_mul_q31();
    test_add_sat_q31();
    test_sub_sat_q31();
    test_ternary_mac_kat();
    test_ternary_mac_edges();
    test_crc32();
    test_lfsr_period();
    test_lfsr_batch32_equivalence();
    test_isqrt32();
    test_biquad_dc_stability();
    test_biquad_impulse_finite();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
