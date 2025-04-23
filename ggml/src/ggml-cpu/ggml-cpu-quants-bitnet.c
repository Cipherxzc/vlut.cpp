#define GGML_COMMON_IMPL_C


#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>   // for GGML_ASSERT
#include <stdlib.h>  // for qsort
#include <string.h>

#include "ggml-cpu-quants-bitnet.h"

#include "ggml-common.h"
#include "ggml-impl.h"


#define UNUSED GGML_UNUSED
#define eps 1e-5


#ifdef BITNET_DEBUG
#include <pthread.h>
#include <time.h>
#endif

#if defined(BITNET_AVX512) // AVX512
    #include <immintrin.h>
    #define ADD_TABLE_ENTRIES(rs, rt, size) \
    do { \
        /* Process 32 int16_t values at a time with AVX512 */ \
        for (int i = 0; i < (size); i += 32) { \
            /* Load 32 elements (512 bits) */ \
            __m512i rs_vec = _mm512_loadu_si512((__m512i*)((rs) + i)); \
            __m512i rt_vec = _mm512_loadu_si512((__m512i*)((rt) + i)); \
            \
            /* Add vectors */ \
            rs_vec = _mm512_add_epi16(rs_vec, rt_vec); \
            \
            /* Store result back */ \
            _mm512_storeu_si512((__m512i*)((rs) + i), rs_vec); \
        } \
    } while(0)
#elif defined(BITNET_SVE) // SVE
    #include <arm_sve.h>
    #define ADD_TABLE_ENTRIES(rs, rt, size) \
    do { \
        for (int i = 0; i < (size); i += svcntw()) { \
            /* Create a predicate for the current chunk */ \
            svbool_t pg = svwhilelt_b16(i, (size)); \
            \
            /* Load vectors */ \
            svint16_t acc = svld1_s16(pg, (rs) + i); \
            svint16_t tab = svld1_s16(pg, (rt) + i); \
            \
            /* Add vectors */ \
            acc = svadd_s16_z(pg, acc, tab); \
            \
            /* Store result */ \
            svst1_s16(pg, (rs) + i, acc); \
        } \
    } while(0)
#elif defined(BITNET_ACCELERATE) // Accelerate framework of Apple
    #include <Accelerate/Accelerate.h>
    #define ADD_TABLE_ENTRIES(rs, rt, size) \
    do { \
        vDSP_vadd((rs), 1, (rt), 1, (rs), 1, (size)); \
    } while(0)
#else // Fallback to auto vectorization by compiler
    #define ADD_TABLE_ENTRIES(rs, rt, size) \
    do { \
        for (int r = 0; r < (size); r++) { \
            (rs)[r] += (rt)[r]; \
        } \
    } while(0)
#endif

// #define ACCUMULATE_TABLE_TRANS(ss, ss2, nc, entry_tile_remain, entry_tile_count) \
//     do { \
//         for (int i = 0; i < nc; i++) { \
//             for (int j = 0; j < entry_tile_remain; j++) { \
//                 ss2[(entry_tile_count * TABLE_ENTRY_SIZE + j) * nc + i] += ss[i * TABLE_ENTRY_SIZE + j]; \
//             } \
//         } \
//     } while(0)

// Accumulate the transposed table entries
#define ACCUMULATE_TABLE_TRANS(ss, ss2, nc, entry_tile_remain, entry_tile_count) \
    do { \
        for (int j = 0; j < entry_tile_remain; j++) { \
            for (int i = 0; i < nc; i++) { \
                ss2[(entry_tile_count * TABLE_ENTRY_SIZE + j) * nc + i] += ss[i * TABLE_ENTRY_SIZE + j]; \
            } \
        } \
    } while(0)

// Access weights in a block of 4, manually unrolling the loop
#define ADD_TABLE_ENTRIES_BLOCK_4(nc, this_x, this_table, sum_i16) \
    do { \
        for (int c = 0; c < (nc) / 4; c++) { \
            uint8_t v0 = this_x[c * 4]; \
            uint8_t v1 = this_x[c * 4 + 1]; \
            uint8_t v2 = this_x[c * 4 + 2]; \
            uint8_t v3 = this_x[c * 4 + 3]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (c * 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (c * 4 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (c * 4 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (c * 4 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
        } \
        \
        for (int c = 0; c < (nc) % 4; c++) { \
            uint8_t v = this_x[(nc) / 4 * 4 + c]; \
            int16_t *restrict rs = sum_i16 + ((nc) / 4 * 4 + c) * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE); \
        } \
    } while(0)

// Block 8 optimized to handle remainders as 8+4+1
#define ADD_TABLE_ENTRIES_BLOCK_8(nc, this_x, this_table, sum_i16) \
    do { \
        for (int c = 0; c < (nc) / 8; c++) { \
            uint8_t v0 = this_x[c * 8]; \
            uint8_t v1 = this_x[c * 8 + 1]; \
            uint8_t v2 = this_x[c * 8 + 2]; \
            uint8_t v3 = this_x[c * 8 + 3]; \
            uint8_t v4 = this_x[c * 8 + 4]; \
            uint8_t v5 = this_x[c * 8 + 5]; \
            uint8_t v6 = this_x[c * 8 + 6]; \
            uint8_t v7 = this_x[c * 8 + 7]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (c * 8) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (c * 8 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (c * 8 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (c * 8 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (c * 8 + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (c * 8 + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (c * 8 + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (c * 8 + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Handle remainder in blocks of 4 */ \
        int offset = ((nc) / 8) * 8; \
        int remainder = (nc) % 8; \
        \
        /* Process blocks of 4 */ \
        for (int c = 0; c < remainder / 4; c++) { \
            uint8_t v0 = this_x[offset + c * 4]; \
            uint8_t v1 = this_x[offset + c * 4 + 1]; \
            uint8_t v2 = this_x[offset + c * 4 + 2]; \
            uint8_t v3 = this_x[offset + c * 4 + 3]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (offset + c * 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + c * 4 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + c * 4 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + c * 4 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Process remaining 1-3 elements */ \
        offset += (remainder / 4) * 4; \
        remainder = remainder % 4; \
        for (int c = 0; c < remainder; c++) { \
            uint8_t v = this_x[offset + c]; \
            int16_t *restrict rs = sum_i16 + (offset + c) * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE); \
        } \
    } while(0)

// Block 16 optimized to handle remainders as 16+8+4+1
#define ADD_TABLE_ENTRIES_BLOCK_16(nc, this_x, this_table, sum_i16) \
    do { \
        for (int c = 0; c < (nc) / 16; c++) { \
            uint8_t v0 = this_x[c * 16]; \
            uint8_t v1 = this_x[c * 16 + 1]; \
            uint8_t v2 = this_x[c * 16 + 2]; \
            uint8_t v3 = this_x[c * 16 + 3]; \
            uint8_t v4 = this_x[c * 16 + 4]; \
            uint8_t v5 = this_x[c * 16 + 5]; \
            uint8_t v6 = this_x[c * 16 + 6]; \
            uint8_t v7 = this_x[c * 16 + 7]; \
            uint8_t v8 = this_x[c * 16 + 8]; \
            uint8_t v9 = this_x[c * 16 + 9]; \
            uint8_t v10 = this_x[c * 16 + 10]; \
            uint8_t v11 = this_x[c * 16 + 11]; \
            uint8_t v12 = this_x[c * 16 + 12]; \
            uint8_t v13 = this_x[c * 16 + 13]; \
            uint8_t v14 = this_x[c * 16 + 14]; \
            uint8_t v15 = this_x[c * 16 + 15]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt8 = this_table + v8 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt9 = this_table + v9 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt10 = this_table + v10 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt11 = this_table + v11 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt12 = this_table + v12 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt13 = this_table + v13 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt14 = this_table + v14 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt15 = this_table + v15 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (c * 16) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (c * 16 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (c * 16 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (c * 16 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (c * 16 + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (c * 16 + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (c * 16 + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (c * 16 + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs8 = sum_i16 + (c * 16 + 8) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs8, rt8, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs9 = sum_i16 + (c * 16 + 9) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs9, rt9, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs10 = sum_i16 + (c * 16 + 10) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs10, rt10, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs11 = sum_i16 + (c * 16 + 11) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs11, rt11, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs12 = sum_i16 + (c * 16 + 12) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs12, rt12, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs13 = sum_i16 + (c * 16 + 13) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs13, rt13, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs14 = sum_i16 + (c * 16 + 14) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs14, rt14, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs15 = sum_i16 + (c * 16 + 15) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs15, rt15, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Handle remainder in graduated steps: first process blocks of 8 */ \
        int offset = ((nc) / 16) * 16; \
        int remainder = (nc) % 16; \
        \
        /* Process blocks of 8 */ \
        for (int c = 0; c < remainder / 8; c++) { \
            uint8_t v0 = this_x[offset + c * 8]; \
            uint8_t v1 = this_x[offset + c * 8 + 1]; \
            uint8_t v2 = this_x[offset + c * 8 + 2]; \
            uint8_t v3 = this_x[offset + c * 8 + 3]; \
            uint8_t v4 = this_x[offset + c * 8 + 4]; \
            uint8_t v5 = this_x[offset + c * 8 + 5]; \
            uint8_t v6 = this_x[offset + c * 8 + 6]; \
            uint8_t v7 = this_x[offset + c * 8 + 7]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (offset + c * 8) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + c * 8 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + c * 8 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + c * 8 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (offset + c * 8 + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (offset + c * 8 + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (offset + c * 8 + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (offset + c * 8 + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Then process blocks of 4 */ \
        offset += (remainder / 8) * 8; \
        remainder = remainder % 8; \
        \
        for (int c = 0; c < remainder / 4; c++) { \
            uint8_t v0 = this_x[offset + c * 4]; \
            uint8_t v1 = this_x[offset + c * 4 + 1]; \
            uint8_t v2 = this_x[offset + c * 4 + 2]; \
            uint8_t v3 = this_x[offset + c * 4 + 3]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (offset + c * 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + c * 4 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + c * 4 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + c * 4 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Finally process remaining 1-3 elements */ \
        offset += (remainder / 4) * 4; \
        remainder = remainder % 4; \
        for (int c = 0; c < remainder; c++) { \
            uint8_t v = this_x[offset + c]; \
            int16_t *restrict rs = sum_i16 + (offset + c) * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE); \
        } \
    } while(0)

// Block 32 with full manual unrolling and graduated remainders (32+16+8+4+1)
#define ADD_TABLE_ENTRIES_BLOCK_32(nc, this_x, this_table, sum_i16) \
    do { \
        for (int c = 0; c < (nc) / 32; c++) { \
            uint8_t v0 = this_x[c * 32]; \
            uint8_t v1 = this_x[c * 32 + 1]; \
            uint8_t v2 = this_x[c * 32 + 2]; \
            uint8_t v3 = this_x[c * 32 + 3]; \
            uint8_t v4 = this_x[c * 32 + 4]; \
            uint8_t v5 = this_x[c * 32 + 5]; \
            uint8_t v6 = this_x[c * 32 + 6]; \
            uint8_t v7 = this_x[c * 32 + 7]; \
            uint8_t v8 = this_x[c * 32 + 8]; \
            uint8_t v9 = this_x[c * 32 + 9]; \
            uint8_t v10 = this_x[c * 32 + 10]; \
            uint8_t v11 = this_x[c * 32 + 11]; \
            uint8_t v12 = this_x[c * 32 + 12]; \
            uint8_t v13 = this_x[c * 32 + 13]; \
            uint8_t v14 = this_x[c * 32 + 14]; \
            uint8_t v15 = this_x[c * 32 + 15]; \
            uint8_t v16 = this_x[c * 32 + 16]; \
            uint8_t v17 = this_x[c * 32 + 17]; \
            uint8_t v18 = this_x[c * 32 + 18]; \
            uint8_t v19 = this_x[c * 32 + 19]; \
            uint8_t v20 = this_x[c * 32 + 20]; \
            uint8_t v21 = this_x[c * 32 + 21]; \
            uint8_t v22 = this_x[c * 32 + 22]; \
            uint8_t v23 = this_x[c * 32 + 23]; \
            uint8_t v24 = this_x[c * 32 + 24]; \
            uint8_t v25 = this_x[c * 32 + 25]; \
            uint8_t v26 = this_x[c * 32 + 26]; \
            uint8_t v27 = this_x[c * 32 + 27]; \
            uint8_t v28 = this_x[c * 32 + 28]; \
            uint8_t v29 = this_x[c * 32 + 29]; \
            uint8_t v30 = this_x[c * 32 + 30]; \
            uint8_t v31 = this_x[c * 32 + 31]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt8 = this_table + v8 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt9 = this_table + v9 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt10 = this_table + v10 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt11 = this_table + v11 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt12 = this_table + v12 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt13 = this_table + v13 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt14 = this_table + v14 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt15 = this_table + v15 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt16 = this_table + v16 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt17 = this_table + v17 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt18 = this_table + v18 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt19 = this_table + v19 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt20 = this_table + v20 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt21 = this_table + v21 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt22 = this_table + v22 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt23 = this_table + v23 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt24 = this_table + v24 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt25 = this_table + v25 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt26 = this_table + v26 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt27 = this_table + v27 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt28 = this_table + v28 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt29 = this_table + v29 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt30 = this_table + v30 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt31 = this_table + v31 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + (c * 32) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (c * 32 + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (c * 32 + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (c * 32 + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (c * 32 + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (c * 32 + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (c * 32 + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (c * 32 + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs8 = sum_i16 + (c * 32 + 8) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs8, rt8, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs9 = sum_i16 + (c * 32 + 9) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs9, rt9, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs10 = sum_i16 + (c * 32 + 10) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs10, rt10, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs11 = sum_i16 + (c * 32 + 11) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs11, rt11, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs12 = sum_i16 + (c * 32 + 12) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs12, rt12, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs13 = sum_i16 + (c * 32 + 13) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs13, rt13, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs14 = sum_i16 + (c * 32 + 14) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs14, rt14, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs15 = sum_i16 + (c * 32 + 15) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs15, rt15, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs16 = sum_i16 + (c * 32 + 16) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs16, rt16, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs17 = sum_i16 + (c * 32 + 17) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs17, rt17, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs18 = sum_i16 + (c * 32 + 18) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs18, rt18, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs19 = sum_i16 + (c * 32 + 19) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs19, rt19, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs20 = sum_i16 + (c * 32 + 20) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs20, rt20, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs21 = sum_i16 + (c * 32 + 21) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs21, rt21, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs22 = sum_i16 + (c * 32 + 22) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs22, rt22, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs23 = sum_i16 + (c * 32 + 23) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs23, rt23, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs24 = sum_i16 + (c * 32 + 24) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs24, rt24, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs25 = sum_i16 + (c * 32 + 25) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs25, rt25, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs26 = sum_i16 + (c * 32 + 26) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs26, rt26, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs27 = sum_i16 + (c * 32 + 27) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs27, rt27, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs28 = sum_i16 + (c * 32 + 28) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs28, rt28, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs29 = sum_i16 + (c * 32 + 29) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs29, rt29, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs30 = sum_i16 + (c * 32 + 30) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs30, rt30, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs31 = sum_i16 + (c * 32 + 31) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs31, rt31, TABLE_ENTRY_SIZE); \
        } \
        \
        /* Handle remainder in graduated steps */ \
        int offset = ((nc) / 32) * 32; \
        int remainder = (nc) % 32; \
        \
        /* Process blocks of 16 */ \
        if (remainder >= 16) { \
            uint8_t v0 = this_x[offset]; \
            uint8_t v1 = this_x[offset + 1]; \
            uint8_t v2 = this_x[offset + 2]; \
            uint8_t v3 = this_x[offset + 3]; \
            uint8_t v4 = this_x[offset + 4]; \
            uint8_t v5 = this_x[offset + 5]; \
            uint8_t v6 = this_x[offset + 6]; \
            uint8_t v7 = this_x[offset + 7]; \
            uint8_t v8 = this_x[offset + 8]; \
            uint8_t v9 = this_x[offset + 9]; \
            uint8_t v10 = this_x[offset + 10]; \
            uint8_t v11 = this_x[offset + 11]; \
            uint8_t v12 = this_x[offset + 12]; \
            uint8_t v13 = this_x[offset + 13]; \
            uint8_t v14 = this_x[offset + 14]; \
            uint8_t v15 = this_x[offset + 15]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt8 = this_table + v8 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt9 = this_table + v9 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt10 = this_table + v10 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt11 = this_table + v11 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt12 = this_table + v12 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt13 = this_table + v13 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt14 = this_table + v14 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt15 = this_table + v15 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + offset * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (offset + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (offset + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (offset + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (offset + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs8 = sum_i16 + (offset + 8) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs8, rt8, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs9 = sum_i16 + (offset + 9) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs9, rt9, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs10 = sum_i16 + (offset + 10) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs10, rt10, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs11 = sum_i16 + (offset + 11) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs11, rt11, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs12 = sum_i16 + (offset + 12) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs12, rt12, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs13 = sum_i16 + (offset + 13) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs13, rt13, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs14 = sum_i16 + (offset + 14) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs14, rt14, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs15 = sum_i16 + (offset + 15) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs15, rt15, TABLE_ENTRY_SIZE); \
            \
            offset += 16; \
            remainder -= 16; \
        } \
        \
        /* Process blocks of 8 */ \
        if (remainder >= 8) { \
            uint8_t v0 = this_x[offset]; \
            uint8_t v1 = this_x[offset + 1]; \
            uint8_t v2 = this_x[offset + 2]; \
            uint8_t v3 = this_x[offset + 3]; \
            uint8_t v4 = this_x[offset + 4]; \
            uint8_t v5 = this_x[offset + 5]; \
            uint8_t v6 = this_x[offset + 6]; \
            uint8_t v7 = this_x[offset + 7]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt4 = this_table + v4 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt5 = this_table + v5 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt6 = this_table + v6 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt7 = this_table + v7 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + offset * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs4 = sum_i16 + (offset + 4) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs4, rt4, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs5 = sum_i16 + (offset + 5) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs5, rt5, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs6 = sum_i16 + (offset + 6) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs6, rt6, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs7 = sum_i16 + (offset + 7) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs7, rt7, TABLE_ENTRY_SIZE); \
            \
            offset += 8; \
            remainder -= 8; \
        } \
        \
        /* Process blocks of 4 */ \
        if (remainder >= 4) { \
            uint8_t v0 = this_x[offset]; \
            uint8_t v1 = this_x[offset + 1]; \
            uint8_t v2 = this_x[offset + 2]; \
            uint8_t v3 = this_x[offset + 3]; \
            \
            const int16_t *restrict rt0 = this_table + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = this_table + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = this_table + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = this_table + v3 * TABLE_ENTRY_SIZE; \
            \
            int16_t *restrict rs0 = sum_i16 + offset * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs0, rt0, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs1 = sum_i16 + (offset + 1) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs1, rt1, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs2 = sum_i16 + (offset + 2) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs2, rt2, TABLE_ENTRY_SIZE); \
            \
            int16_t *restrict rs3 = sum_i16 + (offset + 3) * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs3, rt3, TABLE_ENTRY_SIZE); \
            \
            offset += 4; \
            remainder -= 4; \
        } \
        \
        /* Process remaining 1-3 elements */ \
        for (int i = 0; i < remainder; i++) { \
            uint8_t v = this_x[offset + i]; \
            int16_t *restrict rs = sum_i16 + (offset + i) * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE); \
        } \
    } while(0)

#if WEIGHT_UNROLL_BLOCK == 4
#define ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16) \
    ADD_TABLE_ENTRIES_BLOCK_4(nc, this_x, this_table, sum_i16)
#elif WEIGHT_UNROLL_BLOCK == 8
#define ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16) \
    ADD_TABLE_ENTRIES_BLOCK_8(nc, this_x, this_table, sum_i16)
#elif WEIGHT_UNROLL_BLOCK == 16
#define ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16) \
    ADD_TABLE_ENTRIES_BLOCK_16(nc, this_x, this_table, sum_i16)
#elif WEIGHT_UNROLL_BLOCK == 32
#define ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16) \
    ADD_TABLE_ENTRIES_BLOCK_32(nc, this_x, this_table, sum_i16)
#else
#define ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16) \
    do { \
        for (int c = 0; c < (nc); c++) { \
            uint8_t v = this_x[c]; \
            int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE; \
            ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE); \
        } \
    } while(0)
#endif

#ifdef BITNET_AVX2
#include <immintrin.h>

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i bitnet_mul(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);

    const __m256i dot = _mm256_maddubs_epi16(ax, sy);

    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);

    return summed_pairs;
}
#endif


void quantize_row_i8_b(const float *x, void *y, int64_t n) {
    int8_t *dst = (int8_t *)y;

    double max = eps;
    for (int i = 0; i < n; i++) {
        max = MAX(max, (double)x[i]);
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, eps);

    for (int i = 0; i < n; i++) {
        float v = round((double)x[i] * is);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        dst[i] = (int8_t)v;
    }

    float *scale = (float *)(dst + n);
    *scale = (float)s;
}

void quantize_row_i8_b_tile(const float *x, void *y, int64_t n, float *scale) {
    int8_t *dst = (int8_t *)y;

    double max = eps;
    for (int i = 0; i < n; i++) {
        max = MAX(max, (double)x[i]);
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, eps);

    for (int i = 0, j = 0; i < n; i++, j += TABLE_ENTRY_SIZE) {
        float v = round((double)x[i] * is);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        dst[j] = (int8_t)v;
    }

    *scale = (float)s;
}


inline static void gemm_make_table_i2s(int16_t *restrict table, const int8_t *restrict y);
inline static void gemm_make_table_i1_58s(int16_t *restrict table, const int8_t *restrict y);

#ifdef BITNET_DEBUG
extern long long make_table_time, convert_time, scale_time, LUT_time;
extern pthread_mutex_t time_mutex;

static struct timespec get_thread_cpu_time() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts;
}

static long long get_time_diff(const struct timespec start, const struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
}
#endif


extern int16_t *tables;
extern int8_t *tmp_src;
extern int16_t *sum1;
extern int *sum2;

void ggml_gemm_i2s_i8b_make_table(int ith, int nth, const int8_t *restrict y, int ntables, int nr, int n,
                                  int16_t *restrict table) {
    UNUSED(ith);
    UNUSED(nth);

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * 4 * TABLE_ENTRY_SIZE);

    const int table_stride = n / 4 * 81;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int i = 0; i < entry_tile_count; i++) {
        const int8_t *restrict local_y = y + i * (n + 4) * TABLE_ENTRY_SIZE;
        int16_t *restrict local_table = table + i * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 4;
            for (int k = 0; k < TABLE_ENTRY_SIZE; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
            }
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src);
        }
    }

    // remains
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = y + entry_tile_count * (n + 4) * TABLE_ENTRY_SIZE;
        int16_t *restrict local_table = table + entry_tile_count * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 4;
            for (int k = 0; k < entry_tile_remain; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
            }
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src);
        }
    }

    free(src);
}

void ggml_gemm_i1s_i8b_make_table(int ith, int nth, const int8_t *restrict y, int ntables, int nr, int n,
                                  int16_t *restrict table) {
    UNUSED(ith);
    UNUSED(nth);

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * 5 * TABLE_ENTRY_SIZE);

    const int table_stride = n / 5 * 243;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int i = 0; i < entry_tile_count; i++) {
        const int8_t *restrict local_y = y + i * (n + 4) * TABLE_ENTRY_SIZE; // scale is 4 bytes
        int16_t *restrict local_table = table + i * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 5;
            for (int k = 0; k < TABLE_ENTRY_SIZE; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                src[k + TABLE_ENTRY_SIZE * 4] = y1[k * (n + 4) + 4];
            }
            gemm_make_table_i1_58s(local_table + j *243 * TABLE_ENTRY_SIZE, src);
        }
    }

    // remains
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = y + entry_tile_count * (n + 4) * TABLE_ENTRY_SIZE;
        int16_t *restrict local_table = table + entry_tile_count * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 5;
            for (int k = 0; k < entry_tile_remain; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                src[k + TABLE_ENTRY_SIZE * 4] = y1[k * (n + 4) + 4];
            }
            gemm_make_table_i1_58s(local_table + j *243 * TABLE_ENTRY_SIZE, src);
        }
    }

    free(src);
}

void ggml_gemm_i1m_i8b_make_table(int ith, int nth, const int8_t *restrict y, int ntables, int nr, int n,
                                  int16_t *restrict table) {

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * 5 * TABLE_ENTRY_SIZE);

    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;
    const int blck_num = n / 20 * 4;
    const int blck_remain = n % 20 / 4;
    const int table_stride = blck_num * 243 + blck_remain * 81;

    // tiles
    for (int i = 0; i < entry_tile_count; i++) {
        const int8_t *restrict local_y = y + i * (n + 4) * TABLE_ENTRY_SIZE;  // scale is 4 bytes
        int16_t *restrict local_table = table + i * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 5;
            for (int k = 0; k < TABLE_ENTRY_SIZE; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                src[k + TABLE_ENTRY_SIZE * 4] = y1[k * (n + 4) + 4];
            }
            gemm_make_table_i1_58s(local_table + j * 243 * TABLE_ENTRY_SIZE, src);
        }

        if (ith == nth){
            for (int j = 0; j < blck_remain; j++) {
                const int8_t *restrict y1 = local_y + ntables * 5 + j * 4;
                for (int k = 0; k < TABLE_ENTRY_SIZE; k++) {
                    src[k] = y1[k * (n + 4)];
                    src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                    src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                    src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                }
                gemm_make_table_i2s(local_table + ntables * 243 + j * 81 * TABLE_ENTRY_SIZE, src);
            }
        }
    }

    // remains
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = y + entry_tile_count * (n + 4) * TABLE_ENTRY_SIZE;
        int16_t *restrict local_table = table + entry_tile_count * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < ntables; j++) {
            const int8_t *restrict y1 = local_y + j * 5;
            for (int k = 0; k < entry_tile_remain; k++) {
                src[k] = y1[k * (n + 4)];
                src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                src[k + TABLE_ENTRY_SIZE * 4] = y1[k * (n + 4) + 4];
            }
            gemm_make_table_i1_58s(local_table + j * 243 * TABLE_ENTRY_SIZE, src);
        }

        if (ith == nth) {
            for (int j = 0; j < blck_remain; j++) {
                const int8_t *restrict y1 = local_y + ntables * 5 + j * 4;
                for (int k = 0; k < entry_tile_remain; k++) {
                    src[k] = y1[k * (n + 4)];
                    src[k + TABLE_ENTRY_SIZE] = y1[k * (n + 4) + 1];
                    src[k + TABLE_ENTRY_SIZE * 2] = y1[k * (n + 4) + 2];
                    src[k + TABLE_ENTRY_SIZE * 3] = y1[k * (n + 4) + 3];
                }
                gemm_make_table_i2s(local_table + ntables * 243 + j * 81 * TABLE_ENTRY_SIZE, src);
            }
        }
    }

    free(src);
}

void ggml_gemm_i2s_i8b_make_table_quant(int ith, int nth, const float *restrict y, float *restrict scale, int nr, int n,
                                        int16_t *restrict table) {
    UNUSED(ith);
    UNUSED(nth);

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * n * TABLE_ENTRY_SIZE);
    int8_t *restrict src2 = (int8_t *)malloc(sizeof(int8_t) * 4 * TABLE_ENTRY_SIZE);

    const int table_count = n / 4;
    const int table_stride = n / 4 * 81;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int i = 0; i < entry_tile_count; i++) {
        const float *restrict local_y = y + i * n * TABLE_ENTRY_SIZE;

        for (int j = 0; j < TABLE_ENTRY_SIZE; j++){
            const float *restrict y_row = local_y + j * n;

            double max = eps;
            for (int k = 0; k < n; k++) {
                max = MAX(max, (double)y_row[k]);
            }
            const double s = max / 127;
            const double is = 1e0 / MAX(s, eps);
            scale[i * TABLE_ENTRY_SIZE + j] = s;

            for (int k = 0; k < n; k++) {
                float v = round((double)y_row[k] * is);
                if (v > 127) {
                    v = 127;
                } else if (v < -128) {
                    v = -128;
                }
                src[j * n + k] = (int8_t)v;
            }
        }

        int16_t *restrict local_table = table + i * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < table_count; j++) {
            const int8_t *restrict src_local = src + j * 4;
            for (int k = 0; k < TABLE_ENTRY_SIZE; k++) {
                src2[k] = src_local[k * n];
                src2[k + TABLE_ENTRY_SIZE] = src_local[k * n + 1];
                src2[k + TABLE_ENTRY_SIZE * 2] = src_local[k * n + 2];
                src2[k + TABLE_ENTRY_SIZE * 3] = src_local[k * n + 3];
            }
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src2);
        }
    }

    // remains
    if (entry_tile_remain > 0) {
        memset(src2, 0, sizeof(int8_t) * 4 * TABLE_ENTRY_SIZE);

        const float *restrict local_y = y + entry_tile_count * n * TABLE_ENTRY_SIZE;

        for (int j = 0; j < entry_tile_remain; j++) {
            const float *restrict y_row = local_y + j * n;

            double max = eps;
            for (int k = 0; k < n; k++) {
                max = MAX(max, (double)y_row[k]);
            }
            const double s = max / 127;
            const double is = 1e0 / MAX(s, eps);
            scale[entry_tile_count * TABLE_ENTRY_SIZE + j] = s;

            for (int k = 0; k < n; k++) {
                float v = round((double)y_row[k] * is);
                if (v > 127) {
                    v = 127;
                } else if (v < -128) {
                    v = -128;
                }
                src[j * n + k] = (int8_t)v;
            }
        }

        int16_t *restrict local_table = table + entry_tile_count * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < table_count; j++) {
            const int8_t *restrict src_local = src + j * 4;
            for (int k = 0; k < entry_tile_remain; k++) {
                src2[k] = src_local[k * n];
                src2[k + TABLE_ENTRY_SIZE] = src_local[k * n + 1];
                src2[k + TABLE_ENTRY_SIZE * 2] = src_local[k * n + 2];
                src2[k + TABLE_ENTRY_SIZE * 3] = src_local[k * n + 3];
            }
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src2);
        }
    }

    free(src);
    free(src2);
}

void ggml_gemm_i2s_i8b_make_table_quant2(int ith, int nth, const float *restrict y, float *restrict scale, int nr, int n,
                                         int16_t *restrict table) {
    UNUSED(ith);
    UNUSED(nth);

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * n * TABLE_ENTRY_SIZE);

    const int table_count = n / 4;
    const int table_stride = n / 4 * 81;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int i = 0; i < entry_tile_count; i++) {
        const float *restrict local_y = y + i * n * TABLE_ENTRY_SIZE;

        for (int j = 0; j < TABLE_ENTRY_SIZE; j++) {
            const float *restrict y_row = local_y + j * n;

            double max = eps;
            for (int k = 0; k < n; k++) {
                max = MAX(max, (double)y_row[k]);
            }
            const double s = max / 127;
            const double is = 1e0 / MAX(s, eps);
            scale[i * TABLE_ENTRY_SIZE + j] = s;

            for (int k = 0; k < n; k++) {
                float v = round((double)y_row[k] * is);
                if (v > 127) {
                    v = 127;
                } else if (v < -128) {
                    v = -128;
                }
                src[k * TABLE_ENTRY_SIZE + j] = (int8_t)v;
            }
        }

        int16_t *restrict local_table = table + i * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < table_count; j++) {
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src + j * 4 * TABLE_ENTRY_SIZE);
        }
    }

    // remains
    if (entry_tile_remain > 0) {
        memset(src, 0, sizeof(int8_t) * n * TABLE_ENTRY_SIZE);

        const float *restrict local_y = y + entry_tile_count * n * TABLE_ENTRY_SIZE;

        for (int j = 0; j < entry_tile_remain; j++) {
            const float *restrict y_row = local_y + j * n;

            double max = eps;
            for (int k = 0; k < n; k++) {
                max = MAX(max, (double)y_row[k]);
            }
            const double s = max / 127;
            const double is = 1e0 / MAX(s, eps);
            scale[entry_tile_count * TABLE_ENTRY_SIZE + j] = s;

            for (int k = 0; k < n; k++) {
                float v = round((double)y_row[k] * is);
                if (v > 127) {
                    v = 127;
                } else if (v < -128) {
                    v = -128;
                }
                src[k * TABLE_ENTRY_SIZE + j] = (int8_t)v;
            }
        }

        int16_t *restrict local_table = table + entry_tile_count * table_stride * TABLE_ENTRY_SIZE;

        for (int j = 0; j < table_count; j++) {
            gemm_make_table_i2s(local_table + j * 81 * TABLE_ENTRY_SIZE, src + j * 4 * TABLE_ENTRY_SIZE);
        }
    }

    free(src);
}

void ggml_gemm_i2s_i8b_LUT2(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                            const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict this_table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(this_table + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 512;

    const int group_count = n / group_size; // not including remains
    const int group_size_remain = n % group_size;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE; // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        const int8_t *restrict local_y = (const int8_t *)vy + t * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 4; i++) {
                gemm_make_table_i2s(this_table, local_y + (g * group_size / 4 + i) * 4 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 4 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            for (int i = 0; i < group_size_remain / 4; i++) {
                gemm_make_table_i2s(this_table, local_y + (group_count * group_size / 4 + i) * 4 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // tile remain
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = (const int8_t *)vy + entry_tile_count * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 4; i++) {
                gemm_make_table_i2s(this_table, local_y + (g * group_size / 4 + i) * 4 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 4 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            for (int i = 0; i < group_size_remain / 4; i++) {
                gemm_make_table_i2s(this_table, local_y + (group_count * group_size / 4 + i) * 4 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }
    
    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)((const int8_t *)vy + y_size);
    for (int r = 0; r < nr; r++) {
        const float scale = sc[r];
        float* restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(this_table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1s_i8b_LUT2(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                            const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 5 == 0);

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict this_table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 243);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(this_table + 121 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 640;

    const int group_count = n / group_size; // not including remains
    const int group_size_remain = n % group_size;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE; // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        const int8_t *restrict local_y = (const int8_t *)vy + t * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (g * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            for (int i = 0; i < group_size_remain / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (group_count * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // tile remain
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = (const int8_t *)vy + entry_tile_count * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (g * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            for (int i = 0; i < group_size_remain / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (group_count * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }
    
    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)((const int8_t *)vy + y_size);
    for (int r = 0; r < nr; r++) {
        const float scale = sc[r];
        float* restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(this_table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1m_i8b_LUT2(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                            const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);
    
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict this_table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 243);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(this_table + 121 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 640;

    const int group_size_remain = (n - 1) % group_size + 1;
    const int group_count = (n - group_size_remain) / group_size;  // not including remains
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;            // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;
    const int blck_num = group_size_remain / 20 * 4;
    const int blck_remain = group_size_remain % 20 / 4;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        const int8_t *restrict local_y = (const int8_t *)vy + t * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (g * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }

        // group remain
        for (int i = 0; i < blck_num; i++) {
            gemm_make_table_i1_58s(this_table, local_y + (group_count * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        for (int i = 0; i < blck_remain; i++) {
            gemm_make_table_i2s(this_table, local_y + ((group_count * group_size / 5 + blck_num) * 5 + i * 4) * TABLE_ENTRY_SIZE);
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + blck_num + i) * bs;
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
        memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    }

    // tile remain
    if (entry_tile_remain > 0) {
        const int8_t *restrict local_y = (const int8_t *)vy + entry_tile_count * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            for (int i = 0; i < group_size / 5; i++) {
                gemm_make_table_i1_58s(this_table, local_y + (g * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }

        // group remain
        for (int i = 0; i < blck_num; i++) {
            gemm_make_table_i1_58s(this_table, local_y + (group_count * group_size / 5 + i) * 5 * TABLE_ENTRY_SIZE);
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        for (int i = 0; i < blck_remain; i++) {
            gemm_make_table_i2s(this_table, local_y + ((group_count * group_size / 5 + blck_num) * 5 + i * 4) * TABLE_ENTRY_SIZE);
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + blck_num + i) * bs;
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
        memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    }

    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)((const int8_t *)vy + y_size);
    for (int r = 0; r < nr; r++) {
        const float scale = sc[r];
        float *restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(this_table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i2s_i8b_LUT(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                           const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    // [M, K] * [K, N] = [M, N]
    // nr -> N
    // bs -> M, bs = nc * threads
    // n -> K
    assert(n % 4 == 0);
    assert(bs % nc == 0);

    // printf("ggml_gemm_i2s_i8b_LUT: %d %d %d %d\n", n, nr, nc, bs); // 3200 128 800 3200 (2160 8640)

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);

    static const int group_size = 512;

    const int table_stride = n / 4 * 81;
    const int group_count = n / group_size; // not including remains
    const int group_size_remain = n % group_size;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE; // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 4;
            const int16_t *local_table = tables + (t * table_stride + group_offset * 81) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 4; i++) {
                const int16_t *restrict this_table = local_table + i * 81 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 4 + i) * bs;
                for (int v = 0; v < 81; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            const int group_offset = group_count * group_size / 4;
            const int16_t *local_table = tables + (t * table_stride + group_offset * 81) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size_remain / 4; i++) {
                const int16_t *restrict this_table = local_table + i * 81 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int v = 0; v < 81; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // tile remain
    if (entry_tile_remain > 0) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 4;
            const int16_t *local_table = tables + (entry_tile_count * table_stride + group_offset * 81) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 4; i++) {
                const int16_t *restrict this_table = local_table + i * 81 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 4 + i) * bs;
                for (int v = 0; v < 81; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            const int group_offset = group_count * group_size / 4;
            const int16_t *local_table = tables + (entry_tile_count * table_stride + group_offset * 81) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size_remain / 4; i++) {
                const int16_t *restrict this_table = local_table + i * 81 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int v = 0; v < 81; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }
    
    // copy back to s[N, M]
    // TODO:
    // multiple threads might access the same row (r) of s, although accessing different cols (nc * ith + c)
    // will this cause any performance issue? how to avoid this?
    for (int r = 0; r < nr; r++) {
        const float scale = *((const float *)((const int8_t *)vy + r * (n + 4) + n));
        float* restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1s_i8b_LUT(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                           const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    // [M, K] * [K, N] = [M, N]
    // nr -> N
    // bs -> M, bs = nc * threads
    // n -> K
    assert(n % 5 == 0);
    assert(bs % nc == 0);

    // printf("ggml_gemm_i2s_i8b_LUT: %d %d %d %d\n", n, nr, nc, bs); // 3200 128 800 3200 (2160 8640)

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);

    static const int group_size = 640;

    const int table_stride = n / 5 * 243;
    const int group_count = n / group_size;  // not including remains
    const int group_size_remain = n % group_size;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 5;
            const int16_t *local_table = tables + (t * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            const int group_offset = group_count * group_size / 5;
            const int16_t *local_table = tables + (t * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size_remain / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // tile remain
    if (entry_tile_remain > 0) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 5;
            const int16_t *local_table =
                tables + (entry_tile_count * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            const int group_offset = group_count * group_size / 5;
            const int16_t *local_table =
                tables + (entry_tile_count * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size_remain / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3); // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // copy back to s[N, M]
    // TODO:
    // multiple threads might access the same row (r) of s, although accessing different cols (nc * ith + c)
    // will this cause any performance issue? how to avoid this?
    for (int r = 0; r < nr; r++) {
        const float scale = *((const float *)((const int8_t *)vy + r * (n + 4) + n));
        float *restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1m_i8b_LUT(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                           const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    // [M, K] * [K, N] = [M, N]
    // nr -> N
    // bs -> M, bs = nc * threads
    // n -> K
    assert(n % 4 == 0);
    assert(bs % nc == 0);

    // printf("ggml_gemm_i2s_i8b_LUT: %d %d %d %d\n", n, nr, nc, bs); // 3200 128 800 3200 (2160 8640)

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);

    static const int group_size = 640;

    const int group_size_remain = (n - 1) % group_size + 1;
    const int group_count = (n - group_size_remain) / group_size;  // not including remains
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;            // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;
    const int blck_num = group_size_remain / 20 * 4;
    const int blck_remain = group_size_remain % 20 / 4;
    const int table_stride = (group_count * group_size / 5 + blck_num) * 243 + blck_remain * 81;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 5;
            const int16_t *local_table = tables + (t * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }

        // group remain
        const int group_offset = group_count * group_size / 5;
        const int16_t *local_table = tables + (t * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
        for (int i = 0; i < blck_num; i++) {
            const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
            for (int v = 0; v < 243; v++) {
                __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
            }
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        for (int i = 0; i < blck_remain; i++) {
            const int16_t *restrict this_table = local_table + (blck_num * 243 + i * 81) * TABLE_ENTRY_SIZE;
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + blck_num + i) * bs;
            for (int v = 0; v < 243; v++) {
                __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
            }
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
        memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    }

    // tile remain
    if (entry_tile_remain > 0) {
        // groups
        for (int g = 0; g < group_count; g++) {
            const int group_offset = g * group_size / 5;
            const int16_t *local_table = tables + (entry_tile_count * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
            for (int i = 0; i < group_size / 5; i++) {
                const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
                const uint8_t *restrict this_x = (const uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int v = 0; v < 243; v++) {
                    __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
                }
                ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }

        // group remain
        const int group_offset = group_count * group_size / 5;
        const int16_t *local_table = tables + (entry_tile_count * table_stride + group_offset * 243) * TABLE_ENTRY_SIZE;
        for (int i = 0; i < blck_num; i++) {
            const int16_t *restrict this_table = local_table + i * 243 * TABLE_ENTRY_SIZE;
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
            for (int v = 0; v < 243; v++) {
                __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
            }
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        for (int i = 0; i < blck_remain; i++) {
            const int16_t *restrict this_table = local_table + (blck_num * 243 + i * 81) * TABLE_ENTRY_SIZE;
            const uint8_t *restrict this_x = (const uint8_t *)vx + (group_count * group_size / 5 + blck_num + i) * bs;
            for (int v = 0; v < 243; v++) {
                __builtin_prefetch(this_table + v * TABLE_ENTRY_SIZE, 0, 3);  // read-only(0), temporal locality(3=high)
            }
            ADD_TABLE_ENTRIES_BLOCK(nc, this_x, this_table, sum_i16);
        }
        ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
        memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    }

    // copy back to s[N, M]
    // TODO:
    // multiple threads might access the same row (r) of s, although accessing different cols (nc * ith + c)
    // will this cause any performance issue? how to avoid this?
    for (int r = 0; r < nr; r++) {
        const float scale = *((const float *)((const int8_t *)vy + r * (n + 4) + n));
        float *restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(sum_i16);
    free(sum_i32);
}

inline static void add(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] + y[i];
    }
}

inline static void sub(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] - y[i];
    }
}

inline static void rev(int16_t *restrict t1, const int16_t *restrict t2) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = -t2[i];
    }
}

void gemm_make_table_i2s(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;

    add(table + 41 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y0);
    add(table + 43 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y1);
    add(table + 49 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y2);
    add(table + 67 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y3);
    rev(table + 39 * TABLE_ENTRY_SIZE, table + 41 * TABLE_ENTRY_SIZE);
    rev(table + 37 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE);
    sub(table + 42 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE, y0);
    add(table + 44 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE, y0);
    rev(table + 31 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE);
    sub(table + 46 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y1);
    sub(table + 48 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y0);
    add(table + 50 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y0);
    add(table + 52 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y1);
    rev(table + 13 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE);
    sub(table + 58 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y2);
    sub(table + 64 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y1);
    sub(table + 66 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y0);
    add(table + 68 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y0);
    add(table + 70 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y1);
    add(table + 76 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y2);
    rev(table + 38 * TABLE_ENTRY_SIZE, table + 42 * TABLE_ENTRY_SIZE);
    rev(table + 36 * TABLE_ENTRY_SIZE, table + 44 * TABLE_ENTRY_SIZE);
    rev(table + 34 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE);
    sub(table + 45 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE, y0);
    add(table + 47 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE, y0);
    rev(table + 32 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE);
    rev(table + 30 * TABLE_ENTRY_SIZE, table + 50 * TABLE_ENTRY_SIZE);
    rev(table + 28 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE);
    sub(table + 51 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    add(table + 53 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    rev(table + 22 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE);
    sub(table + 55 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y1);
    sub(table + 57 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y0);
    add(table + 59 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y0);
    add(table + 61 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y1);
    rev(table + 16 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE);
    sub(table + 63 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    add(table + 65 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    rev(table + 14 * TABLE_ENTRY_SIZE, table + 66 * TABLE_ENTRY_SIZE);
    rev(table + 12 * TABLE_ENTRY_SIZE, table + 68 * TABLE_ENTRY_SIZE);
    rev(table + 10 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE);
    sub(table + 69 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE, y0);
    add(table + 71 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE, y0);
    rev(table + 4 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE);
    sub(table + 73 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y1);
    sub(table + 75 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add(table + 77 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add(table + 79 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y1);
    rev(table + 35 * TABLE_ENTRY_SIZE, table + 45 * TABLE_ENTRY_SIZE);
    rev(table + 33 * TABLE_ENTRY_SIZE, table + 47 * TABLE_ENTRY_SIZE);
    rev(table + 29 * TABLE_ENTRY_SIZE, table + 51 * TABLE_ENTRY_SIZE);
    rev(table + 27 * TABLE_ENTRY_SIZE, table + 53 * TABLE_ENTRY_SIZE);
    rev(table + 25 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE);
    sub(table + 54 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE, y0);
    add(table + 56 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE, y0);
    rev(table + 23 * TABLE_ENTRY_SIZE, table + 57 * TABLE_ENTRY_SIZE);
    rev(table + 21 * TABLE_ENTRY_SIZE, table + 59 * TABLE_ENTRY_SIZE);
    rev(table + 19 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE);
    sub(table + 60 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE, y0);
    add(table + 62 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE, y0);
    rev(table + 17 * TABLE_ENTRY_SIZE, table + 63 * TABLE_ENTRY_SIZE);
    rev(table + 15 * TABLE_ENTRY_SIZE, table + 65 * TABLE_ENTRY_SIZE);
    rev(table + 11 * TABLE_ENTRY_SIZE, table + 69 * TABLE_ENTRY_SIZE);
    rev(table + 9 * TABLE_ENTRY_SIZE, table + 71 * TABLE_ENTRY_SIZE);
    rev(table + 7 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE);
    sub(table + 72 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE, y0);
    add(table + 74 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE, y0);
    rev(table + 5 * TABLE_ENTRY_SIZE, table + 75 * TABLE_ENTRY_SIZE);
    rev(table + 3 * TABLE_ENTRY_SIZE, table + 77 * TABLE_ENTRY_SIZE);
    rev(table + 1 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE);
    sub(table + 78 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE, y0);
    add(table + 80 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE, y0);
    rev(table + 26 * TABLE_ENTRY_SIZE, table + 54 * TABLE_ENTRY_SIZE);
    rev(table + 24 * TABLE_ENTRY_SIZE, table + 56 * TABLE_ENTRY_SIZE);
    rev(table + 20 * TABLE_ENTRY_SIZE, table + 60 * TABLE_ENTRY_SIZE);
    rev(table + 18 * TABLE_ENTRY_SIZE, table + 62 * TABLE_ENTRY_SIZE);
    rev(table + 8 * TABLE_ENTRY_SIZE, table + 72 * TABLE_ENTRY_SIZE);
    rev(table + 6 * TABLE_ENTRY_SIZE, table + 74 * TABLE_ENTRY_SIZE);
    rev(table + 2 * TABLE_ENTRY_SIZE, table + 78 * TABLE_ENTRY_SIZE);
    rev(table + 0 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE);
}

void gemm_make_table_i1_58s(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y4 = y3 + TABLE_ENTRY_SIZE;

    add(table + 122 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y0);
    add(table + 124 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y1);
    add(table + 130 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y2);
    add(table + 148 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y3);
    add(table + 202 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y4);
    rev(table + 120 * TABLE_ENTRY_SIZE, table + 122 * TABLE_ENTRY_SIZE);
    rev(table + 118 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE);
    sub(table + 123 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    add(table + 125 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    rev(table + 112 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE);
    sub(table + 127 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y1);
    sub(table + 129 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y0);
    add(table + 131 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y0);
    add(table + 133 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y1);
    rev(table + 94 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE);
    sub(table + 139 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y2);
    sub(table + 145 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y1);
    sub(table + 147 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y0);
    add(table + 149 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y0);
    add(table + 151 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y1);
    add(table + 157 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y2);
    rev(table + 40 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE);
    sub(table + 175 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y3);
    sub(table + 193 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y2);
    sub(table + 199 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y1);
    sub(table + 201 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y0);
    add(table + 203 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y0);
    add(table + 205 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y1);
    add(table + 211 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y2);
    add(table + 229 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y3);
    rev(table + 119 * TABLE_ENTRY_SIZE, table + 123 * TABLE_ENTRY_SIZE);
    rev(table + 117 * TABLE_ENTRY_SIZE, table + 125 * TABLE_ENTRY_SIZE);
    rev(table + 115 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE);
    sub(table + 126 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE, y0);
    add(table + 128 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE, y0);
    rev(table + 113 * TABLE_ENTRY_SIZE, table + 129 * TABLE_ENTRY_SIZE);
    rev(table + 111 * TABLE_ENTRY_SIZE, table + 131 * TABLE_ENTRY_SIZE);
    rev(table + 109 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE);
    sub(table + 132 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE, y0);
    add(table + 134 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE, y0);
    rev(table + 103 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE);
    sub(table + 136 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y1);
    sub(table + 138 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y0);
    add(table + 140 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y0);
    add(table + 142 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y1);
    rev(table + 97 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE);
    sub(table + 144 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE, y0);
    add(table + 146 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE, y0);
    rev(table + 95 * TABLE_ENTRY_SIZE, table + 147 * TABLE_ENTRY_SIZE);
    rev(table + 93 * TABLE_ENTRY_SIZE, table + 149 * TABLE_ENTRY_SIZE);
    rev(table + 91 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE);
    sub(table + 150 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE, y0);
    add(table + 152 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE, y0);
    rev(table + 85 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE);
    sub(table + 154 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y1);
    sub(table + 156 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y0);
    add(table + 158 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y0);
    add(table + 160 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y1);
    rev(table + 67 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE);
    sub(table + 166 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y2);
    sub(table + 172 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y1);
    sub(table + 174 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y0);
    add(table + 176 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y0);
    add(table + 178 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y1);
    add(table + 184 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y2);
    rev(table + 49 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE);
    sub(table + 190 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y1);
    sub(table + 192 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y0);
    add(table + 194 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y0);
    add(table + 196 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y1);
    rev(table + 43 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE);
    sub(table + 198 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE, y0);
    add(table + 200 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE, y0);
    rev(table + 41 * TABLE_ENTRY_SIZE, table + 201 * TABLE_ENTRY_SIZE);
    rev(table + 39 * TABLE_ENTRY_SIZE, table + 203 * TABLE_ENTRY_SIZE);
    rev(table + 37 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE);
    sub(table + 204 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE, y0);
    add(table + 206 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE, y0);
    rev(table + 31 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE);
    sub(table + 208 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y1);
    sub(table + 210 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y0);
    add(table + 212 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y0);
    add(table + 214 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y1);
    rev(table + 13 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE);
    sub(table + 220 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y2);
    sub(table + 226 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y1);
    sub(table + 228 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y0);
    add(table + 230 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y0);
    add(table + 232 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y1);
    add(table + 238 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y2);
    rev(table + 116 * TABLE_ENTRY_SIZE, table + 126 * TABLE_ENTRY_SIZE);
    rev(table + 114 * TABLE_ENTRY_SIZE, table + 128 * TABLE_ENTRY_SIZE);
    rev(table + 110 * TABLE_ENTRY_SIZE, table + 132 * TABLE_ENTRY_SIZE);
    rev(table + 108 * TABLE_ENTRY_SIZE, table + 134 * TABLE_ENTRY_SIZE);
    rev(table + 106 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE);
    sub(table + 135 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE, y0);
    add(table + 137 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE, y0);
    rev(table + 104 * TABLE_ENTRY_SIZE, table + 138 * TABLE_ENTRY_SIZE);
    rev(table + 102 * TABLE_ENTRY_SIZE, table + 140 * TABLE_ENTRY_SIZE);
    rev(table + 100 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE);
    sub(table + 141 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE, y0);
    add(table + 143 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE, y0);
    rev(table + 98 * TABLE_ENTRY_SIZE, table + 144 * TABLE_ENTRY_SIZE);
    rev(table + 96 * TABLE_ENTRY_SIZE, table + 146 * TABLE_ENTRY_SIZE);
    rev(table + 92 * TABLE_ENTRY_SIZE, table + 150 * TABLE_ENTRY_SIZE);
    rev(table + 90 * TABLE_ENTRY_SIZE, table + 152 * TABLE_ENTRY_SIZE);
    rev(table + 88 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE);
    sub(table + 153 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE, y0);
    add(table + 155 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE, y0);
    rev(table + 86 * TABLE_ENTRY_SIZE, table + 156 * TABLE_ENTRY_SIZE);
    rev(table + 84 * TABLE_ENTRY_SIZE, table + 158 * TABLE_ENTRY_SIZE);
    rev(table + 82 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE);
    sub(table + 159 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE, y0);
    add(table + 161 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE, y0);
    rev(table + 76 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE);
    sub(table + 163 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y1);
    sub(table + 165 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y0);
    add(table + 167 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y0);
    add(table + 169 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y1);
    rev(table + 70 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE);
    sub(table + 171 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE, y0);
    add(table + 173 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE, y0);
    rev(table + 68 * TABLE_ENTRY_SIZE, table + 174 * TABLE_ENTRY_SIZE);
    rev(table + 66 * TABLE_ENTRY_SIZE, table + 176 * TABLE_ENTRY_SIZE);
    rev(table + 64 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE);
    sub(table + 177 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE, y0);
    add(table + 179 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE, y0);
    rev(table + 58 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE);
    sub(table + 181 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y1);
    sub(table + 183 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y0);
    add(table + 185 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y0);
    add(table + 187 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y1);
    rev(table + 52 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE);
    sub(table + 189 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE, y0);
    add(table + 191 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE, y0);
    rev(table + 50 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE);
    rev(table + 48 * TABLE_ENTRY_SIZE, table + 194 * TABLE_ENTRY_SIZE);
    rev(table + 46 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE);
    sub(table + 195 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    add(table + 197 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    rev(table + 44 * TABLE_ENTRY_SIZE, table + 198 * TABLE_ENTRY_SIZE);
    rev(table + 42 * TABLE_ENTRY_SIZE, table + 200 * TABLE_ENTRY_SIZE);
    rev(table + 38 * TABLE_ENTRY_SIZE, table + 204 * TABLE_ENTRY_SIZE);
    rev(table + 36 * TABLE_ENTRY_SIZE, table + 206 * TABLE_ENTRY_SIZE);
    rev(table + 34 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE);
    sub(table + 207 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    add(table + 209 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    rev(table + 32 * TABLE_ENTRY_SIZE, table + 210 * TABLE_ENTRY_SIZE);
    rev(table + 30 * TABLE_ENTRY_SIZE, table + 212 * TABLE_ENTRY_SIZE);
    rev(table + 28 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE);
    sub(table + 213 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE, y0);
    add(table + 215 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE, y0);
    rev(table + 22 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE);
    sub(table + 217 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y1);
    sub(table + 219 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    add(table + 221 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    add(table + 223 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y1);
    rev(table + 16 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE);
    sub(table + 225 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE, y0);
    add(table + 227 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE, y0);
    rev(table + 14 * TABLE_ENTRY_SIZE, table + 228 * TABLE_ENTRY_SIZE);
    rev(table + 12 * TABLE_ENTRY_SIZE, table + 230 * TABLE_ENTRY_SIZE);
    rev(table + 10 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE);
    sub(table + 231 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE, y0);
    add(table + 233 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE, y0);
    rev(table + 4 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE);
    sub(table + 235 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y1);
    sub(table + 237 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y0);
    add(table + 239 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y0);
    add(table + 241 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y1);
    rev(table + 107 * TABLE_ENTRY_SIZE, table + 135 * TABLE_ENTRY_SIZE);
    rev(table + 105 * TABLE_ENTRY_SIZE, table + 137 * TABLE_ENTRY_SIZE);
    rev(table + 101 * TABLE_ENTRY_SIZE, table + 141 * TABLE_ENTRY_SIZE);
    rev(table + 99 * TABLE_ENTRY_SIZE, table + 143 * TABLE_ENTRY_SIZE);
    rev(table + 89 * TABLE_ENTRY_SIZE, table + 153 * TABLE_ENTRY_SIZE);
    rev(table + 87 * TABLE_ENTRY_SIZE, table + 155 * TABLE_ENTRY_SIZE);
    rev(table + 83 * TABLE_ENTRY_SIZE, table + 159 * TABLE_ENTRY_SIZE);
    rev(table + 81 * TABLE_ENTRY_SIZE, table + 161 * TABLE_ENTRY_SIZE);
    rev(table + 79 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE);
    sub(table + 162 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE, y0);
    add(table + 164 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE, y0);
    rev(table + 77 * TABLE_ENTRY_SIZE, table + 165 * TABLE_ENTRY_SIZE);
    rev(table + 75 * TABLE_ENTRY_SIZE, table + 167 * TABLE_ENTRY_SIZE);
    rev(table + 73 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE);
    sub(table + 168 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE, y0);
    add(table + 170 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE, y0);
    rev(table + 71 * TABLE_ENTRY_SIZE, table + 171 * TABLE_ENTRY_SIZE);
    rev(table + 69 * TABLE_ENTRY_SIZE, table + 173 * TABLE_ENTRY_SIZE);
    rev(table + 65 * TABLE_ENTRY_SIZE, table + 177 * TABLE_ENTRY_SIZE);
    rev(table + 63 * TABLE_ENTRY_SIZE, table + 179 * TABLE_ENTRY_SIZE);
    rev(table + 61 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE);
    sub(table + 180 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE, y0);
    add(table + 182 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE, y0);
    rev(table + 59 * TABLE_ENTRY_SIZE, table + 183 * TABLE_ENTRY_SIZE);
    rev(table + 57 * TABLE_ENTRY_SIZE, table + 185 * TABLE_ENTRY_SIZE);
    rev(table + 55 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE);
    sub(table + 186 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE, y0);
    add(table + 188 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE, y0);
    rev(table + 53 * TABLE_ENTRY_SIZE, table + 189 * TABLE_ENTRY_SIZE);
    rev(table + 51 * TABLE_ENTRY_SIZE, table + 191 * TABLE_ENTRY_SIZE);
    rev(table + 47 * TABLE_ENTRY_SIZE, table + 195 * TABLE_ENTRY_SIZE);
    rev(table + 45 * TABLE_ENTRY_SIZE, table + 197 * TABLE_ENTRY_SIZE);
    rev(table + 35 * TABLE_ENTRY_SIZE, table + 207 * TABLE_ENTRY_SIZE);
    rev(table + 33 * TABLE_ENTRY_SIZE, table + 209 * TABLE_ENTRY_SIZE);
    rev(table + 29 * TABLE_ENTRY_SIZE, table + 213 * TABLE_ENTRY_SIZE);
    rev(table + 27 * TABLE_ENTRY_SIZE, table + 215 * TABLE_ENTRY_SIZE);
    rev(table + 25 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE);
    sub(table + 216 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE, y0);
    add(table + 218 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE, y0);
    rev(table + 23 * TABLE_ENTRY_SIZE, table + 219 * TABLE_ENTRY_SIZE);
    rev(table + 21 * TABLE_ENTRY_SIZE, table + 221 * TABLE_ENTRY_SIZE);
    rev(table + 19 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE);
    sub(table + 222 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE, y0);
    add(table + 224 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE, y0);
    rev(table + 17 * TABLE_ENTRY_SIZE, table + 225 * TABLE_ENTRY_SIZE);
    rev(table + 15 * TABLE_ENTRY_SIZE, table + 227 * TABLE_ENTRY_SIZE);
    rev(table + 11 * TABLE_ENTRY_SIZE, table + 231 * TABLE_ENTRY_SIZE);
    rev(table + 9 * TABLE_ENTRY_SIZE, table + 233 * TABLE_ENTRY_SIZE);
    rev(table + 7 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE);
    sub(table + 234 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE, y0);
    add(table + 236 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE, y0);
    rev(table + 5 * TABLE_ENTRY_SIZE, table + 237 * TABLE_ENTRY_SIZE);
    rev(table + 3 * TABLE_ENTRY_SIZE, table + 239 * TABLE_ENTRY_SIZE);
    rev(table + 1 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE);
    sub(table + 240 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE, y0);
    add(table + 242 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE, y0);
    rev(table + 80 * TABLE_ENTRY_SIZE, table + 162 * TABLE_ENTRY_SIZE);
    rev(table + 78 * TABLE_ENTRY_SIZE, table + 164 * TABLE_ENTRY_SIZE);
    rev(table + 74 * TABLE_ENTRY_SIZE, table + 168 * TABLE_ENTRY_SIZE);
    rev(table + 72 * TABLE_ENTRY_SIZE, table + 170 * TABLE_ENTRY_SIZE);
    rev(table + 62 * TABLE_ENTRY_SIZE, table + 180 * TABLE_ENTRY_SIZE);
    rev(table + 60 * TABLE_ENTRY_SIZE, table + 182 * TABLE_ENTRY_SIZE);
    rev(table + 56 * TABLE_ENTRY_SIZE, table + 186 * TABLE_ENTRY_SIZE);
    rev(table + 54 * TABLE_ENTRY_SIZE, table + 188 * TABLE_ENTRY_SIZE);
    rev(table + 26 * TABLE_ENTRY_SIZE, table + 216 * TABLE_ENTRY_SIZE);
    rev(table + 24 * TABLE_ENTRY_SIZE, table + 218 * TABLE_ENTRY_SIZE);
    rev(table + 20 * TABLE_ENTRY_SIZE, table + 222 * TABLE_ENTRY_SIZE);
    rev(table + 18 * TABLE_ENTRY_SIZE, table + 224 * TABLE_ENTRY_SIZE);
    rev(table + 8 * TABLE_ENTRY_SIZE, table + 234 * TABLE_ENTRY_SIZE);
    rev(table + 6 * TABLE_ENTRY_SIZE, table + 236 * TABLE_ENTRY_SIZE);
    rev(table + 2 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE);
    rev(table + 0 * TABLE_ENTRY_SIZE, table + 242 * TABLE_ENTRY_SIZE);
}
