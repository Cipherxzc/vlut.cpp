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
#define ACCUMULATE_TABLE_TRANS(ss, ss2, nc, entry_tile_remain, entry_tile_count) \
    do { \
        for (int j = 0; j < entry_tile_remain; j++) { \
            for (int i = 0; i < nc; i++) { \
                ss2[(entry_tile_count * TABLE_ENTRY_SIZE + j) * nc + i] += ss[i * TABLE_ENTRY_SIZE + j]; \
            } \
        } \
    } while(0)

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

    const double eps = 1e-5;
    double max = eps;
    for (int i = 0; i < n; i++) {
        max = MAX(max, (double)x[i]);
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, eps);

    for (int i = 0; i < n; i++) {
        float v = round((double)x[i] * is);
        if (v > 127) v = 127;
#if defined(BITNET_AVX2) || defined(BITNET_ARM)
        if (v < -127) v = -127;
#else
        if (v < -128) v = -128;
#endif
        dst[i] = (int8_t)v;
    }

    float *scale = (float *)(dst + n);
    *scale = (float)s;
}

void quantize_row_i8_b_tile(const float *x, void *y, int64_t n, float *scale) {
    int8_t *dst = (int8_t *)y;

    const double eps = 1e-5;
    double max = eps;
    for (int i = 0; i < n; i++) {
        max = MAX(max, (double)x[i]);
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, eps);

    for (int i = 0, j = 0; i < n; i++, j += TABLE_ENTRY_SIZE) {
        float v = round((double)x[i] * is);
        if (v > 127) v = 127;
#ifdef BITNET_AVX2
        if (v < -127) v = -127;
#else
        if (v < -128) v = -128;
#endif
        dst[j] = (int8_t)v;
    }

    *scale = (float)s;
}


inline static void gemm_make_table_I2_tile(int16_t *restrict table, const int8_t *restrict y);
inline static void gemm_make_table_I2S_tile(int16_t *restrict table, const int8_t *restrict y);
inline static void gemm_make_table_I1_58T_tile(int16_t *restrict table, const int8_t *restrict y);

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


extern int16_t *restrict tables;
extern int8_t *tmp_src;
extern int16_t *sum1;
extern int *sum2;

void ggml_gemm_i2_i8_s_make_table_tile(int ith, const int8_t *restrict y, int ntables, int nr, int n,
                                       int16_t *restrict table) {
    UNUSED(ith);

    // int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * n * TABLE_ENTRY_SIZE);
    // for (int i = 0; i < nr; i += TABLE_ENTRY_SIZE) {
    //     int lim = MIN(i + TABLE_ENTRY_SIZE, nr) - i;

    //     const int8_t *restrict y0 = y + i * (n + 4);
    //     for (int j = 0; j < lim; j++){
    //         for (int k = 0; k < ntables * 4; k++){
    //             src[k * TABLE_ENTRY_SIZE + j] = y0[j * (n + 4) + k];
    //         }
    //     }

    //     int16_t *restrict table0 = table + n / 4 * 81 * i;
    //     for (int j = 0; j < ntables; j++){
    //         gemm_make_table_I2S_tile(table0 + j * 81 * TABLE_ENTRY_SIZE, src + j * 4 * TABLE_ENTRY_SIZE);
    //     }
    // }

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * 4 * TABLE_ENTRY_SIZE);
    for (int i = 0; i < nr; i += TABLE_ENTRY_SIZE) {
        int lim = MIN(i + TABLE_ENTRY_SIZE, nr) - i;

        const int8_t *restrict y0 = y + i * (n + 4);
        int16_t *restrict table0 = table + n / 4 * 81 * i;

        for (int k = 0; k < ntables; k++) {
            const int8_t *restrict y1 = y0 + k * 4;
            for (int j = 0; j < lim; j++) {
                src[j] = y1[j * (n + 4)];
                src[j + TABLE_ENTRY_SIZE] = y1[j * (n + 4) + 1];
                src[j + TABLE_ENTRY_SIZE * 2] = y1[j * (n + 4) + 2];
                src[j + TABLE_ENTRY_SIZE * 3] = y1[j * (n + 4) + 3];
            }
            gemm_make_table_I2S_tile(table0 + k * 81 * TABLE_ENTRY_SIZE, src);
        }
    }

    free(src);
}

void ggml_gemm_i1_58_i8_t_make_table_tile(int ith, const int8_t *restrict y, int ntables, int nr, int n,
                                          int16_t *restrict table) {
    UNUSED(ith);

    // int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * n * TABLE_ENTRY_SIZE);
    // for (int i = 0; i < nr; i += TABLE_ENTRY_SIZE) {
    //     int lim = MIN(i + TABLE_ENTRY_SIZE, nr) - i;

    //     const int8_t *restrict y0 = y + i * (n + 4);
    //     for (int j = 0; j < lim; j++){
    //         for (int k = 0; k < ntables * 5; k++){
    //             src[k * TABLE_ENTRY_SIZE + j] = y0[j * (n + 4) + k];
    //         }
    //     }

    //     int16_t *restrict table0 = table + n / 5 * 243 * i;
    //     for (int j = 0; j < ntables; j++){
    //         gemm_make_table_I1_58T_tile(table0 + j * 243 * TABLE_ENTRY_SIZE, src + j * 4 * TABLE_ENTRY_SIZE);
    //     }
    // }

    int8_t *restrict src = (int8_t *)malloc(sizeof(int8_t) * 4 * TABLE_ENTRY_SIZE);
    for (int i = 0; i < nr; i += TABLE_ENTRY_SIZE) {
        int lim = MIN(i + TABLE_ENTRY_SIZE, nr) - i;

        const int8_t *restrict y0 = y + i * (n + 4);
        int16_t *restrict table0 = table + n / 5 * 243 * i;

        for (int k = 0; k < ntables; k++) {
            const int8_t *restrict y1 = y0 + k * 5;
            for (int j = 0; j < lim; j++) {
                src[j] = y1[j * (n + 4)];
                src[j + TABLE_ENTRY_SIZE] = y1[j * (n + 4) + 1];
                src[j + TABLE_ENTRY_SIZE * 2] = y1[j * (n + 4) + 2];
                src[j + TABLE_ENTRY_SIZE * 3] = y1[j * (n + 4) + 3];
            }
            gemm_make_table_I1_58T_tile(table0 + k * 243 * TABLE_ENTRY_SIZE, src);
        }
    }

    free(src);
}

void ggml_gemm_i2_i8_t_LUT2_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                 const void *restrict vy, int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    UNUSED(ith);

    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    
    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 256);

    // int16_t *restrict sum_i16 = sum1 + ith * TABLE_ENTRY_SIZE * nc;
    // int *restrict sum_i32 = sum2 + ith * nr * nc;
    // int16_t *restrict table = tables + ith * TABLE_ENTRY_SIZE * 256;

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(table, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int t = 0; t < nr; t += TABLE_ENTRY_SIZE){
        const int8_t *restrict y0 = y + t * n;
        const int entry_len = MIN(nr - t, TABLE_ENTRY_SIZE);
        for (int g = 0; g < n; g += group_size) {
            int lim = g + group_size < n ? g + group_size : n;
            for (int i = (g >> 2); i < (lim >> 2); i++) {
#ifdef BITNET_DEBUG
                struct timespec start_make_table = get_thread_cpu_time();
#endif
                gemm_make_table_I2_tile(table, y0 + i * 4 * TABLE_ENTRY_SIZE);
#ifdef BITNET_DEBUG
                struct timespec end_make_table = get_thread_cpu_time();
                make_table_duration += get_time_diff(start_make_table, end_make_table);

                struct timespec start_LUT = get_thread_cpu_time();
#endif
                for (int c = 0; c < nc; c++) {
                    int v = x[i * bs + c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = table + v * TABLE_ENTRY_SIZE;
                    for (int r = 0; r < TABLE_ENTRY_SIZE; r++) {
                        rs[r] += rt[r];
                    }
                }
#ifdef BITNET_DEBUG
                struct timespec end_LUT = get_thread_cpu_time();
                LUT_duration += get_time_diff(start_LUT, end_LUT);
#endif
            }
#ifdef BITNET_DEBUG
            struct timespec start_convert = get_thread_cpu_time();
#endif
            for (int i = 0; i < nc; i++){
                for (int j = 0; j < entry_len; j++) {
                    sum_i32[(t + j) * nc + i] += sum_i16[i * TABLE_ENTRY_SIZE + j];
                }
            }
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
#ifdef BITNET_DEBUG
            struct timespec end_convert = get_thread_cpu_time();
            convert_duration += get_time_diff(start_convert, end_convert);
#endif
        }
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE): nr) * n;
    const float *sc = (const float *)(y + y_size);
    for (int r = 0; r < nr; r++) {
        for (int c = 0; c < nc; c++) {
            s[r * bs + c] = sum_i32[r * nc + c] * sc[r];  // 将输出转置回来
        }
    }
#ifdef BITNET_DEBUG
    struct timespec end_scale = get_thread_cpu_time();
    scale_duration += get_time_diff(start_scale, end_scale);

    pthread_mutex_lock(&time_mutex);
    make_table_time += make_table_duration;
    convert_time += convert_duration;
    scale_time += scale_duration;
    LUT_time += LUT_duration;
    pthread_mutex_unlock(&time_mutex);
#endif

    free(table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i2_i8_s_LUT2_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                 const void *restrict vy, int nr, int nc) {
    UNUSED(ith);

    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);

    // int16_t *restrict sum_i16 = sum1 + ith * TABLE_ENTRY_SIZE * nc;
    // int *restrict sum_i32 = sum2 + ith * nr * nc;
    // int16_t *restrict table = tables + ith * TABLE_ENTRY_SIZE * 256;

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(table + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int t = 0; t < nr; t += TABLE_ENTRY_SIZE) {
        const int8_t *restrict y0 = y + t * n;
        const int entry_len = MIN(nr - t, TABLE_ENTRY_SIZE);
        for (int g = 0; g < n; g += group_size) {
            int lim = g + group_size < n ? g + group_size : n;
            for (int i = (g >> 2); i < (lim >> 2); i++) {
#ifdef BITNET_DEBUG
                struct timespec start_make_table = get_thread_cpu_time();
#endif
                gemm_make_table_I2S_tile(table, y0 + i * 4 * TABLE_ENTRY_SIZE);
#ifdef BITNET_DEBUG
                struct timespec end_make_table = get_thread_cpu_time();
                make_table_duration += get_time_diff(start_make_table, end_make_table);

                struct timespec start_LUT = get_thread_cpu_time();
#endif
                for (int c = 0; c < nc; c++) {
                    int v = x[i * bs + c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = table + v * TABLE_ENTRY_SIZE;
                    for (int r = 0; r < TABLE_ENTRY_SIZE; r++) {
                        rs[r] += rt[r];
                    }
                }
#ifdef BITNET_DEBUG
                struct timespec end_LUT = get_thread_cpu_time();
                LUT_duration += get_time_diff(start_LUT, end_LUT);
#endif
            }
#ifdef BITNET_DEBUG
            struct timespec start_convert = get_thread_cpu_time();
#endif
            for (int i = 0; i < nc; i++) {
                for (int j = 0; j < entry_len; j++) {
                    sum_i32[(t + j) * nc + i] += sum_i16[i * TABLE_ENTRY_SIZE + j];
                }
            }
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
#ifdef BITNET_DEBUG
            struct timespec end_convert = get_thread_cpu_time();
            convert_duration += get_time_diff(start_convert, end_convert);
#endif
        }
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)(y + y_size);
    for (int r = 0; r < nr; r++) {
        for (int c = 0; c < nc; c++) {
            s[r * bs + c] = sum_i32[r * nc + c] * sc[r];  // 将输出转置回来
        }
    }
#ifdef BITNET_DEBUG
    struct timespec end_scale = get_thread_cpu_time();
    scale_duration += get_time_diff(start_scale, end_scale);

    pthread_mutex_lock(&time_mutex);
    make_table_time += make_table_duration;
    convert_time += convert_duration;
    scale_time += scale_duration;
    LUT_time += LUT_duration;
    pthread_mutex_unlock(&time_mutex);
#endif

    free(table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1_58_i8_t_LUT2_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                    const void *restrict vy, int nr, int nc) {
    UNUSED(ith);

    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 5 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 243);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);
    memset(table + 121 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 640;

    for (int t = 0; t < nr; t += TABLE_ENTRY_SIZE) {
        const int8_t *restrict y0 = y + t * n;
        const int entry_len = MIN(nr - t, TABLE_ENTRY_SIZE);
        for (int g = 0; g < n; g += group_size) {
            int lim = g + group_size < n ? g + group_size : n;
            for (int i = g / 5; i < lim / 5; i++) {
                gemm_make_table_I1_58T_tile(table, y0 + i * 5 * TABLE_ENTRY_SIZE);
                
                for (int c = 0; c < nc; c++) {
                    int v = x[i * bs + c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = table + v * TABLE_ENTRY_SIZE;
                    for (int r = 0; r < TABLE_ENTRY_SIZE; r++) {
                        rs[r] += rt[r];
                    }
                }
            }
            for (int i = 0; i < nc; i++) {
                for (int j = 0; j < entry_len; j++) {
                    sum_i32[(t + j) * nc + i] += sum_i16[i * TABLE_ENTRY_SIZE + j];
                }
            }
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)(y + y_size);
    for (int r = 0; r < nr; r++) {
        for (int c = 0; c < nc; c++) {
            s[r * bs + c] = sum_i32[r * nc + c] * sc[r];  // 将输出转置回来
        }
    }

    free(table);
    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i2_i8_s_LUT_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                const void *restrict vy, int nr, int nc) {
    UNUSED(ith);

    // nr: src1->ne[1], nc: src0->ne[1]
    // [M, K] * [K, N] = [M, N]
    // nr -> N
    // bs -> M, bs = nc * threads
    // n -> K
    assert(n % 4 == 0);
    assert(bs % nc == 0);

    // printf("ggml_gemm_i2_i8_s_LUT_tile: %d %d %d %d\n", n, nr, nc, bs); // 3200 128 800 3200 (2160 8640)

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);

    static const int group_size = 512;

    //  *restrict local_table = table;
    const int table_count = n / 4;
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (g * group_size / 4 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (g * group_size / 4 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
        const float scale = *((const float *)((int8_t *)vy + r * (n + 4) + n));
        float* restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(sum_i16);
    free(sum_i32);
}

void ggml_gemm_i1_58_i8_t_LUT_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                   const void *restrict vy, int nr, int nc) {
    UNUSED(ith);

    // nr: src1->ne[1], nc: src0->ne[1]
    // [M, K] * [K, N] = [M, N]
    // nr -> N
    // bs -> M, bs = nc * threads
    // n -> K
    assert(n % 4 == 0);
    assert(bs % nc == 0);

    // printf("ggml_gemm_i2_i8_s_LUT_tile: %d %d %d %d\n", n, nr, nc, bs); // 3200 128 800 3200 (2160 8640)

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);

    static const int group_size = 640;

    //  *restrict local_table = table;
    const int table_count = n / 5;
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (group_count * group_size / 5 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (g * group_size / 5 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
                const uint8_t *restrict this_x = (uint8_t *)vx + (group_count * group_size / 4 + i) * bs;
                for (int c = 0; c < nc; c++) {
                    int v = this_x[c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
                    ADD_TABLE_ENTRIES(rs, rt, TABLE_ENTRY_SIZE);
                }
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
        const float scale = *((const float *)((int8_t *)vy + r * (n + 4) + n));
        float *restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(sum_i16);
    free(sum_i32);
}

inline static void add_tile(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] + y[i];
    }
}

inline static void sub_tile(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] - y[i];
    }
}

inline static void rev_tile(int16_t *restrict t1, const int16_t *restrict t2) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = -t2[i];
    }
}

void gemm_make_table_I2_tile(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;

    add_tile(table + 1 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 3 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 4 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 12 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 16 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 48 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 64 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y3);
    sub_tile(table + 192 * TABLE_ENTRY_SIZE, table + 0 * TABLE_ENTRY_SIZE, y3);
    add_tile(table + 5 * TABLE_ENTRY_SIZE, table + 4 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 7 * TABLE_ENTRY_SIZE, table + 4 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 13 * TABLE_ENTRY_SIZE, table + 12 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 15 * TABLE_ENTRY_SIZE, table + 12 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 17 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 19 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 20 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 28 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 49 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 51 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 52 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 60 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 65 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 67 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 68 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 76 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 80 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 112 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 193 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 195 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 196 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 204 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 208 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 240 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 21 * TABLE_ENTRY_SIZE, table + 20 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 23 * TABLE_ENTRY_SIZE, table + 20 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 29 * TABLE_ENTRY_SIZE, table + 28 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 31 * TABLE_ENTRY_SIZE, table + 28 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 53 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 55 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 61 * TABLE_ENTRY_SIZE, table + 60 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 63 * TABLE_ENTRY_SIZE, table + 60 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 69 * TABLE_ENTRY_SIZE, table + 68 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 71 * TABLE_ENTRY_SIZE, table + 68 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 77 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 79 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 81 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 83 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 84 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 92 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 113 * TABLE_ENTRY_SIZE, table + 112 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 115 * TABLE_ENTRY_SIZE, table + 112 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 116 * TABLE_ENTRY_SIZE, table + 112 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 124 * TABLE_ENTRY_SIZE, table + 112 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 197 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 199 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 205 * TABLE_ENTRY_SIZE, table + 204 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 207 * TABLE_ENTRY_SIZE, table + 204 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 209 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 211 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 212 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 220 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 241 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 243 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 244 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 252 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 85 * TABLE_ENTRY_SIZE, table + 84 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 87 * TABLE_ENTRY_SIZE, table + 84 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 93 * TABLE_ENTRY_SIZE, table + 92 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 95 * TABLE_ENTRY_SIZE, table + 92 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 117 * TABLE_ENTRY_SIZE, table + 116 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 119 * TABLE_ENTRY_SIZE, table + 116 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 125 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 127 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 213 * TABLE_ENTRY_SIZE, table + 212 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 215 * TABLE_ENTRY_SIZE, table + 212 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 221 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 223 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 245 * TABLE_ENTRY_SIZE, table + 244 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 247 * TABLE_ENTRY_SIZE, table + 244 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 253 * TABLE_ENTRY_SIZE, table + 252 * TABLE_ENTRY_SIZE, y0);
    sub_tile(table + 255 * TABLE_ENTRY_SIZE, table + 252 * TABLE_ENTRY_SIZE, y0);
}

void gemm_make_table_I2S_tile(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;

    add_tile(table + 41 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 43 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 49 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 67 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y3);
    rev_tile(table + 39 * TABLE_ENTRY_SIZE, table + 41 * TABLE_ENTRY_SIZE);
    rev_tile(table + 37 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE);
    sub_tile(table + 42 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 44 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 31 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE);
    sub_tile(table + 46 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 48 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 50 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 52 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 13 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE);
    sub_tile(table + 58 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 64 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 66 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 68 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 70 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 76 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y2);
    rev_tile(table + 38 * TABLE_ENTRY_SIZE, table + 42 * TABLE_ENTRY_SIZE);
    rev_tile(table + 36 * TABLE_ENTRY_SIZE, table + 44 * TABLE_ENTRY_SIZE);
    rev_tile(table + 34 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE);
    sub_tile(table + 45 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 47 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 32 * TABLE_ENTRY_SIZE, table + 48 * TABLE_ENTRY_SIZE);
    rev_tile(table + 30 * TABLE_ENTRY_SIZE, table + 50 * TABLE_ENTRY_SIZE);
    rev_tile(table + 28 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE);
    sub_tile(table + 51 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 53 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 22 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE);
    sub_tile(table + 55 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 57 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 59 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 61 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 16 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE);
    sub_tile(table + 63 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 65 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 14 * TABLE_ENTRY_SIZE, table + 66 * TABLE_ENTRY_SIZE);
    rev_tile(table + 12 * TABLE_ENTRY_SIZE, table + 68 * TABLE_ENTRY_SIZE);
    rev_tile(table + 10 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE);
    sub_tile(table + 69 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 71 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 4 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE);
    sub_tile(table + 73 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 75 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 77 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 79 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 35 * TABLE_ENTRY_SIZE, table + 45 * TABLE_ENTRY_SIZE);
    rev_tile(table + 33 * TABLE_ENTRY_SIZE, table + 47 * TABLE_ENTRY_SIZE);
    rev_tile(table + 29 * TABLE_ENTRY_SIZE, table + 51 * TABLE_ENTRY_SIZE);
    rev_tile(table + 27 * TABLE_ENTRY_SIZE, table + 53 * TABLE_ENTRY_SIZE);
    rev_tile(table + 25 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE);
    sub_tile(table + 54 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 56 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 23 * TABLE_ENTRY_SIZE, table + 57 * TABLE_ENTRY_SIZE);
    rev_tile(table + 21 * TABLE_ENTRY_SIZE, table + 59 * TABLE_ENTRY_SIZE);
    rev_tile(table + 19 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE);
    sub_tile(table + 60 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 62 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 17 * TABLE_ENTRY_SIZE, table + 63 * TABLE_ENTRY_SIZE);
    rev_tile(table + 15 * TABLE_ENTRY_SIZE, table + 65 * TABLE_ENTRY_SIZE);
    rev_tile(table + 11 * TABLE_ENTRY_SIZE, table + 69 * TABLE_ENTRY_SIZE);
    rev_tile(table + 9 * TABLE_ENTRY_SIZE, table + 71 * TABLE_ENTRY_SIZE);
    rev_tile(table + 7 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE);
    sub_tile(table + 72 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 74 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 5 * TABLE_ENTRY_SIZE, table + 75 * TABLE_ENTRY_SIZE);
    rev_tile(table + 3 * TABLE_ENTRY_SIZE, table + 77 * TABLE_ENTRY_SIZE);
    rev_tile(table + 1 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE);
    sub_tile(table + 78 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 80 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 26 * TABLE_ENTRY_SIZE, table + 54 * TABLE_ENTRY_SIZE);
    rev_tile(table + 24 * TABLE_ENTRY_SIZE, table + 56 * TABLE_ENTRY_SIZE);
    rev_tile(table + 20 * TABLE_ENTRY_SIZE, table + 60 * TABLE_ENTRY_SIZE);
    rev_tile(table + 18 * TABLE_ENTRY_SIZE, table + 62 * TABLE_ENTRY_SIZE);
    rev_tile(table + 8 * TABLE_ENTRY_SIZE, table + 72 * TABLE_ENTRY_SIZE);
    rev_tile(table + 6 * TABLE_ENTRY_SIZE, table + 74 * TABLE_ENTRY_SIZE);
    rev_tile(table + 2 * TABLE_ENTRY_SIZE, table + 78 * TABLE_ENTRY_SIZE);
    rev_tile(table + 0 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE);
}

void gemm_make_table_I1_58T_tile(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y4 = y3 + TABLE_ENTRY_SIZE;

    add_tile(table + 122 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 124 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 130 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 148 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y3);
    add_tile(table + 202 * TABLE_ENTRY_SIZE, table + 121 * TABLE_ENTRY_SIZE, y4);
    rev_tile(table + 120 * TABLE_ENTRY_SIZE, table + 122 * TABLE_ENTRY_SIZE);
    rev_tile(table + 118 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE);
    sub_tile(table + 123 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 125 * TABLE_ENTRY_SIZE, table + 124 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 112 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE);
    sub_tile(table + 127 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 129 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 131 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 133 * TABLE_ENTRY_SIZE, table + 130 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 94 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE);
    sub_tile(table + 139 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 145 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 147 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 149 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 151 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 157 * TABLE_ENTRY_SIZE, table + 148 * TABLE_ENTRY_SIZE, y2);
    rev_tile(table + 40 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE);
    sub_tile(table + 175 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y3);
    sub_tile(table + 193 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 199 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 201 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 203 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 205 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 211 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y2);
    add_tile(table + 229 * TABLE_ENTRY_SIZE, table + 202 * TABLE_ENTRY_SIZE, y3);
    rev_tile(table + 119 * TABLE_ENTRY_SIZE, table + 123 * TABLE_ENTRY_SIZE);
    rev_tile(table + 117 * TABLE_ENTRY_SIZE, table + 125 * TABLE_ENTRY_SIZE);
    rev_tile(table + 115 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE);
    sub_tile(table + 126 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 128 * TABLE_ENTRY_SIZE, table + 127 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 113 * TABLE_ENTRY_SIZE, table + 129 * TABLE_ENTRY_SIZE);
    rev_tile(table + 111 * TABLE_ENTRY_SIZE, table + 131 * TABLE_ENTRY_SIZE);
    rev_tile(table + 109 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE);
    sub_tile(table + 132 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 134 * TABLE_ENTRY_SIZE, table + 133 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 103 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE);
    sub_tile(table + 136 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 138 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 140 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 142 * TABLE_ENTRY_SIZE, table + 139 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 97 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE);
    sub_tile(table + 144 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 146 * TABLE_ENTRY_SIZE, table + 145 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 95 * TABLE_ENTRY_SIZE, table + 147 * TABLE_ENTRY_SIZE);
    rev_tile(table + 93 * TABLE_ENTRY_SIZE, table + 149 * TABLE_ENTRY_SIZE);
    rev_tile(table + 91 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE);
    sub_tile(table + 150 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 152 * TABLE_ENTRY_SIZE, table + 151 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 85 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE);
    sub_tile(table + 154 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 156 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 158 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 160 * TABLE_ENTRY_SIZE, table + 157 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 67 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE);
    sub_tile(table + 166 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 172 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 174 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 176 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 178 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 184 * TABLE_ENTRY_SIZE, table + 175 * TABLE_ENTRY_SIZE, y2);
    rev_tile(table + 49 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE);
    sub_tile(table + 190 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 192 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 194 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 196 * TABLE_ENTRY_SIZE, table + 193 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 43 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE);
    sub_tile(table + 198 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 200 * TABLE_ENTRY_SIZE, table + 199 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 41 * TABLE_ENTRY_SIZE, table + 201 * TABLE_ENTRY_SIZE);
    rev_tile(table + 39 * TABLE_ENTRY_SIZE, table + 203 * TABLE_ENTRY_SIZE);
    rev_tile(table + 37 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE);
    sub_tile(table + 204 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 206 * TABLE_ENTRY_SIZE, table + 205 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 31 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE);
    sub_tile(table + 208 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 210 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 212 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 214 * TABLE_ENTRY_SIZE, table + 211 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 13 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE);
    sub_tile(table + 220 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y2);
    sub_tile(table + 226 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 228 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 230 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 232 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y1);
    add_tile(table + 238 * TABLE_ENTRY_SIZE, table + 229 * TABLE_ENTRY_SIZE, y2);
    rev_tile(table + 116 * TABLE_ENTRY_SIZE, table + 126 * TABLE_ENTRY_SIZE);
    rev_tile(table + 114 * TABLE_ENTRY_SIZE, table + 128 * TABLE_ENTRY_SIZE);
    rev_tile(table + 110 * TABLE_ENTRY_SIZE, table + 132 * TABLE_ENTRY_SIZE);
    rev_tile(table + 108 * TABLE_ENTRY_SIZE, table + 134 * TABLE_ENTRY_SIZE);
    rev_tile(table + 106 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE);
    sub_tile(table + 135 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 137 * TABLE_ENTRY_SIZE, table + 136 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 104 * TABLE_ENTRY_SIZE, table + 138 * TABLE_ENTRY_SIZE);
    rev_tile(table + 102 * TABLE_ENTRY_SIZE, table + 140 * TABLE_ENTRY_SIZE);
    rev_tile(table + 100 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE);
    sub_tile(table + 141 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 143 * TABLE_ENTRY_SIZE, table + 142 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 98 * TABLE_ENTRY_SIZE, table + 144 * TABLE_ENTRY_SIZE);
    rev_tile(table + 96 * TABLE_ENTRY_SIZE, table + 146 * TABLE_ENTRY_SIZE);
    rev_tile(table + 92 * TABLE_ENTRY_SIZE, table + 150 * TABLE_ENTRY_SIZE);
    rev_tile(table + 90 * TABLE_ENTRY_SIZE, table + 152 * TABLE_ENTRY_SIZE);
    rev_tile(table + 88 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE);
    sub_tile(table + 153 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 155 * TABLE_ENTRY_SIZE, table + 154 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 86 * TABLE_ENTRY_SIZE, table + 156 * TABLE_ENTRY_SIZE);
    rev_tile(table + 84 * TABLE_ENTRY_SIZE, table + 158 * TABLE_ENTRY_SIZE);
    rev_tile(table + 82 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE);
    sub_tile(table + 159 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 161 * TABLE_ENTRY_SIZE, table + 160 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 76 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE);
    sub_tile(table + 163 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 165 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 167 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 169 * TABLE_ENTRY_SIZE, table + 166 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 70 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE);
    sub_tile(table + 171 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 173 * TABLE_ENTRY_SIZE, table + 172 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 68 * TABLE_ENTRY_SIZE, table + 174 * TABLE_ENTRY_SIZE);
    rev_tile(table + 66 * TABLE_ENTRY_SIZE, table + 176 * TABLE_ENTRY_SIZE);
    rev_tile(table + 64 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE);
    sub_tile(table + 177 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 179 * TABLE_ENTRY_SIZE, table + 178 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 58 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE);
    sub_tile(table + 181 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 183 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 185 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 187 * TABLE_ENTRY_SIZE, table + 184 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 52 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE);
    sub_tile(table + 189 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 191 * TABLE_ENTRY_SIZE, table + 190 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 50 * TABLE_ENTRY_SIZE, table + 192 * TABLE_ENTRY_SIZE);
    rev_tile(table + 48 * TABLE_ENTRY_SIZE, table + 194 * TABLE_ENTRY_SIZE);
    rev_tile(table + 46 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE);
    sub_tile(table + 195 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 197 * TABLE_ENTRY_SIZE, table + 196 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 44 * TABLE_ENTRY_SIZE, table + 198 * TABLE_ENTRY_SIZE);
    rev_tile(table + 42 * TABLE_ENTRY_SIZE, table + 200 * TABLE_ENTRY_SIZE);
    rev_tile(table + 38 * TABLE_ENTRY_SIZE, table + 204 * TABLE_ENTRY_SIZE);
    rev_tile(table + 36 * TABLE_ENTRY_SIZE, table + 206 * TABLE_ENTRY_SIZE);
    rev_tile(table + 34 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE);
    sub_tile(table + 207 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 209 * TABLE_ENTRY_SIZE, table + 208 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 32 * TABLE_ENTRY_SIZE, table + 210 * TABLE_ENTRY_SIZE);
    rev_tile(table + 30 * TABLE_ENTRY_SIZE, table + 212 * TABLE_ENTRY_SIZE);
    rev_tile(table + 28 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE);
    sub_tile(table + 213 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 215 * TABLE_ENTRY_SIZE, table + 214 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 22 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE);
    sub_tile(table + 217 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 219 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 221 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 223 * TABLE_ENTRY_SIZE, table + 220 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 16 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE);
    sub_tile(table + 225 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 227 * TABLE_ENTRY_SIZE, table + 226 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 14 * TABLE_ENTRY_SIZE, table + 228 * TABLE_ENTRY_SIZE);
    rev_tile(table + 12 * TABLE_ENTRY_SIZE, table + 230 * TABLE_ENTRY_SIZE);
    rev_tile(table + 10 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE);
    sub_tile(table + 231 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 233 * TABLE_ENTRY_SIZE, table + 232 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 4 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE);
    sub_tile(table + 235 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y1);
    sub_tile(table + 237 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 239 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 241 * TABLE_ENTRY_SIZE, table + 238 * TABLE_ENTRY_SIZE, y1);
    rev_tile(table + 107 * TABLE_ENTRY_SIZE, table + 135 * TABLE_ENTRY_SIZE);
    rev_tile(table + 105 * TABLE_ENTRY_SIZE, table + 137 * TABLE_ENTRY_SIZE);
    rev_tile(table + 101 * TABLE_ENTRY_SIZE, table + 141 * TABLE_ENTRY_SIZE);
    rev_tile(table + 99 * TABLE_ENTRY_SIZE, table + 143 * TABLE_ENTRY_SIZE);
    rev_tile(table + 89 * TABLE_ENTRY_SIZE, table + 153 * TABLE_ENTRY_SIZE);
    rev_tile(table + 87 * TABLE_ENTRY_SIZE, table + 155 * TABLE_ENTRY_SIZE);
    rev_tile(table + 83 * TABLE_ENTRY_SIZE, table + 159 * TABLE_ENTRY_SIZE);
    rev_tile(table + 81 * TABLE_ENTRY_SIZE, table + 161 * TABLE_ENTRY_SIZE);
    rev_tile(table + 79 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE);
    sub_tile(table + 162 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 164 * TABLE_ENTRY_SIZE, table + 163 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 77 * TABLE_ENTRY_SIZE, table + 165 * TABLE_ENTRY_SIZE);
    rev_tile(table + 75 * TABLE_ENTRY_SIZE, table + 167 * TABLE_ENTRY_SIZE);
    rev_tile(table + 73 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE);
    sub_tile(table + 168 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 170 * TABLE_ENTRY_SIZE, table + 169 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 71 * TABLE_ENTRY_SIZE, table + 171 * TABLE_ENTRY_SIZE);
    rev_tile(table + 69 * TABLE_ENTRY_SIZE, table + 173 * TABLE_ENTRY_SIZE);
    rev_tile(table + 65 * TABLE_ENTRY_SIZE, table + 177 * TABLE_ENTRY_SIZE);
    rev_tile(table + 63 * TABLE_ENTRY_SIZE, table + 179 * TABLE_ENTRY_SIZE);
    rev_tile(table + 61 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE);
    sub_tile(table + 180 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 182 * TABLE_ENTRY_SIZE, table + 181 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 59 * TABLE_ENTRY_SIZE, table + 183 * TABLE_ENTRY_SIZE);
    rev_tile(table + 57 * TABLE_ENTRY_SIZE, table + 185 * TABLE_ENTRY_SIZE);
    rev_tile(table + 55 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE);
    sub_tile(table + 186 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 188 * TABLE_ENTRY_SIZE, table + 187 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 53 * TABLE_ENTRY_SIZE, table + 189 * TABLE_ENTRY_SIZE);
    rev_tile(table + 51 * TABLE_ENTRY_SIZE, table + 191 * TABLE_ENTRY_SIZE);
    rev_tile(table + 47 * TABLE_ENTRY_SIZE, table + 195 * TABLE_ENTRY_SIZE);
    rev_tile(table + 45 * TABLE_ENTRY_SIZE, table + 197 * TABLE_ENTRY_SIZE);
    rev_tile(table + 35 * TABLE_ENTRY_SIZE, table + 207 * TABLE_ENTRY_SIZE);
    rev_tile(table + 33 * TABLE_ENTRY_SIZE, table + 209 * TABLE_ENTRY_SIZE);
    rev_tile(table + 29 * TABLE_ENTRY_SIZE, table + 213 * TABLE_ENTRY_SIZE);
    rev_tile(table + 27 * TABLE_ENTRY_SIZE, table + 215 * TABLE_ENTRY_SIZE);
    rev_tile(table + 25 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE);
    sub_tile(table + 216 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 218 * TABLE_ENTRY_SIZE, table + 217 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 23 * TABLE_ENTRY_SIZE, table + 219 * TABLE_ENTRY_SIZE);
    rev_tile(table + 21 * TABLE_ENTRY_SIZE, table + 221 * TABLE_ENTRY_SIZE);
    rev_tile(table + 19 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE);
    sub_tile(table + 222 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 224 * TABLE_ENTRY_SIZE, table + 223 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 17 * TABLE_ENTRY_SIZE, table + 225 * TABLE_ENTRY_SIZE);
    rev_tile(table + 15 * TABLE_ENTRY_SIZE, table + 227 * TABLE_ENTRY_SIZE);
    rev_tile(table + 11 * TABLE_ENTRY_SIZE, table + 231 * TABLE_ENTRY_SIZE);
    rev_tile(table + 9 * TABLE_ENTRY_SIZE, table + 233 * TABLE_ENTRY_SIZE);
    rev_tile(table + 7 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE);
    sub_tile(table + 234 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 236 * TABLE_ENTRY_SIZE, table + 235 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 5 * TABLE_ENTRY_SIZE, table + 237 * TABLE_ENTRY_SIZE);
    rev_tile(table + 3 * TABLE_ENTRY_SIZE, table + 239 * TABLE_ENTRY_SIZE);
    rev_tile(table + 1 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE);
    sub_tile(table + 240 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE, y0);
    add_tile(table + 242 * TABLE_ENTRY_SIZE, table + 241 * TABLE_ENTRY_SIZE, y0);
    rev_tile(table + 80 * TABLE_ENTRY_SIZE, table + 162 * TABLE_ENTRY_SIZE);
    rev_tile(table + 78 * TABLE_ENTRY_SIZE, table + 164 * TABLE_ENTRY_SIZE);
    rev_tile(table + 74 * TABLE_ENTRY_SIZE, table + 168 * TABLE_ENTRY_SIZE);
    rev_tile(table + 72 * TABLE_ENTRY_SIZE, table + 170 * TABLE_ENTRY_SIZE);
    rev_tile(table + 62 * TABLE_ENTRY_SIZE, table + 180 * TABLE_ENTRY_SIZE);
    rev_tile(table + 60 * TABLE_ENTRY_SIZE, table + 182 * TABLE_ENTRY_SIZE);
    rev_tile(table + 56 * TABLE_ENTRY_SIZE, table + 186 * TABLE_ENTRY_SIZE);
    rev_tile(table + 54 * TABLE_ENTRY_SIZE, table + 188 * TABLE_ENTRY_SIZE);
    rev_tile(table + 26 * TABLE_ENTRY_SIZE, table + 216 * TABLE_ENTRY_SIZE);
    rev_tile(table + 24 * TABLE_ENTRY_SIZE, table + 218 * TABLE_ENTRY_SIZE);
    rev_tile(table + 20 * TABLE_ENTRY_SIZE, table + 222 * TABLE_ENTRY_SIZE);
    rev_tile(table + 18 * TABLE_ENTRY_SIZE, table + 224 * TABLE_ENTRY_SIZE);
    rev_tile(table + 8 * TABLE_ENTRY_SIZE, table + 234 * TABLE_ENTRY_SIZE);
    rev_tile(table + 6 * TABLE_ENTRY_SIZE, table + 236 * TABLE_ENTRY_SIZE);
    rev_tile(table + 2 * TABLE_ENTRY_SIZE, table + 240 * TABLE_ENTRY_SIZE);
    rev_tile(table + 0 * TABLE_ENTRY_SIZE, table + 242 * TABLE_ENTRY_SIZE);
}
