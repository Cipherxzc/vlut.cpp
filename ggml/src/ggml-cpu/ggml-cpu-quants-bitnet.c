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

void ggml_gemm_i2_i8_t_LUT2_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                 const void *restrict vy, int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
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

void ggml_gemm_i2_i8_s_LUT_tile(int ith, int n, float *restrict s, size_t bs, const void *restrict vx,
                                const void *restrict vy, int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int *restrict sum_i32 = (int *)malloc(sizeof(int) * nr * nc);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int) * nr * nc);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int t = 0; t < nr; t += TABLE_ENTRY_SIZE) {
        const int16_t *restrict table = tables + t * n / 4 * 81;
        const int entry_len = MIN(nr - t, TABLE_ENTRY_SIZE);
        for (int g = 0; g < n; g += group_size) {
            int lim = g + group_size < n ? g + group_size : n;
            for (int i = (g >> 2); i < (lim >> 2); i++) {
#ifdef BITNET_DEBUG
                struct timespec start_LUT = get_thread_cpu_time();
#endif
            const int16_t *restrict this_table = table + i * 81 * TABLE_ENTRY_SIZE;
                for (int c = 0; c < nc; c++) {
                    int v = x[i * bs + c];
                    int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;
                    const int16_t *restrict rt = this_table + v * TABLE_ENTRY_SIZE;
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

    for (int r = 0; r < nr; r++) {
        const float scale = *((const float *)(y + r * (n + 4) + n));
        for (int c = 0; c < nc; c++) {
            s[r * bs + c] = sum_i32[r * nc + c] * scale;  // 将输出转置回来
        }
    }
#ifdef BITNET_DEBUG
    struct timespec end_scale = get_thread_cpu_time();
    scale_duration += get_time_diff(start_scale, end_scale);

    pthread_mutex_lock(&time_mutex);
    convert_time += convert_duration;
    scale_time += scale_duration;
    LUT_time += LUT_duration;
    pthread_mutex_unlock(&time_mutex);
#endif

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
