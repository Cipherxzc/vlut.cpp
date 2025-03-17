#define GGML_COMMON_IMPL_C

#define BITNET_AVX2
#ifdef BITNET_AVX2
#include <immintrin.h>
#endif

#define BITNET_DEBUG
#ifdef BITNET_DEBUG
#include <pthread.h>
#include <time.h>
#endif


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


#ifdef BITNET_AVX2
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

void quantize_row_i8_b_trans(const float *x, void *y, int64_t n, int64_t row_size, float *scale) {
    int8_t *dst = (int8_t *)y;

    const double eps = 1e-5;
    double max = eps;
    for (int i = 0; i < n; i++) {
        max = MAX(max, (double)x[i]);
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, eps);

    for (int i = 0, j = 0; i < n; i++, j += row_size) {
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

void ggml_vec_dot_i2_i8_b(int n, float *restrict s, size_t bs, const void *restrict vx, size_t bx,
                           const void *restrict vy, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const uint8_t *restrict x = vx;  // int2
    const int8_t *restrict y = vy;   // int8

    assert(n % 32 == 0);

    float sumf = 0;

#ifdef BITNET_AVX2
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        const __m256i qx = _mm256_set_epi32(  // 查表
            i2s_i8s[x[i / 4 + 7]], i2s_i8s[x[i / 4 + 6]], i2s_i8s[x[i / 4 + 5]], i2s_i8s[x[i / 4 + 4]],
            i2s_i8s[x[i / 4 + 3]], i2s_i8s[x[i / 4 + 2]], i2s_i8s[x[i / 4 + 1]], i2s_i8s[x[i / 4 + 0]]);
        const __m256i qy = _mm256_loadu_si256((const __m256i *)(y + i));

        const __m256i q = bitnet_mul(qx, qy);

        acc = _mm256_add_epi32(acc, q);
    }

    sumf = hsum_float_8(_mm256_cvtepi32_ps(acc));
#else
    for (int i = 0; i < n; i += 4) {
        const int8_t *weight = (const int8_t *)(i2s_i8s + x[i >> 2]);  // 查表

        sumf += (int)(y[i + 0] * weight[0]);
        sumf += (int)(y[i + 1] * weight[1]);
        sumf += (int)(y[i + 2] * weight[2]);
        sumf += (int)(y[i + 3] * weight[3]);
    }
#endif

    const float *sc = (const float *)(y + n);
    float scale = *sc;
    sumf *= scale;

    *s = sumf;
}

void ggml_vec_dot_i1_58_i8_b(int n, float *restrict s, size_t bs, const void *restrict vx, size_t bx,
                             const void *restrict vy, size_t by, int nrc) {
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    const uint8_t *restrict x = vx;  // int1.58
    const int8_t *restrict y = vy;   // int8

    assert(n % 5 == 0);

    float sumf = 0;

    for (int i = 0; i < n; i += 5) {
        int weight = x[i / 5];
        for (int j = 0; j < 5; j++){
            int v = weight % 3 - 1;
            sumf += (int)(y[i + j] * v);
            weight /= 3;
        }
    }

    const float *sc = (const float *)(y + n);
    float scale = *sc;
    sumf *= scale;

    *s = sumf;
}


inline static void gemm_make_table_I2(int16_t *restrict table, const int8_t *restrict y, int nr);
inline static void gemm_make_table_I1_58(int16_t *restrict table, const int8_t *restrict y, int nr);
inline static void gemm_look_up_I2(const uint8_t *restrict x, const int16_t *restrict table, int16_t *restrict s, int n, int nc, int nr);
inline static void gemm_look_up_I1_58(const uint8_t *restrict x, const int16_t *restrict table, int16_t *restrict s, int n, int nc, int nr);


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


void ggml_gemm_i2_i8_b_make_table(const int8_t *restrict y, int nrows, int n, int16_t *restrict table){
    for (int i = 0; i < nrows; i += 4){
        gemm_make_table_I2(table + i * 64 * n, y + i * n, n);
    }
}

void ggml_gemm_i1_58_i8_b_make_table(const int8_t *restrict y, int nrows, int n, int16_t *restrict table) {
    for (int i = 0; i < nrows / 5; i++) {
        gemm_make_table_I1_58(table + i * 243 * n, y + i * 5 * n, n);
    }
}

void ggml_gemm_i2_i8_b_LUT(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                           int nr, int nc, const int16_t *restrict table) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
#ifdef BITNET_DEBUG
        struct timespec start_LUT = get_thread_cpu_time();
#endif
        for (int i = (j >> 2); i < (lim >> 2); i++) {
            const uint8_t *restrict nx = x + i;

            gemm_look_up_I2(nx, table + i * 256 * nr, ss, n, nc, nr);
        }
#ifdef BITNET_DEBUG
        struct timespec end_LUT = get_thread_cpu_time();
        LUT_duration += get_time_diff(start_LUT, end_LUT);

        struct timespec start_convert = get_thread_cpu_time();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif

        memset(ss, 0, sizeof(int16_t) * nr * nc);
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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

    free(ss);
    free(ss2);
}

void ggml_gemm_i1_58_i8_b_LUT(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                              int nr, int nc, const int16_t *restrict table) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 5 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);

    static const int group_size = 640;

#ifdef BITNET_DEBUG
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
#ifdef BITNET_DEBUG
        struct timespec start_LUT = get_thread_cpu_time();
#endif
        for (int i = j / 5; i < lim / 5; i++) {
            const uint8_t *restrict nx = x + i;

            gemm_look_up_I1_58(nx, table + i * 243 * nr, ss, n, nc, nr);
        }

#ifdef BITNET_DEBUG
        struct timespec end_LUT = get_thread_cpu_time();
        LUT_duration += get_time_diff(start_LUT, end_LUT);

        struct timespec start_convert = get_thread_cpu_time();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * nr * nc);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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

    free(ss);
    free(ss2);
}

void ggml_gemm_i2_i8_t_LUT(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                           int nr, int nc, const int16_t *restrict table) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
#ifdef BITNET_DEBUG
        struct timespec start_LUT = get_thread_cpu_time();
#endif
        for (int i = (j >> 2); i < (lim >> 2); i++) {
            const int16_t *restrict this_table = table + i * 256 * nr;
            for (int c = 0; c < nc; c++){
                int v = x[i * bs + c];
                int16_t *restrict rs = ss + c * nr;
                const int16_t *restrict rt = this_table + v * nr;
                for (int r = 0; r < nr; r++) {
                    rs[r] += rt[r];
                }
            }
        }
#ifdef BITNET_DEBUG
        struct timespec end_LUT = get_thread_cpu_time();
        LUT_duration += get_time_diff(start_LUT, end_LUT);

        struct timespec start_convert = get_thread_cpu_time();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * nr * nc);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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

    free(ss);
    free(ss2);
}

void ggml_gemm_i2_i8_b_LUT2(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                           int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * nr) << 8);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);
    memset(table, 0, sizeof(int16_t) * nr);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
        for (int i = j; i < lim; i += 4) {
#ifdef BITNET_DEBUG
            struct timespec start_make_table = get_thread_cpu_time();
#endif
            gemm_make_table_I2(table, y + i * nr, nr);
#ifdef BITNET_DEBUG
            struct timespec end_make_table = get_thread_cpu_time();
            make_table_duration += get_time_diff(start_make_table, end_make_table);

            struct timespec start_LUT = get_thread_cpu_time();
#endif
            const uint8_t *restrict nx = x + (i >> 2);
            gemm_look_up_I2(nx, table, ss, n, nc, nr);
#ifdef BITNET_DEBUG
            struct timespec end_LUT = get_thread_cpu_time();
            LUT_duration += get_time_diff(start_LUT, end_LUT);
#endif
        }

#ifdef BITNET_DEBUG
        struct timespec start_convert = get_thread_cpu_time();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * nr * nc);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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
    free(ss);
    free(ss2);
}

void ggml_gemm_i1_58_i8_b_LUT2(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                            int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 5 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * nr) * 243);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);
    memset(table + 121 * nr, 0, sizeof(int16_t) * nr);

    static const int group_size = 640;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
        for (int i = j; i < lim; i += 5) {
#ifdef BITNET_DEBUG
            struct timespec start_make_table = get_thread_cpu_time();
#endif
            gemm_make_table_I1_58(table, y + i * nr, nr);
#ifdef BITNET_DEBUG
            struct timespec end_make_table = get_thread_cpu_time();
            make_table_duration += get_time_diff(start_make_table, end_make_table);

            struct timespec start_LUT = get_thread_cpu_time();
#endif
            const uint8_t *restrict nx = x + (i / 5);
            gemm_look_up_I1_58(nx, table, ss, n, nc, nr);
#ifdef BITNET_DEBUG
            struct timespec end_LUT = get_thread_cpu_time();
            LUT_duration += get_time_diff(start_LUT, end_LUT);
#endif
        }

#ifdef BITNET_DEBUG
        struct timespec start_convert = get_thread_cpu_time();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * nr * nc);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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
    free(ss);
    free(ss2);
}

void ggml_gemm_i2_i8_t_LUT2(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                            int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * nc);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * nc);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * nr) << 8);

    memset(ss, 0, sizeof(int16_t) * nr * nc);
    memset(ss2, 0, sizeof(int) * nr * nc);
    memset(table, 0, sizeof(int16_t) * nr);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
        for (int i = (j >> 2); i < (lim >> 2); i++) {
#ifdef BITNET_DEBUG
            struct timespec start_make_table = get_thread_cpu_time();
#endif
            gemm_make_table_I2(table, y + i * 4 * nr, nr);
#ifdef BITNET_DEBUG
            struct timespec end_make_table = get_thread_cpu_time();
            make_table_duration += get_time_diff(start_make_table, end_make_table);

            struct timespec start_LUT = get_thread_cpu_time();
#endif
            for (int c = 0; c < nc; c++) {
                int v = x[i * bs + c];
                int16_t *restrict rs = ss + c * nr;
                const int16_t *restrict rt = table + v * nr;
                for (int r = 0; r < nr; r++) {
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
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * nr * nc);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
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
    free(ss);
    free(ss2);
}

pthread_mutex_t LUT3_time_mutex = PTHREAD_MUTEX_INITIALIZER;

void ggml_gemm_i2_i8_t_LUT3(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                            int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    const uint8_t *restrict x = vx;
    const int8_t *restrict y = vy;

    int16_t *restrict ss = (int16_t *)malloc(sizeof(int16_t) * nr * bs);
    int *restrict ss2 = (int *)malloc(sizeof(int) * nr * bs);
    int16_t *restrict table = (int16_t *)malloc((sizeof(int16_t) * nr) << 8);

    memset(ss, 0, sizeof(int16_t) * nr * bs);
    memset(ss2, 0, sizeof(int) * nr * bs);
    memset(table, 0, sizeof(int16_t) * nr);

    static const int group_size = 512;

#ifdef BITNET_DEBUG
    double make_table_duration = 0.0;
    double convert_duration = 0.0;
    double scale_duration = 0.0;
    double LUT_duration = 0.0;
#endif

    for (int j = 0; j < nc; j += group_size) {
        int lim = j + group_size < nc ? j + group_size : nc;
        for (int i = (j >> 2); i < (lim >> 2); i++) {
#ifdef BITNET_DEBUG
            struct timespec start_make_table = get_thread_cpu_time();
#endif
            gemm_make_table_I2(table, y + i * 4 * nr, nr);
#ifdef BITNET_DEBUG
            struct timespec end_make_table = get_thread_cpu_time();
            make_table_duration += get_time_diff(start_make_table, end_make_table);

            struct timespec start_LUT = get_thread_cpu_time();
#endif
            for (int c = 0; c < bs; c++) {
                int v = x[i * bs + c];
                int16_t *restrict rs = ss + c * nr;
                const int16_t *restrict rt = table + v * nr;
                for (int r = 0; r < nr; r++) {
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
        for (int i = 0; i < bs * nr; i++) {
            ss2[i] += ss[i];
        }
        memset(ss, 0, sizeof(int16_t) * bs * nr);
#ifdef BITNET_DEBUG
        struct timespec end_convert = get_thread_cpu_time();
        convert_duration += get_time_diff(start_convert, end_convert);
#endif
    }

#ifdef BITNET_DEBUG
    struct timespec start_scale = get_thread_cpu_time();
#endif
    pthread_mutex_lock(&LUT3_time_mutex);
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < bs; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] += ss2[c * nr + r] * sc[r];  // 将输出转置回来
        }
    }
    pthread_mutex_unlock(&LUT3_time_mutex);
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
    free(ss);
    free(ss2);
}

inline static void gemm_look_up_I2(const uint8_t *restrict x, const int16_t *restrict table, int16_t *restrict s, int n,
                                   int nc, int nr) {
    for (int c = 0; c < nc; c++) {
        int v = x[c * n >> 2];
        int16_t *restrict rs = s + c * nr;
        const int16_t *restrict rt = table + v * nr;
        for (int r = 0; r < nr; r++) {
            rs[r] += rt[r];
        }
    }
}

inline static void gemm_look_up_I1_58(const uint8_t *restrict x, const int16_t *restrict table, int16_t *restrict s,
                                      int n, int nc, int nr) {
    for (int c = 0; c < nc; c++) {
        int v = x[c * n / 5];
        int16_t *restrict rs = s + c * nr;
        const int16_t *restrict rt = table + v * nr;
        for (int r = 0; r < nr; r++) {
            rs[r] += rt[r];
        }
    }
}

inline static void add(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y, int nr) {
    for (int i = 0; i < nr; i++) {
        t1[i] = t2[i] + y[i];
    }
}

inline static void sub(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y, int nr) {
    for (int i = 0; i < nr; i++) {
        t1[i] = t2[i] - y[i];
    }
}

#define BITNET_MAKE_TABLE2
#ifndef BITNET_MAKE_TABLE2
void gemm_make_table_I2(int16_t *restrict table, const int8_t *restrict x, int nr) {
    int16_t *r0 = (int16_t *)malloc(sizeof(int16_t) * nr);
    int16_t *r1 = (int16_t *)malloc(sizeof(int16_t) * nr);
    int16_t *r2 = (int16_t *)malloc(sizeof(int16_t) * nr);
    // memset(table, 0, sizeof(int16_t) * nr);  // 初始化第 0 项
    for (int i0 = 0; i0 < 4; i0++) {
        if (i0 == 2) {
            continue;
        }
        switch (i0) {
            case 0:
                memset(r0, 0, sizeof(int16_t) * nr);
                break;
            case 1:
                for (int i = 0; i < nr; i++) {
                    r0[i] = x[i];
                }
                break;
            case 2:
                continue;
            case 3:
                for (int i = 0; i < nr; i++) {
                    r0[i] = -x[i];
                }
                break;
        }
        for (int i1 = 0; i1 < 4; i1++) {
            if (i1 == 2) {
                continue;
            }
            int p1 = i0 | (i1 << 2);
            const int8_t *restrict nx1 = x + nr;

            switch (i1) {
                case 0:
                    for (int i = 0; i < nr; i++) {
                        r1[i] = r0[i];
                    }
                    break;
                case 1:
                    for (int i = 0; i < nr; i++) {
                        r1[i] = r0[i] + nx1[i];
                    }
                    break;
                case 2:
                    continue;
                case 3:
                    for (int i = 0; i < nr; i++) {
                        r1[i] = r0[i] - nx1[i];
                    }
                    break;
            }
            for (int i2 = 0; i2 < 4; i2++) {
                if (i2 == 2) {
                    continue;
                }
                int p2 = p1 | (i2 << 4);
                const int8_t *restrict nx2 = nx1 + nr;

                switch (i2) {
                    case 0:
                        for (int i = 0; i < nr; i++) {
                            r2[i] = r1[i];
                        }
                        break;
                    case 1:
                        for (int i = 0; i < nr; i++) {
                            r2[i] = r1[i] + nx2[i];
                        }
                        break;
                    case 2:
                        continue;
                    case 3:
                        for (int i = 0; i < nr; i++) {
                            r2[i] = r1[i] - nx2[i];
                        }
                        break;
                }
                for (int i3 = 0; i3 < 4; i3++) {
                    if (i3 == 2) {
                        continue;
                    }
                    int p3 = p2 | (i3 << 6);
                    const int8_t *restrict nx3 = nx2 + nr;
                    int16_t *restrict nt = table + p3 * nr;

                    switch (i3) {
                        case 0:
                            for (int i = 0; i < nr; i++) {
                                nt[i] = r2[i];
                            }
                            break;
                        case 1:
                            for (int i = 0; i < nr; i++) {
                                nt[i] = r2[i] + nx3[i];
                            }
                            break;
                        case 2:
                            continue;
                        case 3:
                            for (int i = 0; i < nr; i++) {
                                nt[i] = r2[i] - nx3[i];
                            }
                            break;
                    }
                }
            }
        }
    }
    free(r0);
    free(r1);
    free(r2);
}
#else
void gemm_make_table_I2(int16_t *restrict table, const int8_t *restrict y, int nr) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + nr;
    const int8_t *restrict y2 = y1 + nr;
    const int8_t *restrict y3 = y2 + nr;

    add(table + 1 * nr, table + 0 * nr, y0, nr);
    sub(table + 3 * nr, table + 0 * nr, y0, nr);
    add(table + 4 * nr, table + 0 * nr, y1, nr);
    add(table + 5 * nr, table + 4 * nr, y0, nr);
    sub(table + 7 * nr, table + 4 * nr, y0, nr);
    sub(table + 12 * nr, table + 0 * nr, y1, nr);
    add(table + 13 * nr, table + 12 * nr, y0, nr);
    sub(table + 15 * nr, table + 12 * nr, y0, nr);
    add(table + 16 * nr, table + 0 * nr, y2, nr);
    add(table + 17 * nr, table + 16 * nr, y0, nr);
    sub(table + 19 * nr, table + 16 * nr, y0, nr);
    add(table + 20 * nr, table + 16 * nr, y1, nr);
    add(table + 21 * nr, table + 20 * nr, y0, nr);
    sub(table + 23 * nr, table + 20 * nr, y0, nr);
    sub(table + 28 * nr, table + 16 * nr, y1, nr);
    add(table + 29 * nr, table + 28 * nr, y0, nr);
    sub(table + 31 * nr, table + 28 * nr, y0, nr);
    sub(table + 48 * nr, table + 0 * nr, y2, nr);
    add(table + 49 * nr, table + 48 * nr, y0, nr);
    sub(table + 51 * nr, table + 48 * nr, y0, nr);
    add(table + 52 * nr, table + 48 * nr, y1, nr);
    add(table + 53 * nr, table + 52 * nr, y0, nr);
    sub(table + 55 * nr, table + 52 * nr, y0, nr);
    sub(table + 60 * nr, table + 48 * nr, y1, nr);
    add(table + 61 * nr, table + 60 * nr, y0, nr);
    sub(table + 63 * nr, table + 60 * nr, y0, nr);
    add(table + 64 * nr, table + 0 * nr, y3, nr);
    add(table + 65 * nr, table + 64 * nr, y0, nr);
    sub(table + 67 * nr, table + 64 * nr, y0, nr);
    add(table + 68 * nr, table + 64 * nr, y1, nr);
    add(table + 69 * nr, table + 68 * nr, y0, nr);
    sub(table + 71 * nr, table + 68 * nr, y0, nr);
    sub(table + 76 * nr, table + 64 * nr, y1, nr);
    add(table + 77 * nr, table + 76 * nr, y0, nr);
    sub(table + 79 * nr, table + 76 * nr, y0, nr);
    add(table + 80 * nr, table + 64 * nr, y2, nr);
    add(table + 81 * nr, table + 80 * nr, y0, nr);
    sub(table + 83 * nr, table + 80 * nr, y0, nr);
    add(table + 84 * nr, table + 80 * nr, y1, nr);
    add(table + 85 * nr, table + 84 * nr, y0, nr);
    sub(table + 87 * nr, table + 84 * nr, y0, nr);
    sub(table + 92 * nr, table + 80 * nr, y1, nr);
    add(table + 93 * nr, table + 92 * nr, y0, nr);
    sub(table + 95 * nr, table + 92 * nr, y0, nr);
    sub(table + 112 * nr, table + 64 * nr, y2, nr);
    add(table + 113 * nr, table + 112 * nr, y0, nr);
    sub(table + 115 * nr, table + 112 * nr, y0, nr);
    add(table + 116 * nr, table + 112 * nr, y1, nr);
    add(table + 117 * nr, table + 116 * nr, y0, nr);
    sub(table + 119 * nr, table + 116 * nr, y0, nr);
    sub(table + 124 * nr, table + 112 * nr, y1, nr);
    add(table + 125 * nr, table + 124 * nr, y0, nr);
    sub(table + 127 * nr, table + 124 * nr, y0, nr);
    sub(table + 192 * nr, table + 0 * nr, y3, nr);
    add(table + 193 * nr, table + 192 * nr, y0, nr);
    sub(table + 195 * nr, table + 192 * nr, y0, nr);
    add(table + 196 * nr, table + 192 * nr, y1, nr);
    add(table + 197 * nr, table + 196 * nr, y0, nr);
    sub(table + 199 * nr, table + 196 * nr, y0, nr);
    sub(table + 204 * nr, table + 192 * nr, y1, nr);
    add(table + 205 * nr, table + 204 * nr, y0, nr);
    sub(table + 207 * nr, table + 204 * nr, y0, nr);
    add(table + 208 * nr, table + 192 * nr, y2, nr);
    add(table + 209 * nr, table + 208 * nr, y0, nr);
    sub(table + 211 * nr, table + 208 * nr, y0, nr);
    add(table + 212 * nr, table + 208 * nr, y1, nr);
    add(table + 213 * nr, table + 212 * nr, y0, nr);
    sub(table + 215 * nr, table + 212 * nr, y0, nr);
    sub(table + 220 * nr, table + 208 * nr, y1, nr);
    add(table + 221 * nr, table + 220 * nr, y0, nr);
    sub(table + 223 * nr, table + 220 * nr, y0, nr);
    sub(table + 240 * nr, table + 192 * nr, y2, nr);
    add(table + 241 * nr, table + 240 * nr, y0, nr);
    sub(table + 243 * nr, table + 240 * nr, y0, nr);
    add(table + 244 * nr, table + 240 * nr, y1, nr);
    add(table + 245 * nr, table + 244 * nr, y0, nr);
    sub(table + 247 * nr, table + 244 * nr, y0, nr);
    sub(table + 252 * nr, table + 240 * nr, y1, nr);
    add(table + 253 * nr, table + 252 * nr, y0, nr);
    sub(table + 255 * nr, table + 252 * nr, y0, nr);
}
#endif

void gemm_make_table_I1_58(int16_t *restrict table, const int8_t *restrict y, int nr) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + nr;
    const int8_t *restrict y2 = y1 + nr;
    const int8_t *restrict y3 = y2 + nr;
    const int8_t *restrict y4 = y3 + nr;

    sub(table + 40 * nr, table + 121 * nr, y4, nr);
    sub(table + 94 * nr, table + 121 * nr, y3, nr);
    sub(table + 112 * nr, table + 121 * nr, y2, nr);
    sub(table + 118 * nr, table + 121 * nr, y1, nr);
    sub(table + 120 * nr, table + 121 * nr, y0, nr);
    add(table + 122 * nr, table + 121 * nr, y0, nr);
    add(table + 124 * nr, table + 121 * nr, y1, nr);
    add(table + 130 * nr, table + 121 * nr, y2, nr);
    add(table + 148 * nr, table + 121 * nr, y3, nr);
    add(table + 202 * nr, table + 121 * nr, y4, nr);
    sub(table + 13 * nr, table + 40 * nr, y3, nr);
    sub(table + 31 * nr, table + 40 * nr, y2, nr);
    sub(table + 37 * nr, table + 40 * nr, y1, nr);
    sub(table + 39 * nr, table + 40 * nr, y0, nr);
    add(table + 41 * nr, table + 40 * nr, y0, nr);
    add(table + 43 * nr, table + 40 * nr, y1, nr);
    add(table + 49 * nr, table + 40 * nr, y2, nr);
    add(table + 67 * nr, table + 40 * nr, y3, nr);
    sub(table + 85 * nr, table + 94 * nr, y2, nr);
    sub(table + 91 * nr, table + 94 * nr, y1, nr);
    sub(table + 93 * nr, table + 94 * nr, y0, nr);
    add(table + 95 * nr, table + 94 * nr, y0, nr);
    add(table + 97 * nr, table + 94 * nr, y1, nr);
    add(table + 103 * nr, table + 94 * nr, y2, nr);
    sub(table + 109 * nr, table + 112 * nr, y1, nr);
    sub(table + 111 * nr, table + 112 * nr, y0, nr);
    add(table + 113 * nr, table + 112 * nr, y0, nr);
    add(table + 115 * nr, table + 112 * nr, y1, nr);
    sub(table + 117 * nr, table + 118 * nr, y0, nr);
    add(table + 119 * nr, table + 118 * nr, y0, nr);
    sub(table + 123 * nr, table + 124 * nr, y0, nr);
    add(table + 125 * nr, table + 124 * nr, y0, nr);
    sub(table + 127 * nr, table + 130 * nr, y1, nr);
    sub(table + 129 * nr, table + 130 * nr, y0, nr);
    add(table + 131 * nr, table + 130 * nr, y0, nr);
    add(table + 133 * nr, table + 130 * nr, y1, nr);
    sub(table + 139 * nr, table + 148 * nr, y2, nr);
    sub(table + 145 * nr, table + 148 * nr, y1, nr);
    sub(table + 147 * nr, table + 148 * nr, y0, nr);
    add(table + 149 * nr, table + 148 * nr, y0, nr);
    add(table + 151 * nr, table + 148 * nr, y1, nr);
    add(table + 157 * nr, table + 148 * nr, y2, nr);
    sub(table + 175 * nr, table + 202 * nr, y3, nr);
    sub(table + 193 * nr, table + 202 * nr, y2, nr);
    sub(table + 199 * nr, table + 202 * nr, y1, nr);
    sub(table + 201 * nr, table + 202 * nr, y0, nr);
    add(table + 203 * nr, table + 202 * nr, y0, nr);
    add(table + 205 * nr, table + 202 * nr, y1, nr);
    add(table + 211 * nr, table + 202 * nr, y2, nr);
    add(table + 229 * nr, table + 202 * nr, y3, nr);
    sub(table + 4 * nr, table + 13 * nr, y2, nr);
    sub(table + 10 * nr, table + 13 * nr, y1, nr);
    sub(table + 12 * nr, table + 13 * nr, y0, nr);
    add(table + 14 * nr, table + 13 * nr, y0, nr);
    add(table + 16 * nr, table + 13 * nr, y1, nr);
    add(table + 22 * nr, table + 13 * nr, y2, nr);
    sub(table + 28 * nr, table + 31 * nr, y1, nr);
    sub(table + 30 * nr, table + 31 * nr, y0, nr);
    add(table + 32 * nr, table + 31 * nr, y0, nr);
    add(table + 34 * nr, table + 31 * nr, y1, nr);
    sub(table + 36 * nr, table + 37 * nr, y0, nr);
    add(table + 38 * nr, table + 37 * nr, y0, nr);
    sub(table + 42 * nr, table + 43 * nr, y0, nr);
    add(table + 44 * nr, table + 43 * nr, y0, nr);
    sub(table + 46 * nr, table + 49 * nr, y1, nr);
    sub(table + 48 * nr, table + 49 * nr, y0, nr);
    add(table + 50 * nr, table + 49 * nr, y0, nr);
    add(table + 52 * nr, table + 49 * nr, y1, nr);
    sub(table + 58 * nr, table + 67 * nr, y2, nr);
    sub(table + 64 * nr, table + 67 * nr, y1, nr);
    sub(table + 66 * nr, table + 67 * nr, y0, nr);
    add(table + 68 * nr, table + 67 * nr, y0, nr);
    add(table + 70 * nr, table + 67 * nr, y1, nr);
    add(table + 76 * nr, table + 67 * nr, y2, nr);
    sub(table + 82 * nr, table + 85 * nr, y1, nr);
    sub(table + 84 * nr, table + 85 * nr, y0, nr);
    add(table + 86 * nr, table + 85 * nr, y0, nr);
    add(table + 88 * nr, table + 85 * nr, y1, nr);
    sub(table + 90 * nr, table + 91 * nr, y0, nr);
    add(table + 92 * nr, table + 91 * nr, y0, nr);
    sub(table + 96 * nr, table + 97 * nr, y0, nr);
    add(table + 98 * nr, table + 97 * nr, y0, nr);
    sub(table + 100 * nr, table + 103 * nr, y1, nr);
    sub(table + 102 * nr, table + 103 * nr, y0, nr);
    add(table + 104 * nr, table + 103 * nr, y0, nr);
    add(table + 106 * nr, table + 103 * nr, y1, nr);
    sub(table + 108 * nr, table + 109 * nr, y0, nr);
    add(table + 110 * nr, table + 109 * nr, y0, nr);
    sub(table + 114 * nr, table + 115 * nr, y0, nr);
    add(table + 116 * nr, table + 115 * nr, y0, nr);
    sub(table + 126 * nr, table + 127 * nr, y0, nr);
    add(table + 128 * nr, table + 127 * nr, y0, nr);
    sub(table + 132 * nr, table + 133 * nr, y0, nr);
    add(table + 134 * nr, table + 133 * nr, y0, nr);
    sub(table + 136 * nr, table + 139 * nr, y1, nr);
    sub(table + 138 * nr, table + 139 * nr, y0, nr);
    add(table + 140 * nr, table + 139 * nr, y0, nr);
    add(table + 142 * nr, table + 139 * nr, y1, nr);
    sub(table + 144 * nr, table + 145 * nr, y0, nr);
    add(table + 146 * nr, table + 145 * nr, y0, nr);
    sub(table + 150 * nr, table + 151 * nr, y0, nr);
    add(table + 152 * nr, table + 151 * nr, y0, nr);
    sub(table + 154 * nr, table + 157 * nr, y1, nr);
    sub(table + 156 * nr, table + 157 * nr, y0, nr);
    add(table + 158 * nr, table + 157 * nr, y0, nr);
    add(table + 160 * nr, table + 157 * nr, y1, nr);
    sub(table + 166 * nr, table + 175 * nr, y2, nr);
    sub(table + 172 * nr, table + 175 * nr, y1, nr);
    sub(table + 174 * nr, table + 175 * nr, y0, nr);
    add(table + 176 * nr, table + 175 * nr, y0, nr);
    add(table + 178 * nr, table + 175 * nr, y1, nr);
    add(table + 184 * nr, table + 175 * nr, y2, nr);
    sub(table + 190 * nr, table + 193 * nr, y1, nr);
    sub(table + 192 * nr, table + 193 * nr, y0, nr);
    add(table + 194 * nr, table + 193 * nr, y0, nr);
    add(table + 196 * nr, table + 193 * nr, y1, nr);
    sub(table + 198 * nr, table + 199 * nr, y0, nr);
    add(table + 200 * nr, table + 199 * nr, y0, nr);
    sub(table + 204 * nr, table + 205 * nr, y0, nr);
    add(table + 206 * nr, table + 205 * nr, y0, nr);
    sub(table + 208 * nr, table + 211 * nr, y1, nr);
    sub(table + 210 * nr, table + 211 * nr, y0, nr);
    add(table + 212 * nr, table + 211 * nr, y0, nr);
    add(table + 214 * nr, table + 211 * nr, y1, nr);
    sub(table + 220 * nr, table + 229 * nr, y2, nr);
    sub(table + 226 * nr, table + 229 * nr, y1, nr);
    sub(table + 228 * nr, table + 229 * nr, y0, nr);
    add(table + 230 * nr, table + 229 * nr, y0, nr);
    add(table + 232 * nr, table + 229 * nr, y1, nr);
    add(table + 238 * nr, table + 229 * nr, y2, nr);
    sub(table + 1 * nr, table + 4 * nr, y1, nr);
    sub(table + 3 * nr, table + 4 * nr, y0, nr);
    add(table + 5 * nr, table + 4 * nr, y0, nr);
    add(table + 7 * nr, table + 4 * nr, y1, nr);
    sub(table + 9 * nr, table + 10 * nr, y0, nr);
    add(table + 11 * nr, table + 10 * nr, y0, nr);
    sub(table + 15 * nr, table + 16 * nr, y0, nr);
    add(table + 17 * nr, table + 16 * nr, y0, nr);
    sub(table + 19 * nr, table + 22 * nr, y1, nr);
    sub(table + 21 * nr, table + 22 * nr, y0, nr);
    add(table + 23 * nr, table + 22 * nr, y0, nr);
    add(table + 25 * nr, table + 22 * nr, y1, nr);
    sub(table + 27 * nr, table + 28 * nr, y0, nr);
    add(table + 29 * nr, table + 28 * nr, y0, nr);
    sub(table + 33 * nr, table + 34 * nr, y0, nr);
    add(table + 35 * nr, table + 34 * nr, y0, nr);
    sub(table + 45 * nr, table + 46 * nr, y0, nr);
    add(table + 47 * nr, table + 46 * nr, y0, nr);
    sub(table + 51 * nr, table + 52 * nr, y0, nr);
    add(table + 53 * nr, table + 52 * nr, y0, nr);
    sub(table + 55 * nr, table + 58 * nr, y1, nr);
    sub(table + 57 * nr, table + 58 * nr, y0, nr);
    add(table + 59 * nr, table + 58 * nr, y0, nr);
    add(table + 61 * nr, table + 58 * nr, y1, nr);
    sub(table + 63 * nr, table + 64 * nr, y0, nr);
    add(table + 65 * nr, table + 64 * nr, y0, nr);
    sub(table + 69 * nr, table + 70 * nr, y0, nr);
    add(table + 71 * nr, table + 70 * nr, y0, nr);
    sub(table + 73 * nr, table + 76 * nr, y1, nr);
    sub(table + 75 * nr, table + 76 * nr, y0, nr);
    add(table + 77 * nr, table + 76 * nr, y0, nr);
    add(table + 79 * nr, table + 76 * nr, y1, nr);
    sub(table + 81 * nr, table + 82 * nr, y0, nr);
    add(table + 83 * nr, table + 82 * nr, y0, nr);
    sub(table + 87 * nr, table + 88 * nr, y0, nr);
    add(table + 89 * nr, table + 88 * nr, y0, nr);
    sub(table + 99 * nr, table + 100 * nr, y0, nr);
    add(table + 101 * nr, table + 100 * nr, y0, nr);
    sub(table + 105 * nr, table + 106 * nr, y0, nr);
    add(table + 107 * nr, table + 106 * nr, y0, nr);
    sub(table + 135 * nr, table + 136 * nr, y0, nr);
    add(table + 137 * nr, table + 136 * nr, y0, nr);
    sub(table + 141 * nr, table + 142 * nr, y0, nr);
    add(table + 143 * nr, table + 142 * nr, y0, nr);
    sub(table + 153 * nr, table + 154 * nr, y0, nr);
    add(table + 155 * nr, table + 154 * nr, y0, nr);
    sub(table + 159 * nr, table + 160 * nr, y0, nr);
    add(table + 161 * nr, table + 160 * nr, y0, nr);
    sub(table + 163 * nr, table + 166 * nr, y1, nr);
    sub(table + 165 * nr, table + 166 * nr, y0, nr);
    add(table + 167 * nr, table + 166 * nr, y0, nr);
    add(table + 169 * nr, table + 166 * nr, y1, nr);
    sub(table + 171 * nr, table + 172 * nr, y0, nr);
    add(table + 173 * nr, table + 172 * nr, y0, nr);
    sub(table + 177 * nr, table + 178 * nr, y0, nr);
    add(table + 179 * nr, table + 178 * nr, y0, nr);
    sub(table + 181 * nr, table + 184 * nr, y1, nr);
    sub(table + 183 * nr, table + 184 * nr, y0, nr);
    add(table + 185 * nr, table + 184 * nr, y0, nr);
    add(table + 187 * nr, table + 184 * nr, y1, nr);
    sub(table + 189 * nr, table + 190 * nr, y0, nr);
    add(table + 191 * nr, table + 190 * nr, y0, nr);
    sub(table + 195 * nr, table + 196 * nr, y0, nr);
    add(table + 197 * nr, table + 196 * nr, y0, nr);
    sub(table + 207 * nr, table + 208 * nr, y0, nr);
    add(table + 209 * nr, table + 208 * nr, y0, nr);
    sub(table + 213 * nr, table + 214 * nr, y0, nr);
    add(table + 215 * nr, table + 214 * nr, y0, nr);
    sub(table + 217 * nr, table + 220 * nr, y1, nr);
    sub(table + 219 * nr, table + 220 * nr, y0, nr);
    add(table + 221 * nr, table + 220 * nr, y0, nr);
    add(table + 223 * nr, table + 220 * nr, y1, nr);
    sub(table + 225 * nr, table + 226 * nr, y0, nr);
    add(table + 227 * nr, table + 226 * nr, y0, nr);
    sub(table + 231 * nr, table + 232 * nr, y0, nr);
    add(table + 233 * nr, table + 232 * nr, y0, nr);
    sub(table + 235 * nr, table + 238 * nr, y1, nr);
    sub(table + 237 * nr, table + 238 * nr, y0, nr);
    add(table + 239 * nr, table + 238 * nr, y0, nr);
    add(table + 241 * nr, table + 238 * nr, y1, nr);
    sub(table + 0 * nr, table + 1 * nr, y0, nr);
    add(table + 2 * nr, table + 1 * nr, y0, nr);
    sub(table + 6 * nr, table + 7 * nr, y0, nr);
    add(table + 8 * nr, table + 7 * nr, y0, nr);
    sub(table + 18 * nr, table + 19 * nr, y0, nr);
    add(table + 20 * nr, table + 19 * nr, y0, nr);
    sub(table + 24 * nr, table + 25 * nr, y0, nr);
    add(table + 26 * nr, table + 25 * nr, y0, nr);
    sub(table + 54 * nr, table + 55 * nr, y0, nr);
    add(table + 56 * nr, table + 55 * nr, y0, nr);
    sub(table + 60 * nr, table + 61 * nr, y0, nr);
    add(table + 62 * nr, table + 61 * nr, y0, nr);
    sub(table + 72 * nr, table + 73 * nr, y0, nr);
    add(table + 74 * nr, table + 73 * nr, y0, nr);
    sub(table + 78 * nr, table + 79 * nr, y0, nr);
    add(table + 80 * nr, table + 79 * nr, y0, nr);
    sub(table + 162 * nr, table + 163 * nr, y0, nr);
    add(table + 164 * nr, table + 163 * nr, y0, nr);
    sub(table + 168 * nr, table + 169 * nr, y0, nr);
    add(table + 170 * nr, table + 169 * nr, y0, nr);
    sub(table + 180 * nr, table + 181 * nr, y0, nr);
    add(table + 182 * nr, table + 181 * nr, y0, nr);
    sub(table + 186 * nr, table + 187 * nr, y0, nr);
    add(table + 188 * nr, table + 187 * nr, y0, nr);
    sub(table + 216 * nr, table + 217 * nr, y0, nr);
    add(table + 218 * nr, table + 217 * nr, y0, nr);
    sub(table + 222 * nr, table + 223 * nr, y0, nr);
    add(table + 224 * nr, table + 223 * nr, y0, nr);
    sub(table + 234 * nr, table + 235 * nr, y0, nr);
    add(table + 236 * nr, table + 235 * nr, y0, nr);
    sub(table + 240 * nr, table + 241 * nr, y0, nr);
    add(table + 242 * nr, table + 241 * nr, y0, nr);
}
