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

#define BITNET_MAKE_TABLE2
#ifndef BITNET_MAKE_TABLE2
void gemm_make_table(int16_t *restrict table, const int8_t *restrict x, int nr) {
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
inline void add(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y, int nr) {
    for (int i = 0; i < nr; i++) {
        t1[i] = t2[i] + y[i];
    }
}

inline void sub(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y, int nr) {
    for (int i = 0; i < nr; i++) {
        t1[i] = t2[i] - y[i];
    }
}

void gemm_make_table(int16_t *restrict table, const int8_t *restrict y, int nr) {
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

inline static void gemm_look_up(const uint8_t *restrict x, const int16_t *restrict table, int16_t *restrict s, int n,
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

#ifdef BITNET_DEBUG
double total_time, make_table_time, convert_time, scale_time;
pthread_mutex_t time_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

void ggml_gemm_i2_i8_b_LUT(int n, float *restrict s, size_t bs, const void *restrict vx, const void *restrict vy,
                            int nr, int nc) {
    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);
    // assert(nr % 8 == 0);  // Ensure nr is multiple of 8 for NEON optimization

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
    clock_t start = clock();
    double make_table_duration = 0.0, convert_duration = 0.0, scale_duration = 0.0;
#endif

    for (int j = 0; j < n; j += group_size) {
        int lim = j + group_size < n ? j + group_size : n;
        for (int i = j; i < lim; i += 4) {
#ifdef BITNET_DEBUG
            clock_t start_make_table = clock();
#endif
            gemm_make_table(table, y + i * nr, nr);
#ifdef BITNET_DEBUG
            clock_t end_make_table = clock();
            make_table_duration += (double)(end_make_table - start_make_table) / CLOCKS_PER_SEC * 1000;
#endif

            const uint8_t *restrict nx = x + (i >> 2);

            gemm_look_up(nx, table, ss, n, nc, nr);
        }

#ifdef BITNET_DEBUG
        clock_t start_convert = clock();
#endif
        for (int i = 0; i < nc * nr; i++) {
            ss2[i] += ss[i];
        }
#ifdef BITNET_DEBUG
        clock_t end_convert = clock();
        convert_duration += (double)(end_convert - start_convert) / CLOCKS_PER_SEC * 1000;
#endif

        memset(ss, 0, sizeof(int16_t) * nr * nc);
    }

#ifdef BITNET_DEBUG
    clock_t start_scale = clock();
#endif
    const float *sc = (const float *)(y + nr * n);
    for (int c = 0; c < nc; c++) {
        for (int r = 0; r < nr; r++) {
            s[r * bs + c] = ss2[c * nr + r] * sc[r];  // 将输出转置回来
        }
    }
#ifdef BITNET_DEBUG
    clock_t end_scale = clock();
    scale_duration += (double)(end_scale - start_scale) / CLOCKS_PER_SEC * 1000;

    clock_t end = clock();
    double total_duration = (double)(end - start) / CLOCKS_PER_SEC * 1000;

    pthread_mutex_lock(&time_mutex);
    total_time += total_duration;
    make_table_time += make_table_duration;
    convert_time += convert_duration;
    scale_time += scale_duration;
    pthread_mutex_unlock(&time_mutex);
#endif

    free(table);
    free(ss);
    free(ss2);
}
