#define GGML_COMMON_IMPL_C
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>   // for GGML_ASSERT
#include <stdlib.h>  // for qsort
#include <string.h>

#include "ggml-quants-bitnet.h"

#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#define UNUSED GGML_UNUSED
#define eps 1e-6



size_t quantize_i2_s(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n_per_row; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 3; k >= 0; k--) {
                double v = (double)src[j * n_per_row + i * 4 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i2_weight[i * nrows + j] = w;
        }
    }

    return nrows * ggml_row_size(GGML_TYPE_I2_S, n_per_row);
}

size_t quantize_i2_s_4(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n_per_row; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 3; k >= 0; k--) {
                double v = (double)src[j * n_per_row + i * 4 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i2_weight[i / 4 * nrows * 4 + j * 4 + i % 4] = w;
        }
    }

    return nrows * ggml_row_size(GGML_TYPE_I2_S_4, n_per_row);
}

size_t quantize_i2_s_8(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                       const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n_per_row; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 3; k >= 0; k--) {
                double v = (double)src[j * n_per_row + i * 4 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i2_weight[i / 8 * nrows * 8 + j * 8 + i % 8] = w;
        }
    }

    return nrows * ggml_row_size(GGML_TYPE_I2_S_8, n_per_row);
}



size_t quantize_i1_m(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 1.58 bits per weight
    UNUSED(imatrix);

    int64_t blck_num = n_per_row / 20 * 4;
    int64_t blck_remain = n_per_row % 20 / 4;

    uint8_t *i1m_weight = (uint8_t *)dst;
    for (int i = 0; i < blck_num; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 4; k >= 0; k--) {
                double v = (double)src[j * n_per_row + i * 5 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i1m_weight[i * nrows + j] = w;
        }
    }

    for (int i = 0; i < blck_remain; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 3; k >= 0; k--) {
                double v = (double)src[j * n_per_row + blck_num * 5 + i * 4 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i1m_weight[(blck_num + i) * nrows + j] = w;
        }
    }

    return nrows * ggml_row_size(GGML_TYPE_I1_M, n_per_row);
}

size_t quantize_i1_m_2(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 1.58 bits per weight
    UNUSED(imatrix);

    int64_t blck_num = n_per_row / 20 * 4;
    int64_t blck_remain = n_per_row % 20 / 4;

    uint8_t *i1m_weight = (uint8_t *)dst;
    for (int i = 0; i < blck_num; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 4; k >= 0; k--) {
                double v = (double)src[j * n_per_row + i * 5 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i1m_weight[i / 2 * nrows * 2 + j * 2 + i % 2] = w;
        }
    }

    for (int i = 0; i < blck_remain; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 3; k >= 0; k--) {
                double v = (double)src[j * n_per_row + blck_num * 5 + i * 4 + k];
                uint8_t tmp = 1;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 2 : 0;
                }
                w = w * 3 + tmp;
            }
            i1m_weight[(blck_num + i) * nrows + j] = w;
        }
    }

    return nrows * ggml_row_size(GGML_TYPE_I1_M_2, n_per_row);
}
