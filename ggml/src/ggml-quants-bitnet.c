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


size_t quantize_i2_b(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    size_t row_size = ggml_row_size(GGML_TYPE_I2_B, n_per_row);

    int n = nrows * n_per_row;

    const double eps = 1e-6;

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n; i++) {
        i2_weight[i] = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t tmp = 0;
            double v = (double)src[i * 4 + j];
            if (fabs(v) > eps) {
                tmp = v > 0. ? 1 : 3;
            }
            i2_weight[i] |= tmp << (j * 2);
        }
    }

    return nrows * row_size;
}

size_t quantize_i1_58_b(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 1.58 bits per weight
    UNUSED(imatrix);

    size_t row_size = ggml_row_size(GGML_TYPE_I1_58_B, n_per_row);

    int n = nrows * n_per_row;

    const double eps = 1e-6;

    // 3^5 = 243 < 2^8 = 256
    uint8_t *i1_58_weight = (uint8_t *)dst;
    for (int i = 0; i * 5 < n; i++) {
        uint8_t x = 0;
        for (int j = 4; j >= 0; j--) {
            uint8_t tmp = 1;  // tmp - 1 为真值
            double v = (double)src[i * 5 + j];
            if (fabs(v) > eps) {
                tmp = v > 0. ? 2 : 0;
            }
            x = x * 3 + tmp;
        }
        i1_58_weight[i] = x;
    }

    return nrows * row_size;
}

// UNUSED
// 由于传进来的不是完整的矩阵，无法在这个阶段转置
size_t quantize_i2_t(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    size_t row_size = ggml_row_size(GGML_TYPE_I2_T, n_per_row);

    const double eps = 1e-6;

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n_per_row; i++){
        for (int j = 0; j < nrows; j++){
            int w = 0;
            for (int k = 0; k < 4; k++){
                double v = (double)src[j * n_per_row + i * 4 + k];
                uint8_t tmp = 0;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 1 : 3;
                }
                w |= tmp << (k * 2);
            }
            i2_weight[i * nrows + j] = w;
        }
    }

    return nrows * row_size;
}
