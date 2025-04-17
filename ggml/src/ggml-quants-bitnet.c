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

uint8_t I2_TO_I2S[256] = {
    40, 41, 0, 39, 43, 44, 0, 42, 0, 0, 0, 0, 37, 38, 0, 36, 49, 50, 0, 48, 52, 53, 0, 51, 0, 0, 0, 0, 46, 47, 0, 45,
    0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,  31, 32, 0, 30, 34, 35, 0, 33, 0, 0, 0, 0, 28, 29, 0, 27,
    67, 68, 0, 66, 70, 71, 0, 69, 0, 0, 0, 0, 64, 65, 0, 63, 76, 77, 0, 75, 79, 80, 0, 78, 0, 0, 0, 0, 73, 74, 0, 72,
    0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,  58, 59, 0, 57, 61, 62, 0, 60, 0, 0, 0, 0, 55, 56, 0, 54,
    0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,
    0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,
    13, 14, 0, 12, 16, 17, 0, 15, 0, 0, 0, 0, 10, 11, 0, 9,  22, 23, 0, 21, 25, 26, 0, 24, 0, 0, 0, 0, 19, 20, 0, 18,
    0,  0,  0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0,  0,  0, 0,  4,  5,  0, 3,  7,  8,  0, 6,  0, 0, 0, 0, 1,  2,  0, 0
};

size_t quantize_i2_s(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 2 bits per weight
    UNUSED(imatrix);

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    uint8_t *i2_weight = (uint8_t *)dst;
    for (int i = 0; i * 4 < n_per_row; i++) {
        for (int j = 0; j < nrows; j++) {
            uint8_t w = 0;
            for (int k = 0; k < 4; k++) {
                double v = (double)src[j * n_per_row + i * 4 + k];
                uint8_t tmp = 0;
                if (fabs(v) > eps) {
                    tmp = v > 0. ? 1 : 3;
                }
                w |= tmp << (k * 2);
            }
            i2_weight[i * nrows + j] = I2_TO_I2S[w];
        }
    }

    return nrows * row_size;
}

size_t quantize_i1_s(const float *restrict src, void *restrict dst, int64_t nrows, int64_t n_per_row,
                     const float *imatrix) {
    // 1.58 bits per weight
    UNUSED(imatrix);

    size_t row_size = ggml_row_size(GGML_TYPE_I1_S, n_per_row);


    uint8_t *i1_58_weight = (uint8_t *)dst;
    for (int i = 0; i * 5 < n_per_row; i++) {
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
            i1_58_weight[i * nrows + j] = w;
        }
    }

    return nrows * row_size;
}
