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
