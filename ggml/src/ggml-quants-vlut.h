#pragma once

#define GGML_COMMON_DECL_C

#include "ggml-common.h"
#include "ggml.h"


size_t quantize_i2_v(const float* GGML_RESTRICT src, void* GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float* imatrix);
size_t quantize_i2_v_4(const float* GGML_RESTRICT src, void* GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float* imatrix);
size_t quantize_i2_v_8(const float* GGML_RESTRICT src, void* GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float* imatrix);


size_t quantize_i1_v(const float* GGML_RESTRICT src, void* GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float* imatrix);
size_t quantize_i1_v_2(const float* GGML_RESTRICT src, void* GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float* imatrix);
