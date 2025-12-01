#pragma once

#define GGML_COMMON_DECL_C

#include "ggml-common.h"
#include "ggml.h"


void quantize_row_i8_v(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n);
void quantize_row_i8_v_tile(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n, float* scale);


void ggml_gemm_i2v_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2v2_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2v4_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2v8_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2v16_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_i1v_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1v2_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1v4_i8v_lut(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
