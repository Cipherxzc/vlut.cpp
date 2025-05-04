#pragma once

#define GGML_COMMON_DECL_C

#include "ggml-common.h"
#include "ggml.h"

void quantize_row_i8_b(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n);
void quantize_row_i8_b_tile(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n, float* scale);

void ggml_gemm_i2s_i8b_make_table(int ith, int nth, const int8_t* GGML_RESTRICT y, int ntables, int nr, int n, int16_t* GGML_RESTRICT table);
void ggml_gemm_i1s_i8b_make_table(int ith, int nth, const int8_t* GGML_RESTRICT y, int ntables, int nr, int n, int16_t* GGML_RESTRICT table);
void ggml_gemm_i1m_i8b_make_table(int ith, int nth, const int8_t* GGML_RESTRICT y, int ntables, int nr, int n, int16_t* GGML_RESTRICT table);

void ggml_gemm_i2s_i8b_make_table_quant(int ith, int nth, const float *restrict y, float *restrict scale, int nr, int n, int16_t *restrict table);
void ggml_gemm_i2s_i8b_make_table_quant2(int ith, int nth, const float *restrict y, float *restrict scale, int nr, int n, int16_t *restrict table);

void ggml_gemm_i2s_i8b_LUT(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1s_i8b_LUT(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1m_i8b_LUT(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_i2s_i8b_LUT2(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1s_i8b_LUT2(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1m_i8b_LUT2(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2s2_i8b_LUT2(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2s4_i8b_LUT2(int ith, int nth, int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
