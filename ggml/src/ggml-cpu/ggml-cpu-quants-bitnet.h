#pragma once

#define GGML_COMMON_DECL_C

#include "ggml-common.h"
#include "ggml.h"

#define BITNET_AVX2
#define BITNET_DEBUG
#define BITNET_LUT2
#define BITNET_TILING

#define TABLE_ENTRY_SIZE 32


void quantize_row_i8_b(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n);
void quantize_row_i8_b_trans(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n, int64_t row_size, float* scale);
void quantize_row_i8_b_tile(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t n, float* scale);

void ggml_vec_dot_i2_i8_b(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, size_t bx, const void* GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_i1_58_i8_b(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, size_t bx, const void* GGML_RESTRICT vy, size_t by, int nrc);

void ggml_gemm_i2_i8_b_make_table(const int8_t *GGML_RESTRICT y, int nrows, int n, int16_t *GGML_RESTRICT table);
void ggml_gemm_i1_58_i8_b_make_table(const int8_t* GGML_RESTRICT y, int nrows, int n, int16_t* GGML_RESTRICT table);

// LUT1(LUT && !TILING)：集中打表，按列遍历 weight
// LUT2(LUT2 && !TILING)：边计算边打表，按列遍历 weight
// LUT3(LUT2 && TILING)：边计算边打表，为 activation 添加了tiling
// LUT4(LUT && TILING): 集中打表，为 activation 添加了tiling

void ggml_gemm_i2_i8_b_LUT(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1_58_i8_b_LUT(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_t_LUT(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_i2_i8_b_LUT2(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i1_58_i8_b_LUT2(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_t_LUT2(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

void ggml_gemm_i2_i8_t_LUT3(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s_LUT3(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

// void ggml_gemm_i2_i8_s_LUT4(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy, int nr, int nc);

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };

// clang-format off
GGML_TABLE_BEGIN(uint32_t, i2s_i8s, 256)
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x00000100, 0x00000101, 0x00000100, 0x000001ff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x0000ff00, 0x0000ff01, 0x0000ff00, 0x0000ffff,
    0x00010000, 0x00010001, 0x00010000, 0x000100ff,
    0x00010100, 0x00010101, 0x00010100, 0x000101ff,
    0x00010000, 0x00010001, 0x00010000, 0x000100ff,
    0x0001ff00, 0x0001ff01, 0x0001ff00, 0x0001ffff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x00000100, 0x00000101, 0x00000100, 0x000001ff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x0000ff00, 0x0000ff01, 0x0000ff00, 0x0000ffff,
    0x00ff0000, 0x00ff0001, 0x00ff0000, 0x00ff00ff,
    0x00ff0100, 0x00ff0101, 0x00ff0100, 0x00ff01ff,
    0x00ff0000, 0x00ff0001, 0x00ff0000, 0x00ff00ff,
    0x00ffff00, 0x00ffff01, 0x00ffff00, 0x00ffffff,
    0x01000000, 0x01000001, 0x01000000, 0x010000ff,
    0x01000100, 0x01000101, 0x01000100, 0x010001ff,
    0x01000000, 0x01000001, 0x01000000, 0x010000ff,
    0x0100ff00, 0x0100ff01, 0x0100ff00, 0x0100ffff,
    0x01010000, 0x01010001, 0x01010000, 0x010100ff,
    0x01010100, 0x01010101, 0x01010100, 0x010101ff,
    0x01010000, 0x01010001, 0x01010000, 0x010100ff,
    0x0101ff00, 0x0101ff01, 0x0101ff00, 0x0101ffff,
    0x01000000, 0x01000001, 0x01000000, 0x010000ff,
    0x01000100, 0x01000101, 0x01000100, 0x010001ff,
    0x01000000, 0x01000001, 0x01000000, 0x010000ff,
    0x0100ff00, 0x0100ff01, 0x0100ff00, 0x0100ffff,
    0x01ff0000, 0x01ff0001, 0x01ff0000, 0x01ff00ff,
    0x01ff0100, 0x01ff0101, 0x01ff0100, 0x01ff01ff,
    0x01ff0000, 0x01ff0001, 0x01ff0000, 0x01ff00ff,
    0x01ffff00, 0x01ffff01, 0x01ffff00, 0x01ffffff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x00000100, 0x00000101, 0x00000100, 0x000001ff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x0000ff00, 0x0000ff01, 0x0000ff00, 0x0000ffff,
    0x00010000, 0x00010001, 0x00010000, 0x000100ff,
    0x00010100, 0x00010101, 0x00010100, 0x000101ff,
    0x00010000, 0x00010001, 0x00010000, 0x000100ff,
    0x0001ff00, 0x0001ff01, 0x0001ff00, 0x0001ffff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x00000100, 0x00000101, 0x00000100, 0x000001ff,
    0x00000000, 0x00000001, 0x00000000, 0x000000ff,
    0x0000ff00, 0x0000ff01, 0x0000ff00, 0x0000ffff,
    0x00ff0000, 0x00ff0001, 0x00ff0000, 0x00ff00ff,
    0x00ff0100, 0x00ff0101, 0x00ff0100, 0x00ff01ff,
    0x00ff0000, 0x00ff0001, 0x00ff0000, 0x00ff00ff,
    0x00ffff00, 0x00ffff01, 0x00ffff00, 0x00ffffff,
    0xff000000, 0xff000001, 0xff000000, 0xff0000ff,
    0xff000100, 0xff000101, 0xff000100, 0xff0001ff,
    0xff000000, 0xff000001, 0xff000000, 0xff0000ff,
    0xff00ff00, 0xff00ff01, 0xff00ff00, 0xff00ffff,
    0xff010000, 0xff010001, 0xff010000, 0xff0100ff,
    0xff010100, 0xff010101, 0xff010100, 0xff0101ff,
    0xff010000, 0xff010001, 0xff010000, 0xff0100ff,
    0xff01ff00, 0xff01ff01, 0xff01ff00, 0xff01ffff,
    0xff000000, 0xff000001, 0xff000000, 0xff0000ff,
    0xff000100, 0xff000101, 0xff000100, 0xff0001ff,
    0xff000000, 0xff000001, 0xff000000, 0xff0000ff,
    0xff00ff00, 0xff00ff01, 0xff00ff00, 0xff00ffff,
    0xffff0000, 0xffff0001, 0xffff0000, 0xffff00ff,
    0xffff0100, 0xffff0101, 0xffff0100, 0xffff01ff,
    0xffff0000, 0xffff0001, 0xffff0000, 0xffff00ff,
    0xffffff00, 0xffffff01, 0xffffff00, 0xffffffff,
GGML_TABLE_END()
