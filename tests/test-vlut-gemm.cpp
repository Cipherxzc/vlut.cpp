// This file defines MUL-MAT tests for Vec-LUT on CPU backends.


// ##############################
// ## Section 1: General Setup ##
// ##############################


#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <future>
#include <vector>

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = std::thread::hardware_concurrency();
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto & gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        std::vector<std::future<void>> tasks;
        tasks.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) {
            size_t start =     i*nels/n_threads;
            size_t end   = (i+1)*nels/n_threads;
            tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
        }
        for (auto & t : tasks) {
            t.get();
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        if (tensor->type != GGML_TYPE_I1_V && tensor->type != GGML_TYPE_I1_V_2 && tensor->type != GGML_TYPE_I1_V_4) {
            GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);
        }

         // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            if (tensor->type == GGML_TYPE_I2_V_4) {
                blck_size *= 4;
            } else if (tensor->type == GGML_TYPE_I2_V_8) {
                blck_size *= 8;
            }
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency()/2,
                                                      std::max<size_t>(1, n_blocks / min_blocks_per_thread));
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start =     i*n_blocks/n_threads;
                size_t end   = (i+1)*n_blocks/n_threads;
                tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor)/2;
        ggml_backend_tensor_set(tensor, data.data(), 0*nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1*nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto * tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float * a, const float * b, const size_t n, const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum/nvalid;
}

// utils for printing the variables of the test cases

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_PERF,
    MODE_SEARCH_I1,
    MODE_SEARCH_I2,
};

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual double max_maa_err() {
        return 1e-4;
    }

    virtual float grad_eps() {
        return 1e-1f;
    }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() {
        return false;
    }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() {
        return 10000;
    }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() {
        return {};
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_SEARCH_I1 || mode == MODE_SEARCH_I2) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            ggml_free(ctx);
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            ggml_free(ctx);
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_backend_graph_compute(backend, gf);

        // determine number of runs
        int n_runs;
        bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu =   8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t target_flops = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_flops / op_flops(out)) + 1;
        } else {
            // based on memory size
            const size_t GB = 1ULL << 30;
            const size_t target_size_cpu =  8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t target_size = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = std::min<int>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_size / op_size(out)) + 1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_backend_graph_compute(backend, gf);
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000*1000); // run for at least 1 second

        printf("    %8d runs - %8.2f us/run - ",
            total_runs,
            (double)total_time_us / total_runs);

        if (op_flops(out) > 0) {
            double flops_per_sec = (op_flops(out) * total_runs) / (total_time_us / 1e6);
            auto format_flops = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                } else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                } else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%6.2f KFLOP", flops / 1e3);
                }
                return buf;
            };
            printf("%s/run - \033[1;34m%sS\033[0m",
                format_flops(op_flops(out)).c_str(),
                format_flops(flops_per_sec).c_str());

        } else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m",
                op_size(out) / 1024,
                total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0);
        }
        printf("\n");

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return true;
    }
};


// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, per);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            std::array<int64_t, 4> per = {0, 1, 2, 3})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = {k, m, bs[0],       bs[1]};
            const int64_t ne_b[4] = {k, n, bs[0]*nr[0], bs[1]*nr[1]};

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]]);
            ggml_set_param(ctx, a);
            ggml_set_param(ctx, b);
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else {
            a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0],       bs[1]);
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
            ggml_set_param(ctx, a);
            ggml_set_param(ctx, b);
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################

// Define model configurations with explicit types to test
struct ModelConfig {
    std::string name;
    int d_model;
    int d_ff;
    std::vector<ggml_type> types_to_test;
};

// Test cases for performance evaluation: should be representative of real-world use cases
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf(const char* model_filter = nullptr, const std::vector<int>& test_ns = {}) {
    std::vector<std::unique_ptr<test_case>> test_cases;

    std::vector<ModelConfig> models = {
        {"bitnet_3b",  3200, 8640,  {GGML_TYPE_Q4_0, GGML_TYPE_I2_V_4, GGML_TYPE_I1_V_2}},  // TQ not supported for d_model=3200, fallback to Q4_0
        {"llama3_8b",  4096, 14336, {GGML_TYPE_TQ2_0, GGML_TYPE_I2_V_4, GGML_TYPE_TQ1_0, GGML_TYPE_I1_V_2}},
        {"falcon_1b",  2048, 8192,  {GGML_TYPE_TQ2_0, GGML_TYPE_I2_V_4, GGML_TYPE_TQ1_0, GGML_TYPE_I1_V_2}},
    };

    // Filter by model if specified
    for (const auto& model : models) {
        if (model_filter != nullptr && model.name != model_filter) {
            continue;
        }

        printf("-- Add tests for %s model --\n", model.name.c_str());
        
        for (int n : test_ns) {
            for (ggml_type type_a : model.types_to_test) {
                ggml_type type_b = GGML_TYPE_F32;
                
                printf("  Test %s with type %s, len %d\n", model.name.c_str(), ggml_type_name(type_a), n);
                
                // d_model × d_model
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_model, n, model.d_model, {1, 1}, {1, 1}));
                
                // d_model × d_ff
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_model, n, model.d_ff, {1, 1}, {1, 1}));
                
                // d_ff × d_model
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_ff, n, model.d_model, {1, 1}, {1, 1}));
            }
        }
    }

    return test_cases;
}


// Test cases for configuration search
static std::vector<std::unique_ptr<test_case>> make_test_cases_search(const char* model_filter = nullptr, const std::vector<int>& test_ns = {}, test_mode mode = MODE_SEARCH_I2) {
    std::vector<std::unique_ptr<test_case>> test_cases;
    
    std::vector<ggml_type> test_types;
    if (mode == MODE_SEARCH_I1) {
        test_types = {GGML_TYPE_I1_V_2, GGML_TYPE_I1_V_4};
    } else if (mode == MODE_SEARCH_I2) {
        test_types = {GGML_TYPE_I2_V_2, GGML_TYPE_I2_V_4, GGML_TYPE_I2_V_8, GGML_TYPE_I2_V_16};
    } else {
        GGML_ABORT("fatal error");
    }

    std::vector<ModelConfig> models = {
        {"bitnet_3b", 3200, 8640, test_types},
        // {"llama3_8b",  4096, 14336, test_types},
        // {"falcon_1b",  2048, 8192,  test_types},
    };

    // Filter by model if specified
    for (const auto& model : models) {
        if (model_filter != nullptr && model.name != model_filter) {
            continue;
        }

        printf("-- Add tests for %s model --\n", model.name.c_str());
        
        for (int n : test_ns) {
            for (ggml_type type_a : model.types_to_test) {
                ggml_type type_b = GGML_TYPE_F32;
                
                printf("  Test %s with type %s, len %d\n", model.name.c_str(), ggml_type_name(type_a), n);
                
                // d_model × d_model
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_model, n, model.d_model, {1, 1}, {1, 1}));
                
                // d_model × d_ff
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_model, n, model.d_ff, {1, 1}, {1, 1}));
                
                // d_ff × d_model
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, model.d_ff, n, model.d_model, {1, 1}, {1, 1}));
            }
        }
    }

    return test_cases;
}

// Update the test_backend function to accept a model filter
static bool test_backend(ggml_backend_t backend, test_mode mode, const char* model_filter = nullptr, const std::vector<int>& test_ns = {}) {

    if (mode == MODE_PERF) {
        auto test_cases = make_test_cases_perf(model_filter, test_ns);
        for (auto & test : test_cases) {
            test->eval_perf(backend, "MUL_MAT");
        }
        return true;
    }

    if (mode == MODE_SEARCH_I1 || mode == MODE_SEARCH_I2) {
        auto test_cases = make_test_cases_search(model_filter, test_ns, mode);
        for (auto & test : test_cases) {
            test->eval_perf(backend, "MUL_MAT");
        }
        return true;
    }


    GGML_ABORT("fatal error");
}

static void usage(char** argv) {
    printf("Usage: %s [mode] [-m model] [-t threads] [-ns seq_lens]\n", argv[0]);
    printf("\n");
    printf("  valid modes:\n");
    printf("    - perf   (default)  : run performance evaluation\n");
    printf("    - search1           : tuning mode for I1_V\n");
    printf("    - search2           : tuning mode for I2_V\n");
    printf("\n");
    printf("  options:\n");
    printf("    -m model            : test only the given model\n");
    printf("                          valid models: bitnet_3b, llama3_8b, falcon_1b\n");
    printf("                          if omitted, all models are tested\n");
    printf("\n");
    printf("    -t threads          : number of CPU threads to use\n");
    printf("                          default = hardware concurrency\n");
    printf("\n");
    printf("    -ns seq_lens        : comma-separated list of sequence lengths to test\n");
    printf("                          e.g. -ns 128,256,512\n");
    printf("                          default = 128\n");
    printf("\n");
    printf("  examples:\n");
    printf("    %s perf -m llama3_8b -t 8\n", argv[0]);
    printf("    %s search -ns 128,256\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_PERF;
    const char * op_name_filter = NULL;
    const char * model_filter = NULL;
    int n_threads = std::thread::hardware_concurrency();
    std::vector<int> test_ns = {128}; // default test sequence lengths

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "search1") == 0) {
            mode = MODE_SEARCH_I1;
        } else if (strcmp(argv[i], "search2") == 0) {
            mode = MODE_SEARCH_I2;
        } else if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0) {  // Changed from -n to -t
            if (i + 1 < argc) {
                n_threads = atoi(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-ns") == 0) {  // Added -ns option
            if (i + 1 < argc) {
                test_ns.clear();
                std::string ns_str = argv[++i];
                size_t pos = 0;
                while ((pos = ns_str.find(',')) != std::string::npos) {
                    test_ns.push_back(std::stoi(ns_str.substr(0, pos)));
                    ns_str.erase(0, pos + 1);
                }
                test_ns.push_back(std::stoi(ns_str));
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (argv[i][0] == '-') {
            printf("Unknown option: %s\n", argv[i]);
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    printf("Testing %zu devices\n\n", ggml_backend_dev_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        printf("Backend %zu/%zu: %s\n", i + 1, ggml_backend_dev_count(), ggml_backend_dev_name(dev));

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            printf("  Skipping non-CPU backend\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            int act_n_threads = std::thread::hardware_concurrency();
            if (n_threads > 0 && n_threads <= std::thread::hardware_concurrency()) {
                act_n_threads = n_threads;
            } else if (n_threads > std::thread::hardware_concurrency()) {
                printf("  Warning: requested %d threads, but only %u are available\n", n_threads, std::thread::hardware_concurrency());
            } else {
                printf("  Warning: invalid number of threads %d, using %u instead\n", n_threads, std::thread::hardware_concurrency());
            }
            ggml_backend_set_n_threads_fn(backend, act_n_threads);
            printf("  Using %d threads\n", act_n_threads);
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        bool ok = test_backend(backend, mode, model_filter, test_ns);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_dev_count());

    if (n_ok != ggml_backend_dev_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}