// This file defines tests for various GGML ops and backends.
// For the forward pass it asserts that the results of multiple backends computing the same GGML ops are consistent.
// For the backward pass it asserts that the gradients from backpropagation are consistent
// with the gradients obtained via the method of finite differences ("grad" mode, this is optional).
// It is also possible to check the performance ("perf" mode).
//
// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be tested,
// and section 3 defines which tests to run.
// Quick start for adding a new GGML op: Go to section 2 and create a struct that inherits from test_case,
// then go to section 3 and add an instantiation of your struct.


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
        if (tensor->type != GGML_TYPE_I1_M){
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
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
    MODE_SEARCH,
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
        if (mode == MODE_PERF || mode == MODE_SEARCH || mode == MODE_GRAD) {
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

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("not supported [%s] ", ggml_backend_name(backend));
                    supported = false;
                    break;
                }
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return false;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        mode = mode == MODE_SEARCH ? MODE_SEARCH : MODE_PERF;

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
            //printf("  %s: skipping\n", op_desc(out).c_str());
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

    bool eval_grad(ggml_backend_t backend, const char * op_name) {
        mode = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + 2*ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        gb = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);

        ggml_tensor * out = build_graph(ctx);

        if ((op_name != nullptr && op_desc(out) != op_name) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        if (out->type != GGML_TYPE_F32) {
            ggml_free(ctx);
            printf("not supported [%s->type != FP32]\n", out->name);
            return true;
        }

        // check if the backend supports the ops
        bool supported = true;
        bool any_params = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                printf("not supported [%s] ", ggml_backend_name(backend));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    printf("not supported [%s->type != FP32] ", t->name);
                    supported = false;
                    break;
                }
            }
        }
        if (!any_params) {
            printf("not supported [%s] \n", op_name);
            supported = false;
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        int64_t ngrads = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += ggml_nelements(t);
            }
        }
        if (ngrads > grad_nmax()) {
            printf("skipping large tensors for speed \n");
            ggml_free(ctx);
            return true;
        }


        if (!ggml_is_scalar(out)) {
            out = ggml_sum(ctx, out);
            ggml_set_name(out, "sum_of_out");
        }
        ggml_set_loss(out);

        ggml_build_forward_expand(gf, out);
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx, ctx, gb, false);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) || ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                printf("not supported [%s] ", ggml_backend_name(backend));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                printf("not supported [%s->type != FP32] ", t->name);
                supported = false;
                break;
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend));
            ggml_free(ctx);
            return false;
        }


        initialize_tensors(ctx); // Randomizes all tensors (including gradients).
        ggml_graph_reset(gb);    // Sets gradients to 1 if loss, 0 otherwise.

        ggml_backend_graph_compute(backend, gf);
        ggml_backend_graph_compute(backend, gb);

        bool ok = true;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char * bn = ggml_backend_name(backend);
            const int64_t ne = ggml_nelements(t);

            std::vector<float> ga;
            struct ggml_tensor * grad = ggml_graph_get_grad(gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            } else {
                ga.resize(ne); // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) { // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    printf("[%s] nonfinite gradient at index %" PRId64 " (%s=%f) ", ggml_op_desc(t), i, bn, ga[i]);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne); // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t); // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu  = x0[i] + 1.0f*eps; // x, index i, up
                const float xiuh = x0[i] + 0.5f*eps; // x, index i, up half
                const float xidh = x0[i] - 0.5f*eps; // x, index i, down half
                const float xid  = x0[i] - 1.0f*eps; // x, index i, down

                float fu, fuh, fdh, fd; // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i*sizeof(float), sizeof(float));
                ggml_backend_graph_compute(backend, gf);
                ggml_backend_tensor_get(out, &fu, 0, ggml_nbytes(out));

                ggml_backend_tensor_set(t, &xid, i*sizeof(float), sizeof(float));
                ggml_backend_graph_compute(backend, gf);
                ggml_backend_tensor_get(out, &fd, 0, ggml_nbytes(out));

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i*sizeof(float), sizeof(float));
                    ggml_backend_graph_compute(backend, gf);
                    ggml_backend_tensor_get(out, &fuh, 0, ggml_nbytes(out));

                    ggml_backend_tensor_set(t, &xidh, i*sizeof(float), sizeof(float));
                    ggml_backend_graph_compute(backend, gf);
                    ggml_backend_tensor_get(out, &fdh, 0, ggml_nbytes(out));

                    gn[i] = (8.0*(double)fuh + (double)fd - (8.0*(double)fdh + (double)fu)) / (6.0*(double)eps);
                } else {
                    gn[i] = (fu - fd) / (2.0f*eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, ggml_nbytes(t));
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                printf("[%s] MAA = %.9f > %.9f ", ggml_op_desc(t), err, max_maa_err());
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        if (!ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }
};


// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################


// The following is an example showing the bare minimum for creating a test for a GGML op.

// GGML_OP_EXAMPLE
struct test_example : public test_case {
    // Always define these 2 or variants thereof:
    const ggml_type type; // The type of the input tensors.
    const std::array<int64_t, 4> ne; // The shape of the input tensors.
    // For some ops it's necessary to define multiple types or shapes for the inputs.
    // Or they may need additional parameters.

    // Put all parameters needed to fully define the test into one of the VARS_TO_STR macros.
    // In most cases these are just the properties of the struct that you defined above.
    // This is needed for info prints.
    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    // Define a constructor for the struct.
    // In most cases it will be sufficient to have the same arguments as the struct has properties
    // and just use initializer lists.
    test_example(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    // Define how a simple GGML compute graph can be constructed for the new GGML op.
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Step 1: create input tensors that don't depend on any other tensors:
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a"); // Setting names is optional but it's useful for debugging.

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        // Step 2: use the op that you want to test in the GGML compute graph.
        ggml_tensor * out = ggml_add(ctx, a, b); // For this example we're just doing a simple addition.
        ggml_set_name(out, "out");

        // Step 3: return the output tensor.
        return out;
    }
    // In order to also check the gradients for your op, add calls like ggml_set_param(ctx, a)
    // immediately after you create the tensors.
    // This is optional and only makes sense if a backward pass has actually been implemented for the new op.
};

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
static const ggml_type all_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16,
    GGML_TYPE_Q8_0, // for I8MM tests
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, // for I8MM tests
    GGML_TYPE_Q4_K,
    GGML_TYPE_IQ2_XXS
};

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    // GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : {GGML_TYPE_I2_S, GGML_TYPE_I1_S}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1}, {1, 1}));
            }
        }
    }

    return test_cases;
}

// Test cases for performance evaluation: should be representative of real-world use cases
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf(const char* model_filter = nullptr, const std::vector<int>& test_ns = {}) {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // Define model configurations with explicit types to test
    struct ModelConfig {
        std::string name;
        int d_model;
        int d_ff;
        std::vector<ggml_type> types_to_test;
    };

    std::vector<ModelConfig> models = {
        // {"bitnet_3b",  3200, 8640,  {GGML_TYPE_Q4_0, GGML_TYPE_I2_S, GGML_TYPE_I1_M}},
        // {"llama3_8b",  4096, 14336, {GGML_TYPE_TQ2_0, GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_I1_M}},
        // {"falcon_1b",  2048, 8192,  {GGML_TYPE_TQ2_0, GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_I1_M}},
        // {"trilm_1.5b", 2048, 6144,  {GGML_TYPE_TQ2_0, GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_I1_M}},
        {"llama3_8b", 4096, 14336, {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_TQ2_0, GGML_TYPE_TQ1_0, GGML_TYPE_I2_S, GGML_TYPE_I1_M}},
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
static std::vector<std::unique_ptr<test_case>> make_test_cases_search(const char* model_filter = nullptr, const std::vector<int>& test_ns = {}) {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // Define model configurations with explicit types to test
    struct ModelConfig {
        std::string name;
        int d_model;
        int d_ff;
        std::vector<ggml_type> types_to_test;
    };

    std::vector<ModelConfig> models = {
        {"bitnet_3b",  3200, 8640,  {GGML_TYPE_I2_S}},
        {"llama3_8b",  4096, 14336, {GGML_TYPE_I2_S}},
        {"falcon_1b",  2048, 8192,  {GGML_TYPE_I2_S}},
        {"trilm_1.5b", 2048, 6144,  {GGML_TYPE_I2_S}},
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
static bool test_backend(ggml_backend_t backend, test_mode mode, const char* op_name, const char* model_filter = nullptr, const std::vector<int>& test_ns = {}) {
    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        if (backend_cpu == NULL) {
            printf("  Failed to initialize CPU backend\n");
            return false;
        }

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        auto test_cases = make_test_cases_perf(model_filter, test_ns);
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
    }

    if (mode == MODE_SEARCH) {
        auto test_cases = make_test_cases_search(model_filter, test_ns);
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
    }


    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o op] [-b backend] [-m model]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - perf (performance evaluation)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc)\n");
    printf("    model names for -m are: bitnet_3b, llama3_8b, falcon_1b, trilm_1.5b\n");
    printf("      - if not specified, all models are tested\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    const char * op_name_filter = NULL;
    const char * backend_filter = NULL;
    const char * model_filter = NULL;
    int n_threads = 0;
    std::vector<int> test_ns = {128}; // default test sequence lengths

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "search") == 0) {
            mode = MODE_SEARCH;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0) { // Changed from -n to -t
            if (i + 1 < argc) {
                n_threads = atoi(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-ns") == 0) { // Added -ns option
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

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
            printf("  Skipping CPU backend\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            int act_n_threads = std::thread::hardware_concurrency();
            if (n_threads > 0 && n_threads < std::thread::hardware_concurrency()) {
                act_n_threads = n_threads;
            } else {
                printf("  Warning: requested %d threads, but only %u are available\n", n_threads, std::thread::hardware_concurrency());
            }
            ggml_backend_set_n_threads_fn(backend, act_n_threads);
            printf("  Using %d threads\n", act_n_threads);
        }

        printf("  Device description: %s\n", ggml_backend_dev_description(dev));
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  Device memory: %zu MB (%zu MB free)\n", total / 1024 / 1024, free / 1024 / 1024);
        printf("\n");

        bool ok = test_backend(backend, mode, op_name_filter, model_filter, test_ns);

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