// =============================================================================
// Lux Lattice Library Implementation
// =============================================================================
//
// GPU-accelerated lattice operations using MLX for Metal/CUDA/CPU backends.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include "lux/lattice/lattice.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <mutex>
#include <random>
#include <unordered_map>
#include <memory>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

// =============================================================================
// Internal Utilities
// =============================================================================

namespace {

inline void extended_gcd(uint64_t a, uint64_t b,
                         int64_t& g, int64_t& x, int64_t& y) {
    if (b == 0) { g = a; x = 1; y = 0; return; }
    int64_t g1, x1, y1;
    extended_gcd(b, a % b, g1, x1, y1);
    g = g1; x = y1; y = x1 - (int64_t)(a / b) * y1;
}

inline uint64_t mod_inverse_internal(uint64_t a, uint64_t m) {
    int64_t g, x, y;
    extended_gcd(a, m, g, x, y);
    if (g != 1) return 0;  // Not invertible
    return (x % (int64_t)m + m) % m;
}

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t)a * b % m);
}

inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return sum >= m ? sum - m : sum;
}

inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : m - b + a;
}

inline uint64_t powmod(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod(result, base, m);
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    return result;
}

inline uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (uint64_t i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

uint64_t find_primitive_root_internal(uint32_t N, uint64_t Q) {
    uint64_t order = Q - 1;
    if (order % (2 * N) != 0) return 0;
    for (uint64_t g = 2; g < Q; ++g) {
        if (powmod(g, order / 2, Q) != 1) {
            return powmod(g, order / (2 * N), Q);
        }
    }
    return 0;
}

}  // anonymous namespace

// =============================================================================
// NTT Context Structure
// =============================================================================

struct LatticeNTTContext {
    uint32_t N;
    uint32_t log_N;
    uint64_t Q;
    uint64_t mu;           // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;        // N^{-1} mod Q
    uint64_t omega;        // Primitive 2N-th root of unity

    std::vector<uint64_t> twiddles;       // Forward twiddle factors
    std::vector<uint64_t> inv_twiddles;   // Inverse twiddle factors
    std::vector<uint64_t> tw_precon;      // Barrett precomputation for twiddles
    std::vector<uint64_t> inv_tw_precon;  // Barrett precomputation for inv twiddles
    std::vector<uint32_t> bit_rev;        // Bit-reversal permutation

#ifdef WITH_MLX
    mx::array gpu_twiddles{};
    mx::array gpu_inv_twiddles{};
    mx::array gpu_bit_rev{};
    bool use_gpu = false;
#endif
};

// Cache for NTT contexts
static std::unordered_map<uint64_t, std::unique_ptr<LatticeNTTContext>> g_context_cache;
static std::mutex g_cache_mutex;

// =============================================================================
// Backend Detection
// =============================================================================

#ifdef WITH_MLX
static bool g_gpu_checked = false;
static bool g_gpu_available = false;

static void check_gpu_once() {
    if (!g_gpu_checked) {
        g_gpu_checked = true;
        try {
            // Try to create a small array on GPU
            auto test = mx::zeros({1}, mx::float32);
            g_gpu_available = true;
        } catch (...) {
            g_gpu_available = false;
        }
    }
}
#endif

extern "C" bool lattice_gpu_available(void) {
#ifdef WITH_MLX
    check_gpu_once();
    return g_gpu_available;
#else
    return false;
#endif
}

extern "C" const char* lattice_get_backend(void) {
#ifdef WITH_MLX
    check_gpu_once();
    if (g_gpu_available) {
        #if defined(__APPLE__)
        return "Metal";
        #elif defined(__linux__)
        return "CUDA";
        #else
        return "MLX";
        #endif
    }
#endif
    return "CPU";
}

extern "C" void lattice_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_context_cache.clear();
}

// =============================================================================
// NTT Context Creation
// =============================================================================

static void compute_twiddles(LatticeNTTContext* ctx) {
    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;
    uint32_t log_N = ctx->log_N;

    ctx->twiddles.resize(N);
    ctx->inv_twiddles.resize(N);
    ctx->tw_precon.resize(N);
    ctx->inv_tw_precon.resize(N);
    ctx->bit_rev.resize(N);

    uint64_t omega = ctx->omega;
    uint64_t omega_inv = mod_inverse_internal(omega, Q);

    // Compute bit-reversal permutation
    for (uint32_t i = 0; i < N; ++i) {
        ctx->bit_rev[i] = bit_reverse(i, log_N);
    }

    // OpenFHE-style twiddle factor computation (bit-reversed storage)
    for (uint32_t m = 1; m < N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (N / m) * bit_reverse(i, log_m);
            ctx->twiddles[m + i] = powmod(omega, exp, Q);
            ctx->inv_twiddles[m + i] = powmod(omega_inv, exp, Q);

            // Barrett precomputation
            ctx->tw_precon[m + i] = static_cast<uint64_t>(
                ((__uint128_t)ctx->twiddles[m + i] << 64) / Q);
            ctx->inv_tw_precon[m + i] = static_cast<uint64_t>(
                ((__uint128_t)ctx->inv_twiddles[m + i] << 64) / Q);
        }
    }
    ctx->twiddles[0] = 1;
    ctx->inv_twiddles[0] = 1;
    ctx->tw_precon[0] = static_cast<uint64_t>(((__uint128_t)1 << 64) / Q);
    ctx->inv_tw_precon[0] = ctx->tw_precon[0];
}

extern "C" LatticeNTTContext* lattice_ntt_create(uint32_t N, uint64_t Q) {
    // Validate N is power of 2
    if (N == 0 || (N & (N - 1)) != 0) {
        return nullptr;
    }

    // Validate Q is NTT-friendly prime
    if (!is_prime(Q) || (Q - 1) % (2 * N) != 0) {
        return nullptr;
    }

    // Check cache first
    uint64_t key = ((uint64_t)N << 48) | (Q & 0xFFFFFFFFFFFF);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_context_cache.find(key);
        if (it != g_context_cache.end()) {
            // Return a copy
            LatticeNTTContext* copy = new LatticeNTTContext(*it->second);
            return copy;
        }
    }

    LatticeNTTContext* ctx = new LatticeNTTContext();
    ctx->N = N;
    ctx->Q = Q;
    ctx->log_N = 0;
    while ((1u << ctx->log_N) < N) ++ctx->log_N;

    ctx->mu = static_cast<uint64_t>((__uint128_t)1 << 64) / Q;
    ctx->N_inv = mod_inverse_internal(N, Q);
    ctx->omega = find_primitive_root_internal(N, Q);

    if (ctx->omega == 0) {
        delete ctx;
        return nullptr;
    }

    compute_twiddles(ctx);

#ifdef WITH_MLX
    check_gpu_once();
    ctx->use_gpu = g_gpu_available;
    if (ctx->use_gpu) {
        // Upload twiddles to GPU
        ctx->gpu_twiddles = mx::array(ctx->twiddles.data(), {(int)N}, mx::uint64);
        ctx->gpu_inv_twiddles = mx::array(ctx->inv_twiddles.data(), {(int)N}, mx::uint64);
        ctx->gpu_bit_rev = mx::array(ctx->bit_rev.data(), {(int)N}, mx::uint32);
    }
#endif

    // Add to cache
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_context_cache[key] = std::unique_ptr<LatticeNTTContext>(new LatticeNTTContext(*ctx));
    }

    return ctx;
}

extern "C" void lattice_ntt_destroy(LatticeNTTContext* ctx) {
    delete ctx;
}

extern "C" void lattice_ntt_get_params(const LatticeNTTContext* ctx,
                                        uint32_t* N,
                                        uint64_t* Q) {
    if (ctx) {
        if (N) *N = ctx->N;
        if (Q) *Q = ctx->Q;
    }
}

// =============================================================================
// CPU NTT Implementation (Cooley-Tukey / Gentleman-Sande)
// =============================================================================

// Barrett modular reduction
static inline uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t t = static_cast<uint64_t>(((__uint128_t)x * mu) >> 64);
    uint64_t r = x - t * Q;
    return r >= Q ? r - Q : r;
}

// Forward NTT (Cooley-Tukey, decimation-in-time)
static void ntt_forward_cpu(uint64_t* data, uint32_t N, uint64_t Q,
                            const std::vector<uint64_t>& tw,
                            const std::vector<uint32_t>& bit_rev) {
    // Bit-reversal permutation
    for (uint32_t i = 0; i < N; ++i) {
        if (i < bit_rev[i]) {
            std::swap(data[i], data[bit_rev[i]]);
        }
    }

    // Cooley-Tukey butterfly
    for (uint32_t m = 1; m < N; m <<= 1) {
        for (uint32_t k = 0; k < N; k += 2 * m) {
            for (uint32_t j = 0; j < m; ++j) {
                uint64_t t = mulmod(data[k + j + m], tw[m + j], Q);
                data[k + j + m] = submod(data[k + j], t, Q);
                data[k + j] = addmod(data[k + j], t, Q);
            }
        }
    }
}

// Inverse NTT (Gentleman-Sande, decimation-in-frequency)
static void ntt_inverse_cpu(uint64_t* data, uint32_t N, uint64_t Q, uint64_t N_inv,
                            const std::vector<uint64_t>& inv_tw,
                            const std::vector<uint32_t>& bit_rev) {
    // Gentleman-Sande butterfly
    for (uint32_t m = N / 2; m >= 1; m >>= 1) {
        for (uint32_t k = 0; k < N; k += 2 * m) {
            for (uint32_t j = 0; j < m; ++j) {
                uint64_t u = data[k + j];
                uint64_t v = data[k + j + m];
                data[k + j] = addmod(u, v, Q);
                data[k + j + m] = mulmod(submod(u, v, Q), inv_tw[m + j], Q);
            }
        }
    }

    // Bit-reversal permutation
    for (uint32_t i = 0; i < N; ++i) {
        if (i < bit_rev[i]) {
            std::swap(data[i], data[bit_rev[i]]);
        }
    }

    // Scale by N^{-1}
    for (uint32_t i = 0; i < N; ++i) {
        data[i] = mulmod(data[i], N_inv, Q);
    }
}

// =============================================================================
// Public NTT API
// =============================================================================

extern "C" int lattice_ntt_forward(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !data) return LATTICE_ERROR_NULL_PTR;

    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

#ifdef WITH_MLX
    if (ctx->use_gpu && batch >= 4) {
        // GPU path for large batches
        // TODO: Implement MLX kernel
    }
#endif

    // CPU path
    for (uint32_t b = 0; b < batch; ++b) {
        ntt_forward_cpu(data + b * N, N, Q, ctx->twiddles, ctx->bit_rev);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_ntt_inverse(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !data) return LATTICE_ERROR_NULL_PTR;

    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

#ifdef WITH_MLX
    if (ctx->use_gpu && batch >= 4) {
        // GPU path for large batches
        // TODO: Implement MLX kernel
    }
#endif

    // CPU path
    for (uint32_t b = 0; b < batch; ++b) {
        ntt_inverse_cpu(data + b * N, N, Q, ctx->N_inv, ctx->inv_twiddles, ctx->bit_rev);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_ntt_batch_forward(LatticeNTTContext* ctx,
                                          uint64_t** polys, uint32_t count) {
    if (!ctx || !polys) return LATTICE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < count; ++i) {
        if (!polys[i]) return LATTICE_ERROR_NULL_PTR;
        int err = lattice_ntt_forward(ctx, polys[i], 1);
        if (err != LATTICE_SUCCESS) return err;
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_ntt_batch_inverse(LatticeNTTContext* ctx,
                                          uint64_t** polys, uint32_t count) {
    if (!ctx || !polys) return LATTICE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < count; ++i) {
        if (!polys[i]) return LATTICE_ERROR_NULL_PTR;
        int err = lattice_ntt_inverse(ctx, polys[i], 1);
        if (err != LATTICE_SUCCESS) return err;
    }

    return LATTICE_SUCCESS;
}

// =============================================================================
// Polynomial Arithmetic
// =============================================================================

extern "C" int lattice_poly_mul_ntt(LatticeNTTContext* ctx,
                                     uint64_t* result,
                                     const uint64_t* a,
                                     const uint64_t* b) {
    if (!ctx || !result || !a || !b) return LATTICE_ERROR_NULL_PTR;

    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

    // Element-wise multiplication (Hadamard product)
    for (uint32_t i = 0; i < N; ++i) {
        result[i] = mulmod(a[i], b[i], Q);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_poly_mul(LatticeNTTContext* ctx,
                                 uint64_t* result,
                                 const uint64_t* a,
                                 const uint64_t* b) {
    if (!ctx || !result || !a || !b) return LATTICE_ERROR_NULL_PTR;

    uint32_t N = ctx->N;

    // Allocate temp buffers
    std::vector<uint64_t> a_ntt(N), b_ntt(N);
    std::memcpy(a_ntt.data(), a, N * sizeof(uint64_t));
    std::memcpy(b_ntt.data(), b, N * sizeof(uint64_t));

    // Forward NTT
    int err = lattice_ntt_forward(ctx, a_ntt.data(), 1);
    if (err != LATTICE_SUCCESS) return err;

    err = lattice_ntt_forward(ctx, b_ntt.data(), 1);
    if (err != LATTICE_SUCCESS) return err;

    // Hadamard product
    err = lattice_poly_mul_ntt(ctx, result, a_ntt.data(), b_ntt.data());
    if (err != LATTICE_SUCCESS) return err;

    // Inverse NTT
    return lattice_ntt_inverse(ctx, result, 1);
}

extern "C" int lattice_poly_add(uint64_t* result,
                                 const uint64_t* a,
                                 const uint64_t* b,
                                 uint32_t N,
                                 uint64_t Q) {
    if (!result || !a || !b) return LATTICE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < N; ++i) {
        result[i] = addmod(a[i], b[i], Q);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_poly_sub(uint64_t* result,
                                 const uint64_t* a,
                                 const uint64_t* b,
                                 uint32_t N,
                                 uint64_t Q) {
    if (!result || !a || !b) return LATTICE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < N; ++i) {
        result[i] = submod(a[i], b[i], Q);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_poly_scalar_mul(uint64_t* result,
                                        const uint64_t* a,
                                        uint64_t scalar,
                                        uint32_t N,
                                        uint64_t Q) {
    if (!result || !a) return LATTICE_ERROR_NULL_PTR;

    for (uint32_t i = 0; i < N; ++i) {
        result[i] = mulmod(a[i], scalar, Q);
    }

    return LATTICE_SUCCESS;
}

// =============================================================================
// Sampling
// =============================================================================

static thread_local std::mt19937_64 g_rng(std::random_device{}());

extern "C" int lattice_sample_gaussian(uint64_t* result,
                                        uint32_t N,
                                        uint64_t Q,
                                        double sigma,
                                        const uint8_t* seed) {
    if (!result) return LATTICE_ERROR_NULL_PTR;

    if (seed) {
        std::seed_seq seq(seed, seed + 32);
        g_rng.seed(seq);
    }

    std::normal_distribution<double> dist(0.0, sigma);

    for (uint32_t i = 0; i < N; ++i) {
        double sample = std::round(dist(g_rng));
        if (sample < 0) {
            result[i] = Q - (uint64_t)(-sample) % Q;
        } else {
            result[i] = (uint64_t)sample % Q;
        }
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_sample_uniform(uint64_t* result,
                                       uint32_t N,
                                       uint64_t Q,
                                       const uint8_t* seed) {
    if (!result) return LATTICE_ERROR_NULL_PTR;

    if (seed) {
        std::seed_seq seq(seed, seed + 32);
        g_rng.seed(seq);
    }

    std::uniform_int_distribution<uint64_t> dist(0, Q - 1);

    for (uint32_t i = 0; i < N; ++i) {
        result[i] = dist(g_rng);
    }

    return LATTICE_SUCCESS;
}

extern "C" int lattice_sample_ternary(uint64_t* result,
                                       uint32_t N,
                                       uint64_t Q,
                                       double density,
                                       const uint8_t* seed) {
    if (!result) return LATTICE_ERROR_NULL_PTR;

    if (seed) {
        std::seed_seq seq(seed, seed + 32);
        g_rng.seed(seq);
    }

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    for (uint32_t i = 0; i < N; ++i) {
        if (prob_dist(g_rng) < density) {
            result[i] = sign_dist(g_rng) ? 1 : Q - 1;  // +1 or -1
        } else {
            result[i] = 0;
        }
    }

    return LATTICE_SUCCESS;
}

// =============================================================================
// Utility Functions
// =============================================================================

extern "C" uint64_t lattice_find_primitive_root(uint32_t N, uint64_t Q) {
    return find_primitive_root_internal(N, Q);
}

extern "C" uint64_t lattice_mod_inverse(uint64_t a, uint64_t Q) {
    return mod_inverse_internal(a, Q);
}

extern "C" bool lattice_is_ntt_prime(uint32_t N, uint64_t Q) {
    if (!is_prime(Q)) return false;
    if ((Q - 1) % (2 * N) != 0) return false;
    return find_primitive_root_internal(N, Q) != 0;
}
