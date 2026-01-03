// =============================================================================
// Lux Lattice - Metal NTT Dispatcher Implementation
// =============================================================================
//
// Native Metal GPU acceleration for Number Theoretic Transform.
// Supports both fused kernels (N <= 4096) and staged dispatch (N > 4096).
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ntt.h"
#include <vector>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <memory>

// =============================================================================
// Modular Arithmetic Utilities
// =============================================================================

namespace {

inline void extended_gcd(uint64_t a, uint64_t b, int64_t& g, int64_t& x, int64_t& y) {
    if (b == 0) { g = a; x = 1; y = 0; return; }
    int64_t g1, x1, y1;
    extended_gcd(b, a % b, g1, x1, y1);
    g = g1; x = y1; y = x1 - (int64_t)(a / b) * y1;
}

inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t g, x, y;
    extended_gcd(a, m, g, x, y);
    if (g != 1) return 0;
    return (x % (int64_t)m + m) % m;
}

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t)a * b % m);
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

uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    uint64_t order = Q - 1;
    if (order % (2 * N) != 0) return 0;
    for (uint64_t g = 2; g < Q; ++g) {
        if (powmod(g, order / 2, Q) != 1) {
            return powmod(g, order / (2 * N), Q);
        }
    }
    return 0;
}

} // anonymous namespace

// =============================================================================
// NTT Parameters (matches Metal shader struct)
// =============================================================================

struct NTTParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t stage;
    uint32_t batch;
};

// =============================================================================
// Metal NTT Context Implementation
// =============================================================================

struct MetalNTTContext {
    // Parameters
    uint32_t N;
    uint32_t log_N;
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    bool use_fused;  // N <= 4096

    // Twiddle factors (CPU side for upload)
    std::vector<uint64_t> twiddles;
    std::vector<uint64_t> precon;
    std::vector<uint64_t> inv_twiddles;
    std::vector<uint64_t> inv_precon;

    // Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Compute pipelines
    id<MTLComputePipelineState> forward_fused;
    id<MTLComputePipelineState> inverse_fused;
    id<MTLComputePipelineState> forward_stage;
    id<MTLComputePipelineState> inverse_stage;
    id<MTLComputePipelineState> scale_inverse;
    id<MTLComputePipelineState> pointwise_mul;

    // GPU buffers (persistent)
    id<MTLBuffer> twiddles_buf;
    id<MTLBuffer> precon_buf;
    id<MTLBuffer> inv_twiddles_buf;
    id<MTLBuffer> inv_precon_buf;
    id<MTLBuffer> params_buf;

    bool valid;
};

// =============================================================================
// Shader Source (embedded)
// =============================================================================

static NSString* get_shader_source() {
    // Load from compiled metallib or embed source
    // For simplicity, we compile at runtime from embedded source
    @autoreleasepool {
        NSBundle* bundle = [NSBundle mainBundle];
        NSString* path = [bundle pathForResource:@"ntt_kernels" ofType:@"metallib"];
        if (path) {
            return nil;  // Use precompiled metallib
        }
    }

    // Embed minimal shader source for runtime compilation
    return @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct NTTParams {
    uint64_t Q;
    uint64_t mu;
    uint64_t N_inv;
    uint64_t N_inv_precon;
    uint32_t N;
    uint32_t log_N;
    uint32_t stage;
    uint32_t batch;
};

inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = metal::mulhi(a, precon);
    uint64_t result = a * b - q_approx * Q;
    return result >= Q ? result - Q : result;
}

inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : a + Q - b;
}

kernel void ntt_forward_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = gid;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t log_N = params.log_N;

    device uint64_t* poly = data + batch_idx * N;

    for (uint32_t i = tid; i < N; i += tpg) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N >> (s + 1);

        for (uint32_t butterfly = tid; butterfly < N/2; butterfly += tpg) {
            uint32_t i = butterfly / t;
            uint32_t j = butterfly % t;
            uint32_t idx_lo = (i << (log_N - s)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint64_t omega = twiddles[tw_idx];
            uint64_t pc = precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];
            uint64_t omega_hi = barrett_mul(hi, omega, Q, pc);

            shared[idx_lo] = mod_add(lo, omega_hi, Q);
            shared[idx_hi] = mod_sub(lo, omega_hi, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = shared[i];
    }
}

kernel void ntt_inverse_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* inv_precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = gid;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t log_N = params.log_N;

    device uint64_t* poly = data + batch_idx * N;

    for (uint32_t i = tid; i < N; i += tpg) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = N >> (s + 1);
        uint32_t t = 1u << s;

        for (uint32_t butterfly = tid; butterfly < N/2; butterfly += tpg) {
            uint32_t i = butterfly / t;
            uint32_t j = butterfly % t;
            uint32_t idx_lo = (i << (s + 1)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t pc = inv_precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];
            uint64_t sum = mod_add(lo, hi, Q);
            uint64_t diff = mod_sub(lo, hi, Q);

            shared[idx_lo] = sum;
            shared[idx_hi] = barrett_mul(diff, omega, Q, pc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = barrett_mul(shared[i], params.N_inv, Q, params.N_inv_precon);
    }
}

kernel void ntt_forward_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t stage = params.stage;

    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);
    if (butterfly_idx >= N >> 1) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (params.log_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;

    uint32_t tw_idx = m + i;
    device uint64_t* poly = data + batch_idx * N;

    uint64_t lo = poly[idx_lo];
    uint64_t hi = poly[idx_hi];
    uint64_t omega_hi = barrett_mul(hi, twiddles[tw_idx], Q, precon[tw_idx]);

    poly[idx_lo] = mod_add(lo, omega_hi, Q);
    poly[idx_hi] = mod_sub(lo, omega_hi, Q);
}

kernel void ntt_inverse_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* inv_precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t stage = params.stage;

    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;
    if (butterfly_idx >= N >> 1) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;

    uint32_t tw_idx = m + i;
    device uint64_t* poly = data + batch_idx * N;

    uint64_t lo = poly[idx_lo];
    uint64_t hi = poly[idx_hi];
    poly[idx_lo] = mod_add(lo, hi, Q);
    poly[idx_hi] = barrett_mul(mod_sub(lo, hi, Q), inv_twiddles[tw_idx], Q, inv_precon[tw_idx]);
}

kernel void ntt_scale_inverse(
    device uint64_t* data [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    if (batch_idx >= params.batch || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    data[idx] = barrett_mul(data[idx], params.N_inv, params.Q, params.N_inv_precon);
}

kernel void ntt_pointwise_mul(
    device uint64_t* result [[buffer(0)]],
    constant uint64_t* a [[buffer(1)]],
    constant uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    if (batch_idx >= params.batch || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    uint64_t Q = params.Q;
    uint64_t av = a[idx];
    uint64_t bv = b[idx];

    uint64_t lo = av * bv;
    uint64_t hi = metal::mulhi(av, bv);

    if (hi == 0) {
        result[idx] = lo % Q;
    } else {
        uint64_t two32_mod_q = (uint64_t(1) << 32) % Q;
        uint64_t two64_mod_q = (two32_mod_q * two32_mod_q) % Q;
        result[idx] = (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
    }
}
)METAL";
}

// =============================================================================
// Twiddle Factor Computation
// =============================================================================

static void compute_twiddles(MetalNTTContext* ctx) {
    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

    ctx->twiddles.resize(N);
    ctx->precon.resize(N);
    ctx->inv_twiddles.resize(N);
    ctx->inv_precon.resize(N);

    uint64_t omega = find_primitive_root(N, Q);
    uint64_t omega_inv = mod_inverse(omega, Q);

    // Bit-reversed twiddle storage (OpenFHE compatible)
    for (uint32_t m = 1; m < N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (N / m) * bit_reverse(i, log_m);
            ctx->twiddles[m + i] = powmod(omega, exp, Q);
            ctx->inv_twiddles[m + i] = powmod(omega_inv, exp, Q);

            // Barrett precomputation
            ctx->precon[m + i] = static_cast<uint64_t>(
                ((__uint128_t)ctx->twiddles[m + i] << 64) / Q);
            ctx->inv_precon[m + i] = static_cast<uint64_t>(
                ((__uint128_t)ctx->inv_twiddles[m + i] << 64) / Q);
        }
    }
    ctx->twiddles[0] = 1;
    ctx->inv_twiddles[0] = 1;
    ctx->precon[0] = static_cast<uint64_t>(((__uint128_t)1 << 64) / Q);
    ctx->inv_precon[0] = ctx->precon[0];
}

// =============================================================================
// Metal Pipeline Creation
// =============================================================================

static bool create_pipelines(MetalNTTContext* ctx) {
    @autoreleasepool {
        NSError* error = nil;

        // Try loading precompiled metallib first
        // Check standard install locations
        NSArray* metallibPaths = @[
            @"/usr/local/share/lux/lattice/lux_lattice.metallib",
            @"/usr/local/share/lux/lattice/ntt_kernels.metallib",
            [[NSBundle mainBundle] pathForResource:@"lux_lattice" ofType:@"metallib"] ?: @"",
            [[NSBundle mainBundle] pathForResource:@"ntt_kernels" ofType:@"metallib"] ?: @""
        ];

        for (NSString* libPath in metallibPaths) {
            if (libPath.length > 0 && [[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
                NSURL* libURL = [NSURL fileURLWithPath:libPath];
                ctx->library = [ctx->device newLibraryWithURL:libURL error:&error];
                if (ctx->library) break;
            }
        }

        // Fall back to runtime compilation
        if (!ctx->library) {
            NSString* source = get_shader_source();
            if (!source) return false;

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // Use mathMode on macOS 15.0+, fastMathEnabled on older versions
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            ctx->library = [ctx->device newLibraryWithSource:source options:options error:&error];

            if (!ctx->library) {
                NSLog(@"Metal NTT: Failed to compile shaders: %@", error);
                return false;
            }
        }

        // Create pipelines
        id<MTLFunction> fn;

        fn = [ctx->library newFunctionWithName:@"ntt_forward_fused"];
        if (fn) ctx->forward_fused = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        fn = [ctx->library newFunctionWithName:@"ntt_inverse_fused"];
        if (fn) ctx->inverse_fused = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        fn = [ctx->library newFunctionWithName:@"ntt_forward_stage"];
        if (fn) ctx->forward_stage = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        fn = [ctx->library newFunctionWithName:@"ntt_inverse_stage"];
        if (fn) ctx->inverse_stage = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        fn = [ctx->library newFunctionWithName:@"ntt_scale_inverse"];
        if (fn) ctx->scale_inverse = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        fn = [ctx->library newFunctionWithName:@"ntt_pointwise_mul"];
        if (fn) ctx->pointwise_mul = [ctx->device newComputePipelineStateWithFunction:fn error:&error];

        return ctx->forward_fused != nil || ctx->forward_stage != nil;
    }
}

// =============================================================================
// Public API Implementation
// =============================================================================

bool metal_ntt_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

MetalNTTContext* metal_ntt_create(uint32_t N, uint64_t Q) {
    @autoreleasepool {
        // Validate parameters
        if (N == 0 || (N & (N - 1)) != 0 || N > 16384) return nullptr;

        MetalNTTContext* ctx = new MetalNTTContext();
        ctx->valid = false;
        ctx->N = N;
        ctx->Q = Q;
        ctx->log_N = 0;
        while ((1u << ctx->log_N) < N) ++ctx->log_N;

        ctx->mu = static_cast<uint64_t>((__uint128_t)1 << 64) / Q;
        ctx->N_inv = mod_inverse(N, Q);
        ctx->N_inv_precon = static_cast<uint64_t>(((__uint128_t)ctx->N_inv << 64) / Q);
        ctx->use_fused = (N <= 4096);

        // Create Metal device
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            delete ctx;
            return nullptr;
        }

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            delete ctx;
            return nullptr;
        }

        // Compute twiddles
        compute_twiddles(ctx);

        // Create pipelines
        if (!create_pipelines(ctx)) {
            delete ctx;
            return nullptr;
        }

        // Allocate GPU buffers for twiddles
        ctx->twiddles_buf = [ctx->device newBufferWithBytes:ctx->twiddles.data()
                                                    length:N * sizeof(uint64_t)
                                                   options:MTLResourceStorageModeShared];
        ctx->precon_buf = [ctx->device newBufferWithBytes:ctx->precon.data()
                                                  length:N * sizeof(uint64_t)
                                                 options:MTLResourceStorageModeShared];
        ctx->inv_twiddles_buf = [ctx->device newBufferWithBytes:ctx->inv_twiddles.data()
                                                        length:N * sizeof(uint64_t)
                                                       options:MTLResourceStorageModeShared];
        ctx->inv_precon_buf = [ctx->device newBufferWithBytes:ctx->inv_precon.data()
                                                      length:N * sizeof(uint64_t)
                                                     options:MTLResourceStorageModeShared];
        ctx->params_buf = [ctx->device newBufferWithLength:sizeof(NTTParams)
                                                  options:MTLResourceStorageModeShared];

        ctx->valid = true;
        return ctx;
    }
}

void metal_ntt_destroy(MetalNTTContext* ctx) {
    if (ctx) {
        // ARC handles Metal object cleanup
        delete ctx;
    }
}

int metal_ntt_forward(MetalNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !ctx->valid || !data || batch == 0) return -1;

    @autoreleasepool {
        uint32_t N = ctx->N;

        // Create data buffer (shared memory for zero-copy)
        id<MTLBuffer> data_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                               length:batch * N * sizeof(uint64_t)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!data_buf) return -1;

        // Setup params
        NTTParams params = {ctx->Q, ctx->mu, ctx->N_inv, ctx->N_inv_precon, N, ctx->log_N, 0, batch};
        memcpy([ctx->params_buf contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

        if (ctx->use_fused && ctx->forward_fused) {
            // Fused kernel: one dispatch per batch element
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx->forward_fused];
            [enc setBuffer:data_buf offset:0 atIndex:0];
            [enc setBuffer:ctx->twiddles_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->precon_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->params_buf offset:0 atIndex:3];

            // Threadgroup size: optimal for Apple Silicon
            NSUInteger tpg = MIN(512, N);
            [enc setThreadgroupMemoryLength:N * sizeof(uint64_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        } else {
            // Staged kernel: log(N) dispatches
            for (uint32_t s = 0; s < ctx->log_N; ++s) {
                params.stage = s;
                memcpy([ctx->params_buf contents], &params, sizeof(params));

                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->forward_stage];
                [enc setBuffer:data_buf offset:0 atIndex:0];
                [enc setBuffer:ctx->twiddles_buf offset:0 atIndex:1];
                [enc setBuffer:ctx->precon_buf offset:0 atIndex:2];
                [enc setBuffer:ctx->params_buf offset:0 atIndex:3];

                NSUInteger butterflies = N / 2;
                [enc dispatchThreads:MTLSizeMake(butterflies, batch, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN(256, butterflies), 1, 1)];
                [enc endEncoding];
            }
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -1;
    }
}

int metal_ntt_inverse(MetalNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !ctx->valid || !data || batch == 0) return -1;

    @autoreleasepool {
        uint32_t N = ctx->N;

        id<MTLBuffer> data_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                               length:batch * N * sizeof(uint64_t)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!data_buf) return -1;

        NTTParams params = {ctx->Q, ctx->mu, ctx->N_inv, ctx->N_inv_precon, N, ctx->log_N, 0, batch};
        memcpy([ctx->params_buf contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

        if (ctx->use_fused && ctx->inverse_fused) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:ctx->inverse_fused];
            [enc setBuffer:data_buf offset:0 atIndex:0];
            [enc setBuffer:ctx->inv_twiddles_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->inv_precon_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->params_buf offset:0 atIndex:3];

            NSUInteger tpg = MIN(512, N);
            [enc setThreadgroupMemoryLength:N * sizeof(uint64_t) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
        } else {
            // Staged + scale
            for (uint32_t s = 0; s < ctx->log_N; ++s) {
                params.stage = s;
                memcpy([ctx->params_buf contents], &params, sizeof(params));

                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->inverse_stage];
                [enc setBuffer:data_buf offset:0 atIndex:0];
                [enc setBuffer:ctx->inv_twiddles_buf offset:0 atIndex:1];
                [enc setBuffer:ctx->inv_precon_buf offset:0 atIndex:2];
                [enc setBuffer:ctx->params_buf offset:0 atIndex:3];

                NSUInteger butterflies = N / 2;
                [enc dispatchThreads:MTLSizeMake(butterflies, batch, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN(256, butterflies), 1, 1)];
                [enc endEncoding];
            }

            // Scale by N^{-1}
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->scale_inverse];
                [enc setBuffer:data_buf offset:0 atIndex:0];
                [enc setBuffer:ctx->params_buf offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(N, batch, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN(256, N), 1, 1)];
                [enc endEncoding];
            }
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -1;
    }
}

int metal_ntt_pointwise_mul(MetalNTTContext* ctx, uint64_t* result,
                            const uint64_t* a, const uint64_t* b, uint32_t batch) {
    if (!ctx || !ctx->valid || !result || !a || !b || batch == 0) return -1;

    @autoreleasepool {
        uint32_t N = ctx->N;
        size_t size = batch * N * sizeof(uint64_t);

        id<MTLBuffer> result_buf = [ctx->device newBufferWithBytesNoCopy:result length:size
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        id<MTLBuffer> a_buf = [ctx->device newBufferWithBytes:a length:size
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [ctx->device newBufferWithBytes:b length:size
                                                     options:MTLResourceStorageModeShared];

        NTTParams params = {ctx->Q, ctx->mu, ctx->N_inv, ctx->N_inv_precon, N, ctx->log_N, 0, batch};
        memcpy([ctx->params_buf contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->pointwise_mul];
        [enc setBuffer:result_buf offset:0 atIndex:0];
        [enc setBuffer:a_buf offset:0 atIndex:1];
        [enc setBuffer:b_buf offset:0 atIndex:2];
        [enc setBuffer:ctx->params_buf offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N, batch, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(256, N), 1, 1)];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -1;
    }
}

void metal_ntt_get_params(const MetalNTTContext* ctx, uint32_t* N, uint64_t* Q) {
    if (ctx) {
        if (N) *N = ctx->N;
        if (Q) *Q = ctx->Q;
    }
}

bool metal_ntt_is_fused(const MetalNTTContext* ctx) {
    return ctx ? ctx->use_fused : false;
}
