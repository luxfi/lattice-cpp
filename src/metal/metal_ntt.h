// =============================================================================
// Lux Lattice - Metal NTT Dispatcher Header
// =============================================================================
//
// Native Metal GPU acceleration for Number Theoretic Transform.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_LATTICE_METAL_NTT_H
#define LUX_LATTICE_METAL_NTT_H

#include <cstdint>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Metal NTT Context
// =============================================================================

typedef struct MetalNTTContext MetalNTTContext;

/**
 * Check if Metal GPU is available for NTT operations.
 */
bool metal_ntt_available(void);

/**
 * Create a Metal NTT context for the given ring parameters.
 * @param N Ring dimension (power of 2, max 16384)
 * @param Q Prime modulus
 * @return Context handle, or NULL if Metal unavailable or parameters invalid
 */
MetalNTTContext* metal_ntt_create(uint32_t N, uint64_t Q);

/**
 * Destroy a Metal NTT context and free GPU resources.
 */
void metal_ntt_destroy(MetalNTTContext* ctx);

/**
 * Forward NTT using Metal GPU.
 * @param ctx Context handle
 * @param data Polynomial coefficients (modified in-place)
 * @param batch Number of polynomials
 * @return 0 on success, negative on error
 */
int metal_ntt_forward(MetalNTTContext* ctx, uint64_t* data, uint32_t batch);

/**
 * Inverse NTT using Metal GPU.
 * @param ctx Context handle
 * @param data Polynomial coefficients (modified in-place)
 * @param batch Number of polynomials
 * @return 0 on success, negative on error
 */
int metal_ntt_inverse(MetalNTTContext* ctx, uint64_t* data, uint32_t batch);

/**
 * Pointwise multiplication in NTT domain.
 * @param ctx Context handle
 * @param result Output buffer
 * @param a First polynomial (NTT domain)
 * @param b Second polynomial (NTT domain)
 * @param batch Number of polynomials
 * @return 0 on success, negative on error
 */
int metal_ntt_pointwise_mul(MetalNTTContext* ctx, uint64_t* result,
                            const uint64_t* a, const uint64_t* b, uint32_t batch);

/**
 * Get context parameters.
 */
void metal_ntt_get_params(const MetalNTTContext* ctx, uint32_t* N, uint64_t* Q);

/**
 * Check if this context uses fused kernels (N <= 4096).
 */
bool metal_ntt_is_fused(const MetalNTTContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // LUX_LATTICE_METAL_NTT_H
