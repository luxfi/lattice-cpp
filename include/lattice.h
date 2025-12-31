// =============================================================================
// Lux Lattice Library - GPU-Accelerated Lattice Cryptography
// =============================================================================
//
// High-performance lattice operations with GPU acceleration via MLX.
// This library provides:
// - NTT (Number Theoretic Transform) for polynomial multiplication
// - Ring-LWE operations for threshold signatures
// - Polynomial arithmetic in R_q = Z_q[X]/(X^n + 1)
//
// Backends:
// - Apple Metal (macOS via MLX)
// - CUDA (Linux/NVIDIA via MLX)
// - Optimized CPU fallback
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_LATTICE_H
#define LUX_LATTICE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Library Initialization
// =============================================================================

/**
 * Check if GPU acceleration is available.
 * @return true if GPU (Metal/CUDA) is available
 */
bool lattice_gpu_available(void);

/**
 * Get the name of the active backend.
 * @return "Metal", "CUDA", or "CPU"
 */
const char* lattice_get_backend(void);

/**
 * Clear internal caches (twiddle factors, contexts).
 */
void lattice_clear_cache(void);

// =============================================================================
// NTT Context Management
// =============================================================================

/**
 * Opaque NTT context handle.
 */
typedef struct LatticeNTTContext LatticeNTTContext;

/**
 * Create an NTT context for the given ring parameters.
 * @param N Ring dimension (must be power of 2)
 * @param Q Prime modulus (Q ≡ 1 mod 2N for NTT-friendly)
 * @return Context handle, or NULL on error
 */
LatticeNTTContext* lattice_ntt_create(uint32_t N, uint64_t Q);

/**
 * Free an NTT context.
 * @param ctx Context to free
 */
void lattice_ntt_destroy(LatticeNTTContext* ctx);

// =============================================================================
// NTT Operations
// =============================================================================

/**
 * Forward NTT (time → frequency domain).
 * @param ctx NTT context
 * @param data Input/output polynomial coefficients (modified in-place)
 * @param batch Number of polynomials (for batch processing)
 * @return 0 on success, negative on error
 */
int lattice_ntt_forward(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);

/**
 * Inverse NTT (frequency → time domain).
 * @param ctx NTT context
 * @param data Input/output polynomial coefficients (modified in-place)
 * @param batch Number of polynomials
 * @return 0 on success, negative on error
 */
int lattice_ntt_inverse(LatticeNTTContext* ctx, uint64_t* data, uint32_t batch);

/**
 * Batch forward NTT on multiple polynomials.
 * @param ctx NTT context
 * @param polys Array of polynomial coefficient arrays
 * @param count Number of polynomials
 * @return 0 on success, negative on error
 */
int lattice_ntt_batch_forward(LatticeNTTContext* ctx,
                               uint64_t** polys, uint32_t count);

/**
 * Batch inverse NTT on multiple polynomials.
 * @param ctx NTT context
 * @param polys Array of polynomial coefficient arrays
 * @param count Number of polynomials
 * @return 0 on success, negative on error
 */
int lattice_ntt_batch_inverse(LatticeNTTContext* ctx,
                               uint64_t** polys, uint32_t count);

// =============================================================================
// Polynomial Arithmetic
// =============================================================================

/**
 * Element-wise polynomial multiplication (Hadamard product in NTT domain).
 * @param ctx NTT context
 * @param result Output polynomial (N coefficients)
 * @param a First input polynomial (in NTT domain)
 * @param b Second input polynomial (in NTT domain)
 * @return 0 on success, negative on error
 */
int lattice_poly_mul_ntt(LatticeNTTContext* ctx,
                          uint64_t* result,
                          const uint64_t* a,
                          const uint64_t* b);

/**
 * Full polynomial multiplication (handles NTT conversion internally).
 * @param ctx NTT context
 * @param result Output polynomial (N coefficients)
 * @param a First input polynomial (coefficient form)
 * @param b Second input polynomial (coefficient form)
 * @return 0 on success, negative on error
 */
int lattice_poly_mul(LatticeNTTContext* ctx,
                      uint64_t* result,
                      const uint64_t* a,
                      const uint64_t* b);

/**
 * Polynomial addition: result = a + b (mod Q).
 * @param result Output polynomial (N coefficients)
 * @param a First input polynomial
 * @param b Second input polynomial
 * @param N Ring dimension
 * @param Q Modulus
 * @return 0 on success, negative on error
 */
int lattice_poly_add(uint64_t* result,
                      const uint64_t* a,
                      const uint64_t* b,
                      uint32_t N,
                      uint64_t Q);

/**
 * Polynomial subtraction: result = a - b (mod Q).
 * @param result Output polynomial (N coefficients)
 * @param a First input polynomial
 * @param b Second input polynomial
 * @param N Ring dimension
 * @param Q Modulus
 * @return 0 on success, negative on error
 */
int lattice_poly_sub(uint64_t* result,
                      const uint64_t* a,
                      const uint64_t* b,
                      uint32_t N,
                      uint64_t Q);

/**
 * Scalar multiplication: result = a * scalar (mod Q).
 * @param result Output polynomial (N coefficients)
 * @param a Input polynomial
 * @param scalar Scalar value
 * @param N Ring dimension
 * @param Q Modulus
 * @return 0 on success, negative on error
 */
int lattice_poly_scalar_mul(uint64_t* result,
                             const uint64_t* a,
                             uint64_t scalar,
                             uint32_t N,
                             uint64_t Q);

// =============================================================================
// Ring-LWE Operations (for Ringtail)
// =============================================================================

/**
 * Sample a polynomial with discrete Gaussian distribution.
 * @param result Output polynomial (N coefficients)
 * @param N Ring dimension
 * @param Q Modulus
 * @param sigma Standard deviation
 * @param seed Random seed (NULL for system entropy)
 * @return 0 on success, negative on error
 */
int lattice_sample_gaussian(uint64_t* result,
                             uint32_t N,
                             uint64_t Q,
                             double sigma,
                             const uint8_t* seed);

/**
 * Sample a uniform random polynomial.
 * @param result Output polynomial (N coefficients)
 * @param N Ring dimension
 * @param Q Modulus
 * @param seed Random seed (NULL for system entropy)
 * @return 0 on success, negative on error
 */
int lattice_sample_uniform(uint64_t* result,
                            uint32_t N,
                            uint64_t Q,
                            const uint8_t* seed);

/**
 * Sample a ternary polynomial {-1, 0, 1}.
 * @param result Output polynomial (N coefficients)
 * @param N Ring dimension
 * @param Q Modulus
 * @param density Probability of non-zero coefficient (0.0-1.0)
 * @param seed Random seed (NULL for system entropy)
 * @return 0 on success, negative on error
 */
int lattice_sample_ternary(uint64_t* result,
                            uint32_t N,
                            uint64_t Q,
                            double density,
                            const uint8_t* seed);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Find a primitive 2N-th root of unity modulo Q.
 * @param N Ring dimension
 * @param Q Prime modulus
 * @return Primitive root, or 0 if not found
 */
uint64_t lattice_find_primitive_root(uint32_t N, uint64_t Q);

/**
 * Compute modular inverse: a^{-1} mod Q.
 * @param a Value to invert
 * @param Q Modulus
 * @return Modular inverse, or 0 if not invertible
 */
uint64_t lattice_mod_inverse(uint64_t a, uint64_t Q);

/**
 * Check if Q is a valid NTT-friendly prime for ring dimension N.
 * Q must satisfy: Q is prime and Q ≡ 1 (mod 2N)
 * @param N Ring dimension
 * @param Q Candidate modulus
 * @return true if valid NTT-friendly prime
 */
bool lattice_is_ntt_prime(uint32_t N, uint64_t Q);

/**
 * Get NTT context parameters.
 * @param ctx NTT context
 * @param N Output: ring dimension
 * @param Q Output: modulus
 */
void lattice_ntt_get_params(const LatticeNTTContext* ctx,
                            uint32_t* N,
                            uint64_t* Q);

// =============================================================================
// Error Codes
// =============================================================================

#define LATTICE_SUCCESS          0
#define LATTICE_ERROR_INVALID_N -1
#define LATTICE_ERROR_INVALID_Q -2
#define LATTICE_ERROR_NULL_PTR  -3
#define LATTICE_ERROR_GPU       -4
#define LATTICE_ERROR_MEMORY    -5

#ifdef __cplusplus
}
#endif

#endif // LUX_LATTICE_H
