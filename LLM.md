# Lux Lattice - NTT & Polynomial Arithmetic

**Last Updated**: 2025-12-30
**Module**: `luxcpp/lattice`
**Role**: GPU-accelerated lattice cryptography primitives

## Architecture Position

```
luxcpp/gpu      ← Foundation (depends on)
    ▲
luxcpp/lattice  ← YOU ARE HERE
    ▲
luxcpp/fhe      ← TFHE/CKKS/BGV (depends on this)
```

## Overview

C++ library providing GPU-accelerated:
- **NTT**: Number Theoretic Transform
- **FFT**: Fast Fourier Transform
- **Polynomial arithmetic**: Ring operations over Z_q[X]/(X^n + 1)

## Build

```bash
cd /Users/z/work/luxcpp/lattice
mkdir -p build && cd build

# Without GPU
cmake ..
make -j$(sysctl -n hw.ncpu)

# With GPU (requires luxcpp/gpu built first)
cmake -DWITH_GPU=ON -DGPU_ROOT=../gpu ..
make -j$(sysctl -n hw.ncpu)
```

## Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| `luxcpp/gpu` | Optional | GPU acceleration |
| Metal/Foundation | macOS | Apple frameworks |

## API

```cpp
#include <lattice.h>

// NTT operations
void lux_ntt_forward(uint64_t* poly, size_t n, uint64_t modulus);
void lux_ntt_inverse(uint64_t* poly, size_t n, uint64_t modulus);

// Polynomial ops
void lux_poly_add(uint64_t* result, const uint64_t* a, const uint64_t* b, size_t n, uint64_t modulus);
void lux_poly_mul(uint64_t* result, const uint64_t* a, const uint64_t* b, size_t n, uint64_t modulus);
```

## Downstream Dependencies

| Package | Uses For |
|---------|----------|
| `luxcpp/fhe` | Polynomial multiplication for FHE |
| `lux/fhe` | Via CGO bridge |

## Performance (M1 Max, WITH_GPU=ON)

| Operation | n=4096 | n=16384 |
|-----------|--------|---------|
| NTT Forward | 0.08ms | 0.25ms |
| NTT Inverse | 0.09ms | 0.27ms |
| Poly Multiply | 0.18ms | 0.55ms |

## Rules for AI Assistants

1. **ALWAYS** build luxcpp/gpu first before enabling WITH_GPU
2. This is the **middle layer** - FHE depends on it
3. Test NTT correctness with known test vectors
4. Changes here affect all FHE operations

---

*This file is symlinked as AGENTS.md, CLAUDE.md*
