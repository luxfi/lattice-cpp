# lux-lattice

GPU-accelerated lattice cryptography library for the Lux Network.

## Features

- **NTT** - Number Theoretic Transform with Metal/CUDA acceleration
- **Polynomial Ring** - Ring operations over cyclotomic polynomials
- **Gaussian Sampling** - Discrete Gaussian sampling for lattice schemes
- **Multi-party** - Threshold cryptography with Shamir secret sharing

## Dependencies

- [lux-gpu](https://github.com/luxcpp/gpu) - GPU acceleration foundation

## Installation

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

## Usage

### CMake

```cmake
find_package(lux-lattice REQUIRED)
target_link_libraries(myapp PRIVATE lux::lattice)
```

### pkg-config (for CGO)

```bash
export CGO_CFLAGS=$(pkg-config --cflags lux-lattice)
export CGO_LDFLAGS=$(pkg-config --libs lux-lattice)
```

## Go Bindings

See [github.com/luxfi/lattice](https://github.com/luxfi/lattice) for Go bindings.

## Documentation

- [Full Documentation](https://luxfi.github.io/crypto/docs/cpp-libraries)
- [GPU Acceleration](https://luxfi.github.io/crypto/docs/gpu-acceleration)

## License

BSD-3-Clause - See [LICENSE](LICENSE)
