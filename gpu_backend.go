package lattice

import (
	"errors"
)

// Backend defines the interface for GPU-accelerated operations.
// Implementations can use CUDA, Metal, OpenCL, or other GPU APIs.
type Backend interface {
	// NTT computes the Number Theoretic Transform on a coefficient vector.
	// Input is in coefficient domain, output is in NTT domain.
	NTT(coeffs []uint64) []uint64

	// InverseNTT computes the inverse NTT.
	// Input is in NTT domain, output is in coefficient domain.
	InverseNTT(ntt []uint64) []uint64

	// BatchNTT computes NTT on multiple polynomials in parallel.
	// More efficient than calling NTT multiple times.
	BatchNTT(polys [][]uint64) [][]uint64

	// BatchInverseNTT computes inverse NTT on multiple polynomials in parallel.
	BatchInverseNTT(polys [][]uint64) [][]uint64

	// MulMod computes element-wise modular multiplication of two vectors.
	// Both inputs must be in the same domain (typically NTT).
	MulMod(a, b []uint64) []uint64

	// AddMod computes element-wise modular addition.
	AddMod(a, b []uint64) []uint64

	// SubMod computes element-wise modular subtraction.
	SubMod(a, b []uint64) []uint64

	// Name returns the backend name (e.g., "cuda", "metal", "cpu").
	Name() string

	// DeviceInfo returns information about the GPU device.
	DeviceInfo() DeviceInfo

	// Close releases GPU resources.
	Close() error
}

// DeviceInfo contains information about a GPU device.
type DeviceInfo struct {
	Name       string // Device name
	Vendor     string // Vendor name
	MemoryMB   uint64 // Available memory in MB
	MaxThreads uint32 // Maximum threads per block
}

// ErrNoGPU indicates that no GPU backend is available.
var ErrNoGPU = errors.New("lattice: no GPU backend available")

// NewGPUBackend attempts to create a GPU backend for the given parameters.
// It tries available GPU APIs in order of preference: CUDA, Metal, OpenCL.
// Returns ErrNoGPU if no GPU is available.
func NewGPUBackend(params *Params) (Backend, error) {
	// Try to create GPU backend from available implementations
	// This is a placeholder - actual GPU implementations would be in separate files
	// with build tags for different platforms.

	// Try CUDA first (Linux/Windows with NVIDIA GPU)
	if b, err := newCUDABackend(params); err == nil {
		return b, nil
	}

	// Try Metal (macOS/iOS with Apple GPU)
	if b, err := newMetalBackend(params); err == nil {
		return b, nil
	}

	// Try OpenCL (cross-platform fallback)
	if b, err := newOpenCLBackend(params); err == nil {
		return b, nil
	}

	return nil, ErrNoGPU
}

// newCUDABackend attempts to create a CUDA backend.
// Placeholder implementation - would use github.com/luxcpp/gpu/cuda
func newCUDABackend(params *Params) (Backend, error) {
	// TODO: Implement CUDA backend using luxcpp/gpu
	return nil, ErrNoGPU
}

// newMetalBackend attempts to create a Metal backend.
// Placeholder implementation - would use github.com/luxcpp/gpu/metal
func newMetalBackend(params *Params) (Backend, error) {
	// TODO: Implement Metal backend using luxcpp/gpu
	return nil, ErrNoGPU
}

// newOpenCLBackend attempts to create an OpenCL backend.
// Placeholder implementation - would use github.com/luxcpp/gpu/opencl
func newOpenCLBackend(params *Params) (Backend, error) {
	// TODO: Implement OpenCL backend using luxcpp/gpu
	return nil, ErrNoGPU
}

// CPUBackend implements Backend using CPU-only operations.
// This is useful for testing and as a fallback when no GPU is available.
type CPUBackend struct {
	ring *Ring
}

// NewCPUBackend creates a CPU backend that uses the Ring's CPU implementations.
func NewCPUBackend(params *Params) *CPUBackend {
	return &CPUBackend{
		ring: NewRing(params),
	}
}

// NTT implements Backend.NTT using CPU.
func (b *CPUBackend) NTT(coeffs []uint64) []uint64 {
	p := &Poly{
		Coeffs: make([]uint64, len(coeffs)),
		IsNTT:  false,
	}
	copy(p.Coeffs, coeffs)

	b.ring.NTTInPlace(p)
	return p.Coeffs
}

// InverseNTT implements Backend.InverseNTT using CPU.
func (b *CPUBackend) InverseNTT(ntt []uint64) []uint64 {
	p := &Poly{
		Coeffs: make([]uint64, len(ntt)),
		IsNTT:  true,
	}
	copy(p.Coeffs, ntt)

	b.ring.InverseNTTInPlace(p)
	return p.Coeffs
}

// BatchNTT implements Backend.BatchNTT using CPU.
func (b *CPUBackend) BatchNTT(polys [][]uint64) [][]uint64 {
	results := make([][]uint64, len(polys))
	for i, coeffs := range polys {
		results[i] = b.NTT(coeffs)
	}
	return results
}

// BatchInverseNTT implements Backend.BatchInverseNTT using CPU.
func (b *CPUBackend) BatchInverseNTT(polys [][]uint64) [][]uint64 {
	results := make([][]uint64, len(polys))
	for i, ntt := range polys {
		results[i] = b.InverseNTT(ntt)
	}
	return results
}

// MulMod implements Backend.MulMod using CPU.
func (b *CPUBackend) MulMod(a, b_ []uint64) []uint64 {
	n := len(a)
	if len(b_) != n {
		panic("lattice: mismatched vector lengths")
	}

	result := make([]uint64, n)
	Q := b.ring.params.Q
	QInvNeg := b.ring.params.QInvNeg

	for i := 0; i < n; i++ {
		result[i] = montgomeryMul(a[i], b_[i], Q, QInvNeg)
	}

	return result
}

// AddMod implements Backend.AddMod using CPU.
func (b *CPUBackend) AddMod(a, b_ []uint64) []uint64 {
	n := len(a)
	if len(b_) != n {
		panic("lattice: mismatched vector lengths")
	}

	result := make([]uint64, n)
	Q := b.ring.params.Q

	for i := 0; i < n; i++ {
		sum := a[i] + b_[i]
		if sum >= Q {
			sum -= Q
		}
		result[i] = sum
	}

	return result
}

// SubMod implements Backend.SubMod using CPU.
func (b *CPUBackend) SubMod(a, b_ []uint64) []uint64 {
	n := len(a)
	if len(b_) != n {
		panic("lattice: mismatched vector lengths")
	}

	result := make([]uint64, n)
	Q := b.ring.params.Q

	for i := 0; i < n; i++ {
		if a[i] >= b_[i] {
			result[i] = a[i] - b_[i]
		} else {
			result[i] = Q - b_[i] + a[i]
		}
	}

	return result
}

// Name implements Backend.Name.
func (b *CPUBackend) Name() string {
	return "cpu"
}

// DeviceInfo implements Backend.DeviceInfo.
func (b *CPUBackend) DeviceInfo() DeviceInfo {
	return DeviceInfo{
		Name:   "CPU Fallback",
		Vendor: "Generic",
	}
}

// Close implements Backend.Close.
func (b *CPUBackend) Close() error {
	return nil
}

// Ensure CPUBackend implements Backend interface
var _ Backend = (*CPUBackend)(nil)
