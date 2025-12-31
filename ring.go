package lattice

import (
	"math/bits"
	"sync"
)

// Ring represents a polynomial ring Z_Q[X]/(X^N+1) with NTT support.
type Ring struct {
	params *Params
	gpu    Backend // Optional GPU backend (nil if not available)

	// Precomputed twiddle factors for NTT
	twiddlesNTT    []uint64 // Forward NTT twiddles
	twiddlesInvNTT []uint64 // Inverse NTT twiddles
	twiddlesOnce   sync.Once
}

// Poly represents a polynomial in the ring.
// Coefficients are stored in standard or NTT domain.
type Poly struct {
	Coeffs []uint64
	IsNTT  bool // true if in NTT domain, false if in coefficient domain
}

// NewRing creates a new Ring with the given parameters.
func NewRing(params *Params) *Ring {
	r := &Ring{
		params: params,
	}
	return r
}

// NewRingWithGPU creates a new Ring with GPU acceleration.
func NewRingWithGPU(params *Params, gpu Backend) *Ring {
	r := NewRing(params)
	r.gpu = gpu
	return r
}

// Params returns the ring parameters.
func (r *Ring) Params() *Params {
	return r.params
}

// N returns the ring dimension.
func (r *Ring) N() uint32 {
	return r.params.N
}

// Q returns the ring modulus.
func (r *Ring) Q() uint64 {
	return r.params.Q
}

// NewPoly creates a new zero polynomial in coefficient domain.
func (r *Ring) NewPoly() *Poly {
	return &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  false,
	}
}

// NewPolyNTT creates a new zero polynomial in NTT domain.
func (r *Ring) NewPolyNTT() *Poly {
	return &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  true,
	}
}

// Copy returns a deep copy of the polynomial.
func (p *Poly) Copy() *Poly {
	coeffs := make([]uint64, len(p.Coeffs))
	copy(coeffs, p.Coeffs)
	return &Poly{
		Coeffs: coeffs,
		IsNTT:  p.IsNTT,
	}
}

// CopyTo copies the polynomial to dst.
func (p *Poly) CopyTo(dst *Poly) {
	if len(dst.Coeffs) != len(p.Coeffs) {
		dst.Coeffs = make([]uint64, len(p.Coeffs))
	}
	copy(dst.Coeffs, p.Coeffs)
	dst.IsNTT = p.IsNTT
}

// SetCoeffs sets the polynomial coefficients from the given slice.
// The slice is copied.
func (r *Ring) SetCoeffs(p *Poly, coeffs []uint64) {
	n := int(r.params.N)
	if len(p.Coeffs) != n {
		p.Coeffs = make([]uint64, n)
	}

	copyLen := len(coeffs)
	if copyLen > n {
		copyLen = n
	}
	copy(p.Coeffs[:copyLen], coeffs[:copyLen])

	// Zero out remaining coefficients
	for i := copyLen; i < n; i++ {
		p.Coeffs[i] = 0
	}

	// Reduce coefficients mod Q
	Q := r.params.Q
	for i := 0; i < n; i++ {
		if p.Coeffs[i] >= Q {
			p.Coeffs[i] %= Q
		}
	}

	p.IsNTT = false
}

// Add computes a + b and returns a new polynomial.
// Both polynomials must be in the same domain.
func (r *Ring) Add(a, b *Poly) *Poly {
	if a.IsNTT != b.IsNTT {
		panic("lattice: polynomials must be in the same domain")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  a.IsNTT,
	}

	r.AddTo(a, b, result)
	return result
}

// AddTo computes a + b and stores the result in dst.
func (r *Ring) AddTo(a, b, dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	for i := uint32(0); i < n; i++ {
		sum := a.Coeffs[i] + b.Coeffs[i]
		if sum >= Q {
			sum -= Q
		}
		dst.Coeffs[i] = sum
	}
	dst.IsNTT = a.IsNTT
}

// Sub computes a - b and returns a new polynomial.
// Both polynomials must be in the same domain.
func (r *Ring) Sub(a, b *Poly) *Poly {
	if a.IsNTT != b.IsNTT {
		panic("lattice: polynomials must be in the same domain")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  a.IsNTT,
	}

	r.SubTo(a, b, result)
	return result
}

// SubTo computes a - b and stores the result in dst.
func (r *Ring) SubTo(a, b, dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	for i := uint32(0); i < n; i++ {
		if a.Coeffs[i] >= b.Coeffs[i] {
			dst.Coeffs[i] = a.Coeffs[i] - b.Coeffs[i]
		} else {
			dst.Coeffs[i] = Q - b.Coeffs[i] + a.Coeffs[i]
		}
	}
	dst.IsNTT = a.IsNTT
}

// Neg computes -a and returns a new polynomial.
func (r *Ring) Neg(a *Poly) *Poly {
	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  a.IsNTT,
	}

	r.NegTo(a, result)
	return result
}

// NegTo computes -a and stores the result in dst.
func (r *Ring) NegTo(a, dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	for i := uint32(0); i < n; i++ {
		if a.Coeffs[i] == 0 {
			dst.Coeffs[i] = 0
		} else {
			dst.Coeffs[i] = Q - a.Coeffs[i]
		}
	}
	dst.IsNTT = a.IsNTT
}

// Mul computes a * b using NTT multiplication and returns a new polynomial.
// Both polynomials must be in NTT domain.
func (r *Ring) Mul(a, b *Poly) *Poly {
	if !a.IsNTT || !b.IsNTT {
		panic("lattice: polynomials must be in NTT domain for multiplication")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  true,
	}

	r.MulTo(a, b, result)
	return result
}

// MulTo computes a * b and stores the result in dst.
// Both polynomials must be in NTT domain.
func (r *Ring) MulTo(a, b, dst *Poly) {
	if !a.IsNTT || !b.IsNTT {
		panic("lattice: polynomials must be in NTT domain for multiplication")
	}

	// Use GPU if available
	if r.gpu != nil {
		dst.Coeffs = r.gpu.MulMod(a.Coeffs, b.Coeffs)
		dst.IsNTT = true
		return
	}

	n := r.params.N
	Q := r.params.Q

	for i := uint32(0); i < n; i++ {
		dst.Coeffs[i] = montgomeryMul(a.Coeffs[i], b.Coeffs[i], Q, r.params.QInvNeg)
	}
	dst.IsNTT = true
}

// MulCoeffs computes coefficient-wise multiplication (Hadamard product).
// This is useful for operations that don't require negacyclic convolution.
// Both polynomials must be in the same domain.
func (r *Ring) MulCoeffs(a, b *Poly) *Poly {
	if a.IsNTT != b.IsNTT {
		panic("lattice: polynomials must be in the same domain")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  a.IsNTT,
	}

	r.MulCoeffsTo(a, b, result)
	return result
}

// MulCoeffsTo computes coefficient-wise multiplication and stores in dst.
func (r *Ring) MulCoeffsTo(a, b, dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	for i := uint32(0); i < n; i++ {
		dst.Coeffs[i] = montgomeryMul(a.Coeffs[i], b.Coeffs[i], Q, r.params.QInvNeg)
	}
	dst.IsNTT = a.IsNTT
}

// MulScalar computes a * scalar and returns a new polynomial.
func (r *Ring) MulScalar(a *Poly, scalar uint64) *Poly {
	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  a.IsNTT,
	}

	r.MulScalarTo(a, scalar, result)
	return result
}

// MulScalarTo computes a * scalar and stores the result in dst.
func (r *Ring) MulScalarTo(a *Poly, scalar uint64, dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	// Convert scalar to Montgomery form
	scalarMont := toMontgomery(scalar, Q, r.params.MontgomeryR)

	for i := uint32(0); i < n; i++ {
		dst.Coeffs[i] = montgomeryMul(a.Coeffs[i], scalarMont, Q, r.params.QInvNeg)
	}
	dst.IsNTT = a.IsNTT
}

// MulPoly computes polynomial multiplication in coefficient domain.
// This is slower than NTT multiplication but works in coefficient domain.
// Result is a negacyclic convolution: (a * b) mod (X^N + 1).
func (r *Ring) MulPoly(a, b *Poly) *Poly {
	if a.IsNTT || b.IsNTT {
		panic("lattice: polynomials must be in coefficient domain")
	}

	// Convert to NTT, multiply, convert back
	aNTT := r.NTT(a)
	bNTT := r.NTT(b)
	resultNTT := r.Mul(aNTT, bNTT)
	return r.InverseNTT(resultNTT)
}

// montgomeryMul computes (a * b * R^-1) mod Q using Montgomery reduction.
func montgomeryMul(a, b, Q, QInvNeg uint64) uint64 {
	hi, lo := bits.Mul64(a, b)

	// Montgomery reduction
	m := lo * QInvNeg
	_, carry := bits.Mul64(m, Q)

	result := hi - carry
	if hi < carry {
		result += Q
	}

	// Final reduction
	if result >= Q {
		result -= Q
	}

	return result
}

// toMontgomery converts x to Montgomery form: x * R mod Q.
func toMontgomery(x, Q, R uint64) uint64 {
	hi, lo := bits.Mul64(x, R)
	_, rem := bits.Div64(hi, lo, Q)
	return rem
}

// fromMontgomery converts from Montgomery form to standard form.
func fromMontgomery(x, Q, QInvNeg uint64) uint64 {
	return montgomeryMul(x, 1, Q, QInvNeg)
}

// Equal returns true if two polynomials are equal.
func (r *Ring) Equal(a, b *Poly) bool {
	if a.IsNTT != b.IsNTT {
		return false
	}

	if len(a.Coeffs) != len(b.Coeffs) {
		return false
	}

	for i := range a.Coeffs {
		if a.Coeffs[i] != b.Coeffs[i] {
			return false
		}
	}

	return true
}

// IsZero returns true if the polynomial is zero.
func (r *Ring) IsZero(a *Poly) bool {
	for _, c := range a.Coeffs {
		if c != 0 {
			return false
		}
	}
	return true
}
