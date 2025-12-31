// Package lattice provides lattice-based cryptography primitives for FHE schemes.
// It implements polynomial rings over Z_Q[X]/(X^N+1) with NTT acceleration.
package lattice

import (
	"errors"
	"math/bits"
)

// Params defines parameters for a polynomial ring Z_Q[X]/(X^N+1).
type Params struct {
	N           uint32 // Ring dimension (power of 2)
	LogN        uint8  // log2(N)
	Q           uint64 // Modulus (prime, Q ≡ 1 mod 2N for NTT)
	LogQ        uint8  // Approximate bit-length of Q
	RootOfUnity uint64 // Primitive 2N-th root of unity mod Q

	// Montgomery constants for efficient modular arithmetic
	MontgomeryR    uint64 // R = 2^64 mod Q
	MontgomeryRInv uint64 // R^-1 mod Q
	QInvNeg        uint64 // -Q^-1 mod 2^64 (for Montgomery reduction)

	// Precomputed values
	NInv uint64 // N^-1 mod Q (for inverse NTT)
}

// Pre-defined parameter sets for common FHE schemes.
// These provide ~128-bit security based on standard LWE hardness estimates.
var (
	// TFHE128 provides parameters for TFHE with 128-bit security.
	// N=1024, Q≈2^32, suitable for boolean circuits.
	TFHE128 = Params{
		N:              1024,
		LogN:           10,
		Q:              0xFFFFFFFF00000001, // 2^64 - 2^32 + 1 (Goldilocks prime)
		LogQ:           64,
		RootOfUnity:    1753635133440165772,
		MontgomeryR:    0xFFFFFFFF,
		MontgomeryRInv: 0xFFFFFFFF00000002,
		QInvNeg:        0xFFFFFFFFFFFFFFFF,
		NInv:           0xFFFFFFFB00000005,
	}

	// BFV128 provides parameters for BFV scheme with 128-bit security.
	// N=4096, Q≈2^60, suitable for integer arithmetic.
	BFV128 = Params{
		N:              4096,
		LogN:           12,
		Q:              0x0FFFFEE001, // 68719443969, prime ≡ 1 mod 8192
		LogQ:           37,
		RootOfUnity:    49,
		MontgomeryR:    0x000010011FFF,
		MontgomeryRInv: 0x0F00000E0001,
		QInvNeg:        0xF0000020FFFE1FFF,
		NInv:           0x0003FFFBB801,
	}

	// CKKS128 provides parameters for CKKS scheme with 128-bit security.
	// N=4096, Q≈2^60, suitable for approximate arithmetic on complex numbers.
	CKKS128 = Params{
		N:              4096,
		LogN:           12,
		Q:              0x3FFFFFFFFC0001, // ~2^54, prime ≡ 1 mod 8192
		LogQ:           54,
		RootOfUnity:    17,
		MontgomeryR:    0x3FFC0000400001,
		MontgomeryRInv: 0x0003FFFFFFFC01,
		QInvNeg:        0x3FFFFFFFFBFFFF,
		NInv:           0x000FFFFFFFF001,
	}
)

// Validate checks that the parameters are consistent.
func (p *Params) Validate() error {
	// Check N is a power of 2
	if p.N == 0 || (p.N&(p.N-1)) != 0 {
		return errors.New("lattice: N must be a power of 2")
	}

	// Check LogN
	if uint32(1)<<p.LogN != p.N {
		return errors.New("lattice: LogN must equal log2(N)")
	}

	// Check Q is odd (necessary for NTT-friendly prime)
	if p.Q&1 == 0 {
		return errors.New("lattice: Q must be odd")
	}

	// Check Q > N (necessary for polynomial operations)
	if p.Q <= uint64(p.N) {
		return errors.New("lattice: Q must be greater than N")
	}

	return nil
}

// Copy returns a deep copy of the parameters.
func (p *Params) Copy() *Params {
	cp := *p
	return &cp
}

// modExp computes base^exp mod m using binary exponentiation.
func modExp(base, exp, m uint64) uint64 {
	if m == 1 {
		return 0
	}
	result := uint64(1)
	base = base % m
	for exp > 0 {
		if exp&1 == 1 {
			result = mulMod(result, base, m)
		}
		exp >>= 1
		base = mulMod(base, base, m)
	}
	return result
}

// mulMod computes (a * b) mod m without overflow.
func mulMod(a, b, m uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, rem := bits.Div64(hi, lo, m)
	return rem
}

// modInverse computes the modular inverse of a mod m using extended Euclidean algorithm.
func modInverse(a, m uint64) uint64 {
	if m == 1 {
		return 0
	}

	var t, newT int64 = 0, 1
	var r, newR uint64 = m, a % m

	for newR != 0 {
		q := r / newR
		t, newT = newT, t-int64(q)*newT
		r, newR = newR, r-q*newR
	}

	if r > 1 {
		// a is not invertible mod m
		return 0
	}

	if t < 0 {
		t += int64(m)
	}

	return uint64(t)
}

// ComputeMontgomeryConstants computes Montgomery reduction constants for the given modulus.
func ComputeMontgomeryConstants(Q uint64) (R, RInv, QInvNeg uint64) {
	// R = 2^64 mod Q
	R = (^uint64(0) % Q) + 1
	if R == Q {
		R = 0
	}

	// R^-1 mod Q
	RInv = modInverse(R, Q)

	// -Q^-1 mod 2^64
	// We need x such that Q*x ≡ -1 (mod 2^64)
	QInvNeg = computeQInvNeg(Q)

	return
}

// computeQInvNeg computes -Q^-1 mod 2^64 using Newton's method.
func computeQInvNeg(Q uint64) uint64 {
	// Start with inverse mod 2^3 (Q is odd, so Q^-1 mod 8 = Q mod 8 for Q ≡ 1 mod 8,
	// or we use iterative lifting)
	inv := Q // Initial approximation (works for odd Q)

	// Newton iteration: x_{n+1} = x_n * (2 - Q * x_n) mod 2^64
	// Lift from mod 2^k to mod 2^(2k)
	for i := 0; i < 6; i++ { // 6 iterations to reach 2^64
		inv *= 2 - Q*inv
	}

	// Return -Q^-1 mod 2^64
	return -inv
}

// NewParams creates a new Params with computed Montgomery constants.
func NewParams(N uint32, Q, rootOfUnity uint64) (*Params, error) {
	logN := uint8(bits.TrailingZeros32(N))
	if uint32(1)<<logN != N {
		return nil, errors.New("lattice: N must be a power of 2")
	}

	R, RInv, QInvNeg := ComputeMontgomeryConstants(Q)
	NInv := modInverse(uint64(N), Q)

	p := &Params{
		N:              N,
		LogN:           logN,
		Q:              Q,
		LogQ:           uint8(bits.Len64(Q)),
		RootOfUnity:    rootOfUnity,
		MontgomeryR:    R,
		MontgomeryRInv: RInv,
		QInvNeg:        QInvNeg,
		NInv:           NInv,
	}

	if err := p.Validate(); err != nil {
		return nil, err
	}

	return p, nil
}
