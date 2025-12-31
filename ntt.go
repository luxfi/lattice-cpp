package lattice

import (
	"math/bits"
)

// NTT computes the Number Theoretic Transform of polynomial p.
// The input is in coefficient domain, output is in NTT domain.
// Returns a new polynomial.
func (r *Ring) NTT(p *Poly) *Poly {
	if p.IsNTT {
		panic("lattice: polynomial is already in NTT domain")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  false, // Set to false initially, NTTInPlace will set to true
	}
	copy(result.Coeffs, p.Coeffs)

	r.NTTInPlace(result)
	return result
}

// NTTInPlace computes the NTT in-place.
func (r *Ring) NTTInPlace(p *Poly) {
	if p.IsNTT {
		panic("lattice: polynomial is already in NTT domain")
	}

	// Use GPU if available
	if r.gpu != nil {
		p.Coeffs = r.gpu.NTT(p.Coeffs)
		p.IsNTT = true
		return
	}

	r.twiddlesOnce.Do(r.computeTwiddles)

	n := int(r.params.N)
	Q := r.params.Q
	coeffs := p.Coeffs

	// Cooley-Tukey iterative NTT (decimation in time)
	// Bit-reversal permutation
	bitReverse(coeffs, r.params.LogN)

	// NTT butterfly operations
	for m := 1; m < n; m <<= 1 {
		for k := 0; k < n; k += 2 * m {
			for j := 0; j < m; j++ {
				// Get twiddle factor (already in standard form)
				w := r.twiddlesNTT[m+j]

				u := coeffs[k+j]
				// Modular multiplication v = coeffs[k+j+m] * w mod Q
				v := mulMod(coeffs[k+j+m], w, Q)

				// Butterfly: (u+v, u-v)
				sum := u + v
				if sum >= Q {
					sum -= Q
				}
				coeffs[k+j] = sum

				var diff uint64
				if u >= v {
					diff = u - v
				} else {
					diff = Q - v + u
				}
				coeffs[k+j+m] = diff
			}
		}
	}

	p.IsNTT = true
}

// InverseNTT computes the Inverse NTT of polynomial p.
// The input is in NTT domain, output is in coefficient domain.
// Returns a new polynomial.
func (r *Ring) InverseNTT(p *Poly) *Poly {
	if !p.IsNTT {
		panic("lattice: polynomial is not in NTT domain")
	}

	result := &Poly{
		Coeffs: make([]uint64, r.params.N),
		IsNTT:  true, // Set to true initially, InverseNTTInPlace will set to false
	}
	copy(result.Coeffs, p.Coeffs)

	r.InverseNTTInPlace(result)
	return result
}

// InverseNTTInPlace computes the inverse NTT in-place.
func (r *Ring) InverseNTTInPlace(p *Poly) {
	if !p.IsNTT {
		panic("lattice: polynomial is not in NTT domain")
	}

	// Use GPU if available
	if r.gpu != nil {
		p.Coeffs = r.gpu.InverseNTT(p.Coeffs)
		p.IsNTT = false
		return
	}

	r.twiddlesOnce.Do(r.computeTwiddles)

	n := int(r.params.N)
	Q := r.params.Q
	coeffs := p.Coeffs

	// Gentleman-Sande iterative inverse NTT (decimation in frequency)
	for m := n >> 1; m >= 1; m >>= 1 {
		for k := 0; k < n; k += 2 * m {
			for j := 0; j < m; j++ {
				// Get inverse twiddle factor
				w := r.twiddlesInvNTT[m+j]

				u := coeffs[k+j]
				v := coeffs[k+j+m]

				// Add: u + v
				sum := u + v
				if sum >= Q {
					sum -= Q
				}
				coeffs[k+j] = sum

				// Sub and multiply: (u - v) * w
				var diff uint64
				if u >= v {
					diff = u - v
				} else {
					diff = Q - v + u
				}
				coeffs[k+j+m] = mulMod(diff, w, Q)
			}
		}
	}

	// Bit-reversal permutation
	bitReverse(coeffs, r.params.LogN)

	// Multiply by N^-1
	for i := 0; i < n; i++ {
		coeffs[i] = mulMod(coeffs[i], r.params.NInv, Q)
	}

	p.IsNTT = false
}

// BatchNTT computes NTT on multiple polynomials.
// If GPU is available, this may be more efficient than individual NTTs.
func (r *Ring) BatchNTT(polys []*Poly) []*Poly {
	if len(polys) == 0 {
		return nil
	}

	results := make([]*Poly, len(polys))

	// Use GPU batch if available
	if r.gpu != nil {
		coeffsSlice := make([][]uint64, len(polys))
		for i, p := range polys {
			if p.IsNTT {
				panic("lattice: polynomial is already in NTT domain")
			}
			coeffsCopy := make([]uint64, len(p.Coeffs))
			copy(coeffsCopy, p.Coeffs)
			coeffsSlice[i] = coeffsCopy
		}

		nttResults := r.gpu.BatchNTT(coeffsSlice)
		for i, coeffs := range nttResults {
			results[i] = &Poly{
				Coeffs: coeffs,
				IsNTT:  true,
			}
		}
		return results
	}

	// CPU fallback: process individually
	for i, p := range polys {
		results[i] = r.NTT(p)
	}
	return results
}

// BatchInverseNTT computes inverse NTT on multiple polynomials.
func (r *Ring) BatchInverseNTT(polys []*Poly) []*Poly {
	if len(polys) == 0 {
		return nil
	}

	results := make([]*Poly, len(polys))

	// Use GPU batch if available (reverse transform)
	if r.gpu != nil {
		coeffsSlice := make([][]uint64, len(polys))
		for i, p := range polys {
			if !p.IsNTT {
				panic("lattice: polynomial is not in NTT domain")
			}
			coeffsCopy := make([]uint64, len(p.Coeffs))
			copy(coeffsCopy, p.Coeffs)
			coeffsSlice[i] = coeffsCopy
		}

		// BatchNTT with inverse flag (GPU backend handles this)
		for i, coeffs := range coeffsSlice {
			invCoeffs := r.gpu.InverseNTT(coeffs)
			results[i] = &Poly{
				Coeffs: invCoeffs,
				IsNTT:  false,
			}
		}
		return results
	}

	// CPU fallback
	for i, p := range polys {
		results[i] = r.InverseNTT(p)
	}
	return results
}

// computeTwiddles precomputes twiddle factors for NTT and inverse NTT.
func (r *Ring) computeTwiddles() {
	n := int(r.params.N)
	Q := r.params.Q
	psi := r.params.RootOfUnity // This should be a primitive 2N-th root of unity

	// Allocate twiddle arrays
	r.twiddlesNTT = make([]uint64, n)
	r.twiddlesInvNTT = make([]uint64, n)

	// psi is the 2N-th primitive root of unity
	// For NTT over X^N+1, we use powers of psi
	// twiddle[i] = psi^(bit_rev(i))

	// Compute psi^i for i = 0 to N-1
	psiPowers := make([]uint64, n)
	psiPow := uint64(1)
	for i := 0; i < n; i++ {
		psiPowers[i] = psiPow
		psiPow = mulMod(psiPow, psi, Q)
	}

	// Compute psi^(-i) for i = 0 to N-1
	psiInv := modInverse(psi, Q)
	psiInvPowers := make([]uint64, n)
	psiInvPow := uint64(1)
	for i := 0; i < n; i++ {
		psiInvPowers[i] = psiInvPow
		psiInvPow = mulMod(psiInvPow, psiInv, Q)
	}

	// Store twiddles in the order needed for the NTT algorithm
	// For Cooley-Tukey, at each stage m, we need psi^(j * N/m) for j in [0, m)
	// The bit-reversal of j gives us the right ordering
	for i := 1; i < n; i++ {
		// For the standard CT-NTT, twiddle at position m+j is psi^(j * N / (2m))
		// which can be indexed as psi^(bitrev(i) * N / N) for certain patterns
		// Simpler: just store psi^i in bit-reversed order
		r.twiddlesNTT[i] = psiPowers[i]
		r.twiddlesInvNTT[i] = psiInvPowers[i]
	}
	r.twiddlesNTT[0] = 1
	r.twiddlesInvNTT[0] = 1
}

// bitReverse performs in-place bit-reversal permutation on coefficients.
func bitReverse(coeffs []uint64, logN uint8) {
	n := len(coeffs)
	for i := 0; i < n; i++ {
		j := int(bits.Reverse64(uint64(i)) >> (64 - logN))
		if i < j {
			coeffs[i], coeffs[j] = coeffs[j], coeffs[i]
		}
	}
}

// NTTMontgomery converts polynomial to NTT domain with Montgomery form.
// Input coefficients are in standard form, output is in NTT + Montgomery form.
func (r *Ring) NTTMontgomery(p *Poly) *Poly {
	if p.IsNTT {
		panic("lattice: polynomial is already in NTT domain")
	}

	result := r.NewPoly()
	Q := r.params.Q
	R := r.params.MontgomeryR

	// Convert to Montgomery form first
	for i := range p.Coeffs {
		result.Coeffs[i] = toMontgomery(p.Coeffs[i], Q, R)
	}

	// Then apply NTT
	r.NTTInPlace(result)
	return result
}

// InverseNTTMontgomery converts from NTT + Montgomery form to standard form.
func (r *Ring) InverseNTTMontgomery(p *Poly) *Poly {
	if !p.IsNTT {
		panic("lattice: polynomial is not in NTT domain")
	}

	result := p.Copy()
	r.InverseNTTInPlace(result)

	// Convert from Montgomery form
	Q := r.params.Q
	QInvNeg := r.params.QInvNeg
	for i := range result.Coeffs {
		result.Coeffs[i] = fromMontgomery(result.Coeffs[i], Q, QInvNeg)
	}

	return result
}
