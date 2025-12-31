package lattice

import (
	"crypto/rand"
	"encoding/binary"
	"math"
)

// SampleUniform samples a polynomial with coefficients uniformly random in [0, Q).
func (r *Ring) SampleUniform() *Poly {
	p := r.NewPoly()
	r.SampleUniformTo(p)
	return p
}

// SampleUniformTo samples a uniform polynomial into dst.
func (r *Ring) SampleUniformTo(dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	// Calculate bytes needed per coefficient
	// We use rejection sampling to get uniform distribution mod Q
	bytesPerSample := (r.params.LogQ + 7) / 8

	// Buffer for random bytes
	buf := make([]byte, int(bytesPerSample)*int(n)*2) // Extra space for rejection sampling

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	bufIdx := 0
	for i := uint32(0); i < n; i++ {
		for {
			if bufIdx+int(bytesPerSample) > len(buf) {
				// Refill buffer
				if _, err := rand.Read(buf); err != nil {
					panic("lattice: failed to read random bytes: " + err.Error())
				}
				bufIdx = 0
			}

			// Read bytesPerSample bytes and interpret as uint64
			var val uint64
			switch bytesPerSample {
			case 8:
				val = binary.LittleEndian.Uint64(buf[bufIdx:])
			case 7:
				val = binary.LittleEndian.Uint64(append(buf[bufIdx:bufIdx+7], 0))
			case 6:
				val = binary.LittleEndian.Uint64(append(buf[bufIdx:bufIdx+6], 0, 0))
			case 5:
				val = binary.LittleEndian.Uint64(append(buf[bufIdx:bufIdx+5], 0, 0, 0))
			case 4:
				val = uint64(binary.LittleEndian.Uint32(buf[bufIdx:]))
			default:
				// Generic fallback
				val = 0
				for j := uint8(0); j < bytesPerSample; j++ {
					val |= uint64(buf[bufIdx+int(j)]) << (8 * j)
				}
			}
			bufIdx += int(bytesPerSample)

			// Mask to LogQ bits
			mask := (uint64(1) << r.params.LogQ) - 1
			val &= mask

			// Rejection sampling: accept only if val < Q
			if val < Q {
				dst.Coeffs[i] = val
				break
			}
		}
	}

	dst.IsNTT = false
}

// SampleGaussian samples a polynomial with coefficients from a discrete Gaussian
// distribution with standard deviation sigma, centered at 0.
func (r *Ring) SampleGaussian(sigma float64) *Poly {
	p := r.NewPoly()
	r.SampleGaussianTo(p, sigma)
	return p
}

// SampleGaussianTo samples a Gaussian polynomial into dst.
func (r *Ring) SampleGaussianTo(dst *Poly, sigma float64) {
	n := r.params.N
	Q := r.params.Q

	// Use Box-Muller transform for Gaussian samples
	// We need pairs of uniform random numbers in (0, 1)
	buf := make([]byte, int(n)*8) // 8 bytes per sample for high precision

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	for i := uint32(0); i < n; i += 2 {
		// Read two uniform random values in [0, 1)
		u1 := float64(binary.LittleEndian.Uint64(buf[i*8:])) / float64(1<<64)
		u2 := float64(binary.LittleEndian.Uint64(buf[(i+1)*8:])) / float64(1<<64)

		// Avoid log(0)
		if u1 < 1e-300 {
			u1 = 1e-300
		}

		// Box-Muller transform
		r1 := math.Sqrt(-2.0*math.Log(u1)) * sigma
		theta := 2.0 * math.Pi * u2

		z1 := r1 * math.Cos(theta)
		z2 := r1 * math.Sin(theta)

		// Round to nearest integer and reduce mod Q
		dst.Coeffs[i] = gaussianToModQ(z1, Q)
		if i+1 < n {
			dst.Coeffs[i+1] = gaussianToModQ(z2, Q)
		}
	}

	dst.IsNTT = false
}

// gaussianToModQ converts a Gaussian sample to an element in Z_Q.
// Negative values are mapped to Q - |value|.
func gaussianToModQ(z float64, Q uint64) uint64 {
	// Round to nearest integer
	rounded := math.Round(z)

	if rounded >= 0 {
		val := uint64(rounded)
		if val >= Q {
			val %= Q
		}
		return val
	}

	// Negative value: return Q - |rounded|
	val := uint64(-rounded)
	if val >= Q {
		val %= Q
	}
	if val == 0 {
		return 0
	}
	return Q - val
}

// SampleTernary samples a polynomial with coefficients in {-1, 0, 1}.
// Each coefficient is independently sampled:
//   - P(0) = 1/2
//   - P(1) = 1/4
//   - P(-1) = 1/4
func (r *Ring) SampleTernary() *Poly {
	p := r.NewPoly()
	r.SampleTernaryTo(p)
	return p
}

// SampleTernaryTo samples a ternary polynomial into dst.
func (r *Ring) SampleTernaryTo(dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	// 2 bits per coefficient: 00,01 -> 0, 10 -> 1, 11 -> -1
	bytesNeeded := (n + 3) / 4
	buf := make([]byte, bytesNeeded)

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	for i := uint32(0); i < n; i++ {
		byteIdx := i / 4
		bitIdx := (i % 4) * 2

		bits := (buf[byteIdx] >> bitIdx) & 0x03

		switch bits {
		case 0, 1:
			dst.Coeffs[i] = 0
		case 2:
			dst.Coeffs[i] = 1
		case 3:
			dst.Coeffs[i] = Q - 1 // -1 mod Q
		}
	}

	dst.IsNTT = false
}

// SampleTernaryUniform samples a polynomial with coefficients in {-1, 0, 1}.
// Each value has equal probability 1/3.
func (r *Ring) SampleTernaryUniform() *Poly {
	p := r.NewPoly()
	r.SampleTernaryUniformTo(p)
	return p
}

// SampleTernaryUniformTo samples a uniform ternary polynomial into dst.
func (r *Ring) SampleTernaryUniformTo(dst *Poly) {
	n := r.params.N
	Q := r.params.Q

	// Use rejection sampling to get uniform distribution over {0, 1, 2}
	buf := make([]byte, int(n)*2) // Extra space for rejection

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	bufIdx := 0
	for i := uint32(0); i < n; i++ {
		for {
			if bufIdx >= len(buf) {
				if _, err := rand.Read(buf); err != nil {
					panic("lattice: failed to read random bytes: " + err.Error())
				}
				bufIdx = 0
			}

			// Take 2 bits, reject if >= 3
			val := buf[bufIdx] & 0x03
			bufIdx++

			if val < 3 {
				switch val {
				case 0:
					dst.Coeffs[i] = 0
				case 1:
					dst.Coeffs[i] = 1
				case 2:
					dst.Coeffs[i] = Q - 1 // -1 mod Q
				}
				break
			}
		}
	}

	dst.IsNTT = false
}

// SampleBinary samples a polynomial with coefficients in {0, 1}.
// Each coefficient is independently uniform in {0, 1}.
func (r *Ring) SampleBinary() *Poly {
	p := r.NewPoly()
	r.SampleBinaryTo(p)
	return p
}

// SampleBinaryTo samples a binary polynomial into dst.
func (r *Ring) SampleBinaryTo(dst *Poly) {
	n := r.params.N

	// 1 bit per coefficient
	bytesNeeded := (n + 7) / 8
	buf := make([]byte, bytesNeeded)

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	for i := uint32(0); i < n; i++ {
		byteIdx := i / 8
		bitIdx := i % 8

		if (buf[byteIdx]>>bitIdx)&1 == 1 {
			dst.Coeffs[i] = 1
		} else {
			dst.Coeffs[i] = 0
		}
	}

	dst.IsNTT = false
}

// SampleHWT samples a polynomial with exactly h non-zero coefficients,
// where each non-zero coefficient is in {-1, 1}.
// This is the Hamming Weight distribution used in some schemes.
func (r *Ring) SampleHWT(h uint32) *Poly {
	p := r.NewPoly()
	r.SampleHWTTo(p, h)
	return p
}

// SampleHWTTo samples an HWT polynomial into dst.
func (r *Ring) SampleHWTTo(dst *Poly, h uint32) {
	n := r.params.N
	Q := r.params.Q

	if h > n {
		panic("lattice: h must be <= N")
	}

	// Zero all coefficients first
	for i := range dst.Coeffs {
		dst.Coeffs[i] = 0
	}

	// Fisher-Yates shuffle to select h positions
	positions := make([]uint32, n)
	for i := uint32(0); i < n; i++ {
		positions[i] = i
	}

	buf := make([]byte, 8)
	for i := uint32(0); i < h; i++ {
		// Random index in [i, n)
		if _, err := rand.Read(buf); err != nil {
			panic("lattice: failed to read random bytes: " + err.Error())
		}

		j := i + uint32(binary.LittleEndian.Uint64(buf))%(n-i)
		positions[i], positions[j] = positions[j], positions[i]

		// Random sign
		if _, err := rand.Read(buf[:1]); err != nil {
			panic("lattice: failed to read random bytes: " + err.Error())
		}

		if buf[0]&1 == 0 {
			dst.Coeffs[positions[i]] = 1
		} else {
			dst.Coeffs[positions[i]] = Q - 1 // -1 mod Q
		}
	}

	dst.IsNTT = false
}

// SampleCenteredBinomial samples a polynomial from a centered binomial distribution
// with parameter eta. Each coefficient is sum of eta random bits minus eta random bits,
// giving values in [-eta, eta].
func (r *Ring) SampleCenteredBinomial(eta uint8) *Poly {
	p := r.NewPoly()
	r.SampleCenteredBinomialTo(p, eta)
	return p
}

// SampleCenteredBinomialTo samples a CBD polynomial into dst.
func (r *Ring) SampleCenteredBinomialTo(dst *Poly, eta uint8) {
	n := r.params.N
	Q := r.params.Q

	// Need 2*eta bits per coefficient
	bitsPerCoeff := uint32(eta) * 2
	bytesNeeded := (uint32(n)*bitsPerCoeff + 7) / 8
	buf := make([]byte, bytesNeeded)

	if _, err := rand.Read(buf); err != nil {
		panic("lattice: failed to read random bytes: " + err.Error())
	}

	bitIdx := uint32(0)
	for i := uint32(0); i < n; i++ {
		var a, b uint32

		// Count bits in first eta bits
		for j := uint8(0); j < eta; j++ {
			bytePos := bitIdx / 8
			bitPos := bitIdx % 8
			if (buf[bytePos]>>bitPos)&1 == 1 {
				a++
			}
			bitIdx++
		}

		// Count bits in next eta bits
		for j := uint8(0); j < eta; j++ {
			bytePos := bitIdx / 8
			bitPos := bitIdx % 8
			if (buf[bytePos]>>bitPos)&1 == 1 {
				b++
			}
			bitIdx++
		}

		// Result is a - b, in range [-eta, eta]
		if a >= b {
			dst.Coeffs[i] = uint64(a - b)
		} else {
			dst.Coeffs[i] = Q - uint64(b-a)
		}
	}

	dst.IsNTT = false
}
