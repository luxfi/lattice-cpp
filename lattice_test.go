package lattice

import (
	"testing"
)

// TestParams verifies parameter validation.
func TestParams(t *testing.T) {
	p, err := NewParams(1024, 0xFFFFFFFF00000001, 1753635133440165772)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	if p.N != 1024 {
		t.Errorf("expected N=1024, got %d", p.N)
	}
	if p.LogN != 10 {
		t.Errorf("expected LogN=10, got %d", p.LogN)
	}

	// Test invalid N
	_, err = NewParams(1000, 0xFFFFFFFF00000001, 1753635133440165772)
	if err == nil {
		t.Error("expected error for non-power-of-2 N")
	}
}

// TestRingOperations tests basic ring arithmetic.
func TestRingOperations(t *testing.T) {
	params, err := NewParams(16, 65537, 3)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	ring := NewRing(params)

	// Create test polynomials
	a := ring.NewPoly()
	b := ring.NewPoly()

	// Set some coefficients
	a.Coeffs[0] = 100
	a.Coeffs[1] = 200
	b.Coeffs[0] = 50
	b.Coeffs[1] = 100

	// Test addition
	c := ring.Add(a, b)
	if c.Coeffs[0] != 150 {
		t.Errorf("Add: expected 150, got %d", c.Coeffs[0])
	}
	if c.Coeffs[1] != 300 {
		t.Errorf("Add: expected 300, got %d", c.Coeffs[1])
	}

	// Test subtraction
	d := ring.Sub(a, b)
	if d.Coeffs[0] != 50 {
		t.Errorf("Sub: expected 50, got %d", d.Coeffs[0])
	}
	if d.Coeffs[1] != 100 {
		t.Errorf("Sub: expected 100, got %d", d.Coeffs[1])
	}

	// Test negation
	e := ring.Neg(a)
	f := ring.Add(a, e)
	if !ring.IsZero(f) {
		t.Error("Neg: a + (-a) should be zero")
	}
}

// TestNTT tests NTT forward and inverse transforms.
func TestNTT(t *testing.T) {
	// Use small parameters for testing
	// For N=16, we need Q ≡ 1 (mod 2N=32)
	// Q = 97 is prime and 97 ≡ 1 (mod 32)
	// Find primitive 32nd root of unity: 3^3 = 27 mod 97 works
	// Actually, let's use Q = 769 = 24*32 + 1, primitive root is 11
	// 11^24 mod 769 gives a primitive 32nd root of unity

	// For simplicity, let's compute the 32nd root of unity correctly
	// Q = 769, primitive root is 11, order is 768 = 24 * 32
	// omega_32 = 11^(768/32) = 11^24 mod 769
	Q := uint64(769)
	primRoot := uint64(11)
	// 11^24 mod 769 = ?
	omega := modExp(primRoot, 768/32, Q) // This should give us the 32nd root of unity

	params, err := NewParams(16, Q, omega)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	ring := NewRing(params)

	// Create a polynomial with some coefficients
	p := ring.NewPoly()
	p.Coeffs[0] = 1
	p.Coeffs[1] = 2
	p.Coeffs[2] = 3
	original := p.Copy()

	// Forward NTT
	pNTT := ring.NTT(p)
	if !pNTT.IsNTT {
		t.Error("NTT result should be in NTT domain")
	}

	// Inverse NTT
	pBack := ring.InverseNTT(pNTT)
	if pBack.IsNTT {
		t.Error("InverseNTT result should be in coefficient domain")
	}

	// Check round-trip
	for i := range original.Coeffs {
		if pBack.Coeffs[i] != original.Coeffs[i] {
			t.Errorf("NTT round-trip failed at index %d: expected %d, got %d",
				i, original.Coeffs[i], pBack.Coeffs[i])
		}
	}
}

// TestNTTMul tests polynomial multiplication via NTT.
func TestNTTMul(t *testing.T) {
	Q := uint64(769)
	omega := modExp(11, 768/32, Q)
	params, err := NewParams(16, Q, omega)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	ring := NewRing(params)

	// Create polynomials: (1 + x) * (1 + x) = 1 + 2x + x^2
	a := ring.NewPoly()
	a.Coeffs[0] = 1
	a.Coeffs[1] = 1

	// Forward NTT both
	aNTT := ring.NTT(a)

	// Multiply in NTT domain
	cNTT := ring.Mul(aNTT, aNTT)

	// Convert back
	c := ring.InverseNTT(cNTT)

	// Note: in Z[X]/(X^N+1), we get negacyclic convolution
	// (1+x)^2 = 1 + 2x + x^2 in standard polynomial ring
	// Verify non-zero coefficients exist
	if c.Coeffs[0] == 0 && c.Coeffs[1] == 0 && c.Coeffs[2] == 0 {
		t.Error("Multiplication result should not be all zeros")
	}
}

// TestSampling tests random sampling functions.
func TestSampling(t *testing.T) {
	// Q = 12289 is a popular FHE prime, 12289 = 3 * 4096 + 1
	// For N=64, need Q ≡ 1 (mod 128)
	// 12289 mod 128 = 1, so it works
	Q := uint64(12289)
	// 12288 = 96 * 128, so omega = g^96 where g is primitive root
	// Primitive root of 12289 is 11
	omega := modExp(11, 12288/128, Q)
	params, err := NewParams(64, Q, omega)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	ring := NewRing(params)

	// Test uniform sampling
	u := ring.SampleUniform()
	if u.IsNTT {
		t.Error("SampleUniform should return coefficient domain")
	}
	// Check all coefficients are < Q
	for i, c := range u.Coeffs {
		if c >= params.Q {
			t.Errorf("Uniform sample coefficient %d >= Q: %d", i, c)
		}
	}

	// Test binary sampling
	b := ring.SampleBinary()
	for i, c := range b.Coeffs {
		if c != 0 && c != 1 {
			t.Errorf("Binary sample coefficient %d not in {0,1}: %d", i, c)
		}
	}

	// Test ternary sampling
	ter := ring.SampleTernary()
	for i, c := range ter.Coeffs {
		if c != 0 && c != 1 && c != params.Q-1 {
			t.Errorf("Ternary sample coefficient %d not in {-1,0,1}: %d", i, c)
		}
	}

	// Test Gaussian sampling (just verify it runs)
	g := ring.SampleGaussian(3.2)
	if g == nil {
		t.Error("SampleGaussian returned nil")
	}
}

// TestCPUBackend tests the CPU fallback backend.
func TestCPUBackend(t *testing.T) {
	Q := uint64(769)
	omega := modExp(11, 768/32, Q)
	params, err := NewParams(16, Q, omega)
	if err != nil {
		t.Fatalf("NewParams failed: %v", err)
	}

	backend := NewCPUBackend(params)
	defer backend.Close()

	if backend.Name() != "cpu" {
		t.Errorf("expected name 'cpu', got '%s'", backend.Name())
	}

	// Test NTT via backend
	coeffs := make([]uint64, 16)
	coeffs[0] = 1
	coeffs[1] = 2

	ntt := backend.NTT(coeffs)
	back := backend.InverseNTT(ntt)

	if back[0] != coeffs[0] || back[1] != coeffs[1] {
		t.Errorf("CPU backend NTT round-trip failed: expected [%d,%d], got [%d,%d]",
			coeffs[0], coeffs[1], back[0], back[1])
	}
}

// TestMontgomery tests Montgomery arithmetic.
func TestMontgomery(t *testing.T) {
	Q := uint64(65537)
	R, RInv, QInvNeg := ComputeMontgomeryConstants(Q)

	// Test: (a * R) * (R^-1) ≡ a (mod Q)
	a := uint64(12345)
	aR := toMontgomery(a, Q, R)
	aBack := fromMontgomery(aR, Q, QInvNeg)

	if aBack != a {
		t.Errorf("Montgomery round-trip failed: %d -> %d -> %d", a, aR, aBack)
	}

	// Verify R * RInv ≡ 1 (mod Q)
	product := mulMod(R, RInv, Q)
	if product != 1 {
		t.Errorf("R * RInv should be 1, got %d", product)
	}
}

// BenchmarkNTT benchmarks NTT performance.
func BenchmarkNTT(b *testing.B) {
	params, _ := NewParams(4096, 0x3FFFFFFFFC0001, 17)
	ring := NewRing(params)
	p := ring.SampleUniform()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ring.NTT(p)
	}
}

// BenchmarkInverseNTT benchmarks inverse NTT performance.
func BenchmarkInverseNTT(b *testing.B) {
	params, _ := NewParams(4096, 0x3FFFFFFFFC0001, 17)
	ring := NewRing(params)
	p := ring.SampleUniform()
	pNTT := ring.NTT(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ring.InverseNTT(pNTT)
	}
}

// BenchmarkMul benchmarks NTT multiplication.
func BenchmarkMul(b *testing.B) {
	params, _ := NewParams(4096, 0x3FFFFFFFFC0001, 17)
	ring := NewRing(params)
	a := ring.NTT(ring.SampleUniform())
	bb := ring.NTT(ring.SampleUniform())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ring.Mul(a, bb)
	}
}
