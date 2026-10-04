package superkmeans

import (
	"math/rand"
	"testing"
)

func BenchmarkGEMM(b *testing.B) {
	m, n, k := 4096, 1024, 128
	a := make([]float32, m*k)
	c := make([]float32, n*k)
	out := make([]float32, m*n)
	r := rand.New(rand.NewSource(1))
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	for i := range c {
		c[i] = float32(r.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sgemmNT(m, n, k, a, k, c, k, out, n, 0)
	}
	b.ReportMetric(float64(m)*float64(n)*float64(k)*2/1e9, "gflop")
}

func BenchmarkFit(b *testing.B) {
	n, d, k := 8000, 128, 256
	data := make([]float32, n*d)
	r := rand.New(rand.NewSource(1))
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	cfg := DefaultConfig()
	cfg.Iters = 10
	cfg.SamplingFraction = 0.3
	km, err := New(k, d, &cfg)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := km.FitCentroids(data, n); err != nil {
			b.Fatal(err)
		}
	}
}
