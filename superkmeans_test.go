package superkmeans

import (
	"math"
	"math/rand"
	"testing"
)

func makeBlobs(n, d, k int, seed uint64) []float32 {
	r := rand.New(rand.NewSource(int64(seed)))
	centers := make([]float32, k*d)
	for i := range centers {
		centers[i] = float32(r.NormFloat64()) * 10
	}
	data := make([]float32, n*d)
	for i := 0; i < n; i++ {
		c := i % k
		for j := 0; j < d; j++ {
			data[i*d+j] = centers[c*d+j] + float32(r.NormFloat64())
		}
	}
	return data
}

func bruteAssign(data, centroids []float32, n, k, d int) []uint32 {
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		best := float32(math.MaxFloat32)
		for c := 0; c < k; c++ {
			var s float32
			for j := 0; j < d; j++ {
				diff := data[i*d+j] - centroids[c*d+j]
				s += diff * diff
			}
			if s < best {
				best = s
				out[i] = uint32(c)
			}
		}
	}
	return out
}

func wcss(data, centroids []float32, n, k, d int) float64 {
	assign := bruteAssign(data, centroids, n, k, d)
	var total float64
	for i := 0; i < n; i++ {
		c := int(assign[i])
		var s float32
		for j := 0; j < d; j++ {
			diff := data[i*d+j] - centroids[c*d+j]
			s += diff * diff
		}
		total += float64(s)
	}
	return total
}

func mustNew(t *testing.T, k, d int, cfg *Config) *KMeans {
	km, err := New(k, d, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return km
}

func mustFit(t *testing.T, km *KMeans, data []float32, n int) *Model {
	m, err := km.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func mustFitCentroids(t *testing.T, km *KMeans, data []float32, n int) []float32 {
	c, err := km.FitCentroids(data, n)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestGEMMNT(t *testing.T) {
	m, n, k := 37, 23, 19
	a := make([]float32, m*k)
	b := make([]float32, n*k)
	c := make([]float32, m*n)
	ref := make([]float32, m*n)
	r := rand.New(rand.NewSource(1))
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	for i := range b {
		b[i] = float32(r.NormFloat64())
	}
	sgemmNT(m, n, k, a, k, b, k, c, n, 4)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float32
			for p := 0; p < k; p++ {
				s += a[i*k+p] * b[j*k+p]
			}
			ref[i*n+j] = s
		}
	}
	for i := range c {
		if math.Abs(float64(c[i]-ref[i])) > 1e-3 {
			t.Fatalf("gemm mismatch at %d: got %v want %v", i, c[i], ref[i])
		}
	}
}

func TestRotationRoundtrip(t *testing.T) {
	d := 32
	m := randomOrthonormal(d, 7)
	n := 8
	src := make([]float32, n*d)
	r := rand.New(rand.NewSource(3))
	for i := range src {
		src[i] = float32(r.NormFloat64())
	}
	rot := make([]float32, n*d)
	back := make([]float32, n*d)
	sgemmNT(n, d, d, src, d, m, d, rot, d, 2)
	for i := 0; i < n; i++ {
		s := rot[i*d : i*d+d]
		o := back[i*d : i*d+d]
		for j := 0; j < d; j++ {
			var acc float32
			for l := 0; l < d; l++ {
				acc += s[l] * m[l*d+j]
			}
			o[j] = acc
		}
	}
	for i := range src {
		if math.Abs(float64(src[i]-back[i])) > 1e-3 {
			t.Fatalf("roundtrip mismatch at %d: %v vs %v", i, src[i], back[i])
		}
	}
}

func TestFitDeterministic(t *testing.T) {
	n, d, k := 4000, 64, 32
	data := makeBlobs(n, d, k, 5)
	cfg := DefaultConfig()
	cfg.Iters = 4
	cfg.SamplingFraction = 1
	cfg.Threads = 1
	km1 := mustNew(t, k, d, &cfg)
	m1 := mustFit(t, km1, data, n)
	km2 := mustNew(t, k, d, &cfg)
	m2 := mustFit(t, km2, data, n)
	if len(m1.Centroids) != len(m2.Centroids) {
		t.Fatal("centroid length mismatch")
	}
	for i := range m1.Centroids {
		if m1.Centroids[i] != m2.Centroids[i] {
			t.Fatalf("centroid mismatch at %d", i)
		}
	}
}

func TestFitDeterministicAcrossWorkers(t *testing.T) {
	n, d, k := 4000, 64, 32
	data := makeBlobs(n, d, k, 5)
	cfg := DefaultConfig()
	cfg.Iters = 4
	cfg.SamplingFraction = 1
	cfg.Threads = 1
	m1 := mustFit(t, mustNew(t, k, d, &cfg), data, n)
	cfg.Threads = 4
	m2 := mustFit(t, mustNew(t, k, d, &cfg), data, n)
	for i := range m1.Centroids {
		if m1.Centroids[i] != m2.Centroids[i] {
			t.Fatalf("centroid mismatch at %d across workers", i)
		}
	}
}

func TestFitReducesWCSS(t *testing.T) {
	n, d, k := 6000, 64, 32
	data := makeBlobs(n, d, k, 11)
	cfg := DefaultConfig()
	cfg.Iters = 6
	cfg.SamplingFraction = 1
	cfg.Threads = 0
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	got := wcss(data, m.Centroids, n, k, d)
	assign := m.Assignments
	var init float64
	for i := 0; i < n; i++ {
		c := int(assign[i])
		var s float32
		for j := 0; j < d; j++ {
			diff := data[i*d+j] - m.Centroids[c*d+j]
			s += diff * diff
		}
		init += float64(s)
	}
	if math.Abs(got-init) > 1e-3*math.Abs(init) {
		t.Fatalf("model wcss %v does not match recomputed %v", m.WCSS, init)
	}
	if got <= 0 {
		t.Fatal("wcss should be positive")
	}
	if len(m.ClusterSizes) != k {
		t.Fatalf("cluster sizes length %d != %d", len(m.ClusterSizes), k)
	}
}

func TestAssignMatchesBrute(t *testing.T) {
	n, d, k := 1000, 32, 16
	data := makeBlobs(n, d, k, 13)
	cfg := DefaultConfig()
	cfg.Iters = 3
	cfg.SamplingFraction = 1
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	out := make([]uint32, n)
	dists := make([]float32, n)
	if err := m.Assign(data, n, out, dists); err != nil {
		t.Fatal(err)
	}
	ref := bruteAssign(data, m.Centroids, n, k, d)
	for i := range out {
		if out[i] != ref[i] {
			t.Fatalf("assign mismatch at %d: got %d want %d", i, out[i], ref[i])
		}
	}
}

func TestPrunedMatchesExhaustive(t *testing.T) {
	n, d, k := 8000, 128, 512
	data := makeBlobs(n, d, k, 17)
	cfg := DefaultConfig()
	cfg.Iters = 5
	cfg.SamplingFraction = 1
	cfg.Threads = 0
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)

	pruned := m.Assignments

	cfg.GEMMOnly = true
	km2 := mustNew(t, k, d, &cfg)
	m2 := mustFit(t, km2, data, n)
	ref := m2.Assignments

	agree := 0
	for i := range pruned {
		if pruned[i] == ref[i] {
			agree++
		}
	}
	rate := float64(agree) / float64(n)
	if rate < 0.95 {
		t.Fatalf("pruned vs exhaustive agreement too low: %.3f", rate)
	}
}

func TestHierarchicalRuns(t *testing.T) {
	n, d, k := 8000, 128, 256
	data := makeBlobs(n, d, k, 23)
	cfg := DefaultConfig()
	cfg.Iters = 5
	cfg.Hierarchical = true
	cfg.Threads = 0
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	if len(m.Centroids) != k*d {
		t.Fatalf("centroids length %d != %d", len(m.Centroids), k*d)
	}
	if len(m.Assignments) != n {
		t.Fatalf("assignments length %d != %d", len(m.Assignments), n)
	}
	for _, a := range m.Assignments {
		if int(a) >= k {
			t.Fatalf("assignment %d out of range", a)
		}
	}
	if len(m.ClusterSizes) != k {
		t.Fatalf("cluster sizes length %d != %d", len(m.ClusterSizes), k)
	}
}

func TestFitCentroidsMatchesFit(t *testing.T) {
	n, d, k := 4000, 64, 32
	data := makeBlobs(n, d, k, 5)
	cfg := DefaultConfig()
	cfg.Iters = 4
	cfg.SamplingFraction = 1
	cfg.Threads = 0
	km := mustNew(t, k, d, &cfg)
	c := mustFitCentroids(t, km, data, n)
	km2 := mustNew(t, k, d, &cfg)
	m := mustFit(t, km2, data, n)
	for i := range c {
		if c[i] != m.Centroids[i] {
			t.Fatalf("centroid mismatch at %d between FitCentroids and Fit", i)
		}
	}
}

func TestDataAlreadyRotated(t *testing.T) {
	n, d, k := 3000, 32, 16
	data := makeBlobs(n, d, k, 3)
	cfg := DefaultConfig()
	cfg.Iters = 3
	cfg.SamplingFraction = 1
	cfg.DataAlreadyRotated = true
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	if len(m.Centroids) != k*d {
		t.Fatal("bad centroids")
	}
	if len(m.Assignments) != n {
		t.Fatal("bad assignments")
	}
}

func TestAngular(t *testing.T) {
	n, d, k := 3000, 32, 16
	data := makeBlobs(n, d, k, 3)
	for i := 0; i < n; i++ {
		var s float32
		for j := 0; j < d; j++ {
			s += data[i*d+j] * data[i*d+j]
		}
		inv := float32(1 / math.Sqrt(float64(s)))
		for j := 0; j < d; j++ {
			data[i*d+j] *= inv
		}
	}
	cfg := DefaultConfig()
	cfg.Iters = 3
	cfg.SamplingFraction = 1
	cfg.Angular = true
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	for i := 0; i < k; i++ {
		var s float32
		for j := 0; j < d; j++ {
			s += m.Centroids[i*d+j] * m.Centroids[i*d+j]
		}
		if s < 0.99 || s > 1.01 {
			t.Fatalf("centroid %d norm %v not unit", i, s)
		}
	}
}

func TestNewErrors(t *testing.T) {
	if _, err := New(0, 8, nil); err != ErrInvalidK {
		t.Fatalf("want ErrInvalidK, got %v", err)
	}
	if _, err := New(8, 0, nil); err != ErrInvalidD {
		t.Fatalf("want ErrInvalidD, got %v", err)
	}
	cfg := DefaultConfig()
	cfg.Iters = 0
	if _, err := New(8, 8, &cfg); err != ErrInvalidIters {
		t.Fatalf("want ErrInvalidIters, got %v", err)
	}
}

func TestNonFiniteRejected(t *testing.T) {
	n, d, k := 100, 8, 4
	data := makeBlobs(n, d, k, 1)
	cfg := DefaultConfig()
	km, err := New(k, d, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	data[0] = float32(math.NaN())
	if _, err := km.Fit(data, n); err != ErrNonFinite {
		t.Fatalf("want ErrNonFinite, got %v", err)
	}
	data[0] = float32(math.Inf(1))
	if _, err := km.FitCentroids(data, n); err != ErrNonFinite {
		t.Fatalf("want ErrNonFinite, got %v", err)
	}
	m := &Model{K: k, D: d, Centroids: make([]float32, k*d)}
	out := make([]uint32, n)
	if err := m.Assign(data, n, out, nil); err != ErrNonFinite {
		t.Fatalf("want ErrNonFinite, got %v", err)
	}
}

func TestSamplingAssignsAll(t *testing.T) {
	n, d, k := 6000, 64, 32
	data := makeBlobs(n, d, k, 29)
	cfg := DefaultConfig()
	cfg.Iters = 4
	cfg.SamplingFraction = 0.5
	cfg.MaxPointsPerCluster = 100
	cfg.Threads = 0
	km := mustNew(t, k, d, &cfg)
	m := mustFit(t, km, data, n)
	if len(m.Assignments) != n {
		t.Fatalf("assignments length %d != %d", len(m.Assignments), n)
	}
	if len(m.ClusterSizes) != k {
		t.Fatalf("cluster sizes length %d != %d", len(m.ClusterSizes), k)
	}
}

func TestWCSSMatchesModel(t *testing.T) {
	for _, hier := range []bool{false, true} {
		n, d, k := 6000, 64, 32
		data := makeBlobs(n, d, k, 11)
		cfg := DefaultConfig()
		cfg.Hierarchical = hier
		cfg.Iters = 6
		cfg.SamplingFraction = 1
		km := mustNew(t, k, d, &cfg)
		m := mustFit(t, km, data, n)
		want := wcss(data, m.Centroids, n, k, d)
		if math.Abs(km.WCSS()-want) > 1e-3*want {
			t.Fatalf("hier=%v: km.WCSS()=%v, recomputed=%v", hier, km.WCSS(), want)
		}
		if math.Abs(m.WCSS-want) > 1e-3*want {
			t.Fatalf("hier=%v: m.WCSS=%v, recomputed=%v", hier, m.WCSS, want)
		}
	}
}
