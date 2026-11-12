package superkmeans

import (
	"math"
	"testing"
)

func emBlobs(n, d, k int, seed uint64) []float32 {
	r := newRNG(seed)
	centers := make([]float32, k*d)
	for i := range centers {
		centers[i] = r.norm() * 10
	}
	data := make([]float32, n*d)
	for i := 0; i < n; i++ {
		c := i % k
		for j := 0; j < d; j++ {
			data[i*d+j] = centers[c*d+j] + r.norm()
		}
	}
	return data
}

func TestEMeansRoundPowerOfTwo(t *testing.T) {
	cases := []struct {
		in   float64
		want int
	}{
		{0.5, 1}, {1, 1}, {1.4, 1}, {1.5, 2}, {2.8, 2}, {3, 4}, {3.5, 4},
		{100, 128}, {90, 64}, {1024, 1024},
	}
	for _, c := range cases {
		if got := emRoundPowerOfTwo(c.in); got != c.want {
			t.Errorf("emRoundPowerOfTwo(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestEMeansEpochRows(t *testing.T) {
	n := 1000
	rows := emEpochRows(n, 5)
	if len(rows) != 5 {
		t.Fatalf("rows = %v", rows)
	}
	for _, r := range rows {
		if r != n {
			t.Fatalf("rows = %v", rows)
		}
	}
	rows = emEpochRows(n, 2.5)
	if len(rows) != 3 || rows[0] != n || rows[1] != 500 || rows[2] != n {
		t.Fatalf("rows = %v", rows)
	}
	rows = emEpochRows(n, 0.25)
	if len(rows) != 1 || rows[0] != 250 {
		t.Fatalf("rows = %v", rows)
	}
}

func TestEMeansScheduledAlpha(t *testing.T) {
	a0 := 0.3
	if a := emScheduledAlpha(a0, 0, 100); math.Abs(a-a0) > 1e-12 {
		t.Errorf("alpha(0) = %v, want %v", a, a0)
	}
	if a := emScheduledAlpha(a0, 100, 100); math.Abs(a-0.1*a0) > 1e-12 {
		t.Errorf("alpha(total) = %v, want %v", a, 0.1*a0)
	}
	mid := emScheduledAlpha(a0, 50, 100)
	if mid <= 0.1*a0 || mid >= a0 {
		t.Errorf("alpha(mid) = %v out of range", mid)
	}
}

func TestEMeansRecipeLaws(t *testing.T) {
	n, d, k := 100000, 128, 256
	em, err := NewEMeans(k, d, nil)
	if err != nil {
		t.Fatal(err)
	}
	r := em.resolveRecipe(n)
	if r.BatchLaw != "throughput" {
		t.Errorf("law = %q, want throughput for d=128", r.BatchLaw)
	}
	raw := 0.035134 * math.Pow(float64(k), 0.134749) * math.Pow(float64(n), 0.876473) * math.Pow(5, 0.813344)
	wantB := emRoundPowerOfTwo(raw)
	if r.BatchSize != wantB {
		t.Errorf("batch = %d, want %d (raw %v)", r.BatchSize, wantB, raw)
	}
	wantA := emStepLaw(n, k, 5, r.BatchSize)
	if math.Abs(r.Alpha0-wantA) > 1e-12 {
		t.Errorf("alpha0 = %v, want %v", r.Alpha0, wantA)
	}
	if r.Alpha0 <= 0 || r.Alpha0 > 1 {
		t.Errorf("alpha0 = %v out of (0,1]", r.Alpha0)
	}
	if r.TotalSteps != 5*ceilDiv(n, r.BatchSize) {
		t.Errorf("total steps = %d", r.TotalSteps)
	}

	em2, _ := NewEMeans(k, 8, nil)
	if r2 := em2.resolveRecipe(n); r2.BatchLaw != "quality" {
		t.Errorf("law = %q, want quality for d=8", r2.BatchLaw)
	}
	em3, _ := NewEMeans(k, 2, nil)
	if r3 := em3.resolveRecipe(n); r3.UnusedThreshold != 0 {
		t.Errorf("tau = %v, want 0 for d=2", r3.UnusedThreshold)
	}
	em4, _ := NewEMeans(k, 3, nil)
	if r4 := em4.resolveRecipe(n); r4.UnusedThreshold != emUnusedThreshold {
		t.Errorf("tau = %v, want %v for d=3", r4.UnusedThreshold, emUnusedThreshold)
	}
}

func TestEMeansBlobs(t *testing.T) {
	// k=16 for 8 blobs: e-means revives only starving clusters, so a blob
	// missed by the random init is never rescued when k equals the blob
	// count. The FAISS reference behaves the same way.
	n, d, k := 6000, 32, 16
	data := emBlobs(n, d, 8, 7)
	cfg := DefaultEMeansConfig()
	em, err := NewEMeans(k, d, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	// Well-separated blobs give ~d within-cluster variance per point.
	wcssPerPoint := m.WCSS / float64(n)
	if wcssPerPoint > float64(d)*1.5 {
		t.Errorf("WCSS/point = %v, want ~<=%v for recovered blobs", wcssPerPoint, float64(d)*1.5)
	}
	if len(m.Assignments) != n {
		t.Fatalf("assignments len = %d", len(m.Assignments))
	}
	out := make([]uint32, n)
	dists := make([]float32, n)
	mm := &Model{K: k, D: d, Centroids: m.Centroids}
	if err := mm.Assign(data, n, out, dists); err != nil {
		t.Fatal(err)
	}
	exact := sumFloat32(dists)
	if exact/float64(n) > float64(d)*1.5 {
		t.Errorf("exact WCSS/point = %v", exact/float64(n))
	}
	if exact > m.WCSS*1.05 {
		t.Errorf("exact WCSS %v exceeds moving objective %v", exact, m.WCSS)
	}
	sizes := clusterSizes(out, k)
	for c, s := range sizes {
		if s == 0 {
			t.Errorf("cluster %d empty after revival", c)
		}
	}
}

func TestEMeansBeatsFewLloydIters(t *testing.T) {
	// e-means at 5 epochs should match Lloyd at 5 iterations on the same
	// data; both do 5 assignment passes.
	n, d, k := 8000, 64, 32
	data := emBlobs(n, d, k, 11)
	em, _ := NewEMeans(k, d, nil)
	m, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	emWCSS := exactWCSS(t, data, n, d, k, m.Centroids)

	cfg := DefaultConfig()
	cfg.Iters = 5
	cfg.SamplingFraction = 1
	cfg.EarlyTermination = false
	km, err := New(k, d, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := km.FitCentroids(data, n); err != nil {
		t.Fatal(err)
	}
	lloydWCSS := km.WCSS()
	if emWCSS > lloydWCSS*1.02 {
		t.Errorf("e-means exact WCSS %v > Lloyd(5) %v", emWCSS, lloydWCSS)
	}
}

func exactWCSS(t *testing.T, data []float32, n, d, k int, centroids []float32) float64 {
	t.Helper()
	out := make([]uint32, n)
	dists := make([]float32, n)
	mm := &Model{K: k, D: d, Centroids: centroids}
	if err := mm.Assign(data, n, out, dists); err != nil {
		t.Fatal(err)
	}
	return sumFloat32(dists)
}

func TestEMeansDeterministic(t *testing.T) {
	n, d, k := 3000, 16, 8
	data := emBlobs(n, d, k, 3)
	em, _ := NewEMeans(k, d, nil)
	m1, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	em2, _ := NewEMeans(k, d, nil)
	m2, err := em2.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m1.Centroids {
		if m1.Centroids[i] != m2.Centroids[i] {
			t.Fatalf("non-deterministic at %d: %v vs %v", i, m1.Centroids[i], m2.Centroids[i])
		}
	}
}

func TestEMeansSpherical(t *testing.T) {
	n, d, k := 4000, 32, 8
	data := emBlobs(n, d, k, 5)
	emNormalizeRows(data, n, d)
	cfg := DefaultEMeansConfig()
	cfg.Spherical = true
	em, err := NewEMeans(k, d, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	for c := 0; c < k; c++ {
		row := m.Centroids[c*d : c*d+d]
		norm := math.Sqrt(float64(sqnorm(row)))
		if math.Abs(norm-1) > 1e-4 {
			t.Errorf("centroid %d norm = %v, want 1", c, norm)
		}
	}
	out := make([]uint32, n)
	sims := make([]float32, n)
	if err := m.Assign(data, n, out, sims); err != nil {
		t.Fatal(err)
	}
	var mean float64
	for _, s := range sims {
		mean += float64(s)
	}
	mean /= float64(n)
	if mean < 0.9 {
		t.Errorf("mean cosine similarity = %v, want >0.9 for recovered directions", mean)
	}
}

func TestEMeansMultiTileBatch(t *testing.T) {
	// BatchSize above xBatch (8192 rows) forces multi-tile batch
	// accumulation.
	n, d, k := 20000, 32, 16
	data := emBlobs(n, d, 8, 17)
	cfg := DefaultEMeansConfig()
	cfg.BatchSize = 16384 // > xBatch: two tiles per batch
	em, err := NewEMeans(k, d, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	if got := exactWCSS(t, data, n, d, k, m.Centroids) / float64(n); got > float64(d)*1.5 {
		t.Errorf("exact WCSS/point = %v, want ~<=%v (multi-tile accumulation)", got, float64(d)*1.5)
	}
}

func TestEMeansMaxPointsPerCentroid(t *testing.T) {
	n, d, k := 5000, 16, 8
	data := emBlobs(n, d, k, 13)
	cfg := DefaultEMeansConfig()
	cfg.MaxPointsPerCentroid = 100 // train on 800 of 5000
	em, _ := NewEMeans(k, d, &cfg)
	m, err := em.Fit(data, n)
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, a := range m.Assignments {
		if a != EMeansNoLabel {
			seen++
		}
	}
	if seen != 800 {
		t.Errorf("labeled rows = %d, want 800", seen)
	}
}

func TestEMeansErrors(t *testing.T) {
	if _, err := NewEMeans(0, 8, nil); err != ErrInvalidK {
		t.Errorf("err = %v", err)
	}
	if _, err := NewEMeans(8, 0, nil); err != ErrInvalidD {
		t.Errorf("err = %v", err)
	}
	cfg := DefaultEMeansConfig()
	cfg.NEpochs = 0
	if _, err := NewEMeans(8, 8, &cfg); err != ErrInvalidIters {
		t.Errorf("err = %v", err)
	}
	cfg = DefaultEMeansConfig()
	cfg.BatchLaw = "bogus"
	if _, err := NewEMeans(8, 8, &cfg); err == nil {
		t.Errorf("want error for bad batch law")
	}
	em, _ := NewEMeans(8, 8, nil)
	if _, err := em.Fit(make([]float32, 4*8), 4); err != ErrTooFewPoints {
		t.Errorf("err = %v", err)
	}
}
