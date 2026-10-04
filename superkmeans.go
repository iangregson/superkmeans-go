package superkmeans

import "math"

type Config struct {
	Iters               int
	SamplingFraction    float64
	MaxPointsPerCluster int
	Threads             int
	Seed                uint64
	GEMMOnly            bool
	Tolerance           float64
	EarlyTermination    bool
	MinNotPrunedPct     float64
	MaxNotPrunedPct     float64
	UnrotateCentroids   bool
	Angular             bool
	DataAlreadyRotated  bool
	AggressiveSplit     bool
	Hierarchical        bool
	HierItersMeso       int
	HierItersFine       int
	HierItersRefine     int
	InitialCentroids    []float32
}

func DefaultConfig() Config {
	return Config{
		Iters:               10,
		SamplingFraction:    0.3,
		MaxPointsPerCluster: 256,
		Threads:             0,
		Seed:                42,
		Tolerance:           1e-4,
		EarlyTermination:    true,
		MinNotPrunedPct:     0.03,
		MaxNotPrunedPct:     0.05,
		UnrotateCentroids:   true,
		HierItersMeso:       3,
		HierItersFine:       5,
		HierItersRefine:     0,
	}
}

type IterationStats struct {
	Objective    float64
	Shift        float64
	Splits       int
	NotPrunedPct float64
	PartialD     int
	GEMMOnly     bool
}

type Model struct {
	K              int
	D              int
	Centroids      []float32
	Assignments    []uint32
	ClusterSizes   []uint32
	WCSS           float64
	Iterations     int
	IterationStats []IterationStats
}

type KMeans struct {
	k           int
	d           int
	cfg         Config
	workers     int
	seedInit    uint64
	seedSplit   uint64
	rot         []float32
	rotT        []float32
	ratios      []float32
	horizontalD int
	verticalD   int
	partialD    int

	lastIterations int
	lastWCSS       float64
}

func New(k, d int, cfg *Config) (*KMeans, error) {
	c := DefaultConfig()
	if cfg != nil {
		c = *cfg
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	if d <= 0 {
		return nil, ErrInvalidD
	}
	if c.Iters <= 0 {
		return nil, ErrInvalidIters
	}
	if c.SamplingFraction <= 0 || c.SamplingFraction > 1 {
		return nil, ErrSamplingFraction
	}
	if c.MaxPointsPerCluster <= 0 {
		return nil, ErrMaxPoints
	}
	if c.DataAlreadyRotated {
		c.UnrotateCentroids = false
	}
	if c.Hierarchical {
		c.SamplingFraction = 1
		c.AggressiveSplit = true
	}
	km := &KMeans{
		k:         k,
		d:         d,
		cfg:       c,
		workers:   normWorkers(c.Threads),
		seedInit:  deriveSeed(c.Seed, 2),
		seedSplit: deriveSeed(c.Seed, 3),
	}
	eps := float64(eps0)
	if c.Hierarchical {
		eps = eps0Hier
	}
	if !c.DataAlreadyRotated {
		km.rot = randomOrthonormal(d, deriveSeed(c.Seed, 1))
		km.rotT = transposeSquare(km.rot, d)
	}
	km.ratios = ratios(d, eps)
	km.horizontalD, km.verticalD = dimensionSplit(d)
	km.partialD = min(max(minPartialD, km.verticalD/2), km.verticalD)
	return km, nil
}

func (km *KMeans) Fit(data []float32, n int) (*Model, error) {
	st, centroids, err := km.train(data, n)
	if err != nil {
		return nil, err
	}
	return km.finalize(st, data, n, st.nSamples, centroids), nil
}

func (km *KMeans) FitCentroids(data []float32, n int) ([]float32, error) {
	_, centroids, err := km.train(data, n)
	return centroids, err
}

func (km *KMeans) train(data []float32, n int) (*state, []float32, error) {
	if km.cfg.Hierarchical {
		return km.trainHier(data, n)
	}
	k, d := km.k, km.d
	if n < k {
		return nil, nil, ErrTooFewPoints
	}
	if len(data) < n*d {
		return nil, nil, dataLengthError(len(data), n*d)
	}
	if err := validateFinite(data, n, d); err != nil {
		return nil, nil, err
	}
	nSamples := km.sampleCount(n)
	if nSamples < k {
		return nil, nil, ErrTooFewSamples
	}
	var perm []int
	if km.cfg.InitialCentroids == nil || nSamples < n {
		perm = newRNG(km.seedInit).perm(n)
	}
	st := km.newState(nSamples, k)
	if err := km.initCentroids(data, perm, st.prev); err != nil {
		return nil, nil, err
	}
	st.x = km.sampleAndRotate(data, n, nSamples, perm)
	squaredNorms(st.x, nSamples, d, st.dataNorms, km.workers)

	alwaysGEMM := d < dGate || km.cfg.GEMMOnly || k <= kGate
	partialNorms := false
	for iter := 0; iter < km.cfg.Iters; iter++ {
		gemmOnly := iter == 0 || alwaysGEMM
		if !gemmOnly && !partialNorms {
			squaredNormsPrefix(st.x, nSamples, d, st.partialD, st.dataNorms, km.workers)
			partialNorms = true
		}
		st.runIteration(iter, gemmOnly)
		if km.cfg.EarlyTermination && st.converged(iter) {
			break
		}
	}
	km.lastIterations = len(st.stats)
	km.lastWCSS = st.cost
	centroids := make([]float32, k*d)
	if km.cfg.UnrotateCentroids && km.rot != nil {
		km.unrotateInto(centroids, st.cur, k)
	} else {
		copy(centroids, st.cur)
	}
	return st, centroids, nil
}

func (km *KMeans) Iterations() int {
	return km.lastIterations
}

func (km *KMeans) WCSS() float64 {
	return km.lastWCSS
}

func (km *KMeans) sampleCount(n int) int {
	if km.cfg.SamplingFraction == 1 {
		return n
	}
	byN := int(float64(n) * km.cfg.SamplingFraction)
	byK := km.k * km.cfg.MaxPointsPerCluster
	return min(byN, byK)
}

type state struct {
	km        *KMeans
	k         int
	nSamples  int
	x         []float32
	cur       []float32
	prev      []float32
	sizes     []uint32
	assign    []uint32
	dist      []float32
	dataNorms []float32
	cNorms    []float32
	buf       []float32
	positions [][]uint32
	counts    [][]int64
	partialD  int
	nSplit    int
	cost      float64
	prevCost  float64
	shift     float64
	stats     []IterationStats
	finalized bool
}

func (km *KMeans) newState(nSamples, k int) *state {
	st := &state{
		km:        km,
		k:         k,
		nSamples:  nSamples,
		cur:       make([]float32, k*km.d),
		prev:      make([]float32, k*km.d),
		sizes:     make([]uint32, k),
		assign:    make([]uint32, nSamples),
		dist:      make([]float32, nSamples),
		dataNorms: make([]float32, nSamples),
		cNorms:    make([]float32, k),
		buf:       make([]float32, min(xBatch, nSamples)*min(yBatch, k)),
		positions: make([][]uint32, km.workers),
		counts:    make([][]int64, km.workers),
		partialD:  km.partialD,
	}
	for w := 0; w < km.workers; w++ {
		st.positions[w] = make([]uint32, min(yBatch, k))
		st.counts[w] = make([]int64, 1)
	}
	return st
}

func (km *KMeans) initCentroids(data []float32, perm []int, dst []float32) error {
	if km.cfg.InitialCentroids != nil {
		if len(km.cfg.InitialCentroids) < km.k*km.d {
			return ErrInitialCentroids
		}
		km.rotateInto(dst, km.cfg.InitialCentroids, km.k)
		return nil
	}
	km.gatherRotate(dst, data, perm, km.k)
	return nil
}

func (km *KMeans) gatherRotate(dst, data []float32, idx []int, count int) {
	d := km.d
	if km.rot == nil {
		for i := 0; i < count; i++ {
			copy(dst[i*d:i*d+d], data[idx[i]*d:idx[i]*d+d])
		}
		return
	}
	tmp := make([]float32, count*d)
	for i := 0; i < count; i++ {
		copy(tmp[i*d:i*d+d], data[idx[i]*d:idx[i]*d+d])
	}
	km.rotateInto(dst, tmp, count)
}

func (km *KMeans) rotateInto(dst, src []float32, n int) {
	if km.rot == nil {
		copy(dst, src[:n*km.d])
		return
	}
	sgemmNT(n, km.d, km.d, src, km.d, km.rot, km.d, dst, km.d, km.workers)
}

func (km *KMeans) rotateInPlace(data []float32, n int) {
	d := km.d
	block := (8 << 20) / (d * 4)
	if block < 1 {
		block = 1
	}
	if block > n {
		block = n
	}
	scratch := make([]float32, block*d)
	for i := 0; i < n; i += block {
		rows := min(block, n-i)
		src := data[i*d : (i+rows)*d]
		km.rotateInto(scratch[:rows*d], src, rows)
		copy(src, scratch[:rows*d])
	}
}

func (km *KMeans) unrotateInto(dst, src []float32, n int) {
	if km.rot == nil {
		copy(dst, src[:n*km.d])
		return
	}
	sgemmNT(n, km.d, km.d, src, km.d, km.rotT, km.d, dst, km.d, km.workers)
}

func (km *KMeans) sampleAndRotate(data []float32, n, nSamples int, perm []int) []float32 {
	d := km.d
	if nSamples < n {
		tmp := make([]float32, nSamples*d)
		parallelFor(km.workers, nSamples, d, func(w, lo, hi int) {
			for i := lo; i < hi; i++ {
				copy(tmp[i*d:i*d+d], data[perm[i]*d:perm[i]*d+d])
			}
		})
		if km.rot != nil {
			km.rotateInPlace(tmp, nSamples)
		}
		return tmp
	}
	if km.rot == nil {
		return data[: n*d : n*d]
	}
	out := make([]float32, n*d)
	km.rotateInto(out, data[:n*d], n)
	return out
}

func (st *state) runIteration(iter int, gemmOnly bool) {
	km := st.km
	d := km.d
	if iter > 0 {
		st.cur, st.prev = st.prev, st.cur
	}
	oldPartialD := st.partialD
	st.estep(st.prev, gemmOnly)
	st.sumRows()
	notPruned := 1.0
	if !gemmOnly {
		notPruned = st.tunePartialD()
		if st.partialD != oldPartialD {
			squaredNormsPrefix(st.x, st.nSamples, d, st.partialD, st.dataNorms, km.workers)
		}
	}
	st.consolidate()
	st.prevCost, st.cost = st.cost, sumFloat32(st.dist[:st.nSamples])
	st.shift = st.centroidShift()
	statPartialD := 0
	if !gemmOnly {
		statPartialD = oldPartialD
	}
	st.stats = append(st.stats, IterationStats{
		Objective:    st.cost,
		Shift:        st.shift,
		Splits:       st.nSplit,
		NotPrunedPct: notPruned,
		PartialD:     statPartialD,
		GEMMOnly:     gemmOnly,
	})
}

func (st *state) estep(y []float32, gemmOnly bool) {
	km := st.km
	if gemmOnly {
		squaredNorms(y, st.k, km.d, st.cNorms, km.workers)
		st.exhaustive(y)
		return
	}
	squaredNormsPrefix(y, st.k, km.d, st.partialD, st.cNorms, km.workers)
	for w := range st.counts {
		st.counts[w][0] = 0
	}
	st.pruned(y)
}

func (st *state) exhaustive(y []float32) {
	km := st.km
	d := km.d
	nx, ny := st.nSamples, st.k
	for i := 0; i < nx; i++ {
		st.dist[i] = math.MaxFloat32
		st.assign[i] = 0
	}
	for i := 0; i < nx; i += xBatch {
		bnx := min(xBatch, nx-i)
		xb := st.x[i*d:]
		for j := 0; j < ny; j += yBatch {
			bny := min(yBatch, ny-j)
			gemmNTPost(bnx, bny, d, xb, d, y[j*d:], d, st.buf[:bnx*bny], bny, km.workers, func(w, lo, hi int) {
				for r := lo; r < hi; r++ {
					row := st.buf[r*bny : r*bny+bny]
					nx := st.dataNorms[i+r]
					nyv := st.cNorms[j : j+bny]
					minV := st.dist[i+r]
					minC := st.assign[i+r]
					for c := range row {
						v := -2*row[c] + nx + nyv[c]
						if v < minV {
							minV = v
							minC = uint32(j + c)
						}
					}
					st.dist[i+r] = minV
					st.assign[i+r] = minC
				}
			})
		}
	}
	for i := 0; i < nx; i++ {
		if st.dist[i] < 0 {
			st.dist[i] = 0
		}
	}
}

func (st *state) pruned(y []float32) {
	km := st.km
	d := km.d
	nx, ny := st.nSamples, st.k
	pd := st.partialD
	for i := 0; i < nx; i += xBatch {
		bnx := min(xBatch, nx-i)
		xb := st.x[i*d:]
		for j := 0; j < ny; j += yBatch {
			bny := min(yBatch, ny-j)
			yb := y[j*d:]
			gemmNTPost(bnx, bny, pd, xb, d, yb, d, st.buf[:bnx*bny], bny, km.workers, func(w, lo, hi int) {
				pos := st.positions[w]
				for r := lo; r < hi; r++ {
					row := st.buf[r*bny : r*bny+bny]
					idx := i + r
					nx := st.dataNorms[idx]
					nyv := st.cNorms[j : j+bny]
					convL2(row, nyv, nx)
					q := st.x[idx*d : idx*d+d]
					prevTop := st.assign[idx]
					tau := st.dist[idx]
					if j == 0 {
						o := int(prevTop) * d
						tau = l2Sq(y[o:o+d], q)
					}
					top, np := km.top1(q, yb, bny, pd, row, prevTop, tau, uint32(j), pos)
					st.counts[w][0] += int64(np)
					st.assign[idx] = top.index
					st.dist[idx] = top.dist
				}
			})
		}
	}
	for i := 0; i < nx; i++ {
		if st.dist[i] < 0 {
			st.dist[i] = 0
		}
	}
}

func (st *state) sumRows() {
	km := st.km
	k, d := st.k, km.d
	n := st.nSamples
	assign := st.assign[:n]
	grain := n / max(k, 1) * (d + 1)
	if grain < 1 {
		grain = 1
	}
	parallelFor(km.workers, k, grain, func(w, lo, hi int) {
		for c := lo; c < hi; c++ {
			row := st.cur[c*d : c*d+d]
			for j := range row {
				row[j] = 0
			}
			st.sizes[c] = 0
		}
		for i, a := range assign {
			c := int(a)
			if c < lo || c >= hi {
				continue
			}
			st.sizes[c]++
			dst := st.cur[c*d : c*d+d]
			src := st.x[i*d : i*d+d]
			for j := range dst {
				dst[j] += src[j]
			}
		}
	})
}

func (st *state) tunePartialD() float64 {
	km := st.km
	var total int64
	for _, c := range st.counts {
		total += c[0]
	}
	avg := float64(total) / (float64(st.nSamples) * float64(st.k))
	p := st.partialD
	if avg > km.cfg.MaxNotPrunedPct {
		inc := max(int(float64(p)*0.4), 1)
		p = min(p+inc, km.verticalD)
	} else if avg < km.cfg.MinNotPrunedPct {
		dec := max(int(float64(p)*0.2), 1)
		p = max(p-dec, minPartialD)
	}
	st.partialD = min(p, km.verticalD)
	return avg
}

func (st *state) consolidate() {
	km := st.km
	k, d := st.k, km.d
	for c := 0; c < k; c++ {
		if n := st.sizes[c]; n != 0 {
			scaleRow(1/float32(n), st.cur[c*d:c*d+d])
		}
	}
	st.splitClusters()
	if km.cfg.Angular {
		st.normalizeCentroids()
	}
}

func (st *state) normalizeCentroids() {
	km := st.km
	k, d := st.k, km.d
	parallelFor(km.workers, k, d, func(w, lo, hi int) {
		for c := lo; c < hi; c++ {
			row := st.cur[c*d : c*d+d]
			var s float32
			for _, v := range row {
				s += v * v
			}
			inv := float32(1 / math.Sqrt(float64(maxFloat(s))))
			scaleRow(inv, row)
		}
	})
}

func maxFloat(s float32) float32 {
	if s < 1e-30 {
		return 1e-30
	}
	return s
}

func (st *state) splitClusters() {
	km := st.km
	k, d := st.k, km.d
	st.nSplit = 0
	rng := newRNG(km.seedSplit)
	denom := float64(max(st.nSamples-k, 1))
	for ci := 0; ci < k; ci++ {
		if st.sizes[ci] != 0 {
			continue
		}
		cj := st.pickDonor(rng, ci, func(size uint32) (float64, bool) {
			if size <= 1 {
				return 0, false
			}
			return (float64(size) - 1) / denom, true
		})
		if cj < 0 {
			continue
		}
		dst := st.cur[ci*d : ci*d+d]
		src := st.cur[cj*d : cj*d+d]
		copy(dst, src)
		for j := 0; j < d; j++ {
			if j%2 == 0 {
				dst[j] *= 1 + perturbEps
				src[j] *= 1 - perturbEps
			} else {
				dst[j] *= 1 - perturbEps
				src[j] *= 1 + perturbEps
			}
		}
		half := st.sizes[cj] / 2
		st.sizes[ci] = half
		st.sizes[cj] -= half
		st.nSplit++
	}
	if !km.cfg.AggressiveSplit {
		return
	}
	average := float64(st.nSamples / k)
	threshold := uint32(average * balanceThreshold)
	balanceDenom := float64(st.nSamples) - average*float64(k) + float64(k)
	for ci := 0; ci < k; ci++ {
		size := st.sizes[ci]
		if size == 0 || size > threshold {
			continue
		}
		cj := st.pickDonor(rng, ci, func(s uint32) (float64, bool) {
			if float64(s) < average {
				return 0, false
			}
			return (float64(s) - average + 1) / balanceDenom, true
		})
		if cj < 0 {
			continue
		}
		dst := st.cur[ci*d : ci*d+d]
		src := st.cur[cj*d : cj*d+d]
		w := float64(size)
		if w > balanceWeight {
			w = balanceWeight
		}
		wc := float32(w)
		for j := 0; j < d; j++ {
			dst[j] = (wc*dst[j] + src[j]) / (wc + 1)
		}
		st.nSplit++
	}
}

func (st *state) pickDonor(rng *rng, exclude int, accept func(size uint32) (float64, bool)) int {
	k := st.k
	fallback := -1
	var fallbackSize uint32
	for cj := 0; cj < k; cj++ {
		if cj == exclude {
			continue
		}
		p, ok := accept(st.sizes[cj])
		if ok && p > 0 {
			if rng.float64() < p {
				return cj
			}
			if st.sizes[cj] > fallbackSize {
				fallback = cj
				fallbackSize = st.sizes[cj]
			}
		}
	}
	if fallback >= 0 {
		return fallback
	}
	for cj := 0; cj < k; cj++ {
		if cj != exclude {
			return cj
		}
	}
	return -1
}

func (st *state) centroidShift() float64 {
	km := st.km
	k, d := st.k, km.d
	var total float64
	for c := 0; c < k; c++ {
		cur := st.cur[c*d : c*d+d]
		prev := st.prev[c*d : c*d+d]
		var s float64
		for j := range cur {
			diff := float64(cur[j]) - float64(prev[j])
			s += diff * diff
		}
		total += s
	}
	return total
}

func (st *state) converged(iter int) bool {
	if !st.km.cfg.EarlyTermination {
		return false
	}
	tol := st.km.cfg.Tolerance
	if st.shift < tol {
		return true
	}
	if iter > 0 {
		denom := st.prevCost
		if denom < math.SmallestNonzeroFloat64 {
			denom = math.SmallestNonzeroFloat64
		}
		if st.cost/denom > 1-tol {
			return true
		}
	}
	return false
}

func (km *KMeans) finalize(st *state, data []float32, n, nSamples int, centroids []float32) *Model {
	k, d := km.k, km.d
	m := &Model{
		K:              k,
		D:              d,
		Centroids:      centroids,
		Iterations:     len(st.stats),
		IterationStats: st.stats,
	}
	if nSamples == n {
		gemmOnly := d < dGate || km.cfg.GEMMOnly || k <= kGate
		if !st.finalized {
			if gemmOnly {
				squaredNorms(st.x, nSamples, d, st.dataNorms, km.workers)
			} else {
				squaredNormsPrefix(st.x, nSamples, d, st.partialD, st.dataNorms, km.workers)
			}
			st.estep(st.cur, gemmOnly)
		}
		m.Assignments = append([]uint32(nil), st.assign[:n]...)
		m.ClusterSizes = clusterSizes(m.Assignments, k)
		m.WCSS = sumFloat32(st.dist[:n])
		return m
	}
	assign, dist := km.exhaustive(data, n, centroids, k)
	m.Assignments = assign
	m.ClusterSizes = clusterSizes(assign, k)
	m.WCSS = sumFloat32(dist)
	return m
}

func (km *KMeans) exhaustive(data []float32, n int, centroids []float32, k int) ([]uint32, []float32) {
	return exhaustiveFree(data, n, centroids, k, km.d, km.workers)
}

func (m *Model) Assign(data []float32, n int, out []uint32, dists []float32) error {
	if len(data) < n*m.D {
		return dataLengthError(len(data), n*m.D)
	}
	if len(out) < n {
		return outputLengthError(len(out), n)
	}
	if dists != nil && len(dists) < n {
		return outputLengthError(len(dists), n)
	}
	if err := validateFinite(data, n, m.D); err != nil {
		return err
	}
	assign, dist := exhaustiveFree(data, n, m.Centroids, m.K, m.D, normWorkers(0))
	copy(out, assign)
	if dists != nil {
		copy(dists, dist)
	}
	return nil
}

func validateFinite(data []float32, n, d int) error {
	for _, v := range data[:n*d] {
		if v != v || v > math.MaxFloat32 || v < -math.MaxFloat32 {
			return ErrNonFinite
		}
	}
	return nil
}

func exhaustiveFree(data []float32, n int, centroids []float32, k, d, workers int) ([]uint32, []float32) {
	assign := make([]uint32, n)
	dist := make([]float32, n)
	xn := make([]float32, n)
	cn := make([]float32, k)
	squaredNorms(data, n, d, xn, workers)
	squaredNorms(centroids, k, d, cn, workers)
	buf := make([]float32, min(xBatch, n)*min(yBatch, k))
	for i := range dist {
		dist[i] = math.MaxFloat32
	}
	for i := 0; i < n; i += xBatch {
		bnx := min(xBatch, n-i)
		for j := 0; j < k; j += yBatch {
			bny := min(yBatch, k-j)
			sgemmNT(bnx, bny, d, data[i*d:], d, centroids[j*d:], d, buf[:bnx*bny], bny, workers)
			parallelFor(workers, bnx, bny*d, func(w, lo, hi int) {
				for r := lo; r < hi; r++ {
					row := buf[r*bny : r*bny+bny]
					nx := xn[i+r]
					nyv := cn[j : j+bny]
					minV := dist[i+r]
					minC := assign[i+r]
					for c := range row {
						v := -2*row[c] + nx + nyv[c]
						if v < minV {
							minV = v
							minC = uint32(j + c)
						}
					}
					dist[i+r] = minV
					assign[i+r] = minC
				}
			})
		}
	}
	for i := range dist {
		if dist[i] < 0 {
			dist[i] = 0
		}
	}
	return assign, dist
}
