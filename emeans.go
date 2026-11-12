package superkmeans

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// e-means: a Go port of FAISS contrib/e_means (NumpyEmeans), implementing
// Algorithm 1 of "E-Means: outperforming k-means (in one-tenth the epochs)"
// (Bessa et al., https://arxiv.org/abs/2611.09501).
// https://github.com/facebookresearch/faiss/tree/main/contrib/e_means
//
// Mini-batch k-means with exponential moving average updates. Per batch, each
// cluster j updates a running count N_j and sum S_j:
//
//	N_j <- (1-alpha) * N_j + alpha * n_j
//	S_j <- (1-alpha) * S_j + alpha * s_j
//	c_j  = S_j / N_j
//
// The code stores N_j (usage) and c_j (centroids) instead of S_j. Batch size
// and initial step size come from fitted scaling laws. The step size follows
// a cosine schedule. Every five updates, clusters below a mass threshold are
// revived by splitting heavy clusters. The returned centroids are the exact
// means of the last pass's assignments.

const (
	emAlphaMinFactor   = 0.1     // cosine schedule floor: min step / initial step
	emCheckUnusedEvery = 5       // logical batch updates between revival checks
	emInitUsage        = 0x1p-16 // initial EMA mass N_j (paper uses 1e-10)
	emSplitEpsilon     = 1e-4    // relative coordinate perturbation for splits
	emUnusedThreshold  = 0.1     // revive N_j < tau * mean(N)
	emNoRevivalMaxD    = 2       // unused_threshold default is 0 at or below this d
	emAutoQualityMaxD  = 16      // batch_law "auto" picks "quality" at or below this d
)

// EMeansNoLabel marks rows left out of training by MaxPointsPerCentroid in
// EMeansModel.Assignments.
const EMeansNoLabel = ^uint32(0)

// Batch law B = C_B * k^b_k * n^b_n * T^b_T. Quality uses one eighth of the
// throughput coefficient before power-of-two rounding and batch caps.
var emBatchLaws = map[string][4]float64{
	"throughput": {0.035134, 0.134749, 0.876473, 0.813344},
	"quality":    {0.00439175, 0.134749, 0.876473, 0.813344},
}

// Step law alpha0 = C_a * k^a_k * B^a_B * n^a_n * T^a_T, clamped to [1e-4, 1].
var emAlphaLaw = [5]float64{0.8244, -0.4617, 0.9003, -0.4498, -0.1057}

type EMeansConfig struct {
	NEpochs              float64 // passes over the data, fractional allowed (default 5)
	BatchLaw             string  // "auto" (default), "throughput" or "quality"
	BatchSize            int     // 0 = from the batch law
	Alpha0               float64 // 0 = from the step law at the resolved batch size
	UnusedThreshold      float64 // <0 = default: 0.1, or 0 (no revival) for d <= 2
	MaxPointsPerCentroid int     // 0 = train on all rows
	Seed                 uint64
	Threads              int
	Spherical            bool // cluster by direction (cosine); data must be L2-normalized
	CaptureAssignments   bool // keep each row's last-pass label (default true)
	InitialCentroids     []float32
	Verbose              bool
}

func DefaultEMeansConfig() EMeansConfig {
	return EMeansConfig{
		NEpochs:            5,
		BatchLaw:           "auto",
		UnusedThreshold:    -1,
		Seed:               1234,
		CaptureAssignments: true,
	}
}

type EMeansRecipe struct {
	BatchSize       int
	Alpha0          float64
	BatchLaw        string // resolved law, "" when BatchSize was given
	NEpochs         float64
	EpochRows       []int
	TotalSteps      int
	UnusedThreshold float64
}

type EMeansIterationStats struct {
	Objective float64 // sum of distances measured while centroids move
	Time      float64
	Imbalance float64
	Splits    int
}

type EMeansModel struct {
	K              int
	D              int
	Centroids      []float32
	Assignments    []uint32 // last-pass labels, nil when capture disabled
	WCSS           float64  // objective of the last pass (moving centroids)
	Recipe         EMeansRecipe
	IterationStats []EMeansIterationStats
	NSplits        int
	Spherical      bool
}

type EMeans struct {
	k       int
	d       int
	cfg     EMeansConfig
	workers int
}

func NewEMeans(k, d int, cfg *EMeansConfig) (*EMeans, error) {
	c := DefaultEMeansConfig()
	if cfg != nil {
		c = *cfg
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	if d <= 0 {
		return nil, ErrInvalidD
	}
	if c.NEpochs <= 0 {
		return nil, ErrInvalidIters
	}
	if c.BatchLaw == "" {
		c.BatchLaw = "auto"
	}
	if c.BatchLaw != "auto" {
		if _, ok := emBatchLaws[c.BatchLaw]; !ok {
			return nil, fmt.Errorf("superkmeans: unknown e-means batch law %q", c.BatchLaw)
		}
	}
	if c.MaxPointsPerCentroid < 0 {
		return nil, ErrMaxPoints
	}
	return &EMeans{k: k, d: d, cfg: c, workers: normWorkers(c.Threads)}, nil
}

func emRoundPowerOfTwo(v float64) int {
	if v <= 1 {
		return 1
	}
	low := math.Exp2(math.Floor(math.Log2(v)))
	high := 2 * low
	if v/low < high/v {
		return int(low)
	}
	return int(high)
}

func emStepLaw(n, k int, nepochs float64, batch int) float64 {
	c := emAlphaLaw
	a := c[0] * math.Pow(float64(k), c[1]) * math.Pow(float64(batch), c[2]) *
		math.Pow(float64(n), c[3]) * math.Pow(nepochs, c[4])
	return minF(maxF(a, 1e-4), 1.0)
}

// emEpochRows returns whole passes, a rounded fractional prefix, then the
// moving closing pass. Fractional rows use ties-to-even rounding.
func emEpochRows(n int, nepochs float64) []int {
	whole := int(math.Floor(nepochs))
	rem := int(math.RoundToEven((nepochs - float64(whole)) * float64(n)))
	if whole == 0 {
		return []int{rem}
	}
	rows := make([]int, 0, whole+1)
	for i := 0; i < whole-1; i++ {
		rows = append(rows, n)
	}
	if rem > 0 {
		rows = append(rows, rem)
	}
	return append(rows, n)
}

func emScheduledAlpha(alpha0 float64, step, totalSteps int) float64 {
	frac := float64(min(step, totalSteps)) / float64(max(1, totalSteps))
	cosine := 0.5 * (1 + math.Cos(math.Pi*frac))
	return alpha0 * (emAlphaMinFactor + (1-emAlphaMinFactor)*cosine)
}

func (em *EMeans) resolveRecipe(n int) EMeansRecipe {
	c := em.cfg
	law := c.BatchLaw
	if law == "auto" {
		if em.d <= emAutoQualityMaxD {
			law = "quality"
		} else {
			law = "throughput"
		}
	}
	batch := c.BatchSize
	resolvedLaw := law
	if batch <= 0 {
		coef := emBatchLaws[law]
		raw := coef[0] * math.Pow(float64(em.k), coef[1]) * math.Pow(float64(n), coef[2]) *
			math.Pow(c.NEpochs, coef[3])
		batch = emRoundPowerOfTwo(raw)
		if law == "throughput" {
			batch = min(batch, emRoundPowerOfTwo(float64(n)/2))
		} else {
			batch = min(batch, n/2)
		}
		batch = max(1, min(batch, n))
	} else {
		resolvedLaw = ""
		batch = max(1, min(batch, n))
	}
	alpha0 := c.Alpha0
	if alpha0 <= 0 {
		alpha0 = emStepLaw(n, em.k, c.NEpochs, batch)
	}
	tau := c.UnusedThreshold
	if tau < 0 {
		if em.d <= emNoRevivalMaxD {
			tau = 0
		} else {
			tau = emUnusedThreshold
		}
	}
	rows := emEpochRows(n, c.NEpochs)
	total := 0
	for _, r := range rows {
		total += ceilDiv(r, batch)
	}
	return EMeansRecipe{
		BatchSize:       batch,
		Alpha0:          alpha0,
		BatchLaw:        resolvedLaw,
		NEpochs:         c.NEpochs,
		EpochRows:       rows,
		TotalSteps:      total,
		UnusedThreshold: tau,
	}
}

type emState struct {
	em         *EMeans
	n          int
	recipe     EMeansRecipe
	centroids  []float32 // EMA positions c_j = S_j / N_j
	unit       []float32 // spherical readout: normalized centroids
	usage      []float32 // masses N_j
	sums       []float32
	counts     []float32
	passSums   []float32 // last-pass sums for the closing mean
	passCounts []float64
	dataNorms  []float32
	cNorms     []float32
	gt         []float32 // gathered row tile
	buf        []float32 // GEMM tile
	labels     []uint32
	dist       []float32
	splitRng   *rng
	nSplits    int
	step       int
	spherical  bool
}

// Fit trains on data and returns the closing centroids and last-pass labels.
func (em *EMeans) Fit(data []float32, n int) (*EMeansModel, error) {
	k, d := em.k, em.d
	if n < k {
		return nil, ErrTooFewPoints
	}
	if len(data) < n*d {
		return nil, dataLengthError(len(data), n*d)
	}
	if err := validateFinite(data, n, d); err != nil {
		return nil, err
	}
	train := data[:n*d]
	nTrain := n
	var keep []int
	if em.cfg.MaxPointsPerCentroid > 0 && n > em.cfg.MaxPointsPerCentroid*k {
		limit := em.cfg.MaxPointsPerCentroid * k
		perm := newRNG(deriveSeed(em.cfg.Seed, 5)).perm(n)
		keep = perm[:limit]
		sort.Ints(keep)
		gathered := make([]float32, limit*d)
		parallelFor(em.workers, limit, d, func(w, lo, hi int) {
			for i := lo; i < hi; i++ {
				copy(gathered[i*d:i*d+d], data[keep[i]*d:keep[i]*d+d])
			}
		})
		train, nTrain = gathered, limit
	}

	recipe := em.resolveRecipe(nTrain)
	st := &emState{
		em:         em,
		n:          nTrain,
		recipe:     recipe,
		centroids:  make([]float32, k*d),
		usage:      make([]float32, k),
		sums:       make([]float32, k*d),
		counts:     make([]float32, k),
		passCounts: make([]float64, k),
		cNorms:     make([]float32, k),
		gt:         make([]float32, min(xBatch, recipe.BatchSize)*d),
		buf:        make([]float32, min(xBatch, recipe.BatchSize)*min(yBatch, k)),
		labels:     make([]uint32, recipe.BatchSize),
		dist:       make([]float32, recipe.BatchSize),
		splitRng:   newRNG(deriveSeed(em.cfg.Seed, 2)),
		spherical:  em.cfg.Spherical,
	}
	if st.spherical {
		st.unit = make([]float32, k*d)
	} else {
		st.dataNorms = make([]float32, nTrain)
		squaredNorms(train, nTrain, d, st.dataNorms, em.workers)
	}
	for i := range st.usage {
		st.usage[i] = emInitUsage
	}

	if em.cfg.InitialCentroids != nil {
		if len(em.cfg.InitialCentroids) < k*d {
			return nil, ErrInitialCentroids
		}
		copy(st.centroids, em.cfg.InitialCentroids[:k*d])
	} else {
		perm := newRNG(deriveSeed(em.cfg.Seed, 1)).perm(nTrain)
		parallelFor(em.workers, k, d, func(w, lo, hi int) {
			for c := lo; c < hi; c++ {
				copy(st.centroids[c*d:c*d+d], train[perm[c]*d:perm[c]*d+d])
			}
		})
	}
	if st.spherical {
		emNormalizeRows(st.centroids, k, d)
	}

	if em.cfg.Verbose {
		fmt.Printf("e-means: clustering %d points in %dD to %d clusters, %g epochs "+
			"(batch %d, alpha0 %.3g, law %s)\n",
			nTrain, d, k, recipe.NEpochs, recipe.BatchSize, recipe.Alpha0, recipe.BatchLaw)
	}

	var assignments []uint32
	if em.cfg.CaptureAssignments {
		assignments = make([]uint32, n)
		for i := range assignments {
			assignments[i] = EMeansNoLabel
		}
	}

	start := time.Now()
	stats := make([]EMeansIterationStats, 0, len(recipe.EpochRows))
	lastEpoch := len(recipe.EpochRows) - 1
	for epoch, rows := range recipe.EpochRows {
		last := epoch == lastEpoch
		if last {
			st.passSums = make([]float32, k*d)
		}
		for i := range st.passCounts {
			st.passCounts[i] = 0
		}
		splitsBefore := st.nSplits
		perm := newRNG(deriveSeed(em.cfg.Seed, uint64(10+epoch))).perm(nTrain)
		var objective float64
		for begin := 0; begin < rows; begin += recipe.BatchSize {
			bs := min(recipe.BatchSize, rows-begin)
			objective += st.batchStep(train, perm[begin:begin+bs], last, assignments, keep)
		}
		var countSq float64
		for _, c := range st.passCounts {
			countSq += c * c
		}
		stats = append(stats, EMeansIterationStats{
			Objective: objective,
			Time:      time.Since(start).Seconds(),
			Imbalance: float64(k) * countSq / (float64(rows) * float64(rows)),
			Splits:    st.nSplits - splitsBefore,
		})
		if em.cfg.Verbose {
			fmt.Printf("  epoch %d/%d: objective=%g imbalance=%.3g splits=%d (%.2fs)\n",
				epoch+1, len(recipe.EpochRows), objective, stats[epoch].Imbalance,
				stats[epoch].Splits, stats[epoch].Time)
		}
	}

	// Closing means: exact means of the last pass's assignments.
	parallelFor(em.workers, k, d, func(w, lo, hi int) {
		for c := lo; c < hi; c++ {
			if st.passCounts[c] > 0 {
				inv := 1 / float32(st.passCounts[c])
				row := st.centroids[c*d : c*d+d]
				src := st.passSums[c*d : c*d+d]
				for j := range row {
					row[j] = src[j] * inv
				}
			}
		}
	})
	if st.spherical {
		emNormalizeRows(st.centroids, k, d)
	}

	return &EMeansModel{
		K:              k,
		D:              d,
		Centroids:      st.centroids,
		Assignments:    assignments,
		WCSS:           stats[len(stats)-1].Objective,
		Recipe:         recipe,
		IterationStats: stats,
		NSplits:        st.nSplits,
		Spherical:      st.spherical,
	}, nil
}

// FitCentroids trains and returns the closing centroids only.
func (em *EMeans) FitCentroids(data []float32, n int) ([]float32, error) {
	m, err := em.Fit(data, n)
	if err != nil {
		return nil, err
	}
	return m.Centroids, nil
}

// batchStep assigns one batch against a fixed centroid snapshot, then
// applies the EMA update. keep maps subset row ids back to original row ids
// (nil = identity). Returns the batch objective.
func (st *emState) batchStep(data []float32, idx []int, last bool, assignments []uint32, keep []int) float64 {
	em := st.em
	k, d := em.k, em.d
	nb := len(idx)
	workers := em.workers
	snap := st.centroids
	if st.spherical {
		copy(st.unit, st.centroids)
		emNormalizeRows(st.unit, k, d)
		snap = st.unit
	} else {
		squaredNorms(snap, k, d, st.cNorms, workers)
	}

	parallelFor(workers, k, d+1, func(w, lo, hi int) {
		for c := lo; c < hi; c++ {
			row := st.sums[c*d : c*d+d]
			for j := range row {
				row[j] = 0
			}
			st.counts[c] = 0
		}
	})
	dist := st.dist[:nb]
	labels := st.labels[:nb]
	parallelFor(workers, nb, 1, func(w, lo, hi int) {
		for i := lo; i < hi; i++ {
			dist[i] = math.MaxFloat32
			labels[i] = 0
		}
	})

	spherical := st.spherical
	for i0 := 0; i0 < nb; i0 += xBatch {
		bnx := min(xBatch, nb-i0)
		gt := st.gt[:bnx*d]
		parallelFor(workers, bnx, d, func(w, lo, hi int) {
			for r := lo; r < hi; r++ {
				copy(gt[r*d:r*d+d], data[idx[i0+r]*d:idx[i0+r]*d+d])
			}
		})
		for j0 := 0; j0 < k; j0 += yBatch {
			bny := min(yBatch, k-j0)
			sgemmNT(bnx, bny, d, gt, d, snap[j0*d:], d, st.buf[:bnx*bny], bny, workers)
			parallelFor(workers, bnx, bny, func(w, lo, hi int) {
				for r := lo; r < hi; r++ {
					row := st.buf[r*bny : r*bny+bny]
					minV := dist[i0+r]
					minC := labels[i0+r]
					if spherical {
						for c, v := range row {
							if -v < minV {
								minV = -v
								minC = uint32(j0 + c)
							}
						}
					} else {
						nx := st.dataNorms[idx[i0+r]]
						nyv := st.cNorms[j0 : j0+bny]
						for c, v := range row {
							v = -2*v + nx + nyv[c]
							if v < minV {
								minV = v
								minC = uint32(j0 + c)
							}
						}
					}
					dist[i0+r] = minV
					labels[i0+r] = minC
				}
			})
		}
		// Cluster-partitioned accumulation, as in sumRows.
		grain := bnx / max(k, 1) * (d + 1)
		if grain < 1 {
			grain = 1
		}
		tileLabels := labels[i0 : i0+bnx]
		parallelFor(workers, k, grain, func(w, lo, hi int) {
			for i, a := range tileLabels {
				c := int(a)
				if c < lo || c >= hi {
					continue
				}
				st.counts[c]++
				dst := st.sums[c*d : c*d+d]
				src := gt[i*d : i*d+d]
				for j := range dst {
					dst[j] += src[j]
				}
			}
		})
		if last && assignments != nil {
			parallelFor(workers, bnx, 1, func(w, lo, hi int) {
				for r := lo; r < hi; r++ {
					row := idx[i0+r]
					if keep != nil {
						row = keep[row]
					}
					assignments[row] = tileLabels[r]
				}
			})
		}
	}

	var objective float64
	if spherical {
		objective = 2 * float64(nb)
		for _, v := range dist {
			objective += 2 * float64(v) // dist holds -maxDot
		}
	} else {
		for i, v := range dist {
			if v < 0 {
				v = 0
				dist[i] = 0
			}
			objective += float64(v)
		}
	}

	st.emaUpdate(last)
	st.step++
	if st.step%emCheckUnusedEvery == 0 && st.recipe.UnusedThreshold > 0 {
		st.reviveUnused()
	}
	return objective
}

// emaUpdate applies one EMA step (Algorithm 1, line 8). alpha == 1 replaces
// positions with batch means.
func (st *emState) emaUpdate(last bool) {
	em := st.em
	k, d := em.k, em.d
	alpha := float32(emScheduledAlpha(st.recipe.Alpha0, st.step, st.recipe.TotalSteps))
	decay := 1 - alpha
	parallelFor(em.workers, k, d+1, func(w, lo, hi int) {
		for c := lo; c < hi; c++ {
			nj := st.counts[c]
			st.passCounts[c] += float64(nj)
			u := st.usage[c] * decay
			if nj > 0 {
				src := st.sums[c*d : c*d+d]
				if last {
					dst := st.passSums[c*d : c*d+d]
					for j := range dst {
						dst[j] += src[j]
					}
				}
				mass := u + alpha*nj
				weight := alpha * nj / mass
				row := st.centroids[c*d : c*d+d]
				inv := 1 / nj
				if weight == 1 {
					for j := range row {
						row[j] = src[j] * inv
					}
				} else {
					for j := range row {
						mean := src[j] * inv
						row[j] += weight * (mean - row[j])
					}
				}
				u = mass
			}
			st.usage[c] = u
		}
	})
}

// reviveUnused splits clusters with mass below tau*mean(N). Donors are
// sampled without replacement in proportion to their mass via Gumbel keys
// (paper Appendix J).
func (st *emState) reviveUnused() {
	k, d := st.em.k, st.em.d
	var total float32
	for _, u := range st.usage {
		total += u
	}
	threshold := total * float32(st.recipe.UnusedThreshold) / float32(k)
	type donor struct {
		key float64
		idx int
	}
	var donors []donor
	var unused []int
	for c := 0; c < k; c++ {
		if st.usage[c] < threshold {
			unused = append(unused, c)
		} else if st.usage[c] > 0 {
			u := maxF(st.splitRng.float64(), 2.2250738585072014e-308)
			key := math.Log(float64(st.usage[c])) - math.Log(-math.Log(u))
			donors = append(donors, donor{key, c})
		}
	}
	nsplit := min(len(unused), len(donors))
	if nsplit == 0 {
		return
	}
	sort.Slice(donors, func(i, j int) bool { return donors[i].key > donors[j].key })

	for i := 0; i < nsplit; i++ {
		s, t := donors[i].idx, unused[i]
		src := st.centroids[s*d : s*d+d]
		dst := st.centroids[t*d : t*d+d]
		collapsed := true
		for j := 0; j < d; j++ {
			var off float32 = emSplitEpsilon
			if j%2 == 1 {
				off = -emSplitEpsilon
			}
			donor := src[j] * (1 - off)
			child := src[j] * (1 + off)
			src[j] = donor
			dst[j] = child
			if donor != child {
				collapsed = false
			}
		}
		if collapsed {
			// Zero donors collapse to identical rows; separate by one ulp.
			dst[0] = math.Nextafter32(dst[0], float32(math.Inf(1)))
			if math.IsInf(float64(dst[0]), 0) {
				dst[0] = math.Nextafter32(src[0], float32(math.Inf(-1)))
			}
		}
		half := st.usage[s] / 2
		st.usage[s] = half
		st.usage[t] = half
	}
	st.nSplits += nsplit
}

// emNormalizeRows rescales each row to unit norm in place; zero stays zero.
// Prescaling by the max magnitude avoids overflow.
func emNormalizeRows(x []float32, n, d int) {
	parallelFor(normWorkers(0), n, d, func(w, lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x[i*d : i*d+d]
			var mx float32
			for _, v := range row {
				if a := float32(math.Abs(float64(v))); a > mx {
					mx = a
				}
			}
			if mx == 0 {
				continue
			}
			scaleRow(1/mx, row)
			s := sqnorm(row)
			if s > 0 {
				scaleRow(1/float32(math.Sqrt(float64(s))), row)
			}
		}
	})
}

// Assign returns each row's nearest final centroid. For Euclidean models the
// distances are squared L2; for spherical models they are inner-product
// similarities (higher is closer).
func (m *EMeansModel) Assign(data []float32, n int, out []uint32, dists []float32) error {
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
	var assign []uint32
	var dist []float32
	if m.Spherical {
		assign, dist = sphericalFree(data, n, m.Centroids, m.K, m.D, normWorkers(0))
	} else {
		assign, dist = exhaustiveFree(data, n, m.Centroids, m.K, m.D, normWorkers(0))
	}
	copy(out, assign)
	if dists != nil {
		copy(dists, dist)
	}
	return nil
}

// sphericalFree assigns rows to centroids by maximum inner product.
func sphericalFree(data []float32, n int, centroids []float32, k, d, workers int) ([]uint32, []float32) {
	assign := make([]uint32, n)
	sim := make([]float32, n)
	buf := make([]float32, min(xBatch, n)*min(yBatch, k))
	for i := range sim {
		sim[i] = -math.MaxFloat32
	}
	for i := 0; i < n; i += xBatch {
		bnx := min(xBatch, n-i)
		for j := 0; j < k; j += yBatch {
			bny := min(yBatch, k-j)
			sgemmNT(bnx, bny, d, data[i*d:], d, centroids[j*d:], d, buf[:bnx*bny], bny, workers)
			parallelFor(workers, bnx, bny, func(w, lo, hi int) {
				for r := lo; r < hi; r++ {
					row := buf[r*bny : r*bny+bny]
					maxV := sim[i+r]
					maxC := assign[i+r]
					for c, v := range row {
						if v > maxV {
							maxV = v
							maxC = uint32(j + c)
						}
					}
					sim[i+r] = maxV
					assign[i+r] = maxC
				}
			})
		}
	}
	return assign, sim
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
