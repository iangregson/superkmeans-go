package superkmeans

import "math"

func (km *KMeans) trainHier(data []float32, n int) (*state, []float32, error) {
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
	if nSamples != n {
		return nil, nil, ErrHierSampling
	}
	perm := newRNG(km.seedInit).perm(n)
	nMeso := mesoCount(k)
	st := km.newState(n, k)
	st.k = nMeso
	km.gatherRotate(st.prev, data, perm, nMeso)
	sample := km.sampleAndRotate(data, n, n, perm)
	st.x = sample
	fullNorms := make([]float32, n)
	squaredNorms(sample, n, d, fullNorms, km.workers)
	copy(st.dataNorms, fullNorms)
	st.runPhase(km.cfg.HierItersMeso, false)

	sizes := clusterSizes(st.assign[:n], nMeso)
	offsets := make([]int, nMeso+1)
	for m := 0; m < nMeso; m++ {
		offsets[m+1] = offsets[m] + int(sizes[m])
	}
	members := make([]int, n)
	cursor := make([]int, nMeso)
	copy(cursor, offsets[:nMeso])
	for i := 0; i < n; i++ {
		m := int(st.assign[i])
		members[cursor[m]] = i
		cursor[m]++
	}
	fine := arrangeFineClusters(k, nMeso, n, sizes)
	widest := 0
	for _, s := range sizes {
		if int(s) > widest {
			widest = int(s)
		}
	}
	mesoBuf := make([]float32, widest*d)
	indirect := make([]int, widest)
	finalCentroids := make([]float32, k*d)
	finalAssign := make([]uint32, n)
	offset := 0
	for m := 0; m < nMeso; m++ {
		nFine := fine[m]
		if nFine == 0 {
			continue
		}
		size := int(sizes[m])
		owned := members[offsets[m] : offsets[m]+size]
		parallelFor(km.workers, size, d, func(w, lo, hi int) {
			for j := lo; j < hi; j++ {
				i := owned[j]
				indirect[j] = i
				copy(mesoBuf[j*d:j*d+d], sample[i*d:i*d+d])
				st.dataNorms[j] = fullNorms[i]
			}
		})
		st.x = mesoBuf[: size*d : size*d]
		st.nSamples = size
		st.k = nFine
		st.partialD = km.partialD
		seeds := newRNG(deriveSeed(km.seedInit, 5+uint64(m))).perm(size)
		for i := 0; i < nFine; i++ {
			copy(st.prev[i*d:i*d+d], mesoBuf[seeds[i]*d:seeds[i]*d+d])
		}
		st.runPhase(km.cfg.HierItersFine, false)
		for j := 0; j < size; j++ {
			finalAssign[indirect[j]] = st.assign[j] + uint32(offset)
		}
		copy(finalCentroids[offset*d:(offset+nFine)*d], st.cur[:nFine*d])
		offset += nFine
	}
	if offset != k {
		return nil, nil, ErrHierClusterCount
	}
	st.x = sample
	st.nSamples = n
	st.k = k
	copy(st.prev, finalCentroids)
	copy(st.cur, finalCentroids)
	copy(st.assign, finalAssign)
	copy(st.dataNorms, fullNorms)
	st.partialD = min(max(minPartialD, km.verticalD/3), km.verticalD)
	if km.cfg.HierItersRefine > 0 {
		st.runPhase(km.cfg.HierItersRefine, true)
	}
	st.partialD = km.partialD
	gemmOnly := d < dGate || km.cfg.GEMMOnly || k <= kGate
	if gemmOnly {
		squaredNorms(st.x, n, d, st.dataNorms, km.workers)
	} else {
		squaredNormsPrefix(st.x, n, d, st.partialD, st.dataNorms, km.workers)
	}
	st.estep(st.cur, gemmOnly)
	st.cost = sumFloat32(st.dist[:n])
	st.finalized = true
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

func (st *state) runPhase(iters int, preAssigned bool) {
	km := st.km
	d := km.d
	gemmOnly := d < dGate || km.cfg.GEMMOnly || st.k <= kGate
	prefixNorms := false
	for iter := 0; iter < iters; iter++ {
		exhaustive := gemmOnly || (iter == 0 && !preAssigned)
		if !exhaustive && !prefixNorms {
			squaredNormsPrefix(st.x, st.nSamples, d, st.partialD, st.dataNorms, km.workers)
			prefixNorms = true
		}
		st.runIteration(iter, exhaustive)
		if km.cfg.EarlyTermination && st.converged(iter) {
			break
		}
	}
}

func mesoCount(k int) int {
	return max(int(math.Round(math.Sqrt(float64(k)))), 1)
}

func arrangeFineClusters(k, nMeso, nSamples int, sizes []uint32) []int {
	out := make([]int, nMeso)
	remaining := k
	pending := 0
	for _, s := range sizes {
		if s > 0 {
			pending++
		}
	}
	samplesLeft := nSamples
	for m := 0; m < nMeso; m++ {
		size := int(sizes[m])
		if size == 0 {
			continue
		}
		pending--
		alloc := remaining
		if pending > 0 && samplesLeft > 0 {
			alloc = int(math.Round(float64(remaining) * float64(size) / float64(samplesLeft)))
			if alloc > remaining-pending {
				alloc = remaining - pending
			}
			if alloc < 1 {
				alloc = 1
			}
		}
		if alloc > size {
			alloc = size
		}
		if alloc > remaining {
			alloc = remaining
		}
		out[m] = alloc
		remaining -= alloc
		samplesLeft -= size
	}
	for remaining > 0 {
		moved := false
		for m := 0; m < nMeso && remaining > 0; m++ {
			if out[m] > 0 && out[m] < int(sizes[m]) {
				out[m]++
				remaining--
				moved = true
			}
		}
		if !moved {
			break
		}
	}
	return out
}
