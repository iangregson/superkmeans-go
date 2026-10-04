package superkmeans

type candidate struct {
	index uint32
	dist  float32
}

func (km *KMeans) top1(query, centroids []float32, bny, partialD int, dists []float32, prevTop uint32, prevDist float32, offset uint32, pos []uint32) (candidate, int) {
	top := candidate{index: prevTop, dist: prevDist}
	if bny <= 0 {
		return top, 0
	}
	pos = pos[:bny]
	dv := dists[:bny]
	thr := top.dist * km.ratios[partialD]
	count := survivorCount(dv, thr, pos)
	initial := count
	if count == 1 && offset+pos[0] == prevTop {
		return top, initial
	}
	cur := partialD
	for h := 0; count > 0 && h < km.horizontalD; {
		block := min(64, km.horizontalD-h)
		off := km.verticalD + h
		q := query[off : off+block]
		l2BlockAdd(dv, pos[:count], q, centroids, km.d, off)
		h += block
		cur += block
		count = compact(pos, dv, count, top.dist*km.ratios[cur])
	}
	if count > 0 && partialD < km.verticalD {
		left := km.verticalD - partialD
		q := query[partialD : partialD+left]
		l2BlockAdd(dv, pos[:count], q, centroids, km.d, partialD)
		count = compact(pos, dv, count, top.dist*km.ratios[km.d])
	}
	for _, p := range pos[:count] {
		if v := dv[p]; v < top.dist {
			top.dist = v
			top.index = offset + p
		}
	}
	return top, initial
}

func compact(pos []uint32, dists []float32, count int, threshold float32) int {
	n := 0
	for _, p := range pos[:count] {
		pos[n] = p
		if dists[p] < threshold {
			n++
		}
	}
	return n
}
