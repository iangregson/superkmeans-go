package main

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

func naiveLloyd(data []float32, n, d, k int, seed uint64, iters int) ([]float32, float64) {
	r := rand.New(rand.NewSource(int64(seed)))
	perm := r.Perm(n)
	centroids := make([]float32, k*d)
	for i := 0; i < k; i++ {
		copy(centroids[i*d:i*d+d], data[perm[i]*d:perm[i]*d+d])
	}
	assign := make([]uint32, n)
	sums := make([]float32, k*d)
	counts := make([]int, k)
	workers := runtime.GOMAXPROCS(0)
	for iter := 0; iter < iters; iter++ {
		parFor(workers, n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				p := data[i*d : i*d+d]
				best := float32(math.MaxFloat32)
				bestC := 0
				for c := 0; c < k; c++ {
					s := l2sqNaive(p, centroids[c*d:c*d+d])
					if s < best {
						best = s
						bestC = c
					}
				}
				assign[i] = uint32(bestC)
			}
		})
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < n; i++ {
			c := int(assign[i])
			counts[c]++
			src := data[i*d : i*d+d]
			dst := sums[c*d : c*d+d]
			for j := 0; j < d; j++ {
				dst[j] += src[j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				ri := r.Intn(n)
				copy(centroids[c*d:c*d+d], data[ri*d:ri*d+d])
				continue
			}
			inv := 1 / float32(counts[c])
			for j := 0; j < d; j++ {
				centroids[c*d+j] = sums[c*d+j] * inv
			}
		}
	}
	var wcss float64
	for i := 0; i < n; i++ {
		c := int(assign[i])
		wcss += float64(l2sqNaive(data[i*d:i*d+d], centroids[c*d:c*d+d]))
	}
	return centroids, wcss
}

func l2sqNaive(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var a0, a1, a2, a3, a4, a5, a6, a7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		d0 := x[i] - y[i]
		d1 := x[i+1] - y[i+1]
		d2 := x[i+2] - y[i+2]
		d3 := x[i+3] - y[i+3]
		d4 := x[i+4] - y[i+4]
		d5 := x[i+5] - y[i+5]
		d6 := x[i+6] - y[i+6]
		d7 := x[i+7] - y[i+7]
		a0 += d0 * d0
		a1 += d1 * d1
		a2 += d2 * d2
		a3 += d3 * d3
		a4 += d4 * d4
		a5 += d5 * d5
		a6 += d6 * d6
		a7 += d7 * d7
	}
	s := a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
	for ; i < n; i++ {
		d := x[i] - y[i]
		s += d * d
	}
	return s
}

func parFor(workers, n int, fn func(lo, hi int)) {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 || n <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		lo := w * n / workers
		hi := (w + 1) * n / workers
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
