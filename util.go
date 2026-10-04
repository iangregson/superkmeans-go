package superkmeans

import (
	"math"
	"math/bits"
	"runtime"
	"sync"
)

const (
	xBatch = 8192
	yBatch = 1024

	minPartialD = 16
	dGate       = 128
	kGate       = 256

	perturbEps = 1.0 / 1024

	balanceWeight    = 7
	balanceThreshold = 0.25

	eps0     = 1.5
	eps0Hier = 1.1
)

func normWorkers(w int) int {
	if w <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return w
}

type pool struct {
	tasks chan func()
}

func newPool(n int) *pool {
	if n < 1 {
		n = 1
	}
	p := &pool{tasks: make(chan func(), n*4)}
	for i := 0; i < n; i++ {
		go func() {
			for f := range p.tasks {
				f()
			}
		}()
	}
	return p
}

var (
	globalPoolOnce sync.Once
	globalPool     *pool
)

func getPool() *pool {
	globalPoolOnce.Do(func() {
		globalPool = newPool(runtime.GOMAXPROCS(0))
	})
	return globalPool
}

func parallelFor(workers, n, grain int, fn func(worker, lo, hi int)) {
	if n <= 0 {
		return
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 || n <= 1 || int64(n)*int64(grain) < 1<<16 {
		fn(0, 0, n)
		return
	}
	p := getPool()
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		lo := w * n / workers
		hi := (w + 1) * n / workers
		if lo >= hi {
			wg.Done()
			continue
		}
		w, lo, hi := w, lo, hi
		p.tasks <- func() {
			fn(w, lo, hi)
			wg.Done()
		}
	}
	wg.Wait()
}

func deriveSeed(seed, stream uint64) uint64 {
	x := seed + stream*0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

type rng struct {
	s uint64
}

func newRNG(seed uint64) *rng {
	return &rng{s: seed}
}

func splitmix64(x *uint64) uint64 {
	*x += 0x9e3779b97f4a7c15
	z := *x
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *rng) u64() uint64 {
	return splitmix64(&r.s)
}

func (r *rng) float64() float64 {
	return float64(r.u64()>>11) / (1 << 53)
}

func (r *rng) intn(n int) int {
	if n <= 0 {
		panic("rng: intn <= 0")
	}
	return int(r.u64n(uint64(n)))
}

func (r *rng) u64n(n uint64) uint64 {
	hi, lo := bits.Mul64(r.u64(), n)
	if lo < n {
		thresh := -n % n
		for lo < thresh {
			hi, lo = bits.Mul64(r.u64(), n)
		}
	}
	return hi
}

func (r *rng) norm() float32 {
	u1 := r.float64() + 1e-300
	u2 := r.float64()
	return float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
}

func (r *rng) perm(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	for i := n - 1; i > 0; i-- {
		j := r.intn(i + 1)
		p[i], p[j] = p[j], p[i]
	}
	return p
}

func squaredNormsPrefix(data []float32, n, d, p int, out []float32, workers int) {
	if p > d {
		p = d
	}
	if p < 0 {
		p = 0
	}
	parallelFor(workers, n, p, func(w, lo, hi int) {
		for i := lo; i < hi; i++ {
			row := data[i*d : i*d+p]
			out[i] = sqnorm(row)
		}
	})
}

func squaredNorms(data []float32, n, d int, out []float32, workers int) {
	squaredNormsPrefix(data, n, d, d, out, workers)
}

func sqnorm(v []float32) float32 {
	n := len(v)
	var a0, a1, a2, a3, a4, a5, a6, a7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		vs := v[i : i+8 : i+8]
		a0 += vs[0] * vs[0]
		a1 += vs[1] * vs[1]
		a2 += vs[2] * vs[2]
		a3 += vs[3] * vs[3]
		a4 += vs[4] * vs[4]
		a5 += vs[5] * vs[5]
		a6 += vs[6] * vs[6]
		a7 += vs[7] * vs[7]
	}
	s := a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
	for ; i < n; i++ {
		s += v[i] * v[i]
	}
	return s
}

func l2BlockAdd(dst []float32, idx []uint32, q, y []float32, stride, off int) {
	block := len(q)
	for _, p := range idx {
		base := int(p)*stride + off
		dst[p] += l2Sq(q, y[base:base+block])
	}
}

func scaleRow(s float32, row []float32) {
	for i := range row {
		row[i] *= s
	}
}

func clusterSizes(assign []uint32, k int) []uint32 {
	out := make([]uint32, k)
	for _, a := range assign {
		if int(a) < k {
			out[a]++
		}
	}
	return out
}

func sumFloat32(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v)
	}
	return s
}

func ceilDiv(a, b int) int {
	return (a + b - 1) / b
}

func transposeSquare(m []float32, n int) []float32 {
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[j*n+i] = m[i*n+j]
		}
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
