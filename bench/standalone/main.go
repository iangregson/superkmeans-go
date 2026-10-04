package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	superkmeans "github.com/iangregson/superkmeans-go"
	"github.com/iangregson/superkmeans-go/bench/h5"
)

func main() {
	var dataPath, outPath string
	var n, d, k, seed, iters, threads, runs int
	var sampling, explore float64
	var hier, noRecall, naive bool
	flag.StringVar(&dataPath, "data", "", "path to ann-benchmarks HDF5 file")
	flag.StringVar(&outPath, "out", "", "append JSONL results here (default stdout)")
	flag.IntVar(&n, "n", 20000, "synthetic rows")
	flag.IntVar(&d, "d", 128, "synthetic dimensions")
	flag.IntVar(&k, "k", 0, "clusters (0 = auto: sqrt(n)*4)")
	flag.IntVar(&seed, "seed", 42, "seed")
	flag.IntVar(&iters, "iters", 10, "iterations")
	flag.IntVar(&threads, "threads", 0, "threads (0 = all)")
	flag.IntVar(&runs, "runs", 3, "repeats")
	flag.Float64Var(&sampling, "sampling", 1.0, "sampling fraction")
	flag.Float64Var(&explore, "explore", 0.01, "centroid explore fraction for recall")
	flag.BoolVar(&hier, "hier", false, "hierarchical clustering")
	flag.BoolVar(&noRecall, "no-recall", false, "skip recall and full assignment")
	flag.BoolVar(&naive, "naive", false, "run exact Lloyd's baseline")
	flag.Parse()

	var data []float32
	var test []float32
	var neighbors []int32
	angular := false
	dataset := "blobs"

	if dataPath != "" {
		f, err := h5.Open(dataPath)
		if err != nil {
			fatal(err)
		}
		defer f.Close()
		train, err := f.Lookup("train")
		if err != nil {
			fatal(err)
		}
		n = train.Rows()
		d = train.RowLen()
		data, err = readFloats(f, train, n)
		if err != nil {
			fatal(err)
		}
		if ts, err := f.Lookup("test"); err == nil {
			m := ts.Rows()
			if ts.RowLen() == d {
				test, err = readFloats(f, ts, m)
				if err != nil {
					fatal(err)
				}
			}
		}
		if nb, err := f.Lookup("neighbors"); err == nil && test != nil {
			m := nb.Rows()
			neighbors, err = readInts(f, nb, m)
			if err != nil {
				fatal(err)
			}
		}
		dataset = strings.TrimSuffix(baseName(dataPath), ".hdf5")
		angular = strings.Contains(strings.ToLower(dataset), "angular")
		if k == 0 {
			k = int(math.Round(math.Sqrt(float64(n)) * 4))
		}
	} else {
		if k == 0 {
			k = 1024
		}
		data = makeBlobs(n, d, k, uint64(seed))
	}

	w := os.Stdout
	if outPath != "" {
		f, err := os.OpenFile(outPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			fatal(err)
		}
		defer f.Close()
		w = f
	}

	for run := 0; run < runs; run++ {
		cfg := superkmeans.DefaultConfig()
		cfg.Iters = iters
		cfg.Seed = uint64(seed)
		cfg.Threads = threads
		cfg.SamplingFraction = sampling
		cfg.Hierarchical = hier
		cfg.Angular = angular
		cfg.EarlyTermination = false
		km, err := superkmeans.New(k, d, &cfg)
		if err != nil {
			fatal(err)
		}
		t0 := time.Now()
		centroids, err := km.FitCentroids(data, n)
		if err != nil {
			fatal(err)
		}
		fit := time.Since(t0).Seconds()

		rec := -1.0
		var assignSeconds float64
		if test != nil && len(neighbors) > 0 && !noRecall {
			model := &superkmeans.Model{K: k, D: d, Centroids: centroids}
			trainAssign := make([]uint32, n)
			a0 := time.Now()
			if err := model.Assign(data, n, trainAssign, nil); err != nil {
				fatal(err)
			}
			assignSeconds = time.Since(a0).Seconds()
			rec = computeRecall(test, centroids, trainAssign, neighbors, k, d, explore)
		}

		out := map[string]any{
			"dataset":        dataset,
			"n":              n,
			"d":              d,
			"k":              k,
			"impl":           implName(hier),
			"angular":        angular,
			"threads":        threads,
			"iters":          iters,
			"sampling":       sampling,
			"run":            run + 1,
			"fit_seconds":    fit,
			"wcss":           km.WCSS(),
			"iterations":     km.Iterations(),
			"assign_seconds": assignSeconds,
			"recall":         rec,
			"explore":        explore,
		}
		enc := json.NewEncoder(w)
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}

		if naive {
			nt := usedThreads(threads)
			t0 := time.Now()
			_, nwcss := naiveLloyd(data, n, d, k, uint64(seed), iters)
			nfit := time.Since(t0).Seconds()
			nout := map[string]any{
				"dataset":        dataset,
				"n":              n,
				"d":              d,
				"k":              k,
				"impl":           "naive",
				"angular":        angular,
				"threads":        nt,
				"iters":          iters,
				"sampling":       sampling,
				"run":            run + 1,
				"fit_seconds":    nfit,
				"wcss":           nwcss,
				"iterations":     iters,
				"assign_seconds": 0.0,
				"recall":         -1.0,
				"explore":        explore,
			}
			if err := enc.Encode(nout); err != nil {
				fatal(err)
			}
		}

	}
}

func usedThreads(t int) int {
	if t > 0 {
		return t
	}
	return runtime.GOMAXPROCS(0)
}

func implName(hier bool) string {
	if hier {
		return "hierarchical"
	}
	return "flat"
}

func baseName(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		p = p[i+1:]
	}
	return p
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "bench:", err)
	os.Exit(1)
}

func readFloats(f *h5.File, ds h5.Dataset, rows int) ([]float32, error) {
	n := rows * ds.RowLen()
	out := make([]float32, n)
	switch ds.Kind {
	case h5.F32:
		return out, f.ReadF32(ds, 0, rows, out)
	case h5.F64:
		tmp := make([]float64, n)
		if err := f.ReadF64(ds, 0, rows, tmp); err != nil {
			return nil, err
		}
		for i, v := range tmp {
			out[i] = float32(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("h5: %s is %s, want f32 or f64", ds.Name, ds.Kind)
}

func readInts(f *h5.File, ds h5.Dataset, rows int) ([]int32, error) {
	n := rows * ds.RowLen()
	out := make([]int32, n)
	switch ds.Kind {
	case h5.I32:
		return out, f.ReadI32(ds, 0, rows, out)
	case h5.I64:
		tmp := make([]int64, n)
		if err := f.ReadI64(ds, 0, rows, tmp); err != nil {
			return nil, err
		}
		for i, v := range tmp {
			out[i] = int32(v)
		}
		return out, nil
	}
	return nil, fmt.Errorf("h5: %s is %s, want i32 or i64", ds.Name, ds.Kind)
}

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
	mean := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			mean[j] += float64(data[i*d+j])
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			data[i*d+j] -= float32(mean[j])
		}
	}
	return data
}

func computeRecall(test, centroids []float32, trainAssign []uint32, neighbors []int32, k, d int, explore float64) float64 {
	m := len(test) / d
	gtK := len(neighbors) / m
	if m == 0 || gtK == 0 {
		return -1
	}
	top := max(1, int(float64(k)*explore))
	topIdx := topCentroids(test, centroids, m, k, d, top)

	var total float64
	for q := 0; q < m; q++ {
		inTop := make(map[uint32]bool, top)
		for _, c := range topIdx[q] {
			inTop[uint32(c)] = true
		}
		found := 0
		for g := 0; g < gtK; g++ {
			idx := neighbors[q*gtK+g]
			if idx < 0 || int(idx) >= len(trainAssign) {
				continue
			}
			if inTop[trainAssign[idx]] {
				found++
			}
		}
		total += float64(found) / float64(gtK)
	}
	return total / float64(m)
}

func topCentroids(test, centroids []float32, m, k, d, t int) [][]int32 {
	out := make([][]int32, m)
	workers := runtime.GOMAXPROCS(0)
	if workers > m {
		workers = m
	}
	if workers <= 1 {
		for q := 0; q < m; q++ {
			out[q] = nearestN(test[q*d:q*d+d], centroids, k, d, t)
		}
		return out
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		lo := w * m / workers
		hi := (w + 1) * m / workers
		go func(lo, hi int) {
			defer wg.Done()
			for q := lo; q < hi; q++ {
				out[q] = nearestN(test[q*d:q*d+d], centroids, k, d, t)
			}
		}(lo, hi)
	}
	wg.Wait()
	return out
}

func nearestN(query, centroids []float32, k, d, t int) []int32 {
	type pair struct {
		dist float32
		idx  int32
	}
	dists := make([]pair, k)
	for c := 0; c < k; c++ {
		var s float32
		base := c * d
		for j := 0; j < d; j++ {
			diff := query[j] - centroids[base+j]
			s += diff * diff
		}
		dists[c] = pair{s, int32(c)}
	}
	sort.Slice(dists, func(i, j int) bool { return dists[i].dist < dists[j].dist })
	if t > k {
		t = k
	}
	out := make([]int32, t)
	for i := 0; i < t; i++ {
		out[i] = dists[i].idx
	}
	return out
}
