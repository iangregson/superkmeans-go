package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	superkmeans "github.com/iangregson/superkmeans-go"
)

func main() {
	var dataPath, outCent, outAssign string
	var n, d, k, seed, iters, threads int
	var sampling float64
	var early bool
	flag.StringVar(&dataPath, "data", "", "input f32 data file")
	flag.IntVar(&n, "n", 0, "number of rows")
	flag.IntVar(&d, "d", 0, "dimensions")
	flag.IntVar(&k, "k", 0, "clusters")
	flag.IntVar(&seed, "seed", 42, "seed")
	flag.IntVar(&iters, "iters", 10, "iterations")
	flag.IntVar(&threads, "threads", 0, "threads")
	flag.Float64Var(&sampling, "sampling-fraction", 0.3, "sampling fraction")
	flag.BoolVar(&early, "early-termination", false, "early termination")
	flag.StringVar(&outCent, "out-centroids", "", "centroids output path")
	flag.StringVar(&outAssign, "out-assign", "", "assignments output path")
	flag.Parse()

	if dataPath == "" || n <= 0 || d <= 0 || k <= 0 || outCent == "" {
		fmt.Fprintln(os.Stderr, "missing required flags")
		os.Exit(1)
	}
	data := readF32(dataPath, n*d)

	cfg := superkmeans.DefaultConfig()
	cfg.Iters = iters
	cfg.SamplingFraction = sampling
	cfg.Threads = threads
	cfg.Seed = uint64(seed)
	cfg.EarlyTermination = early
	cfg.UnrotateCentroids = true

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

	writeF32(outCent, centroids)
	assignSeconds := 0.0
	if outAssign != "" {
		a0 := time.Now()
		out := make([]uint32, n)
		m := &superkmeans.Model{K: k, D: d, Centroids: centroids}
		if err := m.Assign(data, n, out, nil); err != nil {
			fatal(err)
		}
		assignSeconds = time.Since(a0).Seconds()
		writeU32(outAssign, out)
	}

	rep := map[string]any{
		"fit_seconds": fit,
		"iterations":  km.Iterations(),
		"impl":        "superkmeans-go (new port)",
		"extra": map[string]any{
			"assign_seconds": assignSeconds,
			"wcss":           km.WCSS(),
			"threads":        threads,
		},
	}
	enc := json.NewEncoder(os.Stdout)
	enc.Encode(rep)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "runner:", err)
	os.Exit(1)
}

func readF32(path string, count int) []float32 {
	b, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	if len(b) != count*4 {
		panic(fmt.Sprintf("data file has %d bytes, want %d", len(b), count*4))
	}
	out := make([]float32, count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func writeF32(path string, vals []float32) {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		panic(err)
	}
}

func writeU32(path string, vals []uint32) {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		panic(err)
	}
}
