# superkmeans-go

A Go implementation of [SuperKMeans](https://github.com/cwida/SuperKMeans), a fast k-means clustering method for
high dimensional vector embeddings.
It aims to be faster than a naive Lloyd's k-means at similar quality.

⚠️ This is a side project for learning, not production code.

## How it works

SuperKMeans adds three ideas on top of Lloyd's algorithm.

1. A random orthogonal rotation of the data.
2. ADSampling pruning. A partial distance over a prefix of the rotated
   dimensions bounds the full distance. Most centroid comparisons stop early.
3. A blocked matrix multiply for the distance work that remains.

The prefix width is tuned each iteration. An optional hierarchical mode
clusters into round(sqrt(k)) mesoclusters first, then subdivides each.

## Build

```bash
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go build ./...
```

On amd64 the SIMD path uses AVX2. Build with GOAMD64=v3 or run on AVX2
hardware.

Without GOEXPERIMENT=simd the code uses the pure Go fallback.

```bash
go test ./...
go build ./...
```

## Example

```go
package main

import (
 "fmt"

 superkmeans "github.com/iangregson/superkmeans-go"
)

func main() {
   n, d, k := 10000, 128, 100
   data := make([]float32, n*d)
   // fill data row-major

   cfg := superkmeans.DefaultConfig()
   cfg.Seed = 42
   cfg.Iters = 10

   km, err := superkmeans.New(k, d, &cfg)
   if err != nil {
      panic(err)
   }
   model, err := km.Fit(data, n)
   if err != nil {
      panic(err)
   }

   fmt.Println(len(model.Centroids), model.WCSS)
   fmt.Println(model.Assignments[:5])
}
```

## API

`Fit` returns centroids, assignments, cluster sizes, WCSS, and per-iteration
stats. `FitCentroids` returns centroids only and skips the final assignment
pass. It is the training-time benchmark path.

## Bench

The standalone benchmark reads ann-benchmarks HDF5 files or generates synthetic
data.

```bash
mkdir -p data && cd data
wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
cd ..

GOEXPERIMENT=simd go run ./bench/standalone -data data/fashion-mnist-784-euclidean.hdf5 -iters 10 -runs 3 -naive
GOEXPERIMENT=simd go run ./bench/standalone -data data/fashion-mnist-784-euclidean.hdf5 -iters 10 -runs 3 -hier
GOEXPERIMENT=simd go run ./bench/standalone -n 20000 -d 128 -k 1024 -iters 10 -runs 3 -naive
GOEXPERIMENT=simd go run ./bench/standalone -n 20000 -d 128 -k 1024 -iters 10 -runs 3 -hier
```

The `-naive` flag runs an exact Lloyd's k-means with the same float32 data, the
same iteration count, and the same Forgy init. It is the baseline. The `-hier`
flag runs hierarchical mode. Angular datasets are detected from the filename.

The benchmark writes one JSON object per run. Fields include fit_seconds,
wcss, iterations, and recall when the HDF5 has test and neighbors datasets.
Use `-no-recall` to skip the recall pass on large datasets.

## Results

Apple M3 Max, 14 cores, Go 1.27.1, GOEXPERIMENT=simd. Flat and naive run 10
iterations. Hierarchical runs 3 meso and 5 fine iterations per phase, so its
iteration count is higher. recall@100 uses 1% centroid explore and is reported
where the dataset has test and neighbors.

| dataset | n | d | k | impl | iters | time | wcss | recall@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blobs | 20,000 | 128 | 1,024 | flat | 10 | 0.076 s | 5.11e7 | - |
| blobs | 20,000 | 128 | 1,024 | hier | 163 | 0.041 s | 4.41e7 | - |
| blobs | 20,000 | 128 | 1,024 | naive | 10 | 0.88 s | 5.28e7 | - |
| fashion-mnist | 60,000 | 784 | 980 | flat | 10 | 0.76 s | 5.80e10 | 0.879 |
| fashion-mnist | 60,000 | 784 | 980 | hier | 158 | 0.42 s | 5.91e10 | 0.861 |
| fashion-mnist | 60,000 | 784 | 980 | naive | 10 | 14.2 s | 5.81e10 | - |
| dbpedia-openai | 990,000 | 1,536 | 3,980 | flat | 10 | 94.8 s | 2.07e5 | 0.903 |
| dbpedia-openai | 990,000 | 1,536 | 3,980 | hier | 318 | 25.0 s | 2.11e5 | 0.889 |

## Limitations

- The HDF5 reader accepts only the ann-benchmarks layout.
- Accelerate / OpenBLAS would be faster.

## Acknowledgements

- [SuperKMeans](https://github.com/cwida/SuperKMeans)
- [pi.dev](https://pi.dev/)
- [Opair](https://www.opairdev.org/)
- [deepseek-flash / deepseek-v4-pro](https://platform.deepseek.com/)

