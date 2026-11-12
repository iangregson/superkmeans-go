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

## e-means

The package also includes a Go port of e-means
([paper](https://arxiv.org/abs/2611.09501),
[code](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means),
[blog](https://e-means.github.io/)). e-means is a mini-batch k-means with
exponential moving average updates:

```
N_j <- (1-alpha) * N_j + alpha * n_j      # running cluster count
S_j <- (1-alpha) * S_j + alpha * s_j      # running sum of assigned points
c_j  = S_j / N_j                          # the centroid
```

Batch size and initial step size come from fitted scaling laws. The step size
follows a cosine schedule down to 0.1x its initial value. Every five updates,
clusters with mass below 0.1x the mean are revived by splitting heavy
clusters. The returned centroids are the exact means of the last pass's
assignments. Assignment uses the same blocked GEMM as superkmeans instead of
faiss.knn.

```go
cfg := superkmeans.DefaultEMeansConfig() // 5 epochs, auto scaling laws
em, err := superkmeans.NewEMeans(k, d, &cfg)
if err != nil {
 panic(err)
}
model, err := em.Fit(data, n) // model.Centroids, model.Assignments, model.Recipe
```

Set `cfg.Spherical = true` on L2-normalized data to cluster by direction
(cosine similarity). Run it from the benchmark with `-emeans` (`-emepochs`
sets the epoch budget):

```bash
GOEXPERIMENT=simd go run ./bench/standalone -data data/fashion-mnist-784-euclidean.hdf5 -runs 1 -emeans
```

The port is validated against the FAISS reference (`NumpyEmeans`). Resolved
recipes (batch size, alpha0) are identical to printed precision. Final
objectives are within 0.1% on fashion-mnist, on a dbpedia subset in both
Euclidean and spherical mode, and on full dbpedia (2.0476e5 vs 2.0482e5).
Per-epoch objective trajectories and split counts also match. RNG streams
differ, so results are not bit-identical.

## Benchmark

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
| blobs | 20,000 | 128 | 1,024 | emeans | 5 | 0.114 s | 3.39e7 | - |
| fashion-mnist | 60,000 | 784 | 980 | flat | 10 | 0.76 s | 5.80e10 | 0.879 |
| fashion-mnist | 60,000 | 784 | 980 | hier | 158 | 0.42 s | 5.91e10 | 0.861 |
| fashion-mnist | 60,000 | 784 | 980 | naive | 10 | 14.2 s | 5.81e10 | - |
| fashion-mnist | 60,000 | 784 | 980 | emeans | 5 | 1.35 s | 5.70e10 | 0.876 |
| dbpedia-openai | 990,000 | 1,536 | 3,980 | flat | 10 | 96.9 s | 2.07e5 | 0.903 |
| dbpedia-openai | 990,000 | 1,536 | 3,980 | hier | 318 | 25.0 s | 2.11e5 | 0.889 |
| dbpedia-openai | 990,000 | 1,536 | 3,980 | emeans (spherical) | 5 | 267.9 s | 2.05e5 | 0.911 |

## e-means vs superkmeans

Same machine, same data. WCSS is the exact L2 error of the final centroids,
measured with a separate assignment pass for both. e-means runs its default 5
epochs with auto scaling laws. Flat superkmeans runs 10 iterations on all
rows.

- e-means reaches lower loss on all three datasets: -34% on blobs, -1.8% on
  fashion-mnist, -0.9% on dbpedia, in half the passes. On dbpedia-angular,
  spherical e-means also gives the best recall (0.911).
- superkmeans is 1.4-2.8x faster in wall-clock. Each pass is cheaper:
  ADSampling pruning skips most distance work, while e-means assigns every
  batch exactly. On dbpedia one superkmeans iteration costs ~10 s vs ~54 s
  for one e-means epoch.
- The two approaches are orthogonal. e-means reduces the number of passes.
  superkmeans reduces the cost per pass. A pruned assignment inside the
  e-means update loop would combine both.
- e-means revives only starving clusters. With unlucky random init on
  well-separated blobs, merged clusters are never rescued. The FAISS
  reference behaves the same way. superkmeans also splits empty and small
  clusters, which handles this case better.
- The closing-mean step lowers the objective below the moving last-epoch
  value (206.0k to 204.8k on dbpedia).

## Limitations

- The HDF5 reader accepts only the ann-benchmarks layout.
- Accelerate / OpenBLAS would be faster.

## Acknowledgements

- [SuperKMeans](https://github.com/cwida/SuperKMeans)
- [pi.dev](https://pi.dev/)
- [Opair](https://www.opairdev.org/)
- [deepseek-flash / deepseek-v4-pro](https://platform.deepseek.com/)

