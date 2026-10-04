//go:build !goexperiment.simd || (!arm64 && !amd64)

package superkmeans

func convL2(row, nyv []float32, nx float32) {
	n := len(row)
	if len(nyv) < n {
		n = len(nyv)
	}
	for c := 0; c < n; c++ {
		row[c] = nx + nyv[c] - 2*row[c]
	}
}
