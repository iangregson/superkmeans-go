//go:build !goexperiment.simd || (!arm64 && !amd64)

package superkmeans

func l2Sq(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var a0, a1, a2, a3, a4, a5, a6, a7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		xs := x[i : i+8 : i+8]
		ys := y[i : i+8 : i+8]
		d0 := xs[0] - ys[0]
		d1 := xs[1] - ys[1]
		d2 := xs[2] - ys[2]
		d3 := xs[3] - ys[3]
		d4 := xs[4] - ys[4]
		d5 := xs[5] - ys[5]
		d6 := xs[6] - ys[6]
		d7 := xs[7] - ys[7]
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
