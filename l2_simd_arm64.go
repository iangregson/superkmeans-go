//go:build goexperiment.simd && arm64

package superkmeans

import "simd/archsimd"

func l2Sq(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var acc archsimd.Float32x4
	i := 0
	for ; i+4 <= n; i += 4 {
		xv := archsimd.LoadFloat32x4(x[i:])
		yv := archsimd.LoadFloat32x4(y[i:])
		d := xv.Sub(yv)
		acc = acc.Add(d.Mul(d))
	}
	t := acc.ConcatAddPairs(acc)
	t = t.ConcatAddPairs(t)
	s := t.GetElem(0)
	for ; i < n; i++ {
		d := x[i] - y[i]
		s += d * d
	}
	return s
}
