//go:build goexperiment.simd && amd64

package superkmeans

import "simd/archsimd"

func l2Sq(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var acc archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		xv := archsimd.LoadFloat32x8(x[i:])
		yv := archsimd.LoadFloat32x8(y[i:])
		d := xv.Sub(yv)
		acc = acc.Add(d.Mul(d))
	}
	var tmp [8]float32
	acc.StoreArray(&tmp)
	s := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	for ; i < n; i++ {
		d := x[i] - y[i]
		s += d * d
	}
	return s
}
