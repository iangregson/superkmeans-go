//go:build goexperiment.simd && amd64

package superkmeans

import "simd/archsimd"

func convL2(row, nyv []float32, nx float32) {
	n := len(row)
	if len(nyv) < n {
		n = len(nyv)
	}
	nxv := archsimd.BroadcastFloat32x8(nx)
	neg2 := archsimd.BroadcastFloat32x8(-2)
	i := 0
	for ; i+8 <= n; i += 8 {
		r := archsimd.LoadFloat32x8(row[i:])
		y := archsimd.LoadFloat32x8(nyv[i:])
		r = nxv.Add(y).Add(r.Mul(neg2))
		r.Store(row[i:])
	}
	for ; i < n; i++ {
		row[i] = nx + nyv[i] - 2*row[i]
	}
}
