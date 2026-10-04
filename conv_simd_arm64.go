//go:build goexperiment.simd && arm64

package superkmeans

import "simd/archsimd"

func convL2(row, nyv []float32, nx float32) {
	n := len(row)
	if len(nyv) < n {
		n = len(nyv)
	}
	nxv := archsimd.BroadcastFloat32x4(nx)
	neg2 := archsimd.BroadcastFloat32x4(-2)
	i := 0
	for ; i+4 <= n; i += 4 {
		r := archsimd.LoadFloat32x4(row[i:])
		y := archsimd.LoadFloat32x4(nyv[i:])
		r = nxv.Add(y).Add(r.Mul(neg2))
		r.Store(row[i:])
	}
	for ; i < n; i++ {
		row[i] = nx + nyv[i] - 2*row[i]
	}
}
