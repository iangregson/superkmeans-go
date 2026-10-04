//go:build goexperiment.simd && amd64

package superkmeans

import "simd/archsimd"

func kernel(m, n, kc int, ap, bp []float32, c []float32, ldc int, acc bool) {
	var a00, a01, a10, a11, a20, a21, a30, a31 archsimd.Float32x8
	ap = ap[:kc*mr]
	bp = bp[:kc*nr]
	for p := 0; p < kc; p++ {
		av := ap[p*mr : p*mr+mr]
		bv0 := archsimd.LoadFloat32x8(bp[p*nr : p*nr+8])
		bv1 := archsimd.LoadFloat32x8(bp[p*nr+8 : p*nr+16])
		v0 := archsimd.BroadcastFloat32x8(av[0])
		v1 := archsimd.BroadcastFloat32x8(av[1])
		v2 := archsimd.BroadcastFloat32x8(av[2])
		v3 := archsimd.BroadcastFloat32x8(av[3])
		a00 = v0.MulAdd(bv0, a00)
		a01 = v0.MulAdd(bv1, a01)
		a10 = v1.MulAdd(bv0, a10)
		a11 = v1.MulAdd(bv1, a11)
		a20 = v2.MulAdd(bv0, a20)
		a21 = v2.MulAdd(bv1, a21)
		a30 = v3.MulAdd(bv0, a30)
		a31 = v3.MulAdd(bv1, a31)
	}
	if m == 4 && n == 16 {
		if acc {
			c00 := archsimd.LoadFloat32x8(c[0*ldc : 0*ldc+8])
			c01 := archsimd.LoadFloat32x8(c[0*ldc+8 : 0*ldc+16])
			c10 := archsimd.LoadFloat32x8(c[1*ldc : 1*ldc+8])
			c11 := archsimd.LoadFloat32x8(c[1*ldc+8 : 1*ldc+16])
			c20 := archsimd.LoadFloat32x8(c[2*ldc : 2*ldc+8])
			c21 := archsimd.LoadFloat32x8(c[2*ldc+8 : 2*ldc+16])
			c30 := archsimd.LoadFloat32x8(c[3*ldc : 3*ldc+8])
			c31 := archsimd.LoadFloat32x8(c[3*ldc+8 : 3*ldc+16])
			a00.Add(c00).Store(c[0*ldc : 0*ldc+8])
			a01.Add(c01).Store(c[0*ldc+8 : 0*ldc+16])
			a10.Add(c10).Store(c[1*ldc : 1*ldc+8])
			a11.Add(c11).Store(c[1*ldc+8 : 1*ldc+16])
			a20.Add(c20).Store(c[2*ldc : 2*ldc+8])
			a21.Add(c21).Store(c[2*ldc+8 : 2*ldc+16])
			a30.Add(c30).Store(c[3*ldc : 3*ldc+8])
			a31.Add(c31).Store(c[3*ldc+8 : 3*ldc+16])
			return
		}
		a00.Store(c[0*ldc : 0*ldc+8])
		a01.Store(c[0*ldc+8 : 0*ldc+16])
		a10.Store(c[1*ldc : 1*ldc+8])
		a11.Store(c[1*ldc+8 : 1*ldc+16])
		a20.Store(c[2*ldc : 2*ldc+8])
		a21.Store(c[2*ldc+8 : 2*ldc+16])
		a30.Store(c[3*ldc : 3*ldc+8])
		a31.Store(c[3*ldc+8 : 3*ldc+16])
		return
	}
	rows := [4][2]archsimd.Float32x8{{a00, a01}, {a10, a11}, {a20, a21}, {a30, a31}}
	for rr := 0; rr < m; rr++ {
		row := c[rr*ldc : rr*ldc+n]
		for cc := 0; cc < n; cc += 8 {
			part := row[cc:min(cc+8, n)]
			prev, _ := archsimd.LoadFloat32x8Part(part)
			v := rows[rr][cc/8]
			if acc {
				v = v.Add(prev)
			}
			v.StorePart(part)
		}
	}
}
