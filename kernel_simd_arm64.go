//go:build goexperiment.simd && arm64

package superkmeans

import "simd/archsimd"

func kernel(m, n, kc int, ap, bp []float32, c []float32, ldc int, acc bool) {
	var r00, r01, r02, r03 archsimd.Float32x4
	var r10, r11, r12, r13 archsimd.Float32x4
	var r20, r21, r22, r23 archsimd.Float32x4
	var r30, r31, r32, r33 archsimd.Float32x4
	ap = ap[:kc*mr]
	bp = bp[:kc*nr]
	for p := 0; p < kc; p++ {
		av := ap[p*mr : p*mr+mr]
		bv0 := archsimd.LoadFloat32x4(bp[p*nr : p*nr+4])
		bv1 := archsimd.LoadFloat32x4(bp[p*nr+4 : p*nr+8])
		bv2 := archsimd.LoadFloat32x4(bp[p*nr+8 : p*nr+12])
		bv3 := archsimd.LoadFloat32x4(bp[p*nr+12 : p*nr+16])
		v0 := archsimd.BroadcastFloat32x4(av[0])
		v1 := archsimd.BroadcastFloat32x4(av[1])
		v2 := archsimd.BroadcastFloat32x4(av[2])
		v3 := archsimd.BroadcastFloat32x4(av[3])
		r00 = v0.MulAdd(bv0, r00)
		r01 = v0.MulAdd(bv1, r01)
		r02 = v0.MulAdd(bv2, r02)
		r03 = v0.MulAdd(bv3, r03)
		r10 = v1.MulAdd(bv0, r10)
		r11 = v1.MulAdd(bv1, r11)
		r12 = v1.MulAdd(bv2, r12)
		r13 = v1.MulAdd(bv3, r13)
		r20 = v2.MulAdd(bv0, r20)
		r21 = v2.MulAdd(bv1, r21)
		r22 = v2.MulAdd(bv2, r22)
		r23 = v2.MulAdd(bv3, r23)
		r30 = v3.MulAdd(bv0, r30)
		r31 = v3.MulAdd(bv1, r31)
		r32 = v3.MulAdd(bv2, r32)
		r33 = v3.MulAdd(bv3, r33)
	}
	if m == 4 && n == 16 {
		if acc {
			c00 := archsimd.LoadFloat32x4(c[0*ldc : 0*ldc+4])
			c01 := archsimd.LoadFloat32x4(c[0*ldc+4 : 0*ldc+8])
			c02 := archsimd.LoadFloat32x4(c[0*ldc+8 : 0*ldc+12])
			c03 := archsimd.LoadFloat32x4(c[0*ldc+12 : 0*ldc+16])
			c10 := archsimd.LoadFloat32x4(c[1*ldc : 1*ldc+4])
			c11 := archsimd.LoadFloat32x4(c[1*ldc+4 : 1*ldc+8])
			c12 := archsimd.LoadFloat32x4(c[1*ldc+8 : 1*ldc+12])
			c13 := archsimd.LoadFloat32x4(c[1*ldc+12 : 1*ldc+16])
			c20 := archsimd.LoadFloat32x4(c[2*ldc : 2*ldc+4])
			c21 := archsimd.LoadFloat32x4(c[2*ldc+4 : 2*ldc+8])
			c22 := archsimd.LoadFloat32x4(c[2*ldc+8 : 2*ldc+12])
			c23 := archsimd.LoadFloat32x4(c[2*ldc+12 : 2*ldc+16])
			c30 := archsimd.LoadFloat32x4(c[3*ldc : 3*ldc+4])
			c31 := archsimd.LoadFloat32x4(c[3*ldc+4 : 3*ldc+8])
			c32 := archsimd.LoadFloat32x4(c[3*ldc+8 : 3*ldc+12])
			c33 := archsimd.LoadFloat32x4(c[3*ldc+12 : 3*ldc+16])
			r00.Add(c00).Store(c[0*ldc : 0*ldc+4])
			r01.Add(c01).Store(c[0*ldc+4 : 0*ldc+8])
			r02.Add(c02).Store(c[0*ldc+8 : 0*ldc+12])
			r03.Add(c03).Store(c[0*ldc+12 : 0*ldc+16])
			r10.Add(c10).Store(c[1*ldc : 1*ldc+4])
			r11.Add(c11).Store(c[1*ldc+4 : 1*ldc+8])
			r12.Add(c12).Store(c[1*ldc+8 : 1*ldc+12])
			r13.Add(c13).Store(c[1*ldc+12 : 1*ldc+16])
			r20.Add(c20).Store(c[2*ldc : 2*ldc+4])
			r21.Add(c21).Store(c[2*ldc+4 : 2*ldc+8])
			r22.Add(c22).Store(c[2*ldc+8 : 2*ldc+12])
			r23.Add(c23).Store(c[2*ldc+12 : 2*ldc+16])
			r30.Add(c30).Store(c[3*ldc : 3*ldc+4])
			r31.Add(c31).Store(c[3*ldc+4 : 3*ldc+8])
			r32.Add(c32).Store(c[3*ldc+8 : 3*ldc+12])
			r33.Add(c33).Store(c[3*ldc+12 : 3*ldc+16])
			return
		}
		r00.Store(c[0*ldc : 0*ldc+4])
		r01.Store(c[0*ldc+4 : 0*ldc+8])
		r02.Store(c[0*ldc+8 : 0*ldc+12])
		r03.Store(c[0*ldc+12 : 0*ldc+16])
		r10.Store(c[1*ldc : 1*ldc+4])
		r11.Store(c[1*ldc+4 : 1*ldc+8])
		r12.Store(c[1*ldc+8 : 1*ldc+12])
		r13.Store(c[1*ldc+12 : 1*ldc+16])
		r20.Store(c[2*ldc : 2*ldc+4])
		r21.Store(c[2*ldc+4 : 2*ldc+8])
		r22.Store(c[2*ldc+8 : 2*ldc+12])
		r23.Store(c[2*ldc+12 : 2*ldc+16])
		r30.Store(c[3*ldc : 3*ldc+4])
		r31.Store(c[3*ldc+4 : 3*ldc+8])
		r32.Store(c[3*ldc+8 : 3*ldc+12])
		r33.Store(c[3*ldc+12 : 3*ldc+16])
		return
	}
	rows := [4][4]archsimd.Float32x4{{r00, r01, r02, r03}, {r10, r11, r12, r13}, {r20, r21, r22, r23}, {r30, r31, r32, r33}}
	for rr := 0; rr < m; rr++ {
		row := c[rr*ldc : rr*ldc+n]
		for cc := 0; cc < n; cc += 4 {
			part := row[cc:min(cc+4, n)]
			prev, _ := archsimd.LoadFloat32x4Part(part)
			v := rows[rr][cc/4]
			if acc {
				v = v.Add(prev)
			}
			v.StorePart(part)
		}
	}
}
