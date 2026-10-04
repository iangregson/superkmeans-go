//go:build !goexperiment.simd || (!arm64 && !amd64)

package superkmeans

func kernel(m, n, kc int, ap, bp []float32, c []float32, ldc int, acc bool) {
	kernelHalf(m, min(n, 4), kc, ap, bp, 0, c, ldc, acc)
	if n > 4 {
		kernelHalf(m, min(n-4, 4), kc, ap, bp, 4, c[4:], ldc, acc)
	}
	if n > 8 {
		kernelHalf(m, min(n-8, 4), kc, ap, bp, 8, c[8:], ldc, acc)
	}
	if n > 12 {
		kernelHalf(m, min(n-12, 4), kc, ap, bp, 12, c[12:], ldc, acc)
	}
}

func kernelHalf(m, n, kc int, ap, bp []float32, colOff int, c []float32, ldc int, acc bool) {
	var s00, s01, s02, s03 float32
	var s10, s11, s12, s13 float32
	var s20, s21, s22, s23 float32
	var s30, s31, s32, s33 float32
	ap = ap[:kc*mr]
	for p := 0; p < kc; p++ {
		av := ap[p*mr : p*mr+mr]
		bv := bp[p*nr+colOff : p*nr+colOff+4]
		a0, a1, a2, a3 := av[0], av[1], av[2], av[3]
		b0, b1, b2, b3 := bv[0], bv[1], bv[2], bv[3]
		s00 += a0 * b0
		s01 += a0 * b1
		s02 += a0 * b2
		s03 += a0 * b3
		s10 += a1 * b0
		s11 += a1 * b1
		s12 += a1 * b2
		s13 += a1 * b3
		s20 += a2 * b0
		s21 += a2 * b1
		s22 += a2 * b2
		s23 += a2 * b3
		s30 += a3 * b0
		s31 += a3 * b1
		s32 += a3 * b2
		s33 += a3 * b3
	}
	sum := [4][4]float32{{s00, s01, s02, s03}, {s10, s11, s12, s13}, {s20, s21, s22, s23}, {s30, s31, s32, s33}}
	if acc {
		for r := 0; r < m; r++ {
			row := c[r*ldc : r*ldc+n]
			for s := 0; s < n; s++ {
				row[s] += sum[r][s]
			}
		}
		return
	}
	for r := 0; r < m; r++ {
		row := c[r*ldc : r*ldc+n]
		for s := 0; s < n; s++ {
			row[s] = sum[r][s]
		}
	}
}
