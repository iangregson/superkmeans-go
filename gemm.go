package superkmeans

const (
	bm = 64
	bn = 1024
	bk = 256
	mr = 4
	nr = 16
)

func gemmNTPost(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, workers int, post func(w, lo, hi int)) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	p := normWorkers(workers)
	if m*n*k < 1<<22 {
		p = 1
	}
	scratch := make([][]float32, p)
	for w := range scratch {
		scratch[w] = make([]float32, bm*bk)
	}
	type tile struct {
		k0 int
		kk int
		bp []float32
	}
	tiles := make([]tile, 0, ceilDiv(k, bk))
	for k0 := 0; k0 < k; k0 += bk {
		kk := min(bk, k-k0)
		tiles = append(tiles, tile{k0, kk, packB(b, ldb, 0, n, k0, kk)})
	}
	blocks := ceilDiv(m, bm)
	parallelFor(p, blocks, bm*n*k, func(w, lo, hi int) {
		ap := scratch[w]
		for blk := lo; blk < hi; blk++ {
			i0 := blk * bm
			im := min(bm, m-i0)
			for _, t := range tiles {
				packA(ap, a, lda, i0, im, t.k0, t.kk)
				macro(im, n, t.kk, ap, t.bp, c[i0*ldc:], ldc, t.k0 > 0)
			}
			post(w, i0, i0+im)
		}
	})
}

func sgemmNT(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	p := normWorkers(workers)
	if m*n*k < 1<<22 {
		p = 1
	}
	scratch := make([][]float32, p)
	for w := range scratch {
		scratch[w] = make([]float32, bm*bk)
	}
	for j0 := 0; j0 < n; j0 += bn {
		jn := min(bn, n-j0)
		for k0 := 0; k0 < k; k0 += bk {
			kk := min(bk, k-k0)
			bp := packB(b, ldb, j0, jn, k0, kk)
			acc := k0 > 0
			blocks := ceilDiv(m, bm)
			parallelFor(p, blocks, bm*kk*jn, func(w, lo, hi int) {
				ap := scratch[w]
				for blk := lo; blk < hi; blk++ {
					i0 := blk * bm
					im := min(bm, m-i0)
					packA(ap, a, lda, i0, im, k0, kk)
					macro(im, jn, kk, ap, bp, c[i0*ldc+j0:], ldc, acc)
				}
			})
		}
	}
}

func packA(dst, a []float32, lda, i0, mc, p0, kc int) {
	for ir := 0; ir < mc; ir += mr {
		rows := min(mr, mc-ir)
		base := ir / mr * kc * mr
		for r := 0; r < rows; r++ {
			src := a[(i0+ir+r)*lda+p0 : (i0+ir+r)*lda+p0+kc]
			for p := 0; p < kc; p++ {
				dst[base+p*mr+r] = src[p]
			}
		}
		for r := rows; r < mr; r++ {
			for p := 0; p < kc; p++ {
				dst[base+p*mr+r] = 0
			}
		}
	}
}

func packB(b []float32, ldb, j0, nc, p0, kc int) []float32 {
	jnp := ceilDiv(nc, nr) * nr
	out := make([]float32, jnp*kc)
	for jr := 0; jr < nc; jr += nr {
		cols := min(nr, nc-jr)
		base := jr / nr * kc * nr
		for s := 0; s < cols; s++ {
			src := b[(j0+jr+s)*ldb+p0 : (j0+jr+s)*ldb+p0+kc]
			for p := 0; p < kc; p++ {
				out[base+p*nr+s] = src[p]
			}
		}
		for s := cols; s < nr; s++ {
			for p := 0; p < kc; p++ {
				out[base+p*nr+s] = 0
			}
		}
	}
	return out
}

func macro(m, n, kc int, ap, bp []float32, c []float32, ldc int, acc bool) {
	for jr := 0; jr < n; jr += nr {
		nc := min(nr, n-jr)
		bpan := bp[jr/nr*kc*nr:]
		for ir := 0; ir < m; ir += mr {
			mc := min(mr, m-ir)
			apan := ap[ir/mr*kc*mr:]
			kernel(mc, nc, kc, apan, bpan, c[ir*ldc+jr:], ldc, acc)
		}
	}
}
