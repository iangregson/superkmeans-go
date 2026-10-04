package superkmeans

import "math"

func randomOrthonormal(d int, seed uint64) []float32 {
	r := newRNG(seed)
	a := make([]float64, d*d)
	for i := range a {
		a[i] = float64(r.norm())
	}
	q := make([]float64, d*d)
	for i := 0; i < d; i++ {
		q[i*d+i] = 1
	}
	v := make([]float64, d)
	for j := 0; j < d; j++ {
		var normSq float64
		for i := j; i < d; i++ {
			normSq += a[i*d+j] * a[i*d+j]
		}
		norm := math.Sqrt(normSq)
		if norm < 1e-12 {
			continue
		}
		alpha := norm
		if a[j*d+j] >= 0 {
			alpha = -norm
		}
		m := d - j
		vv := v[:m]
		vv[0] = a[j*d+j] - alpha
		for i := 1; i < m; i++ {
			vv[i] = a[(j+i)*d+j]
		}
		var vNormSq float64
		for _, x := range vv {
			vNormSq += x * x
		}
		if vNormSq < 1e-30 {
			continue
		}
		inv := 2 / vNormSq
		for c := j; c < d; c++ {
			var dot float64
			for i := 0; i < m; i++ {
				dot += vv[i] * a[(j+i)*d+c]
			}
			scale := inv * dot
			for i := 0; i < m; i++ {
				a[(j+i)*d+c] -= scale * vv[i]
			}
		}
		for rr := 0; rr < d; rr++ {
			var dot float64
			for i := 0; i < m; i++ {
				dot += q[rr*d+j+i] * vv[i]
			}
			scale := inv * dot
			for i := 0; i < m; i++ {
				q[rr*d+j+i] -= scale * vv[i]
			}
		}
	}
	out := make([]float32, d*d)
	for i, x := range q {
		out[i] = float32(x)
	}
	return out
}

func ratios(d int, eps0 float64) []float32 {
	out := make([]float32, d+1)
	eps := float32(eps0)
	t := float64(d)
	out[0] = 1
	out[d] = 1
	for b := 1; b < d; b++ {
		v := float64(b)
		f := 1 + float64(eps)/math.Sqrt(v)
		out[b] = float32((v / t) * f * f)
	}
	return out
}

func dimensionSplit(d int) (horizontal, vertical int) {
	proportion := 0.75
	if d <= 256 {
		proportion = 0.25
	}
	horizontal = int(float64(d) * proportion)
	vertical = d - horizontal
	if horizontal%64 != 0 {
		horizontal = int(math.Round(float64(horizontal)/64)) * 64
		vertical = d - horizontal
	}
	if vertical == 0 {
		horizontal = 64
		vertical = d - horizontal
	}
	if d <= 64 {
		horizontal = 0
		vertical = d
	}
	return horizontal, vertical
}
